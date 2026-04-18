from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import shlex
import shutil
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
BUNDLED_PYTHON_CANDIDATES = [
    ".venv/bin/python",
    ".venv/Scripts/python.exe",
]
DEFAULT_FAILED_SCORE = -1.0
VALID_TRIAL_STATUSES = {"completed"}
FAILURE_PATTERNS = {
    "oom": (
        "out of memory",
        "cuda out of memory",
        "cudnn_status_alloc_failed",
        "cublas_status_alloc_failed",
    ),
    "nan": (
        "loss=nan",
        "loss: nan",
        "train_loss=nan",
        "val_loss=nan",
        "non-finite",
        "overflow",
        "loss exploded",
    ),
    "traceback": ("traceback",),
    "data_error": (
        "filenotfounderror",
        "view path not found",
        "no such file or directory",
        "missing columns in csv",
    ),
}


RUNTIME_PARAM_DEFAULTS: dict[str, Any] = {
    "train.scheduler": "none",
    "train.early_stopping_patience": None,
}


@dataclass
class TrialOutcome:
    trial_number: int
    status: str
    objective: float
    val_accuracy: float | None = None
    threshold_val_accuracy: float | None = None
    val_auc: float | None = None
    val_f1: float | None = None
    train_loss: float | None = None
    val_loss: float | None = None
    total_seconds: float | None = None
    peak_vram_mb: float | None = None
    no_miss_threshold: float | None = None
    threshold_status: str | None = None
    threshold_failure_reason: str | None = None
    log_flags: list[str] = field(default_factory=list)
    failure_reason: str | None = None
    parsed_files: list[str] = field(default_factory=list)
    paths: dict[str, str] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    timestamps: dict[str, Any] = field(default_factory=dict)


@dataclass
class GPUDeviceInfo:
    index: int
    name: str
    memory_used_mb: int
    memory_total_mb: int
    utilization_gpu: int


@dataclass
class PreparedStudy:
    optuna_module: Any
    config_path: Path
    search_cfg: dict[str, Any]
    study_cfg: dict[str, Any]
    base_config: dict[str, Any]
    search_space: dict[str, Any]
    python_executable: str
    study_root: Path
    study: Any
    source_study_dir: Path | None
    target_trials: int
    existing_trials: int
    remaining_trials: int


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def dump_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


def flatten_mapping(data: Any, prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    if isinstance(data, dict):
        for key, value in data.items():
            joined = f"{prefix}.{key}" if prefix else str(key)
            flat.update(flatten_mapping(value, joined))
    elif isinstance(data, list):
        for index, value in enumerate(data):
            joined = f"{prefix}[{index}]"
            flat.update(flatten_mapping(value, joined))
    else:
        flat[prefix.lower()] = data
    return flat


def get_nested(mapping: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def set_nested(mapping: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    current = mapping
    for part in parts[:-1]:
        next_value = current.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            current[part] = next_value
        current = next_value
    current[parts[-1]] = value


def apply_overrides(mapping: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(mapping)
    for key, value in overrides.items():
        set_nested(result, key, value)
    return result


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def path_to_config_string(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def find_repaired_data_path(raw_path: str, base_dir: Path) -> Path | None:
    original = Path(raw_path)
    if original.is_absolute():
        return None

    candidates: list[Path] = []
    parts = list(original.parts)
    if len(parts) >= 2:
        for index in range(1, len(parts) - 1):
            collapsed = Path(*parts[:index], *parts[index + 1 :])
            candidates.append((base_dir / collapsed).resolve())

    direct_parent = (base_dir / original.parent.parent / original.name).resolve() if len(parts) >= 2 else None
    if direct_parent is not None:
        candidates.insert(0, direct_parent)

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate
    return None


def prepare_dataset_inputs(
    config: dict[str, Any],
    study_root: Path,
    max_missing: int = 5,
    allow_csv_rewrite: bool = False,
) -> dict[str, Any]:
    data_cfg = config.get("data", {})
    csv_path = resolve_path(data_cfg["csv_path"])
    if not csv_path.exists():
        raise SystemExit(f"Dataset CSV does not exist: {csv_path}")

    base_dir = resolve_path(data_cfg.get("base_dir", "."))
    required_columns = {"axial_dir", "coronal_dir", "sagittal_dir"}
    missing_paths: list[str] = []
    rewrites: list[dict[str, str]] = []
    rows: list[dict[str, Any]] = []
    fieldnames: list[str] | None = None

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SystemExit(f"Dataset CSV has no header row: {csv_path}")
        fieldnames = list(reader.fieldnames)
        missing_columns = required_columns - set(reader.fieldnames)
        if missing_columns:
            raise SystemExit(f"Dataset CSV is missing required columns: {sorted(missing_columns)}")

        for row_index, row in enumerate(reader, start=2):
            rewritten_row = dict(row)
            for column in ("axial_dir", "coronal_dir", "sagittal_dir"):
                raw_path = row.get(column, "")
                candidate = Path(raw_path)
                if not candidate.is_absolute():
                    candidate = (base_dir / candidate).resolve()
                if not candidate.exists():
                    repaired = find_repaired_data_path(raw_path, base_dir)
                    if repaired is not None:
                        rewritten_row[column] = path_to_config_string(repaired)
                        rewrites.append(
                            {
                                "patient_id": row.get("patient_id", f"row_{row_index}"),
                                "column": column,
                                "from": raw_path,
                                "to": rewritten_row[column],
                            }
                        )
                    else:
                        patient = row.get("patient_id", f"row {row_index}")
                        missing_paths.append(f"{patient} [{column}] -> {candidate}")
                        break
            rows.append(rewritten_row)
            if len(missing_paths) >= max_missing:
                break

    if missing_paths:
        joined = "\n  - ".join(missing_paths)
        raise SystemExit(
            "Dataset preflight failed; the study was not started because some metadata paths do not exist:\n"
            f"  - {joined}"
        )

    report = {
        "source_csv": str(csv_path),
        "base_dir": str(base_dir),
        "rewritten_entries": len(rewrites),
        "rewrites": rewrites[:50],
    }
    save_json(study_root / "dataset_preflight.json", report)

    if rewrites:
        resolved_csv = study_root / "resolved_metadata.csv"
        assert fieldnames is not None
        with resolved_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        if allow_csv_rewrite:
            config = copy.deepcopy(config)
            config.setdefault("data", {})
            config["data"]["csv_path"] = str(resolved_csv)
    return config


def load_search_config(path_like: str | Path) -> tuple[Path, dict[str, Any]]:
    path = resolve_path(path_like)
    data = load_yaml(path)
    if "study" not in data or "search_space" not in data:
        raise ValueError(f"Search config {path} must define 'study' and 'search_space'.")
    return path, data


def normalize_choice(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return value


def short_param_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def compact_params(params: dict[str, Any], max_items: int = 6) -> str:
    pieces = [f"{key}={short_param_value(value)}" for key, value in sorted(params.items())]
    if len(pieces) > max_items:
        visible = pieces[:max_items]
        visible.append(f"...(+{len(pieces) - max_items})")
        pieces = visible
    return ", ".join(pieces)


def is_active_param(spec: dict[str, Any], chosen_params: dict[str, Any]) -> bool:
    conditions = spec.get("condition", {})
    for key, allowed_values in conditions.items():
        if key not in chosen_params:
            return False
        if chosen_params[key] not in allowed_values:
            return False
    return True


def current_template_params(base_config: dict[str, Any], search_space: dict[str, Any]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for key, spec in search_space.items():
        if not is_active_param(spec, params):
            continue
        value = get_nested(base_config, key, RUNTIME_PARAM_DEFAULTS.get(key))
        if value is None:
            continue
        params[key] = normalize_choice(value)
    return params


def build_env(study_cfg: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    for key, value in study_cfg.get("env", {}).items():
        env[str(key)] = str(value)
    return env


def nvidia_smi_query(fields: str) -> list[str]:
    if shutil.which("nvidia-smi") is None:
        return []
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
    except (subprocess.SubprocessError, FileNotFoundError, PermissionError):
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def parse_nvidia_int(raw: str) -> int:
    token = str(raw).strip().split()[0]
    return int(float(token))


def discover_gpu_devices() -> list[GPUDeviceInfo]:
    lines = nvidia_smi_query("index,name,memory.used,memory.total,utilization.gpu")
    devices: list[GPUDeviceInfo] = []
    for line in lines:
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            devices.append(
                GPUDeviceInfo(
                    index=parse_nvidia_int(parts[0]),
                    name=parts[1],
                    memory_used_mb=parse_nvidia_int(parts[2]),
                    memory_total_mb=parse_nvidia_int(parts[3]),
                    utilization_gpu=parse_nvidia_int(parts[4]),
                )
            )
        except (TypeError, ValueError):
            continue
    return devices


def parse_gpu_id_spec(raw_value: str | None) -> list[int] | None:
    if raw_value is None:
        return None
    value = str(raw_value).strip()
    if not value or value.lower() == "auto":
        return None
    gpu_ids: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        gpu_ids.append(int(token))
    return gpu_ids


def parse_max_workers(raw_value: str | int | None, default: int) -> int:
    if raw_value is None:
        return max(1, default)
    if isinstance(raw_value, int):
        return max(1, raw_value)
    value = str(raw_value).strip().lower()
    if not value or value == "auto":
        return max(1, default)
    return max(1, int(value))


def select_gpu_devices(
    gpu_id_spec: str | None,
    max_used_memory_mb: int,
    max_utilization: int,
) -> list[GPUDeviceInfo]:
    devices = discover_gpu_devices()
    if not devices:
        return []

    explicit_ids = parse_gpu_id_spec(gpu_id_spec)
    if explicit_ids is not None:
        selected = [device for device in devices if device.index in explicit_ids]
        selected.sort(key=lambda device: explicit_ids.index(device.index))
        return selected

    idle_devices = [
        device
        for device in devices
        if device.memory_used_mb <= max_used_memory_mb and device.utilization_gpu <= max_utilization
    ]
    if idle_devices:
        idle_devices.sort(key=lambda device: (device.memory_used_mb, device.utilization_gpu, device.index))
        return idle_devices

    return []


def format_gpu_devices(devices: list[GPUDeviceInfo]) -> list[str]:
    return [
        (
            f"GPU {device.index}: {device.name} | "
            f"used={device.memory_used_mb} MiB / {device.memory_total_mb} MiB | "
            f"util={device.utilization_gpu}%"
        )
        for device in devices
    ]


def project_python_candidates(candidates: list[str] | None = None) -> list[str]:
    ordered_candidates: list[str] = []
    ordered_candidates.extend(BUNDLED_PYTHON_CANDIDATES)
    if candidates:
        ordered_candidates.extend(str(candidate) for candidate in candidates)

    resolved_candidates: list[str] = []
    seen: set[str] = set()
    venv_root = (REPO_ROOT / ".venv").absolute()
    for candidate in ordered_candidates:
        resolved = resolve_python_candidate(candidate)
        if not resolved:
            continue
        candidate_path = Path(resolved).absolute()
        normalized = str(candidate_path)
        if normalized in seen:
            continue
        seen.add(normalized)
        try:
            candidate_path.relative_to(venv_root)
        except ValueError:
            continue
        resolved_candidates.append(normalized)
    return resolved_candidates


def resolve_python_candidate(candidate: str) -> str | None:
    candidate = str(candidate).strip()
    if not candidate:
        return None

    candidate_path = Path(candidate)
    if candidate_path.is_absolute():
        resolved_path = candidate_path
    elif any(sep in candidate for sep in (os.sep, "/", "\\")) or candidate.startswith("."):
        resolved_path = (REPO_ROOT / candidate_path).absolute()
    else:
        return shutil.which(candidate)

    if resolved_path.exists():
        return str(resolved_path)
    return None


def python_candidate_works(candidate_to_run: str) -> bool:
    try:
        subprocess.run(
            [candidate_to_run, "-c", "import sys; print(sys.executable)"],
            cwd=REPO_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=10,
        )
        return True
    except (FileNotFoundError, PermissionError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


def detect_python_executable(candidates: list[str] | None = None) -> str:
    resolved_candidates = project_python_candidates(candidates)
    for resolved in resolved_candidates:
        if python_candidate_works(resolved):
            return resolved

    tried_candidates = [
        str((REPO_ROOT / candidate).absolute())
        for candidate in BUNDLED_PYTHON_CANDIDATES
    ]
    raise RuntimeError(
        "Could not find a usable project virtualenv Python executable. "
        "Expected one of: "
        + ", ".join(tried_candidates)
    )


def ensure_running_in_project_python(python_executable: str) -> None:
    current_python = Path(sys.executable).absolute()
    current_prefix = Path(sys.prefix).absolute()
    selected_python = Path(python_executable).absolute()
    venv_root = (REPO_ROOT / ".venv").absolute()

    try:
        current_python.relative_to(venv_root)
        current_uses_project_venv = True
    except ValueError:
        current_uses_project_venv = current_prefix == venv_root

    if current_uses_project_venv:
        return

    raise SystemExit(
        "This Optuna workflow must be launched with the project virtualenv interpreter. "
        f"Current interpreter: {current_python}. "
        f"Current prefix: {current_prefix}. "
        f"Expected under: {venv_root}. "
        f"Detected study interpreter: {selected_python}."
    )


def probe_torch_cuda(python_executable: str, env: dict[str, str] | None = None) -> dict[str, Any]:
    probe_code = (
        "import json, torch; "
        "device_count = int(torch.cuda.device_count()); "
        "payload = {"
        "'cuda_available': bool(torch.cuda.is_available()), "
        "'device_count': device_count, "
        "'device_names': [torch.cuda.get_device_name(i) for i in range(device_count)] "
        "if torch.cuda.is_available() else []"
        "}; "
        "print(json.dumps(payload))"
    )
    try:
        result = subprocess.run(
            [python_executable, "-c", probe_code],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
            env=env,
        )
    except (FileNotFoundError, PermissionError, subprocess.SubprocessError) as exc:
        raise SystemExit(
            "CUDA preflight failed because the project Python environment could not be queried. "
            f"Python={python_executable!r}, error={exc!r}"
        ) from exc

    stdout_lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not stdout_lines:
        raise SystemExit(
            "CUDA preflight failed because the probe returned no output. "
            f"Python={python_executable!r}"
        )

    try:
        return json.loads(stdout_lines[-1])
    except json.JSONDecodeError as exc:
        stderr_text = result.stderr.strip()
        raise SystemExit(
            "CUDA preflight returned an unreadable response. "
            f"stdout_tail={stdout_lines[-1]!r}, stderr={stderr_text!r}"
        ) from exc


def ensure_cuda_ready(python_executable: str, env: dict[str, str] | None = None) -> dict[str, Any]:
    cuda_info = probe_torch_cuda(python_executable, env=env)
    if bool(cuda_info.get("cuda_available")) and int(cuda_info.get("device_count", 0)) > 0:
        return cuda_info

    raise SystemExit(
        "CUDA preflight failed for AutoResearch. "
        f"Python={python_executable!r}, "
        f"cuda_available={cuda_info.get('cuda_available')}, "
        f"device_count={cuda_info.get('device_count')}. "
        "This workflow must not fall back to CPU; fix the CUDA environment first."
    )


def import_optuna():
    try:
        import optuna  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - runtime guard
        message = (
            "Optuna is not installed in the training environment. "
            "Install it in the project environment before running this workflow, for example:\n"
            "  .venv/bin/python -m pip install optuna\n"
            "  or ./.venv/bin/python -m pip install optuna"
        )
        raise SystemExit(message) from exc
    return optuna


def choose_sampler(optuna_module, study_cfg: dict[str, Any]):
    seed = int(study_cfg.get("sampler_seed", 42))
    startup_trials = int(study_cfg.get("startup_trials", 4))
    sampler_name = str(study_cfg.get("sampler", "tpe")).lower()
    if sampler_name == "random":
        return optuna_module.samplers.RandomSampler(seed=seed)
    return optuna_module.samplers.TPESampler(seed=seed, n_startup_trials=startup_trials)


def choose_pruner(optuna_module, study_cfg: dict[str, Any]):
    pruner_name = str(study_cfg.get("pruner", "none")).lower()
    if pruner_name == "median":
        return optuna_module.pruners.MedianPruner(
            n_startup_trials=int(study_cfg.get("pruner_startup_trials", 5))
        )
    return optuna_module.pruners.NopPruner()


def build_study(optuna_module, search_cfg: dict[str, Any], study_root: Path, resume: bool = False):
    study_cfg = search_cfg["study"]
    study_root.mkdir(parents=True, exist_ok=True)
    storage_name = study_cfg.get("storage", "study.sqlite3")
    storage_path = study_root / str(storage_name)
    storage_url = f"sqlite:///{storage_path.as_posix()}"
    if storage_path.exists() and not resume:
        raise FileExistsError(
            f"Study storage already exists at {storage_path}. Use --resume to continue this study or choose a fresh study_root."
        )
    return optuna_module.create_study(
        study_name=str(study_cfg["name"]),
        direction=str(study_cfg.get("direction", "maximize")),
        storage=storage_url,
        load_if_exists=resume,
        sampler=choose_sampler(optuna_module, study_cfg),
        pruner=choose_pruner(optuna_module, study_cfg),
    )


def fail_stale_running_trials(study, study_root: Path, optuna_module) -> int:
    running_state = optuna_module.trial.TrialState.RUNNING
    failed_state = optuna_module.trial.TrialState.FAIL
    stale_trials = [trial for trial in study.trials if trial.state == running_state]
    if not stale_trials:
        return 0

    stale_reason = "Marked failed on resume after a stale RUNNING Optuna trial was detected."
    finished_at = now_iso()
    for trial in stale_trials:
        study.tell(trial.number, state=failed_state)
        trial_file = study_root / "trials" / f"trial_{trial.number:04d}" / "trial.json"
        payload = read_json(trial_file) or {"trial_number": trial.number}
        payload["status"] = "crash"
        payload["failure_reason"] = stale_reason
        payload.setdefault("params", dict(trial.params))
        timestamps = payload.get("timestamps")
        if not isinstance(timestamps, dict):
            timestamps = {}
        timestamps.setdefault("finished_at", finished_at)
        payload["timestamps"] = timestamps
        save_json(trial_file, payload)
    return len(stale_trials)


def suggest_value(trial, key: str, spec: dict[str, Any]) -> Any:
    param_type = str(spec["type"]).lower()
    if param_type == "float":
        return trial.suggest_float(
            key,
            float(spec["low"]),
            float(spec["high"]),
            log=bool(spec.get("log", False)),
            step=spec.get("step"),
        )
    if param_type == "int":
        return trial.suggest_int(
            key,
            int(spec["low"]),
            int(spec["high"]),
            step=int(spec.get("step", 1)),
            log=bool(spec.get("log", False)),
        )
    if param_type == "categorical":
        return trial.suggest_categorical(key, list(spec["choices"]))
    raise ValueError(f"Unsupported search space type for {key}: {param_type}")


def sample_trial_params(trial, search_space: dict[str, Any]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for key, spec in search_space.items():
        if not is_active_param(spec, params):
            continue
        params[key] = suggest_value(trial, key, spec)
    return params


def find_metric_value(flat: dict[str, Any], explicit_paths: list[str], include_tokens: list[str]) -> float | None:
    for path in explicit_paths:
        value = coerce_float(flat.get(path.lower()))
        if value is not None:
            return value

    candidates: list[tuple[str, float]] = []
    include = [token.lower() for token in include_tokens]
    for path, value in flat.items():
        numeric = coerce_float(value)
        if numeric is None:
            continue
        if "test" in path:
            continue
        if all(token in path for token in include):
            candidates.append((path, numeric))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0].count("."))
    return candidates[0][1]


def extract_losses(history_data: dict[str, Any] | None) -> tuple[float | None, float | None]:
    if not history_data:
        return None, None
    history = history_data.get("history")
    if not isinstance(history, list) or not history:
        return None, None

    best_record = None
    best_val_accuracy = -math.inf
    for record in history:
        if not isinstance(record, dict):
            continue
        candidate_accuracy = coerce_float(record.get("val_accuracy"))
        if candidate_accuracy is not None and candidate_accuracy > best_val_accuracy:
            best_val_accuracy = candidate_accuracy
            best_record = record
    if best_record is None:
        best_record = history[-1] if isinstance(history[-1], dict) else None
    if not isinstance(best_record, dict):
        return None, None
    return coerce_float(best_record.get("train_loss")), coerce_float(best_record.get("val_loss"))


def detect_log_flags(log_text: str) -> list[str]:
    lowered = log_text.lower()
    flags: list[str] = []
    for flag, patterns in FAILURE_PATTERNS.items():
        if any(pattern in lowered for pattern in patterns):
            flags.append(flag)
    return flags


def extract_failure_reason(*log_blobs: str) -> str | None:
    for blob in log_blobs:
        lines = [line.strip() for line in blob.splitlines() if line.strip()]
        for line in reversed(lines):
            if line in {"TIMEOUT"} or line.startswith("EXIT_CODE=") or line.startswith("$ "):
                continue
            return line
    return None


def read_log_tail(path: Path, lines: int = 80) -> str:
    if not path.exists():
        return ""
    content = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return "\n".join(content[-lines:])


def parse_trial_outputs(
    trial_number: int,
    params: dict[str, Any],
    run_dir: Path,
    train_log_path: Path,
    threshold_log_path: Path,
    failed_score: float,
) -> TrialOutcome:
    summary_path = run_dir / "summary.json"
    threshold_path = run_dir / "threshold_eval.json"
    history_path = run_dir / "history.json"
    metrics_path = run_dir / "metrics.json"
    results_path = run_dir / "results.json"

    summary_data = read_json(summary_path) or {}
    threshold_data = read_json(threshold_path) or {}
    history_data = read_json(history_path)
    metrics_data = read_json(metrics_path) or {}
    results_data = read_json(results_path) or {}

    parsed_files = [
        str(path.relative_to(REPO_ROOT))
        for path in [summary_path, threshold_path, history_path, metrics_path, results_path]
        if path.exists()
    ]

    flat_summary = flatten_mapping(summary_data)
    flat_threshold = flatten_mapping(threshold_data)
    flat_metrics = flatten_mapping(metrics_data)
    flat_results = flatten_mapping(results_data)
    flat_all = {}
    flat_all.update(flat_results)
    flat_all.update(flat_metrics)
    flat_all.update(flat_summary)
    flat_all.update(flat_threshold)

    val_accuracy = find_metric_value(
        flat_summary,
        ["best_val.accuracy"],
        ["best_val", "accuracy"],
    )
    threshold_val_accuracy = find_metric_value(
        flat_threshold,
        ["val.accuracy"],
        ["val", "accuracy"],
    )
    val_auc = find_metric_value(
        flat_summary,
        ["best_val.auc"],
        ["best_val", "auc"],
    )
    val_f1 = find_metric_value(
        flat_summary,
        ["best_val.f1"],
        ["best_val", "f1"],
    )
    total_seconds = find_metric_value(
        flat_all,
        ["runtime.total_seconds", "total_seconds"],
        ["total_seconds"],
    )
    peak_vram_mb = find_metric_value(
        flat_all,
        ["runtime.peak_vram_mb", "peak_vram_mb"],
        ["peak", "vram"],
    )
    no_miss_threshold = find_metric_value(
        flat_all,
        ["no_miss_threshold"],
        ["no_miss", "threshold"],
    )
    train_loss, val_loss = extract_losses(history_data)

    train_log_tail = read_log_tail(train_log_path)
    threshold_log_tail = read_log_tail(threshold_log_path)
    train_log_flags = detect_log_flags(train_log_tail)
    threshold_log_flags = detect_log_flags(threshold_log_tail)
    log_flags = sorted(set(train_log_flags + threshold_log_flags))

    status = "completed"
    failure_reason = None
    if "oom" in train_log_flags:
        status = "oom"
        failure_reason = "OOM detected from logs."
    elif "data_error" in train_log_flags:
        status = "crash"
        failure_reason = extract_failure_reason(train_log_tail) or "Data path or metadata error detected from training logs."
    elif "nan" in train_log_flags:
        status = "invalid"
        failure_reason = "NaN or non-finite signal detected from training logs."
    elif val_accuracy is None:
        status = "crash"
        failure_reason = extract_failure_reason(train_log_tail) or "Could not parse summary.json.best_val.accuracy from run artifacts."

    objective = val_accuracy if status == "completed" and val_accuracy is not None else failed_score
    return TrialOutcome(
        trial_number=trial_number,
        status=status,
        objective=float(objective),
        val_accuracy=val_accuracy,
        threshold_val_accuracy=threshold_val_accuracy,
        val_auc=val_auc,
        val_f1=val_f1,
        train_loss=train_loss,
        val_loss=val_loss,
        total_seconds=total_seconds,
        peak_vram_mb=peak_vram_mb,
        no_miss_threshold=no_miss_threshold,
        log_flags=log_flags,
        failure_reason=failure_reason,
        parsed_files=parsed_files,
        paths={
            "run_dir": str(run_dir),
            "train_log": str(train_log_path),
            "threshold_log": str(threshold_log_path),
            "summary_json": str(summary_path),
            "threshold_eval_json": str(threshold_path),
            "history_json": str(history_path),
        },
        params=params,
    )


def execute_command(
    command: list[str],
    log_path: Path,
    timeout_seconds: int | None,
    env: dict[str, str],
) -> tuple[str, str | None]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"$ {shlex.join(command)}\n\n")
        handle.flush()
        try:
            subprocess.run(
                command,
                cwd=REPO_ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=env,
                timeout=timeout_seconds,
                check=True,
            )
            return "ok", None
        except subprocess.TimeoutExpired:
            handle.write("\nTIMEOUT\n")
            return "timeout", f"Timed out after {timeout_seconds} seconds."
        except subprocess.CalledProcessError as exc:
            handle.write(f"\nEXIT_CODE={exc.returncode}\n")
            return "crash", f"Command failed with exit code {exc.returncode}."


def prepare_trial_config(
    base_config: dict[str, Any],
    search_cfg: dict[str, Any],
    params: dict[str, Any],
    trial_dir: Path,
    trial_number: int,
) -> dict[str, Any]:
    config = apply_overrides(base_config, search_cfg.get("fixed_overrides", {}))
    config = apply_overrides(config, params)
    study_cfg = search_cfg["study"]
    config["output_dir"] = str((trial_dir / "run").relative_to(REPO_ROOT))
    config["experiment_name"] = f"{study_cfg['name']}_trial_{trial_number:04d}"
    return config


def run_single_trial(
    python_executable: str,
    base_config: dict[str, Any],
    search_cfg: dict[str, Any],
    params: dict[str, Any],
    trial_dir: Path,
    trial_number: int,
    env_overrides: dict[str, str] | None = None,
    worker_label: str | None = None,
) -> TrialOutcome:
    study_cfg = search_cfg["study"]
    env = build_env(study_cfg)
    if env_overrides:
        env.update({str(key): str(value) for key, value in env_overrides.items()})
    trial_timeout_minutes = study_cfg.get("trial_timeout_minutes")
    trial_timeout_seconds = None
    if trial_timeout_minutes is not None:
        trial_timeout_seconds = int(float(trial_timeout_minutes) * 60)
    failed_score = float(study_cfg.get("failed_score", DEFAULT_FAILED_SCORE))

    config = prepare_trial_config(base_config, search_cfg, params, trial_dir, trial_number)
    config_path = trial_dir / "config.yaml"
    dump_yaml(config_path, config)

    metadata = {
        "trial_number": trial_number,
        "status": "running",
        "params": params,
        "config_path": str(config_path),
        "run_dir": config["output_dir"],
        "started_at": now_iso(),
        "worker_label": worker_label,
    }
    save_json(trial_dir / "trial.json", metadata)

    train_log_path = trial_dir / "train.log"
    threshold_log_path = trial_dir / "threshold.log"
    run_dir = resolve_path(config["output_dir"])

    started = time.perf_counter()
    train_result, train_reason = execute_command(
        [python_executable, "train.py", "--config", str(config_path.relative_to(REPO_ROOT))],
        log_path=train_log_path,
        timeout_seconds=trial_timeout_seconds,
        env=env,
    )

    if train_result == "ok":
        threshold_result, threshold_reason = execute_command(
            [
                python_executable,
                "tools/evaluate_threshold.py",
                "--run_dir",
                str(run_dir.relative_to(REPO_ROOT)),
                "--config",
                str(config_path.relative_to(REPO_ROOT)),
            ],
            log_path=threshold_log_path,
            timeout_seconds=trial_timeout_seconds,
            env=env,
        )
    else:
        threshold_result, threshold_reason = "skipped", None

    outcome = parse_trial_outputs(
        trial_number=trial_number,
        params=params,
        run_dir=run_dir,
        train_log_path=train_log_path,
        threshold_log_path=threshold_log_path,
        failed_score=failed_score,
    )
    duration_seconds = time.perf_counter() - started
    outcome.timestamps = {
        "started_at": metadata["started_at"],
        "finished_at": now_iso(),
        "duration_seconds": duration_seconds,
    }
    outcome.threshold_status = threshold_result
    if threshold_result in {"crash", "timeout"}:
        outcome.threshold_failure_reason = threshold_reason or f"Threshold evaluation {threshold_result}."
    if outcome.total_seconds is None:
        outcome.total_seconds = duration_seconds
    outcome.paths["config_yaml"] = str(config_path)
    outcome.paths["trial_dir"] = str(trial_dir)
    if worker_label:
        outcome.paths["worker_label"] = worker_label

    if train_result == "timeout":
        outcome.status = "timeout"
        outcome.failure_reason = train_reason
        outcome.objective = failed_score
    elif train_result == "crash":
        if "oom" not in outcome.log_flags:
            outcome.status = "crash"
        outcome.failure_reason = outcome.failure_reason or train_reason
        outcome.objective = failed_score

    trial_payload = asdict(outcome)
    trial_payload["train_command"] = [python_executable, "train.py", "--config", str(config_path.relative_to(REPO_ROOT))]
    trial_payload["threshold_command"] = [
        python_executable,
        "tools/evaluate_threshold.py",
        "--run_dir",
        str(run_dir.relative_to(REPO_ROOT)),
        "--config",
        str(config_path.relative_to(REPO_ROOT)),
    ]
    save_json(trial_dir / "trial.json", trial_payload)
    return outcome


def load_trial_records(study_root: Path) -> list[dict[str, Any]]:
    trials_dir = study_root / "trials"
    if not trials_dir.exists():
        return []
    records = []
    for trial_file in sorted(trials_dir.glob("trial_*/trial.json")):
        payload = read_json(trial_file)
        if payload:
            records.append(payload)
    return records


def metric_tuple(record: dict[str, Any], primary_key: str = "val_accuracy") -> tuple[float, float] | None:
    primary = coerce_float(record.get(primary_key))
    if primary is None:
        return None
    auc = coerce_float(record.get("val_auc"))
    if auc is None:
        auc = float("-inf")
    return primary, auc


def best_record(records: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    valid = [record for record in records if record.get("status") in VALID_TRIAL_STATUSES]
    if not valid:
        return None
    if key == "val_accuracy":
        ranked = [record for record in valid if metric_tuple(record, "val_accuracy") is not None]
        if not ranked:
            return None
        return max(ranked, key=lambda item: metric_tuple(item, "val_accuracy"))
    ranked = [record for record in valid if record.get(key) is not None]
    if not ranked:
        return None
    return max(ranked, key=lambda item: item[key])


def save_study_status(study_root: Path, search_cfg: dict[str, Any], records: list[dict[str, Any]], state: str) -> None:
    valid_records = [record for record in records if record.get("status") in VALID_TRIAL_STATUSES]
    best = best_record(records, "val_accuracy")
    payload = {
        "study_name": search_cfg["study"]["name"],
        "state": state,
        "updated_at": now_iso(),
        "study_root": str(study_root),
        "total_trials": len(records),
        "valid_trials": len(valid_records),
        "failure_counts": {},
        "selection_rule": "val_accuracy_then_val_auc",
    }
    for record in records:
        status = str(record.get("status", "unknown"))
        payload["failure_counts"][status] = payload["failure_counts"].get(status, 0) + 1
    if best:
        payload["best_trial_number"] = best.get("trial_number")
        payload["best_val_accuracy"] = best.get("val_accuracy")
        payload["best_val_auc"] = best.get("val_auc")
        payload["best_params"] = best.get("params", {})
    save_json(study_root / "study_status.json", payload)


def enqueue_template_trial(study, base_config: dict[str, Any], search_space: dict[str, Any]) -> None:
    params = current_template_params(base_config, search_space)
    if params:
        study.enqueue_trial(params)


def load_top_trial_params(source_study_dir: Path, top_k: int) -> list[dict[str, Any]]:
    records = load_trial_records(source_study_dir)
    valid = [
        record
        for record in records
        if record.get("status") in VALID_TRIAL_STATUSES and metric_tuple(record, "val_accuracy") is not None
    ]
    valid.sort(key=lambda item: metric_tuple(item, "val_accuracy"), reverse=True)
    return [record.get("params", {}) for record in valid[:top_k]]


def enqueue_source_trials(study, source_study_dir: Path, top_k: int, search_space: dict[str, Any]) -> None:
    for params in load_top_trial_params(source_study_dir, top_k):
        filtered = {key: value for key, value in params.items() if key in search_space}
        if filtered:
            study.enqueue_trial(filtered)


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [
        record
        for record in records
        if record.get("status") in VALID_TRIAL_STATUSES and metric_tuple(record, "val_accuracy") is not None
    ]
    best_acc = best_record(records, "val_accuracy")
    best_auc = best_record(records, "val_auc")
    fastest = None
    if valid:
        fastest = min(
            [record for record in valid if record.get("total_seconds") is not None],
            key=lambda item: item["total_seconds"],
            default=None,
        )
    summary = {
        "trial_count": len(records),
        "valid_count": len(valid),
        "best_by_accuracy": best_acc,
        "best_by_auc": best_auc,
        "fastest_valid": fastest,
    }
    return summary


def run_study(
    search_config_path: str | Path,
    source_study_dir: str | Path | None = None,
    top_k: int = 0,
    max_trials_override: int | None = None,
    resume: bool = False,
    cleanup_stale_running_trials_on_resume: bool = True,
    env_overrides: dict[str, str] | None = None,
    worker_label: str | None = None,
) -> Path:
    prepared = prepare_study(
        search_config_path=search_config_path,
        source_study_dir=source_study_dir,
        top_k=top_k,
        max_trials_override=max_trials_override,
        resume=resume,
        cleanup_stale_running_trials_on_resume=cleanup_stale_running_trials_on_resume,
    )
    if prepared.remaining_trials <= 0:
        finalize_study(prepared, state="completed")
        return prepared.study_root

    def objective(trial) -> float:
        params = sample_trial_params(trial, prepared.search_space)
        trial_dir = prepared.study_root / "trials" / f"trial_{trial.number:04d}"
        outcome = run_single_trial(
            python_executable=prepared.python_executable,
            base_config=prepared.base_config,
            search_cfg=prepared.search_cfg,
            params=params,
            trial_dir=trial_dir,
            trial_number=trial.number,
            env_overrides=env_overrides,
            worker_label=worker_label,
        )
        annotate_trial_result(trial, outcome)
        return float(outcome.objective)

    def callback(current_study, _trial) -> None:
        records = load_trial_records(prepared.study_root)
        save_study_status(prepared.study_root, prepared.search_cfg, records, state="running")

    optimize_kwargs: dict[str, Any] = {
        "n_trials": prepared.remaining_trials,
        "callbacks": [callback],
    }
    if prepared.study_cfg.get("timeout_minutes") is not None:
        optimize_kwargs["timeout"] = int(float(prepared.study_cfg["timeout_minutes"]) * 60)
    try:
        prepared.study.optimize(objective, **optimize_kwargs)
    except KeyboardInterrupt:
        finalize_study(prepared, state="interrupted")
        raise
    except Exception:
        finalize_study(prepared, state="failed")
        raise

    finalize_study(prepared, state="completed")
    return prepared.study_root


def prepare_study(
    search_config_path: str | Path,
    source_study_dir: str | Path | None = None,
    top_k: int = 0,
    max_trials_override: int | None = None,
    resume: bool = False,
    cleanup_stale_running_trials_on_resume: bool = True,
) -> PreparedStudy:
    config_path, search_cfg = load_search_config(search_config_path)
    study_cfg = search_cfg["study"]
    objective_metric = str(study_cfg.get("objective_metric", "val_accuracy"))
    if objective_metric != "val_accuracy":
        raise ValueError(
            "Only study.objective_metric=val_accuracy is supported. "
            "The objective is read from summary.json.best_val.accuracy."
        )
    base_config_path = resolve_path(study_cfg["base_config"])
    base_config = load_yaml(base_config_path)
    search_space = search_cfg["search_space"]
    python_executable = detect_python_executable(search_cfg.get("python_candidates"))
    ensure_running_in_project_python(python_executable)
    optuna = import_optuna()
    ensure_cuda_ready(python_executable, env=build_env(study_cfg))

    study_root = resolve_path(study_cfg["study_root"])
    study_root.mkdir(parents=True, exist_ok=True)
    (study_root / "trials").mkdir(parents=True, exist_ok=True)
    preflight_config = apply_overrides(base_config, search_cfg.get("fixed_overrides", {}))
    if bool(study_cfg.get("validate_dataset", True)):
        preflight_config = prepare_dataset_inputs(
            preflight_config,
            study_root,
            allow_csv_rewrite=bool(study_cfg.get("rewrite_dataset_csv", False)),
        )
        base_config = copy.deepcopy(base_config)
        base_config["data"]["csv_path"] = preflight_config["data"]["csv_path"]

    resolved_base_config = study_root / "base_config.resolved.yaml"
    dump_yaml(resolved_base_config, preflight_config)
    resolved_source_study = resolve_path(source_study_dir) if source_study_dir else None

    save_json(
        study_root / "search_config.snapshot.json",
        {
            "config_path": str(config_path),
            "search_config": search_cfg,
            "python_executable": python_executable,
            "resume": resume,
            "resolved_base_config": str(resolved_base_config),
            "source_study_dir": str(resolved_source_study) if resolved_source_study else None,
        },
    )

    study = build_study(optuna, search_cfg, study_root, resume=resume)
    if resume and cleanup_stale_running_trials_on_resume:
        fail_stale_running_trials(study, study_root, optuna)
    existing_trials = len(study.trials)
    target_trials = int(max_trials_override or study_cfg.get("n_trials", 10))
    remaining_trials = max(0, target_trials - existing_trials)
    if bool(study_cfg.get("enqueue_current_template", True)) and existing_trials == 0:
        enqueue_template_trial(study, preflight_config, search_space)
    if resolved_source_study and top_k > 0 and existing_trials == 0:
        enqueue_source_trials(study, resolved_source_study, top_k, search_space)
    save_study_status(study_root, search_cfg, load_trial_records(study_root), state="running")

    return PreparedStudy(
        optuna_module=optuna,
        config_path=config_path,
        search_cfg=search_cfg,
        study_cfg=study_cfg,
        base_config=base_config,
        search_space=search_space,
        python_executable=python_executable,
        study_root=study_root,
        study=study,
        source_study_dir=resolved_source_study,
        target_trials=target_trials,
        existing_trials=existing_trials,
        remaining_trials=remaining_trials,
    )


def is_finished_trial_update_error(exc: BaseException) -> bool:
    return exc.__class__.__name__ == "UpdateFinishedTrialError"


def set_trial_user_attr_best_effort(trial, key: str, value: Any) -> bool:
    try:
        trial.set_user_attr(key, value)
        return True
    except Exception as exc:  # pragma: no cover - defensive runtime guard
        if is_finished_trial_update_error(exc):
            return False
        raise


def annotate_trial_result(trial, outcome: TrialOutcome) -> bool:
    attrs = {
        "status": outcome.status,
        "params": outcome.params,
        "val_accuracy": outcome.val_accuracy,
        "threshold_val_accuracy": outcome.threshold_val_accuracy,
        "val_auc": outcome.val_auc,
        "val_f1": outcome.val_f1,
        "train_loss": outcome.train_loss,
        "val_loss": outcome.val_loss,
        "total_seconds": outcome.total_seconds,
        "peak_vram_mb": outcome.peak_vram_mb,
        "no_miss_threshold": outcome.no_miss_threshold,
    }
    if outcome.failure_reason:
        attrs["failure_reason"] = outcome.failure_reason

    wrote_all_attrs = True
    for key, value in attrs.items():
        if not set_trial_user_attr_best_effort(trial, key, value):
            wrote_all_attrs = False
            break
    return wrote_all_attrs


def record_trial_result(study, trial, outcome: TrialOutcome) -> bool:
    wrote_all_attrs = annotate_trial_result(trial, outcome)
    study.tell(trial, float(outcome.objective), skip_if_finished=True)
    return wrote_all_attrs


def finalize_study(prepared: PreparedStudy, state: str) -> None:
    records = load_trial_records(prepared.study_root)
    save_study_status(prepared.study_root, prepared.search_cfg, records, state=state)
    if state != "completed":
        return

    summary = summarize_records(records)
    save_json(
        prepared.study_root / "study_summary.json",
        {
            "study_name": prepared.study_cfg["name"],
            "updated_at": now_iso(),
            "search_config": str(prepared.config_path),
            "source_study_dir": str(prepared.source_study_dir) if prepared.source_study_dir else None,
            "selection_rule": "val_accuracy_then_val_auc",
            "target_trials": prepared.target_trials,
            "existing_trials": prepared.existing_trials,
            "remaining_trials": prepared.remaining_trials,
            "summary": summary,
        },
    )


def run_study_parallel(
    search_config_path: str | Path,
    source_study_dir: str | Path | None = None,
    top_k: int = 0,
    max_trials_override: int | None = None,
    resume: bool = False,
    cleanup_stale_running_trials_on_resume: bool = True,
    gpu_ids: list[int] | None = None,
    worker_cooldown_seconds: float = 0.0,
) -> Path:
    prepared = prepare_study(
        search_config_path=search_config_path,
        source_study_dir=source_study_dir,
        top_k=top_k,
        max_trials_override=max_trials_override,
        resume=resume,
        cleanup_stale_running_trials_on_resume=cleanup_stale_running_trials_on_resume,
    )
    if prepared.remaining_trials <= 0:
        finalize_study(prepared, state="completed")
        return prepared.study_root

    selected_gpu_ids = [int(gpu_id) for gpu_id in (gpu_ids or [])]
    print("Launching Optuna worker(s) on GPUs: " + ", ".join(str(gpu_id) for gpu_id in selected_gpu_ids))

    study_lock = threading.Lock()
    print_lock = threading.Lock()
    stop_event = threading.Event()
    worker_errors: list[BaseException] = []
    failed_score = float(prepared.study_cfg.get("failed_score", DEFAULT_FAILED_SCORE))

    def log_line(message: str) -> None:
        with print_lock:
            print(message, flush=True)

    def worker_loop(worker_index: int, gpu_id: int) -> None:
        worker_label = f"worker{worker_index}/gpu{gpu_id}"
        env_overrides = {"CUDA_VISIBLE_DEVICES": str(gpu_id)}
        while not stop_event.is_set():
            with study_lock:
                if len(prepared.study.trials) >= prepared.target_trials:
                    return
                trial = prepared.study.ask()
                params = sample_trial_params(trial, prepared.search_space)
                trial_number = trial.number
            log_line(
                f"[{worker_label}] trial {trial_number} started"
                + (f" ({compact_params(params)})" if params else "")
            )

            try:
                outcome = run_single_trial(
                    python_executable=prepared.python_executable,
                    base_config=prepared.base_config,
                    search_cfg=prepared.search_cfg,
                    params=params,
                    trial_dir=prepared.study_root / "trials" / f"trial_{trial_number:04d}",
                    trial_number=trial_number,
                    env_overrides=env_overrides,
                    worker_label=worker_label,
                )
            except Exception as exc:  # pragma: no cover - defensive runtime guard
                with study_lock:
                    attrs_written = True
                    attrs_written &= set_trial_user_attr_best_effort(trial, "status", "crash")
                    attrs_written &= set_trial_user_attr_best_effort(
                        trial,
                        "failure_reason",
                        f"Unhandled worker exception: {exc!r}",
                    )
                    prepared.study.tell(trial, failed_score, skip_if_finished=True)
                    records = load_trial_records(prepared.study_root)
                    save_study_status(prepared.study_root, prepared.search_cfg, records, state="running")
                worker_errors.append(exc)
                stop_event.set()
                if not attrs_written:
                    log_line(
                        f"[{worker_label}] trial {trial_number} was already finished before crash attrs were written."
                    )
                log_line(f"[{worker_label}] trial {trial_number} aborted by worker exception: {exc!r}")
                return

            with study_lock:
                attrs_written = record_trial_result(prepared.study, trial, outcome)
                records = load_trial_records(prepared.study_root)
                save_study_status(prepared.study_root, prepared.search_cfg, records, state="running")
            if not attrs_written:
                log_line(
                    f"[{worker_label}] trial {trial_number} finished before Optuna attrs were written; "
                    "trial.json remains the source of truth."
                )
            metric_text = (
                f"val_acc={outcome.val_accuracy:.6f}"
                if outcome.val_accuracy is not None
                else f"status={outcome.status}"
            )
            log_line(f"[{worker_label}] trial {trial_number} finished: {metric_text}")
            if worker_cooldown_seconds > 0:
                time.sleep(worker_cooldown_seconds)

    threads = [
        threading.Thread(
            target=worker_loop,
            args=(worker_index, gpu_id),
            name=f"optuna-gpu-{gpu_id}",
            daemon=False,
        )
        for worker_index, gpu_id in enumerate(selected_gpu_ids)
    ]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    except KeyboardInterrupt:
        stop_event.set()
        for thread in threads:
            thread.join()
        finalize_study(prepared, state="interrupted")
        raise

    if worker_errors:
        finalize_study(prepared, state="failed")
        raise RuntimeError(f"Parallel Optuna worker failed: {worker_errors[0]!r}")

    finalize_study(prepared, state="completed")
    return prepared.study_root


def run_study_adaptive(
    search_config_path: str | Path,
    source_study_dir: str | Path | None = None,
    top_k: int = 0,
    max_trials_override: int | None = None,
    resume: bool = False,
    cleanup_stale_running_trials_on_resume: bool = True,
    sequential: bool = False,
    gpu_ids: str | None = "auto",
    max_workers: str | int | None = "auto",
    max_used_memory_mb: int = 1024,
    max_utilization: int = 20,
    worker_cooldown_seconds: float = 0.0,
) -> Path:
    explicit_ids = parse_gpu_id_spec(gpu_ids)
    discovered_devices = discover_gpu_devices()
    if explicit_ids is not None:
        discovered_by_id = {device.index: device for device in discovered_devices}
        missing_ids = [gpu_id for gpu_id in explicit_ids if gpu_id not in discovered_by_id]
        if missing_ids:
            raise ValueError(
                f"Some explicit --gpu-ids entries were not found via nvidia-smi: {missing_ids}"
            )
        selected_devices = [discovered_by_id[gpu_id] for gpu_id in explicit_ids]
        busy_devices = [
            device
            for device in selected_devices
            if device.memory_used_mb > max_used_memory_mb or device.utilization_gpu > max_utilization
        ]
        if busy_devices:
            details = "\n".join(f"  {line}" for line in format_gpu_devices(selected_devices))
            raise SystemExit(
                "Explicit --gpu-ids entries do not satisfy the configured idle thresholds. "
                f"Thresholds: used<={max_used_memory_mb} MiB, util<={max_utilization}%.\n"
                f"{details}"
            )
    else:
        selected_devices = select_gpu_devices(
            gpu_id_spec=gpu_ids,
            max_used_memory_mb=max_used_memory_mb,
            max_utilization=max_utilization,
        )
    if explicit_ids is None and str(gpu_ids).strip().lower() == "auto":
        discovered_devices = discover_gpu_devices()
        if discovered_devices and not selected_devices:
            details = "\n".join(f"  {line}" for line in format_gpu_devices(discovered_devices))
            raise SystemExit(
                "Adaptive GPU selection found no idle GPUs that satisfy the configured thresholds. "
                f"Thresholds: used<={max_used_memory_mb} MiB, util<={max_utilization}%. "
                "Refusing to grab a busy GPU.\n"
                f"{details}"
            )

    if selected_devices:
        worker_limit = parse_max_workers(max_workers, default=len(selected_devices))
        selected_devices = selected_devices[:worker_limit]
        selected_gpu_ids = [device.index for device in selected_devices]
        print("Adaptive GPU selection:")
        for line in format_gpu_devices(selected_devices):
            print(f"  {line}")
    else:
        selected_gpu_ids = []
        print("Adaptive GPU selection: nvidia-smi unavailable, falling back to sequential execution.")

    if sequential and len(selected_gpu_ids) > 1:
        print(
            "Sequential mode requested; using the first selected GPU only: "
            f"{selected_gpu_ids[0]}"
        )
        selected_gpu_ids = selected_gpu_ids[:1]

    if len(selected_gpu_ids) <= 1:
        env_overrides = {"CUDA_VISIBLE_DEVICES": str(selected_gpu_ids[0])} if selected_gpu_ids else None
        worker_label = f"gpu{selected_gpu_ids[0]}" if selected_gpu_ids else None
        return run_study(
            search_config_path=search_config_path,
            source_study_dir=source_study_dir,
            top_k=top_k,
            max_trials_override=max_trials_override,
            resume=resume,
            cleanup_stale_running_trials_on_resume=cleanup_stale_running_trials_on_resume,
            env_overrides=env_overrides,
            worker_label=worker_label,
        )

    return run_study_parallel(
        search_config_path=search_config_path,
        source_study_dir=source_study_dir,
        top_k=top_k,
        max_trials_override=max_trials_override,
        resume=resume,
        cleanup_stale_running_trials_on_resume=cleanup_stale_running_trials_on_resume,
        gpu_ids=selected_gpu_ids,
        worker_cooldown_seconds=worker_cooldown_seconds,
    )


def build_cli(default_config: str, description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--search-config",
        default=default_config,
        help="Path to the Optuna search YAML. Defaults to the bundled search config.",
    )
    parser.add_argument(
        "--source-study-dir",
        default=None,
        help="Optional existing study directory to seed the current study with top trials.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="How many top trials to enqueue from --source-study-dir.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help="Optional override for total study.n_trials target in the YAML.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing study in study_root instead of requiring a fresh study directory.",
    )
    parser.add_argument(
        "--skip-stale-running-cleanup",
        action="store_true",
        help="When resuming an active shared study, keep existing RUNNING trials instead of failing them as stale.",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Disable adaptive multi-GPU workers and run the study with a single worker.",
    )
    parser.add_argument(
        "--gpu-ids",
        default="auto",
        help="Comma-separated nvidia-smi GPU indexes to use, or 'auto' to select idle GPUs automatically.",
    )
    parser.add_argument(
        "--max-workers",
        default="auto",
        help="Maximum concurrent Optuna workers. Use an integer or 'auto'.",
    )
    parser.add_argument(
        "--max-used-memory-mb",
        type=int,
        default=1024,
        help="For --gpu-ids auto, only treat GPUs at or below this used-memory threshold as idle.",
    )
    parser.add_argument(
        "--max-utilization",
        type=int,
        default=20,
        help="For --gpu-ids auto, only treat GPUs at or below this utilization threshold as idle.",
    )
    parser.add_argument(
        "--worker-cooldown-seconds",
        type=float,
        default=0.0,
        help="Optional cooldown inserted after each worker finishes a trial.",
    )
    return parser.parse_args()

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import shlex
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
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
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


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
        value = get_nested(base_config, key)
        if value is None:
            continue
        params[key] = normalize_choice(value)
    return params


def build_env(study_cfg: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    for key, value in study_cfg.get("env", {}).items():
        env[str(key)] = str(value)
    return env


def detect_python_executable(candidates: list[str] | None = None) -> str:
    ordered_candidates: list[str] = [
        str(REPO_ROOT / ".venv" / "bin" / "python"),
        str(REPO_ROOT / ".venv" / "Scripts" / "python.exe"),
        sys.executable,
    ]
    if candidates:
        ordered_candidates.extend(str(item) for item in candidates)
    ordered_candidates.extend(["python3", "python"])

    seen: set[str] = set()
    for candidate in ordered_candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        candidate_to_run = candidate
        candidate_path = Path(candidate)
        looks_like_path = candidate_path.is_absolute() or any(token in candidate for token in ("/", "\\"))
        if looks_like_path and not candidate_path.is_absolute():
            candidate_path = (REPO_ROOT / candidate_path).resolve()
            candidate_to_run = str(candidate_path)
        if looks_like_path and not candidate_path.exists():
            continue
        try:
            subprocess.run(
                [candidate_to_run, "-c", "import sys; print(sys.executable)"],
                cwd=REPO_ROOT,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
                timeout=10,
            )
            return candidate_to_run
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            continue
    raise RuntimeError(
        "Could not find a usable Python command. Tried .venv/bin/python, "
        "the current interpreter, python3, and python."
    )


def import_optuna():
    try:
        import optuna  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - runtime guard
        message = (
            "Optuna is not installed in the training environment. "
            "Install it in the project environment before running this workflow, for example:\n"
            "  ./.venv/bin/python -m pip install optuna"
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


def build_study(optuna_module, search_cfg: dict[str, Any], study_root: Path):
    study_cfg = search_cfg["study"]
    study_root.mkdir(parents=True, exist_ok=True)
    storage_name = study_cfg.get("storage", "study.sqlite3")
    storage_path = study_root / str(storage_name)
    storage_url = f"sqlite:///{storage_path.as_posix()}"
    return optuna_module.create_study(
        study_name=str(study_cfg["name"]),
        direction=str(study_cfg.get("direction", "maximize")),
        storage=storage_url,
        load_if_exists=True,
        sampler=choose_sampler(optuna_module, study_cfg),
        pruner=choose_pruner(optuna_module, study_cfg),
    )


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
) -> TrialOutcome:
    study_cfg = search_cfg["study"]
    env = build_env(study_cfg)
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


def best_record(records: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    valid = [record for record in records if record.get("status") in VALID_TRIAL_STATUSES and record.get(key) is not None]
    if not valid:
        return None
    return max(valid, key=lambda item: item[key])


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
    }
    for record in records:
        status = str(record.get("status", "unknown"))
        payload["failure_counts"][status] = payload["failure_counts"].get(status, 0) + 1
    if best:
        payload["best_trial_number"] = best.get("trial_number")
        payload["best_val_accuracy"] = best.get("val_accuracy")
        payload["best_params"] = best.get("params", {})
    save_json(study_root / "study_status.json", payload)


def enqueue_template_trial(study, base_config: dict[str, Any], search_space: dict[str, Any]) -> None:
    params = current_template_params(base_config, search_space)
    if params:
        study.enqueue_trial(params)


def load_top_trial_params(source_study_dir: Path, top_k: int) -> list[dict[str, Any]]:
    records = load_trial_records(source_study_dir)
    valid = [record for record in records if record.get("status") in VALID_TRIAL_STATUSES and record.get("val_accuracy") is not None]
    valid.sort(key=lambda item: (item.get("val_accuracy", -math.inf), item.get("val_auc") or -math.inf), reverse=True)
    return [record.get("params", {}) for record in valid[:top_k]]


def enqueue_source_trials(study, source_study_dir: Path, top_k: int, search_space: dict[str, Any]) -> None:
    for params in load_top_trial_params(source_study_dir, top_k):
        filtered = {key: value for key, value in params.items() if key in search_space}
        if filtered:
            study.enqueue_trial(filtered)


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [record for record in records if record.get("status") in VALID_TRIAL_STATUSES and record.get("val_accuracy") is not None]
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
) -> Path:
    optuna = import_optuna()
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

    study_root = resolve_path(study_cfg["study_root"])
    study_root.mkdir(parents=True, exist_ok=True)
    (study_root / "trials").mkdir(parents=True, exist_ok=True)
    preflight_config = apply_overrides(base_config, search_cfg.get("fixed_overrides", {}))
    if bool(study_cfg.get("validate_dataset", True)):
        preflight_config = prepare_dataset_inputs(preflight_config, study_root)
        base_config = copy.deepcopy(base_config)
        base_config["data"]["csv_path"] = preflight_config["data"]["csv_path"]

    save_json(
        study_root / "search_config.snapshot.json",
        {
            "config_path": str(config_path),
            "search_config": search_cfg,
            "python_executable": python_executable,
        },
    )

    study = build_study(optuna, search_cfg, study_root)
    if bool(study_cfg.get("enqueue_current_template", True)):
        enqueue_template_trial(study, base_config, search_space)
    if source_study_dir and top_k > 0:
        enqueue_source_trials(study, resolve_path(source_study_dir), top_k, search_space)
    save_study_status(study_root, search_cfg, load_trial_records(study_root), state="running")

    def objective(trial) -> float:
        params = sample_trial_params(trial, search_space)
        trial_dir = study_root / "trials" / f"trial_{trial.number:04d}"
        outcome = run_single_trial(
            python_executable=python_executable,
            base_config=base_config,
            search_cfg=search_cfg,
            params=params,
            trial_dir=trial_dir,
            trial_number=trial.number,
        )
        trial.set_user_attr("status", outcome.status)
        trial.set_user_attr("params", outcome.params)
        for key in [
            "val_accuracy",
            "threshold_val_accuracy",
            "val_auc",
            "val_f1",
            "train_loss",
            "val_loss",
            "total_seconds",
            "peak_vram_mb",
            "no_miss_threshold",
        ]:
            trial.set_user_attr(key, getattr(outcome, key))
        if outcome.failure_reason:
            trial.set_user_attr("failure_reason", outcome.failure_reason)
        return float(outcome.objective)

    def callback(current_study, _trial) -> None:
        records = load_trial_records(study_root)
        save_study_status(study_root, search_cfg, records, state="running")

    optimize_kwargs: dict[str, Any] = {
        "n_trials": int(max_trials_override or study_cfg.get("n_trials", 10)),
        "callbacks": [callback],
    }
    if study_cfg.get("timeout_minutes") is not None:
        optimize_kwargs["timeout"] = int(float(study_cfg["timeout_minutes"]) * 60)
    try:
        study.optimize(objective, **optimize_kwargs)
    except KeyboardInterrupt:
        records = load_trial_records(study_root)
        save_study_status(study_root, search_cfg, records, state="interrupted")
        raise
    except Exception:
        records = load_trial_records(study_root)
        save_study_status(study_root, search_cfg, records, state="failed")
        raise

    records = load_trial_records(study_root)
    save_study_status(study_root, search_cfg, records, state="completed")
    summary = summarize_records(records)
    save_json(
        study_root / "study_summary.json",
        {
            "study_name": study_cfg["name"],
            "updated_at": now_iso(),
            "search_config": str(config_path),
            "source_study_dir": str(source_study_dir) if source_study_dir else None,
            "summary": summary,
        },
    )
    return study_root


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
        help="Optional override for study.n_trials in the YAML.",
    )
    return parser.parse_args()

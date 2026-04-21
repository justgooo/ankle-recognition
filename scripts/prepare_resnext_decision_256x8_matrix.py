from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from optuna_workflow import ensure_running_in_project_python
from script_runtime import REPO_ROOT, read_yaml, repo_relative


BASE_CONFIGS = {
    "formal": REPO_ROOT / "configs/autoresearch_formal_resnext_decision_256x8.yaml",
    "proxy": REPO_ROOT / "configs/autoresearch_proxy_resnext_decision_256x8.yaml",
}

LANE_SPECS: dict[str, dict[str, Any]] = {
    "l0_equal": {
        "matrix_id": "L0-equal",
        "description": "Matched equal-weight control.",
        "config_overrides": {"model.equal_weight_fusion": True},
        "runtime_env": {},
    },
    "l1_learned": {
        "matrix_id": "L1-learned",
        "description": "Legacy-style learned-weighting anchor.",
        "config_overrides": {},
        "runtime_env": {},
    },
    "l2_minimal": {
        "matrix_id": "L2-minimal",
        "description": "Minimal learned baseline with raw confidence heads.",
        "config_overrides": {"model.minimal_fusion_baseline": True},
        "runtime_env": {},
    },
    "l3_no_mixer": {
        "matrix_id": "L3-no-mixer",
        "description": "Disable cross-view mixer while keeping calibrator.",
        "config_overrides": {},
        "runtime_env": {"ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER": "1"},
    },
    "l4_no_calibrator": {
        "matrix_id": "L4-no-calibrator",
        "description": "Disable shared calibrator while keeping cross-view mixer.",
        "config_overrides": {},
        "runtime_env": {"ANKLE_DISABLE_FUSION_CALIBRATOR": "1"},
    },
    "l5_temp1p5": {
        "matrix_id": "L5-temp1p5",
        "description": "Global fusion temperature 1.5.",
        "config_overrides": {},
        "runtime_env": {"ANKLE_LEARNED_FUSION_TEMPERATURE": "1.5"},
    },
    "l5_temp2p0": {
        "matrix_id": "L5-temp2p0",
        "description": "Global fusion temperature 2.0.",
        "config_overrides": {},
        "runtime_env": {"ANKLE_LEARNED_FUSION_TEMPERATURE": "2.0"},
    },
}

DEFAULT_SEEDS = [42, 123, 456]


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


def dump_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)


def normalize_output_root(path_like: str) -> Path:
    path = Path(path_like)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def build_config(
    *,
    phase: str,
    lane_key: str,
    seed: int,
    output_root: Path,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    base_path = BASE_CONFIGS[phase]
    base_config = read_yaml(base_path)
    lane_spec = LANE_SPECS[lane_key]

    config = apply_overrides(base_config, lane_spec["config_overrides"])
    runtime_env = dict(lane_spec["runtime_env"])
    if runtime_env:
        config["runtime_env"] = runtime_env
    else:
        config.pop("runtime_env", None)

    stem = f"cmp_resnext_decision_256x8_{lane_key}_{phase}_s{seed}"
    config["seed"] = int(seed)
    config["experiment_name"] = stem
    config["output_dir"] = f"runs/resnext_decision_256x8_matrix/{phase}/{lane_key}/s{seed}"

    config_path = output_root / phase / f"{stem}.yaml"
    record = {
        "phase": phase,
        "lane_key": lane_key,
        "matrix_id": lane_spec["matrix_id"],
        "description": lane_spec["description"],
        "seed": int(seed),
        "config_path": repo_relative(config_path),
        "base_config": repo_relative(base_path),
        "output_dir": config["output_dir"],
        "runtime_env": runtime_env,
        "launch_command": (
            f".venv/bin/python scripts/run_train_with_config_env.py --config "
            f"{repo_relative(config_path)}"
        ),
    }
    return config, config_path, record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize the ResNeXt decision 256x8 module matrix into runnable YAML configs."
    )
    parser.add_argument(
        "--phase",
        nargs="+",
        choices=sorted(BASE_CONFIGS),
        default=["formal"],
        help="Training phases to materialize.",
    )
    parser.add_argument(
        "--lane",
        nargs="+",
        choices=sorted(LANE_SPECS),
        default=sorted(LANE_SPECS),
        help="Matrix lanes to materialize.",
    )
    parser.add_argument(
        "--seed",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help="Seeds to materialize.",
    )
    parser.add_argument(
        "--output-root",
        default="configs/generated_resnext_decision_256x8_matrix",
        help="Directory where generated YAML configs and the manifest will be written.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the configs that would be written without touching the filesystem.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_running_in_project_python(sys.executable)
    output_root = normalize_output_root(args.output_root)

    records: list[dict[str, Any]] = []
    for phase in args.phase:
        for lane_key in args.lane:
            for seed in args.seed:
                config, config_path, record = build_config(
                    phase=phase,
                    lane_key=lane_key,
                    seed=seed,
                    output_root=output_root,
                )
                records.append(record)
                print(
                    f"{'DRY' if args.dry_run else 'WRITE'} "
                    f"{repo_relative(config_path)} | output_dir={record['output_dir']} | "
                    f"runtime_env={json.dumps(record['runtime_env'], ensure_ascii=False, sort_keys=True)}"
                )
                if not args.dry_run:
                    dump_yaml(config_path, config)

    if args.dry_run:
        return 0

    manifest = {
        "generated_from": "scripts/prepare_resnext_decision_256x8_matrix.py",
        "output_root": repo_relative(output_root),
        "records": records,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    print(f"WROTE manifest {repo_relative(manifest_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from optuna_workflow import ensure_running_in_project_python
from script_runtime import (
    REPO_ROOT,
    detect_python_executable,
    read_output_dir,
    read_yaml,
    repo_relative,
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_repo_path(path_like: str) -> Path:
    path = Path(path_like)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def normalize_runtime_env(raw: Any) -> dict[str, str]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise SystemExit(f"config.runtime_env must be a mapping, got {type(raw).__name__}.")
    return {str(key): str(value) for key, value in raw.items()}


def save_launcher_manifest(
    output_dir: Path,
    config_path: Path,
    python_executable: str,
    runtime_env: dict[str, str],
    threshold: bool,
) -> None:
    payload = {
        "launched_at": now_iso(),
        "config_path": str(config_path),
        "python_executable": python_executable,
        "runtime_env": runtime_env,
        "train_command": [
            python_executable,
            "train.py",
            "--config",
            repo_relative(config_path),
        ],
        "threshold_enabled": threshold,
    }
    if threshold:
        payload["threshold_command"] = [
            python_executable,
            "tools/evaluate_threshold.py",
            "--run_dir",
            repo_relative(output_dir),
            "--config",
            repo_relative(config_path),
        ]
    manifest_path = output_dir / "launcher_runtime_env.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch train.py using runtime_env declared inside the YAML config."
    )
    parser.add_argument("--config", required=True, help="Path to the training YAML config.")
    parser.add_argument(
        "--python",
        default=None,
        help="Python executable to use. Defaults to the repo .venv interpreter.",
    )
    parser.add_argument(
        "--threshold",
        action="store_true",
        help="Run tools/evaluate_threshold.py after a successful training run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved command/runtime_env without launching training.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = resolve_repo_path(args.config)
    config = read_yaml(config_path)
    runtime_env = normalize_runtime_env(config.get("runtime_env"))
    python_executable = args.python or detect_python_executable()
    ensure_running_in_project_python(python_executable or sys.executable)
    output_dir = read_output_dir(config_path)

    train_command = [
        python_executable,
        "train.py",
        "--config",
        repo_relative(config_path),
    ]
    threshold_command = [
        python_executable,
        "tools/evaluate_threshold.py",
        "--run_dir",
        repo_relative(output_dir),
        "--config",
        repo_relative(config_path),
    ]

    print(f"config={repo_relative(config_path)}")
    print(f"output_dir={repo_relative(output_dir)}")
    print(f"python={python_executable}")
    print(f"runtime_env={json.dumps(runtime_env, ensure_ascii=False, sort_keys=True)}")
    print("train_command=" + " ".join(train_command))
    if args.threshold:
        print("threshold_command=" + " ".join(threshold_command))

    if args.dry_run:
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    save_launcher_manifest(
        output_dir=output_dir,
        config_path=config_path,
        python_executable=python_executable,
        runtime_env=runtime_env,
        threshold=args.threshold,
    )

    env = os.environ.copy()
    env.update(runtime_env)

    train_result = subprocess.run(train_command, cwd=REPO_ROOT, env=env, check=False)
    if train_result.returncode != 0:
        return train_result.returncode
    if not args.threshold:
        return 0

    threshold_result = subprocess.run(
        threshold_command,
        cwd=REPO_ROOT,
        env=env,
        check=False,
    )
    return threshold_result.returncode


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from script_runtime import detect_python_executable, read_output_dir, repo_relative, tail_lines


REPO_ROOT = SCRIPT_DIR.parent
FORMAL_CONFIGS = [
    "configs/cmp_fair_v100_decision_formal_resunet.yaml",
    "configs/cmp_fair_v100_decision_formal_resnext.yaml",
    "configs/cmp_fair_v100_decision_formal_senet.yaml",
    "configs/cmp_fair_v100_decision_formal_cspnet.yaml",
]


def build_runtime_env(gpu_id: int) -> dict[str, str]:
    env = os.environ.copy()
    cache_root = REPO_ROOT / ".torch-cache"
    torch_home = cache_root / "torch"
    xdg_cache_home = cache_root / "xdg"
    hf_home = cache_root / "hf"
    hf_hub_cache = hf_home / "hub"

    for path in (torch_home, xdg_cache_home, hf_home, hf_hub_cache):
        path.mkdir(parents=True, exist_ok=True)

    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["TORCH_HOME"] = str(torch_home)
    env["XDG_CACHE_HOME"] = str(xdg_cache_home)
    env["HF_HOME"] = str(hf_home)
    env["HUGGINGFACE_HUB_CACHE"] = str(hf_hub_cache)
    return env


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dry-run or execute the fair decision-fusion backbone formal comparison on one or more GPUs."
    )
    parser.add_argument("--gpu-id", type=int, default=1, help="CUDA_VISIBLE_DEVICES value to use.")
    parser.add_argument(
        "--gpu-ids",
        nargs="+",
        type=int,
        help="Optional list of GPU ids to use concurrently, e.g. --gpu-ids 0 1.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=10800,
        help="Timeout per backbone in seconds. Default matches the 180-minute formal budget.",
    )
    parser.add_argument(
        "--log-dir",
        default="autoresearch_logs",
        help="Directory for per-backbone training logs.",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip scripts/backbone_preflight.py before training.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually run preflight and training. Without this flag the script only prints the plan.",
    )
    return parser


def resolve_gpu_ids(args: argparse.Namespace) -> list[int]:
    if args.gpu_ids:
        return [int(gpu_id) for gpu_id in args.gpu_ids]
    return [int(args.gpu_id)]


def print_plan(python_executable: str, gpu_ids: list[int], timeout_seconds: int, log_dir: Path) -> None:
    cache_root = REPO_ROOT / ".torch-cache"
    print("")
    print("============================================================")
    print(" Fair Backbone Formal Compare (Decision Fusion)")
    print(f" Python:   {python_executable}")
    print(f" GPUs:     {', '.join(str(gpu_id) for gpu_id in gpu_ids)}")
    print(f" Cache:    {repo_relative(cache_root)}")
    print(f" Timeout:  {timeout_seconds}s per backbone")
    print(f" Log dir:  {repo_relative(log_dir)}")
    print(" Configs:")
    for config_rel in FORMAL_CONFIGS:
        print(f"   - {config_rel}")
    print("============================================================")
    print("")
    print("Preflight command:")
    print(
        "  CUDA_VISIBLE_DEVICES="
        f"{gpu_ids[0]} TORCH_HOME=.torch-cache/torch HF_HOME=.torch-cache/hf "
        f"XDG_CACHE_HOME=.torch-cache/xdg HUGGINGFACE_HUB_CACHE=.torch-cache/hf/hub "
        f"{python_executable} scripts/backbone_preflight.py --config "
        + " ".join(FORMAL_CONFIGS)
    )
    print("")
    print("Training pool:")
    print(f"  up to {min(len(gpu_ids), len(FORMAL_CONFIGS))} concurrent jobs")
    print(f"  dynamic dispatch across GPUs: {', '.join(str(gpu_id) for gpu_id in gpu_ids)}")
    for config_rel in FORMAL_CONFIGS:
        print(
            "  TORCH_HOME=.torch-cache/torch HF_HOME=.torch-cache/hf "
            f"XDG_CACHE_HOME=.torch-cache/xdg HUGGINGFACE_HUB_CACHE=.torch-cache/hf/hub "
            f"timeout {timeout_seconds} {python_executable} train.py --config {config_rel}"
        )
    print("")


def run_preflight(*, python_executable: str, gpu_id: int) -> None:
    command = [
        python_executable,
        "scripts/backbone_preflight.py",
        "--config",
        *FORMAL_CONFIGS,
    ]
    env = build_runtime_env(gpu_id)
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


def read_summary_line(config_path: Path) -> str:
    summary_path = read_output_dir(config_path) / "summary.json"
    if not summary_path.exists():
        return "summary missing"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    best_val = summary.get("best_val") or {}
    runtime = summary.get("runtime") or {}
    return (
        "val_acc="
        f"{best_val.get('accuracy', float('nan')):.6f}, "
        "val_auc="
        f"{best_val.get('auc', float('nan')):.6f}, "
        "val_f1="
        f"{best_val.get('f1', float('nan')):.6f}, "
        "peak_vram_mb="
        f"{runtime.get('peak_vram_mb', float('nan')):.1f}, "
        "total_seconds="
        f"{runtime.get('total_seconds', float('nan')):.1f}"
    )


def launch_training_job(
    *,
    python_executable: str,
    gpu_id: int,
    timeout_seconds: int,
    log_dir: Path,
    timestamp: str,
    config_rel: str,
) -> tuple[subprocess.Popen[bytes], Path, Path, object]:
    config_path = REPO_ROOT / config_rel
    log_path = log_dir / f"{config_path.stem}_gpu{gpu_id}_{timestamp}.log"
    command = [
        "timeout",
        str(timeout_seconds),
        python_executable,
        "train.py",
        "--config",
        config_rel,
    ]
    env = build_runtime_env(gpu_id)
    handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
    )
    return process, log_path, config_path, handle


def report_finished_job(
    *,
    config_path: Path,
    gpu_id: int,
    log_path: Path,
    returncode: int,
    timeout_seconds: int,
) -> int:
    if returncode == 0:
        print(f"  GPU {gpu_id}: success | {read_summary_line(config_path)}")
        exit_code = 0
    elif returncode == 124:
        print(f"  GPU {gpu_id}: timeout after {timeout_seconds}s")
        exit_code = 1
    else:
        print(f"  GPU {gpu_id}: failed with exit code {returncode}")
        exit_code = 1

    for line in tail_lines(log_path, lines=5):
        print(f"    {line}")
    return exit_code


def run_training(
    *,
    python_executable: str,
    gpu_ids: list[int],
    timeout_seconds: int,
    log_dir: Path,
) -> int:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pending = list(FORMAL_CONFIGS)
    running: dict[int, tuple[subprocess.Popen[bytes], Path, Path, object]] = {}
    overall_exit = 0
    while pending or running:
        for gpu_id in gpu_ids:
            if not pending or gpu_id in running:
                continue
            config_rel = pending.pop(0)
            process, log_path, config_path, handle = launch_training_job(
                python_executable=python_executable,
                gpu_id=gpu_id,
                timeout_seconds=timeout_seconds,
                log_dir=log_dir,
                timestamp=timestamp,
                config_rel=config_rel,
            )
            running[gpu_id] = (process, log_path, config_path, handle)
            print(f"Running {config_rel}")
            print(f"  GPU {gpu_id} | PID {process.pid} | log: {repo_relative(log_path)}")

        if not running:
            continue

        finished_gpu_ids: list[int] = []
        for gpu_id, (process, log_path, config_path, handle) in running.items():
            returncode = process.poll()
            if returncode is None:
                continue
            handle.close()
            overall_exit = max(
                overall_exit,
                report_finished_job(
                    config_path=config_path,
                    gpu_id=gpu_id,
                    log_path=log_path,
                    returncode=returncode,
                    timeout_seconds=timeout_seconds,
                ),
            )
            finished_gpu_ids.append(gpu_id)

        for gpu_id in finished_gpu_ids:
            del running[gpu_id]

        if running:
            time.sleep(5)

    return overall_exit


def main() -> None:
    args = build_parser().parse_args()
    python_executable = detect_python_executable()
    gpu_ids = resolve_gpu_ids(args)
    if shutil.which("timeout") is None:
        raise SystemExit("The `timeout` command is required.")

    log_dir = REPO_ROOT / args.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    print_plan(
        python_executable=python_executable,
        gpu_ids=gpu_ids,
        timeout_seconds=args.timeout,
        log_dir=log_dir,
    )
    if not args.execute:
        print("Dry-run only. Re-run with --execute to start training.")
        return

    if not args.skip_preflight:
        print("Running backbone preflight...")
        run_preflight(python_executable=python_executable, gpu_id=gpu_ids[0])
        print("")

    raise SystemExit(
        run_training(
            python_executable=python_executable,
            gpu_ids=gpu_ids,
            timeout_seconds=args.timeout,
            log_dir=log_dir,
        )
    )


if __name__ == "__main__":
    main()

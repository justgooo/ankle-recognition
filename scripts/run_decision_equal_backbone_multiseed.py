#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
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
COMPARE_CONFIGS = [
    "configs/cmp_decision_equal_cspnet_learned_512x16_e20_s42.yaml",
    "configs/cmp_decision_equal_cspnet_equal_512x16_e20_s42.yaml",
]
ROUND1_MAINLINE_CONFIGS = [
    "configs/cmp_backbone_decision_resnext_512x16_e20_s42.yaml",
    "configs/cmp_backbone_decision_cspnet_512x16_e20_s42.yaml",
]
ROUND2_MAINLINE_CONFIGS = [
    "configs/cmp_backbone_decision_resnext_512x16_e20_s123.yaml",
    "configs/cmp_backbone_decision_resnext_512x16_e20_s456.yaml",
    "configs/cmp_backbone_decision_cspnet_512x16_e20_s123.yaml",
    "configs/cmp_backbone_decision_cspnet_512x16_e20_s456.yaml",
]
ALL_CONFIGS = COMPARE_CONFIGS + ROUND1_MAINLINE_CONFIGS + ROUND2_MAINLINE_CONFIGS


def build_runtime_env() -> dict[str, str]:
    env = os.environ.copy()
    cache_root = REPO_ROOT / ".torch-cache"
    torch_home = cache_root / "torch"
    xdg_cache_home = cache_root / "xdg"
    hf_home = cache_root / "hf"
    hf_hub_cache = hf_home / "hub"

    for path in (torch_home, xdg_cache_home, hf_home, hf_hub_cache):
        path.mkdir(parents=True, exist_ok=True)

    env["TORCH_HOME"] = str(torch_home)
    env["XDG_CACHE_HOME"] = str(xdg_cache_home)
    env["HF_HOME"] = str(hf_home)
    env["HUGGINGFACE_HUB_CACHE"] = str(hf_hub_cache)
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the cspnet learned-vs-equal decision-fusion control pair plus the "
            "matched resnext/cspnet three-seed backbone final in two phases."
        )
    )
    parser.add_argument(
        "--cpus-per-run",
        type=int,
        default=8,
        help="Slurm CPUs to reserve per training sub-step.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=14400,
        help="Timeout per run in seconds. Default: 4 hours.",
    )
    parser.add_argument(
        "--log-dir",
        default="autoresearch_logs/decision_equal_backbone_multiseed",
        help="Directory for aggregated logs and result JSON.",
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=40.0,
        help="Abort early if any visible GPU has less free memory than this threshold.",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip scripts/backbone_preflight.py.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually run the campaign. Without this flag, only print the plan.",
    )
    return parser


def print_plan(python_executable: str, args: argparse.Namespace, log_dir: Path) -> None:
    print("")
    print("============================================================")
    print(" Decision vs Equal-Weight Control + Backbone Multiseed Final")
    print(f" Python:        {python_executable}")
    print(f" Log dir:       {repo_relative(log_dir)}")
    print(f" CPUs / run:    {args.cpus_per_run}")
    print(f" Timeout:       {args.timeout}s")
    print(f" Min free VRAM: {args.min_free_gb:.1f} GiB")
    print(" Phase 1:")
    for config_rel in COMPARE_CONFIGS + ROUND1_MAINLINE_CONFIGS:
        print(f"   - {config_rel}")
    print(" Phase 2:")
    for config_rel in ROUND2_MAINLINE_CONFIGS:
        print(f"   - {config_rel}")
    print("============================================================")
    print("")


def ensure_visible_gpus(*, python_executable: str, env: dict[str, str], min_free_gb: float) -> None:
    command = [
        python_executable,
        "-c",
        (
            "import sys, torch; "
            "count=torch.cuda.device_count(); "
            "print(f'visible_gpus={count}'); "
            "assert count >= 4, f'Need at least 4 visible GPUs, got {count}'; "
            "free_gb=[]; "
            "[free_gb.append(torch.cuda.mem_get_info(i)[0]/1024**3) or print(f'gpu{i}_free_gb={free_gb[-1]:.2f}') for i in range(count)]; "
            f"sys.exit(0 if min(free_gb) >= {min_free_gb} else 2)"
        ),
    ]
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    if result.returncode == 2:
        raise SystemExit(
            f"At least one visible GPU has < {min_free_gb:.1f} GiB free memory; aborting batch."
        )
    if result.returncode != 0:
        raise SystemExit("Failed to verify visible GPU mapping and free memory.")


def run_preflight(*, python_executable: str, env: dict[str, str]) -> None:
    command = [
        python_executable,
        "scripts/backbone_preflight.py",
        "--config",
        *ALL_CONFIGS,
    ]
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


def launch_training_job(
    *,
    python_executable: str,
    env: dict[str, str],
    config_rel: str,
    cpus_per_run: int,
    timeout_seconds: int,
    log_dir: Path,
    timestamp: str,
    phase_name: str,
) -> tuple[subprocess.Popen[bytes], Path, Path, object]:
    config_path = REPO_ROOT / config_rel
    log_path = log_dir / f"{phase_name}_{config_path.stem}_{timestamp}.log"
    command = [
        "srun",
        "--exclusive",
        "-N",
        "1",
        "-n",
        "1",
        "--gres=gpu:1",
        "-c",
        str(cpus_per_run),
        "timeout",
        str(timeout_seconds),
        python_executable,
        "train.py",
        "--config",
        config_rel,
    ]
    handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
    )
    return process, log_path, config_path, handle


def read_summary(config_path: Path) -> dict[str, float | str]:
    summary_path = read_output_dir(config_path) / "summary.json"
    if not summary_path.exists():
        return {
            "status": "missing_summary",
            "val_acc": float("nan"),
            "val_auc": float("nan"),
            "val_f1": float("nan"),
            "peak_vram_mb": float("nan"),
            "total_seconds": float("nan"),
        }
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    best_val = summary.get("best_val") or {}
    runtime = summary.get("runtime") or {}
    return {
        "status": "ok",
        "val_acc": float(best_val.get("accuracy", float("nan"))),
        "val_auc": float(best_val.get("auc", float("nan"))),
        "val_f1": float(best_val.get("f1", float("nan"))),
        "peak_vram_mb": float(runtime.get("peak_vram_mb", float("nan"))),
        "total_seconds": float(runtime.get("total_seconds", float("nan"))),
    }


def report_finished_job(
    *,
    config_path: Path,
    log_path: Path,
    returncode: int,
    timeout_seconds: int,
) -> dict[str, float | int | str]:
    result = read_summary(config_path)
    if returncode == 0:
        print(
            "  success | "
            f"{config_path.name} | "
            f"val_acc={result['val_acc']:.6f} | "
            f"val_auc={result['val_auc']:.6f} | "
            f"val_f1={result['val_f1']:.6f} | "
            f"peak_vram_mb={result['peak_vram_mb']:.1f} | "
            f"total_seconds={result['total_seconds']:.1f}"
        )
    elif returncode == 124:
        print(f"  timeout | {config_path.name} | after {timeout_seconds}s")
    else:
        print(f"  failed | {config_path.name} | exit code {returncode}")

    for line in tail_lines(log_path, lines=5):
        print(f"    {line}")
    result["returncode"] = int(returncode)
    result["log_path"] = repo_relative(log_path)
    return result


def run_phase(
    *,
    phase_name: str,
    configs: list[str],
    python_executable: str,
    env: dict[str, str],
    cpus_per_run: int,
    timeout_seconds: int,
    log_dir: Path,
    aggregate: dict[str, dict[str, float | int | str]],
) -> int:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("")
    print(f"=== {phase_name} ===")
    active: dict[str, tuple[subprocess.Popen[bytes], Path, Path, object]] = {}
    phase_exit_code = 0

    for config_rel in configs:
        print(f"Launching {config_rel}")
        active[config_rel] = launch_training_job(
            python_executable=python_executable,
            env=env,
            config_rel=config_rel,
            cpus_per_run=cpus_per_run,
            timeout_seconds=timeout_seconds,
            log_dir=log_dir,
            timestamp=timestamp,
            phase_name=phase_name,
        )

    while active:
        for config_rel, (process, log_path, config_path, handle) in list(active.items()):
            returncode = process.poll()
            if returncode is None:
                continue
            handle.close()
            aggregate[config_rel] = report_finished_job(
                config_path=config_path,
                log_path=log_path,
                returncode=returncode,
                timeout_seconds=timeout_seconds,
            )
            if returncode != 0:
                phase_exit_code = 1
            active.pop(config_rel)
        if active:
            time.sleep(10)

    return phase_exit_code


def write_aggregate_results(
    *,
    log_dir: Path,
    aggregate: dict[str, dict[str, float | int | str]],
) -> Path:
    result_path = log_dir / "aggregate_results.json"
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "results": aggregate,
    }
    result_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result_path


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    python_executable = detect_python_executable([".venv/bin/python"])
    log_dir = REPO_ROOT / args.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    env = build_runtime_env()

    print_plan(python_executable, args, log_dir)
    if not args.execute:
        return 0

    ensure_visible_gpus(
        python_executable=python_executable,
        env=env,
        min_free_gb=args.min_free_gb,
    )
    if not args.skip_preflight:
        run_preflight(python_executable=python_executable, env=env)

    aggregate: dict[str, dict[str, float | int | str]] = {}
    overall_exit_code = 0
    overall_exit_code |= run_phase(
        phase_name="phase1",
        configs=COMPARE_CONFIGS + ROUND1_MAINLINE_CONFIGS,
        python_executable=python_executable,
        env=env,
        cpus_per_run=args.cpus_per_run,
        timeout_seconds=args.timeout,
        log_dir=log_dir,
        aggregate=aggregate,
    )
    overall_exit_code |= run_phase(
        phase_name="phase2",
        configs=ROUND2_MAINLINE_CONFIGS,
        python_executable=python_executable,
        env=env,
        cpus_per_run=args.cpus_per_run,
        timeout_seconds=args.timeout,
        log_dir=log_dir,
        aggregate=aggregate,
    )
    result_path = write_aggregate_results(log_dir=log_dir, aggregate=aggregate)
    print("")
    print(f"Aggregate results written to {repo_relative(result_path)}")
    return overall_exit_code


if __name__ == "__main__":
    raise SystemExit(main())

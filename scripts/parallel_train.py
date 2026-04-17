from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from script_runtime import (
    REPO_ROOT,
    detect_python_executable,
    disk_free_gb,
    query_nvidia_smi,
    read_output_dir,
    remove_nonbest_checkpoints,
    repo_relative,
    tail_lines,
)


PID_DIR = Path(tempfile.gettempdir()) / "ankle_parallel"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch training on both GPUs in parallel.")
    parser.add_argument(
        "--slot0-config",
        default="configs/autoresearch_proxy_slot0.yaml",
        help="Config path for slot 0 / GPU 0.",
    )
    parser.add_argument(
        "--slot1-config",
        default="configs/autoresearch_proxy.yaml",
        help="Config path for slot 1 / GPU 1.",
    )
    parser.add_argument("--slot0-only", action="store_true", help="Only launch slot 0.")
    parser.add_argument("--slot1-only", action="store_true", help="Only launch slot 1.")
    parser.add_argument("--timeout", type=int, default=3600, help="Timeout for each slot in seconds.")
    parser.add_argument("--log-dir", default="autoresearch_logs", help="Directory for training logs.")
    return parser


def kill_process(proc: subprocess.Popen[bytes] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def launch_slot(
    *,
    slot_id: int,
    gpu_id: int,
    config_path: Path,
    timeout_seconds: int,
    log_dir: Path,
    python_executable: str,
    timestamp: str,
) -> tuple[subprocess.Popen[bytes], Path, Path]:
    log_path = log_dir / f"slot{slot_id}_{timestamp}.log"
    pid_path = PID_DIR / f"slot{slot_id}.pid"

    print(f"  Slot {slot_id}: GPU={gpu_id}, config={repo_relative(config_path)}")
    print(f"  Slot {slot_id}: log -> {repo_relative(log_path)}")

    run_dir = read_output_dir(config_path)
    remove_nonbest_checkpoints(run_dir)

    command = [
        "timeout",
        str(timeout_seconds),
        python_executable,
        "train.py",
        "--config",
        repo_relative(config_path),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    process.log_handle = handle  # type: ignore[attr-defined]
    pid_path.write_text(str(process.pid), encoding="utf-8")
    print(f"  Slot {slot_id}: PID={process.pid}")
    return process, log_path, pid_path


def wait_for_slot(
    *,
    slot_id: int,
    process: subprocess.Popen[bytes],
    log_path: Path,
    pid_path: Path,
    timeout_seconds: int,
) -> int:
    exit_code = process.wait()
    handle = getattr(process, "log_handle", None)
    if handle is not None:
        handle.close()
    if pid_path.exists():
        pid_path.unlink()

    if exit_code == 0:
        print(f"  Slot {slot_id}: completed successfully")
    elif exit_code == 124:
        print(f"  Slot {slot_id}: TIMEOUT after {timeout_seconds}s")
    else:
        print(f"  Slot {slot_id}: failed (exit code {exit_code})")

    lines = tail_lines(log_path, lines=5)
    if lines:
        print(f"  Slot {slot_id}: last 5 log lines:")
        for line in lines:
            print(f"    {line}")
    return exit_code


def main() -> None:
    args = build_parser().parse_args()
    if args.slot0_only and args.slot1_only:
        raise SystemExit("Choose at most one of --slot0-only / --slot1-only.")

    slot0_enabled = not args.slot1_only
    slot1_enabled = not args.slot0_only

    python_executable = detect_python_executable()
    if shutil.which("timeout") is None:
        raise SystemExit("The `timeout` command is required for scripts/parallel_train.py.")

    log_dir = REPO_ROOT / args.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    PID_DIR.mkdir(parents=True, exist_ok=True)

    free_gb = disk_free_gb(REPO_ROOT)
    print("")
    print("============================================================")
    print(" Parallel Training - Ankle CT Classifier")
    print(f" Timestamp:  {datetime.now().strftime('%Y%m%d_%H%M%S')}")
    print(f" Slot 0:     GPU 0 (slot0) - {slot0_enabled}")
    print(f" Slot 1:     GPU 1 (slot1) - {slot1_enabled}")
    print(f" Timeout:    {args.timeout}s per slot")
    print(f" Python:     {python_executable}")
    print(f" Workdir:    {REPO_ROOT}")
    print("============================================================")
    print("")
    print(f"Disk free: {free_gb} GB")
    if free_gb < 10:
        raise SystemExit("Less than 10 GB free disk space. Aborting.")

    gpu_lines = query_nvidia_smi("index,name,memory.used,memory.total,utilization.gpu")
    if gpu_lines:
        print("GPU status before training:")
        for line in gpu_lines:
            print(f"  {line}")
        print("")

    remove_nonbest_checkpoints(REPO_ROOT / "runs")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    launched: dict[int, tuple[subprocess.Popen[bytes], Path, Path]] = {}

    def handle_signal(signum: int, _frame: object) -> None:
        print("")
        print(f"Caught signal {signum} - cleaning up...")
        for process, _log_path, pid_path in launched.values():
            kill_process(process)
            if pid_path.exists():
                pid_path.unlink()
        raise SystemExit(1)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    print("Launching training processes...")
    if slot0_enabled:
        launched[0] = launch_slot(
            slot_id=0,
            gpu_id=0,
            config_path=REPO_ROOT / args.slot0_config,
            timeout_seconds=args.timeout,
            log_dir=log_dir,
            python_executable=python_executable,
            timestamp=timestamp,
        )
    if slot1_enabled:
        launched[1] = launch_slot(
            slot_id=1,
            gpu_id=1,
            config_path=REPO_ROOT / args.slot1_config,
            timeout_seconds=args.timeout,
            log_dir=log_dir,
            python_executable=python_executable,
            timestamp=timestamp,
        )
    print("")
    print("Waiting for training processes to finish...")
    print("(Press Ctrl+C to abort both)")
    print("")

    exit_codes = {0: 0, 1: 0}
    for slot_id in (0, 1):
        if slot_id not in launched:
            continue
        process, log_path, pid_path = launched[slot_id]
        exit_codes[slot_id] = wait_for_slot(
            slot_id=slot_id,
            process=process,
            log_path=log_path,
            pid_path=pid_path,
            timeout_seconds=args.timeout,
        )

    gpu_lines = query_nvidia_smi("index,name,memory.used,memory.total,utilization.gpu")
    if gpu_lines:
        print("")
        print("GPU status after training:")
        for line in gpu_lines:
            print(f"  {line}")

    print("")
    print("============================================================")
    print(" Results")
    print(f"  Slot 0: exit={exit_codes[0]}")
    print(f"  Slot 1: exit={exit_codes[1]}")
    print("============================================================")
    if any(code != 0 for code in exit_codes.values() if code is not None):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

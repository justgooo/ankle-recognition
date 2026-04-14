from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path

from script_runtime import REPO_ROOT, command_output, process_alive, process_info, query_nvidia_smi, tail_lines


PID_DIR = Path(tempfile.gettempdir()) / "ankle_parallel"
LOG_DIR = REPO_ROOT / "autoresearch_logs"


def latest_log_for_slot(slot_id: int) -> Path | None:
    candidates = sorted(LOG_DIR.glob(f"slot{slot_id}_*.log"), key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def main() -> None:
    print("")
    print("============================================================")
    print(f" Parallel Training Status - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("============================================================")
    print("")

    print("=== GPU Status ===")
    gpu_lines = query_nvidia_smi("index,name,memory.used,memory.total,utilization.gpu,temperature.gpu")
    if gpu_lines:
        for line in gpu_lines:
            print(f"  {line}")
    else:
        print("  nvidia-smi unavailable")
    print("")

    for slot_id in (0, 1):
        pid_path = PID_DIR / f"slot{slot_id}.pid"
        print(f"--- Slot {slot_id} ---")
        if pid_path.exists():
            pid = int(pid_path.read_text(encoding="utf-8").strip())
            if process_alive(pid):
                print(f"  Status: RUNNING (PID={pid})")
                info = process_info(pid)
                if info:
                    print(f"  Process: {info}")
            else:
                print(f"  Status: DEAD (PID={pid} no longer running)")
        else:
            print("  Status: NOT STARTED (no PID file)")

        latest_log = latest_log_for_slot(slot_id)
        if latest_log is not None:
            size_kb = latest_log.stat().st_size / 1024
            print(f"  Latest log: {latest_log.name} ({size_kb:.1f} KB)")
            print("  Last 3 lines:")
            for line in tail_lines(latest_log, lines=3):
                print(f"    {line}")
        else:
            print("  No log files found")
        print("")

    print("=== System Resources ===")
    uptime_lines = command_output(["uptime"])
    if uptime_lines:
        print(f"  Uptime: {uptime_lines[0]}")
    print("  Memory:")
    memory_lines = command_output(["free", "-h"])
    if memory_lines:
        for line in memory_lines:
            if line.startswith(("Mem", "Swap")):
                print(f"    {line}")
    print("")

    print("  Disk:")
    disk_lines = command_output(["df", "-h", str(REPO_ROOT)])
    for line in disk_lines:
        print(f"    {line}")
    print("")
    print("============================================================")


if __name__ == "__main__":
    main()

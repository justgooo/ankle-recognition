from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from script_runtime import detect_python_executable, disk_free_gb, query_nvidia_smi, remove_nonbest_checkpoints  # noqa: E402


PID_DIR = Path(tempfile.gettempdir()) / "ankle_parallel"


def repo_python_display(python_executable: str) -> str:
    candidate = Path(python_executable)
    if candidate.is_absolute():
        try:
            return candidate.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            return str(candidate)
    return python_executable


def generate_prompt(slot_id: int, gpu_id: int, gpu_name: str, proxy_config: str, formal_config: str, python_cmd: str) -> str:
    return f"""You are an autonomous ML researcher. Follow the protocol in program.md EXACTLY.

YOUR TASK FOR THIS SESSION (do exactly ONE experiment on SLOT {slot_id}):

CRITICAL GPU ASSIGNMENT: You are running on Slot {slot_id} with {gpu_name} (GPU index={gpu_id}).
- Use CUDA_VISIBLE_DEVICES={gpu_id} for ALL Python commands
- Use proxy config: {proxy_config}
- Use formal config: {formal_config}
- Tag all results with [slot{slot_id}/{gpu_name}] in the description column

1. Read backlog.md - understand current best results, agent state, and priorities
2. Read program.md - understand the full experiment protocol
3. Pick the HIGHEST PRIORITY uncompleted experiment from backlog.md
   - If another slot is currently working on an experiment (check autoresearch_logs/), pick a DIFFERENT one
4. Execute exactly ONE experiment:
   a. Make code changes within the allowed scope (see program.md)
   b. Git commit the changes before training
   c. Run proxy training: CUDA_VISIBLE_DEVICES={gpu_id} {python_cmd} train.py --config {proxy_config} > run.log 2>&1
   d. If crash: handle per program.md crash rules
   e. If success: read summary.json for best_val.accuracy
   f. Compare with current best (val_acc)
   g. Record results in results.tsv (use flock if available)
   h. Update backlog.md (move experiment to completed, update best record if keep)
5. Output a final summary line: EXPERIMENT_DONE: <status> | <description> | val_acc=<value> | slot={slot_id}

CRITICAL CONSTRAINTS:
- Use CUDA_VISIBLE_DEVICES={gpu_id} {python_cmd} for ALL Python commands
- Shell is Bash on Ubuntu
- Follow timeout rules in program.md
- Do NOT ask for human input - decide autonomously
- Read logs with: tail -30 run.log (never read the whole log)
- When reading text files use UTF-8 (Python: Path(...).read_text(encoding="utf-8"))
- Results summary.json is at: runs/$(basename "{proxy_config}" .yaml)/summary.json
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run autoresearch loops on two GPUs in parallel.")
    parser.add_argument("--max-iterations", type=int, default=25, help="Maximum iterations per slot.")
    parser.add_argument("--cooldown-seconds", type=int, default=30, help="Cooldown between iterations.")
    parser.add_argument("--slot-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--slot-id", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--gpu-id", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--gpu-name", help=argparse.SUPPRESS)
    parser.add_argument("--proxy-config", help=argparse.SUPPRESS)
    parser.add_argument("--formal-config", help=argparse.SUPPRESS)
    parser.add_argument("--python-executable", help=argparse.SUPPRESS)
    return parser.parse_args()


def run_slot_worker(args: argparse.Namespace) -> int:
    if shutil.which("qoder") is None:
        raise SystemExit("qoder CLI was not found in PATH.")

    log_dir = REPO_ROOT / "autoresearch_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    python_cmd = repo_python_display(args.python_executable)
    prompt = generate_prompt(args.slot_id, args.gpu_id, args.gpu_name, args.proxy_config, args.formal_config, python_cmd)

    completed = 0
    for iteration in range(1, args.max_iterations + 1):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"slot{args.slot_id}_run_{iteration}_{timestamp}.log"
        last_msg_file = log_dir / f"slot{args.slot_id}_run_{iteration}_{timestamp}.last.txt"

        print(f"[Slot {args.slot_id}] -- Iteration {iteration}/{args.max_iterations} -- {datetime.now().strftime('%H:%M:%S')} --")

        free_gb = disk_free_gb(REPO_ROOT)
        if free_gb < 10:
            print(f"[Slot {args.slot_id}] Less than 10 GB free. Stopping this slot.")
            break

        remove_nonbest_checkpoints(REPO_ROOT / "runs")

        started = time.time()
        with log_file.open("w", encoding="utf-8") as handle:
            result = subprocess.run(
                [
                    "qoder",
                    "exec",
                    "--dangerously-bypass-approvals-and-sandbox",
                    "--color",
                    "never",
                    "-C",
                    str(REPO_ROOT),
                    "-o",
                    str(last_msg_file),
                ],
                cwd=REPO_ROOT,
                env=os.environ.copy(),
                input=prompt,
                text=True,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )

        completed += 1
        duration_minutes = int((time.time() - started) // 60)
        if result.returncode == 0:
            print(f"[Slot {args.slot_id}] Iteration {iteration} done in {duration_minutes} min")
            if last_msg_file.exists():
                lines = last_msg_file.read_text(encoding="utf-8", errors="ignore").splitlines()
                last_line = lines[-1] if lines else ""
                if last_line:
                    print(f"[Slot {args.slot_id}] -> {last_line}")
                if last_line.startswith("CAMPAIGN_COMPLETE:"):
                    print(f"[Slot {args.slot_id}] Campaign complete. Stopping.")
                    break
        else:
            print(f"[Slot {args.slot_id}] Iteration {iteration} FAILED (exit={result.returncode}) after {duration_minutes} min")

        if iteration < args.max_iterations:
            print(f"[Slot {args.slot_id}] Cooling down {args.cooldown_seconds}s...")
            time.sleep(args.cooldown_seconds)

    print(f"[Slot {args.slot_id}] Loop finished: {completed} iteration(s)")
    return 0


def run_parent(args: argparse.Namespace) -> int:
    python_executable = detect_python_executable()
    PID_DIR.mkdir(parents=True, exist_ok=True)
    log_dir = REPO_ROOT / "autoresearch_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    print("")
    print("============================================================")
    print(" Parallel Autoresearch Loop - Ankle CT Classifier")
    print(f" Max iterations per slot: {args.max_iterations}")
    print(f" Cooldown:                {args.cooldown_seconds}s")
    print(f" Slot 0: GPU 0 (RTX 3090)")
    print(f" Slot 1: GPU 1 (RTX 4090)")
    print(f" Python:                  {repo_python_display(python_executable)}")
    print(f" Workdir:                 {REPO_ROOT}")
    print("============================================================")
    print("")

    gpu_lines = query_nvidia_smi("index,name,memory.used,memory.total")
    if gpu_lines:
        print("GPU status:")
        for line in gpu_lines:
            print(f"  {line}")
        print("")

    children: dict[int, subprocess.Popen[str]] = {}
    slot_specs = {
        0: (0, "RTX3090", "configs/autoresearch_proxy_slot0.yaml", "configs/autoresearch_formal_slot0.yaml"),
        1: (1, "RTX4090", "configs/autoresearch_proxy.yaml", "configs/autoresearch_formal.yaml"),
    }

    def cleanup(signum: int, _frame: object) -> None:
        print("")
        print(f"Caught signal {signum} - stopping all slots...")
        for slot_id, proc in children.items():
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
            pid_path = PID_DIR / f"loop_slot{slot_id}.pid"
            if pid_path.exists():
                pid_path.unlink()
        raise SystemExit(1)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    for slot_id, (gpu_id, gpu_name, proxy_config, formal_config) in slot_specs.items():
        print(f"Starting Slot {slot_id} ({gpu_name}, GPU {gpu_id})...")
        command = [
            python_executable,
            str(REPO_ROOT / "autoresearch_parallel_loop.py"),
            "--slot-worker",
            "--slot-id",
            str(slot_id),
            "--gpu-id",
            str(gpu_id),
            "--gpu-name",
            gpu_name,
            "--proxy-config",
            proxy_config,
            "--formal-config",
            formal_config,
            "--python-executable",
            python_executable,
            "--max-iterations",
            str(args.max_iterations),
            "--cooldown-seconds",
            str(args.cooldown_seconds),
        ]
        proc = subprocess.Popen(command, cwd=REPO_ROOT, text=True)
        children[slot_id] = proc
        (PID_DIR / f"loop_slot{slot_id}.pid").write_text(str(proc.pid), encoding="utf-8")

    print("")
    print(f"Both slots running. PIDs: slot0={children[0].pid}, slot1={children[1].pid}")
    print("Monitor with:    python scripts/parallel_status.py")
    print("Stop with:       Ctrl+C")
    print("")

    exit_codes: dict[int, int] = {}
    for slot_id, proc in children.items():
        exit_codes[slot_id] = proc.wait()
        pid_path = PID_DIR / f"loop_slot{slot_id}.pid"
        if pid_path.exists():
            pid_path.unlink()

    print("")
    print("============================================================")
    print(" Parallel loop finished")
    print(f"  Slot 0 exit: {exit_codes.get(0, 0)}")
    print(f"  Slot 1 exit: {exit_codes.get(1, 0)}")
    print("============================================================")
    return 1 if any(code != 0 for code in exit_codes.values()) else 0


def main() -> None:
    args = parse_args()
    if args.slot_worker:
        raise SystemExit(run_slot_worker(args))
    raise SystemExit(run_parent(args))


if __name__ == "__main__":
    main()

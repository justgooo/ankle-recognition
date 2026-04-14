from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from script_runtime import detect_python_executable, disk_free_gb, query_nvidia_smi, remove_nonbest_checkpoints  # noqa: E402


def repo_python_display(python_executable: str) -> str:
    candidate = Path(python_executable)
    if candidate.is_absolute():
        try:
            return candidate.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            return str(candidate)
    return python_executable


def build_prompt(workflow_mode: str, python_cmd: str) -> str:
    if workflow_mode == "joint":
        return f"""You are an autonomous ML researcher. Follow the protocol in program.md EXACTLY.

YOUR TASK FOR THIS SESSION (do exactly ONE research iteration):

1. Read backlog.md - understand current best results, agent state, and priorities
2. Read program.md - understand the full experiment protocol
3. Pick the HIGHEST PRIORITY uncompleted direction from backlog.md
4. Execute exactly ONE high-level AutoResearch change:
   a. Make one minimal but meaningful research change inside the allowed files
   b. Git commit the change before any training
   c. Run one baseline/smoke proxy check with train.py, compare val_acc from summary.json
   d. If the candidate is stable, run a small local Optuna study:
      {python_cmd} scripts/run_optuna_proxy.py
   e. Summarize that study:
      {python_cmd} scripts/monitor_optuna.py --study-dir runs/optuna_proxy
   f. Compare the best tuned candidate against the current keep version using val_acc from summary.json
   g. Only if improved, optionally run the deeper confirmation pass:
      {python_cmd} scripts/run_optuna_main.py --source-study-dir runs/optuna_proxy --top-k 3
   h. Record the outcome in results.tsv and update backlog.md
5. Output a final summary line:
   EXPERIMENT_DONE: <status> | <description> | val_acc=<value>

CRITICAL CONSTRAINTS:
- Use {python_cmd} for ALL Python commands
- Shell is Bash on Ubuntu
- Do not use test-set metrics for selection
- Follow timeout rules in program.md
- Do NOT ask for human input - decide autonomously
- Read logs with: tail -30 <logfile> (never read whole logs)
- Keep the Optuna layer external; do not rewrite train.py unless absolutely necessary
"""

    return f"""You are an autonomous ML researcher. Follow the protocol in program.md EXACTLY.

YOUR TASK FOR THIS SESSION (do exactly ONE experiment):

1. Read backlog.md - understand current best results, agent state, and priorities
2. Read program.md - understand the full experiment protocol
3. Pick the HIGHEST PRIORITY uncompleted experiment from backlog.md
4. Execute exactly ONE experiment:
   a. Make code changes within the allowed scope (see program.md)
   b. Git commit the changes before training
   c. Run proxy training: {python_cmd} train.py --config configs/autoresearch_proxy.yaml > run.log 2>&1
   d. If crash: handle per program.md crash rules
   e. If success: read summary.json for best_val.accuracy
   f. Compare with current best (val_acc)
   g. Record results in results.tsv
   h. Update backlog.md (move experiment to completed, update best record if keep)
5. Output a final summary line: EXPERIMENT_DONE: <status> | <description> | val_acc=<value>

CRITICAL CONSTRAINTS:
- Use {python_cmd} for ALL Python commands
- Shell is Bash on Ubuntu
- Follow timeout rules in program.md
- Do NOT ask for human input - decide autonomously
- Read logs with: tail -30 run.log (never read the whole log)
- When reading text files use UTF-8 (Python: Path(...).read_text(encoding="utf-8"))
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Start one fresh Codex exec session per autoresearch iteration.")
    parser.add_argument("--max-iterations", type=int, default=50, help="Maximum loop iterations.")
    parser.add_argument("--cooldown-seconds", type=int, default=30, help="Cooldown between experiments.")
    parser.add_argument("--workflow", choices=["classic", "joint"], default="classic", help="Workflow prompt mode.")
    args = parser.parse_args()

    if shutil.which("codex") is None:
        raise SystemExit("codex CLI was not found in PATH.")

    python_executable = detect_python_executable()
    python_cmd = repo_python_display(python_executable)
    prompt = build_prompt(args.workflow, python_cmd)

    log_dir = REPO_ROOT / "autoresearch_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "1")

    print("")
    print("============================================================")
    print(" Autoresearch Loop - Ankle CT Classifier")
    print(f" Max iterations:  {args.max_iterations}")
    print(f" Cooldown:        {args.cooldown_seconds}s between experiments")
    print(f" Workflow:        {args.workflow}")
    print(f" Python:          {python_cmd}")
    print(f" Workdir:         {REPO_ROOT}")
    print("============================================================")
    print("")

    completed_iterations = 0
    for iteration in range(1, args.max_iterations + 1):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"run_{iteration}_{timestamp}.log"
        last_msg_file = log_dir / f"run_{iteration}_{timestamp}.last.txt"

        print("")
        print("----------------------------------------")
        print(f" Iteration {iteration} / {args.max_iterations}")
        print(f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("----------------------------------------")

        free_gb = disk_free_gb(REPO_ROOT)
        print(f"  Disk free: {free_gb} GB")
        if free_gb < 10:
            print("  Less than 10 GB free. Stopping loop.")
            break

        gpu_lines = query_nvidia_smi("memory.used")
        if gpu_lines:
            gpu_index = 1 if len(gpu_lines) > 1 else 0
            print(f"  GPU target memory used: {gpu_lines[gpu_index]} MB")

        remove_nonbest_checkpoints(REPO_ROOT / "runs")

        print("  Starting Codex session...")
        print(f"  Session log: autoresearch_logs/{log_file.name}")
        started = time.time()
        with log_file.open("w", encoding="utf-8") as handle:
            result = subprocess.run(
                [
                    "codex",
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
                env=env,
                input=prompt,
                text=True,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )

        completed_iterations += 1
        duration_minutes = int((time.time() - started) // 60)
        if result.returncode == 0:
            print(f"  Iteration {iteration} completed in {duration_minutes} min")
            if last_msg_file.exists():
                lines = last_msg_file.read_text(encoding="utf-8", errors="ignore").splitlines()
                last_line = lines[-1] if lines else ""
                if last_line:
                    print(f"  Final summary: {last_line}")
                if last_line.startswith("CAMPAIGN_COMPLETE:"):
                    print("  Campaign complete. Stopping loop.")
                    break
        else:
            print(f"  Iteration {iteration} failed (exit={result.returncode}) after {duration_minutes} min")

        if iteration < args.max_iterations:
            print(f"  Cooling down {args.cooldown_seconds}s...")
            time.sleep(args.cooldown_seconds)

    print("")
    print("============================================================")
    print(f" Loop finished: {completed_iterations} iteration(s) attempted")
    print("============================================================")


if __name__ == "__main__":
    main()

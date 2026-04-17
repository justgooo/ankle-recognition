from __future__ import annotations

import argparse
import itertools
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from script_runtime import disk_free_gb, query_nvidia_smi, remove_nonbest_checkpoints  # noqa: E402


def detect_repo_python() -> Path:
    candidates = [
        REPO_ROOT / ".venv" / "bin" / "python",
        REPO_ROOT / ".venv" / "Scripts" / "python.exe",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise SystemExit(
        "Project virtualenv python was not found. Expected one of: "
        + ", ".join(str(path) for path in candidates)
    )


def repo_python_display(python_path: Path) -> str:
    try:
        return python_path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(python_path)


def build_prompt(default_lane: str, gpu_id: int, python_cmd: str) -> str:
    formal_command = (
        f"CUDA_VISIBLE_DEVICES={gpu_id} timeout 10800 "
        f"{python_cmd} train.py --config configs/autoresearch_formal.yaml > run.log 2>&1"
    )
    proxy_command = (
        f"CUDA_VISIBLE_DEVICES={gpu_id} timeout 3600 "
        f"{python_cmd} train.py --config configs/autoresearch_proxy.yaml > proxy.log 2>&1"
    )
    default_lane_text = "formal/main" if default_lane == "formal" else "proxy"

    return f"""你是一个完全自主的 ML 研究者。本 session 只做一轮实验迭代，然后退出；外层 autoresearch_loop.py 会在你退出后决定是否再次启动你。

必须严格遵守以下顺序：
1. 先读 backlog.md
2. 再读 program.md

当前环境与硬约束：
- 工作目录：{REPO_ROOT}
- Shell：Bash on Ubuntu
- GPU 默认槽位：CUDA_VISIBLE_DEVICES={gpu_id}
- 所有 Python 命令必须使用：{python_cmd}
- 默认研究 lane：{default_lane_text}
- 默认直接跑 formal/main；proxy 只用于显存不足、快速 smoke、训练 bug 定位或低成本诊断
- 不要修改 train.py、src/dataset.py、src/utils.py、tools/、任何数据文件或数据集划分
- 不要使用测试集指标做模型选择
- 不要阅读全文日志；只允许 tail -30
- 不要问“是否继续”；外层 loop 会继续
- 不要在本 session 里再自行写无限循环；做完一轮就退出

本轮任务（exactly one research iteration）：
1. 阅读 backlog.md 和 program.md，理解当前主线、最新 best record、允许修改范围和实验协议。
2. 自主判断这一轮最值得做的一项实验性改动。backlog 不是严格 machine-readable 队列，你可以根据当前仓库状态自行判断，但必须对齐文档主线。
3. 只做一项离散、可解释的研究改动；不要把多个独立想法混在同一轮。
4. 在训练前 git commit 本轮改动。
5. 默认直接运行 formal：
   {formal_command}
6. 只有在你能明确说明理由时，才改用 proxy：
   {proxy_command}
7. 如果本轮更适合用 Optuna study，也可以使用：
   - {python_cmd} scripts/optuna_main.py ...
   - {python_cmd} scripts/optuna_proxy.py ...
   - {python_cmd} scripts/monitor_optuna.py ...
   但必须遵守 fresh/resume 语义、val_acc 主指标规则，以及当前“默认 direct formal”的总策略。
8. 训练失败时，只读最后 30 行日志，按 OOM / timeout / code bug / data issue 分类处理。
   - data issue：停止本轮并输出阻塞原因
   - code bug：可以修复后重跑 1 次
   - 其他失败：按 crash 记录
9. 训练成功时，从 summary.json 读取 best_val.accuracy 作为主指标；若 val_acc 持平，再比较 best_val.auc。
10. 更新 results.tsv（只追加；并行风险下使用 flock -x /tmp/ankle_results.lock）和 backlog.md，包括 Agent 状态、上次实验、上次结果、下一步。
11. 最后输出一行机器可读总结，格式必须是：
    EXPERIMENT_DONE: <keep|discard|crash> | lane=<formal|proxy|main-study|proxy-study> | description=<...> | val_acc=<...> | val_auc=<...> | commit=<...>

如果你遇到真正需要人类拍板的阻塞，不要继续盲跑；只输出一行：
LOOP_BLOCKED: <reason>
"""


def latest_summary_line(path: Path) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for line in reversed(lines):
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continuously relaunch one Codex experiment iteration at a time.")
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=0,
        help="Maximum loop iterations. Use 0 for unbounded looping until interrupted.",
    )
    parser.add_argument(
        "--cooldown-seconds",
        type=int,
        default=30,
        help="Cooldown between iterations.",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=1,
        help="CUDA_VISIBLE_DEVICES value exported to inner Codex sessions. Defaults to slot 1.",
    )
    parser.add_argument(
        "--default-lane",
        choices=["formal", "proxy"],
        default="formal",
        help="Default lane described to the inner agent. Defaults to direct formal/main.",
    )
    parser.add_argument(
        "--min-disk-free-gb",
        type=int,
        default=10,
        help="Stop the loop if free disk space drops below this threshold.",
    )
    parser.add_argument(
        "--print-prompt",
        action="store_true",
        help="Print the currently generated Codex prompt and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if shutil.which("codex") is None:
        raise SystemExit("codex CLI was not found in PATH.")

    python_path = detect_repo_python()
    python_cmd = repo_python_display(python_path)
    prompt = build_prompt(args.default_lane, args.gpu_id, python_cmd)

    if args.print_prompt:
        print(prompt)
        return

    log_dir = REPO_ROOT / "autoresearch_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = log_dir / "autoresearch_loop.prompt.txt"
    prompt_path.write_text(prompt, encoding="utf-8")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    env.setdefault("PYTHONIOENCODING", "utf-8")

    max_iterations_text = "infinite" if args.max_iterations == 0 else str(args.max_iterations)

    print("")
    print("============================================================")
    print(" Autoresearch Loop - Ankle CT Classifier")
    print(f" Max iterations:  {max_iterations_text}")
    print(f" Cooldown:        {args.cooldown_seconds}s between experiments")
    print(f" GPU slot:        CUDA_VISIBLE_DEVICES={args.gpu_id}")
    print(f" Default lane:    {args.default_lane}")
    print(f" Python:          {python_cmd}")
    print(f" Workdir:         {REPO_ROOT}")
    print(f" Prompt file:     {prompt_path.relative_to(REPO_ROOT).as_posix()}")
    print("============================================================")
    print("")

    completed_iterations = 0
    iteration_source = itertools.count(1) if args.max_iterations == 0 else range(1, args.max_iterations + 1)

    for iteration in iteration_source:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"run_{iteration}_{timestamp}.log"
        last_msg_file = log_dir / f"run_{iteration}_{timestamp}.last.txt"

        print("")
        print("----------------------------------------")
        iteration_header = f" Iteration {iteration}"
        if args.max_iterations > 0:
            iteration_header += f" / {args.max_iterations}"
        print(iteration_header)
        print(f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("----------------------------------------")

        free_gb = disk_free_gb(REPO_ROOT)
        print(f"  Disk free: {free_gb} GB")
        if free_gb < args.min_disk_free_gb:
            print(f"  Less than {args.min_disk_free_gb} GB free. Stopping loop.")
            break

        gpu_lines = query_nvidia_smi("index,name,memory.used,memory.total,utilization.gpu")
        if gpu_lines:
            matching_lines = [line for line in gpu_lines if line.startswith(f"{args.gpu_id},")]
            for line in matching_lines[:1]:
                print(f"  GPU status: {line}")

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
        summary_line = latest_summary_line(last_msg_file)

        if result.returncode == 0:
            print(f"  Iteration {iteration} completed in {duration_minutes} min")
            if summary_line:
                print(f"  Final summary: {summary_line}")
        else:
            print(f"  Iteration {iteration} failed (exit={result.returncode}) after {duration_minutes} min")
            if summary_line:
                print(f"  Last message:  {summary_line}")

        if summary_line.startswith("LOOP_BLOCKED:"):
            print("  Inner agent reported a blocking issue. Stopping loop.")
            break
        if summary_line.startswith("CAMPAIGN_COMPLETE:"):
            print("  Inner agent reported campaign completion. Stopping loop.")
            break

        if args.max_iterations > 0 and iteration >= args.max_iterations:
            break

        print(f"  Cooling down {args.cooldown_seconds}s...")
        time.sleep(max(args.cooldown_seconds, 0))

    print("")
    print("============================================================")
    print(f" Loop finished: {completed_iterations} iteration(s) attempted")
    print("============================================================")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import itertools
import os
import shlex
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


def repo_relative(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def resolve_home_dir() -> Path:
    raw_home = os.environ.get("HOME")
    if raw_home:
        return Path(raw_home).expanduser()
    return Path.home()


def resolve_codex_home(home_dir: Path) -> Path:
    raw_codex_home = os.environ.get("CODEX_HOME")
    if raw_codex_home:
        return Path(raw_codex_home).expanduser()
    return home_dir / ".codex"


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def build_iteration_plan(iteration: int, timestamp: str) -> dict[str, str]:
    tag = f"iter_{iteration:04d}_{timestamp}"
    generated_dir = REPO_ROOT / "autoresearch_logs" / "generated_search_configs"
    return {
        "tag": tag,
        "main_search_config": repo_relative(generated_dir / f"optuna_main_search_{tag}.yaml"),
        "proxy_search_config": repo_relative(generated_dir / f"optuna_proxy_search_{tag}.yaml"),
        "main_study_root": f"runs/optuna_main_autoloop/{tag}",
        "proxy_study_root": f"runs/optuna_proxy_autoloop/{tag}",
    }


def build_prompt(
    default_lane: str,
    fallback_gpu_id: int,
    gpu_ids: str,
    max_workers: str,
    max_used_memory_mb: int,
    max_utilization: int,
    python_cmd: str,
    iteration_plan: dict[str, str],
    main_search_template: str,
    proxy_search_template: str,
    formal_config: str,
    proxy_config: str,
    campaign_note: str,
) -> str:
    main_study_command = shell_join(
        [
            "timeout",
            "43200",
            python_cmd,
            "scripts/autoresearch_main.py",
            "--search-config",
            iteration_plan["main_search_config"],
            "--gpu-ids",
            gpu_ids,
            "--max-workers",
            max_workers,
            "--max-used-memory-mb",
            str(max_used_memory_mb),
            "--max-utilization",
            str(max_utilization),
        ]
    )
    proxy_study_command = shell_join(
        [
            "timeout",
            "14400",
            python_cmd,
            "scripts/autoresearch_proxy.py",
            "--search-config",
            iteration_plan["proxy_search_config"],
            "--gpu-ids",
            gpu_ids,
            "--max-workers",
            max_workers,
            "--max-used-memory-mb",
            str(max_used_memory_mb),
            "--max-utilization",
            str(max_utilization),
        ]
    )
    main_monitor_command = shell_join(
        [python_cmd, "scripts/monitor_optuna.py", "--study-dir", iteration_plan["main_study_root"]]
    )
    proxy_monitor_command = shell_join(
        [python_cmd, "scripts/monitor_optuna.py", "--study-dir", iteration_plan["proxy_study_root"]]
    )
    formal_command = (
        f"CUDA_VISIBLE_DEVICES={fallback_gpu_id} timeout 10800 "
        f"{shell_join([python_cmd, 'train.py', '--config', formal_config])} > run.log 2>&1"
    )
    proxy_command = (
        f"CUDA_VISIBLE_DEVICES={fallback_gpu_id} timeout 3600 "
        f"{shell_join([python_cmd, 'train.py', '--config', proxy_config])} > proxy.log 2>&1"
    )

    lane_labels = {
        "main-study": "adaptive main-study",
        "proxy-study": "adaptive proxy-study",
        "formal": "direct formal",
        "proxy": "direct proxy",
    }
    default_lane_text = lane_labels[default_lane]

    primary_lane_instructions = {
        "main-study": f"""默认优先主流程：
- 把 `{main_search_template}` 复制成 `{iteration_plan["main_search_config"]}`，并把其中 `study.study_root` 改成 `{iteration_plan["main_study_root"]}`。
- 优先运行多卡自适应 main study：
  {main_study_command} > optuna_main.log 2>&1
- study 结束后运行 monitor 汇总：
  {main_monitor_command}
- 以最佳 completed trial 的 `summary.json.best_val.accuracy` 为主指标，`best_val.auc` 为 tie-break；只有在需要最终确认时才补 1 次 direct formal：
  {formal_command}""",
        "proxy-study": f"""默认优先主流程：
- 把 `{proxy_search_template}` 复制成 `{iteration_plan["proxy_search_config"]}`，并把其中 `study.study_root` 改成 `{iteration_plan["proxy_study_root"]}`。
- 优先运行自适应 proxy study：
  {proxy_study_command} > optuna_proxy.log 2>&1
- study 结束后运行 monitor 汇总：
  {proxy_monitor_command}
- proxy 只作低显存 / 快速诊断 / 便宜筛查；若出现强 winner，再决定是否升到 formal/main：
  {formal_command}""",
        "formal": f"""默认优先主流程：
- 先做 direct formal：
  {formal_command}
- 如果当前结构改动明显更适合系统性多卡调参，再升级到 fresh main study：
  {main_study_command} > optuna_main.log 2>&1
  然后运行：
  {main_monitor_command}""",
        "proxy": f"""默认优先主流程：
- 先做 direct proxy：
  {proxy_command}
- 如果当前结构改动需要系统性多卡调参，再升级到 fresh proxy study 或 fresh main study：
  {proxy_study_command} > optuna_proxy.log 2>&1
  或
  {main_study_command} > optuna_main.log 2>&1""",
    }[default_lane]

    return f"""你是这个仓库本轮唯一的 coordinator agent。本 session 只做一轮研究迭代，然后退出；外层 autoresearch_loop.py 会在你退出后决定是否再次启动你。

必须严格遵守以下顺序：
1. 先读 backlog.md
2. 再读 program.md

当前 coordinator 模式与硬约束：
- 工作目录：{REPO_ROOT}
- Shell：Bash on Ubuntu
- 所有 Python 命令必须使用：{python_cmd}
- 你是唯一 agent；不要再起多个 codex exec 或多个并行改代码 agent
- 真正的并行应交给自适应 Optuna worker，而不是多个 agent
- adaptive GPU policy：`--gpu-ids {gpu_ids}`，`--max-workers {max_workers}`，idle 阈值 `used<= {max_used_memory_mb} MiB` 且 `util<= {max_utilization}%`
- 单卡 fallback 槽位：CUDA_VISIBLE_DEVICES={fallback_gpu_id}
- 默认研究 lane：{default_lane_text}
- 默认优先使用 `scripts/autoresearch_main.py` / `scripts/autoresearch_proxy.py` 这两个兼容入口；它们底层分别委托给当前 canonical `scripts/optuna_main.py` / `scripts/optuna_proxy.py`
- 本轮 main search 模板：`{main_search_template}`
- 本轮 proxy search 模板：`{proxy_search_template}`
- 本轮 direct formal 配置：`{formal_config}`
- 本轮 direct proxy 配置：`{proxy_config}`
- 当前唯一主方向：实验 decision fusion，把 axial / coronal / sagittal 各视角的信息还原成它们在 full-fusion 中应有的作用；核心不是机械平均分权，而是让该主导的视角主导、该补充的视角补充，并让真正的 multi-view full-fusion learned `val_acc` 明确超过 matched `equal-weight` control
- 在 full-fusion learned `val_acc` 尚未明确超过 matched `equal-weight` 之前，禁止切换到其他方向；不要切 backbone/geometry family，不要切到 feature fusion，不要把无关 side campaign 或泛化 cleanup 当主线
- 单 seed spike、只提升 `val_auc` / `val_f1`、只改善 single-view 或 leave-one-view-out 结果，都不算完成上述主方向
- fresh run 默认必须使用新的 `study_root`；只有你明确想续跑同一个 study 时才允许 `--resume`
- 不要修改 train.py、src/dataset.py、src/utils.py、tools/、任何数据文件或数据集划分
- 不要使用测试集指标做模型选择
- 不要阅读全文日志；只允许 `tail -30`
- 不要问“是否继续”；外层 loop 会继续
- 不要在本 session 里再自行写无限循环；做完一轮就退出

本轮 campaign 额外约束（高优先级）：
{campaign_note if campaign_note else "- 无额外约束。"}

本轮预留的 fresh study 路径：
- main search-config copy：{iteration_plan["main_search_config"]}
- main study_root：{iteration_plan["main_study_root"]}
- proxy search-config copy：{iteration_plan["proxy_search_config"]}
- proxy study_root：{iteration_plan["proxy_study_root"]}

本轮任务（exactly one research iteration）：
1. 阅读 backlog.md 和 program.md，理解当前主线、最新 best record、允许修改范围和实验协议。
2. 自主判断这一轮最值得做的一项实验性改动。backlog 不是严格 machine-readable 队列，你可以根据当前仓库状态自行判断，但必须对齐文档主线，并且必须直接服务于“让 decision fusion 还原各视角信息应有作用、让 learned full-fusion 超过 equal-weight”这一唯一目标。
3. 在真正行动前，先做两轮自检：
   - 这项改动是否直接帮助“该主导的视角主导、该补充的视角补充”，或直接修复 evidence 与 routing 不一致、增强弱视角在有证据样本上的有效贡献？
   - 如果它成功，为什么它有机会把 multi-view full-fusion learned `val_acc` 推到 matched `equal-weight` 之上，而不是只把权重做得更平均、或只改善单视角表现、稳定性或局部 calibration？
   如果这两问任意一问不能给出具体机制链路，就不要执行该改动，换一个更直接的方向。
4. 只做一项离散、可解释的研究改动；不要把多个独立想法混在同一轮。
5. 在训练或调参前 git commit 本轮改动。
{primary_lane_instructions}
6. 只有在低显存、快速 smoke、训练 bug 定位或更便宜的诊断场景下，才允许改走 proxy：
   {proxy_command}
   或 fresh proxy study：
   {proxy_study_command} > optuna_proxy.log 2>&1
   然后运行：
   {proxy_monitor_command}
7. fresh study 如果因为已有 sqlite / storage 报错，优先理解为你忘了给本轮 fresh study 使用新的 `study_root`；先修正这个问题，不要直接把这种错误记成 crash。
8. 训练或 study 失败时，只读最后 30 行日志，按 OOM / timeout / code bug / data issue 分类处理。
   - data issue：停止本轮并输出阻塞原因
   - code bug：可以修复后重跑 1 次
   - 其他失败：按 crash 记录
9. 更新 results.tsv（只追加；并行风险下使用 `flock -x /tmp/ankle_results.lock`）和 backlog.md，包括 Agent 状态、上次实验、上次结果、下一步。
10. `results.tsv` 的 config 列仍保持 `formal` / `proxy` 语义：main-study 或 direct formal 记为 `formal`；proxy-study 或 direct proxy 记为 `proxy`。
11. 最后输出一行机器可读总结，格式必须是：
    EXPERIMENT_DONE: <keep|discard|crash> | lane=<main-study|proxy-study|formal|proxy> | description=<...> | val_acc=<...> | val_auc=<...> | commit=<...>

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
    parser = argparse.ArgumentParser(
        description="Continuously relaunch one coordinator Codex session at a time while adaptive Optuna workers use the GPUs."
    )
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
        help="Fallback CUDA_VISIBLE_DEVICES slot for direct formal/proxy smoke commands. Defaults to slot 1.",
    )
    parser.add_argument(
        "--default-lane",
        choices=["main-study", "proxy-study", "formal", "proxy"],
        default="main-study",
        help="Default lane described to the inner coordinator agent. Defaults to adaptive multi-GPU main-study.",
    )
    parser.add_argument(
        "--gpu-ids",
        default="auto",
        help="GPU set passed to adaptive Optuna entrypoints. Use 'auto' or a comma-separated nvidia-smi GPU list.",
    )
    parser.add_argument(
        "--max-workers",
        default="auto",
        help="Maximum adaptive Optuna workers. Use 'auto' or an integer.",
    )
    parser.add_argument(
        "--max-used-memory-mb",
        type=int,
        default=1024,
        help="Idle-GPU threshold passed to adaptive Optuna entrypoints.",
    )
    parser.add_argument(
        "--max-utilization",
        type=int,
        default=20,
        help="Idle-GPU utilization threshold passed to adaptive Optuna entrypoints.",
    )
    parser.add_argument(
        "--main-search-template",
        default="configs/optuna_main_search.yaml",
        help="Main-study search-config template copied for each loop iteration.",
    )
    parser.add_argument(
        "--proxy-search-template",
        default="configs/optuna_proxy_search.yaml",
        help="Proxy-study search-config template copied for each loop iteration.",
    )
    parser.add_argument(
        "--formal-config",
        default="configs/autoresearch_formal.yaml",
        help="Direct-formal config used in prompt fallback commands.",
    )
    parser.add_argument(
        "--proxy-config",
        default="configs/autoresearch_proxy.yaml",
        help="Direct-proxy config used in prompt fallback commands.",
    )
    parser.add_argument(
        "--campaign-note",
        default="",
        help="Extra high-priority prompt note injected into every coordinator iteration.",
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

    if not args.print_prompt and shutil.which("codex") is None:
        raise SystemExit("codex CLI was not found in PATH.")

    python_path = detect_repo_python()
    python_cmd = repo_python_display(python_path)
    preview_plan = build_iteration_plan(1, datetime.now().strftime("%Y%m%d_%H%M%S"))
    preview_prompt = build_prompt(
        default_lane=args.default_lane,
        fallback_gpu_id=args.gpu_id,
        gpu_ids=args.gpu_ids,
        max_workers=str(args.max_workers),
        max_used_memory_mb=args.max_used_memory_mb,
        max_utilization=args.max_utilization,
        python_cmd=python_cmd,
        iteration_plan=preview_plan,
        main_search_template=args.main_search_template,
        proxy_search_template=args.proxy_search_template,
        formal_config=args.formal_config,
        proxy_config=args.proxy_config,
        campaign_note=args.campaign_note,
    )

    if args.print_prompt:
        print(preview_prompt)
        return

    log_dir = REPO_ROOT / "autoresearch_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    latest_prompt_path = log_dir / "autoresearch_loop.prompt.txt"
    home_dir = resolve_home_dir()
    codex_home = resolve_codex_home(home_dir)
    codex_config_path = codex_home / "config.toml"
    codex_auth_path = codex_home / "auth.json"

    base_env = os.environ.copy()
    base_env.pop("CUDA_VISIBLE_DEVICES", None)
    base_env.setdefault("PYTHONIOENCODING", "utf-8")
    base_env["HOME"] = str(home_dir)
    base_env["CODEX_HOME"] = str(codex_home)
    base_env["AUTORESEARCH_LOOP_MODE"] = "coordinator"
    base_env["AUTORESEARCH_LOOP_DEFAULT_LANE"] = args.default_lane
    base_env["AUTORESEARCH_LOOP_GPU_IDS"] = str(args.gpu_ids)
    base_env["AUTORESEARCH_LOOP_MAX_WORKERS"] = str(args.max_workers)
    base_env["AUTORESEARCH_LOOP_MAX_USED_MEMORY_MB"] = str(args.max_used_memory_mb)
    base_env["AUTORESEARCH_LOOP_MAX_UTILIZATION"] = str(args.max_utilization)
    base_env["AUTORESEARCH_LOOP_FALLBACK_GPU_ID"] = str(args.gpu_id)
    base_env["AUTORESEARCH_LOOP_MAIN_SEARCH_TEMPLATE"] = args.main_search_template
    base_env["AUTORESEARCH_LOOP_PROXY_SEARCH_TEMPLATE"] = args.proxy_search_template
    base_env["AUTORESEARCH_LOOP_FORMAL_CONFIG"] = args.formal_config
    base_env["AUTORESEARCH_LOOP_PROXY_CONFIG"] = args.proxy_config
    base_env["AUTORESEARCH_LOOP_CAMPAIGN_NOTE"] = args.campaign_note

    max_iterations_text = "infinite" if args.max_iterations == 0 else str(args.max_iterations)

    print("")
    print("============================================================")
    print(" Autoresearch Coordinator Loop - Ankle CT Classifier")
    print(f" Max iterations:  {max_iterations_text}")
    print(f" Cooldown:        {args.cooldown_seconds}s between experiments")
    print(f" Default lane:    {args.default_lane}")
    print(f" Adaptive GPUs:   {args.gpu_ids}")
    print(f" Max workers:     {args.max_workers}")
    print(f" GPU idle rule:   used<={args.max_used_memory_mb} MiB, util<={args.max_utilization}%")
    print(f" Fallback slot:   CUDA_VISIBLE_DEVICES={args.gpu_id}")
    print(f" Main template:   {args.main_search_template}")
    print(f" Proxy template:  {args.proxy_search_template}")
    print(f" Formal config:   {args.formal_config}")
    print(f" Proxy config:    {args.proxy_config}")
    print(f" Python:          {python_cmd}")
    print(f" Workdir:         {REPO_ROOT}")
    print(f" HOME:            {home_dir}")
    print(f" CODEX_HOME:      {codex_home}")
    print(f" Codex config:    {codex_config_path} ({'found' if codex_config_path.is_file() else 'missing'})")
    print(f" Codex auth:      {codex_auth_path} ({'found' if codex_auth_path.is_file() else 'missing'})")
    print(f" Prompt file:     {latest_prompt_path.relative_to(REPO_ROOT).as_posix()}")
    print("============================================================")
    print("")

    completed_iterations = 0
    iteration_source = itertools.count(1) if args.max_iterations == 0 else range(1, args.max_iterations + 1)

    for iteration in iteration_source:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        iteration_plan = build_iteration_plan(iteration, timestamp)
        prompt = build_prompt(
            default_lane=args.default_lane,
            fallback_gpu_id=args.gpu_id,
            gpu_ids=args.gpu_ids,
            max_workers=str(args.max_workers),
            max_used_memory_mb=args.max_used_memory_mb,
            max_utilization=args.max_utilization,
            python_cmd=python_cmd,
            iteration_plan=iteration_plan,
            main_search_template=args.main_search_template,
            proxy_search_template=args.proxy_search_template,
            formal_config=args.formal_config,
            proxy_config=args.proxy_config,
            campaign_note=args.campaign_note,
        )
        log_file = log_dir / f"run_{iteration}_{timestamp}.log"
        last_msg_file = log_dir / f"run_{iteration}_{timestamp}.last.txt"
        prompt_file = log_dir / f"run_{iteration}_{timestamp}.prompt.txt"
        prompt_file.write_text(prompt, encoding="utf-8")
        latest_prompt_path.write_text(prompt, encoding="utf-8")

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
            if str(args.gpu_ids).strip().lower() == "auto":
                selected_lines = gpu_lines
            else:
                prefixes = [f"{token.strip()}," for token in str(args.gpu_ids).split(",") if token.strip()]
                selected_lines = [
                    line for line in gpu_lines if any(line.startswith(prefix) for prefix in prefixes)
                ] or gpu_lines
            for line in selected_lines[:8]:
                print(f"  GPU status: {line}")

        remove_nonbest_checkpoints(REPO_ROOT / "runs")

        env = base_env.copy()
        env["AUTORESEARCH_LOOP_ITERATION_TAG"] = iteration_plan["tag"]
        env["AUTORESEARCH_LOOP_MAIN_SEARCH_CONFIG"] = iteration_plan["main_search_config"]
        env["AUTORESEARCH_LOOP_PROXY_SEARCH_CONFIG"] = iteration_plan["proxy_search_config"]
        env["AUTORESEARCH_LOOP_MAIN_STUDY_ROOT"] = iteration_plan["main_study_root"]
        env["AUTORESEARCH_LOOP_PROXY_STUDY_ROOT"] = iteration_plan["proxy_study_root"]

        print("  Starting Codex session...")
        print(f"  Session log: autoresearch_logs/{log_file.name}")
        print(f"  Prompt copy:  autoresearch_logs/{prompt_file.name}")
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

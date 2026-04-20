#!/usr/bin/env python
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from script_runtime import detect_python_executable, read_output_dir, repo_relative, tail_lines


REPO_ROOT = SCRIPT_DIR.parent
SEEDS = [42, 123, 456]
VARIANTS = ["learned", "equal"]
RUN_ORDER = [
    (42, "learned"),
    (42, "equal"),
    (123, "learned"),
    (123, "equal"),
    (456, "learned"),
    (456, "equal"),
]
CONFIGS = {
    (seed, variant): f"configs/cmp_decision_equal_resnext_attnpool_{variant}_512x16_e20_s{seed}.yaml"
    for seed in SEEDS
    for variant in VARIANTS
}
BACKLOG_HEADING = "## 2026-04-20：ResNeXt AttentionPooling Equal-vs-Learned Control（seeds=42/123/456，node20 V100q）"


@dataclass(frozen=True)
class RunSpec:
    seed: int
    variant: str
    config_rel: str

    @property
    def experiment_tag(self) -> str:
        return (
            f"CMP-DECISION-EQUAL-RESNEXT-ATTNPOOL-{self.variant.upper()}-512X16-E20-S{self.seed}"
        )


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
            "Run the ResNeXt+AttentionPooling learned-vs-equal decision-fusion "
            "control across seeds 42/123/456 inside one 3-GPU Slurm allocation."
        )
    )
    parser.add_argument(
        "--jobid",
        default=os.environ.get("SLURM_JOB_ID", ""),
        help="Target Slurm allocation job id. Defaults to $SLURM_JOB_ID.",
    )
    parser.add_argument(
        "--cpus-per-run",
        type=int,
        default=5,
        help="Slurm CPUs to reserve per training step.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=10800,
        help="Timeout per run in seconds. Default: 3 hours.",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=3,
        help="Maximum number of concurrent 1-GPU training steps.",
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=20.0,
        help="Abort early if any visible GPU in the target allocation has less free memory than this.",
    )
    parser.add_argument(
        "--log-dir",
        default="autoresearch_logs/resnext_attnpool_equal_multiseed",
        help="Directory for campaign logs and aggregate JSON.",
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


def srun_prefix(jobid: str) -> list[str]:
    return ["srun", "--jobid", jobid, "--overlap"]


def print_plan(python_executable: str, args: argparse.Namespace, log_dir: Path) -> None:
    print("")
    print("============================================================")
    print(" ResNeXt AttentionPooling Decision-Fusion Control")
    print(f" Job id:         {args.jobid}")
    print(f" Python:         {python_executable}")
    print(f" Log dir:        {repo_relative(log_dir)}")
    print(f" CPUs / run:     {args.cpus_per_run}")
    print(f" Max parallel:   {args.max_parallel}")
    print(f" Timeout:        {args.timeout}s")
    print(f" Min free VRAM:  {args.min_free_gb:.1f} GiB")
    print(" Queue order:")
    for seed, variant in RUN_ORDER:
        print(f"   - seed={seed} variant={variant} -> {CONFIGS[(seed, variant)]}")
    print("============================================================")
    print("")


def git_short_hash() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def ensure_results_file() -> None:
    results_path = REPO_ROOT / "results.tsv"
    if results_path.exists():
        return
    header = "commit\tval_acc\tval_auc\tval_f1\tmemory_gb\tstatus\tconfig\tdescription\n"
    results_path.write_text(header, encoding="utf-8")


def run_slurm_command(
    *,
    jobid: str,
    env: dict[str, str],
    command: list[str],
    capture_output: bool = False,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    full_command = srun_prefix(jobid) + command
    return subprocess.run(
        full_command,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=capture_output,
        check=check,
    )


def ensure_visible_gpus(
    *,
    jobid: str,
    python_executable: str,
    env: dict[str, str],
    min_free_gb: float,
    max_parallel: int,
) -> None:
    command = [
        python_executable,
        "-c",
        (
            "import sys, torch; "
            "count=torch.cuda.device_count(); "
            "print(f'visible_gpus={count}'); "
            f"assert count >= {max_parallel}, f'Need at least {max_parallel} visible GPUs, got {count}'; "
            "free_gb=[]; "
            "[free_gb.append(torch.cuda.mem_get_info(i)[0]/1024**3) or "
            "print(f'gpu{i}_free_gb={free_gb[-1]:.2f}') for i in range(count)]; "
            f"sys.exit(0 if min(free_gb) >= {min_free_gb} else 2)"
        ),
    ]
    result = run_slurm_command(
        jobid=jobid,
        env=env,
        command=command,
        capture_output=True,
        check=False,
    )
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    if result.returncode == 2:
        raise SystemExit(
            f"At least one visible GPU in job {jobid} has < {min_free_gb:.1f} GiB free memory."
        )
    if result.returncode != 0:
        raise SystemExit(f"Failed to verify visible GPUs inside job {jobid}.")


def run_preflight(*, jobid: str, python_executable: str, env: dict[str, str]) -> None:
    command = [
        python_executable,
        "scripts/backbone_preflight.py",
        "--config",
        *[CONFIGS[(seed, variant)] for seed in SEEDS for variant in VARIANTS],
    ]
    run_slurm_command(jobid=jobid, env=env, command=command, capture_output=False, check=True)


def launch_training_job(
    *,
    jobid: str,
    python_executable: str,
    env: dict[str, str],
    spec: RunSpec,
    cpus_per_run: int,
    timeout_seconds: int,
    log_dir: Path,
) -> tuple[subprocess.Popen[bytes], Path, Path, Any]:
    config_path = REPO_ROOT / spec.config_rel
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{config_path.stem}_{timestamp}.log"
    command = srun_prefix(jobid) + [
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
        spec.config_rel,
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


def is_valid_result(result: dict[str, Any]) -> bool:
    if result.get("returncode") != 0 or result.get("status") != "ok":
        return False
    return not any(
        math.isnan(float(result[key]))
        for key in ("val_acc", "val_auc", "val_f1", "peak_vram_mb", "total_seconds")
    )


def report_finished_job(
    *,
    spec: RunSpec,
    config_path: Path,
    log_path: Path,
    returncode: int,
    timeout_seconds: int,
) -> dict[str, float | int | str]:
    result = read_summary(config_path)
    if returncode == 0 and result["status"] == "ok":
        print(
            "  success | "
            f"{spec.experiment_tag} | "
            f"val_acc={result['val_acc']:.6f} | "
            f"val_auc={result['val_auc']:.6f} | "
            f"val_f1={result['val_f1']:.6f} | "
            f"peak_vram_mb={result['peak_vram_mb']:.1f} | "
            f"total_seconds={result['total_seconds']:.1f}"
        )
    elif returncode == 124:
        print(f"  timeout | {spec.experiment_tag} | after {timeout_seconds}s")
    else:
        print(f"  failed | {spec.experiment_tag} | exit code {returncode}")

    for line in tail_lines(log_path, lines=5):
        print(f"    {line}")
    result["returncode"] = int(returncode)
    result["log_path"] = repo_relative(log_path)
    return result


def choose_pair_status(
    learned: dict[str, Any],
    equal: dict[str, Any],
) -> tuple[str, str, str]:
    learned_ok = is_valid_result(learned)
    equal_ok = is_valid_result(equal)
    if learned_ok and equal_ok:
        learned_acc = float(learned["val_acc"])
        equal_acc = float(equal["val_acc"])
        learned_auc = float(learned["val_auc"])
        equal_auc = float(equal["val_auc"])
        if learned_acc > equal_acc:
            return "keep", "discard", "learned"
        if learned_acc < equal_acc:
            return "discard", "keep", "equal"
        if learned_auc > equal_auc:
            return "keep", "discard", "learned"
        return "discard", "keep", "equal"
    if learned_ok and not equal_ok:
        return "keep", "crash", "learned"
    if not learned_ok and equal_ok:
        return "crash", "keep", "equal"
    return "crash", "crash", "none"


def result_line(
    *,
    commit_hash: str,
    spec: RunSpec,
    result: dict[str, Any],
    status: str,
) -> str:
    valid = is_valid_result(result)
    val_acc = float(result["val_acc"]) if valid else 0.0
    val_auc = float(result["val_auc"]) if valid else 0.0
    val_f1 = float(result["val_f1"]) if valid else 0.0
    memory_gb = float(result["peak_vram_mb"]) / 1024.0 if valid else 0.0
    description = (
        f"{spec.experiment_tag}: "
        f"{'learned decision fusion' if spec.variant == 'learned' else 'fixed equal-weight decision fusion'} "
        f"with AttentionPooling on canonical resnext / seed{spec.seed} in the matched 512x16 20-epoch control"
    )
    return (
        f"{commit_hash}\t{val_acc:.6f}\t{val_auc:.6f}\t{val_f1:.6f}\t{memory_gb:.1f}\t"
        f"{status}\tformal\t{description}\n"
    )


def append_results_rows(
    *,
    commit_hash: str,
    seed: int,
    pair_results: dict[str, dict[str, Any]],
    pair_statuses: dict[str, str],
) -> None:
    results_path = REPO_ROOT / "results.tsv"
    lock_path = Path("/tmp/ankle_results.lock")
    lines = []
    for variant in VARIANTS:
        spec = RunSpec(seed=seed, variant=variant, config_rel=CONFIGS[(seed, variant)])
        lines.append(
            result_line(
                commit_hash=commit_hash,
                spec=spec,
                result=pair_results[variant],
                status=pair_statuses[variant],
            )
        )

    with lock_path.open("w", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        with results_path.open("a", encoding="utf-8") as results_handle:
            results_handle.writelines(lines)
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def safe_metric(result: dict[str, Any], key: str) -> float:
    if not is_valid_result(result):
        return 0.0
    return float(result[key])


def mean_metric(results: dict[int, dict[str, dict[str, Any]]], variant: str, key: str) -> float:
    values = [safe_metric(results[seed][variant], key) for seed in SEEDS if variant in results[seed]]
    return sum(values) / len(values) if values else 0.0


def seed_bullet(seed: int, variant: str, result: dict[str, Any], status: str) -> str:
    spec = RunSpec(seed=seed, variant=variant, config_rel=CONFIGS[(seed, variant)])
    if status == "crash" or not is_valid_result(result):
        return (
            f"- [x] **{spec.experiment_tag}**：`{spec.config_rel}` → **crash**"
            f"（exit={result.get('returncode', 'NA')}；对应 `summary.json` 无有效指标。）"
        )

    peak_vram_gb = float(result["peak_vram_mb"]) / 1024.0
    outcome = "keep" if status == "keep" else "discard"
    explanation = (
        "在 matched 对照里胜出。"
        if status == "keep"
        else "在 matched 对照里落后于另一种 weighting。"
    )
    return (
        f"- [x] **{spec.experiment_tag}**：`{spec.config_rel}` → "
        f"`val_acc={float(result['val_acc']):.16f}`, "
        f"`val_auc={float(result['val_auc']):.16f}`, "
        f"`val_f1={float(result['val_f1']):.16f}`, "
        f"`peak_vram≈{peak_vram_gb:.2f} GiB`, "
        f"`total_seconds≈{float(result['total_seconds']):.1f}` → "
        f"**{outcome}**（{explanation}）"
    )


def build_backlog_section(
    *,
    aggregate: dict[int, dict[str, dict[str, Any]]],
    pair_statuses: dict[int, dict[str, str]],
    jobid: str,
) -> str:
    learned_mean_acc = mean_metric(aggregate, "learned", "val_acc")
    equal_mean_acc = mean_metric(aggregate, "equal", "val_acc")
    learned_mean_auc = mean_metric(aggregate, "learned", "val_auc")
    equal_mean_auc = mean_metric(aggregate, "equal", "val_auc")
    mean_diff = learned_mean_acc - equal_mean_acc

    if learned_mean_acc > equal_mean_acc:
        conclusion = (
            f"在 `resnext + attention pooling` 的 3-seed matched 对照里，"
            f"**learned weighting** 的 mean `val_acc={learned_mean_acc:.16f}`，"
            f"高于 equal-weight 的 `{equal_mean_acc:.16f}`，净提升 `{mean_diff:.16f}`。"
        )
    elif learned_mean_acc < equal_mean_acc:
        conclusion = (
            f"在 `resnext + attention pooling` 的 3-seed matched 对照里，"
            f"**learned weighting** 的 mean `val_acc={learned_mean_acc:.16f}`，"
            f"低于 equal-weight 的 `{equal_mean_acc:.16f}`，净变化 `{mean_diff:.16f}`。"
        )
    else:
        conclusion = (
            f"在 `resnext + attention pooling` 的 3-seed matched 对照里，"
            f"learned 与 equal-weight 的 mean `val_acc` 完全打平，均为 `{learned_mean_acc:.16f}`；"
            f"tie-break 看 mean `val_auc`，learned=`{learned_mean_auc:.16f}`，equal=`{equal_mean_auc:.16f}`。"
        )

    if learned_mean_auc > equal_mean_auc:
        auc_clause = (
            f"同时 learned 的 mean `val_auc={learned_mean_auc:.16f}`，"
            f"高于 equal-weight 的 `{equal_mean_auc:.16f}`。"
        )
    elif learned_mean_auc < equal_mean_auc:
        auc_clause = (
            f"同时 learned 的 mean `val_auc={learned_mean_auc:.16f}`，"
            f"低于 equal-weight 的 `{equal_mean_auc:.16f}`。"
        )
    else:
        auc_clause = f"两者的 mean `val_auc` 也打平在 `{learned_mean_auc:.16f}`。"

    lines = [
        BACKLOG_HEADING,
        "",
        "> **独立 campaign 说明**",
        "> - 这是按人类最新要求补做的 `resnext + attention pooling` 下的 matched 3-seed fusion control：",
        ">   直接比较 learned decision fusion 与 fixed equal-weight。",
        "> - 所有配置保持 `backbone=resnext`、`fusion_type=decision`、`use_attention_pooling=true`、"
        "`image_size=512`、`num_slices_per_view=16`、`trim_edge_slices=2`、`batch_size=2`、"
        "`num_workers=3`、`freeze_layers=3`、`dropout=0.3`、`epochs=20`、`lr=1e-4`、"
        "`weight_decay=1e-4` 不变。",
        f"> - 执行层复用了 `V100q` 的 `node20` allocation `{jobid}`："
        "`1 node / 3 GPU / 15 CPU / 96G / 24h`，节点上无其他用户作业。",
        "",
    ]

    for seed in SEEDS:
        lines.append(seed_bullet(seed, "learned", aggregate[seed]["learned"], pair_statuses[seed]["learned"]))
        lines.append(seed_bullet(seed, "equal", aggregate[seed]["equal"], pair_statuses[seed]["equal"]))

    lines.extend(
        [
            f"- **3-seed mean 对比**：learned 的 mean `val_acc={learned_mean_acc:.16f}`，"
            f"equal-weight 的 mean `val_acc={equal_mean_acc:.16f}`；"
            f"learned 的 mean `val_auc={learned_mean_auc:.16f}`，"
            f"equal-weight 的 mean `val_auc={equal_mean_auc:.16f}`。",
            f"- **控制变量结论**：{conclusion} {auc_clause}",
        ]
    )
    return "\n".join(lines).rstrip()


def replace_table_line(text: str, label: str, value: str) -> str:
    pattern = rf"^\| {re.escape(label)} \|.*\|$"
    replacement = f"| {label} | {value} |"
    return re.sub(pattern, replacement, text, flags=re.MULTILINE)


def update_backlog(
    *,
    aggregate: dict[int, dict[str, dict[str, Any]]],
    pair_statuses: dict[int, dict[str, str]],
    jobid: str,
) -> None:
    backlog_path = REPO_ROOT / "backlog.md"
    lock_path = Path("/tmp/ankle_backlog.lock")
    new_section = build_backlog_section(aggregate=aggregate, pair_statuses=pair_statuses, jobid=jobid)

    learned_mean_acc = mean_metric(aggregate, "learned", "val_acc")
    equal_mean_acc = mean_metric(aggregate, "equal", "val_acc")
    learned_mean_auc = mean_metric(aggregate, "learned", "val_auc")
    equal_mean_auc = mean_metric(aggregate, "equal", "val_auc")

    if learned_mean_acc > equal_mean_acc or (
        math.isclose(learned_mean_acc, equal_mean_acc) and learned_mean_auc > equal_mean_auc
    ):
        next_step = (
            "既然 attention-pooling 下 learned weighting 更强，下一步优先把它与当前 non-attention 的 "
            "minimal learned path 做 matched 对照，拆清收益到底来自 pooling 还是 weighting。"
        )
        last_result = (
            f"keep（这轮 `resnext + attention pooling` 的 3-seed matched 对照里，learned weighting 的 "
            f"mean val_acc={learned_mean_acc:.6f}，对 equal-weight 的 {equal_mean_acc:.6f} 占优；"
            f"mean val_auc 也为 {learned_mean_auc:.6f} 对 {equal_mean_auc:.6f}。）"
        )
    else:
        next_step = (
            "如果主线继续探索 resnext 融合策略，下一步优先拆 attention-pooling 下 learned path 的 "
            "不稳定来源，尤其检查 reliability 分支是否在 attention-pooled view feature 上引入了额外方差。"
        )
        last_result = (
            f"discard（这轮 `resnext + attention pooling` 的 3-seed matched 对照里，learned weighting 的 "
            f"mean val_acc={learned_mean_acc:.6f}，低于或不优于 equal-weight 的 {equal_mean_acc:.6f}；"
            f"mean val_auc 为 {learned_mean_auc:.6f} 对 {equal_mean_auc:.6f}。）"
        )

    last_experiment = (
        f"CMP-DECISION-EQUAL-RESNEXT-ATTNPOOL-512X16-E20-S42/S123/S456"
        f"（V100q node20 allocation {jobid}；1 node / 3 GPU / 15 CPU / 96G / 24h；"
        "固定 resnext + decision + attention pooling + 512x16 + 20 epochs，比较 learned vs equal。)"
    )

    with lock_path.open("w", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        text = backlog_path.read_text(encoding="utf-8")
        section_block = new_section + "\n\n---\n\n"
        if BACKLOG_HEADING in text:
            start = text.index(BACKLOG_HEADING)
            end = text.find("\n---\n", start)
            if end == -1:
                end = len(text)
            else:
                end += len("\n---\n")
            text = text[:start] + section_block + text[end:]
        else:
            marker = "---\n\n"
            insert_at = text.find(marker)
            if insert_at == -1:
                raise SystemExit("Could not find backlog section insertion marker.")
            insert_at += len(marker)
            text = text[:insert_at] + section_block + text[insert_at:]

        text = replace_table_line(text, "上次实验", last_experiment)
        text = replace_table_line(text, "上次结果", last_result)
        text = replace_table_line(text, "下一步", next_step)

        tmp_path = backlog_path.with_suffix(".tmp")
        tmp_path.write_text(text, encoding="utf-8")
        tmp_path.replace(backlog_path)
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def write_aggregate_results(
    *,
    log_dir: Path,
    aggregate: dict[int, dict[str, dict[str, Any]]],
    pair_statuses: dict[int, dict[str, str]],
) -> Path:
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "results": aggregate,
        "pair_statuses": pair_statuses,
        "mean": {
            "learned": {
                "val_acc": mean_metric(aggregate, "learned", "val_acc"),
                "val_auc": mean_metric(aggregate, "learned", "val_auc"),
            },
            "equal": {
                "val_acc": mean_metric(aggregate, "equal", "val_acc"),
                "val_auc": mean_metric(aggregate, "equal", "val_auc"),
            },
        },
    }
    result_path = log_dir / "aggregate_results.json"
    result_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result_path


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not args.jobid:
        raise SystemExit("A Slurm allocation job id is required via --jobid or $SLURM_JOB_ID.")

    python_executable = detect_python_executable([".venv/bin/python"])
    log_dir = REPO_ROOT / args.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    env = build_runtime_env()
    ensure_results_file()

    print_plan(python_executable, args, log_dir)
    if not args.execute:
        return 0

    ensure_visible_gpus(
        jobid=args.jobid,
        python_executable=python_executable,
        env=env,
        min_free_gb=args.min_free_gb,
        max_parallel=args.max_parallel,
    )
    if not args.skip_preflight:
        run_preflight(jobid=args.jobid, python_executable=python_executable, env=env)

    commit_hash = git_short_hash()
    queue = [RunSpec(seed=seed, variant=variant, config_rel=CONFIGS[(seed, variant)]) for seed, variant in RUN_ORDER]
    active: dict[tuple[int, str], tuple[subprocess.Popen[bytes], Path, Path, Any, RunSpec]] = {}
    aggregate: dict[int, dict[str, dict[str, Any]]] = {seed: {} for seed in SEEDS}
    pair_statuses: dict[int, dict[str, str]] = {}
    recorded_pairs: set[int] = set()
    overall_exit_code = 0

    while queue or active:
        while queue and len(active) < args.max_parallel:
            spec = queue.pop(0)
            print(f"Launching {spec.experiment_tag} -> {spec.config_rel}")
            active[(spec.seed, spec.variant)] = launch_training_job(
                jobid=args.jobid,
                python_executable=python_executable,
                env=env,
                spec=spec,
                cpus_per_run=args.cpus_per_run,
                timeout_seconds=args.timeout,
                log_dir=log_dir,
            ) + (spec,)

        any_finished = False
        for key, (process, log_path, config_path, handle, spec) in list(active.items()):
            returncode = process.poll()
            if returncode is None:
                continue
            any_finished = True
            handle.close()
            aggregate[spec.seed][spec.variant] = report_finished_job(
                spec=spec,
                config_path=config_path,
                log_path=log_path,
                returncode=returncode,
                timeout_seconds=args.timeout,
            )
            if returncode != 0:
                overall_exit_code = 1
            active.pop(key)

            if spec.seed not in recorded_pairs and all(
                variant in aggregate[spec.seed] for variant in VARIANTS
            ):
                learned_status, equal_status, winner = choose_pair_status(
                    aggregate[spec.seed]["learned"],
                    aggregate[spec.seed]["equal"],
                )
                pair_statuses[spec.seed] = {
                    "learned": learned_status,
                    "equal": equal_status,
                    "winner": winner,
                }
                append_results_rows(
                    commit_hash=commit_hash,
                    seed=spec.seed,
                    pair_results=aggregate[spec.seed],
                    pair_statuses=pair_statuses[spec.seed],
                )
                print(
                    f"Recorded seed={spec.seed} pair to results.tsv | "
                    f"winner={winner} | learned={learned_status} | equal={equal_status}"
                )
                recorded_pairs.add(spec.seed)

        if not any_finished and active:
            time.sleep(10)

    result_path = write_aggregate_results(
        log_dir=log_dir,
        aggregate=aggregate,
        pair_statuses=pair_statuses,
    )
    update_backlog(aggregate=aggregate, pair_statuses=pair_statuses, jobid=args.jobid)

    print("")
    print(f"Aggregate results written to {repo_relative(result_path)}")
    print(
        "Mean comparison | "
        f"learned val_acc={mean_metric(aggregate, 'learned', 'val_acc'):.6f}, "
        f"equal val_acc={mean_metric(aggregate, 'equal', 'val_acc'):.6f}, "
        f"learned val_auc={mean_metric(aggregate, 'learned', 'val_auc'):.6f}, "
        f"equal val_auc={mean_metric(aggregate, 'equal', 'val_auc'):.6f}"
    )
    return overall_exit_code


if __name__ == "__main__":
    raise SystemExit(main())

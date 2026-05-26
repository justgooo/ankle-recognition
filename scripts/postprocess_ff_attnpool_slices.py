#!/usr/bin/env python
from __future__ import annotations

import argparse
import fcntl
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
STUDY_DIR = REPO_ROOT / "runs/optuna_main_ff_attnpool_slices_3seed"
REPORT_PATH = REPO_ROOT / "autoresearch_logs/ff_attnpool_slices_3seed_summary.json"
BACKLOG_PATH = REPO_ROOT / "backlog.md"
RESULTS_PATH = REPO_ROOT / "results.tsv"
RESULTS_LOCK = Path("/tmp/ankle_results.lock")
BACKLOG_LOCK = Path("/tmp/ankle_backlog.lock")
SEEDS = (42, 123, 456)
SLICE_COUNTS = (8, 12, 16, 20)
ANCHOR = {
    "val_acc": 0.9397163120567376,
    "val_auc": 0.9759090909090910,
    "val_f1": 0.9344151453684922,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate the feature-fusion AttentionPooling slice-count 3-seed campaign."
    )
    parser.add_argument("--study-dir", default=str(STUDY_DIR))
    parser.add_argument("--train-jobid", default="unknown")
    parser.add_argument("--results", default=str(RESULTS_PATH))
    parser.add_argument("--backlog", default=str(BACKLOG_PATH))
    return parser.parse_args()


def numeric(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def git_short_hash() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def load_trial_records(study_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for trial_path in sorted((study_dir / "trials").glob("trial_*/trial.json")):
        with trial_path.open("r", encoding="utf-8") as handle:
            record = json.load(handle)
        record["_trial_json"] = str(trial_path.relative_to(REPO_ROOT))
        records.append(record)
    return records


def seed_from_record(record: dict[str, Any]) -> int | None:
    value = record.get("params", {}).get("seed")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def slices_from_record(record: dict[str, Any]) -> int | None:
    value = record.get("params", {}).get("data.num_slices_per_view")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def is_completed(record: dict[str, Any]) -> bool:
    return (
        record.get("status") == "completed"
        and numeric(record.get("val_accuracy")) is not None
        and numeric(record.get("val_auc")) is not None
        and numeric(record.get("val_f1")) is not None
    )


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def build_group_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_slices: dict[int, list[dict[str, Any]]] = {count: [] for count in SLICE_COUNTS}
    for record in records:
        count = slices_from_record(record)
        if count in by_slices:
            by_slices[count].append(record)

    groups: dict[str, Any] = {}
    for count in SLICE_COUNTS:
        items = sorted(
            by_slices[count],
            key=lambda item: (
                seed_from_record(item) if seed_from_record(item) is not None else -1,
                int(item.get("trial_number") or -1),
            ),
        )
        completed_by_seed: dict[int, dict[str, Any]] = {}
        duplicate_completed_trials = 0
        for item in items:
            seed = seed_from_record(item)
            if seed not in SEEDS or not is_completed(item):
                continue
            if seed in completed_by_seed:
                duplicate_completed_trials += 1
                continue
            completed_by_seed[seed] = item
        completed = [completed_by_seed[seed] for seed in SEEDS if seed in completed_by_seed]

        seed_metrics = []
        for item in completed:
            seed_metrics.append(
                {
                    "trial_number": item.get("trial_number"),
                    "seed": seed_from_record(item),
                    "status": item.get("status"),
                    "val_acc": numeric(item.get("val_accuracy")),
                    "val_auc": numeric(item.get("val_auc")),
                    "val_f1": numeric(item.get("val_f1")),
                    "peak_vram_mb": numeric(item.get("peak_vram_mb")),
                    "total_seconds": numeric(item.get("total_seconds")),
                    "trial_json": item.get("_trial_json"),
                }
            )

        complete = len(completed) == len(SEEDS)
        if complete:
            val_acc = mean([float(item["val_accuracy"]) for item in completed])
            val_auc = mean([float(item["val_auc"]) for item in completed])
            val_f1 = mean([float(item["val_f1"]) for item in completed])
            peak_vram_mb = max(float(item.get("peak_vram_mb") or 0.0) for item in completed)
            total_seconds = sum(float(item.get("total_seconds") or 0.0) for item in completed)
        else:
            val_acc = val_auc = val_f1 = peak_vram_mb = total_seconds = 0.0

        groups[str(count)] = {
            "num_slices_per_view": count,
            "complete": complete,
            "completed_seed_count": len(completed),
            "duplicate_completed_trial_count": duplicate_completed_trials,
            "seed_metrics": seed_metrics,
            "mean": {
                "val_acc": val_acc,
                "val_auc": val_auc,
                "val_f1": val_f1,
                "peak_vram_mb_max": peak_vram_mb,
                "total_seconds_sum": total_seconds,
            },
        }
    return groups


def score_tuple(group: dict[str, Any]) -> tuple[float, float, float]:
    values = group["mean"]
    return (
        float(values["val_acc"]),
        float(values["val_auc"]),
        float(values["val_f1"]),
    )


def beats_anchor(group: dict[str, Any]) -> bool:
    acc, auc, _f1 = score_tuple(group)
    if acc > ANCHOR["val_acc"]:
        return True
    if math.isclose(acc, ANCHOR["val_acc"]) and auc > ANCHOR["val_auc"]:
        return True
    return False


def group_status(group: dict[str, Any], best_count: int) -> str:
    if not group["complete"]:
        return "crash"
    if int(group["num_slices_per_view"]) == best_count and beats_anchor(group):
        return "keep"
    return "discard"


def seed_metric_text(group: dict[str, Any]) -> str:
    parts = []
    for item in group["seed_metrics"]:
        if item["status"] == "completed" and item["val_acc"] is not None:
            parts.append(
                f"s{item['seed']}={item['val_acc']:.6f}/{item['val_auc']:.6f}/{item['val_f1']:.6f}"
            )
        else:
            parts.append(f"s{item['seed']}={item['status']}")
    return ", ".join(parts)


def result_description(group: dict[str, Any], status: str) -> str:
    count = int(group["num_slices_per_view"])
    mean_values = group["mean"]
    if status == "crash":
        actual = (
            f"only {group['completed_seed_count']}/3 seeds completed; "
            f"seed results: {seed_metric_text(group)}"
        )
    else:
        actual = (
            f"seed results: {seed_metric_text(group)}; "
            f"3-seed mean={mean_values['val_acc']:.6f}/"
            f"{mean_values['val_auc']:.6f}/{mean_values['val_f1']:.6f}"
        )
    return (
        f"FF-ATTNPOOL-S{count}-3SEED-FORMAL: "
        "设计思路=turn on learnable slice AttentionPooling in the current ResNeXt feature-fusion "
        f"gated-head anchor and set num_slices_per_view={count}, keeping trim2/freeze3/lr/wd/dropout/clip fixed. "
        "预计改进效果=attention pooling should focus each view token on informative slices; larger slice counts "
        "may expose sparse lesion evidence if the extra context does not add noise. "
        f"实验实际结果={actual}."
    )


def ensure_results_header() -> None:
    if RESULTS_PATH.exists():
        return
    RESULTS_PATH.write_text(
        "commit\tval_acc\tval_auc\tval_f1\tmemory_gb\tstatus\tconfig\tdescription\n",
        encoding="utf-8",
    )


def append_results(groups: dict[str, Any], best_count: int, commit_hash: str, results_path: Path) -> None:
    ensure_results_header()
    existing = results_path.read_text(encoding="utf-8") if results_path.exists() else ""
    lines = []
    for count in SLICE_COUNTS:
        group = groups[str(count)]
        status = group_status(group, best_count)
        key = f"FF-ATTNPOOL-S{count}-3SEED-FORMAL"
        if key in existing:
            continue
        if status == "crash":
            val_acc = val_auc = val_f1 = memory_gb = 0.0
        else:
            mean_values = group["mean"]
            val_acc = float(mean_values["val_acc"])
            val_auc = float(mean_values["val_auc"])
            val_f1 = float(mean_values["val_f1"])
            memory_gb = float(mean_values["peak_vram_mb_max"]) / 1024.0
        lines.append(
            f"{commit_hash}\t{val_acc:.6f}\t{val_auc:.6f}\t{val_f1:.6f}\t"
            f"{memory_gb:.1f}\t{status}\tformal\t{result_description(group, status)}\n"
        )

    if not lines:
        return
    with RESULTS_LOCK.open("w", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        with results_path.open("a", encoding="utf-8") as handle:
            handle.writelines(lines)
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def build_backlog_line(groups: dict[str, Any], best_count: int, train_jobid: str, report_path: Path) -> str:
    fragments = []
    for count in SLICE_COUNTS:
        group = groups[str(count)]
        if group["complete"]:
            values = group["mean"]
            fragments.append(
                f"`S={count}` mean `{values['val_acc']:.6f}/{values['val_auc']:.6f}/{values['val_f1']:.6f}`"
            )
        else:
            fragments.append(f"`S={count}` crash/incomplete `{group['completed_seed_count']}/3`")

    best_group = groups[str(best_count)]
    best_values = best_group["mean"]
    if best_group["complete"] and beats_anchor(best_group):
        conclusion = (
            f"best `S={best_count}` beats current gated-head anchor "
            f"`{ANCHOR['val_acc']:.6f}/{ANCHOR['val_auc']:.6f}/{ANCHOR['val_f1']:.6f}` → **keep**"
        )
    elif best_group["complete"]:
        conclusion = (
            f"best `S={best_count}` remains below current gated-head anchor "
            f"`{ANCHOR['val_acc']:.6f}/{ANCHOR['val_auc']:.6f}/{ANCHOR['val_f1']:.6f}` → **discard family as candidate**"
        )
    else:
        conclusion = "campaign incomplete or crashed → **crash**"

    return (
        "- [x] **FF-ATTNPOOL-SLICE-COUNT-3SEED-FORMAL**：人类要求从 slice pooling / slice count 方向提高当前 "
        "feature-fusion gated-head anchor；使用 `scripts/autoresearch_main.py` + "
        "`configs/optuna_main_search_ff_attnpool_slices_3seed.yaml` 跑 4 组固定 trial："
        "`use_attention_pooling=true` 且 `num_slices_per_view=8/12/16/20`，每组 seeds `42/123/456`，共 12 次 formal 训练；"
        "保持 `trim_edge_slices=2`, `freeze_layers=3`, `lr=1e-4`, `wd=1e-4`, `dropout=0.25`, `clip=1.0`。"
        "设计思路：把每视角 slice mean pooling 改为可学习 AttentionPooling，并比较更多切片是否给 pooling scorer 提供更完整局部证据。"
        "预计改进效果：如果关键病灶只落在少数切片或 8-slice 均匀采样稀释了局部特征，attention pooling 应提高 view token 质量。"
        f"实验实际结果：Slurm job `{train_jobid}` 完成；{'; '.join(fragments)}；"
        f"best `S={best_count}` mean `{best_values['val_acc']:.6f}/{best_values['val_auc']:.6f}/{best_values['val_f1']:.6f}`；"
        f"{conclusion}；report=`{report_path.relative_to(REPO_ROOT)}`。"
    )


def update_backlog(groups: dict[str, Any], best_count: int, train_jobid: str, backlog_path: Path, report_path: Path) -> None:
    if not backlog_path.exists():
        return
    new_line = build_backlog_line(groups, best_count, train_jobid, report_path)
    with BACKLOG_LOCK.open("w", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        text = backlog_path.read_text(encoding="utf-8")
        text = re.sub(
            r"^- \[[ x]\] \*\*FF-ATTNPOOL-SLICE-COUNT-3SEED-FORMAL\*\*：.*$",
            new_line,
            text,
            count=1,
            flags=re.MULTILINE,
        )
        status_line = (
            "- **Agent 状态（2026-05-26 SGT）**：当前主线已切到 feature-fusion exploration；"
            f"last completed=`FF-ATTNPOOL-SLICE-COUNT-3SEED-FORMAL` (Slurm job `{train_jobid}`, "
            f"best `S={best_count}` 3-seed mean `{groups[str(best_count)]['mean']['val_acc']:.6f}/"
            f"{groups[str(best_count)]['mean']['val_auc']:.6f}/"
            f"{groups[str(best_count)]['mean']['val_f1']:.6f}`); "
            "last retained feature-fusion evidence remains full gated-head "
            "(`fe42d90`, mean `0.939716/0.975909/0.934415`) unless the attention-pooling row is marked keep."
        )
        text = re.sub(
            r"^- \*\*Agent 状态（2026-05-26 SGT）\*\*：.*$",
            status_line,
            text,
            count=1,
            flags=re.MULTILINE,
        )
        backlog_path.write_text(text, encoding="utf-8")
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def main() -> int:
    args = parse_args()
    study_dir = Path(args.study_dir)
    if not study_dir.is_absolute():
        study_dir = REPO_ROOT / study_dir
    results_path = Path(args.results)
    if not results_path.is_absolute():
        results_path = REPO_ROOT / results_path
    backlog_path = Path(args.backlog)
    if not backlog_path.is_absolute():
        backlog_path = REPO_ROOT / backlog_path

    records = load_trial_records(study_dir)
    groups = build_group_summary(records)
    complete_groups = {
        count: group for count, group in groups.items() if group["complete"]
    }
    if complete_groups:
        best_count = int(max(complete_groups.items(), key=lambda item: score_tuple(item[1]))[0])
    else:
        best_count = SLICE_COUNTS[0]

    report = {
        "study_dir": str(study_dir.relative_to(REPO_ROOT)),
        "train_jobid": args.train_jobid,
        "anchor": ANCHOR,
        "best_num_slices_per_view": best_count,
        "groups": groups,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    commit_hash = git_short_hash()
    append_results(groups, best_count, commit_hash, results_path)
    update_backlog(groups, best_count, args.train_jobid, backlog_path, REPORT_PATH)

    best = groups[str(best_count)]["mean"]
    print(
        f"best_slices={best_count} "
        f"mean={best['val_acc']:.6f}/{best['val_auc']:.6f}/{best['val_f1']:.6f} "
        f"report={REPORT_PATH.relative_to(REPO_ROOT)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

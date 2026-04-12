from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

from optuna_workflow import REPO_ROOT, VALID_TRIAL_STATUSES, load_trial_records, read_json, resolve_path


def numeric_or_none(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


def short_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_table(records: list[dict[str, Any]]) -> str:
    headers = [
        "trial",
        "status",
        "val_acc",
        "threshold_acc",
        "val_auc",
        "val_f1",
        "val_loss",
        "train_loss",
        "seconds",
        "vram_mb",
        "params",
    ]
    rows = [headers]
    for record in sorted(records, key=lambda item: item.get("trial_number", -1)):
        params = record.get("params", {})
        param_text = ", ".join(f"{key}={value}" for key, value in sorted(params.items()))
        rows.append(
            [
                str(record.get("trial_number", "-")),
                str(record.get("status", "-")),
                short_value(record.get("val_accuracy")),
                short_value(record.get("threshold_val_accuracy")),
                short_value(record.get("val_auc")),
                short_value(record.get("val_f1")),
                short_value(record.get("val_loss")),
                short_value(record.get("train_loss")),
                short_value(record.get("total_seconds")),
                short_value(record.get("peak_vram_mb")),
                param_text or "-",
            ]
        )

    widths = [max(len(str(row[index])) for row in rows) for index in range(len(headers))]
    rendered = []
    for index, row in enumerate(rows):
        rendered.append(" | ".join(str(cell).ljust(widths[idx]) for idx, cell in enumerate(row)))
        if index == 0:
            rendered.append("-+-".join("-" * width for width in widths))
    return "\n".join(rendered)


def valid_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if record.get("status") in VALID_TRIAL_STATUSES and numeric_or_none(record.get("val_accuracy")) is not None
    ]


def read_text_tail(path: Path, lines: int = 40) -> str:
    if not path.exists():
        return ""
    content = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return "\n".join(content[-lines:])


def watchdog_snapshot(study_dir: Path) -> dict[str, Any]:
    records = load_trial_records(study_dir)
    running = [record for record in records if record.get("status") == "running"]

    candidate_paths = [
        study_dir / "study_status.json",
        study_dir / "study.sqlite3",
    ]
    for record in records:
        paths = record.get("paths", {})
        for key in ("train_log", "threshold_log", "summary_json", "threshold_eval_json", "history_json"):
            raw_path = paths.get(key)
            if raw_path:
                candidate_paths.append(resolve_path(raw_path))

    latest_progress = None
    for path in candidate_paths:
        if not path.exists():
            continue
        mtime = path.stat().st_mtime
        if latest_progress is None or mtime > latest_progress:
            latest_progress = mtime

    failure_signals: list[str] = []
    timeout_signals: list[str] = []
    threshold_warning_signals: list[str] = []
    for record in running:
        paths = record.get("paths", {})
        train_tail = read_text_tail(resolve_path(paths.get("train_log", ""))) if paths.get("train_log") else ""
        train_lowered = train_tail.lower()
        if "timeout" in train_lowered:
            timeout_signals.append(str(record.get("trial_number", "-")))
        if "exit_code=" in train_lowered or "traceback" in train_lowered:
            failure_signals.append(str(record.get("trial_number", "-")))

        threshold_tail = read_text_tail(resolve_path(paths.get("threshold_log", ""))) if paths.get("threshold_log") else ""
        threshold_lowered = threshold_tail.lower()
        if threshold_lowered and (
            "timeout" in threshold_lowered or "exit_code=" in threshold_lowered or "traceback" in threshold_lowered
        ):
            threshold_warning_signals.append(str(record.get("trial_number", "-")))

    progress_age_seconds = None
    if latest_progress is not None:
        progress_age_seconds = max(0.0, time.time() - latest_progress)

    return {
        "records": records,
        "running_trials": running,
        "progress_age_seconds": progress_age_seconds,
        "timeout_signals": timeout_signals,
        "failure_signals": failure_signals,
        "threshold_warning_signals": threshold_warning_signals,
    }


def metric_tuple(record: dict[str, Any], primary_metric: str = "val_accuracy") -> tuple[float, float] | None:
    primary = numeric_or_none(record.get(primary_metric))
    if primary is None:
        return None
    auc = numeric_or_none(record.get("val_auc"))
    if auc is None:
        auc = float("-inf")
    return primary, auc


def best_record(records: list[dict[str, Any]], metric: str, fastest: bool = False) -> dict[str, Any] | None:
    if fastest:
        candidates = [record for record in records if numeric_or_none(record.get(metric)) is not None]
        if not candidates:
            return None
        return min(candidates, key=lambda item: float(item[metric]))

    if metric == "val_accuracy":
        candidates = [record for record in records if metric_tuple(record, "val_accuracy") is not None]
        if not candidates:
            return None
        return max(candidates, key=lambda item: metric_tuple(item, "val_accuracy"))

    candidates = [record for record in records if numeric_or_none(record.get(metric)) is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda item: float(item[metric]))


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def analyze_numeric_param(name: str, pairs: list[tuple[float, float]]) -> str | None:
    if len(pairs) < 4:
        return None
    xs = [item[0] for item in pairs]
    ys = [item[1] for item in pairs]
    mean_x = mean(xs)
    mean_y = mean(ys)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x == 0 or var_y == 0:
        return None
    corr = cov / math.sqrt(var_x * var_y)
    if abs(corr) < 0.35:
        return None

    best_pair = max(pairs, key=lambda item: item[1])
    if abs(corr) >= 0.55:
        sensitivity = "strong"
    else:
        sensitivity = "moderate"
    direction = "higher" if corr > 0 else "lower"
    return (
        f"`{name}` shows {sensitivity} sensitivity in this study "
        f"(corr={corr:.2f}); {direction} values worked better, with the best valid trial near {best_pair[0]:.6g}."
    )


def analyze_categorical_param(name: str, grouped: dict[str, list[float]]) -> str | None:
    filtered = {key: values for key, values in grouped.items() if len(values) >= 2}
    if len(filtered) < 2:
        return None
    ranked = sorted(filtered.items(), key=lambda item: mean(item[1]), reverse=True)
    best_key, best_values = ranked[0]
    worst_key, worst_values = ranked[-1]
    gap = mean(best_values) - mean(worst_values)
    if gap < 0.01:
        return None
    return (
        f"`{name}` is trending toward `{best_key}` in this search "
        f"(mean val_accuracy {mean(best_values):.4f} vs `{worst_key}` at {mean(worst_values):.4f})."
    )


def analyze_failures(records: list[dict[str, Any]]) -> list[str]:
    messages: list[str] = []
    failed = [record for record in records if record.get("status") not in VALID_TRIAL_STATUSES]
    if not failed:
        return messages

    grouped: dict[str, dict[str, int]] = {}
    for record in failed:
        params = record.get("params", {})
        for key, value in params.items():
            bucket = grouped.setdefault(key, {})
            label = str(value)
            bucket[label] = bucket.get(label, 0) + 1

    for name, counts in grouped.items():
        label, count = max(counts.items(), key=lambda item: item[1])
        if count >= 2:
            messages.append(f"`{name}={label}` is over-represented in failed trials ({count} failures).")
    return messages


def build_suggestions(records: list[dict[str, Any]]) -> list[str]:
    valid = valid_records(records)
    suggestions: list[str] = []
    if len(valid) < 3:
        return suggestions

    best_acc = max(float(record["val_accuracy"]) for record in valid)
    worst_acc = min(float(record["val_accuracy"]) for record in valid)
    if best_acc - worst_acc >= 0.03:
        suggestions.append(
            f"The current candidate is sensitive to hyperparameters in this range (best val_accuracy {best_acc:.4f}, worst {worst_acc:.4f})."
        )

    param_names = sorted({key for record in valid for key in record.get("params", {})})
    for name in param_names:
        values = [record.get("params", {}).get(name) for record in valid if name in record.get("params", {})]
        metric_values = [float(record["val_accuracy"]) for record in valid if name in record.get("params", {})]
        numeric_pairs = []
        categorical_groups: dict[str, list[float]] = {}
        for value, metric in zip(values, metric_values):
            numeric = numeric_or_none(value)
            if numeric is not None:
                numeric_pairs.append((numeric, metric))
            categorical_groups.setdefault(str(value), []).append(metric)

        numeric_message = analyze_numeric_param(name, numeric_pairs)
        if numeric_message:
            suggestions.append(numeric_message)
            continue

        categorical_message = analyze_categorical_param(name, categorical_groups)
        if categorical_message:
            suggestions.append(categorical_message)

    suggestions.extend(analyze_failures(records))
    return suggestions[:8]


def build_report(study_dir: Path) -> tuple[str, dict[str, Any]]:
    records = load_trial_records(study_dir)
    valid = valid_records(records)
    status_data = read_json(study_dir / "study_status.json") or {}
    best_accuracy = best_record(valid, "val_accuracy")
    best_auc = best_record(valid, "val_auc")
    fastest = best_record(valid, "total_seconds", fastest=True)
    crashes = [record for record in records if record.get("status") in {"crash", "timeout", "oom", "invalid"}]
    degraded = []
    if best_accuracy:
        best_value = float(best_accuracy["val_accuracy"])
        for record in valid:
            current = float(record["val_accuracy"])
            if best_value - current >= 0.03:
                degraded.append(record)

    report = {
        "study_dir": str(study_dir),
        "study_state": status_data.get("state", "unknown"),
        "trial_count": len(records),
        "valid_count": len(valid),
        "best_by_accuracy": best_accuracy,
        "best_by_auc": best_auc,
        "fastest_valid": fastest,
        "crashes": crashes,
        "threshold_warnings": [
            record
            for record in records
            if record.get("threshold_status") in {"crash", "timeout"}
        ],
        "degraded_trials": degraded,
        "suggestions": build_suggestions(records),
        "selection_rule": "val_accuracy_then_val_auc",
    }

    parts = []
    parts.append(f"Study: {study_dir}")
    parts.append(f"State: {report['study_state']}")
    parts.append(f"Trials: {report['trial_count']} total, {report['valid_count']} valid")
    parts.append("")
    parts.append("Trial summary")
    parts.append(render_table(records))
    parts.append("")

    if best_accuracy:
        parts.append(
            "Best by accuracy: "
            f"trial {best_accuracy['trial_number']} | val_acc={best_accuracy['val_accuracy']:.4f} | "
            f"val_auc={short_value(best_accuracy.get('val_auc'))} | "
            f"params={best_accuracy.get('params', {})}"
        )
    if best_auc:
        parts.append(
            "Auxiliary best by AUC: "
            f"trial {best_auc['trial_number']} | val_auc={best_auc['val_auc']:.4f} | "
            f"params={best_auc.get('params', {})}"
        )
    if fastest:
        parts.append(
            "Fastest valid: "
            f"trial {fastest['trial_number']} | total_seconds={fastest['total_seconds']:.1f} | "
            f"val_acc={fastest.get('val_accuracy')}"
        )

    if crashes:
        parts.append("")
        parts.append("Crash / timeout / OOM")
        for record in crashes:
            parts.append(
                f"- trial {record['trial_number']}: status={record['status']} | reason={record.get('failure_reason', '-')}"
            )

    threshold_warnings = report["threshold_warnings"]
    if threshold_warnings:
        parts.append("")
        parts.append("Auxiliary threshold warnings")
        for record in threshold_warnings:
            parts.append(
                f"- trial {record['trial_number']}: threshold_status={record.get('threshold_status', '-')} | "
                f"reason={record.get('threshold_failure_reason', '-')}"
            )

    if degraded:
        parts.append("")
        parts.append("Clearly degraded valid trials")
        for record in degraded:
            parts.append(
                f"- trial {record['trial_number']}: val_acc={record['val_accuracy']:.4f} | params={record.get('params', {})}"
            )

    suggestions = report["suggestions"]
    if suggestions:
        parts.append("")
        parts.append("Suggestions for AutoResearch")
        for suggestion in suggestions:
            parts.append(f"- {suggestion}")

    return "\n".join(parts), report


def write_reports(study_dir: Path, text: str, report: dict[str, Any]) -> None:
    report_dir = study_dir / "monitor"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "report.txt").write_text(text, encoding="utf-8")
    (report_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Optuna trial results for AutoResearch.")
    parser.add_argument("--study-dir", required=True, help="Study directory created by scripts/run_optuna_*.py.")
    parser.add_argument("--watch", action="store_true", help="Poll the study directory until it finishes.")
    parser.add_argument("--interval-seconds", type=int, default=30, help="Polling interval when --watch is enabled.")
    parser.add_argument(
        "--stale-seconds",
        type=int,
        default=1800,
        help="Exit watch mode if study progress files stop changing for this long.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    study_dir = resolve_path(args.study_dir)
    while True:
        text, report = build_report(study_dir)
        print(text)
        write_reports(study_dir, text, report)
        if not args.watch:
            break

        watchdog = watchdog_snapshot(study_dir)
        state = report.get("study_state")
        if state == "completed":
            print("\nWatch exit: study completed.")
            break
        if state in {"failed", "timeout", "crash", "aborted", "stale"}:
            print(f"\nWatch exit: study state is {state}.")
            break
        if watchdog["timeout_signals"]:
            joined = ", ".join(watchdog["timeout_signals"])
            print(f"\nWatch exit: timeout signal detected in running trial logs ({joined}).")
            break
        if watchdog["failure_signals"]:
            joined = ", ".join(watchdog["failure_signals"])
            print(f"\nWatch exit: failure signal detected in running trial logs ({joined}).")
            break

        if watchdog["threshold_warning_signals"]:
            joined = ", ".join(watchdog["threshold_warning_signals"])
            print(f"\nWatch warning: auxiliary threshold evaluation reported issues ({joined}).")

        progress_age_seconds = watchdog["progress_age_seconds"]
        if progress_age_seconds is not None:
            no_running_trials = not watchdog["running_trials"]
            if no_running_trials and report.get("trial_count", 0) > 0 and progress_age_seconds >= max(args.interval_seconds * 2, 10):
                print(
                    "\nWatch exit: no running trials remain, but the study never transitioned to completed "
                    f"(last progress {progress_age_seconds:.0f}s ago)."
                )
                break
            if progress_age_seconds >= max(args.stale_seconds, args.interval_seconds):
                print(f"\nWatch exit: study appears stale (last progress {progress_age_seconds:.0f}s ago).")
                break

        print("\n" + "=" * 80 + "\n")
        time.sleep(max(args.interval_seconds, 5))


if __name__ == "__main__":
    main()

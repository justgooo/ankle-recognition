from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from script_runtime import REPO_ROOT, read_output_dir, read_yaml, repo_relative


MANIFEST_PATH = REPO_ROOT / "configs" / "feature_fusion_structure_5round" / "manifest.json"
SUMMARY_PATH = REPO_ROOT / "autoresearch_logs" / "feature_fusion_structure_5round_summary.json"
REFERENCE = {
    "val_acc": 0.9397163120567376,
    "val_auc": 0.9759090909090910,
    "val_f1": 0.9344151453684922,
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return parsed


def summarize_config(config_path: Path) -> dict[str, Any]:
    config = read_yaml(config_path)
    output_dir = read_output_dir(config_path)
    summary_path = output_dir / "summary.json"
    result: dict[str, Any] = {
        "seed": int(config["seed"]),
        "config": repo_relative(config_path),
        "output_dir": repo_relative(output_dir),
        "summary_path": repo_relative(summary_path),
        "status": "missing",
    }
    if not summary_path.exists():
        return result
    summary = read_json(summary_path)
    best_val = summary.get("best_val") or {}
    runtime = summary.get("runtime") or {}
    result.update(
        {
            "status": "ok",
            "val_acc": safe_float(best_val.get("accuracy")),
            "val_auc": safe_float(best_val.get("auc")),
            "val_f1": safe_float(best_val.get("f1")),
            "peak_vram_mb": safe_float(runtime.get("peak_vram_mb")),
            "total_seconds": safe_float(runtime.get("total_seconds")),
        }
    )
    return result


def mean_metric(runs: list[dict[str, Any]], key: str) -> float:
    values = [float(run[key]) for run in runs if run.get("status") == "ok" and key in run]
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def format_metric(value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    return f"{value:.15f}"


def main() -> int:
    manifest = read_json(MANIFEST_PATH)
    variants: list[dict[str, Any]] = []
    for variant in manifest.get("variants", []):
        runs = [summarize_config(REPO_ROOT / config) for config in variant["configs"]]
        mean = {
            "val_acc": mean_metric(runs, "val_acc"),
            "val_auc": mean_metric(runs, "val_auc"),
            "val_f1": mean_metric(runs, "val_f1"),
            "peak_vram_mb": mean_metric(runs, "peak_vram_mb"),
        }
        ok_count = sum(1 for run in runs if run.get("status") == "ok")
        status = "keep" if ok_count == len(runs) and mean["val_acc"] > REFERENCE["val_acc"] else "discard"
        if ok_count != len(runs):
            status = "crash"
        payload = dict(variant)
        payload.update(
            {
                "runs": runs,
                "ok_count": ok_count,
                "mean": mean,
                "delta_vs_reference": {
                    key: mean[key] - REFERENCE[key]
                    for key in ("val_acc", "val_auc", "val_f1")
                    if math.isfinite(mean[key])
                },
                "status": status,
            }
        )
        variants.append(payload)

    report = {
        "reference": REFERENCE,
        "variants": variants,
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {repo_relative(SUMMARY_PATH)}")
    for variant in variants:
        mean = variant["mean"]
        print(
            f"{variant['title']}\t{variant['status']}\t"
            f"{variant['ok_count']}/3\t"
            f"{format_metric(mean['val_acc'])}\t"
            f"{format_metric(mean['val_auc'])}\t"
            f"{format_metric(mean['val_f1'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

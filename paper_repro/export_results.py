from __future__ import annotations

import argparse
import csv
import fcntl
import json
from datetime import datetime
from pathlib import Path

from paper_repro.common import load_manifest, read_json


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "paper_repro" / "papers.yaml"
EXPORT_DIR = REPO_ROOT / "paper_repro" / "exports"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export paper reproduction summaries.")
    parser.add_argument("--round", type=str, default=None)
    parser.add_argument("--append-root-results", action="store_true")
    return parser.parse_args()


def gather_records(round_name: str | None = None) -> list[dict[str, str]]:
    manifest = load_manifest(MANIFEST_PATH)
    paper_ids = list(manifest["papers"].keys())
    if round_name is not None:
        paper_ids = list(manifest["rounds"][round_name])

    records: list[dict[str, str]] = []
    for paper_id in paper_ids:
        paper_meta = manifest["papers"][paper_id]
        config_path = REPO_ROOT / paper_meta["config"]
        summary_path = REPO_ROOT / paper_meta["output_dir"] / "summary.json"
        if not summary_path.exists():
            records.append(
                {
                    "paper_id": paper_id,
                    "status": "missing_summary",
                    "config_path": str(config_path),
                    "summary_path": str(summary_path),
                }
            )
            continue
        summary = read_json(summary_path)
        best_val = summary.get("best_val") or {}
        runtime = summary.get("runtime") or {}
        records.append(
            {
                "paper_id": paper_id,
                "round": paper_meta["round"],
                "title": paper_meta["title"],
                "config_path": str(config_path),
                "summary_path": str(summary_path),
                "val_acc": f"{float(best_val.get('accuracy', 0.0)):.15f}",
                "val_auc": f"{float(best_val.get('auc', 0.0)):.15f}",
                "val_f1": f"{float(best_val.get('f1', 0.0)):.15f}",
                "peak_vram_gb": f"{float(runtime.get('peak_vram_mb', 0.0)) / 1024:.1f}",
                "status": "ok",
                "budget": paper_meta["budget"],
                "source_doc": paper_meta["source_doc"],
                "reproduction_level": paper_meta["reproduction_level"],
                "description": paper_meta["description"],
            }
            )
    return records


def assign_round_status(records: list[dict[str, str]]) -> None:
    grouped: dict[str, list[dict[str, str]]] = {}
    for record in records:
        if record.get("status") != "ok":
            continue
        round_name = record.get("round", "all")
        grouped.setdefault(round_name, []).append(record)

    for group_records in grouped.values():
        for record in group_records:
            record["status"] = "discard"
        winner = max(
            group_records,
            key=lambda item: (
                float(item.get("val_acc", 0.0)),
                float(item.get("val_auc", 0.0)),
                -float(item.get("peak_vram_gb", 0.0)),
            ),
        )
        winner["status"] = "keep"


def write_exports(records: list[dict[str, str]], round_name: str | None) -> tuple[Path, Path]:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = round_name or "all"
    tsv_path = EXPORT_DIR / f"paper_repro_{suffix}_{stamp}.tsv"
    md_path = EXPORT_DIR / f"paper_repro_{suffix}_{stamp}.md"
    fieldnames = [
        "paper_id",
        "round",
        "title",
        "val_acc",
        "val_auc",
        "val_f1",
        "peak_vram_gb",
        "status",
        "budget",
        "source_doc",
        "reproduction_level",
        "description",
        "config_path",
        "summary_path",
    ]
    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for record in records:
            writer.writerow(record)

    lines = [
        "| paper_id | round | val_acc | val_auc | val_f1 | peak_vram_gb | status | description |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for record in records:
        lines.append(
            f"| {record.get('paper_id', '')} | {record.get('round', '')} | "
            f"{record.get('val_acc', 'n/a')} | {record.get('val_auc', 'n/a')} | "
            f"{record.get('val_f1', 'n/a')} | {record.get('peak_vram_gb', 'n/a')} | "
            f"{record.get('status', '')} | {record.get('description', '')} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tsv_path, md_path


def append_root_results(records: list[dict[str, str]]) -> None:
    results_path = REPO_ROOT / "results.tsv"
    if not results_path.exists():
        results_path.write_text(
            "commit\tval_acc\tval_auc\tval_f1\tmemory_gb\tstatus\tconfig\tdescription\n",
            encoding="utf-8",
        )
    with results_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        for record in records:
            if record.get("status") not in {"keep", "discard"}:
                continue
            line = (
                f"paperrepro-{record['paper_id']}\t{record['val_acc']}\t{record['val_auc']}\t"
                f"{record['val_f1']}\t{record['peak_vram_gb']}\t{record['status']}\t{record['budget']}\t"
                f"{record['description']}\n"
            )
            handle.write(line)
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def main() -> None:
    args = parse_args()
    records = gather_records(args.round)
    assign_round_status(records)
    tsv_path, md_path = write_exports(records, args.round)
    if args.append_root_results:
        append_root_results(records)
    print(f"TSV export: {tsv_path}")
    print(f"MD export: {md_path}")
    print(json.dumps(records, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

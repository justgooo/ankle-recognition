from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print key metrics from a run summary.json file.")
    parser.add_argument("summary_path", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = json.loads(args.summary_path.read_text(encoding="utf-8"))

    best_val = summary.get("best_val", {})
    runtime = summary.get("runtime", {})

    print(f"val_acc={float(best_val.get('accuracy', 0.0)):.15f}")
    print(f"val_auc={float(best_val.get('auc', 0.0)):.15f}")
    print(f"val_f1={float(best_val.get('f1', 0.0)):.15f}")
    print(f"peak_vram_gb={float(runtime.get('peak_vram_mb', 0.0)) / 1024:.1f}")


if __name__ == "__main__":
    main()

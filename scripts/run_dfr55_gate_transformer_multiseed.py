from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from run_dfr44_evidence_gate_multiseed import main as run_multiseed


if __name__ == "__main__":
    default_configs = [
        "configs/cmp_resnext_decision_256x8_dfr55_gate_transformer_context_formal_s42.yaml",
        "configs/cmp_resnext_decision_256x8_dfr55_gate_transformer_context_formal_s123.yaml",
        "configs/cmp_resnext_decision_256x8_dfr55_gate_transformer_context_formal_s456.yaml",
    ]
    if "--configs" not in sys.argv:
        sys.argv.extend(["--configs", *default_configs])
    if "--log-dir" not in sys.argv:
        sys.argv.extend(["--log-dir", "autoresearch_logs/dfr55_gate_transformer_multiseed"])
    raise SystemExit(run_multiseed())

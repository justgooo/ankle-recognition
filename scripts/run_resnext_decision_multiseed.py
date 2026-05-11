from __future__ import annotations

import sys

from run_dfr44_evidence_gate_multiseed import main as run_multiseed


if __name__ == "__main__":
    if "--configs" not in sys.argv:
        raise SystemExit(
            "run_resnext_decision_multiseed.py requires --configs with the three formal seed configs."
        )
    raise SystemExit(run_multiseed())

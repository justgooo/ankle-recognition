from __future__ import annotations

# Compatibility wrapper: prefer this name in coordinator prompts, but keep the
# canonical implementation in optuna_main.py.
from optuna_main import main


if __name__ == "__main__":
    main()

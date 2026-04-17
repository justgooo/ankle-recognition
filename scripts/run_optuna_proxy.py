from __future__ import annotations

# Legacy compatibility wrapper: on 24GB+ GPUs the default lane is main/formal.
from optuna_main import main


if __name__ == "__main__":
    main()

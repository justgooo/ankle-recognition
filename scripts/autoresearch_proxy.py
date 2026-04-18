from __future__ import annotations

# Compatibility wrapper: prefer this name in coordinator prompts, but keep the
# canonical implementation in optuna_proxy.py.
from slurm_optuna_launcher import maybe_dispatch_to_slurm_jobs
from optuna_proxy import main


if __name__ == "__main__":
    if not maybe_dispatch_to_slurm_jobs(
        entrypoint="scripts/optuna_proxy.py",
        default_search_config="configs/optuna_proxy_search.yaml",
    ):
        main()

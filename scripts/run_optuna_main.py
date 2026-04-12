from __future__ import annotations

from optuna_workflow import build_cli, run_study


def main() -> None:
    args = build_cli(
        default_config="configs/optuna_main_search.yaml",
        description="Run a deeper Optuna study or confirmation pass on the formal candidate.",
    )
    study_root = run_study(
        search_config_path=args.search_config,
        source_study_dir=args.source_study_dir,
        top_k=args.top_k,
        max_trials_override=args.trials,
        resume=args.resume,
    )
    print(f"Study finished: {study_root}")


if __name__ == "__main__":
    main()

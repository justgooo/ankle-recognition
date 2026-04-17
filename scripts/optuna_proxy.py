from __future__ import annotations

from optuna_workflow import build_cli, run_study_adaptive


def main() -> None:
    args = build_cli(
        default_config="configs/optuna_proxy_search.yaml",
        description="Run the legacy low-memory AutoResearch proxy Optuna study when a cheaper fallback is needed.",
    )
    study_root = run_study_adaptive(
        search_config_path=args.search_config,
        source_study_dir=args.source_study_dir,
        top_k=args.top_k,
        max_trials_override=args.trials,
        resume=args.resume,
        sequential=args.sequential,
        gpu_ids=args.gpu_ids,
        max_workers=args.max_workers,
        max_used_memory_mb=args.max_used_memory_mb,
        max_utilization=args.max_utilization,
        worker_cooldown_seconds=args.worker_cooldown_seconds,
    )
    print(f"Study finished: {study_root}")


if __name__ == "__main__":
    main()

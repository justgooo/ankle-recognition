from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from script_runtime import REPO_ROOT, read_output_dir, read_yaml, repo_relative


DEFAULT_CONFIGS = [
    "configs/cmp_resnext_decision_256x8_dfr44_evidence_aware_gate_formal_s42.yaml",
    "configs/cmp_resnext_decision_256x8_dfr44_evidence_aware_gate_formal_s123.yaml",
    "configs/cmp_resnext_decision_256x8_dfr44_evidence_aware_gate_formal_s456.yaml",
]


@dataclass(frozen=True)
class RunSpec:
    config_path: Path
    output_dir: Path
    seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DFR-44 evidence-aware gate formal configs on a 3-GPU Slurm allocation."
    )
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--python", default=".venv/bin/python")
    parser.add_argument("--cpus-per-run", type=int, default=12)
    parser.add_argument("--max-parallel", type=int, default=3)
    parser.add_argument("--min-free-gb", type=float, default=20.0)
    parser.add_argument("--log-dir", default="autoresearch_logs/dfr44_evidence_gate_multiseed")
    parser.add_argument("--skip-telemetry", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_path(path_like: str) -> Path:
    path = Path(path_like)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def command_text(command: list[str]) -> str:
    return " ".join(command)


def build_specs(configs: list[str]) -> list[RunSpec]:
    specs: list[RunSpec] = []
    for config in configs:
        config_path = resolve_path(config)
        payload = read_yaml(config_path)
        specs.append(
            RunSpec(
                config_path=config_path,
                output_dir=read_output_dir(config_path),
                seed=int(payload["seed"]),
            )
        )
    return specs


def check_visible_gpus(min_count: int, min_free_gb: float) -> None:
    import torch

    count = torch.cuda.device_count()
    print(f"visible_gpus={count}")
    if count < min_count:
        raise SystemExit(f"Need at least {min_count} visible GPUs, got {count}.")
    min_free_bytes = min_free_gb * 1024**3
    for index in range(count):
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        free_gb = free_bytes / 1024**3
        total_gb = total_bytes / 1024**3
        print(
            f"gpu={index} name={torch.cuda.get_device_name(index)} "
            f"free_gb={free_gb:.2f} total_gb={total_gb:.2f}"
        )
        if index < min_count and free_bytes < min_free_bytes:
            raise SystemExit(
                f"GPU {index} has only {free_gb:.2f} GiB free; need {min_free_gb:.2f} GiB."
            )


def run_config(spec: RunSpec, args: argparse.Namespace, log_dir: Path) -> int:
    log_path = log_dir / f"seed{spec.seed}_train.log"
    command = [
        "srun",
        "--exclusive",
        "-N1",
        "-n1",
        "--gres=gpu:1",
        f"-c{args.cpus_per_run}",
        f"--output={log_path}",
        args.python,
        "scripts/run_train_with_config_env.py",
        "--config",
        repo_relative(spec.config_path),
        "--python",
        args.python,
    ]
    print(f"launch seed={spec.seed} config={repo_relative(spec.config_path)}")
    print(command_text(command))
    if args.dry_run:
        return 0
    result = subprocess.run(command, cwd=REPO_ROOT, check=False)
    print(f"train seed={spec.seed} exit={result.returncode} log={repo_relative(log_path)}")
    return int(result.returncode)


def run_telemetry(spec: RunSpec, args: argparse.Namespace, log_dir: Path) -> int:
    log_path = log_dir / f"seed{spec.seed}_telemetry.log"
    command = [
        args.python,
        "scripts/analyze_fusion_weights.py",
        "--config",
        repo_relative(spec.config_path),
        "--checkpoint",
        repo_relative(spec.output_dir / "best.pt"),
    ]
    print(f"telemetry seed={spec.seed}")
    print(command_text(command))
    if args.dry_run:
        return 0
    with log_path.open("w", encoding="utf-8") as handle:
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    print(f"telemetry seed={spec.seed} exit={result.returncode} log={repo_relative(log_path)}")
    return int(result.returncode)


def read_result(spec: RunSpec) -> dict[str, Any]:
    summary_path = spec.output_dir / "summary.json"
    fusion_path = spec.output_dir / "fusion_weight_analysis.json"
    result: dict[str, Any] = {
        "seed": spec.seed,
        "config": repo_relative(spec.config_path),
        "output_dir": repo_relative(spec.output_dir),
        "summary_path": repo_relative(summary_path),
        "fusion_weight_analysis_path": repo_relative(fusion_path),
    }
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        best_val = summary.get("best_val") or {}
        runtime = summary.get("runtime") or {}
        result.update(
            {
                "val_acc": float(best_val.get("accuracy", float("nan"))),
                "val_auc": float(best_val.get("auc", float("nan"))),
                "val_f1": float(best_val.get("f1", float("nan"))),
                "peak_vram_mb": float(runtime.get("peak_vram_mb", float("nan"))),
                "total_seconds": float(runtime.get("total_seconds", float("nan"))),
            }
        )
    if fusion_path.exists():
        fusion = json.loads(fusion_path.read_text(encoding="utf-8"))
        fusion_summary = fusion.get("summary") or {}
        per_view = fusion.get("per_view") or []
        result["mean_fusion_weight"] = {
            str(item["view"]): float(item["mean_fusion_weight"])
            for item in per_view
        }
        result["top_weight_view_distribution"] = fusion_summary.get(
            "top_weight_view_distribution"
        )
        result["top_true_margin_view_distribution"] = fusion_summary.get(
            "top_true_margin_view_distribution"
        )
        result["top_weight_hit_rate"] = fusion_summary.get("top_weight_hit_rate")
    return result


def summarize(results: list[dict[str, Any]], output_path: Path) -> None:
    valid = [item for item in results if "val_acc" in item]
    aggregate: dict[str, Any] = {"runs": results}
    if valid:
        aggregate["mean"] = {
            key: float(sum(float(item[key]) for item in valid) / len(valid))
            for key in ("val_acc", "val_auc", "val_f1")
        }
    output_path.write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote aggregate={repo_relative(output_path)}")
    if "mean" in aggregate:
        mean = aggregate["mean"]
        print(
            "mean "
            f"val_acc={mean['val_acc']:.15f} "
            f"val_auc={mean['val_auc']:.15f} "
            f"val_f1={mean['val_f1']:.15f}"
        )


def main() -> int:
    args = parse_args()
    os.chdir(REPO_ROOT)
    specs = build_specs(args.configs)
    log_dir = resolve_path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"workdir={REPO_ROOT}")
    print(f"hostname={os.uname().nodename}")
    print(f"SLURM_JOB_ID={os.getenv('SLURM_JOB_ID', 'unset')}")
    print(f"CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES', 'unset')}")
    print(f"configs={[repo_relative(spec.config_path) for spec in specs]}")
    print(f"outputs={[repo_relative(spec.output_dir) for spec in specs]}")
    if not args.dry_run:
        check_visible_gpus(min(len(specs), args.max_parallel), args.min_free_gb)

    preflight = [
        args.python,
        "scripts/backbone_preflight.py",
        "--config",
        *[repo_relative(spec.config_path) for spec in specs],
    ]
    print(command_text(preflight))
    if not args.dry_run:
        preflight_result = subprocess.run(preflight, cwd=REPO_ROOT, check=False)
        if preflight_result.returncode != 0:
            return int(preflight_result.returncode)

    train_status = 0
    with ThreadPoolExecutor(max_workers=args.max_parallel) as executor:
        futures = {
            executor.submit(run_config, spec, args, log_dir): spec
            for spec in specs
        }
        for future in as_completed(futures):
            train_status = max(train_status, int(future.result()))
    if train_status != 0:
        return train_status

    telemetry_status = 0
    if not args.skip_telemetry:
        for spec in specs:
            telemetry_status = max(telemetry_status, run_telemetry(spec, args, log_dir))

    results = [read_result(spec) for spec in specs]
    summarize(results, log_dir / "aggregate_summary.json")
    return telemetry_status


if __name__ == "__main__":
    raise SystemExit(main())

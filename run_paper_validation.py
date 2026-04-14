from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from script_runtime import detect_python_executable, read_output_dir, repo_relative, tail_lines  # noqa: E402


TIMEOUT_PROXY_SECONDS = 3600
TIMEOUT_FORMAL_SECONDS = 10800


def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run_experiment(name: str, config_path: Path, timeout_seconds: int, python_executable: str) -> None:
    run_dir = read_output_dir(config_path)
    log_path = REPO_ROOT / f"{name}.log"
    err_path = REPO_ROOT / f"{name}.err"

    print("")
    print("========================================")
    print(f"[{ts()}] START: {name}")
    print(f"  Config:  {repo_relative(config_path)}")
    print(f"  RunDir:  {repo_relative(run_dir)}")
    print("========================================")

    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "1")

    with log_path.open("w", encoding="utf-8") as log_handle, err_path.open("w", encoding="utf-8") as err_handle:
        try:
            subprocess.run(
                [python_executable, "train.py", "--config", repo_relative(config_path)],
                cwd=REPO_ROOT,
                env=env,
                stdout=log_handle,
                stderr=err_handle,
                check=True,
                timeout=timeout_seconds,
                text=True,
            )
        except subprocess.TimeoutExpired:
            print(f"[{ts()}] TIMEOUT: {name} (killed after {timeout_seconds // 60} min)")
            return
        except subprocess.CalledProcessError as exc:
            print(f"[{ts()}] CRASH: {name} (exit code {exc.returncode})")
            for line in tail_lines(log_path, lines=20):
                print(line)
            return

    print(f"[{ts()}] Training done: {name}")
    subprocess.run(
        [
            python_executable,
            "tools/evaluate_threshold.py",
            "--run_dir",
            repo_relative(run_dir),
            "--config",
            repo_relative(config_path),
        ],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        text=True,
    )
    print(f"[{ts()}] Threshold eval done: {name}")

    summary_path = run_dir / "summary.json"
    threshold_path = run_dir / "threshold_eval.json"
    if summary_path.exists():
        print("  summary.json:")
        print(summary_path.read_text(encoding="utf-8"))
    if threshold_path.exists():
        print("  threshold_eval.json (last 5 lines):")
        for line in tail_lines(threshold_path, lines=5):
            print(line)

    print(f"[{ts()}] FINISHED: {name}")


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def stddev(values: list[float]) -> float:
    average = mean(values)
    return (sum((value - average) ** 2 for value in values) / len(values)) ** 0.5


def main() -> None:
    python_executable = detect_python_executable()

    print("")
    print("################################################################")
    print("#  PHASE 1: Multi-seed validation (VR-MS-01~03, freeze=3)     #")
    print("################################################################")

    run_experiment("VR-MS-01_seed42", REPO_ROOT / "configs/multiseed_s42.yaml", TIMEOUT_PROXY_SECONDS, python_executable)
    run_experiment("VR-MS-02_seed123", REPO_ROOT / "configs/multiseed_s123.yaml", TIMEOUT_PROXY_SECONDS, python_executable)
    run_experiment("VR-MS-03_seed456", REPO_ROOT / "configs/multiseed_s456.yaml", TIMEOUT_PROXY_SECONDS, python_executable)

    print("")
    print("================================================================")
    print("MULTI-SEED SUMMARY")
    print("================================================================")

    seeds = ["42", "123", "456"]
    run_dirs = [REPO_ROOT / "runs/multiseed_s42", REPO_ROOT / "runs/multiseed_s123", REPO_ROOT / "runs/multiseed_s456"]
    accs: list[float] = []
    spes: list[float] = []
    aucs: list[float] = []

    for seed, run_dir in zip(seeds, run_dirs, strict=True):
        threshold_path = run_dir / "threshold_eval.json"
        summary_path = run_dir / "summary.json"
        if threshold_path.exists() and summary_path.exists():
            threshold_data = json.loads(threshold_path.read_text(encoding="utf-8"))
            summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
            acc = float(threshold_data["val"]["accuracy"])
            spe = float(threshold_data["val"]["specificity"])
            auc = float(summary_data["best_val"]["auc"])
            accs.append(acc)
            spes.append(spe)
            aucs.append(auc)
            print(f"  seed={seed}: no_miss_val_acc={acc}  no_miss_val_spe={spe}  val_AUC={auc}")
        else:
            print(f"  seed={seed}: MISSING RESULTS")

    if len(accs) == 3:
        print("")
        print("  === 3-SEED STATISTICS ===")
        print(f"  no_miss_val_acc: {mean(accs):.4f} +/- {stddev(accs):.4f}")
        print(f"  no_miss_val_spe: {mean(spes):.4f} +/- {stddev(spes):.4f}")
        print(f"  val_AUC:         {mean(aucs):.4f} +/- {stddev(aucs):.4f}")

    print("")
    print("################################################################")
    print("#  PHASE 2: Three-fusion formal comparison                    #")
    print("################################################################")

    run_experiment("Formal_Decision", REPO_ROOT / "configs/formal_decision.yaml", TIMEOUT_FORMAL_SECONDS, python_executable)
    run_experiment("Formal_Feature", REPO_ROOT / "configs/formal_feature.yaml", TIMEOUT_FORMAL_SECONDS, python_executable)
    run_experiment("Formal_Attention", REPO_ROOT / "configs/formal_attention.yaml", TIMEOUT_FORMAL_SECONDS, python_executable)

    print("")
    print("================================================================")
    print("FORMAL COMPARISON SUMMARY")
    print("================================================================")

    formal_names = ["Decision", "Feature", "Attention"]
    formal_dirs = [REPO_ROOT / "runs/formal_decision", REPO_ROOT / "runs/formal_feature", REPO_ROOT / "runs/formal_attention"]
    for name, run_dir in zip(formal_names, formal_dirs, strict=True):
        threshold_path = run_dir / "threshold_eval.json"
        summary_path = run_dir / "summary.json"
        if threshold_path.exists() and summary_path.exists():
            threshold_data = json.loads(threshold_path.read_text(encoding="utf-8"))
            summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
            acc = round(float(threshold_data["val"]["accuracy"]), 4)
            spe = round(float(threshold_data["val"]["specificity"]), 4)
            auc = round(float(summary_data["best_val"]["auc"]), 4)
            test_acc = round(float(threshold_data["test"]["accuracy"]), 4)
            test_spe = round(float(threshold_data["test"]["specificity"]), 4)
            test_sen = round(float(threshold_data["test"]["sensitivity"]), 4)
            print(f"  {name} Fusion:")
            print(f"    Val:  no_miss_acc={acc}  no_miss_spe={spe}  AUC={auc}")
            print(f"    Test: acc={test_acc}  spe={test_spe}  sen={test_sen}")
        else:
            print(f"  {name} Fusion: MISSING RESULTS")

    print("")
    print("################################################################")
    print("#  ALL PAPER VALIDATION EXPERIMENTS COMPLETE                   #")
    print(f"#  {ts()}                                         #")
    print("################################################################")


if __name__ == "__main__":
    main()

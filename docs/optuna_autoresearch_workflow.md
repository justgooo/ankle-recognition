# AutoResearch + Optuna + Monitor Workflow

This repository now supports a thin three-layer workflow:

1. AutoResearch Agent makes one discrete, explainable research change.
2. Optuna performs local hyperparameter tuning on top of that candidate.
3. A monitor agent summarizes the trial stream and feeds evidence back into the next AutoResearch step.

The existing training entrypoint stays unchanged:

```bash
./.venv/bin/python train.py --config ...
```

The Optuna layer is optional. If you do not run Optuna, the original training flow still works exactly as before.

GPU selection is inherited from the shell environment. The server has two GPUs:
- GPU 0 = RTX 3090 (slot 0): `CUDA_VISIBLE_DEVICES=0`
- GPU 1 = RTX 4090 (slot 1): `CUDA_VISIBLE_DEVICES=1`

```bash
# Slot 1 (default, RTX 4090)
CUDA_VISIBLE_DEVICES=1 ./.venv/bin/python scripts/run_optuna_proxy.py

# Slot 0 (RTX 3090)
CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python scripts/run_optuna_proxy.py
```

> **⚠️ CPU 限制**：服务器 CPU 已被其他进程占用 ~70%，所有配置的 `num_workers` 必须保持为 1。

## Files

- `scripts/run_optuna_proxy.py`
  - Small, fast search around the current proxy candidate.
- `scripts/run_optuna_main.py`
  - Narrower, deeper search or confirmation on the formal candidate.
- `scripts/monitor_optuna.py`
  - Reads trial artifacts, summarizes progress, flags training failures, emits threshold warnings, and writes monitor reports.
- `scripts/optuna_workflow.py`
  - Shared execution layer: Python detection, temporary YAML generation, run execution, artifact parsing, study bookkeeping, and fresh/resume control.
- `configs/optuna_proxy_search.yaml`
  - Conservative first-pass proxy search space.
- `configs/optuna_main_search.yaml`
  - Narrower formal search space.

## Design rules

- `train.py` stays the training entrypoint.
- The Optuna objective does not patch training internals.
- Every trial is run by generating a temporary YAML file and calling `train.py --config <temp>`.
- Validation-set selection is centered on `val_accuracy`.
- If `val_accuracy` is tied, `val_auc` is the tie-break.
- Test-set metrics are never used for trial ranking or model selection.
- Data split files and metadata semantics remain untouched.
- Dataset preflight may report path issues, but it should not silently rewrite the training CSV by default.

## Python environment

The wrappers now require the project virtualenv.

- Expected interpreter: `./.venv/bin/python` on Linux/macOS, `./.venv/Scripts/python.exe` on Windows
- If the project `.venv` is missing, the Optuna workflow fails closed with a clear error
- The wrappers no longer fall back to `python3`, `python`, or the current interpreter

## Study lifecycle rules

Search YAML files may still define conventional roots such as `runs/optuna_proxy` and `runs/optuna_main`, but reuse is controlled by the workflow:

- fresh run: default mode; if `study.sqlite3` already exists, the workflow refuses to reuse it
- resume run: must be explicit via `--resume`
- `n_trials` means the total target budget for the study, not "add this many more trials on every restart"
- when resuming, the workflow computes the remaining budget from the existing trial count and only launches the missing trials
- `enqueue_current_template: true` uses the effective config after overrides/default normalization, so the baseline trial matches the real runtime template

## What each trial reads

Each trial writes into its own run directory under the chosen study root.

The wrapper reads:

1. `summary.json`
2. `history.json`
3. `metrics.json`
4. `results.json`
5. `threshold_eval.json` as auxiliary evidence only
6. `train.log` / `threshold.log`

Primary objective:

- `summary.json -> best_val.accuracy`

Recorded side metrics:

- `val_auc`
- `val_f1`
- `threshold_eval.json -> val.accuracy`
- `train_loss`
- `val_loss`
- `total_seconds`
- `peak_vram_mb`
- `status`

Failure handling:

- crashed trial -> status `crash`, objective `-1.0`
- timeout -> status `timeout`, objective `-1.0`
- OOM -> status `oom`, objective `-1.0`
- NaN / invalid metrics -> status `invalid`, objective `-1.0`
- missing metadata paths -> preflight stops the study before trials are launched
- `threshold_eval` failure does not override a successful training run; it is tracked as auxiliary output only
- `monitor_optuna.py --watch` exits early only for training failure / timeout signals or stale study state, not for threshold-only failures

## Recommended minimal run order

```bash
# 1) Check the current candidate still trains.
./.venv/bin/python train.py --config configs/autoresearch_proxy.yaml
./.venv/bin/python tools/evaluate_threshold.py \
  --run_dir runs/autoresearch_proxy \
  --config configs/autoresearch_proxy.yaml

# 2) Apply one high-level AutoResearch change.
#    Keep it discrete: one structural or regularization idea at a time.

# 3) Launch a small fresh proxy Optuna study on top of that candidate.
CUDA_VISIBLE_DEVICES=1 ./.venv/bin/python scripts/run_optuna_proxy.py  # or CUDA_VISIBLE_DEVICES=0 for slot 0

# 4) Monitor the running study in another shell.
./.venv/bin/python scripts/monitor_optuna.py \
  --study-dir runs/optuna_proxy \
  --watch \
  --interval-seconds 30

# 5) Resume only when you explicitly want to continue the same study.
CUDA_VISIBLE_DEVICES=1 ./.venv/bin/python scripts/run_optuna_proxy.py --resume

# 6) If the tuned proxy winner improves validation accuracy, run a deeper pass.
CUDA_VISIBLE_DEVICES=1 ./.venv/bin/python scripts/run_optuna_main.py \
  --source-study-dir runs/optuna_proxy \
  --top-k 3

# Note: For parallel training on both GPUs, see:
#   ./scripts/parallel_train.sh
#   ./autoresearch_parallel_loop.sh

# 7) Summarize the formal study.
./.venv/bin/python scripts/monitor_optuna.py \
  --study-dir runs/optuna_main
```

## How AutoResearch should use it

Use this loop for each research iteration:

1. Propose one high-level change.
2. Run one ordinary baseline/smoke check.
3. Run `scripts/run_optuna_proxy.py` as a fresh study unless you intentionally want to resume.
4. Run `scripts/monitor_optuna.py` on the resulting study.
5. Compare the best tuned trial against the current keep version by validation accuracy.
6. If `val_acc` is tied, compare `val_auc`.
7. Only keep the change if the tuned candidate is better.
8. If it wins, run the main/formal pass and monitor it again.
9. Then update `backlog.md`, `results.tsv`, and the best config reference.

## Notes

- All subprocess calls use `pathlib` + `subprocess` and write normal text logs.
- `threshold_eval` remains useful for diagnostics, but it is not a fatal gate for an otherwise successful training trial.
- If Optuna is not installed in the project environment, the default training pipeline still works, but the Optuna wrappers will exit with a clear message.

Example install:

```bash
./.venv/bin/python -m pip install optuna
```

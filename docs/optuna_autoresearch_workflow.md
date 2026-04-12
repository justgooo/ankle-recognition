# AutoResearch + Optuna + Monitor Workflow

This repository now supports a thin three-layer workflow:

1. AutoResearch Agent makes one discrete, explainable research change.
2. Optuna performs local hyperparameter tuning on top of that candidate.
3. A monitor agent summarizes the trial stream and feeds evidence back into the next AutoResearch step.

The existing training entrypoint stays unchanged:

```bash
python train.py --config ...
python3 train.py --config ...
./.venv/bin/python train.py --config ...
```

The Optuna layer is optional. If you do not run Optuna, the original training flow still works exactly as before.

GPU selection is inherited from the shell environment. On Linux, if you want to target GPU 1, prefix the command:

```bash
CUDA_VISIBLE_DEVICES=1 ./.venv/bin/python scripts/run_optuna_proxy.py
```

## Files

- `scripts/run_optuna_proxy.py`
  - Small, fast search around the current proxy candidate.
- `scripts/run_optuna_main.py`
  - Narrower, deeper search or confirmation on the formal candidate.
- `scripts/monitor_optuna.py`
  - Reads trial artifacts, summarizes progress, flags failures, and writes monitor reports.
- `scripts/optuna_workflow.py`
  - Shared execution layer: Python command detection, temporary YAML generation, run execution, artifact parsing, and study bookkeeping.
- `configs/optuna_proxy_search.yaml`
  - Conservative first-pass proxy search space.
- `configs/optuna_main_search.yaml`
  - Narrower formal search space.

## Design rules

- `train.py` stays the training entrypoint.
- The Optuna objective does not patch training internals.
- Every trial is run by generating a temporary YAML file and calling `train.py --config <temp>`.
- Validation-set selection stays centered on `val_accuracy`.
- Test-set metrics are never used for trial ranking or model selection.
- Data split files and metadata semantics remain untouched.

## What each trial reads

Each trial writes into its own run directory under `runs/optuna_proxy/...` or `runs/optuna_main/...`.

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

## Recommended minimal run order

```bash
# 1) Check the current candidate still trains.
./.venv/bin/python train.py --config configs/autoresearch_proxy.yaml
./.venv/bin/python tools/evaluate_threshold.py \
  --run_dir runs/autoresearch_proxy \
  --config configs/autoresearch_proxy.yaml

# 2) Apply one high-level AutoResearch change.
#    Keep it discrete: one structural or regularization idea at a time.

# 3) Launch a small proxy Optuna study on top of that candidate.
CUDA_VISIBLE_DEVICES=1 ./.venv/bin/python scripts/run_optuna_proxy.py

# 4) Monitor the running study in another shell.
./.venv/bin/python scripts/monitor_optuna.py \
  --study-dir runs/optuna_proxy \
  --watch \
  --interval-seconds 30

# 5) If the tuned proxy winner improves validation accuracy, run a deeper pass.
CUDA_VISIBLE_DEVICES=1 ./.venv/bin/python scripts/run_optuna_main.py \
  --source-study-dir runs/optuna_proxy \
  --top-k 3

# 6) Summarize the formal study.
./.venv/bin/python scripts/monitor_optuna.py \
  --study-dir runs/optuna_main
```

## How AutoResearch should use it

Use this loop for each research iteration:

1. Propose one high-level change.
2. Run one ordinary baseline/smoke check.
3. Run `scripts/run_optuna_proxy.py`.
4. Run `scripts/monitor_optuna.py` on the resulting study.
5. Compare the best tuned trial against the current keep version by validation accuracy.
6. Only keep the change if the tuned candidate is better.
7. If it wins, run the main/formal pass and monitor it again.
8. Then update `backlog.md`, `results.tsv`, and the best config reference.

## Linux notes

- The wrappers prefer `./.venv/bin/python`, then `python3`, then `python`.
- They also support fallback to the current interpreter if needed.
- All subprocess calls use `pathlib` + `subprocess` and write normal text logs.
- Example study roots:
  - `runs/optuna_proxy`
  - `runs/optuna_main`

## Optional dependency

The new Optuna layer is intentionally isolated from the default training path.
If Optuna is not installed in the project environment, the training pipeline still works, but the Optuna wrappers will exit with a clear message.

Example install:

```bash
./.venv/bin/python -m pip install optuna
```

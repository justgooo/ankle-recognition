# 实验待办清单 (Experiment Backlog)

> **本文件由 autoresearch Agent 维护。**
> Agent 在每次实验循环启动时必须阅读本文件，并在完成实验后根据结果更新。
> 人类也可以直接编辑本文件来添加想法或调整优先级。
>
> **⚠️ 指标体系切换**：本项目已从「零漏诊 (no_miss_val_acc)」切换到「验证集准确率 (val_acc)」。
> 旧实验记录使用 `no_miss_val_acc`，保留不变。新实验使用 `summary.json` 中的 `best_val.accuracy`。
> 新实验的 keep 判定基于 val_acc，不再使用 evaluate_threshold.py 作为主评估工具。

---

## Agent 状态

> Agent 每次实验后必须更新此表。新 Agent 启动时以此表为起点。

| 字段 | 值 |
|------|-----|
| 上次实验 | CVT-MAN-13（winner-template stability probe；`seed=456`, `weight_decay=8.75e-4`; commit `4363c40`） |
| 上次结果 | discard（`val_acc=0.8297872340425532`, `val_auc=0.899090909090909`, `val_f1=0.8260869565217391`, `peak_vram=14.4 GiB`。较 current proxy winner `0.8829787234042553` 低 `0.0531914893617021`，较 `CVT-MAN-10` 的 exact `seed=456` rerun `0.8404255319148937` 仍低 `0.0106382978723405`，但较 `CVT-MAN-12` 的 `7.5e-4` 点回升 `0.0106382978723404`；同时它与 `CVT-MAN-11` 的 `5e-4` 探针 accuracy 持平，却在 AUC 上再低 `0.0172727272727274`。说明把 `weight_decay` 从 `7.5e-4` 拉回到 `8.75e-4` 只能补回 1 个 accuracy case，却没有保住先前较弱 L2 带来的排序收益。） |
| 下一步 | 若继续完成 `seed=456` 的 `weight_decay` 上侧细化，最高优先级应为 `9e-4`：验证更贴近 canonical winner 时能否把 accuracy 拉回 exact rerun 的 baseline-level `0.8404255319148937`，同时至少保留相对 `1e-3` 的部分 AUC 回升。若 `9e-4` 仍不能达到这一折中，应停止该轴，不再继续做更细的 L2 插值。 |
| 连续 discard 计数 | 13 |
| 累计 proxy keep 数 | 7（当前 `val_acc` 主线新增 1 次 keep：`a60c3e0` / fresh proxy winner） |
| 本地迁移补记 | 2026-04-12 从旧工作副本并入的 legacy 状态：上次实验为 `VR-16`（`9293915`; `no_miss_val_acc=0.872`, `no_miss_val_spe=0.760`, `val_AUC=0.969`）；旧计划下一步为 `VR-MS-01~03`；旧连续 discard 计数为 15。该状态属于旧 `no_miss` / `192x16` campaign，已归档为 legacy，不覆盖当前 canonical `val_acc` 主线。 |

---

## 当前最优纪录

> 当前 canonical baseline 已切换为：**ResUNet encoder + 3 个 Attention Gate + AttentionPooling + VRG / decision fusion**。
> 旧的 `no_miss_*` / `192x16` / 早期 stage-10 结果只作为 **legacy reference only**，不再直接参与当前 keep/discard 比较。

| 指标 | 值 | 来源 commit | 配置 | 备注 |
|------|---:|-------------|------|------|
| **val_acc** | **0.8829787234042553** | `a60c3e0` | `configs/autoresearch_proxy.yaml` + fresh proxy study trial 0 | 4 个 completed trial 中最佳；相对 canonical baseline rerun `0.8404255319148937` 提升 `+0.0425531914893616` |
| **tie-break** | **0.9318181818181819** | `a60c3e0` | fresh proxy study trial 0 | 与上述 winner 同一 trial 的 `best_val.auc`；当 `val_acc` 持平时仍按此决胜 |
| canonical formal 参考 | 0.8617021276595744 | `c30fcae` | `configs/autoresearch_formal.yaml`（trial 0 模板超参） | 当前主线首个 formal confirmation；`val_auc=0.9372727272727273`，准确率低于 proxy winner 但 AUC 更高 |
| legacy no_miss 参考 | 0.915 | `d63b49a` (formal) | 旧 Decision Fusion / 旧实验语义 | **legacy reference only**，不可与当前主线直接比较 |

---

## 阶段 10：当前主线校准与可复现实验 🔴 当前最高优先级

> **当前主线定义（canonical baseline）**
> - 单张 CT 切片先进入以 ResNet18 为编码器的 `ResUNet`
> - 3 个 `Attention Gate` 突出病灶区域
> - 单视角 16 张切片经 `AttentionPooling` 聚合为视角特征
> - 3 个视角通过 `VRG` / `decision fusion` 做可靠度加权融合
> - 主指标是 `summary.json -> best_val.accuracy`
> - 若 `val_acc` 持平，按 `best_val.auc` 决胜
>
> **当前流程约束**
> - `model.freeze_layers` 必须由 YAML 显式声明，不再通过修改 `src/model.py` 常量切换
> - Optuna 必须优先使用项目 `.venv`
> - fresh study 默认不得复用旧 `study.sqlite3`；只有显式 `--resume` 才续跑
> - resume 语义是补足剩余 budget，而不是再次追加完整 `n_trials`
> - `threshold_eval` 仅作辅助输出；失败时记录 warning，不应判定整个 study 失败
>
> **当前判定规则**
> - keep / discard 比较对象：当前 canonical baseline 的 `val_acc`
> - `val_acc` 持平：比较 `val_auc`
> - 二者仍持平：优先更简单的配置 / 代码路径
>
> **执行顺序**
> 1. 跑 1 次 `configs/autoresearch_proxy.yaml`，重新确立当前主线 `val_acc` 基线
> 2. 如 baseline 正常，再启动 fresh proxy Optuna study（不得混入旧 trial）
> 3. 由 `monitor_optuna.py` 汇总 best trial、warning、degraded trial 与建议
> 4. proxy winner 明显更优后，再做 main/formal 确认
>
> **说明**
> - 历史 `VR-01 ~ VR-16`、`no_miss_*`、`192x16` 记录保留供回顾，但不再作为当前主线的直接 comparator
> - 若需要复盘旧阶段，请显式标注为 legacy campaign

### 2026-04-13：一轮基线校准 + capped fresh proxy Optuna + formal confirmation

- [x] **Baseline rerun**：`configs/autoresearch_proxy.yaml` 当前 canonical baseline（未加结构改动）→ `val_acc=0.8404255319148937`, `val_auc=0.9181818181818182`, `val_f1=0.8192771084337349`, `peak_vram=14.4 GiB`。
- [x] **VRG-LOGIT-01**：将 decision fusion 的 reliability head 从 `Linear -> Sigmoid` 改为 `LayerNorm -> Linear` raw logits（commit `a60c3e0`）→ fresh proxy Optuna study 在人工 GPU 切换后于 3090 上重启；收口时保留 4 个 completed trial，忽略已中断的第 5 个 running trial。
- [x] **VRG-LOGIT-01 / winner**：best completed trial 为 **trial 0**（当前模板超参：`batch_size=4`, `dropout=0.3`, `lr=5e-5`, `weight_decay=1e-3`, `augmentation=true`, `scheduler=none`, `label_smoothing=0.0`, `gradient_clip_norm=1.0`, `early_stopping_patience=2`）→ `val_acc=0.8829787234042553`, `val_auc=0.9318181818181819`, `val_f1=0.8764044943820225` → **keep**。
- [x] **Monitor takeaways**：当前结构对超参较敏感；在已完成 trial 中，`batch_size=4`、`dropout=0.3`、`label_smoothing=0.0`、`scheduler=none`、`weight_decay=1e-3` 明显优于小 batch / `cosine` / 更高 `label_smoothing`。
- [x] **Formal confirmation**：`c30fcae` 将 `configs/autoresearch_formal.yaml` 对齐到 proxy winner 模板超参后完成 1 次 formal（best epoch=4，early stop at epoch 6）→ `val_acc=0.8617021276595744`, `val_auc=0.9372727272727273`, `val_f1=0.8505747126436781`, `peak_vram=14.4 GiB` → **keep**（记为当前 canonical formal reference；`val_acc` 低于 proxy winner `0.0213`，但 `val_auc` 提升 `0.0055`）。
- [x] **CVT-MAN-01**：手动 constrained proxy 单点（`lr=4e-5`, `early_stopping_patience=2`；其余固定 `batch_size=4`, `dropout=0.3`, `label_smoothing=0.0`, `scheduler=none`, `weight_decay=1e-3`）→ `val_acc=0.8723404255319149`, `val_auc=0.9209090909090909`, `val_f1=0.8571428571428571`, `peak_vram=14.4 GiB` → **discard**（较 current proxy best `0.8829787234042553` 低 `0.0106382978723404`；accuracy 明显高于 baseline rerun `0.8404`，但仍未追平 winner，AUC 也低 `0.0109`）。
- [x] **CVT-MAN-02**：手动 constrained proxy 单点（`lr=6e-5`, `early_stopping_patience=2`；其余固定 `batch_size=4`, `dropout=0.3`, `label_smoothing=0.0`, `scheduler=none`, `weight_decay=1e-3`）→ `val_acc=0.8085106382978723`, `val_auc=0.9190909090909091`, `val_f1=0.8235294117647058`, `peak_vram=14.4 GiB` → **discard**（较 current proxy best `0.8829787234042553` 低 `0.07446808510638298`；高学习率边界明显退化，且 manual round 1 的较优趋势未能延续）。
- [x] **CVT-MAN-03**：手动 constrained proxy 单点（`lr=5e-5`, `early_stopping_patience=3`；其余固定 `batch_size=4`, `dropout=0.3`, `label_smoothing=0.0`, `scheduler=none`, `weight_decay=1e-3`）→ **crash**（OOM；日志显示目标 GPU 仅剩约 `223 MiB` 空闲，另一个 `.venv/bin/python` 进程已占用约 `16.9 GiB`，在 `src/model.py` attention gate 前向阶段申请额外 `256 MiB` 时失败；需待目标 GPU 清空后再做有效复验）。
- [x] **CVT-MAN-03R**：在确认 `CUDA_VISIBLE_DEVICES=1` 空闲后重跑同一 constrained proxy 单点（`lr=5e-5`, `early_stopping_patience=3`；commit `1f117dd`）→ `val_acc=0.8191489361702128`, `val_auc=0.89`, `val_f1=0.8316831683168316`, `peak_vram=14.4 GiB` → **discard**（验证了原 OOM 属外部占卡，但该 patience=3 点本身比 current proxy best 低 `0.06382978723404253`，也比 `CVT-MAN-01` 的 `lr=4e-5 + patience=2` 低 `0.05319148936170215`；说明当前主线下继续放宽 early stopping 并不有利）。
- [x] **CVT-MAN-04**：手动 constrained proxy 单点（`lr=4.5e-5`, `early_stopping_patience=2`；commit `092eea5`）→ `val_acc=0.851063829787234`, `val_auc=0.9013636363636364`, `val_f1=0.8333333333333334`, `peak_vram=14.4 GiB` → **discard**（较 current proxy best `0.8829787234042553` 低 `0.03191489361702133`，且较 `CVT-MAN-01` 的 `lr=4e-5 + patience=2` 低 `0.021276595744680882`；说明在 4e-5 与 5e-5 之间插值没有优于已知较优低学习率点，手动 sweep 更值得继续向更低 lr 边界探索）。
- [x] **CVT-MAN-05**：手动 constrained proxy 单点（`lr=3.5e-5`, `early_stopping_patience=2`；commit `aabeb35`）→ `val_acc=0.851063829787234`, `val_auc=0.9118181818181819`, `val_f1=0.8292682926829268`, `peak_vram=14.4 GiB` → **discard**（较 current proxy best `0.8829787234042553` 低 `0.03191489361702133`，与 `CVT-MAN-04` 持平 accuracy 但 AUC 高 `0.0104545454545455`，仍低于 `CVT-MAN-01` 的 `0.8723404255319149`；说明当前受限 lr sweep 两侧新边界 `3.5e-5 / 4.5e-5` 都未形成新的 accuracy 峰值，可先结束这一小段手动 lr 搜索）。
- [x] **CVT-MAN-06**：exact winner reproduction（`seed=123`, `lr=5e-5`, `early_stopping_patience=2`；commit `eddf885`）→ `val_acc=0.8404255319148937`, `val_auc=0.9177272727272727`, `val_f1=0.8148148148148148`, `peak_vram=14.4 GiB` → **discard**（较 current proxy winner `0.8829787234042553` 低 `0.0425531914893616`，且 accuracy 与 canonical baseline rerun `0.8404255319148937` 完全相同、AUC 还低 `0.0004545454545455`；说明 trial-0 winner 未在 alternate seed 上复现，当前更需要做稳定性导向的 `weight_decay / dropout` 探针，而不是继续受限 lr 微调）。
- [x] **CVT-MAN-07**：winner-template stability probe（`seed=42`, `weight_decay=5e-4`；commit `336cd96`）→ `val_acc=0.8829787234042553`, `val_auc=0.9263636363636365`, `val_f1=0.8705882352941177`, `peak_vram=14.4 GiB` → **discard**（与 current proxy winner `0.8829787234042553` accuracy 完全持平，但 AUC 低 `0.0054545454545454`，因此按 tie-break 仍输给 `weight_decay=1e-3` 的 trial-0 winner；说明较弱 L2 正则不会立刻伤到 accuracy，但当前 canonical winner 在排序质量上更稳，下一步更值得转测 `dropout=0.25/0.35`，而不是继续下调 weight decay）。
- [x] **CVT-MAN-08**：winner-template stability probe（`seed=42`, `dropout=0.25`；commit `51f8199`）→ `val_acc=0.8404255319148937`, `val_auc=0.9163636363636364`, `val_f1=0.8235294117647058`, `peak_vram=14.4 GiB` → **discard**（较 current proxy winner `0.8829787234042553` 低 `0.0425531914893616`，accuracy 与 canonical baseline rerun `0.8404255319148937` 完全相同，AUC 还低 `0.0154545454545455`；说明把 canonical winner 的 dropout 从 `0.3` 下调到 `0.25` 会明显削弱当前正则平衡，下一步若继续应优先测试 `dropout=0.35` 而不是继续下调 dropout。）
- [x] **CVT-MAN-09**：winner-template stability probe（`seed=42`, `dropout=0.35`；commit `7b25d8d`）→ `val_acc=0.8191489361702128`, `val_auc=0.8727272727272728`, `val_f1=0.7792207792207793`, `peak_vram=14.4 GiB` → **discard**（较 current proxy winner `0.8829787234042553` 低 `0.06382978723404253`，也比 canonical baseline rerun `0.8404255319148937` 再低 `0.021276595744680882`；说明把 canonical winner 的 dropout 从 `0.3` 上调到 `0.35` 会进一步过正则化，至此 `dropout=0.25/0.35` 两侧探针均失败，下一步应停止 dropout 轴并转向 alternate-seed confirmation。）
- [x] **CVT-MAN-10**：exact winner reproduction（`seed=456`, `lr=5e-5`, `weight_decay=1e-3`, `dropout=0.3`, `early_stopping_patience=2`; commit `09cb59b`）→ `val_acc=0.8404255319148937`, `val_auc=0.8718181818181818`, `val_f1=0.810126582278481`, `peak_vram=14.4 GiB` → **discard**（较 current proxy winner `0.8829787234042553` 低 `0.0425531914893616`，accuracy 与 canonical baseline rerun `0.8404255319148937` 完全相同，且 AUC 还低 `0.0463636363636364`；说明 winner 模板在 `seed=456` 上同样回落到 baseline-level accuracy。结合此前 `seed=123` 复验失败，当前应停止继续做 exact rerun，转向稳定性导向的 `weight_decay` 轴探针。）
- [x] **CVT-MAN-11**：winner-template stability probe（`seed=456`, `weight_decay=5e-4`; commit `c8f125c`）→ `val_acc=0.8297872340425532`, `val_auc=0.9163636363636364`, `val_f1=0.7948717948717948`, `peak_vram=14.4 GiB` → **discard**（较 current proxy winner `0.8829787234042553` 低 `0.0531914893617021`，也较 `CVT-MAN-10` 的 exact `seed=456` rerun `0.8404255319148937` 再低 `0.0106382978723405`；但 AUC 较 `CVT-MAN-10` 明显回升 `0.0445454545454546`。说明减弱 L2 正则能修复排序质量，却没有把 accuracy 拉回 baseline-level；下一步应继续留在 `weight_decay` 轴，在 `5e-4` 与 `1e-3` 之间补测中间点，而不是切换到其他超参。）
- [x] **CVT-MAN-12**：winner-template stability probe（`seed=456`, `weight_decay=7.5e-4`; commit `0001398`）→ `val_acc=0.8191489361702128`, `val_auc=0.9059090909090909`, `val_f1=0.7848101265822784`, `peak_vram=14.4 GiB` → **discard**（较 current proxy winner `0.8829787234042553` 低 `0.06382978723404253`，较 `CVT-MAN-10` 的 exact `seed=456` rerun `0.8404255319148937` 低 `0.021276595744680882`，也较 `CVT-MAN-11` 的 `weight_decay=5e-4` 探针再低 `0.0106382978723404`；虽然 `val_auc` 仍较 exact rerun 回升 `0.0340909090909091`，但 accuracy 已落到两个端点之下。说明在 `seed=456` 上，`7.5e-4` 不是可用折中点；若继续沿此轴细化，应把搜索重心移回更靠近 `1e-3` 的上侧，而不是继续往更低 L2 走。）
- [x] **CVT-MAN-13**：winner-template stability probe（`seed=456`, `weight_decay=8.75e-4`; commit `4363c40`）→ `val_acc=0.8297872340425532`, `val_auc=0.899090909090909`, `val_f1=0.8260869565217391`, `peak_vram=14.4 GiB` → **discard**（较 current proxy winner `0.8829787234042553` 低 `0.0531914893617021`，较 `CVT-MAN-10` 的 exact `seed=456` rerun `0.8404255319148937` 仍低 `0.0106382978723405`；虽然较 `CVT-MAN-12` 的 `7.5e-4` 点回升 `0.0106382978723404` accuracy，但它只与 `CVT-MAN-11` 的 `5e-4` 探针持平，并在 AUC 上反而低 `0.0172727272727274`。说明回到更靠近 `1e-3` 的上侧能补回 1 个 case，却没有恢复 baseline-level accuracy，也没保住较弱 L2 的排序收益；若继续留在此轴，剩余最有信息量的点应是 `9e-4`，否则可考虑结束该轴。） 

---


## 用户定向 rerun Campaign ✅ 已完成

> 本轮会话覆盖常规 backlog 优先级，只执行 Plan 2 / Plan 6 的定向 rerun。
> 判定规则（仅本 campaign 生效）：以 `threshold_eval.json` 为准，**仅当 `no_miss_val_acc > 0.75` 时记为 keep**；否则记为 discard。
> 固定顺序：先 `R2-01 → R2-08`，再 `R6-01 → R6-08`。
> 当前状态：**16/16 已完成，rerun queue exhausted**。

### Plan 2：View Reliability Gating（8 次 proxy rerun）

- [x] **R2-01**: baseline（fusion_type=decision, share_backbone=false, aug=true, lr=1e-4, dropout=0.3, label_smoothing=0.0）→ no_miss_val_acc=0.723, no_miss_val_spe=0.480, val_AUC=0.923 → discard
- [x] **R2-02**: lr=5e-5 → no_miss_val_acc=0.777, no_miss_val_spe=0.580, val_AUC=0.951 → keep（campaign keep；未超过全局最佳 0.787）
- [x] **R2-03**: lr=7e-5 → no_miss_val_acc=0.723, no_miss_val_spe=0.480, val_AUC=0.946 → discard
- [x] **R2-04**: dropout=0.2 → no_miss_val_acc=0.532, no_miss_val_spe=0.120, val_AUC=0.945 → discard
- [x] **R2-05**: dropout=0.4 → no_miss_val_acc=0.660, no_miss_val_spe=0.360, val_AUC=0.916 → discard
- [x] **R2-06**: label_smoothing=0.05 → no_miss_val_acc=0.617, no_miss_val_spe=0.280, val_AUC=0.916 → discard
- [x] **R2-07**: lr=5e-5 + label_smoothing=0.05 → no_miss_val_acc=0.723, no_miss_val_spe=0.480, val_AUC=0.935 → discard
- [x] **R2-08**: 基于前 7 次 rerun 最佳方向的组合实验（沿用最佳 lr=5e-5，并恢复 baseline label_smoothing=0.0）→ no_miss_val_acc=0.798, no_miss_val_spe=0.620, val_AUC=0.960 → keep ✅ 当前最优

### Plan 6：Asymmetric Safety-Biased Fusion（8 次 proxy rerun）

- [x] **R6-01**: temperature=0.5 baseline（fusion_type=decision, share_backbone=false, aug=true, lr=1e-4, dropout=0.3, label_smoothing=0.0）→ no_miss_val_acc=0.713, no_miss_val_spe=0.460, val_AUC=0.938 → discard
- [x] **R6-02**: temperature=1.0 → no_miss_val_acc=0.681, no_miss_val_spe=0.400, val_AUC=0.939 → discard
- [x] **R6-03**: temperature=0.3 → no_miss_val_acc=0.660, no_miss_val_spe=0.360, val_AUC=0.957 → discard
- [x] **R6-04**: temperature=2.0 → no_miss_val_acc=0.755, no_miss_val_spe=0.540, val_AUC=0.954 → keep（campaign keep；达到阈值 0.75，但未超过当前最优 0.798）
- [x] **R6-05**: temperature=0.5 + lr=5e-5 → no_miss_val_acc=0.649, no_miss_val_spe=0.340, val_AUC=0.941 → discard
- [x] **R6-06**: temperature=0.5 + dropout=0.2 → no_miss_val_acc=0.660, no_miss_val_spe=0.360, val_AUC=0.945 → discard
- [x] **R6-07**: temperature=0.5 + label_smoothing=0.05 → no_miss_val_acc=0.723, no_miss_val_spe=0.480, val_AUC=0.921 → discard
- [x] **R6-08**: 基于前 7 次 rerun 最佳方向的组合实验（沿用最佳 temperature=2.0，并恢复 baseline lr=1e-4 / dropout=0.3 / label_smoothing=0.0）→ no_miss_val_acc=0.670, no_miss_val_spe=0.380, val_AUC=0.935 → discard

---

## 阶段 10：历史方差缩减记录（legacy）

> 以下 `VR-*` 记录来自旧的 stage-10 / `no_miss_*` / 192x16 语义，保留作历史参考。
> 它们不再作为当前 canonical baseline 的直接比较对象；当前执行口径以上文“当前主线校准与可复现实验”为准。

### legacy：freeze_layers=3 / freeze_layers=2 搜索记录

- [x] **VR-01**: baseline（freeze=3, lr=5e-5, dropout=0.3, ls=0.0, gradient_clip_norm=1.0）→ no_miss_val_acc=0.904, no_miss_val_spe=0.820, val_AUC=0.968 → keep ✅（新阶段 10 proxy 最优；超过 `814f491` 的 0.872）
- [x] **VR-02**: lr=1e-4（冻结后可能需要更高 lr，因为可训练参数更少）→ no_miss_val_acc=0.830, no_miss_val_spe=0.680, val_AUC=0.970 → discard（AUC 略升，但零漏诊 val_acc 明显低于当前 proxy 最优 0.904）
- [x] **VR-03**: lr=3e-5（更保守的 lr）→ no_miss_val_acc=0.819, no_miss_val_spe=0.660, val_AUC=0.958 → discard（较当前 proxy 最优 0.904 下降 0.085，未见更低 lr 带来的稳定性收益）
- [x] **VR-04**: dropout=0.2（冻结后过拟合风险降低，dropout 可减小）→ no_miss_val_acc=0.809, no_miss_val_spe=0.640, val_AUC=0.966 → discard（较当前 proxy 最优 0.904 下降 0.096；降低 dropout 未能提升零漏诊表现，且特异度回落）
- [x] **VR-05**: dropout=0.4 → no_miss_val_acc=0.872, no_miss_val_spe=0.760, val_AUC=0.965 → discard（较 `VR-04` 回升 0.063、特异度回升 0.120，但仍低于当前 proxy 最优 `f6f4ed8` 的 0.904；说明更强 dropout 只能部分修复性能下滑）
- [x] **VR-06**: weight_decay=0.001（更强正则化，配合冻结进一步约束参数）→ no_miss_val_acc=0.883, no_miss_val_spe=0.780, val_AUC=0.966 → discard（较当前 proxy 最优 `f6f4ed8` 的 0.904 下降 0.021，但较 `VR-05` 小幅回升 0.011；更强 weight_decay 略有帮助，但仍未恢复到 freeze=3 baseline）
- [x] **VR-07**: lr=1e-4 + dropout=0.2（冻结后最可能的最优组合）→ no_miss_val_acc=0.904, no_miss_val_spe=0.820, val_AUC=0.968 → discard（与当前 proxy 最优 `f6f4ed8` 的零漏诊 val_acc / val_AUC / val_spe 全部追平，按简洁性原则保留更简单的 `VR-01` baseline）
- [x] **VR-08**: 基于前 7 次最佳方向的组合实验（lr=1e-4 + weight_decay=0.001，保留 baseline dropout=0.3）→ no_miss_val_acc=0.883, no_miss_val_spe=0.780, val_AUC=0.964 → discard（较当前 proxy 最优 `f6f4ed8` 的 0.904 下降 0.021；与 `VR-06` 零漏诊指标完全持平但 AUC 更低，说明高 lr + 更强 weight_decay 组合未带来额外收益）

### legacy：freeze_layers=2 超参搜索（8 次 proxy）

> 以下 freeze=2 记录同样属于 legacy stage-10；其中提到的 `DEFAULT_FREEZE_LAYERS` 只是当时的历史控制方式，不代表当前流程。

- [x] **VR-09**: baseline（freeze=2, lr=5e-5, dropout=0.3, ls=0.0, gradient_clip_norm=1.0）→ no_miss_val_acc=0.883, no_miss_val_spe=0.780, val_AUC=0.961 → discard（较当前 proxy 最优 `f6f4ed8` 的 0.904 下降 0.021；说明仅减少一层冻结并不能自动带来收益，后续转向 lr 调整验证可塑性是否仍可释放）
- [x] **VR-10**: lr=1e-4 → no_miss_val_acc=0.564, no_miss_val_spe=0.180, val_AUC=0.927 → discard（较当前 proxy 最优 `f6f4ed8` 的 0.904 下降 0.340；更高 lr 在 freeze=2 设置下显著压低零漏诊阈值至 0.019，导致 41 个假阳性、特异度崩塌）
- [x] **VR-11**: lr=3e-5 → no_miss_val_acc=0.809, no_miss_val_spe=0.640, val_AUC=0.962 → discard（较当前 proxy 最优 `f6f4ed8` 的 0.904 下降 0.096，且较 freeze=2 baseline `fc760cb` 的 0.883 再降 0.074；说明进一步下调 lr 不能释放 layer3 的额外可塑性，反而扩大假阳性）
- [x] **VR-12**: dropout=0.2 → no_miss_val_acc=0.787, no_miss_val_spe=0.600, val_AUC=0.955 → discard（较当前 proxy 最优 `f6f4ed8` 的 0.904 下降 0.117，且较 `VR-11` 再降 0.021；说明在 freeze=2 设置下减弱 dropout 会进一步放大假阳性，下一步转测 dropout=0.4）
- [x] **VR-13**: dropout=0.4 → no_miss_val_acc=0.766, no_miss_val_spe=0.560, val_AUC=0.961 → discard（较当前 proxy 最优 `f6f4ed8` 的 0.904 下降 0.138，较 freeze=2 baseline `fc760cb` 的 0.883 下降 0.117，且较 `VR-12` 再降 0.021；说明更强 dropout 仍未抑制 layer3 解冻后的假阳性，下一步转测 weight_decay=0.001）
- [x] **VR-14**: weight_decay=0.001 → no_miss_val_acc=0.830, no_miss_val_spe=0.680, val_AUC=0.960 → discard（较当前 proxy 最优 `f6f4ed8` 的 0.904 下降 0.074，较 freeze=2 baseline `fc760cb` 的 0.883 下降 0.053，但较 `VR-13` 的 0.766 回升 0.064；说明更强 L2 正则能部分抑制假阳性，但仍未恢复到 freeze=3/2 baseline 水平）
- [x] **VR-15**: lr=1e-4 + dropout=0.2 → no_miss_val_acc=0.628, no_miss_val_spe=0.300, val_AUC=0.948 → discard（较当前 proxy 最优 `f6f4ed8` 的 0.904 下降 0.277，较 freeze=2 baseline `fc760cb` 的 0.883 下降 0.255，且仅较 `VR-10` 的 0.564 回升 0.064；说明 freeze=2 路线对高 lr 仍极敏感，减小 dropout 不能修复假阳性崩塌）
- [x] **VR-16**: 基于前 7 次（VR-09~VR-15）最佳方向的组合实验（回到 baseline lr=5e-5 / dropout=0.3，并保留最佳正则 `weight_decay=0.001`）→ no_miss_val_acc=0.872, no_miss_val_spe=0.760, val_AUC=0.969 → discard（较当前 proxy 最优 `f6f4ed8` 的 0.904 下降 0.032，但较同配方前次 `VR-14` / `c059aa6` 的 0.830 回升 0.042、较 freeze=2 baseline `fc760cb` 的 0.883 仅差 0.011；说明 regularized freeze=2 rerun 能显著回弹，但仍不足以取代 freeze=3 baseline，下一步转入最优配置 multi-seed 验证）

### 阶段 10C：Multi-seed 验证（最优配置 x 3 seeds）

> 仅在 10A/10B 中产生 keep 后执行

- [ ] **VR-MS-01**: 最优配置 + seed=42
- [ ] **VR-MS-02**: 最优配置 + seed=123
- [ ] **VR-MS-03**: 最优配置 + seed=456

---


## 用户定向 rerun Campaign（Plan 1：Hierarchical Hybrid Fusion）❌ 已放弃

> 9/10 实验全部 discard，用户决定放弃 R1-10。

- [x] **R1-01**: baseline → no_miss_val_acc=0.638 → discard
- [x] **R1-02**: lr=5e-5 → no_miss_val_acc=0.702 → discard
- [x] **R1-03**: lr=7e-5 → no_miss_val_acc=0.670 → discard
- [x] **R1-04**: dropout=0.2 → no_miss_val_acc=0.723 → discard
- [x] **R1-05**: dropout=0.4 → no_miss_val_acc=0.745 → discard
- [x] **R1-06**: label_smoothing=0.05 → no_miss_val_acc=0.564 → discard
- [x] **R1-07**: lr=5e-5 + ls=0.05 → no_miss_val_acc=0.660 → discard
- [x] **R1-08**: fusion_hidden_dim=128 → no_miss_val_acc=0.543 → discard
- [x] **R1-09**: fusion_hidden_dim=384 → no_miss_val_acc=0.691 → discard
- [x] ~~**R1-10**~~: 已放弃（用户决定，2026-04-05）

---

## 阶段 5：AttentionPooling 增强 ❌ 已放弃

> AP-FF 前 3 次全部 discard（最高 0.574），用户决定放弃该方向。

- [x] AP-FF-01 → 0.574 discard | AP-FF-02 → 0.532 discard | AP-FF-03 → 0.553 discard
- 其余实验已取消

---

## 阶段 6：Uncertainty-Weighted Decision Fusion (UWDF) ✅ 已完成（8/8 proxy，全 discard）

> **论文创新点**：不确定性加权决策融合。
> 每个视角不仅输出分类 logits，还输出预测不确定性（log σ²）。
> 不确定性高的视角在融合时权重自动降低。
> 训练损失 = CE(fused_logits) + heteroscedastic auxiliary loss (Kendall & Gal 2017)。
> 基线配置：`share_backbone=false, aug=true, lr=1e-4, label_smoothing=0.0, dropout=0.3, fusion_hidden_dim=256`

### 阶段 6A：UWDF 超参搜索（8 次 proxy）

- [x] **UWDF-01**: baseline（fusion_type=decision, lr=1e-4, dropout=0.3, ls=0.0）→ no_miss_val_acc=0.574, no_miss_val_spe=0.200, val_AUC=0.911 → discard
- [x] **UWDF-02**: lr=5e-5 → no_miss_val_acc=0.723, no_miss_val_spe=0.480, val_AUC=0.922 → discard
- [x] **UWDF-03**: lr=7e-5 → no_miss_val_acc=0.574, no_miss_val_spe=0.200, val_AUC=0.937 → discard
- [x] **UWDF-04**: dropout=0.2 → no_miss_val_acc=0.734, no_miss_val_spe=0.500, val_AUC=0.923 → discard
- [x] **UWDF-05**: dropout=0.4 → no_miss_val_acc=0.596, no_miss_val_spe=0.240, val_AUC=0.913 → discard
- [x] **UWDF-06**: label_smoothing=0.05 → no_miss_val_acc=0.702, no_miss_val_spe=0.440, val_AUC=0.927 → discard
- [x] **UWDF-07**: lr=5e-5 + label_smoothing=0.05 → no_miss_val_acc=0.638, no_miss_val_spe=0.320, val_AUC=0.920 → discard
- [x] **UWDF-08**: dropout=0.2 + lr=5e-5（基于前 7 次最佳方向组合，保持 label_smoothing=0.0）→ no_miss_val_acc=0.660, no_miss_val_spe=0.360, val_AUC=0.926 → discard

---


## 阶段 7：Dual-Granularity Adaptive Fusion (DGAF) ✅ 已完成（跳过末尾 2 次）

> **论文创新点**：双粒度自适应融合。
> 同时在特征级和决策级做融合，用 View-Aware Gate（视角置信度 + 分歧度 + 双分支 logits）
> 为每个样本动态选择最优融合路径。
> 6/8 次 proxy 全 discard，跳过 DGAF-07/08 转入 PFDF。
> 基线配置：share_backbone=false, aug=true, lr=1e-4, label_smoothing=0.0, dropout=0.3, fusion_hidden_dim=256

### 阶段 7A：DGAF 超参搜索（7/8 次 proxy，DGAF-08 跳过）

- [x] **DGAF-01**: baseline（fusion_type=decision, lr=1e-4, dropout=0.3, ls=0.0）→ no_miss_val_acc=0.755, no_miss_val_spe=0.540, val_AUC=0.930 → discard
- [x] **DGAF-02**: lr=5e-5 → no_miss_val_acc=0.723, no_miss_val_spe=0.480, val_AUC=0.946 → discard
- [x] **DGAF-03**: lr=7e-5 → no_miss_val_acc=0.723, no_miss_val_spe=0.480, val_AUC=0.937 → discard
- [x] **DGAF-04**: dropout=0.2 → no_miss_val_acc=0.777, no_miss_val_spe=0.580, val_AUC=0.933 → discard
- [x] **DGAF-05**: dropout=0.4 → no_miss_val_acc=0.777, no_miss_val_spe=0.580, val_AUC=0.933 → discard
- [x] **DGAF-06**: label_smoothing=0.05 → no_miss_val_acc=0.681, no_miss_val_spe=0.400, val_AUC=0.932 → discard
- [x] **DGAF-07**: lr=5e-5 + label_smoothing=0.05 → no_miss_val_acc=0.702, no_miss_val_spe=0.440, val_AUC=0.946 → discard
- [x] ~~**DGAF-08**~~: 已跳过（转入 PFDF，2026-04-06）

---

## 阶段 8：Progressive Feature Distillation Fusion (PFDF) ✅ 已完成（8/8 proxy，全 discard）

> **论文创新点**：渐进式特征蒸馏融合。
> 通过两阶段两两交叉注意力渐进融合三个视角特征：
> Stage 1: Fuse(axial, coronal) → 512D；Stage 2: Fuse(result, sagittal) → 512D。
> 分类器只需处理 512 维（而非暴力拼接的 1536 维），降低过拟合风险。
> 基线配置：share_backbone=false, aug=true, lr=1e-4, label_smoothing=0.0, dropout=0.3, fusion_hidden_dim=256

### 阶段 8A：PFDF 超参搜索（8 次 proxy）

- [x] **PFDF-01**: baseline（fusion_type=decision, lr=1e-4, dropout=0.3, ls=0.0）→ no_miss_val_acc=0.702, no_miss_val_spe=0.440, val_AUC=0.920 → discard
- [x] **PFDF-02**: lr=5e-5 → no_miss_val_acc=0.787, no_miss_val_spe=0.600, val_AUC=0.920 → discard
- [x] **PFDF-03**: lr=7e-5 → no_miss_val_acc=0.670, no_miss_val_spe=0.380, val_AUC=0.893 → discard
- [x] **PFDF-04**: dropout=0.2 → no_miss_val_acc=0.543, no_miss_val_spe=0.140, val_AUC=0.920 → discard
- [x] **PFDF-05**: dropout=0.4 → no_miss_val_acc=0.500, no_miss_val_spe=0.060, val_AUC=0.903 → discard
- [x] **PFDF-06**: label_smoothing=0.05 → no_miss_val_acc=0.745, no_miss_val_spe=0.520, val_AUC=0.934 → discard
- [x] **PFDF-07**: lr=5e-5 + label_smoothing=0.05 → no_miss_val_acc=0.766, no_miss_val_spe=0.560, val_AUC=0.924 → discard
- [x] **PFDF-08**: 基于前 7 次最佳方向的组合实验（沿用最佳 lr=5e-5，并恢复 baseline label_smoothing=0.0）→ no_miss_val_acc=0.755, no_miss_val_spe=0.540, val_AUC=0.916 → discard

---

## 阶段 9：Cross-View Feature Interaction Fusion (CVFI) ✅ 已完成（8/8 proxy，全 discard）

> **论文创新点**：跨视角特征交互融合。
> 在标准特征拼接（一阶）的基础上，增加视角两两之间的**逐元素乘积交互项**（二阶），
> 通过投影层压缩到 128 维。交互项显式建模跨视角特征通道的共激活模式，
> 让分类器能利用视角间的一致性/不一致性信号。
> concat([v1, v2, v3, proj(v1⊙v2), proj(v2⊙v3), proj(v1⊙v3)]) = 1920D → MLP → 2
> 基线配置：share_backbone=false, aug=true, lr=1e-4, label_smoothing=0.0, dropout=0.3, fusion_hidden_dim=256

### 阶段 9A：CVFI 超参搜索（8 次 proxy）

- [x] **CVFI-01**: baseline（fusion_type=decision, lr=1e-4, dropout=0.3, ls=0.0）→ no_miss_val_acc=0.585, no_miss_val_spe=0.220, val_AUC=0.915 → discard
- [x] **CVFI-02**: lr=5e-5 → no_miss_val_acc=0.553, no_miss_val_spe=0.160, val_AUC=0.902 → discard
- [x] **CVFI-03**: lr=7e-5 → no_miss_val_acc=0.628, no_miss_val_spe=0.300, val_AUC=0.911 → discard
- [x] **CVFI-04**: dropout=0.2 → no_miss_val_acc=0.734, no_miss_val_spe=0.500, val_AUC=0.933 → discard
- [x] **CVFI-05**: dropout=0.4 → no_miss_val_acc=0.681, no_miss_val_spe=0.400, val_AUC=0.923 → discard
- [x] **CVFI-06**: label_smoothing=0.05 → no_miss_val_acc=0.553, no_miss_val_spe=0.160, val_AUC=0.896 → discard
- [x] **CVFI-07**: lr=5e-5 + label_smoothing=0.05 → no_miss_val_acc=0.489, no_miss_val_spe=0.040, val_AUC=0.904 → discard
- [x] **CVFI-08**: 基于前 7 次最佳方向的组合实验（沿用最佳 dropout=0.2，并恢复 baseline lr=1e-4 / label_smoothing=0.0）→ no_miss_val_acc=0.723, no_miss_val_spe=0.480, val_AUC=0.933 → discard

---

## 旧阶段（已完成）

### 阶段 1：Feature Fusion 超参优化（已完结）

- 暂无高优先级待办

### 阶段 2：Decision Fusion 对比（已完结）

- 暂无高优先级待办

### 阶段 3：论文就绪验证 🔴 当前最高优先级

- [x] **多 seed 验证**（seed=42, 123, 456 已完成）
  - [x] seed=42（`053cb3a`，当前全局最优基线）→ no_miss_val_acc=0.798, no_miss_val_spe=0.620, val_AUC=0.960 → keep
  - [x] seed=123（`b12f03d`，VRG 当前最优配置 proxy 验证）→ no_miss_val_acc=0.787, no_miss_val_spe=0.600, val_AUC=0.952 → discard
  - [x] seed=456（`f3c0157`，VRG 当前最优配置 proxy 验证）→ no_miss_val_acc=0.521, no_miss_val_spe=0.100, val_AUC=0.923 → discard
- [x] **Bootstrap 置信区间**（1000 次重采样；`7f4d793`，VRG 最优配置 seed=42 rerun）
  - [x] 验证集重采样并重新估计零漏诊阈值：threshold 中位数=0.332，95% CI [0.332, 0.399]
  - [x] 零漏诊验证指标：no_miss_val_acc=0.830（bootstrap 95% CI [0.766, 0.957]），no_miss_val_spe=0.680（95% CI [0.574, 0.927]）
  - [x] 固定阈值测试集指标：acc=0.737（95% CI [0.526, 0.895]），spe=0.900（95% CI [0.667, 1.000]），sen=0.556（95% CI [0.200, 0.875]）
  - [x] proxy rerun 摘要：val_AUC=0.970，no_miss_val_acc=0.830 → keep ✅ 新当前最优
- [ ] **三种融合方式 formal 对比**
  - [ ] Feature Fusion formal
    - [x] proxy gate（`d32d660`，non-shared, aug=true, lr=1e-4, d=0.3, ls=0.05）→ no_miss_val_acc=0.798, no_miss_val_spe=0.620, val_AUC=0.932 → discard（未超过当前最优 0.830，暂不升 formal）
  - [ ] Decision Fusion formal（当前最优 VRG + gradient clip）
    - [x] proxy gate / sanity rerun（`af28838`，non-shared, aug=true, lr=5e-5, d=0.3, ls=0.0）→ no_miss_val_acc=0.755, no_miss_val_spe=0.540, val_AUC=0.949 → discard（未复现当前最优 `7f4d793` 的 0.830，formal 仍待基于 keep 提交推进）
    - [x] proxy stability rerun #2（`03bdce9`，non-shared, aug=true, lr=5e-5, d=0.3, ls=0.0）→ no_miss_val_acc=0.638, no_miss_val_spe=0.320, val_AUC=0.937 → discard（再次低于当前最优 `7f4d793` 的 0.830，formal 暂不推进）
    - [x] proxy stability rerun #3 / raw confidence logits（`ef6008c`，移除 confidence sigmoid 后直接 softmax；non-shared, aug=true, lr=5e-5, d=0.3, ls=0.0）→ no_miss_val_acc=0.734, no_miss_val_spe=0.500, val_AUC=0.914 → discard（较 `03bdce9` 有所回升，但仍低于当前最优 `7f4d793` 的 0.830，formal 暂不推进）
    - [x] proxy stability rerun #4 / restore canonical sigmoid gate（`20a402b`，恢复标准 confidence sigmoid + softmax；non-shared, aug=true, lr=5e-5, d=0.3, ls=0.0）→ no_miss_val_acc=0.585, no_miss_val_spe=0.220, val_AUC=0.941 → discard（与 `7f4d793` 同配方仍未复现，且比 rerun #3 更差，formal 继续搁置）
    - [x] proxy stabilization probe / cosine scheduler（`49e93fb`，non-shared, aug=true, lr=5e-5, d=0.3, ls=0.0, scheduler=cosine）→ no_miss_val_acc=0.628, no_miss_val_spe=0.300, val_AUC=0.938 → discard（较 canonical rerun #4 略有回升，但仍显著低于当前最优 `7f4d793` 的 0.830，formal 继续搁置）
    - [x] proxy stabilization probe / gradient clip（`814f491`，non-shared, aug=true, lr=5e-5, d=0.3, ls=0.0, gradient_clip_norm=1.0）→ no_miss_val_acc=0.872, no_miss_val_spe=0.760, val_AUC=0.966 → keep ✅ 新当前最优（已超过 `7f4d793` 的 0.830，下一步推进 Decision Fusion formal）
    - [x] proxy stability rerun / seed=123 + gradient clip（`1272bc6`，non-shared, aug=true, lr=5e-5, d=0.3, ls=0.0, gradient_clip_norm=1.0, seed=123）→ no_miss_val_acc=0.830, no_miss_val_spe=0.680, val_AUC=0.960 → discard（未超过当前最优 `814f491` 的 0.872，但与 `7f4d793` 的 0.830 持平，说明 clipped VRG 在 alternate seed 上仍具竞争力）
    - [x] proxy formal-preflight / seed=42 clipped VRG rerun + formal config sync（`50be764`，non-shared, aug=true, lr=5e-5, d=0.3, ls=0.0, gradient_clip_norm=1.0, seed=42）→ no_miss_val_acc=0.862, no_miss_val_spe=0.740, val_AUC=0.969 → discard（val_AUC 高于当前最优 `814f491` 的 0.966，但零漏诊 val_acc 仍低 0.011；formal 继续以 `814f491` 为基准推进）
    - [x] proxy formal-geometry smoke（`601351e`，non-shared, aug=true, lr=5e-5, d=0.3, ls=0.0, gradient_clip_norm=1.0, image_size=256, num_slices_per_view=32）→ no_miss_val_acc=0.691, no_miss_val_spe=0.420, val_AUC=0.930 → discard（4 epoch formal geometry proxy 明显退化，较当前最优 `814f491` 低 0.181；已触发连续 3 次 discard，formal 前需先重审稳定化方案）
    - [x] proxy formal-geometry stabilization / edge trimming（`18a19d7`，non-shared, aug=true, lr=5e-5, d=0.3, ls=0.0, gradient_clip_norm=1.0, image_size=256, num_slices_per_view=32, trim_edge_slices=2）→ no_miss_val_acc=0.777, no_miss_val_spe=0.580, val_AUC=0.941 → discard（较 `601351e` 回升 0.085，说明 32-slice formal geometry 的退化部分来自边缘切片稀释；但仍低于当前最优 `814f491` 的 0.872，formal 继续搁置）
    - [x] proxy formal-geometry stabilization / edge trimming + cosine scheduler（`e4b2175`，non-shared, aug=true, lr=5e-5, d=0.3, ls=0.0, gradient_clip_norm=1.0, image_size=256, num_slices_per_view=32, trim_edge_slices=2, scheduler=cosine）→ no_miss_val_acc=0.798, no_miss_val_spe=0.620, val_AUC=0.955 → discard（较 `18a19d7` 再回升 0.021，并恢复到 `7f4d793` / `d32d660` 同级水平；但仍低于当前最优 `814f491` 的 0.872，说明 formal geometry + cosine 仍未补平 256x32 的性能落差）
    - [x] formal run（`d63b49a`，沿用 `814f491` 配方 + native 192x16 geometry，15 epochs；non-shared, aug=true, lr=5e-5, d=0.3, ls=0.0, gradient_clip_norm=1.0）→ no_miss_val_acc=**0.915**, no_miss_val_spe=0.840, val_AUC=0.968 → keep ✅ 新全局最优
  - [ ] Attention Fusion formal
    - [x] proxy gate（`1d0bb82`，shared, aug=false, lr=1e-4, d=0.3, heads=4, layers=2）→ no_miss_val_acc=0.723, no_miss_val_spe=0.480, val_AUC=0.919 → discard（低于当前最优 0.830，暂不升 formal）

### 阶段 4：创新融合架构探索 ✅ 已完成

- 阶段 4A：View Reliability Gating（10 次 proxy，已完结）
- 阶段 4B：Asymmetric Safety-Biased Fusion（5 次 proxy，已完结）

---

## 已完成 / 已废弃实验（历史记录）

### Feature Fusion — 共享 backbone 阶段（Round 1, 已完结）

- [x] 共享 backbone baseline (lr=1e-4, d=0.3) → val_AUC=0.927 → discard
- [x] 共享 backbone lr=5e-5 → val_AUC=0.946 → keep（AUC 体系下最优）
- [x] 共享 backbone lr=3e-4 → discard
- [x] 共享 backbone dropout=0.2 → discard
- [x] 共享 backbone dropout=0.4 → discard
- [x] 共享 backbone CosineAnnealingLR → discard
- [x] 共享 backbone formal (lr=5e-5) → val_AUC=0.940, test_AUC=0.822

### Feature Fusion — 非共享 backbone 阶段（Round 2, 已完结）

- [x] 非共享 baseline (lr=1e-4, d=0.3) → no_miss_val_acc=0.755 → keep
- [x] 非共享 lr=5e-5 → no_miss_val_acc=0.660 → discard
- [x] 非共享 dropout=0.4 → no_miss_val_acc=0.691 → discard
- [x] 非共享 lr=7e-5 → no_miss_val_acc=0.691 → discard
- [x] 非共享 + augmentation=true → no_miss_val_acc=0.777 → keep
- [x] 非共享 + augmentation=true formal 确认 → no_miss_val_acc=0.755, val_AUC=0.946 → keep
- [x] 非共享 + augmentation=true + gradient_clip_norm=1.0 → discard
- [x] **非共享 + augmentation=true + label_smoothing=0.05 → no_miss_val_acc=0.777, val_AUC=0.948 → keep ✅ Feature Fusion 最优**
- [x] 非共享 + aug + lr=5e-5 → discard
- [x] 非共享 + aug + ls=0.05 + wd=5e-4 → discard
- [x] 非共享 + aug + ls=0.05 + d=0.4 → discard
- [x] 非共享 + aug + ls=0.1 → discard
- [x] 非共享 + aug + ls=0.05 + clip=1.0 → discard
- [x] 非共享 + aug + ls=0.05 + clip=0.5 → discard
- [x] 非共享 + aug + ls=0.05 + early_stopping=5 → discard

### Decision Fusion 阶段（Round 3, 已完结）

- [x] Decision Fusion proxy baseline（aug, ls=0.05）→ discard
- [x] Decision Fusion proxy lr=5e-5 → discard
- [x] Decision Fusion proxy lr=7e-5 → discard
- [x] **Decision Fusion proxy 去掉 ls（lr=1e-4）→ no_miss_val_acc=0.787 → keep（曾为全局最优）**
- [x] Decision Fusion proxy lr=3e-4 → discard
- [x] Decision Fusion proxy dropout=0.2 → discard
- [x] Decision Fusion proxy dropout=0.4 → discard
- [x] Decision Fusion proxy clip=0.5 → discard
- [x] Decision Fusion proxy hidden=128 → discard
- [x] Decision Fusion proxy 6 epoch → discard
- [x] Decision Fusion proxy hidden=384 → discard
- [x] Decision Fusion proxy cosine → discard
- [x] Decision Fusion proxy 概率投票 → discard

### 阶段 4A：View Reliability Gating 阶段（Round 4, 已完结）

- [x] VRG-01 ~ VRG-10：均 discard，最佳 no_miss_val_acc=0.755（VRG-02, VRG-03）

### 阶段 4B：Asymmetric Safety-Biased Fusion 阶段（Round 5, 已完结）

- [x] ASF-01: temperature=0.5 → no_miss_val_acc=0.681 → discard
- [x] ASF-02: temperature=1.0 → no_miss_val_acc=0.532 → discard
- [x] ASF-03: temperature=0.3 → no_miss_val_acc=0.617 → discard
- [x] ASF-04: temperature=2.0 → no_miss_val_acc=0.649 → discard
- [x] ASF-05: temperature=0.5 + lr=5e-5 → no_miss_val_acc=0.660 → discard

---

## 维护规则

1. **Agent 每次启动时**：先读本文件，了解当前最优纪录和待办优先级
2. **每完成一个实验后**：
   - 如果 keep：更新"当前最优纪录"表格
   - 将实验从待办移到"已完成"，标注结果
   - 如果结果带来新的实验思路，添加到对应优先级
3. **连续 3 个 discard 后**：重新审视待办清单，考虑换方向
4. **人类编辑后**：Agent 下次启动时以文件内容为准

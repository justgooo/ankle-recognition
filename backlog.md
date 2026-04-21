# 实验待办清单 (Experiment Backlog)

> **本文件由 autoresearch Agent 维护。**
> Agent 在每次实验循环启动时必须阅读本文件，并在完成实验后根据结果更新。
> 人类也可以直接编辑本文件来添加想法或调整优先级。
>
> **⚠️ 指标体系切换**：本项目已从「零漏诊 (no_miss_val_acc)」切换到「验证集准确率 (val_acc)」。
> 旧实验记录使用 `no_miss_val_acc`，保留不变。新实验使用 `summary.json` 中的 `best_val.accuracy`。
> 新实验的 keep 判定基于 val_acc，不再使用 evaluate_threshold.py 作为主评估工具。
>
> **⚠️ 执行策略更新（2026-04-17）**：在单卡显存 `>= 24 GiB` 且当前算力充足的环境里，默认直接运行 `main/formal` lane；`proxy` 不再作为常规筛选步骤，仅保留给低显存 fallback 或快速诊断。

---

## 2026-04-21：人类方向追加约束（主线切回 ResNeXt decision 256x8 learned-weighting）

> **方向约束说明**
> - 人类已明确要求：后续 `decision fusion` 主线不再继续沿当前 `512x16` canonical lane 做方法收敛，而是切回 **`image_size=256` + `num_slices_per_view=8`** 的 `ResNeXt + decision fusion` 几何。
> - 研究目标同步改写为：**以 canonical learned-weighting `256x8` 配方作为唯一主方法方向**，优先做模块贡献分析与性能修复；`equal-weight` 仍必须保留为 matched control，但不再作为人类期望的最终方法叙事，也不再决定 learned 主线是否继续。
> - 上述“先不训练”限制已在同日被人类后续消息“开始启动模块贡献分析”正式解除；当前允许按 `256x8` module matrix 直接启动 matched formal 作业。
>
> **主线基座选择**
> - `256x8` learned-weighting 的主参考应优先锚定到 legacy `7ae19a0` 这条已知最强 learned recipe，而不是当前较弱的 fair-compare `256x8` decision 模板。
> - 当前仓库中的 canonical mainline config 已固定为 [configs/autoresearch_formal_resnext_decision_256x8.yaml](/dataset/HH/ankle-ct/configs/autoresearch_formal_resnext_decision_256x8.yaml)；它对应的 recipe 是：`backbone=resnext`、`fusion_type=decision`、`share_backbone=false`、`use_attention_pooling=false`、`image_size=256`、`num_slices_per_view=8`、`trim_edge_slices=2`、`batch_size=6`、`num_workers=12`、`epochs=15`、`freeze_layers=3`、`lr=1e-4`、`weight_decay=5e-4`、`dropout=0.25`、`gradient_clip_norm=2.0`。
> - 现有最关键的历史证据：
>   - legacy learned 3-seed mean：`0.900709 / 0.939545`
>   - legacy equal 3-seed mean：`0.936170 / 0.965606`
> - 这意味着新一轮模块贡献分析的目的，不是证明 learned 已经优于 equal，而是先回答：**learned weighting 在 `256x8` 上到底差在哪里，哪个模块组合能最稳定地提升 learned 自身的绝对性能与稳定性。**
>
> **待执行的模块贡献分析矩阵（当前进度：`L0/L1/L2/L3/L4` formal 已完成，`L5` 待继续）**
> - `L0 control`: legacy `256x8` equal-weight（只作 control / 报告参考，不作主方法）
> - `L1 anchor`: canonical `256x8` learned weighting baseline
> - `L2 minimal`: 去掉 richer reliability path，仅保留最基础 per-view classifier + raw confidence head
> - `L3 no-mixer`: 保留 calibrator，关闭 cross-view mixer
> - `L4 no-calibrator`: 保留 cross-view mixer，移除 shared low-rank calibrator
> - `L5 temperature`: 对当前最强 learned 变体只加单标量 temperature shrinkage（现阶段默认接到 `L3-no-mixer`）
>
> **分析优先级**
> - 第一优先级是把 **canonical learned `256x8` 配方** 固定成唯一主锚点，之后所有 `L2/L3/L4/L5` 都只围绕这一个 recipe 做 matched ablation，不再在多个 learned recipe family 之间来回切换。
> - 后续模块推进首先看：某个变体是否能提升 learned 主线自己的 `val_acc / val_auc / stability`；`equal-weight` 只负责提供 matched reference 和论文对照，不再作为 learned 主线继续与否的门槛。
> - 如果某个 ablation 能稳定改善 learned 主线，即使仍低于 equal-weight，也应保留为主线候选并继续细化；只有当同一 scalar budget 下多个单模块/低容量变体都无法改善 canonical learned，learned-weighting 主叙事才需要降级为“机制研究”。
> - 工程准备已完成：`256x8` learned lane 的 dedicated formal/proxy/search 模板已补到 `configs/autoresearch_*_resnext_decision_256x8.yaml` 与 `configs/optuna_*_resnext_decision_256x8.yaml`；模块矩阵清单写入 `docs/resnext_decision_256x8_module_matrix.md`。同时，`src/model.py` 已补 runtime toggles：`ANKLE_DISABLE_FUSION_CALIBRATOR=1` 与 `ANKLE_LEARNED_FUSION_TEMPERATURE=<float>`；`scripts/optuna_workflow.py` 与 `scripts/run_train_with_config_env.py` 也已把这些 env overrides 显式写入 trial/config 产物。现在 `scripts/prepare_resnext_decision_256x8_matrix.py` 还已把 `7 lanes × 3 seeds × 2 phases = 42` 份 runnable YAML 与 manifest materialize 到 `configs/generated_resnext_decision_256x8_matrix/`，后续做 `no-calibrator` / `temperature` probe 不再需要依赖隐式 shell 状态。

## 2026-04-22：ResNeXt Decision 256x8 Module Contribution（L3 no-mixer + L4 no-calibrator，formal multiseed，V100q node20）

> **独立 campaign 说明**
> - 这是在 `L2-minimal` 之后继续沿 canonical learned `256x8` 主线做的 matched 单模块 attribution 收口：同一 current scaffold、同一几何、同一 scalar budget 下，把 `L3-no-mixer` 与 `L4-no-calibrator` 一次性补齐到 `seed=42/123/456`。
> - 有效 batch 是 `432589`：`V100q/node20`，`1 node / 3 GPU / 36 CPU / 120G / 4h`；job 内 `CUDA_VISIBLE_DEVICES=0,1,2`，六个 `srun` step `432589.0-.5` 全部 `COMPLETED`。前三个 step 对应 `L3-no-mixer`，后三个 step 对应 `L4-no-calibrator`。本轮运行代码状态保持在 commit `27b557c`。
> - 配方继续保持严格 matched：`backbone=resnext`、`fusion_type=decision`、`image_size=256`、`num_slices_per_view=8`、`trim_edge_slices=2`、`batch_size=6`、`num_workers=12`、`epochs=15`、`freeze_layers=3`、`lr=1e-4`、`weight_decay=5e-4`、`dropout=0.25`、`gradient_clip_norm=2.0`；唯一变量分别是 `ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER=1` 与 `ANKLE_DISABLE_FUSION_CALIBRATOR=1`。

- [x] **CMP-RESNEXT-DECISION-256X8-L3-NO-MIXER-FORMAL-S42**：`runs/resnext_decision_256x8_matrix/formal/l3_no_mixer/s42`（commit `27b557c`，`ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER=1`）→ `val_acc=0.9255319148936170`, `val_auc=0.9777272727272727`, `val_f1=0.9213483146067416`, `peak_vram≈2.15 GiB`, `total_seconds≈896.2` → **keep**（相较 matched `L1-learned` seed42，accuracy / AUC / F1 分别提升 `0.0212765957446808 / 0.0045454545454546 / 0.0224719101123596`；这是当前 `256x8` learned 主线里第一个同时追平 equal accuracy 区间并把 AUC 拉到全批最高的单-seed no-mixer 结果。）
- [x] **CMP-RESNEXT-DECISION-256X8-L3-NO-MIXER-FORMAL-S123**：`runs/resnext_decision_256x8_matrix/formal/l3_no_mixer/s123`（commit `27b557c`，`ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER=1`）→ `val_acc=0.8936170212765957`, `val_auc=0.9395454545454545`, `val_f1=0.8750000000000000`, `peak_vram≈2.15 GiB`, `total_seconds≈887.7` → **keep**（相较 matched `L1-learned` seed123，accuracy / AUC / F1 分别提升 `0.0531914893617020 / 0.0263636363636363 / 0.0398351648351648`；这说明 mixer removal 的收益不是局限于单个低点 seed。）
- [x] **CMP-RESNEXT-DECISION-256X8-L3-NO-MIXER-FORMAL-S456**：`runs/resnext_decision_256x8_matrix/formal/l3_no_mixer/s456`（commit `27b557c`，`ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER=1`）→ `val_acc=0.9255319148936170`, `val_auc=0.9713636363636363`, `val_f1=0.9176470588235294`, `peak_vram≈2.14 GiB`, `total_seconds≈900.1` → **keep**（相较 matched `L1-learned` seed456，accuracy / AUC / F1 分别提升 `0.0319148936170213 / 0.0300000000000000 / 0.0287581699346406`；accuracy 还超过 matched `L0-equal` seed456 `0.0106382978723404`。）
- [x] **CMP-RESNEXT-DECISION-256X8-L4-NO-CALIBRATOR-FORMAL-S42**：`runs/resnext_decision_256x8_matrix/formal/l4_no_calibrator/s42`（commit `27b557c`，`ANKLE_DISABLE_FUSION_CALIBRATOR=1`）→ `val_acc=0.9255319148936170`, `val_auc=0.9509090909090908`, `val_f1=0.9156626506024096`, `peak_vram≈2.17 GiB`, `total_seconds≈887.0` → **keep**（相较 matched `L1-learned` seed42，accuracy / F1 分别提升 `0.0212765957446808 / 0.0167862461080276`，但 AUC 回落 `0.0222727272727273`；这个 seed 说明去掉 calibrator 仍能修复 accuracy 低点，但排序质量没有跟上。）
- [x] **CMP-RESNEXT-DECISION-256X8-L4-NO-CALIBRATOR-FORMAL-S123**：`runs/resnext_decision_256x8_matrix/formal/l4_no_calibrator/s123`（commit `27b557c`，`ANKLE_DISABLE_FUSION_CALIBRATOR=1`）→ `val_acc=0.8404255319148937`, `val_auc=0.9068181818181819`, `val_f1=0.8235294117647058`, `peak_vram≈2.17 GiB`, `total_seconds≈898.1` → **discard**（相较 matched `L1-learned` seed123，accuracy 持平，但 AUC / F1 分别回落 `0.0063636363636363 / 0.0116354234001294`；这说明 calibrator removal 至少没有带来可保留的 seed123 修复。）
- [x] **CMP-RESNEXT-DECISION-256X8-L4-NO-CALIBRATOR-FORMAL-S456**：`runs/resnext_decision_256x8_matrix/formal/l4_no_calibrator/s456`（commit `27b557c`，`ANKLE_DISABLE_FUSION_CALIBRATOR=1`）→ `val_acc=0.9255319148936170`, `val_auc=0.9595454545454545`, `val_f1=0.9176470588235294`, `peak_vram≈2.17 GiB`, `total_seconds≈901.9` → **keep**（相较 matched `L1-learned` seed456，accuracy / AUC / F1 分别提升 `0.0319148936170213 / 0.0181818181818182 / 0.0287581699346406`；但和 `L3` 同 seed 相比仍有稳定差距。）
- **3-seed learned-mainline 结论（L3）**：`L3-no-mixer` 的 mean `val_acc=0.9148936170212766`、mean `val_auc=0.9628787878787879`、mean `val_f1=0.9046651244767570`，`val_acc` population std 为 `0.0150448251316287`。相较 current-scaffold `L1-learned`，`L3` 分别提升 `+0.0354609929078015 val_acc`、`+0.0203030303030303 val_auc`、`+0.0303550816273882 val_f1`，同时把 `val_acc` std 从 `0.0279220137376305` 压到 `0.0150448251316287`。相较 matched `L0-equal`，`L3` 只剩 `-0.0070921985815602 val_acc` 差距，但 `val_auc` 已反超 `+0.0022727272727272`。
- **3-seed learned-mainline 结论（L4）**：`L4-no-calibrator` 的 mean `val_acc=0.8971631205673759`、mean `val_auc=0.9390909090909091`、mean `val_f1=0.8856130403968816`，`val_acc` population std 为 `0.0401195336843431`。相较 `L1-learned`，它虽然还有 `+0.0177304964539008 val_acc` 与 `+0.0113029975475128 val_f1`，但 `val_auc` 反而回落 `0.0034848484848484`，而且 `val_acc` std 恶化到 `0.0401195336843431`；说明 calibrator removal 不是稳定主线。
- **当前判断**：到这里，`L3-no-mixer` 已经成为当前 `256x8` learned 主线里最强、最稳、也最接近 equal 的 branch。相较之下，`L4-no-calibrator` 只能算部分 seed 的 accuracy repair，不能升格为新主线。`L2` 仍然是有效简化修复，但综合均值与 ceiling 都已经被 `L3` 超过。
- **推荐动作**：下一步不再把 `temperature` 直接加在 `L1` 上，而是把 **`L3-no-mixer` 作为 strongest learned branch** 继续做低容量 repair：优先提交 `L5-temp1p5` 与 `L5-temp2p0` 的 matched 3-seed formal，并在运行语义上明确为 **`L3-no-mixer + temperature`**，不是 `L1 + temperature`。

## 2026-04-22：ResNeXt Decision 256x8 Module Contribution（L2 minimal，formal multiseed，V100q node20）

> **独立 campaign 说明**
> - 这是在 canonical learned `256x8` 主线重新固定后的第一轮已完成单模块 ablation 收口：继续沿同一 current scaffold、同一几何、同一 scalar budget，只把 `model.minimal_fusion_baseline=true` 打开，检查 “去掉 richer reliability path” 是否能改善 learned 主线自身。
> - 有效 batch 是 `431651`：`V100q/node20`，`1 node / 3 GPU / 36 CPU / 120G / 4h`；job 内 `CUDA_VISIBLE_DEVICES=0,1,2`，三个 `srun` step `431651.0/.1/.2` 全部 `COMPLETED`。
> - 配方保持严格 matched：`backbone=resnext`、`fusion_type=decision`、`image_size=256`、`num_slices_per_view=8`、`trim_edge_slices=2`、`batch_size=6`、`num_workers=12`、`epochs=15`、`freeze_layers=3`、`lr=1e-4`、`weight_decay=5e-4`、`dropout=0.25`、`gradient_clip_norm=2.0`；唯一实验变量是 `model.minimal_fusion_baseline=true`。本轮 ledger 归档 commit 记为 `03b9211`。

- [x] **CMP-RESNEXT-DECISION-256X8-L2-MINIMAL-FORMAL-S42**：`runs/resnext_decision_256x8_matrix/formal/l2_minimal/s42`（commit `03b9211`，`minimal_fusion_baseline=true`）→ `val_acc=0.9042553191489362`, `val_auc=0.9581818181818182`, `val_f1=0.8860759493670886`, `peak_vram≈2.14 GiB`, `total_seconds≈891.9` → **discard**（相较 matched `L1-learned` seed42，accuracy 持平，但 AUC / F1 分别回落 `0.0150000000000000 / 0.0128004551272934`；这说明 minimal path 没有修复 canonical seed42 低点。）
- [x] **CMP-RESNEXT-DECISION-256X8-L2-MINIMAL-FORMAL-S123**：`runs/resnext_decision_256x8_matrix/formal/l2_minimal/s123`（commit `03b9211`，`minimal_fusion_baseline=true`）→ `val_acc=0.8829787234042553`, `val_auc=0.9540909090909090`, `val_f1=0.8571428571428571`, `peak_vram≈2.14 GiB`, `total_seconds≈901.8` → **keep**（相较 matched `L1-learned` seed123，accuracy / AUC / F1 分别提升 `0.0425531914893617 / 0.0409090909090908 / 0.0219780219780219`；这是当前 `256x8` learned 主线里最明确的单-seed 修复。）
- [x] **CMP-RESNEXT-DECISION-256X8-L2-MINIMAL-FORMAL-S456**：`runs/resnext_decision_256x8_matrix/formal/l2_minimal/s456`（commit `03b9211`，`minimal_fusion_baseline=true`）→ `val_acc=0.9148936170212766`, `val_auc=0.9595454545454545`, `val_f1=0.9047619047619048`, `peak_vram≈2.14 GiB`, `total_seconds≈900.9` → **keep**（相较 matched `L1-learned` seed456，accuracy / AUC / F1 分别提升 `0.0212765957446809 / 0.0181818181818182 / 0.0158730158730159`；accuracy 还追平了 matched `L0-equal`，只是 AUC 仍低 `0.0018181818181819`。）
- **3-seed learned-mainline 结论**：`L2-minimal` 的 mean `val_acc=0.9007092198581560`、mean `val_auc=0.9572727272727273`、mean `val_f1=0.8826602370906168`，`val_acc` population std 为 `0.0132682886055814`。相较 current-scaffold `L1-learned` 的 `0.8794326241134751 / 0.9425757575757575 / 0.8743100428493688`，`L2` 分别提升 `+0.0212765957446808 val_acc`、`+0.0146969696969698 val_auc`、`+0.0083501942412482 val_f1`，同时把 `val_acc` std 从 `0.0279220137376305` 压到 `0.0132682886055814`。
- **对 equal 的参考差距**：`L2-minimal` 仍低于 matched `L0-equal` mean `0.9219858156028368 / 0.9606060606060606 / 0.9112221100424511`，差距是 `-0.0212765957446809 val_acc`、`-0.0033333333333334 val_auc`、`-0.0285618729518343 val_f1`。但按照当前主线口径，这只作为参考差距，不作为是否保留 `L2` 的裁决门槛。
- **当前判断**：`L2-minimal` 是当前 `256x8` learned 主线下第一条 **mean accuracy / mean AUC / stability 同时改善** 的 retained ablation。它还不能替代 `equal` 当 overall winner，但已经足够作为“learned 主线可被结构性修复”的正证据。
- **推荐动作**：下一步不要围绕 `equal` 决策，而是继续按单模块 attribution 收口：优先补 `L3-no-mixer` 与 `L4-no-calibrator` 的 matched 3-seed formal，对齐同一 `L1` canonical anchor，确认 `L2` 的净收益到底更接近“移除 mixer”还是“移除 calibrator / richer path”的哪一部分。

## 2026-04-21：ResNeXt Decision 256x8 Matched Controls（L0 equal vs L1 learned，formal multiseed，V100q node20）

> **独立 campaign 说明**
> - 这是在人类正式放行“开始启动模块贡献分析”后的第一批 `256x8` formal 矩阵作业：直接用 [scripts/slurm_resnext_decision_256x8_matrix.sbatch](/dataset/HH/ankle-ct/scripts/slurm_resnext_decision_256x8_matrix.sbatch) 在同一节点上按 `3 GPU` 并行跑完 `L0-equal` 与 `L1-learned` 的 `seed=42/123/456`。
> - 有效 batch 是 `431122`：`V100q/node20`，`1 node / 3 GPU / 36 CPU / 120G / 4h`；job 内 `CUDA_VISIBLE_DEVICES=0,1,2`，六个 `srun` step `431122.0-.5` 全部 `COMPLETED`。首个提交 `431106` 因 launcher 预检没有导出 `MIN_FREE_GB` 而在训练前 fail-fast，不计入实验结论；该提交已由 commit `8e0bdf4` 修复。
> - 配方保持严格 matched：`backbone=resnext`、`fusion_type=decision`、`image_size=256`、`num_slices_per_view=8`、`trim_edge_slices=2`、`batch_size=6`、`num_workers=12`、`epochs=15`、`freeze_layers=3`、`lr=1e-4`、`weight_decay=5e-4`、`dropout=0.25`、`gradient_clip_norm=2.0`；唯一变量是 `L0` 的 `model.equal_weight_fusion=true` 与 `L1` 的 learned weighting anchor。

- [x] **CMP-RESNEXT-DECISION-256X8-L0-EQUAL-FORMAL-S42**：`runs/resnext_decision_256x8_matrix/formal/l0_equal/s42`（commit `8e0bdf4`，matched control `equal_weight_fusion=true`）→ `val_acc=0.9255319148936170`, `val_auc=0.9486363636363637`, `val_f1=0.9176470588235294`, `peak_vram≈2.14 GiB`, `total_seconds≈900.8` → **keep**（相较 matched `L1-learned` seed42 `0.9042553191489362 / 0.9731818181818181`，accuracy 提升 `0.0212765957446808`，但 AUC 回落 `0.0245454545454544`；control 继续证明 learned 在这个 seed 上仍有 accuracy 劣势。）
- [x] **CMP-RESNEXT-DECISION-256X8-L0-EQUAL-FORMAL-S123**：`runs/resnext_decision_256x8_matrix/formal/l0_equal/s123`（commit `8e0bdf4`，matched control `equal_weight_fusion=true`）→ `val_acc=0.9255319148936170`, `val_auc=0.9718181818181818`, `val_f1=0.9135802469135802`, `peak_vram≈2.14 GiB`, `total_seconds≈903.3` → **keep**（相较 matched `L1-learned` seed123 `0.8404255319148937 / 0.9131818181818182`，accuracy / AUC 分别提升 `0.0851063829787233 / 0.0586363636363636`；这是本轮最大的 cross-seed gap。）
- [x] **CMP-RESNEXT-DECISION-256X8-L0-EQUAL-FORMAL-S456**：`runs/resnext_decision_256x8_matrix/formal/l0_equal/s456`（commit `8e0bdf4`，matched control `equal_weight_fusion=true`）→ `val_acc=0.9148936170212766`, `val_auc=0.9613636363636364`, `val_f1=0.9024390243902439`, `peak_vram≈2.14 GiB`, `total_seconds≈902.4` → **keep**（相较 matched `L1-learned` seed456 `0.8936170212765957 / 0.9413636363636363`，accuracy / AUC 分别提升 `0.0212765957446809 / 0.0200000000000001`。）
- [x] **CMP-RESNEXT-DECISION-256X8-L1-LEARNED-FORMAL-S42**：`runs/resnext_decision_256x8_matrix/formal/l1_learned/s42`（commit `8e0bdf4`，legacy-style learned-weighting anchor）→ `val_acc=0.9042553191489362`, `val_auc=0.9731818181818181`, `val_f1=0.8988764044943820`, `peak_vram≈2.17 GiB`, `total_seconds≈904.2` → **discard**（虽然 AUC 高于 matched `L0-equal` `0.0245454545454544`，但主指标 accuracy 仍落后 `0.0212765957446808`。）
- [x] **CMP-RESNEXT-DECISION-256X8-L1-LEARNED-FORMAL-S123**：`runs/resnext_decision_256x8_matrix/formal/l1_learned/s123`（commit `8e0bdf4`，legacy-style learned-weighting anchor）→ `val_acc=0.8404255319148937`, `val_auc=0.9131818181818182`, `val_f1=0.8351648351648352`, `peak_vram≈2.17 GiB`, `total_seconds≈905.3` → **discard**（相较 matched `L0-equal` seed123 `0.9255319148936170 / 0.9718181818181818`，accuracy / AUC 分别回落 `0.0851063829787233 / 0.0586363636363636`。）
- [x] **CMP-RESNEXT-DECISION-256X8-L1-LEARNED-FORMAL-S456**：`runs/resnext_decision_256x8_matrix/formal/l1_learned/s456`（commit `8e0bdf4`，legacy-style learned-weighting anchor）→ `val_acc=0.8936170212765957`, `val_auc=0.9413636363636363`, `val_f1=0.8888888888888888`, `peak_vram≈2.17 GiB`, `total_seconds≈901.0` → **discard**（相较 matched `L0-equal` seed456 `0.9148936170212766 / 0.9613636363636364`，accuracy / AUC 分别回落 `0.0212765957446809 / 0.0200000000000001`。）
- **3-seed matched control 结论**：`L0-equal` 的 mean `val_acc=0.9219858156028368`、mean `val_auc=0.9606060606060606`、mean `val_f1=0.9112221100424511`，`val_acc` population std 为 `0.0050149417105429`；`L1-learned` 的对应均值是 `0.8794326241134751 / 0.9425757575757575 / 0.8743100428493688`，`val_acc` population std 为 `0.0279220137376305`。在完全 matched 的当前 scaffold 下，equal 相对 learned 的 mean 增益是 `+0.0425531914893617 val_acc`、`+0.0180303030303031 val_auc`、`+0.0369120671930823 val_f1`。
- **当前判断**：这批第一性对照已经把主线问题重新钉死了。即便把几何、budget、seed 集和 runtime provenance 全部对齐到同一个 current scaffold，`learned weighting` 仍然在 3 个 matched seed 上全部输给 `equal-weight` 的主指标，而且 stability 明显更差。下一步没有理由先拆 `L3/L4/L5`；应先继续跑 `L2-minimal`，确认 “更简单的 learned path” 是否至少能缩小这条基础差距。
- **备注**：本轮 current-code `L0/L1` 两条线都低于 detached legacy rerun 的 `1b84e85` equal `0.936170 / 0.965606` 与 `7ae19a0` learned `0.900709 / 0.939545`，说明当前 scaffold 与 detached legacy recipe 之间仍存在一定实现/语义漂移；但由于本轮所有 lane 都在同一 current scaffold 下 matched，对模块贡献结论本身仍然有效。

## 2026-04-21：ResNeXt Fusion-Path Ablation（remove cross-view mixer, keep shared calibrator，seed=123/456，direct single-trial Slurm completion）

> **独立 campaign 说明**
> - 这是在人类明确要求“把剩下的补完”之后，对上一轮 `seed=42` no-mixer keep 的 **剩余 matched seed 收口**：不再通过 `autoresearch_loop.py` 外层 coordinator 调度，而是直接提交两个 fresh、单 trial 的 `scripts/optuna_main.py --sequential` main-study 作业，把 **`remove cross-view mixer + keep shared calibrator`** 一次性补到 `seed=123/456`。
> - 两个 fresh search-config copy 分别为 `autoresearch_logs/generated_search_configs/optuna_main_search_nomixer_seed123_20260421_151406.yaml` 与 `autoresearch_logs/generated_search_configs/optuna_main_search_nomixer_seed456_20260421_151439.yaml`；对应 study_root 为 `runs/optuna_main_manual/nomixer_seed123_20260421_151406` 与 `runs/optuna_main_manual/nomixer_seed456_20260421_151439`。执行 commit 分别是 `2cf4602`（seed123）和 `e3cc52b`（seed456）。
> - 两个 Slurm job 为 `430284` 与 `430285`，都跑在 `V100q/node19`，资源都是 `1 GPU / 12 CPU / 48G / 3h`。两单被 colocate 到同一节点并并发运行，所以单 trial `runtime.total_seconds` 分别拉长到了 `6866.4s / 6840.5s`，明显高于此前 node20 上 `~2900s` 的同类 single-trial；这是 **节点/并发吞吐退化**，不是 `autoresearch_loop` 还在继续迭代。
> - 配方保持严格 matched：`backbone=resnext`、`fusion_type=decision`、`image_size=512`、`num_slices_per_view=16`、`trim_edge_slices=2`、`batch_size=2`、`num_workers=4`、`epochs=20`、`freeze_layers=3`、`lr=1e-4`、`weight_decay=1e-4`、`dropout=0.3`、`gradient_clip_norm=1.0`；唯一实验变量仍然是 `ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER=1`。

- [x] **CMP-FUSION-PATH-RESNEXT-NOMIXER-512X16-E20-S123-MAIN**：fresh direct main-study `runs/optuna_main_manual/nomixer_seed123_20260421_151406`（trial `0`，`seed=123`，remove cross-view mixer while keeping shared calibrator）→ `val_acc=0.8191489361702128`, `val_auc=0.9050000000000000`, `val_f1=0.7848101265822784`, `peak_vram≈4.64 GiB`, `total_seconds≈6866.4` → **discard**（相较 matched `current learned` seed123 `0.8510638297872340 / 0.9109090909090910`，accuracy / AUC 分别回落 `0.0319148936170212 / 0.0059090909090910`；相较 matched `equal-weight` seed123 `0.8404255319148937 / 0.8663636363636364`，accuracy 也回落 `0.0212765957446809`。）
- [x] **CMP-FUSION-PATH-RESNEXT-NOMIXER-512X16-E20-S456-MAIN**：fresh direct main-study `runs/optuna_main_manual/nomixer_seed456_20260421_151439`（trial `0`，`seed=456`，remove cross-view mixer while keeping shared calibrator）→ `val_acc=0.8297872340425532`, `val_auc=0.8795454545454546`, `val_f1=0.8048780487804879`, `peak_vram≈4.64 GiB`, `total_seconds≈6840.5` → **discard**（相较 matched `current learned` seed456 `0.8510638297872340 / 0.9209090909090910`，accuracy / AUC 分别回落 `0.0212765957446808 / 0.0413636363636364`；相较 matched `equal-weight` seed456 `0.8404255319148937 / 0.8890909090909092`，accuracy / AUC 也分别回落 `0.0106382978723405 / 0.0095454545454546`。）
- **3-seed 收口结论（no-mixer + shared calibrator）**：补完 `seed=123/456` 后，这条 simpler learned path 的 canonical 3-seed mean 现为 `val_acc=0.8297872340425533`、`val_auc=0.8971212121212121`、`val_f1=0.7982943268525239`，`val_acc` population std 为 `0.0086861338396567`。相较 `current learned` 的 `0.8475177304964540 / 0.9063636363636364`，mean accuracy / AUC 分别回落 `0.0177304964539007 / 0.0092424242424243`；相较 `equal-weight` 的 `0.8546099290780141 / 0.8972727272727273`，mean accuracy 也回落 `0.0248226950354609`，AUC 仅近乎打平。
- **模块分析总判断**：`seed=42` 上的 no-mixer keep 最终被证明是 **局部 seed low-point 修复**，不是可泛化的主线替代。到这里，模块贡献分析已经足够回答核心问题：`minimal learned` 不成立、`no-calibrator` 不成立、`no-mixer` 也没有跨 seed 站住。因此当前 canonical `512x16` 主线里，不能把“删掉 cross-view mixer”直接升格为新的 learned-path 默认解。
- **推荐动作**：模块贡献分析到这里可以视为完成，后续不必继续沿 `richer reliability path` 做更多 matched ablation。下一步应回到人类批准顺序里的 **低容量 weighting-ratio 优化**，并恢复以 `current learned` 为 learned-path 参考；最优先的剩余问题是补 `seed=456` 上的 `temperature=1.5` 或其它单标量 shrinkage control，而不是继续拆模块。

## 2026-04-21：ResNeXt Fusion-Path Ablation（remove cross-view mixer, keep shared calibrator，seed=42，adaptive main-study fixed trial）

> **独立 campaign 说明**
> - 这是在人类要求“先做 matched 模块贡献分析，再考虑 weighting ratio”之后，严格按上一轮 backlog 推荐动作执行的 **下一步单模块 richer reliability path 拆解**：不再扫 temperature，也不扩容 gating / weighting path，只测试 `current learned` 路径里 **cross-view mixer** 本身是否在制造 `seed=42` 的主指标方差。
> - 为避免把“恢复 canonical current learned”与“本轮 ablation”混成两件事，实现侧在 commit `cf392fe` 中把默认 learned path 恢复成 **`cross-view mixer + shared low-rank calibrator`**，然后仅通过环境变量 `ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER=1` 做本轮对照：**去掉 cross-view mixer、保留 shared calibrator**。
> - 配方继续保持严格 matched：`backbone=resnext`、`fusion_type=decision`、`image_size=512`、`num_slices_per_view=16`、`trim_edge_slices=2`、`batch_size=2`、`num_workers=4`、`epochs=20`、`freeze_layers=3`、`lr=1e-4`、`weight_decay=1e-4`、`dropout=0.3`、`gradient_clip_norm=1.0`；唯一实验变量是 reliability path 从 **`cross-view mixer + shared calibrator`** 改成 **`shared calibrator on pooled per-view features`**。
> - 本轮运行 commit 为 `cf392fe`；fresh search-config copy 为 `autoresearch_logs/generated_search_configs/optuna_main_search_iter_0092_20260421_122116.yaml`，fresh study_root 为 `runs/optuna_main_autoloop/iter_0092_20260421_122116`。
> - 运行前按 adaptive GPU policy 检查了 `GPU 0/1/2` 的 idle 状态，三张 `Tesla V100-PCIE-32GB` 都满足 `used<=1024 MiB`、`util<=20%`；本轮仍只有 1 个 fixed trial，最终只消耗了其中 1 张卡的训练时长，其余卡保持空闲。

- [x] **CMP-FUSION-PATH-RESNEXT-NOMIXER-512X16-E20-S42-MAIN**：fresh adaptive main-study `runs/optuna_main_autoloop/iter_0092_20260421_122116` 的 best completed trial（trial `0`，`seed=42`，remove cross-view mixer while keeping shared calibrator via `ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER=1`）→ `val_acc=0.8404255319148937`, `val_auc=0.9068181818181819`, `val_f1=0.8051948051948052`, `peak_vram≈4.64 GiB`, `total_seconds≈2901.1` → **keep**（相较 matched `current learned` seed42 `0.8404255319148937 / 0.8863636363636364`，accuracy 持平但 AUC 提升 `0.0204545454545455`；相较上一轮 `no-calibrator` seed42 `0.7978723404255319 / 0.8781818181818182`，accuracy / AUC 分别反弹 `0.0425531914893618 / 0.0286363636363637`；虽然仍低于 matched `equal-weight` seed42 `0.8829787234042553 / 0.9363636363636364`，但在 learned-path family 内已经成为更强且更简单的参考。）
- **本轮结论**：在 `seed=42` 这个 canonical 低点上，**cross-view mixer 比 shared calibrator 更像 richer reliability path 的主要坏因子**。把 mixer 去掉以后，主指标至少没有再掉，AUC 还明显高于 matched `current learned`；而上一轮单独去掉 calibrator 却让 accuracy 直接塌到 `0.7979`。这说明 calibrator 更像是在提供必要的温和 logit 校准，而 cross-view token interaction 本身才是当前更值得怀疑的方差来源。
- **推荐动作**：在模块分析真正收口之前，不要恢复“自动优化 weighting 比例”作为默认下一步。既然 simpler learned path 在 `seed=42` 已经优于 matched `current learned`，后续若继续 canonical 主线，应先把 **`no-mixer + shared calibrator`** 当新的 learned-path 参考，优先补 **matched `seed=123/456`** 验证它是否稳定；`equal-weight` 继续保留为 accuracy 对照上界。

## 2026-04-21：ResNeXt Fusion-Path Ablation（remove shared calibrator, keep cross-view mixer，seed=42，adaptive main-study fixed trial）

> **独立 campaign 说明**
> - 这是在人类要求“先做 matched 模块贡献分析，再考虑 weighting ratio”之后，沿着上一轮 backlog 推荐动作执行的 **单模块 richer reliability path 拆解**：不再扫 temperature，也不扩容 gating path，只测试 `current learned` 路径里 **shared low-rank reliability calibrator** 本身是否在制造 `seed=42` 的主指标方差。
> - 配方继续保持严格 matched：`backbone=resnext`、`fusion_type=decision`、`image_size=512`、`num_slices_per_view=16`、`trim_edge_slices=2`、`batch_size=2`、`num_workers=4`、`epochs=20`、`freeze_layers=3`、`lr=1e-4`、`weight_decay=1e-4`、`dropout=0.3`、`gradient_clip_norm=1.0`；唯一实验变量是把 decision-fusion reliability path 从 **`cross-view mixer + shared low-rank calibrator`** 改成 **`cross-view mixer + raw reliability heads`**，并把 fusion softmax 恢复到 matched baseline `temperature=1.0`。
> - 本轮运行 commit 为 `24ac323`；fresh search-config copy 为 `autoresearch_logs/generated_search_configs/optuna_main_search_iter_0088_20260421_112327.yaml`，fresh study_root 为 `runs/optuna_main_autoloop/iter_0088_20260421_112327`。
> - 运行前按 adaptive GPU policy 检查了 `GPU 0/1/2` 的 idle 状态，三张 `Tesla V100-PCIE-32GB` 都满足 `used<=1024 MiB`、`util<=20%`；本轮仍只有 1 个 fixed trial，最终只消耗了其中 1 张卡的训练时长，其余卡保持空闲。

- [x] **CMP-FUSION-PATH-RESNEXT-NOCALIB-512X16-E20-S42-MAIN**：fresh adaptive main-study `runs/optuna_main_autoloop/iter_0088_20260421_112327` 的 best completed trial（trial `0`，`seed=42`，remove shared calibrator while keeping cross-view mixer）→ `val_acc=0.7978723404255319`, `val_auc=0.8781818181818182`, `val_f1=0.7397260273972602`, `peak_vram≈4.67 GiB`, `total_seconds≈2907.0` → **discard**（相较 matched `current learned` seed42 `0.8404255319148937 / 0.8863636363636364`，accuracy / AUC 分别回落 `0.0425531914893618 / 0.0081818181818182`；相较 matched `equal-weight` seed42 `0.8829787234042553 / 0.9363636363636364`，accuracy / AUC 分别回落 `0.0851063829787234 / 0.0581818181818182`；连先前 `25%` equal-prior probe 的 `0.8404255319148937 / 0.8781818181818182` 也没有超越。）
- **本轮结论**：在 `seed=42` 这个 canonical 低点上，**shared low-rank calibrator 并不是主要的坏因子**。把它拿掉以后，AUC 只小幅变差，但 threshold accuracy 直接塌到 `0.7979`，说明 calibrator 至少承担了把 reliability logits 拉回可用决策面的作用；真正值得继续怀疑的，更像是 `cross-view mixer` 与 reliability path 的交互方式，而不是校准器本身。
- **推荐动作**：不要把 `no-calibrator` simpler path 升格为新的 learned weighting 参考，也不要在模块结论未收口前重新回到自动化 weighting-ratio 优化。下一步若继续 matched 模块分析，应优先做 **保留 shared calibrator、去掉 cross-view mixer** 的单模块对照，确认 richer reliability path 里到底是 cross-view token interaction 本身在伤害 `seed=42`，还是必须保留两者组合。

## 2026-04-21：ResNeXt Weighting-Ratio Probe（fusion temperature 1.75，seed=42，adaptive main-study fixed trial）

> **独立 campaign 说明**
> - 这是在 `temperature=1.5` 与 `2.0` 已经对 `seed=42/123` 给出交叉胜负之后，按 backlog 推荐动作执行的 **中间点折中验证**：不新增任何 gating / weighting 容量，只检查同一个全局 fusion-temperature `1.75` 能否在 matched `seed=42` 上同时保住 `2.0` 的 accuracy 修复和 `1.5` 的部分 AUC 回升。
> - 配方继续保持严格 matched：`backbone=resnext`、`fusion_type=decision`、`image_size=512`、`num_slices_per_view=16`、`trim_edge_slices=2`、`batch_size=2`、`num_workers=4`、`epochs=20`、`freeze_layers=3`、`lr=1e-4`、`weight_decay=1e-4`、`dropout=0.3`、`gradient_clip_norm=1.0`；唯一实验变量仍是 learned weighting 路径里的 `softmax(confidences / 1.75)`，本轮只固定 `seed=42`。
> - 本轮运行 commit 为 `6b39d9b`；fresh search-config copy 为 `autoresearch_logs/generated_search_configs/optuna_main_search_iter_0018_20260421_090208.yaml`，fresh study_root 为 `runs/optuna_main_autoloop/iter_0018_20260421_090208`。
> - 运行前按 adaptive GPU policy 检查了 `GPU 0/1/2` 的 idle 状态，三张 `Tesla V100-PCIE-32GB` 都满足 `used<=1024 MiB`、`util<=20%`；本轮仍只有 1 个 fixed trial，最终只消耗了其中 1 张卡的训练时长，其余卡保持空闲。

- [x] **CMP-FUSION-RATIO-RESNEXT-TEMP1P75-512X16-E20-S42-MAIN**：fresh adaptive main-study `runs/optuna_main_autoloop/iter_0018_20260421_090208` 的 best completed trial（trial `0`，`seed=42`，current learned weighting + global fusion temperature `1.75`）→ `val_acc=0.8297872340425532`, `val_auc=0.8704545454545455`, `val_f1=0.8048780487804879`, `peak_vram≈4.67 GiB`, `total_seconds≈2908.1` → **discard**（相较 matched `current learned` seed42 `0.8404255319148937 / 0.8863636363636364`，accuracy / AUC 分别回落 `0.0106382978723405 / 0.0159090909090909`；相较 `temperature=2.0` seed42 `0.8510638297872340 / 0.8840909090909091`，accuracy / AUC 分别回落 `0.0212765957446808 / 0.0136363636363636`；与 `temperature=1.5` seed42 的 accuracy 持平，但 AUC 反而再低 `0.0381818181818182`，因此既没守住 `2.0` 的 accuracy 修复，也没保留 `1.5` 的 ranking 回升。）
- **本轮结论**：`temperature=1.75` 不是 `seed=42` 上的平滑折中点。它在 accuracy 上直接跌回与 `1.5` 相同的低位，却在 AUC 上同时输给 `1.5` 和 `2.0`。这说明 **当前 learned weighting 的单个全局 temperature 并没有给出单调、平滑、可泛化的 trade-off 曲线**；至少在 `seed=42` 上，中间点比两侧邻居都更差。
- **推荐动作**：不要继续把 `seed=42` 上的 global temperature 细扫当作默认主线。既然 `equal-weight` 仍是更强的 accuracy 参考，而 first-wave 模块分析已表明 richer reliability path 的复杂度主要来自 `cross-view mixer + shared low-rank calibrator`，下一步更合理的是回到 **一次只拆一个模块**：优先测试 **去掉 shared low-rank calibrator、保留 cross-view mixer** 的 matched control，看温和校准本身是否就是 `seed=42` 方差来源；`equal-weight` 继续作为主指标参考线，只有单模块 ablation 仍无净收益时才决定 learned weighting 是否值得保留。

## 2026-04-21：ResNeXt Weighting-Ratio Probe（fusion temperature 1.5，seed=42，adaptive main-study fixed trial）

> **独立 campaign 说明**
> - 这是在 `temperature=1.5` 于 matched `seed=123` 明确给出 keep 之后，按 backlog 推荐动作执行的第二条 **matched seed validation**：不新增任何 gating / weighting 容量，只检查同一个全局 fusion-temperature `1.5` 在另一个 canonical seed 上是否也能成立。
> - 配方继续保持严格 matched：`backbone=resnext`、`fusion_type=decision`、`image_size=512`、`num_slices_per_view=16`、`trim_edge_slices=2`、`batch_size=2`、`num_workers=4`、`epochs=20`、`freeze_layers=3`、`lr=1e-4`、`weight_decay=1e-4`、`dropout=0.3`、`gradient_clip_norm=1.0`；唯一实验变量仍是 learned weighting 路径里的 `softmax(confidences / 1.5)`，本轮只把 matched trial seed 从 `123` 切到 `42`。
> - 本轮运行 commit 为 `15ba788`；fresh search-config copy 为 `autoresearch_logs/generated_search_configs/optuna_main_search_iter_0017_20260421_080800.yaml`，fresh study_root 为 `runs/optuna_main_autoloop/iter_0017_20260421_080800`。
> - 运行前按 adaptive GPU policy 检查了 `GPU 0/1/2` 的 idle 状态，三张 `Tesla V100-PCIE-32GB` 都满足 `used<=1024 MiB`、`util<=20%`；由于这轮仍只有 1 个 fixed trial，最终只消耗了其中 1 张卡的训练时长，其余卡保持空闲。

- [x] **CMP-FUSION-RATIO-RESNEXT-TEMP1P5-512X16-E20-S42-MAIN**：fresh adaptive main-study `runs/optuna_main_autoloop/iter_0017_20260421_080800` 的 best completed trial（trial `0`，`seed=42`，current learned weighting + global fusion temperature `1.5`）→ `val_acc=0.8297872340425532`, `val_auc=0.9086363636363637`, `val_f1=0.8`, `peak_vram≈4.67 GiB`, `total_seconds≈2902.7` → **discard**（相较 matched `current learned` seed42 `0.8404255319148937 / 0.8863636363636364`，accuracy 回落 `0.0106382978723405`，虽然 AUC 反而提升 `0.0222727272727273`；相较 `temperature=2.0` seed42 `0.8510638297872340 / 0.8840909090909091`，accuracy 也继续回落 `0.0212765957446808`，只是在 AUC 上回升 `0.0245454545454546`；同时仍明显低于 matched `equal-weight` seed42 `0.8829787234042553 / 0.9363636363636364`。）
- **本轮结论**：`temperature=1.5` 并不能作为跨 seed 的稳定默认值。它在 `seed=123` 上修复了 `temperature=2.0` 的过度 softening，但在 `seed=42` 上却把 threshold accuracy 再次压回到了 baseline 以下，只留下 AUC 提升。这说明 **单个全局 temperature 的最优点已经表现出明显的 seed-sensitive trade-off**：`1.5` 更像是在放大 ranking，而不是稳定提高当前主指标。
- **推荐动作**：下一步仍保持 **单标量、低容量、可解释** 主线，不要扩容 gating / weighting path。既然 `seed=42` 更偏向 `2.0`、`seed=123` 更偏向 `1.5`，最优先的是补 **中间点 `temperature=1.75` on `seed=42`**，看能否在不丢掉 `seed42` accuracy 修复的前提下保留一部分 AUC 回升；只有当 `1.75` 在 `seed=42` 也至少站回 `0.8511` 档后，才值得扩到 `seed=123/456` 验证它是否能成为真正的折中默认值。

## 2026-04-21：ResNeXt Weighting-Ratio Probe（fusion temperature 1.5，seed=123，adaptive main-study fixed trial）

> **独立 campaign 说明**
> - 这是在 `temperature=2.0` 于 matched `seed=123` 明确退化之后，按 backlog 推荐动作执行的 **lighter logit-level shrinkage 回退验证**：不新增任何 gating / weighting 容量，只把同一个全局 fusion-temperature 从 `2.0` 回退到 `1.5`。
> - 配方继续保持严格 matched：`backbone=resnext`、`fusion_type=decision`、`image_size=512`、`num_slices_per_view=16`、`trim_edge_slices=2`、`batch_size=2`、`num_workers=4`、`epochs=20`、`freeze_layers=3`、`lr=1e-4`、`weight_decay=1e-4`、`dropout=0.3`、`gradient_clip_norm=1.0`；唯一实验变量是 learned weighting 路径里的 `softmax(confidences / 1.5)`。
> - 本轮运行 commit 为 `3d780b9`；fresh search-config copy 为 `autoresearch_logs/generated_search_configs/optuna_main_search_iter_0016_20260421_071219.yaml`，fresh study_root 为 `runs/optuna_main_autoloop/iter_0016_20260421_071219`。
> - 运行前按 adaptive GPU policy 检查了 `GPU 0/1/2`：`GPU 0` 当时被外部任务占用（`used=5516 MiB`, `util=99%`），因此显式 `--gpu-ids 0,1,2` 启动被 workflow 的 idle-threshold 保护拒绝；随后按本轮单卡 fallback 规则切到 host `GPU 1` 继续执行，同一 fresh study_root 未变。

- [x] **CMP-FUSION-RATIO-RESNEXT-TEMP1P5-512X16-E20-S123-MAIN**：fresh adaptive main-study `runs/optuna_main_autoloop/iter_0016_20260421_071219` 的 best completed trial（trial `0`，`seed=123`，current learned weighting + global fusion temperature `1.5`）→ `val_acc=0.8723404255319149`, `val_auc=0.9036363636363638`, `val_f1=0.8536585365853658`, `peak_vram≈4.67 GiB`, `total_seconds≈2912.9` → **keep**（相较 matched `current learned` seed123 `0.8510638297872340 / 0.9109090909090910`，accuracy 提升 `0.0212765957446809`，AUC 只回落 `0.0072727272727272`；相较上一轮 `temperature=2.0` seed123 `0.8404255319148937 / 0.8759090909090910`，accuracy / AUC 分别反弹 `0.0319148936170212 / 0.0277272727272728`；同时也高于 matched `equal-weight` seed123 `0.8404255319148937 / 0.8663636363636364`。）
- **本轮结论**：`temperature=1.5` 直接把 `seed=123` 从 `temperature=2.0` 的失败点拉回到了明显高于当前 learned baseline 的区间，而且只付出了很小的 AUC 回撤。这说明 **问题更像是 `2.0` 的 softening 过头，而不是 logit-level shrinkage 方向本身错误**；较轻的温度回退在不增容量的前提下，已经给出比 current learned、equal-weight、以及 `temperature=2.0` 都更强的主指标。
- **推荐动作**：下一步仍保持 **单标量、低容量、可解释** 主线，不要扩容 gating / weighting path。最优先的是补 **matched `temperature=1.5` on `seed=42`**，确认它能否在上一轮 `2.0` 受益的低点上也守住或超过 `0.8511`；只有当 `seed=42` 也站住后，才值得把 `temperature=1.5` 扩到 `seed=456` 做三 seed 收口。

## 2026-04-21：ResNeXt Weighting-Ratio Probe（fusion temperature 2.0，seed=123，adaptive main-study fixed trial）

> **独立 campaign 说明**
> - 这是沿着上一轮 `temperature=2.0` seed42 keep 继续做的第一条 **matched seed validation**：不新增任何 gating / weighting 容量，只检查同一个全局 fusion-temperature 标量在另一个 canonical seed 上是否还能成立。
> - 配方继续保持严格 matched：`backbone=resnext`、`fusion_type=decision`、`image_size=512`、`num_slices_per_view=16`、`trim_edge_slices=2`、`batch_size=2`、`num_workers=4`、`epochs=20`、`freeze_layers=3`、`lr=1e-4`、`weight_decay=1e-4`、`dropout=0.3`、`gradient_clip_norm=1.0`；唯一实验变量仍是 learned weighting 路径里的 `softmax(confidences / 2.0)`，本轮只把 matched trial seed 从 `42` 切到 `123`。
> - 本轮运行 commit 为 `73d68b9`；fresh search-config copy 为 `autoresearch_logs/generated_search_configs/optuna_main_search_iter_0005_20260421_054501.yaml`，fresh study_root 为 `runs/optuna_main_autoloop/iter_0005_20260421_054501`。
> - 运行前按 adaptive GPU policy 检查了 `GPU 0/1/2` 的 idle 状态，三张 `Tesla V100-PCIE-32GB` 都满足 `used<=1024 MiB`、`util<=20%`；由于这轮仍只有 1 个 fixed trial，最终只消耗了其中 1 张卡的训练时长，其余卡保持空闲。

- [x] **CMP-FUSION-RATIO-RESNEXT-TEMP2P0-512X16-E20-S123-MAIN**：fresh adaptive main-study `runs/optuna_main_autoloop/iter_0005_20260421_054501` 的 best completed trial（trial `0`，`seed=123`，current learned weighting + global fusion temperature `2.0`）→ `val_acc=0.8404255319148937`, `val_auc=0.875909090909091`, `val_f1=0.8148148148148148`, `peak_vram≈4.67 GiB`, `total_seconds≈2906.2` → **discard**（相较 matched `current learned` seed123 `0.8510638297872340 / 0.9109090909090910`，temperature `2.0` 的 accuracy / AUC 分别回落 `0.0106382978723403 / 0.0350000000000000`；虽然它与 matched `equal-weight` seed123 的 `0.8404255319148937 / 0.8663636363636364` 在 accuracy 上打平，并把 AUC 微幅抬高 `0.0095454545454546`，但按主指标仍不能保留。）
- **本轮结论**：`temperature=2.0` 对 seed42 的修复并没有稳定泛化到另一个 matched seed。它在 seed123 上没有把 learned weighting 拉回 `0.8511` 档，反而把 accuracy 压回了 equal-weight 水平，同时让 AUC 明显低于当前 learned baseline，说明 **这个 softening 强度已经偏大，更像是在修 seed42 低点时顺便过度收缩了 seed123 的可靠度差异**。
- **推荐动作**：不要把 `temperature=2.0` 升格为新的 learned-path 默认参考，也不要继续扩容 weighting path。既然 backlog 已经预设“其余 seed 若退化则回退更轻的 temperature”，下一步应优先测试 **更轻的 `temperature=1.5`**，先在 `seed=123` 这个失败点上看能否保住 baseline `val_acc`，再决定是否值得扩到 `seed=456`。

## 2026-04-21：ResNeXt Weighting-Ratio Probe（fusion temperature 2.0，seed=42，adaptive main-study fixed trial）

> **独立 campaign 说明**
> - 这是在 `25%` equal-prior interpolation 明确失败之后，按 backlog 推荐动作切换到的第一轮 **logit-level shrinkage** probe：不再做 post-softmax convex blend，而是在当前 canonical `current learned` decision-fusion 上只引入 **1 个全局 fusion-temperature 标量**。
> - 配方继续保持严格 matched：`backbone=resnext`、`fusion_type=decision`、`image_size=512`、`num_slices_per_view=16`、`trim_edge_slices=2`、`batch_size=2`、`num_workers=4`、`epochs=20`、`freeze_layers=3`、`lr=1e-4`、`weight_decay=1e-4`、`dropout=0.3`、`gradient_clip_norm=1.0`；唯一实验变量是在 learned weighting 路径里把 `softmax(confidences)` 改成 `softmax(confidences / 2.0)`。
> - 本轮运行 commit 为 `4b599a8`；fresh search-config copy 为 `autoresearch_logs/generated_search_configs/optuna_main_search_iter_0004_20260421_045018.yaml`，fresh study_root 为 `runs/optuna_main_autoloop/iter_0004_20260421_045018`。
> - 运行前按 adaptive GPU policy 检查了 `GPU 0/1/2` 的 idle 状态，三张 `Tesla V100-PCIE-32GB` 都满足 `used<=1024 MiB`、`util<=20%`；由于这轮仍只有 1 个 fixed trial，最终只消耗了其中 1 张卡的训练时长，其余卡保持空闲。

- [x] **CMP-FUSION-RATIO-RESNEXT-TEMP2P0-512X16-E20-S42-MAIN**：fresh adaptive main-study `runs/optuna_main_autoloop/iter_0004_20260421_045018` 的 best completed trial（trial `0`，`seed=42`，current learned weighting + global fusion temperature `2.0`）→ `val_acc=0.851063829787234`, `val_auc=0.8840909090909091`, `val_f1=0.825`, `peak_vram≈4.67 GiB`, `total_seconds≈2908.2` → **keep**（相较 matched `current learned` seed42 `0.8404255319148937 / 0.8872727272727273`，temperature `2.0` 把 accuracy 提升了 `0.0106382978723403`，AUC 只小幅回落 `0.0031818181818182`；相较上一轮 `25%` equal-prior probe `0.8404255319148937 / 0.8781818181818182`，accuracy 同样提升 `0.0106382978723403`，AUC 也反弹 `0.0059090909090909`。但它仍明显低于 matched `equal-weight` seed42 的 `0.8829787234042553 / 0.9363636363636364`。）
- **本轮结论**：`global fusion-temperature` 是当前 low-capacity weighting-ratio 线上第一个对已知 `seed=42` accuracy low point 给出正收益的单标量改动。它把 learned path 的 `val_acc` 从 `0.8404` 拉到了 `0.8511`，同时避免了 `25%` prior blend 那种更明显的 AUC 伤害，说明 **logit-level softening 比 post-softmax prior mixing 更接近这条线真正需要的收缩方向**。
- **推荐动作**：先不要继续扩容 gating / weighting path，也不要立刻跳到更复杂的 ratio 组合。下一步应把 `temperature=2.0` 当成新的 learned-path 候选参考，优先补 **matched `seed=123/456` 验证**，确认它是在系统性降低 learned weighting 方差，还是只是在 `seed=42` 这一个低点上起作用；只有如果它在其余 seed 上退化，才回退测试更轻的 `temperature=1.5`。

## 2026-04-21：ResNeXt Weighting-Ratio Probe（equal-prior interpolation 25%，seed=42，adaptive main-study fixed trial）

> **独立 campaign 说明**
> - 这是在 **first-wave 模块贡献分析完成后** 启动的第一轮低容量 weighting ratio 优化：不再扩容 gating path，也不回到 `minimal` 对照，而是在当前 canonical `current learned` decision-fusion 上只引入 **1 个标量级 equal-prior interpolation**。
> - 配方继续保持严格 matched：`backbone=resnext`、`fusion_type=decision`、`image_size=512`、`num_slices_per_view=16`、`trim_edge_slices=2`、`batch_size=2`、`num_workers=4`、`epochs=20`、`freeze_layers=3`、`lr=1e-4`、`weight_decay=1e-4`、`dropout=0.3`、`gradient_clip_norm=1.0`；唯一实验变量是在 learned weighting 路径里把 `softmax(confidences)` 改成 `0.75 * learned + 0.25 * uniform(1/3)`。
> - 本轮运行 commit 为 `ab73d77`；fresh search-config copy 为 `autoresearch_logs/generated_search_configs/optuna_main_search_iter_0003_20260421_035452.yaml`，fresh study_root 为 `runs/optuna_main_autoloop/iter_0003_20260421_035452`。
> - 运行前按 adaptive GPU policy 检查了 `GPU 0/1/2` 的 idle 状态，三张 `Tesla V100-PCIE-32GB` 都满足 `used<=1024 MiB`、`util<=20%`；由于这轮仍只有 1 个 fixed trial，最终只消耗了其中 1 张卡的训练时长，其余卡保持空闲。

- [x] **CMP-FUSION-RATIO-RESNEXT-EQPRIOR025-512X16-E20-S42-MAIN**：fresh adaptive main-study `runs/optuna_main_autoloop/iter_0003_20260421_035452` 的 best completed trial（trial `0`，`seed=42`，current learned weighting + `25%` equal-prior interpolation）→ `val_acc=0.8404255319148937`, `val_auc=0.8781818181818182`, `val_f1=0.8051948051948052`, `peak_vram≈4.67 GiB`, `total_seconds≈2902.2` → **discard**（与 matched `current learned` seed42 `0.8404255319148937 / 0.8872727272727273` 在 accuracy 上完全打平，但 AUC 反而回落 `0.0090909090909091`；同时依旧明显低于 matched `equal-weight` seed42 的 `0.8829787234042553 / 0.9363636363636364`，accuracy / AUC 分别落后 `0.0425531914893616 / 0.0581818181818182`。）
- **本轮结论**：直接做 post-softmax `25%` equal-prior shrinkage 并没有修复 seed42 这个 learned weighting 的已知 accuracy 低点，反而把 ranking quality 再往下压了一档。换句话说，**“向 equal-weight 做固定 convex blend” 本身不是当前这条线最有希望的 ratio knob**。
- **推荐动作**：下一步仍留在人类批准的 **单标量 weighting ratio** 轨道里，但不要继续增大 equal-prior blend；更合理的是测试 **global fusion-temperature** 这类更软的 logit-level shrinkage，或者把 equal-prior 改成更轻的 `10%` 级别，而不是直接往 `50%` / 更复杂 gating path 推进。

## 2026-04-21：ResNeXt Fusion-Path Ablation（minimal learned，seed=456，adaptive main-study fixed trial）

> **独立 campaign 说明**
> - 这是按当前 canonical 主线补齐的 **matched 模块贡献分析收口轮**：在已有 `equal-weight vs current learned` 三 seed 控制变量，以及 `minimal learned` 的 `seed=42/123` 结果基础上，补跑 `minimal learned` 的最后一个 matched seed `456`。
> - 为继续遵守“优先 fresh main-study”且不把超参搜索和模块对照混在一起，本轮仍把 `configs/optuna_main_search.yaml` 保持为 **单 trial fixed-config study**：`study.n_trials=1`，并通过 `fixed_overrides` 锁定 canonical recipe（`backbone=resnext`、`fusion_type=decision`、`image_size=512`、`num_slices_per_view=16`、`trim_edge_slices=2`、`batch_size=2`、`num_workers=4`、`epochs=20`、`freeze_layers=3`、`lr=1e-4`、`weight_decay=1e-4`、`dropout=0.3`、`gradient_clip_norm=1.0`），只让 trial 参数显式记录 `seed=456` 和 `model.minimal_fusion_baseline=true`。
> - 本轮运行 commit 为 `4af320c`；fresh search-config copy 为 `autoresearch_logs/generated_search_configs/optuna_main_search_iter_0002_20260421_025952.yaml`，fresh study_root 为 `runs/optuna_main_autoloop/iter_0002_20260421_025952`。
> - 运行前按 adaptive GPU policy 检查了 `GPU 0/1/2` 的 idle 状态，三张 `Tesla V100-PCIE-32GB` 都满足 `used<=1024 MiB`、`util<=20%`；由于这轮仍只有 1 个 fixed trial，最终只消耗了其中 1 张卡的训练时长，其余卡保持空闲。

- [x] **CMP-FUSION-PATH-RESNEXT-MINIMAL-512X16-E20-S456-MAIN**：fresh adaptive main-study `runs/optuna_main_autoloop/iter_0002_20260421_025952` 的 best completed trial（trial `0`，`seed=456`, `minimal_fusion_baseline=true`）→ `val_acc=0.8085106382978723`, `val_auc=0.8636363636363638`, `val_f1=0.7857142857142857`, `peak_vram≈4.64 GiB`, `total_seconds≈2903.1` → **discard**（相较 matched `current learned` seed456 `0.8510638297872340 / 0.9209090909090910`，minimal 的 accuracy / AUC 分别回落 `0.0425531914893617 / 0.0572727272727272`；同时也低于 matched `equal-weight` seed456 的 `0.8404255319148937 / 0.8890909090909092`，因此这轮不只是主指标失败，连辅助排序质量也没有守住。）
- **三 seed 模块分析结论（minimal vs current learned vs equal-weight）**：`minimal learned` 的 canonical 3-seed mean 现为 `val_acc=0.8191489361702127`、`val_auc=0.9059090909090909`；对应 `current learned` 为 `0.8475177304964540 / 0.9063636363636364`，`equal-weight` 为 `0.8546099290780141 / 0.8972727272727273`。这意味着 minimal 虽然在 `seed=42/123` 两次给出更高 AUC，但平均 AUC 也只与 current learned 基本打平，而平均 accuracy 反而低了 `0.0283687943262413`。
- **稳定性结论**：`minimal learned` 的 `val_acc` population std 为 `0.0150448251316287`，明显高于 `current learned` 的 `0.0050149417105429`；它在 `seed=123/456` 都掉到了同一个 `0.8085` accuracy floor。换句话说，first-wave 模块贡献分析已经足够说明：**“去掉 `cross-view mixer + shared low-rank calibrator` 的 minimal path” 不能作为当前 canonical learned weighting 的默认替代。**
- **推荐动作**：first-wave 模块贡献分析至此可以视为完成。下一步不要回到“再补一个 minimal seed”的循环，也不要扩容 gating path；应转入人类已批准顺序里的 **低容量 weighting ratio 优化**，并以 `current learned` 作为 learned-path 参考、`equal-weight` 作为 accuracy 对照，优先测试 **1 个全局 fusion-temperature / equal-prior interpolation** 这类不增容量的标量改动，看看能否向 equal-weight 的 mean accuracy 靠拢，同时尽量保住 current learned 的 AUC 与稳定性。

## 2026-04-21：ResNeXt Fusion-Path Ablation（minimal learned，seed=123，adaptive main-study fixed trial）

> **独立 campaign 说明**
> - 这是按当前 canonical 主线继续补的 **matched 模块贡献分析**：在已有 `equal-weight vs current learned` 三 seed 控制变量和 `minimal learned vs current learned (seed=42)` 的基础上，补跑 `minimal learned` 的 `seed=123`。
> - 为遵守“优先 fresh main-study”但又不把超参搜索和模块对照混在一起，本轮把 `configs/optuna_main_search.yaml` 收成 **单 trial fixed-config study**：`study.n_trials=1`，并通过 `fixed_overrides` 锁定 canonical recipe（`backbone=resnext`、`fusion_type=decision`、`image_size=512`、`num_slices_per_view=16`、`trim_edge_slices=2`、`batch_size=2`、`num_workers=4`、`epochs=20`、`freeze_layers=3`、`lr=1e-4`、`weight_decay=1e-4`、`dropout=0.3`、`gradient_clip_norm=1.0`），只让 trial 参数显式记录 `seed=123` 和 `model.minimal_fusion_baseline=true`。
> - 本轮运行 commit 为 `cc6ebb8`；fresh search-config copy 为 `autoresearch_logs/generated_search_configs/optuna_main_search_iter_0001_20260421_020353.yaml`，fresh study_root 为 `runs/optuna_main_autoloop/iter_0001_20260421_020353`。
> - 运行时按 adaptive GPU policy 检查了 `GPU 0/1/2` 的 idle 状态，三张 `Tesla V100-PCIE-32GB` 都满足 `used<=1024 MiB`、`util<=20%`；由于这轮只有 1 个 fixed trial，最终只在 `GPU 0` 上实际执行训练，其余两张卡保持空闲。

- [x] **CMP-FUSION-PATH-RESNEXT-MINIMAL-512X16-E20-S123-MAIN**：fresh adaptive main-study `runs/optuna_main_autoloop/iter_0001_20260421_020353` 的 best completed trial（trial `0`，`seed=123`, `minimal_fusion_baseline=true`）→ `val_acc=0.8085106382978723`, `val_auc=0.9322727272727273`, `val_f1=0.7428571428571429`, `peak_vram≈4.64 GiB`, `total_seconds≈2897.6` → **discard**（相较 matched `current learned` seed123 `0.8510638297872340 / 0.9109090909090910`，minimal 的 accuracy 回落 `0.0425531914893617`，虽然 AUC 反而提升 `0.0213636363636363`；同时也低于 matched `equal-weight` seed123 的 `0.8404255319148937 / 0.8663636363636364`，因此不能把 simpler path 当成当前默认参考。）
- **模块分析结论（到目前为止）**：`minimal learned` 现在呈现明显的 **seed-sensitive split verdict**。`seed=42` 时它能在不损失 accuracy 的前提下显著优于 `current learned` 的 AUC；但 `seed=123` 时它虽然继续给出更高 AUC，却把 threshold accuracy 拉低到了 `0.8085`。这说明“去掉 `cross-view mixer + shared low-rank calibrator`”并不是稳定单调增益，更像是在降低某些 seed 的过拟合同时，也可能伤害主指标。
- **推荐动作**：下一步应优先补 **`minimal learned` 的 `seed=456` matched run**，把 `equal-weight / current learned / minimal learned` 在 canonical `resnext + decision + 512x16 + 20 epochs` 下的三 seed 证据补齐。只有在 `seed=456` 也落地后，才适合决定 simpler path 是否应该成为主参考，还是保留“`current learned` 与 `minimal learned` 各有一部分净收益”的结论。

## 2026-04-21：7ae19a0 原配方 Equal-Weight 3-seed 对照（V100q node20）

> **独立 campaign 说明**
> - 这是按人类最新要求补做的 **1 次 matched equal-weight 3-seed control**：以上一轮 `7ae19a0` 的 `ResNeXt + learned decision fusion` 稳定性校验为基线，只替换融合权重机制为固定等加权（每个视角 `1/3`），继续跑 `seed=42/123/456`。
> - 为保持 `7ae19a0` 代码路径尽量原样，执行层仍使用 detached worktree；这次 worktree HEAD 为 `1b84e85`，它是从 `7ae19a0` 出发打的最小补丁：仅给 `MultiViewDecisionFusionClassifier` 和 `train.py` 增加 `equal_weight_fusion=true` 开关。`equal_weight_fusion=false` 时行为与上一轮 learned rerun 一致。
> - 配方保持严格 matched：`backbone=resnext`、`fusion_type=decision`、`share_backbone=false`、`use_attention_pooling=false`、`image_size=256`、`num_slices_per_view=8`、`trim_edge_slices=2`、`batch_size=6`、`num_workers=12`、`epochs=15`、`freeze_layers=3`、`lr=1e-4`、`weight_decay=5e-4`、`dropout=0.25`、`gradient_clip_norm=2.0`；唯一实验变量是 `model.equal_weight_fusion=true`。
> - Slurm batch 为 `428936`：`V100q` 的 `node20`，`1 node / 3 GPU / 36 CPU / 120G / 3h`；job 内 `CUDA_VISIBLE_DEVICES=0,1,2`，PyTorch 实测可见 `3` 张 `Tesla V100-PCIE-32GB`；三个 `srun` step `428936.0/.1/.2` 全部 `COMPLETED`。

- [x] **CMP-7AE19A0-RESNEXT-EQUAL-256X8-S42**：在 detached `1b84e85` worktree 中对原 formal 配方仅替换 `equal_weight_fusion=true`、`seed=42` / 独立 `output_dir` 运行 → `val_acc=0.9361702127659575`, `val_auc=0.9459090909090909`, `val_f1=0.9302325581395349`, `peak_vram≈2.14 GiB`, `total_seconds≈907.1` → **keep**（相较上一轮 learned-weighting 同 seed `0.9148936170212766 / 0.9522727272727273`，accuracy 提升 `0.0212765957446809`，AUC 小回落 `0.0063636363636364`；主指标仍明显占优。）
- [x] **CMP-7AE19A0-RESNEXT-EQUAL-256X8-S123**：在 detached `1b84e85` worktree 中对原 formal 配方仅替换 `equal_weight_fusion=true`、`seed=123` / 独立 `output_dir` 运行 → `val_acc=0.9468085106382979`, `val_auc=0.9759090909090908`, `val_f1=0.9397590361445783`, `peak_vram≈2.14 GiB`, `total_seconds≈899.6` → **keep**（相较 learned-weighting 同 seed `0.8617021276595744 / 0.9004545454545455`，accuracy / AUC 分别提升 `0.0851063829787235 / 0.0754545454545453`；直接把上一轮最大低点翻成了全组最高点。）
- [x] **CMP-7AE19A0-RESNEXT-EQUAL-256X8-S456**：在 detached `1b84e85` worktree 中对原 formal 配方仅替换 `equal_weight_fusion=true`、`seed=456` / 独立 `output_dir` 运行 → `val_acc=0.9255319148936170`, `val_auc=0.9750000000000001`, `val_f1=0.9230769230769231`, `peak_vram≈2.14 GiB`, `total_seconds≈901.1` → **keep**（与 learned-weighting 同 seed 的 accuracy `0.9255319148936170` 打平，但 AUC 再升 `0.0090909090909092`。）
- **3-seed mean 对比**：equal-weight 的 mean `val_acc=0.9361702127659575`、mean `val_auc=0.9656060606060606`、mean `val_f1=0.9310228391203455`；learned weighting 的 mean 分别是 `0.9007092198581560 / 0.9395454545454547 / 0.8912655971479500`。equal-weight 相对 learned 的 mean 增益是 `+0.0354609929078015 val_acc`、`+0.0260606060606059 val_auc`、`+0.0397572419723954 val_f1`。
- **稳定性对比**：equal-weight 的 `val_acc` population std 只有 `0.0086861338396570`，明显低于 learned weighting 的 `0.0279220137376310`；equal-weight 三个 seed 的范围是 `0.9255319148936170 -> 0.9468085106382979`，再也没有 learned 版那种 `0.8617` 级别的低点。
- **legacy reference 结论**：对 `7ae19a0` 这条 `256x8` ResNeXt 线，问题看起来不在 backbone / geometry，而更像是 **learned view weighting 本身引入了额外方差并伤害了主指标**。固定 equal-weight 后，不仅 3-seed mean 明显优于上一轮 learned rerun，连 mean `val_acc=0.9361702127659575` / mean `val_auc=0.9656060606060606` 也已经超过了原始单次 anchor `0.9255319148936170 / 0.9640909090909091`。
- **推荐动作**：如果后续还要把 `7ae19a0` 当 legacy 对照，默认应引用这次 **equal-weight 3-seed mean `0.936170 / 0.965606`**，而不是上一轮 learned-weighting 的 `0.900709 / 0.939545`，更不是单次 anchor `0.925532 / 0.964091`。如果要继续把这条 recipe 往 `512x16` 放大，也应优先从 equal-weight 版本出发，再判断 learned weighting 是否值得加回去。

## 2026-04-21：7ae19a0 原配方 3-seed 稳定性校验（V100q node20）

> **独立 campaign 说明**
> - 这是按人类明确要求补做的 **1 次 3-seed stability check**：申请 `3` 张 GPU，对 commit `7ae19a0` 的原始 `ResNeXt + decision fusion` formal 配方做 `seed=42/123/456` 并行复验。
> - 原始 recipe 保持不变：`backbone=resnext`、`fusion_type=decision`、`share_backbone=false`、`use_attention_pooling=false`、`image_size=256`、`num_slices_per_view=8`、`trim_edge_slices=2`、`batch_size=6`、`num_workers=12`、`epochs=15`、`freeze_layers=3`、`lr=1e-4`、`weight_decay=5e-4`、`dropout=0.25`、`gradient_clip_norm=2.0`。
> - 执行层走 detached worktree `7ae19a0` + Slurm batch `428934`：`V100q` 的 `node20`，`1 node / 3 GPU / 36 CPU / 120G / 3h`；job 内 `CUDA_VISIBLE_DEVICES=0,1,2`，PyTorch 实测可见 `3` 张 `Tesla V100-PCIE-32GB`。
> - 首次 batch `428932` 在 epoch 1 因 detached worktree 不含 untracked NIfTI 数据树而触发 `FileNotFoundError`；rerun `428934` 仅把临时 seed config 的 `csv_path/base_dir` 指回 canonical data root，模型与训练超参未变，因此有效结论只取 `428934`。

- [x] **CMP-7AE19A0-RESNEXT-DECISION-256X8-S42**：在 detached `7ae19a0` worktree 中对原 formal 配方仅替换 `seed=42` / 独立 `output_dir` 运行 → `val_acc=0.9148936170212766`, `val_auc=0.9522727272727272`, `val_f1=0.9090909090909091`, `peak_vram≈2.17 GiB`, `total_seconds≈891.4` → **discard**（相较原 anchor `0.925531914893617 / 0.9640909090909091`，accuracy / AUC 分别回落 `0.0106382978723404 / 0.0118181818181819`。）
- [x] **CMP-7AE19A0-RESNEXT-DECISION-256X8-S123**：在 detached `7ae19a0` worktree 中对原 formal 配方仅替换 `seed=123` / 独立 `output_dir` 运行 → `val_acc=0.8617021276595744`, `val_auc=0.9004545454545455`, `val_f1=0.8470588235294118`, `peak_vram≈2.17 GiB`, `total_seconds≈902.4` → **discard**（相较原 anchor `0.925531914893617 / 0.9640909090909091`，accuracy / AUC 分别回落 `0.0638297872340426 / 0.0636363636363636`。）
- [x] **CMP-7AE19A0-RESNEXT-DECISION-256X8-S456**：在 detached `7ae19a0` worktree 中对原 formal 配方仅替换 `seed=456` / 独立 `output_dir` 运行 → `val_acc=0.9255319148936170`, `val_auc=0.9659090909090909`, `val_f1=0.9176470588235294`, `peak_vram≈2.17 GiB`, `total_seconds≈903.1` → **discard**（accuracy 与原 anchor 持平，AUC 反而微升 `0.0018181818181818`，说明高点不是完全不可复现。）
- **3-seed mean**：`mean val_acc=0.9007092198581560`, `mean val_auc=0.9395454545454547`, `mean val_f1=0.8912655971479500`；`val_acc` 的 population std 为 `0.0279220137376310`，范围为 `0.8617021276595744 -> 0.9255319148936170`。
- **稳定性结论**：`7ae19a0` 这条 `256x8` legacy decision-fusion 配方并非完全不可复现，因为 `seed=42/456` 都能达到 `0.9149+`，其中 `seed=456` 甚至把 AUC 微幅抬到 `0.9659`。但它也明显**不是稳定的 `0.9255` 档位**：`seed=123` 会跌到 `0.8617 / 0.9005`，导致 3-seed mean 相较原单次 anchor 仍低 `0.0248226950354610 / 0.0245454545454544`。更准确的解读是：这是一个 **high-ceiling but high-variance** 的 legacy 点，而不是可以直接当 deterministic baseline 的稳定 recipe。
- **推荐动作**：如果后续还要引用 `7ae19a0` 作为 legacy reference，优先使用这次 3-seed mean `0.900709 / 0.939545`，不要再把单次 `0.925532 / 0.964091` 直接当作“稳定基准”；若继续 `512x16` 线，也仍不应直接 transplant 这套旧标量。

---

## 2026-04-21：ResNeXt Decision Direct Geometry Scale-Up（7ae19a0 anchor → 512x16，RTXA6Kq node11）

> **独立 campaign 说明**
> - 这是按人类明确要求补做的 **1 次 direct formal confirmation**：把 `7ae19a0` 的单次峰值 recipe 从 `256x8` 直接放大到 `512x16`，除几何外不改其它标量。
> - 保持 `backbone=resnext`、`fusion_type=decision`、`share_backbone=false`、`use_attention_pooling=false`、`batch_size=6`、`num_workers=12`、`epochs=15`、`freeze_layers=3`、`lr=1e-4`、`weight_decay=5e-4`、`dropout=0.25`、`gradient_clip_norm=2.0` 不变。
> - 唯一几何改动是：`image_size 256 -> 512`、`num_slices_per_view 8 -> 16`，并保留 `trim_edge_slices=2`。
> - 执行层走 Slurm 单卡 formal：`RTXA6Kq` 作业 `428923`，`node11`，`1 node / 1 GPU / 12 CPU / 64G / 4h`；job 内 `CUDA_VISIBLE_DEVICES=2`，PyTorch 实测可见 `1` 张 `NVIDIA RTX A6000`。

- [x] **CMP-RESNEXT-DECISION-CONFIRM-7AE19A0-512X16-B6-NW12**：`configs/cmp_resnext_decision_confirm_7ae19a0_512x16_b6_nw12.yaml` → `val_acc=0.8404255319148937`, `val_auc=0.9136363636363636`, `val_f1=0.8148148148148148`, `peak_vram≈12.67 GiB`, `total_seconds≈1609.9` → **discard**（这次 direct geometry transplant 虽然在 `batch_size=6` 下稳定跑通，但相较 `7ae19a0` 的 `256x8` anchor `0.925531914893617 / 0.9640909090909091`，accuracy / AUC 分别回落 `0.0851063829787233 / 0.0504545454545455`，说明原 256x8 峰值标量不能直接原样放大到 `512x16`。）
- **本轮结论**：`resnext + decision fusion` 这套 `7ae19a0` 标量配方在 `512x16` 下没有复现原本的强度；即使显存足够、吞吐正常，最终 best 也只到 `0.8404 / 0.9136`。这更像是一个**几何迁移失败**信号，而不是执行层问题。
- **推荐动作**：如果后续还要继续这条 `512x16 + full decision-fusion` 线，不要再直接沿用 `256x8` 的固定标量；至少要围绕当前 `batch_size=6` 可运行前提重新搜 `lr / wd / dropout / clip`，或者只把它当成与 `minimal learned` / `attention pooling` 方案做 matched 对照的 reference baseline。

---

## 2026-04-20：ResNeXt AttentionPooling Equal-vs-Learned Control（seeds=42/123/456，node20 V100q）

> **独立 campaign 说明**
> - 这是按人类最新要求补做的 `resnext + attention pooling` 下的 matched 3-seed fusion control：
>   直接比较 learned decision fusion 与 fixed equal-weight。
> - 所有配置保持 `backbone=resnext`、`fusion_type=decision`、`use_attention_pooling=true`、`image_size=512`、`num_slices_per_view=16`、`trim_edge_slices=2`、`batch_size=2`、`num_workers=3`、`freeze_layers=3`、`dropout=0.3`、`epochs=20`、`lr=1e-4`、`weight_decay=1e-4` 不变。
> - 执行层复用了 `V100q` 的 `node20` allocation `428450`：`1 node / 3 GPU / 15 CPU / 96G / 24h`，节点上无其他用户作业。

- [x] **CMP-DECISION-EQUAL-RESNEXT-ATTNPOOL-LEARNED-512X16-E20-S42**：`configs/cmp_decision_equal_resnext_attnpool_learned_512x16_e20_s42.yaml` → `val_acc=0.8723404255319149`, `val_auc=0.8586363636363638`, `val_f1=0.8571428571428571`, `peak_vram≈4.67 GiB`, `total_seconds≈3523.4` → **keep**（在 matched 对照里胜出。）
- [x] **CMP-DECISION-EQUAL-RESNEXT-ATTNPOOL-EQUAL-512X16-E20-S42**：`configs/cmp_decision_equal_resnext_attnpool_equal_512x16_e20_s42.yaml` → `val_acc=0.8085106382978723`, `val_auc=0.9027272727272728`, `val_f1=0.7631578947368421`, `peak_vram≈4.65 GiB`, `total_seconds≈3523.6` → **discard**（在 matched 对照里落后于另一种 weighting。）
- [x] **CMP-DECISION-EQUAL-RESNEXT-ATTNPOOL-LEARNED-512X16-E20-S123**：`configs/cmp_decision_equal_resnext_attnpool_learned_512x16_e20_s123.yaml` → `val_acc=0.8510638297872340`, `val_auc=0.8950000000000001`, `val_f1=0.8157894736842105`, `peak_vram≈4.67 GiB`, `total_seconds≈3546.0` → **discard**（在 matched 对照里落后于另一种 weighting。）
- [x] **CMP-DECISION-EQUAL-RESNEXT-ATTNPOOL-EQUAL-512X16-E20-S123**：`configs/cmp_decision_equal_resnext_attnpool_equal_512x16_e20_s123.yaml` → `val_acc=0.9042553191489362`, `val_auc=0.9204545454545454`, `val_f1=0.8965517241379310`, `peak_vram≈4.65 GiB`, `total_seconds≈3526.5` → **keep**（在 matched 对照里胜出。）
- [x] **CMP-DECISION-EQUAL-RESNEXT-ATTNPOOL-LEARNED-512X16-E20-S456**：`configs/cmp_decision_equal_resnext_attnpool_learned_512x16_e20_s456.yaml` → `val_acc=0.8404255319148937`, `val_auc=0.9090909090909091`, `val_f1=0.8275862068965517`, `peak_vram≈4.67 GiB`, `total_seconds≈3532.1` → **keep**（在 matched 对照里胜出。）
- [x] **CMP-DECISION-EQUAL-RESNEXT-ATTNPOOL-EQUAL-512X16-E20-S456**：`configs/cmp_decision_equal_resnext_attnpool_equal_512x16_e20_s456.yaml` → `val_acc=0.8297872340425532`, `val_auc=0.8677272727272728`, `val_f1=0.8048780487804879`, `peak_vram≈4.65 GiB`, `total_seconds≈3526.7` → **discard**（在 matched 对照里落后于另一种 weighting。）
- **3-seed mean 对比**：learned 的 mean `val_acc=0.8546099290780141`，equal-weight 的 mean `val_acc=0.8475177304964538`；learned 的 mean `val_auc=0.8875757575757577`，equal-weight 的 mean `val_auc=0.8969696969696970`。
- **控制变量结论**：在 `resnext + attention pooling` 的 3-seed matched 对照里，**learned weighting** 的 mean `val_acc=0.8546099290780141`，高于 equal-weight 的 `0.8475177304964538`，净提升 `0.0070921985815603`。 同时 learned 的 mean `val_auc=0.8875757575757577`，低于 equal-weight 的 `0.8969696969696970`。

---



## 2026-04-20：人类方向改动（主线改成探索 ResNeXt 融合策略）

> **最高优先级说明**
> - 人类明确要求：**不能把 fixed equal-weight / 均分方法当作最终结果**。因此上一轮 `equal-weight` 在 3-seed mean `val_acc` 上占优，只能视为一个重要的控制变量证据，**不能直接升格为最终主线方法**。
> - 从现在起，autoresearch 主线不再围绕“learned vs equal 到底谁做最终 default”做终局收束，而是改成：**固定 `resnext` backbone，继续系统性探索 fusion strategy 本身**。
> - 研究优先级改成：
>   1. 先在现有 `resnext + decision fusion` family 内做 fusion-path ablation，拆清楚 current learned VRG 路径里的 `plain per-view classifier`、`raw reliability head`、`cross-view mixer`、`shared low-rank calibrator` 各自是否有净收益
>   2. 只有当 decision family 内部的最优 fusion path 收敛后，再决定是否扩展到更大语义差异的其它 fusion family
> - 下一轮 immediate task：优先比较 **current learned decision fusion** 对 **minimal learned decision baseline**，保持 `512x16 / 20 epochs / freeze=3 / resnext / seed` 等 recipe 完全 matched，只改变 fusion path 复杂度。

---

## 2026-04-20：ResNeXt Fusion-Path Ablation（current learned vs minimal learned，seed=42，node20 V100q）

> **独立 campaign 说明**
> - 这是在人类明确要求“主线改成探索 `resnext` 融合策略、不要把均分方法当最终结果”之后启动的第一轮主线 formal。
> - 研究问题不是再比 `equal-weight`，而是拆当前 `resnext + decision fusion` 内部的 fusion path：**current learned VRG path** 相比 **minimal learned decision baseline** 到底有没有净收益。
> - 两个配置都保持 `backbone=resnext`、`fusion_type=decision`、`image_size=512`、`num_slices_per_view=16`、`trim_edge_slices=2`、`batch_size=2`、`num_workers=4`、`freeze_layers=3`、`dropout=0.3`、`epochs=20`、`lr=1e-4`、`weight_decay=1e-4` 不变。
> - 唯一变量是 fusion path：
>   - `current learned`: 现有 `cross-view mixer + shared low-rank reliability calibrator`
>   - `minimal learned`: 去掉上述 richer path，只保留 plain per-view classifier + raw reliability heads
> - 正式运行是 `V100q` 的 `node20` batch `427812`；申请 `1 node / 3 GPU / 12 CPU / 96G / 5h`，其中 2 张卡并行跑 learned 与 minimal。`427812.0/.1` 为两条训练 step，均 `COMPLETED`；`427812.3` 的 shell step 在 batch 收尾阶段 `CANCELLED 0:15`，不影响实验有效性。

- [x] **CMP-FUSION-PATH-RESNEXT-LEARNED-512X16-E20-S42**：`configs/cmp_fusion_path_resnext_learned_512x16_e20_s42.yaml` → `val_acc=0.8404255319148937`, `val_auc=0.8863636363636364`, `val_f1=0.8235294117647058`, `peak_vram≈4.67 GiB`, `total_seconds≈2949.6` → **discard**（与 minimal 版在 `val_acc` 上完全打平，但 `val_auc` 低 `0.0354545454545454`。）
- [x] **CMP-FUSION-PATH-RESNEXT-MINIMAL-512X16-E20-S42**：`configs/cmp_fusion_path_resnext_minimal_512x16_e20_s42.yaml` → `val_acc=0.8404255319148937`, `val_auc=0.9218181818181818`, `val_f1=0.8192771084337349`, `peak_vram≈4.64 GiB`, `total_seconds≈2919.5` → **keep**（在不损失 accuracy 的前提下，明显提升了 ranking quality。）
- **控制变量结论**：在这轮严格 matched 的 `seed=42` 对照里，**current learned VRG path 没有比 minimal learned baseline 更好**。更具体地说：它既没有带来 `val_acc` 提升，也没有保住 AUC，反而把 `val_auc` 从 `0.9218181818181818` 拉低到 `0.8863636363636364`。
- **结构解释**：当前证据指向一个很具体的怀疑：对 `resnext decision` 而言，fusion path 里的 `cross-view mixer + shared low-rank calibrator` 这层 richer reliability modeling 可能过度复杂，带来了额外方差，但没有换回更好的主指标。
- **主线动作建议**：下一步不要回到 equal-weight 叙事，也不要立刻扩成更大语义差异的 family compare。最合理的动作是先把 **minimal learned vs current learned** 扩到 `seed=123/456`，确认这次“简化 fusion path 反而更稳”的信号是不是可复现。

---

## Agent 状态

> Agent 每次实验后必须更新此表。新 Agent 启动时以此表为起点。

| 字段 | 值 |
|------|-----|
| 上次实验 | `L0-equal` vs `L1-learned` 的 `256x8` formal matched controls（commit `8e0bdf4`；Slurm batch `431122`；`V100q/node20`；用 `scripts/slurm_resnext_decision_256x8_matrix.sbatch` 在同一节点并行跑完 `seed=42/123/456` 六个 current-scaffold formal 配置，run dirs 位于 `runs/resnext_decision_256x8_matrix/formal/l0_equal/*` 与 `runs/resnext_decision_256x8_matrix/formal/l1_learned/*`。） |
| 上次结果 | `L0-equal` keep ×3 / `L1-learned` discard ×3（`L0` 3-seed mean `0.921986 / 0.960606 / 0.911222`，`L1` 3-seed mean `0.879433 / 0.942576 / 0.874310`；equal 相对 learned 的 mean 增益为 `+0.042553 val_acc`、`+0.018030 val_auc`，且 `val_acc` std 从 `0.027922` 降到 `0.005015`。） |
| 下一步 | 继续同一 `256x8` formal matrix，直接跑 **`L2-minimal` 的 matched `seed=42/123/456`**。目标不是证明 minimal 会赢，而是先确认“去掉 richer reliability path 后”是否能比当前 `L1-learned` 更接近 `L0-equal`，从而决定后续还有没有必要继续拆 `L3/L4/L5`。 |
| 默认执行策略 | 24GB+ 显存机器默认直接跑 `main` / `formal`；`proxy` 仅保留作低显存 fallback 与快速 smoke。 |
| 连续 discard 计数 | 3（`L0-equal` 三次 keep 已经把上一轮 streak 清零；随后 `L1-learned` 的 `seed=42/123/456` 三个 matched anchor 连续记 discard，因此当前 discard streak 为 3。） |
| 累计 proxy keep 数 | 7（当前 `val_acc` 主线新增 1 次 keep：`a60c3e0` / fresh proxy winner） |
| 本地迁移补记 | 2026-04-12 从旧工作副本并入的 legacy 状态：上次实验为 `VR-16`（`9293915`; `no_miss_val_acc=0.872`, `no_miss_val_spe=0.760`, `val_AUC=0.969`）；旧计划下一步为 `VR-MS-01~03`；旧连续 discard 计数为 15。该状态属于旧 `no_miss` / `192x16` campaign，已归档为 legacy，不覆盖当前 canonical `val_acc` 主线。 |

---

## 2026-04-20：Paper Reproduction Workflow Correction（closer-to-paper retool）

> **工程纠偏说明**
> - 这不是新的正式对照轮次，也没有向 `results.tsv` 追加“胜负记录”；这次工作的目标是把 `paper_repro/` 从“粗 proxy benchmark”往**更接近原文方法学假设**的方向修正。
> - 触发原因是人工审查后确认：上一版 `paper_repro` 里，`C3 / R1 / R4 / D4` 至少有一条关键原文假设没有保住，因此原先排行榜更适合解读成 paper-inspired proxy ranking，而不适合直接叫“论文复现强弱表”。

- [x] **PAPER-REPRO-C3-OFFLINE-FUSION-RETOOL**：`paper_repro/train.py` + `paper_repro/models.py` + `paper_repro/configs/c3_decision_fusion.yaml` 现已改成 **单专家独立训练 + 验证集启发式离线融合**。不再使用上一版那种“3 个专家分支一起端到端 joint train、再手写固定权重”的实现，方向上更接近 `MMIDFNet` 原文的“先专家、后决策融合”。
- [x] **PAPER-REPRO-R1R4-WEAK-DENSE-PATCH-RETOOL**：`paper_repro/models.py` + `paper_repro/configs/r1_fracnet_weak.yaml` + `paper_repro/configs/r4_dense_vote.yaml` 现已改成 **骨区候选 patch -> patch 内 dense anomaly map -> 病例级 top-k / majority 聚合**。这仍然不是有 voxel label 的真分割复现，但比上一版“整卷直接病例级 CE”更接近 `FracNet / nnU-Net dense prediction -> global label` 的原始建模逻辑。
- [x] **PAPER-REPRO-D4-STAGED-25D3D-RETOOL**：`paper_repro/models.py` + `paper_repro/train.py` + `paper_repro/configs/d4_hybrid_25d_3d.yaml` 现已改成 **multi-slice 2.5D ViT + 3D branch + staged fine-tuning**，并补上 `mixup` 和 `head_only -> partial_25d -> full` 的阶段式训练控制。由于仓库当前没有 domain 标签，所以 `VREx` 仍无法逐字复现，但 2.5D 分支至少不再退化成“只看单张中心切片”。
- [x] **PAPER-REPRO-SMOKE-VALIDATION**：在临时 6 例子集（`train/val/test` 各 `label=0/1` 各 1 例）上做了 smoke：
  - `C3` 的离线三专家 workflow 跑通，三个 branch 和 root fusion `summary.json` 都能正常落盘。
  - `D4` 的 staged 训练链路跑通，且额外检查了 `head_only / partial_25d / full` 三种 stage 的参数解冻切换。
  - `R1 / R4` 的 weak dense patch aggregation 都能完成前向、反向和 `summary.json` 导出。
- **当前状态**：`paper_repro` 的实现语义已经明显比上一版更接近原文，但**全量 12 篇的排行榜已过期**，因为至少 `C3 / R1 / R4 / D4` 的定义发生了实质变化。若后续还要引用 paper lane 的排名，必须整套重跑，不能继续沿用 `paper_repro_all_20260420_111457.tsv`。
- **建议动作**：如果继续 paper lane，下一步应是按新 workflow 重新执行受影响的轮次，最少要补跑：
  - `round1` 的 `C3`
  - `round3` 的 `R1 / R4 / D4`
  - 如果要维持统一比较口径，最好把 `12/12` 全部重新导出一版新的总表，并把旧导出明确标成 legacy proxy ranking。

---

## 2026-04-20：Paper Reproduction Correction Completion（corrected leaderboard closure）

> **收口说明**
> - 受 closer-to-paper retool 影响的最后一个缺口 `C3` 已补跑完成，因此 paper reproduction 这条线现在不是“实现已纠偏但总榜过期”，而是**纠偏后总榜已重新收口**。
> - 旧总表 [paper_repro_all_20260420_111457.tsv](/dataset/HH/ankle-ct/paper_repro/exports/paper_repro_all_20260420_111457.tsv) 仍保留，但现在只能视为 **legacy proxy ranking**；当前应引用的新总表是 [paper_repro_all_20260420_170538.tsv](/dataset/HH/ankle-ct/paper_repro/exports/paper_repro_all_20260420_170538.tsv)。

- [x] **PAPER-REPRO-C3-OFFLINE-FUSION-RERUN**：`paper_repro/configs/c3_decision_fusion.yaml` 在 `node20` / `V100q` 的作业 `427995` 上完成正式补跑，根 [summary.json](/dataset/HH/ankle-ct/runs/paper_repro/c3_decision_fusion/summary.json) 已落盘：`val_acc=0.7553191489361702`, `val_auc=0.8490909090909090`, `val_f1=0.6849315068493150`, `peak_vram≈0.5 GiB` → **keep（corrected round1 winner）**。离线融合权重为 `axial=0.3361 / coronal=0.3773 / sagittal=0.2866`，说明 `coronal` 是主导专家，但 `axial` 仍有实质贡献。
- [x] **PAPER-REPRO-CORRECTED-ROUND1-EXPORT**：[paper_repro_round1_20260420_170538.tsv](/dataset/HH/ankle-ct/paper_repro/exports/paper_repro_round1_20260420_170538.tsv) 和 [paper_repro_round1_20260420_170538.md](/dataset/HH/ankle-ct/paper_repro/exports/paper_repro_round1_20260420_170538.md) 已重导。corrected `round1` 排名仍是 `C3 > D2 > C2`，但 `C3` 的语义已经从旧版 joint proxy 改成单专家独立训练后的离线融合。
- [x] **PAPER-REPRO-CORRECTED-ALL-EXPORT**：[paper_repro_all_20260420_170538.tsv](/dataset/HH/ankle-ct/paper_repro/exports/paper_repro_all_20260420_170538.tsv) 和 [paper_repro_all_20260420_170538.md](/dataset/HH/ankle-ct/paper_repro/exports/paper_repro_all_20260420_170538.md) 已重导。按 corrected 总榜排序，当前第一梯队为：
  - `D4`：`val_acc=0.8510638297872340`, `val_auc=0.9309090909090909`
  - `S1`：`val_acc=0.8085106382978723`, `val_auc=0.9059090909090910`
  - `C3`：`val_acc=0.7553191489361702`, `val_auc=0.8490909090909090`
  - `D1`：`val_acc=0.7553191489361702`, `val_auc=0.8322727272727273`
- **纠偏后结论**：真正把 closer-to-paper 语义补齐后，paper lane 的总冠军不再是旧 proxy 总表里的 `S1`，而是 **`D4` 的 2.5D + 3D staged ensemble**。`S1` 仍然非常强，但在 corrected leaderboard 中退到第二；`C3` 依旧保持第一梯队，只是从“旧版总榜第二”变成了 “corrected 总榜第三、且在与 `D1` 的 accuracy tie-break 中凭更高 AUC 胜出”。
- **当前状态**：paper reproduction corrected lane 现已 **12/12 全部完成且重新导出**，没有剩余待补跑项。

---

## 2026-04-20：Paper Reproduction Side Campaign（round3 / 4）

> **独立 side campaign 说明**
> - 这是 paper reproduction 的第三轮三并行论文结构复现，对象为 `R1`, `R4`, `D4`。
> - round3 继续在 compute node `node03` 的 Slurm job `426869` 内运行；driver 初始仍正确解析：
>   - `requested_gpu_ids=['0','1','2']`
>   - `resolved_gpu_ids=['5','6','7']`
> - 其中 `R4` 的第一次启动在训练入口很早期就 crash：`DenseVoteUNetClassifier(aggregation='majority')` 使用 `(positive_map > 0.5).float()` 作为 majority vote，导致计算图在分类分数处断掉，`loss.backward()` 报 `element 0 of tensors does not require grad`。该问题已用提交 `c1ded40` 修正为可导的 soft-majority proxy，并在同一 Slurm job 内补跑 `R4`，最终 round3 三项都拿到有效 `summary.json`。

- [x] **PAPER-REPRO-R3-R1-FRACNET-WEAK**：`paper_repro/configs/r1_fracnet_weak.yaml` → `val_acc=0.7340425531914894`, `val_auc=0.8709090909090910`, `val_f1=0.6031746031746031`, `peak_vram≈1.6 GiB`, `total_seconds≈512.5` → **discard（本轮内部对照）**。top-k dense anomaly aggregation 的 ranking 质量很好，AUC 是本轮最高，但 fixed-threshold accuracy 仍低于 `D4`。
- [x] **PAPER-REPRO-R3-R4-DENSE-VOTE**：`paper_repro/configs/r4_dense_vote.yaml` → 首次运行 crash（不可导的 hard majority vote）；修复后补跑得到 `val_acc=0.6702127659574468`, `val_auc=0.7386363636363636`, `val_f1=0.5974025974025974`, `peak_vram≈1.6 GiB`, `total_seconds≈661.9` → **discard（本轮内部对照）**。纯 dense vote / majority aggregation 在当前任务上过于粗糙，既没有拿到高 accuracy，也没有保住 R1 的 AUC 优势。
- [x] **PAPER-REPRO-R3-D4-HYBRID-25D-3D**：`paper_repro/configs/d4_hybrid_25d_3d.yaml` → `val_acc=0.7659574468085106`, `val_auc=0.7868181818181819`, `val_f1=0.6666666666666666`, `peak_vram≈2.6 GiB`, `total_seconds≈490.7` → **keep（本轮 side-campaign winner）**。2.5D + 3D ensemble 直接把 accuracy 拉到与 `C3` 并列的全 campaign 最高点，说明混合尺度聚合路线有真实价值。
- **本轮结论**：round3 排序明确为 `D4 > R1 > R4`。`D4` 在 accuracy 上追平了 round1 winner `C3`，但 AUC 低 `0.0445454545454545`；`R1` 则给出了比 `D4` 更高的 AUC，却没能把 threshold accuracy 拉上去。因此当前 paper reproduction 的 top tier 变成 **`C3` 与 `D4`**，而 `R1` 更像一个“排序强、阈值弱”的 dense baseline。
- **工程结论**：这轮首次暴露出“论文语义正确但训练图不可导”的复现问题。对 segmentation-style vote aggregation 来说，训练态不能直接用硬阈值 majority；后续凡是类似 branch，都应优先用 soft proxy，再把 hard vote 仅保留给推理或解释阶段。
- **推荐动作**：继续执行 **round4 = `C1`, `S1`, `S3`**。如果 round4 也没有超过 `C3` 的 AUC tie-break，则 paper lane 可以收束为：`C3` 是当前最稳的总冠军，`D4` 是 accuracy 并列冠军但 ranking 质量偏弱，`D1/R1` 作为各自子范式下的次优参考。

---

## 2026-04-20：Paper Reproduction Side Campaign（round4 / 4）

> **独立 side campaign 说明**
> - 这是 paper reproduction 的最后一轮三并行论文结构复现，对象为 `C1`, `S1`, `S3`。
> - round4 继续在 compute node `node03` 的 Slurm job `426869` 内运行；driver 日志再次确认：
>   - `requested_gpu_ids=['0','1','2']`
>   - `resolved_gpu_ids=['5','6','7']`
>   - 三个 config 全部 `exit=0`
> - 这一轮没有再出现 round3 那样的训练态 bug，三条都一次完成。

- [x] **PAPER-REPRO-R4-C1-XFMAMBA-LITE**：`paper_repro/configs/c1_xfmamba_lite.yaml` → `val_acc=0.7021276595744681`, `val_auc=0.8454545454545455`, `val_f1=0.5483870967741935`, `peak_vram≈1.9 GiB` → **discard（本轮内部对照）**。轻量 cross-view fusion baseline 有一定表达力，但没有追上当前第一梯队。
- [x] **PAPER-REPRO-R4-S1-3D-EFFICIENT**：`paper_repro/configs/s1_3d_efficient.yaml` → `val_acc=0.8085106382978723`, `val_auc=0.9059090909090910`, `val_f1=0.7631578947368421`, `peak_vram≈2.2 GiB` → **keep（本轮 winner，同时是全 paper campaign 总冠军）**。这条 3D EfficientNet-like tri-view classifier 不只是赢了 round4，而是把整个 paper reproduction lane 的 accuracy 和 AUC 都抬到了新高。
- [x] **PAPER-REPRO-R4-S3-ANATOMY-PROTOTYPE**：`paper_repro/configs/s3_anatomy_prototype.yaml` → `val_acc=0.6808510638297872`, `val_auc=0.7793181818181818`, `val_f1=0.6808510638297872`, `peak_vram≈0.4 GiB` → **discard（本轮内部对照）**。prototype head 很轻，但当前二分类任务上判别力不够。
- **本轮结论**：round4 排序明确为 `S1 > C1 > S3`。`S1` 的 `val_acc` 比此前总榜第一 `C3` 高 `0.0425531914893617`，`val_auc` 也高 `0.0745454545454546`，因此不存在 tie-break 歧义，直接成为新的总冠军。
- **全 campaign 收官结论**：paper reproduction side campaign 的 `12/12` 个论文结构复现现已全部完成。最终第一梯队为：
  - **`S1`**：总冠军，`val_acc=0.8085106382978723`, `val_auc=0.9059090909090910`
  - **`C3`**：次优，`val_acc=0.7659574468085106`, `val_auc=0.8313636363636364`
  - **`D4`**：与 `C3` 并列 accuracy 第二，但 `val_auc=0.7868181818181819`，因此排在其后
- **范式层解释**：在这套统一数据和统一 proxy budget 下，最强的不是 snapshot、prototype 或 dense vote，而是 **真正的 3D tri-view classifier（`S1`）**。其次才是多分支决策融合（`C3`）和混合 2.5D+3D ensemble（`D4`）。这说明当前任务最受益的，是能直接利用跨切片 3D 空间结构的模型，而不是更复杂的后融合启发式。
- **推荐动作**：如果继续 paper lane，下一步不再补“未完成论文”，而应做 **`S1 / C3 / D4` 的 fixed-config confirmation**，并导出最终横向对照表，为后续写作或主线借鉴提供稳定证据。

---

## 2026-04-20：Paper Reproduction Side Campaign（round2 / 4）

> **独立 side campaign 说明**
> - 这是在 round1 之后继续执行的第二轮三并行论文结构复现，对象为 `S2`, `D1`, `S4`。
> - round2 继续在 compute node `node03` 的 Slurm job `426869` 内运行；driver 日志再次确认：
>   - `requested_gpu_ids=['0','1','2']`
>   - `resolved_gpu_ids=['5','6','7']`
>   - 三个 config 全部 `exit=0`
> - 运行提交为 `ab80ea4`，唯一执行层改动是把 `paper_repro/configs/base_proxy.yaml` 的默认 `num_workers` 下调到 `2`，避免 round1 暴露出的三并行 I/O 抖动。

- [x] **PAPER-REPRO-R2-S2-SE-RESNET50**：`paper_repro/configs/s2_se_resnet50.yaml` → `val_acc=0.7340425531914894`, `val_auc=0.8243181818181818`, `val_f1=0.7252747252747253`, `peak_vram≈7.9 GiB`, `total_seconds≈679.2` → **discard（本轮内部对照）**。SE-ResNet50 trip-view classifier 有稳定表现，但显存开销远高于同轮其它结构，却没有换来最佳 accuracy。
- [x] **PAPER-REPRO-R2-D1-2P5D-MIL**：`paper_repro/configs/d1_mil_25d.yaml` → `val_acc=0.7553191489361702`, `val_auc=0.8322727272727273`, `val_f1=0.676056338028169`, `peak_vram≈2.0 GiB`, `total_seconds≈679.9` → **keep（本轮 side-campaign winner）**。bag-level attention MIL 在 round2 三者中拿到最高 accuracy，同时保持较低显存，说明“切片级证据聚合”路线对当前二分类任务是成立的。
- [x] **PAPER-REPRO-R2-S4-CNN-LSTM**：`paper_repro/configs/s4_cnn_lstm.yaml` → `val_acc=0.7127659574468085`, `val_auc=0.7979545454545455`, `val_f1=0.6746987951807228`, `peak_vram≈1.9 GiB`, `total_seconds≈678.8` → **discard（本轮内部对照）**。序列化建模能工作，但当前 CNN-LSTM 复现版没有显示出优于 MIL 或 decision fusion 的优势。
- **本轮结论**：round2 排序明确为 `D1 > S2 > S4`。`D1` 的 `val_acc` 比 round1 winner `C3` 低 `0.0106382978723404`，但 `val_auc` 反而高 `0.0009090909090909`；因此当前 paper reproduction side campaign 的第一梯队已经收敛到 **`C3` 与 `D1`** 两条“聚合/融合范式”路线。
- **资源层结论**：`S2` 需要约 `7.9 GiB` 显存，明显重于 `D1`/`S4` 的约 `2 GiB`，但 accuracy 并不占优；后续若还要保留它，只适合作为高成本对照，不应优先扩展。
- **bookkeeping 备注**：本轮发现 `paper_repro/export_results.py` 的 `append_root_results()` 之前把所有 successful record 都硬编码成 `discard`。该脚本已在 round2 后修复，`results.tsv` 采用 append-only 方式补记 `D1` 的 winner 更正行，不回写历史记录。

---

## 2026-04-20：Paper Reproduction Side Campaign（round1 / 4）

> **独立 side campaign 说明**
> - 这是按 2026-04-20 人类新需求新增的**论文复现版 autoresearch lane**，目标不是覆盖当前 canonical 主线，而是把 `/docs` 最新 4 篇综述文档里抽取出的论文结构做成 **task-adapted reproduction**，用于统一数据、统一指标下的并行对照试验。
> - 本轮新建了独立目录 `paper_repro/`，其中包含：
>   - 独立训练入口 `paper_repro/train.py`
>   - 论文模型库 `paper_repro/models.py`
>   - 12 个 paper config
>   - 4 个三并行 round manifest `paper_repro/papers.yaml`
>   - 结果导出器 `paper_repro/export_results.py`
> - round1 运行提交实际使用 `19e7c02`；初始提交 `ac70d78` 建立框架后，发现 Slurm job 内部 `CUDA_VISIBLE_DEVICES=5,6,7` 与 runner 传入 `0,1,2` 会误打到物理卡 `0,1,2`，因此补了 GPU 映射修正再重跑。
> - round1 在 compute node `node03` 的 Slurm job `426869` 上完成，driver 日志显示：
>   - `requested_gpu_ids=['0','1','2']`
>   - `resolved_gpu_ids=['5','6','7']`
>   - 三个 config 全部 `exit=0`

- [x] **PAPER-REPRO-R1-C2-SNAPSHOT-MULTIVIEW**：`paper_repro/configs/c2_snapshot_multiview.yaml` → `val_acc=0.6702127659574468`, `val_auc=0.8120454545454545`, `val_f1=0.7256637168141593`, `peak_vram≈0.8 GiB`, `total_seconds≈498.3` → **discard（本轮内部对照）**。说明 `C2` 风格的 “ROI snapshots + ResNet18” 能在当前数据上学到稳定排序，但 fixed-threshold accuracy 明显落后于 `C3`。
- [x] **PAPER-REPRO-R1-C3-DECISION-FUSION**：`paper_repro/configs/c3_decision_fusion.yaml` → `val_acc=0.7659574468085106`, `val_auc=0.8313636363636364`, `val_f1=0.7317073170731707`, `peak_vram≈1.2 GiB`, `total_seconds≈635.1` → **keep（本轮 side-campaign winner）**。当前 round1 中 accuracy 和 AUC 都最高，说明 `MMIDFNet` 式“专家分支 + 决策级融合”对现有足踝 CT 二分类最匹配。
- [x] **PAPER-REPRO-R1-D2-TRIPLANE-HYBRID**：`paper_repro/configs/d2_triplane_hybrid.yaml` → `val_acc=0.6808510638297872`, `val_auc=0.7502272727272727`, `val_f1=0.5833333333333334`, `peak_vram≈1.0 GiB`, `total_seconds≈557.9` → **discard（本轮内部对照）**。三平面 + handcrafted hybrid 有可行性，但在当前实现下还没体现出比纯决策级融合更强的收益。
- **本轮结论**：在首轮三个可落地、工程风险较低的论文复现结构里，`C3 > D2 > C2` 这一排序没有歧义：`C3` 在 accuracy / AUC 双指标都领先，且显存开销依旧很低；`D2` 的 tri-plane 表征方向有效，但当前 handcrafted late fusion 还偏弱；`C2` 的 snapshot 思路更像一种解释性较好的轻量基线，而不是当前最强者。
- **执行层经验**：NIfTI 数据在三并行下会放大 worker 竞争，round1 实测 `num_workers=4` 容易把单 epoch 拉长到 8-10 分钟级，因此已把 `paper_repro/configs/base_proxy.yaml` 的默认 `num_workers` 从 `4` 下调到 `2`，供 round2 以后沿用。
- **推荐动作**：继续按 manifest 执行 **round2 = `S2`, `D1`, `S4`**。如果 round2 仍以多分支 / MIL / 序列模型获胜，则 paper reproduction 分支后续应把资源优先投入到 `C3`, `D1`, `S4`, `D4` 这一类“聚合/融合范式”上，而不是单纯继续扩展 snapshot 或 handcrafted 支线。

---

## 2026-04-20：人类方向改动（主线切到 decision-fusion backbone 终局赛）

> **最高优先级说明**
> - 从现在起，主线不再继续 `ResNeXt + decision + 512x16 + 30 epochs` 的单点结构创新链路；该链路降级为历史 side campaign。
> - 新主线先只保留两个候选：**`cspnet-decision`** 和 **`resnext-decision`**。其他 backbone 暂不再追加预算。
> - 下一阶段的第一件事是做 **3 次高预算、完全 matched 的终局赛**；除 backbone 外，几何、训练预算、freeze、batch、workers、seed 管理和评估口径都必须保持一致。
> - 终局赛的选择规则：先比较 **多 seed 的 mean `val_acc`**，再用 mean `val_auc` 做 tie-break；如果仍然接近，再参考显存与训练耗时。
> - 一旦选出唯一的 **canonical decision-fusion backbone**，后续所有方法学实验、ablation、formal/test 叙事都只围绕这个 backbone 展开，不再切换 backbone。
> - 该轮现已完成：`resnext-decision` vs `cspnet-decision` 的 `3` seed matched 终局赛已经跑完，最终 **`resnext-decision`** 以 mean `val_acc=0.8581560283687942` 对 `0.8475177304964538`、mean `val_auc=0.9119696969696971` 对 `0.8930303030303031` 胜出；后续主线固定到 `resnext-decision`，不再横向切 backbone。

---

## 2026-04-19：人类方向改动（第 6 轮起切到结构创新）

> **高优先级说明**
> - 第 5 轮结束后，**第 6 轮开始不要再把研究变量放在纯 `lr / wd / dropout / clip / epochs` 标量或预算 retune 上**；保持当前 isolated lane 为 `ResNeXt + decision fusion + 512x16 + trim_edge_slices=2 + 30 epochs`。
> - 从第 6 轮起，研究重点切到 **网络结构层面的单点创新**，但仍然只允许每轮一个离散、可解释的结构改动。
> - **不要切回 ResUNet 直接开新主线**；ResUNet 在这里只作为灵感来源。允许 inner agent 自主决定优先尝试哪个结构点，但必须落在当前 `ResNeXt decision` 路径上。
> - 结构创新优先参考仓库里已经出现过、且在其他 lane 上给出过正向信号的模块化改动，例如：per-view feature recalibration、轻量 gated head、轻量 cross-view interaction 的迁移或重布线；inner agent 可自行判断本轮最值得先测的单一结构点。
> - 几何、融合语义和研究主线不要回退：`backbone=resnext`、`fusion_type=decision`、`image_size=512`、`num_slices_per_view=16`、`share_backbone=false`、`use_attention_pooling=false` 继续保持，除非出现真正的执行层 blocker。

---

## 当前最优纪录

> 当前 ledger 需要区分“主线 canonical 选择”和“历史单次峰值”：
> - 当前主线 canonical backbone 已收束为：**`resnext + decision fusion + 512x16 + 20 epochs`**
> - 历史单次高点、旧 proxy winner 与旧 ResUNet lane 仅保留为 reference，不再覆盖当前 backbone 锁定结论

| 指标 | 值 | 来源 commit | 配置 | 备注 |
|------|---:|-------------|------|------|
| **主线 canonical mean val_acc** | **0.8581560283687942** | `5b286a7` | `configs/cmp_backbone_decision_resnext_512x16_e20_{s42,s123,s456}.yaml` | matched 3-seed backbone final winner；相对 `cspnet-decision` mean `val_acc` 高 `0.0106382978723404` |
| **主线 canonical mean val_auc** | **0.9119696969696971** | `5b286a7` | 同上 | 与上行同一 matched final；相对 `cspnet-decision` mean `val_auc` 高 `0.0189393939393940` |
| **paper reproduction best val_acc** | **0.8510638297872340** | `ed6d535` | `paper_repro/configs/d4_hybrid_25d_3d.yaml` | corrected paper reproduction 总冠军；closer-to-paper rerun 后 `D4` 以 `val_auc=0.9309090909090909` 同时占据该 lane 的最高 AUC |
| **canonical fusion 控制变量参考** | **0.854609929078014** | `3b826ad` | `configs/cmp_decision_equal_resnext_{learned,equal}_512x16_e20_{s42,s123,s456}.yaml` | matched 3-seed fusion control：equal 的 mean `val_acc` 比 learned 高 `0.007092198581560`，但 learned 的 mean `val_auc` 反而高 `0.009090909090909`；综合规则仍先判 equal 胜出，但证据呈 split verdict |
| 全局单次 val_acc 峰值 | 0.893617021276596 | `1695ece` | `configs/cmp_fair_v100_decision_formal_cspnet.yaml` | 历史公平对比单次峰值（`256x8 / 15 epochs`），不是当前 canonical backbone |
| 历史 proxy winner | 0.8829787234042553 | `a60c3e0` | `configs/autoresearch_proxy.yaml` + fresh proxy study trial 0 | 旧 canonical proxy 参考；不再覆盖当前 backbone 锁定 |

---

## 2026-04-20：ResNeXt Equal-vs-Learned Control（seeds=42/123/456，node20 V100q）

> **独立 campaign 说明**
> - 这是按人类追加要求补上的 **canonical backbone fusion control**：不再用 `cspnet` 代替当前主线，而是在 `resnext` 上直接比较当前 learned decision fusion 与 fixed equal-weight。
> - 配方与 `resnext` backbone 终局赛 `seed=42` 完全对齐：`fusion_type=decision`、`image_size=512`、`num_slices_per_view=16`、`trim_edge_slices=2`、`batch_size=2`、`num_workers=4`、`freeze_layers=3`、`dropout=0.3`、`epochs=20`、`lr=1e-4`、`weight_decay=1e-4`。
> - 执行层上先尝试了 `RTXA6Kq` 的 batch `427232`，但 `node16` 的 3 张可见卡实际空闲显存只有 `2.31 / 2.31 / 0.03 GiB`，自检失败，不计入研究结论。
> - 第一批有效运行是 `V100q` 的 `node20` batch `427240`：申请 `1 node / 3 GPU / 12 CPU / 96G / 5h`，其中 2 张卡并行跑 `seed=42` 的 learned 与 equal，第三张保留不用；两个 step 均 `COMPLETED`。
> - 随后用同一类资源在 `node20` 上补跑剩余两组 seed：batch `427395` 同样请求 `1 node / 3 GPU / 12 CPU / 96G / 5h`，按 `seed=123 -> seed=456` 的顺序串行执行两波双并行对照；四个 step `427395.{0,1,2,3}` 均 `COMPLETED`。

- [x] **CMP-DECISION-EQUAL-RESNEXT-LEARNED-512X16-E20-S42**：`configs/cmp_decision_equal_resnext_learned_512x16_e20_s42.yaml` → `val_acc=0.8404255319148937`, `val_auc=0.8872727272727273`, `val_f1=0.810126582278481`, `peak_vram≈4.67 GiB`, `total_seconds≈2921.1` → **discard**（这轮 canonical backbone 对照里，learned weighting 明显输给 equal-weight，不仅 accuracy 低 `0.0425531914893616`，AUC 也低 `0.0490909090909091`。）
- [x] **CMP-DECISION-EQUAL-RESNEXT-EQUAL-512X16-E20-S42**：`configs/cmp_decision_equal_resnext_equal_512x16_e20_s42.yaml` → `val_acc=0.8829787234042553`, `val_auc=0.9363636363636364`, `val_f1=0.8674698795180723`, `peak_vram≈4.64 GiB`, `total_seconds≈2909.7` → **keep**（当前 canonical `resnext` 路线上的单 seed 控制变量 winner。）
- [x] **CMP-DECISION-EQUAL-RESNEXT-LEARNED-512X16-E20-S123**：`configs/cmp_decision_equal_resnext_learned_512x16_e20_s123.yaml` → `val_acc=0.851063829787234`, `val_auc=0.9109090909090910`, `val_f1=0.8444444444444444`, `peak_vram≈4.67 GiB`, `total_seconds≈2923.9` → **keep**（相同 seed 下 learned 相比 equal 把 `val_acc` 拉高 `0.0106382978723403`，AUC 也高 `0.0445454545454546`。）
- [x] **CMP-DECISION-EQUAL-RESNEXT-EQUAL-512X16-E20-S123**：`configs/cmp_decision_equal_resnext_equal_512x16_e20_s123.yaml` → `val_acc=0.8404255319148937`, `val_auc=0.8663636363636364`, `val_f1=0.8101265822784810`, `peak_vram≈4.64 GiB`, `total_seconds≈2921.4` → **discard**（在 `seed=123` 这组 matched 对照里，equal 明显输给 learned。）
- [x] **CMP-DECISION-EQUAL-RESNEXT-LEARNED-512X16-E20-S456**：`configs/cmp_decision_equal_resnext_learned_512x16_e20_s456.yaml` → `val_acc=0.851063829787234`, `val_auc=0.9209090909090910`, `val_f1=0.8250000000000000`, `peak_vram≈4.67 GiB`, `total_seconds≈2932.8` → **keep**（相同 seed 下 learned 相比 equal 再次把 `val_acc` 拉高 `0.0106382978723403`，AUC 也高 `0.0318181818181818`。）
- [x] **CMP-DECISION-EQUAL-RESNEXT-EQUAL-512X16-E20-S456**：`configs/cmp_decision_equal_resnext_equal_512x16_e20_s456.yaml` → `val_acc=0.8404255319148937`, `val_auc=0.8890909090909092`, `val_f1=0.8051948051948052`, `peak_vram≈4.64 GiB`, `total_seconds≈2910.9` → **discard**（在 `seed=456` 这组 matched 对照里，equal 依然输给 learned。）
- **三 seed 控制变量结论**：
  - `learned` mean `val_acc = 0.847517730496454`, mean `val_auc = 0.906363636363636`, mean `val_f1 = 0.826523675574308`
  - `equal` mean `val_acc = 0.854609929078014`, mean `val_auc = 0.897272727272727`, mean `val_f1 = 0.827597088997119`
  - 因此按这轮预先定义的规则（先 mean `val_acc`，再 mean `val_auc`），**equal-weight 仍然整体胜出**：mean accuracy 高 `0.007092198581560`。但这个结论并不“干净”，因为 `learned` 赢了 `2/3` 个新增 seed pair，而且 mean `val_auc` 反而高 `0.009090909090909`。
- **复现实验备注**：`seed=123 learned` 的 fresh rerun（`0.851063829787234`）与早先 backbone final 的同 seed 记录（`0.8829787234042553`）存在显著差距；两份配置 diff 只有 `experiment_name/output_dir`，因此这更像是训练重跑 / 硬件环境带来的波动，而不是显式超参差异。当前 fusion 结论仍可作为 matched compare 证据使用，但不应假装它没有稳定性风险。
- **主线动作建议**：如果现在就要按 protocol 选一个默认 fusion，应该暂时写成 **`resnext + fixed equal-weight decision fusion`**；但下一步更值得做的是 **same-hardware stability confirmation**，优先复核 `seed=123 learned` 的回落是否可复现，再决定是否把 fusion 结论永久锁死。

---

## 2026-04-20：Decision-vs-Equal Control + Backbone Final

> **独立 campaign 说明**
> - 按人类要求，这轮使用 4 GPU 分两波完成：第一波 2 卡做控制变量对照，另外 2 卡并行启动 backbone 终局赛；等第一波结束后，第二波补齐剩余 `4` 个 backbone 实验。
> - 有效运行最终固定在 `node08` 的 4 张 `RTX A6000` 上；由于共享节点显存噪声，matched config 的 `batch_size` 从初始设想收紧到 `2`，其余主变量保持不变：`fusion_type=decision`、`image_size=512`、`num_slices_per_view=16`、`trim_edge_slices=2`、`epochs=20`、`freeze_layers=3`、`dropout=0.3`、`lr=1e-4`、`weight_decay=1e-4`、`num_workers=4`。
> - 早期 `node16` 的 4-GPU batch job `426417` 因可见卡/显存自检失败，随后 `426429~426432` 又因共享繁忙卡 OOM；这些执行层噪声全部**不计入研究结论**。

- [x] **CMP-DECISION-EQUAL-CSPNET-LEARNED-512X16-E20-S42**：`configs/cmp_decision_equal_cspnet_learned_512x16_e20_s42.yaml` → `val_acc=0.851063829787234`, `val_auc=0.8640909090909091`, `val_f1=0.8157894736842105`, `peak_vram≈4.04 GiB`, `total_seconds≈4877.1` → **discard**（与 fixed equal-weight 控制组在主指标 `val_acc` 上完全持平，没有任何准确率收益；而且 `val_auc` 反而低 `0.0522727272727273`。）
- [x] **CMP-DECISION-EQUAL-CSPNET-EQUAL-512X16-E20-S42**：`configs/cmp_decision_equal_cspnet_equal_512x16_e20_s42.yaml` → `val_acc=0.851063829787234`, `val_auc=0.9163636363636364`, `val_f1=0.8333333333333334`, `peak_vram≈4.01 GiB`, `total_seconds≈5385.7` → **keep**（这是这轮控制变量对照里的 winner：准确率与 learned decision fusion 完全相同，但 AUC 更高。）
- **控制变量结论**：在当前这组完全 matched 的 `cspnet / seed=42 / decision-fusion geometry / 20 epochs` 对照里，**当前 learned decision fusion 相比 fixed equal-weight 的准确率提升为 `0.000000`**；也就是没有带来净的 `val_acc` 提升。更强的结论反而是：equal-weight 的 `val_auc` 更高 `0.0522727272727273`，说明“learned weighting 一定更好”在这组设置下不成立。

- [x] **CMP-BACKBONE-DECISION-RESNEXT-512X16-E20-S42**：`configs/cmp_backbone_decision_resnext_512x16_e20_s42.yaml` → `val_acc=0.8404255319148937`, `val_auc=0.9136363636363637`, `val_f1=0.8314606741573034`, `peak_vram≈4.67 GiB`, `total_seconds≈4311.7`
- [x] **CMP-BACKBONE-DECISION-RESNEXT-512X16-E20-S123**：`configs/cmp_backbone_decision_resnext_512x16_e20_s123.yaml` → `val_acc=0.8829787234042553`, `val_auc=0.9181818181818182`, `val_f1=0.8705882352941177`, `peak_vram≈4.67 GiB`, `total_seconds≈2768.8`
- [x] **CMP-BACKBONE-DECISION-RESNEXT-512X16-E20-S456**：`configs/cmp_backbone_decision_resnext_512x16_e20_s456.yaml` → `val_acc=0.851063829787234`, `val_auc=0.9040909090909092`, `val_f1=0.825`, `peak_vram≈4.67 GiB`, `total_seconds≈3478.3`
- [x] **CMP-BACKBONE-DECISION-CSPNET-512X16-E20-S42**：`configs/cmp_backbone_decision_cspnet_512x16_e20_s42.yaml` → `val_acc=0.851063829787234`, `val_auc=0.8640909090909091`, `val_f1=0.8157894736842105`, `peak_vram≈4.04 GiB`, `total_seconds≈5567.6`
- [x] **CMP-BACKBONE-DECISION-CSPNET-512X16-E20-S123**：`configs/cmp_backbone_decision_cspnet_512x16_e20_s123.yaml` → `val_acc=0.8297872340425532`, `val_auc=0.896818181818182`, `val_f1=0.7777777777777778`, `peak_vram≈4.04 GiB`, `total_seconds≈3545.0`
- [x] **CMP-BACKBONE-DECISION-CSPNET-512X16-E20-S456**：`configs/cmp_backbone_decision_cspnet_512x16_e20_s456.yaml` → `val_acc=0.8617021276595744`, `val_auc=0.9181818181818182`, `val_f1=0.8395061728395061`, `peak_vram≈4.04 GiB`, `total_seconds≈2863.8`
- **多 seed backbone 终局赛结论**：
  - `resnext-decision` mean `val_acc = 0.8581560283687942`, mean `val_auc = 0.9119696969696971`
  - `cspnet-decision` mean `val_acc = 0.8475177304964538`, mean `val_auc = 0.8930303030303031`
  - 因此按这轮预先定义的规则（先 mean `val_acc`，再 mean `val_auc`），**`resnext-decision` 胜出**；mean accuracy 领先 `0.0106382978723404`，mean AUC 领先 `0.0189393939393940`。
- **主线动作建议**：这轮 backbone 终局赛已经足够把主线收束到 **`resnext + decision fusion`**。如果后续还要继续写方法学实验，建议不要再横向切 backbone；优先在 `resnext-decision` 上继续做 fusion/weighting 机制本身的分析与改造。与此同时，这次 cspnet 控制变量对照表明“learned weighting 对比 equal-weight 没有准确率净收益”，因此下一步若继续 fusion 研究，应先把这个负面证据解释清楚，而不是默认 learned gating 一定优于均匀投票。

---

## 2026-04-19：Decision-Fusion Fair Backbone Compare（bs=6, nw=4）

> **独立 compare campaign 说明**
> - 这一轮不是主线 `ResNeXt 512x16` 结构创新实验，而是按人类要求对当前仓库里可用的几类 backbone 做一次**重新并行公平对比**。
> - 为避免与旧的 `feature-fusion fair-backbone compare` 混淆，本轮使用新的隔离配置：`configs/cmp_fair_v100_decision_formal_{resunet,resnext,senet,cspnet}.yaml`。
> - 统一 recipe 为：`fusion_type=decision`、`image_size=256`、`num_slices_per_view=8`、`trim_edge_slices=1`、`share_backbone=false`、`use_attention_pooling=false`、`freeze_layers=3`、`epochs=15`、`lr=1e-4`、`weight_decay=1e-4`，并按人类要求固定 `batch_size=6`、`num_workers=4`。
> - 运行入口为 `scripts/run_fair_backbone_decision_formal.py`；由于 `node20` 的 attach shell 实测只暴露 `CUDA_VISIBLE_DEVICES=0,1`，所以本轮实际以**两波双卡并行**完成，而不是三卡并行。

- [x] **CMP-FAIR-V100-DECISION-FORMAL-RESUNET-BS6-NW4**：`configs/cmp_fair_v100_decision_formal_resunet.yaml` → `val_acc=0.872340425531915`, `val_auc=0.889090909090909`, `val_f1=0.853658536585366`, `peak_vram≈3.00 GiB`, `total_seconds=1791.4` → **discard**（高于同轮 `senet`，但明显低于并列第一的 `resnext / cspnet`。）
- [x] **CMP-FAIR-V100-DECISION-FORMAL-RESNEXT-BS6-NW4**：`configs/cmp_fair_v100_decision_formal_resnext.yaml` → `val_acc=0.893617021276596`, `val_auc=0.925909090909091`, `val_f1=0.883720930232558`, `peak_vram≈2.17 GiB`, `total_seconds=1809.2` → **discard**（与 `cspnet` 打平本轮最高 accuracy，但 AUC 低 `0.013636363636364`，因此按 tie-break 输给 `cspnet`。）
- [x] **CMP-FAIR-V100-DECISION-FORMAL-SENET-BS6-NW4**：`configs/cmp_fair_v100_decision_formal_senet.yaml` → `val_acc=0.829787234042553`, `val_auc=0.926818181818182`, `val_f1=0.794871794871795`, `peak_vram≈2.19 GiB`, `total_seconds=1789.6` → **discard**（AUC 不差，但 accuracy 平台明显低于前三者，不能保留。）
- [x] **CMP-FAIR-V100-DECISION-FORMAL-CSPNET-BS6-NW4**：`configs/cmp_fair_v100_decision_formal_cspnet.yaml` → `val_acc=0.893617021276596`, `val_auc=0.939545454545455`, `val_f1=0.875`, `peak_vram≈1.89 GiB`, `total_seconds=1813.0` → **keep**（与 `resnext` 并列本轮最高 accuracy，同时 AUC 更高、显存更低，因此是这轮独立 compare campaign 的 winner。）
- **本轮结论**：把 fair compare 的融合语义从旧 `feature` 切到 `decision`，并固定 `bs=6 / nw=4` 后，backbone 排名发生了实质变化：`cspnet` 与 `resnext` 一起站上第一梯队，而 `cspnet` 在 tie-break 上更强。`resunet` 仍有竞争力，但没有跟上前二；`senet` 则更像“排序质量尚可但阈值准确率不够”的第三梯队。
- **推荐动作**：如果人类还要继续这个 compare 分支，下一步不要把 4 个 backbone 全部重跑；直接对 **`cspnet` 和 `resnext`** 做 `seed=123` 的 matched confirmation，判断这次 `decision-fusion` 下的 `cspnet` 优势是否稳定复现。

---

## 2026-04-19：ResNeXt V100 Dedicated Main-Study（decision fusion + 512x16 scale-only reliability calibrator，fresh rerun）

> **独立 campaign 说明**
> - 这一轮继续承接 `decision fusion` side campaign，只推进 `resnext`，不回到 canonical ResUNet 主线，也不回退到旧 `feature fusion` lane。
> - 保持强制几何 pivot 不变：`image_size=512`, `num_slices_per_view=16`, `trim_edge_slices=2`, `share_backbone=false`, `use_attention_pooling=false`, `batch_size=2`，并继续固定 `train.epochs=30`。
> - 本轮不再引入新的模型结构变量；唯一研究问题是：在 `GPU 0,1,2` 真正空闲后，对上轮因资源阻塞而未能测量的 **identity-init、scale-only 的 temperature-like reliability calibrator** 做第一次完整 fresh adaptive main-study，判断“去掉 affine bias、只保留乘性温度修正”是否真能改善这条 lane。
> - 本轮使用 search-config copy `autoresearch_logs/generated_search_configs/optuna_main_search_iter_0076_20260419_153814.yaml`，fresh study_root `runs/optuna_main_autoloop/iter_0076_20260419_153814`，运行 commit 为 `b818f20`，并在空闲 `GPU 0,1,2` 上完成 1 次 adaptive main-study。

- [x] **CMP-FAIR-V100-RESNEXT-DECISION-MAIN-512X16-SCALE-ONLY-RELIABILITY-CALIBRATOR-RERUN**：保持 `src/model.py` 中已提交的 scale-only / temperature-like reliability calibrator 不变，仅用 fresh study 重新测量这条结构 → wave-aligned 调度共 `6/6` trials completed；best completed trial 为 **trial 1**（`freeze_layers=3`, `lr=3e-5`, `weight_decay=7.5e-4`, `dropout=0.4`, `gradient_clip_norm=2.0`）→ `val_acc=0.8297872340425532`, `val_auc=0.8972727272727272`, `val_f1=0.8`, `peak_vram≈4.67 GiB`, `total_seconds=4282.6` → **discard**（这轮最好结果不仅明显低于上一轮 `affine reliability calibrator` 的 `0.8829787234042553 / 0.9118181818181819`，accuracy / AUC 分别低 `0.0531914893617021 / 0.0145454545454547`；也明显低于 `gated reliability adapter` 的 `0.8829787234042553 / 0.9354545454545455`，accuracy / AUC 分别低 `0.0531914893617021 / 0.0381818181818183`；因此不能保留。）
- **Monitor takeaways**：这轮 `6` 个 valid trials 没有 OOM、timeout 或 workflow crash，`peak_vram` 全部稳定在约 `4.67 GiB`，说明结论是结构层负结果，不是资源噪声。更重要的是，6 个 trial 的 accuracy 只落在 `0.8191489361702128` 或 `0.8297872340425532` 两个平台，最好 AUC 也只有 trial `4` 的 `0.8986363636363637`；这说明一旦把 reliability calibrator 收紧到“纯乘性温度缩放”，这条 lane 的 reliability weighting 几乎失去了区分力，既没保住此前的 accuracy ceiling，也没换来更好的 ranking 质量。
- **本轮结论**：上轮 backlog 中“也许是 additive bias 在伤害 AUC，所以可以继续收紧到 scale-only”这个假设，现在已经拿到干净的否定证据。问题不在于 bias 一项是否过强，而在于 **scale-only 本身把 reliability path 压得过弱**，导致整个 `512x16` decision lane 的表现明显塌陷。因此本轮记 **discard**，且没有必要补 direct formal。
- **推荐动作**：如果外层 loop 继续推进这条 isolated decision-fusion lane，下一步应保持 `512x16`、`trim_edge_slices=2`、`share_backbone=false`、`use_attention_pooling=false` 和 `30`-epoch budget 不变，但不要继续围绕 scale-only / 更弱 calibrator 做 retune。更合理的方向是把唯一结构改动转到**更有表达力但仍低方差的 reliability-side 适配器**，例如共享的低秩 residual gate / offset，同时继续保持 plain per-view classifier 不变。

---

## 2026-04-19：ResNeXt V100 Dedicated Main-Study（decision fusion + 512x16 scale-only reliability calibrator，资源阻塞）

> **独立 campaign 说明**
> - 这一轮继续承接 `decision fusion` side campaign，只推进 `resnext`，不回到 canonical ResUNet 主线，也不回退到旧 `feature fusion` lane。
> - 保持强制几何 pivot 不变：`image_size=512`, `num_slices_per_view=16`, `trim_edge_slices=2`, `share_backbone=false`, `use_attention_pooling=false`, `batch_size=2`，并继续固定 `train.epochs=30`。
> - 唯一离散研究改动：把上一轮 bounded affine reliability calibrator 继续收紧成 **identity-init、scale-only 的 temperature-like reliability calibrator**，只保留对 raw reliability logit 的乘性温度修正，显式去掉 additive bias；plain per-view classifier、轻量 cross-view mixer 与其余融合语义保持不变。
> - 本轮使用 search-config copy `autoresearch_logs/generated_search_configs/optuna_main_search_iter_0011_20260419_142314.yaml`，目标 fresh study_root 为 `runs/optuna_main_autoloop/iter_0011_20260419_142314`，运行 commit 为 `800c0ec`，并按 coordinator 硬约束尝试在 `GPU 0,1,2` 上启动 1 次 adaptive main-study。

- [x] **CMP-FAIR-V100-RESNEXT-DECISION-MAIN-512X16-SCALE-ONLY-RELIABILITY-CALIBRATOR**：`src/model.py`（仅在 `decision fusion` 分支把 bounded affine reliability calibrator 改成 scale-only / temperature-like 版本）+ fresh adaptive main-study 启动尝试（`GPU 0,1,2`）→ **crash**（入口在创建 study 前即退出；`optuna_main.log` 尾部只显示 idle-threshold 拒绝，没有 completed trial、没有 `summary.json`，因此本轮 `val_acc / val_auc / val_f1` 记为 `0`。）
- **Crash takeaways**：当前 `scripts/autoresearch_main.py --gpu-ids 0,1,2 --max-workers 3 --max-used-memory-mb 1024 --max-utilization 20` 会对显式卡集做严格 idle 检查，不会自动只拿 `GPU 2`。本轮启动时 `GPU 0` 占用 `18240 MiB` 且 `util=100%`，`GPU 1` 占用 `17864 MiB`；进一步检查可见两张卡上分别运行着 `.venv/bin/python train.py --config configs/cmp_resnext_feature_minimal_512_b8_e15.yaml` 与 `configs/cmp_resnext_decision_minimal_512_b8_e15.yaml`，所以这是标准的**资源阻塞**，不是模型、数据或 workflow 逻辑崩坏。
- **本轮结论**：这轮不能对 `scale-only reliability calibrator` 做 keep/discard 研究判断。唯一能够确认的是：当前结构改动已经提交并准备完毕，但执行层资源不满足 coordinator 的显式 GPU policy，因此本轮按 **crash** 记录。
- **推荐动作**：如果外层 loop 继续推进这条 isolated decision-fusion lane，下一步应保持本轮结构与 `512x16 / 30`-epoch 几何不变，等 `GPU 0,1,2` 真正通过 idle 阈值后，用新的 fresh `study_root` 原样重开同一 adaptive main-study；不要因为这次 crash 回退到 affine bias、也不要改走 proxy。

---

## 2026-04-19：ResNeXt V100 Dedicated Main-Study（decision fusion + 512x16 affine reliability calibrator）

> **独立 campaign 说明**
> - 这一轮继续承接 `decision fusion` side campaign，只推进 `resnext`，不回到 canonical ResUNet 主线，也不回退到旧 `feature fusion` lane。
> - 保持强制几何 pivot 不变：`image_size=512`, `num_slices_per_view=16`, `trim_edge_slices=2`, `share_backbone=false`, `use_attention_pooling=false`, `batch_size=2`，并继续固定 `train.epochs=30`。
> - 唯一离散研究改动：保留 plain per-view classifier、轻量 cross-view mixer 和 baseline raw-logit VRG 头，仅把上一轮的 zero-init GLU residual confidence adapter 收紧成 **identity-init、bounded 的 affine reliability calibrator**，让 reliability path 只对 raw logit 做小幅 scale/bias 校准。
> - 本轮使用 search-config copy `autoresearch_logs/generated_search_configs/optuna_main_search_iter_0010_20260419_114322.yaml`，fresh study_root `runs/optuna_main_autoloop/iter_0010_20260419_114322`，运行 commit 为 `26cdbeb`，并在空闲 `GPU 0,1,2` 上完成 1 次 adaptive main-study。

- [x] **CMP-FAIR-V100-RESNEXT-DECISION-MAIN-512X16-AFFINE-RELIABILITY-CALIBRATOR**：`src/model.py`（仅在 `decision fusion` 分支把 full-rank GLU confidence adapter 改成 bounded affine reliability calibrator）+ fresh adaptive main-study（`GPU 0,1,2`）→ wave-aligned 调度共 `6` 个 trial，其中 `4` 个 completed / `2` 个 crash；best completed trial 为 **trial 3**（`freeze_layers=3`, `lr=3e-5`, `weight_decay=1e-3`, `dropout=0.35`, `gradient_clip_norm=1.5`）→ `val_acc=0.8829787234042553`, `val_auc=0.9118181818181819`, `val_f1=0.8641975308641975`, `peak_vram≈4.67 GiB`, `total_seconds=4272.3` → **discard**（这轮最好结果在 accuracy 上追平了 `513b94f` 与 `c551967` 的 `0.8829787234042553` ceiling，但 AUC 明显更低：相较 `513b94f` 的 `0.9354545454545455` 低 `0.0236363636363636`，相较 `c551967` 的 `0.9409090909090909` 低 `0.0290909090909090`，因此在主指标持平时按 tie-break 仍不能保留；它也仍明显低于当前 decision-fusion direct-formal anchor `7ae19a0` 的 `0.925531914893617 / 0.9640909090909091`，accuracy / AUC 分别低 `0.0425531914893617 / 0.0522727272727272`。）
- **Monitor takeaways**：这轮 `4` 个 valid trials 的分布说明，简化 reliability path 并没有把这条 lane 拉回更强的 ranking 质量。trial `0/1` 只到 `0.8191489361702128`，trial `2` 更低到 `0.7872340425531915`，而唯一追平 accuracy ceiling 的 trial `3` 恰好落在上一批结构实验也偏好的正则角点（`dropout=0.35`, `clip=1.5`, `lr=3e-5`, `wd=1e-3`）；这说明当前结构变化并没有改变这条 lane 的 scalar preference，只是把 top trial 的 AUC 压低了。
- **Crash takeaways**：wave-aligned tail-fill 的 `trial 4/5` 都在约 `33` 分钟时同时结束，并在各自 `train.log` 里留下 `EXIT_CODE=-15`；日志尾部没有 `CUDA out of memory`、`TIMEOUT`、`Traceback` 或数据错误信号，因此它们更像 worker 被外层流程提前终止，而不是模型或数据 blocker。由于 study 仍然完成了请求的 `4` 个 valid trials，本轮整体仍按有效 study 记 **discard**，不记 crash。
- **本轮结论**：`affine reliability calibrator` 说明“降低 reliability head 的表达力”并不会立刻把 accuracy 打回低点，甚至还能保住 `0.8829787234042553` 的 ceiling；但它同时显著损伤 AUC，意味着当前 affine 形式过于粗糙，尤其是 bias 修正很可能在破坏 ranking 质量。因此这轮仍是 **discard**，且没有必要补 direct formal。
- **推荐动作**：如果外层 loop 继续推进这条 isolated decision-fusion lane，下一步仍应保持 `512x16`、`trim_edge_slices=2`、`share_backbone=false`、`use_attention_pooling=false` 和 `30`-epoch budget 不变，但把 reliability calibrator **继续收紧到 temperature-like / scale-only 版本**：去掉 affine bias，只保留 identity-init 的乘性温度修正，验证本轮 AUC 回落是否主要来自 additive bias。

---

## 2026-04-19：ResNeXt V100 Direct Formal Confirmation（decision fusion + 512x16 gated reliability adapter trial-1 replay）

> **独立 campaign 说明**
> - 这一轮继续承接 `decision fusion` side campaign，只推进 `resnext`，不回到 canonical ResUNet 主线，也不回退到旧 `feature fusion` lane。
> - 保持强制几何 pivot 不变：`image_size=512`, `num_slices_per_view=16`, `trim_edge_slices=2`, `share_backbone=false`, `use_attention_pooling=false`, `batch_size=2`，并继续固定 `train.epochs=30`。
> - 唯一离散研究改动：把 dedicated formal template 对齐到上一轮 gated reliability-adapter main-study best completed trial（`freeze_layers=3`, `lr=3e-5`, `weight_decay=7.5e-4`, `dropout=0.4`, `gradient_clip_norm=2.0`），用单卡 formal 语义验证这条 reliability-side 改善是否可复现。
> - 本轮运行 commit 为 `eef01f1`；由于这是 fixed-config confirmation 而不是 fresh study，直接运行 `CUDA_VISIBLE_DEVICES=0 timeout 10800 .venv/bin/python train.py --config configs/autoresearch_formal_resnext_decision_v100.yaml`。

- [x] **CMP-FAIR-V100-RESNEXT-DECISION-FORMAL-512X16-GATED-RELIABILITY-CONFIRM**：`configs/autoresearch_formal_resnext_decision_v100.yaml`（保持 `512x16 / 30`-epoch 配方不变，仅把 dedicated formal 对齐到 `513b94f` main-study 的 best completed trial：`freeze_layers=3`, `lr=3e-5`, `weight_decay=7.5e-4`, `dropout=0.4`, `gradient_clip_norm=2.0`）→ `val_acc=0.8617021276595744`, `val_auc=0.9372727272727273`, `val_f1=0.8433734939759037`, `peak_vram≈4.68 GiB`, `total_seconds=4260.7` → **discard**（这次 replay 没有复现上一轮 main-study 的 `0.8829787234042553 / 0.9354545454545455`，accuracy 反而低了 `0.0212765957446809`；虽然 AUC 微升 `0.0018181818181818`，但主指标明显回落，因此不能保留。它也仍明显低于当前 decision-fusion direct-formal anchor `7ae19a0` 的 `0.925531914893617 / 0.9640909090909091`，accuracy / AUC 分别低 `0.0638297872340426 / 0.0268181818181818`。）
- **Stability takeaway**：这次 formal confirmation 的落点几乎就是一个“回到旧平台”的信号。它没有延续 `513b94f` 在 fresh main-study 里追平 `c551967` accuracy ceiling 的势头，而是直接落回了 `0.861702 / 0.937273` 这一保守平台；更强的线索是，它与当前 canonical formal 参考 `c30fcae` 的主指标完全相同。这说明当前 full-rank gated reliability adapter 也许能在 study 里帮助搜索找到更好的 trial，但还没有形成稳定的 formal 优势。
- **本轮结论**：这次实验完成了 backlog 中建议的 fixed-config confirmation，结论是：**当前 gated residual reliability adapter 不是稳定可复现的新 anchor**。因此本轮记 **discard**，后续不应继续重复同一 fixed config replay。
- **推荐动作**：如果外层 loop 继续推进这条 isolated decision-fusion lane，下一步仍应保持 `512x16`、`trim_edge_slices=2`、`share_backbone=false`、`use_attention_pooling=false` 和 `30`-epoch budget 不变，但把 reliability-side 结构创新**简化**而不是继续加表达力：保留 plain per-view classifier，只把 current GLU confidence adapter 改成更低方差的 temperature-like / affine reliability calibrator，再做 1 次新的 adaptive main-study。

---

## 2026-04-19：ResNeXt V100 Dedicated Main-Study（decision fusion + 512x16 gated reliability adapter）

> **独立 campaign 说明**
> - 这一轮继续承接 `decision fusion` side campaign，只推进 `resnext`，不回到 canonical ResUNet 主线，也不回退到旧 `feature fusion` lane。
> - 保持强制几何 pivot 不变：`image_size=512`, `num_slices_per_view=16`, `trim_edge_slices=2`, `share_backbone=false`, `use_attention_pooling=false`, `batch_size=2`，并继续固定 `train.epochs=30`。
> - 唯一离散研究改动：把 `MultiViewDecisionFusionClassifier` 的 per-view classifier 恢复到 plain `Linear + ReLU` baseline，并把结构自由度集中到 VRG reliability path：保留 `LayerNorm + Linear(512→1)` raw-logit head，同时叠加零初始化的轻量 GLU gated residual adapter；搜索空间继续保持现有 low-lr / high-regularization family，不再额外引入第二个结构变量。
> - 本轮使用 search-config copy `autoresearch_logs/generated_search_configs/optuna_main_search_iter_0008_20260419_075732.yaml`，fresh study_root `runs/optuna_main_autoloop/iter_0008_20260419_075732`，运行 commit 为 `513b94f`，并在空闲 `GPU 0,1,2` 上完成 1 次 adaptive main-study。

- [x] **CMP-FAIR-V100-RESNEXT-DECISION-MAIN-512X16-GATED-RELIABILITY-ADAPTER**：`src/model.py`（仅在 `decision fusion` 分支恢复 plain per-view classifier，并把唯一结构改动放到 zero-init gated residual confidence adapter）+ fresh adaptive main-study（`GPU 0,1,2`）→ wave-aligned 调度共 `6/6` trials completed，best completed trial 为 **trial 1**（`freeze_layers=3`, `lr=3e-5`, `weight_decay=7.5e-4`, `dropout=0.4`, `gradient_clip_norm=2.0`）→ `val_acc=0.8829787234042553`, `val_auc=0.9354545454545455`, `val_f1=0.8641975308641975`, `peak_vram≈4.68 GiB`, `total_seconds=4283.4` → **discard**（这轮结果把 isolated `512x16` lane 的 best accuracy 拉回到了 long-budget winner `c551967` 的同一高度，但 AUC 仍低 `0.0054545454545454`，因此在主指标持平时按 tie-break 仍然不能保留；同时它仍明显低于当前 decision-fusion direct-formal anchor `7ae19a0` 的 `0.925531914893617 / 0.9640909090909091`，accuracy 低 `0.0425531914893617`、AUC 低 `0.0286363636363636`。）
- **Monitor takeaways**：这轮 6 个 completed trials 没有 OOM、timeout 或 workflow crash，`peak_vram` 全部稳定在约 `4.68 GiB`，说明结论仍然是优化 / 结构层，而不是资源问题。更重要的是，trial 分布不再像上两轮结构创新那样整体塌到 `0.83` 左右：本轮 accuracy 覆盖了 `0.7978723404255319`、`0.8617021276595744`、`0.8723404255319149` 与 `0.8829787234042553` 四个平台，其中 top trial 精确追平了 `c551967` 的 accuracy ceiling。这个信号说明，把表达力从 classifier path 挪到 reliability weighting path，至少比 per-view recalibration 或 gated classifier head 更接近这条 lane 的真实瓶颈。
- **本轮结论**：这次不是 clean negative，但也还不足以 keep。`gated residual reliability adapter` 给出了这轮 isolated decision lane 里最有竞争力的结构创新证据，却仍然没能在 tie-break 上超过 `c551967`，更没有接近当前 direct-formal anchor。因此本轮仍记 **discard**，但它已经把后续优先级从“继续换别的结构点盲试”推向“先验证这条 reliability-side 信号是否可复现”。
- **推荐动作**：如果外层 loop 继续推进这条 isolated decision-fusion lane，下一步应优先做 **1 次 direct formal confirmation**，把本轮 trial 1 的 fixed config（`freeze_layers=3`, `lr=3e-5`, `weight_decay=7.5e-4`, `dropout=0.4`, `gradient_clip_norm=2.0`）在单卡 formal 语义下再跑 1 次。只有在这次 confirmation 仍然回落明显时，才值得继续设计下一个 reliability-path 结构变体。

---

## 2026-04-19：ResNeXt V100 Dedicated Main-Study（decision fusion + 512x16 gated per-view head）

> **独立 campaign 说明**
> - 这一轮继续承接 `decision fusion` side campaign，只推进 `resnext`，不回到 canonical ResUNet 主线，也不回退到旧 `feature fusion` lane。
> - 保持强制几何 pivot 不变：`image_size=512`, `num_slices_per_view=16`, `trim_edge_slices=2`, `share_backbone=false`, `use_attention_pooling=false`, `batch_size=2`，并继续固定 `train.epochs=30`。
> - 唯一离散研究改动：把 `MultiViewDecisionFusionClassifier` 的 per-view classifier 从 plain `Linear + ReLU` MLP 改成轻量 GLU gated head，同时撤掉上轮失败的 per-view recalibration，让 VRG reliability weighting 路径尽量回到 `96fe172` 的 baseline 语义；搜索空间保持现有 low-lr / high-regularization family，不再额外引入第二个结构变量。
> - 本轮使用 search-config copy `autoresearch_logs/generated_search_configs/optuna_main_search_iter_0007_20260419_052750.yaml`，fresh study_root `runs/optuna_main_autoloop/iter_0007_20260419_052750`，运行 commit 为 `41072da`，并在空闲 `GPU 0,1,2` 上完成 1 次 adaptive main-study。

- [x] **CMP-FAIR-V100-RESNEXT-DECISION-MAIN-512X16-GATED-PER-VIEW-HEAD**：`src/model.py`（仅在 `decision fusion` 分支把 per-view classifier 改成 GLU gated head，并移除上轮的 per-view recalibration）+ fresh adaptive main-study（`GPU 0,1,2`）→ wave-aligned 调度共 `6/6` trials completed，best completed trial 为 **trial 5**（`freeze_layers=3`, `lr=4e-5`, `weight_decay=1e-3`, `dropout=0.35`, `gradient_clip_norm=1.5`）→ `val_acc=0.8297872340425532`, `val_auc=0.915`, `val_f1=0.8048780487804879`, `peak_vram≈4.67 GiB`, `total_seconds=4257.3` → **discard**（这轮最好结果不仅低于上一轮 long-budget main-study winner `c551967` 的 `0.8829787234042553 / 0.9409090909090909`，accuracy 低 `0.0531914893617021`、AUC 低 `0.0259090909090909`；也明显低于当前 decision-fusion direct-formal anchor `7ae19a0` 的 `0.925531914893617 / 0.9640909090909091`，accuracy 低 `0.0957446808510638`、AUC 低 `0.0490909090909091`；因此不能保留。）
- **Monitor takeaways**：这轮 6 个 completed trials 没有 OOM、timeout 或 workflow crash，`peak_vram` 都稳定在约 `4.67 GiB`，说明本轮结论仍然是优化/结构层负结果，不是资源问题。更重要的是，6 个 trial 的 accuracy 只落在 `0.8085106382978723`、`0.8191489361702128` 或 `0.8297872340425532` 三个平台，最好 accuracy 比上轮 per-view recalibration 的 `0.8617021276595744` 还再低 `0.0319148936170212`。虽然最好 AUC 回升到 `0.915`，较上轮 `0.8954545454545454` 高 `0.0195454545454546`，但 fixed-threshold accuracy 明显塌陷，说明单纯提高 per-view classifier 的乘性表达力，并没有解决 `512x16` decision lane 的核心问题。
- **本轮结论**：这次结构创新同样给出了干净的否定证据。`gated per-view classifier head` 在当前 `ResNeXt decision 512x16` 路径上既没有改善 validation accuracy，也没有把这条 lane 拉回先前 `30`-epoch 高点，因此本轮记 **discard**，后续不应继续重复这条改动或围绕它做纯标量 retune。
- **推荐动作**：如果外层 loop 继续推进这条 isolated decision-fusion lane，下一步仍应保持 `512x16`、`trim_edge_slices=2`、`share_backbone=false`、`use_attention_pooling=false` 和 `30`-epoch budget 不变，但把结构创新从 classifier path 挪到 **VRG confidence / reliability head**：保留 plain per-view classifier 与当前 baseline pooled-feature path，只把每个视角的 confidence head 从 `LayerNorm + Linear(512→1)` 改成轻量 gated confidence head 或 temperature-like reliability adapter，以便更干净地判断瓶颈是否在融合权重建模，而不是 per-view 分类表达力。

---

## 2026-04-19：ResNeXt V100 Dedicated Main-Study（decision fusion + 512x16 per-view recalibration）

> **独立 campaign 说明**
> - 这一轮继续承接 `decision fusion` side campaign，只推进 `resnext`，不回到 canonical ResUNet 主线，也不回退到旧 `feature fusion` lane。
> - 保持强制几何 pivot 不变：`image_size=512`, `num_slices_per_view=16`, `trim_edge_slices=2`, `share_backbone=false`, `use_attention_pooling=false`, `batch_size=2`，并继续固定 `train.epochs=30`。
> - 唯一离散研究改动：把仓库里已有的 `ViewFeatureRecalibration` 模块接到 `MultiViewDecisionFusionClassifier`，让每个视角的 pooled token 在进入 per-view classifier 与 VRG reliability head 前先做一次 identity-initialized 的轻量通道重标定；搜索空间保持现有 low-lr / high-regularization family，不再额外引入第二个结构变量。
> - 本轮使用 search-config copy `autoresearch_logs/generated_search_configs/optuna_main_search_iter_0006_20260419_025853.yaml`，fresh study_root `runs/optuna_main_autoloop/iter_0006_20260419_025853`，运行 commit 为 `b47975e`，并在空闲 `GPU 0,1,2` 上完成 1 次 adaptive main-study。

- [x] **CMP-FAIR-V100-RESNEXT-DECISION-MAIN-512X16-VIEW-RECALIBRATION**：`src/model.py`（仅在 `decision fusion` 分支加入 per-view recalibration）+ fresh adaptive main-study（`GPU 0,1,2`）→ wave-aligned 调度共 `6/6` trials completed，best completed trial 为 **trial 1**（`freeze_layers=3`, `lr=3e-5`, `weight_decay=7.5e-4`, `dropout=0.4`, `gradient_clip_norm=2.0`）→ `val_acc=0.8617021276595744`, `val_auc=0.8954545454545454`, `val_f1=0.8266666666666667`, `peak_vram≈4.68 GiB`, `total_seconds=4285.1` → **discard**（这轮最好结果仍低于上一轮 long-budget main-study winner `c551967` 的 `0.8829787234042553 / 0.9409090909090909`，accuracy 低 `0.0212765957446809`、AUC 低 `0.0454545454545455`；相较当前 decision-fusion direct-formal anchor `7ae19a0` 的 `0.925531914893617 / 0.9640909090909091` 更低 `0.0638297872340426 / 0.0686363636363637`，因此不能保留。）
- **Monitor takeaways**：这轮 6 个 completed trials 没有 OOM、timeout 或 workflow crash，`peak_vram` 都稳定在约 `4.68 GiB`，说明本轮结论仍然是优化/结构层负结果，不是资源问题。更重要的是，6 个 trial 的 accuracy 只落在两个平台：`0.851063829787234` 或 `0.8617021276595744`；没有任何一个 trial 接近 `c551967` 的 `0.8829787234042553`。同时 tie-high 的最好 AUC 只有 `0.8954545454545454`，显著弱于先前 `512x16 / 30`-epoch family 的 retained high points，说明这次 token-level recalibration 没有起到“稳定 logit”作用，反而更像是把 classifier 和 VRG weighting 两条分支一起过度收缩了。
- **本轮结论**：这次结构创新已经给出了干净的否定证据。`per-view recalibration` 在当前 `ResNeXt decision 512x16` 路径上没有带来更稳的 validation accuracy，也没有改善 AUC，因此本轮记 **discard**，后续不应继续重复这条改动或围绕它做纯标量 retune。
- **推荐动作**：如果外层 loop 继续推进这条 isolated decision-fusion lane，下一步仍应保持 `512x16`、`trim_edge_slices=2`、`share_backbone=false`、`use_attention_pooling=false` 和 `30`-epoch budget 不变，但把结构创新切到**轻量 gated per-view classifier head**，并尽量保持 VRG reliability 分支接近当前 baseline，以便更干净地判断瓶颈到底在 per-view 分类表达力，还是在视角权重估计。

---

## 2026-04-19：ResNeXt V100 Direct Formal Stability Check（decision fusion + 512x16 long-budget winner replay）

> **独立 campaign 说明**
> - 这一轮继续承接 `decision fusion` side campaign，只推进 `resnext`，不回到 canonical ResUNet 主线，也不回退到旧 `feature fusion` lane。
> - 保持强制几何 pivot 不变：`image_size=512`, `num_slices_per_view=16`, `trim_edge_slices=2`, `share_backbone=false`, `use_attention_pooling=false`, `batch_size=2`，并继续固定 `train.epochs=30`。
> - 唯一离散研究改动：停止继续做 winner-centered fresh retune，改为把上一轮 long-budget main-study winner（`freeze_layers=3`, `lr=3e-5`, `weight_decay=1e-3`, `dropout=0.4`, `gradient_clip_norm=1.5`）直接以单卡 fixed-config formal 语义重跑 1 次，判断 `c551967` 的高点是否稳定可复现。
> - 本轮运行 commit 为 `96fe172`；由于这是 fixed-config confirmation 而不是 fresh study，直接运行 `CUDA_VISIBLE_DEVICES=0 timeout 10800 .venv/bin/python train.py --config configs/autoresearch_formal_resnext_decision_v100.yaml`。

- [x] **CMP-FAIR-V100-RESNEXT-DECISION-FORMAL-512X16-STABILITY-CHECK**：`configs/autoresearch_formal_resnext_decision_v100.yaml`（保持 `512x16 / 30`-epoch 配方不变，仅复验上一轮 long-budget winner：`freeze_layers=3`, `lr=3e-5`, `weight_decay=1e-3`, `dropout=0.4`, `gradient_clip_norm=1.5`）→ `val_acc=0.8404255319148937`, `val_auc=0.9354545454545455`, `val_f1=0.8351648351648352`, `peak_vram≈4.67 GiB`, `total_seconds=4247.1` → **discard**（这次 direct formal replay 不仅没有复现 `c551967` 的 `0.8829787234042553 / 0.9409090909090909`，accuracy 还低了 `0.0425531914893616`、AUC 低了 `0.0054545454545454`；相较当前 decision-fusion direct-formal anchor `7ae19a0` 的 `0.925531914893617 / 0.9640909090909091` 更低 `0.0851063829787233 / 0.0286363636363636`，因此不能保留。）
- **Stability takeaway**：这次 direct formal 的结果几乎与最初 `512x16` geometry pivot run `ef62f61` 的 best（`0.8404255319148937 / 0.9354545454545454`）完全重合，也比上一轮 narrow retune 中的 exact winner replay（`481ef67` 的 trial 0：`0.8617021276595744 / 0.9400000000000001`）更低。两次 replay 都明显低于 `c551967` 的 study winner，已经足够说明这个角点当前不具备稳定可复现性。
- **本轮结论**：本轮已经完成 backlog 中建议的 fixed-config stability check，结论是：`c551967` 更像一次高方差 spike，而不是可靠的新基线。后续不应继续在同一 `lr / wd / dropout / clip` 角点上重复 replay。
- **推荐动作**：如果外层 loop 继续推进这条 isolated `512x16` decision lane，下一步应保持几何与 `30`-epoch budget 不变，并按顶部人类约束切到**单点结构创新**，不要再做纯标量 replay。最自然的首个结构方向是给当前 `ResNeXt decision` 路径加入一个轻量 gated head / per-view recalibration 模块，先尝试降低 per-view logit 的方差，再用 1 次新的 adaptive main-study 或更便宜的诊断评估它是否能把这条 lane 从当前不稳定峰值中拉出来。

## 2026-04-19：ResNeXt V100 Dedicated Main-Study（decision fusion + 512x16 winner-centered local retune）

> **独立 campaign 说明**
> - 这一轮继续承接 `decision fusion` side campaign，只推进 `resnext`，不回到 canonical ResUNet 主线，也不回退到旧 `feature fusion` lane。
> - 保持强制几何 pivot 不变：`image_size=512`, `num_slices_per_view=16`, `trim_edge_slices=2`, `share_backbone=false`, `use_attention_pooling=false`, `batch_size=2`，并继续固定 `train.epochs=30`。
> - 唯一离散研究改动：把 dedicated formal template 升格到上一轮 long-budget winner（`freeze_layers=3`, `lr=3e-5`, `weight_decay=1e-3`, `dropout=0.4`, `gradient_clip_norm=1.5`），然后把 fresh adaptive main-study 的搜索空间收窄到这个新 anchor 周围，只测试局部的 `lr / weight_decay / dropout / clip` 扰动。
> - 本轮使用 search-config copy `autoresearch_logs/generated_search_configs/optuna_main_search_iter_0004_20260418_231408.yaml`，fresh study_root `runs/optuna_main_autoloop/iter_0004_20260418_231408`，运行 commit 为 `481ef67`，并在空闲 `GPU 0,1,2` 上完成 1 次 adaptive main-study。

- [x] **CMP-FAIR-V100-RESNEXT-DECISION-MAIN-512X16-WINNER-LOCAL-RETUNE**：`configs/autoresearch_formal_resnext_decision_v100.yaml`（对齐上一轮 `512x16 / 30`-epoch winner）+ fresh adaptive main-study（`GPU 0,1,2`）→ 4/4 trials completed，best completed trial 为 **trial 1**（`freeze_layers=3`, `lr=3e-5`, `weight_decay=7.5e-4`, `dropout=0.4`, `gradient_clip_norm=2.0`）→ `val_acc=0.8723404255319149`, `val_auc=0.9518181818181819`, `val_f1=0.85`, `peak_vram≈4.67 GiB`, `total_seconds=4275.1` → **discard**（这轮 local retune 没有超过上一轮 long-budget winner `c551967` 的 `0.8829787234042553 / 0.9409090909090909`：虽然 AUC 提升了 `0.0109090909090910`，但 accuracy 反而回落 `0.0106382978723404`；它也仍明显低于当前 decision-fusion direct-formal anchor `7ae19a0` 的 `0.925531914893617 / 0.9640909090909091`，以及旧 ResNeXt feature-fusion retained keep `15b1ef6` 的 `0.9468085106382979 / 0.9800000000000001`，因此不能保留。）
- **Monitor takeaways**：这轮 4 个 completed trials 的 `peak_vram` 依旧稳定在约 `4.67 GiB`，说明结论仍然是优化面而不是资源问题。最关键的新信号不是“局部 retune 找到更强角点”，而是 **exact winner replay 本身没有复现**：trial 0 作为 enqueued anchor，参数与上一轮 best 完全一致（`lr=3e-5`, `weight_decay=1e-3`, `dropout=0.4`, `clip=1.5`），却只跑到 `val_acc=0.8617021276595744`, `val_auc=0.9400000000000001`。在此基础上，把 `clip` 提到 `2.0` 并把 `weight_decay` 降到 `7.5e-4` 只能把 accuracy 拉回 `0.8723404255319149`，仍没追上上一轮 best。
- **本轮结论**：这次实验完成了原计划的“围绕新 anchor 做 1 次窄 retune”，但结果表明当前问题已经不再是“还没摸到更好的局部角点”，而是“上一轮高点是否稳定”。既然 exact replay 都明显回落，再继续在同一小网格里扫 `lr / wd / dropout / clip` 的信息增益已经很低，因此本轮仍记 **discard**。
- **推荐动作**：如果外层 loop 还要继续这条 isolated decision-fusion lane，下一步不要再重复 fresh local retune。应保持 `512x16` 和 `30`-epoch budget 不变，先做 **1 次 fixed-config direct formal confirmation / stability check**，把上一轮 long-budget winner（`lr=3e-5`, `weight_decay=1e-3`, `dropout=0.4`, `gradient_clip_norm=1.5`）在单卡固定语义下再跑 1 次，先确认 `0.8829787234042553` 是否可复现，再决定后续是否继续投入这一角点。

---

## 2026-04-18：ResNeXt V100 Dedicated Main-Study（decision fusion + 512x16 budget diagnostic to 30 epochs）

> **独立 campaign 说明**
> - 这一轮继续承接 `decision fusion` side campaign，只推进 `resnext`，不回到 canonical ResUNet 主线，也不回退到旧 `feature fusion` lane。
> - 保持强制几何 pivot 不变：`image_size=512`, `num_slices_per_view=16`, `trim_edge_slices=2`, `share_backbone=false`, `use_attention_pooling=false`, `batch_size=2`。
> - 唯一离散研究改动：把 dedicated formal template 的训练预算从 `15` 提到 `30` epochs，用更长 budget 诊断 `512x16` lane 是预算受限，还是几何/优化面本身失配；局部搜索空间保持上一轮 `512x16` retune 的 low-lr / high-regularization 方向，不再引入新的结构变量。
> - 本轮使用 search-config copy `autoresearch_logs/generated_search_configs/optuna_main_search_iter_0003_20260418_204323.yaml`，fresh study_root `runs/optuna_main_autoloop/iter_0003_20260418_204323`，运行 commit 为 `c551967`，并在空闲 `GPU 0,1,2` 上完成 1 次 adaptive main-study。

- [x] **CMP-FAIR-V100-RESNEXT-DECISION-MAIN-512X16-BUDGET30**：`configs/autoresearch_formal_resnext_decision_v100.yaml`（仅把 `train.epochs` 从 `15` 扩到 `30`）+ fresh adaptive main-study（`GPU 0,1,2`）→ 4/4 trials completed，best completed trial 为 **trial 2**（`freeze_layers=3`, `lr=3e-5`, `weight_decay=1e-3`, `dropout=0.4`, `gradient_clip_norm=1.5`）→ `val_acc=0.8829787234042553`, `val_auc=0.9409090909090909`, `val_f1=0.8641975308641975`, `peak_vram≈4.67 GiB`, `total_seconds=4271.2` → **discard**（虽然这轮长预算把 `512x16` lane 明显抬高了：相较上一轮 `512x16` local-retune best `0.851063829787234 / 0.9313636363636364` 提升 `+0.0319148936170213 / +0.0095454545454545`，并且追平了当前 canonical proxy best `a60c3e0` 的 accuracy、同时在 AUC 上更高；但它仍低于当前 decision-fusion direct-formal anchor `7ae19a0` 的 `0.925531914893617 / 0.9640909090909091`，accuracy 低 `0.0425531914893617`、AUC 低 `0.0231818181818182`；相较旧 ResNeXt feature-fusion retained keep `15b1ef6` 的 `0.9468085106382979 / 0.9800000000000001` 仍低 `0.0638297872340426 / 0.0390909090909092`，因此仍不能保留。）
- **Monitor takeaways**：这轮 30-epoch budget 不是资源问题，4 个 completed trials 的 `peak_vram` 都稳定在约 `4.67 GiB`，单 trial wall time 约 `4249-4276s`。最关键的是，winner 从原先的低正则 anchor（`lr=5e-5`, `weight_decay=5e-4`, `dropout=0.3`, `clip=1.0`）切到了**更低 lr + 更高 dropout / wd / clip** 的组合；monitor 也明确给出同方向信号：`dropout`、`gradient_clip_norm`、`weight_decay` 越高越好，`lr` 越低越好。
- **Budget diagnosis takeaway**：虽然总体 best 明显提升，但 **winner 的 best epoch 仍然是 `epoch 13`**，不是更晚的新峰值。这说明“把预算从 15 拉到 30”帮助了这条 lane 重新排序超参优先级，却没有证明模型需要更长训练才能在后半程冒出更高峰值。
- **本轮结论**：`512x16` lane 确实对 budget 敏感，不能再把 15-epoch 结果当作这条几何的最终上限；但当前 gain 更像是“长预算改变了更强正则角点的可见性”，而不是“模型在 20-30 epoch 自己长出了新峰值”。因此本轮仍记 **discard**，但后续研究变量应从“继续加 budget”切换到“在 30-epoch 预算下围绕新角点做局部 retune”。
- **推荐动作**：如果外层 loop 还要继续这条 isolated decision-fusion lane，下一步不要再继续加 epoch，也不要回退到 15 epochs。应保持 `512x16` 和 `30`-epoch budget 不变，把本轮 winner（`lr=3e-5`, `weight_decay=1e-3`, `dropout=0.4`, `gradient_clip_norm=1.5`）升格为新的 isolated anchor，再做 1 次更窄的 local retune。

---

## 2026-04-18：人类方向追加约束（decision lane geometry pivot）

> **方向说明**
> - 当前 coordinator 的人类高优先级约束已进一步明确：isolated `ResNeXt V100 + decision fusion` lane 后续必须转到 **`image_size=512` + `num_slices_per_view=16`**。
> - 这个 pivot 不是可选探索项，而是下一轮及后续迭代都应遵守的新几何前提。
> - 同时保持：`backbone=resnext`, `fusion_type=decision`, `share_backbone=false`, `use_attention_pooling=false`, `trim_edge_slices=2`。
> - 如果 `512x16` 带来显存或吞吐压力，只允许做安全调整（如降低 `batch_size`、增加 timeout、必要时改走更小预算诊断 lane），**不要**回退到 `256x8`。

---

## 2026-04-18：ResNeXt V100 Dedicated Main-Study（decision fusion + 512x16 local scalar retune）

> **独立 campaign 说明**
> - 这一轮继续承接 `decision fusion` side campaign，只推进 `resnext`，不回到 canonical ResUNet 主线，也不回退到旧 `feature fusion` lane。
> - 保持强制几何 pivot 不变：`image_size=512`, `num_slices_per_view=16`, `trim_edge_slices=2`, `share_backbone=false`, `use_attention_pooling=false`, `batch_size=2`。
> - 唯一离散研究改动：把 dedicated formal template 锚定到上一轮 `512x16` least-bad anchor（`freeze_layers=3`, `lr=5e-5`, `weight_decay=5e-4`, `dropout=0.3`, `gradient_clip_norm=1.0`），并把 fresh adaptive main-study 的搜索空间收窄到同一局部区域，只测试更保守的 `lr / weight_decay / dropout / clip` 标量扰动。
> - 本轮使用 search-config copy `autoresearch_logs/generated_search_configs/optuna_main_search_iter_0002_20260418_192324.yaml`，fresh study_root `runs/optuna_main_autoloop/iter_0002_20260418_192324`，运行 commit 为 `7168b10`，并在空闲 `GPU 0,1,2` 上完成 1 次 adaptive main-study。

- [x] **CMP-FAIR-V100-RESNEXT-DECISION-MAIN-512X16-LOCAL-RETUNE**：`configs/autoresearch_formal_resnext_decision_v100.yaml`（对齐上一轮 `512x16` least-bad anchor）+ fresh adaptive main-study（`GPU 0,1,2`）→ 4/4 trials completed，best completed trial 为 **trial 0**，也就是 enqueued anchor 本身（`freeze_layers=3`, `lr=5e-5`, `weight_decay=5e-4`, `dropout=0.3`, `gradient_clip_norm=1.0`）→ `val_acc=0.851063829787234`, `val_auc=0.9313636363636364`, `val_f1=0.8333333333333334`, `peak_vram≈4.67 GiB`, `total_seconds=2155.6` → **discard**（虽然这比上一轮 `512x16` fresh study best `0.8404255319148937 / 0.9354545454545454` 多了 `+0.0106382978723403` accuracy，但 AUC 还回落了 `0.0040909090909090`；同时它仍明显弱于当前 decision-fusion direct-formal anchor `7ae19a0` 的 `0.925531914893617 / 0.9640909090909091`，也仍明显弱于旧 ResNeXt feature-fusion retained keep `15b1ef6` 的 `0.9468085106382979 / 0.9800000000000001`，因此不能保留。）
- **Monitor takeaways**：这轮 narrower retune 没有找到比 enqueued anchor 更好的角点。把 `dropout` 提到 `0.35`、`gradient_clip_norm` 抬回 `1.5` 会把 accuracy 直接压到 `0.7978723404255319`；把正则进一步推到 `dropout=0.4`, `lr=3e-5`, `weight_decay=1e-3`, `clip=1.5` 同样只得到 `0.7978723404255319`；即便只把 `weight_decay` 从 `5e-4` 提到 `7.5e-4`，accuracy 也会回落到 `0.8297872340425532`。4 个 completed trials 的 `peak_vram` 全都稳定在约 `4.67 GiB`，说明本轮结论是优化面的负结果，不是资源噪声。
- **本轮结论**：`512x16` lane 的当前最优角点仍停留在上一轮 already-known anchor，说明单纯继续做局部标量窄 retune 并不能把这个大几何 lane 拉回到 decision anchor 或旧 feature-fusion keep 的水平。因此本轮仍记 **discard**。
- **推荐动作**：如果外层 loop 还要继续这条 isolated decision-fusion lane，下一步不要再在同一窄区间里重复扫 `lr / dropout / clip / wd`。应继续保持 `512x16` 不回退，但把研究变量切到 **训练预算诊断**（例如更长 epochs 或同 budget 下的 budget-sensitive check），先判断这是几何本身失配，还是当前 15-epoch 预算不足。

---

## 2026-04-18：ResNeXt V100 Dedicated Main-Study（decision fusion + 512x16 geometry pivot）

> **独立 campaign 说明**
> - 这一轮继续承接 `decision fusion` side campaign，只推进 `resnext`，不回到 canonical ResUNet 主线，也不回退到旧 `feature fusion` lane。
> - 唯一研究改动是执行新的强制几何 pivot：把 dedicated decision lane 从 `image_size=256` / `num_slices_per_view=8` 切到 **`image_size=512` / `num_slices_per_view=16`**，同时保持 `trim_edge_slices=2`, `share_backbone=false`, `use_attention_pooling=false` 不变。
> - 为避免把资源问题误判成研究结论，本轮只做安全收紧：`batch_size=2`，并把 Optuna `main` / `proxy` timeout 分别扩到 `480/180` 与 `240/90` 分钟；这些不视为额外研究变量。
> - 本轮使用 search-config copy `autoresearch_logs/generated_search_configs/optuna_main_search_iter_0001_20260418_180333.yaml`，fresh study_root `runs/optuna_main_autoloop/iter_0001_20260418_180333`，运行 commit 为 `ef62f61`，并在空闲 `GPU 0,1,2` 上完成 1 次 adaptive main-study。

- [x] **CMP-FAIR-V100-RESNEXT-DECISION-MAIN-512X16**：`configs/autoresearch_formal_resnext_decision_v100.yaml`（几何 pivot 到 `512x16`，`batch_size=2`）+ fresh adaptive main-study（`GPU 0,1,2`）→ 4/4 trials completed，best completed trial 为 **trial 1**（`freeze_layers=3`, `lr=5e-5`, `weight_decay=5e-4`, `dropout=0.3`, `gradient_clip_norm=1.0`）→ `val_acc=0.8404255319148937`, `val_auc=0.9354545454545454`, `val_f1=0.8235294117647058`, `peak_vram≈4.67 GiB`, `total_seconds=2178.3` → **discard**（虽然这次几何 pivot 在执行层面是成功的，但性能明显弱于当前 decision-fusion direct-formal anchor `7ae19a0` 的 `0.925531914893617 / 0.9640909090909091`，accuracy 低 `0.0851063829787233`、AUC 低 `0.0286363636363637`；相较旧 ResNeXt feature-fusion retained keep `15b1ef6` 的 `0.9468085106382979 / 0.9800000000000001` 更低 `0.1063829787234042 / 0.0445454545454547`，因此不能保留。）
- **Monitor takeaways**：4 个 completed trials 全部有效，没有 OOM、timeout 或 workflow crash。`freeze_layers=3` 的三次试次显存都稳定在约 `4.67 GiB`；唯一的 `freeze_layers=2` 试次也只到约 `15.48 GiB`，说明在 V100 32GB + `batch_size=2` 下，`512x16` 不是资源阻塞问题，而是当前标量/几何组合本身表现不佳。最不差的角点从旧 anchor 的 `lr=1e-4, dropout=0.25, clip=2.0` 转到更保守的 `lr=5e-5, dropout=0.3, clip=1.0`，提示大几何下的优化面已经明显变化。
- **本轮结论**：`512x16` pivot 已被验证为**可稳定执行**，但现阶段不是性能收益；在当前 search budget 下，它反而把 isolated decision lane 的 accuracy 明显压低，所以本轮记 **discard**。
- **推荐动作**：如果外层 loop 还要继续这条 isolated decision-fusion lane，必须保持 `512x16` 不回退，但不要对这次 winner 直接做 formal confirmation。优先围绕 trial-1 角点做 1 次更窄的 retune，必要时用小预算诊断先判断大几何到底需要更强正则、不同学习率，还是更长训练预算。

---

## 2026-04-18：人类方向改动（ResNeXt 转到 Decision Fusion）

> **方向说明**
> - 人类已明确要求：独立 `ResNeXt V100` side campaign 的下一步不再继续当前 `feature-fusion` lane，而是改做 **`resnext + decision fusion`**。
> - 这不是对既有 `feature-fusion` 结果的否定；`15b1ef6` 及其 direct formal confirmation 仍保留为该旧 lane 的最强证据。
> - 为避免混淆历史 ledger、输出目录和搜索记录，新的 decision-fusion 方向必须使用**隔离的配置 / study_root / output_dir**，不能覆盖 `configs/autoresearch_formal_resnext_v100.yaml` 或 `runs/optuna_main_resnext_v100`。
> - 新 lane 的首轮动作应优先保持几何与数据预算不变：`image_size=256`, `num_slices_per_view=8`, `trim_edge_slices=2`, `share_backbone=false`, `use_attention_pooling=false`, `batch_size=6`, `num_workers=12`, 只把 `fusion_type` 切到 `decision`，然后再做 fresh adaptive main-study。

---

## 2026-04-18：ResNeXt V100 Dedicated Main-Study（decision fusion，资源阻塞）

> **独立 campaign 说明**
> - 这一轮显式承接上面的 2026-04-18 人类方向改动，只推进 `resnext`，不回到 canonical ResUNet 主线，也不再继续旧 `feature fusion` 多 seed confirmation。
> - 保持原 ResNeXt V100 side campaign 的几何与预算不变：`image_size=256`, `num_slices_per_view=8`, `trim_edge_slices=2`, `share_backbone=false`, `use_attention_pooling=false`, `batch_size=6`, `num_workers=12`。
> - 唯一离散研究改动：把隔离模板从 `feature fusion` 切到 `decision fusion`，使用 `configs/autoresearch_formal_resnext_decision_v100.yaml` / `configs/optuna_main_search_resnext_decision_v100.yaml` 做 1 次 fresh adaptive main-study。
> - 首跑使用 search-config copy `autoresearch_logs/generated_search_configs/optuna_main_search_iter_0001_20260418_151554.yaml` 与 study_root `runs/optuna_main_autoloop/iter_0001_20260418_151554`，运行 commit 为 `036a6d7`。首跑暴露出 workflow bug：显式 `--gpu-ids 4,5,0,1` 未执行 idle 阈值检查，导致 leader 直接在 busy GPU 上起跑。
> - 为按协议完成“code bug 最多修 1 次再重跑”，本轮随后在 commit `13696b9` 修复 `scripts/optuna_workflow.py`，使显式 GPU 列表也强制 obey `used<=1024 MiB` / `util<=20%`，并改用 search-config copy `autoresearch_logs/generated_search_configs/optuna_main_search_iter_0001_20260418_152350.yaml` 与新的 fresh study_root `runs/optuna_main_autoloop/iter_0001_20260418_152350` 重跑。

- [x] **CMP-FAIR-V100-RESNEXT-DECISION-MAIN-ITER0001**：`configs/autoresearch_formal_resnext_decision_v100.yaml` + fresh adaptive main-study（显式卡集 `4,5,0,1`）→ **crash**（首跑里 `trial 0/1` 在约 `30.8s` 时 OOM，日志显示目标卡只剩约 `163 MiB` 可用；`trial 2/3` 还被遗留为 `running`，说明这不是有效的研究结果。修复显式 GPU idle 检查后，重跑在创建 storage 前就被干净拒绝：leader job 直接报 `Explicit --gpu-ids entries do not satisfy the configured idle thresholds`，其中 `GPU 4/5` 占用约 `78.6 / 81.6 GiB` 且 util `96% / 100%`。现场 `nvidia-smi` 还显示 `GPU 0/1` 也同样远高于阈值，因此本轮没有任何 completed trial，也没有可比较的 `val_acc / val_auc`。）
- **本轮结论**：这次属于 **资源阻塞 / 执行层 crash**，不是对 `resnext + decision fusion` 研究假设的负面证据。当前只能说明：在本轮指定的显式卡集与 Slurm wrapper 环境下，资源当时并不满足 main-study 的 idle policy。
- **推荐动作**：保持同一 isolated decision-fusion lane，不要更换模板、不必回退到 proxy。下一次只需在 `4,5,0,1` 真正空闲时重开同一 fresh adaptive main-study；若资源长期不满足，再由外层 loop 或人类层面调整 GPU policy。

---

## 2026-04-18：ResNeXt V100 Dedicated Main-Study（decision fusion + dropout0.3 template）

> **独立 campaign 说明**
> - 这一轮继续承接上面的 `decision fusion` side campaign，只推进 `resnext`，不回到 canonical ResUNet 主线，也不回退到旧 `feature fusion` lane。
> - 保持原 ResNeXt V100 side campaign 的几何与预算不变：`image_size=256`, `num_slices_per_view=8`, `trim_edge_slices=2`, `share_backbone=false`, `use_attention_pooling=false`, `batch_size=6`, `num_workers=12`。
> - 唯一离散研究改动：把 dedicated formal decision template `configs/autoresearch_formal_resnext_decision_v100.yaml` 的 `dropout` 从 `0.25` 上调到 `0.3`，然后用 search-config copy `autoresearch_logs/generated_search_configs/optuna_main_search_iter_0001_20260418_170826.yaml` 在空闲 `GPU 0,1,2` 上做 1 次 fresh adaptive main-study。
> - 本轮 fresh study 为 `runs/optuna_main_autoloop/iter_0001_20260418_170826`，运行 commit 为 `570f082`，4/4 trials completed。

- [x] **CMP-FAIR-V100-RESNEXT-DECISION-MAIN-DROPOUT030**：`configs/autoresearch_formal_resnext_decision_v100.yaml`（仅把 template `dropout` 调到 `0.3`）+ fresh adaptive main-study（`GPU 0,1,2`）→ best completed trial 为 **trial 3**（`freeze_layers=3`, `lr=1e-4`, `weight_decay=5e-4`, `dropout=0.25`, `gradient_clip_norm=2.0`）→ `val_acc=0.9148936170212766`, `val_auc=0.9763636363636363`, `val_f1=0.9111111111111111`, `peak_vram≈2.17 GiB`, `total_seconds=803.3` → **discard**（这次 fresh main-study 已经干净完成，证明上一轮 crash 只是资源问题；但本轮 winner 仍低于旧 ResNeXt feature-fusion retained keep `15b1ef6` 的 `0.9468085106382979 / 0.9800000000000001`，accuracy 低 `0.0319148936170213`、AUC 低 `0.0036363636363638`，因此还不足以支持把 side campaign 的融合语义正式切到 decision fusion。）
- **Monitor takeaways**：4 个 completed trial 都是有效结果，没有再出现资源/流程层异常。`freeze_layers=3` 依然统治这个 lane；唯一的 `freeze_layers=2` 试次（trial 2）把 `peak_vram` 推到约 `6.27 GiB`，但 accuracy 只到 `0.9042553191489362`。本轮的 template 改动（`dropout=0.3`）本身能把 enqueued trial 0 推到 `0.9148936170212766 / 0.9731818181818181`，但搜索最终还是回到了更低 `dropout=0.25` 且更高 `gradient_clip_norm=2.0` 的组合，说明 decision-fusion lane 的最优正则形状和旧 feature-fusion 稳定家族并不完全相同。
- **本轮结论**：`resnext + decision fusion` 已经拿到了第一批干净的 completed-trial 证据，不再是“只有 crash 没有结果”的方向；但按当前证据，它仍明显落后于既有的 ResNeXt feature-fusion retained keep，因此这轮只能记 **discard**。
- **推荐动作**：如果外层 loop 还要继续这条 isolated decision-fusion lane，优先只做 1 次 direct formal confirmation，验证本轮 trial-3 winner 是否能在 formal 语义下进一步接近或超过旧 feature-fusion keep；在此之前不要再开新的 fresh search。

---

## 2026-04-18：ResNeXt V100 Direct Formal Confirmation（decision-fusion main-study winner）

> **独立 campaign 说明**
> - 这一轮继续承接上面的 `decision fusion` side campaign，只推进 `resnext`，不回到 canonical ResUNet 主线，也不回退到旧 `feature fusion` lane。
> - 保持原 ResNeXt V100 side campaign 的几何与预算不变：`image_size=256`, `num_slices_per_view=8`, `trim_edge_slices=2`, `share_backbone=false`, `use_attention_pooling=false`, `batch_size=6`, `num_workers=12`。
> - 唯一离散研究改动：把 dedicated formal decision template 直接对齐到上一轮 fresh main-study winner（`freeze_layers=3`, `lr=1e-4`, `weight_decay=5e-4`, `dropout=0.25`, `gradient_clip_norm=2.0`），然后在单卡 fallback `GPU 0` 上做 1 次 direct formal confirmation。
> - 本轮运行 commit 为 `7ae19a0`；由于这是 fixed-config confirmation 而不是 fresh study，直接运行 `CUDA_VISIBLE_DEVICES=0 timeout 10800 .venv/bin/python train.py --config configs/autoresearch_formal_resnext_decision_v100.yaml`。

- [x] **CMP-FAIR-V100-RESNEXT-DECISION-FORMAL-CONFIRM**：`configs/autoresearch_formal_resnext_decision_v100.yaml`（对齐上一轮 decision-fusion main-study winner：`freeze_layers=3`, `lr=1e-4`, `weight_decay=5e-4`, `dropout=0.25`, `gradient_clip_norm=2.0`）→ `val_acc=0.925531914893617`, `val_auc=0.9640909090909091`, `val_f1=0.9195402298850575`, `peak_vram≈2.17 GiB`, `total_seconds=797.4` → **discard**（相较上一轮 decision-fusion main-study winner `0.9148936170212766 / 0.9763636363636363`，accuracy 提升 `0.0106382978723404`，说明 search winner 能转化为更强 formal accuracy 证据；但 AUC 回落 `0.0122727272727272`，且相较旧 ResNeXt feature-fusion retained keep `15b1ef6` 的 `0.9468085106382979 / 0.9800000000000001` 仍低 `0.0212765957446809 / 0.0159090909090909`，因此仍不足以把 side campaign 的最强 retained evidence 切换到 decision fusion。）
- **本轮结论**：这次 direct formal confirmation 证明 decision-fusion lane 已经不只是“search 里偶然跑到一个较好 trial”；它在 formal 语义下把 accuracy 进一步抬到 `0.925531914893617`。但在当前 keep/discard 口径下，这个结果仍没有超过既有的 ResNeXt feature-fusion retained keep，所以本轮依旧只能记 **discard**。
- **推荐动作**：如果外层 loop 还要继续这条 isolated decision-fusion lane，下一步不再重复 direct formal 或 broad fresh search；优先把这次 confirmed anchor 作为新的 isolated baseline，只围绕它开 1 次更窄的 fresh adaptive main-study，测试 decision-fusion 是否还能再向 `15b1ef6` 收敛。

---

## 2026-04-16：本地 2070S 基础 Backbone Quick Compare

> **独立 campaign 说明**
> - 这一轮是 **本地兼容性 + backbone 快筛**，不与当前 canonical `val_acc` 主线（`a60c3e0` / `c30fcae`）直接比较。
> - 当前本地环境于 **2026-04-16** 实测为：Windows `.venv\\Scripts\\python.exe` + 单卡 `NVIDIA GeForce RTX 2070 SUPER (8192 MiB)`。
> - 原始 `data/realdata/metadata.csv` 在本地存在 `CTdata2/` 路径漂移；本轮通过 `scripts/prepare_local_metadata.py` 生成 `autoresearch_logs/local_metadata_fixed.csv` 供训练使用，**未修改原始数据文件与 split**。
> - 当日为适配 8GB 显存与本地路径现状，`configs/autoresearch_proxy.yaml` / `configs/autoresearch_formal.yaml` 曾同步切到 `autoresearch_logs/local_metadata_fixed.csv`，并临时收紧到：`image_size=256`, `num_slices_per_view=8`, `batch_size=2`, `num_workers=0`。
> - backbone 对比使用统一 quick-compare 配方：`feature fusion`, `share_backbone=false`, `freeze_layers=3`, `use_pretrained=true`, `dropout=0.3`, `lr=5e-5`, `weight_decay=1e-4`, `epochs=1`。

- [x] **CMP-LOCAL8G-RESUNET**：`configs/cmp_local8g_feature_resunet.yaml` → `val_acc=0.8085106382978723`, `val_auc=0.915`, `val_f1=0.7857142857142857`, `peak_vram=1.23 GiB`, `total_seconds=443.9` → **keep**（本轮四个 backbone 中 accuracy / AUC 均最佳，且显存余量最充足，是当前 2070S 路线的首选 backbone）
- [x] **CMP-LOCAL8G-RESNEXT**：`configs/cmp_local8g_feature_resnext.yaml` → `val_acc=0.46808510638297873`, `val_auc=0.7272727272727272`, `val_f1=0.6376811594202898`, `peak_vram=1.32 GiB`, `total_seconds=607.6` → **discard**（准确率最低，且训练时长最高；在当前预算下没有继续投入价值）
- [x] **CMP-LOCAL8G-SENET**：`configs/cmp_local8g_feature_senet.yaml` → `val_acc=0.5319148936170213`, `val_auc=0.7781818181818181`, `val_f1=0.0`, `peak_vram=1.46 GiB`, `total_seconds=493.5` → **discard**（比 `resnext` 稍好，但模型几乎退化为单类预测，仍明显弱于 `resunet / cspnet`）
- [x] **CMP-LOCAL8G-CSPNET**：`configs/cmp_local8g_feature_cspnet.yaml` → `val_acc=0.7553191489361702`, `val_auc=0.8818181818181818`, `val_f1=0.6567164179104478`, `peak_vram=5.61 GiB`, `total_seconds=475.0` → **discard**（是唯一接近 `resunet` 的备选，但 accuracy 仍低 `0.0531914893617021`；运行中 `nvidia-smi` 一度接近 `7.7 / 8.0 GiB`，说明能跑但显存冗余不大）

- **本轮结论**：`resunet` > `cspnet` >>> `senet` > `resnext`
- **推荐动作**：若继续本地 8GB 路线，只保留 `resunet` 与 `cspnet` 做后续 `4-epoch proxy`；`senet / resnext` 直接停止

---

## 2026-04-17：V100 Fair Backbone Formal Compare

> **独立 campaign 说明**
> - 这一轮按 `docs/fair_backbone_compare.md` 的“equal protocol treatment”语义执行，但把运行平台升级到 `3 x Tesla V100-PCIE-32GB`。
> - 这仍是 **独立 backbone compare campaign**，不与当前 canonical `val_acc` 主线（`a60c3e0` / `c30fcae`）直接比较，也不用于覆盖 decision-fusion 主线最优纪录。
> - 四个 backbone 统一使用 matched formal recipe：`feature fusion`, `share_backbone=false`, `use_attention_pooling=false`, `freeze_layers=3`, `dropout=0.3`, `lr=1e-4`, `weight_decay=1e-4`, `epochs=15`, `image_size=256`, `num_slices_per_view=8`, `batch_size=6`, `num_workers=12`。
> - 运行 commit 为 `25f17f0`。三卡并发脚本成功完成前三个 backbone，但未自动派发最后一个 `cspnet`；该 run 随后在空闲 `GPU 0` 上手动补跑完成。

- [x] **CMP-FAIR-V100-FORMAL-RESUNET**：`configs/cmp_fair_v100_formal_resunet.yaml` → `val_acc=0.9148936170212766`, `val_auc=0.9222727272727274`, `val_f1=0.9047619047619048`, `peak_vram=2.97 GiB`, `total_seconds=870.2` → **discard**（与 `resnext` 同分最高 `val_acc`，但按 campaign tie-break 在 `val_auc` 上低 `0.018181818181818`；说明 `resunet` 在 equal-budget formal 下仍然非常强，但不再像本地 1-epoch quick screen 那样有绝对优势。）
- [x] **CMP-FAIR-V100-FORMAL-RESNEXT**：`configs/cmp_fair_v100_formal_resnext.yaml` → `val_acc=0.9148936170212766`, `val_auc=0.9404545454545454`, `val_f1=0.9090909090909091`, `peak_vram=2.14 GiB`, `total_seconds=888.3` → **keep**（本轮 campaign winner；与 `resunet` 持平最高 `val_acc`，并以更高 `val_auc` 获胜，同时显存也更低。说明在统一 15-epoch matched recipe 下，`resnext` 的上限被此前 quick screen 明显低估。）
- [x] **CMP-FAIR-V100-FORMAL-SENET**：`configs/cmp_fair_v100_formal_senet.yaml` → `val_acc=0.851063829787234`, `val_auc=0.9195454545454544`, `val_f1=0.8157894736842105`, `peak_vram=2.16 GiB`, `total_seconds=878.0` → **discard**（较 campaign winner `0.9148936170212766` 低 `0.0638297872340426`；虽然相比本地 quick screen 有显著回升，但仍明显落后于 `resnext / resunet`。）
- [x] **CMP-FAIR-V100-FORMAL-CSPNET**：`configs/cmp_fair_v100_formal_cspnet.yaml` → `val_acc=0.8829787234042553`, `val_auc=0.9213636363636363`, `val_f1=0.8705882352941177`, `peak_vram=1.87 GiB`, `total_seconds=793.7` → **discard**（本轮最强非冠军；较 campaign winner `0.9148936170212766` 低 `0.0319148936170213`。说明 `cspnet` 在 equal-budget formal 下比本地 8GB 快筛表现更有竞争力，但仍不足以挤进第一梯队。）

- **本轮 formal 排名**：`resnext` >= `resunet` > `cspnet` > `senet`
- **按规则的最终胜者**：`resnext`
- **推荐动作**：Stage 3 稳定性确认现已完成（见下节）；该独立 fair-backbone campaign 可正式收束为 `resnext` stable winner，`resunet` 降为次优对照保留结论

---

## 2026-04-17：V100 Fair Backbone Stage 3 Stability Confirmation

> **独立 campaign 说明**
> - 本节严格承接上面的 `CMP-FAIR-V100-FORMAL`，不修改 recipe，只补做 `seed=123 / 456`。
> - 两个 backbone 都沿用同一 matched formal recipe：`feature fusion`, `share_backbone=false`, `use_attention_pooling=false`, `freeze_layers=3`, `dropout=0.3`, `lr=1e-4`, `weight_decay=1e-4`, `epochs=15`, `image_size=256`, `num_slices_per_view=8`, `batch_size=6`, `num_workers=12`。
> - 本轮运行 commit 为 `2123941`。起跑顺序为三卡并发 `resunet_s123 / resnext_s123 / resunet_s456`，随后补跑 `resnext_s456`。

- [x] **CMP-FAIR-V100-STAGE3-RESUNET-S123**：`configs/cmp_fair_v100_formal_resunet_s123.yaml` → `val_acc=0.8829787234042553`, `val_auc=0.9363636363636363`, `val_f1=0.8607594936708861`, `peak_vram=2.97 GiB`, `total_seconds=859.8` → **discard**（较 `resnext_s123` 低 `0.0319148936170213` accuracy，AUC 也低 `0.0331818181818182`；说明 `resunet` 在 alternate seed 上比 seed-42 更易回落。）
- [x] **CMP-FAIR-V100-STAGE3-RESNEXT-S123**：`configs/cmp_fair_v100_formal_resnext_s123.yaml` → `val_acc=0.9148936170212766`, `val_auc=0.9695454545454545`, `val_f1=0.9047619047619048`, `peak_vram=2.15 GiB`, `total_seconds=880.6` → **keep**（与 seed-42 持平最高 accuracy，并把 AUC 再抬高 `0.0290909090909091`；强力支持 `resnext` 不是一次性赢家。）
- [x] **CMP-FAIR-V100-STAGE3-RESUNET-S456**：`configs/cmp_fair_v100_formal_resunet_s456.yaml` → `val_acc=0.9042553191489362`, `val_auc=0.9309090909090909`, `val_f1=0.8860759493670886`, `peak_vram=2.97 GiB`, `total_seconds=862.6` → **discard**（较 `seed=42` 回落 `0.0106382978723404` accuracy；虽然恢复到竞争区，但与 `resnext_s456` 持平 accuracy 时仍输 AUC。）
- [x] **CMP-FAIR-V100-STAGE3-RESNEXT-S456**：`configs/cmp_fair_v100_formal_resnext_s456.yaml` → `val_acc=0.9042553191489362`, `val_auc=0.9413636363636364`, `val_f1=0.891566265060241`, `peak_vram=2.15 GiB`, `total_seconds=809.6` → **keep**（开局极慢热，前 2 个 epoch 一度掉到 `0.4681`，但最终仍把 best 拉回与 `resunet_s456` 持平的 accuracy，并再赢 `0.0104545454545455` AUC；说明 `resnext` 的优化轨迹更抖，但最终上限依旧稳住。）

- **按 seed 的 head-to-head 结果**
- `seed=42`：`resnext` 与 `resunet` 持平 `val_acc=0.9148936170212766`，AUC `0.9404545454545454 > 0.9222727272727274`
- `seed=123`：`resnext` 以 `0.9148936170212766 / 0.9695454545454545` 明显胜过 `resunet` 的 `0.8829787234042553 / 0.9363636363636363`
- `seed=456`：两者同为 `val_acc=0.9042553191489362`，但 `resnext` 仍以 `val_auc=0.9413636363636364` 胜过 `resunet` 的 `0.9309090909090909`

- **Stage 3 结论**：`resnext` 在 3 个 seed 上 accuracy 从不输给 `resunet`，AUC 则 3/3 全胜，因此该独立 `fair_backbone_compare` campaign 的稳定胜者正式定为 `resnext`
- **附带观察**：`resnext` 的训练曲线明显比 `resunet` 更慢热、更波动，尤其 `seed=456` 前半程多次掉到 `0.8` 以下；但以 `best_val_accuracy_then_auc` 作为规则时，它最终仍保持对 `resunet` 的系统性优势

---

## 2026-04-18：ResNeXt V100 Dedicated Main-Study（fusion_hidden_dim=384）

> **独立 campaign 说明**
> - 这一轮显式延续上面的 `fair_backbone_compare` stable-winner side campaign，只推进 `resnext`，不回到 canonical ResUNet 主线。
> - 保持 matched V100 几何不变：`image_size=256`, `num_slices_per_view=8`, `feature fusion`, `share_backbone=false`, `use_attention_pooling=false`（mean pooling）。
> - 唯一离散改动：把 dedicated formal template `configs/autoresearch_formal_resnext_v100.yaml` 的 `fusion_hidden_dim` 从 `256` 放宽到 `384`，其余由 fresh adaptive Optuna main study 做小预算标量调参。
> - 本轮 fresh study 为 `runs/optuna_main_autoloop/iter_0001_20260418_023423`，运行 commit 为 `992687c`，4/4 trials completed。

- [x] **CMP-FAIR-V100-RESNEXT-MAIN-HEAD384**：`configs/autoresearch_formal_resnext_v100.yaml`（`fusion_hidden_dim=384`）+ fresh adaptive main study → best completed trial 为 **trial 3**（`freeze_layers=3`, `lr=5e-5`, `weight_decay=5e-4`, `dropout=0.25`, `gradient_clip_norm=1.0`）→ `val_acc=0.8829787234042553`, `val_auc=0.9431818181818181`, `val_f1=0.8607594936708861`, `peak_vram=2.15 GiB`, `total_seconds=859.1` → **discard**（相对 widened-head study 内的 template trial 0 `0.8617021276595744 / 0.9663636363636363` 有 accuracy 提升 `+0.0212765957446809`，但仍较 fair-backbone stable winner `CMP-FAIR-V100-FORMAL-RESNEXT` 的 `0.9148936170212766` 低 `0.0319148936170213`；说明仅放宽 fusion bottleneck 并没有把 ResNeXt 的 formal ceiling 推到新高。）
- **Monitor takeaways**：在这 4 个 completed trial 里，`freeze_layers=3` 明显优于 `freeze_layers=2`；更低的 `lr=5e-5`、更高的 `weight_decay=5e-4`、更低的 `dropout=0.25` 共同对应最佳 accuracy。唯一的 `freeze_layers=2` 试次（trial 2）把 `peak_vram` 提到 `6.24 GiB`，但 accuracy 只追平 trial 1 且 AUC 更低，说明解冻更深层并未换来净收益。
- **本轮结论**：`fusion_hidden_dim=384` 不晋升；该独立 ResNeXt V100 side campaign 仍以原始 `256` 宽 head 的 matched formal winner（`val_acc=0.9148936170212766`, `val_auc=0.9404545454545454`）作为当前应保留的最好证据。
- **推荐动作**：若后续继续该独立 side campaign，应把本轮 main-study 暗示较优的标量组合迁回原始 `256` 宽 recipe 做单点验证，而不是继续加宽 head 或重新补 `resunet` 对照。

---

## 2026-04-18：ResNeXt V100 Dedicated Main-Study（回到 head256 + scalar retune）

> **独立 campaign 说明**
> - 这一轮继续承接 `fair_backbone_compare` stable-winner side campaign，只推进 `resnext`，不回到 canonical ResUNet 主线。
> - 保持 matched V100 几何不变：`image_size=256`, `num_slices_per_view=8`, `feature fusion`, `share_backbone=false`, `use_attention_pooling=false`（mean pooling）。
> - 唯一离散改动：把 dedicated formal template `configs/autoresearch_formal_resnext_v100.yaml` 从 widened-head 试验状态迁回原始 `fusion_hidden_dim=256`，并把 base template 标量锚定到上一轮 main-study 暗示较优的组合（`freeze_layers=3`, `lr=5e-5`, `weight_decay=5e-4`, `dropout=0.25`, `gradient_clip_norm=1.0`）。
> - 由于起跑时 `GPU 0` 不满足本轮 idle 阈值（`used≈2152 MiB`, `util≈27%`），本轮 fresh study 改走允许的单卡 fallback `GPU 2`；fresh study 为 `runs/optuna_main_autoloop/iter_0002_20260418_030934`，运行 commit 为 `5a1c113`，4/4 trials completed。

- [x] **CMP-FAIR-V100-RESNEXT-MAIN-HEAD256-RETUNE**：`configs/autoresearch_formal_resnext_v100.yaml`（`fusion_hidden_dim=256`，base template 迁入上一轮较优锚点）+ fresh adaptive main study → best completed trial 为 **trial 3**（`freeze_layers=2`, `lr=1.25e-4`, `weight_decay=1e-4`, `dropout=0.35`, `gradient_clip_norm=2.0`）→ `val_acc=0.9042553191489362`, `val_auc=0.9368181818181818`, `val_f1=0.891566265060241`, `peak_vram=6.23 GiB`, `total_seconds=827.8` → **discard**（较 widened-head study best `0.8829787234042553 / 0.9431818181818181` 回升 `+0.0212765957446809` accuracy，但仍较 fair-backbone stable winner `CMP-FAIR-V100-FORMAL-RESNEXT` 的 `0.9148936170212766 / 0.9404545454545454` 低 `0.0106382978723404` accuracy，且 AUC 也低 `0.0036363636363636`，因此不足以替换当前应保留证据。）
- **Monitor takeaways**：回到 `head=256` 后，搜索上界明显高于上一轮 widened-head study；本轮唯一突破 `0.9` accuracy 的是更激进的 `freeze_layers=2`, `lr=1.25e-4`, `weight_decay=1e-4`, `dropout=0.35`, `gradient_clip_norm=2.0` 角点。作为对照，迁回的锚点组合在 3 个 completed trial 里只落在 `0.8723404255319149 ~ 0.8829787234042553`，说明上一轮 `head384` monitor 所暗示的优选标量并不能直接把原始 `head256` recipe 拉回 stable-winner 水平。
- **附带观察**：trial 0 与 trial 2 被 Optuna 命中了同一组参数（`freeze_layers=3`, `lr=5e-5`, `weight_decay=5e-4`, `dropout=0.25`, `gradient_clip_norm=1.0`），但分别得到 `0.8723404255319149` 与 `0.8829787234042553` accuracy，提示当前 ResNeXt side campaign 仍存在不小的 run-to-run 波动。
- **本轮结论**：把模板迁回原始 `256` 宽 head 是正确方向，但目前找到的更优 retune 角点仍未超过既有 fair-backbone stable winner，因此本轮仍记 **discard**。
- **推荐动作**：若后续继续该独立 side campaign，优先对本轮 trial-3 角点做 1 次 fixed-config formal confirmation，判断 `0.9043` 是否可复现；若仍低于 `0.9149 / 0.9405`，则应收束该 side campaign，不再继续围绕 ResNeXt 标量做小步扫描。

---

## 2026-04-18：ResNeXt V100 Direct Formal Confirmation（head256 retune winner）

> **独立 campaign 说明**
> - 这一轮继续承接 `fair_backbone_compare` stable-winner side campaign，只推进 `resnext`，不回到 canonical ResUNet 主线。
> - 保持 matched V100 几何不变：`image_size=256`, `num_slices_per_view=8`, `feature fusion`, `share_backbone=false`, `use_attention_pooling=false`（mean pooling），`fusion_hidden_dim=256` 不变。
> - 唯一离散改动：把 dedicated formal template `configs/autoresearch_formal_resnext_v100.yaml` 对齐到上一轮 main-study 的 best completed trial（`freeze_layers=2`, `lr=1.25e-4`, `weight_decay=1e-4`, `dropout=0.35`, `gradient_clip_norm=2.0`），然后直接做 1 次 formal confirmation。
> - 本轮运行 commit 为 `072e601`；由于这是 fixed-config confirmation 而不是 fresh study，直接在允许的单卡 fallback `GPU 2` 上运行 `train.py --config configs/autoresearch_formal_resnext_v100.yaml`。

- [x] **CMP-FAIR-V100-RESNEXT-FORMAL-CONFIRM**：`configs/autoresearch_formal_resnext_v100.yaml`（`head256`, `freeze_layers=2`, `lr=1.25e-4`, `weight_decay=1e-4`, `dropout=0.35`, `gradient_clip_norm=2.0`）→ `val_acc=0.925531914893617`, `val_auc=0.9563636363636363`, `val_f1=0.9176470588235294`, `peak_vram=6.23 GiB`, `total_seconds=824.0` → **keep**（较上一轮 main-study winner 的 `0.9042553191489362 / 0.9368181818181818` 再提升 `+0.0212765957446808` accuracy 与 `+0.0195454545454545` AUC，也较此前 fair-backbone stable winner `CMP-FAIR-V100-FORMAL-RESNEXT` 的 `0.9148936170212766 / 0.9404545454545454` 提升 `+0.0106382978723404` accuracy 与 `+0.0159090909090909` AUC；说明这组更激进的解冻与优化标量能够稳定转化为更强的 direct formal 证据。）
- **本轮结论**：上一轮 main-study 的 trial-3 角点不是一次性搜索噪声，而是当前独立 ResNeXt V100 side campaign 的新 best keep；它现在取代旧的 fair-backbone stable winner，成为该 campaign 最强的 retained evidence。
- **推荐动作**：若后续继续该独立 side campaign，应先做 1 次 alternate-seed formal confirmation（优先 `seed=123` 或 `seed=456`）来检验稳定性，再决定是否围绕这一新 anchor 开 fresh adaptive main study；不要回到 `head384`，也不要重新补非 `resnext` backbone。

---

## 2026-04-18：ResNeXt V100 Alternate-Seed Formal Confirmation（seed123）

> **独立 campaign 说明**
> - 这一轮继续承接 `fair_backbone_compare` stable-winner side campaign，只推进 `resnext`，不回到 canonical ResUNet 主线。
> - 保持 matched V100 几何与 retuned winner 标量完全不变：`image_size=256`, `num_slices_per_view=8`, `feature fusion`, `share_backbone=false`, `use_attention_pooling=false`（mean pooling），`fusion_hidden_dim=256`, `freeze_layers=2`, `lr=1.25e-4`, `weight_decay=1e-4`, `dropout=0.35`, `gradient_clip_norm=2.0`。
> - 唯一离散改动：把 dedicated formal template `configs/autoresearch_formal_resnext_v100.yaml` 的 `seed` 从 `42` 改为 `123`，然后直接做 1 次 alternate-seed formal confirmation。
> - 本轮运行 commit 为 `e6f36d0`；由于这是 fixed-config confirmation 而不是 fresh study，直接在允许的单卡 fallback `GPU 2` 上运行 `train.py --config configs/autoresearch_formal_resnext_v100.yaml`。

- [x] **CMP-FAIR-V100-RESNEXT-FORMAL-SEED123**：`configs/autoresearch_formal_resnext_v100.yaml`（仅 `seed=123`，其余保持当前 retuned winner 不变）→ `val_acc=0.8829787234042553`, `val_auc=0.91`, `val_f1=0.8571428571428571`, `peak_vram=6.23 GiB`, `total_seconds=827.9` → **discard**（较当前 retained keep `CMP-FAIR-V100-RESNEXT-FORMAL-CONFIRM` 的 `0.925531914893617 / 0.9563636363636363` 回落 `0.0425531914893617 / 0.0463636363636363`，也较旧 fair-backbone stable winner `CMP-FAIR-V100-FORMAL-RESNEXT` 的 `0.9148936170212766 / 0.9404545454545454` 低 `0.0319148936170213 / 0.0304545454545454`；说明当前新 anchor 至少在 `seed=123` 上并不稳定，不能把上一轮 keep 直接视为已完成多 seed 证实的新配方。）
- **本轮结论**：上一轮 direct formal keep 仍然是该独立 ResNeXt V100 side campaign 的最强单点证据，但它现在应被视为存在明显 seed sensitivity 的 anchor，而不是已经通过 alternate-seed confirmation 的稳定 winner。
- **推荐动作**：若后续继续该独立 side campaign，应优先补 1 次 `seed=456` direct formal confirmation，先判断这次回落是 `seed=123` 特例还是更普遍的不稳定；在此之前不要围绕当前 anchor 再开 fresh adaptive main study。

---

## 2026-04-18：ResNeXt V100 Alternate-Seed Formal Confirmation（seed456）

> **独立 campaign 说明**
> - 这一轮继续承接 `fair_backbone_compare` stable-winner side campaign，只推进 `resnext`，不回到 canonical ResUNet 主线。
> - 保持 matched V100 几何与 retuned winner 标量完全不变：`image_size=256`, `num_slices_per_view=8`, `feature fusion`, `share_backbone=false`, `use_attention_pooling=false`（mean pooling），`fusion_hidden_dim=256`, `freeze_layers=2`, `lr=1.25e-4`, `weight_decay=1e-4`, `dropout=0.35`, `gradient_clip_norm=2.0`。
> - 唯一离散改动：把 dedicated formal template `configs/autoresearch_formal_resnext_v100.yaml` 的 `seed` 从 `42` 改为 `456`，然后直接做 1 次 alternate-seed direct formal confirmation。
> - 本轮运行 commit 为 `b2539f1`；由于这是 fixed-config confirmation 而不是 fresh study，直接在允许的单卡 fallback `GPU 2` 上运行 `train.py --config configs/autoresearch_formal_resnext_v100.yaml`。

- [x] **CMP-FAIR-V100-RESNEXT-FORMAL-SEED456**：`configs/autoresearch_formal_resnext_v100.yaml`（仅 `seed=456`，其余保持当前 retuned winner 不变）→ `val_acc=0.8829787234042553`, `val_auc=0.945`, `val_f1=0.8571428571428571`, `peak_vram=6.23 GiB`, `total_seconds=830.4` → **discard**（较当前 retained keep `CMP-FAIR-V100-RESNEXT-FORMAL-CONFIRM` 的 `0.925531914893617 / 0.9563636363636363` 仍低 `0.0425531914893617 / 0.0113636363636362`；虽然 `val_auc` 较 `seed123` confirmation 的 `0.91` 回升 `0.035`，也较旧 fair-backbone stable winner `CMP-FAIR-V100-FORMAL-RESNEXT` 的 `0.9404545454545454` 略高 `0.0045454545454547`，但 `val_acc` 仍比旧 stable winner 低 `0.0319148936170213`。说明当前 `freeze_layers=2` retuned anchor 的排序质量并非完全崩坏，但它在 alternate seeds 上仍稳定掉回 `0.8829787234042553` accuracy floor，无法复现 `seed=42` 的 direct formal keep。）
- **本轮结论**：两次 alternate-seed direct formal（`seed=123 / 456`）都未复现 `seed=42` 的 `0.925531914893617` accuracy，因此当前 retuned anchor 应被视为“最强单点 keep，但明显 seed-sensitive”的证据；相比之下，旧 fair-backbone stable winner 仍是更可靠的多 seed 参考配方。
- **推荐动作**：若后续继续该独立 side campaign，不要围绕当前 `freeze_layers=2` retuned anchor 再开 fresh adaptive main study；优先把工作锚点降回旧 stable winner（`freeze_layers=3`, `lr=1e-4`, `weight_decay=1e-4`, `dropout=0.3`）再决定是否做新的小步验证，且仍只推进 `resnext`、不回到 `384` head 或非 `resnext` backbone。

---

## 2026-04-18：ResNeXt V100 Dedicated Main-Study（回锚 stable freeze3 lane）

> **独立 campaign 说明**
> - 这一轮显式延续 `fair_backbone_compare` stable-winner side campaign，只推进 `resnext`，不回到 canonical ResUNet 主线。
> - 保持 matched V100 几何不变：`image_size=256`, `num_slices_per_view=8`, `feature fusion`, `share_backbone=false`, `use_attention_pooling=false`（mean pooling），`fusion_hidden_dim=256` 不变。
> - 唯一离散改动：把 dedicated formal template 回锚到旧 fair-backbone stable winner（`freeze_layers=3`, `lr=1e-4`, `weight_decay=1e-4`, `dropout=0.3`, `gradient_clip_norm=1.0`），并把 fresh adaptive main-study 的搜索固定在 `freeze_layers=3`，只继续扫描 stable-anchor 周围的标量。
> - 本轮 fresh study 为 `runs/optuna_main_autoloop/iter_0006_20260418_050606`，search-config copy 为 `autoresearch_logs/generated_search_configs/optuna_main_search_iter_0006_20260418_050606.yaml`，运行 commit 为 `a1baefd`，4/4 trials completed。

- [x] **CMP-FAIR-V100-RESNEXT-MAIN-STABLE-FREEZE3**：`configs/autoresearch_formal_resnext_v100.yaml`（回锚到旧 stable winner）+ `configs/optuna_main_search_resnext_v100.yaml`（`freeze_layers` 固定为 `3`）→ best completed trial 为 **trial 3**（`freeze_layers=3`, `lr=1e-4`, `weight_decay=1e-4`, `dropout=0.35`, `gradient_clip_norm=1.0`）→ `val_acc=0.925531914893617`, `val_auc=0.9486363636363636`, `val_f1=0.9156626506024096`, `peak_vram=2.14 GiB`, `total_seconds=848.4` → **discard**（较旧 fair-backbone stable winner `CMP-FAIR-V100-FORMAL-RESNEXT` 的 `0.9148936170212766 / 0.9404545454545454` 提升 `0.0106382978723404 / 0.0081818181818182`，说明回到 `freeze_layers=3` 并仅上调 `dropout` 到 `0.35` 仍能把 accuracy 拉回当前顶档；但它与当前 retained keep `CMP-FAIR-V100-RESNEXT-FORMAL-CONFIRM` 的 `0.925531914893617 / 0.9563636363636363` 在 accuracy 上完全持平，AUC 仍低 `0.0077272727272727`，因此按 tie-break 不能取代当前 best keep。）
- **Monitor takeaways**：在固定 `freeze_layers=3` 的 4 个 completed trial 里，旧 stable winner 本体（trial 0：`dropout=0.3`, `lr=1e-4`, `wd=1e-4`, `clip=1.0`）恢复到 `0.9042553191489362 / 0.94`，说明把搜索锚点降回稳定配方是正确方向；唯一进一步冲到 `0.925531914893617` 的是只把 `dropout` 上调到 `0.35` 的 trial 3。相反，更低 `lr=7.5e-5` 或更强正则（`lr=5e-5`, `weight_decay=5e-4`, `dropout=0.25`）都退到 `0.8829787234042553` 或 `0.8723404255319149`，说明当前 freeze3 家族最值得继续验证的不是更低学习率，而是“保留 stable winner 标量，只把 dropout 提到 `0.35`”这一单点。
- **本轮结论**：backlog 所建议的“先回锚旧 stable winner，再做小步验证”是有效的；它证明 `freeze_layers=3` 这条较稳的线并没有失去 ceiling，只是当前最有信息量的增量不再是重新解冻到 `freeze_layers=2`，而是更温和的 `dropout` 上调。
- **推荐动作**：若后续继续该独立 side campaign，优先对 trial 3 做 1 次 direct formal confirmation；若它能再次打到 `0.925531914893617` 且 AUC 更接近或超过当前 keep，再考虑把 freeze3/dropout0.35 升级为新的稳定锚点。若直接 confirmation 回落，则停止继续围绕这条 freeze3 标量线做 fresh main-study。

---

## 2026-04-18：ResNeXt V100 Direct Formal Confirmation（stable freeze3 + dropout0.35）

> **独立 campaign 说明**
> - 这一轮继续承接 `fair_backbone_compare` stable-winner side campaign，只推进 `resnext`，不回到 canonical ResUNet 主线。
> - 保持 matched V100 几何与 stable freeze3 锚点完全不变：`image_size=256`, `num_slices_per_view=8`, `feature fusion`, `share_backbone=false`, `use_attention_pooling=false`（mean pooling），`fusion_hidden_dim=256`, `freeze_layers=3`, `lr=1e-4`, `weight_decay=1e-4`, `gradient_clip_norm=1.0`。
> - 唯一离散改动：把 dedicated formal template `configs/autoresearch_formal_resnext_v100.yaml` 的 `dropout` 从 `0.3` 上调到上一轮 fresh main-study winner 所暗示的 `0.35`，然后直接做 1 次 formal confirmation。
> - 本轮运行 commit 为 `eb823b7`；由于这是 fixed-config confirmation 而不是 fresh study，直接在允许的单卡 fallback `GPU 2` 上运行 `train.py --config configs/autoresearch_formal_resnext_v100.yaml`。

- [x] **CMP-FAIR-V100-RESNEXT-FORMAL-STABLE-DROPOUT035**：`configs/autoresearch_formal_resnext_v100.yaml`（仅把 stable freeze3 锚点的 `dropout` 从 `0.3` 上调到 `0.35`）→ `val_acc=0.8936170212765957`, `val_auc=0.9404545454545454`, `val_f1=0.8809523809523809`, `peak_vram=2.14 GiB`, `total_seconds=813.1` → **discard**（较上一轮 fresh main-study winner `CMP-FAIR-V100-RESNEXT-MAIN-STABLE-FREEZE3` 的 `0.925531914893617 / 0.9486363636363636` 回落 `0.0319148936170213 / 0.0081818181818182`，说明 search 中出现的 `dropout=0.35` 高点没有在 direct formal 语义下复现；它也较当前 retained keep `CMP-FAIR-V100-RESNEXT-FORMAL-CONFIRM` 的 `0.925531914893617 / 0.9563636363636363` 低 `0.0319148936170213 / 0.0159090909090909`，并且 accuracy 还比旧 fair-backbone stable winner `CMP-FAIR-V100-FORMAL-RESNEXT` 的 `0.9148936170212766` 低 `0.0212765957446809`，AUC 仅与旧 stable winner 持平。）
- **本轮结论**：`freeze_layers=3 + dropout=0.35` 不能晋升为新的稳定锚点。上一轮 fresh main-study 所揭示的高点更像 search-only 波动，而不是可直接复现的 formal 改进；对这条线继续投入已缺乏回报。
- **推荐动作**：若后续继续该独立 side campaign，应把 dedicated formal template 保持在旧 fair-backbone stable winner（`freeze_layers=3`, `lr=1e-4`, `weight_decay=1e-4`, `dropout=0.3`, `gradient_clip_norm=1.0`）作为默认锚点，不再继续围绕 `dropout=0.35` 做 fresh main-study 或 direct formal。若外层 loop 仍要求再给该 side campaign 1 次 closure check，唯一还算信息充足的动作是做 1 次 exact old stable winner 的 direct formal rerun；否则应收束该 side campaign。

---

## 2026-04-18：ResNeXt V100 Direct Formal Closure Rerun（exact old stable winner）

> **独立 campaign 说明**
> - 这一轮继续承接 `fair_backbone_compare` stable-winner side campaign，只推进 `resnext`，不回到 canonical ResUNet 主线。
> - 保持 matched V100 几何与 old stable winner 完全不变：`image_size=256`, `num_slices_per_view=8`, `feature fusion`, `share_backbone=false`, `use_attention_pooling=false`（mean pooling），`fusion_hidden_dim=256`, `freeze_layers=3`, `lr=1e-4`, `weight_decay=1e-4`, `dropout=0.3`, `gradient_clip_norm=1.0`。
> - 唯一离散动作：不再继续 search 或改动标量，只对 exact old stable winner 做 1 次 direct formal closure rerun，判断它在经历近期 search-only 波动后是否仍能直接回到原始 stable winner 水平。
> - 本轮运行 commit 为 `bf4bd4d`；由于这是 fixed-config rerun 而不是 fresh study，直接在允许的单卡 fallback `GPU 2` 上运行 `train.py --config configs/autoresearch_formal_resnext_v100.yaml`。

- [x] **CMP-FAIR-V100-RESNEXT-FORMAL-STABLE-RERUN**：`configs/autoresearch_formal_resnext_v100.yaml`（exact old stable winner，无任何超参改动）→ `val_acc=0.9042553191489362`, `val_auc=0.9404545454545453`, `val_f1=0.9010989010989011`, `peak_vram=2.14 GiB`, `total_seconds=811.5` → **discard**（较当前 retained keep `CMP-FAIR-V100-RESNEXT-FORMAL-CONFIRM` 的 `0.925531914893617 / 0.9563636363636363` 低 `0.0212765957446808 / 0.015909090909091`；较 old fair-backbone stable winner `CMP-FAIR-V100-FORMAL-RESNEXT` 的 `0.9148936170212766 / 0.9404545454545454` 也低 `0.0106382978723404` accuracy，AUC 基本持平。它虽然比上一轮 `dropout=0.35` direct formal discard 回升 `0.0106382978723405` accuracy，说明把模板留在 old stable winner 上确实优于继续追逐 search-only dropout 波动，但 exact rerun 本身仍未复现原始 stable winner 的 accuracy ceiling。）  
- **本轮结论**：closure check 已完成，而且结果偏负面。当前 ResNeXt V100 side campaign 在既定 `256x8` / feature-fusion + mean pooling / freeze3 stable-family 语义下，已经没有足够强的 direct formal 证据支持继续做新的小步扫描。
- **推荐动作**：默认收束并归档该独立 side campaign。若外层 loop 仍要求 continuation，应先由人类明确给出新的离散假设；按协议，当前已属于“连续 5 个以上实验 discard 且没有新思路”的状态，不应继续盲跑。

---

## 2026-04-18：ResNeXt V100 Dedicated Main-Study（batch_size=12）

> **独立 campaign 说明**
> - 这一轮是在外层 loop 显式要求 continuation 的前提下，继续承接 `fair_backbone_compare` stable-winner side campaign，只推进 `resnext`，不回到 canonical ResUNet 主线。
> - 保持 matched V100 几何不变：`image_size=256`, `num_slices_per_view=8`, `feature fusion`, `share_backbone=false`, `use_attention_pooling=false`（mean pooling），`fusion_hidden_dim=256`，并继续把 main-study 固定在 `freeze_layers=3` stable lane。
> - 唯一离散改动：把 dedicated formal template `configs/autoresearch_formal_resnext_v100.yaml` 的 `batch_size` 从 `6` 提到 `12`，利用 V100 的显存余量测试“更大 batch 是否能降低梯度噪声并抬回 accuracy ceiling”。
> - fresh study 为 `runs/optuna_main_autoloop/iter_0009_20260418_061533`，search-config copy 为 `autoresearch_logs/generated_search_configs/optuna_main_search_iter_0009_20260418_061533.yaml`，运行 commit 为 `d118031`，4/4 trials completed。

- [x] **CMP-FAIR-V100-RESNEXT-MAIN-BS12**：`configs/autoresearch_formal_resnext_v100.yaml`（仅把 `batch_size` 从 `6` 提到 `12`）+ fresh adaptive main-study → best completed trial 为 **trial 0**（`freeze_layers=3`, `lr=1e-4`, `weight_decay=1e-4`, `dropout=0.3`, `gradient_clip_norm=1.0`）→ `val_acc=0.9042553191489362`, `val_auc=0.9504545454545454`, `val_f1=0.8888888888888888`, `peak_vram=3.64 GiB`, `total_seconds=994.1` → **discard**（较当前 retained keep `CMP-FAIR-V100-RESNEXT-FORMAL-CONFIRM` 的 `0.925531914893617 / 0.9563636363636363` 低 `0.0212765957446808 / 0.0059090909090909`；也较旧 fair-backbone stable winner `CMP-FAIR-V100-FORMAL-RESNEXT` 的 `0.9148936170212766 / 0.9404545454545454` 低 `0.0106382978723404` accuracy。虽然 AUC 相比旧 stable winner 提高 `0.01`，但按主指标 `val_acc` 仍不足以晋升。）
- **Monitor takeaways**：在 batch-size 加倍后的 4 个 completed trial 里，enqueued 的 stable-winner 本体（trial 0）仍然是 accuracy 最优，说明更大 batch 并没有把搜索重点推离旧 stable family；更低学习率 `7.5e-5` 或更强正则组合（`lr=5e-5`, `weight_decay=5e-4`, `dropout=0.25`）都把 accuracy 压回 `0.8830 ~ 0.8936`。VRAM 只从此前 freeze3 家族常见的约 `2.14 GiB` 升到 `3.64 GiB`，说明硬件余量确实充足，但 ceiling 仍未抬升。
- **本轮结论**：把 `batch_size` 从 `6` 提到 `12` 没有解决当前 ResNeXt side campaign 的核心问题。近期的 accuracy 回落更像配方本身缺乏可复现增益，而不是单纯由小 batch 引起的训练噪声。
- **推荐动作**：默认正式收束该独立 side campaign。若外层 loop 仍要继续，应先由人类明确新的离散假设；在现有 `resnext` / `256x8` / feature-fusion + mean pooling / stable-freeze3 标量空间内，不再建议继续 fresh main-study。

---

## 2026-04-18：ResNeXt V100 Dedicated Main-Study（reopen freeze2 vs freeze3）

> **独立 campaign 说明**
> - 这一轮是在外层 loop 显式要求 continuation 的前提下，继续承接 `fair_backbone_compare` stable-winner side campaign，只推进 `resnext`，不回到 canonical ResUNet 主线。
> - 保持 matched V100 几何不变：`image_size=256`, `num_slices_per_view=8`, `feature fusion`, `share_backbone=false`, `use_attention_pooling=false`（mean pooling），`fusion_hidden_dim=256`，base formal template 继续锚定 old stable winner（`freeze_layers=3`, `lr=1e-4`, `weight_decay=1e-4`, `dropout=0.3`, `gradient_clip_norm=1.0`）。
> - 唯一离散改动：把 `configs/optuna_main_search_resnext_v100.yaml` 的 `model.freeze_layers` 搜索从固定 `[3]` 重新开放到 `[2, 3]`，用一轮 fresh adaptive main-study 检验“freeze2 本身是否仍有可重复收益”，而不再引入新的几何或 backbone 改动。
> - fresh study 为 `runs/optuna_main_autoloop/iter_0010_20260418_065657`，search-config copy 为 `autoresearch_logs/generated_search_configs/optuna_main_search_iter_0010_20260418_065657.yaml`，运行 commit 为 `87c1e9b`，4/4 trials completed。

- [x] **CMP-FAIR-V100-RESNEXT-MAIN-REOPEN-FREEZE2**：`configs/optuna_main_search_resnext_v100.yaml`（仅把 `model.freeze_layers` 搜索从 `[3]` 重新开放到 `[2, 3]`）+ fresh adaptive main-study → best completed trial 为 **trial 0**（即 enqueued stable template：`freeze_layers=3`, `lr=1e-4`, `weight_decay=1e-4`, `dropout=0.3`, `gradient_clip_norm=1.0`）→ `val_acc=0.925531914893617`, `val_auc=0.94`, `val_f1=0.9213483146067416`, `peak_vram=2.14 GiB`, `total_seconds=855.8` → **discard**（虽然它在 accuracy 上追平当前 retained keep `CMP-FAIR-V100-RESNEXT-FORMAL-CONFIRM` 的 `0.925531914893617`，并较 old fair-backbone stable winner `0.9148936170212766 / 0.9404545454545454` 高 `0.0106382978723404` accuracy，但它的 AUC 仍比当前 retained keep 低 `0.0163636363636363`，且还比 old stable winner 低 `0.0004545454545454` AUC，因此不能替换现有 best keep。）  
- **Monitor takeaways**：这轮 reopen 后唯一的 `freeze_layers=2` completed trial（trial 2：`lr=1e-4`, `weight_decay=5e-4`, `dropout=0.35`, `gradient_clip_norm=2.0`）只得到 `val_acc=0.8829787234042553`, `val_auc=0.9440909090909091`，显存升到约 `6.23 GiB`，但 accuracy 仍比 trial 0 低 `0.0425531914893617`。其余 completed trial 也都停留在 `0.8723 ~ 0.8936` accuracy 区间，说明“freeze2 被先前搜索空间限制掩盖”的假设在当前 budget 下没有得到支持。  
- **本轮结论**：把搜索重新开放到 `freeze_layers=2` 并没有恢复当前 ResNeXt side campaign 的可复现上行空间；反而再次说明目前最强的 search-time 点仍然是 old stable winner 本体，而不是更深解冻。  
- **推荐动作**：默认正式收束该独立 side campaign。若外层 loop 仍要求 continuation，应先由人类明确一个超出当前 `freeze_layers` / stable-scalar 轴的新离散假设；否则不再建议继续围绕 `resnext` 的这条 `256x8` mean-pooling 配方做 fresh main-study。

---

## 2026-04-18：ResNeXt V100 Dedicated Main-Study（AttentionPooling）

> **独立 campaign 说明**
> - 这一轮是在外层 loop 显式要求 continuation 的前提下，继续承接 `fair_backbone_compare` stable-winner side campaign，只推进 `resnext`，不回到 canonical ResUNet 主线。
> - 保持 dedicated V100 lane 的几何与 backbone 其余部分不变：`image_size=256`, `num_slices_per_view=8`, `feature fusion`, `share_backbone=false`, `fusion_hidden_dim=256`。
> - 唯一离散改动：把 `configs/autoresearch_formal_resnext_v100.yaml` 的 `use_attention_pooling` 从 `false` 切到 `true`，显式检验“slice attention 是否能在不改 backbone 与几何的前提下，恢复当前 ResNeXt side campaign 的 accuracy ceiling”。
> - fresh study 为 `runs/optuna_main_autoloop/iter_0011_20260418_073137`，search-config copy 为 `autoresearch_logs/generated_search_configs/optuna_main_search_iter_0011_20260418_073137.yaml`，运行 commit 为 `96bd773`，4/4 trials completed。

- [x] **CMP-FAIR-V100-RESNEXT-MAIN-ATTNPOOL**：`configs/autoresearch_formal_resnext_v100.yaml`（仅把 `use_attention_pooling` 从 `false` 切到 `true`）+ fresh adaptive main-study → best completed trial 为 **trial 2**（`freeze_layers=2`, `lr=1e-4`, `weight_decay=5e-4`, `dropout=0.35`, `gradient_clip_norm=2.0`）→ `val_acc=0.8936170212765957`, `val_auc=0.9640909090909091`, `val_f1=0.8837209302325582`, `peak_vram=6.24 GiB`, `total_seconds=847.7` → **discard**（较当前 retained keep `CMP-FAIR-V100-RESNEXT-FORMAL-CONFIRM` 的 `0.925531914893617 / 0.9563636363636363` 低 `0.0319148936170213` accuracy；虽然 AUC 反而高 `0.0077272727272728`，但按当前规则不能用更低 accuracy 的点晋升。它也较旧 fair-backbone stable winner `CMP-FAIR-V100-FORMAL-RESNEXT` 的 `0.9148936170212766 / 0.9404545454545454` 低 `0.0212765957446809` accuracy。）
- **Monitor takeaways**：在 attention-pooling lane 的 4 个 completed trial 里，所有 `freeze_layers=3` 试次都停在 `0.8404 ~ 0.8723` accuracy，连 enqueued stable template 本体也只得到 `0.851063829787234 / 0.9336363636363636`；最优点再次回到更激进的 `freeze_layers=2 + dropout=0.35 + clip=2.0`，并把 AUC 抬到 `0.9641`，但 accuracy ceiling 仍显著低于当前 retained keep。换言之，AttentionPooling 改变了最优超参形态，却没有把 ResNeXt lane 的主指标拉回冠军区间。
- **本轮结论**：这是一次明确超出 `freeze_layers` / stable-scalar 轴的新结构探针，但结果仍然偏负面。对于当前这条 dedicated ResNeXt `256x8` side campaign，slice AttentionPooling 并没有比原 mean-pooling 语义更强，至少在当前 budget 下没有体现为可用的 `val_acc` 改进。
- **推荐动作**：默认正式收束该独立 side campaign。若外层 loop 仍要继续，必须先由人类给出新的离散假设；不再建议继续围绕这条 ResNeXt lane 的 `mean pooling` / `AttentionPooling` 与旧 freeze/scalar 组合做 fresh main-study。

---

## 2026-04-18：ResNeXt V100 Dedicated Main-Study（mean-pooling + cosine scheduler）

> **独立 campaign 说明**
> - 这一轮是外层 loop 显式要求 continuation 下的单轮 coordinator 迭代，继续承接 `fair_backbone_compare` stable-winner side campaign，只推进 `resnext`，不回到 canonical ResUNet 主线。
> - 先按本轮硬约束把 dedicated formal template 从上一轮残留的 AttentionPooling 状态拉回 `256x8`、feature-fusion、mean-pooling 几何；`fusion_hidden_dim=256`、`share_backbone=false` 保持不变。
> - 唯一新的离散改动：把 `configs/autoresearch_formal_resnext_v100.yaml` 的 `train.scheduler` 从 `none` 切到 `cosine`，显式检验“较平滑的学习率衰减是否能缓解 ResNeXt 在这条 V100 side lane 上慢热且波动大的优化轨迹”。
> - fresh study 为 `runs/optuna_main_autoloop/iter_0012_20260418_080648`，search-config copy 为 `autoresearch_logs/generated_search_configs/optuna_main_search_iter_0012_20260418_080648.yaml`，运行 commit 为 `7b5f95d`，4/4 trials completed。

- [x] **CMP-FAIR-V100-RESNEXT-MAIN-COSINE**：`configs/autoresearch_formal_resnext_v100.yaml`（mean-pooling lane，仅把 `train.scheduler` 从 `none` 切到 `cosine`）+ fresh adaptive main-study → best completed trial 为 **trial 0**（即 enqueued stable template：`freeze_layers=3`, `lr=1e-4`, `weight_decay=1e-4`, `dropout=0.3`, `gradient_clip_norm=1.0`）→ `val_acc=0.9042553191489362`, `val_auc=0.9363636363636364`, `val_f1=0.8888888888888888`, `peak_vram=2.14 GiB`, `total_seconds=854.8` → **discard**（较当前 retained keep `CMP-FAIR-V100-RESNEXT-FORMAL-CONFIRM` 的 `0.925531914893617 / 0.9563636363636363` 低 `0.0212765957446808 / 0.02`；也较旧 fair-backbone stable winner `CMP-FAIR-V100-FORMAL-RESNEXT` 的 `0.9148936170212766 / 0.9404545454545454` 低 `0.0106382978723404 / 0.004090909090909`，因此不能晋升。）
- **Monitor takeaways**：在 cosine lane 的 4 个 completed trial 里，最优点没有离开 old stable winner 本体，说明引入 schedule 并没有改变当前 side campaign 的最优超参骨架。`freeze_layers=2 + dropout=0.35 + clip=2.0` 的 trial 2 虽把 `val_auc` 抬到 `0.9522727272727274`，但 `val_acc` 只有 `0.8723404255319149`；更保守的 `lr=5e-5 + wd=5e-4` 组合则直接退到 `0.8191489361702128 / 0.9254545454545455`。这说明在当前 15-epoch budget 下，cosine 衰减至多改善了部分排序质量，但没有把主指标 accuracy 拉回 keep 区间。
- **本轮结论**：把 ResNeXt side lane 切到 cosine scheduler 没有带来新的可复现上行空间。相反，它再次证明当前这条 `256x8` mean-pooling 配方里最强的 search-time 点仍只是 old stable winner 本体，而且 even that template 在 cosine 下也只能回到 `0.9043 / 0.9364`，低于此前无 scheduler 的 direct/formal 证据。
- **推荐动作**：默认正式收束该独立 side campaign。若外层 loop 仍要求 continuation，必须先给出一个新的离散假设；不再建议继续围绕这条 ResNeXt lane 的 `mean pooling` / `AttentionPooling` / `scheduler` 与旧 freeze-scalar 组合做 fresh main-study。

---

## 2026-04-18：ResNeXt V100 Dedicated Main-Study（feature-fusion head prenorm）

> **独立 campaign 说明**
> - 这一轮是外层 loop 显式要求 continuation 下的单轮 coordinator 迭代，继续承接 `fair_backbone_compare` stable-winner side campaign，只推进 `resnext`，不回到 canonical ResUNet 主线。
> - 保持 dedicated ResNeXt V100 lane 的几何与 backbone 其余部分不变：`image_size=256`, `num_slices_per_view=8`, `feature fusion`, `share_backbone=false`, `use_attention_pooling=false`（mean pooling），`fusion_hidden_dim=256`。
> - 唯一新的离散改动：在 `src/model.py` 的 `MultiViewCTClassifier` 里给拼接后的 `1536D` feature-fusion 向量加入 `LayerNorm(fused_dim)`，显式检验“轻量预归一化是否能稳定 ResNeXt lane 的 view-feature 尺度并恢复 accuracy ceiling”。
> - fresh study 为 `runs/optuna_main_autoloop/iter_0013_20260418_084209`，search-config copy 为 `autoresearch_logs/generated_search_configs/optuna_main_search_iter_0013_20260418_084209.yaml`，运行 commit 为 `7609b02`，4/4 trials completed。

- [x] **CMP-FAIR-V100-RESNEXT-MAIN-PRENORM**：`src/model.py`（feature-fusion 头新增 `LayerNorm(fused_dim)`）+ fresh adaptive main-study → best completed trial 为 **trial 2**（`freeze_layers=2`, `lr=1e-4`, `weight_decay=5e-4`, `dropout=0.35`, `gradient_clip_norm=2.0`）→ `val_acc=0.9148936170212766`, `val_auc=0.9490909090909091`, `val_f1=0.9`, `peak_vram=6.23 GiB`, `total_seconds=866.2` → **discard**（较当前 retained keep `CMP-FAIR-V100-RESNEXT-FORMAL-CONFIRM` 的 `0.925531914893617 / 0.9563636363636363` 仍低 `0.0106382978723404 / 0.0072727272727272`，因此不能晋升；但它较上一轮 cosine lane best `0.9042553191489362 / 0.9363636363636364` 回升 `0.0106382978723404 / 0.0127272727272727`，并以同样的 accuracy 追平旧 fair-backbone stable winner `CMP-FAIR-V100-FORMAL-RESNEXT` 的 `0.9148936170212766`，同时 AUC 更高 `0.0086363636363637`。）
- **Monitor takeaways**：prenorm 明显改写了当前 cosine lane 的最优形态。旧的 `freeze_layers=3` stable-template 本体在 trial 0 只得到 `0.8829787234042553 / 0.9395454545454545`，而更激进的 `freeze_layers=2 + dropout=0.35 + clip=2.0` 在 prenorm 后恢复到 `0.9149 / 0.9491`，说明这个结构改动并非纯负面；但它把 ceiling 拉回到“旧 stable winner 附近”，还没有把主指标拉过当前 retained keep。
- **本轮结论**：feature-fusion head prenorm 是一次有信息量的新结构稳定化探针。它成功修复了上一轮 cosine lane 的部分退化，并让 freeze2 aggressive lane 回到竞争区，但仍不足以替换当前最强 direct-formal keep，因此本轮仍记 **discard**。
- **推荐动作**：默认正式收束该独立 side campaign。若外层 loop 仍要求 continuation，必须先明确另一个新的离散假设，或显式决定是否值得做“保留 prenorm、再把 scheduler 从 `cosine` 解耦回 `none`”的单轮验证；不再建议继续旧 `freeze/scalar/pooling/scheduler` 轴的小步扫描。

---

## 2026-04-18：ResNeXt V100 Dedicated Main-Study（prenorm 解耦 scheduler 回退到 none）

> **独立 campaign 说明**
> - 这一轮是外层 loop 显式要求 continuation 下的单轮 coordinator 迭代，继续承接 `fair_backbone_compare` stable-winner side campaign，只推进 `resnext`，不回到 canonical ResUNet 主线。
> - 保持 dedicated ResNeXt V100 lane 的几何与上一轮结构改动不变：`image_size=256`, `num_slices_per_view=8`, `feature fusion`, `share_backbone=false`, `use_attention_pooling=false`（mean pooling），`fusion_hidden_dim=256`，并保留 feature-fusion 头的 `LayerNorm(fused_dim)` prenorm。
> - 唯一新的离散改动：把 `configs/autoresearch_formal_resnext_v100.yaml` 的 `train.scheduler` 从 carry-over 的 `cosine` 解耦回 `none`，显式检验“上一轮 prenorm 的回升究竟来自 prenorm 本身，还是依赖于与 cosine scheduler 的组合”。
> - fresh study 为 `runs/optuna_main_autoloop/iter_0014_20260418_091749`，search-config copy 为 `autoresearch_logs/generated_search_configs/optuna_main_search_iter_0014_20260418_091749.yaml`，运行 commit 为 `6a39df6`，4/4 trials completed。

- [x] **CMP-FAIR-V100-RESNEXT-MAIN-PRENORM-NOSCHED**：`configs/autoresearch_formal_resnext_v100.yaml`（保留 prenorm，仅把 `train.scheduler` 从 `cosine` 改回 `none`）+ fresh adaptive main-study → best completed trial 为 **trial 2**（`freeze_layers=2`, `lr=1e-4`, `weight_decay=5e-4`, `dropout=0.35`, `gradient_clip_norm=2.0`）→ `val_acc=0.9148936170212766`, `val_auc=0.9336363636363636`, `val_f1=0.9024390243902439`, `peak_vram=6.23 GiB`, `total_seconds=858.9` → **discard**（与上一轮 prenorm+cosine study 的 best `0.9148936170212766 / 0.9490909090909091` 在 accuracy 上持平，但 AUC 低 `0.0154545454545455`；也仍较当前 retained keep `CMP-FAIR-V100-RESNEXT-FORMAL-CONFIRM` 的 `0.925531914893617 / 0.9563636363636363` 低 `0.0106382978723404 / 0.0227272727272727`，因此不能晋升。）
- **Monitor takeaways**：解耦掉 cosine 后，search winner 仍然是上一轮同一组 aggressive freeze2 标量，说明 prenorm 的确改变了这条 lane 的最优超参形态；但它没有继续提高 accuracy ceiling，而且 AUC 明显回落。作为对照，旧 freeze3 stable template（trial 0）恢复到 `0.9042553191489362 / 0.9409090909090909`，比 cosine lane 同模板的 `0.9042553191489362 / 0.9363636363636364` 略好，说明当前负面结果并不是“scheduler=none 普遍更差”，而是 **prenorm 带来的排序质量改善并没有在无 scheduler 条件下保留下来**。
- **本轮结论**：backlog 明示的“prenorm 与 scheduler carry-over 解耦”验证已经完成，而且结果偏负面。它没有带来比 prenorm+cosine 更强的证据，也没有把 direct-formal retained keep 推翻；因此这条 ResNeXt V100 side campaign 现阶段应视为已经完成 closure。
- **推荐动作**：按协议停止继续盲跑。若未来还要重开这条 side campaign，必须先提出另一个真正新的离散假设；在此之前，不再建议继续沿当前 `freeze/scalar/pooling/scheduler/prenorm` 组合轴做 fresh adaptive main-study。

---

## 2026-04-18：ResNeXt V100 Dedicated Main-Study（trim_edge_slices=2）

> **独立 campaign 说明**
> - 这一轮是外层 loop 显式要求 continuation 下的单轮 coordinator 迭代，继续承接 `fair_backbone_compare` stable-winner side campaign，只推进 `resnext`，不回到 canonical ResUNet 主线。
> - 保持 dedicated ResNeXt V100 lane 的几何与当前结构状态不变：`image_size=256`, `num_slices_per_view=8`, `feature fusion`, `share_backbone=false`, `use_attention_pooling=false`（mean pooling），`fusion_hidden_dim=256`，并保留 feature-fusion 头的 `LayerNorm(fused_dim)` prenorm。
> - 唯一新的离散改动：把 `configs/autoresearch_formal_resnext_v100.yaml` 的 `data.trim_edge_slices` 从 `1` 提到 `2`，显式检验“在当前 8-slice mean-pooling lane 里，更强的边缘切片裁剪是否能降低无信息切片噪声并抬回 accuracy ceiling”。
> - fresh study 为 `runs/optuna_main_autoloop/iter_0015_20260418_095144`，search-config copy 为 `autoresearch_logs/generated_search_configs/optuna_main_search_iter_0015_20260418_095144.yaml`，运行 commit 为 `e9e5e43`，4/4 trials completed。

- [x] **CMP-FAIR-V100-RESNEXT-MAIN-EDGE-TRIM2**：`configs/autoresearch_formal_resnext_v100.yaml`（仅把 `trim_edge_slices` 从 `1` 提到 `2`）+ fresh adaptive main-study → best completed trial 为 **trial 3**（`freeze_layers=3`, `lr=1e-4`, `weight_decay=1e-4`, `dropout=0.25`, `gradient_clip_norm=1.0`）→ `val_acc=0.9042553191489362`, `val_auc=0.9777272727272727`, `val_f1=0.8888888888888888`, `peak_vram=2.14 GiB`, `total_seconds=852.1` → **discard**（较当前 retained keep `CMP-FAIR-V100-RESNEXT-FORMAL-CONFIRM` 的 `0.925531914893617 / 0.9563636363636363` 低 `0.0212765957446808` accuracy，因此不能晋升；虽然 AUC 反而高 `0.0213636363636364`，但按当前规则不能用更低 accuracy 的点替换现有 keep。它也较旧 fair-backbone stable winner `CMP-FAIR-V100-FORMAL-RESNEXT` 的 `0.9148936170212766 / 0.9404545454545454` 低 `0.0106382978723404` accuracy。）
- **Monitor takeaways**：在 `trim_edge_slices=2` 的 4 个 completed trial 里，最优点不再落在近期的 aggressive freeze2 lane，而是回到 `freeze_layers=3` 家族；其中 trial 3 只把 template 的 `dropout` 从 `0.3` 下调到 `0.25`，就在与 trial 0 / trial 2 同分的 `0.9042553191489362` accuracy 上，把 AUC 抬到全 study 最高的 `0.9777272727272727`。这说明更强 edge trimming 主要改善了排序质量和阈值敏感性，而不是把固定阈值 accuracy 推过当前 keep。
- **本轮结论**：`trim_edge_slices=2` 是一个真正新的离散假设，而且结果有信息量，但它依然没能把 dedicated ResNeXt V100 side campaign 的主指标拉回冠军区间。当前更合理的判断是：edge trimming 可以作为 ranking-quality probe 保留观察，但不足以成为新的 retained recipe。
- **推荐动作**：默认正式收束该独立 side campaign。若外层 loop 仍强制 continuation，唯一还算信息充足的动作是对 exact trial-3 配方做 1 次 direct formal closure check；否则不要继续在这条 ResNeXt `256x8` mean-pooling lane 上做新的 fresh adaptive main-study。

---

## 2026-04-18：ResNeXt V100 Direct Formal Closure Check（trim_edge_slices=2 trial-3）

> **独立 campaign 说明**
> - 这一轮是外层 loop 显式要求 continuation 下的单轮 coordinator 迭代，继续承接 `fair_backbone_compare` stable-winner side campaign，只推进 `resnext`，不回到 canonical ResUNet 主线。
> - 保持 dedicated ResNeXt V100 lane 的几何与当前结构状态不变：`image_size=256`, `num_slices_per_view=8`, `feature fusion`, `share_backbone=false`, `use_attention_pooling=false`（mean pooling），`fusion_hidden_dim=256`，并保留 feature-fusion 头的 `LayerNorm(fused_dim)` prenorm 与 `trim_edge_slices=2`。
> - 唯一新的离散改动：把 dedicated formal template `configs/autoresearch_formal_resnext_v100.yaml` 从上一轮 `trim_edge_slices=2` main-study 的 template 状态，对齐到 exact trial 3，只把 `model.dropout` 从 `0.3` 下调到 `0.25`，然后直接做 1 次 formal closure check。
> - 本轮运行 commit 为 `72d7ab4`；由于这是 fixed-config confirmation 而不是 fresh study，直接在允许的单卡 fallback `GPU 2` 上运行 `train.py --config configs/autoresearch_formal_resnext_v100.yaml`。

- [x] **CMP-FAIR-V100-RESNEXT-FORMAL-EDGE-TRIM2-CLOSURE**：`configs/autoresearch_formal_resnext_v100.yaml`（保留 `trim_edge_slices=2`，仅把 `dropout` 从 template 的 `0.3` 下调到上一轮 trial-3 所暗示的 `0.25`）→ `val_acc=0.8936170212765957`, `val_auc=0.95`, `val_f1=0.8780487804878049`, `peak_vram=2.14 GiB`, `total_seconds=807.1` → **discard**（较上一轮 fresh main-study `CMP-FAIR-V100-RESNEXT-MAIN-EDGE-TRIM2` 的 trial-3 高点 `0.9042553191489362 / 0.9777272727272727` 低 `0.0106382978723405 / 0.0277272727272727`，说明 `trim_edge_slices=2` lane 的 search-time 最优点没有在 direct formal 语义下复现；它也较当前 retained keep `CMP-FAIR-V100-RESNEXT-FORMAL-CONFIRM` 的 `0.925531914893617 / 0.9563636363636363` 低 `0.0319148936170213 / 0.0063636363636363`，并较旧 fair-backbone stable winner `CMP-FAIR-V100-FORMAL-RESNEXT` 的 `0.9148936170212766 / 0.9404545454545454` 低 `0.0212765957446809` accuracy，因此不能晋升。）
- **本轮结论**：backlog 明示的 exact closure check 已完成，而且结果偏负面。更强 edge trimming 依然更像 ranking-quality probe，而不是能稳定抬升 fixed-threshold accuracy 的 retained recipe。
- **推荐动作**：默认正式收束该独立 side campaign。若外层 loop 仍要求 continuation，必须先提出一个超出当前 `trim_edge_slices / prenorm / scheduler / pooling / freeze-scalar` 组合轴的新离散假设；否则不再建议继续围绕这条 ResNeXt `256x8` mean-pooling lane 盲跑 fresh main-study 或 direct rerun。

---

## 2026-04-18：ResNeXt V100 Dedicated Main-Study（view-wise feature recalibration）

> **独立 campaign 说明**
> - 这一轮是外层 loop 显式要求 continuation 下的单轮 coordinator 迭代，继续承接 `fair_backbone_compare` stable-winner side campaign，只推进 `resnext`，不回到 canonical ResUNet 主线。
> - 保持 dedicated ResNeXt V100 lane 的几何与当前 committed template 状态不变：`image_size=256`, `num_slices_per_view=8`, `feature fusion`, `share_backbone=false`, `use_attention_pooling=false`（mean pooling），`fusion_hidden_dim=256`，并保留 `trim_edge_slices=2` 与 feature-fusion 头的 `LayerNorm(fused_dim)` prenorm。
> - 唯一新的离散改动：在 `src/model.py` 的 `MultiViewCTClassifier` 中加入 identity-initialized 的 per-view feature recalibration gate，让每个 pooled 视角特征先做轻量 channel-wise 重标定再拼接，显式检验“feature-fusion 缺少显式视角可靠度建模”是否是当前 ResNeXt lane 的瓶颈。
> - fresh study 为 `runs/optuna_main_autoloop/iter_0017_20260418_104356`，search-config copy 为 `autoresearch_logs/generated_search_configs/optuna_main_search_iter_0017_20260418_104356.yaml`，运行 commit 为 `f3e9701`，4/4 trials completed。

- [x] **CMP-FAIR-V100-RESNEXT-MAIN-VIEW-RECAL**：`src/model.py`（feature-fusion 头新增 per-view feature recalibration gate）+ fresh adaptive main-study → best completed trial 为 **trial 1**（`freeze_layers=3`, `lr=5e-5`, `weight_decay=5e-4`, `dropout=0.3`, `gradient_clip_norm=1.0`）→ `val_acc=0.9042553191489362`, `val_auc=0.9681818181818183`, `val_f1=0.9032258064516129`, `peak_vram=2.15 GiB`, `total_seconds=850.0` → **discard**（较当前 retained keep `CMP-FAIR-V100-RESNEXT-FORMAL-CONFIRM` 的 `0.925531914893617 / 0.9563636363636363` 低 `0.0212765957446808` accuracy，因此不能晋升；虽然 AUC 反而高 `0.011818181818182`，但按当前规则不能用更低 accuracy 的点替换现有 keep。它也只较上一轮 `trim_edge_slices=2` closure discard 的 `0.8936170212765957 / 0.95` 小幅回升，说明这次结构改动更多是在提升排序质量，而不是恢复固定阈值 accuracy ceiling。）
- **Monitor takeaways**：在这轮 4 个 completed trials 里，最优点回到 `freeze_layers=3` 家族，而且落在更低学习率、更高 weight decay 的保守角点（trial 1）。值得注意的是，trial 1 与 trial 3 被命中了同一组参数，但分别得到 `0.9042553191489362 / 0.9681818181818183` 与 `0.8936170212765957 / 0.9572727272727273`，说明新的 recalibration gate 并没有消除当前 side campaign 的 run-to-run 波动。唯一的 `freeze_layers=2` 试次（trial 2）仍只得到 `0.8829787234042553 / 0.9472727272727274`，同时显存升到约 `6.24 GiB`，说明更深解冻依旧不是这条 lane 的出路。
- **本轮结论**：`view-wise feature recalibration` 是一个真正新的结构假设，而且把 best-trial AUC 再推高了一档，但它仍未把 dedicated ResNeXt V100 side campaign 的主指标拉回冠军区间，因此本轮继续记 **discard**。
- **推荐动作**：默认正式收束该独立 side campaign。若外层 loop 仍要求 continuation，必须先给出另一个真正新的离散假设；不再建议继续沿当前 `view-recalibration / trim_edge_slices / prenorm / scheduler / pooling / freeze-scalar` 组合轴做 fresh adaptive main-study。

---

## 2026-04-18：ResNeXt V100 Dedicated Main-Study（light cross-view token mixer）

> **独立 campaign 说明**
> - 这一轮是外层 loop 显式要求 continuation 下的单轮 coordinator 迭代，继续承接 `fair_backbone_compare` stable-winner side campaign，只推进 `resnext`，不回到 canonical ResUNet 主线。
> - 保持 dedicated ResNeXt V100 lane 的几何与当前 committed template 状态不变：`image_size=256`, `num_slices_per_view=8`, `feature fusion`, `share_backbone=false`, `use_attention_pooling=false`（mean pooling），`fusion_hidden_dim=256`，并保留 `trim_edge_slices=2`、feature-fusion 头的 `LayerNorm(fused_dim)` prenorm，以及拼接前的 per-view feature recalibration。
> - 唯一新的离散改动：在 `src/model.py` 的 `MultiViewCTClassifier` 中，于 3 个 pooled view features 拼接前加入一个轻量 residual `CrossViewAttention` mixer（`attention_dim=256`, `num_heads=4`, `num_layers=1`, `dropout=0.1`, `residual_scale=0.125`），显式检验“当前 ResNeXt lane 的瓶颈是否来自缺少显式跨视角 token interaction，而不是单视角重标定不足”。
> - fresh study 为 `runs/optuna_main_autoloop/iter_0018_20260418_112019`，search-config copy 为 `autoresearch_logs/generated_search_configs/optuna_main_search_iter_0018_20260418_112019.yaml`，运行 commit 为 `139a5b6`，4/4 trials completed。

- [x] **CMP-FAIR-V100-RESNEXT-MAIN-XVIEW-MIXER**：`src/model.py`（feature-fusion 头在拼接前新增轻量 cross-view token mixer）+ fresh adaptive main-study → best completed trial 为 **trial 0**（`freeze_layers=3`, `lr=1e-4`, `weight_decay=1e-4`, `dropout=0.25`, `gradient_clip_norm=1.0`）→ `val_acc=0.9148936170212766`, `val_auc=0.9481818181818182`, `val_f1=0.9069767441860465`, `peak_vram=2.16 GiB`, `total_seconds=855.9` → **discard**（较当前 retained keep `CMP-FAIR-V100-RESNEXT-FORMAL-CONFIRM` 的 `0.925531914893617 / 0.9563636363636363` 低 `0.0106382978723404 / 0.0081818181818181`，因此不能晋升；虽然 accuracy 追平旧 fair-backbone stable winner `CMP-FAIR-V100-FORMAL-RESNEXT` 的 `0.9148936170212766`，并把 AUC 抬高到 `0.9481818181818182`，但按当前规则仍不足以替换现有 best keep。）
- **Monitor takeaways**：在这轮 4 个 completed trials 里，最优点回到 `freeze_layers=3` 家族，而且直接落在 enqueued template 本体（trial 0）；其 duplicate template trial 3 只有 `0.9042553191489362 / 0.9536363636363637`，说明引入 cross-view mixer 后 AUC 还能继续波动上行，但 fixed-threshold accuracy 并没有同步稳定抬升。唯一的 `freeze_layers=2` 试次（trial 2）仍只得到 `0.9042553191489362 / 0.9240909090909091`，同时显存升到约 `6.26 GiB`，说明更深解冻依旧不是这条 lane 的出路。
- **本轮结论**：`light cross-view token mixing` 是又一个真正新的结构假设，而且把 `freeze_layers=3` 家族的 best-trial AUC 推到旧 stable winner 之上；但它依然没有把 dedicated ResNeXt V100 side campaign 的主指标拉回当前冠军区间，因此本轮继续记 **discard**。
- **推荐动作**：默认正式收束该独立 side campaign。若外层 loop 仍要求 continuation，必须先给出另一个真正新的离散假设；不再建议继续沿当前 `cross-view mixer / view-recalibration / trim_edge_slices / prenorm / scheduler / pooling / freeze-scalar` 组合轴做 fresh adaptive main-study。

---

## 2026-04-18：ResNeXt V100 Dedicated Main-Study（gated fusion head）

> **独立 campaign 说明**
> - 这一轮是外层 loop 显式要求 continuation 下的单轮 coordinator 迭代，继续承接 `fair_backbone_compare` stable-winner side campaign，只推进 `resnext`，不回到 canonical ResUNet 主线。
> - 保持 dedicated ResNeXt V100 lane 的几何与当前 committed template 状态不变：`image_size=256`, `num_slices_per_view=8`, `feature fusion`, `share_backbone=false`, `use_attention_pooling=false`（mean pooling），`fusion_hidden_dim=256`，并保留 `trim_edge_slices=2`、feature-fusion 头的 `LayerNorm(fused_dim)` prenorm、拼接前的 per-view feature recalibration 与 light cross-view mixer。
> - 唯一新的离散改动：在 `src/model.py` 的 `MultiViewCTClassifier` 中，把融合头从 plain `Linear + ReLU` MLP 改成轻量 GLU-gated MLP，显式检验“当前 ResNeXt lane 的瓶颈是否在 fused token 级别缺少乘性门控表达，而不是继续往 view-side 叠新模块”。
> - fresh study 为 `runs/optuna_main_autoloop/iter_0019_20260418_115456`，search-config copy 为 `autoresearch_logs/generated_search_configs/optuna_main_search_iter_0019_20260418_115456.yaml`，运行 commit 为 `15b1ef6`，4/4 trials completed。

- [x] **CMP-FAIR-V100-RESNEXT-MAIN-GATED-HEAD**：`src/model.py`（feature-fusion 头把 plain `Linear + ReLU` 改为轻量 GLU gating）+ fresh adaptive main-study → best completed trial 为 **trial 3**（`freeze_layers=3`, `lr=1e-4`, `weight_decay=1e-4`, `dropout=0.25`, `gradient_clip_norm=1.0`）→ `val_acc=0.9468085106382979`, `val_auc=0.9800000000000001`, `val_f1=0.9411764705882353`, `peak_vram=2.17 GiB`, `total_seconds=857.5` → **keep**（较当前 retained keep `CMP-FAIR-V100-RESNEXT-FORMAL-CONFIRM` 的 `0.925531914893617 / 0.9563636363636363` 提升 `0.0212765957446809 / 0.0236363636363638`，也较旧 fair-backbone stable winner `CMP-FAIR-V100-FORMAL-RESNEXT` 的 `0.9148936170212766 / 0.9404545454545454` 提升 `0.0319148936170213 / 0.0395454545454547`；按本轮规则，这是该独立 side campaign 目前最强的新 retained evidence。）
- **Monitor takeaways**：在这轮 4 个 completed trials 里，`freeze_layers=3` 家族显著占优，两个 `freeze_layers=2` / 更高正则角点都落后；更关键的是，trial 0 与 trial 3 被命中了完全相同的 template 标量，却分别得到 `0.925531914893617 / 0.9836363636363636` 与 `0.9468085106382979 / 0.9800000000000001`，说明 gated fusion head 让这条 lane 的 ceiling 显著抬高，但 run-to-run 波动依然存在，且当前最优点就是现有 template 本体而不是新的 scalar 角点。
- **本轮结论**：这是最近连续 discard 序列后第一次明确的正结果。GLU-style fused-token gating 看起来比继续沿 `cross-view mixer / view-recalibration / trim_edge_slices / prenorm / scheduler / pooling / freeze-scalar` 轴小步扫描更有信息量，而且它在不增加显存压力的前提下把该独立 ResNeXt V100 side campaign 的 accuracy ceiling 推到了新高。
- **推荐动作**：若外层 loop 继续，优先对 exact template / commit `15b1ef6` 做 1 次 direct formal confirmation，先判断 `0.9468085106382979` 是否可复现；在此之前不要急着再开新的 fresh main-study。

---

## 2026-04-18：ResNeXt V100 Direct Formal Confirmation（exact gated head template）

> **独立 campaign 说明**
> - 这一轮是外层 loop 显式要求 continuation 下的单轮 coordinator 迭代，继续承接 `fair_backbone_compare` stable-winner side campaign，只推进 `resnext`，不回到 canonical ResUNet 主线。
> - 保持 dedicated ResNeXt V100 lane 的当前 committed 结构与几何完全不变：`image_size=256`, `num_slices_per_view=8`, `feature fusion`, `share_backbone=false`, `use_attention_pooling=false`（mean pooling），`fusion_hidden_dim=256`，并保留 `trim_edge_slices=2`、`LayerNorm(fused_dim)` prenorm、per-view feature recalibration、light cross-view mixer 与 GLU-gated fusion head。
> - 唯一动作：不再开 fresh main-study，只对 exact template / commit `15b1ef6` 做 1 次 direct formal confirmation。为避免覆盖历史产物，本轮仅把运行标识隔离到 `experiment_name=autoresearch_formal_resnext_v100_confirm_iter_0020_20260418_123256`、`output_dir=runs/autoresearch_formal_resnext_v100_confirm/iter_0020_20260418_123256`；训练提交为 `350cc9f`，实际训练仍使用同一组模型与优化超参。

- [x] **CMP-FAIR-V100-RESNEXT-FORMAL-GATED-HEAD-CONFIRM**：`configs/autoresearch_formal_resnext_v100.yaml`（exact `15b1ef6` gated-head template，无任何模型/超参改动）→ `val_acc=0.9468085106382979`, `val_auc=0.9777272727272728`, `val_f1=0.9411764705882353`, `peak_vram=2.17 GiB`, `total_seconds=815.9` → **discard**（按严格 ledger 规则，它与当前 retained keep `CMP-FAIR-V100-RESNEXT-MAIN-GATED-HEAD` 的 `val_acc=0.9468085106382979` 完全持平，但 `val_auc` 比 `0.9800000000000001` 低 `0.0022727272727273`，因此不能替换当前 best keep；不过它也较旧的 direct-formal keep `CMP-FAIR-V100-RESNEXT-FORMAL-CONFIRM` 的 `0.925531914893617 / 0.9563636363636363` 提升 `0.0212765957446809 / 0.0213636363636365`，说明 gated fusion head 的高点并非只存在于 fresh main-study 的单次高波动里，而是在 direct formal 语义下也能复现。）
- **Monitor takeaways**：这次 direct formal 的 `val_acc` 与上一轮 fresh main-study winner 完全一致，`val_f1` 也一致，只是 `val_auc` 轻微回落 `0.0022727272727273`。结合本轮日志尾部可见，训练在 epoch 9 时已经打到 `val_acc=0.9255 / val_auc=0.9782`，最终 best 进一步上探到 `0.9468 / 0.9777`，说明 gated head 带来的 ceiling 提升不是单纯来自 Optuna 搜索或 lucky duplicate trial，而是可以在 direct formal 语义下重现。
- **本轮结论**：按“是否替换当前 best keep”的 strict rule，本轮记 **discard**；但从研究判断上，它是一次偏正面的 confirmation。当前更可靠的结论是：`gated fusion head` 已经把这条 dedicated ResNeXt V100 `256x8` mean-pooling lane 的 ceiling 稳定抬到 `0.9468085106382979` 档位，只是仍需用 alternate seed 判断其多 seed 稳定性。
- **推荐动作**：若外层 loop 继续，优先对同一 exact gated-head template 做 1 次 alternate-seed direct formal confirmation（优先 `seed=123`，次选 `seed=456`）；在此之前不要急着重开 fresh adaptive main-study，也不要回到非 `resnext` backbone 或更换 pooling 几何。

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
> **2026-04-21 主线追加约束**
> - canonical 主线下一阶段不再先做更激进的 weighting-ratio / gating 优化
> - 必须先完成一轮 matched **模块贡献分析**，回答“当前 decision-fusion 路径里到底是哪一层在提供净收益，哪一层在引入方差”
> - 模块分析至少优先覆盖：`equal-weight` 控制、`minimal learned` 路径、`current learned` 路径；必要时再继续拆 richer reliability path
> - 只有当模块分析显示 learned weighting 某一子路径确有正贡献时，才允许继续做训练中自动优化 weighting 比例
>
> **执行顺序**
> 1. 先围绕 canonical `decision fusion` 做 matched 模块贡献分析，优先比较 `equal-weight / minimal learned / current learned`
> 2. 对模块分析中有净收益的路径，再启动 fresh Optuna study 做超参搜索或轻量结构细化
> 3. 由 `monitor_optuna.py` 汇总 best trial、warning、degraded trial 与建议
> 4. candidate winner 明显更优后，再做 main/formal 确认
>
> **说明**
> - 历史 `VR-01 ~ VR-16`、`no_miss_*`、`192x16` 记录保留供回顾，但不再作为当前主线的直接 comparator
> - 若需要复盘旧阶段，请显式标注为 legacy campaign

### 2026-04-21：Canonical Mainline 研究顺序重排（模块贡献优先）

- [ ] **MAIN-MODULE-CONTRIB-01**：对当前 canonical `ResUNet + AttentionPooling + decision fusion` 主线做 matched 模块贡献分析，先回答 `equal-weight`、`minimal learned`、`current learned` 三档的净贡献与方差差异，再决定是否继续拆 richer reliability module。
- [ ] **MAIN-WEIGHT-OPT-01**：仅在 `MAIN-MODULE-CONTRIB-01` 给出明确正信号后，才进入训练中自动优化 weighting 比例；若模块分析未显示净收益，则维持更简单的 weighting baseline，不把“更复杂 gating”当默认方向。

### 2026-04-13：一轮基线校准 + capped fresh proxy Optuna + formal confirmation

- [x] **Baseline rerun**：`configs/autoresearch_proxy.yaml` 当前 canonical baseline（未加结构改动）→ `val_acc=0.8404255319148937`, `val_auc=0.9181818181818182`, `val_f1=0.8192771084337349`, `peak_vram=14.4 GiB`。
- [x] **VRG-LOGIT-01**：将 decision fusion 的 reliability head 从 `Linear -> Sigmoid` 改为 `LayerNorm -> Linear` raw logits（commit `a60c3e0`）→ fresh proxy Optuna study 在人工 GPU 切换后于 3090 上重启；收口时保留 4 个 completed trial，忽略已中断的第 5 个 running trial。
- [x] **VRG-LOGIT-01 / winner**：best completed trial 为 **trial 0**（当前模板超参：`batch_size=4`, `dropout=0.3`, `lr=5e-5`, `weight_decay=1e-3`, `augmentation=true`, `scheduler=none`, `label_smoothing=0.0`, `gradient_clip_norm=1.0`, `early_stopping_patience=2`）→ `val_acc=0.8829787234042553`, `val_auc=0.9318181818181819`, `val_f1=0.8764044943820225` → **keep**。
- [x] **Monitor takeaways**：当前结构对超参较敏感；在已完成 trial 中，`batch_size=4`、`dropout=0.3`、`label_smoothing=0.0`、`scheduler=none`、`weight_decay=1e-3` 明显优于小 batch / `cosine` / 更高 `label_smoothing`。
- [x] **Formal confirmation**：`c30fcae` 将 `configs/autoresearch_formal.yaml` 对齐到 proxy winner 模板超参后完成 1 次 formal（best epoch=4，early stop at epoch 6）→ `val_acc=0.8617021276595744`, `val_auc=0.9372727272727273`, `val_f1=0.8505747126436781`, `peak_vram=14.4 GiB` → **keep**（记为当前 canonical formal reference；`val_acc` 低于 proxy winner `0.0213`，但 `val_auc` 提升 `0.0055`）。
- [x] **Main-sync ledger note**：后续并入 `main@a94350a` 时，补齐了 baseline-only smoke `09531b1`（`val_acc=0.8404255319148937`, `val_auc=0.9181818181818182`）与 3090 五试次 r5 proxy winner `487dbed`（`val_acc=0.851063829787234`, `val_auc=0.9259090909090909`）两条 ledger 记录；它们已保留在 `results.tsv` 中作为 merged history，但均未超过当前本地 training winner `a60c3e0` 的 `0.8829787234042553`。
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
- [x] **CVT-MAN-14**：winner-template stability probe（`seed=456`, `weight_decay=9e-4`; commit `5c45a03`）→ `val_acc=0.8404255319148937`, `val_auc=0.8763636363636363`, `val_f1=0.810126582278481`, `peak_vram=14.4 GiB` → **discard**（较 current proxy winner `0.8829787234042553` 低 `0.0425531914893616`，但与 `CVT-MAN-10` 的 exact `seed=456` rerun accuracy 完全持平，且 AUC 回升 `0.0045454545454545`；同时较 `CVT-MAN-13` 的 `8.75e-4` 点补回 `0.0106382978723405` accuracy，却丢掉 `0.0227272727272727` AUC。说明 `9e-4` 是当前 `seed=456` / upper-side `weight_decay` 细化里唯一把 accuracy 拉回 baseline-level 的折中点，但其排序收益有限、且与 current proxy winner 仍有明显差距；下一步应停止继续细插 `seed=456` 的 L2 轴，转去验证该折中是否能在其他 alternate seed 上复现。）
- [x] **CVT-MAN-15**：winner-template stability probe（`seed=123`, `weight_decay=9e-4`; commit `3093f7f`）→ `val_acc=0.8404255319148937`, `val_auc=0.9209090909090909`, `val_f1=0.8148148148148148`, `peak_vram=14.4 GiB` → **discard**（较 current proxy winner `0.8829787234042553` 低 `0.0425531914893616`；与 `CVT-MAN-06` 的 exact `seed=123` rerun accuracy 完全持平，但 AUC 回升 `0.0031818181818182`；与 `CVT-MAN-14` 的 `seed=456`, `weight_decay=9e-4` 对照点 accuracy 持平，但 AUC 高 `0.0445454545454546`。说明把 `weight_decay` 从 `1e-3` 下调到 `9e-4` 在 `seed=123` 上只修复了少量排序质量，没有恢复 winner-level accuracy；结合 `seed=456` 侧同样仅能回到 baseline-level accuracy，可判定 alternate-seed L2 rescue 已缺乏继续投入价值，后续应转入新的训练稳定化或结构方向。）

- [x] **CVT-STRAT-01**：winner-template optimization-stability probe（`seed=42`, `gradient_clip_norm=0.5`; commit `e3184b6`; 其余固定 `batch_size=4`, `dropout=0.3`, `lr=5e-5`, `weight_decay=1e-3`, `augmentation=true`, `scheduler=none`, `label_smoothing=0.0`, `early_stopping_patience=2`）→ `val_acc=0.8191489361702128`, `val_auc=0.89`, `val_f1=0.8316831683168316`, `peak_vram=14.4 GiB` → **discard**（较 current proxy winner `0.8829787234042553` 低 `0.0638297872340425`，较 canonical baseline rerun `0.8404255319148937` 也低 `0.0212765957446809`；并与 `CVT-MAN-03R` 的 `patience=3` 失败点完全同分。说明把 `gradient_clip_norm` 从 `1.0` 强化到 `0.5` 会显著过度约束当前 winner 模板的优化，未能换来更好的稳定性，应停止继续向更强 clipping 推进。）
- [x] **CVT-STRAT-02**：winner-template optimization-stability probe（`seed=42`, `gradient_clip_norm=0.75`; commit `5402471`; 其余固定 `batch_size=4`, `dropout=0.3`, `lr=5e-5`, `weight_decay=1e-3`, `augmentation=true`, `scheduler=none`, `label_smoothing=0.0`, `early_stopping_patience=2`）→ `val_acc=0.8191489361702128`, `val_auc=0.8909090909090909`, `val_f1=0.8316831683168316`, `peak_vram=14.4 GiB` → **discard**（较 current proxy winner `0.8829787234042553` 低 `0.0638297872340425`，较 canonical baseline rerun `0.8404255319148937` 也低 `0.0212765957446809`；与 `CVT-STRAT-01` 的 `gradient_clip_norm=0.5` 相比 accuracy / F1 完全相同，仅 AUC 微升 `0.0009090909090909`。说明更温和的 clip 中间点仍未把优化拉回 winner-level 轨道，当前 `1.0` 更像可用下界而非可继续下探的冗余余量。）
- [x] **CVT-STRAT-03**：winner-template optimization-stability upper-side probe（`seed=42`, `gradient_clip_norm=1.25`; commit `57bd81c`; 其余固定 `batch_size=4`, `dropout=0.3`, `lr=5e-5`, `weight_decay=1e-3`, `augmentation=true`, `scheduler=none`, `label_smoothing=0.0`, `early_stopping_patience=2`）→ `val_acc=0.8191489361702128`, `val_auc=0.905`, `val_f1=0.8`, `peak_vram=14.4 GiB` → **discard**（较 current proxy winner `0.8829787234042553` 低 `0.0638297872340425`，较 canonical baseline rerun `0.8404255319148937` 也低 `0.0212765957446809`；与 `CVT-STRAT-01/02` 相比 accuracy 完全相同，但 AUC 分别回升 `0.015` / `0.0140909090909091`，同时 F1 降到 `0.8`。说明继续放松 clipping 只能修复部分排序质量，无法把阈值决策拉回 winner-level，因此 `gradient_clip_norm=1.0` 可视为当前 winner 模板的 clipping sweet spot，clipping 轴到此结束。）
- [x] **CVT-REG-01**：winner-template regularization upper-side probe（`seed=42`, `weight_decay=1.25e-3`; commit `6fd3d06`; 其余固定 `batch_size=4`, `dropout=0.3`, `lr=5e-5`, `augmentation=true`, `scheduler=none`, `label_smoothing=0.0`, `gradient_clip_norm=1.0`, `early_stopping_patience=2`）→ `val_acc=0.8191489361702128`, `val_auc=0.8927272727272728`, `val_f1=0.8316831683168316`, `peak_vram=14.4 GiB` → **discard**（较 current proxy winner `0.8829787234042553` 低 `0.0638297872340425`，较 canonical baseline rerun `0.8404255319148937` 也低 `0.0212765957446809`；与 `CVT-STRAT-03` 相比 accuracy 完全相同，但 AUC 低 `0.0122727272727272`、F1 高 `0.0316831683168316`。说明继续增强 L2 正则无法保住 winner-level accuracy，当前 `weight_decay=1e-3` 基本可视为 winner 模板的 upper-side sweet spot，纯标量正则微调应降级。）
- [x] **CVT-VRG-HEAD-01**：winner-template gate-structure probe（`seed=42`; 把每个 VRG reliability head 从 `LayerNorm(feature) -> Linear(512,1)` 改为接收 `concat(view feature, view logits)` 的小型 MLP；commit `5e33d3e`; 其余固定 `batch_size=4`, `dropout=0.3`, `lr=5e-5`, `weight_decay=1e-3`, `augmentation=true`, `scheduler=none`, `label_smoothing=0.0`, `gradient_clip_norm=1.0`, `early_stopping_patience=2`）→ `val_acc=0.8191489361702128`, `val_auc=0.8281818181818182`, `val_f1=0.7848101265822784`, `peak_vram=14.4 GiB` → **discard**（较 current proxy winner `0.8829787234042553` 低 `0.0638297872340425`，较 canonical baseline rerun `0.8404255319148937` 也低 `0.0212765957446809`；与同 accuracy 的 `CVT-REG-01` 相比，AUC 低 `0.0645454545454546`、F1 低 `0.0468730417345532`。说明显式类别证据并没有修复 gate，反而进一步损伤排序质量，结构瓶颈更可能在视角交互不足而不是 reliability head 输入缺失。）
- [x] **CVT-XVIEW-01**：winner-template cross-view context probe（`seed=42`; 在 slice-pooling 后对 3 个 `512D` view features 加 1 层轻量 residual `CrossViewAttention`（4 heads, 1 layer, dropout=0.1），再送入现有 per-view classifier + raw-logit VRG；commit `5b83804`; 其余固定 `batch_size=4`, `dropout=0.3`, `lr=5e-5`, `weight_decay=1e-3`, `augmentation=true`, `scheduler=none`, `label_smoothing=0.0`, `gradient_clip_norm=1.0`, `early_stopping_patience=2`）→ `val_acc=0.851063829787234`, `val_auc=0.9036363636363637`, `val_f1=0.8205128205128205`, `peak_vram=14.5 GiB` → **discard**（较 current proxy winner `0.8829787234042553` 低 `0.0319148936170213`，但较 canonical baseline rerun `0.8404255319148937` 高 `0.0106382978723403`，也较 `CVT-VRG-HEAD-01` 回升 `0.0319148936170212`；不过 `val_auc` 仍较 canonical baseline 低 `0.0145454545454545`。说明跨视角上下文本身可能能补回部分 accuracy，但当前 Transformer-style mixer 仍明显扰动排序质量，信息量足够支持继续沿更低容量的 token-mixing 方向细化，而不是回到 gate head-only 改动。）
- [x] **CVT-XVIEW-02**：winner-template lower-capacity cross-view context probe（`seed=42`; 保留 current winner + raw-logit VRG，把 `CVT-XVIEW-01` 的 1 层 residual `CrossViewAttention` 改为 residual mean-context bottleneck MLP，只在 3 个 pooled `512D` view features 间注入少量上下文后再送入现有 per-view classifier + VRG；commit `d617544`）→ `val_acc=0.8297872340425532`, `val_auc=0.824090909090909`, `val_f1=0.7948717948717948`, `peak_vram=14.4 GiB`, `total_seconds=603.2` → **discard**（较 current proxy winner `0.8829787234042553` 低 `0.0531914893617021`，较 canonical baseline rerun `0.8404255319148937` 也低 `0.0106382978723405`，并较 `CVT-XVIEW-01` 再低 `0.0212765957446808`；`val_auc` 也较 canonical baseline 低 `0.0940909090909092`。说明把 attention 换成更低容量的 mean-context mixing 后，`CVT-XVIEW-01` 的局部 accuracy 回升并没有保住，反而表明“直接改写 pooled 类别特征”这条路径本身不稳；若继续沿跨视角方向，应把上下文限制在 VRG reliability estimation，而不是继续作用在 per-view classifier 输入上。）
- [x] **CVT-XVIEW-03**：winner-template context-for-gating-only probe（`seed=42`; 保留 current winner 的 per-view classifier 完全不接收 cross-view mixing，只给 raw-logit VRG reliability heads 增加 additive other-view mean bias 分支，让跨视角信息只影响 fusion weighting；commit `5bada64`）→ `val_acc=0.7659574468085106`, `val_auc=0.86`, `val_f1=0.725`, `peak_vram=14.4 GiB`, `total_seconds=460.8` → **discard**（较 current proxy winner `0.8829787234042553` 低 `0.1170212765957447`，较 canonical baseline rerun `0.8404255319148937` 也低 `0.0744680851063831`，并较 `CVT-XVIEW-02` 再低 `0.0638297872340426`；虽然 `val_auc` 较 `CVT-XVIEW-02` 回升 `0.035909090909091`，但仍较 canonical baseline 低 `0.0581818181818182`。说明把 other-view mean 仅作为 gating bias 仍会显著破坏准确率，跨视角信息至少不能以这种低容量均值上下文形式直接校准 VRG；`CVT-XVIEW-01` 的局部收益更可能来自 richer token interaction，或只是额外非线性容量带来的偶然回升。）
- [x] **CVT-SELF-01**：winner-template matched-capacity control probe（`seed=42`; 保留 current winner 的 raw-logit VRG，不做任何 cross-view mixing，只在 per-view classifier 前加入一个参数量与 `CVT-XVIEW-02` 接近的 residual self-MLP；commit `3ef1511`; 其余固定 `batch_size=4`, `dropout=0.3`, `lr=5e-5`, `weight_decay=1e-3`, `augmentation=true`, `scheduler=none`, `label_smoothing=0.0`, `gradient_clip_norm=1.0`, `early_stopping_patience=2`）→ `val_acc=0.8297872340425532`, `val_auc=0.8513636363636363`, `val_f1=0.8048780487804879`, `peak_vram=14.4 GiB`, `total_seconds=600.6` → **discard**（较 current proxy winner `0.8829787234042553` 低 `0.0531914893617021`，较 canonical baseline rerun `0.8404255319148937` 也低 `0.0106382978723405`，并较 `CVT-XVIEW-01` 再低 `0.0212765957446808`；虽然较 `CVT-XVIEW-02` 回升 `0.0272727272727273` AUC，但仍较 canonical baseline 低 `0.0668181818181819`。说明单纯增加 matched-capacity per-view 非线性容量不能解释 `CVT-XVIEW-01` 的局部 accuracy 回升；如果继续沿跨视角方向，应保留真实 token interaction，同时进一步限制其残差混合幅度。）
- [x] **CVT-XVIEW-04**：winner-template residual-gated cross-view probe（`seed=42`; 回到 `CVT-XVIEW-01` 的真实 cross-view token interaction，但只在 classifier 分支使用 1 层 residual `CrossViewAttention`（4 heads, 1 layer, dropout=0.1）且固定 `residual_scale=0.25`，VRG reliability head 保持 baseline raw-logit pooled-feature 路径；commit `1a318ad`）→ `val_acc=0.851063829787234`, `val_auc=0.9027272727272727`, `val_f1=0.8292682926829268`, `peak_vram=14.5 GiB`, `total_seconds=596.0` → **discard**（较 current proxy winner `0.8829787234042553` 低 `0.0319148936170213`，但较 canonical baseline rerun `0.8404255319148937` 高 `0.0106382978723403`；accuracy 与 `CVT-XVIEW-01` 完全持平，但 AUC 低 `0.000909090909091`。说明弱化 residual 注入幅度、并把 VRG 保持在 baseline 路径，并没有实质改善 cross-view attention 的排序质量；下一步更值得约束“交互通道秩”而不只是交互幅度。）
- [x] **CVT-XVIEW-05**：winner-template bottlenecked cross-view probe（`seed=42`; 保留 `CVT-XVIEW-04` 的 classifier-only cross-view 路径和 baseline raw-logit VRG，但先把每个 `512D` pooled view token 投影到 `128D` bottleneck 后做 1 层 attention，再投影回 per-view classifier 输入；commit `9690f2e`）→ `val_acc=0.7978723404255319`, `val_auc=0.9254545454545454`, `val_f1=0.7865168539325843`, `peak_vram=14.4 GiB`, `total_seconds=596.5` → **discard**（较 current proxy winner `0.8829787234042553` 低 `0.0851063829787234`，较 canonical baseline rerun `0.8404255319148937` 也低 `0.0425531914893618`；但 `val_auc` 较 `CVT-XVIEW-04` 回升 `0.0227272727272727`，也较 canonical baseline rerun 高 `0.0072727272727272`。说明“低秩真实 token interaction”并非完全无效，它修复了排序质量，却在 `128D` 这个过窄通道上明显损伤了阈值准确率；若继续沿该方向，应优先测试更宽的中等秩 bottleneck，而不是继续压缩。）
- [x] **CVT-XVIEW-06**：winner-template medium-rank bottleneck probe（`seed=42`; 保留 `CVT-XVIEW-04/05` 的 classifier-only cross-view 路径和 baseline raw-logit VRG，但把 pooled `512D` view token 先投影到 `256D` bottleneck 后做 1 层 attention，再投影回 per-view classifier 输入；commit `5c52f70`）→ `val_acc=0.8404255319148937`, `val_auc=0.9104545454545454`, `val_f1=0.810126582278481`, `peak_vram=14.4 GiB`, `total_seconds=594.9` → **discard**（较 current proxy winner `0.8829787234042553` 低 `0.0425531914893616`；与 canonical baseline rerun accuracy 完全持平，但 AUC 低 `0.0077272727272728`。较 `CVT-XVIEW-05` 回升 `0.0425531914893618` accuracy，却回吐 `0.015` AUC；较 `CVT-XVIEW-04` 则补回 `0.0077272727272727` AUC，但又损失 `0.0106382978723403` accuracy。说明 `256D` 只是把 `128D` 的 trade-off 拉回中间态，没有形成新的 accuracy 峰值；若继续沿 bottleneck-rank 轴，应优先测试更靠近 full-rank 的 `384D`，否则可以考虑结束此轴。）
- [x] **CVT-XVIEW-07**：winner-template high-rank bottleneck probe（`seed=42`; 保留 `CVT-XVIEW-04~06` 的 classifier-only cross-view 路径和 baseline raw-logit VRG，但把 pooled `512D` view token 先投影到 `384D` bottleneck 后做 1 层 attention，再投影回 per-view classifier 输入；commit `7d9fad9`）→ `val_acc=0.8191489361702128`, `val_auc=0.875`, `val_f1=0.7671232876712328`, `peak_vram=14.4 GiB`, `total_seconds=596.3` → **discard**（较 current proxy winner `0.8829787234042553` 低 `0.0638297872340425`，较 canonical baseline rerun `0.8404255319148937` 也低 `0.0212765957446809`；相对 `CVT-XVIEW-06` accuracy 再低 `0.0212765957446809`、`val_auc` 再低 `0.0354545454545454`，也同时低于 `CVT-XVIEW-04` 的 `0.851063829787234 / 0.9027272727272727`。说明 `384D` 高秩 bottleneck 没有把性能拉回 full-rank 控制点，反而在 accuracy 与 ranking 上同时退化；由此可结束 bottleneck-rank 轴。）
- [x] **CVT-XVIEW-08**：winner-template minimal-residual full-rank cross-view probe（`seed=42`; 结束 bottleneck-rank 轴，回到 `CVT-XVIEW-04` 的 classifier-only full-rank `CrossViewAttention` + baseline raw-logit VRG，但把 `residual_scale` 从 `0.25` 再下调到 `0.125`；commit `bdf1ef3`）→ `val_acc=0.8404255319148937`, `val_auc=0.8859090909090909`, `val_f1=0.8051948051948052`, `peak_vram=14.5 GiB`, `total_seconds=610.6` → **discard**（较 current proxy winner `0.8829787234042553` 低 `0.0425531914893616`；accuracy 与 canonical baseline rerun 持平，但 `val_auc` 低 `0.0322727272727273`；相对 `CVT-XVIEW-04` 也低 `0.0106382978723403` accuracy 与 `0.0168181818181818` AUC。说明 minimal-residual full-rank 注入同样无法保住此前的局部 accuracy 回升，因此 classifier-side XVIEW 方向到此结束。）
- [x] **CVT-VRG-CTX-01**：winner-template attention-for-gating-only probe（`seed=42`; 保留当前 winner 的 per-view classifier logits 完全走 baseline 路径，只在 reliability estimation 分支对 3 个 pooled `512D` view features 加 1 层轻量 full-rank `CrossViewAttention`，并以弱残差 context 驱动 baseline raw-logit confidence heads；commit `eb8e64b`）→ `val_acc=0.8404255319148937`, `val_auc=0.894090909090909`, `val_f1=0.8192771084337349`, `peak_vram=14.5 GiB`, `total_seconds=623.5` → **discard**（较 current proxy winner `0.8829787234042553` 低 `0.0425531914893616`；accuracy 与 canonical baseline rerun 持平，但 `val_auc` 低 `0.0240909090909092`。相对 `CVT-XVIEW-03` 的 mean-bias gating-only 失败点，accuracy 回升 `0.0744680851063831`、`val_auc` 回升 `0.034090909090909`，说明 richer attention context 确实更稳；但它仍只回到 baseline-level accuracy，没有任何净收益，因此新的 cross-view 结构线到此一起收束。）

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

---

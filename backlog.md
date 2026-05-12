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

## 2026-05-10：人类主线追加（weight collapse → minimal internal structure repair）

> **任务说明**
> - 人类要求把当前 Auto Research 主线明确补上：**针对本项目特点，decision-fusion 中产生了权重坍塌**；后续改进应围绕这个要点，在网络内部处理中增加最小结构改动，以提高准确率。
> - 本项目的具体坍塌形态不是抽象 regularization 问题，而是 `DFR-25` telemetry 暴露出的 residual axial lock-in：虽然 `DFR-25` 已经以 3-seed mean `val_acc=0.9397163120567376` 明确超过 matched equal-weight mean `0.9219858156028368`，但 `seed42/456` 仍保持 `top_weight axial=94/94`；`DFR-44` 进一步证明直接把 detached per-view evidence 注入 gate 会产生 noisy routing，不能稳定提高准确率。
> - 因此新的主线记录为：**从 DFR-25 fork，只做 gate 内部、低容量、可开关的反坍塌结构修复**；不改 backbone family、不改 `256x8` geometry、不改 decision-fusion 语义、不改 DFR-25 winning scalar。

> **自检**
> - 这项改动是否直接帮助 decision fusion 还原各视角应有作用？是。它直接作用于 learned reliability logits 的过度集中问题，限制单个视角在 softmax 前形成近 hard-selection 的尺度优势，目标是让 coronal/sagittal 在已有证据时不被 gate-logit 尺度压死。
> - 如果成功，为什么有机会把 learned full-fusion `val_acc` 推到或保持在 matched equal-weight 之上？因为它不把权重固定平均，也不改 view classifier；排序仍由 learned gate 决定，只削掉过度自信的尾部坍塌。预期是保留 DFR-25 的 learned routing 收益，同时降低 `seed42/456` 的 residual axial lock-in，使 non-axial view 在少量关键样本上进入决策。
> - 它是可复用 learned weighting 机制还是固定权重？是 learned weighting 机制。`GateLogitRMSLimiter` 只约束 gate-logit 相对尺度，不指定 axial/coronal/sagittal 的固定比例。

- [x] **DFR-45-RESNEXT-DECISION-256X8-GATE-LOGIT-RMS-LIMIT-MAIN-S42**：commit `ed6561c`，在 [src/model.py](/dataset/HH/ankle-ct/src/model.py) 新增 `GateLogitRMSLimiter`，通过 `ANKLE_DECISION_ENABLE_GATE_LOGIT_RMS_LIMIT=1` 与 `ANKLE_DECISION_GATE_LOGIT_MAX_CENTERED_RMS=1.3` 开启；运行配置为 [configs/cmp_resnext_decision_256x8_dfr45_gate_logit_rms_limit_formal_s42.yaml](/dataset/HH/ankle-ct/configs/cmp_resnext_decision_256x8_dfr45_gate_logit_rms_limit_formal_s42.yaml)。设计思路：只在 reliability logits softmax 前限制 centered RMS，保留 mean logit 与 view ranking，减少本项目中 axial gate logit 尺度过大造成的权重坍塌。预计改进效果：seed42 的 `top_weight axial` 不应再机械保持 `94/94`，但 mean axial weight 仍可保持主导；理想形态是少量 non-axial evidence 样本获得 top/near-top gate weight，从而在不退回 equal-weight 的前提下守住或超过 DFR-25 seed42 `val_acc=0.9361702127659575`。实际结果：Slurm job `481787` 在 `RTXA6Kq/node16` 完成，best epoch `9`，`val_acc=0.9148936170212766`, `val_auc=0.9772727272727273`, `val_f1=0.9111111111111111`, `peak_vram≈2.15 GiB`, `total_seconds≈1050.0` → **discard**（低于 DFR-25 seed42 `0.9361702127659575 / 0.9786363636363636 / 0.9333333333333333`，也低于 matched equal-weight seed42 `0.9255319148936170` 的主指标。）
- **fusion-weight 结论**：[fusion_weight_analysis.json](/dataset/HH/ankle-ct/runs/resnext_decision_256x8_mainline/dfr45_gate_logit_rms_limit_formal_s42/fusion_weight_analysis.json) 显示 DFR-45 没有解决本项目的权重坍塌。mean fusion weight 虽从 DFR-25 seed42 的近 hard axial 稍降到 `axial/coronal/sagittal = 0.8801514861431528 / 0.05246829055249691 / 0.06738022148133592`，但 `top_weight_view_distribution` 仍是 `axial=94, coronal=0, sagittal=0`；也就是说，RMS limiter 只是压缩了 logit 尺度，没有改变 routing ordering。
- **当前判断**：DFR-45 是机制上合格但结果失败的最小内部结构修复。它证明“只限制 gate-logit concentration、保留原始 view ranking”不足以把 non-axial evidence 推入决策；下一轮若继续做最小网络改动，应直接减少 view-specific gate bias 或让 reliability scorer 在三视角之间共享判别规则，而不是继续单纯压 softmax 尺度。

> **DFR-46 自检**
> - 这项改动是否直接帮助 decision fusion 还原各视角应有作用？是。DFR-45 表明单纯压缩 confidence logits 不会改变 axial top-rank；DFR-46 改成三视角共享同一个 confidence head，让 gate 不能再靠每个视角各自的 head 参数学出固定 axial bias，只能用视角 pooled feature 本身的证据差异来排序。
> - 如果成功，为什么有机会把 learned full-fusion `val_acc` 推到或保持在 matched equal-weight 之上？因为它仍然是 sample-wise learned weighting，不固定平均，也不注入 noisy teacher evidence；它只移除 view-specific reliability scorer 的自由度，预期能降低 seed42 的 axial lock-in，同时保留 DFR-25 的 dominant-gate dropout 训练收益。
> - 它是可复用 learned weighting 机制还是固定权重？是 learned weighting 机制。shared head 仍对每个样本和每个视角分别输出 reliability logit，只是三视角共享打分规则。

- [x] **DFR-46-RESNEXT-DECISION-256X8-SHARED-CONFIDENCE-HEAD-MAIN-S42**：commit `ba1c09d`，在 [src/model.py](/dataset/HH/ankle-ct/src/model.py) 新增 `ANKLE_DECISION_USE_SHARED_CONFIDENCE_HEAD=1`，使 non-equal learned decision fusion 的三个 reliability heads 变为一个共享 `LayerNorm + Linear` scorer；运行配置为 [configs/cmp_resnext_decision_256x8_dfr46_shared_confidence_head_formal_s42.yaml](/dataset/HH/ankle-ct/configs/cmp_resnext_decision_256x8_dfr46_shared_confidence_head_formal_s42.yaml) 与当前 `configs/autoresearch_formal*.yaml`。设计思路：最小化 gate 内部结构改动，直接削弱 view-specific scorer 学出的固定 axial logit bias，而不是改变 classifier、backbone、融合语义或训练标量。预计改进效果：seed42 的 `top_weight axial=94/94` 应至少出现少量 coronal/sagittal top-weight 或 near-top routing；若 non-axial expert 的证据质量不足，accuracy 不应明显低于 matched equal-weight seed42。实际结果：Slurm job `481809` 在 `RTXA6Kq/node16` 完成，best `val_acc=0.9255319148936170`, `val_auc=0.9754545454545455`, `val_f1=0.9156626506024096`, `peak_vram≈2.14 GiB`, `total_seconds≈1163.1` → **discard**（主指标只打平 matched equal-weight seed42 `0.9255319148936170`，低于 DFR-25 seed42 `0.9361702127659575`。）
- **fusion-weight 结论**：[fusion_weight_analysis.json](/dataset/HH/ankle-ct/runs/resnext_decision_256x8_mainline/dfr46_shared_confidence_head_formal_s42/fusion_weight_analysis.json) 显示 shared head 仍未解决权重坍塌。mean fusion weight 为 `axial/coronal/sagittal = 0.8947559893131256 / 0.06149472054490383 / 0.04374929251981542`，`top_weight_view_distribution` 仍是 `axial=94, coronal=0, sagittal=0`；虽然 `top_true_margin` 中有 `coronal=4, sagittal=5` 的样本，gate 仍没有一次把 top-weight 给到 non-axial。
- **当前判断**：DFR-46 证明只把 scorer 参数共享还不够；view pooled features 本身仍携带强 view-specific offset，shared linear scorer 仍能把 axial 排在所有样本 top。下一轮最小结构修复应进一步让 gate scorer 看 **relative view feature**（例如 `feature_i - mean(feature_all)`）或直接在 scorer 前做跨视角去均值，使 gate 排序更依赖样本内相对证据，而不是每个视角固定分布。

> **DFR-47 自检**
> - 这项改动是否直接帮助 decision fusion 还原各视角应有作用？是。DFR-46 表明 gate 的坍塌不只来自 view-specific head 参数，还来自 pooled feature 的固定视角分布 offset；DFR-47 在 reliability path 里加入 `feature_i - mean(feature_all)` 的相对特征 residual，让 gate correction 专门看同一样本内哪个视角相对更有证据。
> - 如果成功，为什么有机会把 learned full-fusion `val_acc` 推到或保持在 matched equal-weight 之上？因为它仍保留 DFR-25 原始 learned logits 和 dominant-gate dropout，只用 identity-initialized 小 residual 修正 routing ordering；预期在 non-axial true-margin 样本上给 coronal/sagittal 少量 top/near-top 机会，同时不把全部样本机械拉平均。
> - 它是可复用 learned weighting 机制还是固定权重？是 learned weighting 机制。relative-view gate 对每个样本、每个视角输出 residual reliability logit，权重仍由 softmax 学出，不指定固定 axial/coronal/sagittal 比例。

- [x] **DFR-47-RESNEXT-DECISION-256X8-RELATIVE-VIEW-GATE-MAIN-S42**：commit `0067869`，在 [src/model.py](/dataset/HH/ankle-ct/src/model.py) 新增 `ANKLE_DECISION_ENABLE_RELATIVE_VIEW_GATE=1`，对 gating features 做 sample-wise 去均值后用低容量、零初始化 residual gate 输出 reliability-logit correction；运行配置为 [configs/cmp_resnext_decision_256x8_dfr47_relative_view_gate_formal_s42.yaml](/dataset/HH/ankle-ct/configs/cmp_resnext_decision_256x8_dfr47_relative_view_gate_formal_s42.yaml) 与当前 `configs/autoresearch_formal*.yaml`。设计思路：最小化 gate 内部结构改动，直接削弱固定视角 feature offset 造成的 axial 排序优势，而不是改变 backbone、classifier、融合语义或训练标量。预计改进效果：seed42 的 `top_weight axial=94/94` 应至少出现少量 non-axial top/near-top routing；若 non-axial feature 质量不足，accuracy 至少应守住 matched equal-weight seed42 附近。实际结果：fresh adaptive `main-study` [runs/optuna_main_autoloop/iter_0001_20260510_200020](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0001_20260510_200020) 的 only completed trial（trial `0`，search-config copy [autoresearch_logs/generated_search_configs/optuna_main_search_iter_0001_20260510_200020.yaml](/dataset/HH/ankle-ct/autoresearch_logs/generated_search_configs/optuna_main_search_iter_0001_20260510_200020.yaml)，runtime env `ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER=1` + `ANKLE_DECISION_TRAIN_DOMINANT_GATE_DROPOUT_PROB=0.25` + `ANKLE_DECISION_ENABLE_RELATIVE_VIEW_GATE=1` + `ANKLE_DECISION_RELATIVE_VIEW_GATE_RESIDUAL_LIMIT=0.75`）→ `val_acc=0.9148936170212766`, `val_auc=0.9704545454545455`, `val_f1=0.9111111111111111`, `peak_vram≈2.15 GiB`, `total_seconds≈1311.3` → **discard**（低于 matched equal-weight seed42 `0.9255319148936170 / 0.9486363636363637 / 0.9176470588235294` 的主指标，也低于 retained `DFR-25/26` seed42 `0.9361702127659575` ceiling。）
- **fusion-weight 结论**：[fusion_weight_analysis.json](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0001_20260510_200020/trials/trial_0000/run/fusion_weight_analysis.json) 显示 DFR-47 不但没有修复 routing ordering，反而比 DFR-46 更接近 hard axial lock-in。mean fusion weight `axial/coronal/sagittal = 0.9949998012248505 / 0.002210335041262566 / 0.0027898584215089364`；`top-weight count/rate axial/coronal/sagittal = 94/1.0000 / 0/0.0000 / 0/0.0000`。同时 `top_true_margin` 分布仍有 `axial/coronal/sagittal = 80 / 4 / 10`，说明至少 `14` 个样本存在 non-axial true-margin signal，但 relative residual gate 没有一次把 top weight 迁移给 non-axial view。
- **当前判断**：DFR-47 否定了“在原始 DFR-25 confidence logits 旁边加一个 relative-feature residual correction 就足够改变排序”的假设。原始 pooled-feature confidence path 的 axial logit bias 仍然压倒 residual correction，且训练会把 residual 推向更尖锐的 axial softmax。下一轮若继续做最小 gate 内部结构修复，不应再做 additive residual；更直接的候选是让 reliability scorer 的主输入本身变成 relative/centered features，或对 raw confidence logits 做显式 per-view de-bias，使原始 axial offset 不能继续作为 dominant path。

> **DFR-48 自检**
> - 这项改动是否直接帮助 decision fusion 还原各视角应有作用？是。DFR-47 说明 additive relative residual 会被原始 axial confidence path 压倒；DFR-48 直接把 reliability scorer / calibrator 的主输入改成 `feature_i - mean(feature_all)`，让 gate 主路径不再直接读取每个视角固定 pooled-feature offset。
> - 如果成功，为什么有机会把 learned full-fusion `val_acc` 推到或保持在 matched `equal-weight` 之上？因为它仍是 sample-wise learned weighting，不固定平均，也不改 view classifier；预期 axial 仍在大多数强证据样本主导，但在 sagittal/coronal relative evidence 更强的少量样本上，top-weight 应从 `axial=94/94` 迁移出一部分，从而把 DFR-25 的 learned routing 收益保住并补回 non-axial 增量。
> - 它是可复用 learned weighting 机制还是固定权重？是 learned weighting 机制。`ANKLE_DECISION_USE_RELATIVE_CONFIDENCE_FEATURES=1` 只改变 confidence scorer 的输入坐标系，softmax 权重仍由样本特征学习得到，不指定三视角固定比例。

- [x] **DFR-48-RESNEXT-DECISION-256X8-RELATIVE-CONFIDENCE-FEATURES-MAIN-S42**：commit `0f37026`，在 [src/model.py](/dataset/HH/ankle-ct/src/model.py) 新增 `ANKLE_DECISION_USE_RELATIVE_CONFIDENCE_FEATURES=1`，让 learned decision-fusion reliability scorer / calibrator 使用 sample-wise centered gating features；运行配置为 [configs/autoresearch_formal.yaml](/dataset/HH/ankle-ct/configs/autoresearch_formal.yaml) 与 fresh search-config copy [autoresearch_logs/generated_search_configs/optuna_main_search_iter_0002_20260510_202918.yaml](/dataset/HH/ankle-ct/autoresearch_logs/generated_search_configs/optuna_main_search_iter_0002_20260510_202918.yaml)，study_root [runs/optuna_main_autoloop/iter_0002_20260510_202918](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0002_20260510_202918)。设计思路：从 DFR-25 fork，只替换 gate confidence 主路径的输入坐标系，直接去掉 fixed view feature offset 对 raw confidence ordering 的主导作用，而不是再叠加 residual correction。预计改进效果：seed42 的 top-weight 不应继续 `axial=94/94`；预期少量 sagittal/coronal true-margin 样本获得 top/near-top routing，mean axial weight 可保持主导但应低于 DFR-47 的近 hard `0.995`，从而有机会让 learned full-fusion 超过 matched equal-weight。实验实际结果：single-GPU adaptive `main-study` only completed trial `0` → `val_acc=0.9148936170212766`, `val_auc=0.9818181818181818`, `val_f1=0.9111111111111111`, `peak_vram≈2.15 GiB`, `total_seconds≈1239.3` → **discard**（主指标低于 matched equal-weight seed42 `0.9255319148936170`，也低于 DFR-25/26 seed42 `0.9361702127659575` ceiling；虽然 AUC 较高，但按协议不能用 AUC 覆盖 val_acc 回落。）
- **fusion-weight 结论**：[fusion_weight_analysis.json](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0002_20260510_202918/trials/trial_0000/run/fusion_weight_analysis.json) 显示 DFR-48 只降低了 axial 平均权重，没有改变 top-weight ordering。mean fusion weight `axial/coronal/sagittal = 0.9172948903225838 / 0.03407463752367395 / 0.0486304713611273`；`top-weight count/rate axial/coronal/sagittal = 94/1.0000 / 0/0.0000 / 0/0.0000`。同时 `top_true_margin` 分布为 `axial/coronal/sagittal = 83 / 0 / 11`，说明 `11` 个 sagittal true-margin 样本仍没有获得 top gate weight。
- **当前判断**：DFR-48 否定了“只把 scorer 主输入换成 centered relative features 就能去除 axial lock-in”的假设。它比 DFR-47 更温和，mean axial weight 从 `0.9950` 降到 `0.9173`，但 confidence ordering 仍在全部样本上选择 axial，accuracy 也回落到 `0.9149`。下一步若继续做最小 gate 内部修复，应转向更显式的 raw confidence per-view de-bias / view-prior removal，或引入极低容量的 contrastive rank correction；不要再只做 feature-space relative residual 或 centered feature main-path 替换。

> **DFR-49 自检**
> - 这项改动是否直接帮助 decision fusion 还原各视角应有作用？是。DFR-48 说明 centered feature path 降低了 axial 平均权重但仍无法改变 raw confidence ordering；DFR-49 直接在 softmax 前减去三视角 centered running confidence-logit prior，让固定 axial reliability offset 不能继续垄断 top-rank。
> - 如果成功，为什么有机会把 learned full-fusion `val_acc` 推到 matched `equal-weight` 之上？因为它仍保留 sample-specific residual confidence logits 与 learned softmax routing，不把三视角机械平均。预期 axial 仍在强 axial-evidence 样本主导，但 sagittal/coronal true-margin 样本应获得 top/near-top 权重迁移，从而把 DFR-25 的 learned routing 收益与弱视角补充收益同时保住。
> - 它是可复用 learned weighting 机制还是固定权重？是 learned weighting 机制。`GateViewPriorDebiaser` 只移除 batch/running centered view prior，剩余 routing 仍由每个样本的 confidence logits 决定，不指定固定 `axial/coronal/sagittal` 比例。

- [x] **DFR-49-RESNEXT-DECISION-256X8-GATE-VIEW-PRIOR-DEBIAS-MAIN-S42**：commit `8b74ea5`，在 [src/model.py](/dataset/HH/ankle-ct/src/model.py) 新增 `GateViewPriorDebiaser`，通过 `ANKLE_DECISION_ENABLE_GATE_VIEW_PRIOR_DEBIAS=1`, `ANKLE_DECISION_GATE_VIEW_PRIOR_DEBIAS_STRENGTH=1.0`, `ANKLE_DECISION_GATE_VIEW_PRIOR_DEBIAS_MOMENTUM=0.1` 开启；运行配置为 [configs/autoresearch_formal.yaml](/dataset/HH/ankle-ct/configs/autoresearch_formal.yaml) 与 fresh search-config copy [autoresearch_logs/generated_search_configs/optuna_main_search_iter_0003_20260510_205859.yaml](/dataset/HH/ankle-ct/autoresearch_logs/generated_search_configs/optuna_main_search_iter_0003_20260510_205859.yaml)，study_root [runs/optuna_main_autoloop/iter_0003_20260510_205859](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0003_20260510_205859)。设计思路：从 DFR-25 fork，只在 reliability logits softmax 前移除三视角 running centered view prior，直接削掉固定 axial logit offset，同时保留样本内 residual logit ranking。预计改进效果：top-weight 不应继续 `axial=94/94`；预期 axial 在强 axial-evidence 样本仍保持最大权重，但部分 sagittal/coronal true-margin 样本应迁移为 top/near-top routing，mean axial weight 应明显低于 DFR-48 且不退化成纯 equal-weight，从而有机会让 learned full-fusion 超过 matched equal-weight。实验实际结果：single-GPU adaptive `main-study` only completed trial `0` → `val_acc=0.8723404255319149`, `val_auc=0.9563636363636364`, `val_f1=0.8461538461538461`, `peak_vram≈2.15 GiB`, `total_seconds≈1226.4` → **discard**（主指标显著低于 matched equal-weight seed42 `0.9255319148936170`，也低于 DFR-25/26 seed42 `0.9361702127659575` ceiling。）
- **fusion-weight 结论**：[fusion_weight_analysis.json](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0003_20260510_205859/trials/trial_0000/run/fusion_weight_analysis.json) 显示 DFR-49 首次真正打破了近期 gate 的 `axial=94/94` top-weight lock-in，但迁移过强且没有转化为正确决策。mean fusion weight `axial/coronal/sagittal = 0.4140417624185694 / 0.31106897619889773 / 0.27488926500874633`；`top-weight count/rate axial/coronal/sagittal = 43/0.4574 / 25/0.2660 / 26/0.2766`。同时 `top_true_margin` 分布为 `axial/coronal/sagittal = 50 / 0 / 44`，`top_weight_hit_rate.true_margin=0.7128`，说明 full-strength prior removal 确实把大量样本路由到 non-axial，但其中相当一部分不是 accuracy-positive routing。
- **当前判断**：DFR-49 给出的是 diagnostic discard。它证明“raw confidence per-view de-bias / view-prior removal”能改变 routing ordering，这比 DFR-45~48 更接近要修的坍塌本体；但 `strength=1.0` 等价于把固定 view prior 全量拿掉，破坏了 axial 在当前数据上的真实强先验与 absolute reliability scale，导致 learned full-fusion 明显低于 equal-weight。下一步若继续这条直接方向，应保留 DFR-49 的 prior-debias 机制但把干预改为更温和或条件化，例如低强度 partial debias（先测 `strength<1`）或只在 non-axial residual evidence 足够接近 axial 时启用；不要再做全量 view-prior subtraction。

> **DFR-50 自检**
> - 这项改动是否直接帮助 decision fusion 还原各视角应有作用？是。DFR-49 已证明 raw confidence per-view prior removal 是近期唯一能改变 top-weight ordering 的杠杆；DFR-50 不换机制，只把 `strength=1.0` 降到 `0.35`，目标是避免全量去先验导致的 non-axial 过度路由，同时给 coronal/sagittal true-margin 样本留下进入 top/near-top 的空间。
> - 如果成功，为什么有机会把 learned full-fusion `val_acc` 推到 matched `equal-weight` 之上？因为 partial debias 仍保留 sample-specific learned reliability logits，不指定固定三视角比例；预期 axial 在多数强证据样本继续主导，但不再像 DFR-48 那样 `94/94` 全部 top-weight axial，也不再像 DFR-49 那样接近均衡分散，从而更接近“强视角主导、弱视角补充”的 routing 形状。
> - 它是可复用 learned weighting 机制还是固定权重？是 learned weighting 机制。`ANKLE_DECISION_GATE_VIEW_PRIOR_DEBIAS_STRENGTH=0.35` 只控制 prior subtraction 幅度，softmax 权重仍由样本内 confidence logits 学出。

- [x] **DFR-50-RESNEXT-DECISION-256X8-PARTIAL-GATE-VIEW-PRIOR-DEBIAS-MAIN-S42**：commit `3b8cac3`，配置只把 `GateViewPriorDebiaser` 的 `ANKLE_DECISION_GATE_VIEW_PRIOR_DEBIAS_STRENGTH` 从 `1.0` 降到 `0.35`，保持 DFR-25 anchor 的 `ResNeXt 256x8 decision + L3-no-mixer + dominant-gate dropout` 与 fixed scalar 不变；运行配置为 [configs/autoresearch_formal.yaml](/dataset/HH/ankle-ct/configs/autoresearch_formal.yaml) 与 fresh search-config copy [autoresearch_logs/generated_search_configs/optuna_main_search_iter_0004_20260510_212811.yaml](/dataset/HH/ankle-ct/autoresearch_logs/generated_search_configs/optuna_main_search_iter_0004_20260510_212811.yaml)，study_root [runs/optuna_main_autoloop/iter_0004_20260510_212811](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0004_20260510_212811)。设计思路：保留 DFR-49 已验证能打破 routing-order lock-in 的 running centered view-prior debias，但只做低强度 partial correction，让 axial 的真实强先验不被完全抹掉。预计改进效果：top-weight 应从 DFR-48 的 `axial=94/94` 迁出少量 coronal/sagittal，但不要重演 DFR-49 的 `43/25/26` 过度分散；理想权重形状是 axial 仍明显多数主导，同时 non-axial true-margin 样本拿到少量 top/near-top routing，使 learned full-fusion 有机会重新超过 matched equal-weight。实验实际结果：single-GPU adaptive `main-study` only completed trial `0` → `val_acc=0.9255319148936170`, `val_auc=0.9622727272727274`, `val_f1=0.9213483146067416`, `peak_vram≈2.15 GiB`, `total_seconds≈1222.9` → **discard**（主指标只打平 matched equal-weight seed42 `0.9255319148936170`，低于 retained DFR-25/26 seed42 `0.9361702127659575`；AUC 也低于 DFR-25/26。）
- **fusion-weight 结论**：[fusion_weight_analysis.json](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0004_20260510_212811/trials/trial_0000/run/fusion_weight_analysis.json) 显示 partial debias 只温和降低了 mean axial mass，没有真正恢复有效 non-axial routing。mean fusion weight `axial/coronal/sagittal = 0.8113602910270082 / 0.11881413876495146 / 0.0698255753897606`；`top-weight count/rate axial/coronal/sagittal = 93/0.9894 / 1/0.0106 / 0/0.0000`。同时 `top_true_margin` 分布为 `axial/coronal/sagittal = 81 / 5 / 8`，说明 `13` 个 non-axial true-margin 样本中只有极少数获得 top gate weight。
- **当前判断**：DFR-50 是机制上合格但结果失败的 partial-debias probe。`strength=0.35` 比 DFR-49 保守得多，避免了 `43/25/26` 的过度迁移，但又退回到几乎 axial lock-in（`93/94`），因此未把 learned full-fusion 推过 matched equal-weight。下一轮若继续 prior-debias family，不应只做另一个盲目中间强度；更直接的方向是 **条件化 debias**，只在 non-axial residual evidence 接近 axial 或 top-true-margin/teacher signal 支持时局部提升 debias，而不是全样本固定 subtraction。
- **Agent 状态（2026-05-10 21:52 SGT）**：本 session 是唯一 coordinator，已完成 exactly one research iteration；last experiment=`DFR-50 partial gate view prior debias`，lane=`main-study`，result=`discard`，last commit=`3b8cac3`；下一步建议=`DFR-51 conditional prior debias / evidence-close trigger`，继续服务 residual axial lock-in 修复，不切 backbone / geometry / fusion family。

> **DFR-51 自检**
> - 这项改动是否直接帮助 decision fusion 还原各视角应有作用？是。DFR-50 显示固定 `strength=0.35` 仍几乎保持 `axial=93/94` top-weight；DFR-51 保留同一 prior-debias 机制，但只在 detached non-axial pred-margin evidence 接近 axial 时把 debias strength 从 `0.35` 连续提高到 `1.0`，目标是专门打开放在 axial lock-in 下面的 evidence-close non-axial 样本。
> - 如果成功，为什么有机会把 learned full-fusion `val_acc` 推到 matched `equal-weight` 之上？因为它不是固定平均权重，也不是全样本强去偏；多数 strong axial 样本仍走 DFR-50 的低强度 prior debias，而 non-axial evidence-close 样本获得更强 anti-lock-in correction。预期 routing 从 DFR-50 的 `93/1/0` 迁移到 axial 仍占多数、sagittal/coronal 有少量 top-weight 的形态，补回 `top_true_margin` 中 non-axial 样本的贡献，从而有机会恢复 DFR-25 learned full-fusion 对 matched equal-weight 的优势。
> - 它是可复用 learned weighting 机制还是固定权重？是 learned weighting 机制。`GateViewPriorDebiaser` 只按 detached evidence-close 程度调节 prior subtraction strength，最终权重仍由 learned confidence logits 经 softmax 得到，不指定固定三视角比例。

- [x] **DFR-51-RESNEXT-DECISION-256X8-EVIDENCE-CLOSE-PRIOR-DEBIAS-MAIN-S42**：commit `5c092a0`，在 [src/model.py](/dataset/HH/ankle-ct/src/model.py) 扩展 `GateViewPriorDebiaser`，新增 `ANKLE_DECISION_GATE_VIEW_PRIOR_DEBIAS_MAX_STRENGTH=1.0`, `ANKLE_DECISION_GATE_VIEW_PRIOR_DEBIAS_EVIDENCE_CLOSE_GAP=0.2`, `ANKLE_DECISION_GATE_VIEW_PRIOR_DEBIAS_EVIDENCE_CLOSE_WINDOW=0.2`；运行配置为 [configs/autoresearch_formal.yaml](/dataset/HH/ankle-ct/configs/autoresearch_formal.yaml) 与 fresh search-config copy [autoresearch_logs/generated_search_configs/optuna_main_search_iter_0005_20260510_215538.yaml](/dataset/HH/ankle-ct/autoresearch_logs/generated_search_configs/optuna_main_search_iter_0005_20260510_215538.yaml)，study_root [runs/optuna_main_autoloop/iter_0005_20260510_215538](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0005_20260510_215538)。设计思路：从 DFR-50 fork，只把 fixed partial prior debias 改为 evidence-close 条件增强；使用 detached per-view pred-margin 作为触发信号，避免把新梯度路径加到 view classifier。预计改进效果：top-weight 应明显离开 DFR-50 的 `axial=93/94`，但不应重演 DFR-49 的近均衡 `43/25/26`；理想形态是 axial 仍多数主导，sagittal/coronal true-margin 样本获得少量 top/near-top routing，使 learned full-fusion 重新超过 matched equal-weight。实验实际结果：single-GPU adaptive `main-study` completed trial `0` → `val_acc=0.9042553191489362`, `val_auc=0.9631818181818181`, `val_f1=0.8965517241379310`, `peak_vram≈2.15 GiB`, `total_seconds≈1152.1` → **discard**（主指标低于 matched equal-weight seed42 `0.9255319148936170`，也低于 retained DFR-25/26 seed42 `0.9361702127659575`。）
- **fusion-weight 结论**：[fusion_weight_analysis.json](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0005_20260510_215538/trials/trial_0000/run/fusion_weight_analysis.json) 显示 DFR-51 确实打破了 DFR-50 的 residual axial lock-in，但过度迁移到 sagittal，且没有转化为 accuracy。mean fusion weight `axial/coronal/sagittal = 0.48874313372591055 / 0.18507649291764403 / 0.3261803809474123`；`top-weight count/rate axial/coronal/sagittal = 60/0.6383 / 1/0.0106 / 33/0.3511`。同时 `top_true_margin` 分布为 `axial/coronal/sagittal = 50 / 0 / 44`，`top_weight_hit_rate.true_margin=0.8617`；routing 与 true-margin 的对齐率提高，但 sagittal branch 单视角 accuracy 只有 `0.4681`，大量迁移仍伤害 full-fusion decision。
- **当前判断**：DFR-51 是一个有诊断价值但结果失败的 conditional-debias probe。它证明 evidence-close trigger 能把 top-weight 从 DFR-50 的 `93/1/0` 推到更真实的 axial+sagittal 混合，但 `gap=0.2/window=0.2/max_strength=1.0` 太激进，等价于把许多 sagittal high-margin but poorly calibrated 样本升为主导，导致 `val_acc` 低于 equal-weight。下一轮若外层 loop 继续该 family，应避免再做同强度条件增强；更合理的是把 correction target 从 pred-margin close 改成 **true-margin proxy 更可靠的 correctness/teacher-aligned close trigger**，或把 max strength 降到中间值并限制 sagittal top-rate，目标是从 DFR-50 的 93/1/0 只迁移少量样本，而不是一次迁到 60/1/33。
- **Agent 状态（2026-05-10 22:28 SGT）**：本 session 是唯一 coordinator，已完成 exactly one research iteration；last experiment=`DFR-51 evidence-close prior debias`，lane=`main-study`，result=`discard`，last commit=`5c092a0`；下一步建议=`DFR-52 teacher-aligned or capped conditional prior debias`，继续服务 residual axial lock-in 修复，不切 backbone / geometry / fusion family。

> **DFR-52 自检**
> - 这项改动是否直接帮助 decision fusion 还原各视角应有作用？是。DFR-51 已证明 evidence-close prior debias 能改变 routing ordering，但 `max_strength=1.0` 过度迁移到 sagittal；DFR-52 只把 conditional debias 上限降到 `0.55`，目标是在 DFR-50 的 axial lock-in 和 DFR-51 的 sagittal over-routing 之间找到低干预区间。
> - 如果成功，为什么有机会把 learned full-fusion `val_acc` 推到 matched `equal-weight` 之上？因为它仍是 sample-wise learned weighting，不固定平均，也不全样本强去偏。预期 axial 在强证据样本继续主导，但少量 coronal/sagittal true-margin 样本获得 top/near-top routing；这种受限迁移若避开 DFR-51 的弱视角过冲，就有机会恢复 DFR-25 learned full-fusion 对 matched equal-weight 的优势。
> - 它是可复用 learned weighting 机制还是固定权重？是 learned weighting 机制。`GateViewPriorDebiaser` 仍只调节 reliability logits 的 prior subtraction strength，最终三视角权重仍由 learned confidence logits 经 softmax 得到。

- [x] **DFR-52-RESNEXT-DECISION-256X8-CAPPED-EVIDENCE-CLOSE-PRIOR-DEBIAS-MAIN-S42**：commit `3422993`，在 [configs/autoresearch_formal.yaml](/dataset/HH/ankle-ct/configs/autoresearch_formal.yaml)、[configs/autoresearch_proxy.yaml](/dataset/HH/ankle-ct/configs/autoresearch_proxy.yaml) 与 Optuna 模板中只把 `ANKLE_DECISION_GATE_VIEW_PRIOR_DEBIAS_MAX_STRENGTH` 从 `1.0` 降到 `0.55`，保持 DFR-51 的 `strength=0.35/gap=0.2/window=0.2` 与 DFR-25 anchor 不变；运行配置为 [autoresearch_logs/generated_search_configs/optuna_main_search_iter_0001_20260510_224652.yaml](/dataset/HH/ankle-ct/autoresearch_logs/generated_search_configs/optuna_main_search_iter_0001_20260510_224652.yaml)，study_root [runs/optuna_main_autoloop/iter_0001_20260510_224652](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0001_20260510_224652)。设计思路：从 DFR-51 fork，只限制 evidence-close 条件 prior-debias 的最大校正强度，避免 `max_strength=1.0` 把过多 high-margin but poorly calibrated sagittal 样本升为主导。预计改进效果：top-weight 应落在 DFR-50 的 `93/1/0` 与 DFR-51 的 `60/1/33` 之间，axial 仍多数主导，但少量 coronal/sagittal true-margin 样本获得 top/near-top routing，从而比 matched equal-weight 更好地利用弱视角补充信息。实验实际结果：single-GPU adaptive `main-study` only completed trial `0` → `val_acc=0.9148936170212766`, `val_auc=0.9804545454545455`, `val_f1=0.9090909090909091`, `peak_vram≈2.15 GiB`, `total_seconds≈1092.9` → **discard**（主指标低于 matched equal-weight seed42 `0.9255319148936170`，也低于 retained DFR-25/26 seed42 `0.9361702127659575`；AUC 较高但不能覆盖 val_acc 回落。）
- **fusion-weight 结论**：[fusion_weight_analysis.json](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0001_20260510_224652/trials/trial_0000/run/fusion_weight_analysis.json) 显示 `max_strength=0.55` 太保守，routing 重新坍塌到 axial。mean fusion weight `axial/coronal/sagittal = 0.9342870243052219 / 0.037763407394448494 / 0.027949566734915086`；`top-weight count/rate axial/coronal/sagittal = 94/1.0000 / 0/0.0000 / 0/0.0000`。同时 `top_true_margin` 分布为 `axial/coronal/sagittal = 84 / 8 / 2`，说明仍有 `10` 个 non-axial true-margin 样本，但 capped conditional debias 没有一次把 top weight 迁移给 non-axial view。
- **当前判断**：DFR-52 是机制上干净但结果失败的 strength-cap probe。它确认 DFR-51 的过冲不是简单把上限降到中间值就能解决：`0.55` 基本回到 DFR-48/DFR-50 式 axial lock-in，`1.0` 又过度迁移到 sagittal。下一轮若继续 prior-debias family，不应再盲扫 `max_strength`；更直接的方向是 teacher/correctness-aligned trigger 或 per-view capped correction，要求只有 non-axial evidence 同时接近 axial 且具备更可靠校准信号时才增强 debias，而不是单靠 pred-margin close。
- **Agent 状态（2026-05-10 23:13 SGT）**：本 session 是唯一 coordinator，已完成 exactly one research iteration；last experiment=`DFR-52 capped evidence-close prior debias`，lane=`main-study`，result=`discard`，last commit=`3422993`；下一步建议=`DFR-53 teacher-aligned conditional prior debias / per-view capped correction`，继续服务 residual axial lock-in 修复，不切 backbone / geometry / fusion family。

> **DFR-53 自检**
> - 这项改动是否直接帮助 decision fusion 还原各视角应有作用？是。DFR-52 的 `max_strength=0.55` 完全回塌到 `axial=94/94`，DFR-51 的 `max_strength=1.0` 又过度迁移到 sagittal；DFR-53 只把同一 evidence-close 条件 debias 上限调到中间档 `0.75`，目标是在 gate 内部直接调节 routing 迁移幅度。
> - 如果成功，为什么有机会把 learned full-fusion `val_acc` 推到 matched `equal-weight` 之上？因为它仍是 sample-wise learned weighting，不固定平均，也不全样本强去偏。预期 axial 在多数强证据样本继续主导，但 coronal/sagittal true-margin 样本获得受限 top/near-top routing；若迁移幅度落在 DFR-50/52 的 axial lock-in 与 DFR-51 的 sagittal over-routing 中间，就可能保住 DFR-25 的 axial 收益并补回弱视角增量。
> - 它是可复用 learned weighting 机制还是固定权重？是 learned weighting 机制。`GateViewPriorDebiaser` 只按 detached evidence-close 程度调节 reliability-logit prior subtraction strength，最终三视角权重仍由 learned confidence logits 经 softmax 得到。

- [x] **DFR-53-RESNEXT-DECISION-256X8-MIDCAP-EVIDENCE-CLOSE-PRIOR-DEBIAS-MAIN-S42**：commit `5055d94`，在 [configs/autoresearch_formal.yaml](/dataset/HH/ankle-ct/configs/autoresearch_formal.yaml)、[configs/autoresearch_proxy.yaml](/dataset/HH/ankle-ct/configs/autoresearch_proxy.yaml) 与 Optuna 模板中只把 `ANKLE_DECISION_GATE_VIEW_PRIOR_DEBIAS_MAX_STRENGTH` 从 `0.55` 提到 `0.75`，保持 DFR-51/52 的 `strength=0.35/gap=0.2/window=0.2` 与 DFR-25 anchor 不变；运行配置为 [autoresearch_logs/generated_search_configs/optuna_main_search_iter_0002_20260510_231151.yaml](/dataset/HH/ankle-ct/autoresearch_logs/generated_search_configs/optuna_main_search_iter_0002_20260510_231151.yaml)，study_root [runs/optuna_main_autoloop/iter_0002_20260510_231151](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0002_20260510_231151)。设计思路：从 DFR-52 fork，只调 evidence-close conditional prior-debias 的最大强度到中间档，检验能否落在 axial recollapse 与 sagittal over-routing 之间。预计改进效果：top-weight 应离开 DFR-52 的 `axial=94/94`，但仍保持 axial majority；少量 coronal/sagittal true-margin 样本应获得 controlled top/near-top routing，从而有机会让 learned full-fusion 超过 matched equal-weight。实验实际结果：single-GPU adaptive `main-study` only completed trial `0` → `val_acc=0.9361702127659575`, `val_auc=0.9568181818181818`, `val_f1=0.9302325581395349`, `peak_vram≈2.15 GiB`, `total_seconds≈1225.1` → **discard**（主指标高于 matched equal-weight seed42 `0.9255319148936170`，并打平 DFR-25/26 seed42 `0.9361702127659575`；但 `val_auc` 低于 DFR-25 seed42 `0.9786363636363636` 与 DFR-26 seed42 `0.9800000000000000`，按 tie-break 不能保留。）
- **fusion-weight 结论**：[fusion_weight_analysis.json](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0002_20260510_231151/trials/trial_0000/run/fusion_weight_analysis.json) 显示 `max_strength=0.75` 确实把 routing 推到 DFR-50/52 与 DFR-51 之间。mean fusion weight `axial/coronal/sagittal = 0.4883510950714984 / 0.21161329661535613 / 0.3000356095999063`；`top-weight count/rate axial/coronal/sagittal = 54/0.5745 / 12/0.1277 / 28/0.2979`。同时 `top_true_margin` 分布为 `33 / 11 / 50`，`top_weight_hit_rate.true_margin=0.5851`，`top_weight_correct_rate=0.9255`；routing 不再坍塌，但与 true-margin 对齐明显低于 DFR-51 的 `0.8617`，说明中间强度修复了权重形状却没有修复 ranking quality。
- **当前判断**：DFR-53 是 diagnostic discard。它证明 blind mid-cap 可以同时避免 DFR-52 的 axial recollapse 和 DFR-51 的 sagittal over-routing，且主指标回到 `0.9362`；但 AUC 明显低于 retained DFR-25/26，说明受限 non-axial top-routing 仍没有稳定地选择真正有增量的样本。下一轮不应继续盲扫 `max_strength`；更直接的下一步是让 conditional debias 具备 per-view cap 或 teacher/correctness-aligned trigger，尤其限制 sagittal/coronal 迁移只发生在更可靠的 non-axial evidence 样本上。
- **Agent 状态（2026-05-10 23:40 SGT）**：本 session 是唯一 coordinator，已完成 exactly one research iteration；last experiment=`DFR-53 midcap evidence-close prior debias`，lane=`main-study`，result=`discard`，last commit=`5055d94`；下一步建议=`DFR-54 per-view capped or teacher-aligned conditional prior debias`，继续服务 residual axial lock-in 修复，不切 backbone / geometry / fusion family。

> **DFR-54 自检**
> - 这项改动是否直接帮助 decision fusion 还原各视角应有作用？是。DFR-52 的 `max_strength=0.55` 重新坍塌到 `axial=94/94`，DFR-53 的 `max_strength=0.75` 虽打破坍塌但 non-axial routing 过多且 AUC 掉线；DFR-54 只把同一条件 debias 上限收窄到 `0.65`，目标是在 gate 内部直接减少 DFR-53 的过度迁移。
> - 如果成功，为什么有机会把 learned full-fusion `val_acc` 推到 matched `equal-weight` 之上？因为它仍保留 sample-wise learned routing，不固定平均，也不改 view classifier。预期 axial 在强 axial evidence 样本重新占主导，但 coronal/sagittal true-margin 样本保留少量 top/near-top routing；若分布落在 DFR-52 的 axial recollapse 与 DFR-53 的 over-routing 之间，就可能保住 DFR-25 axial 收益并补入弱视角增量。
> - 它是可复用 learned weighting 机制还是固定权重？是 learned weighting 机制。`GateViewPriorDebiaser` 只调节 reliability-logit prior subtraction 的最大强度，最终三视角权重仍由 learned confidence logits 经 softmax 得到。

- [x] **DFR-54-RESNEXT-DECISION-256X8-NARROW-MIDCAP-EVIDENCE-CLOSE-PRIOR-DEBIAS-MAIN-S42**：commit `5c64f59`，在 [configs/autoresearch_formal.yaml](/dataset/HH/ankle-ct/configs/autoresearch_formal.yaml)、[configs/autoresearch_proxy.yaml](/dataset/HH/ankle-ct/configs/autoresearch_proxy.yaml) 与 Optuna 模板中只把 `ANKLE_DECISION_GATE_VIEW_PRIOR_DEBIAS_MAX_STRENGTH` 从 `0.75` 降到 `0.65`，保持 DFR-51/52/53 的 `strength=0.35/gap=0.2/window=0.2` 与 DFR-25 anchor 不变；运行配置为 [autoresearch_logs/generated_search_configs/optuna_main_search_iter_0003_20260510_233854.yaml](/dataset/HH/ankle-ct/autoresearch_logs/generated_search_configs/optuna_main_search_iter_0003_20260510_233854.yaml)，study_root [runs/optuna_main_autoloop/iter_0003_20260510_233854](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0003_20260510_233854)。设计思路：从 DFR-53 fork，只收窄 evidence-close conditional prior-debias 的最大强度，检验能否减少 DFR-53 的 non-axial over-routing，同时避免 DFR-52 的 axial recollapse。预计改进效果：top-weight 应落在 DFR-52 的 `94/0/0` 与 DFR-53 的 `54/12/28` 之间，理想形态是 axial-majority，并让少量 coronal/sagittal top 或 near-top routing 样本保留；mean axial weight 应高于 DFR-53 但低于 DFR-52，从而有机会让 learned full-fusion 保住强 axial 样本并利用 non-axial true-margin 样本推过 matched equal-weight。实验实际结果：single-GPU adaptive `main-study` only completed trial `0` → `val_acc=0.9255319148936170`, `val_auc=0.9754545454545454`, `val_f1=0.9230769230769231`, `peak_vram≈2.15 GiB`, `total_seconds≈1226.6` → **discard**（主指标只打平 matched equal-weight seed42 `0.9255319148936170`，低于 retained DFR-25/26 seed42 `0.9361702127659575` 与 DFR-53 `0.9361702127659575`。）
- **fusion-weight 结论**：[fusion_weight_analysis.json](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0003_20260510_233854/trials/trial_0000/run/fusion_weight_analysis.json) 显示 `max_strength=0.65` 没有落在预期中间带，而是重新回到 axial top-routing。mean fusion weight `axial/coronal/sagittal = 0.8680199441757608 / 0.050288894491151294 / 0.08169115956951963`；`top-weight count/rate axial/coronal/sagittal = 94/1.0000 / 0/0.0000 / 0/0.0000`。同时 `top_true_margin` 分布为 `75 / 15 / 4`，`top_weight_hit_rate.true_margin=0.7979`，说明 `19` 个 non-axial true-margin 样本仍没有一次获得 top gate weight；相较 DFR-53，DFR-54 只是把平均权重拉回 axial，而没有形成有效的 controlled weak-view contribution。
- **当前判断**：DFR-54 是 diagnostic discard。它确认 `0.55/0.65/0.75/1.0` 这条 blind `max_strength` 轴是非平滑的：`0.65` 已经 recollapse 到 `94/94`，`0.75` 才能改变 top-routing 但 AUC 掉线，继续细扫 cap 的价值很低。下一轮若继续 prior-debias family，应实现更有选择性的 per-view cap 或 teacher/correctness-aligned trigger，而不是再调一个全局 `max_strength`；核心约束是只让 non-axial 视角在更可靠证据样本上接管 top weight。
- **Agent 状态（2026-05-11 00:03 SGT）**：本 session 是唯一 coordinator，已完成 exactly one research iteration；last experiment=`DFR-54 narrow midcap evidence-close prior debias`，lane=`main-study`，result=`discard`，last commit=`5c64f59`；下一步建议=`DFR-55 per-view capped or teacher/correctness-aligned conditional prior debias`，继续服务 residual axial lock-in 修复，不切 backbone / geometry / fusion family。

> **DFR-55 自检**
> - 这项改动是否直接帮助 decision fusion 还原各视角应有作用？是。当前 ResNeXt learned decision-fusion 的主要不足不是 backbone 表达不够，而是 reliability gate 对三视角证据的相对关系建模不足：DFR-25 虽然 3-seed keep，但 seed42/456 仍接近 axial hard lock-in；DFR-49~54 又证明手写 prior debias 容易在 axial recollapse 和 non-axial over-routing 之间跳变。DFR-55 因此只在 gate/reliability path 上加入低容量 Transformer，让 gate 在打分前读取三视角 token 的相对上下文。
> - 如果成功，为什么有机会把 learned full-fusion `val_acc` 推到 matched `equal-weight` 之上？因为它不固定权重、不改 classifier、不改 DFR-25 winning scalar；Transformer 使用 sample-wise centered view features，并且 output projection zero-init，初始等价于 DFR-25，只允许训练学到小幅 cross-view reliability correction。理想效果是 seed42/456 不再 hard axial，seed123 也不因过强 debias 失稳，从而让 non-axial true-margin 样本获得受控 top/near-top routing。
> - 它是可复用 learned weighting 机制还是固定权重？是 learned weighting 机制。`GateTransformerContextualizer` 只改 learned gate scorer 的上下文输入，最终 fusion weights 仍由 confidence logits softmax 得到，不指定 axial/coronal/sagittal 的固定比例。

- [x] **DFR-55-RESNEXT-DECISION-256X8-GATE-TRANSFORMER-CONTEXT-MULTISEED-FORMAL**：commit `36e1b0e`（模型结构最初在 `85112b4` 引入；后续提交只补 Slurm launcher 与资源 guard），在 [src/model.py](/dataset/HH/ankle-ct/src/model.py) 新增 `GateTransformerContextualizer`，通过 `ANKLE_DECISION_ENABLE_GATE_TRANSFORMER_CONTEXT=1` 开启；运行配置为 [configs/cmp_resnext_decision_256x8_dfr55_gate_transformer_context_formal_s42.yaml](/dataset/HH/ankle-ct/configs/cmp_resnext_decision_256x8_dfr55_gate_transformer_context_formal_s42.yaml)、[configs/cmp_resnext_decision_256x8_dfr55_gate_transformer_context_formal_s123.yaml](/dataset/HH/ankle-ct/configs/cmp_resnext_decision_256x8_dfr55_gate_transformer_context_formal_s123.yaml)、[configs/cmp_resnext_decision_256x8_dfr55_gate_transformer_context_formal_s456.yaml](/dataset/HH/ankle-ct/configs/cmp_resnext_decision_256x8_dfr55_gate_transformer_context_formal_s456.yaml)。设计思路：从 DFR-25 fork，只给 gate/reliability path 增加 1-layer/4-head/128-dim Transformer contextualizer，输入为三视角相对特征，zero-init residual projection 且 `residual_scale=0.2`；保留 ResNeXt 256x8、decision fusion、L3 no-mixer、dominant-gate dropout、`lr=1e-4 / wd=2.5e-4 / dropout=0.25 / clip=2.5 / freeze_layers=3`。预计改进效果：相较 DFR-25 的 seed42/456 axial lock-in，top-weight 应出现少量受控 coronal/sagittal 迁移；相较 DFR-49/51/53，迁移应由 learned token context 决定而不是手写 prior strength，避免 sagittal over-routing，进而让 3-seed learned full-fusion 至少守住 DFR-25 3-seed mean。实验实际结果：Slurm job `482197` 在 `RTXA6Kq/node08` 手动绑定 idle `CUDA_VISIBLE_DEVICES=0` 顺序完成三 seed 训练与 telemetry；此前 `482188/482189/482190/482195/482196` 因 busy GPU/cancel/guard fail 不计入模型结论。`seed42=0.9148936170212766/0.9700000000000000/0.9069767441860465`, `seed123=0.8936170212765957/0.9450000000000000/0.8809523809523809`, `seed456=0.8723404255319149/0.9468181818181819/0.8536585365853658`；3-seed mean `val_acc=0.8936170212765958`, `val_auc=0.9539393939393940`, `val_f1=0.8805292205745978`, `peak_vram≈2.15 GiB` → **discard**（低于 DFR-25 3-seed mean `0.9397163120567376 / 0.9677272727272728 / 0.9358934169278997`，也低于 matched equal-weight mean `0.9219858156028368 / 0.9606060606060606 / 0.9112221100424511`；相对 DFR-25 mean 回落 `val_acc=-0.0460992907801417`, `val_auc=-0.0137878787878788`, `val_f1=-0.0553641963533019`。）
- **fusion-weight 结论**：聚合文件 [autoresearch_logs/dfr55_gate_transformer_multiseed/aggregate_summary.json](/dataset/HH/ankle-ct/autoresearch_logs/dfr55_gate_transformer_multiseed/aggregate_summary.json) 与三条 `fusion_weight_analysis.json` 显示 DFR-55 没有形成稳定的受控 routing。`seed42` mean weight `axial/coronal/sagittal = 0.4368 / 0.0834 / 0.4798`，top-weight `42/0/52`，从 DFR-25 的 axial lock-in 过度迁到 sagittal；`seed123` mean weight `0.9873 / 0.0032 / 0.0095`，top-weight `94/0/0`，完全回到 axial hard lock-in；`seed456` mean weight `0.3344 / 0.2309 / 0.4347`，top-weight `36/18/40`，但 `top_weight_hit_rate.true_margin=0.5213` 且 accuracy 只有 `0.8723`。也就是说，Transformer context 能改变部分 seed 的权重形状，但没有学到可靠的 evidence-to-routing ranking。
- **当前判断**：DFR-55 是明确的 negative result。它回答了人类提出的“用 Transformer 模块改进当前 ResNeXt”问题：在只放入 gate path、zero-init 小 residual、保持 DFR-25 标量的条件下，Transformer 并没有补足当前 ResNeXt learned decision fusion 的不足，反而在 seed 间表现为 `sagittal over-routing / axial recollapse / low true-margin alignment` 三种失稳形态。下一轮若继续结构修复，不应再增加无监督 gate-context 容量；更合理的是先做带明确 ranking 约束的 teacher/correctness-aligned gate supervision，或离线定位 DFR-25 中 non-axial true-margin 样本的可学习触发条件，再把它做成低容量、有上限的 correction。
- **Agent 状态（2026-05-11 03:02 SGT）**：本 session 是唯一 coordinator，已完成用户要求的 `DFR-55 gate Transformer context` 3-seed formal validation；last experiment=`DFR-55 gate transformer context multiseed formal`，lane=`formal 3-seed`，result=`discard`，last commit=`36e1b0e`；下一步建议=`DFR-56 teacher/correctness-aligned gate ranking supervision or offline DFR-25 disagreement telemetry`，继续服务 residual axial lock-in 修复，不切 backbone / geometry / fusion family。

> **DFR-56 自检**
> - 这项改动是否直接帮助 decision fusion 还原各视角应有作用？是。`innovation.md` 的 Pairwise Decomposition / View-GCN 思路强调视角之间的 pairwise 互补和关系建模；DFR-56 只在 gate/reliability path 上加入 pairwise reliability residual，让每个 view 的 reliability logit 读取它与另外两个正交切面的相对关系，而不是继续只看单视角特征或手写 axial debias。
> - 如果成功，为什么有机会把 learned full-fusion `val_acc` 推到 matched `equal-weight` 之上？因为它仍保留 DFR-25 learned decision fusion 与 dominant-gate dropout，不固定平均、不改 classifier；pairwise residual 是 zero-init 且有上限，理论上只在另一个切面提供互补证据时微调 routing，避免无条件压制 axial 或无条件抬高 non-axial。
> - 它是可复用 learned weighting 机制还是固定权重？是 learned weighting 机制。`PairwiseReliabilityGate` 输出 sample-wise、view-wise residual confidence logit，最终权重仍由 softmax 学得，不指定 axial/coronal/sagittal 的固定比例。

- [x] **DFR-56-RESNEXT-DECISION-256X8-PAIRWISE-RELIABILITY-GATE-MULTISEED-FORMAL**：commit `0c2dccb`，在 [src/model.py](/dataset/HH/ankle-ct/src/model.py) 新增 `PairwiseReliabilityGate`，通过 `ANKLE_DECISION_ENABLE_PAIRWISE_RELIABILITY_GATE=1` 开启；运行配置为 [configs/cmp_resnext_decision_256x8_dfr56_pairwise_reliability_gate_formal_s42.yaml](/dataset/HH/ankle-ct/configs/cmp_resnext_decision_256x8_dfr56_pairwise_reliability_gate_formal_s42.yaml)、[configs/cmp_resnext_decision_256x8_dfr56_pairwise_reliability_gate_formal_s123.yaml](/dataset/HH/ankle-ct/configs/cmp_resnext_decision_256x8_dfr56_pairwise_reliability_gate_formal_s123.yaml)、[configs/cmp_resnext_decision_256x8_dfr56_pairwise_reliability_gate_formal_s456.yaml](/dataset/HH/ankle-ct/configs/cmp_resnext_decision_256x8_dfr56_pairwise_reliability_gate_formal_s456.yaml)。设计思路：从 DFR-25 fork，只给 gate/reliability path 增加低容量 pair encoder，对每个 anchor view 聚合 `(anchor, anchor-other, anchor*other)` 两个 pair message 后输出 bounded residual confidence logit；保留 ResNeXt 256x8、decision fusion、L3 no-mixer、dominant-gate dropout、`lr=1e-4 / wd=2.5e-4 / dropout=0.25 / clip=2.5 / freeze_layers=3`。预计改进效果：相较 DFR-25 的 seed42/456 axial lock-in，pairwise 互补特征应让 coronal/sagittal 在相对证据更强的少数样本获得 near-top/top routing，同时不把权重机械拉平均。实验实际结果：用户已申请的 `node20` Slurm job `482369` 可见 3 张 V100 但 PyTorch CUDA probe 因 `/dev/nvidia-uvm` EIO 失败，故本轮改用已验证 CUDA 正常的 `PA100q/node03` Slurm job `482374` 并行跑 seeds `42/123/456`；三 seed 训练与 telemetry 均 exit 0。`seed42=0.9148936170212766/0.9568181818181818/0.9047619047619048`, `seed123=0.8723404255319149/0.9709090909090909/0.8421052631578947`, `seed456=0.9148936170212766/0.9772727272727273/0.9069767441860465`；3-seed mean `val_acc=0.9007092198581560`, `val_auc=0.9683333333333334`, `val_f1=0.8846146373686154`, `peak_vram≈2.20 GiB` → **discard**（主指标低于 DFR-25 3-seed mean `0.9397163120567376 / 0.9677272727272728 / 0.9358934169278997`，也低于 matched equal-weight mean `0.9219858156028368 / 0.9606060606060606 / 0.9112221100424511` 的 `val_acc / val_f1`。）
- **fusion-weight 结论**：三条 `fusion_weight_analysis.json` 显示 DFR-56 没有形成跨 seed 稳定的 pairwise routing。`seed42` mean weight `axial/coronal/sagittal = 0.9416 / 0.0277 / 0.0308`，top-weight `94/0/0`；`seed123` mean weight `0.0081 / 0.9747 / 0.0173`，top-weight `0/94/0`；`seed456` mean weight `0.9671 / 0.0246 / 0.0083`，top-weight `94/0/0`。它避免了“永远 axial”但没有避免“单一视图独占”，而 seed123 的 coronal 独占对应 sensitivity 只有 `0.7273`，说明 pairwise context 学到的是 seed-specific hard routing，不是可靠 evidence-to-routing ranking。
- **当前判断**：DFR-56 是 negative result。它说明仅把 Pairwise Decomposition / View-GCN 的 pair relation 放到 reliability residual 中，仍不足以让三切面各自发挥应有作用；低容量 pair encoder 会把原本的 axial lock-in 换成 seed-specific single-plane lock-in。下一轮应转向 `innovation.md` 中 RotationNet/MVCNN 提到的显式 view-id / anatomical role 建模，或者 GVCNN 式 view discrimination scoring，但必须保持低容量、只改网络结构、避免继续围绕 axial 专门手写规则。
- **Agent 状态（2026-05-11 03:34 SGT）**：本 session 是唯一 coordinator，已完成用户要求的第 1/3 轮 manual autoresearch；last experiment=`DFR-56 pairwise reliability gate multiseed formal`，lane=`formal 3-seed`，result=`discard`，last commit=`0c2dccb`；下一步建议=`DFR-57 view-id/anatomical-role calibrated reliability scoring or GVCNN-style discrimination gate`，继续服务 learned decision-fusion 的单视图独占修复，不切 ResNeXt / 256x8 / decision-fusion family。

> **DFR-57 自检**
> - 这项改动是否直接帮助 decision fusion 还原各视角应有作用？是。`innovation.md` 的 MVCNN/RotationNet 启发是：医学多视图不应把视角身份完全抹掉，但也不能让固定视角 offset 直接主导。DFR-57 因此把 confidence scorer 替换为共享 `ViewRoleConfidenceScorer`：输入是 sample-wise centered view feature + learned anatomical role embedding，显式告诉 gate 三个切面的角色，同时削弱 raw axial feature offset。
> - 如果成功，为什么有机会把 learned full-fusion `val_acc` 推到 matched `equal-weight` 之上？因为它仍输出 sample-wise learned non-equal weights，不固定平均；bounded confidence logits 避免单一切面立即 hard-select，而 role token 允许 axial/coronal/sagittal 在各自可靠场景中承担不同角色。理想结果是保留 DFR-25 的 learned routing 收益，同时减少 seed42/456 的 hard axial lock-in。
> - 它是可复用 learned weighting 机制还是固定权重？是 learned weighting 机制。`ViewRoleConfidenceScorer` 是共享网络 + view-role token 的 gate scorer，最终 fusion weight 仍由样本特征和 view identity 学出，不指定固定三视角比例。

- [x] **DFR-57-RESNEXT-DECISION-256X8-VIEW-ROLE-CONFIDENCE-MULTISEED-FORMAL**：commit `36afe79`，在 [src/model.py](/dataset/HH/ankle-ct/src/model.py) 新增 `ViewRoleConfidenceScorer`，通过 `ANKLE_DECISION_USE_VIEW_ROLE_CONFIDENCE_HEAD=1` 开启；运行配置为 [configs/cmp_resnext_decision_256x8_dfr57_view_role_confidence_formal_s42.yaml](/dataset/HH/ankle-ct/configs/cmp_resnext_decision_256x8_dfr57_view_role_confidence_formal_s42.yaml)、[configs/cmp_resnext_decision_256x8_dfr57_view_role_confidence_formal_s123.yaml](/dataset/HH/ankle-ct/configs/cmp_resnext_decision_256x8_dfr57_view_role_confidence_formal_s123.yaml)、[configs/cmp_resnext_decision_256x8_dfr57_view_role_confidence_formal_s456.yaml](/dataset/HH/ankle-ct/configs/cmp_resnext_decision_256x8_dfr57_view_role_confidence_formal_s456.yaml)。设计思路：从 DFR-25 fork，只把 learned decision-fusion 的 confidence scorer 替换成 role-aware shared scorer，使用 centered view features 和 32-dim role embeddings，logit bounded at `1.5`；保留 ResNeXt 256x8、decision fusion、L3 no-mixer、dominant-gate dropout、`lr=1e-4 / wd=2.5e-4 / dropout=0.25 / clip=2.5 / freeze_layers=3`。预计改进效果：相比 DFR-56 的 seed-specific single-plane lock-in，top-weight 应转向由 view-role + relative evidence 共同决定的非 hard-collapse 分布；若 role scorer 不损失 DFR-25 的强 axial 样本，应在 3-seed mean 上接近或超过 DFR-25。实验实际结果：`PA100q/node03` Slurm 3-GPU manual formal 完成 seeds `42/123/456`，训练与 telemetry 均 exit 0。`seed42=0.9148936170212766/0.9704545454545455/0.9047619047619048`, `seed123=0.9468085106382979/0.9740909090909091/0.9411764705882353`, `seed456=0.9255319148936170/0.9745454545454546/0.9213483146067416`；3-seed mean `val_acc=0.9290780141843971`, `val_auc=0.9730303030303031`, `val_f1=0.9224288966522939`, `peak_vram≈2.20 GiB` → **discard**（虽然高于 matched equal-weight mean `0.9219858156028368 / 0.9606060606060606 / 0.9112221100424511` 的三项均值，但低于当前 DFR-25 3-seed mean `0.9397163120567376 / 0.9677272727272728 / 0.9358934169278997` 的主指标和 F1，不能替代 current best。）
- **fusion-weight 结论**：DFR-57 明显改善了 DFR-55/56 的 hard single-plane collapse，但不是最终解。`seed42` mean weight `axial/coronal/sagittal = 0.4690 / 0.4428 / 0.0882`，top-weight `48/46/0`；`seed123` mean weight `0.4668 / 0.4924 / 0.0408`，top-weight `33/61/0`，且 val_acc 达到 `0.9468`；`seed456` mean weight `0.4589 / 0.2695 / 0.2716`，top-weight `74/9/11`。这说明 view-role scorer 能让 gate 离开 axial hard lock-in，并在 seed123 上超过 DFR-25，但完全替换 DFR-25 scorer 会在 seed42/456 损失强样本 ranking。
- **当前判断**：DFR-57 是 diagnostic discard with positive signal。下一轮不应丢掉 DFR-25 原 scorer，而应把 `ViewRoleConfidenceScorer` 作为 bounded blend/residual 注入 DFR-25 confidence logits：保留 DFR-25 的 strong-evidence ranking，同时借用 DFR-57 的 role-aware anti-collapse routing。
- **Agent 状态（2026-05-11 04:03 SGT）**：本 session 是唯一 coordinator，已完成用户要求的第 2/3 轮 manual autoresearch；last experiment=`DFR-57 view-role confidence multiseed formal`，lane=`formal 3-seed`，result=`discard with positive routing signal`，last commit=`36afe79`；下一步建议=`DFR-58 DFR-25 base scorer + bounded view-role confidence blend`，继续服务 learned decision-fusion 的单视图独占修复，不切 ResNeXt / 256x8 / decision-fusion family。

> **DFR-58 自检**
> - 这项改动是否直接帮助 decision fusion 还原各视角应有作用？是。DFR-57 证明 role-aware scorer 可以显著减少 hard single-plane collapse，但完全替换 DFR-25 scorer 会损失 seed42/456 的强样本 ranking；DFR-58 因此保留 DFR-25 原 confidence heads/calibrator，只把 bounded `ViewRoleConfidenceScorer` 以 `0.45` logit blend 注入，目标是让 role-aware 反坍塌信号成为低强度补充。
> - 如果成功，为什么有机会把 learned full-fusion `val_acc` 推到 matched `equal-weight` 之上并接近 DFR-25？因为 base scorer 继续负责 DFR-25 已验证有效的 strong-evidence ranking，role scorer 只提供显式 view identity + centered relative feature 的 anti-collapse correction；理论上应比 DFR-57 更少损失 axial 强样本，同时比 DFR-25 更少 hard lock-in。
> - 它是可复用 learned weighting 机制还是固定权重？是 learned weighting 机制。`ANKLE_DECISION_VIEW_ROLE_CONFIDENCE_BLEND=0.45` 只固定两个 learned scorer 的 logit 混合比例，最终三视角权重仍由样本特征、原 scorer 和 role-aware scorer 共同决定。

- [x] **DFR-58-RESNEXT-DECISION-256X8-VIEW-ROLE-CONFIDENCE-BLEND-MULTISEED-FORMAL**：commit `254b04c`，在 [src/model.py](/dataset/HH/ankle-ct/src/model.py) 为 `ViewRoleConfidenceScorer` 增加 `ANKLE_DECISION_VIEW_ROLE_CONFIDENCE_BLEND`，当 blend `<1.0` 时同时保留 DFR-25 base confidence heads/calibrator；运行配置为 [configs/cmp_resnext_decision_256x8_dfr58_view_role_blend_formal_s42.yaml](/dataset/HH/ankle-ct/configs/cmp_resnext_decision_256x8_dfr58_view_role_blend_formal_s42.yaml)、[configs/cmp_resnext_decision_256x8_dfr58_view_role_blend_formal_s123.yaml](/dataset/HH/ankle-ct/configs/cmp_resnext_decision_256x8_dfr58_view_role_blend_formal_s123.yaml)、[configs/cmp_resnext_decision_256x8_dfr58_view_role_blend_formal_s456.yaml](/dataset/HH/ankle-ct/configs/cmp_resnext_decision_256x8_dfr58_view_role_blend_formal_s456.yaml)。设计思路：从 DFR-57 fork，不再替换 DFR-25 scorer，而是在 confidence-logit 层做 `0.55 * base + 0.45 * role`，检验 role-aware anti-collapse signal 是否能作为 DFR-25 的补充。预计改进效果：保留 DFR-25 的强样本 ranking，同时减少 seed42/456 的 axial hard lock-in，3-seed mean 应回到 DFR-25 附近并继续高于 matched equal-weight。实验实际结果：`PA100q/node03` Slurm 3-GPU manual formal 完成 seeds `42/123/456`，训练与 telemetry 均 exit 0。`seed42=0.9148936170212766/0.9609090909090909/0.9024390243902439`, `seed123=0.9042553191489362/0.9790909090909091/0.8860759493670886`, `seed456=0.8936170212765957/0.9577272727272729/0.8717948717948718`；3-seed mean `val_acc=0.9042553191489362`, `val_auc=0.9659090909090910`, `val_f1=0.8867699485174015`, `peak_vram≈2.20 GiB` → **discard**（低于 DFR-25 mean，也低于 matched equal-weight mean 的 `val_acc / val_f1`。）
- **fusion-weight 结论**：DFR-58 没有继承 DFR-57 的反坍塌收益。`seed42` mean weight `0.7952 / 0.0827 / 0.1221` 且 top-weight `94/0/0`；`seed123` mean weight `0.9140 / 0.0506 / 0.0354` 且 top-weight `94/0/0`；`seed456` mean weight `0.4704 / 0.4373 / 0.0923`，top-weight `58/36/0`，但 val_acc 只有 `0.8936`。也就是说，base scorer 在 blend 中重新主导了 seed42/123 的 axial lock-in，而 seed456 的 axial/coronal 分裂没有转化为正确 ranking。
- **当前判断**：DFR-58 是 negative result。简单 logit blend 不是合适的桥接方式；如果后续继续 view-role 方向，应优先研究 DFR-57 replacement 的强 seed123 行为和 seed42/456 失败样本，而不是继续调 blend 比例。本次用户要求的 3 轮 manual autoresearch 已完成，当前最优仍是 DFR-25 3-seed formal mean。
- **Agent 状态（2026-05-11 04:30 SGT）**：本 session 是唯一 coordinator，已完成用户要求的第 3/3 轮 manual autoresearch；last experiment=`DFR-58 view-role confidence blend multiseed formal`，lane=`formal 3-seed`，result=`discard`，last commit=`254b04c`；本次三轮结论=`DFR-57 view-role scorer 是唯一 positive routing signal，但未超过 DFR-25`；建议下一步=`offline compare DFR-25 vs DFR-57 sample-level routing/correctness before further network edits`。

### DFR-57 follow-up offline analysis（2026-05-11）

- [x] **DFR-57-VS-DFR-25-SAMPLE-LEVEL-ROUTING-COMPARE**：新增 [scripts/analyze_dfr57_followup.py](/dataset/HH/ankle-ct/scripts/analyze_dfr57_followup.py)，只读取既有 `fusion_weight_analysis.json`，对 DFR-25 与 DFR-57 的 seeds `42/123/456` 做 patient-level join；完整 JSON 报告落在 `autoresearch_logs/dfr57_followup_analysis.json`（日志目录按 `.gitignore` 不入库）。核心结论：DFR-57 相比 DFR-25 共修复 `7` 个样本、打坏 `10` 个样本，净 `-3` correct，正好解释 3-seed mean accuracy 从 `0.939716` 回落到 `0.929078`。
- **按 seed 分解**：`seed42` 从 `0.936170` 降到 `0.914894`，`fixed=2 / broken=4 / net=-2`；`seed123` 从 `0.936170` 升到 `0.946809`，`fixed=4 / broken=3 / net=+1`；`seed456` 从 `0.946809` 降到 `0.925532`，`fixed=1 / broken=3 / net=-2`。DFR-57 的 positive signal 主要是 seed123 的 coronal role routing，而不是全 seed 稳定收益。
- **修复样本形态**：DFR-57 修复的样本多是 DFR-25 的阴性 FP。典型情况是 DFR-25 axial 对阴性样本给出高 abnormal probability，并用高 axial gate weight 放大错误；DFR-57 通过 coronal/role-aware rerouting 或重新训练后的 axial/coronal低异常概率把它们拉回 TN。重复出现的 positive fixed case 是 `CTyin__CT24yin21`（seed42/456）。
- **新增错误形态**：DFR-57 打坏的样本集中在两类。第一类是阳性样本被 coronal 强阴性稀释成 FN，例如 `CTyang__CT24yang1__CT2412yang54` 与 `CTyang__CT24yang1__CT2412yang58` 在 seed42/123 都被打坏；这说明 role-aware scorer 在一部分阳性 case 上过度信任 coronal negative evidence。第二类是 seed456 的阴性样本 classifier 本身漂移成强阳性，例如 `CTyin__CT24yin122/125/188`，即使 top-weight 仍是 axial，也已经不是纯 routing 问题。
- **per-view 诊断**：DFR-57 显著提升 coronal 单视角准确率（seed42 `+0.106383`，seed123 `+0.308511`，seed456 `+0.234043`），但同时压低 sagittal（`-0.063830 / -0.276596 / -0.138298`），并在 seed42/456 压低 axial（`-0.031915 / -0.074468`）。因此 DFR-57 的失败不是简单的 gate 权重分布不好，而是 role-aware confidence replacement 改变了多视角头部训练平衡：coronal 更可用，但 axial/sagittal 稳定性受损。
- **当前判断**：后续不应继续做 DFR-58 式无条件 logit blend，也不应直接把 DFR-57 replacement 当成新 base。更合格的下一轮应保留 DFR-25 scorer/classifier 的强样本 ranking，只在 `DFR-25 高异常阴性 FP 风险` 或 `top gate confidence gap 小且 coronal 低异常证据强` 的样本上启用 role-aware residual；并且必须限制对阳性样本的 coronal negative override，避免 `54/58/147/176` 这类 axial-positive case 被稀释成 FN。

### DFR-59 conditional view-role residual（2026-05-11）

> **实验说明**
> - 本轮按 DFR-57 follow-up 的建议继续 1 轮 3-seed formal，并先取消旧 Slurm 作业 `482369`；随后在 `V100q/node20` 提交失败后修复 base-confidence path，再用 Slurm job `482484` 在 `RTXA6Kq/node16` 完成 seeds `42/123/456`。
> - 严格保持 ResNeXt 256x8、decision fusion、DFR-25 scalar、3-seed formal 与数据划分不变；代码只改 [src/model.py](/dataset/HH/ankle-ct/src/model.py) 的网络决策融合结构，并新增 DFR59 三个 formal config 与 multiseed runner。
> - 结构假设：保留 DFR-25 原 scorer/classifier 的强样本 ranking，把 DFR-57 `ViewRoleConfidenceScorer` 降级为 bounded residual；只有在 base confidence top1-top2 gap 小，或 fused abnormal prob 高且 axial 高异常/coronal 低异常的 FP-risk 样本上启用 residual，并用 axial-positive guard 限制 coronal negative override。
> - 预计改进效果：继承 DFR-57 修复阴性 FP 的 role-aware routing signal，同时避免 DFR-57 在阳性 axial-positive 样本上被 coronal negative evidence 稀释成 FN；因此应至少回到 matched equal-weight mean 之上，并接近 DFR-25 3-seed mean。

- [x] **DFR-59-RESNEXT-DECISION-256X8-CONDITIONAL-VIEW-ROLE-RESIDUAL-MULTISEED-FORMAL**：commit `9570164`，formal seeds `42/123/456`，配置为 [configs/cmp_resnext_decision_256x8_dfr59_conditional_view_role_residual_formal_s42.yaml](/dataset/HH/ankle-ct/configs/cmp_resnext_decision_256x8_dfr59_conditional_view_role_residual_formal_s42.yaml)、[configs/cmp_resnext_decision_256x8_dfr59_conditional_view_role_residual_formal_s123.yaml](/dataset/HH/ankle-ct/configs/cmp_resnext_decision_256x8_dfr59_conditional_view_role_residual_formal_s123.yaml)、[configs/cmp_resnext_decision_256x8_dfr59_conditional_view_role_residual_formal_s456.yaml](/dataset/HH/ankle-ct/configs/cmp_resnext_decision_256x8_dfr59_conditional_view_role_residual_formal_s456.yaml)。实验实际结果：`seed42=0.9148936170212766/0.9704545454545455/0.9130434782608695`，`seed123=0.9042553191489362/0.9740909090909090/0.8915662650602410`，`seed456=0.9042553191489362/0.9813636363636363/0.8860759493670886`；3-seed mean `val_acc=0.9078014184397163`，`val_auc=0.9753030303030302`，`val_f1=0.8968952308960664`，`peak_vram≈2.15 GiB` → **discard**（低于 DFR-25 mean `0.9397163120567376 / 0.9677272727272728 / 0.9358934169278997`，也低于 matched equal-weight mean 的 `val_acc / val_f1`。）
- **fusion-weight 结论**：DFR-59 没有保住 DFR-57 的 anti-collapse signal。三个 seed 的 top-weight 都是 `axial/coronal/sagittal=94/0/0`；mean weights 分别为 seed42 `0.9423/0.0564/0.0013`，seed123 `0.8582/0.0736/0.0682`，seed456 `0.7731/0.1061/0.1207`。虽然平均权重比 DFR-25 略有非轴向质量，但 top-routing 完全回到 axial lock-in，说明 residual gate 触发过窄或 positive guard 过强，没能让 role-aware scorer 在关键样本上改变排序。
- **当前判断**：DFR-59 是 negative result。离线分析给出的“条件化 residual”方向没有直接转化为稳定收益；下一步如果继续 view-role family，不应再只靠 inference-time scorer residual，而应先做 sample-level trigger audit，统计每个 DFR57 fixed/broken case 在 DFR59 中是否实际触发 residual，以及 residual 对 confidence rank 的改变量。当前最优仍是 DFR-25 3-seed formal mean。

### DFR-60 detached view-role confidence（2026-05-11）

> **实验说明**
> - 本轮是用户要求继续 5 轮 3-seed research 的第 `1/5` 轮；在启动前先审查 autoresearch workflow，确认 `scripts/autoresearch_main.py` 只是 `optuna_main.py` 的兼容包装，不适合直接表达“连续 5 轮、每轮 3-seed formal、样本审计驱动”的当前目标。因此新增 generic 3-seed formal Slurm 入口 [scripts/run_resnext_decision_multiseed.py](/dataset/HH/ankle-ct/scripts/run_resnext_decision_multiseed.py) 与 [scripts/slurm_resnext_decision_multiseed.sbatch](/dataset/HH/ankle-ct/scripts/slurm_resnext_decision_multiseed.sbatch)，并把 autoresearch/formal/proxy/Optuna 默认配置恢复到 DFR-25 anchor，避免后续 loop 意外从已 discard 的 DFR54/DFR47 配置继续。
> - 结构假设：DFR-57 的 positive signal 是 role-aware scorer 能明显打破 hard axial collapse，但它完全替换 DFR-25 scorer 后会改变 per-view classifier/head 训练平衡；DFR-60 因此仍使用 DFR-57 replacement，但通过 `ANKLE_DECISION_DETACH_VIEW_ROLE_GATE_FEATURES=1` 将进入 `ViewRoleConfidenceScorer` 的 gate features detach，阻断 role gate loss 对 encoder/classifier 表征的反向塑形。
> - 预计改进效果：保留 DFR-57 seed42/123 的 axial/coronal 双路由与阴性 FP 修复，同时减少 `54/58/147/176` 这类阳性 axial-positive 样本被 coronal negative evidence 稀释成 FN，并降低 seed456 classifier drift。

- [x] **DFR-60-RESNEXT-DECISION-256X8-DETACHED-VIEW-ROLE-CONFIDENCE-MULTISEED-FORMAL**：commit `4d33bfa`，formal seeds `42/123/456`，配置为 [configs/cmp_resnext_decision_256x8_dfr60_detached_view_role_confidence_formal_s42.yaml](/dataset/HH/ankle-ct/configs/cmp_resnext_decision_256x8_dfr60_detached_view_role_confidence_formal_s42.yaml)、[configs/cmp_resnext_decision_256x8_dfr60_detached_view_role_confidence_formal_s123.yaml](/dataset/HH/ankle-ct/configs/cmp_resnext_decision_256x8_dfr60_detached_view_role_confidence_formal_s123.yaml)、[configs/cmp_resnext_decision_256x8_dfr60_detached_view_role_confidence_formal_s456.yaml](/dataset/HH/ankle-ct/configs/cmp_resnext_decision_256x8_dfr60_detached_view_role_confidence_formal_s456.yaml)。实验实际结果：Slurm job `482504` 在 `RTXA6Kq/node16` 完成，`seed42=0.9148936170212766/0.9750000000000000/0.9047619047619048`，`seed123=0.9361702127659575/0.9781818181818183/0.9285714285714286`，`seed456=0.9361702127659575/0.9700000000000000/0.9268292682926830`；3-seed mean `val_acc=0.9290780141843972`，`val_auc=0.9743939393939395`，`val_f1=0.9200542005420055`，`peak_vram≈2.20 GiB` → **discard**（低于 DFR-25 mean `0.9397163120567376 / 0.9677272727272728 / 0.9358934169278997` 的主指标和 F1，也没有超过 DFR-57 的 F1。）
- **fusion-weight 结论**：detach 没有稳定保留 DFR-57 的 useful routing。`seed42` mean weight `0.4847/0.4276/0.0877`，top-weight `48/46/0`；`seed123` mean weight `0.5020/0.4256/0.0723`，top-weight `45/49/0`；`seed456` mean weight `0.5851/0.2617/0.1532`，top-weight `94/0/0`。也就是说，seed42/123 仍有 role-aware axial/coronal split，但 seed456 re-collapse，而且 split 没带来 DFR-25 级别 accuracy。
- **sample-level 结论**：相对 DFR-25，DFR-60 仍是 `fixed=7 / broken=10 / net=-3`，与 DFR-57 的净变化相同。fixed 样本主要仍是阴性 FP（例如 `CTyin__CT24yin21/95/109/113`），broken 样本仍集中在阳性 FN（例如 `CTyang__CT24yang1__CT2412yang29/54/58/176/142`）。典型失败不是单纯 top-weight 变成 coronal，而是 role replacement 后 axial abnormal probability 本身也经常下降，再叠加 coronal 低异常概率，导致阳性证据被融合门稀释。
- **当前判断**：DFR-60 是 negative result。单纯 detach role-gate features 不足以防止 DFR-57 family 的 positive-evidence dilution；下一轮应优先保护阳性证据通道，而不是继续调 role scorer 的梯度或 blend。更合理的 DFR-61 假设是从 DFR-57/60 replacement 出发，在 fusion probability 层加入 learned-routing-compatible 的 positive-evidence floor/guard：当任一 view 给出高置信 abnormal evidence 时，融合结果不能被低异常 view 权重拉到该证据以下过多，同时仍保留 learned non-equal weights用于阴性 FP 修复。

### DFR-61 evidence-margin residual fusion（2026-05-11）

> **实验说明**
> - 本轮是用户要求继续 5 轮 3-seed research 的第 `2/5` 轮；严格保持 ResNeXt 256x8、decision fusion、DFR-25 scalar、DFR-60 role-aware routing 与数据划分不变。
> - 结构假设：DFR-57/60 的主要失败是高异常阳性 view 被低异常 view 通过 learned weights 稀释成 FN；因此在最终二分类 fused logits 上增加一个零初始化、低容量、有界的 `EvidenceMarginResidualFusion`，只读取 detached per-view logits 与 learned fusion weights，学习修正 binary margin，而不直接改 gate weights 或 per-view classifiers。
> - 预计改进效果：保留 DFR-57/60 的阴性 FP 修复与 anti-collapse routing，同时让 `54/58/147/176/29/142` 这类阳性样本在任一 view 有高置信 abnormal evidence 时不被融合门压到 normal。

- [x] **DFR-61-RESNEXT-DECISION-256X8-EVIDENCE-MARGIN-RESIDUAL-FUSION-MULTISEED-FORMAL**：commit `697a414`，formal seeds `42/123/456`，配置为 [configs/cmp_resnext_decision_256x8_dfr61_evidence_margin_residual_fusion_formal_s42.yaml](/dataset/HH/ankle-ct/configs/cmp_resnext_decision_256x8_dfr61_evidence_margin_residual_fusion_formal_s42.yaml)、[configs/cmp_resnext_decision_256x8_dfr61_evidence_margin_residual_fusion_formal_s123.yaml](/dataset/HH/ankle-ct/configs/cmp_resnext_decision_256x8_dfr61_evidence_margin_residual_fusion_formal_s123.yaml)、[configs/cmp_resnext_decision_256x8_dfr61_evidence_margin_residual_fusion_formal_s456.yaml](/dataset/HH/ankle-ct/configs/cmp_resnext_decision_256x8_dfr61_evidence_margin_residual_fusion_formal_s456.yaml)。实验实际结果：Slurm job `482552` 在 `RTXA6Kq/node16` 完成，`seed42=0.9255319148936170/0.9800000000000000/0.9156626506024096`，`seed123=0.9148936170212766/0.9731818181818183/0.9069767441860465`，`seed456=0.9042553191489362/0.9622727272727273/0.8860759493670886`；3-seed mean `val_acc=0.9148936170212766`，`val_auc=0.9718181818181818`，`val_f1=0.9029051147185149`，`peak_vram≈2.20 GiB` → **discard**（低于 DFR-25、DFR-57/60，也低于 matched equal-weight mean 的 `val_acc / val_f1`。）
- **fusion-weight 结论**：DFR-61 进一步打破了 axial lock-in，但没有转化为正确贡献。`seed42` mean weight `0.5204/0.3929/0.0866`，top-weight `68/26/0`；`seed123` mean weight `0.4599/0.3006/0.2395`，top-weight `56/38/0`；`seed456` mean weight `0.4672/0.3803/0.1525`，top-weight `43/51/0`。这说明 freer routing/margin residual 并非瓶颈答案，过度 coronal/axial split 会伤害主指标。
- **sample-level 结论**：相对 DFR-25，DFR-61 是 `fixed=8 / broken=15 / net=-7`；broken 以阳性 FN 为主，seed456 单独新增 `6` 个阳性 FN（例如 `112/142/147/165/29/58`），seed123 还新增阴性 FP（例如 `CTyin__CT24yin122/21`）。相对 DFR-60，DFR-61 只在 seed42 净 `+1`，seed123 净 `-2`，seed456 净 `-3`。
- **当前判断**：DFR-61 是 negative result。自由 signed margin residual 会学习到过强的负向修正，反而加剧 positive-evidence dilution；下一轮不应继续用 unconstrained residual。DFR-62 应改成单向 positive evidence floor/guard：只允许在 fused abnormal probability 低于高置信 view abnormal evidence 时向 abnormal margin 方向补偿，并通过小强度/高阈值约束避免阴性 FP 爆炸。

### DFR-62 positive-evidence floor fusion（2026-05-11）

> **实验说明**
> - 本轮是用户要求继续 5 轮 3-seed research 的第 `3/5` 轮；从 DFR-60 fork，不启用 DFR-61 的自由 signed residual。
> - 结构假设：DFR-61 失败来自 residual 可以学习负向 abnormal-margin 修正；DFR-62 改为 `PositiveEvidenceFloorFusion`，只允许在 fused abnormal probability 低于 `0.5`、且 detached per-view evidence 满足 `max abnormal >= 0.75` 与 second-view support `>= 0.2` 时，向 abnormal margin 做单向 sigmoid boost。模块初始化 bias 为负，初始几乎等价于 DFR-60。
> - 预计改进效果：在不打坏阴性 FP 修复的前提下，救回 DFR-57/60 的阳性 FN；如果 late-fusion floor 足够，seed42/456 应减少 `29/54/58/142/147/165/176` 这些 positive breaks。

- [x] **DFR-62-RESNEXT-DECISION-256X8-POSITIVE-EVIDENCE-FLOOR-FUSION-MULTISEED-FORMAL**：commit `567c4b5`，formal seeds `42/123/456`，配置为 [configs/cmp_resnext_decision_256x8_dfr62_positive_evidence_floor_fusion_formal_s42.yaml](/dataset/HH/ankle-ct/configs/cmp_resnext_decision_256x8_dfr62_positive_evidence_floor_fusion_formal_s42.yaml)、[configs/cmp_resnext_decision_256x8_dfr62_positive_evidence_floor_fusion_formal_s123.yaml](/dataset/HH/ankle-ct/configs/cmp_resnext_decision_256x8_dfr62_positive_evidence_floor_fusion_formal_s123.yaml)、[configs/cmp_resnext_decision_256x8_dfr62_positive_evidence_floor_fusion_formal_s456.yaml](/dataset/HH/ankle-ct/configs/cmp_resnext_decision_256x8_dfr62_positive_evidence_floor_fusion_formal_s456.yaml)。实验实际结果：Slurm job `482601` 在 `RTXA6Kq/node16` 完成，`seed42=0.9148936170212766/0.9700000000000000/0.9024390243902439`，`seed123=0.9255319148936170/0.9718181818181818/0.9213483146067416`，`seed456=0.9148936170212766/0.9586363636363637/0.9000000000000000`；3-seed mean `val_acc=0.9184397163120567`，`val_auc=0.9668181818181818`，`val_f1=0.9079291129989953`，`peak_vram≈2.20 GiB` → **discard**（低于 DFR-25、DFR-57/60，也低于 matched equal-weight mean 的 `val_acc / val_f1`。）
- **fusion-weight 结论**：DFR-62 没有稳定修复 routing。`seed42` top-weight 重新 hard axial `94/0/0`，mean weight `0.6767/0.2167/0.1066`；`seed123` top-weight `51/42/1`；`seed456` top-weight `28/66/0`，表现为 coronal over-routing。不同 seed 的 routing 方向分化，不能作为主线保留。
- **sample-level 结论**：相对 DFR-25，DFR-62 是 `fixed=8 / broken=14 / net=-6`；seed42 broken `5` 个阳性 FN，seed456 broken `5` 个阳性 FN，seed123 则出现 `3` 个阴性 FP（`CTyin__CT24yin122/188/21`）。相对 DFR-60，seed42 净 `0`，seed123 净 `-1`，seed456 净 `-2`。late positive floor 没能救回核心 positive breaks，且会重新打坏阴性样本。
- **当前判断**：DFR-62 是 negative result。DFR-61/62 共同说明问题不适合在最终 fused logits 后补；DFR-57/60 的 broken cases 已经包含 per-view abnormal probability drift，late-fusion guard 只能在已经漂移的概率上补救，容易引入 FP/FN tradeoff。DFR-63 应回到 gate/scorer 训练耦合：保留 role-aware anti-collapse，但减少它对 per-view classifier/head 学习的扰动，例如把 role scorer 的 replacement 变成 stop-gradient gate target / train-time auxiliary，而不改变 fused logits 主梯度。

### DFR-63 decoupled role classifier gradients（2026-05-11）

> **实验说明**
> - 本轮是用户要求继续 5 轮 3-seed research 的第 `4/5` 轮；从 DFR-60 fork，继续使用 detached view-role scorer，但不再在最终 fused logits 后加补丁。
> - 结构假设：DFR-57/60 的坏样本来自 role-aware routing 改变 per-view classifier/head 的训练平衡；因此训练时 fused logits 的数值仍等价于 role-weighted fusion，但 view logits 的主梯度走 DFR-25 base confidence weights，role scorer 只通过 detached view-logit residual 学习 routing。
> - 预计改进效果：保留 role-aware anti-collapse 与阴性 FP 修复，同时让 per-view classifier 更接近 DFR-25 的强阳性证据通道，减少 `54/58/120/165/176/29/47` 这类阳性 FN。

- [x] **DFR-63-RESNEXT-DECISION-256X8-DECOUPLED-ROLE-CLASSIFIER-GRAD-MULTISEED-FORMAL**：commit `6c9dc66`，formal seeds `42/123/456`，配置为 [configs/cmp_resnext_decision_256x8_dfr63_decoupled_role_classifier_grad_formal_s42.yaml](/dataset/HH/ankle-ct/configs/cmp_resnext_decision_256x8_dfr63_decoupled_role_classifier_grad_formal_s42.yaml)、[configs/cmp_resnext_decision_256x8_dfr63_decoupled_role_classifier_grad_formal_s123.yaml](/dataset/HH/ankle-ct/configs/cmp_resnext_decision_256x8_dfr63_decoupled_role_classifier_grad_formal_s123.yaml)、[configs/cmp_resnext_decision_256x8_dfr63_decoupled_role_classifier_grad_formal_s456.yaml](/dataset/HH/ankle-ct/configs/cmp_resnext_decision_256x8_dfr63_decoupled_role_classifier_grad_formal_s456.yaml)。实验实际结果：Slurm job `482624` 在 `RTXA6Kq/node16` 完成，`seed42=0.8936170212765957/0.9550000000000000/0.8750000000000000`，`seed123=0.8829787234042553/0.9277272727272726/0.8607594936708861`，`seed456=0.9148936170212766/0.9522727272727273/0.9024390243902439`；3-seed mean `val_acc=0.8971631205673759`，`val_auc=0.9450000000000000`，`val_f1=0.8793995060203766`，`peak_vram≈2.20 GiB` → **discard**（显著低于 DFR-25、DFR-57/60，也低于 matched equal-weight mean。）
- **fusion-weight 结论**：DFR-63 不是回到 DFR-25 稳定 routing，而是从 axial collapse 过校正成单一弱视角 collapse。`seed42` mean weight `0.3330/0.5190/0.1480`，top-weight `0/94/0`；`seed123` mean weight `0.1127/0.7374/0.1499`，top-weight `0/94/0`；`seed456` mean weight `0.4217/0.0675/0.5108`，top-weight `3/0/91`。seed456 的 `top_weight_hit_rate.true_margin=0.1170`，说明 learned routing 与真正有用视角严重错位。
- **sample-level 结论**：相对 DFR-25，DFR-63 是 `fixed=9 / broken=21 / net=-12`；broken 主要是阳性 FN（总计 `19` 个 FN break + `2` 个 FP break）。seed42/123 的 coronal over-routing 修复了少数阴性 FP（如 `CTyin__CT24yin21/95/113/109`），但打坏大量 DFR-25 强阳性样本（如 `CT2412yang120/165/176/29/47/54/58`）；seed456 则转成 sagittal over-routing。
- **当前判断**：DFR-63 是 strong negative result。role replacement family 的核心问题不是单纯 gate feature 梯度污染，而是 role scorer 在 fused loss 下会找到过强的单弱视角捷径；解耦 classifier 梯度反而让 role scorer 更自由地过路由。第 `5/5` 轮不应继续扩大 role scorer 结构，应回到 DFR-25 内部 gate prior 修复，并做此前最接近主线的 `DFR-53 midcap evidence-close prior debias` 的 3-seed formal confirmation。

### DFR-64 midcap prior debias confirmation（2026-05-11）

> **实验说明**
> - 本轮是用户要求继续 5 轮 3-seed research 的第 `5/5` 轮；不新增模型代码，只对 DFR-53 的 `midcap evidence-close prior debias` 做正式 3-seed confirmation。
> - 结构假设：DFR-63 证明 role replacement family 会过校正到单弱视角捷径；DFR-53 是此前最接近 DFR-25 seed42 ceiling 的 gate-internal prior repair，因此回到 DFR-25 scorer，只开启 `GateViewPriorDebiaser` 的 `strength=0.35 / max_strength=0.75 / gap=0.2 / window=0.2`。
> - 预计改进效果：如果 DFR-53 的 seed42 信号可泛化，seed123/456 应在避免 hard axial lock-in 的同时保持 axial strong-positive 样本，top-weight 应从 DFR-25 的 seed42/456 `axial=94/94` 迁移到 controlled mixed routing，而不是 DFR-63 式 coronal/sagittal collapse。

- [x] **DFR-64-RESNEXT-DECISION-256X8-MIDCAP-PRIOR-DEBIAS-CONFIRM-MULTISEED-FORMAL**：commit `f9d3c8f`，formal seeds `42/123/456`，配置为 [configs/cmp_resnext_decision_256x8_dfr64_midcap_prior_debias_confirm_formal_s42.yaml](/dataset/HH/ankle-ct/configs/cmp_resnext_decision_256x8_dfr64_midcap_prior_debias_confirm_formal_s42.yaml)、[configs/cmp_resnext_decision_256x8_dfr64_midcap_prior_debias_confirm_formal_s123.yaml](/dataset/HH/ankle-ct/configs/cmp_resnext_decision_256x8_dfr64_midcap_prior_debias_confirm_formal_s123.yaml)、[configs/cmp_resnext_decision_256x8_dfr64_midcap_prior_debias_confirm_formal_s456.yaml](/dataset/HH/ankle-ct/configs/cmp_resnext_decision_256x8_dfr64_midcap_prior_debias_confirm_formal_s456.yaml)。实验实际结果：Slurm job `482637` 在 `RTXA6Kq/node16` 完成，`seed42=0.9361702127659575/0.9636363636363636/0.9302325581395349`，`seed123=0.9148936170212766/0.9709090909090909/0.9047619047619048`，`seed456=0.8936170212765957/0.9445454545454546/0.8750000000000000`；3-seed mean `val_acc=0.9148936170212766`，`val_auc=0.9596969696969696`，`val_f1=0.9033314876338133`，`peak_vram≈2.15 GiB` → **discard**（低于 DFR-25 mean `0.9397163120567376 / 0.9677272727272728 / 0.9358934169278997`，也低于 matched equal-weight mean 的 `val_acc / val_auc / val_f1`。）
- **fusion-weight 结论**：DFR-64 复现了 DFR-53 seed42 的表面正信号，但没有跨 seed 泛化。`seed42` mean weight `0.7355/0.2016/0.0629` 且 top-weight 仍 `94/0/0`；`seed123` mean weight `0.4505/0.2927/0.2568`，top-weight `40/33/21`；`seed456` mean weight `0.3009/0.4745/0.2246`，top-weight `15/60/19`。seed456 的 true-margin 分布是 `0/50/44`，但 top-weight hit-rate 只有 `0.7021`，说明 routing 虽然离开 axial lock-in，却转成 coronal-heavy 且仍未对齐有效证据。
- **sample-level 结论**：相对 DFR-25，DFR-64 是 `fixed=6 / broken=13 / net=-7`。fixed 仍主要是阴性 FP（例如 `CTyin__CT24yin21/95/109/122`），broken 主要是阳性 FN（例如 `CT2412yang29/54/58/120/165/176/78`），并且 seed456 额外打坏 `CTyin__CT24yin122`。这说明 midcap prior debias 的收益仍是少量阴性修复，代价是更多阳性 evidence dilution。
- **当前判断**：DFR-64 是 negative confirmation。DFR-53 的单 seed 接近 ceiling 不是可保留主线；blind evidence-close prior debias 可以改变权重分布，但不能可靠判断哪一个 non-axial migration 对分类有益。下一轮不应继续扫 `strength/max_strength`，而应把 gate 修复改成更保守的 **positive-safe** 机制：只在候选 non-axial view 的异常概率不低于 axial 或 fused positive margin 已足够安全时允许 prior-debias 迁移，否则保持 DFR-25 axial strong-positive 通道。

### DFR-65 positive-safe prior debias（2026-05-11）

> **实验说明**
> - 本轮是 DFR-64 后的 autonomous continuation；继续保持 ResNeXt 256x8、decision fusion、DFR-25 scalar、L3-no-mixer 与 dominant-gate dropout。
> - 结构假设：DFR-64 的问题是 prior-debias 在阳性样本上把 low-abnormal non-axial view 推到过高权重，因此在 `GateViewPriorDebiaser` 中新增 positive-safe 条件，只在 best non-axial abnormal probability 不低于 axial 附近，或 max-debias 后 fused abnormal probability 仍高于安全线时，才允许 evidence-close max-strength debias 生效。
> - 预计改进效果：保留 DFR-64 少量阴性 FP 修复，同时避免 `54/58/120/165/176/29` 这类强阳性样本被 debias 后压成 FN。

- [x] **DFR-65-RESNEXT-DECISION-256X8-POSITIVE-SAFE-PRIOR-DEBIAS-MULTISEED-FORMAL**：commit `0d34b88`，formal seeds `42/123/456`，配置为 [configs/cmp_resnext_decision_256x8_dfr65_positive_safe_prior_debias_formal_s42.yaml](/dataset/HH/ankle-ct/configs/cmp_resnext_decision_256x8_dfr65_positive_safe_prior_debias_formal_s42.yaml)、[configs/cmp_resnext_decision_256x8_dfr65_positive_safe_prior_debias_formal_s123.yaml](/dataset/HH/ankle-ct/configs/cmp_resnext_decision_256x8_dfr65_positive_safe_prior_debias_formal_s123.yaml)、[configs/cmp_resnext_decision_256x8_dfr65_positive_safe_prior_debias_formal_s456.yaml](/dataset/HH/ankle-ct/configs/cmp_resnext_decision_256x8_dfr65_positive_safe_prior_debias_formal_s456.yaml)。实验实际结果：Slurm job `482652` 在 `RTXA6Kq/node16` 完成，`seed42=0.9042553191489362/0.9800000000000000/0.8988764044943820`，`seed123=0.9042553191489362/0.9740909090909091/0.8965517241379310`，`seed456=0.9042553191489362/0.9263636363636363/0.8915662650602410`；3-seed mean `val_acc=0.9042553191489362`，`val_auc=0.9601515151515151`，`val_f1=0.8956647978975180`，`peak_vram≈2.15 GiB` → **discard**（低于 DFR-25 mean `0.9397163120567376 / 0.9677272727272728 / 0.9358934169278997`，也低于 matched equal-weight mean 的 `val_acc / val_auc / val_f1`。）
- **fusion-weight 结论**：positive-safe gating 没有可靠保护阳性通道。`seed42` mean weight `0.6053/0.1914/0.2033`，top-weight `71/11/12`；`seed123` mean weight `0.4255/0.3545/0.2200`，top-weight `49/29/16`；`seed456` mean weight `0.2448/0.4145/0.3407`，top-weight `14/50/30`。虽然 `seed42` 的 AUC 达到 `0.9800`，但三个 seed 的 accuracy 都固定在 `0.9043`，说明 routing 迁移仍然没有转成正确分类。
- **sample-level 结论**：相对 DFR-25，DFR-65 是 `fixed=4 / broken=14 / net=-10`。fixed 只有少量 FP/FN 修复（例如 seed42 的 `CTyin__CT24yin122`、seed123 的 `CT2412yang147`、seed456 的 `CTyin__CT24yin21/95`）；broken 同时包含阳性 FN 与阴性 FP，例如 `CT2412yang54/176/29/120/165` 以及 `CTyin__CT24yin109/40/113/122`。这比 DFR-64 的 `fixed=6 / broken=13` 更差。
- **当前判断**：DFR-65 是 negative result。positive-safe 条件仍无法把 prior-debias 的 non-axial migration 限定在真正有益样本上，说明 prior-debias family 的主要问题不是单个安全阈值，而是没有可学习地识别“哪次迁移有增益”。下一轮不应继续调 debias strength/gap/window，也不应做晚期 handcrafted probability floor；离线 sanity check 显示基于 DFR-25 telemetry 的 simple positive floor / negative veto / equal-blend 都不能超过 DFR-25。更合理的 DFR-66 是回到 DFR-25 推理期 gate，尝试**低权重训练期 per-view auxiliary CE**，目标是增强各 view classifier 的独立判别能力而不改变 inference routing。

### DFR-66 low auxiliary view loss（2026-05-11）

> **实验说明**
> - 本轮是 DFR-65 后的 autonomous continuation；不改模型代码，回到 DFR-25 inference-time gate，只额外启用训练期 per-view auxiliary CE。
> - 结构假设：prior-debias 与 role replacement family 的共同失败形态是 routing migration 不能可靠识别有益样本，并且经常造成 positive-evidence dilution；DFR-66 因此不再改推理期权重，只用低权重 auxiliary CE（`ANKLE_DECISION_ENABLE_AUX_VIEW_LOSS=1`, `ANKLE_DECISION_AUX_VIEW_LOSS_WEIGHT=0.1`）增强各 view classifier 的独立判别能力。
> - 预计改进效果：保持 DFR-25 learned gate 与 dominant-gate dropout 的强样本 ranking，同时减少 per-view abnormal probability drift，使 seed42/456 的 axial lock-in 不再成为唯一可用证据通道。

- [x] **DFR-66-RESNEXT-DECISION-256X8-LOW-AUX-VIEW-LOSS-MULTISEED-FORMAL**：commit `07a78ea`，formal seeds `42/123/456`，配置为 [configs/cmp_resnext_decision_256x8_dfr66_low_aux_view_loss_formal_s42.yaml](/dataset/HH/ankle-ct/configs/cmp_resnext_decision_256x8_dfr66_low_aux_view_loss_formal_s42.yaml)、[configs/cmp_resnext_decision_256x8_dfr66_low_aux_view_loss_formal_s123.yaml](/dataset/HH/ankle-ct/configs/cmp_resnext_decision_256x8_dfr66_low_aux_view_loss_formal_s123.yaml)、[configs/cmp_resnext_decision_256x8_dfr66_low_aux_view_loss_formal_s456.yaml](/dataset/HH/ankle-ct/configs/cmp_resnext_decision_256x8_dfr66_low_aux_view_loss_formal_s456.yaml)。实验实际结果：Slurm job `482731` 在 `RTXA6Kq/node16` 完成，`seed42=0.9255319148936170/0.9777272727272727/0.9230769230769231`，`seed123=0.8510638297872340/0.9309090909090909/0.8444444444444444`，`seed456=0.9361702127659575/0.9654545454545455/0.9302325581395349`；3-seed mean `val_acc=0.9042553191489362`，`val_auc=0.9580303030303030`，`val_f1=0.8992513085536341`，`peak_vram≈2.15 GiB` → **discard**（低于 DFR-25 mean `0.9397163120567376 / 0.9677272727272728 / 0.9358934169278997`，也低于 matched equal-weight mean 的 `val_acc / val_auc / val_f1`。）
- **fusion-weight 结论**：低权重 auxiliary CE 没有提升弱视角可用性，反而把 gate 推向 seed-specific 单视图独占。`seed42` mean weight `0.7855/0.1048/0.1097`，top-weight `90/4/0`；`seed123` mean weight `0.0600/0.9235/0.0165`，top-weight `0/94/0`；`seed456` mean weight `0.9149/0.0489/0.0362`，top-weight `94/0/0`。其中 seed123 的 axial 单视角 accuracy `0.8830` 高于融合后的 `0.8511`，说明 auxiliary CE 并没有让 learned gate 更好地选择强视角。
- **sample-level 结论**：相对 DFR-25，DFR-66 是 `fixed=6 / broken=16 / net=-10`。seed42 仅修复 1 个阴性 FP，但新增 2 个阴性 FP；seed123 修复 5 个样本却打坏 13 个，主要因为 coronal 独占把多例 DFR-25 强阳性样本压成 FN（如 `CT2412yang120/165/176/47/54/58`）；seed456 只新增 1 个阳性 FN（`CT2412yang58`）。这说明 auxiliary CE 的主要副作用是训练期改变 per-view/gate 平衡并制造单视图捷径。
- **当前判断**：DFR-66 是 negative result。低权重 per-view auxiliary CE 仍复现了 DFR-02 family 的核心问题：它约束了 view classifiers，却没有提供稳定的 sample-wise routing 改善，并且会把 fusion gate 推成 seed-specific single-plane lock-in。下一轮不应继续加大或调小 auxiliary CE，也不应回到 prior-debias/role replacement。更合理的 DFR-67 是只做更轻的训练期 view robustness：保留 DFR-25 inference gate 和 scalar，使用比 DFR-03 更温和的 `ANKLE_DECISION_TRAIN_VIEW_DROPOUT_PROB=0.05`，不加 axial blur，检验“轻微视角缺失”是否能减少单视图捷径而不破坏 per-view probability calibration。

### DFR-67 light view dropout（2026-05-11）

> **实验说明**
> - 本轮是 DFR-66 后的 autonomous continuation；不改模型代码，回到 DFR-25 inference-time gate，只加入更轻的训练期 view robustness。
> - 结构假设：DFR-66 说明 auxiliary CE 会改变 per-view/gate 训练平衡并制造 seed-specific 单视图捷径；DFR-67 因此停用 auxiliary CE、prior-debias、role replacement 和晚期 probability rule，只启用 `ANKLE_DECISION_TRAIN_VIEW_DROPOUT_PROB=0.05`。
> - 预计改进效果：轻微视角缺失应削弱训练期单视图捷径，让 seed42/456 的 `top_weight axial=94/94` 至少出现少量 non-axial near-top/top routing，同时保持 DFR-25 的强 axial-positive 通道和 winning scalar。

- [x] **DFR-67-RESNEXT-DECISION-256X8-LIGHT-VIEW-DROPOUT-MULTISEED-FORMAL**：commit `ad31836`，formal seeds `42/123/456`，配置为 [configs/cmp_resnext_decision_256x8_dfr67_light_view_dropout_formal_s42.yaml](/dataset/HH/ankle-ct/configs/cmp_resnext_decision_256x8_dfr67_light_view_dropout_formal_s42.yaml)、[configs/cmp_resnext_decision_256x8_dfr67_light_view_dropout_formal_s123.yaml](/dataset/HH/ankle-ct/configs/cmp_resnext_decision_256x8_dfr67_light_view_dropout_formal_s123.yaml)、[configs/cmp_resnext_decision_256x8_dfr67_light_view_dropout_formal_s456.yaml](/dataset/HH/ankle-ct/configs/cmp_resnext_decision_256x8_dfr67_light_view_dropout_formal_s456.yaml)。实验实际结果：Slurm job `482734` 在 `RTXA6Kq/node16` 完成，`seed42=0.9361702127659575/0.9622727272727273/0.9285714285714286`，`seed123=0.9042553191489362/0.9522727272727273/0.8988764044943820`，`seed456=0.9468085106382979/0.9672727272727273/0.9411764705882353`；3-seed mean `val_acc=0.9290780141843972`，`val_auc=0.9606060606060606`，`val_f1=0.9228747678846819`，`peak_vram≈2.18 GiB` → **discard**（高于 DFR-66，但低于 DFR-25 mean `0.9397163120567376 / 0.9677272727272728 / 0.9358934169278997`；主指标虽高于 matched equal-weight mean `0.9219858156028368`，但本轮目标是超过当前 retained DFR-25。）
- **fusion-weight 结论**：轻微 view dropout 没有修复 routing collapse。三个 seed 的 top-weight 都仍是 `axial=94, coronal=0, sagittal=0`；mean weight 分别为 seed42 `0.9249/0.0486/0.0265`、seed123 `0.8785/0.0675/0.0540`、seed456 `0.9209/0.0183/0.0608`。`top_true_margin` 仍包含 non-axial 样本（seed42 `84/6/4`，seed123 `85/4/5`，seed456 `74/8/12`），但 gate 没有给 non-axial 一次 top-weight。
- **sample-level 结论**：相对 DFR-25，DFR-67 是 `fixed=4 / broken=7 / net=-3`。seed42 `fixed=3 / broken=3`，主要是修复部分阴性 FP 但新增阳性 FN；seed123 `fixed=0 / broken=3` 是本轮均值回落的核心；seed456 `fixed=1 / broken=1` 基本持平。说明 light view dropout 主要是概率/分类正则化，不是有效的 learned routing repair。
- **当前判断**：DFR-67 是 diagnostic discard。它比 DFR-66/63~65 稳，但没有改变 `top_weight axial=94/94` 的坍塌形态；下一轮不应继续加大 view dropout 或叠 axial blur。更直接的 DFR-68 是回到 DFR-25 训练语义，只在推理期加极小 `ANKLE_DECISION_FUSION_WEIGHT_FLOOR=0.02`，检验比旧 `floor=0.05` 更保守的 non-axial minimum mass 是否能保住 seed123，同时给 seed42/456 的 non-axial evidence 留出最小贡献。

### DFR-68 tiny fusion floor 0.02（2026-05-11）

> **实验说明**
> - 本轮是 DFR-67 后的 autonomous continuation；保持 ResNeXt 256x8、decision fusion、DFR-25 scalar、L3-no-mixer 与 dominant-gate dropout。
> - 代码自检：`ANKLE_DECISION_FUSION_WEIGHT_FLOOR` 在当前 [src/model.py](/dataset/HH/ankle-ct/src/model.py) 的 fusion forward 路径中直接对 active view softmax weights 做 `floor + residual_mass * learned_weight`，因此本轮实际是训练/验证均生效的 tiny bounded-gating floor，不是纯 post-hoc inference-only 评估。
> - 结构假设：旧 `floor=0.05` 和 rescue family 干预过强，而 DFR-67 完全没有改变 top-weight；DFR-68 用更小的 `0.02` 下限给每个 active view 保留 2% 最小贡献，目标是在不把 learned routing 拉平均的前提下，给 non-axial evidence 留出最小梯度/概率通道。
> - 预计改进效果：seed42/456 应避免完全 hard axial lock-in，至少出现少量 sagittal/coronal top 或 near-top routing；seed123 应比 DFR-67 更稳，不应复制 `floor=0.05` 的 alternate-seed collapse。

- [x] **DFR-68-RESNEXT-DECISION-256X8-TINY-FUSION-FLOOR02-MULTISEED-FORMAL**：commit `f50ed39`，formal seeds `42/123/456`，配置为 [configs/cmp_resnext_decision_256x8_dfr68_tiny_fusion_floor02_formal_s42.yaml](/dataset/HH/ankle-ct/configs/cmp_resnext_decision_256x8_dfr68_tiny_fusion_floor02_formal_s42.yaml)、[configs/cmp_resnext_decision_256x8_dfr68_tiny_fusion_floor02_formal_s123.yaml](/dataset/HH/ankle-ct/configs/cmp_resnext_decision_256x8_dfr68_tiny_fusion_floor02_formal_s123.yaml)、[configs/cmp_resnext_decision_256x8_dfr68_tiny_fusion_floor02_formal_s456.yaml](/dataset/HH/ankle-ct/configs/cmp_resnext_decision_256x8_dfr68_tiny_fusion_floor02_formal_s456.yaml)。实验实际结果：Slurm job `482745` 在 `RTXA6Kq/node16` 完成，`seed42=0.9361702127659575/0.9709090909090909/0.9302325581395349`，`seed123=0.9255319148936170/0.9527272727272728/0.9195402298850575`，`seed456=0.9361702127659575/0.9672727272727273/0.9302325581395349`；3-seed mean `val_acc=0.9326241134751774`，`val_auc=0.9636363636363637`，`val_f1=0.9266684487213758`，`peak_vram≈2.15 GiB` → **discard**（高于 DFR-67/66，但仍低于 DFR-25 mean `0.9397163120567376 / 0.9677272727272728 / 0.9358934169278997`。）
- **fusion-weight 结论**：`floor=0.02` 确实比 DFR-67 更能松动部分 routing，但迁移没有稳定转化为正确样本。`seed42` mean weight `0.6293/0.1473/0.2234`，top-weight `77/0/17`；`seed123` mean weight `0.7956/0.1296/0.0748`，top-weight `94/0/0`；`seed456` mean weight `0.8759/0.0674/0.0568`，top-weight `94/0/0`。seed42 虽有 17 个 sagittal top-weight，但 `top_true_margin` 是 `86/4/4`，说明迁移大多不是 aligned non-axial evidence；seed123/456 仍未摆脱 top axial collapse。
- **sample-level 结论**：相对 DFR-25，DFR-68 是 `fixed=3 / broken=5 / net=-2`。seed42 `fixed=2 / broken=2`，seed123 `fixed=1 / broken=2`，seed456 `fixed=0 / broken=1`。fixed/broken 样本在 DFR25 与 DFR68 下的 top-weight 都仍主要是 axial，说明本轮收益/损失更多来自概率边界和 calibration 变化，不是可靠的 non-axial routing repair。
- **当前判断**：DFR-68 是 diagnostic discard。它是近期 floor/view-robustness 系列里最接近 DFR-25 的结果，但仍未超过 retained DFR-25，也没有稳定修复 seed123/456 的 axial top-weight collapse。plain bounded-gating floor 不能继续宽扫；只允许再做一个更保守的 `0.01` amplitude sanity check，若仍低于 DFR-25，应停止这个 floor family，转向能显式区分“有益 non-axial 迁移”和“弱视角误迁移”的 learned/ranking 机制。

### DFR-69 ultratiny fusion floor 0.01（2026-05-11）

> **实验说明**
> - 本轮是 DFR-68 后的 autonomous continuation；保持 ResNeXt 256x8、decision fusion、DFR-25 scalar、L3-no-mixer 与 dominant-gate dropout。
> - 结构假设：DFR-68 的 `floor=0.02` 已经能松动 seed42 routing，但未能稳定转化为正确样本；DFR-69 只把下限降到 `0.01`，作为 plain bounded-gating floor family 的最后一个 amplitude sanity check。
> - 预计改进效果：如果 DFR-68 的问题主要是下限过大，`floor=0.01` 应保留更接近 DFR-25 的 axial strong routing，同时仍给 non-axial evidence 留出极小贡献；理想形态是 seed42/456 有少量 evidence-aligned non-axial top/near-top 迁移，seed123 不再回落。

- [x] **DFR-69-RESNEXT-DECISION-256X8-ULTRATINY-FUSION-FLOOR01-MULTISEED-FORMAL**：commit `e280c55`，formal seeds `42/123/456`，配置为 [configs/cmp_resnext_decision_256x8_dfr69_ultratiny_fusion_floor01_formal_s42.yaml](/dataset/HH/ankle-ct/configs/cmp_resnext_decision_256x8_dfr69_ultratiny_fusion_floor01_formal_s42.yaml)、[configs/cmp_resnext_decision_256x8_dfr69_ultratiny_fusion_floor01_formal_s123.yaml](/dataset/HH/ankle-ct/configs/cmp_resnext_decision_256x8_dfr69_ultratiny_fusion_floor01_formal_s123.yaml)、[configs/cmp_resnext_decision_256x8_dfr69_ultratiny_fusion_floor01_formal_s456.yaml](/dataset/HH/ankle-ct/configs/cmp_resnext_decision_256x8_dfr69_ultratiny_fusion_floor01_formal_s456.yaml)。实验实际结果：Slurm job `482758` 在 `RTXA6Kq/node16` 完成，`seed42=0.9255319148936170/0.9736363636363636/0.9176470588235294`，`seed123=0.9255319148936170/0.9563636363636364/0.9195402298850575`，`seed456=0.9361702127659575/0.9681818181818183/0.9302325581395349`；3-seed mean `val_acc=0.9290780141843972`，`val_auc=0.9660606060606062`，`val_f1=0.9224732822827072`，`peak_vram≈2.15 GiB` → **discard**（低于 DFR-68，也低于 DFR-25 mean `0.9397163120567376 / 0.9677272727272728 / 0.9358934169278997`。）
- **fusion-weight 结论**：`floor=0.01` 没有比 `floor=0.02` 更接近有效修复。`seed42` mean weight `0.6398/0.1584/0.2018`，top-weight `83/0/11`；`seed123` mean weight `0.9521/0.0164/0.0315`，top-weight `94/0/0`；`seed456` mean weight `0.8949/0.0625/0.0426`，top-weight `94/0/0`。seed42 只有有限 sagittal top migration，且 accuracy 掉到 `0.9255`；seed123/456 仍然 recollapse 到 hard axial top-weight。
- **sample-level 结论**：相对 DFR-25，DFR-69 是 `fixed=3 / broken=6 / net=-3`。seed42 `fixed=2 / broken=3`，seed123 `fixed=1 / broken=2`，seed456 `fixed=0 / broken=1`；fixed/broken 仍主要发生在 DFR25 与 DFR69 都由 axial top-weight 主导的边界样本上，而不是可靠的 non-axial routing repair。
- **当前判断**：DFR-69 是 diagnostic discard，并正式关闭 plain floor/bounded-gating family。`floor=0.02` 和 `floor=0.01` 都不能稳定超过 DFR-25；更小 floor 只是减少 routing 迁移，同时也没有恢复 seed42/123 的主指标。下一轮应转向更显式的 evidence-ranking / teacher-ranking 机制，让 non-axial mass 只在 detached per-view evidence ranking 支持时进入，而不是无条件保底。

### DFR-70 ultralow teacher blend 0.05（2026-05-11）

> **实验说明**
> - 本轮从 DFR-25 anchor fork，保持 ResNeXt 256x8、decision fusion、L3 no-mixer、dominant-gate dropout 与 winning scalar 不变。
> - 结构假设：DFR-68/69 的 plain floor 是无条件给 non-axial 质量，容易在没有证据支持的样本上引入噪声；DFR-70 改成 `ANKLE_DECISION_GATE_TEACHER_BLEND=0.05`，把 detached per-view logit margin softmax 得到的 evidence-rank teacher 以 5% 强度混入 learned gate。
> - 预计改进效果：如果 DFR-25 的 residual axial lock-in 主要是 softmax gate 对非轴向 evidence 过度压制，那么极低 teacher blend 应在 teacher ranking 支持时给 coronal/sagittal 少量质量，seed42/456 应出现 evidence-aligned non-axial top/near-top 迁移，同时保留 seed123 的强样本 ranking。

- [x] **DFR-70-RESNEXT-DECISION-256X8-ULTRALOW-TEACHER-BLEND05-MULTISEED-FORMAL**：commit `0bef411`，formal seeds `42/123/456`，配置为 [configs/cmp_resnext_decision_256x8_dfr70_ultralow_teacher_blend05_formal_s42.yaml](/dataset/HH/ankle-ct/configs/cmp_resnext_decision_256x8_dfr70_ultralow_teacher_blend05_formal_s42.yaml)、[configs/cmp_resnext_decision_256x8_dfr70_ultralow_teacher_blend05_formal_s123.yaml](/dataset/HH/ankle-ct/configs/cmp_resnext_decision_256x8_dfr70_ultralow_teacher_blend05_formal_s123.yaml)、[configs/cmp_resnext_decision_256x8_dfr70_ultralow_teacher_blend05_formal_s456.yaml](/dataset/HH/ankle-ct/configs/cmp_resnext_decision_256x8_dfr70_ultralow_teacher_blend05_formal_s456.yaml)。实验实际结果：Slurm job `482765` 在 `V100q/node21` 完成，`seed42=0.9255319148936170/0.9718181818181818/0.9213483146067416`，`seed123=0.9148936170212766/0.9772727272727273/0.9069767441860465`，`seed456=0.9468085106382979/0.9672727272727273/0.9425287356321839`；3-seed mean `val_acc=0.9290780141843972`，`val_auc=0.9721212121212122`，`val_f1=0.9236179314749906`，`peak_vram≈2.15 GiB` → **discard**（低于 DFR-25 mean `0.9397163120567376 / 0.9677272727272728 / 0.9358934169278997` 的主指标和 F1。）
- **fusion-weight 结论**：DFR-70 的 evidence-rank blend 不稳定。`seed42` mean weight `0.9625/0.0196/0.0179`，top-weight `94/0/0`；`seed123` mean weight `0.2667/0.4444/0.2889`，top-weight `19/46/29`，确实打散了 axial lock-in 但 accuracy 掉到 `0.9149`；`seed456` mean weight `0.9163/0.0620/0.0217`，top-weight `94/0/0`，accuracy 打平 DFR-25 seed456。说明 teacher blend 能在部分 seed 改变 routing，但这种改变是 noisy routing，不是稳定 contribution。
- **sample-level 结论**：相对 DFR-25，DFR-70 是 `fixed=6 / broken=9 / net=-3`。seed42 `fixed=1 / broken=2`，seed123 `fixed=5 / broken=7`，seed456 `fixed=0 / broken=0`；fixed 包含 `4` 个阴性样本和 `2` 个阳性样本，broken 包含 `5` 个阳性样本和 `4` 个阴性样本。seed123 的大幅 routing migration 同时修复和打坏样本，净结果为负。
- **当前判断**：DFR-70 是 diagnostic discard。它证明 detached evidence ranking 作为路由信号确实能影响 gate，但把 teacher blend 放进训练/验证全路径会造成 seed-specific routing drift。下一轮应保留同一 evidence-rank signal，但只在 eval/validation 路径作为极低强度 gate calibration，训练梯度完全回到 DFR-25 learned gate，从而隔离 teacher signal 本身是否有 post-hoc calibration 价值。

### DFR-71 eval-only teacher blend 0.05（2026-05-11）

> **实验说明**
> - 本轮从 DFR-70 的 negative diagnosis 继续，保持 DFR-25 anchor 的 ResNeXt 256x8、decision fusion、L3 no-mixer、dominant-gate dropout 与 winning scalar 不变。
> - 结构假设：DFR-70 失败可能来自 teacher blend 参与训练后改变 gate/classifier 平衡，而不是 detached evidence-rank teacher 本身完全无用；DFR-71 因此新增 `ANKLE_DECISION_GATE_TEACHER_BLEND_EVAL_ONLY=1`，训练时保持 DFR-25 learned gate，只有 eval/validation 时把 5% teacher weights 混入 gate。
> - 预计改进效果：如果 evidence teacher 只适合作为 post-hoc calibration，seed42/456 应出现少量 evidence-aligned non-axial top/near-top routing，同时避免 DFR-70 seed123 的训练期 noisy routing drift，并保住 DFR-25 的强阳性样本。

- [x] **DFR-71-RESNEXT-DECISION-256X8-EVALONLY-TEACHER-BLEND05-MULTISEED-FORMAL**：commit `e848eef`，formal seeds `42/123/456`，配置为 [configs/cmp_resnext_decision_256x8_dfr71_evalonly_teacher_blend05_formal_s42.yaml](/dataset/HH/ankle-ct/configs/cmp_resnext_decision_256x8_dfr71_evalonly_teacher_blend05_formal_s42.yaml)、[configs/cmp_resnext_decision_256x8_dfr71_evalonly_teacher_blend05_formal_s123.yaml](/dataset/HH/ankle-ct/configs/cmp_resnext_decision_256x8_dfr71_evalonly_teacher_blend05_formal_s123.yaml)、[configs/cmp_resnext_decision_256x8_dfr71_evalonly_teacher_blend05_formal_s456.yaml](/dataset/HH/ankle-ct/configs/cmp_resnext_decision_256x8_dfr71_evalonly_teacher_blend05_formal_s456.yaml)。实验实际结果：Slurm job `482773` 在 `V100q/node21` 完成，`seed42=0.9042553191489362/0.9822727272727273/0.8941176470588236`，`seed123=0.9042553191489362/0.9600000000000000/0.8965517241379310`，`seed456=0.9361702127659575/0.9554545454545454/0.9302325581395349`；3-seed mean `val_acc=0.9148936170212766`，`val_auc=0.9659090909090908`，`val_f1=0.9069673097787630`，`peak_vram≈2.15 GiB` → **discard**（低于 DFR-25 mean `0.9397163120567376 / 0.9677272727272728 / 0.9358934169278997`，也低于 matched equal-weight mean 的 `val_acc / val_f1`。）
- **fusion-weight 结论**：eval-only teacher blend 没有把 evidence signal 转成稳定贡献。`seed42` mean weight `0.7995/0.0787/0.1218` 但 top-weight 仍 `94/0/0`；`seed123` mean weight `0.9684/0.0106/0.0209`，top-weight 也仍 `94/0/0`；`seed456` mean weight `0.4919/0.4527/0.0554`，top-weight `44/50/0`，但 accuracy 只有 `0.9362`，低于 DFR-25 seed456 `0.9468`。说明 teacher 只在 eval 混入时仍会造成边界漂移，且 non-axial migration 并不可靠。
- **sample-level 结论**：相对 DFR-25，DFR-71 是 `fixed=4 / broken=11 / net=-7`；seed42 `fixed=2 / broken=5`，seed123 `fixed=1 / broken=4`，seed456 `fixed=1 / broken=2`。fixed 全部是阴性样本，broken 包含 `7` 个阳性样本和 `4` 个阴性样本，说明 eval-only teacher 主要以损伤阳性通道为代价换来少量 FP 修复。
- **当前判断**：DFR-71 是 negative result，并基本关闭“直接把 detached evidence teacher 混入 gate weights”的路径。DFR-70/71 共同说明 teacher-rank signal 能改变 routing，但无法区分有益迁移和弱视角误迁移；下一轮不应继续调 teacher blend 强度或 train/eval 时机，而应转向不直接改 gate 权重的低容量校准变量。DFR-72 因此回到 DFR-25 gate，只启用 per-view logit temperature calibration，验证是否能通过视角 logits 的置信度校准缓解 positive-evidence dilution。

### DFR-72 view-logit temperature on DFR-25 gate（2026-05-11）

> **实验说明**
> - 本轮从 DFR-25 anchor fork，关闭 DFR-70/71 的 direct teacher gate blend，保持 ResNeXt 256x8、decision fusion、L3 no-mixer、dominant-gate dropout 与 winning scalar 不变。
> - 结构假设：如果 DFR-70/71 的失败来自直接改 gate weights 太 noisy，那么更低容量的 per-view classifier logit temperature calibration 可能通过校准各视角置信度来减少 positive-evidence dilution，同时不直接改变 learned gate routing。
> - 预计改进效果：learned temperatures 应明显偏离 `[1,1,1]`，在保持 DFR-25 routing 主体的同时改善 seed42/456 的边界样本；不要求大量 non-axial top migration，但 per-view logits 应校准到足以提升 full fusion。

- [x] **DFR-72-RESNEXT-DECISION-256X8-VIEW-LOGIT-TEMPERATURE-DFR25-MULTISEED-FORMAL**：commit `4601482`，formal seeds `42/123/456`，配置为 [configs/cmp_resnext_decision_256x8_dfr72_view_logit_temperature_dfr25_formal_s42.yaml](/dataset/HH/ankle-ct/configs/cmp_resnext_decision_256x8_dfr72_view_logit_temperature_dfr25_formal_s42.yaml)、[configs/cmp_resnext_decision_256x8_dfr72_view_logit_temperature_dfr25_formal_s123.yaml](/dataset/HH/ankle-ct/configs/cmp_resnext_decision_256x8_dfr72_view_logit_temperature_dfr25_formal_s123.yaml)、[configs/cmp_resnext_decision_256x8_dfr72_view_logit_temperature_dfr25_formal_s456.yaml](/dataset/HH/ankle-ct/configs/cmp_resnext_decision_256x8_dfr72_view_logit_temperature_dfr25_formal_s456.yaml)。实验实际结果：Slurm job `482783` 在 `V100q/node21` 完成，`seed42=0.9148936170212766/0.9800000000000000/0.9090909090909091`，`seed123=0.9361702127659575/0.9568181818181819/0.9302325581395349`，`seed456=0.9042553191489362/0.9654545454545455/0.9032258064516129`；3-seed mean `val_acc=0.9184397163120567`，`val_auc=0.9674242424242424`，`val_f1=0.9141830912273523`，`peak_vram≈2.15 GiB` → **discard**（低于 DFR-25 mean `0.9397163120567376 / 0.9677272727272728 / 0.9358934169278997`，也低于 matched equal-weight mean 的 `val_acc / val_f1`。）
- **fusion-weight 结论**：view-logit temperature 没有实质改变 DFR-25 的 routing。`seed42` mean weight `0.8325/0.0826/0.0850`，top-weight `94/0/0`；`seed123` mean weight `0.9327/0.0482/0.0191`，top-weight `94/0/0`；`seed456` mean weight `0.7790/0.1334/0.0876`，top-weight `92/2/0`。checkpoint 中 learned temperatures 近似 identity：seed42 `0.9930/1.0031/1.0047`，seed123 `0.9969/1.0056/1.0038`，seed456 `0.9962/1.0017/1.0029`，说明该变量几乎没有学到可用的视角尺度修正。
- **sample-level 结论**：相对 DFR-25，DFR-72 是 `fixed=3 / broken=9 / net=-6`；seed42 `fixed=1 / broken=3`，seed123 `fixed=1 / broken=1`，seed456 `fixed=1 / broken=5`。fixed 包含 `2` 个阴性样本和 `1` 个阳性样本，broken 包含 `6` 个阴性样本和 `3` 个阳性样本；seed456 新增 `5` 个阴性 FP，说明 temperature 的轻微训练扰动没有带来可靠校准。
- **当前判断**：DFR-72 是 negative result。低容量 scalar / temperature / calibration 变量在 DFR-25 anchor 下不足以修复 residual axial lock-in；DFR-70/71/72 共同关闭了 direct teacher blend 与 per-view logit temperature 这两条低容量证据校准路径。下一轮不应继续调 teacher blend 或 scalar temperature，而应回到近期唯一有正向 routing signal 的 DFR-57 view-role scorer，测试更严格 bounded role scorer 是否能保留 anti-collapse 同时减少 weak-view over-routing。

### DFR-73 bounded view-role scorer logit limit 1.0（2026-05-12）

> **实验说明**
> - 本轮从 DFR-57 view-role scorer 回来，但把 `ANKLE_DECISION_VIEW_ROLE_CONFIDENCE_LOGIT_LIMIT` 从 `1.5` 收紧到 `1.0`，其余保持 DFR-25 anchor：ResNeXt 256x8、decision fusion、L3 no-mixer、dominant-gate dropout 与 winning scalar 不变。
> - 结构假设：DFR-57/60 是近期唯一能稳定提供 anti-collapse routing signal 的 family，但会产生 weak-view over-routing 和 positive-evidence dilution；更严格的 confidence-logit bound 可能保留 axial/coronal split，同时削弱低置信弱视角对阳性样本的过度稀释。
> - 预计改进效果：seed42/456 应避免完全 hard axial lock-in，同时 seed123 不应像 DFR-57 一样出现过强 coronal routing；理想形态是 axial 仍为多数 top-weight，但 coronal 在有 true-margin 支持的样本上获得少量 top/near-top routing，从而把 DFR-57 的 FP 修复收益转成超过 DFR-25 的净收益。

- [x] **DFR-73-RESNEXT-DECISION-256X8-BOUNDED-VIEW-ROLE-LIMIT10-MULTISEED-FORMAL**：commit `4f3fa00`，formal seeds `42/123/456`，配置为 [configs/cmp_resnext_decision_256x8_dfr73_bounded_view_role_limit10_formal_s42.yaml](/dataset/HH/ankle-ct/configs/cmp_resnext_decision_256x8_dfr73_bounded_view_role_limit10_formal_s42.yaml)、[configs/cmp_resnext_decision_256x8_dfr73_bounded_view_role_limit10_formal_s123.yaml](/dataset/HH/ankle-ct/configs/cmp_resnext_decision_256x8_dfr73_bounded_view_role_limit10_formal_s123.yaml)、[configs/cmp_resnext_decision_256x8_dfr73_bounded_view_role_limit10_formal_s456.yaml](/dataset/HH/ankle-ct/configs/cmp_resnext_decision_256x8_dfr73_bounded_view_role_limit10_formal_s456.yaml)。实验实际结果：Slurm job `482792` 在 `V100q/node21` 完成，`seed42=0.9255319148936170/0.9540909090909091/0.9156626506024096`，`seed123=0.9255319148936170/0.9690909090909092/0.9176470588235294`，`seed456=0.9361702127659575/0.9781818181818182/0.9285714285714286`；3-seed mean `val_acc=0.9290780141843972`，`val_auc=0.9671212121212122`，`val_f1=0.9206270459991225`，`peak_vram≈2.15 GiB` → **discard**（高于 matched equal-weight mean 的 `val_acc`，但仍低于 DFR-25 mean `0.9397163120567376 / 0.9677272727272728 / 0.9358934169278997` 的主指标和 F1。）
- **fusion-weight 结论**：limit `1.0` 确实把 DFR-57/60 的 routing 变得更保守，但没有形成正确净收益。`seed42` mean weight `0.4686/0.4653/0.0662`，top-weight `90/4/0`；`seed123` mean weight `0.4722/0.4626/0.0653`，top-weight 仍 `94/0/0`；`seed456` mean weight `0.6229/0.2724/0.1048`，top-weight `89/5/0`。也就是说，coronal 获得了接近 axial 的平均质量，但 top-routing 仍主要由 axial 控制，且 sagittal 仍没有稳定进入 top-weight。
- **sample-level 结论**：相对 DFR-25，DFR-73 是 `fixed=6 / broken=9 / net=-3`；seed42 `fixed=3 / broken=4`，seed123 `fixed=2 / broken=3`，seed456 `fixed=1 / broken=2`。fixed 主要是阴性 FP（总计 `5` 个阴性、`1` 个阳性），broken 全部是阳性 FN（`9` 个阳性），说明 bounded role scorer 继续以损伤 positive-evidence channel 为代价修复一部分阴性 FP。
- **当前判断**：DFR-73 是 negative result，并基本关闭继续盲调 view-role confidence-logit amplitude 的路径。更严格 bound 能降低 over-routing，但不能解决 role family 的核心问题：阳性样本上 axial/sagittal positive evidence 仍会被 coronal/role negative evidence 稀释。下一轮不应继续把 limit 从 `1.0` 往下扫；应回到尚未 3-seed formal 确认的 DFR-37 hard mismatch dominant-dropout，验证这个更保守的训练期 mismatch repair 是否有跨 seed 稳定性。

### DFR-74 hard mismatch dominant-gate dropout confirmation（2026-05-12）

> **实验说明**
> - 本轮按人类要求从 DFR-25 anchor 出发，对 DFR-37 hard mismatch dominant-gate dropout 做 formal 3-seed confirmation；保持 ResNeXt 256x8、decision fusion、L3 no-mixer、dominant-gate dropout prob=0.25 与 winning scalar 不变，只额外启用 `ANKLE_DECISION_TRAIN_AXIAL_TEACHER_MISALIGNMENT_THRESHOLD=0.1`。
> - 设计思路：DFR-37 的 seed42 positive-but-not-promoted 信号说明，把 dominant-gate dropout 限定到 learned gate 过度相信 axial、而 detached per-view evidence teacher 明确不同意的样本，可能比 view-role/floor/prior-debias family 更保守地修复 single-view dependence。
> - 预计改进效果：seed42/456 的 residual axial lock-in 应不再完全 `top_weight axial=94/94`；预期 axial 仍在强 axial-evidence 样本主导，但 coronal/sagittal 在 teacher/evidence 支持的样本上获得 top/near-top routing，使 learned full-fusion mean `val_acc` 保持或超过 DFR-25，并明确高于 matched equal-weight，而不是机械平均。

- [x] **DFR-74-RESNEXT-DECISION-256X8-HARD-MISMATCH-DOMINANT-DROPOUT-MULTISEED-FORMAL**：commit `ac64885`，fresh adaptive `main-study` 使用 [autoresearch_logs/generated_search_configs/optuna_main_search_iter_0001_20260512_031151.yaml](/dataset/HH/ankle-ct/autoresearch_logs/generated_search_configs/optuna_main_search_iter_0001_20260512_031151.yaml)，study root [runs/optuna_main_autoloop/iter_0001_20260512_031151](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0001_20260512_031151)，GPU policy `--gpu-ids 0,2,3 --max-workers 3`，完成 seeds `42/123/456`。实验实际结果：`seed42=0.9148936170212766/0.9754545454545454/0.9090909090909091`，`seed123=0.9042553191489362/0.9536363636363637/0.9010989010989011`，`seed456=0.9468085106382979/0.9668181818181818/0.9425287356321839`；3-seed mean `val_acc=0.9219858156028369`，`val_auc=0.9653030303030303`，`val_f1=0.9175728486073313`，`peak_vram≈2.19 GiB` → **discard**（seed456 是单 seed spike，但 3-seed mean 低于 DFR-25 mean，且 `val_acc` 只等于 matched equal-weight mean，没有证明 learned full-fusion 超过 matched equal-weight。）
- **fusion-weight 结论**：hard mismatch trigger 没有修复 routing collapse。`seed42` mean weight `0.9748/0.0140/0.0112`，top-weight `94/0/0`；`seed123` mean weight `0.9210/0.0486/0.0304`，top-weight `94/0/0`；`seed456` mean weight `0.9676/0.0137/0.0188`，top-weight `94/0/0`。三 seed 合计 top-weight `282/0/0`，multi-seed mean weight `0.9545/0.0254/0.0201`；即使 true-margin telemetry 中存在 non-axial 支持样本，最终 routing 仍全部选择 axial。
- **当前判断**：DFR-74 正式关闭 DFR-37 hard mismatch threshold 作为可推广主线修复的假设。它没有把 weak-view contribution 转化成稳定 routing，也没有让 learned full-fusion 明确超过 matched equal-weight；后续不应继续调这个 threshold。下一轮若继续 DFR-25 anchor，应优先做一项 positive-signal 机制或样本级 trigger audit，聚焦“哪些样本确实需要 non-axial routing、当前 trigger 为什么没有覆盖/转化”，而不是再做 broad amplitude sweep。

### DFR-75 candidate-view reliability gate seed42 candidates（2026-05-12）

> **实验说明**
> - 本轮继续严格从 DFR-25 anchor 出发：ResNeXt 256x8、learned decision fusion、L3 no-mixer、dominant-gate dropout 与 winning scalar 不变；唯一新增机制是在 reliability path 中加入低容量 `CandidateViewReliabilityGate`。
> - 设计思路：用 detached per-view classifier logit margin 识别 candidate non-axial view；只有当 coronal/sagittal 的 predicted margin 相对 axial 至少高出 `0.5` 时，才允许该 non-axial view 获得 bounded residual reliability logit。这样修复的是 evidence 与 routing 不一致，而不是把三视角固定平均。
> - 预计改进效果：相对 DFR-25 seed42 的 `top_weight axial=94/94`，预期 axial 仍主导大多数强 axial-evidence 样本，但 coronal/sagittal 在自身 margin 明确强于 axial 的样本上获得少量 top/near-top routing；如果这些候选样本确实对应 right-view evidence，则 learned full-fusion 有机会超过 matched equal-weight seed42 `0.9255319148936170`，而不是只提升单视角或把权重拉平均。

- [x] **DFR-75-RESNEXT-DECISION-256X8-CANDIDATE-VIEW-RELIABILITY-GATE-SEED42-MAIN-STUDY**：commit `2c53420`，fresh adaptive `main-study` 使用 [autoresearch_logs/generated_search_configs/optuna_main_search_iter_0002_20260512_033321.yaml](/dataset/HH/ankle-ct/autoresearch_logs/generated_search_configs/optuna_main_search_iter_0002_20260512_033321.yaml)，study root [runs/optuna_main_autoloop/iter_0002_20260512_033321](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0002_20260512_033321)，GPU policy `--gpu-ids 0,2,3 --max-workers 3`，只跑 seed42 三个保守候选 `ANKLE_DECISION_CANDIDATE_VIEW_GATE_RESIDUAL_LIMIT=0.25/0.35/0.50`。实验实际结果：trial0 `0.8936170212765957/0.9368181818181818/0.8837209302325582`，trial1 `0.9042553191489362/0.9468181818181818/0.8915662650602410`，trial2 `0.9148936170212766/0.9468181818181818/0.9090909090909091`；best trial2 `peak_vram≈2.19 GiB` → **discard**（best seed42 仍低于 DFR-25 seed42 `0.9361702127659575/0.9786363636363636/0.9333333333333333`，也低于 matched equal-weight seed42 `0.9255319148936170` 的主指标。）
- **fusion-weight 结论**：best trial2 的 [fusion_weight_analysis.json](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0002_20260512_033321/trials/trial_0002/run/fusion_weight_analysis.json) 显示 candidate gate 确实打破了 axial collapse，但迁移质量不够。mean fusion weight `axial/coronal/sagittal = 0.4241/0.5587/0.0173`；top-weight count/rate `axial=42/0.4468`，`coronal=52/0.5532`，`sagittal=0/0.0000`。`top_true_margin` 分布为 `32/46/16`，`top_weight_hit_rate.true_margin=0.5851`；也就是说，coronal 获得大量 top-routing，但并没有把 true-margin 中的 sagittal 支持样本纳入，也没有转化为 accuracy。
- **当前判断**：DFR-75 是 negative seed42 candidate study，不应 promotion 到 3-seed formal。它提供的诊断是：直接用 predicted-margin advantage 选 candidate view 可以移动 routing，但该触发仍会把 learned fusion 推向 coronal over-routing，而不是 right-view-at-right-sample。后续若继续 candidate/hierarchical selection，必须先做样本级 trigger audit 或加更可靠的 supervised/prototype alignment signal；不要把 `residual_limit`、`margin_gap` 当成新的 broad sweep 主线。

### DFR-76 class-consensus candidate-view reliability gate（2026-05-12）

> **实验说明**
> - 本轮继续从 DFR-25 anchor 出发，并只在 DFR-75 的 candidate-view reliability gate 上增加一个默认关闭的 signed class-consensus guard；ResNeXt 256x8、learned decision fusion、L3 no-mixer、dominant-gate dropout、winning scalar 与 DFR-75 的 candidate margin trigger 均保持不变。
> - 设计思路：DFR-75 已证明 class-agnostic predicted-margin advantage 会打破 axial collapse，但也会把高置信、方向可能错误的 coronal evidence 放大成 over-routing。DFR-76 因此要求候选 non-axial view 的 signed abnormal/normal margin 至少得到另一个视角同方向支持，避免孤立弱视角仅凭大 margin 抢走 routing。
> - 预计改进效果：相对 DFR-75 seed42 best 的 `top_weight axial/coronal/sagittal=42/52/0`，预期 coronal top-routing 会收窄，axial 保留强证据样本主导，同时只有 class-consensus 支持的 non-axial 样本获得 top/near-top weight；如果这些迁移更接近 right-view evidence，learned full-fusion 应能超过 matched equal-weight seed42 `0.9255319148936170`，而不是只把权重平均化或改善单视角表现。

- [x] **DFR-76-RESNEXT-DECISION-256X8-CLASS-CONSENSUS-CANDIDATE-GATE-SEED42-MAIN-STUDY**：commit `5424ff4`，fresh adaptive `main-study` 使用 [autoresearch_logs/generated_search_configs/optuna_main_search_iter_0003_20260512_040326.yaml](/dataset/HH/ankle-ct/autoresearch_logs/generated_search_configs/optuna_main_search_iter_0003_20260512_040326.yaml)，study root [runs/optuna_main_autoloop/iter_0003_20260512_040326](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0003_20260512_040326)，GPU policy `--gpu-ids 0,2,3 --max-workers 3`，只跑 seed42 三个保守候选 `ANKLE_DECISION_CANDIDATE_VIEW_GATE_CONSENSUS_MARGIN=0.15/0.25/0.35`，并固定 `residual_limit=0.50`, `margin_gap=0.5`, `consensus_window=0.20`。实验实际结果：trial0 `0.9148936170212766/0.9654545454545456/0.9069767441860465`，trial1 `0.8936170212765957/0.9372727272727274/0.8837209302325582`，trial2 `0.9255319148936170/0.9750000000000000/0.9176470588235294`；best trial2 `peak_vram≈2.19 GiB`, `total_seconds≈606.0` → **discard**（best seed42 只打平 matched equal-weight seed42 `0.9255319148936170`，仍低于 DFR-25 seed42 `0.9361702127659575/0.9786363636363636/0.9333333333333333`；AUC 较高但不能覆盖主指标未超过。）
- **fusion-weight 结论**：best trial2 的 [fusion_weight_analysis.json](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0003_20260512_040326/trials/trial_0002/run/fusion_weight_analysis.json) 显示 class-consensus guard 对 DFR-75 的 coronal over-routing 有所收窄，但仍没有产生 right-view-at-right-sample 的净收益。mean fusion weight `axial/coronal/sagittal = 0.4746/0.3865/0.1390`；top-weight count/rate `axial=51/0.5426`，`coronal=43/0.4574`，`sagittal=0/0.0000`。`top_true_margin` 分布为 `36/48/10`，`top_weight_hit_rate.true_margin=0.7128`；也就是说，routing alignment 比 DFR-75 的 `0.5851` 改善，但 sagittal true-margin 样本仍没有一次获得 top-weight，accuracy 只到 equal-weight tie。
- **当前判断**：DFR-76 是 negative seed42 candidate study，不应 promotion 到 3-seed formal。它证明 signed class-consensus 可以过滤一部分 DFR-75 的孤立 coronal over-routing，但仍不足以识别 sagittal 支持样本或把 learned full-fusion 推过 matched equal-weight。下一轮如果继续 candidate/hierarchical selection，不能继续扫 consensus margin；更值得做样本级 trigger audit，或引入更可靠的 supervised/prototype alignment signal，让 candidate selection 具备类别语义而不是只读 per-view logit geometry。

### DFR-77 detached class-consensus candidate-view reliability gate（2026-05-12）

> **实验说明**
> - 本轮继续从 DFR-25 anchor 出发，并严格只改 DFR-76 的 candidate-view reliability gate：新增 `ANKLE_DECISION_CANDIDATE_VIEW_GATE_DETACH_FEATURES=1`，让 candidate gate 的 feature adapter 读取 detached view features；ResNeXt 256x8、learned decision fusion、L3 no-mixer、dominant-gate dropout、winning scalar、class-consensus 条件与 seed42 candidate-study 语义保持不变。
> - 设计思路：DFR-76 的 signed class-consensus guard 已经比 DFR-75 更少 coronal over-routing，但 accuracy 只打平 equal-weight。一个可能失败点是 candidate gate 的 residual loss 反向改动 per-view feature/classifier，使 non-axial routing 信号和原本强 axial evidence 互相干扰。DFR-77 因此只把 candidate gate feature path detach，检验它能否作为纯 reliability-side correction 保留 DFR-25 强样本 ranking。
> - 预计改进效果：相对 DFR-76 best 的 `top_weight axial/coronal/sagittal=51/43/0`，预期 detach 会削弱错误 coronal 迁移，但不应回到 `94/0/0`；理想形态是 axial 重新主导强 axial-evidence 样本，同时 consensus-supported non-axial 样本仍获得少量 top/near-top weight。如果 DFR-76 的损失来自 feature/classifier 干扰而不是 candidate 触发本身，这条纯 routing correction 有机会把 learned full-fusion 推过 matched equal-weight seed42，而不是只把权重平均化或只改善单视角表现。

- [x] **DFR-77-RESNEXT-DECISION-256X8-DETACHED-CLASS-CONSENSUS-CANDIDATE-GATE-SEED42-MAIN-STUDY**：commit `800c21e`，fresh adaptive `main-study` 使用 [autoresearch_logs/generated_search_configs/optuna_main_search_iter_0004_20260512_042419.yaml](/dataset/HH/ankle-ct/autoresearch_logs/generated_search_configs/optuna_main_search_iter_0004_20260512_042419.yaml)，study root [runs/optuna_main_autoloop/iter_0004_20260512_042419](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0004_20260512_042419)，GPU policy `--gpu-ids 0,2,3 --max-workers 3`，只跑 seed42 三个保守候选 `ANKLE_DECISION_CANDIDATE_VIEW_GATE_RESIDUAL_LIMIT=0.35/0.50/0.65`，并固定 `margin_gap=0.5`, `require_class_consensus=1`, `consensus_margin=0.35`, `consensus_window=0.20`, `detach_features=1`。实验实际结果：trial0/1/2 全部为 `0.9042553191489362/0.9786363636363636/0.8888888888888888`；best trial0 `total_seconds≈601.2`, monitor `vram_mb≈2245` → **discard**（低于 DFR-25 seed42 `0.9361702127659575/0.9786363636363636/0.9333333333333333`，也低于 matched equal-weight seed42 `0.9255319148936170` 的主指标；AUC 持平 DFR-25 seed42 不能覆盖 accuracy/F1 回落。）
- **fusion-weight 结论**：best trial0 的 [fusion_weight_analysis.json](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0004_20260512_042419/trials/trial_0000/run/fusion_weight_analysis.json) 显示 detach 后 candidate gate 基本失去 DFR-76 的 anti-collapse routing。mean fusion weight `axial/coronal/sagittal = 0.8991/0.0521/0.0487`；top-weight count/rate `axial=94/1.0000`，`coronal=0/0.0000`，`sagittal=0/0.0000`。`top_true_margin` 分布为 `85/1/8`，`top_weight_hit_rate.true_margin=0.9043`；也就是说，routing 和 true-margin 的表面 hit rate 变高主要来自 axial 独占，而不是 right-view-at-right-sample 的 non-axial contribution。
- **当前判断**：DFR-77 是 negative seed42 candidate study，不应 promotion 到 3-seed formal。它说明 DFR-76 的 residual routing 迁移依赖可训练 feature path；一旦 detach，candidate selection 回到 axial single-view dependence，accuracy 反而低于 equal-weight。这关闭了“只把 candidate gate 做成纯 detached reliability correction”这一最小修复。下一轮如果继续 candidate/hierarchical selection，应先做样本级 trigger audit 或引入显式 supervised/prototype alignment，让 non-axial candidate 的触发目标更可靠；不要继续扫 `residual_limit` 或仅在 detach/不 detach 间切换。

### DFR-78 same-class probability-advantage candidate gate（2026-05-12）

> **实验说明**
> - 本轮继续从 DFR-25 anchor 与 DFR-76 class-consensus candidate gate 出发；ResNeXt 256x8、learned decision fusion、L3 no-mixer、dominant-gate dropout、winning scalar、candidate margin trigger 与 signed class-consensus 条件均保持不变。
> - 设计思路：DFR-76 虽把 DFR-75 的 coronal over-routing 收窄到 `top_weight axial/coronal/sagittal=51/43/0`，但仍只打平 matched equal-weight。DFR-78 因此新增一个默认关闭的 same-class probability-advantage guard：non-axial candidate 除了 margin advantage 与 class consensus 外，还必须在自己预测的类别上比 axial 对同一类别的概率更高，才允许获得 bounded residual reliability。
> - 预计改进效果：相对 DFR-76 的 `51/43/0`，预期 coronal top-routing 进一步回落，axial 在强 axial-evidence 样本重新保持多数 top-weight；只有同时满足 margin、方向一致和同类概率优势的 non-axial 样本获得 top/near-top routing。若这个 guard 能过滤孤立高 margin 的错误 coronal 样本，同时保留真实 non-axial evidence，它才有机会把 learned full-fusion `val_acc` 推过 matched equal-weight seed42，而不是把权重机械平均或只改善单视角表现。

- [x] **DFR-78-RESNEXT-DECISION-256X8-SAME-CLASS-PROB-ADVANTAGE-CANDIDATE-GATE-SEED42-MAIN-STUDY**：commit `45f6bbf`，fresh adaptive `main-study` 使用 [autoresearch_logs/generated_search_configs/optuna_main_search_iter_0005_20260512_044723.yaml](/dataset/HH/ankle-ct/autoresearch_logs/generated_search_configs/optuna_main_search_iter_0005_20260512_044723.yaml)，study root [runs/optuna_main_autoloop/iter_0005_20260512_044723](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0005_20260512_044723)，GPU policy `--gpu-ids 0,2,3 --max-workers 3`，只跑 seed42 三个保守候选 `ANKLE_DECISION_CANDIDATE_VIEW_GATE_PROB_ADVANTAGE=0.02/0.05/0.08`，并固定 `prob_window=0.05`, `residual_limit=0.50`, `margin_gap=0.5`, `require_class_consensus=1`, `consensus_margin=0.35`, `consensus_window=0.20`。实验实际结果：trial0 `0.9042553191489362/0.9713636363636364/0.8888888888888888`，trial1 `0.8936170212765957/0.9536363636363636/0.8913043478260869`，trial2 `0.8936170212765957/0.9395454545454546/0.8837209302325582`；best trial0 `total_seconds≈608.4`, monitor `vram_mb≈2245` → **discard**（低于 DFR-25 seed42 `0.9361702127659575/0.9786363636363636/0.9333333333333333`，也低于 matched equal-weight seed42 `0.9255319148936170` 的主指标。）
- **fusion-weight 结论**：三条候选均已刷新 `fusion_weight_analysis.json`。best trial0 的 [fusion_weight_analysis.json](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0005_20260512_044723/trials/trial_0000/run/fusion_weight_analysis.json) 显示 same-class probability guard 没有把 DFR-76 的 coronal over-routing 拉回正确样本级选择：mean fusion weight `axial/coronal/sagittal = 0.4087/0.5371/0.0542`；top-weight count/rate `axial=39/0.4149`，`coronal=55/0.5851`，`sagittal=0/0.0000`；`top_weight_hit_rate.true_margin=0.5638`，低于 DFR-76 best 的 `0.7128`。更强阈值反而进一步 coronal collapse：trial1 mean `0.2212/0.7361/0.0427`、top-weight `14/80/0`；trial2 mean `0.0712/0.8844/0.0444`、top-weight `1/93/0`。
- **当前判断**：DFR-78 是 negative seed42 candidate study，不应 promotion 到 3-seed formal。它说明 same-class probability advantage 不是可靠的 right-view trigger；阈值越强越偏向 coronal，表明该概率比较会放大训练后 coronal 的同类置信偏置，而不是筛出真实 non-axial contribution。下一轮如果继续 candidate/hierarchical selection，不应再扫 probability/margin/consensus 阈值；应先做样本级 trigger audit 或引入 supervised/prototype alignment signal，让 candidate trigger 对应真实样本级证据。

### DFR-79 prototype-aligned candidate gate（2026-05-12）

> **实验说明**
> - 本轮继续从 DFR-25 anchor 与 DFR-76 class-consensus candidate gate 出发；保持 ResNeXt 256x8、learned decision fusion、L3 no-mixer、dominant-gate dropout、winning scalar、candidate margin trigger 与 signed class-consensus 条件不变。
> - 设计思路：DFR-75/76/78 证明 candidate trigger 能改变 routing，但 margin/probability geometry 无法保证迁移到真正有用视角；本轮新增 class-prototype alignment guard，让 candidate gate 学到两个轻量 class prototype，并要求 non-axial 视角在自己预测类别的 prototype 相似度上相对 axial 有优势后才允许 residual reliability。
> - 预计改进效果：相对 DFR-78 best 的 coronal-heavy `top_weight axial/coronal/sagittal=39/55/0`，预期 prototype guard 会过滤孤立 coronal confidence bias，使 routing 回到 axial-majority，但仍保留少量 prototype-aligned coronal/sagittal top/near-top 样本；如果这种 prototype 相似度更接近真实 right-view evidence，则 learned full-fusion 有机会超过 matched equal-weight seed42，而不是把权重机械平均或只提升单视角表现。

- [x] **DFR-79-RESNEXT-DECISION-256X8-PROTOTYPE-ALIGNED-CANDIDATE-GATE-SEED42-MAIN-STUDY**：commit `65d15aa`，fresh adaptive `main-study` 使用 [autoresearch_logs/generated_search_configs/optuna_main_search_iter_0006_20260512_051002.yaml](/dataset/HH/ankle-ct/autoresearch_logs/generated_search_configs/optuna_main_search_iter_0006_20260512_051002.yaml)，study root [runs/optuna_main_autoloop/iter_0006_20260512_051002](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0006_20260512_051002)，GPU policy `--gpu-ids 0,2,3 --max-workers 3`，只跑 seed42 三个保守候选 `ANKLE_DECISION_CANDIDATE_VIEW_GATE_PROTOTYPE_ADVANTAGE=0.00/0.05/0.10`，并固定 `prototype_window=0.05`, `residual_limit=0.50`, `margin_gap=0.5`, `require_class_consensus=1`, `consensus_margin=0.35`, `consensus_window=0.20`。实验实际结果：trial0 `0.9042553191489362/0.9354545454545455/0.8915662650602410`，trial1 `0.9042553191489362/0.9568181818181818/0.8888888888888888`，trial2 `0.9042553191489362/0.9650000000000000/0.8965517241379310`；best by `val_acc` tie-break AUC 为 trial2，`peak_vram≈2.19 GiB`, `total_seconds≈614.8` → **discard**（三个候选主指标全部低于 DFR-25 seed42 `0.9361702127659575/0.9786363636363636/0.9333333333333333`，也低于 matched equal-weight seed42 `0.9255319148936170`。）
- **fusion-weight 结论**：三条候选均已刷新 `fusion_weight_analysis.json`。best trial2 的 [fusion_weight_analysis.json](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0006_20260512_051002/trials/trial_0002/run/fusion_weight_analysis.json) 显示 prototype guard 没有产生 right-view-at-right-sample 迁移，而是基本恢复 axial dependence。mean fusion weight `axial/coronal/sagittal = 0.8861/0.0898/0.0240`；top-weight count/rate `axial=93/0.9894`，`coronal=1/0.0106`，`sagittal=0/0.0000`；`top_true_margin` 仍有 `axial/coronal/sagittal = 77/12/5`，`top_weight_hit_rate.true_margin=0.8085`，说明 `17` 个 non-axial true-margin 样本几乎没有进入 top routing。候选趋势为 trial0 mean `0.9059/0.0503/0.0438`、top `94/0/0`；trial1 mean `0.8226/0.1493/0.0281`、top `94/0/0`；trial2 mean `0.8861/0.0898/0.0240`、top `93/1/0`。
- **当前判断**：DFR-79 是 negative seed42 candidate study，不应 promotion 到 3-seed formal。轻量 learned prototype guard 过滤了 DFR-78 的 coronal over-routing，但也把 candidate gate 拉回 axial single-view dependence，没有让 sagittal/coronal 在真实有用样本上贡献。下一轮不应继续扫 prototype advantage/window；若继续 candidate/hierarchical selection，应先做样本级 trigger audit，或把 prototype alignment 做成显式 supervised/center-style auxiliary signal 后再用于 routing，而不是仅靠 gate 内部自由 prototype。

### DFR-80 supervised prototype auxiliary candidate gate（2026-05-12）

> **实验说明**
> - 本轮继续从 DFR-25 anchor 与 DFR-76/79 class-consensus candidate gate 出发；保持 ResNeXt 256x8、learned decision fusion、L3 no-mixer、dominant-gate dropout、winning scalar、candidate margin trigger、signed class-consensus 条件不变。
> - 设计思路：DFR-79 的 learned prototype guard 能过滤 DFR-78 的 coronal over-routing，但 prototype 是 gate 内自由几何，最终又回到 axial dependence。本轮只新增 supervised prototype auxiliary loss：candidate gate 输出三视角 class-prototype logits，并复用现有辅助 CE hook 让这些 prototype logits 直接受标签监督，目标是让 prototype alignment 更接近“哪个视角在该样本上有可用类别证据”。
> - 预计改进效果：相对 DFR-79 best 的 `top_weight axial/coronal/sagittal=93/1/0`，预期 supervised prototypes 会使 coronal/sagittal 在自身 class evidence 正确且与 consensus/margin trigger 一致的样本上获得更可信的 residual routing，top-weight 从 axial `93-94/94` 迁移到少量 evidence-selected non-axial 样本，而不是机械平均权重。若监督后的 prototype 几何能把 DFR-79 的 `17` 个 non-axial true-margin 样本中一部分送入 top/near-top routing，learned full-fusion 才有机会超过 matched equal-weight seed42。

- [x] **DFR-80-RESNEXT-DECISION-256X8-SUPERVISED-PROTOTYPE-AUX-CANDIDATE-GATE-SEED42-MAIN-STUDY**：commit `0d0fae5`，fresh adaptive `main-study` 使用 [autoresearch_logs/generated_search_configs/optuna_main_search_iter_0007_20260512_053305.yaml](/dataset/HH/ankle-ct/autoresearch_logs/generated_search_configs/optuna_main_search_iter_0007_20260512_053305.yaml)，study root [runs/optuna_main_autoloop/iter_0007_20260512_053305](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0007_20260512_053305)，GPU policy `--gpu-ids 0,2,3 --max-workers 3`，只跑 seed42 三个保守候选 `ANKLE_DECISION_CANDIDATE_VIEW_PROTOTYPE_AUX_WEIGHT=0.02/0.05/0.10`，并固定 `temperature=0.2`, `residual_limit=0.50`, `margin_gap=0.5`, `require_class_consensus=1`, `consensus_margin=0.35`, `consensus_window=0.20`, `prototype_advantage=0.0`, `prototype_window=0.05`。实验实际结果：trial0 `0.8936170212765957/0.9500000000000001/0.8780487804878049`，trial1 `0.8936170212765957/0.9550000000000001/0.8837209302325582`，trial2 `0.9148936170212766/0.9472727272727273/0.9000000000000000`；best by `val_acc` 为 trial2，`peak_vram≈2.19 GiB`, `total_seconds≈613.4` → **discard**（低于 DFR-25 seed42 `0.9361702127659575/0.9786363636363636/0.9333333333333333`，也低于 matched equal-weight seed42 `0.9255319148936170` 的主指标。）
- **fusion-weight 结论**：三条候选均已刷新 `fusion_weight_analysis.json`。best trial2 的 [fusion_weight_analysis.json](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0007_20260512_053305/trials/trial_0002/run/fusion_weight_analysis.json) 显示 supervised prototype aux 没有把 non-axial true-margin 样本送入 routing，而是保持 axial top-weight 全独占。mean fusion weight `axial/coronal/sagittal = 0.9176/0.0320/0.0504`；top-weight count/rate `axial=94/1.0000`，`coronal=0/0.0000`，`sagittal=0/0.0000`；`top_true_margin` 仍有 `axial/coronal/sagittal = 83/4/7`，说明 `11` 个 non-axial true-margin 样本没有一次进入 top routing。候选趋势为 trial0 mean `0.7745/0.1163/0.1092`、top `94/0/0`；trial1 mean `0.9164/0.0380/0.0456`、top `94/0/0`；trial2 mean `0.9176/0.0320/0.0504`、top `94/0/0`。
- **当前判断**：DFR-80 是 negative seed42 candidate study，不应 promotion 到 3-seed formal。它说明把 prototype alignment 直接做成 per-view supervised auxiliary CE 仍不能让 candidate gate 学到 right-view-at-right-sample routing；更强 aux weight 只提升到 `0.9149`，但仍低于 equal-weight，且 best checkpoint 完全 axial top-weight。下一轮不应继续扫 prototype aux weight/temperature 或回到 candidate threshold 微调；更合理的方向是先做样本级 trigger audit，或换成更直接的 pairwise reliability / view-graph evidence agreement 机制，要求 non-axial 触发同时对应真实样本级增益。

### DFR-81 pairwise agreement candidate gate（2026-05-12）

> **实验说明**
> - 本轮继续从 DFR-25 anchor 与 DFR-76 class-consensus candidate gate 出发；保持 ResNeXt 256x8、learned decision fusion、L3 no-mixer、dominant-gate dropout、winning scalar、candidate margin trigger 与 signed class-consensus 条件不变。
> - 设计思路：DFR-75/76/78 证明 candidate gate 能移动 routing，但孤立 coronal 高置信很容易变成 over-routing；DFR-79/80 又证明 prototype/prototype-aux 会把 gate 拉回 axial dependence。本轮只新增一个 pairwise non-axial evidence-agreement guard：coronal candidate 必须得到 sagittal 同方向 signed-margin 支持，sagittal candidate 必须得到 coronal 支持，axial 不额外加分。
> - 预计改进效果：相对 DFR-80 的 `top_weight axial/coronal/sagittal=94/0/0` 与 DFR-76 的 `51/43/0`，预期 routing 会保持 axial-majority，但只在两个 non-axial view 对同一 abnormal/normal 方向同时有证据时，给 coronal/sagittal 少量 top/near-top 机会。若这种 pairwise agreement 能过滤孤立 coronal 偏置并覆盖真实 non-axial true-margin 样本，learned full-fusion 才有机会超过 matched equal-weight seed42，而不是把权重机械平均或只改善单视角。

- [x] **DFR-81-RESNEXT-DECISION-256X8-PAIRWISE-AGREEMENT-CANDIDATE-GATE-SEED42-MAIN-STUDY**：commit `199b412`，fresh adaptive `main-study` 使用 [autoresearch_logs/generated_search_configs/optuna_main_search_iter_0008_20260512_055427.yaml](/dataset/HH/ankle-ct/autoresearch_logs/generated_search_configs/optuna_main_search_iter_0008_20260512_055427.yaml)，study root [runs/optuna_main_autoloop/iter_0008_20260512_055427](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0008_20260512_055427)，GPU policy `--gpu-ids 0,2,3 --max-workers 3`，只跑 seed42 三个保守候选 `ANKLE_DECISION_CANDIDATE_VIEW_GATE_NONAXIAL_PAIR_CONSENSUS_MARGIN=0.10/0.20/0.30`，并固定 `residual_limit=0.50`, `margin_gap=0.5`, `require_class_consensus=1`, `consensus_margin=0.35`, `consensus_window=0.20`, `nonaxial_pair_consensus_window=0.20`。实验实际结果：trial0 `0.8936170212765957/0.9509090909090909/0.8780487804878049`，trial1 `0.9148936170212766/0.9427272727272727/0.9024390243902439`，trial2 `0.9042553191489362/0.9354545454545454/0.8915662650602410`；best by `val_acc` 为 trial1，monitor `vram_mb≈2245`, `total_seconds≈621.2` → **discard**（best seed42 低于 DFR-25 seed42 `0.9361702127659575/0.9786363636363636/0.9333333333333333`，也低于 matched equal-weight seed42 `0.9255319148936170` 的主指标；AUC/F1 也没有补偿信号。）
- **fusion-weight 结论**：三条候选均已刷新 `fusion_weight_analysis.json`。best trial1 的 [fusion_weight_analysis.json](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0008_20260512_055427/trials/trial_0001/run/fusion_weight_analysis.json) 显示 pairwise agreement guard 只产生很弱的 coronal routing 迁移，仍未让 sagittal 进入 top-weight。mean fusion weight `axial/coronal/sagittal = 0.7997/0.1571/0.0432`；top-weight count/rate `axial=88/0.9362`，`coronal=6/0.0638`，`sagittal=0/0.0000`；`top_true_margin` 分布为 `78/9/7`，`top_weight_hit_rate.true_margin=0.7660`。候选趋势为 trial0 mean `0.8359/0.1375/0.0267`、top `94/0/0`；trial2 mean `0.8347/0.1256/0.0397`、top `92/2/0`。也就是说，margin `0.20` 确实从 axial collapse 中释放了 6 个 coronal top-weight 样本，但没有把 `7` 个 sagittal true-margin 样本纳入 routing，且 accuracy 仍低于 equal-weight。
- **当前判断**：DFR-81 是 negative seed42 candidate study，不应 promotion 到 3-seed formal。pairwise non-axial agreement 比 DFR-80 更能打破少量 axial top-weight 独占，也比 DFR-78 的 coronal over-routing更保守，但 signal 太弱且没有转化为主指标；继续细扫 pairwise margin/window 很可能只是 threshold tuning。下一轮如果继续同一主线，应优先做样本级 trigger audit / offline threshold study，找出 `top_true_margin` 为 sagittal/coronal 的样本里哪些非轴向证据能真实修正 DFR-25 错误，再决定是否做 view-graph 或 hierarchical selection，而不是继续添加新的 candidate guard。

### DFR-82 DFR-25 non-axial trigger audit（2026-05-12）

> **实验说明**
> - 本轮按 DFR-81 后的建议做 analysis-only trigger audit，不训练新模型、不新增 gate、不改 backbone / geometry / fusion family；新增只读脚本 [scripts/analyze_dfr25_trigger_audit.py](/dataset/HH/ankle-ct/scripts/analyze_dfr25_trigger_audit.py)，读取已有 `fusion_weight_analysis.json` telemetry。
> - 设计思路：DFR-75~81 已经证明 margin / consensus / prototype / pairwise candidate gate 可以移动 routing，但没有把 non-axial migration 转成正确样本级贡献；因此先审计 DFR-25 错样本里到底有多少可由 coronal/sagittal 修复，以及这些样本是否能被 label-free trigger 捕捉。
> - 预计改进效果：如果存在合格触发器，它应该只在 DFR-25 错误且 non-axial view 确实正确的少量样本上建议 coronal/sagittal top/near-top routing；预期全局权重仍保持 DFR-25 的 axial-majority，而不是平均化。这样的 high-precision trigger 才可能让后续 learned full-fusion 在保留 DFR-25 强 axial 通道的同时补回 non-axial 错样本，推过 matched equal-weight 和 DFR-25 ceiling。

- [x] **DFR-82-RESNEXT-DECISION-256X8-DFR25-TRIGGER-AUDIT-ANALYSIS**：commit `d52919f`，分析输出为 [autoresearch_logs/dfr82_trigger_audit/report.json](/dataset/HH/ankle-ct/autoresearch_logs/dfr82_trigger_audit/report.json)（ignored log artifact，仓库内可复查）。实验实际结果：不产生新训练指标，引用 DFR-25 3-seed reference `val_acc/val_auc/val_f1 = 0.9397163120567376 / 0.9677272727272728 / 0.9358934169278997`。DFR-25 权重 telemetry 为 seed42 mean `0.8886/0.0791/0.0324`, top `94/0/0`；seed123 mean `0.5253/0.1159/0.3588`, top `49/0/45`；seed456 mean `0.8701/0.0658/0.0640`, top `94/0/0`。
- **trigger audit 结论**：三 seed 聚合的 DFR-25 错样本共 `17` 个（`FN=8`, `FP=9`），其中 `top_true_margin` 为 `axial/coronal/sagittal=1/6/10`，说明确实存在 non-axial oracle 修复空间。oracle `view_correct` trigger 可修复 `11/17` 个 DFR-25 错误且 `harm=0`；但可训练前可用的 label-free triggers 全部失败：`pred_margin_gap_0.5` fired `60`、helpful/harmful `0/4`；`class_consensus_0.35` fired `45`、`0/1`；`pair_consensus_0.2` fired `8`、`0/1`；`nonaxial_abnormal_over_axial` fired `41`、`2/39`。也就是说，现有 detached logit 几何能证明有非轴向修复上限，却不能可靠识别该在何处迁移 routing。
- **comparator 复核**：DFR-76 seed42 best 权重 mean `0.4746/0.3865/0.1390`, top `51/43/0`，相对 DFR-25 seed42 fixed `4`、broken `5`、net `-1`；DFR-81 seed42 best mean `0.7997/0.1571/0.0432`, top `88/6/0`，fixed `4`、broken `6`、net `-2`。这与 audit 一致：candidate migration 能修复少数 DFR-25 错误，但同时打坏更多已正确样本。
- **当前判断**：DFR-82 是 **discard / analysis-only negative**，不 promotion 任何 DFR-75~81 trigger，也不继续 margin/window/threshold 微调。下一轮若继续主线，应转向更直接的 supervised reliability target 或 sample-level learned selector：训练目标必须逼近“view correctness / true-margin oracle”的 11 个可修复样本，而不是继续从 detached pred-margin、同向 consensus 或 abnormal-prob heuristic 中手写触发器。

### DFR-83 gate view-correctness auxiliary loss（2026-05-12）

> **实验说明**
> - 本轮从 DFR-82 的 audit 结论出发，继续严格保持 DFR-25 anchor：ResNeXt 256x8、learned decision fusion、L3 no-mixer、dominant-gate dropout、winning scalar 与 seed42 candidate-study 语义不变。
> - 唯一新增机制是在 [src/model.py](/dataset/HH/ankle-ct/src/model.py) 中加入默认关闭的 `ANKLE_DECISION_ENABLE_GATE_VIEW_CORRECTNESS_AUX_LOSS`。该 aux 不改 fused logits，不新增 candidate gate；它把 detached per-view logits 乘以 learned raw fusion weight 派生的 view scale，再复用现有辅助 CE hook，让标签监督只作用到 reliability/gate 权重路径，避免直接改变 per-view classifier 梯度。
> - 设计思路：DFR-82 说明 oracle view-correctness 可修复 `11/17` 个 DFR-25 错样本，但 label-free trigger 不能可靠识别。DFR-83 因此用轻量监督信号逼近 “view correctness / true-margin oracle”，目标是让 gate 学到哪一个 view 的 detached class evidence 更应该被信任，而不是继续手写 margin/consensus 阈值。
> - 预计改进效果：相对 DFR-25 seed42 的 hard axial `top_weight=94/0/0`，预期全局仍保持 axial-majority，但 coronal/sagittal 在少量标签监督支持、且自身 detached evidence 与正确类别一致的样本上获得 top/near-top routing；理想形态是 `top_weight axial` 从 `94` 降到约 `75-90`，coronal/sagittal 获得少量 top-weight，同时 `val_acc` 超过 matched equal-weight seed42 `0.9255319148936170`，证明不是机械平均或单视角改善。
> - 执行备注：第一次生成的 search-config copy 把 `fixed_overrides.runtime_env` 写成覆盖 base runtime env，导致无效预跑漏掉 `ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER` 与 `ANKLE_DECISION_TRAIN_DOMINANT_GATE_DROPOUT_PROB`；该无效 study 已清理，不纳入结论。随后 commit `4dbea28` 修复 search-config 并 fresh 重跑同一预留 `study_root`。

- [x] **DFR-83-RESNEXT-DECISION-256X8-GATE-VIEW-CORRECTNESS-AUX-SEED42-MAIN-STUDY**：implementation commit `9eb1314`，valid run commit `4dbea28`，fresh adaptive `main-study` 使用 [autoresearch_logs/generated_search_configs/optuna_main_search_iter_0010_20260512_062824.yaml](/dataset/HH/ankle-ct/autoresearch_logs/generated_search_configs/optuna_main_search_iter_0010_20260512_062824.yaml)，study root [runs/optuna_main_autoloop/iter_0010_20260512_062824](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0010_20260512_062824)，GPU policy `--gpu-ids 0,2,3 --max-workers 3`，只跑 seed42 三个保守候选 `ANKLE_DECISION_GATE_VIEW_CORRECTNESS_AUX_WEIGHT=0.005/0.01/0.02`。实验实际结果：trial0 `0.9148936170212766/0.9772727272727273/0.9111111111111111`，trial1 `0.9255319148936170/0.9736363636363636/0.9176470588235294`，trial2 `0.9148936170212766/0.9768181818181818/0.9111111111111111`；best by `val_acc` 为 trial1，`peak_vram≈2.19 GiB`, `total_seconds≈605.3` → **discard**（best seed42 只打平 matched equal-weight seed42，仍低于 DFR-25 seed42 `0.9361702127659575/0.9786363636363636/0.9333333333333333`；AUC/F1 也没有超过 DFR-25。）
- **fusion-weight 结论**：三条候选均已刷新 `fusion_weight_analysis.json`。best trial1 的 [fusion_weight_analysis.json](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0010_20260512_062824/trials/trial_0001/run/fusion_weight_analysis.json) 显示监督 gate aux 能明显移动 routing，但没有转成 accuracy。mean fusion weight `axial/coronal/sagittal = 0.6412/0.2104/0.1484`；top-weight count/rate `axial=77/0.8191`, `coronal=13/0.1383`, `sagittal=4/0.0426`；`top_true_margin` 分布为 `85/4/5`，`top_weight_hit_rate.true_margin=0.7660`。候选趋势为 trial0 mean `0.8941/0.0194/0.0865`、top `94/0/0`；trial2 mean `0.9251/0.0620/0.0129`、top `94/0/0`。也就是说，`0.01` 权重确实把少量 coronal/sagittal 推进 top-routing，但迁移没有命中足够多的 true-margin 有益样本，且更弱/更强 aux 都重新回到 axial lock-in。
- **当前判断**：DFR-83 是 negative seed42 candidate study，不应 promotion 到 3-seed formal。它比 DFR-75~81 更直接地证明“监督 reliability target 可以移动权重”，但当前实现仍只是通过 detached view logits 和 learned weight scale 做软 CE，不能区分 “迁移会修复 DFR-25 错误” 与 “迁移会打坏 DFR-25 已正确样本”。下一轮若继续监督式方向，应优先做 sample-level selector / pairwise ranking target 的离线可学习性审计，或者把监督信号限制在 DFR-82 指出的 mixed-view-correctness 样本上；不要继续扫 aux weight/base-scale/weight-scale。
- **Agent 状态（2026-05-12 07:11 SGT）**：本 session 是唯一 coordinator，已完成 exactly one research iteration；last experiment=`DFR-83 gate view-correctness auxiliary loss`，lane=`main-study`，result=`discard`，last commit=`4dbea28`；下一步建议=`sample-level selector / pairwise ranking target audit from DFR-82 oracle-view-correct cases`，继续服务 DFR-25 decision-fusion routing 修复，不切 backbone / geometry / fusion family。

### DFR-84 disagreement-gated view-correctness auxiliary loss（2026-05-12）

> **实验说明**
> - 本轮继续从 DFR-25 anchor 与 DFR-83 supervised gate view-correctness aux 出发；保持 ResNeXt 256x8、learned decision fusion、L3 no-mixer、dominant-gate dropout、winning scalar 不变。
> - 唯一新增机制是在 [src/model.py](/dataset/HH/ankle-ct/src/model.py) 给 DFR-83 aux 增加默认关闭开关 `ANKLE_DECISION_GATE_VIEW_CORRECTNESS_AUX_REQUIRE_DISAGREEMENT=1`：只有当 detached per-view logits 的预测类别在三视角之间出现分歧时，aux 才使用 learned fusion weight 放大对应 view logits；all-agree 样本退回 base scale，不再让 aux 全局推动 gate。
> - 设计思路：DFR-82 显示 oracle view-correctness 能修复 `11/17` 个 DFR-25 错样本，但 label-free trigger 会打坏已正确样本；DFR-83 则证明监督 gate aux 能移动 routing，却仍无法区分有益迁移与破坏性迁移。DFR-84 因此把监督信号限制到更接近 mixed-view-correctness 的 disagreement 场景，避免在三视角同向样本上继续扰动 DFR-25 强 axial 通道。
> - 预计改进效果：相对 DFR-83 best 的 `top_weight axial/coronal/sagittal=77/13/4`，预期全局 routing 会回到 axial-majority，但不应完全回到 DFR-25 的 `94/0/0`；理想形态是 `top_weight axial≈80-90`，coronal/sagittal 只在 disagreement 且有标签支持的样本上获得少量 top/near-top weight。如果这些迁移对应 DFR-82 的 oracle-correct non-axial 样本，就有机会把 learned full-fusion seed42 `val_acc` 推过 matched equal-weight `0.9255319148936170`，而不是机械平均或只改善 AUC。

- [x] **DFR-84-RESNEXT-DECISION-256X8-DISAGREEMENT-GATED-VIEW-CORRECTNESS-AUX-SEED42-MAIN-STUDY**：commit `3c457c0`，fresh adaptive `main-study` 使用 [autoresearch_logs/generated_search_configs/optuna_main_search_iter_0011_20260512_071305.yaml](/dataset/HH/ankle-ct/autoresearch_logs/generated_search_configs/optuna_main_search_iter_0011_20260512_071305.yaml)，study root [runs/optuna_main_autoloop/iter_0011_20260512_071305](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0011_20260512_071305)，GPU policy `--gpu-ids 0,2,3 --max-workers 3`，只跑 seed42 三个保守候选 `ANKLE_DECISION_GATE_VIEW_CORRECTNESS_AUX_WEIGHT=0.005/0.01/0.02` 且固定 `ANKLE_DECISION_GATE_VIEW_CORRECTNESS_AUX_REQUIRE_DISAGREEMENT=1`。实验实际结果：trial0 `0.9148936170212766/0.9736363636363636/0.9111111111111111`，trial1 `0.9148936170212766/0.9809090909090910/0.9090909090909091`，trial2 `0.9255319148936170/0.9813636363636364/0.9230769230769231`；best by `val_acc` 为 trial2，`peak_vram≈2.19 GiB`, `total_seconds≈613.6` → **discard**（best seed42 只打平 matched equal-weight seed42 `0.9255319148936170`，仍低于 DFR-25 seed42 `0.9361702127659575/0.9786363636363636/0.9333333333333333`；AUC 虽高但不能覆盖主指标未超过。）
- **fusion-weight 结论**：三条候选均已刷新 `fusion_weight_analysis.json`。best trial2 的 [fusion_weight_analysis.json](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0011_20260512_071305/trials/trial_0002/run/fusion_weight_analysis.json) 显示 disagreement mask 没有保留 DFR-83 的有效 routing 迁移，而是重新回到 axial top-weight 独占。mean fusion weight `axial/coronal/sagittal = 0.8981/0.0339/0.0681`；top-weight count/rate `axial=94/1.0000`, `coronal=0/0.0000`, `sagittal=0/0.0000`；`top_true_margin` 仍有 `axial/coronal/sagittal = 84/2/8`，`top_weight_hit_rate.true_margin=0.8936`。候选趋势为 trial0 mean `0.7034/0.1348/0.1618`, top `84/0/10`, accuracy `0.9149`；trial1 mean `0.8863/0.0401/0.0736`, top `94/0/0`, accuracy `0.9149`；trial2 mean `0.8981/0.0339/0.0681`, top `94/0/0`, accuracy `0.9255`。也就是说，唯一出现 sagittal top-routing 的低权重候选反而更差，best accuracy 候选没有 non-axial top contribution。
- **当前判断**：DFR-84 是 negative seed42 candidate study，不应 promotion 到 3-seed formal。简单 disagreement mask 太粗：它减少了 DFR-83 的全局扰动，但没有把监督信号精准对齐到 DFR-82 的 oracle-correct non-axial 错样本；要么回到 axial dependence，要么少量 sagittal migration 未转成 accuracy。下一轮若继续监督式 routing，应先做离线 pairwise ranking / selector 可学习性审计，直接围绕 DFR-25 错样本中的 oracle view-correct target 构造训练目标；不要继续扫 DFR-83/84 的 aux weight、base scale 或 disagreement 阈值。
- **Agent 状态（2026-05-12 07:36 SGT）**：本 session 是唯一 coordinator，已完成 exactly one research iteration；last experiment=`DFR-84 disagreement-gated view-correctness auxiliary loss`，lane=`main-study`，result=`discard`，last commit=`3c457c0`；下一步建议=`offline pairwise ranking / sample-level selector target audit from DFR-82 oracle-view-correct cases`，继续服务 DFR-25 decision-fusion routing 修复，不切 backbone / geometry / fusion family。

### DFR-85 DFR-25 selector audit（2026-05-12）

> **实验说明**
> - 本轮按 DFR-84 的推荐动作做 analysis-only 可学习性审计，不训练新 checkpoint、不改 backbone / geometry / fusion family，也不再扫 margin / consensus / probability 阈值。
> - 新增只读脚本 [scripts/analyze_dfr25_selector_audit.py](/dataset/HH/ankle-ct/scripts/analyze_dfr25_selector_audit.py)，直接读取已有 DFR-25 三 seed `fusion_weight_analysis.json`，评估两类 selector target：`oracle view-correct / true-margin` 上限，以及跨 seed 可学习的 multiclass / pairwise selector 是否能在 leave-one-seed-out 下保持样本级净收益。
> - 设计思路：DFR-82 已证明 oracle view-correctness 可以修复 DFR-25 的一部分错误，但 label-free trigger 都失败；因此本轮不是再发明 trigger，而是审计“如果把目标改成 selector 学习，现有 telemetry 是否真的支持一个能推广的 sample-level selector”。
> - 预计改进效果：如果 selector 目标足够可学，leave-one-seed-out 的 learned selector 应该能把 DFR-25 错误中的 oracle-correct non-axial 样本选出来，同时保护 axial 已正确样本，最终在样本级上表现出明显正净收益；否则这条路线只能停留在 analysis，不应进入下一轮训练。

- [x] **DFR-85-RESNEXT-DECISION-256X8-DFR25-SELECTOR-AUDIT-ANALYSIS**：commit `9f4c238`，analysis output [autoresearch_logs/dfr85_selector_audit/report.json](/dataset/HH/ankle-ct/autoresearch_logs/dfr85_selector_audit/report.json)。实验实际结果：不训练新 checkpoint；引用 DFR-25 3-seed reference `val_acc/val_auc/val_f1 = 0.9397163120567376 / 0.9677272727272728 / 0.9358934169278997`。oracle 上限显示 `any_view_correct_accuracy_upper_bound=0.982270`，Dfr25 错样本 `17` 个中 `12` 个可由任一正确视角修复，且 `11` 个属于 axial wrong + non-axial correct 的可迁移样本；但跨 seed learned selector 仍不成立：multiclass leave-one-seed-out `selected_view_accuracy=0.925532`, `fixed_dfr25_errors=1`, `broken_dfr25_correct=5`, `net=-4`，top 选择 `axial/coronal/sagittal=220/7/55`；pairwise leave-one-seed-out `selected_view_accuracy=0.904255`, `fixed_dfr25_errors=3`, `broken_dfr25_correct=13`, `net=-10`，top 选择 `axial/coronal/sagittal=214/6/62`。这说明 selector 目标在 oracle 层面有信号，但当前可学习实现会同时打坏更多已正确样本，因此本轮 **analysis-only discard**。
- **权重 / selector 结论**：DFR-25 三 seed telemetry 继续显示明显的样本异质性。oracle selector 仅作为上限时可将 212/17/53 的 sample 级选择分布映射到 `fixed=12 / broken=0`，其中 fixed 主要是 DFR-25 错样本里的 `FN=3 / FP=9`，且 `broken_summary` 为 0；但实际 multiclass / pairwise selector 都出现负净收益，说明目前缺少一个能稳定保留 axial strong-positive 与 true-negative 的可学习 ranking target。
- **当前判断**：DFR-85 是 **analysis-only negative**。它比 DFR-82 更具体地证明：oracle view-correct / true-margin 里确实存在可修复样本，但现有 sample-level selector 目标跨 seed 不可推广，且会打坏更多 DFR-25 已正确样本。下一轮若继续主线，应优先围绕 oracle-correct non-axial 错样本设计更保守的 supervised selector target 或 pairwise ranking target，但必须先证明它能同时覆盖 DFR-25 的 11 个 oracle-fixable non-axial 错样本并保护 axial 已正确样本；在此之前，不应把任何 selector 训练线推进到正式实验。
- **Agent 状态（2026-05-12 08:18 SGT）**：本 session 是唯一 coordinator，已完成 exactly one research iteration；last experiment=`DFR-85 DFR-25 selector audit`，lane=`analysis-only`, result=`discard`, last commit=`9f4c238`；下一步建议=`design a more conservative supervised selector target that is restricted to DFR-82 oracle-correct non-axial errors and explicitly protected axial strong-positive / TN cases`，继续服务 DFR-25 decision-fusion routing 修复，不切 backbone / geometry / fusion family。

### DFR-86 non-axial-only gate view-correctness auxiliary loss（2026-05-12）

> **实验说明**
> - 本轮继续从 DFR-25 anchor 与 DFR-83/84 supervised gate view-correctness aux 出发；保持 ResNeXt 256x8、learned decision fusion、L3 no-mixer、dominant-gate dropout、winning scalar 不变。
> - 唯一新增机制是在 [src/model.py](/dataset/HH/ankle-ct/src/model.py) 给 gate view-correctness aux 增加默认关闭开关 `ANKLE_DECISION_GATE_VIEW_CORRECTNESS_AUX_NONAXIAL_ONLY=1`：aux 仍不改 fused logits，只是把 learned raw fusion weight 的 aux scaling 限定到 coronal/sagittal，axial 只保留 base scale。
> - 设计思路：DFR-83 证明全局监督 gate aux 能移动 routing，但会同时扰动 axial 已正确样本；DFR-84 的 disagreement mask 又过粗，best 候选重新回到 axial `94/0/0`。DFR-86 只让标签监督通过 non-axial view 的 fusion weight 进入 aux CE，目标是奖励 coronal/sagittal 在自身 detached logits 对标签有帮助时获得可靠性，而不再全局强化 axial 或机械平均权重。
> - 预计改进效果：相对 DFR-84 best 的 `top_weight axial/coronal/sagittal=94/0/0`，预期 axial 仍主导强样本，但 `top_weight axial` 应降到约 `75-90`，coronal/sagittal 在少量 DFR-82 oracle-fixable 样本上获得 top/near-top routing；如果 non-axial-only 监督比 DFR-83 更少打坏 axial strong-positive / TN 样本，则 seed42 learned full-fusion 有机会超过 matched equal-weight `0.9255319148936170`，而不是只把权重做平均或只提升 AUC。

- [x] **DFR-86-RESNEXT-DECISION-256X8-NONAXIAL-ONLY-GATE-CORRECTNESS-AUX-SEED42-MAIN-STUDY**：commit `4c4b9cf`，fresh adaptive `main-study` 使用 [autoresearch_logs/generated_search_configs/optuna_main_search_iter_0013_20260512_074837.yaml](/dataset/HH/ankle-ct/autoresearch_logs/generated_search_configs/optuna_main_search_iter_0013_20260512_074837.yaml)，study root [runs/optuna_main_autoloop/iter_0013_20260512_074837](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0013_20260512_074837)，GPU policy `--gpu-ids 0,2,3 --max-workers 3`，只跑 seed42 三个保守候选 `ANKLE_DECISION_GATE_VIEW_CORRECTNESS_AUX_WEIGHT=0.005/0.01/0.02` 且固定 `ANKLE_DECISION_GATE_VIEW_CORRECTNESS_AUX_NONAXIAL_ONLY=1`。实验实际结果：trial0 `0.9255319148936170/0.9754545454545455/0.9230769230769231`，trial1 `0.9042553191489362/0.9786363636363636/0.8988764044943820`，trial2 `0.9255319148936170/0.9772727272727273/0.9176470588235294`；best by `val_acc` then `val_auc` 为 trial2，`peak_vram≈2.19 GiB`, `total_seconds≈606.4` → **discard**（best seed42 只打平 matched equal-weight seed42，仍低于 DFR-25 seed42 `0.9361702127659575/0.9786363636363636/0.9333333333333333`；AUC/F1 也未超过 DFR-25。）
- **fusion-weight 结论**：三条候选均已刷新 `fusion_weight_analysis.json`。best trial2 的 [fusion_weight_analysis.json](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0013_20260512_074837/trials/trial_0002/run/fusion_weight_analysis.json) 显示 non-axial-only aux 能打破 axial top-weight 独占，但迁移明显不对齐 true-margin。mean fusion weight `axial/coronal/sagittal = 0.5663/0.1515/0.2822`；top-weight count/rate `axial=60/0.6383`, `coronal=0/0.0000`, `sagittal=34/0.3617`；`top_true_margin` 分布却是 `axial/coronal/sagittal = 87/4/3`，`top_weight_hit_rate.true_margin=0.5957`。候选趋势为 trial0 mean `0.8882/0.0384/0.0734`, top `94/0/0`, accuracy `0.9255`；trial1 mean `0.8894/0.0392/0.0714`, top `94/0/0`, accuracy `0.9043`；trial2 才释放 sagittal top-routing，但没有提高 accuracy。
- **当前判断**：DFR-86 是 negative seed42 candidate study，不应 promotion 到 3-seed formal。它证明“只把 supervised aux 压到 non-axial”可以制造 sagittal routing，但这不是 right-view-at-right-sample：sagittal top-weight 远多于 true-margin 支持，且 coronal oracle/fixable 信号没有进入 top routing。下一轮不应继续扫 aux weight/base-scale/weight-scale；若继续监督式方向，需要在 aux/selector 中显式保护 DFR-25 已正确样本，或先做 sample-level error-aware audit 来定义更干净的 pairwise ranking target。
- **Agent 状态（2026-05-12 08:13 SGT）**：本 session 是唯一 coordinator，已完成 exactly one research iteration；last experiment=`DFR-86 non-axial-only gate view-correctness auxiliary loss`，lane=`main-study`，result=`discard`，last commit=`4c4b9cf`；下一步建议=`error-aware supervised pairwise ranking target / selector that explicitly protects DFR-25-correct samples before any promotion to training`，继续服务 DFR-25 decision-fusion routing 修复，不切 backbone / geometry / fusion family。

### DFR-87 axial-protected non-axial gate view-correctness auxiliary loss（2026-05-12）

> **实验说明**
> - 本轮继续从 DFR-25 anchor 与 DFR-86 supervised non-axial-only gate aux 出发；保持 ResNeXt 256x8、learned decision fusion、L3 no-mixer、dominant-gate dropout、winning scalar 不变。
> - 唯一新增机制是在 [src/model.py](/dataset/HH/ankle-ct/src/model.py) 给 gate view-correctness aux 增加默认关闭保护开关 `ANKLE_DECISION_GATE_VIEW_CORRECTNESS_AUX_PROTECT_STRONG_AXIAL=1` 与 `ANKLE_DECISION_GATE_VIEW_CORRECTNESS_AUX_AXIAL_MARGIN_CAP`：当 detached axial view logits 的预测 margin 高于 cap 时，coronal/sagittal 的 aux fusion-weight scaling 置零，只在 axial evidence 较弱的样本上允许 non-axial aux 学习 reliability。
> - 设计思路：DFR-86 证明 non-axial-only supervised aux 可以打破 axial collapse，但 best 候选把 `34/94` 个样本顶权重迁到 sagittal，而 true-margin 只有 `3/94` 个样本支持 sagittal，说明问题是“释放太多弱视角”而不是“仍然释放不够”。DFR-87 因此用 label-free axial margin 做保护栏，只在 axial 自身不强时给 coronal/sagittal 学习机会，避免扰动 DFR-25 已正确的强 axial / TN 样本。
> - 预计改进效果：相对 DFR-86 best 的 top-weight `60/0/34`，预期 routing 回到 axial-majority 但不完全回到 `94/0/0`；理想形态是 `top_weight axial≈75-90`，coronal/sagittal 只在 weak-axial 且标签监督支持的样本上获得 top/near-top weight。如果这些样本覆盖 DFR-82 oracle-fixable non-axial 错误，同时保护 DFR-25-correct strong axial 样本，则 seed42 learned full-fusion 有机会超过 matched equal-weight `0.9255319148936170`，而不是只把权重平均化或只提升 AUC。

- [x] **DFR-87-RESNEXT-DECISION-256X8-AXIAL-PROTECTED-NONAXIAL-GATE-AUX-SEED42-MAIN-STUDY**：commit `367b1d7`，fresh adaptive `main-study` 使用 [autoresearch_logs/generated_search_configs/optuna_main_search_iter_0014_20260512_080950.yaml](/dataset/HH/ankle-ct/autoresearch_logs/generated_search_configs/optuna_main_search_iter_0014_20260512_080950.yaml)，study root [runs/optuna_main_autoloop/iter_0014_20260512_080950](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0014_20260512_080950)，GPU policy `--gpu-ids 0,2,3 --max-workers 3`，只跑 seed42 三个保守候选 `ANKLE_DECISION_GATE_VIEW_CORRECTNESS_AUX_AXIAL_MARGIN_CAP=2.0/2.5/3.0`，固定 `ANKLE_DECISION_GATE_VIEW_CORRECTNESS_AUX_NONAXIAL_ONLY=1` 与 `ANKLE_DECISION_GATE_VIEW_CORRECTNESS_AUX_PROTECT_STRONG_AXIAL=1`。实验实际结果：trial0 `0.9148936170212766/0.9840909090909091/0.9111111111111111`，trial1 `0.9148936170212766/0.9763636363636363/0.9111111111111111`，trial2 `0.9042553191489362/0.9631818181818181/0.8988764044943820`；best by `val_acc` then `val_auc` 为 trial0，`peak_vram≈2.19 GiB`, `total_seconds≈608.3` → **discard**（best seed42 低于 matched equal-weight seed42 `0.9255319148936170` 与 DFR-25 seed42 `0.9361702127659575/0.9786363636363636/0.9333333333333333`；AUC 虽高但主指标与 F1 均未达标。）
- **fusion-weight 结论**：三条候选均已刷新 `fusion_weight_analysis.json`。best trial0 的 [fusion_weight_analysis.json](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0014_20260512_080950/trials/trial_0000/run/fusion_weight_analysis.json) 显示 axial protection 没有形成保守的 evidence-selected non-axial routing，而是重新恢复 axial top-weight 独占。mean fusion weight `axial/coronal/sagittal = 0.9070/0.0411/0.0520`；top-weight count/rate `axial=94/1.0000`, `coronal=0/0.0000`, `sagittal=0/0.0000`；`top_true_margin` 分布为 `axial/coronal/sagittal = 85/6/3`，`top_weight_hit_rate.true_margin=0.9043`。候选趋势为 trial1 mean `0.9161/0.0273/0.0566`, top `94/0/0`, true-margin `82/2/10`；trial2 mean `0.7756/0.1454/0.0790`, top `94/0/0`, true-margin `84/2/8`。也就是说，更宽 cap 只增加 non-axial 平均质量，但仍没有任何 non-axial top-routing，也没有 accuracy gain。
- **当前判断**：DFR-87 是 negative seed42 candidate study，不应 promotion 到 3-seed formal。它说明用 axial confidence 做硬保护会过度抑制 DFR-86 中仅有的 non-axial routing 迁移，回到 single-view axial dependence；同时 best accuracy 已低于 matched equal-weight。下一轮不要继续扫 axial-margin cap，也不要继续在 DFR-83/86 aux family 上做幅度微调；如果外层 loop 继续，应转向更直接的 error-aware pairwise / selector target，目标必须同时证明能保护 DFR-25-correct 样本并只释放 oracle-correct non-axial 错样本。
- **Agent 状态（2026-05-12 08:30 SGT）**：本 session 是唯一 coordinator，已完成 exactly one research iteration；last experiment=`DFR-87 axial-protected non-axial gate view-correctness auxiliary loss`，lane=`main-study`，result=`discard`，last commit=`367b1d7`；下一步建议=`stop axial-margin cap / aux amplitude sweeps; use error-aware supervised pairwise or selector target with explicit DFR-25-correct protection and oracle-fixable non-axial release`，继续服务 DFR-25 decision-fusion routing 修复，不切 backbone / geometry / fusion family。

---

## 2026-05-10：Decision-Fusion Repair（DFR-44 evidence-aware residual gate，3-seed formal，RTXA6Kq/node16）

> **实验说明**
> - 本轮严格从当前 operational mainline `DFR-25 ResNeXt 256x8 learned decision fusion` fork：不改 backbone family、不改 geometry、不切 feature fusion、不改 DFR-25 winning scalar；唯一结构变量是在 decision-fusion reliability path 上新增 **evidence-aware residual gate**。
> - 代码落点是 [src/model.py](/dataset/HH/ankle-ct/src/model.py)：新增 `EvidenceAwareReliabilityGate`，用 detached per-view logits 派生 `max_prob / margin / entropy / abnormal_prob` 及其跨视角 centered evidence，再与每个视角的 gating feature 拼接，输出一个 identity-initialized residual reliability-logit correction。初始时 residual 为 0，等价于 DFR-25；训练中才学习 evidence-to-routing correction。
> - runtime env 保持 DFR-25 的 `ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER=1` + `ANKLE_DECISION_TRAIN_DOMINANT_GATE_DROPOUT_PROB=0.25`，只额外开启 `ANKLE_DECISION_ENABLE_EVIDENCE_AWARE_GATE=1` 与 `ANKLE_DECISION_EVIDENCE_AWARE_GATE_RESIDUAL_LIMIT=0.25`。
> - 设计思路 / 预计改进效果：DFR-25 已完成 learned full-fusion > matched equal-weight 的 3-seed 主指标目标，但 telemetry 仍显示 seed42/456 存在 residual axial lock-in。相比继续加训练期 floor / teacher redistribution，本轮把改动收敛成一个可解释的网络结构模块，让 gate 在推理期也能读到 per-view classifier evidence，目标是让非 axial 视角在自身 margin / confidence 明确更强时获得更稳定的 routing 权重，同时不把权重机械拉平均。
> - 执行记录：先逐节点排查 `V100q` 的 3-GPU 空闲资源；`node19` 只剩 2 张空闲 GPU，`node21` 已被 3-GPU 作业占满，唯一 strict idle 的 `node20` 在 Slurm 可分配但 PyTorch CUDA probe 失败（单卡也 `torch.cuda.is_available=False`）。因此改用已验证可用的 `RTXA6Kq/node16` Slurm 3-GPU allocation（visible physical GPU `5,6,7`），不是整节点独占，但三张卡空闲、lane probe 通过。
> - Slurm job `481492`：`COMPLETED 0:0`，elapsed `00:12:29`；三条 seed 训练和三条 `fusion_weight_analysis` telemetry 均 exit 0。聚合文件：[autoresearch_logs/dfr44_evidence_gate_multiseed/aggregate_summary.json](/dataset/HH/ankle-ct/autoresearch_logs/dfr44_evidence_gate_multiseed/aggregate_summary.json)。

- [x] **DFR-44-RESNEXT-DECISION-256X8-EVIDENCE-AWARE-RESIDUAL-GATE-MULTISEED-FORMAL**：commit `5da110b`，formal seeds `42/123/456`，保持 DFR-25 anchor scalar 与 runtime，只新增 inference-time evidence-aware residual gate → `seed42=0.882979/0.943636/0.873563`, `seed123=0.914894/0.958636/0.911111`, `seed456=0.882979/0.949545/0.867470`，3-seed mean `val_acc=0.8936170212765958`, `val_auc=0.9506060606060606`, `val_f1=0.8840480696733293`, `peak_vram≈2.20 GiB` → **discard**（低于 DFR-25 3-seed mean `0.9397163120567376 / 0.9677272727272728 / 0.9358934169278997`，`val_acc / val_auc / val_f1` 分别回落 `0.0460992907801418 / 0.0171212121212122 / 0.0518453472545704`；也低于 matched equal-weight 3-seed mean `0.9219858156028368 / 0.9606060606060606 / 0.9112221100424511`。）
- **fusion-weight 结论**：DFR-44 没有稳定解决 DFR-25 的 residual axial lock-in。`seed42` 与 `seed123` 仍是 `top_weight axial=94/94`，mean axial weight 分别为 `0.9878` 与 `0.9122`；只有 `seed456` 出现 `top_weight axial/coronal=42/52`、mean weight `0.4756/0.4542/0.0702`，但该 seed accuracy 仍只有 `0.882979`。也就是说，evidence-aware residual gate 能改变部分 seed 的 routing，但目前改变没有转化为主指标收益。
- **当前判断**：这条结构创新方向作为“论文可解释模块”是成立的，但作为当前主线优化失败；它证明“直接把 per-view classifier evidence 注入 gate”不够，需要更强约束或更干净的监督信号，否则容易把 gate 改动变成 noisy routing 而不是稳定 contribution。下一轮若继续结构创新，应优先做 `teacher/evidence supervised gate auxiliary loss` 或 `monotonic evidence-calibrated gate`，而不是继续增大 residual gate 容量。

---

## 2026-05-10：人类主线同步（DFR-25 ResNeXt minimal-variable）

> **本次动作**
> - 按人类要求，先完成 GitHub 同步，再把当前 operational mainline 改为 **DFR-25 ResNeXt 256x8 learned decision fusion**。
> - 本次只更新配置与协议，不启动训练、不启动 Optuna、不写入新的实验结果。
>
> **当前 DFR-25 锚点**
> - model / geometry：`backbone=resnext`, `fusion_type=decision`, `use_attention_pooling=false`, `image_size=256`, `num_slices_per_view=8`, `trim_edge_slices=2`。
> - runtime env：`ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER=1` + `ANKLE_DECISION_TRAIN_DOMINANT_GATE_DROPOUT_PROB=0.25`。
> - winning scalar：`seed=42`, `freeze_layers=3`, `lr=1e-4`, `weight_decay=2.5e-4`, `dropout=0.25`, `gradient_clip_norm=2.5`。
> - formal / proxy 真值来源：`configs/autoresearch_formal.yaml`, `configs/autoresearch_proxy.yaml`, `configs/autoresearch_formal_resnext_decision_256x8.yaml`, `configs/autoresearch_proxy_resnext_decision_256x8.yaml`。
> - 默认 Optuna 入口现在是 fixed anchor：`configs/optuna_main_search.yaml`, `configs/optuna_proxy_search.yaml`, `configs/optuna_main_search_resnext_decision_256x8.yaml`, `configs/optuna_proxy_search_resnext_decision_256x8.yaml` 的 `search_space` 均为空。
>
> **后续 ResNeXt 改进约束**
> - 后续任何 “ResNeXt 改进” 必须从 DFR-25 锚点 fork，只改一个明确声明的变量。
> - 默认不得同时改变 backbone family、geometry、fusion family、runtime repair 和训练标量；如果需要改搜索空间，必须说明它是哪一个变量轴，以及为什么直接服务于 DFR-25 的 decision-routing 机制。
> - 直接训练入口必须经 `scripts/run_train_with_config_env.py` 或等价方式注入 `runtime_env`，避免 YAML 表达的是 DFR-25、实际运行却漏掉 dominant-gate dropout。

## 2026-04-23：人类最新主线锁定（最高优先级，覆盖旧规则）

> **主方向硬约束**
> - 当前 autoresearch 的唯一主方向，改为：**实验 decision fusion，把 axial / coronal / sagittal 各个视角的信息还原成它们在 full-fusion 中该有的作用**。核心不是把权重机械拉平均，也不是只压制 axial，而是让该主导的视角在有真实证据时主导、该补充的视角在提供增量信息时被真正用上，并把真正的 **multi-view full-fusion learned accuracy** 做到明确高于 matched `equal-weight` control。
> - 在完成上面这个目标之前，**不允许切换到其他方向**。禁止把 backbone/geometry family 切换、feature fusion、无关 side campaign、泛化清理、与单视角依赖无直接关系的 calibration 小修小补，当成新的主线。
> - 本主线的完成标准不是单个 seed spike，也不是只提升 `val_auc` / `val_f1`。只有当 **同一 geometry / budget / seed protocol 下的 multi-view full-fusion learned 分支**，在主指标 `val_acc` 上明确高于 matched `equal-weight` control，才算完成，才允许切换方向。
> - 任何只证明“single-view expert 更强”“equal-weight 更强”“leave-one-view-out 结果更稳定”“某个视角继续错误独占但 accuracy 偶然更高”“AUC/F1 更好但 accuracy 没过 equal”的实验，都**不算**完成这个主方向。
> - 目标不是把 `equal-weight` 本身当成答案，也不是把三视角硬拉成平均分。`equal-weight` 只允许作为 matched control / 外部门槛；真正要找的是能解释或实现“各视角信息按其应有作用进入决策”的 **learned weighting**，或作为过渡研究工具的 **non-equal fixed weighting with clear interpretability**。
> - 如果某条 learned weighting 或 non-equal fixed weighting 首次在单个 seed 上翻过 `equal-weight`，后续优先做 matched confirmation、multi-seed validation 与权重 / 证据 telemetry，先确认它确实在让各视角信息回到应有作用，而不是一次高方差 spike 或一次表面去塌缩；在这个确认完成前，不要切去无关方向。
> - 每一轮实验完成后，**必须验证各视角权重比例**。默认要求是对本轮 best completed checkpoint / best trial 产出或补齐 `fusion_weight_analysis.json` 一类权重 telemetry，至少落盘并记录三视角 `mean fusion weight axial/coronal/sagittal` 与 `top-weight count/rate`；不能只报 `val_acc / val_auc / val_f1` 而不报权重。
> - 从提出设计思路开始，到实验结束的整条链路都必须写入实验记录。每条主线实验记录至少要明确写出三件事：`设计思路`（这次改动想修什么机制问题）、`预计改进效果`（预期哪一视角权重比例 / routing 行为会如何变化，以及为什么这有机会把 learned full-fusion `val_acc` 推过 matched `equal-weight`）、`实验实际结果`（实际验证到的权重比例 / top-weight 分布 / 关键指标 / keep-discard 结论）。如果结果与预期不符，必须明确写出来，不得省略。
>
> **inner agent 行动前自检要求**
> - 每一轮在决定改动前，必须至少做两轮自检：
>   1. 这项改动是否**直接**帮助 decision fusion 还原各视角信息在 full-fusion 中应有的作用，例如减弱错误的 axial 独占、增强 coronal/sagittal 在有证据样本上的有效进入、修复 evidence 与 routing 不一致、或让“该主导的视角主导、该补充的视角补充”？
>   2. 如果它成功，为什么它有机会把 **multi-view full-fusion learned `val_acc`** 推到 `equal-weight` 之上，而不是只把权重做得更平均、只改善单视角表现、训练稳定性、或局部 calibration？
>   3. 这项改动是在逼近一种可复用的 learned weighting 机制，还是在验证一种 **非 `equal-weight` 固定权重** 的可解释性？如果它只是无差别抹平视角差异，或说不清为什么这种权重形状能让各视角信息发挥应有作用，则该方向不合格。
> - 如果这两问里任意一问不能给出具体机制链路，则该改动视为**不合格方向**，本轮不得执行。
>
> **允许优先探索的修复类型**
> - 直接改变 evidence routing 的修复
> - 直接缓解弱视角 starvation 的修复
> - 直接修复 evidence-to-routing 对不齐、让视角信息回到应有作用的修复
> - 能让 stronger per-view experts 在 full-fusion 下真正形成超过 `equal-weight` 的净收益的对齐/训练修复
>
> **解释优先级**
> - 本节与 backlog 中任何旧表述冲突时，一律以本节为准。

---

## 2026-04-23：Decision-Fusion Repair（DFR-25 dominant-gate dropout，adaptive main-study，RTX A6000）

> **实验说明**
> - 本轮严格对齐当前唯一主线：不切 backbone / 几何 family，不切到 feature fusion，只在 `256x8 ResNeXt decision + L3-no-mixer` 上做一个**直接作用于 routing** 的修复。
> - 唯一离散改动是 commit `bf4256e` 在 `src/model.py` 新增 `ANKLE_DECISION_TRAIN_DOMINANT_GATE_DROPOUT_PROB`：仅在训练态、仅对当前样本的**最高 gate confidence** 做随机压平，迫使 full-fusion loss 穿过次强视角；`view_logits` 本身不被破坏，推理语义仍保持 learned late fusion。
> - 这条线与 `DFR-03` 的关键区别是：`DFR-03` 通过输入 view dropout / axial blur 同时扰动 classifier 与 gate；本轮只扰动 gate，因此更直接服务于“减弱 axial dominance、让弱视角拿到真实融合梯度”，同时避免再次伤害 per-view expert。
> - 首次 fresh main-study 尝试使用预留 search-config copy [autoresearch_logs/generated_search_configs/optuna_main_search_iter_0001_20260423_021418.yaml](/dataset/HH/ankle-ct/autoresearch_logs/generated_search_configs/optuna_main_search_iter_0001_20260423_021418.yaml) 与 study_root [runs/optuna_main_autoloop/iter_0001_20260423_021418](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0001_20260423_021418)，但 commit `1eed9e9` 的首版实现因为训练态 `scatter_` 原地写入破坏 autograd，在 `epoch 1` 全量 crash；这被判定为 **code bug**，不记作模型结论。随后在 commit `bf4256e` 去掉原地写入，并按 fresh-run 规则换到新的 search-config copy [autoresearch_logs/generated_search_configs/optuna_main_search_iter_0001_20260423_021418_retry1.yaml](/dataset/HH/ankle-ct/autoresearch_logs/generated_search_configs/optuna_main_search_iter_0001_20260423_021418_retry1.yaml) 与新的 study_root [runs/optuna_main_autoloop/iter_0001_20260423_021418_retry1](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0001_20260423_021418_retry1) 重跑。

- [x] **DFR-25-RESNEXT-DECISION-256X8-DOMINANT-GATE-DROPOUT-MAIN-S42**：fresh adaptive `main-study` [runs/optuna_main_autoloop/iter_0001_20260423_021418_retry1](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0001_20260423_021418_retry1) 的 best completed trial（trial `1`，commit `bf4256e`，runtime env `ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER=1` + `ANKLE_DECISION_TRAIN_DOMINANT_GATE_DROPOUT_PROB=0.25`）→ `val_acc=0.9361702127659575`, `val_auc=0.9786363636363636`, `val_f1=0.9333333333333333`, `peak_vram≈2.15 GiB`, `total_seconds≈1071.0` → **keep**（相较 matched `equal-weight` control `seed=42` `0.9255319148936170 / 0.9486363636363637 / 0.9176470588235294`，主指标 `val_acc` 提升 `0.0106382978723405`；相较当前 `L3-no-mixer` seed42 anchor `0.9255319148936170 / 0.9750000000000000 / 0.9213483146067416`，`val_acc / val_auc / val_f1` 分别提升 `0.0106382978723405 / 0.0036363636363636 / 0.0119850187265917`。这也是当前 `256x8 decision-only` 主线下，首次有**真正的 multi-view learned full-fusion** 在同 geometry / budget / seed protocol 上明确超过 matched `equal-weight` control 的单轮正结果。）
- **study 内部分布**：requested `4` 个 valid trial 中，只有 trial `1` 站上 `0.9362 / 0.9786 / 0.9333`；其余 3 个 valid trial (`0/2/3`) 全都停在 `0.9042553191489362 val_acc` 档，说明这条新 repair 已经给出强正信号，但**当前仍对超参敏感**。monitor 也给出一致方向：`gradient_clip_norm=2.5` 与更低的 `weight_decay=2.5e-4` 更有利。
- **tail-fill 说明**：wave-aligned 额外补齐的 trial `4/5` 在 requested `4` 个 valid trial 已经全部完成后仍明显落后当前 best；为按“本 session 只做一轮后退出”收口，本轮手动结束了这两个 tail-fill trial。它们在 ledger 里显示为 `crash`，但这不是新的模型故障，不影响本轮 best completed trial 的 keep / discard 判断。
- **当前判断**：`DFR-25` 是一条**机制上合格、结果上偏正面**的新 repair。它直接削弱 dominant view lock-in、避免像 `DFR-03` 那样同时伤害 classifier，并且在 seed42 上把 learned full-fusion `val_acc` 真正推到了 matched `equal-weight` 之上；但按 2026-04-23 的主线完成标准，这仍只是**单 seed keep**，还不能宣告主线完成。
- **推荐动作**：下一步优先做 **matched alternate-seed validation**，按 `seed=123 -> seed=456` 的顺序复验同一 runtime env 与同一 winning scalar（`lr=1e-4`, `weight_decay=2.5e-4`, `dropout=0.25`, `gradient_clip_norm=2.5`）。只有当这条 dominant-gate-dropout repair 在 `seed=123/456` 上也能稳定让 learned full-fusion `val_acc` 超过 matched `equal-weight`，当前唯一主方向才算真正完成。若 alternate seed 仍有正信号，再补一轮 FWR-style telemetry，确认收益来自实际的 routing 改善，而不是偶然 calibration spike。

---

## 2026-04-23：Decision-Fusion Repair Follow-up（DFR-25 dominant-gate dropout，seed123 exact confirmation，adaptive main-study，RTX A6000）

> **实验说明**
> - 本轮不引入任何新结构或新超参，只做 `DFR-25` 的 **matched alternate-seed validation**，目的是回答这条 routing-only repair 是否能从 `seed=42` 迁移到新的随机种子，而不是再开一轮会混入标量漂移的搜索。
> - 使用 fresh single-trial search-config copy [autoresearch_logs/generated_search_configs/optuna_main_search_iter_0002_20260423_031222.yaml](/dataset/HH/ankle-ct/autoresearch_logs/generated_search_configs/optuna_main_search_iter_0002_20260423_031222.yaml) 与 study_root [runs/optuna_main_autoloop/iter_0002_20260423_031222](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0002_20260423_031222)，通过 `fixed_overrides` 固定 `seed=123` 与 seed42 winner scalar（`lr=1e-4`, `weight_decay=2.5e-4`, `dropout=0.25`, `gradient_clip_norm=2.5`），runtime env 保持 `ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER=1` + `ANKLE_DECISION_TRAIN_DOMINANT_GATE_DROPOUT_PROB=0.25`。
> - 这轮的唯一研究问题是：dominant-gate dropout 如果真的在训练期削弱了 axial lock-in、让弱视角拿到有效融合梯度，那么在不改标量的前提下换到 `seed=123` 时，learned full-fusion `val_acc` 应继续高于 matched `equal-weight`，而不是只留下 `seed=42` 的一次高方差 spike。

- [x] **DFR-25-RESNEXT-DECISION-256X8-DOMINANT-GATE-DROPOUT-MAIN-S123**：fresh adaptive `main-study` [runs/optuna_main_autoloop/iter_0002_20260423_031222](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0002_20260423_031222) 的 only completed trial（commit `8085b55`，fixed exact-confirmation config，runtime env `ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER=1` + `ANKLE_DECISION_TRAIN_DOMINANT_GATE_DROPOUT_PROB=0.25`）→ `val_acc=0.9361702127659575`, `val_auc=0.9554545454545454`, `val_f1=0.9318181818181818`, `peak_vram≈2.15 GiB`, `total_seconds≈747.6` → **keep**（相较 matched `equal-weight` control `seed=123` `0.9255319148936170 / 0.9718181818181818 / 0.9135802469135802`，主指标 `val_acc` 再次提升 `0.0106382978723405`；相较当前 `L3-no-mixer` seed123 anchor `0.8936170212765957 / 0.9395454545454545 / 0.8750000000000000`，`val_acc / val_auc / val_f1` 分别提升 `0.0425531914893618 / 0.0159090909090909 / 0.0568181818181818`。虽然 `val_auc` 低于 matched `equal-weight`，但按当前主指标规则，这仍是一次明确的 learned full-fusion 胜出。）
- **当前判断**：`DFR-25` 已从“单 seed 正结果”升级为 **`seed=42 + seed=123` 两个 matched keep**。更关键的是，这两颗 seed 的 learned full-fusion `val_acc` 都落在同一 `0.9361702127659575` 档位，并且都稳定高于各自 matched `equal-weight` control `0.9255319148936170`；这让 dominant-gate dropout 更像一个真实的 routing repair，而不是单点 calibration 偶然值。
- **推荐动作**：下一步只剩 **`seed=456` exact confirmation**。继续保持同一 runtime env 与同一 winning scalar，不要再引入新的 repair 或新的搜索空间；若 `seed=456` 也能在 `val_acc` 上高于 matched `equal-weight`，再补一轮 FWR-style telemetry，确认收益确实来自 routing 改善而不是 seed-specific ranking 偶然性。

---

## 2026-04-23：Decision-Fusion Repair Follow-up（DFR-25 dominant-gate dropout，seed456 exact confirmation，adaptive main-study，RTX A6000）

> **实验说明**
> - 本轮继续严格保持同一主线与同一 repair，不引入任何新结构、新搜索轴或新 runtime path；唯一目的，是把 `DFR-25 dominant-gate dropout` 的 exact confirmation 从 `seed=42/123` 补到最后一颗 matched seed。
> - 使用 fresh single-trial search-config copy [autoresearch_logs/generated_search_configs/optuna_main_search_iter_0003_20260423_033048.yaml](/dataset/HH/ankle-ct/autoresearch_logs/generated_search_configs/optuna_main_search_iter_0003_20260423_033048.yaml) 与 study_root [runs/optuna_main_autoloop/iter_0003_20260423_033048](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0003_20260423_033048)，通过 `fixed_overrides` 固定 `seed=456` 与 seed42 winner scalar（`lr=1e-4`, `weight_decay=2.5e-4`, `dropout=0.25`, `gradient_clip_norm=2.5`），runtime env 继续保持 `ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER=1` + `ANKLE_DECISION_TRAIN_DOMINANT_GATE_DROPOUT_PROB=0.25`。
> - 本轮唯一研究问题是：如果 dominant-gate dropout 确实通过训练期压平 dominant confidence、让弱视角拿到有效融合梯度，那么在第三颗 matched seed 上它也应继续把 learned full-fusion `val_acc` 推到 matched `equal-weight` 之上，而不是只在前两颗 seed 上成立。

- [x] **DFR-25-RESNEXT-DECISION-256X8-DOMINANT-GATE-DROPOUT-MAIN-S456**：fresh adaptive `main-study` [runs/optuna_main_autoloop/iter_0003_20260423_033048](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0003_20260423_033048) 的 only completed trial（commit `3fb1e11`，fixed exact-confirmation config，runtime env `ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER=1` + `ANKLE_DECISION_TRAIN_DOMINANT_GATE_DROPOUT_PROB=0.25`）→ `val_acc=0.9468085106382979`, `val_auc=0.9690909090909091`, `val_f1=0.9425287356321839`, `peak_vram≈2.14 GiB`, `total_seconds≈996.0` → **keep**（相较 matched `equal-weight` control `seed=456` `0.9148936170212766 / 0.9613636363636364 / 0.9024390243902439`，`val_acc / val_auc / val_f1` 分别提升 `0.0319148936170213 / 0.0077272727272727 / 0.0400897112419400`；相较当前 `L3-no-mixer` seed456 anchor `0.9255319148936170 / 0.9713636363636363 / 0.9176470588235294`，`val_acc / val_auc / val_f1` 分别提升 `0.0212765957446809 / -0.0022727272727272 / 0.0248816768086545`。）
- **当前判断**：`DFR-25` 现在已经完成 **`seed=42 + seed=123 + seed=456` 的 `3/3` matched keep**。其 3-seed mean 达到 `val_acc=0.9397163120567376`, `val_auc=0.9677272727272728`, `val_f1=0.9358934169278997`，明确高于 matched `equal-weight` control 的 `0.9219858156028368 / 0.9606060606060606 / 0.9112221100424511`；同时 `val_acc` population std 为 `0.0050149417105429`，没有出现“靠更高方差换来单点翻盘”的坏迹象。按 2026-04-23 的主线完成标准，这已经把 dominant-gate dropout 从“promising repair”推进成了**完成 matched multiseed confirmation 的 learned full-fusion repair**。
- **推荐动作**：下一步不该再切新 repair，而应优先补 **FWR-style telemetry**。重点是对 `DFR-25` 的 retained checkpoints 做 fusion-weight / leave-one-view-out / perturbation 级分析，确认这次 `3/3` 胜出确实来自弱视角 contribution 提升与 routing 迁移，而不是仅靠 seed-specific ranking 偶然性或 calibration 偏置。

## 2026-04-23：Decision-Fusion Repair Follow-up（DFR-25 multiseed routing telemetry，existing main-study checkpoints，RTX A6000）

> **实验说明**
> - 本轮不新开 study、不重训；唯一离散改动是在 commit `4eddcd3` 新增 `scripts/analyze_multiseed_decision_routing.py`，把已有 `DFR-25` retained checkpoints `seed=42/123/456` 的三类后验分析统一串起来：`fusion_weight_analysis.json`、`view_ablation_summary.json` 与 `perturbation_summary_axial_blur17_sigma4.json`。
> - 分析对象分别是 [runs/optuna_main_autoloop/iter_0001_20260423_021418_retry1/trials/trial_0001/run](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0001_20260423_021418_retry1/trials/trial_0001/run)、[runs/optuna_main_autoloop/iter_0002_20260423_031222/trials/trial_0000/run](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0002_20260423_031222/trials/trial_0000/run)、[runs/optuna_main_autoloop/iter_0003_20260423_033048/trials/trial_0000/run](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0003_20260423_033048/trials/trial_0000/run)；aggregate 报告写入 [autoresearch_logs/dfr25_multiseed_routing_iter_0004_20260423_035328.json](/dataset/HH/ankle-ct/autoresearch_logs/dfr25_multiseed_routing_iter_0004_20260423_035328.json)。
> - 本轮唯一研究问题是：`DFR-25` 的 `3/3` matched keep，究竟是不是因为它真的减弱了 axial dominance、提升了 coronal / sagittal 的有效贡献；还是说它虽然在 accuracy 上赢了 `equal-weight`，但 routing 仍然大体停留在单视角主导。

- [x] **DFR-25-RESNEXT-DECISION-256X8-DOMINANT-GATE-DROPOUT-MULTISEED-ROUTING-TELEMETRY**：commit `4eddcd3` 用新增 wrapper 脚本分析既有 `DFR-25` retained checkpoints（训练指标仍是 matched 3-seed mean `val_acc=0.9397163120567376`, `val_auc=0.9677272727272728`, `val_f1=0.9358934169278997`, `peak_vram≈2.15 GiB`）并产出 multiseed aggregate report → **discard**（本轮没有产生新的更优 checkpoint；它回答的是机制问题而不是主指标提升，因此只能记为 analysis-only discard。）
- **fusion-weight 结论**：相较 `FWR-02` 在旧 `L3-no-mixer` seed42 上看到的 `mean axial weight=0.997426`、`top-weight axial=94/94`，`DFR-25` 的 3-seed mean 已明显去塌缩到 `mean weight axial/coronal/sagittal = 0.7613 / 0.0869 / 0.1517`，`non-axial top-weight rate=0.1596`。但这种去塌缩并不均匀：真正的迁移几乎都来自 `seed=123`，其 `top-weight rate axial/sagittal = 0.5213 / 0.4787`；而 `seed=42` 与 `seed=456` 仍是 `axial top-weight 94/94`。
- **view-control 结论**：`DFR-25` 的 3-seed mean full fusion 已经比各自 **best single-view** 高 `+0.02837 val_acc / +0.00182 val_auc / +0.02552 val_f1`；其中 `seed=123` 对 best single-view 的优势最大（`+0.07447 val_acc`），`seed=456` 也有小幅净收益（`+0.01064 val_acc`），只有 `seed=42` 仍与 `single_view_axial` 完全同分。与此同时，`leave_out_coronal` 与 `leave_out_sagittal` 的 mean accuracy penalty 只有 `0.00355 / 0.02482`，并且仍存在 exact tie，这说明弱视角贡献已经出现，但还没有稳定到“移除就必掉分”。
- **perturbation 结论**：在与 `FWR-03` 对齐的 axial blur（`kernel=17`, `sigma=4.0`）下，`DFR-25` 的 3-seed mean `val_acc / val_auc / val_f1` 分别回落 `0.28369 / 0.16667 / 0.44431`；同时 axial mean weight 会下降 `0.09828`，但 `top-weight switch rate` 仍只有 `0.07092`，`migrated_away_rate` 只有 `0.05319`。更关键的是，这些权重迁移 **全部来自 `seed=123`，且迁移目的地全部是 sagittal**；`seed=42/456` 即使在 axial 证据被明显打坏后，top-weight 仍保持 `94/94` axial，不会迁向 coronal / sagittal。
- **当前判断**：这轮 telemetry 给出的结论是 **mixed-positive**。`DFR-25` 的 accuracy keep 不是纯 calibration 偶然值，因为它已经在 3-seed mean 上明显高于 matched `equal-weight`，并且 `seed=123` 确实表现出真实的多视角净收益与 axial→sagittal routing 迁移；但它也还没有彻底摆脱 single-view dependence，`seed=42/456` 仍保留明显的 axial lock-in。也就是说，`DFR-25` 已经完成了“learned full-fusion > matched equal-weight”的主指标目标，但还没有完成“弱视角贡献在机制上稳定且不可替代”的更强版本。
- **推荐动作**：下一步仍应留在 `DFR-25` 家族内，不切 backbone、不切到 feature fusion，也不要回到无关 cleanup。最直接的 follow-up 是一个 **training-only 的弱视角保活修复**，例如只在训练态对 gate 增加最小 non-dominant mass / entropy floor 一类约束，专门针对 telemetry 暴露出来的 `seed=42/456 residual axial lock-in`，目标是把 `seed123` 上已经出现的 routing migration 扩展到其余种子，而不是改写 inference 语义。

---

## 2026-04-23：Decision-Fusion Repair Follow-up（DFR-26 train-time non-dominant weight floor，adaptive main-study，RTX A6000）

> **实验说明**
> - 本轮严格保持当前唯一主线与 `256x8 ResNeXt decision + L3-no-mixer + DFR-25` 语义不变，不切 backbone / geometry，不切到 feature fusion；唯一离散改动是在 commit `9ca2de1` 的 [src/model.py](/dataset/HH/ankle-ct/src/model.py) 新增 `ANKLE_DECISION_TRAIN_NONDOMINANT_WEIGHT_FLOOR`。
> - 这个修复只在**训练态**、只在 gate softmax / teacher blend 之后生效：对每个样本保留一小段非 dominant 视角的最小融合质量，让 weak views 即使在 `axial` gate 已经塌缩时也持续收到 fused loss；推理期不保留该 floor，因此 inference 仍是原始 learned late fusion，而不是退回 `equal-weight` 或引入新的 inference-time floor。
> - fresh adaptive `main-study` 使用 search-config copy [autoresearch_logs/generated_search_configs/optuna_main_search_iter_0005_20260423_040441.yaml](/dataset/HH/ankle-ct/autoresearch_logs/generated_search_configs/optuna_main_search_iter_0005_20260423_040441.yaml) 与 study_root [runs/optuna_main_autoloop/iter_0005_20260423_040441](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0005_20260423_040441)，runtime env 为 `ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER=1` + `ANKLE_DECISION_TRAIN_DOMINANT_GATE_DROPOUT_PROB=0.25` + `ANKLE_DECISION_TRAIN_NONDOMINANT_WEIGHT_FLOOR=0.05`。
> - 本轮唯一研究问题是：如果 telemetry 暴露出来的 `seed=42/456 residual axial lock-in` 主要来自 weak-view starvation，那么在不改 inference 语义的前提下，训练期给 non-dominant views 保留固定最小 mass，是否能在 matched seed42 上继续让 learned full-fusion 站住并优于 matched `equal-weight`，而不是只把 gate 朝平均分配拉平。

- [x] **DFR-26-RESNEXT-DECISION-256X8-NONDOMINANT-WEIGHT-FLOOR-MAIN-S42**：fresh adaptive `main-study` [runs/optuna_main_autoloop/iter_0005_20260423_040441](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0005_20260423_040441) 的 best completed trial（trial `1`，commit `9ca2de1`，runtime env `ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER=1` + `ANKLE_DECISION_TRAIN_DOMINANT_GATE_DROPOUT_PROB=0.25` + `ANKLE_DECISION_TRAIN_NONDOMINANT_WEIGHT_FLOOR=0.05`）→ `val_acc=0.9361702127659575`, `val_auc=0.9800000000000000`, `val_f1=0.9302325581395349`, `peak_vram≈2.15 GiB`, `total_seconds≈1081.8` → **keep**（相较 matched `equal-weight` control `seed=42` `0.9255319148936170 / 0.9486363636363637 / 0.9176470588235294`，`val_acc / val_auc / val_f1` 分别提升 `0.0106382978723405 / 0.0313636363636363 / 0.0125854993160055`；相较 `DFR-25` seed42 winner `0.9361702127659575 / 0.9786363636363636 / 0.9333333333333333`，`val_acc` 持平、`val_auc` 再升 `0.0013636363636363`、`val_f1` 回落 `0.0031007751937985`。按当前选择规则，主指标 tie 时以更高 `val_auc` 胜出，因此这是一条**窄幅但有效的 seed42 keep**。）
- **study 内部分布**：requested `4` 个 valid trial 最终补齐成 `6` 个 valid trial；其中 trial `1` 与 tail-fill trial `5` 都站上 `0.9361702127659575 val_acc`，但 trial `1` 以更高 `val_auc=0.9800` 胜出。更关键的是，当前 best trial 的 scalar **没有偏离 `DFR-25` 的 seed42 winner**，仍是 `lr=1e-4`, `weight_decay=2.5e-4`, `dropout=0.25`, `gradient_clip_norm=2.5`；其余 trial 分别落在 `0.9149` 与 `0.9255` 档，说明 training-only weak-view floor 没有抬高 seed42 ceiling，但在既有 winning corner 上提供了小幅更好的 ranking quality。
- **当前判断**：`DFR-26` 是一条**机制上合格、结果上小幅正向**的 follow-up repair。它没有把 `seed=42` learned full-fusion `val_acc` 进一步推高到超过 `DFR-25`，因此还不能宣称已经修复了 telemetry 暴露出来的 residual axial lock-in；但它在不改 inference 语义、也不牺牲 matched `equal-weight` 胜出的前提下，把 `seed=42` 的 `val_auc` 做到了当前该主线上新的局部最优，因此按 ledger 规则应保留。
- **推荐动作**：下一步不要再回到 broad search，也不要切去别的方向。最直接的后续是做 **`seed=123` exact confirmation**：保持同一 runtime env 与同一 winning scalar（`lr=1e-4`, `weight_decay=2.5e-4`, `dropout=0.25`, `gradient_clip_norm=2.5`），只把 `seed` 固定到 `123`，确认这条 training-only non-dominant floor 是真实的 routing repair，还是只在 `seed=42` 上给出 tie-break 级好处。若 `seed=123` 继续 keep，再补 `seed=456`；若 multiseed 站住，再回到 telemetry 验证它是否真的把 `seed=42/456` 的 routing migration 拉起来。

---

## 2026-04-23：Decision-Fusion Repair Follow-up（DFR-26 train-time non-dominant weight floor，seed123 exact confirmation，adaptive main-study，RTX A6000）

> **实验说明**
> - 本轮不引入任何新结构或新超参，只做 `DFR-26` 的 **matched alternate-seed validation**，目标是判断这个 training-only weak-view survival repair 能否从 `seed=42` 迁移到新的随机种子，而不是只留下一个 tie-break 级的 seed42 局部最优。
> - 使用 fresh single-trial search-config copy [autoresearch_logs/generated_search_configs/optuna_main_search_iter_0006_20260423_045515.yaml](/dataset/HH/ankle-ct/autoresearch_logs/generated_search_configs/optuna_main_search_iter_0006_20260423_045515.yaml) 与 study_root [runs/optuna_main_autoloop/iter_0006_20260423_045515](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0006_20260423_045515)，通过 `fixed_overrides` 固定 `seed=123` 与 `DFR-26` seed42 winner scalar（`lr=1e-4`, `weight_decay=2.5e-4`, `dropout=0.25`, `gradient_clip_norm=2.5`），runtime env 保持 `ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER=1` + `ANKLE_DECISION_TRAIN_DOMINANT_GATE_DROPOUT_PROB=0.25` + `ANKLE_DECISION_TRAIN_NONDOMINANT_WEIGHT_FLOOR=0.05`。
> - 本轮唯一研究问题是：如果固定 non-dominant floor 真能在不改 inference 语义的前提下修复 `seed=42/456 residual axial lock-in`，那么它在 `seed=123` 上至少应继续守住 matched `equal-weight`，而不是把已经存在的有益 routing migration 再次压平。

- [x] **DFR-26-RESNEXT-DECISION-256X8-NONDOMINANT-WEIGHT-FLOOR-MAIN-S123**：fresh adaptive `main-study` [runs/optuna_main_autoloop/iter_0006_20260423_045515](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0006_20260423_045515) 的 only completed trial（trial `0`，commit `4c8328b`，fixed exact-confirmation config，runtime env `ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER=1` + `ANKLE_DECISION_TRAIN_DOMINANT_GATE_DROPOUT_PROB=0.25` + `ANKLE_DECISION_TRAIN_NONDOMINANT_WEIGHT_FLOOR=0.05`）→ `val_acc=0.8404255319148937`, `val_auc=0.9190909090909091`, `val_f1=0.8275862068965517`, `peak_vram≈2.15 GiB`, `total_seconds≈910.0` → **discard**（相较 matched `equal-weight` control `seed=123` `0.9255319148936170 / 0.9718181818181818 / 0.9135802469135802`，`val_acc / val_auc / val_f1` 分别回落 `0.0851063829787233 / 0.0527272727272727 / 0.0859940400170285`；相较 `DFR-25` seed123 confirm keep `0.9361702127659575 / 0.9554545454545454 / 0.9318181818181818`，`val_acc / val_auc / val_f1` 分别回落 `0.0957446808510638 / 0.0363636363636363 / 0.1042319749216301`；也低于 `L3-no-mixer` seed123 anchor `0.8936170212765957 / 0.9395454545454545 / 0.8750000000000000`。）
- **当前判断**：`DFR-26` 的固定 `0.05` non-dominant floor **没有迁移到 `seed=123`**。这说明“无条件给弱视角保底质量”并不是一个稳健的 multiseed routing repair；更可能的情况是，它在 `seed=42` 上能轻微修正 lock-in，但在 `seed=123` 这种已经出现真实 axial→sagittal migration 的情形下，反而会把有益的 learned routing 压平，直接伤害 full-fusion accuracy。
- **推荐动作**：不要继续补 **同一个固定 floor 配方** 的 `seed=456` exact confirmation。更直接的 follow-up 是把这条弱视角保活修复改成 **dominance-triggered / conditional** 版本，例如只在 top gate 过度集中、或 gate entropy 低于阈值时才注入额外 non-dominant mass；这样才有机会专门救 `seed=42/456` 的塌缩样本，而不破坏 `seed=123` 上已经存在的有益 routing migration。

---

## 2026-04-23：Decision-Fusion Repair Follow-up（DFR-27 conditional axial-dominance floor，adaptive main-study，RTX A6000）

> **实验说明**
> - 本轮严格保持当前唯一主线与 `256x8 ResNeXt decision + L3-no-mixer + DFR-25` 语义不变，不切 backbone / geometry，不切到 feature fusion；唯一离散改动是在 commit `1176eb2` 的 [src/model.py](/dataset/HH/ankle-ct/src/model.py) 新增 `ANKLE_DECISION_TRAIN_AXIAL_DOMINANCE_THRESHOLD`。
> - 这个修复不再像 `DFR-26` 那样对所有样本无条件注入 `non-dominant floor`，而是只在**训练态**、只在 blended gate 中 `axial` 仍是 dominant、且其权重至少达到 `0.8` 时，才对已有的 `ANKLE_DECISION_TRAIN_NONDOMINANT_WEIGHT_FLOOR=0.05` 触发弱视角保活；推理期仍保持原始 learned late fusion。
> - fresh adaptive `main-study` 使用 search-config copy [autoresearch_logs/generated_search_configs/optuna_main_search_iter_0007_20260423_051629.yaml](/dataset/HH/ankle-ct/autoresearch_logs/generated_search_configs/optuna_main_search_iter_0007_20260423_051629.yaml) 与 study_root [runs/optuna_main_autoloop/iter_0007_20260423_051629](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0007_20260423_051629)，runtime env 为 `ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER=1` + `ANKLE_DECISION_TRAIN_DOMINANT_GATE_DROPOUT_PROB=0.25` + `ANKLE_DECISION_TRAIN_NONDOMINANT_WEIGHT_FLOOR=0.05` + `ANKLE_DECISION_TRAIN_AXIAL_DOMINANCE_THRESHOLD=0.8`。
> - 本轮唯一研究问题是：如果 `DFR-26` 的失败主要来自“对所有样本一刀切地压平有益 routing migration”，那么把 weak-view floor 改成只作用于 **high-confidence axial lock-in** 的条件式版本，是否能在 `seed=42` 上保留 `DFR-26` 的局部收益，同时不再把 learned branch 推离已验证的多视角净收益。

- [x] **DFR-27-RESNEXT-DECISION-256X8-CONDITIONAL-AXIAL-DOMINANCE-FLOOR-MAIN-S42**：fresh adaptive `main-study` [runs/optuna_main_autoloop/iter_0007_20260423_051629](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0007_20260423_051629) 的 best completed trial（trial `3`，commit `1176eb2`，runtime env `ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER=1` + `ANKLE_DECISION_TRAIN_DOMINANT_GATE_DROPOUT_PROB=0.25` + `ANKLE_DECISION_TRAIN_NONDOMINANT_WEIGHT_FLOOR=0.05` + `ANKLE_DECISION_TRAIN_AXIAL_DOMINANCE_THRESHOLD=0.8`）→ `val_acc=0.9361702127659575`, `val_auc=0.9659090909090909`, `val_f1=0.9318181818181818`, `peak_vram≈2.15 GiB`, `total_seconds≈896.4` → **discard**（相较 matched `equal-weight` control `seed=42` `0.9255319148936170 / 0.9486363636363637 / 0.9176470588235294`，仍有 `+0.0106382978723405 / +0.0172727272727272 / +0.0141711229946524` 的正差，但相较 retained `DFR-26` seed42 keep `0.9361702127659575 / 0.9800000000000000 / 0.9302325581395349`，主指标只打平、`val_auc` 回落 `0.0140909090909091`；相较 `DFR-25` seed42 winner `0.9361702127659575 / 0.9786363636363636 / 0.9333333333333333`，同样只打平 `val_acc` 且 `val_auc / val_f1` 都更差，因此按当前 tie-break 规则不能保留。）
- **study 内部分布**：6 个 completed trial 形成 `0.9043 / 0.9255 / 0.9149 / 0.9362 / 0.9255 / 0.9043` 的离散分布；只有 trial `3`（`dropout=0.2`, `gradient_clip_norm=2.0`, `lr=1e-4`, `weight_decay=5e-4`）站上 `0.9362`。更关键的是，`DFR-25/26` 已知的 seed42 winning corner（`dropout=0.25`, `gradient_clip_norm=2.5`, `lr=1e-4`, `weight_decay=2.5e-4`）在本轮对应的 trial `1` 只得到 `0.9255319148936170 val_acc`，说明这条硬阈值 repair 没能保住原先已知的最好角点。
- **当前判断**：`DFR-27` 的 hard conditional floor 在机制上是合格的，因为它确实只对 `axial` 高集中塌缩样本介入，而不是再次全局压平 routing；但结果上它仍然**没有超过当前 retained keep**。它既没把 `seed=42` ceiling 从 `0.9362` 再往上推，也没保住 `DFR-26` 的 `0.9800 val_auc`，更像是一个“避免了 `DFR-26 seed123` 那种明显错误方向、但触发太硬太稀疏”的中间版本。
- **推荐动作**：不要把这条 **hard threshold** 版 conditional floor 升格为新基线，也不要直接补 multiseed validation。若继续沿这条 family 前进，更合理的下一步仍应保持 training-only routing repair，但把触发从二值阈值改成 **continuous concentration-scaled rescue**（例如随 axial dominance excess 连续增减注入 mass，或用 entropy 低点平滑控制），先在 `seed=42` 回答“能否同时保住 `0.9362 val_acc` 与 `DFR-26` 的 AUC 收益”，再决定是否值得回到 `seed=123` exact confirmation。

---

## 2026-04-23：Decision-Fusion Repair Follow-up（DFR-28 continuous axial-dominance-scaled rescue，adaptive main-study，RTX A6000）

> **实验说明**
> - 本轮继续严格保持当前唯一主线与 `256x8 ResNeXt decision + L3-no-mixer + DFR-25` 语义不变，不切 backbone / geometry，不切到 feature fusion；唯一离散改动是在 commit `5f75da6` 的 [src/model.py](/dataset/HH/ankle-ct/src/model.py) 把 `ANKLE_DECISION_TRAIN_AXIAL_DOMINANCE_THRESHOLD` 从 `DFR-27` 的**二值触发**改成**连续缩放触发**。
> - 具体来说，训练态 `non-dominant floor` 不再是“axial weight >= 0.8 就全量注入 floor”，而是只对 `axial` dominant 样本按 `axial dominance excess` 线性缩放 rescue 量；dominance 越强，注入的 weak-view mass 越接近 `ANKLE_DECISION_TRAIN_NONDOMINANT_WEIGHT_FLOOR=0.05`，而刚越过阈值的样本只受到轻量干预。推理期仍保持原始 learned late fusion。
> - fresh adaptive `main-study` 使用 search-config copy [autoresearch_logs/generated_search_configs/optuna_main_search_iter_0008_20260423_060711.yaml](/dataset/HH/ankle-ct/autoresearch_logs/generated_search_configs/optuna_main_search_iter_0008_20260423_060711.yaml) 与 study_root [runs/optuna_main_autoloop/iter_0008_20260423_060711](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0008_20260423_060711)，runtime env 为 `ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER=1` + `ANKLE_DECISION_TRAIN_DOMINANT_GATE_DROPOUT_PROB=0.25` + `ANKLE_DECISION_TRAIN_NONDOMINANT_WEIGHT_FLOOR=0.05` + `ANKLE_DECISION_TRAIN_AXIAL_DOMINANCE_THRESHOLD=0.8`。
> - 本轮唯一研究问题是：如果 `DFR-27` 的问题主要在于触发太硬、把“轻度 axial dominance”与“完全塌缩”混成同一种干预，那么把 rescue 量改成随 dominance excess 连续变化，是否能在 `seed=42` 上同时保住 `0.9362` 档 `val_acc` 与 `DFR-26` 的 AUC 收益。

- [x] **DFR-28-RESNEXT-DECISION-256X8-CONTINUOUS-AXIAL-DOMINANCE-SCALED-RESCUE-MAIN-S42**：fresh adaptive `main-study` [runs/optuna_main_autoloop/iter_0008_20260423_060711](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0008_20260423_060711) 的 best completed trial（trial `1`，commit `5f75da6`，runtime env `ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER=1` + `ANKLE_DECISION_TRAIN_DOMINANT_GATE_DROPOUT_PROB=0.25` + `ANKLE_DECISION_TRAIN_NONDOMINANT_WEIGHT_FLOOR=0.05` + `ANKLE_DECISION_TRAIN_AXIAL_DOMINANCE_THRESHOLD=0.8`）→ `val_acc=0.9361702127659575`, `val_auc=0.9772727272727272`, `val_f1=0.9333333333333333`, `peak_vram≈2.15 GiB`, `total_seconds≈1343.4` → **discard**（相较 matched `equal-weight` control `seed=42` `0.9255319148936170 / 0.9486363636363637 / 0.9176470588235294`，仍有 `+0.0106382978723405 / +0.0286363636363635 / +0.0156862745098039` 的正差；但相较 retained `DFR-26` seed42 keep `0.9361702127659575 / 0.9800000000000000 / 0.9302325581395349`，主指标只打平、`val_auc` 回落 `0.0027272727272728`；相较 `DFR-25` seed42 winner `0.9361702127659575 / 0.9786363636363636 / 0.9333333333333333`，同样只打平 `val_acc / val_f1` 且 `val_auc` 回落 `0.0013636363636364`，因此按当前 tie-break 规则不能保留。）
- **study 内部分布**：requested `4` 个 valid trial 最终补齐成 `6` 个 valid trial；`val_acc` 分布为 `0.9149 / 0.9255 / 0.9362` 三档。best completed trial `1` 恰好回到 `DFR-25/26` 已知的 winning corner（`lr=1e-4`, `weight_decay=2.5e-4`, `dropout=0.25`, `gradient_clip_norm=2.5`），说明 continuous rescue 至少**没有像 `DFR-27` 那样把旧最好角点直接打坏**；但它也没有把这个角点的 ranking quality 再往上抬。AUC-best trial `5` 虽然达到 `val_auc=0.9831818181818183`，但 `val_acc` 只有 `0.9255319148936170`，仍低于当前 retained keep。
- **当前判断**：`DFR-28` 在机制上仍然是合格方向，因为它确实只对 residual axial lock-in 样本按集中程度平滑介入，而没有回退到 blanket floor；结果上它比 `DFR-27` 更健康，至少保住了 `DFR-25/26` 的 winning scalar corner，但**仍然没有超过当前 retained keep**。这说明“触发平滑化”本身还不够，当前 `0.05` rescue cap 很可能仍然偏重，或者 dominance-based signal 本身还不够精确。
- **推荐动作**：继续留在 `DFR-25` 家族内，不切 backbone、不切到 feature fusion，也**不要**因为这轮只是 tie-on-accuracy 就直接补 multiseed validation。若继续沿 training-only rescue family 前进，更直接的下一步应是把 rescue 强度从固定 `0.05` 进一步解耦出来，例如补一个 **更小的 continuous rescue cap**（如 `<0.05`）或显式搜索 `train_non_dominant_weight_floor` 的低幅区间，先回答“能否在 seed42 上保住 `0.9362 val_acc` 的同时重新拿回 `DFR-26` 级别的 AUC 收益”，再决定是否值得回到 `seed=123`。

---

## 2026-04-23：Decision-Fusion Repair Follow-up（DFR-29 reduced continuous rescue scale，adaptive main-study，RTX A6000）

> **实验说明**
> - 本轮继续严格保持当前唯一主线与 `256x8 ResNeXt decision + L3-no-mixer + DFR-25` 语义不变，不切 backbone / geometry，不切到 feature fusion；唯一离散改动是在 commit `72f0bf3` 的 [src/model.py](/dataset/HH/ankle-ct/src/model.py) 新增 `ANKLE_DECISION_TRAIN_NONDOMINANT_RESCUE_SCALE`，把 `DFR-28` 的 continuous rescue 最大混合强度从实现里解耦出来。
> - 具体来说，训练态仍保持 `ANKLE_DECISION_TRAIN_NONDOMINANT_WEIGHT_FLOOR=0.05` 与 `ANKLE_DECISION_TRAIN_AXIAL_DOMINANCE_THRESHOLD=0.8` 的 `DFR-28` 语义，但通过新的 `ANKLE_DECISION_TRAIN_NONDOMINANT_RESCUE_SCALE=0.5` 把最大 rescue 混合强度减半；未显式设置该 env 时，现有行为保持不变。
> - fresh adaptive `main-study` 使用 search-config copy [autoresearch_logs/generated_search_configs/optuna_main_search_iter_0009_20260423_065356.yaml](/dataset/HH/ankle-ct/autoresearch_logs/generated_search_configs/optuna_main_search_iter_0009_20260423_065356.yaml) 与 study_root [runs/optuna_main_autoloop/iter_0009_20260423_065356](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0009_20260423_065356)，runtime env 为 `ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER=1` + `ANKLE_DECISION_TRAIN_DOMINANT_GATE_DROPOUT_PROB=0.25` + `ANKLE_DECISION_TRAIN_NONDOMINANT_WEIGHT_FLOOR=0.05` + `ANKLE_DECISION_TRAIN_AXIAL_DOMINANCE_THRESHOLD=0.8` + `ANKLE_DECISION_TRAIN_NONDOMINANT_RESCUE_SCALE=0.5`。
> - 本轮唯一研究问题是：如果 `DFR-28` 仍然没有超过 retained keep 的原因主要是 rescue 量过重，那么把 continuous rescue cap 减半，是否能保住 `DFR-25/26` 的 winning corner、同时重新拿回更高的 ranking quality，而不是把 learned routing 再次压回接近 single-view 行为。

- [x] **DFR-29-RESNEXT-DECISION-256X8-REDUCED-CONTINUOUS-RESCUE-SCALE-MAIN-S42**：fresh adaptive `main-study` [runs/optuna_main_autoloop/iter_0009_20260423_065356](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0009_20260423_065356) 的 best completed trial（trial `0`，commit `72f0bf3`，runtime env `ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER=1` + `ANKLE_DECISION_TRAIN_DOMINANT_GATE_DROPOUT_PROB=0.25` + `ANKLE_DECISION_TRAIN_NONDOMINANT_WEIGHT_FLOOR=0.05` + `ANKLE_DECISION_TRAIN_AXIAL_DOMINANCE_THRESHOLD=0.8` + `ANKLE_DECISION_TRAIN_NONDOMINANT_RESCUE_SCALE=0.5`）→ `val_acc=0.9148936170212766`, `val_auc=0.9836363636363636`, `val_f1=0.9090909090909091`, `peak_vram≈2.15 GiB`, `total_seconds≈972.7` → **discard**（相较 matched `equal-weight` control `seed=42` `0.9255319148936170 / 0.9486363636363637 / 0.9176470588235294`，主指标 `val_acc` 反而回落 `0.0106382978723404`；相较 retained `DFR-26` seed42 keep `0.9361702127659575 / 0.9800000000000000 / 0.9302325581395349`，`val_acc / val_f1` 分别回落 `0.0212765957446809 / 0.0211426490486258`，虽然 `val_auc` 只高 `0.0036363636363636`。按当前主指标规则，这是一条明确不能保留的回退。）
- **study 内部分布**：`6/6` completed trial 的 `val_acc` 全部锁在 `0.9148936170212766`，没有任何一个角点回到 `DFR-25/26/28` 已经证明可达的 `0.9361702127659575` 档。best trial `0`（`lr=1e-4`, `weight_decay=5e-4`, `dropout=0.25`, `gradient_clip_norm=2.0`）只是在这个统一回落平台里拿到最高 `val_auc=0.9836363636363636`；就连 `DFR-25/26` 的 seed42 winning corner（trial `1`: `lr=1e-4`, `weight_decay=2.5e-4`, `dropout=0.25`, `gradient_clip_norm=2.5`）也同样掉到 `0.9149`。
- **当前判断**：`DFR-29` 直接否定了“`DFR-28` 主要是 rescue cap 过重，所以把 cap 降低就能守住 accuracy”的假设。结果不是高方差式的小幅回落，而是 across-the-board 的一致退化：一旦把 continuous rescue 最大强度减半，整个 study 的 learned full-fusion ceiling 都退回到 `0.9149` 平台，甚至低于 matched `equal-weight`。这说明当前 training-only rescue family 的问题不在于“量太大”，而更像是“弱视角梯度一旦给得不够，repair 会整体失效”。
- **推荐动作**：不要继续沿 **更小 rescue cap / 更低 floor** 这条 amplitude 轴前进。若继续留在 `DFR-25` 家族，更合理的下一步应转向 **更精准的 full-strength trigger**，例如保持 `0.05` rescue 量不变，但把触发从单纯的 `axial weight / dominance excess` 改成更能识别真实塌缩样本的 `gate entropy` 或 `top1-top2 margin` 条件；目标是只在真正的 axial lock-in 样本上保留 `DFR-26` 那种 full-strength 弱视角梯度，而不是继续削弱干预幅度。

---

## 2026-04-23：Decision-Fusion Repair Follow-up（DFR-30 axial top1-top2 margin trigger，adaptive main-study，RTX A6000）

> **实验说明**
> - 本轮继续严格保持当前唯一主线与 `256x8 ResNeXt decision + L3-no-mixer + DFR-25` 语义不变，不切 backbone / geometry，不切到 feature fusion；唯一离散改动是在 commit `ef48be9` 的 [src/model.py](/dataset/HH/ankle-ct/src/model.py) 新增 `ANKLE_DECISION_TRAIN_AXIAL_TOP12_MARGIN_THRESHOLD`。
> - 这个修复仍然只在**训练态**生效，并保留 `DFR-26` 的 full-strength `ANKLE_DECISION_TRAIN_NONDOMINANT_WEIGHT_FLOOR=0.05` 语义；但触发条件不再看绝对 `axial weight`，而是只在 `axial` 为 dominant 且 `top1-top2` gate margin 至少达到 `0.7` 时才注入 weak-view rescue。推理期仍保持原始 learned late fusion，不引入任何 inference-time floor。
> - fresh adaptive `main-study` 使用 search-config copy [autoresearch_logs/generated_search_configs/optuna_main_search_iter_0010_20260423_074115.yaml](/dataset/HH/ankle-ct/autoresearch_logs/generated_search_configs/optuna_main_search_iter_0010_20260423_074115.yaml) 与 study_root [runs/optuna_main_autoloop/iter_0010_20260423_074115](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0010_20260423_074115)，runtime env 为 `ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER=1` + `ANKLE_DECISION_TRAIN_DOMINANT_GATE_DROPOUT_PROB=0.25` + `ANKLE_DECISION_TRAIN_NONDOMINANT_WEIGHT_FLOOR=0.05` + `ANKLE_DECISION_TRAIN_AXIAL_TOP12_MARGIN_THRESHOLD=0.7`。
> - 本轮唯一研究问题是：如果 `DFR-27~29` 的问题主要在于用 `absolute axial mass` 近似 collapse，导致要么误伤已迁移样本、要么错过真正的 starvation case，那么把 trigger 改成更直接刻画 “axial 与次强视角拉开多远” 的 `top1-top2 margin`，是否能在 seed42 上重新保住 `DFR-25/26` 的 `0.9362` 档，同时仍维持 learned routing 的净收益。

- [x] **DFR-30-RESNEXT-DECISION-256X8-AXIAL-TOP12-MARGIN-TRIGGER-MAIN-S42**：fresh adaptive `main-study` [runs/optuna_main_autoloop/iter_0010_20260423_074115](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0010_20260423_074115) 的 best completed trial（trial `0`，commit `ef48be9`，runtime env `ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER=1` + `ANKLE_DECISION_TRAIN_DOMINANT_GATE_DROPOUT_PROB=0.25` + `ANKLE_DECISION_TRAIN_NONDOMINANT_WEIGHT_FLOOR=0.05` + `ANKLE_DECISION_TRAIN_AXIAL_TOP12_MARGIN_THRESHOLD=0.7`）→ `val_acc=0.9255319148936170`, `val_auc=0.9772727272727273`, `val_f1=0.9213483146067416`, `peak_vram≈2.15 GiB`, `total_seconds≈928.8` → **discard**（相较 matched `equal-weight` control `seed=42` `0.9255319148936170 / 0.9486363636363637 / 0.9176470588235294`，只做到 `val_acc` 打平，虽然 `val_auc / val_f1` 仍高 `0.0286363636363636 / 0.0037012557832122`；但相较 retained `DFR-26` seed42 keep `0.9361702127659575 / 0.9800000000000000 / 0.9302325581395349`，`val_acc / val_auc / val_f1` 分别回落 `0.0106382978723405 / 0.0027272727272727 / 0.0088842435327933`；相较 `DFR-25` seed42 winner `0.9361702127659575 / 0.9786363636363636 / 0.9333333333333333`，同样回落 `0.0106382978723405 / 0.0013636363636363 / 0.0119850187265917`。按当前主指标规则，这条 precise-trigger repair 没有达到保留门槛。）
- **study 内部分布**：`6/6` completed trial 只形成 `0.9149` 与 `0.9255` 两档，没有任何一个角点回到 `DFR-25/26/28` 已经证明可达的 `0.9361702127659575`。更关键的是，`DFR-25/26` 的 seed42 winning corner（trial `1`: `lr=1e-4`, `weight_decay=2.5e-4`, `dropout=0.25`, `gradient_clip_norm=2.5`）在本轮只得到 `0.9255319148936170 / 0.9745454545454545 / 0.9195402298850575`；而 AUC-best trial `4` 虽然到 `0.9790909090909091`，`val_acc` 却只有 `0.9148936170212766`。这说明 hard `top1-top2 margin` trigger 不只是没超过 retained keep，而是把整个 study ceiling 一起压回了 `equal-weight tie` / `0.9149` 平台。
- **当前判断**：`DFR-30` 在机制上依然是合格方向，因为它直接针对 `weak-view starvation` 的 gate gap，而不是再次回到 amplitude 轴；但结果上它比预期更差，说明 **hard margin trigger at `0.7` 太稀疏**。从本轮 study 看，一旦 rescue 只覆盖“极端拉开”的 axial lock-in 样本，`DFR-25` 建立起来的弱视角梯度通路就不足以维持 `0.9362` ceiling，seed42 又退回到了“至多与 matched `equal-weight` 打平”的状态。
- **推荐动作**：保持 `256x8 ResNeXt decision + L3-no-mixer + DFR-25` 主线不变，也**不要**回到更小 rescue cap / 更低 floor。若继续沿 precise-trigger family 前进，更直接的下一步应是测试 **coverage 更平滑的 collapse score**，例如 `gate entropy` 或 **continuous top1-top2 margin scaling**：目标不是削弱 `0.05` rescue 量，而是让 near-collapse 样本也能收到足够的 weak-view gradient，同时仍避免 `DFR-26` 那种 blanket floor 对已迁移样本的无差别压平。

---

## 2026-04-22：人类追加任务（融合权重合理性 side campaign，限定 3 轮 autoresearch）

> **高优先级 side campaign 说明**
> - 人类已明确要求：把“融合权重合理性”补成一个 **限定 3 轮 outer autoresearch iterations** 的独立 side campaign；本轮只跑 3 轮，然后退出，**不要无限循环**。
> - 当前已有独立作业 **`432763` / `ankle-rx256x8-l6s123`** 在 `node20` 上运行，负责 `L6` scalar repair（`L3-no-mixer` 的 `seed=123` tuning）。这个 side campaign **不得重复 `L6`**，不得复用该作业的 `study_root`、`output_dir`、日志路径或任何正在运行的 trial 目录。
> - 当前 side campaign 的默认主锚点不是 `L1`，而是 **`256x8 ResNeXt decision fusion + L3-no-mixer`**（即 `ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER=1` 的 strongest learned branch）。如需对照，可派生 `equal-weight`、single-view 或 leave-one-view-out control，但**不要切回 `512x16`**。
> - 资源约束：仅申请 **V100q 32GB** 卡；单节点运行；不超过 4 张 GPU。本轮计划优先用 **2 张 V100 32GB** 启动 bounded loop，既避免与 `432763` 冲突，也避免单卡串行把 3 轮 side campaign 拖得过慢。
> - 若某一轮需要新增轻量 instrumentation，优先把分析产物写到各自 `output_dir`（例如 `fusion_weight_analysis.json`、`view_ablation_summary.json`、`perturbation_summary.json`），避免把结论只留在临时 shell 输出里。
>
> **本轮 3 个固定任务（按顺序执行）**
> - [x] **FWR-01 single-view / leave-one-view-out matched control**：已在 fresh adaptive `main-study` `runs/optuna_main_autoloop/iter_0001_20260422_020718` 上完成；结论是当前 `L3-no-mixer` `seed=42` 的 full fusion 与 `single_view_axial`、`leave_out_coronal`、`leave_out_sagittal` 完全同分，收益主要表现为 **axial dominance**，而不是多视角间更细粒度的 learned redistribution。
> - [x] **FWR-02 fusion-weight telemetry**：已对 `runs/optuna_main_autoloop/iter_0001_20260422_020718` 的 `L3-no-mixer` `trial_0000` checkpoint 生成 `fusion_weight_analysis.json`；结论是 learned fusion 的权重几乎完全塌缩到 axial（`mean weight=0.997426`, `top-weight axial=94/94`），`top-weight hit rate` 虽然对 `pred_margin / true_margin` 仍有 `0.861702 / 0.882979`，但这更像 **“axial 恒定主导”** 而不是按样本把权重迁到最有证据的视角。
> - [x] **FWR-03 perturbation-based weight migration**：已对同一 `L3-no-mixer` winner checkpoint 做 `axial` 可复现高斯模糊扰动（`kernel=17`, `sigma=4.0`）；结论是即便 axial 证据被明显打坏，fusion top-weight 仍然 `94/94` 固定留在 axial，learned fusion 并不会把权重迁向 coronal / sagittal。
>
> **执行规则**
> - 外层 loop 固定为 `max-iterations=3`，每轮只做一个离散实验，不要在同一轮混多个独立想法。
> - 除非 implementation risk 很高，否则优先用当前 canonical `256x8` formal/main lane；只有 smoke/debug 才允许先走 proxy。
> - 每轮都必须更新 `backlog.md` 与 `results.tsv`；记录时明确标注这是 `fusion-weight rationality` side campaign，而不是 `L6` scalar repair 延续。

## 2026-04-22：人类更正后的后续主线（decision fusion only）

> **主线修正说明**
> - 人类已明确要求：后续主线 **不要切到 feature fusion**；最终输出语义必须保持 `decision fusion / late fusion`。
> - feature-fusion 既有实现与历史 keep（尤其 `15b1ef6` 的 `per-view recalibration + light cross-view mixer + gated head`）只作为**模块参考**，允许把其中局部结构迁入 decision-fusion 的 `per-view classifier` 或 `reliability` 路径，但**不允许**把整条主线改成 `fusion_type=feature`。
> - 因此，新的结构性修复优先级改成：先修 **per-view logits 的上下文建模能力**，再补 **auxiliary per-view supervision**，最后补 **train-time 视角鲁棒性**；三步都必须保持最终输出仍是“3 个视角 logits 经 late fusion 合成”。

- [x] **DFR-01 contextual per-view classifier path on top of decision late fusion**：已在 `node19` 上完成 `seed=42` formal（job `433578`，commit `942b1e9`，config `configs/cmp_resnext_decision_256x8_dfr01_classifier_context_formal_s42.yaml`，runtime env `ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER=1` + `ANKLE_DECISION_ENABLE_CLASSIFIER_VIEW_CONTEXT=1`）→ best `val_acc=0.8723404255319149`, `val_auc=0.9350000000000000`, `val_f1=0.8604651162790697`，明显低于当前 `L3-no-mixer` seed42 anchor `0.9255319148936170 / 0.9777272727272727 / 0.9213483146067416`，因此 **discard**。
- [x] **DFR-02 auxiliary per-view supervision for decision fusion**：已在 `node19` 上完成 `seed=42` formal（job `433581`，commit `feba55e`，config `configs/cmp_resnext_decision_256x8_dfr02_aux_view_loss_formal_s42.yaml`，runtime env `ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER=1` + `ANKLE_DECISION_ENABLE_AUX_VIEW_LOSS=1` + `ANKLE_DECISION_AUX_VIEW_LOSS_WEIGHT=0.5`）→ best `val_acc=0.8723404255319149`, `val_auc=0.9468181818181818`, `val_f1=0.8723404255319149`，同样明显低于当前 `L3-no-mixer` seed42 anchor `0.9255319148936170 / 0.9777272727272727 / 0.9213483146067416`，因此 **discard**。
- [x] **DFR-03 train-time view robustness for decision fusion**：已在 `node19` 上完成 `seed=42` formal（job `433591`，commit `ecf7d47`，config `configs/cmp_resnext_decision_256x8_dfr03_view_robustness_formal_s42.yaml`，runtime env `ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER=1` + `ANKLE_DECISION_TRAIN_VIEW_DROPOUT_PROB=0.15` + `ANKLE_DECISION_TRAIN_AXIAL_BLUR_PROB=0.30` + `ANKLE_DECISION_TRAIN_AXIAL_BLUR_KERNEL=9`）→ best `val_acc=0.9042553191489362`, `val_auc=0.9600000000000001`, `val_f1=0.8988764044943820`，优于 `DFR-01/02`，但仍低于当前 `L3-no-mixer` seed42 anchor `0.9255319148936170 / 0.9777272727272727 / 0.9213483146067416`，因此 **discard**。
- [x] **DFR-04 bounded late-fusion gating for decision fusion**：已在 `node19` 上完成 `seed=42` formal（job `433646`，commit `70d22ac`，config `configs/cmp_resnext_decision_256x8_dfr04_bounded_gating_formal_s42.yaml`，runtime env `ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER=1` + `ANKLE_DECISION_FUSION_WEIGHT_FLOOR=0.10`）→ best `val_acc=0.9255319148936170`, `val_auc=0.9777272727272727`, `val_f1=0.9156626506024096`，在 accuracy / AUC 上追平当前 `L3-no-mixer` seed42 anchor，但 `val_f1` 仍低 `0.0056856640043320`，且代码更复杂，因此 **discard**。
- [x] **DFR-05 mild bounded late-fusion gating for decision fusion**：已在 `node19` 上完成 `seed=42` formal（job `433674`，commit `0415da9`，config `configs/cmp_resnext_decision_256x8_dfr05_bounded_gating_floor05_formal_s42.yaml`，runtime env `ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER=1` + `ANKLE_DECISION_FUSION_WEIGHT_FLOOR=0.05`）→ best `val_acc=0.9468085106382979`, `val_auc=0.9736363636363636`, `val_f1=0.9425287356321839`，相较当前 retained `L3-no-mixer` seed42 anchor `0.9255319148936170 / 0.9777272727272727 / 0.9213483146067416` 提升 `+0.0212765957446809 val_acc` 与 `+0.0211804210254423 val_f1`；但该结论已被后续 `DFR-07~09` multiseed validation 修正为 **single-seed positive signal only**，不再升格为新的 decision-only mainline reference。
- [x] **DFR-07 exact formal confirmation on bounded-gating floor=0.05**：已在 `node19` 上完成同配方 `seed=42` 复跑（job `433760`，commit `102460a`，config `configs/cmp_resnext_decision_256x8_dfr07_bounded_gating_floor05_confirm_formal_s42.yaml`）→ best `val_acc=0.9255319148936170`, `val_auc=0.9809090909090908`, `val_f1=0.9213483146067416`；相较 matched `L3-no-mixer` seed42 只是在 `val_acc / val_f1` 打平的前提下，用更高 `val_auc` 小幅占优，但**没有复现** `DFR-05` 的 `0.9468` spike。
- [x] **DFR-08 alternate-seed validation on bounded-gating floor=0.05 (`seed=123`)**：已在 `node20` 上完成 formal（job `433764`，commit `3f7bc35`，config `configs/cmp_resnext_decision_256x8_dfr08_bounded_gating_floor05_formal_s123.yaml`）→ best `val_acc=0.8617021276595744`, `val_auc=0.9263636363636363`, `val_f1=0.8433734939759037`，明显低于 matched `L3-no-mixer` seed123 `0.8936170212765957 / 0.9395454545454545 / 0.8750000000000000`，因此 **discard**。
- [x] **DFR-09 alternate-seed validation on bounded-gating floor=0.05 (`seed=456`)**：已在 `node20` 上完成 formal（job `433765`，commit `3f7bc35`，config `configs/cmp_resnext_decision_256x8_dfr09_bounded_gating_floor05_formal_s456.yaml`）→ best `val_acc=0.9361702127659575`, `val_auc=0.9690909090909091`, `val_f1=0.9285714285714286`；相较 matched `L3-no-mixer` seed456，`val_acc / val_f1` 分别提升 `0.0106382978723405 / 0.0109243697478992`，但 `val_auc` 小幅回落 `0.0022727272727272`，可记为 **keep**。
- [x] **DFR-11 per-view logit temperature calibration on top of decision late fusion**：已在 `node20` 上完成 `seed=42` formal（job `433864`，commit `a8994ea`，config `configs/cmp_resnext_decision_256x8_dfr11_view_logit_temperature_formal_s42.yaml`，runtime env `ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER=1` + `ANKLE_DECISION_ENABLE_VIEW_LOGIT_TEMPERATURE=1`）→ best `val_acc=0.9148936170212766`, `val_auc=0.9745454545454546`, `val_f1=0.9111111111111111`，仍低于当前 `L3-no-mixer` seed42 anchor `0.9255319148936170 / 0.9777272727272727 / 0.9213483146067416`，因此 **discard**。
- [x] **DFR-12 teacher-guided gate blend on top of calibrated decision late fusion**：已在 `node20` 上完成 `seed=42` formal（job `433863`，commit `a8994ea`，config `configs/cmp_resnext_decision_256x8_dfr12_teacher_guided_gate_formal_s42.yaml`，runtime env `ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER=1` + `ANKLE_DECISION_ENABLE_VIEW_LOGIT_TEMPERATURE=1` + `ANKLE_DECISION_GATE_TEACHER_BLEND=0.30` + `ANKLE_DECISION_GATE_TEACHER_TEMPERATURE=1.0`）→ best `val_acc=0.9148936170212766`, `val_auc=0.9731818181818181`, `val_f1=0.9000000000000000`，同样低于当前 `L3-no-mixer` seed42 anchor，因此 **discard**。
- **DFR-11/12 calibration-gating 结论**：两条新线都只把 seed42 ceiling 拉到 `0.9148936170212766 val_acc`，没有翻过当前 `L3-no-mixer` anchor `0.9255319148936170`。其中 `DFR-11` 以更高 `val_auc / val_f1`（`0.9745454545454546 / 0.9111111111111111`）赢过 `DFR-12`（`0.9731818181818181 / 0.9000000000000000`），但快速 checkpoint inspection 显示学到的 per-view temperature 仍近乎恒等 `~[0.9990, 1.0006, 1.0013]`；因此当前更合理的下一步不是继续放大 teacher-guided routing，而是把 `DFR-11` 作为较稳的基底，再与已知有机制价值的 `floor=0.05` 做组合验证。
- **bounded-gating floor=0.05 multiseed 结论**：用 `DFR-07/08/09` 组成的 3-seed formal mean 为 `val_acc=0.9078014184397163`, `val_auc=0.9587878787878786`, `val_f1=0.8977644123846913`，`val_acc` population std 为 `0.0328851719698429`；整体低于 `L3-no-mixer` 的 `0.9148936170212766 / 0.9628787878787879 / 0.9046651244767570` 与 `0.0150448251316287`，因此当前只能认定 bounded-gating 在 **减弱 axial 独占** 上有机制价值，但还不够稳，不能替代 `L3-no-mixer` 作为主线锚点。

## 2026-04-22：Decision-Fusion Repair（DFR-01 contextual per-view classifier path，formal，node19）

> **实验说明**
> - 人类已更正主线：后续修复必须保持 `decision fusion`，不能切到 `feature fusion`；因此本轮只把 feature-fusion 里已验证过的轻量 `per-view recalibration + cross-view mixer` 借到 `view_classifiers` 之前，最终输出仍保持 `view_logits -> fusion_weights -> fused logits`。
> - 代码落点是 commit `942b1e9`：在 `src/model.py` 里新增 `ANKLE_DECISION_ENABLE_CLASSIFIER_VIEW_CONTEXT`，只上下文化 per-view classifier path；reliability path 仍保留 `L3-no-mixer` 语义（通过 `ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER=1` 关闭 reliability mixer）。
> - 运行使用隔离 formal config `configs/cmp_resnext_decision_256x8_dfr01_classifier_context_formal_s42.yaml` 与 Slurm job `433578`（`V100q / node19 / 1xV100 32GB / 24 CPU / 96G`），输出目录 `runs/resnext_decision_256x8_mainline/dfr01_classifier_context_formal_s42`。

- [x] **DFR-01-RESNEXT-DECISION-256X8-CLASSIFIER-CONTEXT-FORMAL-S42**：`runs/resnext_decision_256x8_mainline/dfr01_classifier_context_formal_s42`（commit `942b1e9`）→ `val_acc=0.8723404255319149`, `val_auc=0.9350000000000000`, `val_f1=0.8604651162790697`, `peak_vram≈2.21 GiB`, `total_seconds≈828.4` → **discard**（相较当前 retained `L3-no-mixer` seed42 anchor `27b557c` 的 `0.9255319148936170 / 0.9777272727272727 / 0.9213483146067416`，accuracy / AUC / F1 分别回落 `0.0531914893617021 / 0.0427272727272727 / 0.0608831983276719`，因此没有主线保留价值。）
- **训练轨迹观察**：首轮验证直接掉到 `val_acc=0.4681 / val_auc=0.7005`，之后虽回升到 `epoch 3` 的 `0.8404 / 0.8791`，最终 best 也只到 `0.8723 / 0.9350`；中后期还出现多次明显震荡，例如 `epoch 11` 一度掉到 `val_acc=0.5106`、`val_loss=1.8744`。这说明“仅在 classifier path 引入 cross-view context”并没有修复当前 decision-fusion 的 axial-dominant failure mode，反而带来了更强的不稳定性。
- **当前判断**：`DFR-01` 给出的结论是负面的。feature-fusion 里有效的轻量 cross-view context，直接迁到 decision path 并不能自然转化为更强的 late fusion；至少在当前 `256x8 ResNeXt` 几何与 `seed=42` formal 语义下，这条修复方向会明显伤害主指标。
- **推荐动作**：下一步进入 **`DFR-02 auxiliary per-view supervision`**，但不要再保留本轮的 classifier-context 结构。更合理的下一轮是回到当前 strongest `L3-no-mixer` anchor，在不引入 classifier-path cross-view mixing 的前提下，只给 per-view logits 加显式辅助监督，先测试 coronal / sagittal 是否能被拉回有效判别。

## 2026-04-22：Decision-Fusion Repair（DFR-02 auxiliary per-view supervision，formal，node19）

> **实验说明**
> - 本轮回到当前 strongest decision anchor `256x8 ResNeXt + L3-no-mixer`，不保留 `DFR-01` 的 classifier-path context，只利用现有 `train.py` 里的 `_view_logits/_log_vars` 钩子，把 per-view CE 作为显式 auxiliary supervision 叠加回训练损失。
> - 代码落点是 commit `feba55e`：在 `src/model.py` 里新增 `ANKLE_DECISION_ENABLE_AUX_VIEW_LOSS` 与 `ANKLE_DECISION_AUX_VIEW_LOSS_WEIGHT`，通过已有 UWDF loss hook 暴露固定权重的 per-view auxiliary CE；late-fusion 输出语义仍保持 `view_logits -> fusion_weights -> fused logits`。
> - 运行使用隔离 formal config `configs/cmp_resnext_decision_256x8_dfr02_aux_view_loss_formal_s42.yaml` 与 Slurm job `433581`（`V100q / node19 / 1xV100 32GB / 24 CPU / 96G`），输出目录 `runs/resnext_decision_256x8_mainline/dfr02_aux_view_loss_formal_s42`。

- [x] **DFR-02-RESNEXT-DECISION-256X8-AUX-VIEW-LOSS-FORMAL-S42**：`runs/resnext_decision_256x8_mainline/dfr02_aux_view_loss_formal_s42`（commit `feba55e`）→ `val_acc=0.8723404255319149`, `val_auc=0.9468181818181818`, `val_f1=0.8723404255319149`, `peak_vram≈2.20 GiB`, `total_seconds≈820.1` → **discard**（相较当前 retained `L3-no-mixer` seed42 anchor `27b557c` 的 `0.9255319148936170 / 0.9777272727272727 / 0.9213483146067416`，accuracy / AUC / F1 分别回落 `0.0531914893617021 / 0.0309090909090909 / 0.0490078890748267`，因此没有主线保留价值。）
- **训练轨迹观察**：辅助监督把早期 collapse 稍微拉回来了，`epoch 7` 左右一度达到 `val_acc=0.8511 / val_auc=0.9268`，最终 best 也抬到 `0.8723 / 0.9468`，比 `DFR-01` 的 `0.8723 / 0.9350` 在 ranking 上更好；但它始终没有逼近 `L3-no-mixer` anchor 的 accuracy ceiling，而且整体训练 loss 长时间停留在 `>1.0`，说明这类显式 per-view CE 并没有真正把弱视角转化成可用的 late-fusion 增益。
- **当前判断**：`DFR-02` 相比 `DFR-01` 更稳一些，但结论仍然是否定的。单纯给三视角 logits 加辅助监督，不足以修复 `FWR` 暴露出来的 axial hard-selection；它更多像是在训练期给 view classifiers 加了额外约束，却没有改变 inference 时 evidence routing 的基本格局。
- **推荐动作**：继续进入 **`DFR-03 train-time view robustness`**。既然 `DFR-01/02` 都没有解决“模型过度依赖 axial”这个根因，下一步应直接在训练期制造视角缺失 / axial 退化样本，测试 late-fusion 是否能被迫学出更稳的路由策略。

## 2026-04-22：Decision-Fusion Repair（DFR-03 train-time view robustness，formal，node19）

> **实验说明**
> - 本轮继续保持 strongest decision anchor `256x8 ResNeXt + L3-no-mixer` 不变，不叠加 `DFR-01/02` 的结构改动，只在训练态对输入视角做轻量鲁棒性扰动，直接针对 `FWR-03` 暴露出的 axial hard-selection。
> - 代码落点是 commit `ecf7d47`：在 `src/model.py` 里新增 `ANKLE_DECISION_TRAIN_VIEW_DROPOUT_PROB`、`ANKLE_DECISION_TRAIN_AXIAL_BLUR_PROB`、`ANKLE_DECISION_TRAIN_AXIAL_BLUR_KERNEL`，仅在 `model.training` 时对输入做“随机单视角 dropout + 受控 axial avg-pool blur”；inference 与 late-fusion 语义保持不变。
> - 运行使用隔离 formal config `configs/cmp_resnext_decision_256x8_dfr03_view_robustness_formal_s42.yaml` 与 Slurm job `433591`（`V100q / node19 / 1xV100 32GB / 24 CPU / 96G`），输出目录 `runs/resnext_decision_256x8_mainline/dfr03_view_robustness_formal_s42`。

- [x] **DFR-03-RESNEXT-DECISION-256X8-VIEW-ROBUSTNESS-FORMAL-S42**：`runs/resnext_decision_256x8_mainline/dfr03_view_robustness_formal_s42`（commit `ecf7d47`）→ `val_acc=0.9042553191489362`, `val_auc=0.9600000000000001`, `val_f1=0.8988764044943820`, `peak_vram≈2.23 GiB`, `total_seconds≈810.3` → **discard**（相较当前 retained `L3-no-mixer` seed42 anchor `27b557c` 的 `0.9255319148936170 / 0.9777272727272727 / 0.9213483146067416`，accuracy / AUC / F1 分别回落 `0.0212765957446808 / 0.0177272727272726 / 0.0224719101123596`，因此仍不足以晋升主线。）
- **训练轨迹观察**：这条线是三轮 DFR 里唯一把 val_acc 拉回 `0.90+` 的修复。早期 `epoch 1-2` 先明显退化到 `0.4681`，但中后期逐步回升，在 `epoch 10` 左右达到 best `val_acc=0.9043 / val_auc=0.9600`；之后还有较大波动，末段重新回落到 `0.89` 左右，说明训练期鲁棒扰动确实能缓解一部分 axial 依赖，但稳定性仍不够。
- **当前判断**：`DFR-03` 是目前三条 decision-only 修复里最有信息量的一条。它表明“训练时强迫模型见到 axial 退化样本”比单纯做 classifier-context 或 auxiliary CE 更接近问题根因；但按当前这组扰动强度（`view_dropout=0.15`, `axial_blur_prob=0.30`, `kernel=9`），提升还不足以超过原始 `L3-no-mixer` anchor。
- **推荐动作**：本轮 `DFR-01~03` 已全部完成。若后续继续沿 decision-only 主线优化，优先级应落在 **以 `DFR-03` 为起点做小范围强度搜索**，而不是回退到 `DFR-01/02`：只调 `view_dropout_prob / axial_blur_prob / blur kernel`，并补一轮 matched `FWR-03` 式后验分析，确认 robustness training 是否真的让权重迁移而不仅是局部 accuracy repair。

## 2026-04-22：Decision-Fusion Repair（DFR-04 bounded late-fusion gating，formal，node19）

> **实验说明**
> - 本轮继续保持 strongest decision anchor `256x8 ResNeXt + L3-no-mixer` 不变，不叠加 `DFR-01~03` 的结构或训练扰动，只在最终 late-fusion 权重上加入一个轻量的 **minimum-weight floor**，直接约束 `fusion_weights` 不再塌到近似 `1.0 / 0.0 / 0.0`。
> - 代码落点是 commit `70d22ac`：在 `src/model.py` 里新增 `ANKLE_DECISION_FUSION_WEIGHT_FLOOR`，把 raw softmax 权重映射到“每个 active view 至少保留固定最小质量”的 bounded-gating 语义；同时对 `active_view_mask` 控制路径做了兼容，保证 leave-one-view-out 分析不会把被屏蔽视角重新加回来。
> - 运行使用隔离 formal config `configs/cmp_resnext_decision_256x8_dfr04_bounded_gating_formal_s42.yaml` 与 Slurm job `433646`（`V100q / node19 / 1xV100 32GB / 24 CPU / 96G`），输出目录 `runs/resnext_decision_256x8_mainline/dfr04_bounded_gating_formal_s42`；runtime env 为 `ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER=1` + `ANKLE_DECISION_FUSION_WEIGHT_FLOOR=0.10`。

- [x] **DFR-04-RESNEXT-DECISION-256X8-BOUNDED-GATING-FORMAL-S42**：`runs/resnext_decision_256x8_mainline/dfr04_bounded_gating_formal_s42`（commit `70d22ac`）→ `val_acc=0.9255319148936170`, `val_auc=0.9777272727272727`, `val_f1=0.9156626506024096`, `peak_vram≈2.15 GiB`, `total_seconds≈822.3` → **discard**（与当前 retained `L3-no-mixer` seed42 anchor `27b557c` 在 `val_acc` 与 `val_auc` 上完全打平，但 `val_f1` 仍低 `0.0056856640043320`，且 bounded-gating 比 anchor 多了一层 late-fusion 特殊逻辑，因此按 tie-break 不能保留。）
- **训练轨迹观察**：这条线的前期收敛明显更慢，`epoch 1/2` 只有 `0.5213 / 0.6918` 与 `0.4787 / 0.8936`；但中段开始快速回升，`epoch 7` 达到 `val_acc=0.9043 / val_auc=0.9736`，`epoch 11` 进一步冲到本轮 best `0.9255 / 0.9777`。后半程依然有明显波动，`epoch 13` 还一度掉到 `val_acc=0.6915`，说明 bounded-gating 的确能压住极端权重塌缩，但训练稳定性并没有根本解决。
- **当前判断**：`DFR-04` 是当前 decision-only 修复里第一条 **真正追平 anchor accuracy / AUC** 的路线，因此它比 `DFR-01~03` 更接近有效修复；但它仍没有把主指标抬过 anchor，也没有把 F1 拉平，按当前模型选择规则只能记为 **promising discard**，不能晋升主线。
- **推荐动作**：下一步不要立刻把 bounded-gating 和更多结构/损失叠在一起。更合理的是先在同一语义下做一个 **更温和的 floor 强度搜索**，优先测试 `ANKLE_DECISION_FUSION_WEIGHT_FLOOR=0.05` 这类更小的 bounded-gating；如果更温和的边界仍然只能打平 anchor，再考虑把 bounded-gating 作为 `DFR-03` robustness training 的配套约束，而不是单独扩张 late-fusion 逻辑。

## 2026-04-22：Decision-Fusion Repair（DFR-05 mild bounded late-fusion gating，formal，node19）

> **实验说明**
> - 本轮延续 `DFR-04` 的 bounded-gating 语义，不改模型结构、不叠加训练期扰动，只把 late-fusion 的 minimum-weight floor 从 `0.10` 收到更温和的 `0.05`，直接回答“上一轮只是边界压得太狠，还是 bounded-gating 本身有价值”。
> - 代码与 `DFR-04` 共用同一实现，只通过新的隔离 formal config `configs/cmp_resnext_decision_256x8_dfr05_bounded_gating_floor05_formal_s42.yaml` 和 Slurm 脚本 `scripts/slurm_resnext_decision_256x8_dfr05_bounded_gating_floor05.sbatch` 覆写 runtime env；实验 commit 是 `0415da9`，运行环境为 `ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER=1` + `ANKLE_DECISION_FUSION_WEIGHT_FLOOR=0.05`。
> - 运行使用 Slurm job `433674`（`V100q / node19 / 1xV100 32GB / 24 CPU / 96G`），输出目录 `runs/resnext_decision_256x8_mainline/dfr05_bounded_gating_floor05_formal_s42`。

- [x] **DFR-05-RESNEXT-DECISION-256X8-BOUNDED-GATING-FLOOR05-FORMAL-S42**：`runs/resnext_decision_256x8_mainline/dfr05_bounded_gating_floor05_formal_s42`（commit `0415da9`）→ `val_acc=0.9468085106382979`, `val_auc=0.9736363636363636`, `val_f1=0.9425287356321839`, `peak_vram≈2.15 GiB`, `total_seconds≈874.3` → **keep**（相较当前 retained `L3-no-mixer` seed42 anchor `27b557c` 的 `0.9255319148936170 / 0.9777272727272727 / 0.9213483146067416`，accuracy 提升 `0.0212765957446809`、F1 提升 `0.0211804210254423`，虽然 AUC 回落 `0.0040909090909091`，但按主指标 `val_acc` 规则这是明确的新主线胜出。）
- **训练轨迹观察**：这轮的收敛仍然保留 bounded-gating 的“先抖后升”特征，但比 `floor=0.10` 更健康。`epoch 1` 就先站到 `val_acc=0.7340`，随后在 `epoch 6/7/9` 依次抬到 `0.8617 / 0.9043 / 0.9149`，最终在 `epoch 12` 打到 best `val_acc=0.9468 / val_auc=0.9736 / val_f1=0.9425`。后段仍有波动，但本轮 ceiling 已显著高于旧 anchor，也高于 `DFR-04`。
- **当前判断**：`DFR-05` 仍然是一个有效的 **single-seed positive keep**。它证明 bounded-gating 的思路不是完全无效，`0.05` 也确实比 `0.10` 合理；但这个“新主线胜出”判断已被后续 `DFR-07~09` 修正，当前不能仅凭这一条单 seed 结果替换 `L3-no-mixer` 主锚点。
- **推荐动作**：后续不要再把 `DFR-05` 当成已经确认的新 anchor；所有围绕 bounded-gating 的后续工作，都应转成“如何进一步减弱单视角依赖、尤其是解决 sagittal starvation”的局部修复，而不是默认它已经整体胜过 `L3-no-mixer`。

## 2026-04-22：Decision-Fusion Follow-up（DFR-06 fusion-weight telemetry on DFR-05 winner，existing checkpoint telemetry，node19）

> **实验说明**
> - 这是对新 keep `DFR-05` 的机制侧复核，不新开训练，只在现有 winner checkpoint 上跑离线 telemetry，检查 bounded-gating 的收益是否真的伴随 “axial hard-selection 缓解”。
> - 使用 `scripts/analyze_fusion_weights.py`，输入 config 为 `configs/cmp_resnext_decision_256x8_dfr05_bounded_gating_floor05_formal_s42.yaml`，checkpoint 为 `runs/resnext_decision_256x8_mainline/dfr05_bounded_gating_floor05_formal_s42/best.pt`；runtime env 继续保持 `ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER=1` + `ANKLE_DECISION_FUSION_WEIGHT_FLOOR=0.05`。
> - 输出文件为 `runs/resnext_decision_256x8_mainline/dfr05_bounded_gating_floor05_formal_s42/fusion_weight_analysis.json`，运行 commit 为 `36d99ee`，不产生新的模型 checkpoint。

- [x] **DFR-06-RESNEXT-DECISION-256X8-BOUNDED-GATING-FLOOR05-TELEMETRY**：分析 `DFR-05` winner checkpoint（训练指标仍是 `val_acc=0.9468085106382979`, `val_auc=0.9736363636363636`, `val_f1=0.9425287356321839`）并产出 `fusion_weight_analysis.json` → **discard**（本轮是机制验证，不产生新的更优 checkpoint；其价值在于回答 `DFR-05` 为什么有效，而不是替换当前 keep。）
- **权重分布变化**：相较旧 `L3-no-mixer` winner 的 `axial top-weight 94/94` 与 `mean axial weight=0.997426`，`DFR-05` 已显著去塌缩。新的 top-weight 分布变成 `axial=57 / coronal=37 / sagittal=0`，mean fusion weight 则是 `axial=0.5558 / coronal=0.3920 / sagittal=0.0522`。这说明 bounded-gating 确实把 late-fusion 从“几乎纯 axial selector”拉回到了“axial+coronal 共治”的状态。
- **证据对齐情况**：`top_weight hit rate` 对 `true_margin` 是 `0.9043`，高于旧 winner 的 `0.8830`；同时 `top_true_margin` 分布也从过去几乎固定 axial，变成 `axial=50 / coronal=44 / sagittal=0`。这说明新 winner 的权重已经更接近“谁对当前样本更有真实判别证据”。但 `top_pred_margin` 几乎都落在 `coronal=83/94`，而 top-weight 只给了 coronal `37/94`，说明 coronal 分支仍有明显的 over-confident tendency，需要继续谨慎对待。
- **当前判断**：`DFR-06` 给出了一个偏正面的机制证据。`DFR-05` 的收益不是纯偶然，它确实缓解了旧主线的极端 axial dominance；不过当前新 winner 也没有完全学成理想的三视角平衡，更多像是从 “axial hard-selection” 修到了 “axial-coronal 双主导、sagittal 仍弱”。
- **推荐动作**：下一步优先做 **exact direct formal confirmation**，先确认 `DFR-05` 这组 mild bounded-gating 在同一 formal 语义下可复现；如果 confirmation 仍站住 `0.94+ val_acc`，再决定是沿 `floor` 轴做 alternate-seed 扩展，还是把 `DFR-05` 当新 anchor 去叠更轻的 robustness training。

## 2026-04-22：Decision-Fusion Follow-up（DFR-07~09 bounded-gating floor=0.05 validation，formal multiseed，node19/node20）

> **实验说明**
> - 这是对 `DFR-05` 的正式复核，不再引入新变量，只做同机制下的 `seed=42 exact confirmation` 与 `seed=123/456 alternate-seed validation`。
> - `DFR-07` 使用 commit `102460a`、job `433760`、config `configs/cmp_resnext_decision_256x8_dfr07_bounded_gating_floor05_confirm_formal_s42.yaml` 在 `node19` 上复跑；`DFR-08/09` 使用 commit `3f7bc35`、jobs `433764/433765`、configs `configs/cmp_resnext_decision_256x8_dfr08_bounded_gating_floor05_formal_s123.yaml` 与 `configs/cmp_resnext_decision_256x8_dfr09_bounded_gating_floor05_formal_s456.yaml` 在 `node20` 上完成。
> - 三轮都保持相同 runtime path：`ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER=1` + `ANKLE_DECISION_FUSION_WEIGHT_FLOOR=0.05`；目标只回答一件事：`DFR-05` 的改善是否足够稳定，能否替换 `L3-no-mixer` 成为新的 decision-only mainline。

- [x] **DFR-07-RESNEXT-DECISION-256X8-BOUNDED-GATING-FLOOR05-CONFIRM-FORMAL-S42**：`runs/resnext_decision_256x8_mainline/dfr07_bounded_gating_floor05_confirm_formal_s42`（commit `102460a`）→ `val_acc=0.9255319148936170`, `val_auc=0.9809090909090908`, `val_f1=0.9213483146067416`, `peak_vram≈2.15 GiB`, `total_seconds≈822.8` → **keep**（相较 matched `L3-no-mixer` seed42，`val_acc / val_f1` 打平，`val_auc` 提升 `0.0031818181818181`；但它没有复现 `DFR-05` 的 `0.9468` 峰值，说明这条线在同 seed 下也存在明显不稳定性。）
- [x] **DFR-08-RESNEXT-DECISION-256X8-BOUNDED-GATING-FLOOR05-FORMAL-S123**：`runs/resnext_decision_256x8_mainline/dfr08_bounded_gating_floor05_formal_s123`（commit `3f7bc35`）→ `val_acc=0.8617021276595744`, `val_auc=0.9263636363636363`, `val_f1=0.8433734939759037`, `peak_vram≈2.15 GiB`, `total_seconds≈881.6` → **discard**（相较 matched `L3-no-mixer` seed123，accuracy / AUC / F1 分别回落 `0.0319148936170213 / 0.0131818181818182 / 0.0316265060240963`；这是当前 bounded-gating floor=0.05 最清楚的负例。）
- [x] **DFR-09-RESNEXT-DECISION-256X8-BOUNDED-GATING-FLOOR05-FORMAL-S456**：`runs/resnext_decision_256x8_mainline/dfr09_bounded_gating_floor05_formal_s456`（commit `3f7bc35`）→ `val_acc=0.9361702127659575`, `val_auc=0.9690909090909091`, `val_f1=0.9285714285714286`, `peak_vram≈2.15 GiB`, `total_seconds≈889.8` → **keep**（相较 matched `L3-no-mixer` seed456，accuracy / F1 分别提升 `0.0106382978723405 / 0.0109243697478992`，`val_auc` 仅小幅回落 `0.0022727272727272`；说明 bounded-gating 仍然有局部 seed 价值。）
- **3-seed validation 结论**：把 `DFR-07/08/09` 作为 bounded-gating floor=0.05 的 formal multiseed 结果来看，mean `val_acc=0.9078014184397163`、mean `val_auc=0.9587878787878786`、mean `val_f1=0.8977644123846913`，`val_acc` population std 为 `0.0328851719698429`。相较 retained `L3-no-mixer` multiseed `0.9148936170212766 / 0.9628787878787879 / 0.9046651244767570` 与 `0.0150448251316287`，bounded-gating 在均值与稳定性上都更差。
- **当前判断**：这条线的最终结论是 **机制正向、主线负向**。bounded-gating 确实能缓解 `axial-only` collapse，并在部分 seed 上带来收益；但它当前更像高方差修复，而不是足够稳的新主线。结合 `DFR-06` telemetry，它把问题从“axial 独占”修到了“axial+coronal 共治”，但 `sagittal` 仍然基本贴着 `floor`，即典型的 `sagittal starvation`。
- **推荐动作**：后续如果还沿 decision-only 主线推进，不应继续做泛化意义上的 floor 扫描，而应围绕“如何让 `sagittal` 真正进入 late-fusion 决策”设计更窄的修复，并继续保持不引入 feature fusion。

## 2026-04-22：Decision-Fusion Repair（DFR-11 per-view logit temperature + DFR-12 teacher-guided gate，formal，node20）

> **实验说明**
> - 这是在人类确认“先做 per-view calibration，再做 gate supervision”之后补的两条 matched formal。两条都继续固定 `256x8 ResNeXt + decision fusion + L3-no-mixer`，不引入 feature fusion，不改最终 `view_logits -> fusion_weights -> fused logits` 语义。
> - 两条实验共用代码 commit `a8994ea`，都在 `node20` 用单卡 `V100q 32GB` 完成：`DFR-11` 对应 job `433864`，只打开 `ANKLE_DECISION_ENABLE_VIEW_LOGIT_TEMPERATURE=1`；`DFR-12` 对应 job `433863`，在相同 calibration 路径上再加 `ANKLE_DECISION_GATE_TEACHER_BLEND=0.30` 与 `ANKLE_DECISION_GATE_TEACHER_TEMPERATURE=1.0`。
> - 运行使用隔离 formal config [configs/cmp_resnext_decision_256x8_dfr11_view_logit_temperature_formal_s42.yaml](/dataset/HH/ankle-ct/configs/cmp_resnext_decision_256x8_dfr11_view_logit_temperature_formal_s42.yaml) 与 [configs/cmp_resnext_decision_256x8_dfr12_teacher_guided_gate_formal_s42.yaml](/dataset/HH/ankle-ct/configs/cmp_resnext_decision_256x8_dfr12_teacher_guided_gate_formal_s42.yaml)，输出目录分别是 [runs/resnext_decision_256x8_mainline/dfr11_view_logit_temperature_formal_s42](/dataset/HH/ankle-ct/runs/resnext_decision_256x8_mainline/dfr11_view_logit_temperature_formal_s42) 与 [runs/resnext_decision_256x8_mainline/dfr12_teacher_guided_gate_formal_s42](/dataset/HH/ankle-ct/runs/resnext_decision_256x8_mainline/dfr12_teacher_guided_gate_formal_s42)。

- [x] **DFR-11-RESNEXT-DECISION-256X8-VIEW-LOGIT-TEMPERATURE-FORMAL-S42**：`runs/resnext_decision_256x8_mainline/dfr11_view_logit_temperature_formal_s42`（commit `a8994ea`）→ `val_acc=0.9148936170212766`, `val_auc=0.9745454545454546`, `val_f1=0.9111111111111111`, `peak_vram≈2.15 GiB`, `total_seconds≈838.6` → **discard**（相较当前 retained `L3-no-mixer` seed42 anchor `27b557c` 的 `0.9255319148936170 / 0.9777272727272727 / 0.9213483146067416`，accuracy / AUC / F1 分别回落 `0.0106382978723404 / 0.0031818181818181 / 0.0102372034956305`。）
- [x] **DFR-12-RESNEXT-DECISION-256X8-TEACHER-GUIDED-GATE-FORMAL-S42**：`runs/resnext_decision_256x8_mainline/dfr12_teacher_guided_gate_formal_s42`（commit `a8994ea`）→ `val_acc=0.9148936170212766`, `val_auc=0.9731818181818181`, `val_f1=0.9000000000000000`, `peak_vram≈2.15 GiB`, `total_seconds≈841.0` → **discard**（相较同一 anchor，accuracy / AUC / F1 分别回落 `0.0106382978723404 / 0.0045454545454546 / 0.0213483146067416`；在相同 `val_acc` 下也弱于 `DFR-11`。）
- **训练轨迹观察**：两条线都不是“跑不起来”，而是都在中后期一度冲到 `val_acc=0.9149`，随后明显回撤。`DFR-11` 的 best 出现在 `epoch 9/12/14` 一带，对应 `val_auc=0.9745`；`DFR-12` 的 best 出现在 `epoch 9`，随后波动更大，`epoch 13` 一度掉到 `val_acc=0.7021`。这说明校准线有一定正向信号，但 teacher-guided gate 在当前配方下没有带来额外稳定收益。
- **快速 checkpoint 观察**：直接读取 `DFR-11` best checkpoint 后，学到的 per-view temperature 约为 `0.998987 / 1.000632 / 1.001273`（`axial / coronal / sagittal`），几乎仍是恒等映射。这意味着当前 `DFR-11` 的提升并不是来自明显的全局 logit rescaling，更像是“加了一个低容量校准自由度后的轻微训练轨迹变化”。
- **当前判断**：`DFR-11/12` 都没能替换 `L3-no-mixer` seed42 anchor。更细一点说，`DFR-11` 可以记为 **promising discard**，因为它在同一 `val_acc` 下至少比 `DFR-12` 更稳、`val_auc / val_f1` 也更好；`DFR-12` 则基本否定了“在当前 decision-only 语义下，直接给 gate 混入 detached teacher weight”这条修复方向。
- **推荐动作**：如果继续沿这条主线推进，下一步优先做 **`DFR-13 = DFR-11 calibration + DFR-05 floor=0.05`** 的组合验证，而不是继续单独扩 teacher-guided gate。原因很直接：`DFR-11` 至少证明“保持 late fusion 语义不变时，加一点 calibration 自由度不会伤太多”；而 `floor=0.05` 已经有清晰的机制证据能缓解 axial 独占，把两者合起来比继续押 `DFR-12` 更合理。

## 2026-04-22：Decision-Fusion Follow-up（DFR-13/14/15 stagewise gate-only fine-tuning，formal，node20）

> **实验说明**
> - 这是在人类确认“继续按一般多视角论文步骤推进”之后补的三条 matched formal，但仍严格保持 `decision fusion only`。目标不是再引入新的 feature 交互，而是测试：在当前最强 `L3-no-mixer` checkpoint 已经给定的前提下，只解冻 late-fusion gate 相关参数，是否能把权重从 axial dominance 进一步修到更合理的 routing。
> - 三条实验基于新增脚本 `scripts/train_decision_stagewise.py`。`DFR-13` 使用 commit `747c1c9`，从 `runs/resnext_decision_256x8_matrix/formal/l3_no_mixer/s42/best.pt` 初始化，并且只训练 `confidence_heads` 与 `confidence_calibrator`；`DFR-14/15` 在 commit `06fe604` 上分别叠加 `DFR-03` 风格的 train-time robustness 和 `DFR-05` 风格的 `floor=0.05`，但仍保持“冻结 experts、只调 gate”的 stagewise 语义。
> - 三条作业都在 `node20` 上用单卡 `V100q 32GB` 完成：`DFR-13`=`435192`、`DFR-14`=`435198`、`DFR-15`=`435199`。三条都会先对 init checkpoint 做 `epoch 0` 基线评估，再进入最多 `4` 个 fine-tune epoch；因此如果后续训练只会变差，best checkpoint 会停在初始化点。

- [x] **DFR-13-RESNEXT-DECISION-256X8-STAGEWISE-GATE-FORMAL-S42**：`runs/resnext_decision_256x8_mainline/dfr13_stagewise_gate_formal_s42`（commit `747c1c9`）→ `val_acc=0.9255319148936170`, `val_auc=0.9777272727272727`, `val_f1=0.9213483146067416`, `peak_vram≈1.22 GiB`, `total_seconds≈256.9` → **discard**（best 完全等于 init checkpoint；既没有超过当前 retained `L3-no-mixer` seed42 anchor，也额外引入了 stagewise fine-tune 路径。）
- [x] **DFR-14-RESNEXT-DECISION-256X8-STAGEWISE-GATE-ROBUSTNESS-FORMAL-S42**：`runs/resnext_decision_256x8_mainline/dfr14_stagewise_gate_robustness_formal_s42`（commit `06fe604`）→ `val_acc=0.9255319148936170`, `val_auc=0.9777272727272727`, `val_f1=0.9213483146067416`, `peak_vram≈1.26 GiB`, `total_seconds≈262.7` → **discard**（同样 best 停在 init checkpoint；把 robustness 叠到“只调 gate”的 regime 里并没有产生任何超越 init 的新收益。）
- [x] **DFR-15-RESNEXT-DECISION-256X8-STAGEWISE-GATE-FLOOR05-FORMAL-S42**：`runs/resnext_decision_256x8_mainline/dfr15_stagewise_gate_floor05_formal_s42`（commit `06fe604`）→ `val_acc=0.9255319148936170`, `val_auc=0.9777272727272727`, `val_f1=0.9213483146067416`, `peak_vram≈1.22 GiB`, `total_seconds≈261.8` → **discard**（即便把 `floor=0.05` 搬进 stagewise gate-only setting，best 也仍然只是 init checkpoint，本身没有新增主线价值。）
- **训练轨迹观察**：三条线的 pattern 完全一致，都是 `epoch 0` 最好，后面一训练就开始退化并在 `epoch 4` 触发 early stop。最终验证准确率分别回落到 `DFR-13=0.8297872340425532`、`DFR-14=0.7446808510638298`、`DFR-15=0.8191489361702128`。这说明当前 `L3-no-mixer` checkpoint 里的 gate 已经处在一个局部稳定点，单独继续推 gate，并不能从既有 biased experts 里再榨出额外多视角收益。
- **当前判断**：这三条结果基本否定了“只修 gate 就能修好单视角依赖”这条线。更准确地说，当前瓶颈已经不再是 gate 参数没调够，而是 experts 本身就带着明显的 axial bias；在这种前提下，无论是 stagewise、robustness 还是 bounded-gating，单独作用在 gate 上都只能维持初始化点，或者训练后更差。
- **推荐动作**：后续主线应改成先做 **`single-view expert strengthening`**，也就是先把 `axial / coronal / sagittal` 三个 masked single-view expert 单独训练出来，确认弱视角的 standalone ceiling 是否能显著高于当前 FWR-control 里的“从多视角 checkpoint 裁出来的单视角表现”；只有在这一步确认 `coronal/sagittal` 具备可用专家能力后，再回到真正的两阶段 `decision fusion`。

## 2026-04-22：Decision-Fusion Follow-up（DFR-16/17/18 single-view expert strengthening，formal，node20+node19）

> **实验说明**
> - 这是沿着“先修弱视角 experts，再修 late-fusion routing”这条主线补的第一步。仍然保持 `decision fusion only`，但通过新增 `ANKLE_DECISION_FORCE_ACTIVE_VIEW_MASK`，在标准 `train.py` 语义下强制只保留一个视角参与 late fusion，从而把当前 `256x8 ResNeXt + L3-no-mixer` scaffold 退化成真正的单视角 expert 训练。
> - 代码落点是 commit `d690119`（在 `src/model.py` 里新增固定 active-view mask）与提交修正 commit `80e0557`（修复 Slurm `--export` 逗号会截断 mask 的 launch bug，改为 `ACTIVE_VIEW_INDEX` 在脚本内展开）。三条 formal 共用通用脚本 `scripts/slurm_resnext_decision_256x8_single_view_masked.sbatch`，但分别指向独立 config：`DFR-16=axial`、`DFR-17=coronal`、`DFR-18=sagittal`。
> - 首次提交 `435477/435478/435479` 因 `--export` 逗号截断在启动前即失败，不计入实验结果。修复后有效作业是：`DFR-16=435481`（`node20`）、`DFR-17=435486`（重定向到 `node19`）、`DFR-18=435482`（`node20`）；三条都在单卡 `V100q 32GB` 上正常完赛。

- [x] **DFR-16-RESNEXT-DECISION-256X8-SINGLE-VIEW-AXIAL-FORMAL-S42**：`runs/resnext_decision_256x8_mainline/dfr16_single_view_axial_formal_s42`（commit `80e0557`）→ `val_acc=0.9255319148936170`, `val_auc=0.9781818181818183`, `val_f1=0.9230769230769231`, `peak_vram≈2.15 GiB`, `total_seconds≈838.4` → **keep**（相较 `FWR-01` 里“从多视角 winner 裁出来的 `single_view_axial` control”`0.9255319148936170 / 0.9750000000000000 / 0.9213483146067416`，本轮 direct single-view training 在不降主指标的前提下，进一步抬高了 `val_auc / val_f1`。）
- [x] **DFR-17-RESNEXT-DECISION-256X8-SINGLE-VIEW-CORONAL-FORMAL-S42**：`runs/resnext_decision_256x8_mainline/dfr17_single_view_coronal_formal_s42`（commit `80e0557`）→ `val_acc=0.8936170212765957`, `val_auc=0.9372727272727274`, `val_f1=0.8863636363636364`, `peak_vram≈2.15 GiB`, `total_seconds≈831.9` → **keep**（相较 `FWR-01` 里同 seed 的 `single_view_coronal` control `0.4680851063829787 / 0.7013636363636363 / 0.1071428571428571`，这是一次大幅恢复，说明 coronal 不是天然无效视角，而是此前在 joint late-fusion 里没有被训练成像样 expert。）
- [x] **DFR-18-RESNEXT-DECISION-256X8-SINGLE-VIEW-SAGITTAL-FORMAL-S42**：`runs/resnext_decision_256x8_mainline/dfr18_single_view_sagittal_formal_s42`（commit `80e0557`）→ `val_acc=0.8617021276595744`, `val_auc=0.9040909090909091`, `val_f1=0.8354430379746836`, `peak_vram≈2.15 GiB`, `total_seconds≈840.9` → **keep**（相较 `FWR-01` 里同 seed 的 `single_view_sagittal` control `0.4680851063829787 / 0.6295454545454545 / 0.1071428571428571`，也出现了显著恢复，说明 sagittal 的问题更像 starvation，而不是“这个视角没有信息”。）
- **训练轨迹观察**：三条 single-view experts 都不是靠“偶然 spike”赢的。`axial` 最终站上 `0.9255 / 0.9782`；`coronal` 在 `epoch 13` 达到 `0.8936 / 0.9373`；`sagittal` 最终在 `epoch 15` 回到 `0.8617 / 0.9041`。虽然 `coronal/sagittal` 仍然落后于 axial，但它们与 `FWR-01` control 相比的提升幅度已经足够说明：过去的问题不是视角本身没判别力，而是多视角 joint 训练把弱视角饿死了。
- **当前判断**：这一轮是关键正结果。它直接把“是否值得继续修 single-view dependence”从不确定变成了明确的 **值得**。因为现在已经有证据表明：`coronal` 和 `sagittal` 在 direct expert training 下都能学成有信息量的独立专家，而当前 late-fusion 主线没有把这两个视角的潜力转化出来。
- **推荐动作**：下一步不再做新的单视角扫参，而是进入真正的 **two-stage decision fusion**：把 `DFR-16/17/18` 的三个 expert checkpoint 按视角回填到同一个 3-view decision model 里，先冻结 per-view experts、只训练 `confidence_heads / confidence_calibrator` 做 gate warmup，再小学习率 joint finetune。只有这一步能回答“当三个视角都先被训成像样 expert 后，late fusion 还能不能摆脱 axial dominance”。

## 2026-04-22：Decision-Fusion Follow-up（DFR-19 merged single-view experts + gate warmup，formal，node20）

> **实验说明**
> - 这是 `DFR-16/17/18` 之后的第一条真正 two-stage decision-fusion 验证。先在 commit `77cb402` 上新增 `scripts/merge_single_view_experts.py`，用 retained `L3-no-mixer` seed42 checkpoint 作为 gate / 非视角专属模块底座，再把 `DFR-16/17/18` 各自训练好的 `view_encoders.{i}` 与 `view_classifiers.{i}` 回填到对应视角，生成 merged init checkpoint `runs/resnext_decision_256x8_mainline/artifacts/dfr19_merged_single_view_experts_s42.pt`。
> - 随后复用 `scripts/train_decision_stagewise.py`，只解冻 `confidence_heads` 与 `confidence_calibrator`，做一轮 gate warmup formal；对应 config 是 `configs/cmp_resnext_decision_256x8_dfr19_merged_experts_gate_warmup_formal_s42.yaml`，Slurm job `435551` 跑在 `node20` 的单卡 `V100q 32GB` 上。
> - merge 脚本成功从三个 single-view checkpoints 各拷回 `326` 个视角专属参数键，说明 expert transplant 本身没有漏拷主要分支。

- [x] **DFR-19-RESNEXT-DECISION-256X8-MERGED-EXPERTS-GATE-WARMUP-FORMAL-S42**：`runs/resnext_decision_256x8_mainline/dfr19_merged_experts_gate_warmup_formal_s42`（commit `77cb402`）→ `init_val=0.9042553191489362 / 0.9740909090909091 / 0.9010989010989011`，`best_val=0.9042553191489362 / 0.9740909090909091 / 0.9010989010989011`, `peak_vram≈1.22 GiB`, `total_seconds≈248.2` → **discard**（merged init 本身就低于当前 retained `L3-no-mixer` seed42 anchor `0.9255319148936170 / 0.9777272727272727 / 0.9213483146067416`，而 gate warmup 后也没有任何 epoch 超过 init。）
- **训练轨迹观察**：这条线的 pattern 很明确：`epoch 0` 的 merged init 是全程最好点，后面 gate warmup 只会更差。验证准确率依次走成 `epoch1=0.8617`、`epoch2=0.8617`、`epoch3=0.8723`、`epoch4=0.8830`，随后因连续 `4` 轮无提升触发 early stop。也就是说，把三条 expert 直接拼回一个 3-view model 后，单独重训 gate 并不能把它们重新协调起来。
- **当前判断**：`DFR-19` 否定了“merged experts + gate-only warmup”这条最保守的 two-stage 变体。问题不是 merge 失败，而是 merged experts 的相对 logit / feature geometry 与旧 gate 不匹配，且这种不匹配不能只靠 `confidence_heads / confidence_calibrator` 重新拟合。
- **推荐动作**：下一步不要再重复 gate-only warmup；应该直接试 **merged-expert joint finetune**，至少把 `view_classifiers` 一并解冻，必要时连 `view_encoders` 也低学习率共同调整。当前更合理的问题是“如何把三个已经各自变强的 experts 重新对齐到一个共同 late-fusion 空间”，而不是继续把责任全部压给 gate。

## 2026-04-22：Decision-Fusion Follow-up（DFR-20 merged single-view experts + joint finetune，formal，node20）

> **实验说明**
> - 这是在 `DFR-19` 否掉 gate-only warmup 之后补的更激进 two-stage 变体：继续使用同一个 merged init checkpoint `runs/resnext_decision_256x8_mainline/artifacts/dfr19_merged_single_view_experts_s42.pt`，但不再只解冻 gate，而是把 `view_encoders + view_classifiers + confidence_heads + confidence_calibrator` 全部放开，用较小学习率 `5e-5` 做 joint finetune。
> - 对应 config 是 `configs/cmp_resnext_decision_256x8_dfr20_merged_experts_joint_finetune_formal_s42.yaml`，Slurm job `435554` 跑在 `node20` 的单卡 `V100q 32GB` 上；代码落点是 commit `86df7d9`。

- [x] **DFR-20-RESNEXT-DECISION-256X8-MERGED-EXPERTS-JOINT-FINETUNE-FORMAL-S42**：`runs/resnext_decision_256x8_mainline/dfr20_merged_experts_joint_finetune_formal_s42`（commit `86df7d9`）→ `init_val=0.9042553191489362 / 0.9740909090909091 / 0.9010989010989011`，`best_val=0.9042553191489362 / 0.9740909090909091 / 0.9010989010989011`, `peak_vram≈20.78 GiB`, `total_seconds≈283.2` → **discard**（joint finetune 同样没有超过 merged init，更没有回到当前 retained `L3-no-mixer` seed42 anchor `0.9255319148936170 / 0.9777272727272727 / 0.9213483146067416`。）
- **训练轨迹观察**：这条线比 `DFR-19` 更激进，但趋势仍然是否定的。验证准确率走成 `epoch1=0.8298`、`epoch2=0.7660`、`epoch3=0.8936`，直到 `epoch4` 仍未超过 `init=0.9043`，最终同样 early stop。说明把三条 stronger experts 生硬拼回一个 3-view model 后，不仅 gate 不好重对齐，连 full/joint finetune 也会在短程内先明显破坏原有 calibration。
- **当前判断**：`DFR-20` 否定了“直接 merge 后整体一起微调”这条一步到位方案。到这里更清楚了：当前 merged-expert 方案缺的不是更大训练自由度，而是一个更平滑的对齐过渡。
- **推荐动作**：下一步应转到 **中间态 realignment**，也就是保持 `view_encoders` 固定，只解冻 `view_classifiers + confidence_heads + confidence_calibrator`，先把每个视角 head 与 late-fusion 空间重新对齐；如果这一层还不行，再考虑更细的分阶段 unfreeze，而不是继续做 full-joint。

## 2026-04-22：Decision-Fusion Follow-up（DFR-21 merged single-view experts + classifier/gate finetune，formal，node20）

> **实验说明**
> - 这是在 `DFR-20` 之后补的中间态对齐实验：继续使用同一个 merged init checkpoint `runs/resnext_decision_256x8_mainline/artifacts/dfr19_merged_single_view_experts_s42.pt`，但不再放开 `view_encoders`，只解冻 `view_classifiers + confidence_heads + confidence_calibrator`，让每个视角 head 与 late-fusion routing 先重新对齐。
> - 对应 config 是 `configs/cmp_resnext_decision_256x8_dfr21_merged_experts_classifier_gate_finetune_formal_s42.yaml`，Slurm job `435617` 跑在 `node20` 的单卡 `V100q 32GB` 上；代码落点是 commit `156c6a1`。

- [x] **DFR-21-RESNEXT-DECISION-256X8-MERGED-EXPERTS-CLASSIFIER-GATE-FINETUNE-FORMAL-S42**：`runs/resnext_decision_256x8_mainline/dfr21_merged_experts_classifier_gate_finetune_formal_s42`（commit `156c6a1`）→ `init_val=0.9042553191489362 / 0.9740909090909091 / 0.9010989010989011`，`best_val=0.9042553191489362 / 0.9740909090909091 / 0.9010989010989011`, `peak_vram≈1.22 GiB`, `total_seconds≈245.0` → **discard**（与 `DFR-19/20` 一样，没有任何 epoch 超过 merged init，更没有逼近当前 retained `L3-no-mixer` seed42 anchor `0.9255319148936170 / 0.9777272727272727 / 0.9213483146067416`。）
- **训练轨迹观察**：中间态对齐也没有把 merged experts 救回来。验证准确率依次走成 `epoch1=0.8511`、`epoch2=0.8936`、`epoch3=0.8723`、`epoch4=0.8617`，最终 early stop。也就是说，问题并不只是“encoder 解冻过多”或“joint finetune 太猛”；即便只允许 classifier+gate 重新拟合，merged init 依然没有恢复到 `0.9043` 以上。
- **当前判断**：到这里，`DFR-19/20/21` 三条 together 已经把“merge 后再继续学”这条线压得很窄了。共同结论是：直接把三条 stronger single-view experts 拼回多视角模型，会形成一个初始 `0.9043` 的 mixed geometry，而无论只调 gate、调 classifier+gate，还是 full-joint，都没法在短程内把它拉回当前主线 anchor。
- **推荐动作**：下一步不该继续在 merged checkpoint 上盲调解冻范围，而是先回答一个更基本的问题：**merged experts 在 equal-weight 下是不是本来就比 old learned gate 更好**。如果 `equal-weight on merged experts` 明显高于 `0.9043`，那说明专家本身没问题，问题主要在 learned routing；反之，如果 equal-weight 也不行，那 merge 方案本身就需要重写。 

## 2026-04-23：Decision-Fusion Follow-up（DFR-22/23/24 merged experts + equal-weight control / realignment，formal，node15→node20）

> **实验说明**
> - 这是沿着 `DFR-21` 的推荐动作补的三条 equal-weight follow-up：`DFR-22` 先回答“merged experts 在纯 `equal-weight` 下是不是本来就比旧 learned routing 更好”；`DFR-23` 在 equal-weight 固定下只解冻 `view_classifiers` 做轻量 realignment；`DFR-24` 再把 `view_encoders + view_classifiers` 一并放开，测试更激进的 realignment。
> - 三条原始作业最初并行落在同一台 `RTXA6Kq / node15`：`DFR-22=435848`、`DFR-23=435849`、`DFR-24=435850/435876/435899`。它们共享同一数据集并都使用 `num_workers=12`，结果在 `init eval` 和少数训练 batch 上出现了极端 DataLoader / I/O stall：例如 `DFR-23` 的首个验证 batch 耗时约 `595s`，第 `13/16` 个验证 batch 再卡约 `1030s`；`DFR-22` 也出现了同型的 `812s / 1040s` 级停顿。`DFR-23` 因此在 `01:30:00` walltime 下于 `epoch 2` 刚开始时被 Slurm 超时取消，而不是模型本身算不动。
> - 为确认这不是模型计算瓶颈，而是并发资源噪声，后续把 `DFR-23` 的 saved `best.pt` 单独放到空闲的 `node20 / V100q` 上做 isolated formalization：当前 HEAD 的 eval-only job `435971` 与原始 commit `318f7f0` worktree 的 legacy eval-only job `435985` 都在 `1~2` 分钟内完成，而且两者对同一个 checkpoint 的重评估结果完全一致，都是 `0.9361702127659575 / 0.9740909090909091 / 0.9318181818181818`。因此 timeout 的根因可以确定为**同节点三作业并发带来的 DataLoader/I/O 拖慢 + 原始 `01:30:00` walltime 过紧**；这不是模型语义问题。
> - `DFR-24` 还额外暴露了独立的显存问题：在 `RTX 6000 Ada 49GB` 上，`batch_size=6` 与 `4` 都会 OOM；降到 `batch_size=2`（commit `23c7a4d`）后才完整跑完。这个 OOM 与 `DFR-23` 的 timeout 是两件事。

- [x] **DFR-22-RESNEXT-DECISION-256X8-MERGED-EXPERTS-EQUAL-WEIGHT-FORMAL-S42**：`runs/resnext_decision_256x8_mainline/dfr22_merged_experts_equal_weight_formal_s42`（commit `318f7f0`，job `435848`）→ `init_val=0.9042553191489362 / 0.9686363636363636 / 0.8965517241379310`，`best_val=0.9042553191489362 / 0.9686363636363636 / 0.8965517241379310`, `peak_vram≈1.21 GiB`, `total_seconds≈4601.3` → **discard**（pure equal-weight control 完全停在 merged init，本身没有回答出“专家很好、只是 learned routing 错了”这个乐观结论。）
- [x] **DFR-23-RESNEXT-DECISION-256X8-MERGED-EXPERTS-EQUAL-WEIGHT-CLASSIFIER-FINETUNE-FORMAL-S42**：原始 fine-tune job `435849`（commit `318f7f0`）在 `node15` 因 walltime 超时，没有写出 `summary.json`；随后对其 saved `best.pt` 做 isolated formalization（current-code job `435971` + legacy-code job `435985`，两次重评估完全一致）→ `best_val=0.9361702127659575 / 0.9740909090909091 / 0.9318181818181818`, `peak_vram≈1.21 GiB` → **keep**（相较 `DFR-22`/`DFR-24` 的 `0.9043` 平台，classifier-only realignment 确实把 merged experts 拉上来了；但正式值应以这次可复现的 `0.9362 / 0.9741 / 0.9318` 为准，不再使用原 timeout 日志里那条未复现的 `0.9468 / 0.9736`。）
- [x] **DFR-24-RESNEXT-DECISION-256X8-MERGED-EXPERTS-EQUAL-WEIGHT-JOINT-FINETUNE-FORMAL-S42**：`runs/resnext_decision_256x8_mainline/dfr24_merged_experts_equal_weight_joint_finetune_formal_s42`（最终完成版本 commit `23c7a4d`，job `435899`；此前 `435850/435876` 分别在 `batch_size=6/4` OOM）→ `init_val=0.9042553191489362 / 0.9686363636363636 / 0.8965517241379310`，`best_val=0.9042553191489362 / 0.9686363636363636 / 0.8965517241379310`, `peak_vram≈7.68 GiB`, `total_seconds≈5733.7` → **discard**（更大范围的 joint realignment 也没把 merged experts 拉出 `0.9043` 平台，说明这条线的问题不是“解冻得还不够多”。）
- **当前判断**：这组三条里，真正成立的只有 `DFR-23`。结论结构很清楚：`equal-weight` 本身不能救 merged experts，full-joint 也不能；但**在 equal-weight 固定下，只做 classifier-level realignment 是有效的**。同时，这轮还额外暴露了一个执行层事实：如果继续在同节点并行跑多条 stagewise line，`num_workers=12` 会把 walltime 风险放大，后续同类作业应避免再复用这种同节点三并发模式。
- **推荐动作**：如果未来还要回到这条 merged-expert family，优先沿 `DFR-23` 的“equal-weight + classifier-only realignment”展开，而不是再重复 `DFR-22` 或 `DFR-24`。执行层面则应默认把这类 stagewise/eval-heavy job 放到**隔离节点**，并把 walltime 设到不少于 `03:00:00`。

## 2026-04-22：Fusion-Weight Rationality（FWR-01 single-view / leave-one-view-out matched control，adaptive main-study，node19）

> **独立 side campaign 说明**
> - 这是人类限定 `3` 轮外层 autoresearch 的第 `1/3` 轮，只做 `FWR-01`：围绕当前 strongest learned branch `256x8 ResNeXt decision fusion + L3-no-mixer`，补一轮 matched `single-view` / `leave-one-view-out` 控制，并把分析结果落到本轮 run dir 的 `view_ablation_summary.json`。
> - 首次 fresh study `runs/optuna_main_autoloop/iter_0001_20260422_020107` 在 commit `51967c9` 上暴露了 workflow code bug：`scripts/optuna_workflow.py` 的并行 worker 提前把 `WAITING` template trial 算进停止条件，导致 `n_trials=1` 且 `max-workers=2` 时 study 直接 `0 trial completed`。该空跑不计入实验结果；随后在 commit `c3ba210` 修复后，改用新的 fresh search-config copy `autoresearch_logs/generated_search_configs/optuna_main_search_iter_0001_20260422_020718.yaml` 与新的 study_root `runs/optuna_main_autoloop/iter_0001_20260422_020718` 重跑。
> - 有效运行环境是 `node19`，adaptive policy 可见 `GPU 1/2` 两张空闲 `Tesla V100-PCIE-32GB`；fresh study 最终只执行 1 个 template trial，trial config 继续保持 canonical `256x8` 几何与标量：`freeze_layers=3`、`lr=1e-4`、`weight_decay=5e-4`、`dropout=0.25`、`gradient_clip_norm=2.0`，唯一 runtime path 是 `ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER=1`。

- [x] **FWR-01-RESNEXT-DECISION-256X8-L3-NOMIXER-MAIN-S42-CONTROL**：fresh adaptive `main-study` `runs/optuna_main_autoloop/iter_0001_20260422_020718` 的唯一 completed trial（trial `0`，commit `c3ba210`，best checkpoint 路径 `runs/optuna_main_autoloop/iter_0001_20260422_020718/trials/trial_0000/run/best.pt`）→ `val_acc=0.9255319148936170`, `val_auc=0.9750000000000000`, `val_f1=0.9213483146067416`, `peak_vram≈2.15 GiB`, `total_seconds≈818.6`；随后用同一 trial config 运行 `scripts/analyze_view_controls.py`，产出 `runs/optuna_main_autoloop/iter_0001_20260422_020718/trials/trial_0000/run/view_ablation_summary.json` → **discard**（相较历史 best `L3-no-mixer` seed42 keep `27b557c` 的 `0.9255319148936170 / 0.9777272727272727`，这次只是在主指标上打平，且 `val_auc` 回落 `0.0027272727272727`，因此不能作为新的 retained model checkpoint。）
- **matched control 结果**：`full_fusion`、`single_view_axial`、`leave_out_coronal`、`leave_out_sagittal` 四条路径在验证集上给出完全相同的 `0.9255319148936170 val_acc / 0.9750000000000000 val_auc / 0.9213483146067416 val_f1`；相反，`single_view_coronal`、`single_view_sagittal`、`leave_out_axial` 全部掉到 `0.4680851063829787 val_acc`，其中 `leave_out_axial` 的 `val_auc=0.7109090909090909`，单独 coronal / sagittal 则只有 `0.7013636363636363 / 0.6295454545454545`。
- **翻转统计**：相较 full fusion，`single_view_axial`、`leave_out_coronal`、`leave_out_sagittal` 的 changed-prediction count 都是 `0`；而 `single_view_coronal`、`single_view_sagittal`、`leave_out_axial` 都有 `49` 个预测翻转，其中 `46` 个是从 full-fusion 的正确样本退化成错误，只带来 `3` 个纠错样本。这说明当前 winner 的有效信息几乎被 axial 一条视角吃满，coronal / sagittal 在这个 seed 上没有表现出可见的主指标增益。
- **当前判断**：`FWR-01` 已经足够回答第一轮问题。对这次 `L3-no-mixer seed42` winner 而言，learned fusion 的收益更接近 **“贴着 best single view（axial）工作”**，而不是“通过 learned weights 稳定压低拖后腿视角后仍保留多视角净收益”。也就是说，在当前 strongest branch 的至少这个 matched seed 上，fusion 机制还没有给出“超越 best single view”的额外证据。
- **推荐动作**：下一轮按固定顺序转入 **`FWR-02 fusion-weight telemetry`**。重点不再是继续重训 `L6` 标量修复，而是补样本级 `fusion_weights` 记录与最小一致性统计，先回答“最高权重是否真的落在最有证据的视角上”；如果 telemetry 继续显示 axial 几乎恒定主导，再决定 `FWR-03` 的扰动设计优先对 axial 还是弱视角下手。

## 2026-04-22：Fusion-Weight Rationality（FWR-02 fusion-weight telemetry，existing main-study checkpoint telemetry，node19）

> **独立 side campaign 说明**
> - 这是人类限定 `3` 轮外层 autoresearch 的第 `2/3` 轮，只做 `FWR-02`：不新开 study、不重训，而是在 commit `478f371` 上新增 `scripts/analyze_fusion_weights.py`，对上一轮已完成的 `L3-no-mixer` adaptive `main-study` `runs/optuna_main_autoloop/iter_0001_20260422_020718/trials/trial_0000` 做离线 telemetry。
> - 分析输入继续使用上一轮 trial config `runs/optuna_main_autoloop/iter_0001_20260422_020718/trials/trial_0000/config.yaml` 与 checkpoint `.../run/best.pt`；runtime path 保持 `ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER=1`。本轮只额外生成 `runs/optuna_main_autoloop/iter_0001_20260422_020718/trials/trial_0000/run/fusion_weight_analysis.json`，不复用 `432763` 的任何运行目录，也不重新触发 `L6`。
> - 运行环境是 `node19`；因 host `GPU 0` 当时已有占用（`2940 MiB / 25% util`），实际 telemetry 通过 `CUDA_VISIBLE_DEVICES=1` 在空闲 `V100 32GB` 上完成，仅做离线前向推理。

- [x] **FWR-02-RESNEXT-DECISION-256X8-L3-NOMIXER-MAIN-S42-TELEMETRY**：commit `478f371` 用新增脚本分析既有 adaptive `main-study` checkpoint `runs/optuna_main_autoloop/iter_0001_20260422_020718/trials/trial_0000/run/best.pt`（训练指标仍是 `val_acc=0.9255319148936170`, `val_auc=0.9750000000000000`, `val_f1=0.9213483146067416`, `peak_vram≈2.15 GiB`）并产出 `fusion_weight_analysis.json` → **discard**（本轮没有产生新的更优 checkpoint；它回答的是机制问题而不是主指标提升，相较历史 best `L3-no-mixer` seed42 keep `27b557c` 仍然保持 `val_acc` 打平但 `val_auc` 落后 `0.0027272727272727`。）
- **权重塌缩结论**：sample-level telemetry 显示 learned fusion 几乎是一个硬性的 axial selector。三视角平均权重分别为 `axial=0.997426`、`coronal=0.001593`、`sagittal=0.000981`；`axial` 在 `94/94` 个验证样本上都是 top-weight view，且最小 axial 权重仍有 `0.992064`。这说明当前 strongest learned branch 基本没有发生样本级的权重迁移。
- **top-weight hit rate / margin 对齐**：`top-weight hit rate` 对 `pred_margin` 是 `81/94 = 0.861702`，对 `true_margin` 是 `83/94 = 0.882979`。`top_pred_margin` 分布是 `axial=81 / coronal=13 / sagittal=0`，而 `top_true_margin` 分布是 `axial=83 / coronal=7 / sagittal=4`；也就是说，即使有 `11` 个样本的最高 `true_margin` 已经落到非 axial 视角，fusion 仍然把最高权重固定给 axial。
- **correctness 对齐的真实来源**：`top-weight correct rate = 0.925532`，`mixed-correctness` 子集里的 `top-weight correct rate = 46/49 = 0.938776`，看起来很高；但拆开看，`49` 个 mixed-correctness 样本里有 `46` 个是 **只有 axial 一条视角预测正确**。因此这条“高一致性”更多来自 axial 本来就是 best single view，而不是 learned weights 在不同样本之间做了有意义的 evidence routing。按扁平样本-视角对统计，`weight vs pred_margin / true_margin / correctness` 的 Pearson 分别是 `0.7063 / 0.6907 / 0.4451`，`correct-view mean weight=0.4968` 也高于 `incorrect-view mean weight=0.0660`；但这些 aggregate 相关性依然被全局 axial prior 主导。
- **最关键的反例**：共有 `11` 个样本出现“非 axial 视角拥有最高 true-margin”，其中 `7` 个最终 full fusion 仍然预测错误。更尖锐的是，有 `3` 个阳性样本上 **axial 错、coronal+sagittal 对**，且 `coronal` 还是最高 true-margin view，但 fusion weight 依旧维持在 `>0.998 axial`，导致 full fusion 跟着 axial 一起错。这说明当前 learned fusion 没有在最需要的时候把权重迁到更有证据的非 axial 分支。
- **当前判断**：`FWR-02` 已经足够回答第二轮问题。当前 `256x8 + L3-no-mixer` winner 的 learned fusion 更像是 **近乎静态的 axial hard-selection**，不是“按样本根据证据重新分配权重”的自适应机制。它在 aggregate 上看起来“top-weight 和 correctness 对齐”，主要因为 axial 本来就最强，而不是因为权重学会了真正的跨视角迁移。
- **推荐动作**：下一轮按固定顺序转入 **`FWR-03 perturbation-based weight migration`**。既然 telemetry 已经证明权重几乎不离开 axial，最有信息量的设计就是优先对 axial 做可控退化（首选低容量、可复现的 blur / noise / slice-drop 之一），直接测试在 axial 质量下降时，权重是否会从 `~0.997` 真正迁向 coronal / sagittal。

## 2026-04-22：Fusion-Weight Rationality（FWR-03 perturbation-based weight migration，existing main-study checkpoint perturbation analysis，node19）

> **独立 side campaign 说明**
> - 这是人类限定 `3` 轮外层 autoresearch 的第 `3/3` 轮，也是本 bounded side campaign 的收口轮次：不新开 study、不重训，而是在 commit `27219b2` 上新增 `scripts/analyze_fusion_perturbations.py`，对同一 `L3-no-mixer` adaptive `main-study` `runs/optuna_main_autoloop/iter_0001_20260422_020718/trials/trial_0000` checkpoint 做离线扰动分析。
> - 分析输入继续使用 `runs/optuna_main_autoloop/iter_0001_20260422_020718/trials/trial_0000/config.yaml` 与 `.../run/best.pt`；runtime path 仍是 `ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER=1`。本轮只额外生成 `runs/optuna_main_autoloop/iter_0001_20260422_020718/trials/trial_0000/run/perturbation_summary.json`，不复用 `432763` 的任何运行目录，也不重新触发 `L6`。
> - 扰动设计固定为 **axial 单视角 deterministic Gaussian blur**（`kernel=17`, `sigma=4.0`），运行环境是 `node19` 的 `CUDA_VISIBLE_DEVICES=0`。之所以选这条设计，是因为 `FWR-01/02` 已经表明 axial 是当前 winner 的绝对主导视角；最有信息量的问题不再是“谁最强”，而是“当 axial 变差时，权重会不会真的迁走”。

- [x] **FWR-03-RESNEXT-DECISION-256X8-L3-NOMIXER-MAIN-S42-PERTURB**：commit `27219b2` 用新增脚本分析既有 adaptive `main-study` checkpoint `runs/optuna_main_autoloop/iter_0001_20260422_020718/trials/trial_0000/run/best.pt`（训练指标仍是 `val_acc=0.9255319148936170`, `val_auc=0.9750000000000000`, `val_f1=0.9213483146067416`, `peak_vram≈2.15 GiB`）并产出 `perturbation_summary.json` → **discard**（本轮没有产生新的更优 checkpoint；它回答的是机制问题而不是主指标提升，相较历史 retained `L3-no-mixer` seed42 keep `27b557c` 仍然只是 `val_acc` 打平但 `val_auc` 落后 `0.0027272727272727`。）
- **扰动后的融合退化**：对 axial 做 blur 后，full-fusion 验证集指标直接降到 `val_acc=0.6170212765957447`, `val_auc=0.8404545454545455`, `val_f1=0.3333333333333333`，相较 baseline 分别回落 `0.3085106382978723 / 0.1345454545454545 / 0.5880149812734083`。共有 `37/94` 个样本发生预测翻转，其中 `33` 个是从 baseline 的正确样本退化成错误，只带来 `4` 个纠错样本。
- **最关键的权重迁移结论**：尽管 axial 的证据被明显打坏，fusion top-weight 仍然 **没有任何一次** 从 axial 迁走。扰动前后 `top-weight axial` 都是 `94/94`；三视角平均权重从 `axial/coronal/sagittal = 0.997426 / 0.001593 / 0.000981` 变成 `0.997567 / 0.001488 / 0.000945`。也就是说，当前 learned fusion 并不是“弱化了 axial 但仍不够多”，而是 **在显著退化场景下依旧维持硬性的 axial selector**。
- **证据退化与权重变化脱钩**：axial 本身的 per-view 指标也和 full fusion 一起大幅下滑，`axial val_acc / val_auc` 同样掉到 `0.617021 / 0.840000`，平均 `pred_margin / true_margin` 分别下降 `1.461722 / 2.363960`。但这些退化并没有触发权重迁移：虽然 `67/94` 个样本的 axial weight 有小幅下降，但平均变化只有 `+0.0001409`，而且 `66` 个 axial `pred_margin` 下降样本里有 `25` 个反而出现 axial weight **上升**；更尖锐的是，`33` 个从对变错的 regressed 样本里有 `18` 个也出现 axial weight 上升。
- **当前判断**：到这里，bounded `fusion-weight rationality` side campaign 已经足够回答问题。当前 `256x8 + L3-no-mixer` winner 的 learned fusion 不仅在静态 telemetry 上表现为 axial collapse，在显式的 axial degradation 下也依旧**不会把权重迁给更稳定的非 axial 分支**。这说明它更像“几乎固定的 axial hard-selection”，而不是具备样本级证据路由能力的 adaptive fusion。
- **推荐动作**：本 side campaign 到此完成 `3/3`，外层 autoresearch loop 应按人类要求退出这一支。若未来还要继续追问 learned fusion 的合理性，不要再重复 control / telemetry / perturbation 轮次；只有真正会改变 evidence routing 的**结构性修复**才值得继续，比如显式限制 axial prior、引入跨视角竞争正则或辅助迁移目标。当前常规主线则应把控制权交回外层 queue，并继续避免与独立运行中的 `432763` / `L6` 冲突。

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
> **待执行的模块贡献分析矩阵（当前进度：`L0/L1/L2/L3/L4/L5` formal 已完成；`temperature` 线已封口，后续转入 `L3-no-mixer` 的非-temp 标量修复）**
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
> - **主线叙事补充规则（2026-04-22）**：`temperature` 只保留为当前 `L5` 的一次性低容量校准收口，不再扩展为后续常规消融维度。也就是说，`L5-temp1p5 / L5-temp2p0` 跑完后，后续主线模块分析默认不再新增新的 `temp=*` 探针，除非出现明确机制证据表明必须重新检查 temperature 才能解释主指标变化。

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

## 2026-04-22：ResNeXt Decision 256x8 Module Contribution（L5 temperature closeout，formal multiseed，V100q node20）

> **独立 campaign 说明**
> - 这是在 `L3-no-mixer` 被确认成 strongest learned branch 之后，对 `temperature` 这条低容量校准线做的一次性收口，而不是新的长期 ablation family。按人类最新规则，`L5-temp1p5 / L5-temp2p0` 跑完后，后续主线默认不再新增新的 `temp=*` probe。
> - 有效 batch 是 `432737`：`V100q/node20`，`1 node / 3 GPU / 36 CPU / 120G / 4h`；job 内 `CUDA_VISIBLE_DEVICES=0,1,2`，前 3 个 `srun` step `432737.0-.2` 对应 `L5-temp1p5`，后 3 个 step `432737.3-.5` 对应 `L5-temp2p0`，六个 step 全部 `COMPLETED`。本轮运行代码状态保持在 commit `f84df46`。
> - 配方继续保持严格 matched：`backbone=resnext`、`fusion_type=decision`、`image_size=256`、`num_slices_per_view=8`、`trim_edge_slices=2`、`batch_size=6`、`num_workers=12`、`epochs=15`、`freeze_layers=3`、`lr=1e-4`、`weight_decay=5e-4`、`dropout=0.25`、`gradient_clip_norm=2.0`；共同 runtime path 是 `ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER=1`，唯一变量是 `ANKLE_LEARNED_FUSION_TEMPERATURE=1.5` 或 `2.0`。

- [x] **CMP-RESNEXT-DECISION-256X8-L5-TEMP1P5-FORMAL-S42**：`runs/resnext_decision_256x8_matrix/formal/l5_temp1p5/s42`（commit `f84df46`，`ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER=1`，`ANKLE_LEARNED_FUSION_TEMPERATURE=1.5`）→ `val_acc=0.9255319148936170`, `val_auc=0.9800000000000000`, `val_f1=0.9176470588235294`, `peak_vram≈2.15 GiB`, `total_seconds≈897.3` → **keep**（与 matched `L3-no-mixer` seed42 在 `val_acc` 上打平，同时把 AUC 再抬高 `0.0022727272727273`；但 F1 略低 `0.0037012557832122`，属于“更软的 ranking 修正”而不是主指标突破。）
- [x] **CMP-RESNEXT-DECISION-256X8-L5-TEMP1P5-FORMAL-S123**：`runs/resnext_decision_256x8_matrix/formal/l5_temp1p5/s123`（commit `f84df46`，`ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER=1`，`ANKLE_LEARNED_FUSION_TEMPERATURE=1.5`）→ `val_acc=0.8829787234042553`, `val_auc=0.9472727272727273`, `val_f1=0.8791208791208791`, `peak_vram≈2.15 GiB`, `total_seconds≈896.9` → **discard**（相较 matched `L3-no-mixer` seed123，AUC 虽提升 `0.0077272727272728`，但主指标 accuracy 回落 `0.0106382978723404`；说明 temperature=1.5 没有修复当前最关键的低点 seed。）
- [x] **CMP-RESNEXT-DECISION-256X8-L5-TEMP1P5-FORMAL-S456**：`runs/resnext_decision_256x8_matrix/formal/l5_temp1p5/s456`（commit `f84df46`，`ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER=1`，`ANKLE_LEARNED_FUSION_TEMPERATURE=1.5`）→ `val_acc=0.9361702127659575`, `val_auc=0.9686363636363636`, `val_f1=0.9285714285714286`, `peak_vram≈2.14 GiB`, `total_seconds≈902.1` → **keep**（相较 matched `L3-no-mixer` seed456，accuracy / F1 分别提升 `0.0106382978723405 / 0.0109243697478992`，但 AUC 回落 `0.0027272727272727`；这更像局部 seed 的 softmax-shrinkage 收益，不是稳定主线替代。）
- [x] **CMP-RESNEXT-DECISION-256X8-L5-TEMP2P0-FORMAL-S42**：`runs/resnext_decision_256x8_matrix/formal/l5_temp2p0/s42`（commit `f84df46`，`ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER=1`，`ANKLE_LEARNED_FUSION_TEMPERATURE=2.0`）→ `val_acc=0.9042553191489362`, `val_auc=0.9827272727272728`, `val_f1=0.8988764044943820`, `peak_vram≈2.15 GiB`, `total_seconds≈898.0` → **discard**（虽然 AUC 进一步升到本轮最高，但相较 matched `L3-no-mixer` seed42，accuracy / F1 分别回落 `0.0212765957446808 / 0.0224719101123596`；主指标代价过大。）
- [x] **CMP-RESNEXT-DECISION-256X8-L5-TEMP2P0-FORMAL-S123**：`runs/resnext_decision_256x8_matrix/formal/l5_temp2p0/s123`（commit `f84df46`，`ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER=1`，`ANKLE_LEARNED_FUSION_TEMPERATURE=2.0`）→ `val_acc=0.8617021276595744`, `val_auc=0.9500000000000000`, `val_f1=0.8266666666666667`, `peak_vram≈2.15 GiB`, `total_seconds≈900.5` → **discard**（相较 matched `L3-no-mixer` seed123，accuracy / F1 分别回落 `0.0319148936170213 / 0.0483333333333333`；这条线没有修复低点，反而把 low-point 拉得更低。）
- [x] **CMP-RESNEXT-DECISION-256X8-L5-TEMP2P0-FORMAL-S456**：`runs/resnext_decision_256x8_matrix/formal/l5_temp2p0/s456`（commit `f84df46`，`ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER=1`，`ANKLE_LEARNED_FUSION_TEMPERATURE=2.0`）→ `val_acc=0.9255319148936170`, `val_auc=0.9618181818181817`, `val_f1=0.9195402298850575`, `peak_vram≈2.14 GiB`, `total_seconds≈904.0` → **discard**（与 matched `L3-no-mixer` seed456 在 `val_acc` 上打平，但 AUC 回落 `0.0095454545454546`；因此不能保留。）
- **3-seed learned-mainline 结论（L5-temp1p5）**：`L5-temp1p5` 的 mean `val_acc=0.9148936170212766`、mean `val_auc=0.9653030303030303`、mean `val_f1=0.9084464555052790`，`val_acc` population std 为 `0.0229813499943541`。相较 `L3-no-mixer`，它没有提升主指标 mean accuracy，但提升了 `+0.0024242424242424 val_auc` 与 `+0.0037813310285220 val_f1`，同时把 `val_acc` std 从 `0.0150448251316287` 拉高到 `0.0229813499943541`。这说明 `temperature=1.5` 更像是 ranking-side softening，而不是更强的主线 recipe。
- **3-seed learned-mainline 结论（L5-temp2p0）**：`L5-temp2p0` 的 mean `val_acc=0.8971631205673759`、mean `val_auc=0.9648484848484848`、mean `val_f1=0.8816944336820353`，`val_acc` population std 为 `0.0265365772111627`。相较 `L3-no-mixer`，它虽然把 mean AUC 再抬了 `0.0019696969696969`，但 mean accuracy 回落 `0.0177304964539007`，mean F1 回落 `0.0229706907947217`；不具备主线保留价值。
- **当前判断**：到这里，`temperature` 线已经足够回答问题了。`temp1.5` 只能提供轻度 ranking 修正，`temp2.0` 则明显伤害主指标 accuracy；两者都没有把 `L3-no-mixer` 从 “当前 strongest learned branch” 的位置上替换掉。按主线规则，`temperature` 现在正式封口，后续模块消融不再默认继续新增 `temp=*`。
- **推荐动作**：下一步不再继续任何 `temperature` probe，而是把 **`L3-no-mixer` 固定为新的 canonical learned branch**，转入非-temp 的小范围标量修复。最高优先级是围绕当前低点 `seed=123` 发起一轮 fresh `main-study`，只在 `lr / weight_decay / dropout / gradient_clip_norm` 上做小范围搜索，看能否在不回引 mixer 的前提下把 accuracy 从 `0.893617` 往上抬。

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

## 2026-04-23：Decision-Fusion Repair Follow-up（DFR-31 continuous top1-top2 margin-scaled rescue，adaptive main-study，RTX A6000）

> **实验说明**
> - 本轮继续严格保持当前唯一主线与 `256x8 ResNeXt decision + L3-no-mixer + DFR-25` 语义不变，不切 backbone / geometry，不切到 feature fusion；唯一离散改动是在 commit `fbf46bf` 的 [src/model.py](/dataset/HH/ankle-ct/src/model.py) 新增 `ANKLE_DECISION_TRAIN_AXIAL_TOP12_MARGIN_SCALE_THRESHOLD`。
> - 这个修复仍然只在**训练态**生效，并保留 `DFR-26` 的 full-strength `ANKLE_DECISION_TRAIN_NONDOMINANT_WEIGHT_FLOOR=0.05` 与 `DFR-25` 的 dominant-gate dropout；但触发不再是 `DFR-30` 那种 hard `top1-top2 margin >= 0.7`。新的语义是：只要 `axial` 仍是 dominant，就让 weak-view rescue 随 `top1-top2` margin 从阈值 `0.4` 开始连续放大，目标是覆盖 near-collapse 样本，同时避免 `DFR-26` blanket floor 对已迁移样本的无差别压平。
> - fresh adaptive `main-study` 使用 search-config copy [autoresearch_logs/generated_search_configs/optuna_main_search_iter_0011_20260423_082437.yaml](/dataset/HH/ankle-ct/autoresearch_logs/generated_search_configs/optuna_main_search_iter_0011_20260423_082437.yaml) 与 study_root [runs/optuna_main_autoloop/iter_0011_20260423_082437](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0011_20260423_082437)，runtime env 为 `ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER=1` + `ANKLE_DECISION_TRAIN_DOMINANT_GATE_DROPOUT_PROB=0.25` + `ANKLE_DECISION_TRAIN_NONDOMINANT_WEIGHT_FLOOR=0.05` + `ANKLE_DECISION_TRAIN_AXIAL_TOP12_MARGIN_SCALE_THRESHOLD=0.4`。
> - 本轮唯一研究问题是：如果 `DFR-30` 的失败主要来自 hard margin trigger 过于稀疏，那么把同一个 collapse score 改成**连续覆盖**，是否能在不削弱 `0.05` rescue 量的前提下，把 seed42 重新拉回 `DFR-25/26` 的 `0.9362` ceiling，而不是继续停在 `equal-weight tie` 附近。

- [x] **DFR-31-RESNEXT-DECISION-256X8-CONTINUOUS-TOP12-MARGIN-SCALED-RESCUE-MAIN-S42**：fresh adaptive `main-study` [runs/optuna_main_autoloop/iter_0011_20260423_082437](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0011_20260423_082437) 的 best completed trial（trial `0`，commit `fbf46bf`，runtime env `ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER=1` + `ANKLE_DECISION_TRAIN_DOMINANT_GATE_DROPOUT_PROB=0.25` + `ANKLE_DECISION_TRAIN_NONDOMINANT_WEIGHT_FLOOR=0.05` + `ANKLE_DECISION_TRAIN_AXIAL_TOP12_MARGIN_SCALE_THRESHOLD=0.4`）→ `val_acc=0.9148936170212766`, `val_auc=0.9790909090909091`, `val_f1=0.9090909090909091`, `peak_vram≈2.15 GiB`, `total_seconds≈937.5` → **discard**（相较 matched `equal-weight` control `seed=42` `0.9255319148936170 / 0.9486363636363637 / 0.9176470588235294`，主指标 `val_acc` 反而回落 `0.0106382978723404`；相较 retained `DFR-26` seed42 keep `0.9361702127659575 / 0.9800000000000000 / 0.9302325581395349`，`val_acc / val_auc / val_f1` 分别回落 `0.0212765957446809 / 0.0009090909090909 / 0.0211416490486258`；相较 `DFR-25` seed42 winner `0.9361702127659575 / 0.9786363636363636 / 0.9333333333333333`，则回落 `0.0212765957446809 / -0.0004545454545455 / 0.0242424242424242`。按当前主指标规则，这是一条明确不能保留的回退。）
- **study 内部分布**：requested `4` 个 valid trial 与 `+2` 个 wave-aligned tail-fill 全部完成；`6/6` completed trial 只形成 `0.9042553191489362` 与 `0.9148936170212766` 两档，没有任何一个角点回到 `DFR-25/26/28` 已经证明可达的 `0.9361702127659575`。best trial `0`（`lr=1e-4`, `weight_decay=5e-4`, `dropout=0.25`, `gradient_clip_norm=2.0`）虽然拿到本轮最高 `val_auc=0.9790909090909091`，但 accuracy ceiling 全 study 一致停在 `0.9149`。
- **当前判断**：`DFR-31` 直接否定了“只要把 `DFR-30` 的 hard margin trigger 变平滑，就能重新守住 retained keep”的假设。和 `DFR-29/30` 一样，这轮不是单点 unlucky spike，而是 across-the-board 的平台式回落：同一条 continuous margin score 既没保住 `DFR-26` 的 seed42 keep，也没把 learned full-fusion 拉回 matched `equal-weight` 之上。
- **推荐动作**：不要继续沿 **top1-top2 margin family** 再细扫阈值或曲线形状。若继续留在当前唯一主线，更直接的下一步应改测 **gate entropy-based collapse score**：仍保持 `DFR-25` + full-strength `floor=0.05` 不变，但把 trigger 从二元 margin 排名改成使用整个 gate 分布集中度，验证“full-distribution concentration”是否比 `top1-top2` gap 更能识别真正的 axial starvation 样本。

---

## 2026-04-23：Decision-Fusion Repair Follow-up（DFR-32 axial low-entropy trigger，adaptive main-study，RTX A6000）

> **实验说明**
> - 本轮继续严格保持当前唯一主线与 `256x8 ResNeXt decision + L3-no-mixer + DFR-25` 语义不变，不切 backbone / geometry，不切到 feature fusion；唯一离散改动是在 commit `3b2d06e` 的 [src/model.py](/dataset/HH/ankle-ct/src/model.py) 新增 `ANKLE_DECISION_TRAIN_AXIAL_ENTROPY_THRESHOLD`。
> - 这个修复仍然只在**训练态**生效，并保留 `DFR-25` 的 dominant-gate dropout 与 `DFR-26` 的 full-strength `ANKLE_DECISION_TRAIN_NONDOMINANT_WEIGHT_FLOOR=0.05`；但触发条件不再看 `axial weight` 或 `top1-top2 margin`，而是只在 `axial` 仍是 dominant、且整个 gate 分布的**归一化 entropy** 不高于 `0.65` 时才注入 weak-view rescue。目标是用 full-distribution concentration 更准确地识别真实 starvation case，避免再把已经出现 axial→sagittal 迁移的样本误判成 collapse。
> - fresh adaptive `main-study` 使用 search-config copy [autoresearch_logs/generated_search_configs/optuna_main_search_iter_0012_20260423_090521.yaml](/dataset/HH/ankle-ct/autoresearch_logs/generated_search_configs/optuna_main_search_iter_0012_20260423_090521.yaml) 与 study_root [runs/optuna_main_autoloop/iter_0012_20260423_090521](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0012_20260423_090521)，runtime env 为 `ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER=1` + `ANKLE_DECISION_TRAIN_DOMINANT_GATE_DROPOUT_PROB=0.25` + `ANKLE_DECISION_TRAIN_NONDOMINANT_WEIGHT_FLOOR=0.05` + `ANKLE_DECISION_TRAIN_AXIAL_ENTROPY_THRESHOLD=0.65`。
> - 本轮唯一研究问题是：如果 `DFR-30/31` 的 margin family 失败主要因为只看 `top1-top2` gap，遗漏了第三视角 starvation 与整分布塌缩信息，那么把 trigger 换成 full-distribution low-entropy detector，是否能在 seed42 上重新守住 `DFR-25/26` 的 `0.9362` ceiling，而不是再次掉回 `0.9149` 平台。

- [x] **DFR-32-RESNEXT-DECISION-256X8-AXIAL-LOW-ENTROPY-TRIGGER-MAIN-S42**：fresh adaptive `main-study` [runs/optuna_main_autoloop/iter_0012_20260423_090521](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0012_20260423_090521) 的 best completed trial（trial `4`，commit `3b2d06e`，runtime env `ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER=1` + `ANKLE_DECISION_TRAIN_DOMINANT_GATE_DROPOUT_PROB=0.25` + `ANKLE_DECISION_TRAIN_NONDOMINANT_WEIGHT_FLOOR=0.05` + `ANKLE_DECISION_TRAIN_AXIAL_ENTROPY_THRESHOLD=0.65`）→ `val_acc=0.9255319148936170`, `val_auc=0.9777272727272728`, `val_f1=0.9230769230769231`, `peak_vram≈2.15 GiB`, `total_seconds≈954.1` → **discard**（相较 matched `equal-weight` control `seed=42` `0.9255319148936170 / 0.9486363636363637 / 0.9176470588235294`，只做到 `val_acc` 打平，虽然 `val_auc / val_f1` 仍高 `0.0290909090909091 / 0.0054298642533937`；但相较 retained `DFR-26` seed42 keep `0.9361702127659575 / 0.9800000000000000 / 0.9302325581395349`，`val_acc / val_auc / val_f1` 分别回落 `0.0106382978723405 / 0.0022727272727272 / 0.0071556350626118`；相较 `DFR-25` seed42 winner `0.9361702127659575 / 0.9786363636363636 / 0.9333333333333333`，同样回落 `0.0106382978723405 / 0.0009090909090908 / 0.0102564102564102`。按当前主指标规则，这条 entropy-trigger repair 仍未达到保留门槛。）
- **study 内部分布**：`6/6` completed trial 形成 `0.9043 / 0.9149 / 0.9255` 三档，best trial `4` 出现在 tail-fill duplicate 的 `lr=1e-4`, `weight_decay=5e-4`, `dropout=0.25`, `gradient_clip_norm=2.0` 角点；而 `DFR-25/26` 已知的 seed42 winning corner（trial `1`: `lr=1e-4`, `weight_decay=2.5e-4`, `dropout=0.25`, `gradient_clip_norm=2.5`）在本轮只得到 `0.9148936170212766 val_acc`。这说明 entropy trigger 虽然比 `DFR-30/31` 的 margin family 更健康，但仍然没把已知 best corner 保住。
- **当前判断**：`DFR-32` 给出的结论是 **mixed-negative**。正面的一点是，它没有像 `DFR-31` 那样把整个 study 压回纯 `0.9149` 平台，而是至少把 best trial 拉回了 matched `equal-weight tie`；这说明 **full-distribution concentration 确实比 `top1-top2 margin` 更接近真实 collapse detector**。但负面同样明确：`hard low-entropy threshold` 仍然太稀疏，既没保住 `DFR-25/26` 的 `0.9362` seed42 ceiling，也没有产生任何一个明确高于 matched `equal-weight` 的新 keep。
- **推荐动作**：保持 `256x8 ResNeXt decision + L3-no-mixer + DFR-25` 主线不变，也不要回到 `margin` family。若继续沿 entropy family 前进，更直接的下一步应是测试 **continuous low-entropy scaling**：仍保持 `ANKLE_DECISION_TRAIN_NONDOMINANT_WEIGHT_FLOOR=0.05` 不变，但让 rescue 强度随 normalized entropy 低于阈值的程度连续增长，目标是在保留 entropy detector 精度的同时，把 near-collapse 样本也纳入足够的 weak-view gradient 覆盖，而不是继续用硬阈值遗漏这些样本。

---

## 2026-04-23：Decision-Fusion Repair Follow-up（DFR-33 continuous low-entropy scaling，adaptive main-study，RTX A6000）

> **实验说明**
> - 本轮继续严格保持当前唯一主线与 `256x8 ResNeXt decision + L3-no-mixer + DFR-25` 语义不变，不切 backbone / geometry，不切到 feature fusion；唯一离散改动是在 commit `1c1900d` 的 [src/model.py](/dataset/HH/ankle-ct/src/model.py) 新增 `ANKLE_DECISION_TRAIN_AXIAL_ENTROPY_SCALE_THRESHOLD`。
> - 这个修复仍然只在**训练态**生效，并保留 `DFR-25` 的 dominant-gate dropout 与 `DFR-26` 的 full-strength `ANKLE_DECISION_TRAIN_NONDOMINANT_WEIGHT_FLOOR=0.05`；但不再像 `DFR-32` 那样对 low-entropy axial collapse 做 hard on/off 触发，而是让 rescue 强度随 normalized entropy 低于阈值 `0.65` 的程度连续放大，目标是把 near-collapse 样本也纳入 weak-view gradient 覆盖。
> - fresh adaptive `main-study` 使用 search-config copy [autoresearch_logs/generated_search_configs/optuna_main_search_iter_0013_20260423_094918.yaml](/dataset/HH/ankle-ct/autoresearch_logs/generated_search_configs/optuna_main_search_iter_0013_20260423_094918.yaml) 与 study_root [runs/optuna_main_autoloop/iter_0013_20260423_094918](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0013_20260423_094918)，runtime env 为 `ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER=1` + `ANKLE_DECISION_TRAIN_DOMINANT_GATE_DROPOUT_PROB=0.25` + `ANKLE_DECISION_TRAIN_NONDOMINANT_WEIGHT_FLOOR=0.05` + `ANKLE_DECISION_TRAIN_AXIAL_ENTROPY_SCALE_THRESHOLD=0.65`。
> - 本轮唯一研究问题是：如果 `DFR-32` 的失败主要来自 hard low-entropy trigger 仍然过稀，那么把同一 entropy detector 改成 continuous scaling，是否能在不回到 blanket floor 的前提下，把 seed42 重新拉回 `DFR-25/26` 的 `0.9362` ceiling 并再次明确超过 matched `equal-weight`。

- [x] **DFR-33-RESNEXT-DECISION-256X8-CONTINUOUS-LOW-ENTROPY-SCALING-MAIN-S42**：fresh adaptive `main-study` [runs/optuna_main_autoloop/iter_0013_20260423_094918](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0013_20260423_094918) 的 best completed trial（trial `4`，commit `1c1900d`，runtime env `ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER=1` + `ANKLE_DECISION_TRAIN_DOMINANT_GATE_DROPOUT_PROB=0.25` + `ANKLE_DECISION_TRAIN_NONDOMINANT_WEIGHT_FLOOR=0.05` + `ANKLE_DECISION_TRAIN_AXIAL_ENTROPY_SCALE_THRESHOLD=0.65`）→ `val_acc=0.9255319148936170`, `val_auc=0.9827272727272727`, `val_f1=0.9176470588235294`, `peak_vram≈2.15 GiB`, `total_seconds≈841.1` → **discard**（相较 matched `equal-weight` control `seed=42` `0.9255319148936170 / 0.9486363636363637 / 0.9176470588235294`，只做到 `val_acc / val_f1` 打平，虽然 `val_auc` 仍高 `0.0340909090909090`；但相较 retained `DFR-26` seed42 keep `0.9361702127659575 / 0.9800000000000000 / 0.9302325581395349`，`val_acc / val_auc / val_f1` 分别回落 `0.0106382978723405 / -0.0027272727272727 / 0.0125854993160055`；相较 `DFR-25` seed42 winner `0.9361702127659575 / 0.9786363636363636 / 0.9333333333333333`，同样回落 `0.0106382978723405 / -0.0040909090909091 / 0.0156862745098039`。按当前主指标规则，这条 continuous entropy repair 仍未达到保留门槛。）
- **study 内部分布**：`6/6` completed trial 形成 `0.8936 / 0.9043 / 0.9149 / 0.9255` 四档。best trial `4` 与 trial `1` 都只到 `0.9255319148936170 val_acc`；其中 trial `4`（`lr=1e-4`, `weight_decay=5e-4`, `dropout=0.25`, `gradient_clip_norm=2.0`）以更高 `val_auc=0.9827272727272727` 胜出，而 `DFR-25` 已知的 seed42 winning corner（trial `1`: `lr=1e-4`, `weight_decay=2.5e-4`, `dropout=0.25`, `gradient_clip_norm=2.5`）本轮也只恢复到 `equal-weight tie`，没有任何 trial 回到 `0.9362` ceiling。
- **当前判断**：`DFR-33` 比 `DFR-32` 更像正确方向，但结论仍然是 **negative**。continuous scaling 确实把 `DFR-32` 里被 hard entropy trigger 压坏的 winning corner 从 `0.9149` 拉回到了 `0.9255`，说明“entropy detector + smoother coverage”比 hard trigger 更健康；但 ceiling 依旧停在 matched `equal-weight tie`，表明问题已经不只是 detector 稀疏，而更像是 **uniform non-dominant rescue 本身把宝贵的 weak-view mass 平均摊给两个非 dominant 视角，导致真正有用的 fallback expert 仍然拿不到足够集中的梯度**。
- **推荐动作**：保持 `256x8 ResNeXt decision + L3-no-mixer + DFR-25` 主线不变，也**不要**继续沿 entropy threshold / scaling family 细扫。更直接的下一步应改测 **targeted non-dominant rescue redistribution**：仍保持 training-only semantics 与总 rescue mass 不变，但不要把 floor 平均分到两个 non-dominant views，而是按当前 non-dominant gate 排名或 detached teacher evidence，把 rescued mass 更集中地分给最强 fallback view，验证这是否能把 `equal-weight tie` 再推回 learned full-fusion 的净收益。

---

## 2026-04-23：Decision-Fusion Repair Follow-up（DFR-34 targeted strongest-fallback rescue redistribution，adaptive main-study，RTX A6000）

> **实验说明**
> - 本轮继续严格保持当前唯一主线与 `256x8 ResNeXt decision + L3-no-mixer + DFR-25` 语义不变，不切 backbone / geometry，不切到 feature fusion；唯一离散改动是在 commit `dbdc290` 的 [src/model.py](/dataset/HH/ankle-ct/src/model.py) 新增 `ANKLE_DECISION_TRAIN_TARGET_TOP_NONDOMINANT_RESCUE`。
> - 这个修复仍然只在**训练态**生效，并保留 `DFR-25` 的 dominant-gate dropout 与 `DFR-26` 的 `ANKLE_DECISION_TRAIN_NONDOMINANT_WEIGHT_FLOOR=0.05`；但不再把 rescue mass 平均分给两个 non-dominant views，而是把**同样的总 rescue mass**集中投给当前 gate 排名最高的 fallback view。推理期仍保持原始 learned late fusion，不引入任何 inference-time floor 或固定权重。
> - fresh adaptive `main-study` 使用 search-config copy [autoresearch_logs/generated_search_configs/optuna_main_search_iter_0014_20260423_103248.yaml](/dataset/HH/ankle-ct/autoresearch_logs/generated_search_configs/optuna_main_search_iter_0014_20260423_103248.yaml) 与 study_root [runs/optuna_main_autoloop/iter_0014_20260423_103248](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0014_20260423_103248)，runtime env 为 `ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER=1` + `ANKLE_DECISION_TRAIN_DOMINANT_GATE_DROPOUT_PROB=0.25` + `ANKLE_DECISION_TRAIN_NONDOMINANT_WEIGHT_FLOOR=0.05` + `ANKLE_DECISION_TRAIN_TARGET_TOP_NONDOMINANT_RESCUE=1`。
> - 本轮唯一研究问题是：如果 `DFR-33` 的 ceiling 被压在 `equal-weight tie` 的根因主要是“uniform rescue 把最有价值的 fallback 梯度稀释给了另一个弱视角”，那么把 rescue mass 定向集中到 strongest fallback view，是否能把 learned full-fusion 从 `equal-weight tie` 重新推回到明确更高的 `val_acc`。

- [x] **DFR-34-RESNEXT-DECISION-256X8-TARGETED-STRONGEST-FALLBACK-RESCUE-MAIN-S42**：fresh adaptive `main-study` [runs/optuna_main_autoloop/iter_0014_20260423_103248](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0014_20260423_103248) 的 best completed trial（trial `5`，commit `dbdc290`，runtime env `ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER=1` + `ANKLE_DECISION_TRAIN_DOMINANT_GATE_DROPOUT_PROB=0.25` + `ANKLE_DECISION_TRAIN_NONDOMINANT_WEIGHT_FLOOR=0.05` + `ANKLE_DECISION_TRAIN_TARGET_TOP_NONDOMINANT_RESCUE=1`）→ `val_acc=0.9255319148936170`, `val_auc=0.9813636363636363`, `val_f1=0.9263157894736842`, `peak_vram≈2.15 GiB`, `total_seconds≈869.1` → **discard**（相较 matched `equal-weight` control `seed=42` `0.9255319148936170 / 0.9486363636363637 / 0.9176470588235294`，只做到 `val_acc` 打平，虽然 `val_auc / val_f1` 仍高 `0.0327272727272726 / 0.0086687306501548`；但相较 retained `DFR-26` seed42 keep `0.9361702127659575 / 0.9800000000000000 / 0.9302325581395349`，`val_acc / val_f1` 分别回落 `0.0106382978723405 / 0.0039167686658507`，仅 `val_auc` 小幅更高 `0.0013636363636363`；相较 `DFR-25` seed42 winner `0.9361702127659575 / 0.9786363636363636 / 0.9333333333333333`，同样是 `val_acc / val_f1` 回落 `0.0106382978723405 / 0.0070175438596491`，只有 `val_auc` 更高 `0.0027272727272727`。按当前主指标规则，这条 targeted rescue redistribution 仍未达到保留门槛。）
- **study 内部分布**：`6/6` completed trial 形成 `0.8830 / 0.9043 / 0.9149 / 0.9255` 四档。更关键的是，`DFR-25/26` 已知的 seed42 winning corner（trial `1`: `lr=1e-4`, `weight_decay=2.5e-4`, `dropout=0.25`, `gradient_clip_norm=2.5`）在本轮只得到 `0.9042553191489362 val_acc`；而最终 best trial `5` 其实是 trial `2` 参数的 tail-fill duplicate（`lr=7.5e-5`, `weight_decay=7.5e-4`, `dropout=0.3`, `gradient_clip_norm=2.5`），trial `2` 本体只有 `0.9148936170212766`。这说明 gate-ranked targeted rescue 不仅没恢复 retained ceiling，还带来了明显的 run-to-run 波动。
- **当前判断**：`DFR-34` 给出的结论是 **negative**。把 rescue mass 从“均分给两个弱视角”改成“全部给当前 gate 认为最强的 fallback view”，最佳情况下确实能把 ceiling 从 `0.9149` 拉回到 `equal-weight tie`；但它仍然没有把 learned full-fusion `val_acc` 推回 matched `equal-weight` 之上，也没保住 `DFR-25/26` 的 seed42 best corner。更像的解释是：**当前 gate 自己仍带着 residual collapse bias，用 gate 排名来决定 rescue 目标，会把一部分训练态补偿继续投向并不真正最有价值的 fallback route**。
- **推荐动作**：保持 `256x8 ResNeXt decision + L3-no-mixer + DFR-25` 主线不变，也不要回到 detector family 的阈值细扫。若继续沿 targeted redistribution family 前进，更直接的下一步应是测试 **detached teacher-evidence targeted rescue**：仍保持 training-only semantics 与总 rescue mass 不变，但不要再由当前 gate 排名决定 rescue target，而是按 detached per-view evidence / teacher weight 把 rescue mass 集中投给真正更强的 fallback expert，验证这是否能减少 gate 自身偏置带来的误导分配。

---

## 2026-04-23：Decision-Fusion Repair Follow-up（DFR-35 detached teacher-evidence targeted rescue，adaptive main-study，RTX A6000）

> **实验说明**
> - 本轮继续严格保持当前唯一主线与 `256x8 ResNeXt decision + L3-no-mixer + DFR-25` 语义不变，不切 backbone / geometry，不切到 feature fusion；唯一离散改动是在 commit `6a20314` 的 [src/model.py](/dataset/HH/ankle-ct/src/model.py) 新增 `ANKLE_DECISION_TRAIN_TARGET_TEACHER_NONDOMINANT_RESCUE`。
> - 这个修复仍然只在**训练态**生效，并保留 `DFR-25` 的 dominant-gate dropout 与 `DFR-26` 的 `ANKLE_DECISION_TRAIN_NONDOMINANT_WEIGHT_FLOOR=0.05`；但 `DFR-34` 用当前 gate 排名选 fallback target 的逻辑被替换成 **detached teacher evidence / teacher weight** 选 target。也就是说，总 rescue mass 不变，只把“投给哪个 non-dominant view”的决策从已塌缩 gate 中解耦出来，改由 detached per-view classifier evidence 决定。推理期仍保持原始 learned late fusion，不引入任何 inference-time floor 或固定权重。
> - fresh adaptive `main-study` 使用 search-config copy [autoresearch_logs/generated_search_configs/optuna_main_search_iter_0015_20260423_111428.yaml](/dataset/HH/ankle-ct/autoresearch_logs/generated_search_configs/optuna_main_search_iter_0015_20260423_111428.yaml) 与 study_root [runs/optuna_main_autoloop/iter_0015_20260423_111428](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0015_20260423_111428)，runtime env 为 `ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER=1` + `ANKLE_DECISION_TRAIN_DOMINANT_GATE_DROPOUT_PROB=0.25` + `ANKLE_DECISION_TRAIN_NONDOMINANT_WEIGHT_FLOOR=0.05` + `ANKLE_DECISION_TRAIN_TARGET_TEACHER_NONDOMINANT_RESCUE=1`。
> - 本轮唯一研究问题是：如果 `DFR-34` 的失败主要因为“用 gate 自己来挑 fallback target 仍会继承 residual collapse bias”，那么把 rescue target 改成 detached teacher-evidence ranking，是否能把 learned full-fusion 从 matched `equal-weight tie` 重新推回到明确更高的 `val_acc`。

- [x] **DFR-35-RESNEXT-DECISION-256X8-DETACHED-TEACHER-EVIDENCE-TARGETED-RESCUE-MAIN-S42**：fresh adaptive `main-study` [runs/optuna_main_autoloop/iter_0015_20260423_111428](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0015_20260423_111428) 的 best completed trial（trial `2`，commit `6a20314`，runtime env `ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER=1` + `ANKLE_DECISION_TRAIN_DOMINANT_GATE_DROPOUT_PROB=0.25` + `ANKLE_DECISION_TRAIN_NONDOMINANT_WEIGHT_FLOOR=0.05` + `ANKLE_DECISION_TRAIN_TARGET_TEACHER_NONDOMINANT_RESCUE=1`）→ `val_acc=0.9255319148936170`, `val_auc=0.9813636363636364`, `val_f1=0.9176470588235294`, `peak_vram≈2.15 GiB`, `total_seconds≈931.5` → **discard**（相较 matched `equal-weight` control `seed=42` `0.9255319148936170 / 0.9486363636363637 / 0.9176470588235294`，只做到 `val_acc / val_f1` 打平，虽然 `val_auc` 仍高 `0.0327272727272727`；相较 retained `DFR-26` seed42 keep `0.9361702127659575 / 0.9800000000000000 / 0.9302325581395349`，`val_acc / val_f1` 分别回落 `0.0106382978723405 / 0.0125854993160055`，仅 `val_auc` 小幅更高 `0.0013636363636364`；相较 `DFR-25` seed42 winner `0.9361702127659575 / 0.9786363636363636 / 0.9333333333333333`，同样是 `val_acc / val_f1` 回落 `0.0106382978723405 / 0.0156862745098039`，只有 `val_auc` 更高 `0.0027272727272728`。按当前主指标规则，这条 detached teacher-targeted rescue 仍未达到保留门槛。）
- **study 内部分布**：`6/6` completed trial 形成 `0.9043 / 0.9149 / 0.9255` 三档，其中 `0.9255` 出现了 3 次（trial `0/2/5`），但没有任何一个角点回到 retained `DFR-25/26` 的 `0.9362`。更关键的是，`DFR-25/26` 已知的 seed42 winning corner（trial `1`: `lr=1e-4`, `weight_decay=2.5e-4`, `dropout=0.25`, `gradient_clip_norm=2.5`）在本轮再次只得到 `0.9148936170212766 val_acc`；而 best trial `2`（`lr=7.5e-5`, `weight_decay=7.5e-4`, `dropout=0.3`, `gradient_clip_norm=2.5`）也只恢复到 matched `equal-weight tie`，与 `DFR-34` 的 best ceiling 实质相同。
- **当前判断**：`DFR-35` 给出的结论仍然是 **negative**。把 rescue target 从 gate-ranked 改成 detached teacher-evidence ranked，并没有把 learned full-fusion `val_acc` 推回 matched `equal-weight` 之上，也没有恢复 `DFR-25/26` 的 seed42 best corner。这说明 `DFR-34` 的失败**不只是 rescue target 被 gate 偏置误导**；更像是当前整条 `non-dominant floor / rescue mass` family 本身就在压制 `DFR-25` 已经建立起来的更优 routing dynamics。
- **推荐动作**：保持 `256x8 ResNeXt decision + L3-no-mixer + DFR-25` 主线不变，但**不要继续沿 floor / targeted-rescue family 细扫**。更直接的下一步应回到 `DFR-25` 自身，测试 **teacher-guided dominant-gate dropout redistribution**：仍保持 training-only semantics，但不再额外注入 `floor=0.05` 的 rescue mass，而是在 dominant-gate dropout 触发时，把被压平的 dominant mass 按 detached teacher-evidence 重新分配给更强 fallback expert，验证“修复对象应该是 dropout redistribution 本身，而不是 dropout 之后再叠一层 non-dominant floor”。

---

## 2026-04-23：Decision-Fusion Repair Follow-up（DFR-36 teacher-guided dominant-gate dropout redistribution，adaptive main-study，RTX A6000）

> **实验说明**
> - 本轮继续严格保持当前唯一主线与 `256x8 ResNeXt decision + L3-no-mixer + DFR-25` 语义不变，不切 backbone / geometry，不切到 feature fusion；唯一离散改动是在 commit `c3e881b` 的 [src/model.py](/dataset/HH/ankle-ct/src/model.py) 新增 `ANKLE_DECISION_TRAIN_TARGET_TEACHER_DROPOUT_REDISTRIBUTION`。
> - 这个修复仍然只在**训练态**生效，并保留 `DFR-25` 的 `ANKLE_DECISION_TRAIN_DOMINANT_GATE_DROPOUT_PROB=0.25`；但不再像 `DFR-25` 那样把 dominant-gate dropout 后的梯度机会平均撒给两个 non-dominant views，而是把被压平的 dominant gate mass 重定向到 **detached teacher-evidence** 选出的 strongest fallback view。推理期仍保持原始 learned late fusion，不引入任何 inference-time floor 或固定权重。
> - fresh adaptive `main-study` 使用预留 search-config copy [autoresearch_logs/generated_search_configs/optuna_main_search_iter_0001_20260423_141632.yaml](/dataset/HH/ankle-ct/autoresearch_logs/generated_search_configs/optuna_main_search_iter_0001_20260423_141632.yaml) 与 study_root [runs/optuna_main_autoloop/iter_0001_20260423_141632](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0001_20260423_141632)，runtime env 为 `ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER=1` + `ANKLE_DECISION_TRAIN_DOMINANT_GATE_DROPOUT_PROB=0.25` + `ANKLE_DECISION_TRAIN_TARGET_TEACHER_DROPOUT_REDISTRIBUTION=1`。
> - 本轮唯一研究问题是：如果 `DFR-25` 已经证明“训练期打断 dominant lock-in”本身有价值，而后续失败主要来自 dropout 后的梯度分配过于平均，那么把 dropped dominant mass 直接投向 detached teacher 认为更有 fallback evidence 的视角，是否能保住 `DFR-25` 的 winning dynamics，并把 learned full-fusion 再次推到 matched `equal-weight` 之上。

- [x] **DFR-36-RESNEXT-DECISION-256X8-TEACHER-GUIDED-DROPOUT-REDISTRIBUTION-MAIN-S42**：fresh adaptive `main-study` [runs/optuna_main_autoloop/iter_0001_20260423_141632](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0001_20260423_141632) 的 best completed trial（trial `4`，commit `c3e881b`，runtime env `ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER=1` + `ANKLE_DECISION_TRAIN_DOMINANT_GATE_DROPOUT_PROB=0.25` + `ANKLE_DECISION_TRAIN_TARGET_TEACHER_DROPOUT_REDISTRIBUTION=1`）→ `val_acc=0.9148936170212766`, `val_auc=0.9390909090909091`, `val_f1=0.9024390243902439`, `peak_vram≈2.15 GiB`, `total_seconds≈994.4` → **discard**（相较 matched `equal-weight` control `seed=42` `0.9255319148936170 / 0.9486363636363637 / 0.9176470588235294`，`val_acc / val_auc / val_f1` 分别回落 `0.0106382978723404 / 0.0095454545454546 / 0.0152080344332855`；相较 retained `DFR-26` seed42 keep `0.9361702127659575 / 0.9800000000000000 / 0.9302325581395349`，`val_acc / val_auc / val_f1` 分别回落 `0.0212765957446809 / 0.0409090909090909 / 0.0277935337492910`；相较 `DFR-25` seed42 winner `0.9361702127659575 / 0.9786363636363636 / 0.9333333333333333`，同样回落 `0.0212765957446809 / 0.0395454545454545 / 0.0308943089430894`。按当前主指标规则，这是一条明确不能保留的回退。）
- **study 内部分布**：requested `4` 个 valid trial 最终补齐成 `6` 个 valid trial，形成 `0.8404 / 0.8617 / 0.9043 / 0.9149` 四档，没有任何一个角点回到 `DFR-25/26` 已经证明可达的 `0.9362`。更关键的是，`DFR-25/26` 的 seed42 winning corner（trial `1`: `lr=1e-4`, `weight_decay=2.5e-4`, `dropout=0.25`, `gradient_clip_norm=2.5`）在本轮只得到 `0.9042553191489362 / 0.9622727272727273 / 0.8988764044943820`；best completed trial `4`（`lr=1e-4`, `weight_decay=5e-4`, `dropout=0.25`, `gradient_clip_norm=2.0`）虽然把 `val_acc` 拉到本轮最高 `0.9149`，但整体 ceiling 仍低于 matched `equal-weight`。
- **当前判断**：`DFR-36` 给出的结论是 **negative**。把 dominant-dropout 的训练期梯度机会从“平均打散”改成“定向投给 teacher 认为更强的 fallback expert”并没有恢复 `DFR-25` 的 keep，反而让整个 study ceiling 从此前 floor-family 常见的 `equal-weight tie` 进一步掉到 `0.9149`。这说明当前问题**不只是 dropout 之后把 mass 投给谁**；更像是 `DFR-25` 的收益本身依赖于更柔和的 gate flattening，而 teacher-targeted 的单点重分配会过早把 routing 压向单一路径，伤到 learned full-fusion 的净收益。
- **推荐动作**：保持 `256x8 ResNeXt decision + L3-no-mixer + DFR-25` 主线不变，但**不要继续沿 dominant-dropout 后的 redistribution family 细扫**。若外层 loop 继续，下一条合格方向应改成 **evidence-misalignment-conditioned dominant-gate dropout**：不是再改“dropout 后把 mass 给谁”，而是让 dropout 只在 gate 与 detached teacher evidence 明显不一致、或 axial dominance 明显越过 teacher evidence 的样本上触发，从根源上缩小错误干预覆盖面。

---

## 2026-04-23：Decision-Fusion Repair Follow-up（DFR-37 evidence-misalignment-conditioned dominant-gate dropout，adaptive main-study，V100 32GB x3）

> **实验说明**
> - 本轮继续严格保持当前唯一主线与 `256x8 ResNeXt decision + L3-no-mixer + DFR-25` 语义不变，不切 backbone / geometry，不切到 feature fusion；唯一离散改动是在 commit `e48302a` 的 [src/model.py](/dataset/HH/ankle-ct/src/model.py) 新增 `ANKLE_DECISION_TRAIN_AXIAL_TEACHER_MISALIGNMENT_THRESHOLD`。
> - 这个修复仍然只在**训练态**生效，并保留 `DFR-25` 的 `ANKLE_DECISION_TRAIN_DOMINANT_GATE_DROPOUT_PROB=0.25`；但 dominant-gate dropout 不再对所有 sampled 样本一视同仁，而是只在 **axial 仍是 gate top-view，且 gate 的 axial mass 至少比 detached teacher evidence 高 `0.1`** 时才允许触发。也就是说，这轮改的不是 dropout 后把 mass 给谁，而是把干预覆盖面收缩到明确的 evidence-to-routing mismatch 样本。
> - fresh adaptive `main-study` 使用预留 search-config copy [autoresearch_logs/generated_search_configs/optuna_main_search_iter_0002_20260423_150121.yaml](/dataset/HH/ankle-ct/autoresearch_logs/generated_search_configs/optuna_main_search_iter_0002_20260423_150121.yaml) 与 study_root [runs/optuna_main_autoloop/iter_0002_20260423_150121](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0002_20260423_150121)，runtime env 为 `ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER=1` + `ANKLE_DECISION_TRAIN_DOMINANT_GATE_DROPOUT_PROB=0.25` + `ANKLE_DECISION_TRAIN_AXIAL_TEACHER_MISALIGNMENT_THRESHOLD=0.1`。
> - 本轮唯一研究问题是：如果 `DFR-25` 之后的连续失败主要来自训练期 dropout 覆盖面过宽、误伤了原本已经 healthy 的 axial-leading 样本，那么把 dropout 收缩到 “gate axial 明显高于 teacher evidence” 的 mismatch case，是否能在不重新引入 floor / redistribution 的前提下，保住 learned full-fusion 超过 matched `equal-weight` 的净收益，并进一步超过 retained `DFR-25/26`。

- [x] **DFR-37-RESNEXT-DECISION-256X8-MISALIGNMENT-CONDITIONED-DOMINANT-DROPOUT-MAIN-S42**：fresh adaptive `main-study` [runs/optuna_main_autoloop/iter_0002_20260423_150121](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0002_20260423_150121) 的 best completed trial（trial `0`，commit `e48302a`，runtime env `ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER=1` + `ANKLE_DECISION_TRAIN_DOMINANT_GATE_DROPOUT_PROB=0.25` + `ANKLE_DECISION_TRAIN_AXIAL_TEACHER_MISALIGNMENT_THRESHOLD=0.1`）→ `val_acc=0.9361702127659575`, `val_auc=0.9745454545454546`, `val_f1=0.9318181818181818`, `peak_vram≈2.15 GiB`, `total_seconds≈1060.4` → **discard**（相较 matched `equal-weight` control `seed=42` `0.9255319148936170 / 0.9486363636363637 / 0.9176470588235294`，`val_acc / val_auc / val_f1` 分别提升 `0.0106382978723405 / 0.0259090909090909 / 0.0141711229946524`；但相较 retained `DFR-25` seed42 winner `0.9361702127659575 / 0.9786363636363636 / 0.9333333333333333`，主指标 `val_acc` 只做到打平，`val_auc / val_f1` 分别回落 `0.0040909090909090 / 0.0015151515151515`；相较 retained `DFR-26` seed42 keep `0.9361702127659575 / 0.9800000000000000 / 0.9302325581395349`，虽然 `val_f1` 小幅更高 `0.0015856236786469`，但 tie-break `val_auc` 仍低 `0.0054545454545454`。按当前 protocol，这轮没有超过已有 keep。）
- **study 内部分布**：`6/6` completed trial 形成 `0.9043 / 0.9255 / 0.9362` 三档。最关键的是，best trial `0`（`lr=1e-4`, `weight_decay=5e-4`, `dropout=0.25`, `gradient_clip_norm=2.0`）把 ceiling 重新拉回了 retained `DFR-25/26` 的 `0.9362` 档；同时另外 2 个 completed trial (`1/2`) 也都保持在 `0.9255`，没有再像 `DFR-36` 那样整体掉到 `0.9149` 以下。这说明“只在 evidence-mismatch case 触发 dropout”确实比后续 `floor / targeted-rescue / redistribution` 家族更接近 `DFR-25` 的正确信号。
- **当前判断**：`DFR-37` 的机制结论是 **positive-but-not-promoted**。它直接验证了 backlog 上一轮的假设：问题更像是错误干预覆盖面，而不是 dropout 后把 mass 投给谁。把 dominant-gate dropout 缩到 detached teacher 明确不同意的 axial lock-in 样本后，seed42 ceiling 可以恢复到 `0.9362`，并再次明确高于 matched `equal-weight`。但按 ledger 规则，这轮仍是 **discard**，因为它没有在 `val_acc` 上超过 retained `DFR-25/26`，且在 tie-break `val_auc` 上落后于已有 keep。
- **推荐动作**：保持 `256x8 ResNeXt decision + L3-no-mixer + DFR-25` 主线不变，也不要回到 `floor / redistribution` family。若外层 loop 继续，更直接的下一步应是测试 **continuous evidence-misalignment-scaled dominant-gate dropout**：不是继续细扫 hard threshold `0.1`，而是让 dropout 强度或触发概率随 `gate axial - teacher axial` excess 连续变化。目标是保留 `DFR-37` 已经验证的“只修 mismatch case”优点，同时把 near-mismatch 样本也纳入更平滑的训练覆盖，争取在不误伤 healthy axial-leading 样本的前提下，把 tie-break `val_auc` 与主指标 ceiling 一起抬过 retained `DFR-25/26`。

---

## 2026-04-23：Decision-Fusion Repair Follow-up（DFR-38 continuous evidence-misalignment-scaled dominant-gate dropout，adaptive main-study，V100 32GB x3）

> **实验说明**
> - 本轮继续严格保持当前唯一主线与 `256x8 ResNeXt decision + L3-no-mixer + DFR-25` 语义不变，不切 backbone / geometry，不切到 feature fusion；唯一离散改动是在 commit `6ce6405` 的 [src/model.py](/dataset/HH/ankle-ct/src/model.py) 新增 `ANKLE_DECISION_TRAIN_AXIAL_TEACHER_MISALIGNMENT_SCALE_THRESHOLD`。
> - 这个修复仍然只在**训练态**生效，并保留 `DFR-25` 的 `ANKLE_DECISION_TRAIN_DOMINANT_GATE_DROPOUT_PROB=0.25`；但与 `DFR-37` 的 hard mismatch gate 不同，这一轮不再要求 `gate axial - detached teacher axial` 先硬过 `0.1` 才触发 dropout。相反，只要 `axial` 仍是 gate top-view，dominant-gate dropout 的**触发概率**就按 misalignment 大小连续缩放，并在 `0.1` 处达到满强度。也就是说，这轮改的是训练期干预覆盖曲线，而不是 dropout 后的重分配语义。
> - fresh adaptive `main-study` 使用预留 search-config copy [autoresearch_logs/generated_search_configs/optuna_main_search_iter_0003_20260423_154327.yaml](/dataset/HH/ankle-ct/autoresearch_logs/generated_search_configs/optuna_main_search_iter_0003_20260423_154327.yaml) 与 study_root [runs/optuna_main_autoloop/iter_0003_20260423_154327](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0003_20260423_154327)，runtime env 为 `ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER=1` + `ANKLE_DECISION_TRAIN_DOMINANT_GATE_DROPOUT_PROB=0.25` + `ANKLE_DECISION_TRAIN_AXIAL_TEACHER_MISALIGNMENT_SCALE_THRESHOLD=0.1`。
> - 本轮唯一研究问题是：如果 `DFR-37` 还没超过 retained keep 的原因主要来自 hard mismatch gate 太稀疏、漏掉了 near-mismatch 样本，那么把 dropout coverage 改成随 misalignment 连续放大，是否能在不回到 blanket intervention 的前提下，保住 `DFR-37` 的 `0.9362` ceiling 并把 tie-break `val_auc` 再往上抬。

- [x] **DFR-38-RESNEXT-DECISION-256X8-CONTINUOUS-MISALIGNMENT-SCALED-DOMINANT-DROPOUT-MAIN-S42**：fresh adaptive `main-study` [runs/optuna_main_autoloop/iter_0003_20260423_154327](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0003_20260423_154327) 的 best completed trial（trial `4`，commit `6ce6405`，runtime env `ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER=1` + `ANKLE_DECISION_TRAIN_DOMINANT_GATE_DROPOUT_PROB=0.25` + `ANKLE_DECISION_TRAIN_AXIAL_TEACHER_MISALIGNMENT_SCALE_THRESHOLD=0.1`）→ `val_acc=0.9255319148936170`, `val_auc=0.9740909090909091`, `val_f1=0.9230769230769231`, `peak_vram≈2.15 GiB`, `total_seconds≈974.5` → **discard**（相较 matched `equal-weight` control `seed=42` `0.9255319148936170 / 0.9486363636363637 / 0.9176470588235294`，只做到 `val_acc` 打平，虽然 `val_auc / val_f1` 仍高 `0.0254545454545454 / 0.0054298642533937`；但相较 retained `DFR-25` seed42 winner `0.9361702127659575 / 0.9786363636363636 / 0.9333333333333333`，`val_acc / val_auc / val_f1` 分别回落 `0.0106382978723405 / 0.0045454545454545 / 0.0102564102564102`；相较 retained `DFR-26` seed42 keep `0.9361702127659575 / 0.9800000000000000 / 0.9302325581395349`，`val_acc / val_auc` 分别回落 `0.0106382978723405 / 0.0059090909090909`，虽然 `val_f1` 小幅更高 `0.0078443649373882`。按当前 protocol，这轮没有保住 `DFR-37` 的 ceiling，更没有超过已有 keep。）
- **study 内部分布**：`6/6` completed trial 形成 `0.9043 / 0.9149 / 0.9255` 三档，没有任何一个角点回到 `DFR-37` 已恢复出的 `0.9362` ceiling。更关键的是，`DFR-37` 最接近 retained keep 的两个角点在本轮都被压低：trial `1`（旧 seed42 winning corner：`lr=1e-4`, `weight_decay=2.5e-4`, `dropout=0.25`, `gradient_clip_norm=2.5`）只得到 `0.9148936170212766 / 0.9740909090909091 / 0.9130434782608695`；best completed trial `4` 也只是 `gradient_clip_norm=1.5` 下回到 matched `equal-weight tie`。
- **当前判断**：`DFR-38` 给出的结论是 **negative**。把 `DFR-37` 的 hard mismatch gate 改成 pre-threshold continuous coverage，并没有把 near-mismatch 样本转化为新的净收益，反而把整个 study ceiling 从 `0.9362` 压回到 `0.9255`。这说明当前问题**不是 hard gate 单纯“太稀疏”**；更像是当 dropout 干预开始覆盖那些只是轻度 misalignment 的 axial-leading 样本时，会重新误伤 `DFR-25/37` 已经保住的 healthy routing dynamics。
- **推荐动作**：保持 `256x8 ResNeXt decision + L3-no-mixer + DFR-25` 主线不变，也不要回到 `floor / redistribution` family。若外层 loop 继续，**不要再扩大 pre-threshold mismatch coverage**。更合理的下一步应是保留 `DFR-37` 的 hard evidence-mismatch eligibility，只在**已经过 hard gate 的真实 mismatch 样本**上测试连续强度控制（例如按超过 `0.1` 的 misalignment excess 去缩放 dropout flattening 强度，而不是放宽触发覆盖面），以避免再次误伤 near-mismatch / healthy axial-leading 样本。

---

## 2026-04-23：Decision-Fusion Repair Follow-up（DFR-39 hard-gated misalignment-excess-scaled dominant-gate dropout，adaptive main-study，V100 32GB x3）

> **实验说明**
> - 本轮继续严格保持当前唯一主线与 `256x8 ResNeXt decision + L3-no-mixer + DFR-25` 语义不变，不切 backbone / geometry，不切到 feature fusion；唯一离散改动是在 commit `cbb77a1` 的 [src/model.py](/dataset/HH/ankle-ct/src/model.py) 新增 `ANKLE_DECISION_TRAIN_AXIAL_TEACHER_MISALIGNMENT_EXCESS_SCALE_WINDOW`。
> - 这个修复仍然只在**训练态**生效，并保留 `DFR-37` 已验证更接近正确方向的 hard mismatch eligibility：`axial` 仍需是 gate top-view，且 `gate axial - detached teacher axial >= 0.1`。与 `DFR-37` 唯一不同的是，一旦样本已经过 hard gate，不再对这些样本统一做 full-strength dominant-dropout flattening；而是让 flattening 强度按超过 `0.1` 的 misalignment excess 连续缩放，在 `window=0.05` 处达到满强度。也就是说，这轮改的是**已过 hard gate 样本内部**的干预强度曲线，而不是再去扩大 coverage。
> - fresh adaptive `main-study` 使用预留 search-config copy [autoresearch_logs/generated_search_configs/optuna_main_search_iter_0004_20260423_162502.yaml](/dataset/HH/ankle-ct/autoresearch_logs/generated_search_configs/optuna_main_search_iter_0004_20260423_162502.yaml) 与 study_root [runs/optuna_main_autoloop/iter_0004_20260423_162502](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0004_20260423_162502)，runtime env 为 `ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER=1` + `ANKLE_DECISION_TRAIN_DOMINANT_GATE_DROPOUT_PROB=0.25` + `ANKLE_DECISION_TRAIN_AXIAL_TEACHER_MISALIGNMENT_THRESHOLD=0.1` + `ANKLE_DECISION_TRAIN_AXIAL_TEACHER_MISALIGNMENT_EXCESS_SCALE_WINDOW=0.05`。
> - 本轮唯一研究问题是：如果 `DFR-37` 没有超过 retained keep 的原因主要来自“hard mismatch 样本里也存在干预过重”，那么在**不放宽 eligibility** 的前提下，仅把这些已确认 mismatch 样本的 flattening 强度做得更平滑，是否能保住 `0.9362` ceiling 并把 tie-break `val_auc` 往上抬。

- [x] **DFR-39-RESNEXT-DECISION-256X8-HARD-GATED-MISALIGNMENT-EXCESS-SCALED-DOMINANT-DROPOUT-MAIN-S42**：fresh adaptive `main-study` [runs/optuna_main_autoloop/iter_0004_20260423_162502](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0004_20260423_162502) 的 best completed trial（trial `4`，commit `cbb77a1`，runtime env `ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER=1` + `ANKLE_DECISION_TRAIN_DOMINANT_GATE_DROPOUT_PROB=0.25` + `ANKLE_DECISION_TRAIN_AXIAL_TEACHER_MISALIGNMENT_THRESHOLD=0.1` + `ANKLE_DECISION_TRAIN_AXIAL_TEACHER_MISALIGNMENT_EXCESS_SCALE_WINDOW=0.05`）→ `val_acc=0.9255319148936170`, `val_auc=0.9772727272727272`, `val_f1=0.9176470588235294`, `peak_vram≈2.15 GiB`, `total_seconds≈941.6` → **discard**（相较 matched `equal-weight` control `seed=42` `0.9255319148936170 / 0.9486363636363637 / 0.9176470588235294`，只做到 `val_acc / val_f1` 打平，虽然 `val_auc` 仍高 `0.0286363636363635`；但相较 `DFR-37` seed42 best `0.9361702127659575 / 0.9745454545454546 / 0.9318181818181818`，`val_acc / val_f1` 分别回落 `0.0106382978723405 / 0.0141711229946524`；相较 retained `DFR-25` seed42 winner `0.9361702127659575 / 0.9786363636363636 / 0.9333333333333333`，`val_acc / val_auc / val_f1` 分别回落 `0.0106382978723405 / 0.0013636363636364 / 0.0156862745098039`；相较 retained `DFR-26` seed42 keep `0.9361702127659575 / 0.9800000000000000 / 0.9302325581395349`，同样在 `val_acc / val_auc / val_f1` 上全部落后。按当前 protocol，这轮没有保住 `DFR-37` 的 ceiling，更没有超过已有 keep。）
- **study 内部分布**：`6/6` completed trial 形成 `0.9043 / 0.9149 / 0.9255` 三档，没有任何一个角点回到 `DFR-37` 已恢复出的 `0.9362` ceiling。更关键的是，`DFR-25/26` 的 seed42 winning corner（trial `1`: `lr=1e-4`, `weight_decay=2.5e-4`, `dropout=0.25`, `gradient_clip_norm=2.5`）在本轮只得到 `0.9042553191489362 / 0.9781818181818183 / 0.8988764044943820`；best completed trial `4` 也只是以 `lr=7.5e-5`, `weight_decay=7.5e-4`, `dropout=0.3`, `gradient_clip_norm=1.5` 回到 matched `equal-weight tie`。
- **权重 telemetry**：best trial `4` 的 [fusion_weight_analysis.json](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0004_20260423_162502/trials/trial_0004/run/fusion_weight_analysis.json) 显示 `mean fusion weight axial/coronal/sagittal = 0.9359 / 0.0270 / 0.0370`，并且 `top-weight count/rate axial/coronal/sagittal = 94/1.0000 / 0/0.0000 / 0/0.0000`。这说明一旦把 hard-gated mismatch 样本里的 flattening 做成 excess-scaled partial intervention，weak-view mass 很快就重新饿死，routing 直接塌回“axial 全样本主导”。
- **当前判断**：`DFR-39` 给出的结论是 **negative**。问题已经不只是“是否扩大 coverage”，而是 **对已确认 mismatch 的样本，dominant-dropout 也不能被软化得太多**。`DFR-37` 的正信号更像来自“只在真实 mismatch case 上做 full-strength correction”；一旦把这些样本内部再分成强弱层级，弱视角拿到的有效融合梯度又不够了，best trial 的 routing 与主指标都会一起退回 `equal-weight tie` / axial re-collapse。
- **推荐动作**：保持 `256x8 ResNeXt decision + L3-no-mixer + DFR-25` 主线不变，也不要继续沿 **mismatch intensity scaling** 前进。若外层 loop 继续，更合理的下一步应改成 **更严格的 hard mismatch eligibility**，而不是继续软化强度：例如保留 `DFR-37` 的 full-strength dropout，但要求 detached teacher 的 top-view 必须是非 axial，或要求 teacher 的 strongest non-axial evidence 明确超过 axial，再触发 correction。目标是只在“teacher 明确认为非 axial 更该主导”的真实跨视角分歧样本上做完整干预，而不是在已确认 mismatch 样本内部继续削弱修复力度。

---

## 2026-04-23：Decision-Fusion Repair Follow-up（DFR-40 teacher-nonaxial-top hard mismatch eligibility，adaptive main-study，V100 32GB x3）

> **实验说明**
> - 本轮继续严格保持当前唯一主线与 `256x8 ResNeXt decision + L3-no-mixer + DFR-25` 语义不变，不切 backbone / geometry，不切到 feature fusion；唯一离散改动是在 commit `94c4dcf` 的 [src/model.py](/dataset/HH/ankle-ct/src/model.py) 新增 `ANKLE_DECISION_TRAIN_REQUIRE_TEACHER_NONAXIAL_TOP`。
> - 这个修复仍然只在**训练态**生效，并保留 `DFR-37` 已验证最接近正确方向的 full-strength dominant-dropout 与 hard mismatch gate：`axial` 仍需是 gate top-view，且 `gate axial - detached teacher axial >= 0.1`。与 `DFR-37` 唯一不同的是，本轮再加一层更严格 eligibility：只有当 **detached teacher 的 top-view 已经是非 axial** 时，dominant-dropout 才允许触发。也就是说，这轮改的不是 dropout 强度，也不是重分配语义，而是把 correction 覆盖面进一步缩到 “teacher 已明确认为该由非 axial 主导” 的样本。
> - 设计思路 / 预计改进效果：`DFR-39` 表明已确认 mismatch 样本内部的 partial intervention 会让 weak-view gradient 再次不足，因此本轮回到 `DFR-37` 的 full-strength dropout；同时为了避免 `DFR-37` 仍可能误伤“teacher 只是认为 axial 该降一点、但 axial 仍应主导”的 healthy axial-leading 样本，再加 `teacher top-view non-axial` 这道门。若假设正确，预期结果应是：只有真正的跨视角 routing disagreement 样本才会被纠偏，seed42 ceiling 至少应保住 `0.9362`，并且 best trial 的权重 telemetry 不该再是 `94/94 axial top-weight` 的全塌缩形态。
> - fresh adaptive `main-study` 使用预留 search-config copy [autoresearch_logs/generated_search_configs/optuna_main_search_iter_0005_20260423_170759.yaml](/dataset/HH/ankle-ct/autoresearch_logs/generated_search_configs/optuna_main_search_iter_0005_20260423_170759.yaml) 与 study_root [runs/optuna_main_autoloop/iter_0005_20260423_170759](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0005_20260423_170759)，runtime env 为 `ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER=1` + `ANKLE_DECISION_TRAIN_DOMINANT_GATE_DROPOUT_PROB=0.25` + `ANKLE_DECISION_TRAIN_AXIAL_TEACHER_MISALIGNMENT_THRESHOLD=0.1` + `ANKLE_DECISION_TRAIN_REQUIRE_TEACHER_NONAXIAL_TOP=1`。

- [x] **DFR-40-RESNEXT-DECISION-256X8-TEACHER-NONAXIAL-TOP-HARD-MISMATCH-DROPOUT-MAIN-S42**：fresh adaptive `main-study` [runs/optuna_main_autoloop/iter_0005_20260423_170759](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0005_20260423_170759) 的 best completed trial（trial `1`，commit `94c4dcf`，runtime env `ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER=1` + `ANKLE_DECISION_TRAIN_DOMINANT_GATE_DROPOUT_PROB=0.25` + `ANKLE_DECISION_TRAIN_AXIAL_TEACHER_MISALIGNMENT_THRESHOLD=0.1` + `ANKLE_DECISION_TRAIN_REQUIRE_TEACHER_NONAXIAL_TOP=1`）→ `val_acc=0.9148936170212766`, `val_auc=0.9809090909090910`, `val_f1=0.9090909090909091`, `peak_vram≈2.15 GiB`, `total_seconds≈989.7` → **discard**（相较 matched `equal-weight` control `seed=42` `0.9255319148936170 / 0.9486363636363637 / 0.9176470588235294`，`val_acc / val_f1` 分别回落 `0.0106382978723404 / 0.0085561497326203`，仅 `val_auc` 更高 `0.0322727272727273`；相较 `DFR-37` seed42 best `0.9361702127659575 / 0.9745454545454546 / 0.9318181818181818`，`val_acc / val_f1` 分别回落 `0.0212765957446809 / 0.0227272727272727`；相较 retained `DFR-25` seed42 winner `0.9361702127659575 / 0.9786363636363636 / 0.9333333333333333`，同样回落 `0.0212765957446809 / 0.0040909090909091 / 0.0242424242424242`。按当前 protocol，这条更严格 eligibility 没有达到保留门槛。）
- **study 内部分布**：`6/6` completed trial 只形成 `0.9043` 与 `0.9149` 两档，没有任何一个角点回到 `DFR-37` 已恢复出的 `0.9362` ceiling。更关键的是，`DFR-25/26` 已知的 seed42 winning corner（trial `1`: `lr=1e-4`, `weight_decay=2.5e-4`, `dropout=0.25`, `gradient_clip_norm=2.5`）在本轮只得到 `0.9148936170212766 / 0.9809090909090910 / 0.9090909090909091`；其余 3 个 `0.9149` trial 也没有任何一个在 accuracy 上超过 matched `equal-weight`。这说明把 correction 再收窄到 “teacher top-view 已翻到非 axial” 后，干预覆盖面已经窄到不足以维持 `DFR-37` 的 keep-ceiling。
- **权重 telemetry**：best trial `1` 的 [fusion_weight_analysis.json](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0005_20260423_170759/trials/trial_0001/run/fusion_weight_analysis.json) 显示 `mean fusion weight axial/coronal/sagittal = 0.9611 / 0.0271 / 0.0118`，并且 `top-weight count/rate axial/coronal/sagittal = 94/1.0000 / 0/0.0000 / 0/0.0000`。同时 `top_true_margin_view_distribution = axial 84 / coronal 5 / sagittal 5`，却没有任何样本把 top fusion weight 迁走，说明这条 strict teacher-top gate 让 non-axial evidence 即使在 detached classifier 上已出现，也拿不到足够的 routing control。
- **当前判断**：`DFR-40` 给出的结论是 **negative**。这轮的失败不是因为 full-strength dropout 本身无效，而是因为 **`teacher top-view must be non-axial` 过于严格**：它把 `DFR-37` 原本还能修到的 residual mismatch case 一并过滤掉了，导致 weak-view 梯度再次饥饿，best trial 的 routing 重新塌回 `axial 94/94` 全样本主导。
- **推荐动作**：保持 `256x8 ResNeXt decision + L3-no-mixer + DFR-25` 主线不变，但**不要继续沿 “teacher top-view 必须非 axial” 这条 binary stricter gate** 前进。若外层 loop 继续，更合理的下一步应是回到 `DFR-37` 的 hard mismatch base，并只加入一种 **比 top-view flip 更宽、但仍比原始 `DFR-37` 更有针对性的非 axial competitiveness 条件**，例如要求 detached teacher 的 strongest non-axial evidence 与 axial 足够接近或达到最小强度，而不是必须已经完全翻成非 axial top-view；目标是保留对 residual mismatch case 的 full-strength correction，同时避免再次把 coverage 收窄到直接 re-collapse。

---

## 2026-04-23：Decision-Fusion Repair Follow-up（DFR-41 teacher-nonaxial competitiveness-gap hard mismatch eligibility，adaptive main-study，V100 32GB x3）

> **实验说明**
> - 本轮继续严格保持当前唯一主线与 `256x8 ResNeXt decision + L3-no-mixer + DFR-25` 语义不变，不切 backbone / geometry，不切到 feature fusion；唯一离散改动是在 commit `98804e1` 的 [src/model.py](/dataset/HH/ankle-ct/src/model.py) 新增 `ANKLE_DECISION_TRAIN_TEACHER_NONAXIAL_COMPETITIVE_GAP`。
> - 这个修复仍然只在**训练态**生效，并保留 `DFR-37` 已验证更接近正确方向的 full-strength dominant-dropout 与 hard mismatch gate：`axial` 仍需是 gate top-view，且 `gate axial - detached teacher axial >= 0.1`。与 `DFR-40` 唯一不同的是，本轮不再要求 detached teacher top-view 已经翻成 non-axial；只要 detached teacher 的 **strongest non-axial weight 与 axial 的差距不超过 `0.2`**，correction 就允许触发。也就是说，这轮改的仍然是 eligibility，但从“已经翻盘”放宽到“已经具备明确竞争力”。
> - 设计思路 / 预计改进效果：`DFR-40` 的失败说明 `teacher top-view non-axial` 把 coverage 收得过窄；如果真正需要的是“teacher 已经看到 non-axial 足够有竞争力”而不是“teacher 已经完全翻盘”，那么把 eligibility 放宽到 strongest non-axial 与 axial 足够接近，理论上应能保留 `DFR-37` 对 residual mismatch case 的 full-strength correction，同时减少对 healthy axial-leading sample 的误干预。若假设成立，预期结果至少应把 seed42 ceiling 拉回 `0.9362`，并让 best trial 的权重 telemetry 不再是 `94/94 axial top-weight` 的单视角塌缩形态。
> - fresh adaptive `main-study` 使用预留 search-config copy [autoresearch_logs/generated_search_configs/optuna_main_search_iter_0006_20260423_174928.yaml](/dataset/HH/ankle-ct/autoresearch_logs/generated_search_configs/optuna_main_search_iter_0006_20260423_174928.yaml) 与 study_root [runs/optuna_main_autoloop/iter_0006_20260423_174928](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0006_20260423_174928)，runtime env 为 `ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER=1` + `ANKLE_DECISION_TRAIN_DOMINANT_GATE_DROPOUT_PROB=0.25` + `ANKLE_DECISION_TRAIN_AXIAL_TEACHER_MISALIGNMENT_THRESHOLD=0.1` + `ANKLE_DECISION_TRAIN_TEACHER_NONAXIAL_COMPETITIVE_GAP=0.2`。

- [x] **DFR-41-RESNEXT-DECISION-256X8-TEACHER-NONAXIAL-COMPETITIVE-GAP-HARD-MISMATCH-DROPOUT-MAIN-S42**：fresh adaptive `main-study` [runs/optuna_main_autoloop/iter_0006_20260423_174928](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0006_20260423_174928) 的 best completed trial（trial `2`，commit `98804e1`，runtime env `ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER=1` + `ANKLE_DECISION_TRAIN_DOMINANT_GATE_DROPOUT_PROB=0.25` + `ANKLE_DECISION_TRAIN_AXIAL_TEACHER_MISALIGNMENT_THRESHOLD=0.1` + `ANKLE_DECISION_TRAIN_TEACHER_NONAXIAL_COMPETITIVE_GAP=0.2`）→ `val_acc=0.9255319148936170`, `val_auc=0.9831818181818183`, `val_f1=0.9176470588235294`, `peak_vram≈2.15 GiB`, `total_seconds≈1081.0` → **discard**（相较 matched `equal-weight` control `seed=42` `0.9255319148936170 / 0.9486363636363637 / 0.9176470588235294`，只做到 `val_acc / val_f1` 打平，虽然 `val_auc` 仍高 `0.0345454545454546`；但相较 `DFR-37` seed42 best `0.9361702127659575 / 0.9745454545454546 / 0.9318181818181818`，`val_acc / val_f1` 分别回落 `0.0106382978723405 / 0.0141711229946524`，仅 `val_auc` 更高 `0.0086363636363637`；相较 retained `DFR-25` seed42 winner `0.9361702127659575 / 0.9786363636363636 / 0.9333333333333333`，同样回落 `0.0106382978723405 / 0.0015151515151515 / 0.0156862745098039`；相较 retained `DFR-26` seed42 keep `0.9361702127659575 / 0.9800000000000000 / 0.9302325581395349`，则在 `val_acc / val_auc / val_f1` 上分别回落 `0.0106382978723405 / -0.0031818181818183 / 0.0125854993160055`。按当前 protocol，这轮没有把 learned full-fusion 拉回 matched `equal-weight` 之上，更没有超过当前 keep。）
- **study 内部分布**：`6/6` completed trial 形成 `0.9043 / 0.9149 / 0.9255` 三档，没有任何一个角点回到 `DFR-37` 或 retained `DFR-25/26` 的 `0.9362` ceiling。best completed trial 出现在 `trial 2`（`lr=7.5e-5`, `weight_decay=7.5e-4`, `dropout=0.3`, `gradient_clip_norm=2.5`），而 `DFR-25/26` 已知的 seed42 winning corner（trial `1`: `lr=1e-4`, `weight_decay=2.5e-4`, `dropout=0.25`, `gradient_clip_norm=2.5`）在本轮只得到 `0.9042553191489362 / 0.9813636363636364 / 0.8965517241379310`。这说明即便把 `DFR-40` 的 top-view flip 放宽成 `gap<=0.2`，coverage 仍然窄到不足以恢复 `DFR-37` 的 keep-ceiling。
- **权重 telemetry**：best trial `2` 的 [fusion_weight_analysis.json](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0006_20260423_174928/trials/trial_0002/run/fusion_weight_analysis.json) 显示 `mean fusion weight axial/coronal/sagittal = 0.9689 / 0.0177 / 0.0134`，并且 `top-weight count/rate axial/coronal/sagittal = 94/1.0000 / 0/0.0000 / 0/0.0000`。与此同时，`top_true_margin_view_distribution = axial 84 / coronal 6 / sagittal 4`，说明 detached classifier 上仍存在 non-axial 更有证据的样本，但 hard competitiveness gate 依旧没让这些样本拿到足够的 routing control，best trial 最终重新塌回了 `axial 94/94`。
- **当前判断**：`DFR-41` 给出的结论仍然是 **negative**。这轮相较 `DFR-40` 的确更健康，至少没有全盘掉回 `0.9149` 平台；但结果上它仍只把 ceiling 拉回 matched `equal-weight tie`，并且权重 telemetry 比 `DFR-40` 还更偏 axial。这说明问题不在于 `teacher top-view must flip` 这个 binary gate 的符号方向，而在于 **任何额外的 hard teacher-competitiveness gate 都会把 `DFR-37` 需要修正的 residual mismatch coverage 收窄过头**。
- **推荐动作**：保持 `256x8 ResNeXt decision + L3-no-mixer + DFR-25` 主线不变，但**不要继续沿 binary teacher-competitiveness eligibility 家族前进**，包括 `teacher top-view flip` 与 `strongest non-axial gap` 两种硬门。若外层 loop 继续，更合理的下一步应回到 `DFR-37` 的 hard mismatch base，不再额外收窄 coverage；如果仍要利用 detached teacher 的 non-axial competitiveness 信息，应优先先做 telemetry / offline threshold study，再考虑不会进一步压缩 correction coverage 的非二值方案。

---

## 2026-04-23：Decision-Fusion Repair Follow-up（DFR-42 additive mismatch dropout boost，adaptive main-study，V100 32GB x3）

> **实验说明**
> - 本轮继续严格保持当前唯一主线与 `256x8 ResNeXt decision + L3-no-mixer + DFR-25` 语义不变，不切 backbone / geometry，不切到 feature fusion；唯一离散改动是在 commit `1d816ab` 的 [src/model.py](/dataset/HH/ankle-ct/src/model.py) 新增 `ANKLE_DECISION_TRAIN_AXIAL_TEACHER_MISALIGNMENT_DROPOUT_BOOST`。
> - 设计思路：`DFR-40/41` 说明 hard teacher-competitiveness gate 会把 correction coverage 收窄过头，导致 weak-view starvation 与 axial re-collapse；本轮改成 **additive boost**，保留 `DFR-25` 原始 `0.25` 随机 dominant-gate dropout 覆盖，只在 gate 过度相信 axial、且 detached teacher 显示 strongest non-axial 与 axial 仍有竞争力的样本上额外增加 dropout 概率。
> - 预计改进效果：如果失败根因确实是 hard eligibility 压缩 coverage，那么 additive boost 应至少保住 `DFR-25/26` 的 `0.9362` seed42 ceiling，同时把一部分真实 mismatch 样本的训练梯度导向 sagittal/coronal，使 best checkpoint 的 top-weight 不再是 `94/94 axial`，并有机会让 learned full-fusion 在 matched `equal-weight` 之上获得更稳定的样本级 routing 净收益，而不是机械平均。
> - fresh adaptive `main-study` 使用预留 search-config copy [autoresearch_logs/generated_search_configs/optuna_main_search_iter_0007_20260423_183115.yaml](/dataset/HH/ankle-ct/autoresearch_logs/generated_search_configs/optuna_main_search_iter_0007_20260423_183115.yaml) 与 study_root [runs/optuna_main_autoloop/iter_0007_20260423_183115](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0007_20260423_183115)，runtime env 为 `ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER=1` + `ANKLE_DECISION_TRAIN_DOMINANT_GATE_DROPOUT_PROB=0.25` + `ANKLE_DECISION_TRAIN_AXIAL_TEACHER_MISALIGNMENT_THRESHOLD=0.1` + `ANKLE_DECISION_TRAIN_TEACHER_NONAXIAL_COMPETITIVE_GAP=0.2` + `ANKLE_DECISION_TRAIN_AXIAL_TEACHER_MISALIGNMENT_DROPOUT_BOOST=0.25`。

- [x] **DFR-42-RESNEXT-DECISION-256X8-ADDITIVE-MISMATCH-DROPOUT-BOOST-MAIN-S42**：fresh adaptive `main-study` [runs/optuna_main_autoloop/iter_0007_20260423_183115](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0007_20260423_183115) 的 best completed trial（trial `4`，commit `1d816ab`，runtime env 如上）→ `val_acc=0.9148936170212766`, `val_auc=0.9754545454545455`, `val_f1=0.9047619047619048`, `peak_vram≈2.15 GiB`, `total_seconds≈957.3` → **discard**（相较 matched `equal-weight` control `seed=42` `0.9255319148936170 / 0.9486363636363637 / 0.9176470588235294`，`val_acc / val_f1` 分别回落 `0.0106382978723404 / 0.0128851540616246`，仅 `val_auc` 更高；相较 retained `DFR-25` seed42 winner `0.9361702127659575 / 0.9786363636363636 / 0.9333333333333333` 与 `DFR-26` seed42 keep `0.9361702127659575 / 0.9800000000000000 / 0.9302325581395349`，主指标均回落 `0.0212765957446809`。）
- **study 内部分布**：`6/6` completed trial 中，trial `0/1/2/3/4` 全部停在 `0.9148936170212766 val_acc`，trial `5` 进一步降到 `0.8936170212765957`；没有任何一个角点回到 `DFR-25/26/37` 的 `0.9362` ceiling。best trial `4` 与 trial `0` 是同一 template scalar（`lr=1e-4`, `weight_decay=5e-4`, `dropout=0.25`, `gradient_clip_norm=2.0`）的 wave-fill rerun，只是 AUC 更高，因此按 monitor tie-break 成为 best。
- **权重 telemetry**：best trial `4` 的 [fusion_weight_analysis.json](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0007_20260423_183115/trials/trial_0004/run/fusion_weight_analysis.json) 显示 `mean fusion weight axial/coronal/sagittal = 0.7482 / 0.0699 / 0.1819`，`top-weight count/rate axial/coronal/sagittal = 84/0.8936 / 0/0.0000 / 10/0.1064`。这比 `DFR-41` 的 `axial=0.9689` 与 `94/94 axial top-weight` 确实更不塌缩，但 top-weight hit rate 对 `true_margin` 只有 `0.7979`，且 sagittal single-view metrics 只有 `accuracy=0.5319`, `f1=0.0`；也就是说，本轮把一部分权重迁给了弱视角，但这些迁移没有变成正确决策。
- **当前判断**：`DFR-42` 给出的结论是 **negative but diagnostic**。它证明“保留 base coverage 的 additive boost”确实避免了 DFR-40/41 的纯 axial re-collapse，但也暴露出更关键的问题：仅凭 detached teacher/non-axial competitiveness 增加 dropout，容易把 routing 推向当前并不可靠的 sagittal fallback，导致 learned full-fusion accuracy 低于 matched `equal-weight`。这不是 amplitude 太小的问题，而是 evidence target 本身仍不够可信。
- **推荐动作**：不要继续沿 **更大 additive boost / 更宽 non-axial competitiveness** 方向推进。下一步如果仍使用 teacher evidence，必须先做 sample-level telemetry / offline threshold study，比较 `DFR-25/37/42` 中被 boost 或迁移的样本到底哪些 non-axial evidence 是真增益、哪些是 sagittal false fallback；在没有这个证据前，不应再加大 dropout/boost 覆盖。若必须直接做训练修复，更合格的方向应改为保守地保护 `DFR-25` base，同时避免把额外梯度集中到当前明显不可靠的 sagittal fallback。

---

## 2026-04-23：Decision-Fusion Repair Follow-up（DFR-43 fallback-disagreement-only teacher dropout redistribution，adaptive main-study，V100 32GB x3）

> **实验说明**
> - 本轮继续严格保持当前唯一主线与 `256x8 ResNeXt decision + L3-no-mixer + DFR-25` 语义不变，不切 backbone / geometry，不切到 feature fusion；唯一离散改动是在 commit `2c803ab` 的 [src/model.py](/dataset/HH/ankle-ct/src/model.py) 新增 `ANKLE_DECISION_TRAIN_TARGET_TEACHER_DROPOUT_ONLY_ON_FALLBACK_DISAGREEMENT`。
> - 设计思路：`DFR-42` 的样本级对比显示，hard mismatch 样本里的问题不只是“要不要额外干预”，更像是 **gate 的 non-axial fallback 自己也可能选错**。因此本轮不扩大 coverage，也不再增加 teacher 竞争力 hard gate；而是保留 `DFR-37` 已验证更接近正确方向的 hard mismatch eligibility，只在 dropped 样本里 **gate fallback 与 detached teacher fallback 明确分歧** 时，才让 teacher-targeted redistribution 覆盖原始 flattening。其余 dropped 样本继续走 `DFR-25/37` 的平均 flattening。
> - 预计改进效果：如果 residual mismatch 的核心问题确实是“extra gradient 仍流向错误 fallback view”而不是 coverage 不足，那么这条 disagreement-only redistribution 至少应保住 `DFR-37` 已恢复出的 `0.9362` ceiling，同时把一小部分 “teacher 已看见更强 non-axial evidence、但 gate fallback 仍走错” 的样本纠正过来，使 best checkpoint 的 `top-weight axial` 不再是 `94/94` 全塌缩，并把 learned full-fusion `val_acc` 再次推到 matched `equal-weight` 之上，而不是仅提高局部 calibration。
> - fresh adaptive `main-study` 使用预留 search-config copy [autoresearch_logs/generated_search_configs/optuna_main_search_iter_0008_20260423_191146.yaml](/dataset/HH/ankle-ct/autoresearch_logs/generated_search_configs/optuna_main_search_iter_0008_20260423_191146.yaml) 与 study_root [runs/optuna_main_autoloop/iter_0008_20260423_191146](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0008_20260423_191146)，runtime env 为 `ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER=1` + `ANKLE_DECISION_TRAIN_DOMINANT_GATE_DROPOUT_PROB=0.25` + `ANKLE_DECISION_TRAIN_AXIAL_TEACHER_MISALIGNMENT_THRESHOLD=0.1` + `ANKLE_DECISION_TRAIN_TARGET_TEACHER_DROPOUT_REDISTRIBUTION=1` + `ANKLE_DECISION_TRAIN_TARGET_TEACHER_DROPOUT_ONLY_ON_FALLBACK_DISAGREEMENT=1`。

- [x] **DFR-43-RESNEXT-DECISION-256X8-FALLBACK-DISAGREEMENT-ONLY-TEACHER-DROPOUT-REDISTRIBUTION-MAIN-S42**：fresh adaptive `main-study` [runs/optuna_main_autoloop/iter_0008_20260423_191146](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0008_20260423_191146) 的 best completed trial（trial `0`，commit `2c803ab`，runtime env 如上）→ `val_acc=0.9255319148936170`, `val_auc=0.9654545454545456`, `val_f1=0.9213483146067416`, `peak_vram≈2.15 GiB`, `total_seconds≈1119.1` → **discard**（相较 matched `equal-weight` control `seed=42` `0.9255319148936170 / 0.9486363636363637 / 0.9176470588235294`，只做到 `val_acc` 打平，虽然 `val_auc / val_f1` 仍高 `0.0168181818181819 / 0.0037012557832122`；但相较 `DFR-37` seed42 best `0.9361702127659575 / 0.9745454545454546 / 0.9318181818181818`，`val_acc / val_auc / val_f1` 分别回落 `0.0106382978723405 / 0.0090909090909090 / 0.0104698672114402`；相较 retained `DFR-25` seed42 winner `0.9361702127659575 / 0.9786363636363636 / 0.9333333333333333` 与 retained `DFR-26` seed42 keep `0.9361702127659575 / 0.9800000000000000 / 0.9302325581395349`，主指标同样都回落 `0.0106382978723405`。按当前 protocol，这轮没有把 learned full-fusion 拉回 matched `equal-weight` 之上，更没有回到当前 keep ceiling。）
- **study 内部分布**：`6/6` completed trial 形成 `0.9043 / 0.9149 / 0.9255` 三档，没有任何一个角点回到 `DFR-25/26/37` 的 `0.9362` ceiling。更关键的是，`DFR-37` 的 seed42 best corner（trial `0`: `lr=1e-4`, `weight_decay=5e-4`, `dropout=0.25`, `gradient_clip_norm=2.0`）在本轮只得到 `0.9255319148936170 / 0.9654545454545456 / 0.9213483146067416`；而 `DFR-25/26` 的 retained winning corner（trial `1`: `lr=1e-4`, `weight_decay=2.5e-4`, `dropout=0.25`, `gradient_clip_norm=2.5`）更是只剩 `0.9042553191489362 / 0.9777272727272727 / 0.9032258064516129`。这说明即便只在 fallback disagreement 子集上启用 teacher-targeted redistribution，也没能保住 `DFR-37` 的 positive ceiling。
- **权重 telemetry**：best trial `0` 的 [fusion_weight_analysis.json](/dataset/HH/ankle-ct/runs/optuna_main_autoloop/iter_0008_20260423_191146/trials/trial_0000/run/fusion_weight_analysis.json) 显示 `mean fusion weight axial/coronal/sagittal = 0.9227 / 0.0739 / 0.0034`，`top-weight count/rate axial/coronal/sagittal = 94/1.0000 / 0/0.0000 / 0/0.0000`，而 `top_true_margin` 分布却是 `83 / 8 / 3`。也就是说，本轮几乎重新塌回了 `94/94 axial top-weight`；即便 `coronal` 在 `8` 个样本上拥有最高 `true_margin`，fusion 也没有一次把 top-weight 给到 coronal。`top_weight_hit_rate.true_margin=0.8830` 仍明显低于理想的 evidence-aligned routing。
- **当前判断**：`DFR-43` 给出的结论是 **negative and more decisive**。这轮说明问题已经不只是 “fallback expert 选错了几个样本”；只要在训练态把 `DFR-25/37` 的平均 flattening 替换成任何 teacher-targeted single-view redistribution，即便只发生在 hard mismatch 且 fallback disagreement 的更窄子集上，也会把 learned full-fusion ceiling 压回 matched `equal-weight tie`，并重新诱发 axial re-collapse。
- **推荐动作**：保持 `256x8 ResNeXt decision + L3-no-mixer + DFR-25` 主线不变，但**不要继续沿 teacher-targeted dropout redistribution family 前进**，包括更强 teacher targeting、更多 disagreement/advantage 细分或更宽的 redistribution coverage。若外层 loop 继续，更合理的下一步应是先做 sample-level telemetry / offline threshold study，对比 `DFR-25 / DFR-37 / DFR-43` 的 disagreement 样本，验证 detached teacher fallback 优势是否真的对应 `true_margin` 增益；在没有这层证据前，不应再提交新的 redistribution 训练修复。

---

## Agent 状态

> Agent 每次实验后必须更新此表。新 Agent 启动时以此表为起点。

| 字段 | 值 |
|------|-----|
| 上次实验 | `DFR-87 axial-protected non-axial gate view-correctness auxiliary loss`：main-study；在 [src/model.py](/dataset/HH/ankle-ct/src/model.py) 新增 default-off `ANKLE_DECISION_GATE_VIEW_CORRECTNESS_AUX_PROTECT_STRONG_AXIAL` 与 `ANKLE_DECISION_GATE_VIEW_CORRECTNESS_AUX_AXIAL_MARGIN_CAP`，只在 detached axial margin 不强时允许 non-axial gate view-correctness aux scaling；seed42 三候选 `axial_margin_cap=2.0/2.5/3.0`。 |
| 上次结果 | commit `367b1d7`，fresh main-study `iter_0014_20260512_080950` 完成 3/3 trials：trial0 `val_acc/val_auc/val_f1=0.914894/0.984091/0.911111`，trial1 `0.914894/0.976364/0.911111`，trial2 `0.904255/0.963182/0.898876`。best trial0 低于 matched equal-weight seed42 与 DFR-25 seed42，因此 discard；权重验证 mean axial/coronal/sagittal=`0.9070/0.0411/0.0520`，top-weight=`94/0/0`，true-margin=`85/6/3`，`hit_rate.true_margin=0.9043`，说明 axial-margin protection 过度保护并恢复 single-view axial dependence，没有把 weak-axial 样本转成有效 non-axial contribution。 |
| 下一步 | 不再继续扫 axial-margin cap，也不要继续在 DFR-83/86 aux family 上做幅度微调。若外层 loop 继续，优先转向 error-aware supervised pairwise / selector target：必须显式保护 DFR-25-correct 样本，并只在 oracle-correct non-axial error / true-margin 有证据样本上推动非 axial contribution；继续避免 broad view-role amplitude、plain floor、direct teacher blend、scalar temperature、blind prior-debias 和 candidate-threshold sweep。 |
| 默认执行策略 | 24GB+ 显存机器默认直接跑 `main` / `formal`；`proxy` 仅保留作低显存 fallback 与快速 smoke。 |
| 连续 discard 计数 | 62（自 `DFR-26 seed123` 起至 `DFR-87 axial-protected non-axial gate view-correctness auxiliary loss` 连续为 discard；DFR-57/60/61/62/63/64/65/66/67/68/69/70/71/72/73/74/75/76/77/78/79/80/81/82/83/84/85 follow-up 离线分析不改变 discard 计数语义，但本轮 ledger 仍按 discard 记录。） |
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
> - **2026-04-23 当前唯一主线** 已锁到：**`resnext + decision fusion + 256x8 + L3-no-mixer / routing repair`**
> - 旧 `512x16` backbone compare、paper lane 与旧 proxy winner 仅保留为 historical reference，不再覆盖当前主线判断

| 指标 | 值 | 来源 commit | 配置 | 备注 |
|------|---:|-------------|------|------|
| **当前主线 seed42 learned > equal keep** | **0.9361702127659575** | `bf4256e` | `autoresearch_logs/generated_search_configs/optuna_main_search_iter_0001_20260423_021418_retry1.yaml` | `DFR-25 dominant-gate dropout` 首个 positive keep；在当前 `256x8 decision-only` 主线上首次把 multi-view learned full-fusion `val_acc` 明确推到 matched `equal-weight` 之上 |
| **当前主线 seed123 learned > equal confirm keep** | **0.9361702127659575** | `8085b55` | `autoresearch_logs/generated_search_configs/optuna_main_search_iter_0002_20260423_031222.yaml` | `DFR-25 dominant-gate dropout` exact confirmation；保持与 seed42 相同的 winning scalar，再次 beat matched `equal-weight`，使该 repair 升级为 `2/2` matched keep |
| **当前主线 seed456 learned > equal confirm keep** | **0.9468085106382979** | `3fb1e11` | `autoresearch_logs/generated_search_configs/optuna_main_search_iter_0003_20260423_033048.yaml` | `DFR-25 dominant-gate dropout` final exact confirmation；保持同一 winning scalar 并在第三颗 matched seed 上再次 beat `equal-weight`，完成 `3/3` matched keep |
| **当前主线 DFR-25 matched 3-seed mean val_acc** | **0.9397163120567376** | `bf4256e/8085b55/3fb1e11` | `autoresearch_logs/generated_search_configs/optuna_main_search_iter_{0001_retry1,0002,0003}_20260423_*.yaml` | `DFR-25 dominant-gate dropout` 的 matched 3-seed mean；现已明确高于 matched `equal-weight` control mean `0.9219858156028368` |
| **当前主线 DFR-25 matched 3-seed mean val_auc** | **0.9677272727272728** | `bf4256e/8085b55/3fb1e11` | 同上 | 与上行同一 `DFR-25` 3-seed keep；也高于 matched `equal-weight` control mean `0.9606060606060606` |
| **当前主线 matched equal-weight control (s42)** | **0.9255319148936170** | `8e0bdf4` | `configs/generated_resnext_decision_256x8_matrix/formal/cmp_resnext_decision_256x8_l0_equal_formal_s42.yaml` | 当前 256x8 locked mainline 的 fixed equal-weight accuracy 参考 |
| **当前主线 matched equal-weight control mean val_acc** | **0.9219858156028368** | `8e0bdf4` | `configs/generated_resnext_decision_256x8_matrix/formal/cmp_resnext_decision_256x8_l0_equal_formal_{s42,s123,s456}.yaml` | 当前 `DFR-25` confirmation 系列所对齐的 fixed equal-weight 3-seed control mean |
| **当前主线 pre-repair learned anchor mean val_acc** | **0.9148936170212766** | `27b557c/3f7bc35` | `configs/generated_resnext_decision_256x8_matrix/formal/cmp_resnext_decision_256x8_l3_no_mixer_formal_{s42,s123,s456}.yaml` | `L3-no-mixer` 的 matched 3-seed mean；作为 dominant-gate-dropout 之前的 strongest learned anchor |
| **当前主线 pre-repair learned anchor mean val_auc** | **0.9628787878787879** | `27b557c/3f7bc35` | 同上 | 与上行同一 matched 3-seed `L3-no-mixer` anchor |
| legacy `512x16` canonical mean val_acc | 0.8581560283687942 | `5b286a7` | `configs/cmp_backbone_decision_resnext_512x16_e20_{s42,s123,s456}.yaml` | 旧 backbone compare winner；保留作历史 reference |
| legacy `512x16` canonical mean val_auc | 0.9119696969696971 | `5b286a7` | 同上 | 与上行同一 matched final；保留作历史 reference |
| **paper reproduction best val_acc** | **0.8510638297872340** | `ed6d535` | `paper_repro/configs/d4_hybrid_25d_3d.yaml` | corrected paper reproduction 总冠军；closer-to-paper rerun 后 `D4` 以 `val_auc=0.9309090909090909` 同时占据该 lane 的最高 AUC |
| legacy `512x16` fusion 控制变量参考 | 0.854609929078014 | `3b826ad` | `configs/cmp_decision_equal_resnext_{learned,equal}_512x16_e20_{s42,s123,s456}.yaml` | 旧 `512x16` matched fusion control；保留作 historical reference |
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
> - 2026-05-10 起，operational baseline 改为 `DFR-25 ResNeXt 256x8 learned decision fusion`
> - `ResNeXt + decision fusion + L3 no-mixer + train-time dominant-gate dropout`
> - `image_size=256`, `num_slices_per_view=8`, `trim_edge_slices=2`, `use_attention_pooling=false`
> - runtime env 固定为 `ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER=1` + `ANKLE_DECISION_TRAIN_DOMINANT_GATE_DROPOUT_PROB=0.25`
> - winning scalar 固定为 `freeze_layers=3`, `lr=1e-4`, `weight_decay=2.5e-4`, `dropout=0.25`, `gradient_clip_norm=2.5`
> - 主指标是 `summary.json -> best_val.accuracy`
> - 若 `val_acc` 持平，按 `best_val.auc` 决胜
>
> **当前流程约束**
> - `model.freeze_layers` 必须由 YAML 显式声明，不再通过修改 `src/model.py` 常量切换
> - Optuna 必须优先使用项目 `.venv`
> - 默认 Optuna search config 只表达 DFR-25 fixed anchor；`search_space: {}` 是有意收敛，不代表要开始宽搜
> - 任何后续 ResNeXt 改进都必须从 DFR-25 fork，并且一次只改一个明确声明的变量
> - 直接训练必须使用 `scripts/run_train_with_config_env.py` 或等价注入，保证 `runtime_env` 生效
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
> - 该节是历史约束；若与 2026-05-10 DFR-25 mainline 冲突，以 DFR-25 minimal-variable 约束为准
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

- [ ] **MAIN-MODULE-CONTRIB-01**：如需继续模块贡献分析，必须以当前 canonical `DFR-25 ResNeXt 256x8 decision fusion` 为锚点，先回答 `equal-weight`、`minimal learned`、`current learned` 三档的净贡献与方差差异，再决定是否继续拆 richer reliability module。
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

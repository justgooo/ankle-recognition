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
| 上次实验 | Optuna / monitor workflow hardening（统一 `val_acc` 主指标、`val_auc` 平手决胜；`freeze_layers` 显式配置；fresh/resume 语义收紧；monitor 将 threshold_eval 故障降级为 warning） |
| 上次结果 | keep（代码与 workflow 已对齐当前主线：ResUNet encoder + 3 个 Attention Gate + 单视角 16 张切片 AttentionPooling + 3 视角 VRG / decision fusion；新一轮比较以 `summary.json -> best_val.accuracy` 为准，平手按 `best_val.auc` 决胜） |
| 下一步 | 先跑 1 次当前 canonical baseline / smoke train，重新确立 `val_acc` 基线；随后再基于同一配置启动 fresh Optuna proxy study，避免与历史 192x16 / no_miss 记录混比 |
| 连续 discard 计数 | 0（当前工作以流程修复为主，不计入实验 discard） |
| 累计 proxy keep 数 | 6（历史记录保留；等待新 `val_acc` 体系下重新建账） |

---

## 当前最优纪录

> 当前 canonical baseline 已切换为：**ResUNet encoder + 3 个 Attention Gate + AttentionPooling + VRG / decision fusion**。
> 旧的 `no_miss_*` / `192x16` / 早期 stage-10 结果只作为 **legacy reference only**，不再直接参与当前 keep/discard 比较。

| 指标 | 值 | 来源 commit | 配置 | 备注 |
|------|---:|-------------|------|------|
| **val_acc** | **待重新确立** | — | `configs/autoresearch_proxy.yaml` 当前主线 | 必须先跑 1 次当前 canonical baseline，读取 `summary.json -> best_val.accuracy` |
| **tie-break** | **val_auc** | — | 统一规则 | 当 `val_acc` 持平时，按 `best_val.auc` 决胜 |
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

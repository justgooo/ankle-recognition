# 实验待办清单 (Experiment Backlog)

> **本文件由 autoresearch Agent 维护。**
> Agent 在每次实验循环启动时必须阅读本文件，并在完成实验后根据结果更新。
> 人类也可以直接编辑本文件来添加想法或调整优先级。

---

## Agent 状态

> Agent 每次实验后必须更新此表。新 Agent 启动时以此表为起点。

| 字段 | 值 |
|------|-----|
| 上次实验 | AP-FF-03：Feature Fusion + AttentionPooling lr=7e-5 proxy |
| 上次结果 | discard（no_miss_val_acc=0.553, no_miss_val_spe=0.160, val_AUC=0.893） |
| 下一步 | AP-FF-04：Feature Fusion + AttentionPooling dropout=0.2 proxy（按规则切换超参维度） |
| 连续 discard 计数 | 3 |
| 累计 proxy keep 数 | 0（阶段 5 尚未出现 keep） |

---

## 当前最优纪录

| 指标 | 值 | 来源 commit | 配置 |
|------|---:|-------------|------|
| **零漏诊 val_Acc** | **0.787** | `7b184e0` | Decision, non-shared, lr=1e-4, d=0.3, aug=true, ls=0.0 |
| 零漏诊 val_Spe | 0.600 | `7b184e0` | 同上 |
| 不漏诊阈值 | 0.155 | `7b184e0` | 同上 |
| val_AUC (参考) | 0.940 | `7b184e0` | 同上 |

---

## 用户定向 rerun Campaign 🔴 当前最高优先级

> 本轮会话覆盖常规 backlog 优先级，只执行 Plan 2 / Plan 6 的定向 rerun。
> 判定规则（仅本 campaign 生效）：以 `threshold_eval.json` 为准，**仅当 `no_miss_val_acc > 0.75` 时记为 keep**；否则记为 discard。
> 固定顺序：先 `R2-01 → R2-08`，再 `R6-01 → R6-08`。

### Plan 2：View Reliability Gating（8 次 proxy rerun）

- [ ] **R2-01**: baseline（fusion_type=decision, share_backbone=false, aug=true, lr=1e-4, dropout=0.3, label_smoothing=0.0）
- [ ] **R2-02**: lr=5e-5
- [ ] **R2-03**: lr=7e-5
- [ ] **R2-04**: dropout=0.2
- [ ] **R2-05**: dropout=0.4
- [ ] **R2-06**: label_smoothing=0.05
- [ ] **R2-07**: lr=5e-5 + label_smoothing=0.05
- [ ] **R2-08**: 基于前 7 次 rerun 最佳方向的组合实验

### Plan 6：Asymmetric Safety-Biased Fusion（8 次 proxy rerun）

- [ ] **R6-01**: temperature=0.5 baseline（fusion_type=decision, share_backbone=false, aug=true, lr=1e-4, dropout=0.3, label_smoothing=0.0）
- [ ] **R6-02**: temperature=1.0
- [ ] **R6-03**: temperature=0.3
- [ ] **R6-04**: temperature=2.0
- [ ] **R6-05**: temperature=0.5 + lr=5e-5
- [ ] **R6-06**: temperature=0.5 + dropout=0.2
- [ ] **R6-07**: temperature=0.5 + label_smoothing=0.05
- [ ] **R6-08**: 基于前 7 次 rerun 最佳方向的组合实验

---

## 阶段 5：AttentionPooling 增强 🔴 当前执行中

> 将 Feature Fusion 和 Decision Fusion 的切片聚合从 mean pooling 升级为 AttentionPooling。
> 通过 YAML 配置 `model.use_attention_pooling: true` 启用。
> 基线配置：`share_backbone=false, aug=true, lr=1e-4, label_smoothing=0.0, dropout=0.3`

### 阶段 5A：Feature Fusion + AttentionPooling（8 次 proxy）

- [x] **AP-FF-01**: baseline（fusion_type=feature, use_attention_pooling=true, lr=1e-4, dropout=0.3）→ no_miss_val_acc=0.574, val_AUC=0.910 → discard
- [x] **AP-FF-02**: lr=5e-5 → no_miss_val_acc=0.532, val_AUC=0.939 → discard
- [x] **AP-FF-03**: lr=7e-5 → no_miss_val_acc=0.553, val_AUC=0.893 → discard
- [ ] **AP-FF-04**: dropout=0.2
- [ ] **AP-FF-05**: dropout=0.4
- [ ] **AP-FF-06**: label_smoothing=0.05
- [ ] **AP-FF-07**: lr=5e-5 + label_smoothing=0.05
- [ ] **AP-FF-08**: 基于前 7 次最佳方向的组合实验

### 阶段 5B：Decision Fusion + AttentionPooling（8 次 proxy）

- [ ] **AP-DF-01**: baseline（fusion_type=decision, use_attention_pooling=true, lr=1e-4, dropout=0.3）
- [ ] **AP-DF-02**: lr=5e-5
- [ ] **AP-DF-03**: lr=7e-5
- [ ] **AP-DF-04**: dropout=0.2
- [ ] **AP-DF-05**: dropout=0.4
- [ ] **AP-DF-06**: label_smoothing=0.05
- [ ] **AP-DF-07**: lr=5e-5 + label_smoothing=0.05
- [ ] **AP-DF-08**: 基于前 7 次最佳方向的组合实验

---

## 旧阶段（已完成）

### 阶段 1：Feature Fusion 超参优化（已完结）

- 暂无高优先级待办

### 阶段 2：Decision Fusion 对比（已完结）

- 暂无高优先级待办

### 阶段 3：论文就绪验证

- [ ] **多 seed 验证**（seed=42, 123, 456）
- [ ] **Bootstrap 置信区间**（1000 次重采样）
- [ ] **三种融合方式 formal 对比**

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
- [x] **Decision Fusion proxy 去掉 ls（lr=1e-4）→ no_miss_val_acc=0.787 → keep ✅ 当前最优**
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

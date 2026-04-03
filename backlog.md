# 实验待办清单 (Experiment Backlog)

> **本文件由 autoresearch Agent 维护。**
> Agent 在每次实验循环启动时必须阅读本文件，并在完成实验后根据结果更新。
> 人类也可以直接编辑本文件来添加想法或调整优先级。

---

## Agent 状态

> Agent 每次实验后必须更新此表。新 Agent 启动时以此表为起点。

| 字段 | 值 |
|------|-----|
| 上次实验 | VRG-10：View Reliability Gating 组合实验（lr=5e-5, label_smoothing=0.05）proxy |
| 上次结果 | discard（no_miss_val_acc=0.670, no_miss_val_spe=0.380, val_AUC=0.951） |
| 下一步 | ASF-01：Asymmetric Safety-Biased Fusion baseline（temperature=0.5）proxy |
| 连续 discard 计数 | 0（新架构重置） |
| 累计 proxy keep 数 | 0（新阶段重置） |

---

## 当前最优纪录

| 指标 | 值 | 来源 commit | 配置 |
|------|---:|-------------|------|
| **零漏诊 val_Acc** | **0.787** | `7b184e0` | Decision, non-shared, lr=1e-4, d=0.3, aug=true, ls=0.0 |
| 零漏诊 val_Spe | 0.600 | `7b184e0` | 同上 |
| 不漏诊阈值 | 0.155 | `7b184e0` | 同上 |
| val_AUC (参考) | 0.940 | `7b184e0` | 同上 |

---

## 阶段 1：Feature Fusion 超参优化

### 🔴 高优先级

- 暂无（`augmentation + label_smoothing=0.1` 已完成并 discard；Feature Fusion 已连续 5 次 discard，建议只保留 formal-only cosine 重评或转向 Decision Fusion formal 触发条件）

### 🟡 中优先级

- 暂无

### 🟢 低优先级

- [ ] **CosineAnnealingLR 在 formal 阶段重新评估**
  - 4 epoch proxy 阶段 cosine 效果不好（太短），但 15 epoch formal 下可能有效
  - 配置：`scheduler: cosine, scheduler_t_max: 15, scheduler_eta_min: 1e-6`

- 暂无（保守组合的 `gradient_clip_norm=1.0 / 0.5` 与 `early_stopping_patience=5` 均已完成并 discard；Feature Fusion proxy 暂无剩余高信号组合）

---

## 阶段 2：Decision Fusion 对比

> 在 Feature Fusion 超参优化告一段落后执行。
> 用 Feature Fusion 找到的最佳训练策略，直接切换 `fusion_type: decision` 跑对比。

### 🔴 高优先级

- 暂无（Decision Fusion 的 `fusion_hidden_dim=128 / 256 / 384` 已补齐，256 仍为最佳 head 宽度）

### 🟡 中优先级

- [ ] **Decision Fusion formal 确认**
  - 触发条件：再出现 1 次 proxy keep，或未来单次 proxy 提升超过 `0.03`
  - 默认沿用当前 best commit `7b184e0` 的训练策略

---

## 阶段 3：论文就绪验证

- [ ] **多 seed 验证**（seed=42, 123, 456）
  - 对最终最优配置跑 3 次，报告均值 ± 标准差
- [ ] **Bootstrap 置信区间**（1000 次重采样）
- [ ] **三种融合方式 formal 对比**
  - Feature / Decision / Attention 使用相同训练策略
  - 统一使用零漏诊阈值评估，生成论文用表格

---

## 阶段 4：创新融合架构探索

> 超参搜索已连续 12 次 discard，通过结构创新打破瓶颈。

### 阶段 4A：方案 2 — 视角可靠度门控 (View Reliability Gating) ✅ 已完成

> 将 Decision Fusion 中固定的全局视角权重替换为基于输入的动态 confidence 门控。
> 基线：`share_backbone=false, aug=true, lr=1e-4, label_smoothing=0.0, dropout=0.3`

- 暂无（10 个 VRG proxy 实验已完成；最佳信号来自 VRG-02 的高 AUC 与 VRG-07 的较好零漏诊校准，但组合实验 VRG-10 仍未超过当前 best 0.787，转向阶段 4B）

### 阶段 4B：方案 6 — 非对称安全融合 (Asymmetric Safety-Biased Fusion) 🔴 当前执行中

> 异常 logit 使用 temperature-scaled logsumexp 融合，直接对齐"宁可误诊不可漏诊"的临床需求。
> model.py 已切换为方案 6 实现。temperature 在 `model.py` 的 `DEFAULT_TEMPERATURE` 中修改。

- [ ] ASF-01: temperature=0.5
- [ ] ASF-02: temperature=1.0
- [ ] ASF-03: temperature=0.3
- [ ] ASF-04: temperature=2.0
- [ ] ASF-05: temperature=0.5 + lr=5e-5
- [ ] ASF-06: temperature=0.5 + dropout=0.2
- [ ] ASF-07: temperature=0.5 + label_smoothing=0.05
- [ ] ASF-08: temperature=1.0 + lr=7e-5
- [ ] ASF-09: temperature=0.3 + dropout=0.4
- [ ] ASF-10: 基于前 9 次最佳方向的组合实验

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

### Feature Fusion — 非共享 backbone 阶段（Round 2, 进行中）

- [x] 非共享 baseline (lr=1e-4, d=0.3) → no_miss_val_acc=0.755 → keep
- [x] 非共享 lr=5e-5 → no_miss_val_acc=0.660 → discard
- [x] 非共享 dropout=0.4 → no_miss_val_acc=0.691 → discard
- [x] 非共享 lr=7e-5 → no_miss_val_acc=0.691 → discard
- [x] 非共享 + augmentation=true → no_miss_val_acc=0.777 → keep
- [x] 非共享 + augmentation=true formal 确认（15 epochs / 256px / 32 slices）→ no_miss_val_acc=0.755, no_miss_val_spe=0.540, val_AUC=0.946 → keep
- [x] 非共享 + augmentation=true + gradient_clip_norm=1.0 → no_miss_val_acc=0.766, no_miss_val_spe=0.560, val_AUC=0.948 → discard
- [x] **非共享 + augmentation=true + label_smoothing=0.05 → no_miss_val_acc=0.777（持平）, val_AUC=0.948（更优）→ keep ✅ Feature Fusion 最优**
- [x] 非共享 + augmentation=true + lr=5e-5（无 label smoothing）→ no_miss_val_acc=0.745, val_AUC=0.947 → discard
- [x] 非共享 + augmentation=true + label_smoothing=0.05 + weight_decay=5e-4 → no_miss_val_acc=0.681, val_AUC=0.933 → discard
- [x] 非共享 + augmentation=true + label_smoothing=0.05 + dropout=0.4 → no_miss_val_acc=0.777（持平）, val_AUC=0.940 → discard（AUC 低于当前最优 0.948）
- [x] 非共享 + augmentation=true + label_smoothing=0.1 → no_miss_val_acc=0.777（持平）, val_AUC=0.940 → discard（AUC 低于当前最优 0.948）
- [x] 非共享 + augmentation=true + label_smoothing=0.05 + gradient_clip_norm=1.0 → no_miss_val_acc=0.691, val_AUC=0.934 → discard（联合正则明显退化）
- [x] 非共享 + augmentation=true + label_smoothing=0.05 + gradient_clip_norm=0.5 → no_miss_val_acc=0.585, val_AUC=0.917 → discard（较 1.0 clip 更差，零漏诊阈值几乎塌陷）
- [x] 非共享 + augmentation=true + label_smoothing=0.05 + early_stopping_patience=5 → no_miss_val_acc=0.734, val_AUC=0.946 → discard（4 epoch proxy 下 `patience > epochs`，基本无实际影响）

### Decision Fusion 阶段（Round 3, 进行中）

- [x] Decision Fusion proxy baseline（share_backbone=false, aug=true, label_smoothing=0.05）→ no_miss_val_acc=0.585, val_AUC=0.949 → discard（AUC 高但零漏诊阈值过低，校准差）
- [x] Decision Fusion proxy lr=5e-5（share_backbone=false, aug=true, label_smoothing=0.05）→ no_miss_val_acc=0.617, val_AUC=0.930 → discard
- [x] Decision Fusion proxy lr=7e-5（share_backbone=false, aug=true, label_smoothing=0.05）→ no_miss_val_acc=0.638, val_AUC=0.949 → discard
- [x] **Decision Fusion proxy 去掉 label smoothing（share_backbone=false, aug=true, lr=1e-4）→ no_miss_val_acc=0.787, no_miss_val_spe=0.600, val_AUC=0.940 → keep ✅ 当前最优**
- [x] Decision Fusion proxy lr=3e-4（share_backbone=false, aug=true, label_smoothing=0.0）→ no_miss_val_acc=0.755, no_miss_val_spe=0.540, val_AUC=0.901 → discard
- [x] Decision Fusion proxy dropout=0.2（share_backbone=false, aug=true, lr=1e-4, label_smoothing=0.0）→ no_miss_val_acc=0.755, no_miss_val_spe=0.540, val_AUC=0.959 → discard（AUC 提升，但零漏诊 acc 低于当前最优 0.787）
- [x] Decision Fusion proxy dropout=0.4（share_backbone=false, aug=true, lr=1e-4, label_smoothing=0.0）→ no_miss_val_acc=0.638, no_miss_val_spe=0.320, val_AUC=0.909 → discard（明显退化，dropout 过高）
- [x] Decision Fusion proxy + gradient_clip_norm=0.5（share_backbone=false, aug=true, lr=1e-4, label_smoothing=0.0）→ no_miss_val_acc=0.713, no_miss_val_spe=0.460, val_AUC=0.925 → discard（额外梯度裁剪降低了零漏诊阈值表现）
- [x] Decision Fusion proxy `fusion_hidden_dim=128`（share_backbone=false, aug=true, lr=1e-4, label_smoothing=0.0）→ no_miss_val_acc=0.553, no_miss_val_spe=0.160, val_AUC=0.931 → discard（更小 head 明显欠拟合，零漏诊阈值表现大幅下降）
- [x] Decision Fusion proxy 6 epoch 复核（share_backbone=false, aug=true, lr=1e-4, label_smoothing=0.0）→ no_miss_val_acc=0.691, no_miss_val_spe=0.420, val_AUC=0.932 → discard（延长 proxy 未能改善零漏诊 acc，4 epoch 信号并非主要问题）
- [x] Decision Fusion proxy `fusion_hidden_dim=384`（share_backbone=false, aug=true, lr=1e-4, label_smoothing=0.0）→ no_miss_val_acc=0.660, no_miss_val_spe=0.360, val_AUC=0.928 → discard（更宽 head 同样退化，256 仍是最佳 hidden_dim）
- [x] Decision Fusion proxy `scheduler=cosine, scheduler_t_max=4, scheduler_eta_min=1e-6`（share_backbone=false, aug=true, lr=1e-4, label_smoothing=0.0）→ no_miss_val_acc=0.734, no_miss_val_spe=0.500, val_AUC=0.941 → discard（4 epoch proxy 上 cosine 仍未改善零漏诊阈值表现，未能触发 formal）
- [x] Decision Fusion proxy 概率空间加权投票（share_backbone=false, aug=true, lr=1e-4, label_smoothing=0.0）→ no_miss_val_acc=0.691, no_miss_val_spe=0.420, val_AUC=0.918 → discard（概率投票虽然更符合决策融合直觉，但零漏诊阈值抬高到 0.301，验证集零漏诊 acc 明显退化）

### 阶段 4A：View Reliability Gating 阶段（Round 4, 已完成）

- [x] **VRG-01: baseline**（share_backbone=false, aug=true, lr=1e-4, dropout=0.3, label_smoothing=0.0） → no_miss_val_acc=0.681, no_miss_val_spe=0.400, val_AUC=0.941 → discard（低于当前 best 0.787）
- [x] **VRG-02: lr=5e-5**（share_backbone=false, aug=true, dropout=0.3, label_smoothing=0.0） → no_miss_val_acc=0.755, no_miss_val_spe=0.540, val_AUC=0.958 → discard（AUC 明显升高，但零漏诊 val_acc 仍低于当前 best 0.787）
- [x] **VRG-03: lr=7e-5**（share_backbone=false, aug=true, dropout=0.3, label_smoothing=0.0） → no_miss_val_acc=0.755, no_miss_val_spe=0.540, val_AUC=0.940 → discard（与 VRG-02 相同的零漏诊表现，但 AUC 回落，仍低于当前 best 0.787）
- [x] **VRG-04: lr=3e-4**（share_backbone=false, aug=true, dropout=0.3, label_smoothing=0.0） → no_miss_val_acc=0.585, no_miss_val_spe=0.220, val_AUC=0.885 → discard（大学习率明显破坏校准与零漏诊阈值表现，较当前 best 0.787 大幅退化）
- [x] **VRG-05: dropout=0.2**（share_backbone=false, aug=true, lr=1e-4, label_smoothing=0.0） → no_miss_val_acc=0.617, no_miss_val_spe=0.280, val_AUC=0.931 → discard（较当前 best 0.787 明显退化；虽较 VRG-04 恢复部分校准，但零漏诊阈值表现仍偏弱）
- [x] **VRG-06: dropout=0.4**（share_backbone=false, aug=true, lr=1e-4, label_smoothing=0.0） → no_miss_val_acc=0.479, no_miss_val_spe=0.020, val_AUC=0.928 → discard（相较当前 best 0.787 明显退化；阈值降到 0.069 后特异度几乎归零，说明更高 dropout 严重破坏零漏诊校准）
- [x] **VRG-07: label_smoothing=0.05**（share_backbone=false, aug=true, lr=1e-4, dropout=0.3） → no_miss_val_acc=0.745, no_miss_val_spe=0.520, val_AUC=0.943 → discard（较当前 best 0.787 仍有差距；零漏诊阈值升至 0.295，特异度回升到 0.520，但零漏诊 acc 仍未超过最佳 Decision Fusion）
- [x] **VRG-08: weight_decay=5e-4**（share_backbone=false, aug=true, lr=1e-4, dropout=0.3, label_smoothing=0.0） → no_miss_val_acc=0.489, no_miss_val_spe=0.040, val_AUC=0.916 → discard（较当前 best 0.787 大幅退化；零漏诊阈值降到 0.091，特异度几乎清零，说明更强 L2 正则显著破坏了 VRG 的校准）
- [x] **VRG-09: scheduler=cosine, T_max=4, eta_min=1e-6**（share_backbone=false, aug=true, lr=1e-4, dropout=0.3, label_smoothing=0.0） → no_miss_val_acc=0.574, no_miss_val_spe=0.200, val_AUC=0.940 → discard（较当前 best 0.787 明显退化；虽然 AUC 回到 0.940，但零漏诊 acc 进一步下滑，说明 4 epoch proxy 下 cosine 调度未改善 VRG 的阈值校准）
- [x] **VRG-10: lr=5e-5 + label_smoothing=0.05**（share_backbone=false, aug=true, dropout=0.3） → no_miss_val_acc=0.670, no_miss_val_spe=0.380, val_AUC=0.951 → discard（AUC 仍较高，但零漏诊阈值降到 0.166 后 FP 明显增多，未能同时继承 VRG-02 的高 AUC 与 VRG-07 的零漏诊校准优势）

### Attention Fusion 阶段（旧实验，已完结）

- [x] 约 17 组 proxy 实验 → 已转入 Feature Fusion 为主力

---

## 维护规则

1. **Agent 每次启动时**：先读本文件，了解当前最优纪录和待办优先级
2. **每完成一个实验后**：
   - 如果 keep：更新"当前最优纪录"表格
   - 将实验从待办移到"已完成"，标注结果
   - 如果结果带来新的实验思路，添加到对应优先级
3. **连续 3 个 discard 后**：重新审视待办清单，考虑换方向
4. **人类编辑后**：Agent 下次启动时以文件内容为准





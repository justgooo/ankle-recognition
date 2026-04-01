# 用于踝关节识别的自动研究流程

这个仓库已经为踝关节 CT 分类器配置好了 autoresearch 风格的实验流程。
整体思路与 `karpathy/autoresearch` 相同，但适配了这个医学影像项目、
它的评估指标，以及当前这台 Windows + NVIDIA CUDA 工作站环境。

本文中的命令默认都在仓库根目录的 PowerShell 中执行，并统一使用
`.\.venv\Scripts\python.exe`。不要直接使用系统 PATH 里的 `python.exe`：
当前系统 Python 没有安装 `torch`，训练环境在项目 `.venv` 里。

## 目标

主目标：在**验证集 FN=0（零漏诊）约束**下，最大化验证集准确率。

具体做法：
1. 训练完成后，对 best.pt 在验证集上计算每个样本的正类概率
2. 找到「零漏诊阈值」= 验证集所有阳性样本中的最小正类概率
3. 用该阈值计算验证集 Accuracy
4. 以该 Accuracy 作为模型选择的主指标

辅助指标（跟踪但不作为主选择标准）：
- `best_val.auc`
- `best_val.f1`
- 零漏诊阈值下的验证集 Specificity

保留或丢弃实验时，使用零漏诊 Accuracy 做主决策，AUC 做辅助参考。
**绝对不要**用测试集指标来做模型选择。

## 研究策略

**主力方案**：特征融合 (Feature Fusion) + 决策融合 (Decision Fusion)
**补充分析**：注意力融合 (Attention Fusion)，在主力方案稳定后做 1-2 组 formal 对比

工作流程：
1. 先以 Feature Fusion 做主轮 autoresearch 循环
2. 找到最佳训练策略后，复用同样策略跑 Decision Fusion formal 确认
3. 在论文准备阶段，用相同策略跑 Attention Fusion formal 一组，作为补充对比
4. 最终论文报告三种融合方式在相同训练策略下的对比结果

## 硬件限制

- 操作系统：Windows，Shell：PowerShell
- CPU：13th Gen Intel(R) Core(TM) i5-13490F（10 核 / 16 线程）
- GPU：NVIDIA GeForce RTX 2070 SUPER（8GB 显存，CUDA 可用，驱动 581.80）
- 系统内存：32GB
- 训练默认走 `.venv` 中的 PyTorch CUDA 环境（`torch 2.10.0+cu130`，`device: auto` 会选 `cuda`）
- proxy 实验（4 epochs，feature fusion，非共享 backbone）预计：peak_vram ≈ 2 GB，约 30 分钟
- formal 实验（15 epochs，feature fusion，非共享 backbone）预计：peak_vram ≈ 5.5 GB，约 90-120 分钟
- 在当前 8GB 显存设备上，`batch_size=2` 已验证可稳定运行
- 每次实验前确认 `runs/` 下没有残留的大 checkpoint 文件，并顺手检查 C 盘剩余空间

## 准备工作

开始一次新的运行时，需要与用户一起完成以下事项：

1. 根据本地日期确定一个运行标签，例如 `2026-04-01-ankle-feature`。
2. 基于当前主分支创建一个新的分支：
   - `git checkout -b autoresearch/<tag>`
3. 先确认训练环境可用：
   - `Test-Path .\.venv\Scripts\python.exe`
   - `.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"`
   - 如果 `torch.cuda.is_available()` 不是 `True`，先停下来修环境，不要盲跑 CPU
4. 阅读以下文件获取完整上下文：
   - `README.md`
   - `train.py`
   - `src/model.py`
   - `configs/autoresearch_proxy.yaml`
   - `configs/autoresearch_formal.yaml`
5. 确认数据已经准备就绪：
   - 必须存在 `data/realdata/metadata.csv`
   - 其中包含 `patient_id`、`label`、`axial_dir`、`coronal_dir`、`sagittal_dir`
   - `split` 字段已固定为 `train` / `val` / `test`
6. 如果 `results.tsv` 不存在，初始化它（只含表头）。
7. 确认以上准备完成后，再开始实验。

## 允许修改的范围

你可以修改：
- `src/model.py`
- `src/attention_pooling.py`
- `src/cross_view_attention.py`
- `configs/autoresearch_proxy.yaml`
- `configs/autoresearch_formal.yaml`

你可以追加写入：
- `results.tsv`

不要修改：
- `train.py`（训练策略增强已内置，通过配置控制）
- `src/dataset.py`
- `src/utils.py`
- `tools/`
- 任何 CSV 数据文件
- 评估逻辑

不要新增依赖。

## 基线

第一次运行必须是没有任何实验性改动的基线版本：

```powershell
.\.venv\Scripts\python.exe train.py --config configs/autoresearch_proxy.yaml > run.log 2>&1
```

然后读取：
- `runs/autoresearch_proxy/summary.json`
- `run.log`

接着运行阈值评估：
```powershell
.\.venv\Scripts\python.exe tools/evaluate_threshold.py --run_dir runs/autoresearch_proxy --config configs/autoresearch_proxy.yaml
```

然后读取：
- `runs/autoresearch_proxy/threshold_eval.json`

`train.py` 日志行：
- `best_val_auc=...`
- `best_val_f1=...`
- `best_val_accuracy=...`
- `peak_vram_mb=...`
- `total_seconds=...`

`evaluate_threshold.py` 日志行：
- `no_miss_threshold=...`
- `no_miss_val_acc=...`
- `no_miss_val_spe=...`

在开始长时间无人值守运行之前：
- 先确保手动执行的一次基线运行能够顺利完成
- 前 3 到 4 个实验要先盯着看
- 只有当整个循环已经明显运行稳定时，才开始隔夜运行

## 结果文件

`results.tsv` 使用制表符分隔，表头如下：

```tsv
commit	val_auc	val_f1	no_miss_threshold	no_miss_val_acc	no_miss_val_spe	memory_gb	status	config	description
```

规则：
- `commit`：短 git hash
- `val_auc`：如果崩溃，记为 `0.000000`
- `val_f1`：如果崩溃，记为 `0.000000`
- `no_miss_threshold`：验证集零漏诊阈值；如果崩溃或未计算，记为空
- `no_miss_val_acc`：零漏诊阈值下的验证集准确率；如果崩溃或未计算，记为空
- `no_miss_val_spe`：零漏诊阈值下的验证集特异度；如果崩溃或未计算，记为空
- `memory_gb`：用 `runtime.peak_vram_mb` 除以 `1024`，保留一位小数；如果崩溃则记为 `0.0`
- `status`：只能是 `keep`、`discard`、`crash` 之一
- `config`：`proxy` 或 `formal`
- `description`：实验的简短描述

旧的 results.tsv 记录保留，新实验追加到末尾。
旧行没有 `no_miss_*` 列是正常的。

不要提交 `results.tsv`。

## 实验循环

完成准备后，持续循环执行：

1. 检查当前分支和提交。
2. 在允许修改的范围内做一项实验性改动。
3. 在运行前先提交这次实验改动。
4. 运行 proxy 实验：
   - `.\.venv\Scripts\python.exe train.py --config configs/autoresearch_proxy.yaml > run.log 2>&1`
5. 如果运行崩溃：
   - 用 `Get-Content run.log -Tail 30` 查看最后几行（不要读整个日志）
   - OOM → 标记为 `crash`
   - 代码 bug → 修复后重跑一次（最多重试 1 次）
   - 数据问题 → 立即停止
   - 其他 → 标记为 `crash`，写入 `results.tsv`，回退
6. 如果运行成功：
   - 读取 `runs/autoresearch_proxy/summary.json`
   - 运行阈值评估：`.\.venv\Scripts\python.exe tools/evaluate_threshold.py --run_dir runs/autoresearch_proxy --config configs/autoresearch_proxy.yaml`
   - 读取 `runs/autoresearch_proxy/threshold_eval.json`
   - 将零漏诊 `val_acc` 与当前保留的最佳结果比较
   - 只有当零漏诊 `val_acc` 提升时，才保留这个提交
   - 如果零漏诊 `val_acc` 持平，优先选择 `val_auc` 更高的版本
   - 如果两者都持平，优先选择更简单的代码
   - 否则回退这次实验
7. 每出现 3 次 proxy 胜出后，做一次 formal 确认：
   - `.\.venv\Scripts\python.exe train.py --config configs/autoresearch_formal.yaml > formal.log 2>&1`
   - `.\.venv\Scripts\python.exe tools/evaluate_threshold.py --run_dir runs/autoresearch_formal --config configs/autoresearch_formal.yaml`
   - 如果 formal 也提升了，视为新的 formal 最优结果
   - 如果 proxy 提升但 formal 退化，优先保留上一个 formal 最优
   - formal 结果也记入 `results.tsv`
   - 额外：如果 proxy 零漏诊 acc 单次提升超过 0.03，可以立即做 formal

## 超时规则

- proxy 实验（4 epochs）预计约 **30-35 分钟**
- 超时阈值：**60 分钟**
- formal 实验（15 epochs）预计约 **90-120 分钟**
- 超时阈值：**180 分钟**
- 使用以下方式监控运行时间：
  ```powershell
  $proc = Start-Process .\.venv\Scripts\python.exe -ArgumentList "train.py","--config","configs/autoresearch_proxy.yaml" -RedirectStandardOutput run.log -RedirectStandardError err.log -PassThru
  if (-not $proc.WaitForExit(3600000)) { $proc.Kill(); Write-Host "TIMEOUT" }
  ```

## 简洁性原则

在零漏诊 val_acc 相同或差距极小（< 0.005）时：
- 代码行数更少的版本胜出
- 微小提升（< 0.003）但增加了大量复杂代码？放弃
- 永远追求更少的特殊逻辑、更干净的模型结构

## 实验优先级

以 Feature Fusion 为主，优先做高信号、低风险的实验：

1. 学习率调优（5e-5 / 7e-5 / 1e-4 / 3e-4）
2. Dropout 比率（0.2 / 0.3 / 0.4 / 0.5）
3. CosineAnnealingLR 调度器（搭配 eta_min 参数）
4. 数据增强开关（`train.augmentation: true`）
5. 梯度裁剪（`train.gradient_clip_norm: 1.0 / 0.5`）
6. Label Smoothing（`train.label_smoothing: 0.05 / 0.1`）
7. EarlyStopping（`train.early_stopping_patience: 5 / 10 / 15`）
8. 权重衰减（`train.weight_decay: 1e-4 / 5e-4 / 1e-3`）
9. `fusion_hidden_dim`（128 / 256 / 384）
10. 增加 proxy epochs 至 6（如果 4 epochs 信号不稳定）

当 Feature Fusion 优化告一段落后，对 Decision Fusion 做对比：
- 用 Feature Fusion 找到的最佳超参（lr、dropout、scheduler 等）
- 直接切换 `fusion_type` 为 `decision`，跑一组 formal 确认

避免一次改动太多内容。

## 可使用的训练策略配置项

以下配置项已内置于 train.py，通过 YAML 配置控制：

```yaml
train:
  scheduler: cosine              # 学习率调度器：cosine / none（默认 none）
  scheduler_t_max: 15            # CosineAnnealing 的 T_max（默认 = epochs）
  scheduler_eta_min: 0.0         # CosineAnnealing 最小学习率（默认 0.0）
  gradient_clip_norm: 1.0        # 梯度裁剪阈值（默认不启用）
  early_stopping_patience: 10    # 早停的耐心值（默认不启用）
  label_smoothing: 0.1           # 标签平滑（默认 0.0）
  augmentation: true             # 数据增强：随机翻转+旋转±10°（默认 false）
```

## 如果陷入瓶颈

当连续 5 个以上实验都是 discard 时：
1. 重新阅读所有可修改的源文件，寻找被忽略的维度
2. 检查 `results.tsv` 中所有 keep 的实验，寻找可组合的改进
3. 尝试不同的训练策略组合（scheduler + augmentation + clip）
4. 切换到 Decision Fusion 看是否能打破瓶颈
5. 回顾基础超参是否合理
6. 尝试不同的 dropout 策略或正则化手段

不要在同一个方向反复尝试超过 3 次。

## 上下文管理

- 读日志时只读最后 30 行：`Get-Content run.log -Tail 30`
- 不要把整个日志或完整代码文件粘贴到对话中
- 只在排错时才读更多内容
- 每次实验只关注：`summary.json` + `threshold_eval.json` 的指标 + 简短的 diff 描述
- 大约每 15-20 次实验后，Agent 上下文可能接近饱和，此时需要重启 Agent

## 自主运行规则

**一旦进入实验循环，不要停下来询问是否继续。**
你是完全自主的研究者。如果跑完一轮还有思路，就继续下一轮。
只有当人类手动中断你时才停止。
如果实验思路用尽了，重新阅读代码和文档寻找新方向。
**永远不要主动暂停**。人类可能在睡觉或不在电脑前，期望你持续工作。

## 安全规则

- 绝不要用测试集指标来选择实验。
- 实验过程中绝不要修改数据集划分。
- 绝不要在实验中覆盖或重新标注元数据。
- 如果工作区里存在与当前任务无关的用户改动，不要回退它们。
- 如果卡住了，继续寻找新的模型侧思路，而不是去改数据或评估逻辑。

## 资源保护

- 如果系统内存使用率超过 90%，标记为 crash 并回退
- 如果训练 loss 在前 2 个 epoch 完全没有下降，可以提前终止该实验
- 每次实验结束后确认 `runs/` 目录下没有残留的大 checkpoint 积累
- 定期检查磁盘空间：`Get-PSDrive C | Select-Object Free`

## 论文就绪验证阶段

当 autoresearch 循环找到最优配置后，进入论文就绪验证：
1. 用 3 个不同 seed（42, 123, 456）分别训练，报告均值 ± 标准差
2. 对测试集做 Bootstrap 置信区间（1000 次重采样）
3. 用最佳训练策略分别跑 Feature / Decision / Attention 三种融合的 formal
4. 统一使用零漏诊阈值评估，生成论文用表格

## 实践经验

- 大多数实验都会失败或被丢弃，这是正常现象。
- 关键杠杆点是 `program.md`，不是在每次运行之间随意手工调整。
- 说明应该具体、简单，并且始终围绕可衡量的零漏诊 val_acc 变化。
- 如果某条实验路线反复失败，就换个方向，不要硬推下去。

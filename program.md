# 用于踝关节识别的自动研究流程

这个仓库已经为踝关节 CT 分类器配置好了 autoresearch 风格的实验流程。
整体思路与 `karpathy/autoresearch` 相同，但适配了这个医学影像项目、
它的评估指标，以及当前这台 Windows + NVIDIA CUDA 工作站环境。

本文中的命令默认都在仓库根目录的 PowerShell 中执行，并统一使用
`.\.venv\Scripts\python.exe`。不要直接使用系统 PATH 里的 `python.exe`：
当前系统 Python 没有安装 `torch`，训练环境在项目 `.venv` 里。

## 目标

最大化 `runs/autoresearch_proxy/summary.json -> best_val.auc`。

次要指标：
- `best_val.f1`

保留或丢弃实验时，只使用验证集指标做决策。
不要**绝对不要**用测试集指标来做模型选择。

## 硬件限制

- 操作系统：Windows，Shell：PowerShell
- CPU：13th Gen Intel(R) Core(TM) i5-13490F（10 核 / 16 线程）
- GPU：NVIDIA GeForce RTX 2070 SUPER（8GB 显存，CUDA 可用，驱动 581.80）
- 系统内存：32GB
- 训练默认走 `.venv` 中的 PyTorch CUDA 环境（`torch 2.10.0+cu130`，`device: auto` 会选 `cuda`）
- 当前 `autoresearch_proxy` 最近一次实测：`peak_vram_mb ≈ 1782.6`，`total_seconds ≈ 1708.8`（约 28.5 分钟）
- 当前 `autoresearch_formal` 最近一次实测：`peak_vram_mb ≈ 5725.3`，`total_seconds ≈ 5964.6`（约 99.4 分钟）
- 在当前 8GB 显存设备上，`batch_size=2` 已验证可稳定运行；如果你增大 `image_size`、`num_slices_per_view` 或模型规模，优先先降到 1
- 在 attention + formal（256 / 32 slices / 10 epochs）配置下，不要尝试 resnet34/50 等更大的骨干网络，`resnet18` 仍是默认上限
- 每次实验前确认 `runs/` 下没有残留的大 checkpoint 文件，并顺手检查 C 盘剩余空间

## 准备工作

开始一次新的运行时，需要与用户一起完成以下事项：

1. 根据本地日期确定一个运行标签，例如 `2026-03-31-ankle`。
2. 基于当前主分支创建一个新的分支：
   - `git checkout -b autoresearch/<tag>`
3. 先确认训练环境可用：
   - `Test-Path .\.venv\Scripts\python.exe`
   - `.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"`
   - 如果 `torch.cuda.is_available()` 不是 `True`，先停下来修环境，不要盲跑 CPU
4. 阅读以下范围内的文件，获得完整上下文：
   - `README.md`
   - `train.py`
   - `src/model.py`
   - `src/attention_pooling.py`
   - `src/cross_view_attention.py`
   - `configs/autoresearch_proxy.yaml`
   - `configs/autoresearch_formal.yaml`
5. 确认数据已经准备就绪：
   - 必须存在 `data/realdata/metadata.csv`。
   - 其中必须包含 `patient_id`、`label`、`axial_dir`、`coronal_dir`、`sagittal_dir`。
   - 如果存在 `split` 字段，它应该已经固定为 `train` / `val` / `test`。
   - 如果缺少 `metadata.csv`，但原始 NIfTI 文件已经位于 `data/realdata/` 下，
     则使用下面的命令生成模板：
     - `.\.venv\Scripts\python.exe tools/create_realdata_metadata.py --input_dir data/realdata --output_csv data/realdata/metadata.csv`
   - 如果缺少标签或数据划分，先停止实验，并请人工补全后再继续。
6. 如果 `results.tsv` 不存在，就初始化它。文件中应该只包含表头这一行。
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
- `train.py`
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

`train.py` 会打印一些有用的日志行：
- `best_val_auc=...`
- `best_val_f1=...`
- `best_val_accuracy=...`
- `peak_vram_mb=...`
- `total_seconds=...`

在开始长时间无人值守运行之前：
- 先确保手动执行的一次基线运行能够顺利完成
- 前 3 到 4 个实验要先盯着看，不要一开始就直接离开
- 只有当整个循环已经明显运行稳定时，才开始隔夜运行

## 结果文件

`results.tsv` 使用制表符分隔，表头如下：

```tsv
commit	val_auc	val_f1	memory_gb	status	config	description
```

规则：
- `commit`：短 git hash
- `val_auc`：如果崩溃，记为 `0.000000`
- `val_f1`：如果崩溃，记为 `0.000000`
- `memory_gb`：用 `runtime.peak_vram_mb` 除以 `1024`，保留一位小数；如果崩溃则记为 `0.0`
- `status`：只能是 `keep`、`discard`、`crash` 之一
- `config`：`proxy` 或 `formal`
- `description`：实验的简短描述

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
   - 如果是 OOM → 在 description 中记录大致内存用量，标记为 `crash`
   - 如果是代码 bug → 修复后重跑一次（最多重试 1 次）
   - 如果是数据问题 → 立即停止，不要自己修数据
   - 其他情况 → 标记为 `crash`，写入 `results.tsv`，回退这次实验
6. 如果运行成功：
   - 读取 `runs/autoresearch_proxy/summary.json`
   - 将 `best_val.auc` 与当前保留的最佳结果比较
   - 只有当 `best_val.auc` 提升时，才保留这个提交
   - 如果 `best_val.auc` 持平，优先选择更简单的代码
   - 否则回退这次实验
7. 每出现 3 次 proxy 胜出后，做一次 formal 确认：
   - `.\.venv\Scripts\python.exe train.py --config configs/autoresearch_formal.yaml > formal.log 2>&1`
   - 读取 `runs/autoresearch_formal/summary.json`
   - 如果 formal 运行也提升了，就把这个提交视为新的 formal 最优结果
   - 如果 proxy 提升了，但 formal 明显退化，则优先保留上一个 formal 最优结果
   - 额外规则：如果 proxy AUC 单次提升超过 0.02，可以立即做 formal 确认
   - formal 结果也要记入 `results.tsv`（config 列标记为 `formal`）
   - 如果 formal 与 proxy 结果严重不一致（差异 > 0.03），考虑增加 proxy 的 epochs

## 超时规则

- proxy 实验（4 epochs）在当前机器上实测约 **28-30 分钟**
- 超时阈值：**60 分钟**。如果一次 proxy 运行超过 60 分钟还未结束，终止它并视为失败
- formal 实验（10 epochs）在当前机器上实测约 **95-105 分钟**
- 超时阈值：**150 分钟**
- 使用以下方式监控运行时间：
  ```powershell
  $proc = Start-Process .\.venv\Scripts\python.exe -ArgumentList "train.py","--config","configs/autoresearch_proxy.yaml" -RedirectStandardOutput run.log -RedirectStandardError err.log -PassThru
  if (-not $proc.WaitForExit(3600000)) { $proc.Kill(); Write-Host "TIMEOUT" }
  ```

## 简洁性原则

在 AUC 相同或差距极小（< 0.005）时：
- 代码行数更少的版本胜出
- 删掉组件后 AUC 不变？这是简化胜利，保留
- 微小的提升（< 0.003）但增加了大量复杂代码？放弃
- 永远追求更少的特殊逻辑、更干净的模型结构

## 实验优先级

优先做高信号、低风险的实验：

1. 跨视角注意力的深度、头数和池化行为
2. 共享骨干网络与独立骨干网络的比较
3. 切片数量、图像尺寸、dropout 和隐藏维度
4. 学习率和权重衰减
5. 在保持或提升 AUC 的前提下做小规模结构简化
6. 数据增强策略（随机旋转、翻转、亮度对比度变化）
   — 需在允许修改的代码文件中实现，不要改 `dataset.py`

避免一次改动太多内容。

## 如果陷入瓶颈

当连续 5 个以上实验都是 discard 时：
1. 重新阅读所有可修改的源文件，寻找被忽略的维度
2. 检查 `results.tsv` 中所有 keep 的实验，寻找可组合的改进
3. 尝试更大胆的结构性改变（如融合方式、注意力机制变体）
4. 回顾学习率/权重衰减等基础超参是否合理
5. 尝试不同的 dropout 策略或正则化手段

不要在同一个方向反复尝试超过 3 次。

## 上下文管理

- 读日志时只读最后 30 行：`Get-Content run.log -Tail 30`
- 不要把整个日志或完整代码文件粘贴到对话中
- 只在排错时才读更多内容
- 每次实验只关注：`summary.json` 的指标 + 简短的 diff 描述
- 大约每 15-20 次实验后，Agent 上下文可能接近饱和，此时需要重启 Agent

## 自主运行规则

**一旦进入实验循环，不要停下来询问是否继续。**
你是完全自主的研究者。如果跑完一轮还有思路，就继续下一轮。
只有当人类手动中断你时才停止。
如果实验思路用尽了，重新阅读代码和文档寻找新方向，参考上方"如果陷入瓶颈"一节。
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

## 实践经验

- 大多数实验都会失败或被丢弃，这是正常现象。
- 关键杠杆点是 `program.md`，不是在每次运行之间随意手工调整。
- 说明应该具体、简单，并且始终围绕可衡量的验证集 AUC 变化。
- 如果某条实验路线反复失败，就换个方向，不要硬推下去。


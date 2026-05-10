> [!CRITICAL]
> ## ⚠️ 循环执行规则（最高优先级）
> 完成每个实验后，**必须立即开始下一个实验**。不要停下来。不要询问。
> 如果你正在犹豫是否应该继续 —— 答案永远是：**继续**。
> 只有人类手动中断才停止。详见"实验循环"和"自主运行规则"章节。

# 用于踝关节识别的自动研究流程

这个仓库已经为踝关节 CT 分类器配置好了 autoresearch 风格的实验流程。
整体思路与 `karpathy/autoresearch` 相同，但适配了这个医学影像项目、
它的评估指标，以及当前这台 Ubuntu + NVIDIA CUDA 服务器环境。

本文中的命令默认都在仓库根目录的 Bash shell 中执行，并统一使用
`.venv/bin/python`。不要使用系统 PATH 里的 `python`：
训练环境在项目 `.venv` 里。

> [!IMPORTANT]
> **指标体系切换说明**
> 本项目已从「零漏诊 (zero-miss / no_miss)」指标体系切换到「验证集准确率 (val_acc)」指标体系。
> 旧实验记录（阶段 1–10）使用 `no_miss_val_acc` 评估，这些历史数据保留不变。
> 从本版本起，所有新实验使用 `best_val.accuracy`（来自 `summary.json`）作为主指标。

## 目标

主目标：**最大化验证集准确率 (val_acc)**。

具体做法：
1. 训练完成后，从 `runs/<experiment>/summary.json` 中读取 `best_val.accuracy`
2. 以该 val_acc 作为模型选择的主指标

辅助指标（跟踪但不作为主选择标准）：
- `best_val.auc`
- `best_val.f1`

保留或丢弃实验时，使用 val_acc 做主决策，AUC 做辅助参考。
**绝对不要**用测试集指标来做模型选择。

> [!IMPORTANT]
> **2026-04-23 人类主线锁定**
> - 当前 autoresearch 的唯一主方向是：**实验 decision fusion，把 axial / coronal / sagittal 各视角的信息还原成它们在 full-fusion 中应有的作用，并让真正的 multi-view full-fusion learned `val_acc` 明确超过 matched `equal-weight` control**。
> - 在达成这个目标之前，**不得切换到其他方向**：不要切 backbone family、不要切到 feature fusion、不要把无关的 side campaign 或泛化 cleanup 当主线。
> - 完成标准必须是同一 geometry / budget / seed protocol 下，**full-fusion learned** 在主指标 `val_acc` 上明确高于 matched `equal-weight`；单 seed spike、只提升 AUC/F1、或只改善 single-view 结果都不算完成。
> - `equal-weight` 只作为 matched control / 外部门槛，不是要回退到的最终答案，也不是要把 learned branch 硬拉平均。当前真正要找的是能解释或实现“各视角信息按其应有作用进入决策”的 **learned weighting**，或用于解释该收益来源的 **non-equal fixed weighting**。
> - 如果某条 learned weighting 或 non-equal fixed weighting 首次在单 seed 上翻过 `equal-weight`，后续优先做 matched confirmation、multi-seed validation 与权重 / 证据 telemetry，先确认它确实在让各视角信息回到应有作用；在这个确认完成前，不应切去无关方向。
> - 每轮行动前必须先自检：这项改动是否直接帮助 decision fusion 还原各视角信息的应有作用，以及它为什么有机会把 full-fusion learned accuracy 推到 `equal-weight` 之上；如果不能明确回答，这轮改动就不应执行。
> - 每轮实验完成后必须验证各视角权重比例。默认做法是对本轮 best completed checkpoint / best trial 生成或补齐 `fusion_weight_analysis.json`，至少记录三视角 `mean fusion weight` 与 `top-weight count/rate`；不得只汇报主指标而不汇报视角权重。
> - 每条实验记录都必须完整写下：`设计思路`、`预计改进效果`、`实验实际结果`。其中“预计改进效果”必须明确说明预期哪一视角权重比例 / routing 行为会如何变化，以及为什么这种变化有机会把 learned full-fusion `val_acc` 推过 matched `equal-weight`；“实验实际结果”必须写出实际权重比例与指标，且要明确说明是否符合预期。

> [!IMPORTANT]
> **2026-05-10 人类主线重定向**
> - 当前 operational mainline 固定为 **DFR-25 ResNeXt 256x8 learned decision fusion**：`ResNeXt + decision fusion + L3 no-mixer + train-time dominant-gate dropout`。
> - 人类已追加当前机制解释：本项目的 learned decision-fusion 仍存在 **权重坍塌**，主要表现为 residual axial lock-in；后续主线改进应优先采用最小网络结构改动，在 gate 内部缓解过度集中，而不是回退到 fixed equal-weight 或切换融合语义。
> - 默认配置真值来源为 `configs/autoresearch_formal.yaml`、`configs/autoresearch_proxy.yaml`、`configs/autoresearch_formal_resnext_decision_256x8.yaml` 与 `configs/autoresearch_proxy_resnext_decision_256x8.yaml`。
> - 后续若继续“对 ResNeXt 做改进”，必须从 DFR-25 锚点 fork，只改一个明确声明的变量；默认不得同时改 backbone family、geometry、fusion family、训练标量和 runtime repair。
> - `configs/optuna_main_search*.yaml` 与 `configs/optuna_proxy_search*.yaml` 默认只表达 DFR-25 fixed anchor，`search_space: {}`；任何新搜索轴必须作为新的单变量假设显式加入。
> - 2026-05-10 的这次动作只同步 GitHub 并更新配置 / 协议，**不启动实验**。

## 研究策略

**当前阶段**：主线校准 + 可复现实验 workflow 对齐

当前 canonical baseline 不再是旧的 `192x16 + stage-10 freeze` 或 `ResUNet + AttentionPooling` 叙事，而是：

- **ResNeXt** backbone
- **256x8** geometry：`image_size=256`、每视角 `8` 张切片、`trim_edge_slices=2`
- **learned decision fusion**，关闭 fusion cross-view mixer (`ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER=1`)
- **DFR-25 train-time dominant-gate dropout** (`ANKLE_DECISION_TRAIN_DOMINANT_GATE_DROPOUT_PROB=0.25`)
- 固定 winning scalar：`lr=1e-4`、`weight_decay=2.5e-4`、`dropout=0.25`、`gradient_clip_norm=2.5`、`freeze_layers=3`

当前工作重点不是继续沿旧 backlog 语义比较 `no_miss_*` 记录，而是先保证：

1. 文档、配置、训练选择规则一致
2. `model.freeze_layers` 由 YAML 显式控制，可复现
3. Optuna fresh / resume 语义清晰，不混入旧 trial
4. monitor 仅把训练故障视为 fatal，`threshold_eval` 只做辅助 warning

### 当前模型选择规则

- 主指标：`summary.json -> best_val.accuracy`
- 辅助指标：`best_val.auc`、`best_val.f1`
- keep / discard：先比较 `val_acc`
- 如果 `val_acc` 持平，优先选择 `val_auc` 更高的版本
- 如果 `val_acc` 与 `val_auc` 都持平，优先更简单的代码或配置

### 当前 baseline / Optuna 对齐规则

- `configs/autoresearch_proxy.yaml` 与 `configs/autoresearch_formal.yaml` 是当前 baseline 真值来源
- DFR-25 的 `runtime_env` 是 baseline 的一部分；直接跑训练时必须使用 `scripts/run_train_with_config_env.py`，或手动确保同等环境变量已注入
- 在单卡显存 `>= 24 GiB` 且当前算力充足时，默认直接运行 `main/formal` lane（`scripts/optuna_main.py` / `configs/autoresearch_formal.yaml`）；`proxy` 仅保留作低显存 fallback 或快速诊断
- `model.freeze_layers` 必须在 YAML 中显式声明，不再依赖修改源码常量
- Optuna baseline trial 必须基于 **effective config**（包含 runtime 默认值），不能只复制 base YAML 中显式写出的键
- 默认 Optuna search config 是 DFR-25 fixed anchor；后续 ResNeXt 改进必须以单变量 hypothesis 新开或修改搜索空间，不得默认宽搜
- dataset preflight 可以报告路径修复，但默认不应静默改写训练输入 CSV
- Optuna 必须使用项目 `.venv`，找不到就直接报错


## 硬件限制

- 操作系统：Ubuntu，Shell：Bash
- CPU / RAM（2026-04-16 实测）：Intel Xeon Gold 6426Y，2 sockets / 32 物理核 / 64 线程；内存 125 GiB
- GPU（申请目标）：4 张 NVIDIA GPU，单卡显存约 24 GB 或以上
- 当前仓库的训练配置仍按 `slot 0` / `slot 1` 两个主训练槽位组织；自适应 Optuna 入口会自动探测空闲 GPU 并按可见卡数并行调参，如需固定卡集则显式传 `--gpu-ids`
- 设备映射注意：不要假定 `nvidia-smi` 与 PyTorch/CUDA 设备编号一致；长跑前先用 `.venv/bin/python -c "import torch; print(torch.cuda.device_count()); [print(i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"` 实测当前可见 GPU
- 训练默认走 `.venv` 中的 PyTorch CUDA 环境
- formal / main 实验（15 epochs，当前 canonical DFR-25 ResNeXt 256x8 decision-fusion 路径）预计以实际 GPU 为准；历史 RTX A6000 记录约 12-18 分钟
- proxy 实验（4 epochs，当前 canonical DFR-25 ResNeXt 256x8 decision-fusion 路径）仅在显存不足或需要快速 smoke 时使用
- `batch_size=6` 是当前 DFR-25 配置锚点
- 每次实验前确认 `runs/` 下没有残留的大 checkpoint 文件，并顺手检查磁盘剩余空间

## 准备工作

开始一次新的运行时，需要先完成以下事项：

1. 根据本地日期确定一个运行标签，例如 `2026-04-01-ankle-feature`。
2. 如当前工作树是干净的，可基于当前主分支创建一个新的分支：
   - `git checkout -b autoresearch/<tag>`
   - 如果当前分支上已经有未提交的 experiment ledger / protocol 更新，不要为了切分支而打断记录流程
3. 先确认训练环境可用：
   - `test -f .venv/bin/python && echo OK`
   - `.venv/bin/python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count()); [print(i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else print('NO CUDA')"`
   - `CUDA_VISIBLE_DEVICES=1 .venv/bin/python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"`
   - 第一条命令用于全量枚举 GPU：目标申请环境下应能看到 4 张卡；第二条命令用于确认当前训练槽位实际命中的设备
   - 如果 `torch.cuda.is_available()` 不是 `True`，先停下来修环境，不要盲跑 CPU
4. 阅读以下文件获取完整上下文：
   - `backlog.md`（**必须最先读**，了解当前最优纪录、待办优先级和已完成实验）
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
- `scripts/optuna_proxy.py`
- `scripts/optuna_main.py`
- `configs/optuna_proxy_search.yaml`
- `configs/optuna_main_search.yaml`
- `scripts/`（Optuna 工作流脚本）

你可以追加写入：
- `results.tsv`

你可以读取和更新（按维护规则）：
- `backlog.md`（实验待办清单，每次实验后必须更新）
- `program.md`（协议文本；修改前需先得到人类许可）

你可以提交：
- `backlog.md`
- `results.tsv`
- `program.md`（仅限已获人类许可的协议修订）

不要修改：
- `train.py`
- `src/dataset.py`
- `src/utils.py`
- `tools/`
- 任何 CSV 数据文件
- 评估逻辑

可以新增依赖（限 `optuna` 等实验工具），但需记录在 `requirements.txt` 中。

## 基线

第一次运行必须是没有任何实验性改动的基线版本。若当前机器单卡显存 `>= 24 GiB`，默认直接验证 formal 配置：

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/run_train_with_config_env.py --config configs/autoresearch_formal.yaml --python .venv/bin/python > run.log 2>&1
```

然后读取：
- `runs/autoresearch_formal_dfr25_resnext_decision_256x8/summary.json`
- `run.log`（只读最后 30 行：`tail -30 run.log`）

`train.py` 日志行：
- `best_val_auc=...`
- `best_val_f1=...`
- `best_val_accuracy=...`
- `peak_vram_mb=...`
- `total_seconds=...`

（可选）运行阈值评估获取辅助参考指标：
```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/run_train_with_config_env.py --config configs/autoresearch_formal.yaml --python .venv/bin/python --threshold > run.log 2>&1
```

在开始长时间无人值守运行之前：
- 先确保手动执行的一次基线运行能够顺利完成
- 前 3 到 4 个实验要先盯着看
- 只有当整个循环已经明显运行稳定时，才开始隔夜运行

## 结果文件

`results.tsv` 使用制表符分隔，表头如下：

```tsv
commit	val_acc	val_auc	val_f1	memory_gb	status	config	description
```

规则：
- `commit`：短 git hash
- `val_acc`：`summary.json` 中的 `best_val.accuracy`；如果崩溃，记为 `0.000000`
- `val_auc`：`summary.json` 中的 `best_val.auc`；如果崩溃，记为 `0.000000`
- `val_f1`：`summary.json` 中的 `best_val.f1`；如果崩溃，记为 `0.000000`
- `memory_gb`：用 `runtime.peak_vram_mb` 除以 `1024`，保留一位小数；如果崩溃则记为 `0.0`
- `status`：只能是 `keep`、`discard`、`crash` 之一
- `config`：`proxy` 或 `formal`
- `description`：实验的简短描述

旧的 results.tsv 记录保留，新实验追加到末尾。
旧行使用 `no_miss_*` 列是正常的（历史指标体系）。
如果发现当前 DFR-25 输出目录（如 `runs/autoresearch_formal_dfr25_resnext_decision_256x8/summary.json` 或 `runs/autoresearch_proxy_dfr25_resnext_decision_256x8/summary.json`）已经存在但 ledger 尚未同步，先补记 `results.tsv` 与 `backlog.md`，再开始下一轮实验。
可以提交 `results.tsv`、`backlog.md` 与 `program.md`，但不要把 `runs/`、checkpoint 或大日志提交进仓库。

## 实验循环

完成准备后，持续循环执行：

> **重要**：每完成一个实验后，必须更新 `backlog.md`：
> - keep 的实验：更新"当前最优纪录"表格，将实验从待办移到"已完成"并标注结果
> - discard 的实验：将实验移到"已完成"并标注结果
> - 如果结果带来新的实验思路，添加到 backlog 对应优先级
> - 连续 3 个 discard 后，重新审视 backlog 待办清单

1. 检查当前分支和提交。
2. 在允许修改的范围内做一项实验性改动。
3. 在运行前先提交这次实验改动。
4. 默认直接运行 main / formal 实验：
   - `CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/run_train_with_config_env.py --config configs/autoresearch_formal.yaml --python .venv/bin/python > run.log 2>&1`
5. 如果运行崩溃：
   - 用 `tail -30 run.log` 查看最后几行（不要读整个日志）
   - OOM → 标记为 `crash`
   - 代码 bug → 修复后重跑一次（最多重试 1 次）
   - 数据问题 → 立即停止
   - 其他 → 标记为 `crash`，写入 `results.tsv`，回退
6. 如果运行成功：
   - 读取 `runs/autoresearch_formal_dfr25_resnext_decision_256x8/summary.json`
   - 对本轮 best completed checkpoint / best trial 生成或补齐 `fusion_weight_analysis.json`，至少提取三视角 `mean fusion weight axial/coronal/sagittal` 与 `top-weight count/rate`
   - 将 `best_val.accuracy` 与当前保留的最佳结果比较
   - 只有当 val_acc 提升时，才保留这个提交
   - 如果 val_acc 持平，优先选择 `val_auc` 更高的版本
   - 如果两者都持平，优先选择更简单的代码
   - 否则回退这次实验
   - 更新实验记录时，必须同时写入 `设计思路`、`预计改进效果`、`实验实际结果（含权重比例验证）`
7. 只有在显存不足、需要快速 smoke，或定位训练 bug 时，才临时回退到 proxy：
   - `CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/run_train_with_config_env.py --config configs/autoresearch_proxy.yaml --python .venv/bin/python > proxy.log 2>&1`
   - proxy 结果只作为快速诊断或低成本筛查证据，不再是当前默认门控步骤

## 超时规则

- formal / main 实验（15 epochs）预计约 **90-120 分钟**
- 超时阈值：**180 分钟**
- proxy 实验（4 epochs）预计约 **30-35 分钟**
- 超时阈值：**60 分钟**
- 使用以下方式监控运行时间：
  ```bash
  timeout 10800 .venv/bin/python scripts/run_train_with_config_env.py --config configs/autoresearch_formal.yaml --python .venv/bin/python > run.log 2>&1
  EXIT_CODE=$?
  if [ $EXIT_CODE -eq 124 ]; then echo "TIMEOUT"; fi
  ```

## 简洁性原则

在 val_acc 相同或差距极小（< 0.005）时：
- 代码行数更少的版本胜出
- 微小提升（< 0.003）但增加了大量复杂代码？放弃
- 永远追求更少的特殊逻辑、更干净的模型结构

## 实验优先级

### 阶段 4A：方案 2 — 视角可靠度门控 (View Reliability Gating) ✅ 已完成

（10 次 proxy 实验已完结）

### 阶段 4B：方案 6 — 非对称安全融合 (Asymmetric Safety-Biased Fusion) ✅ 已完成

（10 次 proxy 实验已完结）

### 阶段 5：AttentionPooling 增强 ❌ 已放弃

（AP-FF 前 3 次全部 discard，用户决定放弃）

### 阶段 6：UWDF（Uncertainty-Weighted Decision Fusion）✅ 已完成

（8 次 proxy 实验全 discard，已完结）

### 阶段 7：DGAF（Dual-Granularity Adaptive Fusion）✅ 已完成（跳过末尾 2 次）

（6/8 次 proxy 实验已完成，全 discard，跳过 DGAF-07/08 转入 PFDF）

### 阶段 8：PFDF（Progressive Feature Distillation Fusion）✅ 已完成

（8/8 次 proxy 实验全 discard，已完结）

### 阶段 9：CVFI（Cross-View Feature Interaction Fusion）✅ 已完成


### 阶段 10：legacy 方差缩减记录（已冻结）

> 以下内容保留作历史参考。
> 其中的 `freeze_layers=2/3`、`DEFAULT_FREEZE_LAYERS`、旧 stage-10 试验顺序，都不再代表当前执行口径。
> 当前执行口径以上文“主线校准 + 可复现实验 workflow 对齐”为准。

#### legacy：freeze_layers=3 超参搜索（8 次 proxy）

1. **VR-01**: baseline（freeze=3, lr=5e-5, dropout=0.3）
2. **VR-02**: lr=1e-4
3. **VR-03**: lr=3e-5
4. **VR-04**: dropout=0.2
5. **VR-05**: dropout=0.4
6. **VR-06**: weight_decay=0.001
7. **VR-07**: lr=1e-4 + dropout=0.2
8. **VR-08**: 基于前 7 次最佳方向的组合实验

#### legacy：freeze_layers=2 超参搜索（8 次 proxy）

1. **VR-09**: baseline（freeze=2, lr=5e-5, dropout=0.3）
2. **VR-10**: lr=1e-4
3. **VR-11**: lr=3e-5
4. **VR-12**: dropout=0.2
5. **VR-13**: dropout=0.4
6. **VR-14**: weight_decay=0.001
7. **VR-15**: lr=1e-4 + dropout=0.2
8. **VR-16**: 基于前 7 次最佳方向的组合实验

> 下方 CVFI 记录同样属于历史阶段总结，不代表当前 canonical baseline。

#### 阶段 9A：CVFI 超参搜索（8 次 proxy）

1. **CVFI-01**: baseline（fusion_type=decision, lr=1e-4, dropout=0.3）
2. **CVFI-02**: lr=5e-5
3. **CVFI-03**: lr=7e-5
4. **CVFI-04**: dropout=0.2
5. **CVFI-05**: dropout=0.4
6. **CVFI-06**: label_smoothing=0.05
7. **CVFI-07**: lr=5e-5 + label_smoothing=0.05
8. **CVFI-08**: 基于前 7 次最佳方向的组合实验


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

model:
  use_attention_pooling: true     # 注意力池化（默认 false，替代 mean pooling）
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

- 读日志时只读最后 30 行：`tail -30 run.log`
- 不要把整个日志或完整代码文件粘贴到对话中
- 只在排错时才读更多内容
- 每次实验只关注：`summary.json` 的指标 + 简短的 diff 描述
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
- 定期检查磁盘空间：`df -h .`

## 论文就绪验证阶段

当 autoresearch 循环找到最优配置后，进入论文就绪验证：
1. 用 3 个不同 seed（42, 123, 456）分别训练，报告均值 ± 标准差
2. 对测试集做 Bootstrap 置信区间（1000 次重采样）
3. 用最佳训练策略分别跑 Feature / Decision / Attention 三种融合的 formal
4. 使用验证集准确率评估，生成论文用表格

## 实践经验

- 大多数实验都会失败或被丢弃，这是正常现象。
- 关键杠杆点是 `program.md`，不是在每次运行之间随意手工调整。
- 说明应该具体、简单，并且始终围绕可衡量的 val_acc 变化。
- 如果某条实验路线反复失败，就换个方向，不要硬推下去。

# 足踝多视角 CT 二分类入门项目

这是一个面向医学影像二分类任务的最小可运行多视角 CT 基线仓库。
目标很简单：先把“每位病人的 3 个标准视角 CT 影像 → 有病 / 无病”这件事稳定跑通，再逐步升级到更复杂的实验设置。

当前仓库已经支持：

- `2.5D` 多视角分类，不是完整 3D 网络
- 三个标准视角：`axial / coronal / sagittal`
- 三种融合模式：`feature / decision / attention`
- `png / jpg / jpeg / bmp / dcm` 切片输入
- `nii / nii.gz` 体数据输入
- 手工划分 `train / val / test`，或自动 `train / val`
- 注意力融合：`Attention Pooling + Cross-View Attention`
- **增强特性**：`View Reliability Gating (VRG)` 置信度门控、`LayerNorm` 特征归一化、动态骨干网络冻结（防过拟合策略）

如果你是第一次做医学影像深度学习，建议先用 demo 数据把流程跑通，再切到真实数据。

## 1. 项目结构

```text
ankle-recognition/
├─ configs/
│  ├─ default.yaml
│  ├─ default_decision.yaml
│  ├─ attention_fusion.yaml
│  ├─ paper_baseline.yaml
│  ├─ realdata.yaml
│  └─ formal_attention_26_10_2.yaml
├─ data/
│  └─ template_metadata.csv
├─ src/
│  ├─ __init__.py
│  ├─ attention_pooling.py
│  ├─ cross_view_attention.py
│  ├─ dataset.py
│  ├─ model.py
│  └─ utils.py
├─ tools/
│  ├─ create_dummy_dataset.py
│  └─ create_realdata_metadata.py
├─ requirements.txt
├─ train.py
├─ README.md
├─ 总结.md
└─ 改进.md
```

常用配置说明：

- `configs/default.yaml`：demo 数据 + 特征融合，适合先在 CPU 上冒烟测试
- `configs/default_decision.yaml`：demo 数据 + 决策融合
- `configs/attention_fusion.yaml`：demo 数据 + 注意力融合
- `configs/paper_baseline.yaml`：更接近正式实验的 baseline 配置
- `configs/realdata.yaml`：真实数据特征融合配置
- `configs/formal_attention_26_10_2.yaml`：真实数据注意力融合实验配置示例

## 2. 环境安装

建议 Python `3.10` 或 `3.11`。

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

如果你要使用 GPU，建议先按 PyTorch 官方方式安装与你设备匹配的 `torch / torchvision`，再补装其余依赖。

## 3. 从 GitHub clone 到新设备后，还需要做什么

这个仓库**不会**把体积大或本地运行产物上传到 GitHub。clone 之后，除了代码本身，你通常还需要准备以下内容：

### 3.1 不会随仓库一起下来的内容

下面这些目录默认被 `.gitignore` 忽略：

- `.venv/`
- `.torch-cache/`
- `data/demo/`
- `data/realdata/`
- `runs/`
- `__pycache__/`

也就是说，新设备 clone 完后：

- 需要重新创建 Python 虚拟环境
- demo 数据需要重新生成
- 真实数据和对应的 `metadata.csv` 需要你自己放回去
- 历史训练结果（如 `best.pt`、`history.json`、`summary.json`）如果想复用，需要手动拷贝 `runs/`
- 当配置里 `use_pretrained: true` 时，首次运行通常需要联网下载 ResNet18 预训练权重

### 3.2 最小可运行步骤

如果你只是想确认代码在新设备上能跑，最短流程是：

```powershell
git clone https://github.com/loxgtxy/ankle-recognition.git
cd ankle-recognition
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
python tools/create_dummy_dataset.py
python train.py --config configs/default.yaml
```

## 4. 先用 demo 数据跑通

先确认环境、依赖和训练流程都是通的：

```powershell
python tools/create_dummy_dataset.py
python train.py --config configs/default.yaml
```

训练完成后会在对应的 `runs/.../` 目录下生成：

- `best.pt`：最优模型权重
- `history.json`：每个 epoch 的训练记录
- `summary.json`：最终验证集 / 测试集指标

## 5. 真实数据怎么组织

### 5.1 方式一：每个视角一个切片文件夹

推荐目录结构：

```text
data/
└─ real/
   ├─ metadata.csv
   └─ patients/
      ├─ P001/
      │  ├─ axial/
      │  │  ├─ 0001.png
      │  │  ├─ 0002.png
      │  │  └─ ...
      │  ├─ coronal/
      │  └─ sagittal/
      └─ P002/
```

支持的切片格式：

- `png`
- `jpg / jpeg`
- `bmp`
- `dcm`

注意：同一视角里的切片文件名最好补零，例如 `0001.png`、`0002.png`，这样排序更稳定。

### 5.2 方式二：每个视角一个 NIfTI 体数据文件

每个视角也可以直接给一个 3D 体数据文件：

- `axial_dir -> xxx.nii` 或 `xxx.nii.gz`
- `coronal_dir -> xxx.nii` 或 `xxx.nii.gz`
- `sagittal_dir -> xxx.nii` 或 `xxx.nii.gz`

代码会按视角对应轴自动抽取切片，再做归一化与 resize。

### 5.3 metadata.csv 格式

必须是一行一个病人，而不是一行一张切片：

```csv
patient_id,label,axial_dir,coronal_dir,sagittal_dir,split
P001,1,data/real/patients/P001/axial,data/real/patients/P001/coronal,data/real/patients/P001/sagittal,train
P002,0,data/real/patients/P002/case_axial.nii.gz,data/real/patients/P002/case_coronal.nii.gz,data/real/patients/P002/case_sagittal.nii.gz,val
P003,1,data/real/patients/P003/axial,data/real/patients/P003/coronal,data/real/patients/P003/sagittal,test
```

字段说明：

- `patient_id`：病人编号
- `label`：标签，`0=无病`，`1=有病`
- `axial_dir / coronal_dir / sagittal_dir`：三个视角各自的路径，可以是文件夹，也可以是单个 `.nii/.nii.gz`
- `split`：可选，推荐写成 `train / val / test`

兼容说明：

- 如果不写 `split`，代码会自动做一次 `train / val` 划分
- 如果历史 CSV 使用的是 `view1_dir / view2_dir / view3_dir`，代码会自动映射到标准列名

### 5.4 用脚本生成真实数据 metadata 模板

如果你的真实数据是 NIfTI 文件，可以先生成一个 metadata 模板再补标签：

```powershell
python tools/create_realdata_metadata.py --input_dir data/realdata
```

默认输出：

- `data/realdata/metadata_template.csv`

你也可以手动指定输出路径：

```powershell
python tools/create_realdata_metadata.py --input_dir data/realdata --output_csv data/realdata/metadata.csv
```

## 6. 训练怎么跑

### 6.1 demo 数据

```powershell
python train.py --config configs/default.yaml
python train.py --config configs/default_decision.yaml
python train.py --config configs/attention_fusion.yaml
```

### 6.2 真实数据

先把配置里的 `data.csv_path` 改成你自己的 CSV 路径，再运行：

```powershell
python train.py --config configs/realdata.yaml
```

如果你要复现注意力融合实验配置，可以使用：

```powershell
python train.py --config configs/formal_attention_26_10_2.yaml
```

## 7. 当前支持的模型模式

`configs/*.yaml` 里通过 `model.fusion_type` 切换模式：

```yaml
model:
  fusion_type: feature
```

可选值：

- `feature`：特征融合
- `decision`：决策融合
- `attention`：注意力融合

### 7.1 特征融合（Feature Fusion）

流程：

1. 每个视角固定抽取 `num_slices_per_view` 张切片
2. 每张切片 resize 到统一大小
3. 每个视角用 `ResNet18` 提特征
4. 对同一视角的多张切片做平均池化
5. 把三个视角特征拼接起来
6. 经过分类头输出 2 分类

这是当前最直接的 baseline。

### 7.2 决策融合（Decision Fusion）及其增强版

流程：

1. 每个视角独立提特征（推荐设置 `share_backbone: false` 让视角学习独立模式）
2. 特征首先通过 `LayerNorm` 稳定分布（对较小样本训练非常有帮助）
3. 每个视角各自输出一组 2 分类 logits
4. 三个视角的输出通过可学习的置信度门控 `View Reliability Gating (VRG)` 进行加权融合
5. 最终对融合后的输出计算主损失

目前这种带有 VRG 和层归一化的决策融合，是模型在多视角下降低“漏诊率”并行之有效的稳定 baseline。

适合做与特征融合的对照实验。

### 7.3 注意力融合（Attention Fusion）

相比特征融合，注意力模式主要多了两层机制：

1. `Attention Pooling`：替代简单平均池化，自动聚焦更关键的切片
2. `Cross-View Attention`：让三个视角在融合前先交换信息

额外可配置项：

- `cross_view_heads`
- `cross_view_layers`

对应代码位置：

- `src/attention_pooling.py`
- `src/cross_view_attention.py`
- `src/model.py`

## 8. 常用配置项

`data` 部分常用项：

- `csv_path`：病例表路径
- `base_dir`：相对路径基准目录
- `image_size`：resize 后边长
- `num_slices_per_view`：每个视角抽几张切片
- `trim_edge_slices`：抽样前忽略两端若干切片
- `hu_min / hu_max`：CT 窗宽窗位裁剪范围
- `batch_size`
- `num_workers`
- `val_ratio`

`model` 部分常用项：

- `fusion_type`：`feature / decision / attention`
- `share_backbone`：三个视角是否共享同一个 `ResNet18`
- `use_pretrained`：是否使用 ImageNet 预训练权重
- `fusion_hidden_dim`：分类头隐藏层维度
- `dropout`
- `cross_view_heads`：注意力头数（仅 attention 模式使用）
- `cross_view_layers`：Transformer 层数（仅 attention 模式使用）

`train` 部分常用项：

- `epochs`
- `lr`
- `weight_decay`
- `gradient_clip_norm`：梯度裁剪（如设为 `1.0`，对稳定收敛十分有效）
- `label_smoothing`：标签平滑
- `class_weight`
- `device`

## 9. 当前训练与评估逻辑

- 有 `split` 列时，按 `train / val / test` 使用
- 没有 `split` 列时，自动按标签分层切出 `train / val`
- 训练时使用交叉熵损失
- 如果 `class_weight: true`，会自动按训练集类别频次计算权重
- 模型选择优先按验证集 `AUC`，如果 `AUC` 不可用则退化为 `accuracy`
- `device: auto` 当前会在 `cuda` 和 `cpu` 之间自动选择

## 10. 小白最容易踩的坑

- 不要把同一病人的不同切片拆到训练集和验证集
- 不要一开始就上完整 3D 三分支模型
- 数据量较小且参数量大时，极易因过拟合导致指标崩盘。建议直接修改代码内的配置（如冻结骨干网络的大部分层：设 `DEFAULT_FREEZE_LAYERS = 2 或 3`），只放开最后的层与分类头。
- 不要只看准确率，至少看 `AUC`，同时留意阈值与`零漏诊准确性（无病特异度）`的平衡
- 验证集太小时，`AUC` 波动很大是正常现象，强烈建议看多个随机种子 (Seed)
- `dcm` 依赖 `pydicom`，`nii/nii.gz` 必须是 3D 体数据
- 如果你在新设备上直接运行默认配置报找不到 `data/demo/metadata.csv`，先执行 `python tools/create_dummy_dataset.py`
- 如果你想直接复用旧实验结果，记得把旧设备上的 `runs/` 一并拷过来

## 11. 进阶探索与提分建议

这一版跑通以后，基于医学影像中样本量通常较小的痛点（如只有几百例），重点可往如下方向深挖以保证指标稳定、置信度高：

1. **方差缩减 (Variance Reduction)**：3 个独立 ResNet18 的参数组合很容易在小数据集上引起巨大方差。建议结合网络截断冻结策略、增大强正则化 (`dropout=0.3/0.4`, `weight_decay=0.001`, `gradient_clip`) 来提升稳健性。
2. **多随机种子评价 (Multi-Seed Validation)**：每次新改进后至少测试 3 个以上的 Seed 并计算指标标准差。单次的高准确率可能只是撞上的幸运分布。
3. 简单划分 -> 5-fold 交叉验证。
4. 最后才是考虑将 2.5D 切片扩展为 3D 模型实现。

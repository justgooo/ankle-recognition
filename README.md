# 足踝多视角 CT 二分类入门项目

这是一个最小可运行的多视角 CT 二分类框架，目标是先把下面这件事稳定跑通：

- 输入：每个病人的 3 个视角影像
- 输出：`有病 / 无病`

当前版本已经支持：

- `2.5D` 多视角分类，不是完整 3D 网络
- 三个标准视角：`axial / coronal / sagittal`
- 两种融合模式
- `png/jpg/bmp/dcm` 切片输入
- `nii/nii.gz` 体数据输入
- 手工划分 `train/val/test` 或自动 `train/val`

如果你是第一次做医学影像深度学习，先把这一版跑通，再考虑升级到 MONAI、3D 模型或更复杂的多分支结构。

## 1. 当前项目包含什么

项目结构：

```text
Playground/
├─ configs/
│  ├─ default.yaml
│  ├─ paper_baseline.yaml
│  └─ realdata.yaml
├─ data/
│  ├─ demo/
│  │  ├─ metadata.csv
│  │  └─ patients/
│  └─ template_metadata.csv
├─ src/
│  ├─ __init__.py
│  ├─ dataset.py
│  ├─ model.py
│  └─ utils.py
├─ tools/
│  └─ create_dummy_dataset.py
├─ requirements.txt
└─ train.py
```

几个配置文件的定位：

- `configs/default.yaml`：轻量版，适合先在 CPU 上跑通
- `configs/paper_baseline.yaml`：更接近正式实验设置
- `configs/realdata.yaml`：面向真实数据的更大输入尺寸和切片数

## 2. 安装环境

建议 Python `3.10` 或 `3.11`。

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 3. 先用假数据跑通

先确认环境和训练流程是通的：

```powershell
python tools/create_dummy_dataset.py
python train.py --config configs/default.yaml
```

训练完成后会在对应的 `runs/.../` 目录下生成：

- `best.pt`：最佳模型权重
- `history.json`：每个 epoch 的训练记录
- `summary.json`：最终验证集/测试集指标

## 4. 数据怎么组织

### 4.1 方式一：每个视角一个切片文件夹

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
- `jpg/jpeg`
- `bmp`
- `dcm`

注意：

- 同一个视角里的切片文件名最好补零，比如 `0001.png`、`0002.png`
- 这样排序才稳定

### 4.2 方式二：每个视角一个 NIfTI 体数据文件

每个视角也可以直接给一个 3D 体数据文件：

- `axial_dir -> xxx.nii` 或 `xxx.nii.gz`
- `coronal_dir -> xxx.nii` 或 `xxx.nii.gz`
- `sagittal_dir -> xxx.nii` 或 `xxx.nii.gz`

代码会按视角对应轴自动抽取切片，再做归一化和 resize。

### 4.3 metadata.csv 格式

必须是一行一个病人，不能一行一张切片。

```csv
patient_id,label,axial_dir,coronal_dir,sagittal_dir,split
P001,1,data/real/patients/P001/axial,data/real/patients/P001/coronal,data/real/patients/P001/sagittal,train
P002,0,data/real/patients/P002/case_axial.nii.gz,data/real/patients/P002/case_coronal.nii.gz,data/real/patients/P002/case_sagittal.nii.gz,val
P003,1,data/real/patients/P003/axial,data/real/patients/P003/coronal,data/real/patients/P003/sagittal,test
```

字段说明：

- `patient_id`：病人编号
- `label`：标签，`0=无病`，`1=有病`
- `axial_dir/coronal_dir/sagittal_dir`：三个视角各自的路径，既可以是文件夹，也可以是单个 `.nii/.nii.gz`
- `split`：可选，推荐写 `train/val/test`

兼容说明：

- 如果不写 `split`，代码会自动做一次 `train/val` 划分
- 如果历史 CSV 用的是 `view1_dir/view2_dir/view3_dir`，代码会自动映射到标准列名

## 5. 训练怎么跑

先把配置里的 `data.csv_path` 改成你自己的 CSV 路径，再运行：

```powershell
python train.py --config configs/default.yaml
```

如果你已经准备好真实数据，可以直接换配置：

```powershell
python train.py --config configs/realdata.yaml
```

## 6. 现在支持的模型模式

`configs/*.yaml` 里支持 `model.fusion_type`：

```yaml
model:
  fusion_type: feature
```

可选值：

- `feature`：特征融合
- `decision`：决策融合

### 6.1 特征融合

流程是：

1. 每个视角固定抽取 `num_slices_per_view` 张切片
2. 每张切片 resize 到统一大小
3. 每个视角用 `ResNet18` 提特征
4. 对同一视角的多张切片做平均池化
5. 把三个视角特征拼接起来
6. 经过一个分类头输出 2 分类

这是当前默认模式，适合先做 baseline。

### 6.2 决策融合

流程是：

1. 每个视角先独立提特征
2. 每个视角各自输出一组 2 分类 logits
3. 三个视角的输出通过可学习权重做加权融合
4. 最终只对融合后的输出计算主损失

这里的融合权重是模型参数，会在训练中自动学习。

## 7. 常用配置项

`data` 部分常用项：

- `csv_path`：病例表路径
- `base_dir`：相对路径基准目录
- `image_size`：resize 后的边长
- `num_slices_per_view`：每个视角抽几张切片
- `trim_edge_slices`：抽样前忽略两端若干张切片
- `hu_min/hu_max`：CT 窗宽窗位裁剪范围
- `batch_size`
- `num_workers`
- `val_ratio`

`model` 部分常用项：

- `fusion_type`：`feature` 或 `decision`
- `share_backbone`：三个视角是否共享同一个 `ResNet18`
- `use_pretrained`：是否使用 ImageNet 预训练权重
- `fusion_hidden_dim`：分类头隐藏层维度
- `dropout`

`train` 部分常用项：

- `epochs`
- `lr`
- `weight_decay`
- `class_weight`
- `device`

## 8. 当前训练与评估逻辑

- 有 `split` 列时，按 `train/val/test` 使用
- 没有 `split` 列时，自动按标签分层切出 `train/val`
- 训练时使用交叉熵损失
- 如果 `class_weight: true`，会自动按训练集类别频次计算权重
- 模型选择优先按验证集 `AUC`，如果 `AUC` 不可用则退化为 `accuracy`

## 9. 小白最容易踩的坑

- 不要把同一个病人的不同切片拆到训练集和验证集
- 不要一开始就上完整 3D 三分支模型
- 不要只看准确率，至少看 `AUC`
- 验证集太小时，`AUC` 波动很大是正常现象
- `dcm` 会依赖 `pydicom`，`nii/nii.gz` 必须是 3D 体数据

## 10. 下一步可以怎么升级

这一版跑通以后，再按下面顺序升级更稳：

1. 特征融合 vs 决策融合做对比实验
2. 简单划分 -> 5-fold 交叉验证
3. 2.5D -> 3D
4. 纯 PyTorch -> MONAI
5. 加可解释性，比如 `Grad-CAM`

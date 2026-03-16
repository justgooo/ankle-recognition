# 足踝多视角 CT 二分类入门项目

这是一套给小白准备的最小可运行框架，目标是先把下面这件事跑通：

- 输入：每个病人的 3 个视角 CT 切片序列
- 输出：`有病 / 无病`

这版代码故意保持简单：

- 先做 `2.5D`，不是完整 3D
- 每个视角固定抽取若干张切片
- 每个视角用同一个 `ResNet18` 提特征
- 最后拼接做分类

如果你是第一次做医学影像深度学习，先把这一版跑通，再考虑升级到 MONAI 或 3D 模型。

## 1. 你会得到什么

项目结构：

```text
Playground/
├─ configs/
│  └─ default.yaml
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

## 2. 先装 Python 包

建议先用 Python 3.10 或 3.11。

### Windows 创建环境

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

如果你已经装了 Anaconda，也可以直接新建一个环境再安装 `requirements.txt`。

## 3. 先用假数据测试环境

先生成一套假数据，确认代码和环境都能跑：

```powershell
python tools/create_dummy_dataset.py
python train.py --config configs/default.yaml
```

说明：

- `configs/default.yaml` 是轻量版，适合先在 CPU 上跑通
- `configs/paper_baseline.yaml` 更接近正式实验配置，但更慢

跑完后会在 `runs/beginner_multiview_ct/` 下看到：

- `best.pt`：最佳模型权重
- `history.json`：每个 epoch 的训练记录
- `summary.json`：最终验证集/测试集结果

## 4. 你自己的数据怎么放

### 4.1 推荐目录结构

每个病人一行 CSV，每个视角一个文件夹，文件夹里放这一视角的所有切片。

```text
data/
└─ real/
   ├─ metadata.csv
   └─ patients/
      ├─ P001/
      │  ├─ view1/
      │  │  ├─ 0001.png
      │  │  ├─ 0002.png
      │  │  └─ ...
      │  ├─ view2/
      │  └─ view3/
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

### 4.2 metadata.csv 格式

必须是一行一个病人，不能一行一张切片。

```csv
patient_id,label,view1_dir,view2_dir,view3_dir,split
P001,1,data/real/patients/P001/view1,data/real/patients/P001/view2,data/real/patients/P001/view3,train
P002,0,data/real/patients/P002/view1,data/real/patients/P002/view2,data/real/patients/P002/view3,val
P003,1,data/real/patients/P003/view1,data/real/patients/P003/view2,data/real/patients/P003/view3,test
```

字段说明：

- `patient_id`：病人编号
- `label`：标签，`0=无病`，`1=有病`
- `view1_dir/view2_dir/view3_dir`：三个视角各自的切片文件夹
- `split`：可选，推荐写 `train/val/test`

如果你不写 `split`，代码会自动按病人行做一次 `train/val` 划分。

## 5. 用你自己的数据训练

先把 `configs/default.yaml` 或 `configs/paper_baseline.yaml` 里的 `csv_path` 改成你自己的 CSV 路径，然后运行：

```powershell
python train.py --config configs/default.yaml
```

## 6. 这版模型做了什么

模型逻辑如下：

1. 每个视角固定抽取 `num_slices_per_view` 张切片
2. 每张切片 resize 到统一大小
3. 每个视角用 `ResNet18` 提特征
4. 对同一视角的多张切片做平均池化
5. 三个视角的特征拼接，输出 2 分类

这版非常适合先做 baseline。

## 7. 小白最容易踩的坑

- 不要把同一个病人的不同切片分到训练集和验证集
- 不要一开始就上完整 3D 三分支模型
- 不要只看准确率，至少看 `AUC`
- 如果验证集很小，AUC 波动会很大，这很正常

## 8. 下一步怎么升级

这一版跑通以后，你再按下面顺序升级：

1. 单视角 -> 三视角
2. 2.5D -> 3D
3. 纯 PyTorch -> MONAI
4. 简单划分 -> 5-fold 交叉验证

如果你想，我下一步可以继续给你：

- 一版更适合论文的 `5-fold` 版本
- 一版支持 `MONAI` 的医学影像版本
- 一版支持 `Grad-CAM` 的可解释性版本

# 服务器环境配置 Agent 指令

> **目标**：在一台 Ubuntu + 4 张 24GB+ GPU 服务器上，将「足踝 CT 分类」项目配置到可运行状态。
> 项目 GitHub 仓库已克隆到服务器，数据集已传输到位。

---

## 前置假设

- OS：Ubuntu（22.04 或更高）
- GPU：4 × NVIDIA GPU（单卡显存约 24 GB 或以上；具体型号可不同）
- CPU：Intel Xeon Gold 6426Y，2 sockets / 32 物理核 / 64 线程（按当前宿主机实测）
- RAM：125 GiB
- 项目仓库已通过 `git clone` 拉取到服务器本地
- 数据集目录 `data/realdata/` 已放置在项目根目录下
- 服务器有 sudo 权限

---

## 第 1 步：确认 NVIDIA 驱动与 CUDA

```bash
nvidia-smi
```

- 确认输出中能看到 **4 张 GPU**、每张显存约 **24GB 或以上**，以及 **Driver Version**。
- 如果 `nvidia-smi` 不可用或驱动版本过低（< 470），需要安装/升级驱动：

```bash
sudo apt update
sudo apt install -y nvidia-driver-535  # 或当前推荐的稳定版本
sudo reboot
```

重启后重新运行 `nvidia-smi` 验证。

---

## 第 2 步：安装 Python 3.12

项目使用 **Python 3.12**。

```bash
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3.12-dev
python3.12 --version  # 确认输出 Python 3.12.x
```

---

## 第 3 步：创建虚拟环境

在项目根目录下执行：

```bash
cd /path/to/ankle-recognition   # 替换为实际项目路径
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

后续所有 Python 命令均在此 venv 下执行（即使用 `.venv/bin/python`）。

---

## 第 4 步：安装 PyTorch（GPU 版）

项目原环境为 `torch==2.10.0+cu130`。根据服务器 CUDA 驱动版本选择合适的安装命令：

```bash
# 方案 A：CUDA 13.0（与原环境一致，推荐）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# 方案 B：如果服务器驱动不支持 CUDA 13.0，退而选择 CUDA 12.4
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

安装后验证：

```bash
python -c "import torch; print(f'torch={torch.__version__}, cuda={torch.version.cuda}, available={torch.cuda.is_available()}, count={torch.cuda.device_count()}'); [print(i, torch.cuda.get_device_name(i), round(torch.cuda.get_device_properties(i).total_memory / 1024**3, 1), 'GiB') for i in range(torch.cuda.device_count())]"
```

预期输出应包含 `count=4`，且每张卡的显存约 `24 GiB` 或以上；具体 GPU 型号不作硬编码要求。

---

## 第 5 步：安装其他依赖

```bash
pip install -r requirements.txt
```

`requirements.txt` 内容如下，无需修改：

```
torch>=2.2
torchvision>=0.17
numpy>=1.26
pandas>=2.2
scikit-learn>=1.4
Pillow>=10.0
PyYAML>=6.0
tqdm>=4.66
pydicom>=2.4
```

---

## 第 6 步：修复 Windows → Linux 路径

### 6.1 检查 `data/realdata/metadata.csv`

```bash
head -3 data/realdata/metadata.csv
```

如果路径中包含 Windows 反斜杠 `\`，执行替换：

```bash
sed -i 's|\\|/|g' data/realdata/metadata.csv
```

### 6.2 检查所有 YAML 配置文件

配置文件中的路径（`csv_path`、`base_dir`、`output_dir`）应使用正斜杠 `/`。
当前项目使用的都是相对路径（如 `data/realdata/metadata.csv`、`.`、`runs/...`），
在 Linux 下相对路径本身就是正斜杠格式，**通常无需修改**。但仍需确认：

```bash
grep -r '\\\\' configs/*.yaml   # 检查是否有反斜杠
```

如有输出，手动将对应的 `\\` 替换为 `/`。

---

## 第 7 步：调整 AGENTS.md 配置

项目的 `AGENTS.md` 中有 Windows 特定配置，需要修改为 Linux 版本：

| 原内容 (Windows) | 替换为 (Linux) |
|---|---|
| `.\.venv\Scripts\python.exe` | `.venv/bin/python` |
| `OS: Windows，Shell: PowerShell` | `OS: Ubuntu，Shell: Bash` |

执行：

```bash
sed -i 's|.\\.venv\\Scripts\\python.exe|.venv/bin/python|g' AGENTS.md
sed -i 's|OS: Windows，Shell: PowerShell|OS: Ubuntu，Shell: Bash|g' AGENTS.md
```

---

## 第 8 步：统一为 Python 入口

服务器目标环境不依赖可执行脚本入口，项目中的调度/批处理入口统一改为 `.py`：

- `run_paper_validation.py`
- `scripts/parallel_train.py`
- `scripts/parallel_status.py`
- `scripts/optuna_proxy.py`
- `scripts/optuna_main.py`

调用方式统一为：

```bash
python run_paper_validation.py
python scripts/parallel_train.py
python scripts/parallel_status.py
python scripts/optuna_proxy.py
python scripts/optuna_main.py
```

这样部署时只要求服务器能运行 Python，不再依赖 `.sh` / `.ps1` 的执行权限与 shell 兼容性。

---

## 第 9 步：调整训练参数

建议先用保守的数据加载配置启动，再根据 CPU 占用情况逐步放宽。

修改所有 `configs/autoresearch_*.yaml`：

```yaml
data:
  batch_size: 4      # 24GB 级别显存通常足够
  num_workers: 1     # 先从保守值起步，观察 CPU 使用率后再决定是否提高
```

> **注意**：当前仓库的并行脚本仍以双槽位为主，建议先用 `num_workers=1` 验证稳定性；
> 如果 CPU 占用明显不高，可以逐步提升到 `2` 或 `4`，但每次调整前都要先观察是否出现 DataLoader 抖动或整体吞吐下降。

---

## 第 10 步：验证运行

### 10.1 快速冒烟测试

```bash
source .venv/bin/activate
python train.py --config configs/autoresearch_proxy.yaml
```

确认：
- 输出 `Using device: cuda`
- 训练循环正常启动，损失在下降
- 无报错（特别注意路径错误和 CUDA 错误）
- 运行结束后 `runs/autoresearch_proxy/summary.json` 存在且内容合理

### 10.2 检查 VRAM 使用

训练过程中在另一个终端运行：

```bash
nvidia-smi
```

确认 GPU 利用率 > 0%，且显存使用量合理（`batch_size=4` 时通常预计 10-16GB）。

### 10.3 双 GPU 并行测试

```bash
python scripts/parallel_train.py
```

确认：
- 当前实现默认会拉起两个训练槽位；确认至少两张 GPU 有负载（`nvidia-smi` 显示两卡均有显存占用）
- CPU 使用率没有飙到 100%
- 两个进程各自的日志正常输出

其余第 3 / 4 张卡可先做单卡可见性检查：

```bash
CUDA_VISIBLE_DEVICES=2 python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.get_device_name(0))"
CUDA_VISIBLE_DEVICES=3 python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.get_device_name(0))"
```

监控状态：
```bash
python scripts/parallel_status.py
```

---

## 完成标志

以下条件全部满足即视为配置成功：

- [x] `nvidia-smi` 正常显示 4 张 GPU，且每张显存约 24GB 或以上
- [x] `python -c "import torch; print(torch.cuda.is_available())"` 输出 `True`
- [x] `python -c "import torch; print(torch.cuda.device_count())"` 输出 `4`
- [x] `pip list` 中包含 torch、torchvision、numpy、pandas、scikit-learn、Pillow、PyYAML、tqdm、pydicom
- [x] `python train.py --config configs/autoresearch_proxy.yaml` 能完整跑完 4 个 epoch 且生成 `summary.json`
- [x] AGENTS.md 中的路径已改为 Linux 格式
- [x] `data/realdata/metadata.csv` 中的路径均为正斜杠格式
- [ ] `python scripts/parallel_train.py` 能同时在两个主训练槽位上启动训练
- [ ] `CUDA_VISIBLE_DEVICES=2/3` 的单卡检查均能正常识别 GPU

---

## 项目关键文件速查

| 文件 | 说明 | 可修改？ |
|---|---|---|
| `train.py` | 训练主入口 | ❌ |
| `src/model.py` | 模型定义 | ✅ |
| `src/dataset.py` | 数据加载 | ❌ |
| `src/utils.py` | 工具函数 | ❌ |
| `src/attention_pooling.py` | 注意力池化 | ❌ |
| `src/cross_view_attention.py` | 跨视角注意力 | ❌ |
| `configs/*.yaml` | 实验配置 | ✅ |
| `tools/evaluate_threshold.py` | 阈值评估 | ❌ |
| `requirements.txt` | Python 依赖 | ❌ |
| `AGENTS.md` | Agent 规则 | ✅ |
| `backlog.md` | 实验待办 | ✅ |
| `program.md` | 实验协议 | ⚠️ 需确认 |
| `results.tsv` | 实验结果 | 只追加 |
| `scripts/parallel_train.py` | 双 GPU 并行训练启动器 | ✅ |
| `scripts/parallel_status.py` | 双槽位状态监控 | ✅ |

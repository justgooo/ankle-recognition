# 服务器环境配置 Agent 指令

> **目标**：在一台全新的 Ubuntu + RTX 3090 服务器上，将「足踝 CT 分类」项目配置到可运行状态。
> 项目 GitHub 仓库已克隆到服务器，数据集已传输到位。

---

## 前置假设

- OS：Ubuntu（22.04 或更高）
- GPU：NVIDIA RTX 3090（24GB VRAM）
- 项目仓库已通过 `git clone` 拉取到服务器本地（分支 `autoresearch/2026-04-08-ankle-vr01`）
- 数据集目录 `data/realdata/` 已放置在项目根目录下
- 服务器有 sudo 权限

---

## 第 1 步：确认 NVIDIA 驱动与 CUDA

```bash
nvidia-smi
```

- 确认输出中能看到 **RTX 3090** 和 **Driver Version**。
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
python -c "import torch; print(f'torch={torch.__version__}, cuda={torch.version.cuda}, gpu={torch.cuda.get_device_name(0)}')"
```

预期输出应包含 `NVIDIA GeForce RTX 3090` 且 `cuda` 版本非空。

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

## 第 8 步：将 PowerShell 脚本转换为 Bash

项目有两个 `.ps1`（PowerShell）脚本需要转为 `.sh`：

- `autoresearch_loop.ps1` → `autoresearch_loop.sh`
- `run_paper_validation.ps1` → `run_paper_validation.sh`

**转换要点**：

| PowerShell 语法 | Bash 等价 |
|---|---|
| `$variable = "value"` | `variable="value"` |
| `Write-Host "msg"` | `echo "msg"` |
| `Start-Process` | 直接执行或 `nohup ... &` |
| `Get-Content file` | `cat file` |
| `Test-Path file` | `[ -f file ]` 或 `[ -d dir ]` |
| `.\.venv\Scripts\python.exe` | `.venv/bin/python` |
| `2>$null` | `2>/dev/null` |
| 条件 `-eq`, `-ne`, `-gt` | `==`, `!=`, `-gt`（在 `[[ ]]` 中可用 `>`） |

脚本头部添加：

```bash
#!/usr/bin/env bash
set -euo pipefail
```

转换完成后赋予执行权限：

```bash
chmod +x autoresearch_loop.sh run_paper_validation.sh
```

---

## 第 9 步：（可选）利用 3090 显存优势调参

3090 有 24GB VRAM（原 2070 Super 仅 8GB），可以在配置文件中提升以下参数以加速训练：

修改 `configs/autoresearch_proxy.yaml` 和 `configs/autoresearch_formal.yaml`：

```yaml
data:
  batch_size: 6      # 原值 2，3090 可提升到 4-8
  num_workers: 4     # 原值 0，多进程加载数据加速
```

> **注意**：`num_workers` 设置过高可能导致内存不足，建议从 4 开始，如果 RAM 充裕可提升到 8。

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

确认 GPU 利用率 > 0%，且显存使用量合理（batch_size=6 时预计 8-12GB）。

---

## 完成标志

以下条件全部满足即视为配置成功：

- [x] `nvidia-smi` 正常显示 3090
- [x] `python -c "import torch; print(torch.cuda.is_available())"` 输出 `True`
- [x] `pip list` 中包含 torch、torchvision、numpy、pandas、scikit-learn、Pillow、PyYAML、tqdm、pydicom
- [x] `python train.py --config configs/autoresearch_proxy.yaml` 能完整跑完 4 个 epoch 且生成 `summary.json`
- [x] AGENTS.md 中的路径已改为 Linux 格式
- [x] `data/realdata/metadata.csv` 中的路径均为正斜杠格式

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
| `AGENTS.md` | Agent 规则 | ✅（仅路径） |
| `backlog.md` | 实验待办 | ✅ |
| `program.md` | 实验协议 | ⚠️ 需确认 |
| `results.tsv` | 实验结果 | 只追加 |

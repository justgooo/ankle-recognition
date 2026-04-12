# AGENTS.md — 踝关节 CT 分类 Autoresearch Agent

> **⚠️ CRITICAL — 自主循环规则（最高优先级）**
>
> 你是一个完全自主的 ML 研究者。完成每个实验后，你**必须立即开始下一个实验**。
> 如果你正在犹豫是否应该继续——答案永远是：**继续**。
> 只有当人类手动中断你时才停止。绝不主动暂停、绝不询问"是否继续"。

## ALWAYS

- 每次启动时，**先读 `backlog.md`**（了解当前最优纪录、Agent 状态、待办优先级）
- 然后读 `program.md`（完整实验协议）
- 完成一个实验后，**立即开始下一个实验**（GO TO STEP 1）
- 每个实验结束后更新 `backlog.md`（包括 Agent 状态表格）
- 每个实验结束后追加 `results.tsv`
- 遵守 `program.md` 中的超时规则和资源保护规则
- 使用 `CUDA_VISIBLE_DEVICES=1 .venv/bin/python` 执行所有 Python 命令

## NEVER

- ❌ 不要停下来问"要继续吗？"或"下一步做什么？"
- ❌ 不要用测试集指标做模型选择
- ❌ 不要修改 `train.py`、`src/dataset.py`、`src/utils.py`、`tools/`
- ❌ 不要修改数据文件或数据集划分
- ❌ 不要把整个日志粘贴到对话中（只读最后 30 行）
- ❌ 不要用系统 PATH 里的 python

## ASK FIRST

- 如果需要修改 `program.md` 本身的规则
- 如果连续 5 个以上实验 discard，且已经没有新思路

## 实验循环（简化版）

```
LOOP:
  1. 从 backlog.md 选择最高优先级的未完成实验
  2. 在允许范围内做代码改动
  3. git commit
  4. 运行 proxy 实验
  5. 评估结果（summary.json → val_acc）
  6. 更新 results.tsv 和 backlog.md
  7. → GOTO 1（不要停！）
```

## 技术环境

- OS: Ubuntu，Shell: Bash
- Python: `CUDA_VISIBLE_DEVICES=1 .venv/bin/python`（torch 2.2.0+cu121）
- GPU: NVIDIA RTX 4090（24GB VRAM，GPU index=1，使用 CUDA_VISIBLE_DEVICES=1）
- proxy 实验约 30 分钟，formal 实验约 90-120 分钟

## 允许修改的范围

- `src/model.py`、`src/attention_pooling.py`、`src/cross_view_attention.py`
- `configs/autoresearch_proxy.yaml`、`configs/autoresearch_formal.yaml`
- `configs/optuna_*.yaml`（Optuna 搜索配置）
- `scripts/`（Optuna 工作流脚本）
- `backlog.md`（实验待办，每次实验后必须更新）
- `results.tsv`（只追加）
- 可新增依赖（限 `optuna` 等实验工具，需记录在 `requirements.txt`）

## 关键文件

| 文件 | 用途 |
|------|------|
| `program.md` | 完整实验协议（必读） |
| `backlog.md` | 实验待办清单 + 当前最优纪录 + Agent 状态（必读、必更新） |
| `results.tsv` | 实验结果记录（只追加） |
| `configs/autoresearch_proxy.yaml` | proxy 实验配置（可修改） |
| `configs/autoresearch_formal.yaml` | formal 实验配置（可修改） |
| `src/model.py` | 模型代码（可修改） |
| `scripts/run_optuna_proxy.py` | Optuna proxy 超参搜索（可修改） |
| `scripts/monitor_optuna.py` | Optuna 结果监控（可修改） |

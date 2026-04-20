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
- 使用分配给你的 GPU 执行 Python 命令（slot 0: `CUDA_VISIBLE_DEVICES=0`，slot 1: `CUDA_VISIBLE_DEVICES=1`）
- 如果未明确分配 slot，默认使用 `CUDA_VISIBLE_DEVICES=1`（slot 1 / 默认主训练槽位）
- 并行模式下，写 `results.tsv` 时使用 `flock -x /tmp/ankle_results.lock`
- 跨节点 / 多 Slurm job 并行时，**必须先确认每个 job 内可见 GPU 映射**，再启动 coordinator
- 跨节点 / 多 Slurm job 并行时，**必须从 compute node 内的 shell 启动 loop / coordinator**，不要从登录节点或文件视图不一致的控制端直接启动
- 涉及作业申请、启动、attach、监控、取消时，**默认先选 Slurm 原生命令**（如 `srun`、`sbatch`、`squeue`、`sacct`、`scancel`）；只有本地只读检查、临时环境变量整理或 Slurm 没有等价语义时才退回普通 Bash
- 如果命令必须借助 shell 组合多步逻辑，优先写成 `srun ... bash -lc '...'` 或 `sbatch` 脚本，让 **Slurm 负责资源与生命周期**；不要先进入裸 `bash` 再手动后台管理长任务

## NEVER

- ❌ 不要停下来问"要继续吗？"或"下一步做什么？"
- ❌ 不要用测试集指标做模型选择
- ❌ 不要修改 `train.py`、`src/dataset.py`、`src/utils.py`、`tools/`
- ❌ 不要修改数据文件或数据集划分
- ❌ 不要把整个日志粘贴到对话中（只读最后 30 行）
- ❌ 不要用系统 PATH 里的 python
- ❌ 不要在并行模式下使用另一个 slot 的 output_dir
- ❌ 不要盲目把 `num_workers` 设得太高（必须先评估 CPU 占用。如果 CPU 占用不高，可以自动调高 `num_workers` 的水平）
- ❌ 不要假设多个 Slurm job 会自动组成“单机 4 卡”；跨节点时必须显式做 leader/follower 分发
- ❌ 不要在 H100/V100 等异构卡混跑前跳过空闲显存检查；被分配到 job 不等于实际显存空闲
- ❌ 不要把长时间训练、Optuna 搜索或 coordinator 直接用裸 `bash` / `nohup` / 后台 `&` 提交；优先走 `srun` / `sbatch`
- ❌ 不要把 `bash -lc` 当成调度器；它只能作为 `srun` / `sbatch` 内部的执行载体，不能替代 Slurm 的作业控制

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
- Python: `.venv/bin/python`（torch 2.2.0+cu121）
- CPU: Intel Xeon Gold 6426Y，2 sockets / 32 物理核 / 64 线程（2026-04-16 `lscpu` 实测；⚠️ 注意监控 CPU 占用率，占用不高时可尝试调高 num_workers）
- RAM: 125 GiB（2026-04-16 `free -h` 实测）
- GPU（申请目标）: 4 张 NVIDIA GPU，单卡显存约 24 GB 或以上
- 槽位约定：主训练配置仍保留 `slot 0 -> CUDA_VISIBLE_DEVICES=0`、`slot 1 -> CUDA_VISIBLE_DEVICES=1` 两个槽位；自适应 Optuna 入口会按当前可见且空闲的 GPU 数量自动并行调参，也可用 `--gpu-ids` 显式指定卡列表
- 设备映射提醒：某些宿主环境里 `nvidia-smi` 与 PyTorch 的设备顺序可能不一致；长跑前先用 `.venv/bin/python -c "import torch; print(torch.cuda.device_count()); [print(i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"` 实测当前可见 GPU
- `num_workers`: 默认 1（注意：如果 CPU 占用不高，Agent 可以自动调整 num_workers 的水平以加速训练）
- proxy 实验约 30 分钟，formal 实验约 90-120 分钟

## 跨节点 / 多 Slurm Job 操作

- 当 4 张 GPU 分散在多个 Slurm job / 多个节点上时，不要把它当成单机 `--gpu-ids 0,1,2,3` 问题；正确语义是：**多个 job 共同服务同一个 Optuna study**。
- 当前仓库的跨节点分发入口是 `scripts/slurm_optuna_launcher.py`；它会把 `scripts/autoresearch_main.py` / `scripts/autoresearch_proxy.py` 包装成：
  - 第一个 job = leader：fresh create study
  - 后续 job = follower：对同一个 `study_root/study.sqlite3` 使用 `--resume`
- 启动跨节点 coordinator 前，先在每个 job 的 shell 中确认：
  - `hostname`
  - `echo $CUDA_VISIBLE_DEVICES`
  - `.venv/bin/python -c "import torch; print(torch.cuda.device_count()); [print(i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"`
- 如果某个 attach shell 只暴露了部分 GPU，可以临时手动覆盖 `CUDA_VISIBLE_DEVICES` 到该 job 实际持有的卡；不要直接假设 step 里的 `SLURM_GPUS_ON_NODE` 一定完整。
- 跨节点运行时，优先在 **leader 所在 compute node** 内启动 `autoresearch_loop.py`，并显式导出：
  - `AUTORESEARCH_SLURM_JOB_IDS`
  - `AUTORESEARCH_SLURM_JOB_GPU_IDS`
  - `AUTORESEARCH_SLURM_JOB_CUDA_VISIBLE_DEVICES`
- 变量格式规则：
  - `AUTORESEARCH_SLURM_JOB_IDS`: 逗号分隔的 job id，例如 `423677,423003`
  - `AUTORESEARCH_SLURM_JOB_GPU_IDS`: 按 job 顺序、用分号分组的 GPU id，例如 `'4,5;0,1'`
  - `AUTORESEARCH_SLURM_JOB_CUDA_VISIBLE_DEVICES`: 按 job 顺序、用分号分组的 CUDA 可见卡，例如 `'4,5;0,1'`
- 典型启动方式：优先在 leader job 内用 `srun` 启动；如果需要新申请资源，再把同样命令封装进 `sbatch` 脚本提交，而不是直接 `nohup`
```bash
AUTORESEARCH_SLURM_JOB_IDS=423677,423003 \
AUTORESEARCH_SLURM_JOB_GPU_IDS='4,5;0,1' \
AUTORESEARCH_SLURM_JOB_CUDA_VISIBLE_DEVICES='4,5;0,1' \
srun --jobid=423677 --overlap \
  --output=autoresearch_logs/cross_node_loop.log \
  .venv/bin/python autoresearch_loop.py \
    --max-iterations 8 \
    --default-lane main-study \
    --gpu-id 4 \
    --gpu-ids 4,5,0,1 \
    --max-workers 4 \
    --main-search-template configs/optuna_main_search_resnext_decision_v100.yaml \
    --proxy-search-template configs/optuna_proxy_search_resnext_decision_v100.yaml \
    --formal-config configs/autoresearch_formal_resnext_decision_v100.yaml \
    --proxy-config configs/autoresearch_proxy_resnext_decision_v100.yaml
```
- 运行后，`optuna_main.log` / `optuna_proxy.log` 中应看到：
  - `Dispatching Optuna study across Slurm jobs`
  - `leader job ...`
  - `follower job ...`
- 如果控制端看不到 compute node 新生成的 `runs/` 或 `autoresearch_logs/` 文件，以 compute node shell 里的视图为准；这是启动点选错的信号。
- 如果某个 job 上出现 OOM，而另一个 job 还在跑，不要把该轮结果直接当作有效 keep/discard；先记录故障类型，再决定是否降 batch 或拆分异构卡。

## 允许修改的范围

- `src/model.py`、`src/attention_pooling.py`、`src/cross_view_attention.py`
- `configs/autoresearch_proxy.yaml`、`configs/autoresearch_formal.yaml`
- `configs/autoresearch_proxy_slot0.yaml`、`configs/autoresearch_formal_slot0.yaml`
- `configs/optuna_*.yaml`（Optuna 搜索配置）
- `scripts/`（Optuna 工作流脚本 + 并行训练脚本）
- `backlog.md`（实验待办，每次实验后必须更新）
- `results.tsv`（只追加）
- 可新增依赖（限 `optuna` 等实验工具，需记录在 `requirements.txt`）

## 关键文件

| 文件 | 用途 |
|------|------|
| `program.md` | 完整实验协议（必读） |
| `backlog.md` | 实验待办清单 + 当前最优纪录 + Agent 状态（必读、必更新） |
| `results.tsv` | 实验结果记录（只追加，并行时加 flock） |
| `configs/autoresearch_proxy.yaml` | 主训练槽位（slot 1）proxy 配置 |
| `configs/autoresearch_formal.yaml` | 主训练槽位（slot 1）formal 配置 |
| `configs/autoresearch_proxy_slot0.yaml` | 辅助训练槽位（slot 0）proxy 配置 |
| `configs/autoresearch_formal_slot0.yaml` | 辅助训练槽位（slot 0）formal 配置 |
| `src/model.py` | 模型代码（可修改） |
| `scripts/parallel_train.py` | 双槽位并行训练启动器（当前实现） |
| `scripts/parallel_status.py` | 双槽位状态监控 |
| `scripts/optuna_proxy.py` | 自适应 GPU proxy 超参搜索入口（可修改） |
| `scripts/optuna_main.py` | 自适应 GPU formal 超参搜索入口（可修改） |
| `scripts/monitor_optuna.py` | Optuna 结果监控（可修改） |

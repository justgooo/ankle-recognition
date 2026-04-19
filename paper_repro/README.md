# Paper Reproduction AutoResearch

这个目录是给 `/docs` 最新四篇综述文档准备的“论文复现版”训练入口。

目标不是逐字逐代码重建原论文的全部数据、标注和 workflow，而是在**当前足踝 CT 二分类数据条件下**，把每篇论文的核心网络结构或决策范式做成 **task-adapted reproduction**，用于并行对照试验。

## 设计原则

- 与主线 `train.py` 隔离，避免污染当前 autoresearch 主干。
- 复用仓库现有数据读取与评估逻辑，保证指标口径一致。
- 每轮固定 3 个 paper config 并行运行。
- 默认提供 `proxy` 预算；同一套 config 可以再扩到 `formal`。
- 对需要额外标注或额外模态的论文，采用 **task-adapted** 近似复现，并在 `papers.yaml` 中明确标记。

## 目录

- `configs/`: 每篇论文一个 YAML
- `papers.yaml`: 12 篇论文到 4 个 round 的映射
- `train.py`: 独立训练入口
- `run_round.py`: 三并行 round 调度器
- `export_results.py`: 汇总导出 TSV / Markdown，并可追加到根目录 `results.tsv`

## 运行方式

先在有 GPU 的 Slurm job shell 里确认设备映射，再启动一轮：

```bash
srun --jobid=<JOB_ID> --overlap bash -lc '
  cd /dataset/HH/ankle-ct &&
  .venv/bin/python -m paper_repro.run_round \
    --round round1 \
    --gpu-ids 0,1,2 \
    --timeout 10800 \
    --append-root-results
'
```

如果只想导出某一轮已有结果：

```bash
.venv/bin/python -m paper_repro.export_results --round round1
```

## 12 个论文配置

- `R1`: FracNet 风格 `3D U-Net + top-k dense anomaly aggregation`
- `R4`: nnU-Net / dense-vote 风格 `3D U-Net + majority voxel voting`
- `S1`: `3D EfficientNet-like` 三视图体积分类
- `S2`: `SE-ResNet50` 三视图切片聚合
- `S3`: anatomy-aware `ROI crop + prototype head`
- `S4`: `CNN + LSTM` 切片序列建模
- `D1`: `2.5D + MIL`
- `D2`: `tri-plane deep features + handcrafted features`
- `D4`: `2.5D + 3D logit ensemble`
- `C1`: `XFMamba-lite` 两级跨视图融合
- `C2`: `ROI snapshots + ResNet18` 多视角快照分类
- `C3`: `expert branches + decision fusion`

## 说明

- `S3`, `R1`, `R4`, `C2` 这类原论文依赖额外分割或 dense label 的方法，在这里使用骨区/ROI 启发式与弱 dense aggregation 做 task-adapted reproduction。
- `C1` 使用 `XFMamba-lite`，保留“两级跨视图融合 + state-space 风格 token mixer”的结构语义，不依赖额外的 Mamba 第三方包。
- `D4` 的 2.5D 分支采用强预训练 2D encoder，3D 分支采用独立体积分支，最终在 logit 层集成。


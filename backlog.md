# 实验待办清单 (Experiment Backlog)

> **本文件由 autoresearch Agent 维护。**
> Agent 在每次实验循环启动时必须阅读本文件，并在完成实验后根据结果更新。
> 人类也可以直接编辑本文件来添加想法或调整优先级。
>
> **⚠️ 指标体系切换**：本项目已从「零漏诊 (no_miss_val_acc)」切换到「验证集准确率 (val_acc)」。
> 旧实验记录使用 `no_miss_val_acc`，保留不变。新实验使用 `summary.json` 中的 `best_val.accuracy`。
> 新实验的 keep 判定基于 val_acc，不再使用 evaluate_threshold.py 作为主评估工具。
>
> **⚠️ 执行策略更新（2026-04-17）**：在单卡显存 `>= 24 GiB` 且当前算力充足的环境里，默认直接运行 `main/formal` lane；`proxy` 不再作为常规筛选步骤，仅保留给低显存 fallback 或快速诊断。

---

## Agent 状态

> Agent 每次实验后必须更新此表。新 Agent 启动时以此表为起点。

| 字段 | 值 |
|------|-----|
| 上次实验 | CMP-FAIR-V100-RESNEXT-FORMAL-SEED123（保持 dedicated ResNeXt V100 retuned formal winner 的 `freeze_layers=2`, `lr=1.25e-4`, `weight_decay=1e-4`, `dropout=0.35`, `gradient_clip_norm=2.0`, `fusion_hidden_dim=256` / `256x8` / mean pooling 几何完全不变，仅把 `seed` 从 `42` 改为 `123`，并在 `GPU 2` 上做 1 次 alternate-seed direct formal confirmation；commit `e6f36d0`） |
| 上次结果 | discard（alternate-seed direct formal 只得到 `val_acc=0.8829787234042553`, `val_auc=0.91`, `val_f1=0.8571428571428571`，较当前 retained keep `0.925531914893617 / 0.9563636363636363` 明显回落 `0.0425531914893617 / 0.0463636363636363`，也低于旧 fair-backbone stable winner `0.9148936170212766 / 0.9404545454545454`。说明当前新 anchor 至少在 `seed=123` 上没有稳定复现，应暂时视为 seed-sensitive 证据，而不是已确认稳定的新配方。） |
| 下一步 | 若继续该独立 ResNeXt V100 side campaign，优先补 1 次 `seed=456` direct formal confirmation；在得到第二个 alternate-seed 结果前，不要围绕当前 anchor 再开 fresh adaptive main study，也不要回到 `384` head 或补非 `resnext` backbone。 |
| 默认执行策略 | 24GB+ 显存机器默认直接跑 `main` / `formal`；`proxy` 仅保留作低显存 fallback 与快速 smoke。 |
| 连续 discard 计数 | 1 |
| 累计 proxy keep 数 | 7（当前 `val_acc` 主线新增 1 次 keep：`a60c3e0` / fresh proxy winner） |
| 本地迁移补记 | 2026-04-12 从旧工作副本并入的 legacy 状态：上次实验为 `VR-16`（`9293915`; `no_miss_val_acc=0.872`, `no_miss_val_spe=0.760`, `val_AUC=0.969`）；旧计划下一步为 `VR-MS-01~03`；旧连续 discard 计数为 15。该状态属于旧 `no_miss` / `192x16` campaign，已归档为 legacy，不覆盖当前 canonical `val_acc` 主线。 |

---

## 当前最优纪录

> 当前 canonical baseline 已切换为：**ResUNet encoder + 3 个 Attention Gate + AttentionPooling + VRG / decision fusion**。
> 旧的 `no_miss_*` / `192x16` / 早期 stage-10 结果只作为 **legacy reference only**，不再直接参与当前 keep/discard 比较。

| 指标 | 值 | 来源 commit | 配置 | 备注 |
|------|---:|-------------|------|------|
| **val_acc** | **0.8829787234042553** | `a60c3e0` | `configs/autoresearch_proxy.yaml` + fresh proxy study trial 0 | 4 个 completed trial 中最佳；相对 canonical baseline rerun `0.8404255319148937` 提升 `+0.0425531914893616` |
| **tie-break** | **0.9318181818181819** | `a60c3e0` | fresh proxy study trial 0 | 与上述 winner 同一 trial 的 `best_val.auc`；当 `val_acc` 持平时仍按此决胜 |
| canonical formal 参考 | 0.8617021276595744 | `c30fcae` | `configs/autoresearch_formal.yaml`（trial 0 模板超参） | 当前主线首个 formal confirmation；`val_auc=0.9372727272727273`，准确率低于 proxy winner 但 AUC 更高 |
| legacy no_miss 参考 | 0.915 | `d63b49a` (formal) | 旧 Decision Fusion / 旧实验语义 | **legacy reference only**，不可与当前主线直接比较 |

---

## 2026-04-16：本地 2070S 基础 Backbone Quick Compare

> **独立 campaign 说明**
> - 这一轮是 **本地兼容性 + backbone 快筛**，不与当前 canonical `val_acc` 主线（`a60c3e0` / `c30fcae`）直接比较。
> - 当前本地环境于 **2026-04-16** 实测为：Windows `.venv\\Scripts\\python.exe` + 单卡 `NVIDIA GeForce RTX 2070 SUPER (8192 MiB)`。
> - 原始 `data/realdata/metadata.csv` 在本地存在 `CTdata2/` 路径漂移；本轮通过 `scripts/prepare_local_metadata.py` 生成 `autoresearch_logs/local_metadata_fixed.csv` 供训练使用，**未修改原始数据文件与 split**。
> - 当日为适配 8GB 显存与本地路径现状，`configs/autoresearch_proxy.yaml` / `configs/autoresearch_formal.yaml` 曾同步切到 `autoresearch_logs/local_metadata_fixed.csv`，并临时收紧到：`image_size=256`, `num_slices_per_view=8`, `batch_size=2`, `num_workers=0`。
> - backbone 对比使用统一 quick-compare 配方：`feature fusion`, `share_backbone=false`, `freeze_layers=3`, `use_pretrained=true`, `dropout=0.3`, `lr=5e-5`, `weight_decay=1e-4`, `epochs=1`。

- [x] **CMP-LOCAL8G-RESUNET**：`configs/cmp_local8g_feature_resunet.yaml` → `val_acc=0.8085106382978723`, `val_auc=0.915`, `val_f1=0.7857142857142857`, `peak_vram=1.23 GiB`, `total_seconds=443.9` → **keep**（本轮四个 backbone 中 accuracy / AUC 均最佳，且显存余量最充足，是当前 2070S 路线的首选 backbone）
- [x] **CMP-LOCAL8G-RESNEXT**：`configs/cmp_local8g_feature_resnext.yaml` → `val_acc=0.46808510638297873`, `val_auc=0.7272727272727272`, `val_f1=0.6376811594202898`, `peak_vram=1.32 GiB`, `total_seconds=607.6` → **discard**（准确率最低，且训练时长最高；在当前预算下没有继续投入价值）
- [x] **CMP-LOCAL8G-SENET**：`configs/cmp_local8g_feature_senet.yaml` → `val_acc=0.5319148936170213`, `val_auc=0.7781818181818181`, `val_f1=0.0`, `peak_vram=1.46 GiB`, `total_seconds=493.5` → **discard**（比 `resnext` 稍好，但模型几乎退化为单类预测，仍明显弱于 `resunet / cspnet`）
- [x] **CMP-LOCAL8G-CSPNET**：`configs/cmp_local8g_feature_cspnet.yaml` → `val_acc=0.7553191489361702`, `val_auc=0.8818181818181818`, `val_f1=0.6567164179104478`, `peak_vram=5.61 GiB`, `total_seconds=475.0` → **discard**（是唯一接近 `resunet` 的备选，但 accuracy 仍低 `0.0531914893617021`；运行中 `nvidia-smi` 一度接近 `7.7 / 8.0 GiB`，说明能跑但显存冗余不大）

- **本轮结论**：`resunet` > `cspnet` >>> `senet` > `resnext`
- **推荐动作**：若继续本地 8GB 路线，只保留 `resunet` 与 `cspnet` 做后续 `4-epoch proxy`；`senet / resnext` 直接停止

---

## 2026-04-17：V100 Fair Backbone Formal Compare

> **独立 campaign 说明**
> - 这一轮按 `docs/fair_backbone_compare.md` 的“equal protocol treatment”语义执行，但把运行平台升级到 `3 x Tesla V100-PCIE-32GB`。
> - 这仍是 **独立 backbone compare campaign**，不与当前 canonical `val_acc` 主线（`a60c3e0` / `c30fcae`）直接比较，也不用于覆盖 decision-fusion 主线最优纪录。
> - 四个 backbone 统一使用 matched formal recipe：`feature fusion`, `share_backbone=false`, `use_attention_pooling=false`, `freeze_layers=3`, `dropout=0.3`, `lr=1e-4`, `weight_decay=1e-4`, `epochs=15`, `image_size=256`, `num_slices_per_view=8`, `batch_size=6`, `num_workers=12`。
> - 运行 commit 为 `25f17f0`。三卡并发脚本成功完成前三个 backbone，但未自动派发最后一个 `cspnet`；该 run 随后在空闲 `GPU 0` 上手动补跑完成。

- [x] **CMP-FAIR-V100-FORMAL-RESUNET**：`configs/cmp_fair_v100_formal_resunet.yaml` → `val_acc=0.9148936170212766`, `val_auc=0.9222727272727274`, `val_f1=0.9047619047619048`, `peak_vram=2.97 GiB`, `total_seconds=870.2` → **discard**（与 `resnext` 同分最高 `val_acc`，但按 campaign tie-break 在 `val_auc` 上低 `0.018181818181818`；说明 `resunet` 在 equal-budget formal 下仍然非常强，但不再像本地 1-epoch quick screen 那样有绝对优势。）
- [x] **CMP-FAIR-V100-FORMAL-RESNEXT**：`configs/cmp_fair_v100_formal_resnext.yaml` → `val_acc=0.9148936170212766`, `val_auc=0.9404545454545454`, `val_f1=0.9090909090909091`, `peak_vram=2.14 GiB`, `total_seconds=888.3` → **keep**（本轮 campaign winner；与 `resunet` 持平最高 `val_acc`，并以更高 `val_auc` 获胜，同时显存也更低。说明在统一 15-epoch matched recipe 下，`resnext` 的上限被此前 quick screen 明显低估。）
- [x] **CMP-FAIR-V100-FORMAL-SENET**：`configs/cmp_fair_v100_formal_senet.yaml` → `val_acc=0.851063829787234`, `val_auc=0.9195454545454544`, `val_f1=0.8157894736842105`, `peak_vram=2.16 GiB`, `total_seconds=878.0` → **discard**（较 campaign winner `0.9148936170212766` 低 `0.0638297872340426`；虽然相比本地 quick screen 有显著回升，但仍明显落后于 `resnext / resunet`。）
- [x] **CMP-FAIR-V100-FORMAL-CSPNET**：`configs/cmp_fair_v100_formal_cspnet.yaml` → `val_acc=0.8829787234042553`, `val_auc=0.9213636363636363`, `val_f1=0.8705882352941177`, `peak_vram=1.87 GiB`, `total_seconds=793.7` → **discard**（本轮最强非冠军；较 campaign winner `0.9148936170212766` 低 `0.0319148936170213`。说明 `cspnet` 在 equal-budget formal 下比本地 8GB 快筛表现更有竞争力，但仍不足以挤进第一梯队。）

- **本轮 formal 排名**：`resnext` >= `resunet` > `cspnet` > `senet`
- **按规则的最终胜者**：`resnext`
- **推荐动作**：Stage 3 稳定性确认现已完成（见下节）；该独立 fair-backbone campaign 可正式收束为 `resnext` stable winner，`resunet` 降为次优对照保留结论

---

## 2026-04-17：V100 Fair Backbone Stage 3 Stability Confirmation

> **独立 campaign 说明**
> - 本节严格承接上面的 `CMP-FAIR-V100-FORMAL`，不修改 recipe，只补做 `seed=123 / 456`。
> - 两个 backbone 都沿用同一 matched formal recipe：`feature fusion`, `share_backbone=false`, `use_attention_pooling=false`, `freeze_layers=3`, `dropout=0.3`, `lr=1e-4`, `weight_decay=1e-4`, `epochs=15`, `image_size=256`, `num_slices_per_view=8`, `batch_size=6`, `num_workers=12`。
> - 本轮运行 commit 为 `2123941`。起跑顺序为三卡并发 `resunet_s123 / resnext_s123 / resunet_s456`，随后补跑 `resnext_s456`。

- [x] **CMP-FAIR-V100-STAGE3-RESUNET-S123**：`configs/cmp_fair_v100_formal_resunet_s123.yaml` → `val_acc=0.8829787234042553`, `val_auc=0.9363636363636363`, `val_f1=0.8607594936708861`, `peak_vram=2.97 GiB`, `total_seconds=859.8` → **discard**（较 `resnext_s123` 低 `0.0319148936170213` accuracy，AUC 也低 `0.0331818181818182`；说明 `resunet` 在 alternate seed 上比 seed-42 更易回落。）
- [x] **CMP-FAIR-V100-STAGE3-RESNEXT-S123**：`configs/cmp_fair_v100_formal_resnext_s123.yaml` → `val_acc=0.9148936170212766`, `val_auc=0.9695454545454545`, `val_f1=0.9047619047619048`, `peak_vram=2.15 GiB`, `total_seconds=880.6` → **keep**（与 seed-42 持平最高 accuracy，并把 AUC 再抬高 `0.0290909090909091`；强力支持 `resnext` 不是一次性赢家。）
- [x] **CMP-FAIR-V100-STAGE3-RESUNET-S456**：`configs/cmp_fair_v100_formal_resunet_s456.yaml` → `val_acc=0.9042553191489362`, `val_auc=0.9309090909090909`, `val_f1=0.8860759493670886`, `peak_vram=2.97 GiB`, `total_seconds=862.6` → **discard**（较 `seed=42` 回落 `0.0106382978723404` accuracy；虽然恢复到竞争区，但与 `resnext_s456` 持平 accuracy 时仍输 AUC。）
- [x] **CMP-FAIR-V100-STAGE3-RESNEXT-S456**：`configs/cmp_fair_v100_formal_resnext_s456.yaml` → `val_acc=0.9042553191489362`, `val_auc=0.9413636363636364`, `val_f1=0.891566265060241`, `peak_vram=2.15 GiB`, `total_seconds=809.6` → **keep**（开局极慢热，前 2 个 epoch 一度掉到 `0.4681`，但最终仍把 best 拉回与 `resunet_s456` 持平的 accuracy，并再赢 `0.0104545454545455` AUC；说明 `resnext` 的优化轨迹更抖，但最终上限依旧稳住。）

- **按 seed 的 head-to-head 结果**
- `seed=42`：`resnext` 与 `resunet` 持平 `val_acc=0.9148936170212766`，AUC `0.9404545454545454 > 0.9222727272727274`
- `seed=123`：`resnext` 以 `0.9148936170212766 / 0.9695454545454545` 明显胜过 `resunet` 的 `0.8829787234042553 / 0.9363636363636363`
- `seed=456`：两者同为 `val_acc=0.9042553191489362`，但 `resnext` 仍以 `val_auc=0.9413636363636364` 胜过 `resunet` 的 `0.9309090909090909`

- **Stage 3 结论**：`resnext` 在 3 个 seed 上 accuracy 从不输给 `resunet`，AUC 则 3/3 全胜，因此该独立 `fair_backbone_compare` campaign 的稳定胜者正式定为 `resnext`
- **附带观察**：`resnext` 的训练曲线明显比 `resunet` 更慢热、更波动，尤其 `seed=456` 前半程多次掉到 `0.8` 以下；但以 `best_val_accuracy_then_auc` 作为规则时，它最终仍保持对 `resunet` 的系统性优势

---

## 2026-04-18：ResNeXt V100 Dedicated Main-Study（fusion_hidden_dim=384）

> **独立 campaign 说明**
> - 这一轮显式延续上面的 `fair_backbone_compare` stable-winner side campaign，只推进 `resnext`，不回到 canonical ResUNet 主线。
> - 保持 matched V100 几何不变：`image_size=256`, `num_slices_per_view=8`, `feature fusion`, `share_backbone=false`, `use_attention_pooling=false`（mean pooling）。
> - 唯一离散改动：把 dedicated formal template `configs/autoresearch_formal_resnext_v100.yaml` 的 `fusion_hidden_dim` 从 `256` 放宽到 `384`，其余由 fresh adaptive Optuna main study 做小预算标量调参。
> - 本轮 fresh study 为 `runs/optuna_main_autoloop/iter_0001_20260418_023423`，运行 commit 为 `992687c`，4/4 trials completed。

- [x] **CMP-FAIR-V100-RESNEXT-MAIN-HEAD384**：`configs/autoresearch_formal_resnext_v100.yaml`（`fusion_hidden_dim=384`）+ fresh adaptive main study → best completed trial 为 **trial 3**（`freeze_layers=3`, `lr=5e-5`, `weight_decay=5e-4`, `dropout=0.25`, `gradient_clip_norm=1.0`）→ `val_acc=0.8829787234042553`, `val_auc=0.9431818181818181`, `val_f1=0.8607594936708861`, `peak_vram=2.15 GiB`, `total_seconds=859.1` → **discard**（相对 widened-head study 内的 template trial 0 `0.8617021276595744 / 0.9663636363636363` 有 accuracy 提升 `+0.0212765957446809`，但仍较 fair-backbone stable winner `CMP-FAIR-V100-FORMAL-RESNEXT` 的 `0.9148936170212766` 低 `0.0319148936170213`；说明仅放宽 fusion bottleneck 并没有把 ResNeXt 的 formal ceiling 推到新高。）
- **Monitor takeaways**：在这 4 个 completed trial 里，`freeze_layers=3` 明显优于 `freeze_layers=2`；更低的 `lr=5e-5`、更高的 `weight_decay=5e-4`、更低的 `dropout=0.25` 共同对应最佳 accuracy。唯一的 `freeze_layers=2` 试次（trial 2）把 `peak_vram` 提到 `6.24 GiB`，但 accuracy 只追平 trial 1 且 AUC 更低，说明解冻更深层并未换来净收益。
- **本轮结论**：`fusion_hidden_dim=384` 不晋升；该独立 ResNeXt V100 side campaign 仍以原始 `256` 宽 head 的 matched formal winner（`val_acc=0.9148936170212766`, `val_auc=0.9404545454545454`）作为当前应保留的最好证据。
- **推荐动作**：若后续继续该独立 side campaign，应把本轮 main-study 暗示较优的标量组合迁回原始 `256` 宽 recipe 做单点验证，而不是继续加宽 head 或重新补 `resunet` 对照。

---

## 2026-04-18：ResNeXt V100 Dedicated Main-Study（回到 head256 + scalar retune）

> **独立 campaign 说明**
> - 这一轮继续承接 `fair_backbone_compare` stable-winner side campaign，只推进 `resnext`，不回到 canonical ResUNet 主线。
> - 保持 matched V100 几何不变：`image_size=256`, `num_slices_per_view=8`, `feature fusion`, `share_backbone=false`, `use_attention_pooling=false`（mean pooling）。
> - 唯一离散改动：把 dedicated formal template `configs/autoresearch_formal_resnext_v100.yaml` 从 widened-head 试验状态迁回原始 `fusion_hidden_dim=256`，并把 base template 标量锚定到上一轮 main-study 暗示较优的组合（`freeze_layers=3`, `lr=5e-5`, `weight_decay=5e-4`, `dropout=0.25`, `gradient_clip_norm=1.0`）。
> - 由于起跑时 `GPU 0` 不满足本轮 idle 阈值（`used≈2152 MiB`, `util≈27%`），本轮 fresh study 改走允许的单卡 fallback `GPU 2`；fresh study 为 `runs/optuna_main_autoloop/iter_0002_20260418_030934`，运行 commit 为 `5a1c113`，4/4 trials completed。

- [x] **CMP-FAIR-V100-RESNEXT-MAIN-HEAD256-RETUNE**：`configs/autoresearch_formal_resnext_v100.yaml`（`fusion_hidden_dim=256`，base template 迁入上一轮较优锚点）+ fresh adaptive main study → best completed trial 为 **trial 3**（`freeze_layers=2`, `lr=1.25e-4`, `weight_decay=1e-4`, `dropout=0.35`, `gradient_clip_norm=2.0`）→ `val_acc=0.9042553191489362`, `val_auc=0.9368181818181818`, `val_f1=0.891566265060241`, `peak_vram=6.23 GiB`, `total_seconds=827.8` → **discard**（较 widened-head study best `0.8829787234042553 / 0.9431818181818181` 回升 `+0.0212765957446809` accuracy，但仍较 fair-backbone stable winner `CMP-FAIR-V100-FORMAL-RESNEXT` 的 `0.9148936170212766 / 0.9404545454545454` 低 `0.0106382978723404` accuracy，且 AUC 也低 `0.0036363636363636`，因此不足以替换当前应保留证据。）
- **Monitor takeaways**：回到 `head=256` 后，搜索上界明显高于上一轮 widened-head study；本轮唯一突破 `0.9` accuracy 的是更激进的 `freeze_layers=2`, `lr=1.25e-4`, `weight_decay=1e-4`, `dropout=0.35`, `gradient_clip_norm=2.0` 角点。作为对照，迁回的锚点组合在 3 个 completed trial 里只落在 `0.8723404255319149 ~ 0.8829787234042553`，说明上一轮 `head384` monitor 所暗示的优选标量并不能直接把原始 `head256` recipe 拉回 stable-winner 水平。
- **附带观察**：trial 0 与 trial 2 被 Optuna 命中了同一组参数（`freeze_layers=3`, `lr=5e-5`, `weight_decay=5e-4`, `dropout=0.25`, `gradient_clip_norm=1.0`），但分别得到 `0.8723404255319149` 与 `0.8829787234042553` accuracy，提示当前 ResNeXt side campaign 仍存在不小的 run-to-run 波动。
- **本轮结论**：把模板迁回原始 `256` 宽 head 是正确方向，但目前找到的更优 retune 角点仍未超过既有 fair-backbone stable winner，因此本轮仍记 **discard**。
- **推荐动作**：若后续继续该独立 side campaign，优先对本轮 trial-3 角点做 1 次 fixed-config formal confirmation，判断 `0.9043` 是否可复现；若仍低于 `0.9149 / 0.9405`，则应收束该 side campaign，不再继续围绕 ResNeXt 标量做小步扫描。

---

## 2026-04-18：ResNeXt V100 Direct Formal Confirmation（head256 retune winner）

> **独立 campaign 说明**
> - 这一轮继续承接 `fair_backbone_compare` stable-winner side campaign，只推进 `resnext`，不回到 canonical ResUNet 主线。
> - 保持 matched V100 几何不变：`image_size=256`, `num_slices_per_view=8`, `feature fusion`, `share_backbone=false`, `use_attention_pooling=false`（mean pooling），`fusion_hidden_dim=256` 不变。
> - 唯一离散改动：把 dedicated formal template `configs/autoresearch_formal_resnext_v100.yaml` 对齐到上一轮 main-study 的 best completed trial（`freeze_layers=2`, `lr=1.25e-4`, `weight_decay=1e-4`, `dropout=0.35`, `gradient_clip_norm=2.0`），然后直接做 1 次 formal confirmation。
> - 本轮运行 commit 为 `072e601`；由于这是 fixed-config confirmation 而不是 fresh study，直接在允许的单卡 fallback `GPU 2` 上运行 `train.py --config configs/autoresearch_formal_resnext_v100.yaml`。

- [x] **CMP-FAIR-V100-RESNEXT-FORMAL-CONFIRM**：`configs/autoresearch_formal_resnext_v100.yaml`（`head256`, `freeze_layers=2`, `lr=1.25e-4`, `weight_decay=1e-4`, `dropout=0.35`, `gradient_clip_norm=2.0`）→ `val_acc=0.925531914893617`, `val_auc=0.9563636363636363`, `val_f1=0.9176470588235294`, `peak_vram=6.23 GiB`, `total_seconds=824.0` → **keep**（较上一轮 main-study winner 的 `0.9042553191489362 / 0.9368181818181818` 再提升 `+0.0212765957446808` accuracy 与 `+0.0195454545454545` AUC，也较此前 fair-backbone stable winner `CMP-FAIR-V100-FORMAL-RESNEXT` 的 `0.9148936170212766 / 0.9404545454545454` 提升 `+0.0106382978723404` accuracy 与 `+0.0159090909090909` AUC；说明这组更激进的解冻与优化标量能够稳定转化为更强的 direct formal 证据。）
- **本轮结论**：上一轮 main-study 的 trial-3 角点不是一次性搜索噪声，而是当前独立 ResNeXt V100 side campaign 的新 best keep；它现在取代旧的 fair-backbone stable winner，成为该 campaign 最强的 retained evidence。
- **推荐动作**：若后续继续该独立 side campaign，应先做 1 次 alternate-seed formal confirmation（优先 `seed=123` 或 `seed=456`）来检验稳定性，再决定是否围绕这一新 anchor 开 fresh adaptive main study；不要回到 `head384`，也不要重新补非 `resnext` backbone。

---

## 2026-04-18：ResNeXt V100 Alternate-Seed Formal Confirmation（seed123）

> **独立 campaign 说明**
> - 这一轮继续承接 `fair_backbone_compare` stable-winner side campaign，只推进 `resnext`，不回到 canonical ResUNet 主线。
> - 保持 matched V100 几何与 retuned winner 标量完全不变：`image_size=256`, `num_slices_per_view=8`, `feature fusion`, `share_backbone=false`, `use_attention_pooling=false`（mean pooling），`fusion_hidden_dim=256`, `freeze_layers=2`, `lr=1.25e-4`, `weight_decay=1e-4`, `dropout=0.35`, `gradient_clip_norm=2.0`。
> - 唯一离散改动：把 dedicated formal template `configs/autoresearch_formal_resnext_v100.yaml` 的 `seed` 从 `42` 改为 `123`，然后直接做 1 次 alternate-seed formal confirmation。
> - 本轮运行 commit 为 `e6f36d0`；由于这是 fixed-config confirmation 而不是 fresh study，直接在允许的单卡 fallback `GPU 2` 上运行 `train.py --config configs/autoresearch_formal_resnext_v100.yaml`。

- [x] **CMP-FAIR-V100-RESNEXT-FORMAL-SEED123**：`configs/autoresearch_formal_resnext_v100.yaml`（仅 `seed=123`，其余保持当前 retuned winner 不变）→ `val_acc=0.8829787234042553`, `val_auc=0.91`, `val_f1=0.8571428571428571`, `peak_vram=6.23 GiB`, `total_seconds=827.9` → **discard**（较当前 retained keep `CMP-FAIR-V100-RESNEXT-FORMAL-CONFIRM` 的 `0.925531914893617 / 0.9563636363636363` 回落 `0.0425531914893617 / 0.0463636363636363`，也较旧 fair-backbone stable winner `CMP-FAIR-V100-FORMAL-RESNEXT` 的 `0.9148936170212766 / 0.9404545454545454` 低 `0.0319148936170213 / 0.0304545454545454`；说明当前新 anchor 至少在 `seed=123` 上并不稳定，不能把上一轮 keep 直接视为已完成多 seed 证实的新配方。）
- **本轮结论**：上一轮 direct formal keep 仍然是该独立 ResNeXt V100 side campaign 的最强单点证据，但它现在应被视为存在明显 seed sensitivity 的 anchor，而不是已经通过 alternate-seed confirmation 的稳定 winner。
- **推荐动作**：若后续继续该独立 side campaign，应优先补 1 次 `seed=456` direct formal confirmation，先判断这次回落是 `seed=123` 特例还是更普遍的不稳定；在此之前不要围绕当前 anchor 再开 fresh adaptive main study。

---

## 阶段 10：当前主线校准与可复现实验 🔴 当前最高优先级

> **当前主线定义（canonical baseline）**
> - 单张 CT 切片先进入以 ResNet18 为编码器的 `ResUNet`
> - 3 个 `Attention Gate` 突出病灶区域
> - 单视角 16 张切片经 `AttentionPooling` 聚合为视角特征
> - 3 个视角通过 `VRG` / `decision fusion` 做可靠度加权融合
> - 主指标是 `summary.json -> best_val.accuracy`
> - 若 `val_acc` 持平，按 `best_val.auc` 决胜
>
> **当前流程约束**
> - `model.freeze_layers` 必须由 YAML 显式声明，不再通过修改 `src/model.py` 常量切换
> - Optuna 必须优先使用项目 `.venv`
> - fresh study 默认不得复用旧 `study.sqlite3`；只有显式 `--resume` 才续跑
> - resume 语义是补足剩余 budget，而不是再次追加完整 `n_trials`
> - `threshold_eval` 仅作辅助输出；失败时记录 warning，不应判定整个 study 失败
>
> **当前判定规则**
> - keep / discard 比较对象：当前 canonical baseline 的 `val_acc`
> - `val_acc` 持平：比较 `val_auc`
> - 二者仍持平：优先更简单的配置 / 代码路径
>
> **执行顺序**
> 1. 跑 1 次 `configs/autoresearch_proxy.yaml`，重新确立当前主线 `val_acc` 基线
> 2. 如 baseline 正常，再启动 fresh proxy Optuna study（不得混入旧 trial）
> 3. 由 `monitor_optuna.py` 汇总 best trial、warning、degraded trial 与建议
> 4. proxy winner 明显更优后，再做 main/formal 确认
>
> **说明**
> - 历史 `VR-01 ~ VR-16`、`no_miss_*`、`192x16` 记录保留供回顾，但不再作为当前主线的直接 comparator
> - 若需要复盘旧阶段，请显式标注为 legacy campaign

### 2026-04-13：一轮基线校准 + capped fresh proxy Optuna + formal confirmation

- [x] **Baseline rerun**：`configs/autoresearch_proxy.yaml` 当前 canonical baseline（未加结构改动）→ `val_acc=0.8404255319148937`, `val_auc=0.9181818181818182`, `val_f1=0.8192771084337349`, `peak_vram=14.4 GiB`。
- [x] **VRG-LOGIT-01**：将 decision fusion 的 reliability head 从 `Linear -> Sigmoid` 改为 `LayerNorm -> Linear` raw logits（commit `a60c3e0`）→ fresh proxy Optuna study 在人工 GPU 切换后于 3090 上重启；收口时保留 4 个 completed trial，忽略已中断的第 5 个 running trial。
- [x] **VRG-LOGIT-01 / winner**：best completed trial 为 **trial 0**（当前模板超参：`batch_size=4`, `dropout=0.3`, `lr=5e-5`, `weight_decay=1e-3`, `augmentation=true`, `scheduler=none`, `label_smoothing=0.0`, `gradient_clip_norm=1.0`, `early_stopping_patience=2`）→ `val_acc=0.8829787234042553`, `val_auc=0.9318181818181819`, `val_f1=0.8764044943820225` → **keep**。
- [x] **Monitor takeaways**：当前结构对超参较敏感；在已完成 trial 中，`batch_size=4`、`dropout=0.3`、`label_smoothing=0.0`、`scheduler=none`、`weight_decay=1e-3` 明显优于小 batch / `cosine` / 更高 `label_smoothing`。
- [x] **Formal confirmation**：`c30fcae` 将 `configs/autoresearch_formal.yaml` 对齐到 proxy winner 模板超参后完成 1 次 formal（best epoch=4，early stop at epoch 6）→ `val_acc=0.8617021276595744`, `val_auc=0.9372727272727273`, `val_f1=0.8505747126436781`, `peak_vram=14.4 GiB` → **keep**（记为当前 canonical formal reference；`val_acc` 低于 proxy winner `0.0213`，但 `val_auc` 提升 `0.0055`）。
- [x] **Main-sync ledger note**：后续并入 `main@a94350a` 时，补齐了 baseline-only smoke `09531b1`（`val_acc=0.8404255319148937`, `val_auc=0.9181818181818182`）与 3090 五试次 r5 proxy winner `487dbed`（`val_acc=0.851063829787234`, `val_auc=0.9259090909090909`）两条 ledger 记录；它们已保留在 `results.tsv` 中作为 merged history，但均未超过当前本地 training winner `a60c3e0` 的 `0.8829787234042553`。
- [x] **CVT-MAN-01**：手动 constrained proxy 单点（`lr=4e-5`, `early_stopping_patience=2`；其余固定 `batch_size=4`, `dropout=0.3`, `label_smoothing=0.0`, `scheduler=none`, `weight_decay=1e-3`）→ `val_acc=0.8723404255319149`, `val_auc=0.9209090909090909`, `val_f1=0.8571428571428571`, `peak_vram=14.4 GiB` → **discard**（较 current proxy best `0.8829787234042553` 低 `0.0106382978723404`；accuracy 明显高于 baseline rerun `0.8404`，但仍未追平 winner，AUC 也低 `0.0109`）。
- [x] **CVT-MAN-02**：手动 constrained proxy 单点（`lr=6e-5`, `early_stopping_patience=2`；其余固定 `batch_size=4`, `dropout=0.3`, `label_smoothing=0.0`, `scheduler=none`, `weight_decay=1e-3`）→ `val_acc=0.8085106382978723`, `val_auc=0.9190909090909091`, `val_f1=0.8235294117647058`, `peak_vram=14.4 GiB` → **discard**（较 current proxy best `0.8829787234042553` 低 `0.07446808510638298`；高学习率边界明显退化，且 manual round 1 的较优趋势未能延续）。
- [x] **CVT-MAN-03**：手动 constrained proxy 单点（`lr=5e-5`, `early_stopping_patience=3`；其余固定 `batch_size=4`, `dropout=0.3`, `label_smoothing=0.0`, `scheduler=none`, `weight_decay=1e-3`）→ **crash**（OOM；日志显示目标 GPU 仅剩约 `223 MiB` 空闲，另一个 `.venv/bin/python` 进程已占用约 `16.9 GiB`，在 `src/model.py` attention gate 前向阶段申请额外 `256 MiB` 时失败；需待目标 GPU 清空后再做有效复验）。
- [x] **CVT-MAN-03R**：在确认 `CUDA_VISIBLE_DEVICES=1` 空闲后重跑同一 constrained proxy 单点（`lr=5e-5`, `early_stopping_patience=3`；commit `1f117dd`）→ `val_acc=0.8191489361702128`, `val_auc=0.89`, `val_f1=0.8316831683168316`, `peak_vram=14.4 GiB` → **discard**（验证了原 OOM 属外部占卡，但该 patience=3 点本身比 current proxy best 低 `0.06382978723404253`，也比 `CVT-MAN-01` 的 `lr=4e-5 + patience=2` 低 `0.05319148936170215`；说明当前主线下继续放宽 early stopping 并不有利）。
- [x] **CVT-MAN-04**：手动 constrained proxy 单点（`lr=4.5e-5`, `early_stopping_patience=2`；commit `092eea5`）→ `val_acc=0.851063829787234`, `val_auc=0.9013636363636364`, `val_f1=0.8333333333333334`, `peak_vram=14.4 GiB` → **discard**（较 current proxy best `0.8829787234042553` 低 `0.03191489361702133`，且较 `CVT-MAN-01` 的 `lr=4e-5 + patience=2` 低 `0.021276595744680882`；说明在 4e-5 与 5e-5 之间插值没有优于已知较优低学习率点，手动 sweep 更值得继续向更低 lr 边界探索）。
- [x] **CVT-MAN-05**：手动 constrained proxy 单点（`lr=3.5e-5`, `early_stopping_patience=2`；commit `aabeb35`）→ `val_acc=0.851063829787234`, `val_auc=0.9118181818181819`, `val_f1=0.8292682926829268`, `peak_vram=14.4 GiB` → **discard**（较 current proxy best `0.8829787234042553` 低 `0.03191489361702133`，与 `CVT-MAN-04` 持平 accuracy 但 AUC 高 `0.0104545454545455`，仍低于 `CVT-MAN-01` 的 `0.8723404255319149`；说明当前受限 lr sweep 两侧新边界 `3.5e-5 / 4.5e-5` 都未形成新的 accuracy 峰值，可先结束这一小段手动 lr 搜索）。
- [x] **CVT-MAN-06**：exact winner reproduction（`seed=123`, `lr=5e-5`, `early_stopping_patience=2`；commit `eddf885`）→ `val_acc=0.8404255319148937`, `val_auc=0.9177272727272727`, `val_f1=0.8148148148148148`, `peak_vram=14.4 GiB` → **discard**（较 current proxy winner `0.8829787234042553` 低 `0.0425531914893616`，且 accuracy 与 canonical baseline rerun `0.8404255319148937` 完全相同、AUC 还低 `0.0004545454545455`；说明 trial-0 winner 未在 alternate seed 上复现，当前更需要做稳定性导向的 `weight_decay / dropout` 探针，而不是继续受限 lr 微调）。
- [x] **CVT-MAN-07**：winner-template stability probe（`seed=42`, `weight_decay=5e-4`；commit `336cd96`）→ `val_acc=0.8829787234042553`, `val_auc=0.9263636363636365`, `val_f1=0.8705882352941177`, `peak_vram=14.4 GiB` → **discard**（与 current proxy winner `0.8829787234042553` accuracy 完全持平，但 AUC 低 `0.0054545454545454`，因此按 tie-break 仍输给 `weight_decay=1e-3` 的 trial-0 winner；说明较弱 L2 正则不会立刻伤到 accuracy，但当前 canonical winner 在排序质量上更稳，下一步更值得转测 `dropout=0.25/0.35`，而不是继续下调 weight decay）。
- [x] **CVT-MAN-08**：winner-template stability probe（`seed=42`, `dropout=0.25`；commit `51f8199`）→ `val_acc=0.8404255319148937`, `val_auc=0.9163636363636364`, `val_f1=0.8235294117647058`, `peak_vram=14.4 GiB` → **discard**（较 current proxy winner `0.8829787234042553` 低 `0.0425531914893616`，accuracy 与 canonical baseline rerun `0.8404255319148937` 完全相同，AUC 还低 `0.0154545454545455`；说明把 canonical winner 的 dropout 从 `0.3` 下调到 `0.25` 会明显削弱当前正则平衡，下一步若继续应优先测试 `dropout=0.35` 而不是继续下调 dropout。）
- [x] **CVT-MAN-09**：winner-template stability probe（`seed=42`, `dropout=0.35`；commit `7b25d8d`）→ `val_acc=0.8191489361702128`, `val_auc=0.8727272727272728`, `val_f1=0.7792207792207793`, `peak_vram=14.4 GiB` → **discard**（较 current proxy winner `0.8829787234042553` 低 `0.06382978723404253`，也比 canonical baseline rerun `0.8404255319148937` 再低 `0.021276595744680882`；说明把 canonical winner 的 dropout 从 `0.3` 上调到 `0.35` 会进一步过正则化，至此 `dropout=0.25/0.35` 两侧探针均失败，下一步应停止 dropout 轴并转向 alternate-seed confirmation。）
- [x] **CVT-MAN-10**：exact winner reproduction（`seed=456`, `lr=5e-5`, `weight_decay=1e-3`, `dropout=0.3`, `early_stopping_patience=2`; commit `09cb59b`）→ `val_acc=0.8404255319148937`, `val_auc=0.8718181818181818`, `val_f1=0.810126582278481`, `peak_vram=14.4 GiB` → **discard**（较 current proxy winner `0.8829787234042553` 低 `0.0425531914893616`，accuracy 与 canonical baseline rerun `0.8404255319148937` 完全相同，且 AUC 还低 `0.0463636363636364`；说明 winner 模板在 `seed=456` 上同样回落到 baseline-level accuracy。结合此前 `seed=123` 复验失败，当前应停止继续做 exact rerun，转向稳定性导向的 `weight_decay` 轴探针。）
- [x] **CVT-MAN-11**：winner-template stability probe（`seed=456`, `weight_decay=5e-4`; commit `c8f125c`）→ `val_acc=0.8297872340425532`, `val_auc=0.9163636363636364`, `val_f1=0.7948717948717948`, `peak_vram=14.4 GiB` → **discard**（较 current proxy winner `0.8829787234042553` 低 `0.0531914893617021`，也较 `CVT-MAN-10` 的 exact `seed=456` rerun `0.8404255319148937` 再低 `0.0106382978723405`；但 AUC 较 `CVT-MAN-10` 明显回升 `0.0445454545454546`。说明减弱 L2 正则能修复排序质量，却没有把 accuracy 拉回 baseline-level；下一步应继续留在 `weight_decay` 轴，在 `5e-4` 与 `1e-3` 之间补测中间点，而不是切换到其他超参。）
- [x] **CVT-MAN-12**：winner-template stability probe（`seed=456`, `weight_decay=7.5e-4`; commit `0001398`）→ `val_acc=0.8191489361702128`, `val_auc=0.9059090909090909`, `val_f1=0.7848101265822784`, `peak_vram=14.4 GiB` → **discard**（较 current proxy winner `0.8829787234042553` 低 `0.06382978723404253`，较 `CVT-MAN-10` 的 exact `seed=456` rerun `0.8404255319148937` 低 `0.021276595744680882`，也较 `CVT-MAN-11` 的 `weight_decay=5e-4` 探针再低 `0.0106382978723404`；虽然 `val_auc` 仍较 exact rerun 回升 `0.0340909090909091`，但 accuracy 已落到两个端点之下。说明在 `seed=456` 上，`7.5e-4` 不是可用折中点；若继续沿此轴细化，应把搜索重心移回更靠近 `1e-3` 的上侧，而不是继续往更低 L2 走。）
- [x] **CVT-MAN-13**：winner-template stability probe（`seed=456`, `weight_decay=8.75e-4`; commit `4363c40`）→ `val_acc=0.8297872340425532`, `val_auc=0.899090909090909`, `val_f1=0.8260869565217391`, `peak_vram=14.4 GiB` → **discard**（较 current proxy winner `0.8829787234042553` 低 `0.0531914893617021`，较 `CVT-MAN-10` 的 exact `seed=456` rerun `0.8404255319148937` 仍低 `0.0106382978723405`；虽然较 `CVT-MAN-12` 的 `7.5e-4` 点回升 `0.0106382978723404` accuracy，但它只与 `CVT-MAN-11` 的 `5e-4` 探针持平，并在 AUC 上反而低 `0.0172727272727274`。说明回到更靠近 `1e-3` 的上侧能补回 1 个 case，却没有恢复 baseline-level accuracy，也没保住较弱 L2 的排序收益；若继续留在此轴，剩余最有信息量的点应是 `9e-4`，否则可考虑结束该轴。）
- [x] **CVT-MAN-14**：winner-template stability probe（`seed=456`, `weight_decay=9e-4`; commit `5c45a03`）→ `val_acc=0.8404255319148937`, `val_auc=0.8763636363636363`, `val_f1=0.810126582278481`, `peak_vram=14.4 GiB` → **discard**（较 current proxy winner `0.8829787234042553` 低 `0.0425531914893616`，但与 `CVT-MAN-10` 的 exact `seed=456` rerun accuracy 完全持平，且 AUC 回升 `0.0045454545454545`；同时较 `CVT-MAN-13` 的 `8.75e-4` 点补回 `0.0106382978723405` accuracy，却丢掉 `0.0227272727272727` AUC。说明 `9e-4` 是当前 `seed=456` / upper-side `weight_decay` 细化里唯一把 accuracy 拉回 baseline-level 的折中点，但其排序收益有限、且与 current proxy winner 仍有明显差距；下一步应停止继续细插 `seed=456` 的 L2 轴，转去验证该折中是否能在其他 alternate seed 上复现。）
- [x] **CVT-MAN-15**：winner-template stability probe（`seed=123`, `weight_decay=9e-4`; commit `3093f7f`）→ `val_acc=0.8404255319148937`, `val_auc=0.9209090909090909`, `val_f1=0.8148148148148148`, `peak_vram=14.4 GiB` → **discard**（较 current proxy winner `0.8829787234042553` 低 `0.0425531914893616`；与 `CVT-MAN-06` 的 exact `seed=123` rerun accuracy 完全持平，但 AUC 回升 `0.0031818181818182`；与 `CVT-MAN-14` 的 `seed=456`, `weight_decay=9e-4` 对照点 accuracy 持平，但 AUC 高 `0.0445454545454546`。说明把 `weight_decay` 从 `1e-3` 下调到 `9e-4` 在 `seed=123` 上只修复了少量排序质量，没有恢复 winner-level accuracy；结合 `seed=456` 侧同样仅能回到 baseline-level accuracy，可判定 alternate-seed L2 rescue 已缺乏继续投入价值，后续应转入新的训练稳定化或结构方向。）

- [x] **CVT-STRAT-01**：winner-template optimization-stability probe（`seed=42`, `gradient_clip_norm=0.5`; commit `e3184b6`; 其余固定 `batch_size=4`, `dropout=0.3`, `lr=5e-5`, `weight_decay=1e-3`, `augmentation=true`, `scheduler=none`, `label_smoothing=0.0`, `early_stopping_patience=2`）→ `val_acc=0.8191489361702128`, `val_auc=0.89`, `val_f1=0.8316831683168316`, `peak_vram=14.4 GiB` → **discard**（较 current proxy winner `0.8829787234042553` 低 `0.0638297872340425`，较 canonical baseline rerun `0.8404255319148937` 也低 `0.0212765957446809`；并与 `CVT-MAN-03R` 的 `patience=3` 失败点完全同分。说明把 `gradient_clip_norm` 从 `1.0` 强化到 `0.5` 会显著过度约束当前 winner 模板的优化，未能换来更好的稳定性，应停止继续向更强 clipping 推进。）
- [x] **CVT-STRAT-02**：winner-template optimization-stability probe（`seed=42`, `gradient_clip_norm=0.75`; commit `5402471`; 其余固定 `batch_size=4`, `dropout=0.3`, `lr=5e-5`, `weight_decay=1e-3`, `augmentation=true`, `scheduler=none`, `label_smoothing=0.0`, `early_stopping_patience=2`）→ `val_acc=0.8191489361702128`, `val_auc=0.8909090909090909`, `val_f1=0.8316831683168316`, `peak_vram=14.4 GiB` → **discard**（较 current proxy winner `0.8829787234042553` 低 `0.0638297872340425`，较 canonical baseline rerun `0.8404255319148937` 也低 `0.0212765957446809`；与 `CVT-STRAT-01` 的 `gradient_clip_norm=0.5` 相比 accuracy / F1 完全相同，仅 AUC 微升 `0.0009090909090909`。说明更温和的 clip 中间点仍未把优化拉回 winner-level 轨道，当前 `1.0` 更像可用下界而非可继续下探的冗余余量。）
- [x] **CVT-STRAT-03**：winner-template optimization-stability upper-side probe（`seed=42`, `gradient_clip_norm=1.25`; commit `57bd81c`; 其余固定 `batch_size=4`, `dropout=0.3`, `lr=5e-5`, `weight_decay=1e-3`, `augmentation=true`, `scheduler=none`, `label_smoothing=0.0`, `early_stopping_patience=2`）→ `val_acc=0.8191489361702128`, `val_auc=0.905`, `val_f1=0.8`, `peak_vram=14.4 GiB` → **discard**（较 current proxy winner `0.8829787234042553` 低 `0.0638297872340425`，较 canonical baseline rerun `0.8404255319148937` 也低 `0.0212765957446809`；与 `CVT-STRAT-01/02` 相比 accuracy 完全相同，但 AUC 分别回升 `0.015` / `0.0140909090909091`，同时 F1 降到 `0.8`。说明继续放松 clipping 只能修复部分排序质量，无法把阈值决策拉回 winner-level，因此 `gradient_clip_norm=1.0` 可视为当前 winner 模板的 clipping sweet spot，clipping 轴到此结束。）
- [x] **CVT-REG-01**：winner-template regularization upper-side probe（`seed=42`, `weight_decay=1.25e-3`; commit `6fd3d06`; 其余固定 `batch_size=4`, `dropout=0.3`, `lr=5e-5`, `augmentation=true`, `scheduler=none`, `label_smoothing=0.0`, `gradient_clip_norm=1.0`, `early_stopping_patience=2`）→ `val_acc=0.8191489361702128`, `val_auc=0.8927272727272728`, `val_f1=0.8316831683168316`, `peak_vram=14.4 GiB` → **discard**（较 current proxy winner `0.8829787234042553` 低 `0.0638297872340425`，较 canonical baseline rerun `0.8404255319148937` 也低 `0.0212765957446809`；与 `CVT-STRAT-03` 相比 accuracy 完全相同，但 AUC 低 `0.0122727272727272`、F1 高 `0.0316831683168316`。说明继续增强 L2 正则无法保住 winner-level accuracy，当前 `weight_decay=1e-3` 基本可视为 winner 模板的 upper-side sweet spot，纯标量正则微调应降级。）
- [x] **CVT-VRG-HEAD-01**：winner-template gate-structure probe（`seed=42`; 把每个 VRG reliability head 从 `LayerNorm(feature) -> Linear(512,1)` 改为接收 `concat(view feature, view logits)` 的小型 MLP；commit `5e33d3e`; 其余固定 `batch_size=4`, `dropout=0.3`, `lr=5e-5`, `weight_decay=1e-3`, `augmentation=true`, `scheduler=none`, `label_smoothing=0.0`, `gradient_clip_norm=1.0`, `early_stopping_patience=2`）→ `val_acc=0.8191489361702128`, `val_auc=0.8281818181818182`, `val_f1=0.7848101265822784`, `peak_vram=14.4 GiB` → **discard**（较 current proxy winner `0.8829787234042553` 低 `0.0638297872340425`，较 canonical baseline rerun `0.8404255319148937` 也低 `0.0212765957446809`；与同 accuracy 的 `CVT-REG-01` 相比，AUC 低 `0.0645454545454546`、F1 低 `0.0468730417345532`。说明显式类别证据并没有修复 gate，反而进一步损伤排序质量，结构瓶颈更可能在视角交互不足而不是 reliability head 输入缺失。）
- [x] **CVT-XVIEW-01**：winner-template cross-view context probe（`seed=42`; 在 slice-pooling 后对 3 个 `512D` view features 加 1 层轻量 residual `CrossViewAttention`（4 heads, 1 layer, dropout=0.1），再送入现有 per-view classifier + raw-logit VRG；commit `5b83804`; 其余固定 `batch_size=4`, `dropout=0.3`, `lr=5e-5`, `weight_decay=1e-3`, `augmentation=true`, `scheduler=none`, `label_smoothing=0.0`, `gradient_clip_norm=1.0`, `early_stopping_patience=2`）→ `val_acc=0.851063829787234`, `val_auc=0.9036363636363637`, `val_f1=0.8205128205128205`, `peak_vram=14.5 GiB` → **discard**（较 current proxy winner `0.8829787234042553` 低 `0.0319148936170213`，但较 canonical baseline rerun `0.8404255319148937` 高 `0.0106382978723403`，也较 `CVT-VRG-HEAD-01` 回升 `0.0319148936170212`；不过 `val_auc` 仍较 canonical baseline 低 `0.0145454545454545`。说明跨视角上下文本身可能能补回部分 accuracy，但当前 Transformer-style mixer 仍明显扰动排序质量，信息量足够支持继续沿更低容量的 token-mixing 方向细化，而不是回到 gate head-only 改动。）
- [x] **CVT-XVIEW-02**：winner-template lower-capacity cross-view context probe（`seed=42`; 保留 current winner + raw-logit VRG，把 `CVT-XVIEW-01` 的 1 层 residual `CrossViewAttention` 改为 residual mean-context bottleneck MLP，只在 3 个 pooled `512D` view features 间注入少量上下文后再送入现有 per-view classifier + VRG；commit `d617544`）→ `val_acc=0.8297872340425532`, `val_auc=0.824090909090909`, `val_f1=0.7948717948717948`, `peak_vram=14.4 GiB`, `total_seconds=603.2` → **discard**（较 current proxy winner `0.8829787234042553` 低 `0.0531914893617021`，较 canonical baseline rerun `0.8404255319148937` 也低 `0.0106382978723405`，并较 `CVT-XVIEW-01` 再低 `0.0212765957446808`；`val_auc` 也较 canonical baseline 低 `0.0940909090909092`。说明把 attention 换成更低容量的 mean-context mixing 后，`CVT-XVIEW-01` 的局部 accuracy 回升并没有保住，反而表明“直接改写 pooled 类别特征”这条路径本身不稳；若继续沿跨视角方向，应把上下文限制在 VRG reliability estimation，而不是继续作用在 per-view classifier 输入上。）
- [x] **CVT-XVIEW-03**：winner-template context-for-gating-only probe（`seed=42`; 保留 current winner 的 per-view classifier 完全不接收 cross-view mixing，只给 raw-logit VRG reliability heads 增加 additive other-view mean bias 分支，让跨视角信息只影响 fusion weighting；commit `5bada64`）→ `val_acc=0.7659574468085106`, `val_auc=0.86`, `val_f1=0.725`, `peak_vram=14.4 GiB`, `total_seconds=460.8` → **discard**（较 current proxy winner `0.8829787234042553` 低 `0.1170212765957447`，较 canonical baseline rerun `0.8404255319148937` 也低 `0.0744680851063831`，并较 `CVT-XVIEW-02` 再低 `0.0638297872340426`；虽然 `val_auc` 较 `CVT-XVIEW-02` 回升 `0.035909090909091`，但仍较 canonical baseline 低 `0.0581818181818182`。说明把 other-view mean 仅作为 gating bias 仍会显著破坏准确率，跨视角信息至少不能以这种低容量均值上下文形式直接校准 VRG；`CVT-XVIEW-01` 的局部收益更可能来自 richer token interaction，或只是额外非线性容量带来的偶然回升。）
- [x] **CVT-SELF-01**：winner-template matched-capacity control probe（`seed=42`; 保留 current winner 的 raw-logit VRG，不做任何 cross-view mixing，只在 per-view classifier 前加入一个参数量与 `CVT-XVIEW-02` 接近的 residual self-MLP；commit `3ef1511`; 其余固定 `batch_size=4`, `dropout=0.3`, `lr=5e-5`, `weight_decay=1e-3`, `augmentation=true`, `scheduler=none`, `label_smoothing=0.0`, `gradient_clip_norm=1.0`, `early_stopping_patience=2`）→ `val_acc=0.8297872340425532`, `val_auc=0.8513636363636363`, `val_f1=0.8048780487804879`, `peak_vram=14.4 GiB`, `total_seconds=600.6` → **discard**（较 current proxy winner `0.8829787234042553` 低 `0.0531914893617021`，较 canonical baseline rerun `0.8404255319148937` 也低 `0.0106382978723405`，并较 `CVT-XVIEW-01` 再低 `0.0212765957446808`；虽然较 `CVT-XVIEW-02` 回升 `0.0272727272727273` AUC，但仍较 canonical baseline 低 `0.0668181818181819`。说明单纯增加 matched-capacity per-view 非线性容量不能解释 `CVT-XVIEW-01` 的局部 accuracy 回升；如果继续沿跨视角方向，应保留真实 token interaction，同时进一步限制其残差混合幅度。）
- [x] **CVT-XVIEW-04**：winner-template residual-gated cross-view probe（`seed=42`; 回到 `CVT-XVIEW-01` 的真实 cross-view token interaction，但只在 classifier 分支使用 1 层 residual `CrossViewAttention`（4 heads, 1 layer, dropout=0.1）且固定 `residual_scale=0.25`，VRG reliability head 保持 baseline raw-logit pooled-feature 路径；commit `1a318ad`）→ `val_acc=0.851063829787234`, `val_auc=0.9027272727272727`, `val_f1=0.8292682926829268`, `peak_vram=14.5 GiB`, `total_seconds=596.0` → **discard**（较 current proxy winner `0.8829787234042553` 低 `0.0319148936170213`，但较 canonical baseline rerun `0.8404255319148937` 高 `0.0106382978723403`；accuracy 与 `CVT-XVIEW-01` 完全持平，但 AUC 低 `0.000909090909091`。说明弱化 residual 注入幅度、并把 VRG 保持在 baseline 路径，并没有实质改善 cross-view attention 的排序质量；下一步更值得约束“交互通道秩”而不只是交互幅度。）
- [x] **CVT-XVIEW-05**：winner-template bottlenecked cross-view probe（`seed=42`; 保留 `CVT-XVIEW-04` 的 classifier-only cross-view 路径和 baseline raw-logit VRG，但先把每个 `512D` pooled view token 投影到 `128D` bottleneck 后做 1 层 attention，再投影回 per-view classifier 输入；commit `9690f2e`）→ `val_acc=0.7978723404255319`, `val_auc=0.9254545454545454`, `val_f1=0.7865168539325843`, `peak_vram=14.4 GiB`, `total_seconds=596.5` → **discard**（较 current proxy winner `0.8829787234042553` 低 `0.0851063829787234`，较 canonical baseline rerun `0.8404255319148937` 也低 `0.0425531914893618`；但 `val_auc` 较 `CVT-XVIEW-04` 回升 `0.0227272727272727`，也较 canonical baseline rerun 高 `0.0072727272727272`。说明“低秩真实 token interaction”并非完全无效，它修复了排序质量，却在 `128D` 这个过窄通道上明显损伤了阈值准确率；若继续沿该方向，应优先测试更宽的中等秩 bottleneck，而不是继续压缩。）
- [x] **CVT-XVIEW-06**：winner-template medium-rank bottleneck probe（`seed=42`; 保留 `CVT-XVIEW-04/05` 的 classifier-only cross-view 路径和 baseline raw-logit VRG，但把 pooled `512D` view token 先投影到 `256D` bottleneck 后做 1 层 attention，再投影回 per-view classifier 输入；commit `5c52f70`）→ `val_acc=0.8404255319148937`, `val_auc=0.9104545454545454`, `val_f1=0.810126582278481`, `peak_vram=14.4 GiB`, `total_seconds=594.9` → **discard**（较 current proxy winner `0.8829787234042553` 低 `0.0425531914893616`；与 canonical baseline rerun accuracy 完全持平，但 AUC 低 `0.0077272727272728`。较 `CVT-XVIEW-05` 回升 `0.0425531914893618` accuracy，却回吐 `0.015` AUC；较 `CVT-XVIEW-04` 则补回 `0.0077272727272727` AUC，但又损失 `0.0106382978723403` accuracy。说明 `256D` 只是把 `128D` 的 trade-off 拉回中间态，没有形成新的 accuracy 峰值；若继续沿 bottleneck-rank 轴，应优先测试更靠近 full-rank 的 `384D`，否则可以考虑结束此轴。）
- [x] **CVT-XVIEW-07**：winner-template high-rank bottleneck probe（`seed=42`; 保留 `CVT-XVIEW-04~06` 的 classifier-only cross-view 路径和 baseline raw-logit VRG，但把 pooled `512D` view token 先投影到 `384D` bottleneck 后做 1 层 attention，再投影回 per-view classifier 输入；commit `7d9fad9`）→ `val_acc=0.8191489361702128`, `val_auc=0.875`, `val_f1=0.7671232876712328`, `peak_vram=14.4 GiB`, `total_seconds=596.3` → **discard**（较 current proxy winner `0.8829787234042553` 低 `0.0638297872340425`，较 canonical baseline rerun `0.8404255319148937` 也低 `0.0212765957446809`；相对 `CVT-XVIEW-06` accuracy 再低 `0.0212765957446809`、`val_auc` 再低 `0.0354545454545454`，也同时低于 `CVT-XVIEW-04` 的 `0.851063829787234 / 0.9027272727272727`。说明 `384D` 高秩 bottleneck 没有把性能拉回 full-rank 控制点，反而在 accuracy 与 ranking 上同时退化；由此可结束 bottleneck-rank 轴。）
- [x] **CVT-XVIEW-08**：winner-template minimal-residual full-rank cross-view probe（`seed=42`; 结束 bottleneck-rank 轴，回到 `CVT-XVIEW-04` 的 classifier-only full-rank `CrossViewAttention` + baseline raw-logit VRG，但把 `residual_scale` 从 `0.25` 再下调到 `0.125`；commit `bdf1ef3`）→ `val_acc=0.8404255319148937`, `val_auc=0.8859090909090909`, `val_f1=0.8051948051948052`, `peak_vram=14.5 GiB`, `total_seconds=610.6` → **discard**（较 current proxy winner `0.8829787234042553` 低 `0.0425531914893616`；accuracy 与 canonical baseline rerun 持平，但 `val_auc` 低 `0.0322727272727273`；相对 `CVT-XVIEW-04` 也低 `0.0106382978723403` accuracy 与 `0.0168181818181818` AUC。说明 minimal-residual full-rank 注入同样无法保住此前的局部 accuracy 回升，因此 classifier-side XVIEW 方向到此结束。）
- [x] **CVT-VRG-CTX-01**：winner-template attention-for-gating-only probe（`seed=42`; 保留当前 winner 的 per-view classifier logits 完全走 baseline 路径，只在 reliability estimation 分支对 3 个 pooled `512D` view features 加 1 层轻量 full-rank `CrossViewAttention`，并以弱残差 context 驱动 baseline raw-logit confidence heads；commit `eb8e64b`）→ `val_acc=0.8404255319148937`, `val_auc=0.894090909090909`, `val_f1=0.8192771084337349`, `peak_vram=14.5 GiB`, `total_seconds=623.5` → **discard**（较 current proxy winner `0.8829787234042553` 低 `0.0425531914893616`；accuracy 与 canonical baseline rerun 持平，但 `val_auc` 低 `0.0240909090909092`。相对 `CVT-XVIEW-03` 的 mean-bias gating-only 失败点，accuracy 回升 `0.0744680851063831`、`val_auc` 回升 `0.034090909090909`，说明 richer attention context 确实更稳；但它仍只回到 baseline-level accuracy，没有任何净收益，因此新的 cross-view 结构线到此一起收束。）

---


## 用户定向 rerun Campaign ✅ 已完成

> 本轮会话覆盖常规 backlog 优先级，只执行 Plan 2 / Plan 6 的定向 rerun。
> 判定规则（仅本 campaign 生效）：以 `threshold_eval.json` 为准，**仅当 `no_miss_val_acc > 0.75` 时记为 keep**；否则记为 discard。
> 固定顺序：先 `R2-01 → R2-08`，再 `R6-01 → R6-08`。
> 当前状态：**16/16 已完成，rerun queue exhausted**。

### Plan 2：View Reliability Gating（8 次 proxy rerun）

- [x] **R2-01**: baseline（fusion_type=decision, share_backbone=false, aug=true, lr=1e-4, dropout=0.3, label_smoothing=0.0）→ no_miss_val_acc=0.723, no_miss_val_spe=0.480, val_AUC=0.923 → discard
- [x] **R2-02**: lr=5e-5 → no_miss_val_acc=0.777, no_miss_val_spe=0.580, val_AUC=0.951 → keep（campaign keep；未超过全局最佳 0.787）
- [x] **R2-03**: lr=7e-5 → no_miss_val_acc=0.723, no_miss_val_spe=0.480, val_AUC=0.946 → discard
- [x] **R2-04**: dropout=0.2 → no_miss_val_acc=0.532, no_miss_val_spe=0.120, val_AUC=0.945 → discard
- [x] **R2-05**: dropout=0.4 → no_miss_val_acc=0.660, no_miss_val_spe=0.360, val_AUC=0.916 → discard
- [x] **R2-06**: label_smoothing=0.05 → no_miss_val_acc=0.617, no_miss_val_spe=0.280, val_AUC=0.916 → discard
- [x] **R2-07**: lr=5e-5 + label_smoothing=0.05 → no_miss_val_acc=0.723, no_miss_val_spe=0.480, val_AUC=0.935 → discard
- [x] **R2-08**: 基于前 7 次 rerun 最佳方向的组合实验（沿用最佳 lr=5e-5，并恢复 baseline label_smoothing=0.0）→ no_miss_val_acc=0.798, no_miss_val_spe=0.620, val_AUC=0.960 → keep ✅ 当前最优

### Plan 6：Asymmetric Safety-Biased Fusion（8 次 proxy rerun）

- [x] **R6-01**: temperature=0.5 baseline（fusion_type=decision, share_backbone=false, aug=true, lr=1e-4, dropout=0.3, label_smoothing=0.0）→ no_miss_val_acc=0.713, no_miss_val_spe=0.460, val_AUC=0.938 → discard
- [x] **R6-02**: temperature=1.0 → no_miss_val_acc=0.681, no_miss_val_spe=0.400, val_AUC=0.939 → discard
- [x] **R6-03**: temperature=0.3 → no_miss_val_acc=0.660, no_miss_val_spe=0.360, val_AUC=0.957 → discard
- [x] **R6-04**: temperature=2.0 → no_miss_val_acc=0.755, no_miss_val_spe=0.540, val_AUC=0.954 → keep（campaign keep；达到阈值 0.75，但未超过当前最优 0.798）
- [x] **R6-05**: temperature=0.5 + lr=5e-5 → no_miss_val_acc=0.649, no_miss_val_spe=0.340, val_AUC=0.941 → discard
- [x] **R6-06**: temperature=0.5 + dropout=0.2 → no_miss_val_acc=0.660, no_miss_val_spe=0.360, val_AUC=0.945 → discard
- [x] **R6-07**: temperature=0.5 + label_smoothing=0.05 → no_miss_val_acc=0.723, no_miss_val_spe=0.480, val_AUC=0.921 → discard
- [x] **R6-08**: 基于前 7 次 rerun 最佳方向的组合实验（沿用最佳 temperature=2.0，并恢复 baseline lr=1e-4 / dropout=0.3 / label_smoothing=0.0）→ no_miss_val_acc=0.670, no_miss_val_spe=0.380, val_AUC=0.935 → discard

---

## 阶段 10：历史方差缩减记录（legacy）

> 以下 `VR-*` 记录来自旧的 stage-10 / `no_miss_*` / 192x16 语义，保留作历史参考。
> 它们不再作为当前 canonical baseline 的直接比较对象；当前执行口径以上文“当前主线校准与可复现实验”为准。

### legacy：freeze_layers=3 / freeze_layers=2 搜索记录

- [x] **VR-01**: baseline（freeze=3, lr=5e-5, dropout=0.3, ls=0.0, gradient_clip_norm=1.0）→ no_miss_val_acc=0.904, no_miss_val_spe=0.820, val_AUC=0.968 → keep ✅（新阶段 10 proxy 最优；超过 `814f491` 的 0.872）
- [x] **VR-02**: lr=1e-4（冻结后可能需要更高 lr，因为可训练参数更少）→ no_miss_val_acc=0.830, no_miss_val_spe=0.680, val_AUC=0.970 → discard（AUC 略升，但零漏诊 val_acc 明显低于当前 proxy 最优 0.904）
- [x] **VR-03**: lr=3e-5（更保守的 lr）→ no_miss_val_acc=0.819, no_miss_val_spe=0.660, val_AUC=0.958 → discard（较当前 proxy 最优 0.904 下降 0.085，未见更低 lr 带来的稳定性收益）
- [x] **VR-04**: dropout=0.2（冻结后过拟合风险降低，dropout 可减小）→ no_miss_val_acc=0.809, no_miss_val_spe=0.640, val_AUC=0.966 → discard（较当前 proxy 最优 0.904 下降 0.096；降低 dropout 未能提升零漏诊表现，且特异度回落）
- [x] **VR-05**: dropout=0.4 → no_miss_val_acc=0.872, no_miss_val_spe=0.760, val_AUC=0.965 → discard（较 `VR-04` 回升 0.063、特异度回升 0.120，但仍低于当前 proxy 最优 `f6f4ed8` 的 0.904；说明更强 dropout 只能部分修复性能下滑）
- [x] **VR-06**: weight_decay=0.001（更强正则化，配合冻结进一步约束参数）→ no_miss_val_acc=0.883, no_miss_val_spe=0.780, val_AUC=0.966 → discard（较当前 proxy 最优 `f6f4ed8` 的 0.904 下降 0.021，但较 `VR-05` 小幅回升 0.011；更强 weight_decay 略有帮助，但仍未恢复到 freeze=3 baseline）
- [x] **VR-07**: lr=1e-4 + dropout=0.2（冻结后最可能的最优组合）→ no_miss_val_acc=0.904, no_miss_val_spe=0.820, val_AUC=0.968 → discard（与当前 proxy 最优 `f6f4ed8` 的零漏诊 val_acc / val_AUC / val_spe 全部追平，按简洁性原则保留更简单的 `VR-01` baseline）
- [x] **VR-08**: 基于前 7 次最佳方向的组合实验（lr=1e-4 + weight_decay=0.001，保留 baseline dropout=0.3）→ no_miss_val_acc=0.883, no_miss_val_spe=0.780, val_AUC=0.964 → discard（较当前 proxy 最优 `f6f4ed8` 的 0.904 下降 0.021；与 `VR-06` 零漏诊指标完全持平但 AUC 更低，说明高 lr + 更强 weight_decay 组合未带来额外收益）

### legacy：freeze_layers=2 超参搜索（8 次 proxy）

> 以下 freeze=2 记录同样属于 legacy stage-10；其中提到的 `DEFAULT_FREEZE_LAYERS` 只是当时的历史控制方式，不代表当前流程。

- [x] **VR-09**: baseline（freeze=2, lr=5e-5, dropout=0.3, ls=0.0, gradient_clip_norm=1.0）→ no_miss_val_acc=0.883, no_miss_val_spe=0.780, val_AUC=0.961 → discard（较当前 proxy 最优 `f6f4ed8` 的 0.904 下降 0.021；说明仅减少一层冻结并不能自动带来收益，后续转向 lr 调整验证可塑性是否仍可释放）
- [x] **VR-10**: lr=1e-4 → no_miss_val_acc=0.564, no_miss_val_spe=0.180, val_AUC=0.927 → discard（较当前 proxy 最优 `f6f4ed8` 的 0.904 下降 0.340；更高 lr 在 freeze=2 设置下显著压低零漏诊阈值至 0.019，导致 41 个假阳性、特异度崩塌）
- [x] **VR-11**: lr=3e-5 → no_miss_val_acc=0.809, no_miss_val_spe=0.640, val_AUC=0.962 → discard（较当前 proxy 最优 `f6f4ed8` 的 0.904 下降 0.096，且较 freeze=2 baseline `fc760cb` 的 0.883 再降 0.074；说明进一步下调 lr 不能释放 layer3 的额外可塑性，反而扩大假阳性）
- [x] **VR-12**: dropout=0.2 → no_miss_val_acc=0.787, no_miss_val_spe=0.600, val_AUC=0.955 → discard（较当前 proxy 最优 `f6f4ed8` 的 0.904 下降 0.117，且较 `VR-11` 再降 0.021；说明在 freeze=2 设置下减弱 dropout 会进一步放大假阳性，下一步转测 dropout=0.4）
- [x] **VR-13**: dropout=0.4 → no_miss_val_acc=0.766, no_miss_val_spe=0.560, val_AUC=0.961 → discard（较当前 proxy 最优 `f6f4ed8` 的 0.904 下降 0.138，较 freeze=2 baseline `fc760cb` 的 0.883 下降 0.117，且较 `VR-12` 再降 0.021；说明更强 dropout 仍未抑制 layer3 解冻后的假阳性，下一步转测 weight_decay=0.001）
- [x] **VR-14**: weight_decay=0.001 → no_miss_val_acc=0.830, no_miss_val_spe=0.680, val_AUC=0.960 → discard（较当前 proxy 最优 `f6f4ed8` 的 0.904 下降 0.074，较 freeze=2 baseline `fc760cb` 的 0.883 下降 0.053，但较 `VR-13` 的 0.766 回升 0.064；说明更强 L2 正则能部分抑制假阳性，但仍未恢复到 freeze=3/2 baseline 水平）
- [x] **VR-15**: lr=1e-4 + dropout=0.2 → no_miss_val_acc=0.628, no_miss_val_spe=0.300, val_AUC=0.948 → discard（较当前 proxy 最优 `f6f4ed8` 的 0.904 下降 0.277，较 freeze=2 baseline `fc760cb` 的 0.883 下降 0.255，且仅较 `VR-10` 的 0.564 回升 0.064；说明 freeze=2 路线对高 lr 仍极敏感，减小 dropout 不能修复假阳性崩塌）
- [x] **VR-16**: 基于前 7 次（VR-09~VR-15）最佳方向的组合实验（回到 baseline lr=5e-5 / dropout=0.3，并保留最佳正则 `weight_decay=0.001`）→ no_miss_val_acc=0.872, no_miss_val_spe=0.760, val_AUC=0.969 → discard（较当前 proxy 最优 `f6f4ed8` 的 0.904 下降 0.032，但较同配方前次 `VR-14` / `c059aa6` 的 0.830 回升 0.042、较 freeze=2 baseline `fc760cb` 的 0.883 仅差 0.011；说明 regularized freeze=2 rerun 能显著回弹，但仍不足以取代 freeze=3 baseline，下一步转入最优配置 multi-seed 验证）

### 阶段 10C：Multi-seed 验证（最优配置 x 3 seeds）

> 仅在 10A/10B 中产生 keep 后执行

- [ ] **VR-MS-01**: 最优配置 + seed=42
- [ ] **VR-MS-02**: 最优配置 + seed=123
- [ ] **VR-MS-03**: 最优配置 + seed=456

---


## 用户定向 rerun Campaign（Plan 1：Hierarchical Hybrid Fusion）❌ 已放弃

> 9/10 实验全部 discard，用户决定放弃 R1-10。

- [x] **R1-01**: baseline → no_miss_val_acc=0.638 → discard
- [x] **R1-02**: lr=5e-5 → no_miss_val_acc=0.702 → discard
- [x] **R1-03**: lr=7e-5 → no_miss_val_acc=0.670 → discard
- [x] **R1-04**: dropout=0.2 → no_miss_val_acc=0.723 → discard
- [x] **R1-05**: dropout=0.4 → no_miss_val_acc=0.745 → discard
- [x] **R1-06**: label_smoothing=0.05 → no_miss_val_acc=0.564 → discard
- [x] **R1-07**: lr=5e-5 + ls=0.05 → no_miss_val_acc=0.660 → discard
- [x] **R1-08**: fusion_hidden_dim=128 → no_miss_val_acc=0.543 → discard
- [x] **R1-09**: fusion_hidden_dim=384 → no_miss_val_acc=0.691 → discard
- [x] ~~**R1-10**~~: 已放弃（用户决定，2026-04-05）

---

## 阶段 5：AttentionPooling 增强 ❌ 已放弃

> AP-FF 前 3 次全部 discard（最高 0.574），用户决定放弃该方向。

- [x] AP-FF-01 → 0.574 discard | AP-FF-02 → 0.532 discard | AP-FF-03 → 0.553 discard
- 其余实验已取消

---

## 阶段 6：Uncertainty-Weighted Decision Fusion (UWDF) ✅ 已完成（8/8 proxy，全 discard）

> **论文创新点**：不确定性加权决策融合。
> 每个视角不仅输出分类 logits，还输出预测不确定性（log σ²）。
> 不确定性高的视角在融合时权重自动降低。
> 训练损失 = CE(fused_logits) + heteroscedastic auxiliary loss (Kendall & Gal 2017)。
> 基线配置：`share_backbone=false, aug=true, lr=1e-4, label_smoothing=0.0, dropout=0.3, fusion_hidden_dim=256`

### 阶段 6A：UWDF 超参搜索（8 次 proxy）

- [x] **UWDF-01**: baseline（fusion_type=decision, lr=1e-4, dropout=0.3, ls=0.0）→ no_miss_val_acc=0.574, no_miss_val_spe=0.200, val_AUC=0.911 → discard
- [x] **UWDF-02**: lr=5e-5 → no_miss_val_acc=0.723, no_miss_val_spe=0.480, val_AUC=0.922 → discard
- [x] **UWDF-03**: lr=7e-5 → no_miss_val_acc=0.574, no_miss_val_spe=0.200, val_AUC=0.937 → discard
- [x] **UWDF-04**: dropout=0.2 → no_miss_val_acc=0.734, no_miss_val_spe=0.500, val_AUC=0.923 → discard
- [x] **UWDF-05**: dropout=0.4 → no_miss_val_acc=0.596, no_miss_val_spe=0.240, val_AUC=0.913 → discard
- [x] **UWDF-06**: label_smoothing=0.05 → no_miss_val_acc=0.702, no_miss_val_spe=0.440, val_AUC=0.927 → discard
- [x] **UWDF-07**: lr=5e-5 + label_smoothing=0.05 → no_miss_val_acc=0.638, no_miss_val_spe=0.320, val_AUC=0.920 → discard
- [x] **UWDF-08**: dropout=0.2 + lr=5e-5（基于前 7 次最佳方向组合，保持 label_smoothing=0.0）→ no_miss_val_acc=0.660, no_miss_val_spe=0.360, val_AUC=0.926 → discard

---


## 阶段 7：Dual-Granularity Adaptive Fusion (DGAF) ✅ 已完成（跳过末尾 2 次）

> **论文创新点**：双粒度自适应融合。
> 同时在特征级和决策级做融合，用 View-Aware Gate（视角置信度 + 分歧度 + 双分支 logits）
> 为每个样本动态选择最优融合路径。
> 6/8 次 proxy 全 discard，跳过 DGAF-07/08 转入 PFDF。
> 基线配置：share_backbone=false, aug=true, lr=1e-4, label_smoothing=0.0, dropout=0.3, fusion_hidden_dim=256

### 阶段 7A：DGAF 超参搜索（7/8 次 proxy，DGAF-08 跳过）

- [x] **DGAF-01**: baseline（fusion_type=decision, lr=1e-4, dropout=0.3, ls=0.0）→ no_miss_val_acc=0.755, no_miss_val_spe=0.540, val_AUC=0.930 → discard
- [x] **DGAF-02**: lr=5e-5 → no_miss_val_acc=0.723, no_miss_val_spe=0.480, val_AUC=0.946 → discard
- [x] **DGAF-03**: lr=7e-5 → no_miss_val_acc=0.723, no_miss_val_spe=0.480, val_AUC=0.937 → discard
- [x] **DGAF-04**: dropout=0.2 → no_miss_val_acc=0.777, no_miss_val_spe=0.580, val_AUC=0.933 → discard
- [x] **DGAF-05**: dropout=0.4 → no_miss_val_acc=0.777, no_miss_val_spe=0.580, val_AUC=0.933 → discard
- [x] **DGAF-06**: label_smoothing=0.05 → no_miss_val_acc=0.681, no_miss_val_spe=0.400, val_AUC=0.932 → discard
- [x] **DGAF-07**: lr=5e-5 + label_smoothing=0.05 → no_miss_val_acc=0.702, no_miss_val_spe=0.440, val_AUC=0.946 → discard
- [x] ~~**DGAF-08**~~: 已跳过（转入 PFDF，2026-04-06）

---

## 阶段 8：Progressive Feature Distillation Fusion (PFDF) ✅ 已完成（8/8 proxy，全 discard）

> **论文创新点**：渐进式特征蒸馏融合。
> 通过两阶段两两交叉注意力渐进融合三个视角特征：
> Stage 1: Fuse(axial, coronal) → 512D；Stage 2: Fuse(result, sagittal) → 512D。
> 分类器只需处理 512 维（而非暴力拼接的 1536 维），降低过拟合风险。
> 基线配置：share_backbone=false, aug=true, lr=1e-4, label_smoothing=0.0, dropout=0.3, fusion_hidden_dim=256

### 阶段 8A：PFDF 超参搜索（8 次 proxy）

- [x] **PFDF-01**: baseline（fusion_type=decision, lr=1e-4, dropout=0.3, ls=0.0）→ no_miss_val_acc=0.702, no_miss_val_spe=0.440, val_AUC=0.920 → discard
- [x] **PFDF-02**: lr=5e-5 → no_miss_val_acc=0.787, no_miss_val_spe=0.600, val_AUC=0.920 → discard
- [x] **PFDF-03**: lr=7e-5 → no_miss_val_acc=0.670, no_miss_val_spe=0.380, val_AUC=0.893 → discard
- [x] **PFDF-04**: dropout=0.2 → no_miss_val_acc=0.543, no_miss_val_spe=0.140, val_AUC=0.920 → discard
- [x] **PFDF-05**: dropout=0.4 → no_miss_val_acc=0.500, no_miss_val_spe=0.060, val_AUC=0.903 → discard
- [x] **PFDF-06**: label_smoothing=0.05 → no_miss_val_acc=0.745, no_miss_val_spe=0.520, val_AUC=0.934 → discard
- [x] **PFDF-07**: lr=5e-5 + label_smoothing=0.05 → no_miss_val_acc=0.766, no_miss_val_spe=0.560, val_AUC=0.924 → discard
- [x] **PFDF-08**: 基于前 7 次最佳方向的组合实验（沿用最佳 lr=5e-5，并恢复 baseline label_smoothing=0.0）→ no_miss_val_acc=0.755, no_miss_val_spe=0.540, val_AUC=0.916 → discard

---

## 阶段 9：Cross-View Feature Interaction Fusion (CVFI) ✅ 已完成（8/8 proxy，全 discard）

> **论文创新点**：跨视角特征交互融合。
> 在标准特征拼接（一阶）的基础上，增加视角两两之间的**逐元素乘积交互项**（二阶），
> 通过投影层压缩到 128 维。交互项显式建模跨视角特征通道的共激活模式，
> 让分类器能利用视角间的一致性/不一致性信号。
> concat([v1, v2, v3, proj(v1⊙v2), proj(v2⊙v3), proj(v1⊙v3)]) = 1920D → MLP → 2
> 基线配置：share_backbone=false, aug=true, lr=1e-4, label_smoothing=0.0, dropout=0.3, fusion_hidden_dim=256

### 阶段 9A：CVFI 超参搜索（8 次 proxy）

- [x] **CVFI-01**: baseline（fusion_type=decision, lr=1e-4, dropout=0.3, ls=0.0）→ no_miss_val_acc=0.585, no_miss_val_spe=0.220, val_AUC=0.915 → discard
- [x] **CVFI-02**: lr=5e-5 → no_miss_val_acc=0.553, no_miss_val_spe=0.160, val_AUC=0.902 → discard
- [x] **CVFI-03**: lr=7e-5 → no_miss_val_acc=0.628, no_miss_val_spe=0.300, val_AUC=0.911 → discard
- [x] **CVFI-04**: dropout=0.2 → no_miss_val_acc=0.734, no_miss_val_spe=0.500, val_AUC=0.933 → discard
- [x] **CVFI-05**: dropout=0.4 → no_miss_val_acc=0.681, no_miss_val_spe=0.400, val_AUC=0.923 → discard
- [x] **CVFI-06**: label_smoothing=0.05 → no_miss_val_acc=0.553, no_miss_val_spe=0.160, val_AUC=0.896 → discard
- [x] **CVFI-07**: lr=5e-5 + label_smoothing=0.05 → no_miss_val_acc=0.489, no_miss_val_spe=0.040, val_AUC=0.904 → discard
- [x] **CVFI-08**: 基于前 7 次最佳方向的组合实验（沿用最佳 dropout=0.2，并恢复 baseline lr=1e-4 / label_smoothing=0.0）→ no_miss_val_acc=0.723, no_miss_val_spe=0.480, val_AUC=0.933 → discard

---

## 旧阶段（已完成）

### 阶段 1：Feature Fusion 超参优化（已完结）

- 暂无高优先级待办

### 阶段 2：Decision Fusion 对比（已完结）

- 暂无高优先级待办

### 阶段 3：论文就绪验证 🔴 当前最高优先级

- [x] **多 seed 验证**（seed=42, 123, 456 已完成）
  - [x] seed=42（`053cb3a`，当前全局最优基线）→ no_miss_val_acc=0.798, no_miss_val_spe=0.620, val_AUC=0.960 → keep
  - [x] seed=123（`b12f03d`，VRG 当前最优配置 proxy 验证）→ no_miss_val_acc=0.787, no_miss_val_spe=0.600, val_AUC=0.952 → discard
  - [x] seed=456（`f3c0157`，VRG 当前最优配置 proxy 验证）→ no_miss_val_acc=0.521, no_miss_val_spe=0.100, val_AUC=0.923 → discard
- [x] **Bootstrap 置信区间**（1000 次重采样；`7f4d793`，VRG 最优配置 seed=42 rerun）
  - [x] 验证集重采样并重新估计零漏诊阈值：threshold 中位数=0.332，95% CI [0.332, 0.399]
  - [x] 零漏诊验证指标：no_miss_val_acc=0.830（bootstrap 95% CI [0.766, 0.957]），no_miss_val_spe=0.680（95% CI [0.574, 0.927]）
  - [x] 固定阈值测试集指标：acc=0.737（95% CI [0.526, 0.895]），spe=0.900（95% CI [0.667, 1.000]），sen=0.556（95% CI [0.200, 0.875]）
  - [x] proxy rerun 摘要：val_AUC=0.970，no_miss_val_acc=0.830 → keep ✅ 新当前最优
- [ ] **三种融合方式 formal 对比**
  - [ ] Feature Fusion formal
    - [x] proxy gate（`d32d660`，non-shared, aug=true, lr=1e-4, d=0.3, ls=0.05）→ no_miss_val_acc=0.798, no_miss_val_spe=0.620, val_AUC=0.932 → discard（未超过当前最优 0.830，暂不升 formal）
  - [ ] Decision Fusion formal（当前最优 VRG + gradient clip）
    - [x] proxy gate / sanity rerun（`af28838`，non-shared, aug=true, lr=5e-5, d=0.3, ls=0.0）→ no_miss_val_acc=0.755, no_miss_val_spe=0.540, val_AUC=0.949 → discard（未复现当前最优 `7f4d793` 的 0.830，formal 仍待基于 keep 提交推进）
    - [x] proxy stability rerun #2（`03bdce9`，non-shared, aug=true, lr=5e-5, d=0.3, ls=0.0）→ no_miss_val_acc=0.638, no_miss_val_spe=0.320, val_AUC=0.937 → discard（再次低于当前最优 `7f4d793` 的 0.830，formal 暂不推进）
    - [x] proxy stability rerun #3 / raw confidence logits（`ef6008c`，移除 confidence sigmoid 后直接 softmax；non-shared, aug=true, lr=5e-5, d=0.3, ls=0.0）→ no_miss_val_acc=0.734, no_miss_val_spe=0.500, val_AUC=0.914 → discard（较 `03bdce9` 有所回升，但仍低于当前最优 `7f4d793` 的 0.830，formal 暂不推进）
    - [x] proxy stability rerun #4 / restore canonical sigmoid gate（`20a402b`，恢复标准 confidence sigmoid + softmax；non-shared, aug=true, lr=5e-5, d=0.3, ls=0.0）→ no_miss_val_acc=0.585, no_miss_val_spe=0.220, val_AUC=0.941 → discard（与 `7f4d793` 同配方仍未复现，且比 rerun #3 更差，formal 继续搁置）
    - [x] proxy stabilization probe / cosine scheduler（`49e93fb`，non-shared, aug=true, lr=5e-5, d=0.3, ls=0.0, scheduler=cosine）→ no_miss_val_acc=0.628, no_miss_val_spe=0.300, val_AUC=0.938 → discard（较 canonical rerun #4 略有回升，但仍显著低于当前最优 `7f4d793` 的 0.830，formal 继续搁置）
    - [x] proxy stabilization probe / gradient clip（`814f491`，non-shared, aug=true, lr=5e-5, d=0.3, ls=0.0, gradient_clip_norm=1.0）→ no_miss_val_acc=0.872, no_miss_val_spe=0.760, val_AUC=0.966 → keep ✅ 新当前最优（已超过 `7f4d793` 的 0.830，下一步推进 Decision Fusion formal）
    - [x] proxy stability rerun / seed=123 + gradient clip（`1272bc6`，non-shared, aug=true, lr=5e-5, d=0.3, ls=0.0, gradient_clip_norm=1.0, seed=123）→ no_miss_val_acc=0.830, no_miss_val_spe=0.680, val_AUC=0.960 → discard（未超过当前最优 `814f491` 的 0.872，但与 `7f4d793` 的 0.830 持平，说明 clipped VRG 在 alternate seed 上仍具竞争力）
    - [x] proxy formal-preflight / seed=42 clipped VRG rerun + formal config sync（`50be764`，non-shared, aug=true, lr=5e-5, d=0.3, ls=0.0, gradient_clip_norm=1.0, seed=42）→ no_miss_val_acc=0.862, no_miss_val_spe=0.740, val_AUC=0.969 → discard（val_AUC 高于当前最优 `814f491` 的 0.966，但零漏诊 val_acc 仍低 0.011；formal 继续以 `814f491` 为基准推进）
    - [x] proxy formal-geometry smoke（`601351e`，non-shared, aug=true, lr=5e-5, d=0.3, ls=0.0, gradient_clip_norm=1.0, image_size=256, num_slices_per_view=32）→ no_miss_val_acc=0.691, no_miss_val_spe=0.420, val_AUC=0.930 → discard（4 epoch formal geometry proxy 明显退化，较当前最优 `814f491` 低 0.181；已触发连续 3 次 discard，formal 前需先重审稳定化方案）
    - [x] proxy formal-geometry stabilization / edge trimming（`18a19d7`，non-shared, aug=true, lr=5e-5, d=0.3, ls=0.0, gradient_clip_norm=1.0, image_size=256, num_slices_per_view=32, trim_edge_slices=2）→ no_miss_val_acc=0.777, no_miss_val_spe=0.580, val_AUC=0.941 → discard（较 `601351e` 回升 0.085，说明 32-slice formal geometry 的退化部分来自边缘切片稀释；但仍低于当前最优 `814f491` 的 0.872，formal 继续搁置）
    - [x] proxy formal-geometry stabilization / edge trimming + cosine scheduler（`e4b2175`，non-shared, aug=true, lr=5e-5, d=0.3, ls=0.0, gradient_clip_norm=1.0, image_size=256, num_slices_per_view=32, trim_edge_slices=2, scheduler=cosine）→ no_miss_val_acc=0.798, no_miss_val_spe=0.620, val_AUC=0.955 → discard（较 `18a19d7` 再回升 0.021，并恢复到 `7f4d793` / `d32d660` 同级水平；但仍低于当前最优 `814f491` 的 0.872，说明 formal geometry + cosine 仍未补平 256x32 的性能落差）
    - [x] formal run（`d63b49a`，沿用 `814f491` 配方 + native 192x16 geometry，15 epochs；non-shared, aug=true, lr=5e-5, d=0.3, ls=0.0, gradient_clip_norm=1.0）→ no_miss_val_acc=**0.915**, no_miss_val_spe=0.840, val_AUC=0.968 → keep ✅ 新全局最优
  - [ ] Attention Fusion formal
    - [x] proxy gate（`1d0bb82`，shared, aug=false, lr=1e-4, d=0.3, heads=4, layers=2）→ no_miss_val_acc=0.723, no_miss_val_spe=0.480, val_AUC=0.919 → discard（低于当前最优 0.830，暂不升 formal）

### 阶段 4：创新融合架构探索 ✅ 已完成

- 阶段 4A：View Reliability Gating（10 次 proxy，已完结）
- 阶段 4B：Asymmetric Safety-Biased Fusion（5 次 proxy，已完结）

---

## 已完成 / 已废弃实验（历史记录）

### Feature Fusion — 共享 backbone 阶段（Round 1, 已完结）

- [x] 共享 backbone baseline (lr=1e-4, d=0.3) → val_AUC=0.927 → discard
- [x] 共享 backbone lr=5e-5 → val_AUC=0.946 → keep（AUC 体系下最优）
- [x] 共享 backbone lr=3e-4 → discard
- [x] 共享 backbone dropout=0.2 → discard
- [x] 共享 backbone dropout=0.4 → discard
- [x] 共享 backbone CosineAnnealingLR → discard
- [x] 共享 backbone formal (lr=5e-5) → val_AUC=0.940, test_AUC=0.822

### Feature Fusion — 非共享 backbone 阶段（Round 2, 已完结）

- [x] 非共享 baseline (lr=1e-4, d=0.3) → no_miss_val_acc=0.755 → keep
- [x] 非共享 lr=5e-5 → no_miss_val_acc=0.660 → discard
- [x] 非共享 dropout=0.4 → no_miss_val_acc=0.691 → discard
- [x] 非共享 lr=7e-5 → no_miss_val_acc=0.691 → discard
- [x] 非共享 + augmentation=true → no_miss_val_acc=0.777 → keep
- [x] 非共享 + augmentation=true formal 确认 → no_miss_val_acc=0.755, val_AUC=0.946 → keep
- [x] 非共享 + augmentation=true + gradient_clip_norm=1.0 → discard
- [x] **非共享 + augmentation=true + label_smoothing=0.05 → no_miss_val_acc=0.777, val_AUC=0.948 → keep ✅ Feature Fusion 最优**
- [x] 非共享 + aug + lr=5e-5 → discard
- [x] 非共享 + aug + ls=0.05 + wd=5e-4 → discard
- [x] 非共享 + aug + ls=0.05 + d=0.4 → discard
- [x] 非共享 + aug + ls=0.1 → discard
- [x] 非共享 + aug + ls=0.05 + clip=1.0 → discard
- [x] 非共享 + aug + ls=0.05 + clip=0.5 → discard
- [x] 非共享 + aug + ls=0.05 + early_stopping=5 → discard

### Decision Fusion 阶段（Round 3, 已完结）

- [x] Decision Fusion proxy baseline（aug, ls=0.05）→ discard
- [x] Decision Fusion proxy lr=5e-5 → discard
- [x] Decision Fusion proxy lr=7e-5 → discard
- [x] **Decision Fusion proxy 去掉 ls（lr=1e-4）→ no_miss_val_acc=0.787 → keep（曾为全局最优）**
- [x] Decision Fusion proxy lr=3e-4 → discard
- [x] Decision Fusion proxy dropout=0.2 → discard
- [x] Decision Fusion proxy dropout=0.4 → discard
- [x] Decision Fusion proxy clip=0.5 → discard
- [x] Decision Fusion proxy hidden=128 → discard
- [x] Decision Fusion proxy 6 epoch → discard
- [x] Decision Fusion proxy hidden=384 → discard
- [x] Decision Fusion proxy cosine → discard
- [x] Decision Fusion proxy 概率投票 → discard

### 阶段 4A：View Reliability Gating 阶段（Round 4, 已完结）

- [x] VRG-01 ~ VRG-10：均 discard，最佳 no_miss_val_acc=0.755（VRG-02, VRG-03）

### 阶段 4B：Asymmetric Safety-Biased Fusion 阶段（Round 5, 已完结）

- [x] ASF-01: temperature=0.5 → no_miss_val_acc=0.681 → discard
- [x] ASF-02: temperature=1.0 → no_miss_val_acc=0.532 → discard
- [x] ASF-03: temperature=0.3 → no_miss_val_acc=0.617 → discard
- [x] ASF-04: temperature=2.0 → no_miss_val_acc=0.649 → discard
- [x] ASF-05: temperature=0.5 + lr=5e-5 → no_miss_val_acc=0.660 → discard

---

## 维护规则

1. **Agent 每次启动时**：先读本文件，了解当前最优纪录和待办优先级
2. **每完成一个实验后**：
   - 如果 keep：更新"当前最优纪录"表格
   - 将实验从待办移到"已完成"，标注结果
   - 如果结果带来新的实验思路，添加到对应优先级
3. **连续 3 个 discard 后**：重新审视待办清单，考虑换方向
4. **人类编辑后**：Agent 下次启动时以文件内容为准

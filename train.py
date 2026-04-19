"""
train.py — 模型训练主程序
============================
这是整个项目的"主入口"，运行这个文件就开始训练模型。

完整的训练流程如下：
    1. 读取配置文件（configs/default.yaml）
    2. 加载数据并划分为"训练集"和"验证集"
    3. 创建模型
    4. 开始训练循环（多个 epoch）：
        - 每个 epoch 遍历一遍所有训练数据，更新模型参数
        - 用验证集评估当前模型表现
        - 保存表现最好的模型
    5. 训练结束后，用最好的模型在验证集/测试集上做最终评估
    6. 保存训练历史和评估结果

用法：
    python train.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse         # 命令行参数解析
import time
from pathlib import Path

import numpy as np
import pandas as pd     # Pandas：用来读取和操作 CSV 表格数据
import torch
from sklearn.model_selection import train_test_split  # 数据集切分工具
from torch.utils.data import DataLoader               # PyTorch 的数据加载器
from tqdm import tqdm                                  # 进度条显示

# 从我们自己的代码中导入需要的类和函数
from src.dataset import VIEW_COLUMNS, PatientCTDataset, canonicalize_view_columns
from src.model import MultiViewCTClassifier, MultiViewDecisionFusionClassifier, MultiViewAttentionClassifier
from src.utils import (
    choose_device,          # 自动选择运行设备（GPU 或 CPU）
    compute_class_weights,  # 计算类别权重（处理数据不平衡问题）
    compute_metrics,        # 计算评估指标（准确率、AUC 等）
    load_config,            # 加载 YAML 配置文件
    save_json,              # 保存 JSON 文件
    set_seed,               # 设置随机种子（保证实验可重复）
)




# ==================== 数据增强 ====================


def apply_augmentation(images: torch.Tensor) -> torch.Tensor:
    """对一个 batch 的 CT 图像做随机增强（仅训练时调用）。

    Args:
        images: (B, V, S, H, W) 张量，已在 device 上。

    Returns:
        增强后的图像张量，形状不变。
    """
    import torchvision.transforms.functional as TF

    # 随机水平翻转 (p=0.5)
    if torch.rand(1).item() < 0.5:
        images = images.flip(-1)

    # 随机旋转 ±10 度
    angle = (torch.rand(1).item() - 0.5) * 20  # [-10, 10]
    if abs(angle) > 0.5:
        B, V, S, H, W = images.shape
        flat = images.reshape(B * V * S, 1, H, W)
        flat = TF.rotate(flat, angle)
        images = flat.reshape(B, V, S, H, W)

    return images


# ==================== 命令行参数 ====================


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    目前只有一个参数：
        --config: 配置文件路径，默认是 configs/default.yaml

    运行示例：
        python train.py                              # 使用默认配置
        python train.py --config configs/custom.yaml  # 使用自定义配置
    """
    parser = argparse.ArgumentParser(description="Train a beginner multi-view CT classifier.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    return parser.parse_args()


# ==================== 数据加载与划分 ====================


def load_and_split_dataframe(
    csv_path: str,
    seed: int,
    val_ratio: float,
    test_ratio: float = 0.0,
    use_existing_split: bool = True,
):
    """Load a CSV file and split it into train/val/test dataframes."""
    df = canonicalize_view_columns(pd.read_csv(csv_path))

    required_columns = {
        "patient_id",
        "label",
        *VIEW_COLUMNS,
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {sorted(missing)}")

    df["label"] = df["label"].astype(int)

    if use_existing_split and "split" in df.columns:
        train_df = df[df["split"] == "train"].copy()
        val_df = df[df["split"] == "val"].copy()
        test_df = df[df["split"] == "test"].copy()
        if len(train_df) == 0:
            raise ValueError("When using a split column, you need at least train rows.")
        if len(val_df) == 0 and len(test_df) == 0:
            raise ValueError("When using a split column, provide at least val or test rows.")
        return train_df, val_df, test_df

    if val_ratio < 0 or test_ratio < 0:
        raise ValueError("val_ratio and test_ratio must be >= 0.")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1.")

    if test_ratio > 0:
        train_df, remaining_df = train_test_split(
            df,
            test_size=val_ratio + test_ratio,
            stratify=df["label"],
            random_state=seed,
        )
        relative_test_ratio = test_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(
            remaining_df,
            test_size=relative_test_ratio,
            stratify=remaining_df["label"],
            random_state=seed,
        )
        return (
            train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True),
        )

    train_df, val_df = train_test_split(
        df,
        test_size=val_ratio,
        stratify=df["label"],
        random_state=seed,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), pd.DataFrame()


def build_dataloaders(config: dict):
    """Build dataloaders from config."""
    data_cfg = config["data"]

    train_df, val_df, test_df = load_and_split_dataframe(
        csv_path=data_cfg["csv_path"],
        seed=config["seed"],
        val_ratio=data_cfg["val_ratio"],
        test_ratio=data_cfg.get("test_ratio", 0.0),
        use_existing_split=data_cfg.get("use_existing_split", True),
    )

    total_samples = len(train_df) + len(val_df) + len(test_df)
    if total_samples > 0:
        print(
            "Dataset split: "
            f"train={len(train_df)} ({len(train_df) / total_samples:.1%}), "
            f"val={len(val_df)} ({len(val_df) / total_samples:.1%}), "
            f"test={len(test_df)} ({len(test_df) / total_samples:.1%})"
        )

    common_kwargs = {
        "base_dir": data_cfg["base_dir"],
        "image_size": data_cfg["image_size"],
        "num_slices_per_view": data_cfg["num_slices_per_view"],
        "trim_edge_slices": data_cfg.get("trim_edge_slices", 0),
        "hu_min": data_cfg["hu_min"],
        "hu_max": data_cfg["hu_max"],
    }

    train_dataset = PatientCTDataset(train_df, **common_kwargs)
    val_dataset = PatientCTDataset(val_df, **common_kwargs) if len(val_df) > 0 else None
    test_dataset = PatientCTDataset(test_df, **common_kwargs) if len(test_df) > 0 else None

    batch_size = data_cfg["batch_size"]
    num_workers = data_cfg["num_workers"]
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    return train_loader, val_loader, test_loader, train_df


def build_model(config: dict):
    """
    根据配置文件创建模型。

    支持两种融合方式：
        - "feature" （特征融合）: 拼接 3 个视角特征后一起分类（MultiViewCTClassifier）
        - "decision"（决策融合）: 每个视角单独分类后加权投票（MultiViewDecisionFusionClassifier）

    参数：
        config: 配置字典

    返回：
        创建好的模型对象
    """
    model_cfg = config["model"]
    common_kwargs = {
        "share_backbone": model_cfg["share_backbone"],       # 是否共享 backbone
        "use_pretrained": model_cfg["use_pretrained"],       # 是否使用预训练权重
        "freeze_layers": int(model_cfg.get("freeze_layers", 3)),  # 显式冻结层配置
        "fusion_hidden_dim": model_cfg["fusion_hidden_dim"], # 分类器隐藏层维度
        "dropout": model_cfg["dropout"],                     # Dropout 比率
        "use_attention_pooling": model_cfg.get("use_attention_pooling", False),  # 注意力池化
        "backbone": model_cfg.get("backbone", "resnet18"),   # backbone 类型
        "minimal_fusion_baseline": model_cfg.get("minimal_fusion_baseline", False),
    }
    fusion_type = model_cfg.get("fusion_type", "feature")

    if fusion_type == "feature":
        return MultiViewCTClassifier(**common_kwargs)
    if fusion_type == "decision":
        return MultiViewDecisionFusionClassifier(**common_kwargs)
    if fusion_type == "attention":
        backbone = model_cfg.get("backbone", "resnet18")
        if backbone != "resnet18":
            raise ValueError(
                "Attention fusion currently supports model.backbone='resnet18' only."
            )
        # 注意力融合模式内部固定使用 AttentionPooling，只透传实际支持的参数
        attention_kwargs = {
            "share_backbone": model_cfg["share_backbone"],
            "use_pretrained": model_cfg["use_pretrained"],
            "freeze_layers": int(model_cfg.get("freeze_layers", 3)),
            "fusion_hidden_dim": model_cfg["fusion_hidden_dim"],
            "dropout": model_cfg["dropout"],
            "cross_view_heads": model_cfg.get("cross_view_heads", 8),
            "cross_view_layers": model_cfg.get("cross_view_layers", 2),
        }
        return MultiViewAttentionClassifier(**attention_kwargs)

    raise ValueError(
        "Unsupported model.fusion_type. Expected one of: 'feature', 'decision', 'attention'."
    )


def build_scheduler(config: dict, optimizer):
    """Build an optional learning-rate scheduler from config."""
    train_cfg = config["train"]
    scheduler_name = train_cfg.get("scheduler")
    if scheduler_name in (None, "", "none"):
        return None

    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(train_cfg.get("scheduler_t_max", train_cfg["epochs"])),
            eta_min=float(train_cfg.get("scheduler_eta_min", 0.0)),
        )

    raise ValueError(
        "Unsupported train.scheduler. Expected one of: none, cosine."
    )


def get_current_lr(optimizer) -> float:
    """Get current learning rate from optimizer."""
    return float(optimizer.param_groups[0]["lr"])


# ==================== 训练逻辑 ====================


def train_one_epoch(model, loader, criterion, optimizer, device, gradient_clip_norm=None, augmentation=False):
    """
    训练一个 epoch（遍历一遍所有训练数据）。

    一个 epoch 的流程：
        对每个 batch（小批量数据）：
            1. 把数据送进模型，得到预测结果（前向传播）
            2. 计算损失（预测结果 vs 真实标签的差距）
            3. 反向传播：计算每个参数对损失的贡献（梯度）
            4. 更新参数：根据梯度调整模型参数，让损失变小

    参数：
        model:     模型
        loader:    训练数据加载器
        criterion: 损失函数（CrossEntropyLoss）
        optimizer: 优化器（Adam）
        device:    运算设备（GPU 或 CPU）

    返回：
        平均损失值（越小表示模型学得越好）
    """
    model.train()  # 设为训练模式（启用 Dropout 等训练专用行为）
    running_loss = 0.0
    progress = tqdm(loader, desc="train", leave=False)  # 显示进度条

    for batch in progress:
        # ---------- 准备数据 ----------
        images = batch["images"].to(device)  # 把图像数据传到 GPU（如果有的话）
        labels = batch["label"].to(device)   # 标签也传到 GPU

        # ---------- 数据增强（仅训练时） ----------
        if augmentation:
            images = apply_augmentation(images)

        # ---------- 前向传播 ----------
        optimizer.zero_grad()     # 清零上一步的梯度（不清零会累加）
        logits = model(images)    # 模型预测，得到 (B, 2) 的分数
        loss = criterion(logits, labels)  # 计算损失（交叉熵损失）

        # ---------- 不确定性校准辅助损失 (UWDF) ----------
        # 如果模型是 UWDF（暴露了 _view_logits 和 _log_vars），
        # 则计算 Kendall & Gal 2017 的 heteroscedastic uncertainty loss：
        #   L_aux = (1/V) * Σ_v [exp(-s_v) * CE_v + s_v]
        # 其中 s_v = log_var_v（视角 v 的预测不确定性）
        _vl = getattr(model, '_view_logits', None)
        _lv = getattr(model, '_log_vars', None)
        if _vl is not None and _lv is not None:
            import torch.nn.functional as _F
            _num_views = _vl.size(1)
            _aux = torch.zeros(1, device=labels.device)
            for _v in range(_num_views):
                _vce = _F.cross_entropy(_vl[:, _v], labels, reduction='none')  # (B,)
                _sv = _lv[:, _v, 0]  # (B,)
                _aux = _aux + (torch.exp(-_sv) * _vce + _sv).mean()
            loss = loss + _aux / _num_views

        # ---------- 反向传播 + 参数更新 ----------
        loss.backward()           # 反向传播：自动计算每个参数的梯度

        # ---------- 梯度裁剪 ----------
        if gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)

        optimizer.step()          # 根据梯度更新模型参数

        # ---------- 记录损失 ----------
        running_loss += loss.item() * labels.size(0)  # 累加总损失
        progress.set_postfix(loss=f"{loss.item():.4f}")  # 在进度条上显示当前 batch 的损失

    # 返回平均损失 = 总损失 / 样本总数
    return running_loss / len(loader.dataset)


# ==================== 评估逻辑 ====================


@torch.no_grad()  # 评估时不需要计算梯度，关闭梯度可以节省显存和加速
def evaluate(model, loader, criterion, device):
    """
    在验证集或测试集上评估模型。

    参数：
        model:     模型
        loader:    验证/测试数据加载器
        criterion: 损失函数
        device:    运算设备

    返回：
        average_loss: 平均损失
        metrics:      评估指标字典（包含 accuracy, auc, f1, sensitivity, specificity）
    """
    model.eval()  # 设为评估模式（关闭 Dropout，对所有数据做完整预测）
    running_loss = 0.0
    all_labels = []  # 存放所有真实标签
    all_preds = []   # 存放所有预测标签
    all_probs = []   # 存放所有预测概率

    for batch in tqdm(loader, desc="eval", leave=False):
        images = batch["images"].to(device)
        labels = batch["label"].to(device)

        logits = model(images)                        # 模型预测
        loss = criterion(logits, labels)              # 计算损失
        probs = torch.softmax(logits, dim=1)          # 把分数转换为概率（和为 1）
        preds = torch.argmax(probs, dim=1)            # 取概率最大的类别作为预测结果

        running_loss += loss.item() * labels.size(0)
        # 收集所有样本的真实标签、预测标签、预测概率
        all_labels.extend(labels.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())

    average_loss = running_loss / len(loader.dataset)
    # 计算评估指标（准确率、AUC、F1 等）
    metrics = compute_metrics(all_labels, all_preds, all_probs)
    return average_loss, metrics


@torch.no_grad()
def collect_attention_output(model, loader, device, view_index: int):
    """Collect attention outputs for a given view."""
    if loader is None or not hasattr(model, "forward_with_attention"):
        return None

    model.eval()
    all_view_weights = []
    example_rows = []

    for batch in tqdm(loader, desc=f"attention_v{view_index + 1}", leave=False):
        images = batch["images"].to(device)
        labels = batch["label"]
        patient_ids = batch["patient_id"]

        logits, attention_info = model.forward_with_attention(images)
        abnormal_probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        view_weights = attention_info["slice_attention_weights"][:, view_index, :].cpu().numpy()
        all_view_weights.append(view_weights)

        remaining_examples = 5 - len(example_rows)
        if remaining_examples > 0:
            for patient_id, label, abnormal_prob, sample_weights in zip(
                patient_ids[:remaining_examples],
                labels[:remaining_examples].tolist(),
                abnormal_probs[:remaining_examples].tolist(),
                view_weights[:remaining_examples].tolist(),
            ):
                example_rows.append(
                    {
                        "patient_id": patient_id,
                        "label": int(label),
                        "abnormal_prob": float(abnormal_prob),
                        "slice_weights": [float(weight) for weight in sample_weights],
                    }
                )

    if not all_view_weights:
        return None

    stacked_weights = np.concatenate(all_view_weights, axis=0)
    mean_slice_weights = stacked_weights.mean(axis=0)
    return {
        "view_index": int(view_index + 1),
        "view_name": VIEW_COLUMNS[view_index].removesuffix("_dir"),
        "num_samples": int(stacked_weights.shape[0]),
        "top_slice_index": int(np.argmax(mean_slice_weights) + 1),
        "mean_slice_weights": [float(weight) for weight in mean_slice_weights.tolist()],
        "examples": example_rows,
    }


def score_for_model_selection(metrics: dict) -> tuple[float, float]:
    """
    计算一个分数，用于选择"最佳模型"。

    主指标是验证集 accuracy；当 val_acc 持平时，用 val_auc 作为 tie-break。
    返回的 tuple 会按 (accuracy, auc) 的字典序比较。
    """
    return float(metrics["accuracy"]), float(metrics.get("auc", float("-inf")))


# ==================== 主训练流程 ====================


def main() -> None:
    """
    主函数 — 完整的训练流程。

    流程：
        1. 解析命令行参数，读取配置文件
        2. 设置随机种子（保证实验可重复）
        3. 创建数据加载器（训练集 + 验证集 + 测试集）
        4. 创建模型并放到 GPU 上
        5. 设置损失函数和优化器
        6. 开始训练循环（多个 epoch）
        7. 保存最佳模型和训练历史
        8. 最终评估并保存结果
    """
    # ---------- 第 1 步：读取配置 ----------
    args = parse_args()
    config = load_config(args.config)
    run_start_time = time.perf_counter()
    set_seed(config["seed"])  # 固定随机种子，确保实验可重复

    # 创建输出目录（用来保存模型和结果）
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_name in ("best.pt", "history.json", "summary.json"):
        stale_path = output_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()

    # 选择设备（自动检测是否有 GPU）
    device = choose_device(config["train"]["device"])
    print(f"Using device: {device}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # ---------- 第 2 步：加载数据 ----------
    train_loader, val_loader, test_loader, train_df = build_dataloaders(config)

    # ---------- 第 3 步：创建模型 ----------
    model = build_model(config).to(device)  # .to(device) 把模型搬到 GPU 上

    # ---------- 第 4 步：设置损失函数 ----------
    # CrossEntropyLoss = 交叉熵损失，用于分类任务
    # 如果数据不平衡（正常样本 >> 异常样本），可以给少数类更高的权重
    label_smoothing = float(config["train"].get("label_smoothing", 0.0))
    if config["train"]["class_weight"]:
        class_weights = compute_class_weights(train_df["label"].to_numpy()).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    if label_smoothing > 0:
        print(f"Label smoothing enabled: {label_smoothing}")

    # ---------- 第 5 步：设置优化器 ----------
    # Adam 优化器：自动调整学习率的优化算法
    # lr = 学习率（控制每次更新的步长大小）
    # weight_decay = 权重衰减（L2 正则化，防止过拟合）
    optimizer = torch.optim.Adam(
        model.parameters(),                    # 告诉优化器要更新哪些参数
        lr=config["train"]["lr"],              # 学习率
        weight_decay=config["train"]["weight_decay"],  # 权重衰减
    )
    scheduler = build_scheduler(config, optimizer)


    # ---------- 训练策略配置 ----------
    gradient_clip_norm = config["train"].get("gradient_clip_norm")
    if gradient_clip_norm is not None:
        gradient_clip_norm = float(gradient_clip_norm)
        print(f"Gradient clipping enabled: max_norm={gradient_clip_norm}")
    augmentation_enabled = bool(config["train"].get("augmentation", False))
    if augmentation_enabled:
        print("Data augmentation enabled (random flip + rotation)")
    early_stopping_patience = config["train"].get("early_stopping_patience")
    if early_stopping_patience is not None:
        early_stopping_patience = int(early_stopping_patience)
        print(f"Early stopping enabled: patience={early_stopping_patience}")
    epochs_without_improvement = 0

    # ---------- 第 6 步：训练循环 ----------
    history = []         # 记录每个 epoch 的训练历史
    if val_loader is not None:
        best_score: tuple[float, float] | float = (float("-inf"), float("-inf"))
    else:
        best_score = float("-inf")
    best_path = output_dir / "best.pt"  # 最佳模型的保存路径

    for epoch in range(1, config["train"]["epochs"] + 1):
        print(f"\nEpoch {epoch}/{config['train']['epochs']}")
        current_lr = get_current_lr(optimizer)

        # 训练一轮
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            gradient_clip_norm=gradient_clip_norm,
            augmentation=augmentation_enabled,
        )
        record = {"epoch": epoch, "train_loss": train_loss, "lr": current_lr}

        # 在验证集上评估
        if val_loader is not None:
            val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
            record.update(
                {
                    "val_loss": val_loss,
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                    # ↑ 把 {"accuracy": 0.9} 变成 {"val_accuracy": 0.9}
                }
            )
            print(
                f"lr={current_lr:.6g} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"val_auc={val_metrics['auc']:.4f} | val_acc={val_metrics['accuracy']:.4f}"
            )
            current_score = score_for_model_selection(val_metrics)
        else:
            print(f"lr={current_lr:.6g} | train_loss={train_loss:.4f}")
            current_score = -train_loss  # 没有验证集时，用训练损失的负值作为分数（损失越小分数越高）

        history.append(record)

        # 如果当前模型比历史最佳还好，就保存它
        if current_score > best_score:
            best_score = current_score
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),  # 保存模型参数
                    "config": config,                         # 也保存配置文件，方便复现
                },
                best_path,
            )
            print(f"Saved best model to: {best_path}")
        else:
            epochs_without_improvement += 1

        # Early stopping
        if early_stopping_patience is not None and epochs_without_improvement >= early_stopping_patience:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {early_stopping_patience} epochs)")
            break

        if scheduler is not None:
            scheduler.step()

    # ---------- 第 7 步：保存训练历史 ----------
    save_json({"history": history}, output_dir / "history.json")

    # ---------- 第 8 步：最终评估 ----------
    # 加载训练过程中表现最好的模型（不一定是最后一个 epoch 的）
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    summary = {}

    # 在验证集上做最终评估
    if val_loader is not None:
        _, final_val_metrics = evaluate(model, val_loader, criterion, device)
        summary["best_val"] = final_val_metrics
        summary["model_selection"] = "best_val_accuracy_then_auc"
    else:
        summary["model_selection"] = "lowest_train_loss"

    # 在测试集上做最终评估（如果有测试集的话）
    if test_loader is not None:
        _, final_test_metrics = evaluate(model, test_loader, criterion, device)
        summary["test"] = final_test_metrics

    attention_output_view = config["model"].get("attention_output_view")
    if attention_output_view is not None:
        view_index = int(attention_output_view) - 1
        if not 0 <= view_index < len(VIEW_COLUMNS):
            raise ValueError(
                f"model.attention_output_view must be between 1 and {len(VIEW_COLUMNS)}."
            )
        attention_output = {
            "view_index": int(attention_output_view),
            "view_name": VIEW_COLUMNS[view_index].removesuffix("_dir"),
        }
        if val_loader is not None:
            attention_output["val"] = collect_attention_output(model, val_loader, device, view_index)
        if test_loader is not None:
            attention_output["test"] = collect_attention_output(model, test_loader, device, view_index)
        attention_output_path = output_dir / f"attention_view_{attention_output_view}.json"
        save_json(attention_output, attention_output_path)
        summary["attention_output_path"] = str(attention_output_path)

    total_seconds = time.perf_counter() - run_start_time
    peak_vram_mb = float("nan")
    if device.type == "cuda":
        peak_vram_mb = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
    summary["runtime"] = {
        "total_seconds": total_seconds,
        "peak_vram_mb": peak_vram_mb,
    }

    # 保存最终评估结果
    save_json(summary, output_dir / "summary.json")
    print("\nTraining finished.")
    print(f"Summary saved to: {output_dir / 'summary.json'}")
    if "best_val" in summary:
        print(f"best_val_auc={summary['best_val'].get('auc', float('nan')):.6f}")
        print(f"best_val_f1={summary['best_val'].get('f1', float('nan')):.6f}")
        print(f"best_val_accuracy={summary['best_val'].get('accuracy', float('nan')):.6f}")
    print(f"peak_vram_mb={peak_vram_mb:.1f}")
    print(f"total_seconds={total_seconds:.1f}")


# ==================== 入口点 ====================
# 当直接运行 "python train.py" 时，会执行 main() 函数
if __name__ == "__main__":
    main()

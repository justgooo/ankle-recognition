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


def load_and_split_dataframe(csv_path: str, seed: int, val_ratio: float):
    """
    从 CSV 文件中加载数据，并划分为训练集、验证集、测试集。

    CSV 文件必须包含以下列：
        - patient_id:    病人 ID
        - label:         标签（0=正常，1=异常）
        - axial_dir:     轴状面切片目录
        - coronal_dir:   冠状面切片目录
        - sagittal_dir:  矢状面切片目录

    划分方式有两种：
        1. 如果 CSV 已经有 "split" 列（值为 "train"/"val"/"test"），就按这个列划分
        2. 否则，自动按 val_ratio 比例随机划分（保持正负样本比例一致 = 分层抽样）

    参数：
        csv_path:  CSV 文件路径
        seed:      随机种子（保证每次划分结果相同）
        val_ratio: 验证集占总数据的比例（比如 0.2 表示 20%）

    返回：
        train_df, val_df, test_df 三个 DataFrame
    """
    df = canonicalize_view_columns(pd.read_csv(csv_path))

    # 检查必须有的列是否存在
    required_columns = {
        "patient_id",
        "label",
        *VIEW_COLUMNS,  # axial_dir, coronal_dir, sagittal_dir
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {sorted(missing)}")

    df["label"] = df["label"].astype(int)

    # 方式 1：如果 CSV 有 "split" 列，按它来划分
    if "split" in df.columns:
        train_df = df[df["split"] == "train"].copy()
        val_df = df[df["split"] == "val"].copy()
        test_df = df[df["split"] == "test"].copy()
        if len(train_df) == 0:
            raise ValueError("When using a split column, you need at least train rows.")
        if len(val_df) == 0 and len(test_df) == 0:
            raise ValueError("When using a split column, provide at least val or test rows.")
        return train_df, val_df, test_df

    # 方式 2：自动随机划分（分层抽样，保持正负样本比例一致）
    train_df, val_df = train_test_split(
        df,
        test_size=val_ratio,          # 验证集占比
        stratify=df["label"],         # 按标签分层，保证训练集和验证集中正负比例一致
        random_state=seed,            # 随机种子
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), pd.DataFrame()


def build_dataloaders(config: dict):
    """
    根据配置文件创建数据加载器（DataLoader）。

    DataLoader 的作用是：
        - 把数据集中的样本自动分成一个个小批次（batch）
        - 训练时自动打乱数据顺序（shuffle=True）
        - 支持多线程预加载数据（num_workers）

    返回：
        train_loader: 训练数据加载器
        val_loader:   验证数据加载器（可能为 None）
        test_loader:  测试数据加载器（可能为 None）
        train_df:     训练集的 DataFrame（后面计算类别权重要用）
    """
    data_cfg = config["data"]

    # 加载 CSV 并划分数据集
    train_df, val_df, test_df = load_and_split_dataframe(
        csv_path=data_cfg["csv_path"],
        seed=config["seed"],
        val_ratio=data_cfg["val_ratio"],
    )

    # 所有数据集共用的参数
    common_kwargs = {
        "base_dir": data_cfg["base_dir"],
        "image_size": data_cfg["image_size"],
        "num_slices_per_view": data_cfg["num_slices_per_view"],
        "trim_edge_slices": data_cfg.get("trim_edge_slices", 0),
        "hu_min": data_cfg["hu_min"],
        "hu_max": data_cfg["hu_max"],
    }

    # 创建 PyTorch 数据集对象
    train_dataset = PatientCTDataset(train_df, **common_kwargs)
    val_dataset = PatientCTDataset(val_df, **common_kwargs) if len(val_df) > 0 else None
    test_dataset = PatientCTDataset(test_df, **common_kwargs) if len(test_df) > 0 else None

    batch_size = data_cfg["batch_size"]    # 每批处理多少个样本（比如 4）
    num_workers = data_cfg["num_workers"]  # 用几个线程并行加载数据
    pin_memory = torch.cuda.is_available()  # 如果有 GPU，开启 pin_memory 加速数据传输

    # 创建训练集 DataLoader（训练时需要打乱数据顺序）
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,            # 打乱顺序！每个 epoch 数据顺序不同，防止模型记住顺序
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # 创建验证集 DataLoader（验证时不需要打乱）
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,       # 不打乱
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    # 创建测试集 DataLoader（如果有的话）
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


# ==================== 模型构建 ====================


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
        "fusion_hidden_dim": model_cfg["fusion_hidden_dim"], # 分类器隐藏层维度
        "dropout": model_cfg["dropout"],                     # Dropout 比率
    }
    fusion_type = model_cfg.get("fusion_type", "feature")

    if fusion_type == "feature":
        return MultiViewCTClassifier(**common_kwargs)
    if fusion_type == "decision":
        return MultiViewDecisionFusionClassifier(**common_kwargs)
    if fusion_type == "attention":
        # 注意力融合模式额外支持 cross_view_heads / cross_view_layers 配置
        attention_kwargs = {
            **common_kwargs,
            "cross_view_heads": model_cfg.get("cross_view_heads", 8),
            "cross_view_layers": model_cfg.get("cross_view_layers", 2),
        }
        return MultiViewAttentionClassifier(**attention_kwargs)

    raise ValueError(
        "Unsupported model.fusion_type. Expected one of: 'feature', 'decision', 'attention'."
    )


# ==================== 训练逻辑 ====================


def train_one_epoch(model, loader, criterion, optimizer, device):
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

        # ---------- 前向传播 ----------
        optimizer.zero_grad()     # 清零上一步的梯度（不清零会累加）
        logits = model(images)    # 模型预测，得到 (B, 2) 的分数
        loss = criterion(logits, labels)  # 计算损失（交叉熵损失）

        # ---------- 反向传播 + 参数更新 ----------
        loss.backward()           # 反向传播：自动计算每个参数的梯度
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


def score_for_model_selection(metrics: dict) -> float:
    """
    计算一个分数，用于选择"最佳模型"。

    优先用 AUC 作为选择标准（AUC 越高越好）。
    如果 AUC 不可用（比如验证集只有一个类别），则退而求其次用准确率。
    """
    auc = metrics.get("auc", float("nan"))
    if np.isnan(auc):
        return metrics["accuracy"]
    return auc


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
    set_seed(config["seed"])  # 固定随机种子，确保实验可重复

    # 创建输出目录（用来保存模型和结果）
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # 选择设备（自动检测是否有 GPU）
    device = choose_device(config["train"]["device"])
    print(f"Using device: {device}")

    # ---------- 第 2 步：加载数据 ----------
    train_loader, val_loader, test_loader, train_df = build_dataloaders(config)

    # ---------- 第 3 步：创建模型 ----------
    model = build_model(config).to(device)  # .to(device) 把模型搬到 GPU 上

    # ---------- 第 4 步：设置损失函数 ----------
    # CrossEntropyLoss = 交叉熵损失，用于分类任务
    # 如果数据不平衡（正常样本 >> 异常样本），可以给少数类更高的权重
    if config["train"]["class_weight"]:
        class_weights = compute_class_weights(train_df["label"].to_numpy()).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    # ---------- 第 5 步：设置优化器 ----------
    # Adam 优化器：自动调整学习率的优化算法
    # lr = 学习率（控制每次更新的步长大小）
    # weight_decay = 权重衰减（L2 正则化，防止过拟合）
    optimizer = torch.optim.Adam(
        model.parameters(),                    # 告诉优化器要更新哪些参数
        lr=config["train"]["lr"],              # 学习率
        weight_decay=config["train"]["weight_decay"],  # 权重衰减
    )

    # ---------- 第 6 步：训练循环 ----------
    history = []         # 记录每个 epoch 的训练历史
    best_score = -1.0    # 记录历史最佳分数
    best_path = output_dir / "best.pt"  # 最佳模型的保存路径

    for epoch in range(1, config["train"]["epochs"] + 1):
        print(f"\nEpoch {epoch}/{config['train']['epochs']}")

        # 训练一轮
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        record = {"epoch": epoch, "train_loss": train_loss}

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
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"val_auc={val_metrics['auc']:.4f} | val_acc={val_metrics['accuracy']:.4f}"
            )
            current_score = score_for_model_selection(val_metrics)
        else:
            print(f"train_loss={train_loss:.4f}")
            current_score = -train_loss  # 没有验证集时，用训练损失的负值作为分数（损失越小分数越高）

        history.append(record)

        # 如果当前模型比历史最佳还好，就保存它
        if current_score > best_score:
            best_score = current_score
            torch.save(
                {
                    "model_state_dict": model.state_dict(),  # 保存模型参数
                    "config": config,                         # 也保存配置文件，方便复现
                },
                best_path,
            )
            print(f"Saved best model to: {best_path}")

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
    else:
        summary["model_selection"] = "lowest_train_loss"

    # 在测试集上做最终评估（如果有测试集的话）
    if test_loader is not None:
        _, final_test_metrics = evaluate(model, test_loader, criterion, device)
        summary["test"] = final_test_metrics

    # 保存最终评估结果
    save_json(summary, output_dir / "summary.json")
    print("\nTraining finished.")
    print(f"Summary saved to: {output_dir / 'summary.json'}")


# ==================== 入口点 ====================
# 当直接运行 "python train.py" 时，会执行 main() 函数
if __name__ == "__main__":
    main()

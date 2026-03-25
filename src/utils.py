"""
utils.py — 通用工具函数
========================
这个文件包含一些在整个项目中都会用到的"小工具"函数。

包含的功能：
    - load_config:          读取 YAML 配置文件
    - set_seed:             设置随机种子（保证实验可重复）
    - choose_device:        自动选择运行设备（GPU 或 CPU）
    - compute_class_weights: 计算类别权重（处理数据不平衡）
    - compute_metrics:       计算分类评估指标
    - save_json:            保存数据为 JSON 文件
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import yaml  # 用来读取 YAML 格式的配置文件
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
# ↑ sklearn 提供的各种评估指标计算函数


def load_config(path: str | Path) -> Dict:
    """
    读取 YAML 格式的配置文件。

    YAML 是一种人类友好的配置文件格式，比如：
        train:
          lr: 0.001
          epochs: 50
    
    读取后会变成 Python 字典：
        {"train": {"lr": 0.001, "epochs": 50}}

    参数：
        path: 配置文件路径

    返回：
        配置字典
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """
    设置所有随机数生成器的种子。

    为什么要设置随机种子？
        深度学习中有很多随机操作（数据打乱、参数初始化、Dropout 等），
        设置固定的种子后，每次运行的结果都一样，方便复现和对比实验。

    参数：
        seed: 随机种子（任意整数，比如 42）
    """
    random.seed(seed)            # Python 原生随机库
    np.random.seed(seed)         # NumPy 随机库
    torch.manual_seed(seed)      # PyTorch CPU 随机库
    torch.cuda.manual_seed_all(seed)  # PyTorch GPU 随机库（所有 GPU）


def choose_device(device_name: str) -> torch.device:
    """
    选择运行设备。

    参数：
        device_name: 设备名称
            - "auto": 自动选择（有 GPU 就用 GPU，没有就用 CPU）
            - "cuda": 强制使用 GPU
            - "cpu":  强制使用 CPU

    返回：
        PyTorch 设备对象
    """
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def compute_class_weights(labels) -> torch.Tensor:
    """
    计算类别权重，用于处理数据不平衡问题。

    什么是数据不平衡？
        比如有 900 个正常样本但只有 100 个异常样本，
        如果不做处理，模型可能全部预测为"正常"就能有 90% 准确率，
        但实际上完全没有学到识别异常的能力。

    解决办法：
        给少数类（异常样本）更高的权重，让模型更重视对少数类的分类。

    计算公式：
        weight_i = 总样本数 / (类别数 × 该类样本数)

    示例：
        900 个正常 + 100 个异常
        → weight_正常 = 1000 / (2 × 900) ≈ 0.56
        → weight_异常 = 1000 / (2 × 100) = 5.0
        → 异常样本的权重是正常的约 9 倍

    参数：
        labels: 所有样本的标签数组

    返回：
        长度为 2 的权重张量 [正常类权重, 异常类权重]
    """
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=2).astype(np.float32)  # 统计每个类别的样本数
    counts[counts == 0] = 1.0  # 防止除以零
    weights = counts.sum() / (2.0 * counts)  # 计算权重
    return torch.tensor(weights, dtype=torch.float32)


def compute_metrics(y_true, y_pred, y_prob) -> Dict[str, float]:
    """
    计算分类任务的各种评估指标。

    指标说明：
        - accuracy（准确率）：预测正确的比例 = (TP+TN) / 总数
        - f1（F1 分数）：precision 和 recall 的调和平均，综合考虑精确率和召回率
        - auc（AUC）：ROC 曲线下面积，衡量模型区分正负样本的能力，1.0 表示完美，0.5 表示随机猜
        - sensitivity（灵敏度/召回率）：真阳性率 = TP / (TP+FN)，即"有病的人中被检出的比例"
        - specificity（特异度）：真阴性率 = TN / (TN+FP)，即"没病的人中被正确排除的比例"

    参数：
        y_true: 真实标签列表
        y_pred: 预测标签列表
        y_prob: 预测概率列表（每个样本 2 个概率 [正常概率, 异常概率]）

    返回：
        包含各项指标的字典
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    # 计算准确率和 F1
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    # 计算 AUC（需要至少有两个类别的样本才能计算）
    if len(np.unique(y_true)) == 2:
        metrics["auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))  # 用"异常"类的概率计算 AUC
    else:
        metrics["auc"] = float("nan")  # 只有一个类别时 AUC 无意义

    # 计算混淆矩阵并提取灵敏度和特异度
    # 混淆矩阵：
    #              预测正常  预测异常
    # 实际正常      TN        FP
    # 实际异常      FN        TP
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()  # 展平为 4 个值
    metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
    metrics["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
    return metrics


def save_json(data: Dict, path: str | Path) -> None:
    """
    把数据保存为 JSON 文件。

    参数：
        data: 要保存的数据（字典格式）
        path: 保存的文件路径
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)  # 自动创建父目录
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)  # 格式化输出，支持中文

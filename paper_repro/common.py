from __future__ import annotations

import copy
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset import PatientCTDataset, canonicalize_view_columns  # noqa: E402
from src.utils import choose_device, compute_class_weights, compute_metrics, save_json  # noqa: E402


def merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config_with_inheritance(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    inherit_from = config.pop("inherit_from", None)
    if inherit_from is None:
        return config
    parent_path = (path.parent / inherit_from).resolve()
    parent = load_config_with_inheritance(parent_path)
    return merge_dicts(parent, config)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_dataframe(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_cfg = config["data"]
    csv_path = REPO_ROOT / data_cfg["csv_path"]
    df = canonicalize_view_columns(pd.read_csv(csv_path))
    if data_cfg.get("use_existing_split", True) and "split" in df.columns:
        train_df = df[df["split"] == "train"].copy().reset_index(drop=True)
        val_df = df[df["split"] == "val"].copy().reset_index(drop=True)
        test_df = df[df["split"] == "test"].copy().reset_index(drop=True)
        return train_df, val_df, test_df
    raise ValueError("paper_repro currently expects an existing split column in the metadata CSV.")


def build_dataloaders(config: dict[str, Any]) -> tuple[DataLoader, DataLoader | None, DataLoader | None, pd.DataFrame]:
    data_cfg = config["data"]
    train_df, val_df, test_df = read_dataframe(config)

    dataset_kwargs = {
        "base_dir": REPO_ROOT / data_cfg["base_dir"],
        "image_size": int(data_cfg["image_size"]),
        "num_slices_per_view": int(data_cfg["num_slices_per_view"]),
        "trim_edge_slices": int(data_cfg.get("trim_edge_slices", 0)),
        "hu_min": float(data_cfg["hu_min"]),
        "hu_max": float(data_cfg["hu_max"]),
    }

    train_dataset = PatientCTDataset(train_df, **dataset_kwargs)
    val_dataset = PatientCTDataset(val_df, **dataset_kwargs) if len(val_df) > 0 else None
    test_dataset = PatientCTDataset(test_df, **dataset_kwargs) if len(test_df) > 0 else None

    batch_size = int(data_cfg["batch_size"])
    num_workers = int(data_cfg.get("num_workers", 1))
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


class FocalCrossEntropy(torch.nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = torch.nn.functional.cross_entropy(logits, targets, reduction="none", weight=self.weight)
        prob = torch.softmax(logits, dim=1)
        pt = prob.gather(1, targets.unsqueeze(1)).squeeze(1).clamp_min(1e-6)
        return (((1 - pt) ** self.gamma) * ce).mean()


def build_loss(config: dict[str, Any], train_df: pd.DataFrame, device: torch.device) -> torch.nn.Module:
    loss_name = str(config["train"].get("loss", "ce")).lower()
    class_weights = compute_class_weights(train_df["label"].tolist()).to(device)
    if loss_name == "ce":
        return torch.nn.CrossEntropyLoss(weight=class_weights)
    if loss_name == "focal":
        return FocalCrossEntropy(
            gamma=float(config["train"].get("focal_gamma", 2.0)),
            weight=class_weights,
        )
    if loss_name == "ce_focal":
        focal = FocalCrossEntropy(
            gamma=float(config["train"].get("focal_gamma", 2.0)),
            weight=class_weights,
        )
        ce = torch.nn.CrossEntropyLoss(weight=class_weights)

        class HybridLoss(torch.nn.Module):
            def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
                return 0.5 * ce(logits, targets) + 0.5 * focal(logits, targets)

        return HybridLoss()
    raise ValueError(f"Unsupported loss: {loss_name}")


def build_optimizer(config: dict[str, Any], model: torch.nn.Module) -> torch.optim.Optimizer:
    train_cfg = config["train"]
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )


def build_scheduler(config: dict[str, Any], optimizer: torch.optim.Optimizer):
    name = str(config["train"].get("scheduler", "plateau")).lower()
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=max(1, int(config["train"].get("lr_patience", 1))),
        )
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(config["train"]["epochs"])),
        )
    raise ValueError(f"Unsupported scheduler: {name}")


def metric_score(metrics: dict[str, float]) -> tuple[float, float]:
    return (float(metrics.get("accuracy", 0.0)), float(metrics.get("auc", float("-inf"))))


def format_float(value: float) -> str:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "nan"
    return f"{value:.6f}"


def save_history(history: list[dict[str, Any]], output_dir: Path) -> None:
    save_json(history, output_dir / "history.json")


def load_manifest(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def latest_export_timestamp() -> str:
    return Path(".").resolve().name


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_output_dir(config: dict[str, Any]) -> Path:
    output_dir = REPO_ROOT / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def detect_device(config: dict[str, Any]) -> torch.device:
    return choose_device(str(config["train"].get("device", "auto")))


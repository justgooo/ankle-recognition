from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import PatientCTDataset
from src.model import MultiViewCTClassifier
from src.utils import (
    choose_device,
    compute_class_weights,
    compute_metrics,
    load_config,
    save_json,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a beginner multi-view CT classifier.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    return parser.parse_args()


def load_and_split_dataframe(csv_path: str, seed: int, val_ratio: float):
    df = pd.read_csv(csv_path)
    required_columns = {
        "patient_id",
        "label",
        "view1_dir",
        "view2_dir",
        "view3_dir",
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {sorted(missing)}")

    df["label"] = df["label"].astype(int)

    if "split" in df.columns:
        train_df = df[df["split"] == "train"].copy()
        val_df = df[df["split"] == "val"].copy()
        test_df = df[df["split"] == "test"].copy()
        if len(train_df) == 0:
            raise ValueError("When using a split column, you need at least train rows.")
        if len(val_df) == 0 and len(test_df) == 0:
            raise ValueError("When using a split column, provide at least val or test rows.")
        return train_df, val_df, test_df

    train_df, val_df = train_test_split(
        df,
        test_size=val_ratio,
        stratify=df["label"],
        random_state=seed,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), pd.DataFrame()


def build_dataloaders(config: dict):
    data_cfg = config["data"]
    train_df, val_df, test_df = load_and_split_dataframe(
        csv_path=data_cfg["csv_path"],
        seed=config["seed"],
        val_ratio=data_cfg["val_ratio"],
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


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    progress = tqdm(loader, desc="train", leave=False)

    for batch in progress:
        images = batch["images"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []

    for batch in tqdm(loader, desc="eval", leave=False):
        images = batch["images"].to(device)
        labels = batch["label"].to(device)

        logits = model(images)
        loss = criterion(logits, labels)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        running_loss += loss.item() * labels.size(0)
        all_labels.extend(labels.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())

    average_loss = running_loss / len(loader.dataset)
    metrics = compute_metrics(all_labels, all_preds, all_probs)
    return average_loss, metrics


def score_for_model_selection(metrics: dict) -> float:
    auc = metrics.get("auc", float("nan"))
    if np.isnan(auc):
        return metrics["accuracy"]
    return auc


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(config["seed"])

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(config["train"]["device"])
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, train_df = build_dataloaders(config)

    model = MultiViewCTClassifier(
        share_backbone=config["model"]["share_backbone"],
        use_pretrained=config["model"]["use_pretrained"],
        fusion_hidden_dim=config["model"]["fusion_hidden_dim"],
        dropout=config["model"]["dropout"],
    ).to(device)

    if config["train"]["class_weight"]:
        class_weights = compute_class_weights(train_df["label"].to_numpy()).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"],
    )

    history = []
    best_score = -1.0
    best_path = output_dir / "best.pt"

    for epoch in range(1, config["train"]["epochs"] + 1):
        print(f"\nEpoch {epoch}/{config['train']['epochs']}")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        record = {"epoch": epoch, "train_loss": train_loss}

        if val_loader is not None:
            val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
            record.update(
                {
                    "val_loss": val_loss,
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                }
            )
            print(
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"val_auc={val_metrics['auc']:.4f} | val_acc={val_metrics['accuracy']:.4f}"
            )
            current_score = score_for_model_selection(val_metrics)
        else:
            print(f"train_loss={train_loss:.4f}")
            current_score = -train_loss

        history.append(record)
        if current_score > best_score:
            best_score = current_score
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                },
                best_path,
            )
            print(f"Saved best model to: {best_path}")

    save_json({"history": history}, output_dir / "history.json")

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    summary = {}

    if val_loader is not None:
        _, final_val_metrics = evaluate(model, val_loader, criterion, device)
        summary["best_val"] = final_val_metrics
    else:
        summary["model_selection"] = "lowest_train_loss"

    if test_loader is not None:
        _, final_test_metrics = evaluate(model, test_loader, criterion, device)
        summary["test"] = final_test_metrics

    save_json(summary, output_dir / "summary.json")
    print("\nTraining finished.")
    print(f"Summary saved to: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()

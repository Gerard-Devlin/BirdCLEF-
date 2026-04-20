from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Tuple
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from birdclef_plus import (
    BirdCLEFNet,
    BirdTrainDataset,
    MelSpectrogramFrontend,
    build_label_mapping,
    load_train_csv,
    macro_auc_skip_empty,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BirdCLEF+ 2026 baseline training")
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/baseline"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--segment-seconds", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixup-alpha", type=float, default=0.2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def assign_folds(df: pd.DataFrame, n_splits: int, seed: int) -> pd.DataFrame:
    if n_splits < 2:
        df = df.copy()
        df["fold"] = 0
        return df

    df = df.copy()
    df["fold"] = -1
    label_count = df["primary_label"].value_counts()
    rare_mask = df["primary_label"].map(label_count) < n_splits

    regular_index = df.index[~rare_mask].to_numpy()
    rare_index = df.index[rare_mask].to_numpy()

    if regular_index.size > 0:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        regular_labels = df.loc[regular_index, "primary_label"].values
        for fold, (_, valid_local) in enumerate(skf.split(regular_index, regular_labels)):
            valid_index = regular_index[valid_local]
            df.loc[valid_index, "fold"] = fold

    rng = np.random.default_rng(seed)
    rng.shuffle(rare_index)
    for i, idx in enumerate(rare_index):
        df.loc[idx, "fold"] = i % n_splits

    return df


def mixup_batch(
    features: torch.Tensor, targets: torch.Tensor, alpha: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    if alpha <= 0 or features.size(0) < 2:
        return features, targets

    lam = np.random.beta(alpha, alpha)
    perm = torch.randperm(features.size(0), device=features.device)
    mixed_features = lam * features + (1.0 - lam) * features[perm]
    mixed_targets = lam * targets + (1.0 - lam) * targets[perm]
    return mixed_features, mixed_targets


def train_one_epoch(
    model: nn.Module,
    frontend: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
    use_amp: bool,
    mixup_alpha: float,
) -> float:
    model.train()
    frontend.train()
    running_loss = 0.0

    progress = tqdm(loader, desc="train", leave=False)
    for waveforms, targets in progress:
        waveforms = waveforms.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            features = frontend(waveforms)
            features, targets = mixup_batch(features, targets, alpha=mixup_alpha)
            logits = model(features)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += float(loss.item()) * waveforms.size(0)
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / len(loader.dataset)


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    frontend: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    frontend.eval()

    running_loss = 0.0
    predictions = []
    targets_all = []

    progress = tqdm(loader, desc="valid", leave=False)
    for waveforms, targets in progress:
        waveforms = waveforms.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            features = frontend(waveforms)
            logits = model(features)
            loss = criterion(logits, targets)

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        gt = targets.detach().cpu().numpy()
        predictions.append(probs)
        targets_all.append(gt)
        running_loss += float(loss.item()) * waveforms.size(0)

    y_pred = np.concatenate(predictions, axis=0)
    y_true = np.concatenate(targets_all, axis=0)
    score = macro_auc_skip_empty(y_true, y_pred)
    val_loss = running_loss / len(loader.dataset)
    return val_loss, score, y_true, y_pred


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.amp and device.type == "cuda"
    print(f"Device: {device}, AMP: {use_amp}")

    df = load_train_csv(args.train_csv)
    df = assign_folds(df, n_splits=args.folds, seed=args.seed)
    if args.fold >= args.folds:
        raise ValueError(f"--fold must be < --folds ({args.folds})")

    train_df = df[df["fold"] != args.fold].reset_index(drop=True)
    valid_df = df[df["fold"] == args.fold].reset_index(drop=True)
    print(f"Train rows: {len(train_df)}, Valid rows: {len(valid_df)}, Fold: {args.fold}")

    labels = build_label_mapping(df)
    label_to_idx = {label: i for i, label in enumerate(labels)}
    (args.output_dir / "labels.json").write_text(
        json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    train_dataset = BirdTrainDataset(
        dataframe=train_df,
        audio_dir=args.audio_dir,
        label_to_idx=label_to_idx,
        segment_seconds=args.segment_seconds,
        is_train=True,
    )
    valid_dataset = BirdTrainDataset(
        dataframe=valid_df,
        audio_dir=args.audio_dir,
        label_to_idx=label_to_idx,
        segment_seconds=args.segment_seconds,
        is_train=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    frontend = MelSpectrogramFrontend(augment=True).to(device)
    model = BirdCLEFNet(num_classes=len(labels), dropout=args.dropout).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.05
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_score = -1.0
    best_path = args.output_dir / f"best_fold{args.fold}.pth"
    last_path = args.output_dir / f"last_fold{args.fold}.pth"

    train_start = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss = train_one_epoch(
            model=model,
            frontend=frontend,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
            mixup_alpha=args.mixup_alpha,
        )
        val_loss, val_score, y_true, y_pred = validate_one_epoch(
            model=model,
            frontend=frontend,
            loader=valid_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
        )
        scheduler.step()

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "frontend_state_dict": frontend.state_dict(),
            "labels": labels,
            "fold": args.fold,
            "best_val_auc": best_score,
            "args": vars(args),
        }
        torch.save(checkpoint, last_path)

        if val_score > best_score:
            best_score = val_score
            checkpoint["best_val_auc"] = best_score
            torch.save(checkpoint, best_path)
            np.savez_compressed(
                args.output_dir / f"oof_fold{args.fold}.npz",
                y_true=y_true,
                y_pred=y_pred,
            )

        elapsed = time.time() - epoch_start
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_auc={val_score:.5f} | best_auc={best_score:.5f} | "
            f"epoch_time={elapsed:.1f}s"
        )

    total_minutes = (time.time() - train_start) / 60.0
    print(f"Training done. best_auc={best_score:.5f} | total_time={total_minutes:.1f} min")
    print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()

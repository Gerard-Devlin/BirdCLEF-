from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable, Optional, Tuple
import sys
import site

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
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=1.0,
        help="Clip gradient norm. Set <=0 to disable clipping.",
    )
    parser.add_argument(
        "--save-every-steps",
        type=int,
        default=0,
        help="Save rolling step checkpoint every N optimizer steps. 0 disables it.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to checkpoint (.pth) to resume training from.",
    )
    parser.add_argument(
        "--no-save-on-interrupt",
        action="store_true",
        help="Disable automatic interrupt checkpoint on Ctrl+C or termination.",
    )
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
    scaler,
    use_amp: bool,
    mixup_alpha: float,
    grad_clip_norm: float = 1.0,
    global_step_start: int = 0,
    save_every_steps: int = 0,
    on_step_checkpoint: Optional[Callable[[int], None]] = None,
) -> Tuple[float, int]:
    model.train()
    frontend.train()
    running_loss = 0.0
    seen_samples = 0
    global_step = int(global_step_start)

    progress = tqdm(loader, desc="train", leave=False)
    skipped_batches = 0
    for waveforms, targets in progress:
        waveforms = waveforms.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            features = frontend(waveforms)
            features, targets = mixup_batch(features, targets, alpha=mixup_alpha)
            logits = model(features)
            loss = criterion(logits, targets)

        if not torch.isfinite(loss):
            skipped_batches += 1
            print(
                f"[WARN] Non-finite loss detected ({loss.item()}). Skipping this batch.",
                flush=True,
            )
            continue

        scaler.scale(loss).backward()
        if grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        global_step += 1
        running_loss += float(loss.item()) * waveforms.size(0)
        seen_samples += int(waveforms.size(0))
        progress.set_postfix(loss=f"{loss.item():.4f}")

        if (
            save_every_steps > 0
            and on_step_checkpoint is not None
            and global_step % save_every_steps == 0
        ):
            on_step_checkpoint(global_step)

    if skipped_batches > 0:
        print(f"[WARN] Skipped {skipped_batches} non-finite batches in this epoch.", flush=True)

    return running_loss / max(1, seen_samples), global_step


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
    non_finite_val_batches = 0

    progress = tqdm(loader, desc="valid", leave=False)
    for waveforms, targets in progress:
        waveforms = waveforms.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            features = frontend(waveforms)
            logits = model(features)
            loss = criterion(logits, targets)

        if not torch.isfinite(logits).all():
            non_finite_val_batches += 1
            logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
        if not torch.isfinite(loss):
            non_finite_val_batches += 1
            loss = torch.zeros((), device=targets.device, dtype=targets.dtype)

        probs = torch.sigmoid(logits)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0).detach().cpu().numpy()
        gt = targets.detach().cpu().numpy()
        predictions.append(probs)
        targets_all.append(gt)
        running_loss += float(loss.item()) * waveforms.size(0)

    y_pred = np.concatenate(predictions, axis=0)
    y_true = np.concatenate(targets_all, axis=0)
    if non_finite_val_batches > 0:
        print(
            f"[WARN] Non-finite tensors appeared in {non_finite_val_batches} validation batches. "
            "Sanitized to finite values for scoring.",
            flush=True,
        )
    score = macro_auc_skip_empty(y_true, y_pred)
    val_loss = running_loss / len(loader.dataset)
    return val_loss, score, y_true, y_pred


def build_checkpoint(
    *,
    epoch: int,
    global_step: int,
    best_score: float,
    model: nn.Module,
    frontend: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    labels,
    args: argparse.Namespace,
) -> dict:
    scaler_state = None
    if scaler is not None:
        try:
            scaler_state = scaler.state_dict()
        except Exception:
            scaler_state = None

    return {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "frontend_state_dict": frontend.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler_state,
        "labels": labels,
        "fold": args.fold,
        "best_val_auc": float(best_score),
        "args": vars(args),
        "saved_at_unix": time.time(),
    }


def atomic_torch_save(obj: dict, path: Path) -> None:
    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp_path)
    tmp_path.replace(path)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    try:
        user_site = site.getusersitepackages()
    except Exception:
        user_site = None
    if user_site and user_site in sys.path:
        print(
            f"[WARN] User site-packages is active: {user_site}. "
            "This can mix Python 3.13 local packages into conda env. "
            "Use `PYTHONNOUSERSITE=1` when launching training.",
            flush=True,
        )

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
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    else:  # pragma: no cover - compatibility fallback
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_score = -1.0
    start_epoch = 1
    global_step = 0
    best_path = args.output_dir / f"best_fold{args.fold}.pth"
    last_path = args.output_dir / f"last_fold{args.fold}.pth"
    step_path = args.output_dir / f"step_last_fold{args.fold}.pth"
    interrupt_path = args.output_dir / f"interrupt_fold{args.fold}.pth"

    if args.resume is not None:
        if not args.resume.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        print(f"Resuming from: {args.resume}")
        resume_ckpt = torch.load(args.resume, map_location="cpu")

        resume_labels = resume_ckpt.get("labels")
        if resume_labels is not None and list(resume_labels) != list(labels):
            raise ValueError(
                "Label mapping mismatch between resume checkpoint and current dataset."
            )

        model.load_state_dict(resume_ckpt["model_state_dict"], strict=True)
        if "frontend_state_dict" in resume_ckpt:
            frontend.load_state_dict(resume_ckpt["frontend_state_dict"], strict=False)
        if "optimizer_state_dict" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in resume_ckpt:
            scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])
        scaler_state = resume_ckpt.get("scaler_state_dict")
        if scaler_state:
            try:
                scaler.load_state_dict(scaler_state)
            except Exception as exc:
                print(f"[WARN] Failed to load scaler state: {exc}")

        best_score = float(resume_ckpt.get("best_val_auc", best_score))
        start_epoch = int(resume_ckpt.get("epoch", 0)) + 1
        global_step = int(resume_ckpt.get("global_step", 0))
        print(
            f"Resume state -> start_epoch={start_epoch}, global_step={global_step}, best_auc={best_score:.5f}"
        )

    if start_epoch > args.epochs:
        print(
            f"Nothing to train: start_epoch={start_epoch} is greater than --epochs={args.epochs}."
        )
        return

    train_start = time.time()
    current_epoch = start_epoch - 1
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            current_epoch = epoch
            epoch_start = time.time()

            def on_step_checkpoint(step_value: int) -> None:
                checkpoint = build_checkpoint(
                    epoch=epoch,
                    global_step=step_value,
                    best_score=best_score,
                    model=model,
                    frontend=frontend,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    labels=labels,
                    args=args,
                )
                atomic_torch_save(checkpoint, step_path)

            train_loss, global_step = train_one_epoch(
                model=model,
                frontend=frontend,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                scaler=scaler,
                use_amp=use_amp,
                mixup_alpha=args.mixup_alpha,
                grad_clip_norm=args.grad_clip_norm,
                global_step_start=global_step,
                save_every_steps=args.save_every_steps,
                on_step_checkpoint=on_step_checkpoint,
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

            if val_score > best_score:
                best_score = val_score
                best_checkpoint = build_checkpoint(
                    epoch=epoch,
                    global_step=global_step,
                    best_score=best_score,
                    model=model,
                    frontend=frontend,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    labels=labels,
                    args=args,
                )
                atomic_torch_save(best_checkpoint, best_path)
                np.savez_compressed(
                    args.output_dir / f"oof_fold{args.fold}.npz",
                    y_true=y_true,
                    y_pred=y_pred,
                )

            last_checkpoint = build_checkpoint(
                epoch=epoch,
                global_step=global_step,
                best_score=best_score,
                model=model,
                frontend=frontend,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                labels=labels,
                args=args,
            )
            atomic_torch_save(last_checkpoint, last_path)

            elapsed = time.time() - epoch_start
            print(
                f"Epoch {epoch:02d}/{args.epochs} | "
                f"step={global_step} | "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"val_auc={val_score:.5f} | best_auc={best_score:.5f} | "
                f"epoch_time={elapsed:.1f}s"
            )
    except (KeyboardInterrupt, SystemExit):
        if not args.no_save_on_interrupt:
            print("\n[WARN] Training interrupted, saving interrupt checkpoint...", flush=True)
            interrupt_checkpoint = build_checkpoint(
                epoch=current_epoch,
                global_step=global_step,
                best_score=best_score,
                model=model,
                frontend=frontend,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                labels=labels,
                args=args,
            )
            atomic_torch_save(interrupt_checkpoint, interrupt_path)
            print(f"[WARN] Interrupt checkpoint saved: {interrupt_path}", flush=True)
        raise

    total_minutes = (time.time() - train_start) / 60.0
    print(f"Training done. best_auc={best_score:.5f} | total_time={total_minutes:.1f} min")
    print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()

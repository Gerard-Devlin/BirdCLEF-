from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple
import sys
import site

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from birdclef_plus import (
    BirdCLEFNet,
    BirdSoundscapeDataset,
    BirdTrainDataset,
    MelSpectrogramFrontend,
    build_label_mapping,
    load_soundscape_labels_csv,
    load_train_csv,
    macro_auc_skip_empty,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BirdCLEF+ 2026 baseline training")
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument(
        "--submission-csv",
        type=Path,
        default=None,
        help=(
            "Optional sample_submission.csv path. If provided, model output labels will "
            "match submission columns exactly."
        ),
    )
    parser.add_argument(
        "--soundscape-labels-csv",
        type=Path,
        default=None,
        help="Optional train_soundscapes_labels.csv path for soundscape fine-tuning rows.",
    )
    parser.add_argument(
        "--soundscape-audio-dir",
        type=Path,
        default=None,
        help="Optional train_soundscapes directory for soundscape fine-tuning rows.",
    )
    parser.add_argument(
        "--soundscape-repeat",
        type=int,
        default=4,
        help=(
            "Repeat the soundscape train dataset this many times inside one epoch "
            "to increase its sampling weight."
        ),
    )
    parser.add_argument(
        "--soundscape-pseudo-csv",
        type=Path,
        default=None,
        help=(
            "Optional pseudo-labeled soundscape CSV. Pseudo rows are added to TRAIN split only "
            "(validation remains real-labeled soundscapes)."
        ),
    )
    parser.add_argument(
        "--soundscape-pseudo-repeat",
        type=int,
        default=1,
        help=(
            "Repeat the pseudo-labeled soundscape train dataset this many times "
            "inside one epoch."
        ),
    )
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
    parser.add_argument(
        "--model-name",
        type=str,
        default="custom_cnn",
        choices=("custom_cnn", "efficientnet_b0", "efficientnet_b2"),
        help="Backbone model architecture.",
    )
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--select-best-on",
        type=str,
        choices=("all", "soundscape"),
        default="soundscape",
        help=(
            "Metric used to decide best checkpoint. "
            "'all' uses combined validation set; 'soundscape' uses soundscape-only validation."
        ),
    )
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
        "--resume-model-only",
        action="store_true",
        help=(
            "Load only model/frontend weights from --resume and reset optimizer/scheduler/scaler "
            "states. Useful when previous optimizer state causes instability."
        ),
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


def assign_group_folds(
    df: pd.DataFrame, group_col: str, n_splits: int, seed: int
) -> pd.DataFrame:
    if n_splits < 2:
        df = df.copy()
        df["fold"] = 0
        return df

    df = df.copy()
    groups = sorted(df[group_col].astype(str).unique().tolist())
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    group_to_fold = {group: i % n_splits for i, group in enumerate(groups)}
    df["fold"] = df[group_col].astype(str).map(group_to_fold).astype(int)
    return df


def build_group_fold_mapping(groups, n_splits: int, seed: int) -> Dict[str, int]:
    groups = sorted({str(group) for group in groups})
    if not groups:
        return {}
    if n_splits < 2:
        return {group: 0 for group in groups}

    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    return {group: i % n_splits for i, group in enumerate(groups)}


def apply_group_fold_mapping(
    df: pd.DataFrame, group_col: str, group_to_fold: Dict[str, int]
) -> pd.DataFrame:
    df = df.copy()
    if not group_to_fold:
        df["fold"] = 0
        return df

    mapped = df[group_col].astype(str).map(group_to_fold)
    if mapped.isna().any():
        missing_groups = sorted(df.loc[mapped.isna(), group_col].astype(str).unique().tolist())
        raise ValueError(
            "Found groups without fold assignment: "
            f"{missing_groups[:10]} (total {len(missing_groups)})"
        )
    df["fold"] = mapped.astype(int)
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
) -> Tuple[float, float, np.ndarray, np.ndarray, int, int]:
    model.eval()
    frontend.eval()

    running_loss = 0.0
    predictions = []
    targets_all = []
    bad_logits_batches = 0
    bad_loss_batches = 0

    progress = tqdm(loader, desc="valid", leave=False)
    for waveforms, targets in progress:
        waveforms = waveforms.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            features = frontend(waveforms)
            logits = model(features)
            loss = criterion(logits, targets)

        if not torch.isfinite(logits).all():
            bad_logits_batches += 1
            logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
        if not torch.isfinite(loss):
            bad_loss_batches += 1
            loss = torch.zeros((), device=targets.device, dtype=targets.dtype)

        probs = torch.sigmoid(logits)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0).detach().cpu().numpy()
        gt = targets.detach().cpu().numpy()
        predictions.append(probs)
        targets_all.append(gt)
        running_loss += float(loss.item()) * waveforms.size(0)

    y_pred = np.concatenate(predictions, axis=0)
    y_true = np.concatenate(targets_all, axis=0)
    if bad_logits_batches > 0 or bad_loss_batches > 0:
        print(
            f"[WARN] Non-finite tensors in validation batches: logits={bad_logits_batches}, "
            f"loss={bad_loss_batches}. Sanitized to finite values for scoring.",
            flush=True,
        )
    score = macro_auc_skip_empty(y_true, y_pred)
    val_loss = running_loss / len(loader.dataset)
    return val_loss, score, y_true, y_pred, bad_logits_batches, bad_loss_batches


def count_non_finite_params(module: nn.Module) -> int:
    total_bad = 0
    for parameter in module.parameters():
        bad = ~torch.isfinite(parameter.data)
        total_bad += int(bad.sum().item())
    return total_bad


def adapt_state_dict_prefix_for_model(
    model: nn.Module, checkpoint_state_dict: dict
) -> Tuple[dict, Optional[str]]:
    model_keys = list(model.state_dict().keys())
    ckpt_keys = list(checkpoint_state_dict.keys())

    model_has_backbone_prefix = any(key.startswith("backbone.") for key in model_keys)
    ckpt_has_backbone_prefix = any(key.startswith("backbone.") for key in ckpt_keys)

    if model_has_backbone_prefix and not ckpt_has_backbone_prefix:
        adapted = {}
        for key, value in checkpoint_state_dict.items():
            if key.startswith("backbone."):
                adapted[key] = value
            else:
                adapted[f"backbone.{key}"] = value
        return (
            adapted,
            "[INFO] Adapted resume checkpoint state_dict from legacy key format to 'backbone.*' format.",
        )

    if (not model_has_backbone_prefix) and ckpt_has_backbone_prefix:
        adapted = {}
        for key, value in checkpoint_state_dict.items():
            if key.startswith("backbone."):
                adapted[key[len("backbone.") :]] = value
            else:
                adapted[key] = value
        return (
            adapted,
            "[INFO] Adapted resume checkpoint state_dict from 'backbone.*' format to legacy key format.",
        )

    return checkpoint_state_dict, None


def load_model_with_label_alignment(
    model: nn.Module,
    checkpoint_state: dict,
    checkpoint_labels,
    current_labels,
) -> int:
    model_state = model.state_dict()
    copied_labels = 0

    backbone_state = {}
    for key, value in checkpoint_state.items():
        if key.startswith("head."):
            continue
        if key in model_state and model_state[key].shape == value.shape:
            backbone_state[key] = value
    model.load_state_dict(backbone_state, strict=False)

    src_w = checkpoint_state.get("head.weight")
    src_b = checkpoint_state.get("head.bias")
    if src_w is None or src_b is None:
        return copied_labels
    if checkpoint_labels is None:
        return copied_labels

    checkpoint_label_to_idx = {str(label): i for i, label in enumerate(checkpoint_labels)}
    dst_w = model_state["head.weight"].clone()
    dst_b = model_state["head.bias"].clone()

    for dst_idx, label in enumerate(current_labels):
        src_idx = checkpoint_label_to_idx.get(str(label))
        if src_idx is None:
            continue
        if src_idx >= src_w.shape[0]:
            continue
        dst_w[dst_idx] = src_w[src_idx]
        dst_b[dst_idx] = src_b[src_idx]
        copied_labels += 1

    model_state["head.weight"] = dst_w
    model_state["head.bias"] = dst_b
    model.load_state_dict(model_state, strict=False)
    return copied_labels


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
    best_metric_name: str,
) -> dict:
    def _to_serializable(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {k: _to_serializable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_to_serializable(v) for v in value]
        return value

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
        "best_metric_name": str(best_metric_name),
        "args": _to_serializable(vars(args)),
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

    submission_csv = args.submission_csv
    if submission_csv is None:
        auto_submission_csv = args.train_csv.parent / "sample_submission.csv"
        if auto_submission_csv.exists():
            submission_csv = auto_submission_csv

    submission_labels = None
    if submission_csv is not None:
        if not submission_csv.exists():
            raise FileNotFoundError(f"Submission CSV not found: {submission_csv}")
        submission_df = pd.read_csv(submission_csv, nrows=1)
        submission_labels = [str(column) for column in submission_df.columns.tolist()[1:]]
        print(
            f"Submission label space loaded from {submission_csv} "
            f"({len(submission_labels)} classes)."
        )

    soundscape_labels_csv = args.soundscape_labels_csv
    if soundscape_labels_csv is None:
        auto_soundscape_labels_csv = args.train_csv.parent / "train_soundscapes_labels.csv"
        if auto_soundscape_labels_csv.exists():
            soundscape_labels_csv = auto_soundscape_labels_csv

    soundscape_audio_dir = args.soundscape_audio_dir
    if soundscape_audio_dir is None:
        auto_soundscape_audio_dir = args.train_csv.parent / "train_soundscapes"
        if auto_soundscape_audio_dir.exists():
            soundscape_audio_dir = auto_soundscape_audio_dir

    soundscape_pseudo_csv = args.soundscape_pseudo_csv
    if soundscape_pseudo_csv is None:
        auto_soundscape_pseudo_csv = args.train_csv.parent / "train_soundscapes_pseudo.csv"
        if auto_soundscape_pseudo_csv.exists():
            soundscape_pseudo_csv = auto_soundscape_pseudo_csv

    soundscape_df = None
    soundscape_real_df = None
    soundscape_pseudo_df = None
    train_soundscape_df = None
    valid_soundscape_df = None
    train_soundscape_pseudo_df = None
    if (
        soundscape_labels_csv is not None
        or soundscape_pseudo_csv is not None
        or soundscape_audio_dir is not None
    ):
        if (soundscape_labels_csv is None and soundscape_pseudo_csv is None) or soundscape_audio_dir is None:
            raise ValueError(
                "To enable soundscape training, provide --soundscape-audio-dir and at least one "
                "of --soundscape-labels-csv / --soundscape-pseudo-csv "
                "(or keep files in dataset dir for auto-detection)."
            )
        if not soundscape_audio_dir.exists():
            raise FileNotFoundError(f"Soundscape audio directory not found: {soundscape_audio_dir}")
        if soundscape_labels_csv is not None:
            if not soundscape_labels_csv.exists():
                raise FileNotFoundError(f"Soundscape labels CSV not found: {soundscape_labels_csv}")
            soundscape_real_df = load_soundscape_labels_csv(soundscape_labels_csv)

        if soundscape_pseudo_csv is not None:
            if not soundscape_pseudo_csv.exists():
                raise FileNotFoundError(f"Soundscape pseudo CSV not found: {soundscape_pseudo_csv}")
            soundscape_pseudo_df = load_soundscape_labels_csv(soundscape_pseudo_csv)

        all_groups = []
        if soundscape_real_df is not None:
            all_groups.extend(soundscape_real_df["filename"].astype(str).tolist())
        if soundscape_pseudo_df is not None:
            all_groups.extend(soundscape_pseudo_df["filename"].astype(str).tolist())
        group_to_fold = build_group_fold_mapping(
            all_groups, n_splits=args.folds, seed=args.seed
        )

        if soundscape_real_df is not None:
            soundscape_real_df = apply_group_fold_mapping(
                soundscape_real_df, group_col="filename", group_to_fold=group_to_fold
            )
            train_soundscape_df = soundscape_real_df[
                soundscape_real_df["fold"] != args.fold
            ].reset_index(drop=True)
            valid_soundscape_df = soundscape_real_df[
                soundscape_real_df["fold"] == args.fold
            ].reset_index(drop=True)
            print(
                f"Soundscape rows (real): train={len(train_soundscape_df)}, "
                f"valid={len(valid_soundscape_df)} (source={soundscape_labels_csv})"
            )

        if soundscape_pseudo_df is not None:
            soundscape_pseudo_df = apply_group_fold_mapping(
                soundscape_pseudo_df, group_col="filename", group_to_fold=group_to_fold
            )
            train_soundscape_pseudo_df = soundscape_pseudo_df[
                soundscape_pseudo_df["fold"] != args.fold
            ].reset_index(drop=True)
            print(
                f"Soundscape rows (pseudo): train={len(train_soundscape_pseudo_df)}, "
                f"valid=0 (source={soundscape_pseudo_csv})"
            )

        if soundscape_real_df is not None and soundscape_pseudo_df is not None:
            soundscape_df = pd.concat(
                [soundscape_real_df, soundscape_pseudo_df], ignore_index=True
            )
        elif soundscape_real_df is not None:
            soundscape_df = soundscape_real_df
        elif soundscape_pseudo_df is not None:
            soundscape_df = soundscape_pseudo_df

    labels = build_label_mapping(
        df, soundscape_df=soundscape_df, submission_labels=submission_labels
    )
    label_to_idx = {label: i for i, label in enumerate(labels)}
    print(f"Model label space: {len(labels)} classes.")
    (args.output_dir / "labels.json").write_text(
        json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    train_audio_dataset = BirdTrainDataset(
        dataframe=train_df,
        audio_dir=args.audio_dir,
        label_to_idx=label_to_idx,
        segment_seconds=args.segment_seconds,
        is_train=True,
    )
    valid_audio_dataset = BirdTrainDataset(
        dataframe=valid_df,
        audio_dir=args.audio_dir,
        label_to_idx=label_to_idx,
        segment_seconds=args.segment_seconds,
        is_train=False,
    )

    train_datasets = [train_audio_dataset]
    valid_datasets = [valid_audio_dataset]
    valid_soundscape_dataset = None

    if train_soundscape_df is not None and valid_soundscape_df is not None:
        train_soundscape_dataset = BirdSoundscapeDataset(
            dataframe=train_soundscape_df,
            audio_dir=soundscape_audio_dir,
            label_to_idx=label_to_idx,
            segment_seconds=args.segment_seconds,
            is_train=True,
        )
        valid_soundscape_dataset = BirdSoundscapeDataset(
            dataframe=valid_soundscape_df,
            audio_dir=soundscape_audio_dir,
            label_to_idx=label_to_idx,
            segment_seconds=args.segment_seconds,
            is_train=False,
        )

        repeat_times = max(1, int(args.soundscape_repeat))
        for _ in range(repeat_times):
            train_datasets.append(train_soundscape_dataset)
        valid_datasets.append(valid_soundscape_dataset)
        print(
            f"Enabled soundscape training with repeat={repeat_times}. "
            f"Train samples total={len(train_audio_dataset) + repeat_times * len(train_soundscape_dataset)}; "
            f"Valid samples total={len(valid_audio_dataset) + len(valid_soundscape_dataset)}"
        )

    if train_soundscape_pseudo_df is not None and len(train_soundscape_pseudo_df) > 0:
        train_soundscape_pseudo_dataset = BirdSoundscapeDataset(
            dataframe=train_soundscape_pseudo_df,
            audio_dir=soundscape_audio_dir,
            label_to_idx=label_to_idx,
            segment_seconds=args.segment_seconds,
            is_train=True,
        )
        pseudo_repeat_times = max(1, int(args.soundscape_pseudo_repeat))
        for _ in range(pseudo_repeat_times):
            train_datasets.append(train_soundscape_pseudo_dataset)
        print(
            f"Enabled pseudo soundscape training with repeat={pseudo_repeat_times}. "
            f"Pseudo train samples total={pseudo_repeat_times * len(train_soundscape_pseudo_dataset)}"
        )

    train_dataset = train_datasets[0] if len(train_datasets) == 1 else ConcatDataset(train_datasets)
    valid_dataset = valid_datasets[0] if len(valid_datasets) == 1 else ConcatDataset(valid_datasets)
    print(
        f"Dataset sizes per epoch: train={len(train_dataset)}, valid={len(valid_dataset)}",
        flush=True,
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
    valid_soundscape_loader = None
    if valid_soundscape_dataset is not None:
        valid_soundscape_loader = DataLoader(
            valid_soundscape_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )
    selected_best_metric = args.select_best_on
    if selected_best_metric == "soundscape" and valid_soundscape_loader is None:
        print(
            "[WARN] --select-best-on soundscape requested but no soundscape validation data. "
            "Falling back to combined validation metric.",
            flush=True,
        )
        selected_best_metric = "all"

    frontend = MelSpectrogramFrontend(augment=True).to(device)
    model = BirdCLEFNet(
        num_classes=len(labels), dropout=args.dropout, model_name=args.model_name
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.05
    )
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
        try:
            # PyTorch >=2.6 defaults to weights_only=True, which can reject
            # trusted training checkpoints that contain argparse/pathlib objects.
            resume_ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        except TypeError:
            # Backward compatibility for older PyTorch versions.
            resume_ckpt = torch.load(args.resume, map_location="cpu")

        resume_labels = resume_ckpt.get("labels")
        if resume_labels is not None and list(resume_labels) != list(labels):
            raise ValueError(
                "Label mapping mismatch between resume checkpoint and current dataset."
            )
        resume_args = resume_ckpt.get("args", {})
        if isinstance(resume_args, dict):
            resume_model_name = str(resume_args.get("model_name", "custom_cnn")).lower()
            if resume_model_name != str(args.model_name).lower():
                raise ValueError(
                    "Model architecture mismatch between resume checkpoint and current run: "
                    f"resume={resume_model_name}, current={args.model_name}"
                )

        resume_model_state_dict = resume_ckpt["model_state_dict"]
        try:
            model.load_state_dict(resume_model_state_dict, strict=True)
        except RuntimeError as exc:
            adapted_state_dict, adaptation_msg = adapt_state_dict_prefix_for_model(
                model, resume_model_state_dict
            )
            if adaptation_msg is not None:
                print(adaptation_msg, flush=True)
                model.load_state_dict(adapted_state_dict, strict=True)
            else:
                raise RuntimeError(
                    "Failed to load resume checkpoint model_state_dict. "
                    "Likely architecture mismatch."
                ) from exc
        if "frontend_state_dict" in resume_ckpt:
            frontend.load_state_dict(resume_ckpt["frontend_state_dict"], strict=False)

        resume_best_metric_name = str(resume_ckpt.get("best_metric_name", "all"))
        if resume_best_metric_name != selected_best_metric:
            print(
                f"[WARN] Resume checkpoint best metric is '{resume_best_metric_name}', "
                f"but current run uses '{selected_best_metric}'. Resetting best score tracker.",
                flush=True,
            )
            best_score = -1.0
        else:
            best_score = float(resume_ckpt.get("best_val_auc", best_score))
        start_epoch = int(resume_ckpt.get("epoch", 0)) + 1
        global_step = int(resume_ckpt.get("global_step", 0))
        if args.resume_model_only:
            print(
                "[INFO] --resume-model-only enabled: optimizer/scheduler/scaler states were reset.",
                flush=True,
            )
        else:
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

        bad_params = count_non_finite_params(model) + count_non_finite_params(frontend)
        if bad_params > 0:
            raise RuntimeError(
                f"Resume checkpoint has {bad_params} non-finite parameters. "
                "Try a different checkpoint (for example best_fold*.pth)."
            )
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
                    best_metric_name=selected_best_metric,
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
            bad_params = count_non_finite_params(model) + count_non_finite_params(frontend)
            if bad_params > 0:
                raise RuntimeError(
                    f"Detected {bad_params} non-finite parameters after training epoch {epoch}. "
                    "Model diverged. Lower LR / disable AMP / use --resume-model-only from best checkpoint."
                )

            val_loss, val_score, y_true, y_pred, bad_logits_batches, bad_loss_batches = validate_one_epoch(
                model=model,
                frontend=frontend,
                loader=valid_loader,
                criterion=criterion,
                device=device,
                use_amp=use_amp,
            )
            if bad_logits_batches >= len(valid_loader):
                raise RuntimeError(
                    "Validation logits are non-finite in all batches. "
                    "Training has diverged. Stop and resume from best checkpoint with "
                    "--resume-model-only and a lower learning rate."
                )
            val_soundscape_loss = None
            val_soundscape_score = None
            y_true_soundscape = None
            y_pred_soundscape = None
            if valid_soundscape_loader is not None:
                (
                    val_soundscape_loss,
                    val_soundscape_score,
                    y_true_soundscape,
                    y_pred_soundscape,
                    bad_logits_batches_ss,
                    _bad_loss_batches_ss,
                ) = validate_one_epoch(
                    model=model,
                    frontend=frontend,
                    loader=valid_soundscape_loader,
                    criterion=criterion,
                    device=device,
                    use_amp=use_amp,
                )
                if bad_logits_batches_ss >= len(valid_soundscape_loader):
                    raise RuntimeError(
                        "Soundscape validation logits are non-finite in all batches. "
                        "Training has diverged. Stop and resume from best checkpoint with "
                        "--resume-model-only and a lower learning rate."
                    )

            selected_metric_name = "all"
            selected_score = val_score
            selected_y_true = y_true
            selected_y_pred = y_pred
            if selected_best_metric == "soundscape" and val_soundscape_score is not None:
                selected_metric_name = "soundscape"
                selected_score = float(val_soundscape_score)
                selected_y_true = y_true_soundscape
                selected_y_pred = y_pred_soundscape
            scheduler.step()

            if selected_score > best_score:
                best_score = selected_score
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
                    best_metric_name=selected_best_metric,
                )
                atomic_torch_save(best_checkpoint, best_path)
                np.savez_compressed(
                    args.output_dir / f"oof_fold{args.fold}.npz",
                    y_true=selected_y_true,
                    y_pred=selected_y_pred,
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
                best_metric_name=selected_best_metric,
            )
            atomic_torch_save(last_checkpoint, last_path)

            elapsed = time.time() - epoch_start
            message = (
                f"Epoch {epoch:02d}/{args.epochs} | "
                f"step={global_step} | "
                f"train_loss={train_loss:.4f} | val_all_loss={val_loss:.4f} | "
                f"val_all_auc={val_score:.5f}"
            )
            if val_soundscape_score is not None and val_soundscape_loss is not None:
                message += (
                    f" | val_soundscape_loss={val_soundscape_loss:.4f} "
                    f"| val_soundscape_auc={val_soundscape_score:.5f}"
                )
            message += (
                f" | best_{selected_metric_name}_auc={best_score:.5f} | "
                f"epoch_time={elapsed:.1f}s"
            )
            print(message)
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
                best_metric_name=selected_best_metric,
            )
            atomic_torch_save(interrupt_checkpoint, interrupt_path)
            print(f"[WARN] Interrupt checkpoint saved: {interrupt_path}", flush=True)
        raise

    total_minutes = (time.time() - train_start) / 60.0
    metric_name = selected_best_metric
    print(
        f"Training done. best_{metric_name}_auc={best_score:.5f} | total_time={total_minutes:.1f} min"
    )
    print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()

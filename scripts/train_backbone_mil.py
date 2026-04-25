"""
Train a stronger audio backbone with file-level noisy-or MIL and export window logits npz.

Goal:
  1) Train backbone on weak labels from train.csv + train_audio.
  2) Export window-level logits for train_soundscapes / test_soundscapes.
  3) Plug exported npz into two_pass_ssm_pipeline_v2.py via BC26_EXTERNAL_* envs.
"""

from __future__ import annotations

import argparse
import ast
import math
import random
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

try:
    import torchaudio
except Exception as e:  # pragma: no cover
    raise RuntimeError("torchaudio is required for train_backbone_mil.py") from e

try:
    from torchvision.models import resnet34
except Exception as e:  # pragma: no cover
    raise RuntimeError("torchvision is required for train_backbone_mil.py") from e


SR = 32000
WINDOW_SEC = 5
WINDOW_SAMPLES = SR * WINDOW_SEC
N_WINDOWS = 12
EPS = 1e-7


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train strong backbone MIL and export logits npz.")
    p.add_argument("--train-csv", type=Path, default=Path("dataset/train.csv"))
    p.add_argument("--audio-dir", type=Path, default=Path("dataset/train_audio"))
    p.add_argument("--taxonomy-csv", type=Path, default=Path("dataset/taxonomy.csv"))
    p.add_argument("--sample-submission-csv", type=Path, default=Path("dataset/sample_submission.csv"))
    p.add_argument("--base-dir", type=Path, default=Path("dataset"), help="Contains train_soundscapes / test_soundscapes.")
    p.add_argument("--output-ckpt", type=Path, required=True)
    p.add_argument("--output-train-npz", type=Path, required=True)
    p.add_argument("--output-test-npz", type=Path, required=True)
    p.add_argument("--segments-per-file", type=int, default=8)
    p.add_argument("--batch-files", type=int, default=12)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--n-mels", type=int, default=128)
    p.add_argument("--fmin", type=float, default=40.0)
    p.add_argument("--fmax", type=float, default=14000.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use-gpu", action="store_true")
    p.add_argument("--include-secondary", action="store_true")
    p.add_argument("--max-files", type=int, default=0)
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_secondary_labels(raw: str) -> list[str]:
    if raw is None:
        return []
    s = str(raw).strip()
    if not s or s == "[]" or s.lower() == "nan":
        return []
    try:
        value = ast.literal_eval(s)
        if isinstance(value, (list, tuple)):
            return [str(x).strip() for x in value if str(x).strip()]
    except Exception:
        pass
    parts = [x.strip() for x in s.replace(";", ",").split(",")]
    return [x for x in parts if x]


def load_audio_mono_32k(path: Path) -> np.ndarray:
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if sr != SR:
        raise ValueError(f"Expected sr={SR}, got {sr}: {path}")
    return y


def sample_random_windows(y: np.ndarray, n_seg: int) -> np.ndarray:
    if len(y) <= WINDOW_SAMPLES:
        z = np.pad(y, (0, max(0, WINDOW_SAMPLES - len(y))))
        return np.tile(z[:WINDOW_SAMPLES][None, :], (n_seg, 1)).astype(np.float32, copy=False)
    max_start = len(y) - WINDOW_SAMPLES
    starts = np.random.randint(0, max_start + 1, size=n_seg, dtype=np.int64)
    out = np.empty((n_seg, WINDOW_SAMPLES), dtype=np.float32)
    for i, s in enumerate(starts.tolist()):
        out[i] = y[s : s + WINDOW_SAMPLES]
    return out


def split_fixed_windows_60s(y: np.ndarray) -> np.ndarray:
    total = N_WINDOWS * WINDOW_SAMPLES
    if len(y) < total:
        y = np.pad(y, (0, total - len(y)))
    elif len(y) > total:
        y = y[:total]
    return y.reshape(N_WINDOWS, WINDOW_SAMPLES).astype(np.float32, copy=False)


class AudioBackboneMIL(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.backbone = resnet34(weights=None)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone.fc = nn.Identity()
        self.head = nn.Linear(512, n_classes)

    def forward_window_logits(self, mel: torch.Tensor) -> torch.Tensor:
        # mel: [B, 1, M, T]
        feat = self.backbone(mel)
        return self.head(feat)

    def forward_file_logits(self, mel_segments: torch.Tensor) -> torch.Tensor:
        # mel_segments: [B, K, 1, M, T]
        b, k = mel_segments.shape[:2]
        x = mel_segments.reshape(b * k, *mel_segments.shape[2:])
        logits_seg = self.forward_window_logits(x).reshape(b, k, -1)  # [B,K,C]
        probs_seg = torch.sigmoid(logits_seg).clamp(EPS, 1.0 - EPS)
        file_probs = 1.0 - torch.prod(1.0 - probs_seg, dim=1)
        file_logits = torch.log(file_probs.clamp(EPS, 1.0 - EPS)) - torch.log((1.0 - file_probs).clamp(EPS, 1.0))
        return file_logits


def macro_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    keep = y_true.sum(axis=0) > 0
    if int(keep.sum()) == 0:
        return 0.0
    return float(roc_auc_score(y_true[:, keep], y_score[:, keep], average="macro"))


def make_targets(df: pd.DataFrame, label_set: set[str], include_secondary: bool) -> list[list[str]]:
    out: list[list[str]] = []
    for row in df.itertuples(index=False):
        labels: list[str] = []
        p = str(getattr(row, "primary_label", "")).strip()
        if p in label_set:
            labels.append(p)
        if include_secondary:
            sec_raw = getattr(row, "secondary_labels", "")
            for x in parse_secondary_labels(sec_raw):
                if x in label_set and x not in labels:
                    labels.append(x)
        out.append(labels)
    return out


def build_file_table(train_df: pd.DataFrame, audio_dir: Path, label_to_idx: dict[str, int]) -> list[dict]:
    rows = []
    filenames = train_df["filename"].astype(str).tolist()
    labels_col = train_df["__labels"].tolist()
    for rel, labels in zip(filenames, labels_col):
        rel = str(rel).strip()
        path = audio_dir / rel
        if not path.exists():
            continue
        y = np.zeros(len(label_to_idx), dtype=np.float32)
        for lbl in labels:
            y[label_to_idx[lbl]] = 1.0
        rows.append({"path": path, "y": y})
    return rows


def mel_transform(n_mels: int, fmin: float, fmax: float, device: torch.device):
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=SR,
        n_fft=1024,
        hop_length=320,
        win_length=1024,
        n_mels=n_mels,
        f_min=fmin,
        f_max=fmax,
        power=2.0,
    ).to(device)


def audio_to_logmel(x: torch.Tensor, mel_fn) -> torch.Tensor:
    # x: [N, T]
    mel = mel_fn(x)
    mel = torch.log(mel + 1e-6)
    mel = (mel - mel.mean(dim=(-2, -1), keepdim=True)) / (mel.std(dim=(-2, -1), keepdim=True) + 1e-6)
    return mel.unsqueeze(1)  # [N,1,M,T]


def iter_batches(index: np.ndarray, batch_size: int, shuffle: bool) -> Iterable[np.ndarray]:
    idx = index.copy()
    if shuffle:
        np.random.shuffle(idx)
    for i in range(0, len(idx), batch_size):
        yield idx[i : i + batch_size]


@torch.no_grad()
def infer_soundscape_npz(
    model: AudioBackboneMIL,
    paths: list[Path],
    mel_fn,
    device: torch.device,
    out_npz: Path,
) -> None:
    model.eval()
    all_scores = []
    for p in tqdm(paths, desc=f"Infer {out_npz.name}"):
        y = load_audio_mono_32k(p)
        win = split_fixed_windows_60s(y)  # [12,T]
        x = torch.from_numpy(win).to(device)
        mel = audio_to_logmel(x, mel_fn)
        logits = model.forward_window_logits(mel).detach().cpu().numpy().astype(np.float32)  # [12,C]
        all_scores.append(logits)
    if all_scores:
        arr = np.concatenate(all_scores, axis=0).astype(np.float32)
    else:
        arr = np.zeros((0, 234), dtype=np.float32)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, scores=arr)
    print(f"[OK] Saved logits npz: {out_npz} shape={arr.shape}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    if args.use_gpu and device.type != "cuda":
        raise RuntimeError("--use-gpu is set, but torch.cuda.is_available() is False.")
    print(f"[INFO] device={device}")

    sample_sub = pd.read_csv(args.sample_submission_csv)
    primary_labels = sample_sub.columns[1:].tolist()
    label_to_idx = {c: i for i, c in enumerate(primary_labels)}
    n_classes = len(primary_labels)

    train_df = pd.read_csv(args.train_csv)
    train_df = train_df.copy()
    train_df["__labels"] = make_targets(train_df, set(primary_labels), args.include_secondary)
    train_df = train_df[train_df["__labels"].map(len) > 0].reset_index(drop=True)
    if args.max_files > 0:
        train_df = train_df.iloc[: args.max_files].reset_index(drop=True)
    print(f"[INFO] rows with valid labels={len(train_df)}")

    files = build_file_table(train_df, args.audio_dir, label_to_idx)
    if not files:
        raise RuntimeError("No training files resolved. Check --audio-dir and train.csv filename column.")
    print(f"[INFO] resolved train files={len(files)}")

    all_idx = np.arange(len(files), dtype=np.int32)
    np.random.shuffle(all_idx)
    n_val = max(1, int(len(all_idx) * float(args.val_ratio)))
    va_idx = all_idx[:n_val]
    tr_idx = all_idx[n_val:]
    print(f"[INFO] split train={len(tr_idx)} val={len(va_idx)}")

    y_train = np.stack([files[i]["y"] for i in tr_idx], axis=0).astype(np.float32)
    pos = y_train.sum(axis=0).astype(np.float32)
    neg = float(len(tr_idx)) - pos
    pos_weight = np.clip(neg / (pos + 1.0), 1.0, 50.0)
    pos_weight_t = torch.from_numpy(pos_weight).to(device)

    model = AudioBackboneMIL(n_classes=n_classes).to(device)
    mel_fn = mel_transform(args.n_mels, args.fmin, args.fmax, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_auc = -1.0
    best_state = None
    wait = 0
    t0 = time.time()

    for ep in range(1, args.epochs + 1):
        model.train()
        tr_losses: list[float] = []
        for b in tqdm(list(iter_batches(tr_idx, args.batch_files, shuffle=True)), desc=f"Train ep{ep:02d}", leave=False):
            ys = []
            xs = []
            for bi in b.tolist():
                item = files[bi]
                y = load_audio_mono_32k(item["path"])
                seg = sample_random_windows(y, args.segments_per_file)  # [K,T]
                xs.append(seg)
                ys.append(item["y"])
            x_np = np.stack(xs, axis=0).astype(np.float32)  # [B,K,T]
            y_np = np.stack(ys, axis=0).astype(np.float32)  # [B,C]

            x_t = torch.from_numpy(x_np).to(device)
            y_t = torch.from_numpy(y_np).to(device)
            bsz, kseg = x_t.shape[:2]
            mel = audio_to_logmel(x_t.reshape(bsz * kseg, -1), mel_fn).reshape(bsz, kseg, 1, args.n_mels, -1)
            logits = model.forward_file_logits(mel)
            loss = F.binary_cross_entropy_with_logits(logits, y_t, pos_weight=pos_weight_t)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()
            tr_losses.append(float(loss.item()))

        model.eval()
        va_losses: list[float] = []
        va_probs: list[np.ndarray] = []
        with torch.no_grad():
            for b in tqdm(list(iter_batches(va_idx, args.batch_files, shuffle=False)), desc=f"Val ep{ep:02d}", leave=False):
                ys = []
                xs = []
                for bi in b.tolist():
                    item = files[bi]
                    y = load_audio_mono_32k(item["path"])
                    seg = sample_random_windows(y, args.segments_per_file)
                    xs.append(seg)
                    ys.append(item["y"])
                x_np = np.stack(xs, axis=0).astype(np.float32)
                y_np = np.stack(ys, axis=0).astype(np.float32)

                x_t = torch.from_numpy(x_np).to(device)
                y_t = torch.from_numpy(y_np).to(device)
                bsz, kseg = x_t.shape[:2]
                mel = audio_to_logmel(x_t.reshape(bsz * kseg, -1), mel_fn).reshape(bsz, kseg, 1, args.n_mels, -1)
                logits = model.forward_file_logits(mel)
                loss = F.binary_cross_entropy_with_logits(logits, y_t, pos_weight=pos_weight_t)
                va_losses.append(float(loss.item()))
                va_probs.append(torch.sigmoid(logits).cpu().numpy().astype(np.float32))

        y_va = np.stack([files[i]["y"] for i in va_idx], axis=0).astype(np.float32)
        p_va = np.concatenate(va_probs, axis=0) if va_probs else np.zeros_like(y_va)
        va_auc = macro_auc(y_va, p_va)
        print(
            f"Epoch {ep:02d}/{args.epochs} | train_loss={np.mean(tr_losses):.5f} "
            f"| val_loss={np.mean(va_losses):.5f} | val_auc={va_auc:.6f}"
        )

        if va_auc > best_auc:
            best_auc = va_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                print(f"[INFO] Early stop at epoch {ep}.")
                break

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
    print(f"[INFO] training done in {(time.time()-t0)/60:.1f} min | best_val_auc={best_auc:.6f}")

    args.output_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "n_classes": int(n_classes),
            "n_mels": int(args.n_mels),
            "fmin": float(args.fmin),
            "fmax": float(args.fmax),
            "primary_labels": primary_labels,
            "best_val_auc": float(best_auc),
            "segments_per_file": int(args.segments_per_file),
            "seed": int(args.seed),
        },
        args.output_ckpt,
    )
    print(f"[OK] saved ckpt: {args.output_ckpt}")

    # Export logits for two-pass external blend.
    train_soundscapes = sorted((args.base_dir / "train_soundscapes").glob("*.ogg"))
    test_soundscapes = sorted((args.base_dir / "test_soundscapes").glob("*.ogg"))
    if not test_soundscapes:
        # local dryrun fallback
        test_soundscapes = train_soundscapes[:20]

    infer_soundscape_npz(model, train_soundscapes, mel_fn, device, args.output_train_npz)
    infer_soundscape_npz(model, test_soundscapes, mel_fn, device, args.output_test_npz)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Fine-tune a residual adapter on top of Google Perch v2 features.

This script uses the full `train_audio` + `train.csv` split (weak labels) to
train a residual head:

    adapted_logits = perch_logits + adapter([perch_embedding, perch_logits])

The adapter can be injected into `two_pass_ssm_pipeline_v2.py` through:
  - BC26_PERCH_ADAPTER_CKPT
  - BC26_PERCH_ADAPTER_WEIGHT
"""

from __future__ import annotations

import argparse
import ast
import random
import time
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

try:
    import onnxruntime as ort
except Exception:
    ort = None

try:
    import tensorflow as tf
except Exception:
    tf = None


SR = 32000
WINDOW_SEC = 5
WINDOW_SAMPLES = SR * WINDOW_SEC


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune Perch residual adapter on train_audio.")
    p.add_argument("--train-csv", type=Path, default=Path("dataset/train.csv"))
    p.add_argument("--audio-dir", type=Path, default=Path("dataset/train_audio"))
    p.add_argument("--taxonomy-csv", type=Path, default=Path("dataset/taxonomy.csv"))
    p.add_argument("--sample-submission-csv", type=Path, default=Path("dataset/sample_submission.csv"))
    p.add_argument("--model-dir", type=Path, required=True, help="Perch SavedModel dir (contains assets/labels.csv).")
    p.add_argument("--onnx-path", type=Path, default=None, help="Optional Perch ONNX model path.")
    p.add_argument("--output-ckpt", type=Path, required=True)
    p.add_argument(
        "--feature-cache",
        type=Path,
        default=None,
        help="Optional .npz cache for extracted features. If exists, skip extract.",
    )
    p.add_argument("--segments-per-file", type=int, default=2)
    p.add_argument("--max-files", type=int, default=0, help="0 means all files.")
    p.add_argument("--batch-size", type=int, default=256, help="Perch forward batch size.")
    p.add_argument("--train-batch-size", type=int, default=1024, help="Adapter train batch size.")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-dim", type=int, default=640)
    p.add_argument(
        "--hidden-dim2",
        type=int,
        default=0,
        help="Second hidden dim for mlp3_gated. <=0 means auto: max(128, hidden_dim//2).",
    )
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument(
        "--adapter-arch",
        type=str,
        default="mlp3_gated",
        choices=["linear", "mlp3_gated", "mlp2_legacy"],
        help="Adapter architecture.",
    )
    p.add_argument(
        "--gate-bias",
        type=float,
        default=-2.0,
        help="Initial bias for sigmoid gate in mlp3_gated.",
    )
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--include-secondary", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use-gpu", action="store_true")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_secondary_labels(raw: str) -> List[str]:
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
        raise ValueError(f"Expected sample rate {SR}, got {sr} for {path}")
    return y


def extract_windows(y: np.ndarray, segments_per_file: int) -> List[np.ndarray]:
    if len(y) <= WINDOW_SAMPLES:
        z = np.pad(y, (0, max(0, WINDOW_SAMPLES - len(y))))
        return [z[:WINDOW_SAMPLES]]
    max_start = len(y) - WINDOW_SAMPLES
    n_seg = max(1, int(segments_per_file))
    if n_seg == 1:
        starts = [max_start // 2]
    else:
        starts = np.linspace(0, max_start, n_seg, dtype=np.int64).tolist()
    return [y[s : s + WINDOW_SAMPLES].astype(np.float32, copy=False) for s in starts]


class PerchRunner:
    def __init__(self, model_dir: Path, onnx_path: Path | None, use_gpu: bool):
        self.use_onnx = False
        self.session = None
        self.input_name = ""
        self.out_map = {}
        self.infer_fn = None
        self.model_dir = model_dir

        if onnx_path and str(onnx_path).strip() and onnx_path.exists():
            if ort is None:
                raise RuntimeError("onnxruntime is not available, but --onnx-path was provided.")
            providers = ["CPUExecutionProvider"]
            try:
                avail = ort.get_available_providers()
                if use_gpu:
                    if "CUDAExecutionProvider" not in avail:
                        raise RuntimeError(
                            f"--use-gpu is set, but onnxruntime has no CUDAExecutionProvider. "
                            f"Available providers: {avail}"
                        )
                    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            except Exception:
                raise
            sess_opt = ort.SessionOptions()
            # Avoid noisy pthread_setaffinity_np warnings on some cluster schedulers.
            sess_opt.intra_op_num_threads = 1
            sess_opt.inter_op_num_threads = 1
            try:
                self.session = ort.InferenceSession(
                    str(onnx_path),
                    sess_options=sess_opt,
                    providers=providers,
                )
                self.input_name = self.session.get_inputs()[0].name
                self.out_map = {o.name: i for i, o in enumerate(self.session.get_outputs())}
                self.use_onnx = True
                active_providers = self.session.get_providers()
                if use_gpu and "CUDAExecutionProvider" not in active_providers:
                    raise RuntimeError(
                        "--use-gpu is set, but ONNX session did not activate CUDAExecutionProvider. "
                        f"Active providers: {active_providers}"
                    )
                print(f"[INFO] Using ONNX Perch | providers={active_providers}")
                return
            except Exception as e:
                print(f"[WARN] ONNX session init failed: {e}")
                if tf is None:
                    raise RuntimeError(
                        "ONNX load failed and TensorFlow is unavailable. "
                        "Try a compatible ONNX model/runtime pair."
                    ) from e
                print("[WARN] Falling back to TensorFlow SavedModel Perch.")
        else:
            if tf is None:
                raise RuntimeError("Neither ONNX path provided nor TensorFlow available.")

        birdclassifier = tf.saved_model.load(str(model_dir))
        self.infer_fn = birdclassifier.signatures["serving_default"]
        self.use_onnx = False
        print("[INFO] Using TensorFlow SavedModel Perch")

    def infer(self, batch_audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.use_onnx:
            outs = self.session.run(None, {self.input_name: batch_audio})
            logits = outs[self.out_map["label"]].astype(np.float32)
            emb = outs[self.out_map["embedding"]].astype(np.float32)
        else:
            out = self.infer_fn(inputs=tf.convert_to_tensor(batch_audio))
            logits = out["label"].numpy().astype(np.float32)
            emb = out["embedding"].numpy().astype(np.float32)
        return logits, emb


class PerchAdapterHeadLegacy(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PerchAdapterHeadLinear(nn.Module):
    """Single-layer linear residual adapter."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

        # Conservative init: start near identity-over-base (tiny delta), then learn.
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class PerchAdapterHeadGated(nn.Module):
    """3-layer gated residual adapter.

    delta = sigmoid(gate(h2)) * proj(h2)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        hidden_dim2: int,
        output_dim: int,
        dropout: float,
        gate_bias: float = -2.0,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim2)
        self.norm2 = nn.LayerNorm(hidden_dim2)
        self.delta = nn.Linear(hidden_dim2, output_dim)
        self.gate = nn.Linear(hidden_dim2, output_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        # Start from a conservative adapter and let training learn to amplify.
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, float(gate_bias))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dropout(self.act(self.norm1(self.fc1(x))))
        h = self.dropout(self.act(self.norm2(self.fc2(h))))
        g = torch.sigmoid(self.gate(h))
        return g * self.delta(h)


def build_adapter_model(input_dim: int, output_dim: int, args: argparse.Namespace) -> tuple[nn.Module, dict]:
    arch = str(args.adapter_arch).strip().lower()
    if arch == "linear":
        model = PerchAdapterHeadLinear(
            input_dim=input_dim,
            output_dim=output_dim,
        )
        meta = {
            "adapter_arch": "linear",
            "hidden_dim": 0,
            "hidden_dim2": 0,
            "gate_bias": 0.0,
            "dropout": 0.0,
        }
        return model, meta

    if arch == "mlp3_gated":
        hidden_dim2 = int(args.hidden_dim2) if int(args.hidden_dim2) > 0 else max(128, int(args.hidden_dim) // 2)
        model = PerchAdapterHeadGated(
            input_dim=input_dim,
            hidden_dim=int(args.hidden_dim),
            hidden_dim2=hidden_dim2,
            output_dim=output_dim,
            dropout=float(args.dropout),
            gate_bias=float(args.gate_bias),
        )
        meta = {
            "adapter_arch": "mlp3_gated",
            "hidden_dim": int(args.hidden_dim),
            "hidden_dim2": int(hidden_dim2),
            "gate_bias": float(args.gate_bias),
            "dropout": float(args.dropout),
        }
        return model, meta

    model = PerchAdapterHeadLegacy(
        input_dim=input_dim,
        hidden_dim=int(args.hidden_dim),
        output_dim=output_dim,
        dropout=float(args.dropout),
    )
    meta = {
        "adapter_arch": "mlp2_legacy",
        "hidden_dim": int(args.hidden_dim),
        "hidden_dim2": 0,
        "gate_bias": 0.0,
        "dropout": float(args.dropout),
    }
    return model, meta


def macro_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    keep = y_true.sum(axis=0) > 0
    if keep.sum() == 0:
        return 0.0
    return float(roc_auc_score(y_true[:, keep], y_score[:, keep], average="macro"))


def build_label_mapping(
    sample_submission_csv: Path,
    taxonomy_csv: Path,
    model_dir: Path,
) -> tuple[List[str], np.ndarray, np.ndarray]:
    sample_sub = pd.read_csv(sample_submission_csv)
    taxonomy = pd.read_csv(taxonomy_csv)
    primary_labels = sample_sub.columns[1:].tolist()
    label_to_idx = {c: i for i, c in enumerate(primary_labels)}

    bc_labels = pd.read_csv(model_dir / "assets" / "labels.csv").reset_index()
    name_col = "inat2024_fsd50k" if "inat2024_fsd50k" in bc_labels.columns else "scientific_name"
    if name_col not in bc_labels.columns:
        raise RuntimeError("Perch labels.csv missing expected scientific-name column.")
    bc_labels = bc_labels.rename(columns={"index": "bc_index", name_col: "scientific_name"})

    mapping = taxonomy.merge(bc_labels[["scientific_name", "bc_index"]], on="scientific_name", how="left")
    mapping["bc_index"] = mapping["bc_index"].fillna(-1).astype(int)
    lbl2bc = mapping.set_index("primary_label")["bc_index"]

    bc_indices = np.array([int(lbl2bc.get(lbl, -1)) for lbl in primary_labels], dtype=np.int32)
    mapped_pos = np.where(bc_indices >= 0)[0].astype(np.int32)
    mapped_bc_idx = bc_indices[mapped_pos].astype(np.int32)
    print(f"[INFO] Label mapping: matched {len(mapped_pos)}/{len(primary_labels)} competition classes to Perch.")
    return primary_labels, mapped_pos, mapped_bc_idx


def make_targets(train_df: pd.DataFrame, valid_labels: set[str], include_secondary: bool) -> List[List[str]]:
    all_labels = []
    for row in train_df.itertuples(index=False):
        labels = set()
        p = str(getattr(row, "primary_label", "")).strip()
        if p in valid_labels:
            labels.add(p)
        if include_secondary:
            sec_raw = getattr(row, "secondary_labels", "")
            for x in parse_secondary_labels(sec_raw):
                if x in valid_labels:
                    labels.add(x)
        all_labels.append(sorted(labels))
    return all_labels


def train_adapter(
    X: np.ndarray,
    base_logits: np.ndarray,
    Y: np.ndarray,
    groups: np.ndarray,
    args: argparse.Namespace,
    n_classes: int,
) -> tuple[nn.Module, float, dict]:
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    if args.use_gpu and device.type != "cuda":
        raise RuntimeError("--use-gpu is set, but torch.cuda.is_available() is False.")
    print(f"[INFO] Adapter train device: {device}")

    unique_files = np.unique(groups)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(unique_files)
    n_val_files = max(1, int(len(unique_files) * args.val_ratio))
    val_file_set = set(unique_files[:n_val_files].tolist())
    val_mask = np.array([g in val_file_set for g in groups], dtype=bool)
    tr_idx = np.where(~val_mask)[0]
    va_idx = np.where(val_mask)[0]
    print(f"[INFO] Split: train_windows={len(tr_idx)}, val_windows={len(va_idx)}, files={len(unique_files)}")

    y_tr = Y[tr_idx]
    pos = y_tr.sum(axis=0).astype(np.float32)
    neg = float(len(tr_idx)) - pos
    pos_weight = np.clip(neg / (pos + 1.0), 1.0, 50.0)
    pos_weight_t = torch.from_numpy(pos_weight).to(device)

    model, adapter_meta = build_adapter_model(
        input_dim=int(X.shape[1]),
        output_dim=n_classes,
        args=args,
    )
    model = model.to(device)
    print(
        "[INFO] Adapter arch="
        f"{adapter_meta['adapter_arch']} "
        f"(hidden={adapter_meta['hidden_dim']}, "
        f"hidden2={adapter_meta['hidden_dim2']}, "
        f"dropout={adapter_meta['dropout']}, "
        f"gate_bias={adapter_meta['gate_bias']})"
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_auc = -1.0
    best_state = None
    wait = 0

    def _iter_batches(index: np.ndarray, batch_size: int, shuffle: bool) -> Iterable[np.ndarray]:
        idx = index.copy()
        if shuffle:
            rng.shuffle(idx)
        for i in range(0, len(idx), batch_size):
            yield idx[i : i + batch_size]

    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        tr_losses = []
        for b in _iter_batches(tr_idx, args.train_batch_size, shuffle=True):
            x = torch.from_numpy(X[b]).to(device)
            base = torch.from_numpy(base_logits[b]).to(device)
            y = torch.from_numpy(Y[b]).to(device)
            delta = model(x)
            logits = base + delta
            loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight_t)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_losses.append(float(loss.item()))

        model.eval()
        va_losses = []
        va_preds = []
        with torch.no_grad():
            for b in _iter_batches(va_idx, args.train_batch_size, shuffle=False):
                x = torch.from_numpy(X[b]).to(device)
                base = torch.from_numpy(base_logits[b]).to(device)
                y = torch.from_numpy(Y[b]).to(device)
                delta = model(x)
                logits = base + delta
                loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight_t)
                va_losses.append(float(loss.item()))
                va_preds.append(torch.sigmoid(logits).cpu().numpy())
        va_pred = np.concatenate(va_preds, axis=0) if va_preds else np.zeros((0, n_classes), dtype=np.float32)
        va_auc = macro_auc(Y[va_idx], va_pred) if len(va_idx) > 0 else 0.0
        print(
            f"Epoch {ep:02d}/{args.epochs} | "
            f"train_loss={np.mean(tr_losses):.5f} | val_loss={np.mean(va_losses):.5f} | val_auc={va_auc:.6f}"
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
    print(f"[INFO] Adapter training done in {(time.time()-t0)/60:.1f} min | best_val_auc={best_auc:.6f}")
    return model, best_auc, adapter_meta


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if not args.train_csv.exists():
        raise FileNotFoundError(args.train_csv)
    if not args.audio_dir.exists():
        raise FileNotFoundError(args.audio_dir)
    if not args.model_dir.exists():
        raise FileNotFoundError(args.model_dir)

    primary_labels, mapped_pos, mapped_bc_idx = build_label_mapping(
        args.sample_submission_csv,
        args.taxonomy_csv,
        args.model_dir,
    )
    n_classes = len(primary_labels)
    label_to_idx = {c: i for i, c in enumerate(primary_labels)}

    train_df = pd.read_csv(args.train_csv)
    targets_label_names = make_targets(train_df, set(primary_labels), args.include_secondary)
    train_df = train_df.copy()
    train_df["__labels"] = targets_label_names
    train_df = train_df[train_df["__labels"].map(len) > 0].reset_index(drop=True)
    if args.max_files > 0:
        train_df = train_df.iloc[: args.max_files].reset_index(drop=True)

    print(f"[INFO] train rows with valid labels: {len(train_df)}")

    onnx_path = None
    if args.onnx_path is not None:
        raw = str(args.onnx_path).strip()
        if raw and raw != ".":
            onnx_path = args.onnx_path
    cache_loaded = False
    if args.feature_cache is not None and args.feature_cache.exists():
        arr = np.load(args.feature_cache)
        X = arr["X"].astype(np.float32, copy=False)
        B = arr["B"].astype(np.float32, copy=False)
        Y = arr["Y"].astype(np.float32, copy=False)
        G = arr["G"].astype(np.int32, copy=False)
        if Y.shape[1] != n_classes:
            raise RuntimeError(
                f"Feature cache label dim {Y.shape[1]} does not match current n_classes {n_classes}."
            )
        cache_loaded = True
        print(
            f"[INFO] Loaded feature cache: {args.feature_cache} | "
            f"windows={len(X)} files={len(np.unique(G))} feature_dim={X.shape[1]}"
        )

    if not cache_loaded:
        runner = PerchRunner(args.model_dir, onnx_path, args.use_gpu)

        pending_audio: List[np.ndarray] = []
        pending_targets: List[np.ndarray] = []
        pending_groups: List[int] = []

        feats_all = []
        base_all = []
        y_all = []
        groups_all: List[int] = []

        def infer_with_auto_chunk(batch_audio: np.ndarray, min_chunk: int = 4) -> tuple[np.ndarray, np.ndarray]:
            """Run Perch inference with OOM-aware chunk fallback."""
            try:
                return runner.infer(batch_audio)
            except Exception as e:
                msg = str(e).lower()
                oom_like = (
                    "failed to allocate memory" in msg
                    or "cuda out of memory" in msg
                    or "bfc_arena" in msg
                )
                if (not oom_like) or len(batch_audio) <= min_chunk:
                    raise

                n = len(batch_audio)
                mid = n // 2
                print(
                    f"[WARN] OOM at batch={n}, retry with chunks {mid}+{n-mid}.",
                    flush=True,
                )
                l_logit, l_emb = infer_with_auto_chunk(batch_audio[:mid], min_chunk=min_chunk)
                r_logit, r_emb = infer_with_auto_chunk(batch_audio[mid:], min_chunk=min_chunk)
                return (
                    np.concatenate([l_logit, r_logit], axis=0),
                    np.concatenate([l_emb, r_emb], axis=0),
                )

        def flush_batch() -> None:
            if not pending_audio:
                return
            x = np.stack(pending_audio, axis=0).astype(np.float32, copy=False)
            logits_raw, emb = infer_with_auto_chunk(x)
            mapped_scores = np.zeros((len(x), n_classes), dtype=np.float32)
            if len(mapped_pos) > 0:
                mapped_scores[:, mapped_pos] = logits_raw[:, mapped_bc_idx]
            feat = np.concatenate([emb, mapped_scores], axis=1).astype(np.float32, copy=False)
            feats_all.append(feat)
            base_all.append(mapped_scores)
            y_all.append(np.stack(pending_targets, axis=0).astype(np.float32, copy=False))
            groups_all.extend(pending_groups)
            pending_audio.clear()
            pending_targets.clear()
            pending_groups.clear()

        skipped = 0
        total_windows = 0
        file_list = train_df["filename"].astype(str).tolist()
        label_list = train_df["__labels"].tolist()
        pbar = tqdm(
            enumerate(zip(file_list, label_list)),
            total=len(file_list),
            desc="Extract",
        )
        for file_idx, (rel, labels_for_row) in pbar:
            path = args.audio_dir / rel
            if not path.exists():
                skipped += 1
                continue
            try:
                y = load_audio_mono_32k(path)
                windows = extract_windows(y, args.segments_per_file)
            except Exception:
                skipped += 1
                continue
            target = np.zeros(n_classes, dtype=np.float32)
            for lbl in labels_for_row:
                target[label_to_idx[lbl]] = 1.0
            for w in windows:
                pending_audio.append(w)
                pending_targets.append(target)
                pending_groups.append(file_idx)
                total_windows += 1
                if len(pending_audio) >= args.batch_size:
                    flush_batch()
            pbar.set_postfix(windows=total_windows, skipped=skipped)
        flush_batch()

        if not feats_all:
            raise RuntimeError("No features extracted. Check dataset paths / sample rate.")

        X = np.concatenate(feats_all, axis=0)
        B = np.concatenate(base_all, axis=0)
        Y = np.concatenate(y_all, axis=0)
        G = np.asarray(groups_all, dtype=np.int32)

        print(
            f"[INFO] Extracted windows={len(X)} | files={len(np.unique(G))} | "
            f"feature_dim={X.shape[1]} | labels={Y.shape[1]} | skipped_files={skipped}"
        )
        if args.feature_cache is not None:
            args.feature_cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                args.feature_cache,
                X=X.astype(np.float32),
                B=B.astype(np.float32),
                Y=Y.astype(np.float32),
                G=G.astype(np.int32),
            )
            print(f"[OK] Saved feature cache: {args.feature_cache}")

    model, best_auc, adapter_meta = train_adapter(X, B, Y, G, args, n_classes=n_classes)

    args.output_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "adapter_state_dict": model.state_dict(),
            "adapter_arch": str(adapter_meta.get("adapter_arch", "mlp2_legacy")),
            "input_dim": int(X.shape[1]),
            "hidden_dim": int(adapter_meta.get("hidden_dim", args.hidden_dim)),
            "hidden_dim2": int(adapter_meta.get("hidden_dim2", 0)),
            "gate_bias": float(adapter_meta.get("gate_bias", 0.0)),
            "output_dim": int(n_classes),
            "dropout": float(args.dropout),
            "primary_labels": primary_labels,
            "best_val_auc": float(best_auc),
            "segments_per_file": int(args.segments_per_file),
            "train_windows": int(len(X)),
            "train_files": int(len(np.unique(G))),
            "seed": int(args.seed),
        },
        args.output_ckpt,
    )
    print(f"[OK] Saved adapter ckpt: {args.output_ckpt}")


if __name__ == "__main__":
    main()

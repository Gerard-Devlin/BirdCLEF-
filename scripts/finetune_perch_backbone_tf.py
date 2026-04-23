#!/usr/bin/env python
"""Experimental Perch v2 backbone fine-tuning (TensorFlow SavedModel).

This script is intentionally separated from the adapter flow. It attempts
true backbone updates by optimizing selected variables inside the loaded
SavedModel function graph.

Workflow:
1) Load Perch TF SavedModel (`--model-dir`)
2) Build weak-label training targets from `train.csv`
3) Stream audio windows from `train_audio`
4) Fine-tune selected backbone variables on mapped competition classes
5) Export a new tuned SavedModel (`--output-model-dir`)

Notes:
- This is a high-risk/high-reward path. Keep experiments isolated.
- If gradients are all None, your SavedModel graph is effectively frozen for
  training under this runtime; the script will fail loudly.
"""

from __future__ import annotations

import argparse
import ast
import os
import random
import re
import time
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.metrics import roc_auc_score

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

try:
    import tensorflow as tf
except Exception:
    tf = None


SR = 32000
WINDOW_SEC = 5
WINDOW_SAMPLES = SR * WINDOW_SEC


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experimental Perch backbone fine-tune on train_audio.")
    p.add_argument("--train-csv", type=Path, default=Path("dataset/train.csv"))
    p.add_argument("--audio-dir", type=Path, default=Path("dataset/train_audio"))
    p.add_argument("--taxonomy-csv", type=Path, default=Path("dataset/taxonomy.csv"))
    p.add_argument("--sample-submission-csv", type=Path, default=Path("dataset/sample_submission.csv"))
    p.add_argument("--model-dir", type=Path, required=True, help="Input Perch SavedModel directory.")
    p.add_argument("--output-model-dir", type=Path, required=True, help="Output tuned SavedModel directory.")

    p.add_argument("--segments-per-file", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=32, help="Audio windows per optimization step.")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--val-max-batches", type=int, default=200, help="0 means full validation pass.")
    p.add_argument("--max-files", type=int, default=0, help="0 means all train rows.")
    p.add_argument("--include-secondary", action="store_true")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument(
        "--unfreeze-patterns",
        type=str,
        default="heads_,classifier,logit,dense,proj,attn,ffn",
        help="Comma-separated substrings to select trainable variables by name.",
    )
    p.add_argument(
        "--unfreeze-last-n",
        type=int,
        default=0,
        help="Additionally unfreeze the last N variables by order in SavedModel.",
    )
    p.add_argument("--use-gpu", action="store_true")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if tf is not None:
        try:
            tf.random.set_seed(seed)
        except Exception:
            pass


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


def extract_windows(y: np.ndarray, segments_per_file: int, rng: np.random.Generator) -> List[np.ndarray]:
    if len(y) <= WINDOW_SAMPLES:
        z = np.pad(y, (0, max(0, WINDOW_SAMPLES - len(y))))
        return [z[:WINDOW_SAMPLES]]
    max_start = len(y) - WINDOW_SAMPLES
    n_seg = max(1, int(segments_per_file))
    if n_seg == 1:
        starts = [rng.integers(0, max_start + 1)]
    else:
        # Random multi-crop; sort for deterministic read order.
        starts = sorted(rng.integers(0, max_start + 1, size=n_seg).tolist())
    return [y[s : s + WINDOW_SAMPLES].astype(np.float32, copy=False) for s in starts]


def macro_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    keep = y_true.sum(axis=0) > 0
    if int(keep.sum()) == 0:
        return 0.0
    return float(roc_auc_score(y_true[:, keep], y_score[:, keep], average="macro"))


def build_label_mapping(
    sample_submission_csv: Path,
    taxonomy_csv: Path,
    model_dir: Path,
) -> tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    sample_sub = pd.read_csv(sample_submission_csv)
    taxonomy = pd.read_csv(taxonomy_csv)
    primary_labels = sample_sub.columns[1:].tolist()

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

    comp_to_mapped = np.full(len(primary_labels), -1, dtype=np.int32)
    comp_to_mapped[mapped_pos] = np.arange(len(mapped_pos), dtype=np.int32)
    print(f"[INFO] Label mapping: matched {len(mapped_pos)}/{len(primary_labels)} competition classes to Perch.")
    return primary_labels, mapped_pos, mapped_bc_idx, comp_to_mapped


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


class PerchBackboneTuner:
    def __init__(
        self,
        model_dir: Path,
        mapped_bc_idx: np.ndarray,
        unfreeze_tokens: Sequence[str],
        unfreeze_last_n: int,
        use_gpu: bool,
    ) -> None:
        if tf is None:
            raise RuntimeError("TensorFlow is not available in this environment.")

        if not use_gpu:
            try:
                tf.config.set_visible_devices([], "GPU")
            except Exception:
                pass

        self.model = tf.saved_model.load(str(model_dir))
        self.infer_fn = self.model.signatures["serving_default"]
        self.mapped_bc_idx_tf = tf.constant(mapped_bc_idx, dtype=tf.int32)

        vars_all = list(getattr(self.model, "variables", []))
        if not vars_all:
            raise RuntimeError("No variables found in loaded SavedModel.")

        selected = []
        tokens = [t.strip().lower() for t in unfreeze_tokens if t.strip()]
        for v in vars_all:
            vn = v.name.lower()
            if any(tok in vn for tok in tokens):
                selected.append(v)
        if unfreeze_last_n > 0:
            selected.extend(vars_all[-int(unfreeze_last_n) :])

        # Keep order, deduplicate by ref.
        seen = set()
        train_vars = []
        for v in selected:
            key = id(v)
            if key not in seen:
                seen.add(key)
                train_vars.append(v)

        if not train_vars:
            raise RuntimeError(
                "No trainable variables selected. Adjust --unfreeze-patterns or --unfreeze-last-n."
            )

        self.train_vars = train_vars
        print(f"[INFO] SavedModel vars total={len(vars_all)} | selected train_vars={len(self.train_vars)}")
        preview = [v.name for v in self.train_vars[:12]]
        for n in preview:
            print(f"  [train] {n}")
        if len(self.train_vars) > len(preview):
            print(f"  ... ({len(self.train_vars)-len(preview)} more)")

    def forward_mapped_logits(self, x_np: np.ndarray) -> tf.Tensor:
        outs = self.infer_fn(inputs=tf.convert_to_tensor(x_np))
        logits_raw = outs["label"]
        logits_mapped = tf.gather(logits_raw, self.mapped_bc_idx_tf, axis=1)
        return logits_mapped


def iter_file_batches(
    file_indices: np.ndarray,
    file_list: List[str],
    target_list: List[np.ndarray],
    audio_dir: Path,
    segments_per_file: int,
    batch_size: int,
    rng: np.random.Generator,
    shuffle: bool,
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    order = file_indices.copy()
    if shuffle:
        rng.shuffle(order)

    xb = []
    yb = []
    for fi in order:
        rel = file_list[int(fi)]
        path = audio_dir / rel
        if not path.exists():
            continue
        try:
            y = load_audio_mono_32k(path)
            windows = extract_windows(y, segments_per_file, rng)
        except Exception:
            continue
        target = target_list[int(fi)]
        for w in windows:
            xb.append(w)
            yb.append(target)
            if len(xb) >= batch_size:
                yield np.stack(xb, axis=0).astype(np.float32), np.stack(yb, axis=0).astype(np.float32)
                xb.clear()
                yb.clear()
    if xb:
        yield np.stack(xb, axis=0).astype(np.float32), np.stack(yb, axis=0).astype(np.float32)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if tf is None:
        raise RuntimeError("TensorFlow import failed. Install TensorFlow before backbone fine-tuning.")
    if not args.train_csv.exists():
        raise FileNotFoundError(args.train_csv)
    if not args.audio_dir.exists():
        raise FileNotFoundError(args.audio_dir)
    if not args.model_dir.exists():
        raise FileNotFoundError(args.model_dir)

    primary_labels, mapped_pos, mapped_bc_idx, comp_to_mapped = build_label_mapping(
        args.sample_submission_csv,
        args.taxonomy_csv,
        args.model_dir,
    )
    n_mapped = len(mapped_pos)
    if n_mapped == 0:
        raise RuntimeError("No mapped classes found. Cannot train backbone.")

    train_df = pd.read_csv(args.train_csv)
    targets_label_names = make_targets(train_df, set(primary_labels), args.include_secondary)
    train_df = train_df.copy()
    train_df["__labels"] = targets_label_names
    train_df = train_df[train_df["__labels"].map(len) > 0].reset_index(drop=True)
    if args.max_files > 0:
        train_df = train_df.iloc[: args.max_files].reset_index(drop=True)
    print(f"[INFO] train rows with valid labels: {len(train_df)}")

    file_list = train_df["filename"].astype(str).tolist()
    target_list = []
    skipped_unmapped = 0
    for labels in train_df["__labels"].tolist():
        y = np.zeros(n_mapped, dtype=np.float32)
        touched = False
        for lbl in labels:
            comp_idx = primary_labels.index(lbl)
            mapped_idx = int(comp_to_mapped[comp_idx])
            if mapped_idx >= 0:
                y[mapped_idx] = 1.0
                touched = True
        if not touched:
            skipped_unmapped += 1
        target_list.append(y)
    if skipped_unmapped > 0:
        print(f"[INFO] rows with only unmapped labels (still kept as all-zero targets): {skipped_unmapped}")

    # File-level split to avoid leakage.
    file_indices = np.arange(len(file_list), dtype=np.int32)
    uniq = file_indices.copy()
    rng_split = np.random.default_rng(args.seed)
    rng_split.shuffle(uniq)
    n_val = max(1, int(len(uniq) * args.val_ratio))
    val_idx = np.sort(uniq[:n_val])
    tr_idx = np.sort(uniq[n_val:])
    print(f"[INFO] Split: train_files={len(tr_idx)}, val_files={len(val_idx)}")

    # Build class-wise pos_weight from a quick pass over train targets.
    y_tr = np.stack([target_list[i] for i in tr_idx], axis=0).astype(np.float32)
    pos = y_tr.sum(axis=0)
    neg = float(len(y_tr)) - pos
    pos_weight = np.clip(neg / (pos + 1.0), 1.0, 50.0).astype(np.float32)
    pos_weight_tf = tf.constant(pos_weight, dtype=tf.float32)

    unfreeze_tokens = [x for x in args.unfreeze_patterns.split(",") if x.strip()]
    tuner = PerchBackboneTuner(
        model_dir=args.model_dir,
        mapped_bc_idx=mapped_bc_idx,
        unfreeze_tokens=unfreeze_tokens,
        unfreeze_last_n=args.unfreeze_last_n,
        use_gpu=args.use_gpu,
    )

    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
    )

    best_auc = -1.0
    best_vals = [v.numpy().copy() for v in tuner.train_vars]
    wait = 0

    @tf.function(reduce_retracing=True)
    def train_step(x_batch, y_batch):
        with tf.GradientTape() as tape:
            for v in tuner.train_vars:
                tape.watch(v)
            logits = tuner.forward_mapped_logits(x_batch)
            per_elem = tf.nn.weighted_cross_entropy_with_logits(
                labels=y_batch,
                logits=logits,
                pos_weight=pos_weight_tf,
            )
            loss = tf.reduce_mean(per_elem)
        grads = tape.gradient(loss, tuner.train_vars)
        grads_vars = [(g, v) for g, v in zip(grads, tuner.train_vars) if g is not None]
        if grads_vars:
            if args.grad_clip_norm > 0:
                clipped = []
                for g, v in grads_vars:
                    clipped.append((tf.clip_by_norm(g, args.grad_clip_norm), v))
                grads_vars = clipped
            optimizer.apply_gradients(grads_vars)
        return loss, tf.constant(len(grads_vars), dtype=tf.int32)

    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        rng_ep = np.random.default_rng(args.seed + ep * 17)
        train_losses = []
        grad_steps = 0
        non_none_steps = 0

        for xb, yb in iter_file_batches(
            tr_idx,
            file_list,
            target_list,
            args.audio_dir,
            args.segments_per_file,
            args.batch_size,
            rng_ep,
            shuffle=True,
        ):
            loss, n_grad = train_step(xb, yb)
            train_losses.append(float(loss.numpy()))
            grad_steps += 1
            if int(n_grad.numpy()) > 0:
                non_none_steps += 1

        if grad_steps == 0:
            raise RuntimeError("No train batches produced; check dataset paths and audio files.")
        if non_none_steps == 0:
            raise RuntimeError(
                "All gradient steps produced None gradients. This SavedModel may be non-trainable in this setup."
            )

        # Validation
        val_losses = []
        val_true = []
        val_pred = []
        rng_val = np.random.default_rng(args.seed + 999 + ep)
        for b_idx, (xb, yb) in enumerate(
            iter_file_batches(
                val_idx,
                file_list,
                target_list,
                args.audio_dir,
                args.segments_per_file,
                args.batch_size,
                rng_val,
                shuffle=False,
            ),
            1,
        ):
            logits = tuner.forward_mapped_logits(xb)
            per_elem = tf.nn.weighted_cross_entropy_with_logits(
                labels=tf.convert_to_tensor(yb),
                logits=logits,
                pos_weight=pos_weight_tf,
            )
            loss = tf.reduce_mean(per_elem)
            probs = tf.sigmoid(logits).numpy()
            val_losses.append(float(loss.numpy()))
            val_true.append(yb)
            val_pred.append(probs)
            if args.val_max_batches > 0 and b_idx >= args.val_max_batches:
                break

        yv = np.concatenate(val_true, axis=0) if val_true else np.zeros((0, n_mapped), dtype=np.float32)
        pv = np.concatenate(val_pred, axis=0) if val_pred else np.zeros((0, n_mapped), dtype=np.float32)
        val_auc = macro_auc(yv, pv) if len(yv) > 0 else 0.0
        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses)) if val_losses else 0.0

        print(
            f"Epoch {ep:02d}/{args.epochs} | train_loss={train_loss:.5f} "
            f"| val_loss={val_loss:.5f} | val_auc={val_auc:.6f} "
            f"| grad_steps(non_none/all)={non_none_steps}/{grad_steps}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            best_vals = [v.numpy().copy() for v in tuner.train_vars]
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                print(f"[INFO] Early stop at epoch {ep}.")
                break

    # Restore best values before export.
    for v, best in zip(tuner.train_vars, best_vals):
        v.assign(best)

    args.output_model_dir.mkdir(parents=True, exist_ok=True)
    tf.saved_model.save(tuner.model, str(args.output_model_dir), signatures=tuner.model.signatures)
    print(f"[OK] Saved tuned model: {args.output_model_dir}")
    print(f"[INFO] Done in {(time.time()-t0)/60:.1f} min | best_val_auc={best_auc:.6f}")


if __name__ == "__main__":
    main()


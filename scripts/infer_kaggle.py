from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from birdclef_plus import BirdCLEFNet, MelSpectrogramFrontend
from birdclef_plus.audio import TARGET_SAMPLE_RATE, crop_or_pad, load_audio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BirdCLEF+ 2026 Kaggle inference")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--competition-dir",
        type=Path,
        default=Path("/kaggle/input/birdclef-2026"),
        help="Directory containing sample_submission.csv and test_soundscapes/",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--segment-seconds", type=float, default=5.0)
    parser.add_argument("--tta-flip", action="store_true", help="Average with time-flip TTA")
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    parser.add_argument(
        "--strict-missing",
        action="store_true",
        help="Fail if a soundscape file referenced by row_id is missing.",
    )
    return parser.parse_args()


def parse_row_id(row_id: str) -> Tuple[str, int]:
    soundscape_id, end_second_token = row_id.rsplit("_", 1)
    try:
        end_second = int(end_second_token)
    except ValueError:
        end_second = int(float(end_second_token))
    return soundscape_id, end_second


def build_soundscape_index(test_dir: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for file_path in test_dir.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in {".ogg", ".wav", ".flac", ".mp3"}:
            index.setdefault(file_path.stem, file_path)
    return index


def resolve_soundscape_path(
    test_dir: Path, soundscape_id: str, soundscape_index: Optional[Dict[str, Path]] = None
) -> Path:
    if soundscape_index and soundscape_id in soundscape_index:
        return soundscape_index[soundscape_id]

    # Fuzzy stem match: some runtimes can expose soundscape files with
    # slight naming differences (prefix/suffix variants).
    if soundscape_index:
        fuzzy_matches = []
        for stem, path in soundscape_index.items():
            if stem.startswith(soundscape_id) or soundscape_id.startswith(stem):
                fuzzy_matches.append(path)
        if len(fuzzy_matches) == 1:
            return fuzzy_matches[0]
        if len(fuzzy_matches) > 1:
            fuzzy_matches.sort(key=lambda p: abs(len(p.stem) - len(soundscape_id)))
            return fuzzy_matches[0]

    for ext in (".ogg", ".wav", ".flac", ".mp3"):
        candidate = test_dir / f"{soundscape_id}{ext}"
        if candidate.exists():
            return candidate
    # Some competition inputs can nest files by folders.
    for ext in (".ogg", ".wav", ".flac", ".mp3"):
        matches = list(test_dir.rglob(f"{soundscape_id}{ext}"))
        if matches:
            return matches[0]
    matches = list(test_dir.glob(f"{soundscape_id}.*"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Soundscape file not found for row_id prefix: {soundscape_id}")


@torch.no_grad()
def predict_file(
    model: torch.nn.Module,
    frontend: torch.nn.Module,
    waveform: torch.Tensor,
    end_seconds: List[int],
    batch_size: int,
    segment_seconds: float,
    tta_flip: bool,
    device: torch.device,
) -> np.ndarray:
    segment_samples = int(segment_seconds * TARGET_SAMPLE_RATE)
    segments = []
    for end_second in end_seconds:
        start_second = max(0.0, float(end_second) - segment_seconds)
        start_sample = int(start_second * TARGET_SAMPLE_RATE)
        stop_sample = start_sample + segment_samples
        segment = waveform[start_sample:stop_sample]
        segment = crop_or_pad(segment, segment_samples, random_crop=False)
        segments.append(segment)

    segments_tensor = torch.stack(segments, dim=0)
    all_probs = []
    for i in range(0, len(segments_tensor), batch_size):
        batch = segments_tensor[i : i + batch_size].to(device)
        features = frontend(batch)
        logits = model(features)
        probs = torch.sigmoid(logits)

        if tta_flip:
            flipped = torch.flip(batch, dims=[1])
            features_flip = frontend(flipped)
            logits_flip = model(features_flip)
            probs = (probs + torch.sigmoid(logits_flip)) * 0.5

        all_probs.append(probs.cpu().numpy())

    return np.concatenate(all_probs, axis=0)


def main() -> None:
    args = parse_args()
    start_time = time.time()

    sample_submission_path = args.competition_dir / "sample_submission.csv"
    test_soundscapes_dir = args.competition_dir / "test_soundscapes"
    if not sample_submission_path.exists():
        raise FileNotFoundError(f"Missing {sample_submission_path}")
    if not test_soundscapes_dir.exists():
        raise FileNotFoundError(f"Missing {test_soundscapes_dir}")

    submission = pd.read_csv(sample_submission_path)
    submission_columns = submission.columns.tolist()[1:]
    num_rows = len(submission)

    try:
        # PyTorch >=2.6 defaults to weights_only=True and may reject extra python objects
        # (for example pathlib.PosixPath) stored in training args. This checkpoint is trusted.
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        # Backward compatibility for older PyTorch versions without weights_only argument.
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
    labels: List[str] = checkpoint["labels"]
    model = BirdCLEFNet(num_classes=len(labels))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    frontend = MelSpectrogramFrontend(augment=False)
    if "frontend_state_dict" in checkpoint:
        frontend.load_state_dict(checkpoint["frontend_state_dict"], strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    frontend = frontend.to(device).eval()
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Device: {device}")

    grouped_rows: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    for row_index, row_id in enumerate(submission["row_id"].astype(str).tolist()):
        soundscape_id, end_second = parse_row_id(row_id)
        grouped_rows[soundscape_id].append((row_index, end_second))

    model_to_submission = np.array(
        [submission_columns.index(label) if label in submission_columns else -1 for label in labels],
        dtype=np.int32,
    )
    covered_labels = int((model_to_submission >= 0).sum())
    print(
        f"Label coverage: model={len(labels)} | submission={len(submission_columns)} | "
        f"matched={covered_labels} | unmatched_in_submission={len(submission_columns) - covered_labels}"
    )
    # Start from sample_submission defaults so local dry-run still outputs valid values
    # even when competition test audio files are not exposed in draft sessions.
    output_probs = submission.iloc[:, 1:].to_numpy(copy=True, dtype=np.float32)
    default_probs = output_probs.copy()
    soundscape_index = build_soundscape_index(test_soundscapes_dir)
    missing_soundscapes = []
    updated_rows = 0

    for soundscape_id, items in tqdm(grouped_rows.items(), desc="soundscapes"):
        try:
            soundscape_path = resolve_soundscape_path(
                test_soundscapes_dir, soundscape_id, soundscape_index=soundscape_index
            )
        except FileNotFoundError as exc:
            if args.strict_missing:
                raise
            missing_soundscapes.append(soundscape_id)
            print(f"[WARN] {exc}. Keep sample_submission defaults for these rows.")
            continue

        waveform = load_audio(soundscape_path, target_sr=TARGET_SAMPLE_RATE, mono=True)

        row_indices = [idx for idx, _ in items]
        end_seconds = [sec for _, sec in items]
        probs = predict_file(
            model=model,
            frontend=frontend,
            waveform=waveform,
            end_seconds=end_seconds,
            batch_size=args.batch_size,
            segment_seconds=args.segment_seconds,
            tta_flip=args.tta_flip,
            device=device,
        )

        for local_idx, row_idx in enumerate(row_indices):
            row_prob = probs[local_idx]
            valid_mask = model_to_submission >= 0
            output_probs[row_idx, model_to_submission[valid_mask]] = row_prob[valid_mask]
            updated_rows += 1

    submission.iloc[:, 1:] = output_probs
    submission.to_csv(args.output, index=False, float_format="%.6f")

    elapsed = time.time() - start_time
    if missing_soundscapes:
        print(
            f"[WARN] Missing soundscape files: {len(missing_soundscapes)} groups "
            f"(strict mode off)."
        )
    changed_values = int(np.count_nonzero(np.abs(output_probs - default_probs) > 1e-12))
    print(
        f"Prediction updates: updated_rows={updated_rows}/{num_rows} | "
        f"changed_values={changed_values}"
    )
    print(f"Saved submission to {args.output} | rows={len(submission)} | time={elapsed:.1f}s")


if __name__ == "__main__":
    main()

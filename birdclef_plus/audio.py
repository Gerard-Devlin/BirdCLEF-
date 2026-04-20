from __future__ import annotations

import ast
import random
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F

try:
    import torchaudio
except ImportError:  # pragma: no cover - fallback for environments without torchaudio
    torchaudio = None

try:
    import librosa
except ImportError:  # pragma: no cover - fallback for environments without librosa
    librosa = None


TARGET_SAMPLE_RATE = 32_000


def parse_secondary_labels(raw_value: str) -> List[str]:
    if not raw_value or raw_value == "[]":
        return []
    try:
        parsed = ast.literal_eval(raw_value)
    except (ValueError, SyntaxError):
        return []
    if not isinstance(parsed, list):
        return []
    return [str(label) for label in parsed]


def load_audio(path: Path, target_sr: int = TARGET_SAMPLE_RATE, mono: bool = True) -> torch.Tensor:
    path = Path(path)
    if torchaudio is not None:
        waveform, sample_rate = torchaudio.load(str(path))
        waveform = waveform.float()
        if mono and waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != target_sr:
            waveform = torchaudio.functional.resample(waveform, sample_rate, target_sr)
        return waveform.squeeze(0)

    if librosa is not None:
        waveform, _ = librosa.load(str(path), sr=target_sr, mono=mono)
        return torch.tensor(waveform, dtype=torch.float32)

    raise RuntimeError(
        "No audio backend available. Install torchaudio (recommended) or librosa."
    )


def crop_or_pad(waveform: torch.Tensor, num_samples: int, random_crop: bool = False) -> torch.Tensor:
    current = int(waveform.numel())
    if current == num_samples:
        return waveform

    if current > num_samples:
        if random_crop:
            start = random.randint(0, current - num_samples)
        else:
            start = (current - num_samples) // 2
        return waveform[start : start + num_samples]

    padding = num_samples - current
    if random_crop:
        left = random.randint(0, padding)
    else:
        left = padding // 2
    right = padding - left
    return F.pad(waveform, (left, right))

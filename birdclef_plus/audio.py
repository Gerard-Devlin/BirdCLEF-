from __future__ import annotations

import ast
import math
import random
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    np = None

try:
    import torchaudio
except ImportError:  # pragma: no cover - fallback for environments without torchaudio
    torchaudio = None

try:
    import soundfile as sf
except ImportError:  # pragma: no cover - optional fallback backend
    sf = None

try:
    import librosa
except ImportError:  # pragma: no cover - fallback for environments without librosa
    librosa = None

try:
    from scipy.signal import resample_poly
except ImportError:  # pragma: no cover - optional fallback resampler
    resample_poly = None


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


def _resample_waveform(
    waveform: torch.Tensor, sample_rate: int, target_sr: int
) -> torch.Tensor:
    if sample_rate == target_sr:
        return waveform

    if torchaudio is not None:
        return torchaudio.functional.resample(waveform.unsqueeze(0), sample_rate, target_sr).squeeze(0)

    if resample_poly is not None and np is not None:
        gcd = math.gcd(sample_rate, target_sr)
        up = target_sr // gcd
        down = sample_rate // gcd
        out = resample_poly(waveform.cpu().numpy(), up=up, down=down)
        return torch.tensor(out, dtype=torch.float32)

    raise RuntimeError(
        f"Need resampling from {sample_rate} to {target_sr}, but no resampler backend is available."
    )


def _load_with_torchaudio(path: Path) -> tuple[torch.Tensor, int]:
    if torchaudio is None:
        raise RuntimeError("torchaudio is not installed")

    try:
        waveform, sample_rate = torchaudio.load(str(path))
        return waveform.float(), int(sample_rate)
    except Exception:
        pass

    for backend in ("ffmpeg", "soundfile"):
        try:
            waveform, sample_rate = torchaudio.load(str(path), backend=backend)
            return waveform.float(), int(sample_rate)
        except Exception:
            continue

    raise RuntimeError(f"torchaudio failed to decode audio: {path}")


def _load_with_soundfile(path: Path) -> tuple[torch.Tensor, int]:
    if sf is None:
        raise RuntimeError("soundfile is not installed")
    waveform, sample_rate = sf.read(str(path), always_2d=False, dtype="float32")
    tensor = torch.tensor(waveform, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 2:
        # soundfile returns [num_samples, channels]
        tensor = tensor.transpose(0, 1).contiguous()
    else:
        raise RuntimeError(f"Unexpected waveform ndim from soundfile: {tensor.ndim}")
    return tensor, int(sample_rate)


def load_audio(path: Path, target_sr: int = TARGET_SAMPLE_RATE, mono: bool = True) -> torch.Tensor:
    path = Path(path)

    waveform = None
    sample_rate = None

    if torchaudio is not None:
        try:
            waveform, sample_rate = _load_with_torchaudio(path)
        except Exception:
            waveform, sample_rate = None, None

    if waveform is None and sf is not None:
        try:
            waveform, sample_rate = _load_with_soundfile(path)
        except Exception:
            waveform, sample_rate = None, None

    if waveform is not None and sample_rate is not None:
        if mono and waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = _resample_waveform(waveform.squeeze(0), sample_rate, target_sr)
        return waveform

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

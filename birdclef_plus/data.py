from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence
import random

import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset

from .audio import TARGET_SAMPLE_RATE, crop_or_pad, load_audio, parse_secondary_labels

try:
    import torchaudio
except ImportError as exc:  # pragma: no cover
    raise ImportError("torchaudio is required for mel-spectrogram feature extraction.") from exc


def load_train_csv(train_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(train_csv)
    df["primary_label"] = df["primary_label"].astype(str)
    df["secondary_labels"] = df["secondary_labels"].fillna("[]")
    return df


def parse_soundscape_label_tokens(raw_value: str) -> List[str]:
    if raw_value is None:
        return []
    tokens = [token.strip() for token in str(raw_value).split(";")]
    return [token for token in tokens if token]


def parse_hhmmss_to_seconds(text: str) -> float:
    parts = str(text).strip().split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid HH:MM:SS format: {text}")
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    return float(hours * 3600 + minutes * 60) + seconds


def load_soundscape_labels_csv(soundscape_labels_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(soundscape_labels_csv)
    required_columns = {"filename", "start", "end", "primary_label"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing columns in {soundscape_labels_csv}: {sorted(missing)}"
        )

    df = df.copy()
    df["filename"] = df["filename"].astype(str)
    df["primary_label"] = df["primary_label"].astype(str)
    df["start_seconds"] = df["start"].map(parse_hhmmss_to_seconds).astype(float)
    df["end_seconds"] = df["end"].map(parse_hhmmss_to_seconds).astype(float)

    # The CSV can contain exact duplicate rows. Keep only one copy.
    df = df.drop_duplicates(
        subset=["filename", "start_seconds", "end_seconds", "primary_label"]
    ).reset_index(drop=True)
    return df


def build_label_mapping(
    train_df: pd.DataFrame,
    soundscape_df: Optional[pd.DataFrame] = None,
    submission_labels: Optional[Sequence[str]] = None,
) -> List[str]:
    labels = set(train_df["primary_label"].astype(str).unique().tolist())

    if soundscape_df is not None and len(soundscape_df) > 0:
        for raw_value in soundscape_df["primary_label"].astype(str).tolist():
            labels.update(parse_soundscape_label_tokens(raw_value))

    if submission_labels is not None:
        labels.update([str(label) for label in submission_labels if str(label)])

    return sorted(labels)


class BirdTrainDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        audio_dir: Path,
        label_to_idx: Dict[str, int],
        segment_seconds: float = 5.0,
        sample_rate: int = TARGET_SAMPLE_RATE,
        is_train: bool = True,
        max_decode_retries: int = 3,
        log_decode_errors: bool = True,
    ) -> None:
        self.df = dataframe.reset_index(drop=True)
        self.audio_dir = Path(audio_dir)
        self.label_to_idx = label_to_idx
        self.segment_samples = int(segment_seconds * sample_rate)
        self.sample_rate = sample_rate
        self.is_train = is_train
        self.num_classes = len(label_to_idx)
        self.max_decode_retries = max(1, int(max_decode_retries))
        self.log_decode_errors = log_decode_errors

    def __len__(self) -> int:
        return len(self.df)

    def _build_target(self, row: pd.Series) -> torch.Tensor:
        target = torch.zeros(self.num_classes, dtype=torch.float32)
        primary = str(row["primary_label"])
        if primary in self.label_to_idx:
            target[self.label_to_idx[primary]] = 1.0

        for secondary in parse_secondary_labels(row["secondary_labels"]):
            if secondary in self.label_to_idx:
                target[self.label_to_idx[secondary]] = 1.0
        return target

    def __getitem__(self, index: int):
        row_index = int(index)
        row = self.df.iloc[row_index]
        waveform = None
        last_error = None
        last_audio_path = None

        for attempt in range(self.max_decode_retries):
            row = self.df.iloc[row_index]
            audio_path = self.audio_dir / row["filename"]
            last_audio_path = audio_path
            try:
                waveform = load_audio(audio_path, target_sr=self.sample_rate, mono=True)
                break
            except Exception as exc:
                last_error = exc
                if self.is_train and attempt < self.max_decode_retries - 1:
                    row_index = random.randint(0, len(self.df) - 1)
                    continue
                waveform = torch.zeros(self.segment_samples, dtype=torch.float32)
                if self.log_decode_errors:
                    print(
                        f"[WARN] decode failed for {audio_path}: {exc}. Using silence fallback.",
                        flush=True,
                    )
                break

        if waveform is None:
            # Safety fallback, should never be reached because we set silence above.
            waveform = torch.zeros(self.segment_samples, dtype=torch.float32)
            if self.log_decode_errors:
                print(
                    f"[WARN] decode fallback triggered for {last_audio_path}: {last_error}",
                    flush=True,
                )

        waveform = crop_or_pad(
            waveform, num_samples=self.segment_samples, random_crop=self.is_train
        )
        waveform = torch.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)
        target = self._build_target(row)
        return waveform, target


class BirdSoundscapeDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        audio_dir: Path,
        label_to_idx: Dict[str, int],
        segment_seconds: float = 5.0,
        sample_rate: int = TARGET_SAMPLE_RATE,
        is_train: bool = True,
        max_decode_retries: int = 3,
        log_decode_errors: bool = True,
        cache_size: int = 8,
    ) -> None:
        self.df = dataframe.reset_index(drop=True)
        self.audio_dir = Path(audio_dir)
        self.label_to_idx = label_to_idx
        self.segment_samples = int(segment_seconds * sample_rate)
        self.sample_rate = sample_rate
        self.is_train = is_train
        self.num_classes = len(label_to_idx)
        self.max_decode_retries = max(1, int(max_decode_retries))
        self.log_decode_errors = log_decode_errors
        self.cache_size = max(0, int(cache_size))
        self._waveform_cache: Dict[str, torch.Tensor] = {}
        self._cache_order: List[str] = []

    def __len__(self) -> int:
        return len(self.df)

    def _get_cached_waveform(self, audio_path: Path) -> Optional[torch.Tensor]:
        key = str(audio_path)
        waveform = self._waveform_cache.get(key)
        if waveform is None:
            return None

        # refresh LRU order
        if key in self._cache_order:
            self._cache_order.remove(key)
        self._cache_order.append(key)
        return waveform

    def _put_cached_waveform(self, audio_path: Path, waveform: torch.Tensor) -> None:
        if self.cache_size <= 0:
            return
        key = str(audio_path)
        if key in self._waveform_cache:
            if key in self._cache_order:
                self._cache_order.remove(key)
        self._waveform_cache[key] = waveform
        self._cache_order.append(key)

        while len(self._cache_order) > self.cache_size:
            drop_key = self._cache_order.pop(0)
            self._waveform_cache.pop(drop_key, None)

    def _build_target(self, raw_labels: str) -> torch.Tensor:
        target = torch.zeros(self.num_classes, dtype=torch.float32)
        for label in parse_soundscape_label_tokens(raw_labels):
            if label in self.label_to_idx:
                target[self.label_to_idx[label]] = 1.0
        return target

    def __getitem__(self, index: int):
        row_index = int(index)
        waveform = None
        last_error = None
        last_audio_path = None
        row = self.df.iloc[row_index]

        for attempt in range(self.max_decode_retries):
            row = self.df.iloc[row_index]
            audio_path = self.audio_dir / str(row["filename"])
            last_audio_path = audio_path
            cached = self._get_cached_waveform(audio_path)
            if cached is not None:
                waveform = cached
                break

            try:
                waveform = load_audio(audio_path, target_sr=self.sample_rate, mono=True)
                self._put_cached_waveform(audio_path, waveform)
                break
            except Exception as exc:
                last_error = exc
                if self.is_train and attempt < self.max_decode_retries - 1:
                    row_index = random.randint(0, len(self.df) - 1)
                    continue
                waveform = torch.zeros(self.segment_samples, dtype=torch.float32)
                if self.log_decode_errors:
                    print(
                        f"[WARN] decode failed for {audio_path}: {exc}. Using silence fallback.",
                        flush=True,
                    )
                break

        if waveform is None:
            waveform = torch.zeros(self.segment_samples, dtype=torch.float32)
            if self.log_decode_errors:
                print(
                    f"[WARN] decode fallback triggered for {last_audio_path}: {last_error}",
                    flush=True,
                )

        start_second = float(row["start_seconds"])
        end_second = float(row["end_seconds"])
        start_sample = max(0, int(start_second * self.sample_rate))
        end_sample = max(start_sample + 1, int(end_second * self.sample_rate))
        segment = waveform[start_sample:end_sample]
        segment = crop_or_pad(
            segment, num_samples=self.segment_samples, random_crop=self.is_train
        )
        segment = torch.nan_to_num(segment, nan=0.0, posinf=0.0, neginf=0.0)
        target = self._build_target(str(row["primary_label"]))
        return segment, target


class MelSpectrogramFrontend(nn.Module):
    def __init__(
        self,
        sample_rate: int = TARGET_SAMPLE_RATE,
        n_mels: int = 128,
        n_fft: int = 1024,
        hop_length: int = 320,
        fmin: int = 20,
        fmax: int = 16_000,
        augment: bool = False,
    ) -> None:
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=fmin,
            f_max=fmax,
            power=2.0,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80.0)
        self.augment = augment
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=16)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=24)

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        if waveforms.ndim == 1:
            waveforms = waveforms.unsqueeze(0)

        mels = self.mel(waveforms)  # [B, M, T]
        mels = self.to_db(mels)
        mels = (mels - mels.mean(dim=(-2, -1), keepdim=True)) / (
            mels.std(dim=(-2, -1), keepdim=True) + 1e-6
        )

        if self.training and self.augment:
            mels = self.freq_mask(mels)
            mels = self.time_mask(mels)

        return mels.unsqueeze(1)  # [B, 1, M, T]

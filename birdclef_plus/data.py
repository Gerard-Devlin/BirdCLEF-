from __future__ import annotations

from pathlib import Path
from typing import Dict, List
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


def build_label_mapping(df: pd.DataFrame) -> List[str]:
    labels = sorted(df["primary_label"].astype(str).unique().tolist())
    return labels


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
        target = self._build_target(row)
        return waveform, target


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

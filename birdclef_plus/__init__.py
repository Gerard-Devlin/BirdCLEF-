"""BirdCLEF+ baseline package."""

from .audio import TARGET_SAMPLE_RATE, crop_or_pad, load_audio, parse_secondary_labels
from .data import (
    BirdSoundscapeDataset,
    BirdTrainDataset,
    MelSpectrogramFrontend,
    build_label_mapping,
    load_soundscape_labels_csv,
    load_train_csv,
)
from .metrics import macro_auc_skip_empty
from .model import BirdCLEFNet
from .utils import seed_everything

__all__ = [
    "TARGET_SAMPLE_RATE",
    "crop_or_pad",
    "load_audio",
    "parse_secondary_labels",
    "BirdSoundscapeDataset",
    "BirdTrainDataset",
    "MelSpectrogramFrontend",
    "build_label_mapping",
    "load_soundscape_labels_csv",
    "load_train_csv",
    "macro_auc_skip_empty",
    "BirdCLEFNet",
    "seed_everything",
]

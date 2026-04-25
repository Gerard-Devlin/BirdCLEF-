# Auto-extracted from birdclef-26-two-pass-ssm-advanced-pp.ipynb
# Source notebook should be treated as canonical reference.

# ===== Cell 1 =====
# Install ONNX Runtime and TensorFlow 2.20 from local wheel files.
# ONNX is preferred because it is significantly faster than the TF SavedModel.
import subprocess, sys, os
from pathlib import Path

INPUT_ROOT = Path(os.environ.get("BC26_INPUT_ROOT", "/kaggle/input"))

def find_wheel(pattern):
    for p in INPUT_ROOT.rglob(pattern):
        return p
    return None

def maybe_install_wheel(pattern, label):
    whl = find_wheel(pattern)
    if whl is None:
        print(f"[WARN] wheel not found for {label}: pattern={pattern}")
        return False
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--no-deps", str(whl)],
        check=True,
    )
    print(f"{label} installed from {whl}")
    return True

ONNX_WHL = Path(
    os.environ.get(
        "BC26_ONNX_WHL",
        "/kaggle/input/datasets/rishikeshjani/perch-onnx-for-birdclef-2026/onnxruntime-1.24.4-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
    )
)
if not ONNX_WHL.exists():
    _onnx_wheel_from_input = find_wheel("onnxruntime-*.whl")
    if _onnx_wheel_from_input is not None:
        ONNX_WHL = _onnx_wheel_from_input
if ONNX_WHL.exists():
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--no-deps", str(ONNX_WHL)],
        check=True,
    )
    print(f"ONNX Runtime installed from {ONNX_WHL}")

try:
    import onnxruntime as ort
    _ONNX_AVAILABLE = True
    print("ONNX Runtime available")
except ImportError:
    _ONNX_AVAILABLE = False
    print("ONNX not available, falling back to TF")

_TF_AVAILABLE = False
try:
    import tensorflow as _tf_test  # noqa: F401
    _TF_AVAILABLE = True
except Exception:
    _TF_AVAILABLE = False

if not _TF_AVAILABLE:
    ok_tb = maybe_install_wheel("tensorboard-2.20.0-*.whl", "tensorboard-2.20.0")
    ok_tf = maybe_install_wheel("tensorflow-2.20.0-*.whl", "tensorflow-2.20.0")
    if ok_tb and ok_tf:
        try:
            import tensorflow as _tf_test  # noqa: F401
            _TF_AVAILABLE = True
            print("TF 2.20 installed")
        except Exception:
            _TF_AVAILABLE = False
            print("[WARN] TensorFlow install attempt failed; continuing in ONNX-only mode if possible.")
    else:
        print("[WARN] TensorFlow wheels not found; continuing in ONNX-only mode if possible.")
else:
    print("TensorFlow already available; skip local wheel install")

if (not _ONNX_AVAILABLE) and (not _TF_AVAILABLE):
    raise RuntimeError(
        "Neither ONNX Runtime nor TensorFlow is available. Install at least one backend."
    )

# ===== Cell 2 =====
# Toggle between local cross-validation ('train') and Kaggle submission ('submit').
# Downstream config values (epoch counts, OOF splits, dry-run size) all branch on this flag.
# You can override via env var: BC26_MODE=train|submit
MODE = os.environ.get("BC26_MODE", "submit").strip().lower()
 
assert MODE in {"train", "submit"}
print("MODE =", MODE)
USE_GPU = os.environ.get("BC26_USE_GPU", "1").strip().lower() not in {"0", "false", "no"}
print("USE_GPU =", USE_GPU)


def _env_int(name, default):
    raw = os.environ.get(name, "").strip()
    if raw == "":
        return int(default)
    return int(raw)


def _env_float(name, default):
    raw = os.environ.get(name, "").strip()
    if raw == "":
        return float(default)
    return float(raw)


def _env_int_list(name, default):
    raw = os.environ.get(name, "").strip()
    if raw == "":
        return list(default)
    vals = [int(x.strip()) for x in raw.split(",") if x.strip() != ""]
    return vals if vals else list(default)


def _env_float_list(name, default):
    raw = os.environ.get(name, "").strip()
    if raw == "":
        return list(default)
    vals = [float(x.strip()) for x in raw.split(",") if x.strip() != ""]
    return vals if vals else list(default)


def _env_bool(name, default=False):
    raw = os.environ.get(name, "").strip().lower()
    if raw == "":
        return bool(default)
    return raw in {"1", "true", "yes", "y", "on"}

# ===== Cell 3 =====
# Core imports, environment setup, path constants, audio parameters, and the main CFG dict.
# CFG values are conditional on MODE so the same notebook runs for both CV and submission.
import os, re, gc, time, warnings, random
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
 
import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from tqdm.auto import tqdm

if _TF_AVAILABLE:
    import tensorflow as tf
else:
    tf = None
 
if _TF_AVAILABLE:
    tf.experimental.numpy.experimental_enable_numpy_behavior()
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass
 
_WALL_START = time.time()
 
BASE      = Path(os.environ.get("BC26_BASE", "/kaggle/input/competitions/birdclef-2026"))
MODEL_DIR = Path(
    os.environ.get(
        "BC26_MODEL_DIR",
        "/kaggle/input/models/google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1",
    )
)
WORK_DIR  = Path(os.environ.get("BC26_WORK_DIR", "/kaggle/working/cache"))
WORK_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSION_PATH = Path(os.environ.get("BC26_SUBMISSION_PATH", "submission.csv"))
 
# Each recording is exactly 60 s; we slice it into 12 non-overlapping 5-second windows.
SR             = 32_000
WINDOW_SEC     = 5
WINDOW_SAMPLES = SR * WINDOW_SEC
FILE_SAMPLES   = 60 * SR
N_WINDOWS      = 12
 
CFG = {
    "batch_files": _env_int("BC26_BATCH_FILES", 16),
    "oof_n_splits": 5   if MODE == "train" else 3,
    "dryrun_n_files": 20 if MODE == "train" else 0,
    "run_oof": MODE == "train",
    "verbose": MODE == "train",
    "proto_ssm_train": {
        "n_epochs":        80  if MODE == "train" else 40,
        "lr":              8e-4,
        "weight_decay":    1e-3,
        "val_ratio":       0.15,
        "patience":        20  if MODE == "train" else 8,
        "pos_weight_cap":  25.0,
        "distill_weight":  0.15,
        "proto_margin":    0.15,
        "label_smoothing": 0.03,
        "oof_n_splits":    5   if MODE == "train" else 3,
        "mixup_alpha":     0.4,
        "focal_gamma":     2.5,
        "swa_start_frac":  0.65,
        "swa_lr":          4e-4,
        "use_cosine_restart": True,
        "restart_period":  20,
    },
    "residual_ssm": {
        "d_model": 128, "d_state": 16, "n_ssm_layers": 2,
        "dropout": 0.1, "correction_weight": 0.35,
        "n_epochs": 40  if MODE == "train" else 20,
        "lr": 8e-4,
        "patience": 12  if MODE == "train" else 6,
    },
    "mlp_params": {
        "hidden_layer_sizes": (256, 128), "activation": "relu",
        "max_iter": 500  if MODE == "train" else 200,
        "early_stopping": True,
        "validation_fraction": 0.15,
        "n_iter_no_change": 20  if MODE == "train" else 10,
        "random_state": 42,
        "learning_rate_init": 5e-4,
        "alpha": 0.005,
    },
}

TUNE = {
    "proto_epochs": _env_int("BC26_PROTO_EPOCHS", 40),
    "proto_patience": _env_int("BC26_PROTO_PATIENCE", 8),
    "proto_lr": _env_float("BC26_PROTO_LR", 1e-3),
    "proto_d_model": _env_int("BC26_PROTO_D_MODEL", 128),
    "proto_d_state": _env_int("BC26_PROTO_D_STATE", 16),
    "proto_meta_dim": _env_int("BC26_PROTO_META_DIM", 16),
    "proto_dropout": _env_float("BC26_PROTO_DROPOUT", 0.15),
    "proto_n_layers": _env_int("BC26_PROTO_N_LAYERS", 2),
    "proto_cross_attn_heads": _env_int("BC26_PROTO_CROSS_ATTN_HEADS", 2),
    "res_epochs": _env_int("BC26_RES_EPOCHS", 30),
    "res_patience": _env_int("BC26_RES_PATIENCE", 8),
    "res_lr": _env_float("BC26_RES_LR", 1e-3),
    "res_d_model": _env_int("BC26_RES_D_MODEL", 64),
    "res_d_state": _env_int("BC26_RES_D_STATE", 8),
    "res_meta_dim": _env_int("BC26_RES_META_DIM", 8),
    "res_dropout": _env_float("BC26_RES_DROPOUT", 0.1),
    "res_correction_weight": _env_float("BC26_RES_CORRECTION_WEIGHT", 0.30),
    "mlp_min_pos": _env_int("BC26_MLP_MIN_POS", 5),
    "mlp_pca_dim": _env_int("BC26_MLP_PCA_DIM", 64),
    "mlp_alpha_blend": _env_float("BC26_MLP_ALPHA_BLEND", 0.4),
    "mlp_probe_hidden1": _env_int("BC26_MLP_PROBE_HIDDEN1", 128),
    "mlp_probe_hidden2": _env_int("BC26_MLP_PROBE_HIDDEN2", 64),
    "mlp_probe_max_iter": _env_int("BC26_MLP_PROBE_MAX_ITER", 300),
    "mlp_probe_max_rows": _env_int("BC26_MLP_PROBE_MAX_ROWS", 3000),
    "mlp_probe_n_iter_no_change": _env_int("BC26_MLP_PROBE_N_ITER_NO_CHANGE", 15),
    "mlp_probe_lr_init": _env_float("BC26_MLP_PROBE_LR_INIT", 5e-4),
    "mlp_probe_alpha": _env_float("BC26_MLP_PROBE_ALPHA", 0.005),
    "mlp_probe_max_repeat": _env_int("BC26_MLP_PROBE_MAX_REPEAT", 8),
    "calib_min_pos_files": _env_int("BC26_CALIB_MIN_POS_FILES", 3),
    "calib_default_threshold": _env_float("BC26_CALIB_DEFAULT_THRESHOLD", 0.5),
    "prior_lambda": _env_float("BC26_PRIOR_LAMBDA", 0.4),
    "ensemble_w": 0.5,
    "tta_shifts": _env_int_list("BC26_TTA_SHIFTS", [0, 1, -1, 2, -2]),
    "threshold_grid": _env_float_list(
        "BC26_THRESHOLD_GRID",
        [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
    ),
    "threshold_grid_rare": _env_float_list(
        "BC26_THRESHOLD_GRID_RARE",
        [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
    ),
    "threshold_grid_common": _env_float_list(
        "BC26_THRESHOLD_GRID_COMMON",
        [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
    ),
    "calib_bucketed": _env_bool("BC26_CALIB_BUCKETED", False),
    "calib_use_oof": _env_bool("BC26_CALIB_USE_OOF", True),
    "calib_rare_pos_max": _env_int("BC26_CALIB_RARE_POS_MAX", 5),
    "calib_common_pos_min": _env_int("BC26_CALIB_COMMON_POS_MIN", 20),
    "post_topk": _env_int("BC26_POST_TOPK", 2),
    "post_conf_power": _env_float("BC26_POST_CONF_POWER", 0.4),
    "post_rank_power": _env_float("BC26_POST_RANK_POWER", 0.4),
    "post_smooth_alpha": _env_float("BC26_POST_SMOOTH_ALPHA", 0.20),
}

DISABLE_EARLY_STOP = _env_bool("BC26_DISABLE_EARLY_STOP", False)
GLOBAL_SEED = _env_int("BC26_SEED", 42)
DISABLE_GENUS_PROXY = _env_bool("BC26_DISABLE_GENUS_PROXY", False)
STOP_AFTER_RAW_OOF = _env_bool("BC26_STOP_AFTER_RAW_OOF", False)
PERCH_ADAPTER_CKPT_RAW = os.environ.get("BC26_PERCH_ADAPTER_CKPT", "").strip()
PERCH_ADAPTER_WEIGHT_RAW = os.environ.get("BC26_PERCH_ADAPTER_WEIGHT", "1.0").strip()
PERCH_ADAPTER_WEIGHT_AUTO = PERCH_ADAPTER_WEIGHT_RAW.lower() in {"auto", "cv", "soundscape"}
PERCH_ADAPTER_WEIGHT = 1.0 if PERCH_ADAPTER_WEIGHT_AUTO else float(PERCH_ADAPTER_WEIGHT_RAW)
PERCH_ADAPTER_WEIGHT_GRID = _env_float_list(
    "BC26_PERCH_ADAPTER_WEIGHT_GRID",
    [0.0, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.30],
)
PERCH_ADAPTER_PER_CLASS_WEIGHT_RAW = os.environ.get("BC26_PERCH_ADAPTER_PER_CLASS_WEIGHT", "").strip()
PERCH_ADAPTER_PER_CLASS_WEIGHT_AUTO = PERCH_ADAPTER_PER_CLASS_WEIGHT_RAW.lower() in {"auto", "cv", "soundscape"}
PERCH_ADAPTER_PER_CLASS_WEIGHT_GRID = _env_float_list(
    "BC26_PERCH_ADAPTER_PER_CLASS_WEIGHT_GRID",
    [0.0, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.30],
)
PERCH_ADAPTER_PER_CLASS_WEIGHT = None
PERCH_ADAPTER_MODEL = None
PERCH_EMB_CLS_CKPT_RAW = os.environ.get("BC26_PERCH_EMB_CLS_CKPT", "").strip()
PERCH_EMB_CLS_WEIGHT = _env_float("BC26_PERCH_EMB_CLS_WEIGHT", 0.35)
PERCH_EMB_CLS_MODEL = None
PERCH_MIL_CLS_CKPT_RAW = os.environ.get("BC26_PERCH_MIL_CLS_CKPT", "").strip()
PERCH_MIL_CLS_WEIGHT_RAW = os.environ.get("BC26_PERCH_MIL_CLS_WEIGHT", "0.35").strip()
PERCH_MIL_CLS_WEIGHT_AUTO = PERCH_MIL_CLS_WEIGHT_RAW.lower() in {"auto", "cv", "soundscape"}
PERCH_MIL_CLS_WEIGHT = 0.35 if PERCH_MIL_CLS_WEIGHT_AUTO else float(PERCH_MIL_CLS_WEIGHT_RAW)
PERCH_MIL_CLS_WEIGHT_GRID = _env_float_list(
    "BC26_PERCH_MIL_CLS_WEIGHT_GRID",
    [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.65, 0.80, 1.0],
)
PERCH_MIL_CLS_MODEL = None
UNMAPPED_HEAD_CKPT_RAW = os.environ.get("BC26_UNMAPPED_HEAD_CKPT", "").strip()
UNMAPPED_HEAD_WEIGHT = _env_float("BC26_UNMAPPED_HEAD_WEIGHT", 0.35)
UNMAPPED_HEAD_MODEL = None

ENSEMBLE_W_RAW = os.environ.get("BC26_ENSEMBLE_W", "").strip()
ENSEMBLE_W_AUTO = ENSEMBLE_W_RAW.lower() in {"auto", "cv", "soundscape"}
if not ENSEMBLE_W_AUTO and ENSEMBLE_W_RAW != "":
    TUNE["ensemble_w"] = float(ENSEMBLE_W_RAW)
ENSEMBLE_W_GRID = _env_float_list(
    "BC26_ENSEMBLE_W_GRID",
    [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80],
)
ENSEMBLE_PER_CLASS = _env_bool("BC26_ENSEMBLE_PER_CLASS", False)
ENSEMBLE_PER_FOLD = _env_bool("BC26_ENSEMBLE_PER_FOLD", False)
ENSEMBLE_PER_CLASS_MIN_POS = _env_int("BC26_ENSEMBLE_PER_CLASS_MIN_POS", 6)
ENSEMBLE_PER_CLASS_SHRINK = _env_float("BC26_ENSEMBLE_PER_CLASS_SHRINK", 0.5)
STACKING_ENABLE = _env_bool("BC26_STACKING_ENABLE", False)
STACKING_MIN_POS = _env_int("BC26_STACKING_MIN_POS", 6)
STACKING_LOGREG_C = _env_float("BC26_STACKING_LOGREG_C", 1.0)

np.random.seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)
CFG["mlp_params"]["random_state"] = GLOBAL_SEED

print("V18 CFG loaded")
print(f"  n_epochs={CFG['proto_ssm_train']['n_epochs']}  "
      f"patience={CFG['proto_ssm_train']['patience']}  "
      f"oof_n_splits={CFG['proto_ssm_train']['oof_n_splits']}  "
      f"mlp_max_iter={CFG['mlp_params']['max_iter']}")
print(f"  seed={GLOBAL_SEED}")
if PERCH_ADAPTER_CKPT_RAW:
    weight_msg = (
        "auto[" + ",".join(str(float(x)) for x in PERCH_ADAPTER_WEIGHT_GRID) + "]"
        if PERCH_ADAPTER_WEIGHT_AUTO
        else str(PERCH_ADAPTER_WEIGHT)
    )
    print(
        f"  perch_adapter_ckpt={PERCH_ADAPTER_CKPT_RAW} "
        f"(weight={weight_msg})"
    )
    if PERCH_ADAPTER_PER_CLASS_WEIGHT_AUTO:
        print(
            "  perch_adapter_per_class_weight=auto["
            + ",".join(str(float(x)) for x in PERCH_ADAPTER_PER_CLASS_WEIGHT_GRID)
            + "]"
        )
if PERCH_EMB_CLS_CKPT_RAW:
    print(
        f"  perch_emb_cls_ckpt={PERCH_EMB_CLS_CKPT_RAW} "
        f"(blend_weight={PERCH_EMB_CLS_WEIGHT})"
    )
if PERCH_MIL_CLS_CKPT_RAW:
    mil_weight_msg = (
        "auto[" + ",".join(str(float(x)) for x in PERCH_MIL_CLS_WEIGHT_GRID) + "]"
        if PERCH_MIL_CLS_WEIGHT_AUTO
        else str(PERCH_MIL_CLS_WEIGHT)
    )
    print(
        f"  perch_mil_cls_ckpt={PERCH_MIL_CLS_CKPT_RAW} "
        f"(blend_weight={mil_weight_msg})"
    )
if UNMAPPED_HEAD_CKPT_RAW:
    print(
        f"  unmapped_head_ckpt={UNMAPPED_HEAD_CKPT_RAW} "
        f"(blend_weight={UNMAPPED_HEAD_WEIGHT})"
    )
 
print("Config ready")
print(f"  run_oof={CFG['run_oof']}  verbose={CFG['verbose']}  dryrun={CFG['dryrun_n_files']}")
print(f"  BASE={BASE}")
print(f"  MODEL_DIR={MODEL_DIR}")
print(f"  WORK_DIR={WORK_DIR}")
print(f"  SUBMISSION_PATH={SUBMISSION_PATH}")
print(
    "TUNE: "
    f"proto={TUNE['proto_epochs']}ep/{TUNE['proto_patience']}pat@{TUNE['proto_lr']} "
    f"(d_model={TUNE['proto_d_model']},d_state={TUNE['proto_d_state']},layers={TUNE['proto_n_layers']},heads={TUNE['proto_cross_attn_heads']}) "
    f"res={TUNE['res_epochs']}ep/{TUNE['res_patience']}pat@{TUNE['res_lr']} "
    f"(d_model={TUNE['res_d_model']},d_state={TUNE['res_d_state']}) "
    f"ens_w={'auto' if ENSEMBLE_W_AUTO else TUNE['ensemble_w']} "
    f"mlp(min_pos={TUNE['mlp_min_pos']},pca={TUNE['mlp_pca_dim']},alpha={TUNE['mlp_alpha_blend']},"
    f"hidden=({TUNE['mlp_probe_hidden1']},{TUNE['mlp_probe_hidden2']}),max_iter={TUNE['mlp_probe_max_iter']}) "
    f"calib(min_pos_files={TUNE['calib_min_pos_files']},default_t={TUNE['calib_default_threshold']}) "
    f"tta(shifts={TUNE['tta_shifts']})"
)
if ENSEMBLE_PER_CLASS:
    print("TUNE: per-class ensemble weight enabled via BC26_ENSEMBLE_PER_CLASS=1")
if ENSEMBLE_PER_FOLD:
    print("TUNE: per-fold ensemble weight enabled via BC26_ENSEMBLE_PER_FOLD=1")
if STACKING_ENABLE:
    print(
        f"TUNE: OOF stacking enabled via BC26_STACKING_ENABLE=1 "
        f"(min_pos={STACKING_MIN_POS}, C={STACKING_LOGREG_C})"
    )
if TUNE["calib_bucketed"]:
    print(
        "TUNE: bucketed calibration enabled "
        f"(rare<= {TUNE['calib_rare_pos_max']}, common>= {TUNE['calib_common_pos_min']})"
    )
if TUNE["calib_use_oof"]:
    print("TUNE: calibration uses OOF probs when available via BC26_CALIB_USE_OOF=1")
if DISABLE_EARLY_STOP:
    print("TUNE: early stopping disabled via BC26_DISABLE_EARLY_STOP=1")
if DISABLE_GENUS_PROXY:
    print("TUNE: genus proxy disabled via BC26_DISABLE_GENUS_PROXY=1")
if STOP_AFTER_RAW_OOF:
    print("TUNE: stop after raw OOF via BC26_STOP_AFTER_RAW_OOF=1")

# ===== Cell 4 =====
# Load competition CSVs and derive per-window label arrays.
# Only files that have annotations for all 12 windows are kept as 'fully labeled';
# these are the only rows used for supervised training.
taxonomy          = pd.read_csv(BASE / "taxonomy.csv")
sample_sub        = pd.read_csv(BASE / "sample_submission.csv")
soundscape_labels = pd.read_csv(BASE / "train_soundscapes_labels.csv")
 
PRIMARY_LABELS = sample_sub.columns[1:].tolist()
N_CLASSES      = len(PRIMARY_LABELS)
label_to_idx   = {c: i for i, c in enumerate(PRIMARY_LABELS)}
 
# Regex to extract site code and UTC hour from BirdCLEF 2026 filenames.
FNAME_RE = re.compile(r"BC2026_(?:Train|Test)_(\d+)_(S\d+)_(\d{8})_(\d{6})\.ogg")
 
def parse_fname(name):
    m = FNAME_RE.match(name)
    if not m: return {"site": "unknown", "hour_utc": -1}
    _, site, _, hms = m.groups()
    return {"site": site, "hour_utc": int(hms[:2])}
 
def union_labels(series):
    out = set()
    for x in series:
        if pd.notna(x):
            for t in str(x).split(";"):
                t = t.strip()
                if t: out.add(t)
    return sorted(out)
 
sc = (soundscape_labels
      .groupby(["filename", "start", "end"])["primary_label"]
      .apply(union_labels)
      .reset_index(name="label_list"))
 
sc["end_sec"] = pd.to_timedelta(sc["end"]).dt.total_seconds().astype(int)
sc["row_id"]  = sc["filename"].str.replace(".ogg", "", regex=False) + "_" + sc["end_sec"].astype(str)
 
_meta = sc["filename"].apply(parse_fname).apply(pd.Series)
sc = pd.concat([sc, _meta], axis=1)
 
# Build a binary label matrix Y_SC aligned with the sc DataFrame rows.
Y_SC = np.zeros((len(sc), N_CLASSES), dtype=np.uint8)
for i, lbls in enumerate(sc["label_list"]):
    for lbl in lbls:
        if lbl in label_to_idx:
            Y_SC[i, label_to_idx[lbl]] = 1
 
# Keep only files annotated across all 12 windows to avoid partial-label noise.
windows_per_file = sc.groupby("filename").size()
full_files = sorted(windows_per_file[windows_per_file == N_WINDOWS].index.tolist())
sc["fully_labeled"] = sc["filename"].isin(full_files)
 
full_rows = (sc[sc["fully_labeled"]]
             .sort_values(["filename", "end_sec"])
             .reset_index(drop=False))
Y_FULL = Y_SC[full_rows["index"].to_numpy()]
 
print(f"Classes: {N_CLASSES} | Fully-labeled files: {len(full_files)}")
print(f"Full-file windows: {len(full_rows)} | Active classes: {int((Y_FULL.sum(0) > 0).sum())}")

# ===== Cell 5 =====
ONNX_PERCH_PATH = Path(
    os.environ.get(
        "BC26_ONNX_PATH",
        "/kaggle/input/datasets/rishikeshjani/perch-onnx-for-birdclef-2026/perch_v2.onnx",
    )
)
USE_ONNX = _ONNX_AVAILABLE and ONNX_PERCH_PATH.exists()
infer_fn = None

if USE_ONNX:
    _so = ort.SessionOptions()
    _so.intra_op_num_threads = 4
    _providers = ["CPUExecutionProvider"]
    try:
        _avail = ort.get_available_providers()
        if USE_GPU and "CUDAExecutionProvider" in _avail:
            _providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    except Exception:
        _providers = ["CPUExecutionProvider"]
    ONNX_SESSION    = ort.InferenceSession(str(ONNX_PERCH_PATH), sess_options=_so,
                                            providers=_providers)
    ONNX_INPUT_NAME = ONNX_SESSION.get_inputs()[0].name
    ONNX_OUT_MAP    = {o.name: i for i, o in enumerate(ONNX_SESSION.get_outputs())}
    print(f"Using ONNX Perch (150x faster) | providers={ONNX_SESSION.get_providers()}")
else:
    if not _TF_AVAILABLE:
        raise RuntimeError(
            f"ONNX model not found at {ONNX_PERCH_PATH} and TensorFlow is unavailable."
        )
    birdclassifier = tf.saved_model.load(str(MODEL_DIR))
    infer_fn = birdclassifier.signatures["serving_default"]
    print("Using TF SavedModel Perch")

bc_labels = (pd.read_csv(MODEL_DIR / "assets" / "labels.csv")
             .reset_index()
             .rename(columns={"index": "bc_index", "inat2024_fsd50k": "scientific_name"}))
NO_LABEL = len(bc_labels)

# Map each competition species to its Perch output index via scientific name.
# Species not in Perch's vocabulary receive the sentinel value NO_LABEL.
mapping = (taxonomy
           .merge(bc_labels.rename(columns={"scientific_name": "scientific_name"}),
                  on="scientific_name", how="left"))
mapping["bc_index"] = mapping["bc_index"].fillna(NO_LABEL).astype(int)
lbl2bc = mapping.set_index("primary_label")["bc_index"]

BC_INDICES    = np.array([int(lbl2bc.loc[c]) for c in PRIMARY_LABELS], dtype=np.int32)
MAPPED_MASK   = BC_INDICES != NO_LABEL
MAPPED_POS    = np.where(MAPPED_MASK)[0].astype(np.int32)
MAPPED_BC_IDX = BC_INDICES[MAPPED_MASK].astype(np.int32)

print(f"Mapped: {MAPPED_MASK.sum()} / {N_CLASSES} species have a Perch logit")

# ===== Cell 6 =====
# Build genus-level proxy logits for species that Perch does not recognise directly.
# For each unmapped species we find all Perch entries from the same genus and
# take their max logit as a soft signal. Only biologically plausible taxa are proxied.
import re as _re

UNMAPPED_POS  = np.where(~MAPPED_MASK)[0].astype(np.int32)

CLASS_NAME_MAP = taxonomy.set_index("primary_label")["class_name"].to_dict()
TEXTURE_TAXA   = {"Amphibia", "Insecta"}

proxy_map = {}

unmapped_df = (taxonomy[taxonomy["primary_label"]
               .isin([PRIMARY_LABELS[i] for i in UNMAPPED_POS])]
               .copy())

for _, row in unmapped_df.iterrows():
    target = row["primary_label"]
    sci    = str(row["scientific_name"])
    genus  = sci.split()[0]
    
    hits = bc_labels[
        bc_labels["scientific_name"]
        .astype(str)
        .str.match(rf"^{_re.escape(genus)}\s", na=False)
    ]
    
    if len(hits) > 0:
        proxy_map[label_to_idx[target]] = hits["bc_index"].astype(int).tolist()

# Restrict proxies to taxa where genus-level similarity is biologically meaningful.
PROXY_TAXA = {"Amphibia", "Insecta", "Aves"}
proxy_map  = {
    idx: bc_idxs
    for idx, bc_idxs in proxy_map.items()
    if CLASS_NAME_MAP.get(PRIMARY_LABELS[idx]) in PROXY_TAXA
}
if DISABLE_GENUS_PROXY:
    proxy_map = {}

print(f"Unmapped species total:        {len(UNMAPPED_POS)}")
print(f"Species with genus proxy:      {len(proxy_map)}")
print(f"Species still without signal:  {len(UNMAPPED_POS) - len(proxy_map)}")
print("\nProxy targets:")
for idx, bc_idxs in list(proxy_map.items())[:8]:
    label = PRIMARY_LABELS[idx]
    cls   = CLASS_NAME_MAP.get(label, "?")
    print(f"  {label:12s} ({cls:10s}) ->{len(bc_idxs)} Perch genus matches")

# ===== Cell 7 =====
# Perch inference engine: reads audio in batches with async I/O prefetch,
# runs ONNX or TF inference, and fills score + embedding matrices.
# Mapped species get their direct Perch logit; unmapped species get the
# best genus-level proxy logit via proxy_map.
import concurrent.futures

def read_60s(path):
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim == 2: y = y.mean(axis=1)
    if len(y) < FILE_SAMPLES: y = np.pad(y, (0, FILE_SAMPLES - len(y)))
    else:                      y = y[:FILE_SAMPLES]
    return y

def run_perch(paths, batch_files=16, verbose=True):
    paths  = [Path(p) for p in paths]
    n_rows = len(paths) * N_WINDOWS

    row_ids   = np.empty(n_rows, dtype=object)
    filenames = np.empty(n_rows, dtype=object)
    sites     = np.empty(n_rows, dtype=object)
    hours     = np.zeros(n_rows, dtype=np.int16)
    scores    = np.zeros((n_rows, N_CLASSES), dtype=np.float32)
    embs      = np.zeros((n_rows, 1536),      dtype=np.float32)

    wr  = 0
    itr = tqdm(range(0, len(paths), batch_files), desc="Perch") if verbose else range(0, len(paths), batch_files)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as io_executor:
        next_paths   = paths[0:batch_files]
        future_audio = [io_executor.submit(read_60s, p) for p in next_paths]

        for start in itr:
            batch_paths  = next_paths
            batch_n      = len(batch_paths)
            batch_audio  = [f.result() for f in future_audio]

            next_start = start + batch_files
            if next_start < len(paths):
                next_paths   = paths[next_start:next_start + batch_files]
                future_audio = [io_executor.submit(read_60s, p) for p in next_paths]

            x  = np.empty((batch_n * N_WINDOWS, WINDOW_SAMPLES), dtype=np.float32)
            br = wr

            for bi, path in enumerate(batch_paths):
                y    = batch_audio[bi]
                meta = parse_fname(path.name)
                stem = path.stem
                x[bi * N_WINDOWS:(bi + 1) * N_WINDOWS] = y.reshape(N_WINDOWS, WINDOW_SAMPLES)
                row_ids  [wr:wr + N_WINDOWS] = [f"{stem}_{t}" for t in range(5, 65, 5)]
                filenames[wr:wr + N_WINDOWS] = path.name
                sites    [wr:wr + N_WINDOWS] = meta["site"]
                hours    [wr:wr + N_WINDOWS] = meta["hour_utc"]
                wr += N_WINDOWS

            if USE_ONNX:
                def _run_onnx_with_auto_chunk(batch_audio, min_chunk=8):
                    try:
                        _outs = ONNX_SESSION.run(None, {ONNX_INPUT_NAME: batch_audio})
                        _logits = _outs[ONNX_OUT_MAP["label"]].astype(np.float32)
                        _emb = _outs[ONNX_OUT_MAP["embedding"]].astype(np.float32)
                        return _logits, _emb
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
                        if verbose:
                            print(
                                f"[WARN] ONNX OOM at batch={n}, retry chunks {mid}+{n-mid}",
                                flush=True,
                            )
                        l_logit, l_emb = _run_onnx_with_auto_chunk(batch_audio[:mid], min_chunk=min_chunk)
                        r_logit, r_emb = _run_onnx_with_auto_chunk(batch_audio[mid:], min_chunk=min_chunk)
                        return (
                            np.concatenate([l_logit, r_logit], axis=0),
                            np.concatenate([l_emb, r_emb], axis=0),
                        )

                logits, emb = _run_onnx_with_auto_chunk(x)
            else:
                out    = infer_fn(inputs=tf.convert_to_tensor(x))
                logits = out["label"].numpy().astype(np.float32)
                emb    = out["embedding"].numpy().astype(np.float32)

            scores[br:wr, MAPPED_POS] = logits[:, MAPPED_BC_IDX]
            embs  [br:wr]             = emb

            for pos_idx, bc_idxs in proxy_map.items():
                bc_arr = np.array(bc_idxs, dtype=np.int32)
                scores[br:wr, pos_idx] = logits[:, bc_arr].max(axis=1)

            del x, logits, emb, batch_audio
            gc.collect()

    meta_df = pd.DataFrame({"row_id": row_ids, "filename": filenames,
                             "site": sites, "hour_utc": hours})
    return meta_df, scores, embs

print("[OK] Perch inference engine (ONNX + multithreaded I/O) defined")

# ===== Cell 8 =====
# Load Perch embeddings and scores for the training split from cache.
# Priority: external mounted dataset ->local working-dir cache ->build from scratch.
# On a cache miss the full run_perch() pass is executed and results are persisted
# to /kaggle/working/cache for reuse within the session.

print(f"USE_ONNX = {USE_ONNX}  "
      f"(cache will be built with {'ONNX' if USE_ONNX else 'TF SavedModel'})")

EXTERNAL_CACHE_DIRS = [
    Path("/kaggle/input/notebooks/vyankteshdwivedi/notebook1b25083f0d"),
    Path("/kaggle/input/datasets/jaejohn/perch-meta"),
]
extra_cache_dirs = os.environ.get("BC26_EXTRA_CACHE_DIRS", "").strip()
if extra_cache_dirs:
    for raw_path in extra_cache_dirs.split(os.pathsep):
        raw_path = raw_path.strip()
        if raw_path:
            EXTERNAL_CACHE_DIRS.append(Path(raw_path))

CACHE_META_LOCAL = WORK_DIR / "perch_meta.parquet"
CACHE_NPZ_LOCAL  = WORK_DIR / "perch_arrays.npz"

META_NAME_CANDIDATES = ("perch_meta.parquet", "full_perch_meta.parquet")
NPZ_NAME_CANDIDATES = ("perch_arrays.npz", "full_perch_arrays.npz")


def _find_cache_pair_in_dir(root: Path):
    for meta_name in META_NAME_CANDIDATES:
        for npz_name in NPZ_NAME_CANDIDATES:
            meta = root / meta_name
            npz = root / npz_name
            if meta.exists() and npz.exists():
                return meta, npz
    return None, None


def _find_external_cache():
    for d in EXTERNAL_CACHE_DIRS:
        meta, npz = _find_cache_pair_in_dir(d)
        if meta is not None and npz is not None:
            return meta, npz
    return None, None

# Fallback key lists handle npz files saved under different key naming conventions.
SCORE_KEYS = ["scores", "sc", "logits", "perch_scores", "preds", "arr_0"]
EMB_KEYS   = ["embs", "emb", "embeddings", "features", "perch_embs", "arr_1"]

def _pick_array(arr, candidates, shape_hint_cols):
    """Try known key names first, then shape-based fallback."""
    for k in candidates:
        if k in arr.files:
            return arr[k], k
    for k in arr.files:
        v = arr[k]
        if v.ndim == 2 and v.shape[1] == shape_hint_cols:
            return v, k
    raise KeyError(
        f"None of {candidates} found in npz. Available keys: {arr.files}"
    )

def _build_cache():
    print(f"Building Perch cache from {len(full_files)} fully-labeled "
          f"train_soundscape files...")
    train_paths = [BASE / "train_soundscapes" / fn for fn in full_files]
    missing = [p for p in train_paths if not p.exists()]
    if missing:
        print(f"  WARNING: {len(missing)} files listed but not on disk; skipping")
        train_paths = [p for p in train_paths if p.exists()]

    t0 = time.time()
    meta_built, sc_built, emb_built = run_perch(
        train_paths,
        batch_files=CFG["batch_files"],
        verbose=True,
    )
    print(f"  Perch pass finished in {time.time()-t0:.1f}s  "
          f"scores={sc_built.shape} embs={emb_built.shape}")

    meta_built.to_parquet(CACHE_META_LOCAL)
    np.savez(
        CACHE_NPZ_LOCAL,
        scores=sc_built.astype(np.float32),
        embs=emb_built.astype(np.float32),
        primary_labels=np.array(PRIMARY_LABELS),
    )
    print(f"  Cache saved to {WORK_DIR}")
    return CACHE_META_LOCAL, CACHE_NPZ_LOCAL

ext_meta, ext_npz = _find_external_cache()
if ext_meta is not None:
    CACHE_META, CACHE_NPZ = ext_meta, ext_npz
    print(f"Using external cache: {CACHE_META.parent}")
else:
    local_meta, local_npz = _find_cache_pair_in_dir(WORK_DIR)
    if local_meta is not None:
        CACHE_META, CACHE_NPZ = local_meta, local_npz
        print(f"Using local cache: {WORK_DIR}")
    else:
        print("No cache found - building from scratch")
        CACHE_META, CACHE_NPZ = _build_cache()

print("Loading Perch cache from:", CACHE_META.parent)
meta_tr = pd.read_parquet(CACHE_META)
_arr    = np.load(CACHE_NPZ)
print("  npz keys      :", list(_arr.keys()))
print("  parquet cols  :", meta_tr.columns.tolist())

sc_tr_raw,  sk = _pick_array(_arr, SCORE_KEYS, N_CLASSES)
emb_tr_raw, ek = _pick_array(_arr, EMB_KEYS,   1536)
print(f"  scores ->'{sk}'  shape={sc_tr_raw.shape}")
print(f"  embs   ->'{ek}'  shape={emb_tr_raw.shape}")

sc_tr  = sc_tr_raw.astype(np.float32)
emb_tr = emb_tr_raw.astype(np.float32)

# Warn if the cached label order differs from the current sample_submission.
if "primary_labels" in _arr.files:
    cached_labels = _arr["primary_labels"].tolist()
    if cached_labels != PRIMARY_LABELS:
        print("  WARNING: cached primary_labels differ from current "
              "sample_submission -scores columns may not align!")
    else:
        print("  primary_labels schema OK")

# Reconstruct row_id from available columns if it is absent from the parquet.
if "row_id" not in meta_tr.columns:
    print("  row_id missing in parquet -reconstructing")
    if "end_sec" in meta_tr.columns:
        end_sec = meta_tr["end_sec"].astype(int)
    elif "window_idx" in meta_tr.columns:
        end_sec = (meta_tr["window_idx"].astype(int) + 1) * 5
    else:
        n_files_cache = len(meta_tr) // N_WINDOWS
        end_sec = np.tile(np.arange(5, 65, 5), n_files_cache)
    meta_tr["row_id"] = (
        meta_tr["filename"].str.replace(".ogg", "", regex=False)
        + "_" + end_sec.astype(str)
    )

# Align Y_FULL to the row order used by the cache rather than the original sc sort order.
row_id_to_index = full_rows.set_index("row_id")["index"]
missing_rows = set(meta_tr["row_id"]) - set(row_id_to_index.index)
if missing_rows:
    raise RuntimeError(
        f"Cache contains {len(missing_rows)} row_ids not present in current "
        f"fully-labeled set. Example: {list(missing_rows)[:3]}. "
        f"This usually means the cache was built against a different competition "
        f"data version -rebuild the cache by deleting {CACHE_META_LOCAL} and "
        f"{CACHE_NPZ_LOCAL}, then rerunning this cell."
    )

Y_FULL_aligned = Y_SC[row_id_to_index.loc[meta_tr["row_id"]].to_numpy()]

expected_rows = len(full_files) * N_WINDOWS
if len(meta_tr) != expected_rows:
    print(f"  NOTE: cache has {len(meta_tr)} rows, current full_files implies "
          f"{expected_rows}. Proceeding with cache's own coverage.")

print(f"sc_tr: {sc_tr.shape}  emb_tr: {emb_tr.shape}  "
      f"Y_FULL_aligned: {Y_FULL_aligned.shape}")

if PERCH_ADAPTER_MODEL is not None:
    t0 = time.time()
    sc_tr = _apply_perch_adapter(
        sc_tr,
        emb_tr,
        PERCH_ADAPTER_MODEL,
        weight=PERCH_ADAPTER_WEIGHT,
    )
    print(
        f"[OK] Applied Perch adapter to training cache in {time.time()-t0:.1f}s "
        f"| score range [{sc_tr.min():.3f}, {sc_tr.max():.3f}]"
    )

# ===== Cell 9 =====
# Competition metric (macro ROC-AUC over classes with at least one positive)
# and an honest GroupKFold OOF evaluator that never splits a file across folds.
def macro_auc(y_true, y_score):
    """
    Exact replica of the competition metric:
    macro-averaged ROC-AUC, skipping classes with no positive labels.
    This is the ONLY number you should track locally.
    """
    keep = y_true.sum(axis=0) > 0
    return roc_auc_score(y_true[:, keep], y_score[:, keep], average="macro")
 
 
def honest_oof_auc(scores, Y, meta_df, n_splits=5, label="scores"):
    """
    GroupKFold by filename -files never split across folds.
    This is the only correct way to estimate LB performance locally.
    Leaking a file across train/val inflates AUC by ~0.01-.03.
    """
    groups = meta_df["filename"].to_numpy()
    gkf    = GroupKFold(n_splits=n_splits)
    oof    = np.zeros_like(scores, dtype=np.float32)
 
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(scores, groups=groups), 1):
        oof[va_idx] = scores[va_idx]
 
    auc = macro_auc(Y, oof)
    print(f"[{label}] honest OOF macro-AUC: {auc:.6f}")
    return auc, oof

# ===== Cell 10 =====
# Temporal smoothing: blends each 5-second window with its immediate neighbours.
# alpha controls the strength -0 means no smoothing, higher values lean more on neighbours.
# Edge windows are padded by repeating the boundary window.
def smooth_predictions(probs, n_windows=12, alpha=0.3):
    """
    For each file's 12 windows, blend each window with its neighbors.
    
    new[t] = (1 - alpha) * old[t] + 0.5*alpha * (old[t-1] + old[t+1])
    
    alpha=0: no smoothing (your current baseline)
    alpha=0.3: moderate smoothing (good starting point)
    
    Shape: (n_files * 12, n_classes) ->same shape output
    """
    N, C = probs.shape
    assert N % n_windows == 0, f"Expected multiple of {n_windows}, got {N}"
    
    view = probs.reshape(-1, n_windows, C).copy()
    
    prev_w = np.concatenate([view[:, :1, :],  view[:, :-1, :]], axis=1)
    next_w = np.concatenate([view[:, 1:,  :], view[:, -1:, :]], axis=1)
    
    smoothed = (1 - alpha) * view + 0.5 * alpha * (prev_w + next_w)
    
    return smoothed.reshape(N, C)


print("[OK] Temporal smoothing helper defined")

# ===== Cell 11 =====
# Build site-level and hour-level species frequency tables from training labels,
# then apply them as an additive logit prior to raw Perch scores.
# Shrinkage toward the global mean is applied when a stratum has little data.
def build_prior_tables(sc_df, Y_labels):
    """
    Build site-level and hour-level species frequency tables.
    
    These answer: "How often is species X observed at site S at hour H?"
    
    We use these as a soft prior: add them to raw Perch logits.
    """
    sc_df = sc_df.reset_index(drop=True)
    global_p = Y_labels.mean(axis=0).astype(np.float32)
    
    site_keys = sorted(sc_df["site"].dropna().astype(str).unique())
    site_to_i = {k: i for i, k in enumerate(site_keys)}
    site_p    = np.zeros((len(site_keys), Y_labels.shape[1]), dtype=np.float32)
    site_n    = np.zeros(len(site_keys), dtype=np.float32)
    
    for s in site_keys:
        i     = site_to_i[s]
        mask  = sc_df["site"].astype(str).values == s
        site_n[i] = mask.sum()
        site_p[i] = Y_labels[mask].mean(axis=0)
    
    hour_keys = sorted(sc_df["hour_utc"].dropna().astype(int).unique())
    hour_to_i = {h: i for i, h in enumerate(hour_keys)}
    hour_p    = np.zeros((len(hour_keys), Y_labels.shape[1]), dtype=np.float32)
    hour_n    = np.zeros(len(hour_keys), dtype=np.float32)
    
    for h in hour_keys:
        i     = hour_to_i[h]
        mask  = sc_df["hour_utc"].astype(int).values == h
        hour_n[i] = mask.sum()
        hour_p[i] = Y_labels[mask].mean(axis=0)
    
    return {
        "global_p": global_p,
        "site_to_i": site_to_i, "site_p": site_p, "site_n": site_n,
        "hour_to_i": hour_to_i, "hour_p": hour_p, "hour_n": hour_n,
    }


def apply_prior(scores, sites, hours, tables, lambda_prior=0.4):
    """
    Add a scaled prior logit to the raw Perch scores.
    
    lambda_prior=0: no effect (your baseline)
    lambda_prior=0.4: moderate influence from location/time
    
    The prior is converted to a logit (log-odds) before adding.
    This is mathematically correct -you add logits, not probabilities.
    """
    eps = 1e-4
    n   = len(scores)
    out = scores.copy()
    
    p = np.tile(tables["global_p"], (n, 1))
    
    for i, h in enumerate(hours):
        h = int(h)
        if h in tables["hour_to_i"]:
            j   = tables["hour_to_i"][h]
            nh  = tables["hour_n"][j]
            w   = nh / (nh + 8.0)
            p[i] = w * tables["hour_p"][j] + (1 - w) * tables["global_p"]
    
    for i, s in enumerate(sites):
        s = str(s)
        if s in tables["site_to_i"]:
            j   = tables["site_to_i"][s]
            ns  = tables["site_n"][j]
            w   = ns / (ns + 8.0)
            p[i] = w * tables["site_p"][j] + (1 - w) * p[i]
    
    p      = np.clip(p, eps, 1 - eps)
    logit_prior = np.log(p) - np.log1p(-p)
    out   += lambda_prior * logit_prior
    
    return out.astype(np.float32)


print("[OK] Prior table functions defined")

# ===== Cell 12 =====
# File-level confidence scaling: suppresses low-confidence files by multiplying
# each window's scores by a power of the file's top-k mean score.
# Files where no window ever fires confidently get damped across all windows.
def file_confidence_scale(probs, n_windows=12, top_k=2, power=0.4):
    """
    Scale each window's predictions by how confident the file is overall.
    
    Steps:
    1. For each file, find the top-k highest scores across all 12 windows
    2. Compute their mean ->"file confidence"
    3. Multiply every window's scores by (file_confidence ** power)
    
    power=0: no effect (baseline)
    power=0.4: moderate suppression of uncertain files
    
    Why top-k and not max?
    Max is noisy (one lucky spike). Top-2 mean is more robust.
    """
    N, C = probs.shape
    assert N % n_windows == 0
    
    view      = probs.reshape(-1, n_windows, C)
    sorted_v  = np.sort(view, axis=1)
    top_k_mean = sorted_v[:, -top_k:, :].mean(axis=1, keepdims=True)
    
    scale  = np.power(top_k_mean, power)
    scaled = view * scale
    
    return scaled.reshape(N, C)


print("[OK] File-level confidence scaling defined")

# ===== Cell 13 =====
# Assign per-class temperature scalars: slightly sharper for continuous-calling taxa
# (frogs, insects) and slightly softer for birds, then apply via logit division.
CLASS_NAME_MAP = taxonomy.set_index("primary_label")["class_name"].to_dict()
TEXTURE_TAXA   = {"Amphibia", "Insecta"}

temperatures = np.ones(N_CLASSES, dtype=np.float32)
for ci, label in enumerate(PRIMARY_LABELS):
    cls = CLASS_NAME_MAP.get(label, "Aves")
    if cls in TEXTURE_TAXA:
        temperatures[ci] = 0.95
    else:
        temperatures[ci] = 1.10

n_texture = (temperatures < 1.0).sum()
n_event   = (temperatures > 1.0).sum()
print(f"[OK] Temperatures: {n_event} event species (T=1.10), {n_texture} texture species (T=0.95)")

# ===== Cell 14 =====
# MLP probes trained per-species on PCA-compressed Perch embeddings plus
# sequential score features (prev, next, mean, max, std across the 12 windows).
# Rare species are oversampled proportionally to their class weight before fitting.
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

def build_class_freq_weights(Y, cap=10.0):
    total     = Y.shape[0]
    pos_count = Y.sum(axis=0).astype(np.float32) + 1.0
    freq      = pos_count / total
    weights   = 1.0 / (freq ** 0.5)
    weights   = np.clip(weights, 1.0, cap)
    weights   = weights / weights.mean()
    return weights.astype(np.float32)


def build_sequential_features(scores_col, n_windows=12):
    N = len(scores_col)
    assert N % n_windows == 0
    x     = scores_col.reshape(-1, n_windows)
    prev  = np.concatenate([x[:, :1], x[:, :-1]], axis=1)
    next_ = np.concatenate([x[:, 1:], x[:, -1:]], axis=1)
    mean  = np.repeat(x.mean(axis=1), n_windows)
    max_  = np.repeat(x.max(axis=1),  n_windows)
    std   = np.repeat(x.std(axis=1),  n_windows)
    return prev.reshape(-1), next_.reshape(-1), mean, max_, std


def train_mlp_probes(emb, scores_raw, Y, min_pos=5, pca_dim=64, alpha_blend=0.4):
    """
    CHANGE 1: Upgraded MLP probe.
    - pca_dim: 32 ->64  (more embedding information)
    - hidden:  (32,) ->(128, 64)  (more capacity)
    - max_iter: 100 ->300  (longer training)
    - min_pos: 8 ->5  (catches more rare species)
    """
    scaler = StandardScaler()
    emb_s  = scaler.fit_transform(emb)
    pca    = PCA(n_components=min(pca_dim, emb_s.shape[1] - 1))
    Z      = pca.fit_transform(emb_s).astype(np.float32)
    print(f"Embedding: {emb.shape} ->PCA: {Z.shape}  "
          f"(variance retained: {pca.explained_variance_ratio_.sum():.2%})")

    class_weights = build_class_freq_weights(Y, cap=10.0)

    probe_models = {}
    active = np.where(Y.sum(axis=0) >= min_pos)[0]
    print(f"Training MLP probes for {len(active)} species (>= {min_pos} pos windows)...")

    MAX_ROWS = int(TUNE["mlp_probe_max_rows"])
    hidden1 = int(TUNE["mlp_probe_hidden1"])
    hidden2 = int(TUNE["mlp_probe_hidden2"])
    max_iter = int(TUNE["mlp_probe_max_iter"])
    n_iter_no_change = int(TUNE["mlp_probe_n_iter_no_change"])
    lr_init = float(TUNE["mlp_probe_lr_init"])
    reg_alpha = float(TUNE["mlp_probe_alpha"])
    max_repeat = int(TUNE["mlp_probe_max_repeat"])

    for ci in tqdm(active, desc="MLP probes"):
        y = Y[:, ci]
        if y.sum() == 0 or y.sum() == len(y):
            continue

        prev, next_, mean, max_, std = build_sequential_features(scores_raw[:, ci])
        X = np.hstack([
            Z,
            scores_raw[:, ci:ci+1],
            prev[:, None], next_[:, None],
            mean[:, None], max_[:, None], std[:, None],
        ])

        n_pos = int(y.sum()); n_neg = len(y) - n_pos
        pos_idx = np.where(y == 1)[0]

        w      = float(class_weights[ci])
        repeat = max(1, int(round(w * n_neg / max(n_pos, 1))))
        repeat = min(repeat, max_repeat)
        if n_pos * repeat + len(y) > MAX_ROWS:
            repeat = max(1, (MAX_ROWS - len(y)) // max(n_pos, 1))

        X_bal = np.vstack([X, np.tile(X[pos_idx], (repeat, 1))])
        y_bal = np.concatenate([y, np.ones(n_pos * repeat, dtype=y.dtype)])

        clf = MLPClassifier(
            hidden_layer_sizes=(hidden1, hidden2),
            activation="relu",
            max_iter=max_iter,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=n_iter_no_change,
            random_state=GLOBAL_SEED,
            learning_rate_init=lr_init,
            alpha=reg_alpha,
        )
        clf.fit(X_bal, y_bal)
        probe_models[ci] = clf

    print(f"Trained {len(probe_models)} MLP probes")
    return probe_models, scaler, pca, alpha_blend


def apply_mlp_probes(emb_test, scores_test, probe_models, scaler, pca, alpha_blend=0.4):
    emb_s  = scaler.transform(emb_test)
    Z_test = pca.transform(emb_s).astype(np.float32)
    result = scores_test.copy()
    for ci, clf in probe_models.items():
        prev, next_, mean, max_, std = build_sequential_features(scores_test[:, ci])
        X_test = np.hstack([
            Z_test, scores_test[:, ci:ci+1],
            prev[:, None], next_[:, None],
            mean[:, None], max_[:, None], std[:, None],
        ])
        prob  = clf.predict_proba(X_test)[:, 1].astype(np.float32)
        logit = np.log(prob + 1e-7) - np.log(1 - prob + 1e-7)
        result[:, ci] = (1 - alpha_blend) * scores_test[:, ci] + alpha_blend * logit
    return result

print(
    "[OK] CHANGE 1: Upgraded MLP probe "
    f"(pca_dim={TUNE['mlp_pca_dim']}, hidden=({TUNE['mlp_probe_hidden1']},{TUNE['mlp_probe_hidden2']}), "
    f"max_iter={TUNE['mlp_probe_max_iter']}, min_pos={TUNE['mlp_min_pos']})"
)

# ===== Cell 15 =====
# Vectorized MLP probe inference using batched PyTorch matrix multiplies.
# Replaces the per-class Python loop at inference time for a 10-50x speedup.
import torch
import torch.nn as nn

torch.manual_seed(GLOBAL_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(GLOBAL_SEED)

TORCH_DEVICE = torch.device(
    "cuda" if (USE_GPU and torch.cuda.is_available()) else "cpu"
)
print(f"Torch device: {TORCH_DEVICE}")


class PerchAdapterHeadLegacy(nn.Module):
    """Legacy 2-layer residual adapter."""

    def __init__(
        self,
        input_dim=1536 + 234,
        hidden_dim=512,
        output_dim=234,
        dropout=0.2,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class PerchAdapterHeadLinear(nn.Module):
    """Single-layer linear residual adapter."""

    def __init__(
        self,
        input_dim=1536 + 234,
        output_dim=234,
    ):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x)


class PerchAdapterHeadGated(nn.Module):
    """3-layer gated residual adapter.

    delta = sigmoid(gate(h2)) * proj(h2)
    """

    def __init__(
        self,
        input_dim=1536 + 234,
        hidden_dim=512,
        hidden_dim2=256,
        output_dim=234,
        dropout=0.2,
        gate_bias=-2.0,
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

        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, float(gate_bias))

    def forward(self, x):
        h = self.dropout(self.act(self.norm1(self.fc1(x))))
        h = self.dropout(self.act(self.norm2(self.fc2(h))))
        g = torch.sigmoid(self.gate(h))
        return g * self.delta(h)


class PerchAdapterHeadSepGated(nn.Module):
    """Two-layer separated gated residual adapter.

    delta = gate(emb) * scaled_logits + bias(emb)
    """

    def __init__(
        self,
        emb_dim=1536,
        output_dim=234,
        hidden_dim=512,
        dropout=0.2,
    ):
        super().__init__()
        self.emb_dim = int(emb_dim)
        self.output_dim = int(output_dim)
        self.emb_ln = nn.LayerNorm(self.emb_dim)
        self.logit_ln = nn.LayerNorm(self.output_dim)
        self.gate_mlp = nn.Sequential(
            nn.Linear(self.emb_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.output_dim),
        )
        self.bias_mlp = nn.Sequential(
            nn.Linear(self.emb_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.output_dim),
        )
        self.logit_scale = nn.Parameter(torch.zeros(self.output_dim))

        nn.init.zeros_(self.gate_mlp[-1].weight)
        nn.init.constant_(self.gate_mlp[-1].bias, -2.0)
        nn.init.zeros_(self.bias_mlp[-1].weight)
        nn.init.zeros_(self.bias_mlp[-1].bias)

    def forward(self, x):
        emb = x[:, : self.emb_dim]
        logits_in = x[:, self.emb_dim : self.emb_dim + self.output_dim]
        emb_n = self.emb_ln(emb)
        logits_n = self.logit_ln(logits_in)
        gate = torch.sigmoid(self.gate_mlp(emb_n))
        bias = self.bias_mlp(emb_n)
        scaled_logits = logits_n * torch.tanh(self.logit_scale)[None, :]
        return gate * scaled_logits + bias


class PerchEmbeddingClassifier(nn.Module):
    """Classifier head using only Perch embeddings."""

    def __init__(
        self,
        emb_dim=1536,
        output_dim=234,
        hidden_dim=512,
        dropout=0.2,
        arch="mlp2",
    ):
        super().__init__()
        self.arch = str(arch).lower()
        if self.arch == "linear":
            self.net = nn.Linear(emb_dim, output_dim)
        elif self.arch == "mlp2":
            self.net = nn.Sequential(
                nn.LayerNorm(emb_dim),
                nn.Linear(emb_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )
        else:
            raise ValueError(f"Unknown embedding classifier arch: {arch}")

    def forward(self, emb):
        return self.net(emb)


class PerchFeatureClassifier(nn.Module):
    """Classifier using concatenated Perch embedding and Perch logits."""

    def __init__(self, input_dim=1770, output_dim=234, hidden_dim=512, dropout=0.2, arch="mlp2"):
        super().__init__()
        self.arch = str(arch).lower()
        if self.arch == "linear":
            self.net = nn.Linear(input_dim, output_dim)
        elif self.arch == "mlp2":
            self.net = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )
        else:
            raise ValueError(f"Unknown feature classifier arch: {arch}")

    def forward(self, x):
        return self.net(x)


def _sanitize_state_dict_local(state_dict):
    cleaned = {}
    for k, v in state_dict.items():
        if k == "n_averaged":
            continue
        if k.startswith("module."):
            cleaned[k[len("module."):]] = v
        else:
            cleaned[k] = v
    return cleaned


def _detect_adapter_arch_from_ckpt(ckpt):
    arch = str(ckpt.get("adapter_arch", "")).strip().lower()
    if arch:
        return arch
    state_keys = list(ckpt.get("adapter_state_dict", {}).keys())
    has_linear_keys = any(k.startswith("linear.") for k in state_keys)
    if has_linear_keys:
        return "linear"
    has_sep_keys = any(
        k.startswith("emb_ln.")
        or k.startswith("logit_ln.")
        or k.startswith("gate_mlp.")
        or k.startswith("bias_mlp.")
        or k.startswith("logit_scale")
        for k in state_keys
    )
    if has_sep_keys:
        return "sep_gated2"
    has_gated_keys = any(
        k.startswith("fc1.")
        or k.startswith("fc2.")
        or k.startswith("delta.")
        or k.startswith("gate.")
        for k in state_keys
    )
    return "mlp3_gated" if has_gated_keys else "mlp2_legacy"


def _load_perch_adapter_from_env():
    if not PERCH_ADAPTER_CKPT_RAW:
        return None
    ckpt_path = Path(PERCH_ADAPTER_CKPT_RAW)
    if not ckpt_path.exists():
        print(f"[WARN] Perch adapter ckpt not found: {ckpt_path}. Skip adapter.")
        return None

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_labels = ckpt.get("primary_labels")
    if ckpt_labels is not None and list(ckpt_labels) != list(PRIMARY_LABELS):
        raise RuntimeError(
            "Perch adapter label space mismatch with current sample_submission."
        )

    arch = _detect_adapter_arch_from_ckpt(ckpt)
    input_dim = int(ckpt.get("input_dim", 1536 + N_CLASSES))
    hidden_dim = int(ckpt.get("hidden_dim", 512))
    output_dim = int(ckpt.get("output_dim", N_CLASSES))
    dropout = float(ckpt.get("dropout", 0.2))
    hidden_dim2 = int(ckpt.get("hidden_dim2", max(128, hidden_dim // 2)))
    gate_bias = float(ckpt.get("gate_bias", -2.0))
    emb_dim = int(ckpt.get("emb_dim", max(1, input_dim - output_dim)))
    adapter_class_weights = ckpt.get("adapter_class_weights")

    if arch == "linear":
        model = PerchAdapterHeadLinear(
            input_dim=input_dim,
            output_dim=output_dim,
        ).to(TORCH_DEVICE)
    elif arch == "sep_gated2":
        model = PerchAdapterHeadSepGated(
            emb_dim=emb_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        ).to(TORCH_DEVICE)
    elif arch == "mlp3_gated":
        model = PerchAdapterHeadGated(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            hidden_dim2=hidden_dim2,
            output_dim=output_dim,
            dropout=dropout,
            gate_bias=gate_bias,
        ).to(TORCH_DEVICE)
    else:
        arch = "mlp2_legacy"
        model = PerchAdapterHeadLegacy(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
        ).to(TORCH_DEVICE)

    model.load_state_dict(_sanitize_state_dict_local(ckpt["adapter_state_dict"]), strict=True)
    if adapter_class_weights is not None:
        adapter_class_weights = np.asarray(adapter_class_weights, dtype=np.float32)
        if adapter_class_weights.shape[0] != output_dim:
            raise RuntimeError(
                f"Perch adapter class weights dim {adapter_class_weights.shape[0]} "
                f"does not match output_dim {output_dim}."
            )
        model.adapter_class_weights_np = adapter_class_weights
    else:
        model.adapter_class_weights_np = None
    model.eval()
    class_weight_msg = ""
    if getattr(model, "adapter_class_weights_np", None) is not None:
        w_arr = model.adapter_class_weights_np
        class_weight_msg = (
            f", class_w_mean={float(w_arr.mean()):.3f}, "
            f"class_w_range=[{float(w_arr.min()):.3f},{float(w_arr.max()):.3f}]"
        )
    print(
        f"[OK] Loaded Perch adapter: {ckpt_path} "
        f"(arch={arch}, hidden={hidden_dim}, hidden2={hidden_dim2}, emb_dim={emb_dim}, "
        f"weight={PERCH_ADAPTER_WEIGHT}{class_weight_msg})"
    )
    return model


def _apply_perch_adapter(
    scores_raw,
    emb_raw,
    adapter_model,
    weight=1.0,
    per_class_weight=None,
    batch_size=2048,
):
    if adapter_model is None or abs(weight) < 1e-12:
        return scores_raw
    delta = _predict_perch_adapter_delta(scores_raw, emb_raw, adapter_model, batch_size=batch_size)
    if per_class_weight is not None:
        w = np.asarray(per_class_weight, dtype=np.float32)
        if w.shape[0] != delta.shape[1]:
            raise RuntimeError(
                f"Per-class adapter weight dim {w.shape[0]} does not match n_classes {delta.shape[1]}."
            )
        delta = delta * w[None, :]
    return scores_raw + float(weight) * delta


def _predict_perch_adapter_delta(scores_raw, emb_raw, adapter_model, batch_size=2048):
    if adapter_model is None:
        return np.zeros_like(scores_raw, dtype=np.float32)
    feat = np.concatenate([emb_raw, scores_raw], axis=1).astype(np.float32, copy=False)
    out = np.empty_like(scores_raw, dtype=np.float32)
    n = len(feat)
    for i in range(0, n, batch_size):
        j = min(n, i + batch_size)
        x = torch.from_numpy(feat[i:j]).to(TORCH_DEVICE)
        with torch.no_grad():
            delta = adapter_model(x).detach().cpu().numpy().astype(np.float32)
        class_weights = getattr(adapter_model, "adapter_class_weights_np", None)
        if class_weights is not None:
            delta = delta * class_weights[None, :]
        out[i:j] = delta
    return out


def _auto_tune_perch_adapter_weight(scores_raw, emb_raw, y_true, adapter_model):
    if adapter_model is None:
        return float(PERCH_ADAPTER_WEIGHT)
    delta = _predict_perch_adapter_delta(scores_raw, emb_raw, adapter_model)
    best_w = float(PERCH_ADAPTER_WEIGHT_GRID[0])
    best_auc = -1.0
    rows = []
    for w in PERCH_ADAPTER_WEIGHT_GRID:
        logits = scores_raw + float(w) * delta
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
        auc = macro_auc(y_true, probs)
        rows.append(f"{float(w):.3g}:{auc:.6f}")
        if auc > best_auc + 1e-12:
            best_auc = float(auc)
            best_w = float(w)
    print("[INFO] Adapter weight soundscape search: " + " ".join(rows))
    print(f"[INFO] Adapter weight selected on train_soundscapes: {best_w:.4g} (auc={best_auc:.6f})")
    return best_w


def _auto_tune_perch_adapter_per_class_weight(scores_raw, emb_raw, y_true, adapter_model):
    if adapter_model is None:
        return None
    delta = _predict_perch_adapter_delta(scores_raw, emb_raw, adapter_model)
    n_classes = scores_raw.shape[1]
    out = np.ones(n_classes, dtype=np.float32)
    searched = 0
    for c in range(n_classes):
        yc = y_true[:, c].astype(np.int32)
        pos = int(yc.sum())
        neg = int(len(yc) - pos)
        if pos < 3 or neg < 3:
            continue
        base = scores_raw[:, c]
        d = delta[:, c]
        best_w = 1.0
        best_auc = -1.0
        for w in PERCH_ADAPTER_PER_CLASS_WEIGHT_GRID:
            logits = base + float(w) * d
            probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
            try:
                auc = roc_auc_score(yc, probs)
            except Exception:
                continue
            if auc > best_auc + 1e-12:
                best_auc = float(auc)
                best_w = float(w)
        out[c] = best_w
        searched += 1
    print(
        f"[INFO] Per-class adapter weight tuned for {searched}/{n_classes} classes "
        f"| mean={float(out.mean()):.4f} min={float(out.min()):.4f} max={float(out.max()):.4f}"
    )
    return out


def _auto_tune_ensemble_weight(proto_logits, mlp_logits, y_true):
    best_w = float(ENSEMBLE_W_GRID[0])
    best_auc = -1.0
    rows = []
    for w in ENSEMBLE_W_GRID:
        logits = float(w) * proto_logits + (1.0 - float(w)) * mlp_logits
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
        auc = macro_auc(y_true, probs)
        rows.append(f"{float(w):.3g}:{auc:.6f}")
        if auc > best_auc + 1e-12:
            best_auc = float(auc)
            best_w = float(w)
    print("[INFO] Ensemble weight search: " + " ".join(rows))
    print(f"[INFO] Ensemble weight selected: {best_w:.4g} (auc={best_auc:.6f})")
    return best_w


def _auto_tune_ensemble_weight_per_class(proto_logits, mlp_logits, y_true, default_w):
    n_classes = proto_logits.shape[1]
    out = np.full(n_classes, float(default_w), dtype=np.float32)
    tuned = 0
    for c in range(n_classes):
        yc = y_true[:, c].astype(np.int32)
        pos = int(yc.sum())
        neg = int(len(yc) - pos)
        if pos < int(ENSEMBLE_PER_CLASS_MIN_POS) or neg < int(ENSEMBLE_PER_CLASS_MIN_POS):
            continue
        p = proto_logits[:, c]
        m = mlp_logits[:, c]
        best_w = float(default_w)
        best_auc = -1.0
        for w in ENSEMBLE_W_GRID:
            logits = float(w) * p + (1.0 - float(w)) * m
            probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
            try:
                auc = roc_auc_score(yc, probs)
            except Exception:
                continue
            if auc > best_auc + 1e-12:
                best_auc = float(auc)
                best_w = float(w)
        shrink = float(np.clip(ENSEMBLE_PER_CLASS_SHRINK, 0.0, 1.0))
        out[c] = np.float32(shrink * best_w + (1.0 - shrink) * float(default_w))
        tuned += 1
    print(
        f"[INFO] Per-class ensemble weight tuned for {tuned}/{n_classes} classes "
        f"| mean={float(out.mean()):.4f} min={float(out.min()):.4f} max={float(out.max()):.4f} "
        f"| min_pos>={int(ENSEMBLE_PER_CLASS_MIN_POS)} shrink={float(ENSEMBLE_PER_CLASS_SHRINK):.2f}"
    )
    return out


def _blend_ensemble_logits(proto_logits, mlp_logits, weight):
    if np.isscalar(weight):
        w = float(weight)
        return (w * proto_logits + (1.0 - w) * mlp_logits).astype(np.float32, copy=False)
    w_arr = np.asarray(weight, dtype=np.float32)
    if w_arr.ndim != 1 or w_arr.shape[0] != proto_logits.shape[1]:
        raise RuntimeError(
            f"Ensemble per-class weight shape mismatch: got {w_arr.shape}, "
            f"expected ({proto_logits.shape[1]},)."
        )
    return (proto_logits * w_arr[None, :] + mlp_logits * (1.0 - w_arr[None, :])).astype(np.float32, copy=False)


def train_oof_logit_stacker(proto_logits, mlp_logits, y_true, min_pos=6, c_value=1.0):
    n_classes = proto_logits.shape[1]
    coef = np.zeros((n_classes, 2), dtype=np.float32)
    bias = np.zeros(n_classes, dtype=np.float32)
    default_coef = np.array([0.5, 0.5], dtype=np.float32)
    tuned = 0
    for c in range(n_classes):
        yc = y_true[:, c].astype(np.int32)
        pos = int(yc.sum())
        neg = int(len(yc) - pos)
        if pos < int(min_pos) or neg < int(min_pos):
            coef[c] = default_coef
            bias[c] = 0.0
            continue
        x = np.stack([proto_logits[:, c], mlp_logits[:, c]], axis=1).astype(np.float32, copy=False)
        try:
            clf = LogisticRegression(
                penalty="l2",
                C=float(c_value),
                solver="liblinear",
                max_iter=300,
            )
            clf.fit(x, yc)
            coef[c] = clf.coef_[0].astype(np.float32)
            bias[c] = np.float32(clf.intercept_[0])
            tuned += 1
        except Exception:
            coef[c] = default_coef
            bias[c] = 0.0
    print(
        f"[INFO] OOF stacker trained for {tuned}/{n_classes} classes "
        f"| min_pos>={int(min_pos)} C={float(c_value):.4g}"
    )
    return {"coef": coef, "bias": bias}


def apply_logit_stacker(proto_logits, mlp_logits, stacker):
    if stacker is None:
        return _blend_ensemble_logits(proto_logits, mlp_logits, float(TUNE["ensemble_w"]))
    coef = np.asarray(stacker["coef"], dtype=np.float32)
    bias = np.asarray(stacker["bias"], dtype=np.float32)
    if coef.shape != (proto_logits.shape[1], 2):
        raise RuntimeError(
            f"Stacker coef shape mismatch: got {coef.shape}, expected ({proto_logits.shape[1]}, 2)"
        )
    if bias.shape != (proto_logits.shape[1],):
        raise RuntimeError(
            f"Stacker bias shape mismatch: got {bias.shape}, expected ({proto_logits.shape[1]},)"
        )
    out = (
        proto_logits * coef[:, 0][None, :]
        + mlp_logits * coef[:, 1][None, :]
        + bias[None, :]
    )
    return out.astype(np.float32, copy=False)


def _load_perch_embedding_classifier_from_env():
    if not PERCH_EMB_CLS_CKPT_RAW:
        return None
    ckpt_path = Path(PERCH_EMB_CLS_CKPT_RAW)
    if not ckpt_path.exists():
        print(f"[WARN] Perch embedding classifier ckpt not found: {ckpt_path}. Skip classifier.")
        return None

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_labels = ckpt.get("primary_labels")
    if ckpt_labels is not None and list(ckpt_labels) != list(PRIMARY_LABELS):
        raise RuntimeError(
            "Perch embedding classifier label space mismatch with current sample_submission."
        )

    state_dict = ckpt.get("classifier_state_dict")
    if state_dict is None:
        raise RuntimeError(
            f"Checkpoint {ckpt_path} does not contain classifier_state_dict. "
            "Train with --head-type embedding_classifier."
        )

    emb_dim = int(ckpt.get("emb_dim", 1536))
    output_dim = int(ckpt.get("output_dim", N_CLASSES))
    hidden_dim = int(ckpt.get("hidden_dim", 512))
    dropout = float(ckpt.get("dropout", 0.2))
    arch = str(ckpt.get("classifier_arch", "mlp2")).strip().lower() or "mlp2"

    model = PerchEmbeddingClassifier(
        emb_dim=emb_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        arch=arch,
    ).to(TORCH_DEVICE)
    model.load_state_dict(_sanitize_state_dict_local(state_dict), strict=True)
    model.eval()
    print(
        f"[OK] Loaded Perch embedding classifier: {ckpt_path} "
        f"(arch={arch}, hidden={hidden_dim}, emb_dim={emb_dim}, blend_weight={PERCH_EMB_CLS_WEIGHT})"
    )
    return model


def _apply_perch_embedding_classifier(scores_raw, emb_raw, classifier_model, weight=0.35, batch_size=2048):
    if classifier_model is None or abs(weight) < 1e-12:
        return scores_raw
    out = np.empty_like(scores_raw, dtype=np.float32)
    n = len(emb_raw)
    for i in range(0, n, batch_size):
        j = min(n, i + batch_size)
        emb = torch.from_numpy(emb_raw[i:j].astype(np.float32, copy=False)).to(TORCH_DEVICE)
        with torch.no_grad():
            cls_logits = classifier_model(emb).detach().cpu().numpy().astype(np.float32)
        out[i:j] = (1.0 - float(weight)) * scores_raw[i:j] + float(weight) * cls_logits
    return out


def _load_perch_mil_classifier_from_env():
    if not PERCH_MIL_CLS_CKPT_RAW:
        return None
    ckpt_path = Path(PERCH_MIL_CLS_CKPT_RAW)
    if not ckpt_path.exists():
        print(f"[WARN] Perch MIL classifier ckpt not found: {ckpt_path}. Skip classifier.")
        return None

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_labels = ckpt.get("primary_labels")
    if ckpt_labels is not None and list(ckpt_labels) != list(PRIMARY_LABELS):
        raise RuntimeError(
            "Perch MIL classifier label space mismatch with current sample_submission."
        )

    state_dict = ckpt.get("mil_classifier_state_dict")
    if state_dict is None:
        raise RuntimeError(
            f"Checkpoint {ckpt_path} does not contain mil_classifier_state_dict. "
            "Train with --head-type mil_classifier."
        )

    input_dim = int(ckpt.get("input_dim", 1536 + N_CLASSES))
    output_dim = int(ckpt.get("output_dim", N_CLASSES))
    hidden_dim = int(ckpt.get("hidden_dim", 512))
    dropout = float(ckpt.get("dropout", 0.2))
    arch = str(ckpt.get("classifier_arch", "mlp2")).strip().lower() or "mlp2"

    model = PerchFeatureClassifier(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        arch=arch,
    ).to(TORCH_DEVICE)
    model.load_state_dict(_sanitize_state_dict_local(state_dict), strict=True)
    model.eval()
    print(
        f"[OK] Loaded Perch MIL classifier: {ckpt_path} "
        f"(arch={arch}, input_dim={input_dim}, hidden={hidden_dim}, blend_weight={PERCH_MIL_CLS_WEIGHT})"
    )
    return model


def _apply_perch_mil_classifier(scores_raw, emb_raw, classifier_model, weight=0.35, batch_size=2048):
    if classifier_model is None or abs(weight) < 1e-12:
        return scores_raw
    cls_logits = _predict_perch_mil_classifier_logits(scores_raw, emb_raw, classifier_model, batch_size=batch_size)
    return ((1.0 - float(weight)) * scores_raw + float(weight) * cls_logits).astype(np.float32, copy=False)


def _predict_perch_mil_classifier_logits(scores_raw, emb_raw, classifier_model, batch_size=2048):
    if classifier_model is None:
        return np.zeros_like(scores_raw, dtype=np.float32)
    feat = np.concatenate([emb_raw, scores_raw], axis=1).astype(np.float32, copy=False)
    out = np.empty_like(scores_raw, dtype=np.float32)
    n = len(feat)
    for i in range(0, n, batch_size):
        j = min(n, i + batch_size)
        x = torch.from_numpy(feat[i:j]).to(TORCH_DEVICE)
        with torch.no_grad():
            cls_logits = classifier_model(x).detach().cpu().numpy().astype(np.float32)
        out[i:j] = cls_logits
    return out


def _auto_tune_perch_mil_classifier_weight(scores_raw, emb_raw, y_true, classifier_model):
    if classifier_model is None:
        return float(PERCH_MIL_CLS_WEIGHT)
    cls_logits = _predict_perch_mil_classifier_logits(scores_raw, emb_raw, classifier_model)
    best_w = float(PERCH_MIL_CLS_WEIGHT_GRID[0])
    best_auc = -1.0
    rows = []
    for w in PERCH_MIL_CLS_WEIGHT_GRID:
        logits = (1.0 - float(w)) * scores_raw + float(w) * cls_logits
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
        auc = macro_auc(y_true, probs)
        rows.append(f"{float(w):.3g}:{auc:.6f}")
        if auc > best_auc + 1e-12:
            best_auc = float(auc)
            best_w = float(w)
    print("[INFO] MIL classifier weight soundscape search: " + " ".join(rows))
    print(f"[INFO] MIL classifier weight selected on train_soundscapes: {best_w:.4g} (auc={best_auc:.6f})")
    return best_w


def _load_unmapped_head_from_env():
    if not UNMAPPED_HEAD_CKPT_RAW:
        return None
    ckpt_path = Path(UNMAPPED_HEAD_CKPT_RAW)
    if not ckpt_path.exists():
        print(f"[WARN] Unmapped head ckpt not found: {ckpt_path}. Skip unmapped head.")
        return None

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_labels = ckpt.get("primary_labels")
    if ckpt_labels is not None and list(ckpt_labels) != list(PRIMARY_LABELS):
        raise RuntimeError("Unmapped head label space mismatch with current sample_submission.")

    state_dict = ckpt.get("unmapped_head_state_dict")
    if state_dict is None:
        raise RuntimeError(
            f"Checkpoint {ckpt_path} does not contain unmapped_head_state_dict. "
            "Train with --head-type unmapped_head."
        )

    class_indices = np.asarray(ckpt.get("unmapped_class_indices", []), dtype=np.int32)
    if len(class_indices) == 0:
        raise RuntimeError(f"Checkpoint {ckpt_path} has no unmapped_class_indices.")
    if class_indices.min() < 0 or class_indices.max() >= N_CLASSES:
        raise RuntimeError(f"Checkpoint {ckpt_path} has out-of-range unmapped_class_indices.")

    emb_dim = int(ckpt.get("emb_dim", 1536))
    hidden_dim = int(ckpt.get("hidden_dim", 512))
    dropout = float(ckpt.get("dropout", 0.2))
    arch = str(ckpt.get("classifier_arch", "mlp2")).strip().lower() or "mlp2"

    model = PerchEmbeddingClassifier(
        emb_dim=emb_dim,
        output_dim=len(class_indices),
        hidden_dim=hidden_dim,
        dropout=dropout,
        arch=arch,
    ).to(TORCH_DEVICE)
    model.load_state_dict(_sanitize_state_dict_local(state_dict), strict=True)
    model.unmapped_class_indices_np = class_indices
    model.eval()
    print(
        f"[OK] Loaded Unmapped head: {ckpt_path} "
        f"(arch={arch}, hidden={hidden_dim}, emb_dim={emb_dim}, classes={len(class_indices)}, "
        f"blend_weight={UNMAPPED_HEAD_WEIGHT})"
    )
    return model


def _apply_unmapped_head(scores_raw, emb_raw, unmapped_model, weight=0.35, batch_size=2048):
    if unmapped_model is None or abs(weight) < 1e-12:
        return scores_raw
    class_indices = getattr(unmapped_model, "unmapped_class_indices_np", None)
    if class_indices is None or len(class_indices) == 0:
        return scores_raw
    out = scores_raw.copy()
    n = len(emb_raw)
    for i in range(0, n, batch_size):
        j = min(n, i + batch_size)
        emb = torch.from_numpy(emb_raw[i:j].astype(np.float32, copy=False)).to(TORCH_DEVICE)
        with torch.no_grad():
            logits = unmapped_model(emb).detach().cpu().numpy().astype(np.float32)
        out[i:j, class_indices] = (
            (1.0 - float(weight)) * scores_raw[i:j, class_indices]
            + float(weight) * logits
        )
    return out


PERCH_ADAPTER_MODEL = _load_perch_adapter_from_env()
if PERCH_ADAPTER_MODEL is not None and "sc_tr" in globals() and "emb_tr" in globals():
    t0 = time.time()
    if PERCH_ADAPTER_WEIGHT_AUTO and MODE == "train":
        PERCH_ADAPTER_WEIGHT = _auto_tune_perch_adapter_weight(
            sc_tr,
            emb_tr,
            Y_FULL_aligned,
            PERCH_ADAPTER_MODEL,
        )
    elif PERCH_ADAPTER_WEIGHT_AUTO:
        print(
            "[WARN] BC26_PERCH_ADAPTER_WEIGHT=auto requires train labels; "
            f"using default weight={PERCH_ADAPTER_WEIGHT} in {MODE} mode."
        )
    if PERCH_ADAPTER_PER_CLASS_WEIGHT_AUTO and MODE == "train":
        PERCH_ADAPTER_PER_CLASS_WEIGHT = _auto_tune_perch_adapter_per_class_weight(
            sc_tr,
            emb_tr,
            Y_FULL_aligned,
            PERCH_ADAPTER_MODEL,
        )
    elif PERCH_ADAPTER_PER_CLASS_WEIGHT_AUTO:
        print(
            "[WARN] BC26_PERCH_ADAPTER_PER_CLASS_WEIGHT=auto requires train labels; "
            f"skip in {MODE} mode."
        )
    sc_tr = _apply_perch_adapter(
        sc_tr,
        emb_tr,
        PERCH_ADAPTER_MODEL,
        weight=PERCH_ADAPTER_WEIGHT,
        per_class_weight=PERCH_ADAPTER_PER_CLASS_WEIGHT,
    )
    print(
        f"[OK] Applied Perch adapter to training cache in {time.time()-t0:.1f}s "
        f"| score range [{sc_tr.min():.3f}, {sc_tr.max():.3f}]"
    )

PERCH_EMB_CLS_MODEL = _load_perch_embedding_classifier_from_env()
if PERCH_EMB_CLS_MODEL is not None and "sc_tr" in globals() and "emb_tr" in globals():
    t0 = time.time()
    sc_tr = _apply_perch_embedding_classifier(
        sc_tr,
        emb_tr,
        PERCH_EMB_CLS_MODEL,
        weight=PERCH_EMB_CLS_WEIGHT,
    )
    print(
        f"[OK] Applied Perch embedding classifier to training cache in {time.time()-t0:.1f}s "
        f"| score range [{sc_tr.min():.3f}, {sc_tr.max():.3f}]"
    )

PERCH_MIL_CLS_MODEL = _load_perch_mil_classifier_from_env()
if PERCH_MIL_CLS_MODEL is not None and "sc_tr" in globals() and "emb_tr" in globals():
    t0 = time.time()
    if PERCH_MIL_CLS_WEIGHT_AUTO and MODE == "train":
        PERCH_MIL_CLS_WEIGHT = _auto_tune_perch_mil_classifier_weight(
            sc_tr,
            emb_tr,
            Y_FULL_aligned,
            PERCH_MIL_CLS_MODEL,
        )
    elif PERCH_MIL_CLS_WEIGHT_AUTO:
        print(
            "[WARN] BC26_PERCH_MIL_CLS_WEIGHT=auto requires train labels; "
            f"using default weight={PERCH_MIL_CLS_WEIGHT} in {MODE} mode."
        )
    sc_tr = _apply_perch_mil_classifier(
        sc_tr,
        emb_tr,
        PERCH_MIL_CLS_MODEL,
        weight=PERCH_MIL_CLS_WEIGHT,
    )
    print(
        f"[OK] Applied Perch MIL classifier to training cache in {time.time()-t0:.1f}s "
        f"| score range [{sc_tr.min():.3f}, {sc_tr.max():.3f}]"
    )

UNMAPPED_HEAD_MODEL = _load_unmapped_head_from_env()
if UNMAPPED_HEAD_MODEL is not None and "sc_tr" in globals() and "emb_tr" in globals():
    t0 = time.time()
    sc_tr = _apply_unmapped_head(
        sc_tr,
        emb_tr,
        UNMAPPED_HEAD_MODEL,
        weight=UNMAPPED_HEAD_WEIGHT,
    )
    print(
        f"[OK] Applied Unmapped head to training cache in {time.time()-t0:.1f}s "
        f"| score range [{sc_tr.min():.3f}, {sc_tr.max():.3f}]"
    )

class VectorizedMLPProbes(nn.Module):
    """Stacks all per-class MLP weights into a single batched PyTorch model.
    Replaces the slow Python for-loop over probe_models at inference time."""
    def __init__(self, probe_models):
        super().__init__()
        self.valid_classes = sorted(probe_models.keys())
        V = len(self.valid_classes)
        if V == 0:
            self.weights = nn.ParameterList()
            self.biases  = nn.ParameterList()
            self.n_layers = 0
            return

        sample = probe_models[self.valid_classes[0]]
        self.n_layers = len(sample.coefs_)
        self.weights  = nn.ParameterList()
        self.biases   = nn.ParameterList()

        for layer_idx in range(self.n_layers):
            W = np.stack([probe_models[c].coefs_[layer_idx]
                          for c in self.valid_classes], axis=0)
            b = np.stack([probe_models[c].intercepts_[layer_idx]
                          for c in self.valid_classes], axis=0)
            self.weights.append(nn.Parameter(
                torch.tensor(W, dtype=torch.float32), requires_grad=False))
            self.biases.append(nn.Parameter(
                torch.tensor(b, dtype=torch.float32), requires_grad=False))

    def forward(self, x):
        h = x
        for i in range(self.n_layers):
            h = torch.bmm(h, self.weights[i]) + self.biases[i].unsqueeze(1)
            if i < self.n_layers - 1:
                h = torch.relu(h)
        return h.squeeze(-1)


def apply_mlp_probes_vectorized(emb_test, scores_test, probe_models,
                                 scaler, pca, alpha_blend=0.4):
    """
    Drop-in replacement for apply_mlp_probes().
    Uses batched PyTorch matrix multiply instead of a Python for-loop -
    ~10-50x faster at inference time.
    """
    if len(probe_models) == 0:
        return scores_test.copy()

    emb_s  = scaler.transform(emb_test)
    Z_test = pca.transform(emb_s).astype(np.float32)

    valid_classes = sorted(probe_models.keys())
    V = len(valid_classes)
    N = len(scores_test)

    raw  = scores_test[:, valid_classes].T
    n_files = N // N_WINDOWS
    raw_view = raw.reshape(V, n_files, N_WINDOWS)
    prev = np.concatenate([raw_view[:, :, :1], raw_view[:, :, :-1]], axis=2).reshape(V, N)
    nxt  = np.concatenate([raw_view[:, :, 1:], raw_view[:, :, -1:]], axis=2).reshape(V, N)
    mean = np.repeat(raw_view.mean(axis=2), N_WINDOWS, axis=1)
    mx   = np.repeat(raw_view.max(axis=2),  N_WINDOWS, axis=1)
    std  = np.repeat(raw_view.std(axis=2),  N_WINDOWS, axis=1)

    scalar_feats = np.stack([raw, prev, nxt, mean, mx, std], axis=-1).astype(np.float32)

    Z_expanded = np.broadcast_to(Z_test, (V, N, Z_test.shape[1]))

    X_all = np.concatenate(
        [Z_expanded.astype(np.float32), scalar_feats], axis=-1)

    vec_probe = VectorizedMLPProbes(probe_models).to(TORCH_DEVICE)
    vec_probe.eval()
    with torch.no_grad():
        preds = vec_probe(torch.tensor(X_all, device=TORCH_DEVICE)).detach().cpu().numpy()

    result = scores_test.copy()
    base_valid = scores_test[:, valid_classes]
    result[:, valid_classes] = (
        (1.0 - alpha_blend) * base_valid +
        alpha_blend * preds.T
    )
    return result

print("[OK] Vectorized MLP probe inference defined")

# ===== Cell 16 =====
# Isotonic calibration fits a monotone regressor per class on OOF scores,
# then a grid search finds the F1-maximising decision threshold.
# apply_per_class_thresholds sharpens probabilities around those thresholds.
from sklearn.isotonic import IsotonicRegression

def calibrate_and_optimize_thresholds(oof_probs, Y_FULL,
                                       threshold_grid=None, n_windows=12,
                                       min_pos_files=3, default_threshold=0.5,
                                       bucketed=False,
                                       rare_pos_max=5,
                                       common_pos_min=20,
                                       threshold_grid_rare=None,
                                       threshold_grid_common=None):
    """
    CHANGE 2: For each species:
    1. Fit isotonic regression on OOF scores (calibrates overconfident/underconfident classes)
    2. Grid-search F1-optimal threshold over calibrated probs
    Returns: thresholds array of shape (n_classes,)
    """
    if threshold_grid is None:
        threshold_grid = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    
    n_samples, n_cls = oof_probs.shape
    thresholds = np.full(n_cls, float(default_threshold), dtype=np.float32)
    n_files    = n_samples // n_windows
    file_oof   = oof_probs.reshape(n_files, n_windows, n_cls).max(axis=1)
    file_y     = Y_FULL.reshape(n_files, n_windows, n_cls).max(axis=1)
    
    n_calibrated = 0
    for c in range(n_cls):
        y_true = file_y[:, c]
        y_prob = file_oof[:, c]
        if y_true.sum() < min_pos_files:
            continue
        try:
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(y_prob, y_true)
            y_cal = ir.transform(y_prob)
        except Exception:
            y_cal = y_prob
        
        best_f1, best_t = 0.0, 0.5
        grid = threshold_grid
        pos_count = int(y_true.sum())
        if bucketed:
            if pos_count <= int(rare_pos_max) and threshold_grid_rare is not None:
                grid = threshold_grid_rare
            elif pos_count >= int(common_pos_min) and threshold_grid_common is not None:
                grid = threshold_grid_common

        for t in grid:
            pred = (y_cal >= t).astype(int)
            tp = ((pred==1) & (y_true==1)).sum()
            fp = ((pred==1) & (y_true==0)).sum()
            fn = ((pred==0) & (y_true==1)).sum()
            prec = tp / (tp + fp + 1e-8)
            rec  = tp / (tp + fn + 1e-8)
            f1   = 2 * prec * rec / (prec + rec + 1e-8)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[c] = best_t
        n_calibrated += 1
    
    print(f"Calibrated {n_calibrated} classes")
    print(f"Mean threshold: {thresholds.mean():.3f}")
    print(f"Range: [{thresholds.min():.2f}, {thresholds.max():.2f}]")
    return thresholds


def apply_per_class_thresholds(scores, thresholds):
    """
    Sharpens probabilities around the per-class threshold:
    - above threshold ->push toward 1
    - below threshold ->push toward 0
    """
    C = scores.shape[1]
    assert C == len(thresholds)
    scaled = np.copy(scores)
    for c in range(C):
        t = thresholds[c]
        above = scores[:, c] > t
        scaled[ above, c] = 0.5 + 0.5 * (scores[ above, c] - t) / (1 - t + 1e-8)
        scaled[~above, c] = 0.5 * scores[~above, c] / (t + 1e-8)
    return np.clip(scaled, 0.0, 1.0)

print("[OK] CHANGE 2: Isotonic calibration + per-class threshold optimization defined")

# ===== Cell 17 =====
# Rank-aware scaling: multiplies each window by the file's single peak confidence
# raised to a given power. Suppresses uncertain files more aggressively than
# top-2 mean scaling (file_confidence_scale) because it asks whether ANY
# window shows strong evidence rather than how consistently it fires.
def rank_aware_scaling(probs, n_windows=12, power=0.4):
    """
    CHANGE 6: Scale each window by the file's single peak confidence.

    How it works:
      1. For each file, find the MAX score across all 12 windows (per species)
      2. Raise it to power ->scale factor
      3. Multiply every window's score by that scale factor

    Example for one species across 12 windows:
      Confident file:  max=0.90 ->scale=0.90^0.4=0.96 ->mild boost
      Uncertain file:  max=0.10 ->scale=0.10^0.4=0.40 ->strong suppression

    How this differs from Change 3 (file_confidence_scale):
      Change 3 uses top-2 MEAN ->smoother, less aggressive
      Change 6 uses single MAX  ->asks "does ANY window have strong evidence?"

    power=0.0 ->no effect (baseline)
    power=0.4 ->moderate suppression of uncertain files (recommended start)
    power=1.0 ->multiply directly by file max (very aggressive)
    """
    N, C = probs.shape
    assert N % n_windows == 0, f"Expected multiple of {n_windows}, got {N}"

    view     = probs.reshape(-1, n_windows, C)
    file_max = view.max(axis=1, keepdims=True)

    scale  = np.power(file_max, power)
    scaled = view * scale

    return scaled.reshape(N, C)


print("[OK] Rank-aware scaling defined")

# ===== Cell 18 =====
# Adaptive delta smoothing: blends uncertain windows toward their neighbours
# while leaving confident windows nearly unchanged.
# The smoothing strength scales with (1 - window_confidence), so high-scoring
# windows are protected and low-scoring noisy windows are attenuated.
def adaptive_delta_smooth(probs, n_windows=12, base_alpha=0.20):
    """
    CHANGE 7: Smooth uncertain windows toward their neighbors,
    while leaving confident windows almost untouched.

    How it works:
      For each window t:
        conf  = max probability across all 234 species at window t
        alpha = base_alpha * (1 - conf)   ->KEY: adapts to confidence
        new[t] = (1 - alpha) * old[t] + alpha * avg(old[t-1], old[t+1])

    Why alpha adapts to confidence:
      Confident window (max=0.90):
        alpha = 0.20 * (1 - 0.90) = 0.02  ->barely smoothed, peak preserved
      Uncertain window (max=0.10):
        alpha = 0.20 * (1 - 0.10) = 0.18  ->smoothed more, noise reduced

    This is exactly why your Change 1 hurt (-0.005) but this one should help:
      Change 1 used fixed alpha=0.3 ->diluted confident peaks equally
      Change 7 uses adaptive alpha  ->protects confident peaks, smooths noise

    base_alpha=0.0  ->no smoothing (baseline)
    base_alpha=0.20 ->recommended starting point
    """
    N, C = probs.shape
    assert N % n_windows == 0, f"Expected multiple of {n_windows}, got {N}"

    result = probs.copy()
    view   = probs.reshape(-1, n_windows, C)
    out    = result.reshape(-1, n_windows, C)

    for t in range(n_windows):

        conf = view[:, t, :].max(axis=-1, keepdims=True)

        alpha = base_alpha * (1.0 - conf)

        if t == 0:
            neighbor_avg = (view[:, t, :] + view[:, t+1, :]) / 2.0
        elif t == n_windows - 1:
            neighbor_avg = (view[:, t-1, :] + view[:, t, :]) / 2.0
        else:
            neighbor_avg = (view[:, t-1, :] + view[:, t+1, :]) / 2.0

        out[:, t, :] = (1.0 - alpha) * view[:, t, :] + alpha * neighbor_avg

    return result


print("[OK] Adaptive delta smoothing defined")

# ===== Cell 19 =====
# LightProtoSSM: a bidirectional selective-state-space model with learnable
# class prototypes, site/hour meta embeddings, optional cross-attention between
# SSM layers, and a learnable per-class fusion weight with Perch logits.
# train_light_proto_ssm wraps training with SWA and OneCycleLR scheduling.
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveSSM(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.in_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.conv1d = nn.Conv1d(
            d_model, d_model, d_conv, padding=d_conv - 1, groups=d_model
        )
        self.dt_proj = nn.Linear(d_model, d_model, bias=True)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(
            d_model, -1
        )
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_model))
        self.B_proj = nn.Linear(d_model, d_state, bias=False)
        self.C_proj = nn.Linear(d_model, d_state, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B_sz, T, D = x.shape
        xz = self.in_proj(x)
        x_ssm, z = xz.chunk(2, dim=-1)

        x_conv = self.conv1d(x_ssm.transpose(1, 2))[:, :, :T].transpose(1, 2)
        x_conv = F.silu(x_conv)

        dt = F.softplus(self.dt_proj(x_conv))
        A = -torch.exp(self.A_log)
        B = self.B_proj(x_conv)
        C = self.C_proj(x_conv)

        h = torch.zeros(B_sz, D, self.d_state, device=x.device, dtype=x.dtype)
        ys = []

        for t in range(T):
            dA = torch.exp(A[None] * dt[:, t, :, None])
            dB = dt[:, t, :, None] * B[:, t, None, :]
            h = h * dA + x[:, t, :, None] * dB
            ys.append((h * C[:, t, None, :]).sum(-1))

        y = torch.stack(ys, dim=1)
        return y + x * self.D[None, None, :]


class LightProtoSSM(nn.Module):
    """
    CHANGE 4: LightProtoSSM with cross-attention between SSM layers.
    """

    def __init__(
        self,
        d_input=1536,
        d_model=128,
        d_state=16,
        n_classes=234,
        n_windows=12,
        dropout=0.15,
        n_sites=20,
        meta_dim=16,
        use_cross_attn=True,
        cross_attn_heads=2,
        n_ssm_layers=2,
    ):
        super().__init__()

        self.n_classes = n_classes
        self.n_windows = n_windows
        self.use_cross_attn = use_cross_attn
        self.n_ssm_layers = n_ssm_layers

        self.input_proj = nn.Sequential(
            nn.Linear(d_input, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.pos_enc = nn.Parameter(torch.randn(1, n_windows, d_model) * 0.02)
        self.site_emb = nn.Embedding(n_sites, meta_dim)
        self.hour_emb = nn.Embedding(24, meta_dim)
        self.meta_proj = nn.Linear(2 * meta_dim, d_model)

        self.ssm_fwd = nn.ModuleList(
            [SelectiveSSM(d_model, d_state) for _ in range(n_ssm_layers)]
        )
        self.ssm_bwd = nn.ModuleList(
            [SelectiveSSM(d_model, d_state) for _ in range(n_ssm_layers)]
        )
        self.ssm_merge = nn.ModuleList(
            [nn.Linear(2 * d_model, d_model) for _ in range(n_ssm_layers)]
        )
        self.ssm_norm = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(n_ssm_layers)]
        )
        self.drop = nn.Dropout(dropout)

        if use_cross_attn:
            self.cross_attn = nn.ModuleList(
                [
                    nn.MultiheadAttention(
                        d_model,
                        num_heads=cross_attn_heads,
                        dropout=dropout,
                        batch_first=True,
                    )
                    for _ in range(n_ssm_layers)
                ]
            )
            self.cross_norm = nn.ModuleList(
                [nn.LayerNorm(d_model) for _ in range(n_ssm_layers)]
            )

        self.prototypes = nn.Parameter(
            torch.randn(n_classes, d_model) * 0.02
        )
        self.proto_temp = nn.Parameter(torch.tensor(5.0))
        self.class_bias = nn.Parameter(torch.zeros(n_classes))
        self.fusion_alpha = nn.Parameter(torch.zeros(n_classes))

    def init_prototypes(self, emb_tensor, labels_tensor):
        with torch.no_grad():
            h = self.input_proj(emb_tensor)
            for c in range(self.n_classes):
                mask = labels_tensor[:, c] > 0.5
                if mask.sum() > 0:
                    self.prototypes.data[c] = F.normalize(
                        h[mask].mean(0), dim=0
                    )

    def forward(self, emb, perch_logits=None, site_ids=None, hours=None):
        B, T, _ = emb.shape

        h = self.input_proj(emb) + self.pos_enc[:, :T, :]

        if site_ids is not None and hours is not None:
            meta = self.meta_proj(
                torch.cat(
                    [self.site_emb(site_ids), self.hour_emb(hours)], dim=-1
                )
            )
            h = h + meta[:, None, :]

        for i, (fwd, bwd, merge, norm) in enumerate(
            zip(
                self.ssm_fwd,
                self.ssm_bwd,
                self.ssm_merge,
                self.ssm_norm,
            )
        ):
            res = h
            h_f = fwd(h)
            h_b = bwd(h.flip(1)).flip(1)

            h = self.drop(merge(torch.cat([h_f, h_b], dim=-1)))
            h = norm(h + res)

            if self.use_cross_attn:
                attn_out, _ = self.cross_attn[i](h, h, h)
                h = self.cross_norm[i](h + attn_out)

        h_n = F.normalize(h, dim=-1)
        p_n = F.normalize(self.prototypes, dim=-1)

        sim = (
            torch.matmul(h_n, p_n.T)
            * F.softplus(self.proto_temp)
            + self.class_bias[None, None, :]
        )

        if perch_logits is not None:
            alpha = torch.sigmoid(self.fusion_alpha)[None, None, :]
            out = alpha * sim + (1 - alpha) * perch_logits
        else:
            out = sim

        return out

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def train_light_proto_ssm(
    emb_full,
    scores_full,
    Y_full,
    meta_full,
    n_epochs=40,
    patience=8,
    lr=1e-3,
    d_model=128,
    d_state=16,
    meta_dim=16,
    dropout=0.15,
    n_layers=2,
    cross_attn_heads=2,
    n_sites=20,
    verbose=False,
):
    """Train LightProtoSSM with cross-attention."""

    n_files = len(emb_full) // N_WINDOWS
    emb_f = emb_full.reshape(n_files, N_WINDOWS, -1)
    log_f = scores_full.reshape(n_files, N_WINDOWS, -1)
    lab_f = Y_full.reshape(n_files, N_WINDOWS, -1).astype(np.float32)

    fnames = meta_full["filename"].unique()
    sites_u = sorted(meta_full["site"].unique())
    site2i = {s: i + 1 for i, s in enumerate(sites_u)}

    site_ids = np.array(
        [
            min(
                site2i.get(
                    meta_full.loc[
                        meta_full["filename"] == fn, "site"
                    ].iloc[0],
                    0,
                ),
                n_sites - 1,
            )
            for fn in fnames
        ],
        dtype=np.int64,
    )

    hour_ids = np.array(
        [
            int(
                meta_full.loc[
                    meta_full["filename"] == fn, "hour_utc"
                ].iloc[0]
            )
            % 24
            for fn in fnames
        ],
        dtype=np.int64,
    )

    model = LightProtoSSM(
        d_model=d_model,
        d_state=d_state,
        n_classes=N_CLASSES,
        n_sites=n_sites,
        meta_dim=meta_dim,
        dropout=dropout,
        use_cross_attn=True,
        cross_attn_heads=cross_attn_heads,
        n_ssm_layers=n_layers,
    ).to(TORCH_DEVICE)

    model.init_prototypes(
        torch.tensor(emb_full, dtype=torch.float32, device=TORCH_DEVICE),
        torch.tensor(Y_full, dtype=torch.float32, device=TORCH_DEVICE),
    )

    print(
        f"LightProtoSSM params: {model.count_parameters():,} "
        f"(d_model={d_model}, d_state={d_state}, meta_dim={meta_dim}, "
        f"dropout={dropout}, layers={n_layers}, heads={cross_attn_heads})"
    )

    emb_t = torch.tensor(emb_f, dtype=torch.float32, device=TORCH_DEVICE)
    log_t = torch.tensor(log_f, dtype=torch.float32, device=TORCH_DEVICE)
    lab_t = torch.tensor(lab_f, dtype=torch.float32, device=TORCH_DEVICE)
    site_t = torch.tensor(site_ids, dtype=torch.long, device=TORCH_DEVICE)
    hour_t = torch.tensor(hour_ids, dtype=torch.long, device=TORCH_DEVICE)

    pos_cnt = lab_t.sum(dim=(0, 1))
    total = lab_t.shape[0] * lab_t.shape[1]
    pos_weight = ((total - pos_cnt) / (pos_cnt + 1)).clamp(max=25.0)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=lr,
        epochs=n_epochs,
        steps_per_epoch=1,
        pct_start=0.1,
        anneal_strategy="cos",
    )

    best_loss, best_state, wait = float("inf"), None, 0
    effective_patience = n_epochs + 1 if DISABLE_EARLY_STOP else patience
    early_stopped = False
    epochs_ran = 0

    swa_model = torch.optim.swa_utils.AveragedModel(model)
    swa_start = int(n_epochs * 0.65)
    swa_sched = torch.optim.swa_utils.SWALR(opt, swa_lr=4e-4)

    for ep in range(n_epochs):
        epochs_ran = ep + 1
        model.train()

        out = model(emb_t, log_t, site_ids=site_t, hours=hour_t)

        loss = (
            F.binary_cross_entropy_with_logits(
                out, lab_t, pos_weight=pos_weight[None, None, :]
            )
            + 0.15 * F.mse_loss(out, log_t)
        )

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if ep >= swa_start:
            swa_model.update_parameters(model)
            swa_sched.step()
        else:
            sched.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {
                k: v.clone() for k, v in model.state_dict().items()
            }
            wait = 0
        else:
            wait += 1

        if wait >= effective_patience:
            if verbose:
                print(f"  Early stop ep {ep+1}")
            early_stopped = True
            break

    if ep >= swa_start:
        torch.optim.swa_utils.update_bn(emb_t.unsqueeze(0), swa_model)
        model = swa_model
    else:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        out = model(emb_t, log_t, site_ids=site_t, hours=hour_t)

    print(
        f"LightProtoSSM trained -best loss={best_loss:.4f} "
        f"| epochs_ran={epochs_ran}/{n_epochs} | early_stop={early_stopped}"
    )
    return model, site2i


print("[OK] CHANGE 4: LightProtoSSM with cross-attention (2 heads) defined")

# ===== Cell 20 =====
# Test-time augmentation via circular window shifts: run ProtoSSM on 5 shifted
# versions of each file's 12-window sequence and average the predictions.
# Predictions are counter-shifted before averaging to realign time steps.
def run_tta_proto(proto_model, emb_files, sc_files,
                  site_t, hour_t, shifts=[0, 1, -1, 2, -2]):
    """
    CHANGE 3: TTA by circular-shifting 12-window sequences.
    
    For each shift s:
      1. Roll embeddings and perch logits by s windows
      2. Run ProtoSSM ->get predictions
      3. Roll predictions back by -s (undo shift)
    
    Finally average all predictions across shifts.
    
    Why this works:
      - ProtoSSM sees temporal context across all 12 windows
      - Different starting points expose different context patterns
      - Averaging over 5 views reduces temporal boundary artifacts
    """
    proto_model.eval()
    all_preds = []
    
    emb_t  = torch.tensor(emb_files, dtype=torch.float32, device=TORCH_DEVICE)
    sc_t   = torch.tensor(sc_files,  dtype=torch.float32, device=TORCH_DEVICE)
    site_t = site_t.to(TORCH_DEVICE)
    hour_t = hour_t.to(TORCH_DEVICE)
    
    for shift in shifts:
        if shift == 0:
            e_shifted = emb_t
            s_shifted = sc_t
        else:
            e_shifted = torch.roll(emb_t, shift, dims=1)
            s_shifted = torch.roll(sc_t,  shift, dims=1)
        
        with torch.no_grad():
            out = proto_model(
                e_shifted, s_shifted,
                site_ids=site_t, hours=hour_t
            ).detach().cpu().numpy()
        
        if shift != 0:
            out = np.roll(out, -shift, axis=1)
        
        all_preds.append(out)
    
    return np.mean(all_preds, axis=0)

print("[OK] CHANGE 3: TTA with 5 circular shifts defined")

# ===== Cell 21 =====
# ResidualSSM: a lightweight second-pass model that learns to correct systematic
# errors from the first-pass ensemble by predicting (Y - sigmoid(first_pass)).
# The output head is zero-initialised so corrections begin small.
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualSSM(nn.Module):
    """
    Lightweight second-pass model that learns to correct
    systematic errors from the first-pass ensemble.
    
    Input:  embeddings + first-pass scores (concatenated)
    Output: additive correction to first-pass scores
    
    Key design: output head initialized to zero
    so corrections start small and only grow if helpful.
    ~25s training on 59 files.
    """
    def __init__(self, d_input=1536, d_scores=234,
                 d_model=64, d_state=8,
                 n_classes=234, n_windows=12,
                 dropout=0.1, n_sites=20, meta_dim=8):
        super().__init__()
        self.n_classes = n_classes

        self.input_proj = nn.Sequential(
            nn.Linear(d_input + d_scores, d_model),
            nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout))

        self.site_emb  = nn.Embedding(n_sites, meta_dim)
        self.hour_emb  = nn.Embedding(24,      meta_dim)
        self.meta_proj = nn.Linear(2 * meta_dim, d_model)
        self.pos_enc   = nn.Parameter(
            torch.randn(1, n_windows, d_model) * 0.02)

        self.ssm_fwd   = SelectiveSSM(d_model, d_state)
        self.ssm_bwd   = SelectiveSSM(d_model, d_state)
        self.ssm_merge = nn.Linear(2 * d_model, d_model)
        self.ssm_norm  = nn.LayerNorm(d_model)
        self.ssm_drop  = nn.Dropout(dropout)

        self.output_head = nn.Linear(d_model, n_classes)
        nn.init.zeros_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)

    def forward(self, emb, first_pass, site_ids=None, hours=None):
        B, T, _ = emb.shape
        x = torch.cat([emb, first_pass], dim=-1)
        h = self.input_proj(x) + self.pos_enc[:, :T, :]

        if site_ids is not None and hours is not None:
            meta = self.meta_proj(torch.cat(
                [self.site_emb(site_ids.clamp(0, self.site_emb.num_embeddings-1)),
                 self.hour_emb(hours.clamp(0, 23))], dim=-1))
            h = h + meta.unsqueeze(1)

        res = h
        h_f = self.ssm_fwd(h)
        h_b = self.ssm_bwd(h.flip(1)).flip(1)
        h   = self.ssm_drop(self.ssm_merge(
            torch.cat([h_f, h_b], dim=-1)))
        h   = self.ssm_norm(h + res)

        return self.output_head(h)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters()
                   if p.requires_grad)


def train_residual_ssm(emb_full, first_pass_flat, Y_full,
                       site_ids, hour_ids,
                       n_epochs=30, patience=8, lr=1e-3,
                       d_model=64, d_state=8, meta_dim=8, dropout=0.1,
                       correction_weight=0.30,
                       verbose=False):
    """
    Train ResidualSSM to predict (Y - sigmoid(first_pass)).
    Returns corrected flat scores (n_rows, n_classes).
    ~20s on CPU.
    """
    n_files    = len(emb_full) // N_WINDOWS
    emb_f      = emb_full.reshape(n_files, N_WINDOWS, -1)
    fp_f       = first_pass_flat.reshape(n_files, N_WINDOWS, -1)
    lab_f      = Y_full.reshape(n_files, N_WINDOWS, -1).astype(np.float32)

    fp_prob    = 1.0 / (1.0 + np.exp(-np.clip(fp_f, -30, 30)))
    residuals  = lab_f - fp_prob

    print(f"Residuals: mean={residuals.mean():.4f}  "
          f"std={residuals.std():.4f}  "
          f"abs_mean={np.abs(residuals).mean():.4f}")

    n_val    = max(1, int(n_files * 0.15))
    rng      = torch.Generator(); rng.manual_seed(GLOBAL_SEED)
    perm     = torch.randperm(n_files, generator=rng).numpy()
    val_i    = perm[:n_val];  train_i = perm[n_val:]

    emb_t    = torch.tensor(emb_f,    dtype=torch.float32, device=TORCH_DEVICE)
    fp_t     = torch.tensor(fp_f,     dtype=torch.float32, device=TORCH_DEVICE)
    res_t    = torch.tensor(residuals, dtype=torch.float32, device=TORCH_DEVICE)
    site_t   = torch.tensor(site_ids, dtype=torch.long, device=TORCH_DEVICE)
    hour_t   = torch.tensor(hour_ids, dtype=torch.long, device=TORCH_DEVICE)

    model = ResidualSSM(
        n_classes=N_CLASSES,
        n_sites=20,
        d_model=d_model,
        d_state=d_state,
        meta_dim=meta_dim,
        dropout=dropout,
    ).to(TORCH_DEVICE)
    print(
        f"ResidualSSM params: {model.count_parameters():,} "
        f"(d_model={d_model}, d_state={d_state}, meta_dim={meta_dim}, dropout={dropout})"
    )

    opt      = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=1e-3)
    sched    = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, epochs=n_epochs, steps_per_epoch=1,
        pct_start=0.1, anneal_strategy="cos")

    best_loss, best_state, wait = float("inf"), None, 0
    effective_patience = n_epochs + 1 if DISABLE_EARLY_STOP else patience
    early_stopped = False
    epochs_ran = 0

    for ep in range(n_epochs):
        epochs_ran = ep + 1
        model.train()
        corr = model(emb_t[train_i], fp_t[train_i],
                     site_ids=site_t[train_i],
                     hours   =hour_t[train_i])
        loss = F.mse_loss(corr, res_t[train_i])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

        model.eval()
        with torch.no_grad():
            val_corr = model(emb_t[val_i], fp_t[val_i],
                             site_ids=site_t[val_i],
                             hours   =hour_t[val_i])
            val_loss = F.mse_loss(val_corr, res_t[val_i])

        if val_loss.item() < best_loss:
            best_loss  = val_loss.item()
            best_state = {k: v.clone()
                          for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= effective_patience:
            if verbose: print(f"  Early stop ep {ep+1}")
            early_stopped = True
            break

    model.load_state_dict(best_state)
    print(
        f"ResidualSSM trained -best val MSE={best_loss:.6f} "
        f"| epochs_ran={epochs_ran}/{n_epochs} | early_stop={early_stopped}"
    )

    model.eval()
    with torch.no_grad():
        all_corr = model(emb_t, fp_t,
                         site_ids=site_t,
                         hours   =hour_t).detach().cpu().numpy()
    print(f"Correction magnitude: "
          f"mean_abs={np.abs(all_corr).mean():.4f}  "
          f"max={np.abs(all_corr).max():.4f}")

    return model, correction_weight


print("[OK] ResidualSSM defined (~439K params, ~20s training)")

# ===== Cell 22 =====
# OOF evaluation in train mode: report raw Perch macro-AUC as a baseline.
# Skipped entirely in submit mode to save time.
baseline_auc = None
oof_raw      = None
 
if CFG["run_oof"]:
    print("Running honest OOF evaluation on training data...")
    baseline_auc, oof_raw = honest_oof_auc(
        sc_tr, Y_FULL_aligned, meta_tr,
        n_splits=CFG["oof_n_splits"],
        label="raw Perch"
    )
    print(f"\nBaseline OOF AUC: {baseline_auc:.6f}  ->your starting point")
    if STOP_AFTER_RAW_OOF:
        print("[INFO] Exiting early after raw OOF as requested.")
        raise SystemExit(0)
else:
    print("Submit mode: skipping OOF evaluation")

# ===== Cell 23 =====
# Full-pipeline GroupKFold OOF: trains ProtoSSM + MLP probes on K-1 folds
# and predicts on the held-out fold. This is the only way to get an unbiased
# estimate of the final ensemble AUC without test-set leakage.

def run_pipeline_oof(emb_full, sc_full, Y_full, meta_full, n_splits=5):
    """
    Proper full-pipeline OOF.
    Trains ProtoSSM + MLP on K-1 folds, predicts on held-out fold.
    ~3-4 min total on CPU. Use this instead of the raw-Perch OOF.
    """
    file_meta = (
        meta_full.drop_duplicates("filename")
        .reset_index(drop=True)
    )

    gkf = GroupKFold(n_splits=n_splits)
    oof_probs = np.zeros((len(sc_full), N_CLASSES), dtype=np.float32)
    oof_proto_logits = np.zeros((len(sc_full), N_CLASSES), dtype=np.float32)
    oof_mlp_logits = np.zeros((len(sc_full), N_CLASSES), dtype=np.float32)

    for fold, (tr_f, va_f) in enumerate(
        gkf.split(file_meta, groups=file_meta["filename"]), 1
    ):
        tr_fnames = set(file_meta.iloc[tr_f]["filename"])
        va_fnames = set(file_meta.iloc[va_f]["filename"])

        tr_mask = meta_full["filename"].isin(tr_fnames).values
        va_mask = meta_full["filename"].isin(va_fnames).values

        emb_tr_f = emb_full[tr_mask]
        sc_tr_f = sc_full[tr_mask]
        Y_tr_f = Y_full[tr_mask]
        meta_tr_f = meta_full[tr_mask].reset_index(drop=True)

        emb_va_f = emb_full[va_mask]
        sc_va_f = sc_full[va_mask]
        meta_va_f = meta_full[va_mask].reset_index(drop=True)

        proto_model, site2i = train_light_proto_ssm(
            emb_tr_f,
            sc_tr_f,
            Y_tr_f,
            meta_tr_f,
            n_epochs=TUNE["proto_epochs"],
            patience=TUNE["proto_patience"],
            lr=TUNE["proto_lr"],
            d_model=TUNE["proto_d_model"],
            d_state=TUNE["proto_d_state"],
            meta_dim=TUNE["proto_meta_dim"],
            dropout=TUNE["proto_dropout"],
            n_layers=TUNE["proto_n_layers"],
            cross_attn_heads=TUNE["proto_cross_attn_heads"],
            verbose=False,
        )

        n_va = len(emb_va_f) // N_WINDOWS

        va_fn_list = (
            meta_va_f.drop_duplicates("filename")["filename"].tolist()
        )

        va_site_ids = np.array(
            [
                min(
                    site2i.get(
                        meta_va_f.loc[
                            meta_va_f["filename"] == fn, "site"
                        ].iloc[0],
                        0,
                    ),
                    19,
                )
                for fn in va_fn_list
            ],
            dtype=np.int64,
        )

        va_hour_ids = np.array(
            [
                int(
                    meta_va_f.loc[
                        meta_va_f["filename"] == fn, "hour_utc"
                    ].iloc[0]
                )
                % 24
                for fn in va_fn_list
            ],
            dtype=np.int64,
        )

        proto_model.eval()
        with torch.no_grad():
            proto_va = proto_model(
                torch.tensor(
                    emb_va_f.reshape(n_va, N_WINDOWS, -1),
                    dtype=torch.float32,
                    device=TORCH_DEVICE,
                ),
                torch.tensor(
                    sc_va_f.reshape(n_va, N_WINDOWS, -1),
                    dtype=torch.float32,
                    device=TORCH_DEVICE,
                ),
                site_ids=torch.tensor(va_site_ids, dtype=torch.long, device=TORCH_DEVICE),
                hours=torch.tensor(va_hour_ids, dtype=torch.long, device=TORCH_DEVICE),
            ).detach().cpu().numpy().reshape(-1, N_CLASSES)

        probe_models, emb_scaler, emb_pca, alpha_blend = train_mlp_probes(
            emb_tr_f,
            sc_tr_f,
            Y_tr_f,
            min_pos=TUNE["mlp_min_pos"],
            pca_dim=TUNE["mlp_pca_dim"],
            alpha_blend=TUNE["mlp_alpha_blend"],
        )

        sc_va_mlp = apply_mlp_probes_vectorized(
            emb_va_f,
            sc_va_f,
            probe_models,
            emb_scaler,
            emb_pca,
            alpha_blend,
        )
        oof_proto_logits[va_mask] = proto_va.astype(np.float32, copy=False)
        oof_mlp_logits[va_mask] = sc_va_mlp.astype(np.float32, copy=False)

        fold_weight = float(TUNE["ensemble_w"])
        if ENSEMBLE_PER_FOLD:
            n_tr = len(emb_tr_f) // N_WINDOWS
            tr_fn_list = (
                meta_tr_f.drop_duplicates("filename")["filename"].tolist()
            )
            tr_site_ids = np.array(
                [
                    min(
                        site2i.get(
                            meta_tr_f.loc[
                                meta_tr_f["filename"] == fn, "site"
                            ].iloc[0],
                            0,
                        ),
                        19,
                    )
                    for fn in tr_fn_list
                ],
                dtype=np.int64,
            )
            tr_hour_ids = np.array(
                [
                    int(
                        meta_tr_f.loc[
                            meta_tr_f["filename"] == fn, "hour_utc"
                        ].iloc[0]
                    )
                    % 24
                    for fn in tr_fn_list
                ],
                dtype=np.int64,
            )
            proto_model.eval()
            with torch.no_grad():
                proto_tr = proto_model(
                    torch.tensor(
                        emb_tr_f.reshape(n_tr, N_WINDOWS, -1),
                        dtype=torch.float32,
                        device=TORCH_DEVICE,
                    ),
                    torch.tensor(
                        sc_tr_f.reshape(n_tr, N_WINDOWS, -1),
                        dtype=torch.float32,
                        device=TORCH_DEVICE,
                    ),
                    site_ids=torch.tensor(tr_site_ids, dtype=torch.long, device=TORCH_DEVICE),
                    hours=torch.tensor(tr_hour_ids, dtype=torch.long, device=TORCH_DEVICE),
                ).detach().cpu().numpy().reshape(-1, N_CLASSES)
            sc_tr_mlp_fold = apply_mlp_probes_vectorized(
                emb_tr_f,
                sc_tr_f,
                probe_models,
                emb_scaler,
                emb_pca,
                alpha_blend,
            )
            if ENSEMBLE_PER_CLASS:
                fold_weight = _auto_tune_ensemble_weight_per_class(
                    proto_tr,
                    sc_tr_mlp_fold,
                    Y_tr_f,
                    default_w=float(TUNE["ensemble_w"]),
                )
            else:
                fold_weight = _auto_tune_ensemble_weight(
                    proto_tr,
                    sc_tr_mlp_fold,
                    Y_tr_f,
                )

        first_pass = _blend_ensemble_logits(proto_va, sc_va_mlp, fold_weight)
        probs_va = 1.0 / (1.0 + np.exp(-np.clip(first_pass, -30, 30)))
        oof_probs[va_mask] = probs_va

        fold_auc = macro_auc(Y_full[va_mask], probs_va)
        print(
            f"  Fold {fold}/{n_splits}  val files={len(va_fnames)}  AUC={fold_auc:.6f}"
        )

    overall = macro_auc(Y_full, oof_probs)
    print(f"\nFull pipeline OOF AUC: {overall:.6f}")
    return overall, oof_probs, oof_proto_logits, oof_mlp_logits


pipeline_auc, oof_pipeline = None, None
oof_proto_logits, oof_mlp_logits = None, None
if CFG["run_oof"]:
    pipeline_auc, oof_pipeline, oof_proto_logits, oof_mlp_logits = run_pipeline_oof(
        emb_tr,
        sc_tr,
        Y_FULL_aligned,
        meta_tr,
        n_splits=5,
    )

# ===== Cell 24 =====
# Run Perch on hidden test soundscapes.
# If no test files exist (local run), fall back to a dry-run subset of train files.
test_paths = sorted((BASE / "test_soundscapes").glob("*.ogg"))
 
if not test_paths:
    n = CFG["dryrun_n_files"] or 20
    print(f"No hidden test -dry-run on {n} train files")
    test_paths = sorted((BASE / "train_soundscapes").glob("*.ogg"))[:n]
else:
    print(f"Hidden test files: {len(test_paths)}")
 
meta_te, sc_te, emb_te = run_perch(test_paths, CFG["batch_files"], verbose=CFG["verbose"])
print(f"Test scores: {sc_te.shape}")
if PERCH_ADAPTER_MODEL is not None:
    t0 = time.time()
    sc_te = _apply_perch_adapter(
        sc_te,
        emb_te,
        PERCH_ADAPTER_MODEL,
        weight=PERCH_ADAPTER_WEIGHT,
        per_class_weight=PERCH_ADAPTER_PER_CLASS_WEIGHT,
    )
    print(
        f"[OK] Applied Perch adapter to test scores in {time.time()-t0:.1f}s "
        f"| score range [{sc_te.min():.3f}, {sc_te.max():.3f}]"
    )
if PERCH_EMB_CLS_MODEL is not None:
    t0 = time.time()
    sc_te = _apply_perch_embedding_classifier(
        sc_te,
        emb_te,
        PERCH_EMB_CLS_MODEL,
        weight=PERCH_EMB_CLS_WEIGHT,
    )
    print(
        f"[OK] Applied Perch embedding classifier to test scores in {time.time()-t0:.1f}s "
        f"| score range [{sc_te.min():.3f}, {sc_te.max():.3f}]"
    )
if PERCH_MIL_CLS_MODEL is not None:
    t0 = time.time()
    sc_te = _apply_perch_mil_classifier(
        sc_te,
        emb_te,
        PERCH_MIL_CLS_MODEL,
        weight=PERCH_MIL_CLS_WEIGHT,
    )
    print(
        f"[OK] Applied Perch MIL classifier to test scores in {time.time()-t0:.1f}s "
        f"| score range [{sc_te.min():.3f}, {sc_te.max():.3f}]"
    )
if UNMAPPED_HEAD_MODEL is not None:
    t0 = time.time()
    sc_te = _apply_unmapped_head(
        sc_te,
        emb_te,
        UNMAPPED_HEAD_MODEL,
        weight=UNMAPPED_HEAD_WEIGHT,
    )
    print(
        f"[OK] Applied Unmapped head to test scores in {time.time()-t0:.1f}s "
        f"| score range [{sc_te.min():.3f}, {sc_te.max():.3f}]"
    )

# ===== Cell 25 =====
# Full inference pipeline:
#   A. Train LightProtoSSM on training embeddings.
#   B. Run ProtoSSM on test set.
#   C. Apply site/hour prior logits to raw Perch test scores.
#   D. Blend in MLP probe corrections.
#   E. Average ProtoSSM and MLP scores (first pass).
#   F. Train ResidualSSM on training errors; apply correction to test scores.
#   G. Divide by per-taxon temperature.
#   H. Sigmoid ->probabilities.
#   I. Post-processing: file confidence scaling, rank-aware scaling, adaptive smoothing.
#   J. Apply per-class thresholds and write submission.csv.

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

CKPT_PATH_RAW = os.environ.get("BC26_CKPT_PATH", "").strip()
CKPT_PATH = Path(CKPT_PATH_RAW) if CKPT_PATH_RAW else None
LOAD_FROM_CKPT_IN_TRAIN = os.environ.get("BC26_LOAD_CKPT_IN_TRAIN", "0").strip().lower() in {"1", "true", "yes"}
LOAD_FROM_CKPT = bool(
    CKPT_PATH is not None
    and CKPT_PATH.exists()
    and (MODE == "submit" or LOAD_FROM_CKPT_IN_TRAIN)
)
SAVE_TO_CKPT = bool(CKPT_PATH is not None and MODE == "train")


def _sanitize_torch_state_dict(state_dict):
    cleaned = {}
    for k, v in state_dict.items():
        if k == "n_averaged":
            continue
        if k.startswith("module."):
            cleaned[k[len("module."):]] = v
        else:
            cleaned[k] = v
    return cleaned

n_sites_cap = 20
ENSEMBLE_W = float(TUNE["ensemble_w"])
ENSEMBLE_W_VEC = None
STACKER = None

if LOAD_FROM_CKPT:
    print(f"Loading pipeline checkpoint: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    ckpt_labels = ckpt.get("primary_labels")
    if ckpt_labels is not None and list(ckpt_labels) != list(PRIMARY_LABELS):
        raise RuntimeError(
            "Checkpoint label space mismatch with current sample_submission."
        )

    site2i_tr = dict(ckpt["site2i_tr"])
    prior_tables = ckpt["prior_tables"]
    probe_models = ckpt["probe_models"]
    emb_scaler = ckpt["emb_scaler"]
    emb_pca = ckpt["emb_pca"]
    alpha_blend = float(ckpt.get("alpha_blend", TUNE["mlp_alpha_blend"]))
    ENSEMBLE_W = float(ckpt.get("ensemble_w", TUNE["ensemble_w"]))
    _ewv = ckpt.get("ensemble_w_vec")
    if _ewv is not None:
        ENSEMBLE_W_VEC = np.asarray(_ewv, dtype=np.float32)
    _stack_coef = ckpt.get("stacking_coef")
    _stack_bias = ckpt.get("stacking_bias")
    if _stack_coef is not None and _stack_bias is not None:
        STACKER = {
            "coef": np.asarray(_stack_coef, dtype=np.float32),
            "bias": np.asarray(_stack_bias, dtype=np.float32),
        }
    correction_weight = float(ckpt.get("correction_weight", TUNE["res_correction_weight"]))
    _pcw = ckpt.get("perch_adapter_per_class_weight")
    if _pcw is not None:
        PERCH_ADAPTER_PER_CLASS_WEIGHT = np.asarray(_pcw, dtype=np.float32)
    temperatures = np.asarray(ckpt["temperatures"], dtype=np.float32)
    PER_CLASS_THRESHOLDS = np.asarray(ckpt["per_class_thresholds"], dtype=np.float32)
    n_sites_cap = int(ckpt.get("n_sites_cap", 20))
    proto_arch = ckpt.get("proto_arch", {})
    proto_d_model = int(proto_arch.get("d_model", 128))
    proto_d_state = int(proto_arch.get("d_state", 16))
    proto_meta_dim = int(proto_arch.get("meta_dim", 16))
    proto_dropout = float(proto_arch.get("dropout", 0.15))
    proto_n_layers = int(proto_arch.get("n_layers", 2))
    proto_cross_attn_heads = int(proto_arch.get("cross_attn_heads", 2))
    res_arch = ckpt.get("res_arch", {})
    res_d_model = int(res_arch.get("d_model", 64))
    res_d_state = int(res_arch.get("d_state", 8))
    res_meta_dim = int(res_arch.get("meta_dim", 8))
    res_dropout = float(res_arch.get("dropout", 0.1))

    proto_model = LightProtoSSM(
        d_model=proto_d_model,
        d_state=proto_d_state,
        n_classes=N_CLASSES,
        n_sites=n_sites_cap,
        meta_dim=proto_meta_dim,
        dropout=proto_dropout,
        use_cross_attn=True,
        cross_attn_heads=proto_cross_attn_heads,
        n_ssm_layers=proto_n_layers,
    ).to(TORCH_DEVICE)
    proto_sd = _sanitize_torch_state_dict(ckpt["proto_state_dict"])
    proto_model.load_state_dict(proto_sd, strict=True)
    proto_model.eval()

    res_model = ResidualSSM(
        n_classes=N_CLASSES,
        n_sites=n_sites_cap,
        d_model=res_d_model,
        d_state=res_d_state,
        meta_dim=res_meta_dim,
        dropout=res_dropout,
    ).to(TORCH_DEVICE)
    res_sd = _sanitize_torch_state_dict(ckpt["residual_state_dict"])
    res_model.load_state_dict(res_sd, strict=True)
    res_model.eval()

    if STACKER is not None:
        ens_msg = "stacking"
    else:
        ens_msg = (
            "per-class"
            if ENSEMBLE_W_VEC is not None
            else f"{ENSEMBLE_W:.3f}"
        )
    print(
        f"Checkpoint loaded: probes={len(probe_models)} "
        f"| n_sites_cap={n_sites_cap} | alpha_blend={alpha_blend:.3f} "
        f"| ensemble={ens_msg} "
        f"| proto_arch(d_model={proto_d_model},d_state={proto_d_state},layers={proto_n_layers},heads={proto_cross_attn_heads}) "
        f"| res_arch(d_model={res_d_model},d_state={res_d_state})"
    )
else:
    t0 = time.time()
    proto_model, site2i_tr = train_light_proto_ssm(
        emb_tr, sc_tr, Y_FULL_aligned, meta_tr,
        n_epochs=TUNE["proto_epochs"],
        patience=TUNE["proto_patience"],
        lr=TUNE["proto_lr"],
        d_model=TUNE["proto_d_model"],
        d_state=TUNE["proto_d_state"],
        meta_dim=TUNE["proto_meta_dim"],
        dropout=TUNE["proto_dropout"],
        n_layers=TUNE["proto_n_layers"],
        cross_attn_heads=TUNE["proto_cross_attn_heads"],
        verbose=False,
    )
    print(f"ProtoSSM training: {time.time()-t0:.1f}s")

    prior_tables = build_prior_tables(sc, Y_SC)
    probe_models, emb_scaler, emb_pca, alpha_blend = train_mlp_probes(
        emb=emb_tr, scores_raw=sc_tr, Y=Y_FULL_aligned,
        min_pos=TUNE["mlp_min_pos"],
        pca_dim=TUNE["mlp_pca_dim"],
        alpha_blend=TUNE["mlp_alpha_blend"],
    )

    n_tr_files    = len(sc_tr) // N_WINDOWS
    emb_tr_f      = emb_tr.reshape(n_tr_files, N_WINDOWS, -1)
    sc_tr_f       = sc_tr.reshape(n_tr_files, N_WINDOWS, -1)

    tr_fnames     = meta_tr.drop_duplicates("filename")["filename"].tolist()
    tr_site_ids   = np.array([
        min(site2i_tr.get(
            meta_tr.loc[meta_tr["filename"]==fn,"site"].iloc[0], 0),
            n_sites_cap-1)
        for fn in tr_fnames], dtype=np.int64)
    tr_hour_ids   = np.array([
        int(meta_tr.loc[meta_tr["filename"]==fn,"hour_utc"].iloc[0]) % 24
        for fn in tr_fnames], dtype=np.int64)

    proto_tr_out = run_tta_proto(
        proto_model, emb_tr_f, sc_tr_f,
        site_t=torch.tensor(tr_site_ids, dtype=torch.long),
        hour_t=torch.tensor(tr_hour_ids, dtype=torch.long),
        shifts=TUNE["tta_shifts"],
    )
    proto_tr_flat = proto_tr_out.reshape(-1, N_CLASSES).astype(np.float32)

    sc_tr_prior   = apply_prior(
        sc_tr,
        sites=meta_tr["site"].to_numpy(),
        hours=meta_tr["hour_utc"].to_numpy(),
        tables=prior_tables,
        lambda_prior=TUNE["prior_lambda"],
    )
    sc_tr_mlp = apply_mlp_probes_vectorized(
        emb_tr, sc_tr_prior,
        probe_models, emb_scaler, emb_pca, alpha_blend,
    )
    if ENSEMBLE_W_AUTO:
        ENSEMBLE_W = _auto_tune_ensemble_weight(
            proto_tr_flat,
            sc_tr_mlp,
            Y_FULL_aligned,
        )
    if ENSEMBLE_PER_CLASS:
        ENSEMBLE_W_VEC = _auto_tune_ensemble_weight_per_class(
            proto_tr_flat,
            sc_tr_mlp,
            Y_FULL_aligned,
            default_w=float(ENSEMBLE_W),
        )
    if STACKING_ENABLE and oof_proto_logits is not None and oof_mlp_logits is not None:
        STACKER = train_oof_logit_stacker(
            oof_proto_logits,
            oof_mlp_logits,
            Y_FULL_aligned,
            min_pos=STACKING_MIN_POS,
            c_value=STACKING_LOGREG_C,
        )
        oof_stacked_logits = apply_logit_stacker(oof_proto_logits, oof_mlp_logits, STACKER)
        oof_pipeline = sigmoid(oof_stacked_logits)
        stack_auc = macro_auc(Y_FULL_aligned, oof_pipeline)
        print(f"[INFO] OOF stacker macro-AUC: {stack_auc:.6f}")
    elif STACKING_ENABLE:
        print("[WARN] Stacking enabled but OOF branch logits unavailable; fallback to weighted blend.")
    first_pass_tr = (
        apply_logit_stacker(proto_tr_flat, sc_tr_mlp, STACKER)
        if STACKER is not None
        else _blend_ensemble_logits(
            proto_tr_flat,
            sc_tr_mlp,
            ENSEMBLE_W_VEC if ENSEMBLE_W_VEC is not None else ENSEMBLE_W,
        )
    )

    train_probs_for_calib = sigmoid(first_pass_tr)
    calib_probs = train_probs_for_calib
    if (
        TUNE["calib_use_oof"]
        and oof_pipeline is not None
        and oof_pipeline.shape == train_probs_for_calib.shape
    ):
        calib_probs = oof_pipeline.astype(np.float32, copy=False)
        print("[INFO] Calibration source: OOF pipeline probabilities")
    else:
        print("[INFO] Calibration source: in-sample train probabilities")
    PER_CLASS_THRESHOLDS = calibrate_and_optimize_thresholds(
        oof_probs=calib_probs,
        Y_FULL=Y_FULL_aligned,
        threshold_grid=TUNE["threshold_grid"],
        n_windows=N_WINDOWS,
        min_pos_files=TUNE["calib_min_pos_files"],
        default_threshold=TUNE["calib_default_threshold"],
        bucketed=TUNE["calib_bucketed"],
        rare_pos_max=TUNE["calib_rare_pos_max"],
        common_pos_min=TUNE["calib_common_pos_min"],
        threshold_grid_rare=TUNE["threshold_grid_rare"],
        threshold_grid_common=TUNE["threshold_grid_common"],
    )

    t0 = time.time()
    res_model, correction_weight = train_residual_ssm(
        emb_full=emb_tr,
        first_pass_flat=first_pass_tr,
        Y_full=Y_FULL_aligned,
        site_ids=tr_site_ids,
        hour_ids=tr_hour_ids,
        n_epochs=TUNE["res_epochs"],
        patience=TUNE["res_patience"],
        lr=TUNE["res_lr"],
        d_model=TUNE["res_d_model"],
        d_state=TUNE["res_d_state"],
        meta_dim=TUNE["res_meta_dim"],
        dropout=TUNE["res_dropout"],
        correction_weight=TUNE["res_correction_weight"],
        verbose=False,
    )
    print(f"ResidualSSM training: {time.time()-t0:.1f}s")

    if SAVE_TO_CKPT:
        CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "primary_labels": PRIMARY_LABELS,
                "n_classes": N_CLASSES,
                "n_sites_cap": n_sites_cap,
                "site2i_tr": site2i_tr,
                "proto_state_dict": proto_model.state_dict(),
                "proto_arch": {
                    "d_model": int(TUNE["proto_d_model"]),
                    "d_state": int(TUNE["proto_d_state"]),
                    "meta_dim": int(TUNE["proto_meta_dim"]),
                    "dropout": float(TUNE["proto_dropout"]),
                    "n_layers": int(TUNE["proto_n_layers"]),
                    "cross_attn_heads": int(TUNE["proto_cross_attn_heads"]),
                },
                "residual_state_dict": res_model.state_dict(),
                "res_arch": {
                    "d_model": int(TUNE["res_d_model"]),
                    "d_state": int(TUNE["res_d_state"]),
                    "meta_dim": int(TUNE["res_meta_dim"]),
                    "dropout": float(TUNE["res_dropout"]),
                },
                "prior_tables": prior_tables,
                "probe_models": probe_models,
                "emb_scaler": emb_scaler,
                "emb_pca": emb_pca,
                "alpha_blend": float(alpha_blend),
                "ensemble_w": float(ENSEMBLE_W),
                "ensemble_w_vec": (
                    ENSEMBLE_W_VEC.astype(np.float32)
                    if ENSEMBLE_W_VEC is not None
                    else None
                ),
                "stacking_coef": (
                    STACKER["coef"].astype(np.float32)
                    if STACKER is not None
                    else None
                ),
                "stacking_bias": (
                    STACKER["bias"].astype(np.float32)
                    if STACKER is not None
                    else None
                ),
                "correction_weight": float(correction_weight),
                "temperatures": temperatures.astype(np.float32),
                "per_class_thresholds": PER_CLASS_THRESHOLDS.astype(np.float32),
                "tune": dict(TUNE),
                "perch_adapter_ckpt": PERCH_ADAPTER_CKPT_RAW,
                "perch_adapter_weight": float(PERCH_ADAPTER_WEIGHT),
                "perch_adapter_per_class_weight": (
                    PERCH_ADAPTER_PER_CLASS_WEIGHT.astype(np.float32)
                    if PERCH_ADAPTER_PER_CLASS_WEIGHT is not None
                    else None
                ),
                "perch_mil_cls_ckpt": PERCH_MIL_CLS_CKPT_RAW,
                "perch_mil_cls_weight": float(PERCH_MIL_CLS_WEIGHT),
            },
            CKPT_PATH,
        )
        print(f"Pipeline checkpoint saved to: {CKPT_PATH}")

n_test_files  = len(sc_te) // N_WINDOWS
emb_te_f      = emb_te.reshape(n_test_files, N_WINDOWS, -1)
sc_te_f       = sc_te.reshape(n_test_files, N_WINDOWS, -1)

test_fnames   = meta_te.drop_duplicates("filename")["filename"].tolist()
test_site_ids = np.array([
    min(site2i_tr.get(
        meta_te.loc[meta_te["filename"]==fn,"site"].iloc[0], 0),
        n_sites_cap-1)
    for fn in test_fnames], dtype=np.int64)
test_hour_ids = np.array([
    int(meta_te.loc[meta_te["filename"]==fn,"hour_utc"].iloc[0]) % 24
    for fn in test_fnames], dtype=np.int64)

proto_model.eval()
with torch.no_grad():
    proto_out = proto_model(
        torch.tensor(emb_te_f, dtype=torch.float32, device=TORCH_DEVICE),
        torch.tensor(sc_te_f,  dtype=torch.float32, device=TORCH_DEVICE),
        site_ids=torch.tensor(test_site_ids, dtype=torch.long, device=TORCH_DEVICE),
        hours   =torch.tensor(test_hour_ids, dtype=torch.long, device=TORCH_DEVICE),
    ).detach().cpu().numpy()
proto_scores_flat = proto_out.reshape(-1, N_CLASSES).astype(np.float32)

sc_te_adjusted = apply_prior(
    sc_te,
    sites=meta_te["site"].to_numpy(),
    hours=meta_te["hour_utc"].to_numpy(),
    tables=prior_tables,
    lambda_prior=TUNE["prior_lambda"],
)
sc_te_adjusted = apply_mlp_probes_vectorized(
    emb_te, sc_te_adjusted,
    probe_models, emb_scaler, emb_pca, alpha_blend,
)
first_pass_flat = (
    apply_logit_stacker(proto_scores_flat, sc_te_adjusted, STACKER)
    if STACKER is not None
    else _blend_ensemble_logits(
        proto_scores_flat,
        sc_te_adjusted,
        ENSEMBLE_W_VEC if ENSEMBLE_W_VEC is not None else ENSEMBLE_W,
    )
)

first_pass_te_f  = first_pass_flat.reshape(n_test_files, N_WINDOWS, -1)
res_model.eval()
with torch.no_grad():
    test_correction = res_model(
        torch.tensor(emb_te_f,         dtype=torch.float32, device=TORCH_DEVICE),
        torch.tensor(first_pass_te_f,  dtype=torch.float32, device=TORCH_DEVICE),
        site_ids=torch.tensor(test_site_ids, dtype=torch.long, device=TORCH_DEVICE),
        hours   =torch.tensor(test_hour_ids, dtype=torch.long, device=TORCH_DEVICE),
    ).detach().cpu().numpy()

correction_flat = test_correction.reshape(-1, N_CLASSES).astype(np.float32)
final_scores    = (first_pass_flat
                   + correction_weight * correction_flat)

print(f"Correction applied -"
      f"mean_abs={np.abs(correction_flat).mean():.4f}  "
      f"score range [{final_scores.min():.3f}, {final_scores.max():.3f}]")

final_scores = final_scores / temperatures[None, :]

probs = sigmoid(final_scores)

probs = file_confidence_scale(probs, n_windows=N_WINDOWS,
                               top_k=TUNE["post_topk"], power=TUNE["post_conf_power"])
probs = rank_aware_scaling(   probs, n_windows=N_WINDOWS,
                               power=TUNE["post_rank_power"])
probs = adaptive_delta_smooth(probs, n_windows=N_WINDOWS,
                               base_alpha=TUNE["post_smooth_alpha"])
probs = np.clip(probs, 0.0, 1.0)

probs = apply_per_class_thresholds(probs, PER_CLASS_THRESHOLDS)

sub = pd.DataFrame(probs.astype(np.float32), columns=PRIMARY_LABELS)
sub.insert(0, "row_id", meta_te["row_id"].values)
assert list(sub.columns) == ["row_id"] + PRIMARY_LABELS
assert len(sub) == len(test_paths) * N_WINDOWS
assert not sub.isna().any().any()
sub.to_csv(SUBMISSION_PATH, index=False)

print(f"\n{SUBMISSION_PATH} saved -shape {sub.shape}")
print(f"Total wall time: {(time.time() - _WALL_START)/60:.1f} min")

# ===== Cell 26 =====


# ===== Cell 27 =====




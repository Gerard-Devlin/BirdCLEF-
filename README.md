# BirdCLEF+ 2026

This repo now keeps only the gold-inspired pipeline:

- `scripts/two_pass_ssm_pipeline_v2.py`
- source notebook: `birdclef-26-two-pass-ssm-advanced-pp.ipynb`

All old baseline training/inference code was removed.

## Required Inputs

You need these assets:

1. Competition data (`birdclef-2026`)
    - `taxonomy.csv`
    - `sample_submission.csv`
    - `train_soundscapes_labels.csv`
    - `train_soundscapes/*.ogg`
    - `test_soundscapes/*.ogg` (for real submit rerun)
2. Perch pretrained model
    - `.../perch_v2_cpu/1/` with `saved_model.pb`, `variables`, `assets/labels.csv`
3. Perch cache dataset (strongly recommended)
    - `perch_meta.parquet`
    - `perch_arrays.npz`
4. Optional speedup
    - `perch_v2.onnx`
    - `onnxruntime` wheel

## Environment Variables Used By Script

- `BC26_MODE`: `train` or `submit` (default `submit`)
- `BC26_BASE`: competition folder path
- `BC26_MODEL_DIR`: Perch model folder
- `BC26_WORK_DIR`: cache/output workspace
- `BC26_SUBMISSION_PATH`: output CSV path (default `submission.csv`)
- `BC26_ONNX_PATH`: optional ONNX model path
- `BC26_EXTRA_CACHE_DIRS`: extra cache dirs separated by `:` on Linux or `;` on Windows
- `BC26_CKPT_PATH`: optional pipeline ckpt path (save when training, load when exists)
- `BC26_USE_GPU`: `1` to prefer GPU for torch/onnxruntime, `0` to force CPU

### Optional Tunables (env vars)

- `BC26_PROTO_EPOCHS`, `BC26_PROTO_PATIENCE`, `BC26_PROTO_LR`
- `BC26_RES_EPOCHS`, `BC26_RES_PATIENCE`, `BC26_RES_LR`, `BC26_RES_CORRECTION_WEIGHT`
- `BC26_MLP_MIN_POS`, `BC26_MLP_PCA_DIM`, `BC26_MLP_ALPHA_BLEND`
- `BC26_PRIOR_LAMBDA`, `BC26_ENSEMBLE_W`
- `BC26_TTA_SHIFTS` (comma-separated, e.g. `0,1,-1,2,-2`)
- `BC26_THRESHOLD_GRID` (comma-separated, e.g. `0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70`)
- `BC26_POST_TOPK`, `BC26_POST_CONF_POWER`, `BC26_POST_RANK_POWER`, `BC26_POST_SMOOTH_ALPHA`

## Local Fine-Tune + Export CKPT (Linux)

```bash
BC26_MODE=train \
BC26_BASE=/data/birdclef-2026 \
BC26_MODEL_DIR=/data/perch_v2_cpu/1 \
BC26_WORK_DIR=/data/work/cache \
BC26_SUBMISSION_PATH=/data/work/submission_local.csv \
BC26_ONNX_PATH=/data/perch-onnx-for-birdclef-2026/perch_v2.onnx \
BC26_EXTRA_CACHE_DIRS=/data/perch-meta \
BC26_CKPT_PATH=/data/work/two_pass_pipeline_ckpt.pth \
BC26_USE_GPU=1 \
python scripts/two_pass_ssm_pipeline_v2.py
```

After this run, `BC26_CKPT_PATH` is your offline fine-tuned checkpoint bundle.

## Server Training Command (nohup)

Use this exact command on server:

```bash
cd ~/birdclef+
conda activate birdclef
ROOT="$PWD"
mkdir -p "$ROOT/logs" "$ROOT/work/cache" "$ROOT/work/ckpt"

CKPT="$ROOT/work/ckpt/two_pass_pipeline_ckpt_v2.pth"
nohup env PYTHONNOUSERSITE=1 \
BC26_USE_GPU=1 \
BC26_MODE=train \
BC26_BASE="$ROOT/dataset" \
BC26_MODEL_DIR="$ROOT/models/bird-vocalization-classifier-tensorflow2-perch_v2_cpu-v1" \
BC26_WORK_DIR="$ROOT/work/cache" \
BC26_SUBMISSION_PATH="$ROOT/work/submission_local_v2.csv" \
BC26_ONNX_PATH="$ROOT/source/Perch-onnx-for-birdclef+2026/perch_v2.onnx" \
BC26_EXTRA_CACHE_DIRS="$ROOT/source/Perch_meta" \
BC26_CKPT_PATH="$CKPT" \
BC26_PROTO_EPOCHS=80 \
BC26_PROTO_PATIENCE=16 \
BC26_PROTO_LR=8e-4 \
BC26_RES_EPOCHS=60 \
BC26_RES_PATIENCE=14 \
BC26_RES_LR=8e-4 \
BC26_MLP_MIN_POS=3 \
python scripts/two_pass_ssm_pipeline_v2.py > "$ROOT/logs/train_ckpt_v2.log" 2>&1 &

echo $! > "$ROOT/logs/train_ckpt_v2.pid"
tail -f "$ROOT/logs/train_ckpt_v2.log"
```

Completion signals:

- `Full pipeline OOF AUC: ...`
- `Pipeline checkpoint saved to: ...`
- `submission_local_v2.csv saved ...`

Common issues:

- `Permission denied: /work`: `ROOT` was empty; run `ROOT="$PWD"` first.
- `Unable to find a usable engine ... parquet`: install in current env via `python -m pip install pyarrow`.
- `cuda:0 and cpu mismatch`: pull latest code (`git pull`) and rerun.
- If `BC26_CKPT_PATH` already exists, `BC26_MODE=train` now ignores loading it by default and only saves new ckpt at the end.  
  To force loading ckpt during train mode, set `BC26_LOAD_CKPT_IN_TRAIN=1`.

## Kaggle Pure Inference (No Fine-Tune)

Use `scripts/infer_pt.py` so Kaggle only loads ckpt and runs inference.

Notebook Cell 1 (path setup):

```python
import os

bundle_root = "/kaggle/input/<your-code-dataset-root>"
ckpt_path = "/kaggle/input/<your-ckpt-dataset-root>/two_pass_pipeline_ckpt.pth"
base = "/kaggle/input/competitions/birdclef-2026"
model_dir = "/kaggle/input/models/google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1"
onnx_path = "/kaggle/input/perch-onnx-for-birdclef+2026/perch_v2.onnx"
extra_cache_dirs = "/kaggle/input/perch_meta"

print(os.path.exists(bundle_root), os.path.exists(ckpt_path), os.path.exists(base))
```

Notebook Cell 2 (inference only):

```python
!PYTHONNOUSERSITE=1 python {bundle_root}/scripts/infer_pt.py \
  --checkpoint {ckpt_path} \
  --base {base} \
  --model-dir {model_dir} \
  --onnx-path {onnx_path} \
  --extra-cache-dirs {extra_cache_dirs} \
  --output /kaggle/working/submission.csv
```

The notebook version must output `/kaggle/working/submission.csv`.

## Direct Submit-Mode Run (Legacy)

```python
!PYTHONNOUSERSITE=1 BC26_MODE=submit python /kaggle/working/scripts/two_pass_ssm_pipeline_v2.py
```

Output file is written to `BC26_SUBMISSION_PATH` (default `submission.csv`).

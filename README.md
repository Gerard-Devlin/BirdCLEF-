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
python scripts/two_pass_ssm_pipeline_v2.py
```

After this run, `BC26_CKPT_PATH` is your offline fine-tuned checkpoint bundle.

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

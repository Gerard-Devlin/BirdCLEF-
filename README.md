# BirdCLEF+ 2026 (Gold-Pipeline Only)

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

## Local Run Example (Linux)

```bash
BC26_MODE=train \
BC26_BASE=/data/birdclef-2026 \
BC26_MODEL_DIR=/data/perch_v2_cpu/1 \
BC26_WORK_DIR=/data/work/cache \
BC26_SUBMISSION_PATH=/data/work/submission_local.csv \
BC26_ONNX_PATH=/data/perch-onnx-for-birdclef-2026/perch_v2.onnx \
BC26_EXTRA_CACHE_DIRS=/data/perch-meta \
python scripts/two_pass_ssm_pipeline_v2.py
```

## Kaggle Run Example

```python
!PYTHONNOUSERSITE=1 BC26_MODE=submit python /kaggle/working/scripts/two_pass_ssm_pipeline_v2.py
```

Output file is written to `BC26_SUBMISSION_PATH` (default `submission.csv`).

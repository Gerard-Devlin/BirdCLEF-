#!/usr/bin/env python
"""Minimal pure-inference entrypoint for Kaggle.

This wrapper runs the original two-pass pipeline script in submit mode, but
forces it to load a pre-trained pipeline checkpoint and skip local fine-tuning.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BirdCLEF two-pass inference from checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to exported pipeline ckpt (.pt/.pth).")
    parser.add_argument("--base", required=True, help="BirdCLEF competition data directory.")
    parser.add_argument("--model-dir", required=True, help="Perch model directory.")
    parser.add_argument("--output", default="/kaggle/working/submission.csv", help="Submission CSV output path.")
    parser.add_argument("--work-dir", default="/kaggle/working/cache", help="Working/cache directory.")
    parser.add_argument("--onnx-path", default="", help="Optional Perch ONNX path.")
    parser.add_argument("--extra-cache-dirs", default="", help="Optional extra cache dirs (os.pathsep separated).")
    parser.add_argument("--input-root", default="/kaggle/input", help="Root path for wheel auto-discovery.")
    parser.add_argument(
        "--pipeline-script",
        default="",
        help="Path to two_pass_ssm_pipeline_v2.py; defaults to sibling script.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    here = Path(__file__).resolve().parent
    pipeline_script = Path(args.pipeline_script) if args.pipeline_script else here / "two_pass_ssm_pipeline_v2.py"
    if not pipeline_script.exists():
        raise FileNotFoundError(f"Pipeline script not found: {pipeline_script}")

    env = os.environ.copy()
    env["BC26_MODE"] = "submit"
    env["BC26_CKPT_PATH"] = str(Path(args.checkpoint))
    env["BC26_BASE"] = str(Path(args.base))
    env["BC26_MODEL_DIR"] = str(Path(args.model_dir))
    env["BC26_WORK_DIR"] = str(Path(args.work_dir))
    env["BC26_SUBMISSION_PATH"] = str(Path(args.output))
    env["BC26_INPUT_ROOT"] = str(Path(args.input_root))
    if args.onnx_path:
        env["BC26_ONNX_PATH"] = str(Path(args.onnx_path))
    if args.extra_cache_dirs:
        env["BC26_EXTRA_CACHE_DIRS"] = args.extra_cache_dirs

    cmd = [sys.executable, str(pipeline_script)]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, env=env, check=True)


if __name__ == "__main__":
    main()


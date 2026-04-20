from __future__ import annotations

import argparse
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare minimal bundle for Kaggle notebook inference"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dst = output_dir / "checkpoint.pth"
    shutil.copy2(args.checkpoint, checkpoint_dst)

    scripts_dir = output_dir / "scripts"
    scripts_dir.mkdir(exist_ok=True)
    shutil.copy2(PROJECT_ROOT / "scripts/infer_kaggle.py", scripts_dir / "infer_kaggle.py")

    package_dst = output_dir / "birdclef_plus"
    if package_dst.exists():
        shutil.rmtree(package_dst)
    shutil.copytree(PROJECT_ROOT / "birdclef_plus", package_dst)

    print(f"Bundle ready: {output_dir}")
    print(f"- checkpoint: {checkpoint_dst}")
    print(f"- inference script: {scripts_dir / 'infer_kaggle.py'}")
    print(f"- package: {package_dst}")


if __name__ == "__main__":
    main()

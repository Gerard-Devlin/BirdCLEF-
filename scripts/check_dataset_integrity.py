from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

try:
    import soundfile as sf
except ImportError as exc:  # pragma: no cover
    raise ImportError("Please install soundfile: python -m pip install soundfile") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check BirdCLEF dataset file integrity")
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="0 means check all files; otherwise check first N rows from train.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    with args.train_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            if args.max_files > 0 and len(rows) >= args.max_files:
                break

    missing = []
    zero_size = []
    decode_fail = []
    ext_counter = Counter()

    for row in rows:
        rel = row["filename"]
        path = args.audio_dir / rel
        ext_counter[path.suffix.lower()] += 1

        if not path.exists():
            missing.append(rel)
            continue

        if path.stat().st_size <= 0:
            zero_size.append(rel)
            continue

        try:
            sf.info(str(path))
        except Exception as exc:
            decode_fail.append((rel, str(exc)))

    total = len(rows)
    print(f"Checked rows: {total}")
    print(f"Missing files: {len(missing)}")
    print(f"Zero-size files: {len(zero_size)}")
    print(f"Decode failures: {len(decode_fail)}")
    print(f"Extension stats: {dict(ext_counter)}")

    if missing:
        print("\nFirst 20 missing:")
        for rel in missing[:20]:
            print(f"  {rel}")

    if zero_size:
        print("\nFirst 20 zero-size:")
        for rel in zero_size[:20]:
            print(f"  {rel}")

    if decode_fail:
        print("\nFirst 20 decode failures:")
        for rel, err in decode_fail[:20]:
            print(f"  {rel} -> {err}")


if __name__ == "__main__":
    main()

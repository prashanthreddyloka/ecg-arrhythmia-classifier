from __future__ import annotations

import argparse
from pathlib import Path

from data import (
    DS1_RECORDS,
    DS2_RECORDS,
    build_split,
    download_mit_bih,
    save_processed_splits,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare MIT-BIH beat windows for ECG arrhythmia detection.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/raw/mitdb"))
    parser.add_argument("--output-path", type=Path, default=Path("data/processed/mitbih_beats.npz"))
    parser.add_argument("--window-size", type=int, default=31)
    parser.add_argument("--fft-bins", type=int, default=32)
    parser.add_argument("--pre-samples", type=int, default=180)
    parser.add_argument("--post-samples", type=int, default=180)
    parser.add_argument("--lead-index", type=int, default=0)
    parser.add_argument("--download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.download:
        download_mit_bih(args.dataset_dir)

    train_split = build_split(
        dataset_dir=args.dataset_dir,
        records=DS1_RECORDS,
        window_size=args.window_size,
        fft_bins=args.fft_bins,
        pre_samples=args.pre_samples,
        post_samples=args.post_samples,
        lead_index=args.lead_index,
    )
    test_split = build_split(
        dataset_dir=args.dataset_dir,
        records=DS2_RECORDS,
        window_size=args.window_size,
        fft_bins=args.fft_bins,
        pre_samples=args.pre_samples,
        post_samples=args.post_samples,
        lead_index=args.lead_index,
    )

    save_processed_splits(args.output_path, train_split, test_split)
    print(f"Saved processed data to {args.output_path}")
    print(
        f"Train beats: {train_split.labels.shape[0]} | "
        f"Test beats: {test_split.labels.shape[0]}"
    )


if __name__ == "__main__":
    main()

"""Step 1: retain the 50 most frequent SkinCAP disease labels."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Cleaned SkinCAP CSV")
    parser.add_argument("--output", required=True, help="Top-label CSV")
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    frame = pd.read_csv(input_path)
    if "disease" not in frame.columns:
        raise ValueError("Input CSV does not contain a disease column")
    if args.top_k <= 1:
        raise ValueError("--top-k must be greater than one")


    disease_counts = frame["disease"].value_counts()
    if len(disease_counts) < args.top_k:
        raise ValueError(
            f"Requested {args.top_k} labels, found only {len(disease_counts)}"
        )
    top_labels = disease_counts.head(args.top_k).index
    filtered = frame[frame["disease"].isin(top_labels)].copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(output_path, index=False)
    print(
        {
            "stage": "top_labels",
            "input_labels": len(disease_counts),
            "selected_labels": len(top_labels),
            "rows": len(filtered),
            "minimum_class_count": int(
                filtered["disease"].value_counts().min()
            ),
            "output": str(output_path.resolve()),
        }
    )


if __name__ == "__main__":
    main()

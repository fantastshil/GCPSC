"""Step 2: convert the selected SkinCAP disease label to one-hot columns."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Top-label SkinCAP CSV")
    parser.add_argument("--output", required=True, help="One-hot SkinCAP CSV")
    parser.add_argument("--expected-labels", type=int, default=50)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    frame = pd.read_csv(input_path)
    required = {
        "skincap_file_path",
        "caption_zh_polish_en",
        "disease",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    disease_dummies = pd.get_dummies(
        frame["disease"], dtype=int
    ).sort_index(axis=1)
    if disease_dummies.shape[1] != args.expected_labels:
        raise ValueError(
            f"Expected {args.expected_labels} labels, "
            f"found {disease_dummies.shape[1]}"
        )
    final = pd.concat(
        [
            frame[["skincap_file_path", "caption_zh_polish_en"]],
            disease_dummies,
        ],
        axis=1,
    )
    if not final.iloc[:, 2:].sum(axis=1).eq(1).all():
        raise ValueError("One-hot conversion did not produce one label per row")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(output_path, index=False)
    print(
        {
            "stage": "one_hot",
            "rows": len(final),
            "labels": disease_dummies.shape[1],
            "output": str(output_path.resolve()),
        }
    )


if __name__ == "__main__":
    main()

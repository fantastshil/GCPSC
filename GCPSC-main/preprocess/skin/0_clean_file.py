

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = [
    "skincap_file_path",
    "disease",
    "caption_zh_polish_en",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Original SkinCAP CSV")
    parser.add_argument("--output", required=True, help="Cleaned CSV")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    frame = pd.read_csv(input_path)
    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in frame.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    cleaned = frame[REQUIRED_COLUMNS].copy()
    missing_rows = cleaned.isna().any(axis=1)
    if missing_rows.any():
        raise ValueError(
            f"{int(missing_rows.sum())} rows contain missing required fields"
        )
    for column in REQUIRED_COLUMNS:
        cleaned[column] = cleaned[column].astype(str).str.strip()
    empty_rows = cleaned.eq("").any(axis=1)
    if empty_rows.any():
        raise ValueError(
            f"{int(empty_rows.sum())} rows contain empty required fields"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False)
    print(
        {
            "stage": "clean",
            "input": str(input_path.resolve()),
            "output": str(output_path.resolve()),
            "rows": len(cleaned),
        }
    )


if __name__ == "__main__":
    main()

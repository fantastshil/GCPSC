"""Step 3: create reproducible stratified SkinCAP train/val/test splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def label_counts(frame: pd.DataFrame, label_columns: list[str]) -> dict[str, int]:
    return {
        label: int(frame[label].sum())
        for label in label_columns
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="One-hot SkinCAP CSV")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--image-root",
        default="",
        help="Optional image directory used to verify every CSV path",
    )
    parser.add_argument("--val-size", type=float, default=0.10)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not 0.0 < args.val_size < 1.0:
        raise ValueError("--val-size must be in (0, 1)")
    if not 0.0 < args.test_size < 1.0:
        raise ValueError("--test-size must be in (0, 1)")
    if args.val_size + args.test_size >= 1.0:
        raise ValueError("--val-size + --test-size must be less than one")

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    frame = pd.read_csv(input_path)
    label_columns = frame.columns[2:].tolist()
    if not label_columns:
        raise ValueError("No label columns found after the first two columns")

    label_matrix = frame[label_columns]
    row_label_counts = label_matrix.sum(axis=1)
    if not row_label_counts.eq(1).all():
        raise ValueError(
            "SkinCAP split expects exactly one positive disease per row"
        )
    disease = label_matrix.idxmax(axis=1)

    if args.image_root:
        image_root = Path(args.image_root)
        missing_images = [
            str(relative_path)
            for relative_path in frame["skincap_file_path"]
            if not (image_root / str(relative_path)).is_file()
        ]
        if missing_images:
            raise FileNotFoundError(
                f"{len(missing_images)} images are missing; "
                f"first examples={missing_images[:5]}"
            )

    train_val, test = train_test_split(
        frame,
        test_size=args.test_size,
        random_state=args.seed,
        shuffle=True,
        stratify=disease,
    )
    relative_val_size = args.val_size / (1.0 - args.test_size)
    train_val_disease = train_val[label_columns].idxmax(axis=1)
    train, val = train_test_split(
        train_val,
        test_size=relative_val_size,
        random_state=args.seed,
        shuffle=True,
        stratify=train_val_disease,
    )

    # Stable row order makes checksums and later missing-mask generation
    # reproducible without changing the stratified membership.
    train = train.sort_index().reset_index(drop=True)
    val = val.sort_index().reset_index(drop=True)
    test = test.sort_index().reset_index(drop=True)
    splits = {"train": train, "val": val, "test": test}

    for split_name, split_frame in splits.items():
        absent_labels = [
            label
            for label in label_columns
            if int(split_frame[label].sum()) == 0
        ]
        if absent_labels:
            raise ValueError(
                f"{split_name} split lacks labels: {absent_labels}"
            )

    path_sets = {
        name: set(split["skincap_file_path"].astype(str))
        for name, split in splits.items()
    }
    if (
        path_sets["train"] & path_sets["val"]
        or path_sets["train"] & path_sets["test"]
        or path_sets["val"] & path_sets["test"]
    ):
        raise ValueError("Image paths overlap across data splits")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "train": output_dir / "skincap_50_train_set.csv",
        "val": output_dir / "skincap_50_val_set.csv",
        "test": output_dir / "skincap_50_test_set.csv",
    }
    for name, path in output_paths.items():
        splits[name].to_csv(path, index=False)

    summary = {
        "source": str(input_path.resolve()),
        "seed": args.seed,
        "requested_ratios": {
            "train": 1.0 - args.val_size - args.test_size,
            "val": args.val_size,
            "test": args.test_size,
        },
        "total_rows": len(frame),
        "label_count": len(label_columns),
        "label_order": label_columns,
        "splits": {
            name: {
                "rows": len(split),
                "ratio": len(split) / len(frame),
                "minimum_class_count": min(
                    label_counts(split, label_columns).values()
                ),
                "maximum_class_count": max(
                    label_counts(split, label_columns).values()
                ),
                "class_counts": label_counts(split, label_columns),
                "output": str(output_paths[name].resolve()),
            }
            for name, split in splits.items()
        },
    }
    summary_path = output_dir / "skincap_50_split_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()



from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np
import yaml


EXPECTED_COUNTS = {"train": 42628, "val": 4802, "test": 11914}
ID_COLUMNS = {"stay", "period_length", "stay_id"}


def listfile_info(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    labels = [name for name in rows[0] if name not in ID_COLUMNS]
    return rows, labels


def main(config_path, output_path):
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    root = Path(config["dataset_root"])
    image_root = Path(config["image_root"])
    with (root / config["paired_metadata"]).open("rb") as handle:
        metadata = pickle.load(handle)

    report = {"splits": {}, "subject_overlap": {}, "errors": []}
    subject_sets = {}
    common_labels = None

    for split in ("train", "val", "test"):
        rows, labels = listfile_info(root / config[f"{split}_listfile"])
        common_labels = labels if common_labels is None else common_labels
        if labels != common_labels:
            report["errors"].append(f"{split}: label order differs")
        with (root / config[f"{split}_ehr_pkl"]).open("rb") as handle:
            ehr = pickle.load(handle)

        ordered_stays = [row["stay"] for row in rows]
        missing = set(ordered_stays) - set(ehr)
        extra = set(ehr) - set(ordered_stays)
        lengths, bad_features, bad_labels, nonfinite = [], 0, 0, 0
        for value in ehr.values():
            data = np.asarray(value["data"])
            target = np.asarray(value["ys"])
            lengths.append(int(data.shape[0]))
            bad_features += int(data.ndim != 2 or data.shape[1] != 76)
            bad_labels += int(target.shape != (25,))
            nonfinite += int(not np.isfinite(data).all())

        frame = metadata[split]
        duplicated = int(frame["stay"].duplicated().sum())
        missing_images = 0
        label_mismatches = 0
        for row in frame.to_dict(orient="records"):
            missing_images += int(not (image_root / f"{row['dicom_id']}.jpg").is_file())
            stay = str(row["stay"])
            if stay in ehr:
                metadata_target = np.asarray([row[name] for name in labels], dtype=np.float32)
                label_mismatches += int(
                    not np.array_equal(np.asarray(ehr[stay]["ys"], dtype=np.float32), metadata_target)
                )

        subject_sets[split] = {str(stay).split("_", 1)[0] for stay in ehr}
        split_report = {
            "samples": len(ehr),
            "expected_samples": EXPECTED_COUNTS[split],
            "listfile_rows": len(rows),
            "paired_rows": len(frame),
            "sequence_length_min": min(lengths),
            "sequence_length_max": max(lengths),
            "missing_from_pickle": len(missing),
            "extra_in_pickle": len(extra),
            "bad_feature_shapes": bad_features,
            "bad_label_shapes": bad_labels,
            "nonfinite_samples": nonfinite,
            "duplicate_paired_stays": duplicated,
            "missing_images": missing_images,
            "paired_label_mismatches": label_mismatches,
        }
        report["splits"][split] = split_report
        for key, value in split_report.items():
            if key.startswith(("missing_", "extra_", "bad_", "nonfinite_", "duplicate_")) and value:
                report["errors"].append(f"{split}: {key}={value}")
        if len(ehr) != EXPECTED_COUNTS[split]:
            report["errors"].append(f"{split}: unexpected sample count {len(ehr)}")

        del ehr

    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = len(subject_sets[left] & subject_sets[right])
        report["subject_overlap"][f"{left}_{right}"] = overlap
        if overlap:
            report["errors"].append(f"subject leakage: {left}/{right}={overlap}")

    guideline_path = Path(config["guideline_path"])
    with guideline_path.open("r", encoding="utf-8") as handle:
        guideline_queries = [json.loads(line)["query"] for line in handle if line.strip()]
    report["guideline_order_matches"] = guideline_queries == common_labels
    if not report["guideline_order_matches"]:
        report["errors"].append("guideline order differs from label order")
    report["ok"] = not report["errors"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/mimic_full.yaml"))
    parser.add_argument("--output", type=Path, default=Path("outputs/mimic_data_audit.json"))
    arguments = parser.parse_args()
    main(arguments.config, arguments.output)


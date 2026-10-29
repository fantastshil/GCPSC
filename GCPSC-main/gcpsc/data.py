

from __future__ import annotations

import csv
import json
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IDENTIFIER_COLUMNS = {"stay", "period_length", "stay_id"}
STAY_NAME_PATTERN = re.compile(
    r"^(?P<subject_id>\d+)_episode\d+_timeseries\.csv$"
)


def parse_subject_id(stay: str) -> int:
    """Extract the MIMIC subject identifier from a MEDFuse stay filename."""
    match = STAY_NAME_PATTERN.fullmatch(str(stay))
    if match is None:
        raise ValueError(
            "Cannot parse subject_id from stay name "
            f"{stay!r}; expected '<subject>_episode<n>_timeseries.csv'"
        )
    return int(match.group("subject_id"))


def read_label_names(listfile: str | Path) -> List[str]:
    with Path(listfile).open("r", encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle))
    labels = [column for column in header if column not in IDENTIFIER_COLUMNS]
    if len(labels) != 25:
        raise ValueError(f"Expected 25 phenotyping labels, found {len(labels)} in {listfile}")
    return labels


def read_stay_order(listfile: str | Path) -> List[str]:
    with Path(listfile).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [row["stay"] for row in reader]


def build_cxr_transform(image_size: int, training: bool):
    operations = [transforms.Resize(256)]
    if training:
        operations.append(
            transforms.RandomAffine(
                degrees=7,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
            )
        )
    operations.extend(
        [
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    return transforms.Compose(operations)


class GCPSCPhenotypingDataset(Dataset):


    def __init__(
        self,
        ehr_pickle: str | Path,
        listfile: str | Path,
        paired_metadata: str | Path,
        image_root: str | Path,
        split: str,
        image_size: int = 224,
        sample_mode: str = "all",
        augment: bool = False,
        strict_images: bool = False,
        verify_labels: bool = True,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split: {split}")
        if sample_mode not in {"all", "paired", "ehr_only"}:
            raise ValueError(f"Unsupported sample_mode: {sample_mode}")

        self.split = split
        self.sample_mode = sample_mode
        self.image_size = image_size
        self.image_root = Path(image_root)
        self.transform = build_cxr_transform(image_size, training=augment)
        self.label_names = read_label_names(listfile)

        with Path(ehr_pickle).open("rb") as handle:
            self.ehr: Dict[str, Dict[str, np.ndarray]] = pickle.load(handle)

        stay_order = read_stay_order(listfile)
        missing_ehr = [stay for stay in stay_order if stay not in self.ehr]
        if missing_ehr:
            raise KeyError(f"{len(missing_ehr)} listfile stays are absent from {ehr_pickle}")

        with Path(paired_metadata).open("rb") as handle:
            metadata_by_split = pickle.load(handle)
        metadata = metadata_by_split[split]
        if metadata["stay"].duplicated().any():
            duplicates = int(metadata["stay"].duplicated().sum())
            raise ValueError(f"Paired metadata contains {duplicates} duplicate stays in {split}")

        cxr_by_stay: Dict[str, dict] = {}
        for row in metadata.to_dict(orient="records"):
            stay = str(row["stay"])
            dicom_id = str(row["dicom_id"])
            image_path = self.image_root / f"{dicom_id}.jpg"
            exists = image_path.is_file()
            if strict_images and not exists:
                raise FileNotFoundError(image_path)
            row["image_path"] = image_path
            row["image_exists"] = exists
            cxr_by_stay[stay] = row

            if verify_labels:
                ehr_labels = np.asarray(self.ehr[stay]["ys"], dtype=np.float32)
                metadata_labels = np.asarray([row[name] for name in self.label_names], dtype=np.float32)
                if not np.array_equal(ehr_labels, metadata_labels):
                    raise ValueError(f"Label mismatch between EHR and CXR metadata for {stay}")

        self.records = []
        for stay in stay_order:
            cxr = cxr_by_stay.get(stay)
            available = bool(cxr and cxr["image_exists"] and sample_mode != "ehr_only")
            if sample_mode == "paired" and not available:
                continue
            # Parse during initialization so malformed identifiers fail before
            # a long training or evaluation run begins.
            self.records.append(
                {
                    "stay": stay,
                    "subject_id": parse_subject_id(stay),
                    "cxr": cxr,
                    "image_available": available,
                }
            )

    @property
    def paired_count(self) -> int:
        return sum(record["image_available"] for record in self.records)

    def __len__(self) -> int:
        return len(self.records)

    def ehr_tensors(self, index: int):
        """Build the padded (48, 76) EHR tensor, its time mask and the labels."""
        record = self.records[index]
        stay = record["stay"]
        ehr_item = self.ehr[stay]
        ehr_array = np.asarray(ehr_item["data"], dtype=np.float32)
        label = torch.as_tensor(np.asarray(ehr_item["ys"], dtype=np.float32))

        if ehr_array.ndim != 2 or ehr_array.shape[1] != 76:
            raise ValueError(f"Unexpected EHR shape {tuple(ehr_array.shape)} for {stay}")
        if label.shape != (25,):
            raise ValueError(f"Unexpected label shape {tuple(label.shape)} for {stay}")

        valid_length = min(int(ehr_array.shape[0]), 48)
        ehr = torch.zeros(48, 76, dtype=torch.float32)
        ehr[:valid_length] = torch.from_numpy(ehr_array[:valid_length])
        ehr_mask = torch.zeros(48, dtype=torch.bool)
        ehr_mask[:valid_length] = True
        return ehr, ehr_mask, label

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        stay = record["stay"]
        ehr, ehr_mask, label = self.ehr_tensors(index)

        cxr = record["cxr"]
        if record["image_available"]:
            with Image.open(cxr["image_path"]) as image_file:
                image = self.transform(image_file.convert("RGB"))
            dicom_id = str(cxr["dicom_id"])
            stay_id = int(cxr["stay_id"])
        else:
            image = torch.zeros(3, self.image_size, self.image_size, dtype=torch.float32)
            dicom_id = ""
            stay_id = int(cxr["stay_id"]) if cxr is not None else -1

        return {
            "ehr": ehr,
            "ehr_mask": ehr_mask,
            "image": image,
            "image_available": torch.tensor(record["image_available"], dtype=torch.bool),
            "label": label,
            "stay": stay,
            # Patient identifier: repeated ICU stays form one bootstrap cluster.
            "subject_id": record["subject_id"],
            "stay_id": stay_id,
            "dicom_id": dicom_id,
        }


def load_guidelines(path: str | Path, label_names: Sequence[str]) -> tuple[List[str], List[str]]:
    guidelines: List[str] = []
    radiology_sections: List[str] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if len(rows) != len(label_names):
        raise ValueError(f"Expected {len(label_names)} guidelines, found {len(rows)}")

    for expected, row in zip(label_names, rows):
        if row.get("query") != expected:
            raise ValueError(f"Guideline order mismatch: expected {expected!r}, got {row.get('query')!r}")
        content = row["guideline_1_content"]
        sections = [section.strip() for section in content.split("\n\n") if section.strip()]
        radiology = next(
            (section for section in sections if "Radiological" in section or "Imaging" in section),
            sections[-1],
        )
        guidelines.append(content)
        radiology_sections.append(radiology)
    return guidelines, radiology_sections


# ---------------------------------------------------------------- SkinCAP
SPLIT_SEED_OFFSETS = {"train": 0, "val": 10_000, "test": 20_000}


def build_stratified_missing_mask(
    labels: np.ndarray,
    missing_rate: float,
    seed: int,
) -> np.ndarray:
    """Create a deterministic class-stratified mask nested across rates."""
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("labels must be a one-dimensional class-index array")
    if not 0.0 <= missing_rate <= 1.0:
        raise ValueError("missing_rate must be between zero and one")

    sample_count = labels.shape[0]
    missing = np.zeros(sample_count, dtype=bool)
    if missing_rate == 0.0:
        return missing
    if missing_rate == 1.0:
        return ~missing


    seed_sequence = np.random.SeedSequence(seed)
    classes = np.unique(labels)
    class_seeds = seed_sequence.spawn(len(classes))
    priorities = np.empty(sample_count, dtype=np.float64)
    for class_value, class_seed in zip(classes, class_seeds):
        indices = np.flatnonzero(labels == class_value)
        order = np.random.default_rng(class_seed).permutation(indices)
        quantiles = (np.arange(len(indices), dtype=np.float64) + 0.5) / len(
            indices
        )
        priorities[order] = quantiles
    target_count = int(np.floor(sample_count * missing_rate + 0.5))
    ranked = np.lexsort((np.arange(sample_count), priorities))
    missing[ranked[:target_count]] = True
    return missing


class SkinCAPMissingDataset(Dataset):

    """Return tokenized caption, optional skin image, and 50-way one-hot label."""

    def __init__(
        self,
        csv_path: str | Path,
        image_root: str | Path,
        tokenizer,
        split: str,
        missing_rate: float,
        missing_seed: int = 42,
        image_size: int = 224,
        caption_max_length: int = 128,
        caption_column: str = "caption_zh_polish_en",
        augment: bool = False,
        expected_label_names: Sequence[str] | None = None,
    ) -> None:
        if split not in SPLIT_SEED_OFFSETS:
            raise ValueError(f"Unsupported split: {split}")
        self.csv_path = Path(csv_path)
        self.image_root = Path(image_root)
        self.split = split
        self.missing_rate = float(missing_rate)
        self.missing_seed = (
            int(missing_seed) + SPLIT_SEED_OFFSETS[split]
        )
        self.sample_mode = f"mcar_{self.missing_rate:.2f}"
        self.image_size = int(image_size)
        self.transform = build_cxr_transform(
            self.image_size,
            training=bool(augment),
        )

        frame = pd.read_csv(self.csv_path)
        if frame.shape[1] < 3:
            raise ValueError(f"{self.csv_path} has no label columns")
        self.label_names = frame.columns[2:].tolist()
        if expected_label_names is not None and list(
            expected_label_names
        ) != self.label_names:
            raise ValueError(
                f"Label order mismatch in {self.csv_path}"
            )

        label_matrix = frame.iloc[:, 2:].to_numpy(dtype=np.float32)
        if not np.isin(label_matrix, (0.0, 1.0)).all():
            raise ValueError(f"{self.csv_path} labels are not binary")
        if not np.equal(label_matrix.sum(axis=1), 1.0).all():
            raise ValueError(
                "SkinCAP adapter expects exactly one positive disease per row"
            )
        class_indices = label_matrix.argmax(axis=1)
        missing = build_stratified_missing_mask(
            class_indices,
            self.missing_rate,
            self.missing_seed,
        )

        relative_paths = frame.iloc[:, 0].astype(str).tolist()
        image_paths = [self.image_root / path for path in relative_paths]
        absent = [
            str(path)
            for path in image_paths
            if not path.is_file()
        ]
        if absent:
            raise FileNotFoundError(
                f"{len(absent)} SkinCAP images are missing; "
                f"first examples={absent[:5]}"
            )

        if frame.columns[1] != caption_column:
            raise ValueError(
                f"{self.csv_path} expected second column "
                f"{caption_column!r}, found {frame.columns[1]!r}"
            )
        captions = frame[caption_column].astype(str).tolist()
        encoded = tokenizer(
            captions,
            padding="max_length",
            truncation=True,
            max_length=int(caption_max_length),
            return_tensors="pt",
        )
        self.input_ids = encoded["input_ids"]
        self.attention_mask = encoded["attention_mask"].bool()
        self.labels = torch.from_numpy(label_matrix)
        self.relative_paths = relative_paths
        self.image_paths = image_paths
        self.image_available = torch.from_numpy(~missing)

    @property
    def paired_count(self) -> int:
        return int(self.image_available.sum().item())

    @property
    def actual_missing_rate(self) -> float:
        return 1.0 - self.paired_count / max(len(self), 1)

    def __len__(self) -> int:
        return len(self.relative_paths)

    def __getitem__(self, index: int) -> dict:
        available = bool(self.image_available[index])
        if available:
            with Image.open(self.image_paths[index]) as image_file:
                image = self.transform(image_file.convert("RGB"))
        else:
            image = torch.zeros(
                3,
                self.image_size,
                self.image_size,
                dtype=torch.float32,
            )

        relative_path = self.relative_paths[index]
        stem = Path(relative_path).stem
        subject_id = int(stem) if stem.isdigit() else int(index)
        return {
            "ehr": self.input_ids[index],
            "ehr_mask": self.attention_mask[index],
            "image": image,
            "image_available": self.image_available[index],
            "label": self.labels[index],
            "stay": relative_path,

            "subject_id": subject_id,
            "stay_id": int(index),
            "dicom_id": stem,
        }

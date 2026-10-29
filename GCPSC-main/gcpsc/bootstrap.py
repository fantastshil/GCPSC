

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm.auto import tqdm


SUPPORTED_METRICS = ("macro_auroc", "macro_auprc")
SUPPORTED_UNITS = ("patient", "sample")
SUBGROUP_MASKS = {
    "all": lambda available: np.ones(available.shape[0], dtype=bool),
    "paired": lambda available: available,
    "ehr_only": lambda available: ~available,
}


@dataclass(frozen=True)
class BootstrapSettings:
    iterations: int = 1000
    confidence_level: float = 0.95
    method: str = "percentile"
    unit: str = "patient"
    seed: int = 42
    max_attempt_multiplier: int = 10
    show_progress: bool = True
    metrics: tuple[str, ...] = SUPPORTED_METRICS
    subgroups: tuple[str, ...] = tuple(SUBGROUP_MASKS)

    @classmethod
    def from_mapping(cls, values: Mapping | None) -> "BootstrapSettings":
        values = values or {}
        settings = cls(
            iterations=int(values.get("iterations", 1000)),
            confidence_level=float(values.get("confidence_level", 0.95)),
            method=str(values.get("method", "percentile")),
            unit=str(values.get("unit", "patient")),
            seed=int(values.get("seed", 42)),
            max_attempt_multiplier=int(
                values.get("max_attempt_multiplier", 10)
            ),
            show_progress=bool(values.get("show_progress", True)),
            metrics=tuple(values.get("metrics", SUPPORTED_METRICS)),
            subgroups=tuple(values.get("subgroups", SUBGROUP_MASKS)),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.iterations <= 0:
            raise ValueError("bootstrap iterations must be positive")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("bootstrap confidence_level must be between 0 and 1")
        if self.method != "percentile":
            raise ValueError(
                "Only percentile bootstrap confidence intervals are supported"
            )
        if self.unit not in SUPPORTED_UNITS:
            raise ValueError(
                f"Unsupported bootstrap unit {self.unit!r}; "
                f"choose one of {SUPPORTED_UNITS}"
            )
        if self.max_attempt_multiplier <= 0:
            raise ValueError("bootstrap max_attempt_multiplier must be positive")
        unknown_metrics = set(self.metrics) - set(SUPPORTED_METRICS)
        if unknown_metrics:
            raise ValueError(f"Unsupported bootstrap metrics: {unknown_metrics}")
        unknown_subgroups = set(self.subgroups) - set(SUBGROUP_MASKS)
        if unknown_subgroups:
            raise ValueError(f"Unsupported bootstrap subgroups: {unknown_subgroups}")
        if not self.metrics:
            raise ValueError("At least one bootstrap metric is required")
        if not self.subgroups:
            raise ValueError("At least one bootstrap subgroup is required")


def _validate_arrays(labels, predictions, availability, subject_ids):
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    availability = np.asarray(availability, dtype=bool)
    subject_ids = np.asarray(subject_ids)
    if labels.ndim != 2 or predictions.shape != labels.shape:
        raise ValueError(
            "labels and predictions must have the same [samples, labels] shape"
        )
    sample_count = labels.shape[0]
    if availability.shape != (sample_count,):
        raise ValueError("availability must have shape [samples]")
    if subject_ids.shape != (sample_count,):
        raise ValueError("subject_ids must have shape [samples]")
    if not np.isfinite(labels).all() or not np.isfinite(predictions).all():
        raise ValueError("Bootstrap inputs contain NaN or Inf")
    return labels, predictions, availability, subject_ids


def _invalid_label_indices(labels: np.ndarray) -> list[int]:
    """Return labels without both a positive and a negative observation."""
    positive = labels.sum(axis=0)
    return np.flatnonzero(
        (positive <= 0) | (positive >= labels.shape[0])
    ).astype(int).tolist()


def _macro_scores(
    labels: np.ndarray,
    predictions: np.ndarray,
    metrics: Sequence[str],
) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    if "macro_auroc" in metrics:
        scores["macro_auroc"] = float(
            np.mean(
                [
                    roc_auc_score(labels[:, index], predictions[:, index])
                    for index in range(labels.shape[1])
                ]
            )
        )
    if "macro_auprc" in metrics:
        scores["macro_auprc"] = float(
            np.mean(
                [
                    average_precision_score(
                        labels[:, index], predictions[:, index]
                    )
                    for index in range(labels.shape[1])
                ]
            )
        )
    return scores


def _cluster_rows(subject_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique_subjects, inverse = np.unique(subject_ids, return_inverse=True)
    return unique_subjects, inverse


def _draw_indices(
    rng: np.random.Generator,
    sample_count: int,
    unit: str,
    patient_inverse: np.ndarray | None,
) -> np.ndarray:
    if unit == "sample":
        return rng.integers(0, sample_count, size=sample_count)
    if patient_inverse is None:
        raise RuntimeError("patient_inverse is required for patient bootstrap")
    patient_count = int(patient_inverse.max()) + 1
    sampled_patients = rng.integers(
        0, patient_count, size=patient_count
    )
    multiplicities = np.bincount(
        sampled_patients, minlength=patient_count
    )
    # Repeated patients duplicate every one of their stays. Using per-row
    # multiplicities avoids concatenating thousands of tiny arrays per draw.
    return np.repeat(
        np.arange(sample_count),
        multiplicities[patient_inverse],
    )


def _bootstrap_one_subgroup(
    labels: np.ndarray,
    predictions: np.ndarray,
    subject_ids: np.ndarray,
    settings: BootstrapSettings,
    rng: np.random.Generator,
    description: str,
) -> tuple[dict, Dict[str, np.ndarray]]:
    invalid_original = _invalid_label_indices(labels)
    if invalid_original:
        raise ValueError(
            "Cannot compute all-label bootstrap metrics because the original "
            f"subgroup lacks both classes for label indices {invalid_original}"
        )

    unique_subjects, patient_inverse = _cluster_rows(subject_ids)
    distributions = {
        metric: np.empty(settings.iterations, dtype=np.float64)
        for metric in settings.metrics
    }
    valid = 0
    attempts = 0
    maximum_attempts = settings.iterations * settings.max_attempt_multiplier
    progress = tqdm(
        total=settings.iterations,
        desc=f"Bootstrap [{description}]",
        dynamic_ncols=True,
        mininterval=0.5,
        disable=not settings.show_progress,
    )
    try:
        while valid < settings.iterations and attempts < maximum_attempts:
            attempts += 1
            indices = _draw_indices(
                rng,
                labels.shape[0],
                settings.unit,
                patient_inverse if settings.unit == "patient" else None,
            )
            sampled_labels = labels[indices]
            if _invalid_label_indices(sampled_labels):
                continue
            sampled_scores = _macro_scores(
                sampled_labels, predictions[indices], settings.metrics
            )
            for metric, value in sampled_scores.items():
                distributions[metric][valid] = value
            valid += 1
            progress.update(1)
            if attempts != valid:
                progress.set_postfix(rejected=attempts - valid)
    finally:
        progress.close()

    if valid < settings.iterations:
        raise RuntimeError(
            f"Only obtained {valid}/{settings.iterations} valid bootstrap "
            f"replicates after {attempts} attempts"
        )

    alpha = 1.0 - settings.confidence_level
    lower_percentile = 100.0 * alpha / 2.0
    upper_percentile = 100.0 * (1.0 - alpha / 2.0)
    point_estimates = _macro_scores(labels, predictions, settings.metrics)
    summary = {
        "sample_count": int(labels.shape[0]),
        "patient_count": int(unique_subjects.shape[0]),
        "valid_iterations": valid,
        "rejected_iterations": attempts - valid,
        "metrics": {
            metric: {
                "point_estimate": point_estimates[metric],
                "lower": float(
                    np.percentile(distributions[metric], lower_percentile)
                ),
                "upper": float(
                    np.percentile(distributions[metric], upper_percentile)
                ),
            }
            for metric in settings.metrics
        },
    }
    return summary, distributions


def bootstrap_confidence_intervals(
    labels,
    predictions,
    availability,
    subject_ids,
    config: Mapping | None = None,
) -> tuple[dict, Dict[str, np.ndarray]]:
    """Compute percentile CIs for configured test subgroups and metrics."""
    labels, predictions, availability, subject_ids = _validate_arrays(
        labels, predictions, availability, subject_ids
    )
    settings = BootstrapSettings.from_mapping(config)
    result = {
        "settings": {
            "iterations": settings.iterations,
            "confidence_level": settings.confidence_level,
            "method": settings.method,
            "unit": settings.unit,
            "seed": settings.seed,
            "max_attempt_multiplier": settings.max_attempt_multiplier,
            "show_progress": settings.show_progress,
            "metrics": list(settings.metrics),
            "subgroups": list(settings.subgroups),
        },
        "subgroups": {},
    }
    saved_distributions: Dict[str, np.ndarray] = {}
    # Separate deterministic streams keep one subgroup's result unchanged when
    # another subgroup is added to or removed from the configuration.
    seed_sequence = np.random.SeedSequence(settings.seed)
    child_seeds = seed_sequence.spawn(len(settings.subgroups))
    for subgroup, child_seed in zip(settings.subgroups, child_seeds):
        mask = SUBGROUP_MASKS[subgroup](availability)
        if not mask.any():
            raise ValueError(f"Bootstrap subgroup {subgroup!r} is empty")
        subgroup_summary, distributions = _bootstrap_one_subgroup(
            labels[mask],
            predictions[mask],
            subject_ids[mask],
            settings,
            np.random.default_rng(child_seed),
            subgroup,
        )
        result["subgroups"][subgroup] = subgroup_summary
        for metric, values in distributions.items():
            saved_distributions[f"{subgroup}_{metric}"] = values
    return result, saved_distributions


def main() -> None:
    """Recompute intervals from a saved test.npz without model inference."""
    parser = argparse.ArgumentParser(
        description="Bootstrap confidence intervals from fixed test predictions"
    )
    parser.add_argument("--input", required=True, help="Path to test.npz")
    parser.add_argument("--output", help="Output JSON path")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--unit", choices=SUPPORTED_UNITS, default="patient")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-attempt-multiplier", type=int, default=10)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = (
        Path(args.output)
        if args.output
        else input_path.with_name(f"{input_path.stem}_bootstrap.json")
    )
    with np.load(input_path) as arrays:
        required = {
            "labels",
            "predictions",
            "image_available",
            "subject_ids",
        }
        missing = required - set(arrays.files)
        if missing:
            raise KeyError(
                f"{input_path} is missing arrays required for bootstrap: "
                f"{sorted(missing)}"
            )
        summary, distributions = bootstrap_confidence_intervals(
            arrays["labels"],
            arrays["predictions"],
            arrays["image_available"],
            arrays["subject_ids"],
            {
                "iterations": args.iterations,
                "confidence_level": args.confidence_level,
                "method": "percentile",
                "unit": args.unit,
                "seed": args.seed,
                "max_attempt_multiplier": args.max_attempt_multiplier,
            },
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
    scores_path = output_path.with_name(
        f"{output_path.stem}_scores.npz"
    )
    np.savez_compressed(scores_path, **distributions)
    print(f"Saved bootstrap summary to {output_path}")
    print(f"Saved bootstrap distributions to {scores_path}")


if __name__ == "__main__":
    main()



from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


PRIMARY_LESIONS = {
    "Vesicle": "vesicle",
    "Papule": "papule",
    "Macule": "macule",
    "Plaque": "plaque",
    "Abscess": "abscess",
    "Pustule": "pustule",
    "Bulla": "bulla",
    "Patch": "patch",
    "Nodule": "nodule",
    "Ulcer": "ulcer",
    "Wheal": "wheal",
    "Cyst": "cyst",
}

SURFACE_FEATURES = {
    "Crust": "crusting",
    "Erosion": "erosion",
    "Excoriation": "excoriation",
    "Atrophy": "atrophy",
    "Exudate": "exudate",
    "Purpura/Petechiae": "purpura or petechiae",
    "Fissure": "fissuring",
    "Induration": "induration",
    "Xerosis": "xerosis",
    "Telangiectasia": "telangiectatic change",
    "Scale": "scaling",
    "Scar": "scarring",
    "Friable": "friability",
    "Sclerosis": "sclerosis",
    "Comedo": "comedonal change",
    "Lichenification": "lichenification",
}

ARCHITECTURAL_FEATURES = {
    "Pedunculated": "pedunculated",
    "Exophytic/Fungating": "exophytic or fungating",
    "Warty/Papillomatous": "warty or papillomatous",
    "Dome-shaped": "dome-shaped",
    "Flat topped": "flat-topped",
    "Translucent": "translucent",
    "Umbilicated": "umbilicated",
    "Poikiloderma": "poikilodermatous",
    "Acuminate": "acuminate",
    "Burrow": "burrow-like",
}

COLOR_FEATURES = {
    "Brown(Hyperpigmentation)": "increased brown pigment",
    "White(Hypopigmentation)": "reduced white pigment",
    "Purple": "purple coloration",
    "Yellow": "yellow coloration",
    "Black": "black coloration",
    "Erythema": "erythema",
    "Blue": "blue coloration",
    "Salmon": "salmon coloration",
    "Gray": "gray coloration",
    "Pigmented": "pigmentation",
}

ATTRIBUTE_GROUPS = {
    "primary lesion morphology": PRIMARY_LESIONS,
    "surface features": SURFACE_FEATURES,
    "architectural features": ARCHITECTURAL_FEATURES,
    "color and pigmentation features": COLOR_FEATURES,
}

WHITELIST_CLINICAL_GROUPS = {
    "anatomical locations": {
        "face": (r"\bface\b", r"\bfacial\b"),
        "scalp": (r"\bscalp\b",),
        "forehead": (r"\bforehead\b",),
        "eyelid": (r"\beyelids?\b",),
        "periocular region": (r"\bperiocular\b", r"\baround the eyes?\b"),
        "nose": (r"\bnose\b", r"\bnasal\b"),
        "mouth or lips": (r"\bmouth\b", r"\boral\b", r"\blips?\b"),
        "neck": (r"\bneck\b",),
        "chest": (r"\bchest\b",),
        "back": (r"\bback\b",),
        "trunk": (r"\btrunk\b", r"\btorso\b"),
        "abdomen": (r"\babdomen\b", r"\babdominal\b"),
        "groin": (r"\bgroin\b", r"\binguinal\b"),
        "buttock": (r"\bbuttocks?\b", r"\bgluteal\b"),
        "upper limb": (r"\bupper limbs?\b", r"\bupper extremit(?:y|ies)\b"),
        "arm": (r"\barms?\b",),
        "elbow": (r"\belbows?\b",),
        "forearm": (r"\bforearms?\b",),
        "wrist": (r"\bwrists?\b",),
        "hand": (r"\bhands?\b",),
        "palm": (r"\bpalms?\b",),
        "finger": (r"\bfingers?\b",),
        "lower limb": (r"\blower limbs?\b", r"\blower extremit(?:y|ies)\b"),
        "leg": (r"\blegs?\b",),
        "thigh": (r"\bthighs?\b",),
        "knee": (r"\bknees?\b",),
        "ankle": (r"\bankles?\b",),
        "foot": (r"\bfoot\b", r"\bfeet\b"),
        "sole": (r"\bsole\b", r"\bplantar\b"),
        "toe": (r"\btoes?\b",),
    },
    "distribution and geometry": {
        "solitary": (r"\bsolitary\b", r"\bsingle lesion\b"),
        "multiple": (r"\bmultiple\b", r"\bnumerous\b", r"\bseveral\b"),
        "clustered": (r"\bcluster(?:ed|s)?\b", r"\bgrouped\b"),
        "scattered": (r"\bscattered\b", r"\bdisseminated\b"),
        "diffuse": (r"\bdiffuse\b", r"\bwidespread\b", r"\bgeneralized\b"),
        "bilateral": (r"\bbilateral\b",),
        "symmetric": (r"\bsymmetric(?:al)?\b",),
        "asymmetric": (r"\basymmetric(?:al)?\b",),
        "localized": (r"\blocali[sz]ed\b",),
        "linear": (r"\blinear\b",),
        "annular": (r"\bannular\b", r"\bring[- ]shaped\b"),
        "irregular": (r"\birregular(?:ly)?\b",),
        "well-demarcated": (
            r"\bwell[- ]defined\b",
            r"\bsharply demarcated\b",
            r"\bclear boundaries\b",
            r"\bclear border\b",
        ),
        "ill-defined": (
            r"\bill[- ]defined\b",
            r"\bpoorly defined\b",
            r"\bblurred boundaries\b",
        ),
    },
    "size descriptors": {
        "tiny": (r"\btiny\b",),
        "small": (r"\bsmall\b",),
        "large": (r"\blarge\b",),
        "giant": (r"\bgiant\b",),
    },
    "symptoms and course": {
        "pruritic": (r"\bpruritic\b", r"\bitch(?:y|ing)\b"),
        "painful": (r"\bpainful\b", r"\bpain\b"),
        "tender": (r"\btender(?:ness)?\b",),
        "burning": (r"\bburning\b",),
        "asymptomatic": (r"\basymptomatic\b",),
        "acute": (r"\bacute\b",),
        "chronic": (r"\bchronic\b",),
        "recurrent": (r"\brecurrent\b", r"\brecurring\b"),
        "progressive": (r"\bprogressive\b", r"\bprogressing\b"),
    },
}

OUTPUT_CAPTION_COLUMN = "caption_hybrid_en"
NO_ATTRIBUTE_CAPTION = (
    "No reliable structured or whitelisted clinical lesion attributes are "
    "available for this sample."
)
SPLIT_FILENAMES = {
    "train": "skincap_50_train_set.csv",
    "val": "skincap_50_val_set.csv",
    "test": "skincap_50_test_set.csv",
}


def _join_terms(terms: list[str]) -> str:
    if len(terms) == 1:
        return terms[0]
    if len(terms) == 2:
        return f"{terms[0]} and {terms[1]}"
    return f"{', '.join(terms[:-1])}, and {terms[-1]}"


def extract_whitelist_clinical_features(
    source_caption: str,
) -> dict[str, list[str]]:
    """Extract only fixed, pre-diagnostic concepts from source prose."""

    source_caption = str(source_caption).lower()
    extracted = {}
    for group_name, mapping in WHITELIST_CLINICAL_GROUPS.items():
        values = [
            normalized_value
            for normalized_value, patterns in mapping.items()
            if any(
                re.search(pattern, source_caption, flags=re.IGNORECASE)
                for pattern in patterns
            )
        ]
        if values:
            extracted[group_name] = values
    return extracted


def build_caption(row: pd.Series) -> str:

    sentences = []
    for group_name, mapping in ATTRIBUTE_GROUPS.items():
        positive_terms = [
            text
            for column, text in mapping.items()
            if int(row[column]) == 1
        ]
        if positive_terms:
            sentences.append(
                f"Observed {group_name}: {_join_terms(positive_terms)}."
            )
    whitelist_features = extract_whitelist_clinical_features(
        row["caption_zh_polish_en"]
    )
    for group_name, values in whitelist_features.items():
        sentences.append(
            f"Recorded {group_name}: {_join_terms(values)}."
        )
    return " ".join(sentences) if sentences else NO_ATTRIBUTE_CAPTION


def normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def contains_disease_name(caption: str, disease_names: list[str]) -> bool:
    normalized_caption = f" {normalize_text(caption)} "
    return any(
        f" {normalize_text(disease)} " in normalized_caption
        for disease in disease_names
        if normalize_text(disease)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-csv", required=True)
    parser.add_argument(
        "--split-dir",
        required=True,
        help="Directory containing the existing split CSVs",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    raw_path = Path(args.raw_csv)
    split_dir = Path(args.split_dir)
    output_dir = Path(args.output_dir)
    raw = pd.read_csv(raw_path)

    required = {
        "skincap_file_path",
        "disease",
        "caption_zh_polish_en",
        "Do not consider this image",
    }
    for mapping in ATTRIBUTE_GROUPS.values():
        required.update(mapping)
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"Raw SkinCAP CSV lacks columns: {missing}")
    if raw["skincap_file_path"].duplicated().any():
        raise ValueError("Raw SkinCAP image paths are not unique")

    attribute_columns = [
        column
        for mapping in ATTRIBUTE_GROUPS.values()
        for column in mapping
    ]
    non_binary = [
        column
        for column in attribute_columns
        if not set(raw[column].dropna().unique()).issubset({0, 1})
    ]
    if non_binary:
        raise ValueError(f"Non-binary structured attributes: {non_binary}")

    raw = raw.copy()
    raw[OUTPUT_CAPTION_COLUMN] = raw.apply(build_caption, axis=1)
    caption_by_path = raw.set_index("skincap_file_path")[
        OUTPUT_CAPTION_COLUMN
    ]
    disease_names = sorted(raw["disease"].astype(str).unique().tolist())

    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}
    output_frames = {}
    for split, filename in SPLIT_FILENAMES.items():
        input_path = split_dir / filename
        frame = pd.read_csv(input_path)
        paths = frame.iloc[:, 0].astype(str)
        missing_paths = sorted(set(paths).difference(caption_by_path.index))
        if missing_paths:
            raise ValueError(
                f"{split} contains paths absent from raw CSV: "
                f"{missing_paths[:5]}"
            )

        rebuilt = pd.concat(
            [
                frame.iloc[:, [0]].reset_index(drop=True),
                paths.map(caption_by_path)
                .rename(OUTPUT_CAPTION_COLUMN)
                .reset_index(drop=True),
                frame.iloc[:, 2:].reset_index(drop=True),
            ],
            axis=1,
        )
        leaked = rebuilt[OUTPUT_CAPTION_COLUMN].map(
            lambda caption: contains_disease_name(
                caption,
                disease_names,
            )
        )
        if leaked.any():
            examples = rebuilt.loc[
                leaked, [rebuilt.columns[0], OUTPUT_CAPTION_COLUMN]
            ].head(5)
            raise ValueError(
                "Disease-name leakage remains in structured captions: "
                f"{examples.to_dict(orient='records')}"
            )

        output_path = output_dir / filename
        rebuilt.to_csv(output_path, index=False)
        output_frames[split] = rebuilt
        summaries[split] = {
            "rows": len(rebuilt),
            "unique_captions": int(
                rebuilt[OUTPUT_CAPTION_COLUMN].nunique()
            ),
            "no_attribute_rows": int(
                rebuilt[OUTPUT_CAPTION_COLUMN].eq(
                    NO_ATTRIBUTE_CAPTION
                ).sum()
            ),
            "rows_with_whitelist_clinical_information": int(
                paths.map(
                    raw.set_index("skincap_file_path")[
                        "caption_zh_polish_en"
                    ]
                )
                .map(extract_whitelist_clinical_features)
                .map(bool)
                .sum()
            ),
            "disease_name_leakage_rows": int(leaked.sum()),
            "output": str(output_path.resolve()),
        }

    path_sets = {
        split: set(frame.iloc[:, 0].astype(str))
        for split, frame in output_frames.items()
    }
    if (
        path_sets["train"] & path_sets["val"]
        or path_sets["train"] & path_sets["test"]
        or path_sets["val"] & path_sets["test"]
    ):
        raise ValueError("Image paths overlap across rebuilt splits")

    audit = {
        "caption_source": "structured_plus_whitelist_clinical_v1",
        "raw_source": str(raw_path.resolve()),
        "preserved_split_source": str(split_dir.resolve()),
        "caption_column": OUTPUT_CAPTION_COLUMN,
        "uses_disease_to_generate_caption": False,
        "excluded_input_fields": [
            "disease",
            "malignant",
            "source",
            "caption_zh",
            "caption_zh_polish",
            "caption_zh_polish_en",
            "remark",
            "Do not consider this image",
        ],
        "attribute_groups": {
            group: list(mapping)
            for group, mapping in ATTRIBUTE_GROUPS.items()
        },
        "whitelist_clinical_groups": {
            group: list(mapping)
            for group, mapping in WHITELIST_CLINICAL_GROUPS.items()
        },
        "splits": summaries,
    }
    summary_path = output_dir / "hybrid_caption_audit.json"
    summary_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

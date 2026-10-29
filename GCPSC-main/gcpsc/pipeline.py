

from __future__ import annotations

import copy
import csv
import gc
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from gcpsc.data import GCPSCPhenotypingDataset, SkinCAPMissingDataset, load_guidelines
from gcpsc.decoder import KnowledgeQueryDecoder
from gcpsc.encoders import ClinicalTextSequenceEncoder, ImageEncoder, KnowledgeTextEncoder, StructuredEHREncoder
from gcpsc.engine import LOSS_WEIGHT_NAMES, evaluate, train_one_epoch
from gcpsc.model import GCPSCModel

try:
    from tensorboardX import SummaryWriter
except ImportError:  # optional
    SummaryWriter = None

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("mimic_full", "mimic_paired", "skincap")
CHECKPOINT_FORMAT = "gcpsc-core-v1"
LEGACY_PREFIXES = (
    "propensity_estimator.", "propensity_router_adapter.", "conditional_availability_discriminator.",
    "clinical_adapter.", "acquisition_projection.", "independent_acquisition_encoder.",
    "acquisition_head.", "clinical_availability_head.",
)
ABLATION_DEFAULTS = {"completion_guideline_conditioning": True, "router_mode": "learned"}
OBJECTIVE_KEYS = LOSS_WEIGHT_NAMES + ("classification_warmup_epochs", "guidance_warmup_epochs", "num_classes")


def parse_overrides(items):
    overrides = {}
    for item in items or ():
        key, sep, value = str(item).partition("=")
        if not sep or not key.strip():
            raise ValueError(f"--set expects KEY=VALUE, got {item!r}")
        overrides[key.strip()] = yaml.safe_load(value)
    return overrides


def apply_overrides(config, overrides):
    for key, value in overrides.items():
        target, parts = config, key.split(".")
        for part in parts[:-1]:
            if not isinstance(target.get(part), dict):
                raise KeyError(f"--set {key}: {part!r} is not a section of the config")
            target = target[part]
        if parts[-1] not in target:
            raise KeyError(f"--set {key}: no such key in the config")
        target[parts[-1]] = value
    return config


def load_config(dataset, config_path=None, seed=None, overrides=None):
    path = Path(config_path) if config_path else ROOT / "configs" / f"{dataset}.yaml"
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config.get("dataset") != dataset:
        raise ValueError(f"{path} is a config for {config.get('dataset')!r}, not {dataset!r}")
    for key, value in ABLATION_DEFAULTS.items():
        config.setdefault(key, value)
    if overrides:
        if "dataset" in overrides:
            raise KeyError("--set cannot change 'dataset'; use --dataset")
        apply_overrides(config, overrides)
    if seed is not None:
        config["seed"] = int(seed)
    for key in ("guideline_path", "dataset_root"):
        if key in config and not Path(config[key]).is_absolute():
            config[key] = str((ROOT / config[key]).resolve())
    return config


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_mimic(config):
    return config["dataset"].startswith("mimic")


def make_mimic_dataset(config, split):
    root = Path(config["dataset_root"])
    return GCPSCPhenotypingDataset(
        ehr_pickle=root / config[f"{split}_ehr_pkl"],
        listfile=root / config[f"{split}_listfile"],
        paired_metadata=root / config["paired_metadata"],
        image_root=config["image_root"],
        split=split,
        image_size=int(config["image_res"]),
        sample_mode=config[f"{split}_sample_mode"],
        augment=bool(split == "train" and config["augment"]),
        strict_images=bool(config["strict_images"]),
        verify_labels=bool(config["verify_labels"]),
    )


def make_skincap_dataset(config, tokenizer, split, missing_rate, expected_label_names=None):
    return SkinCAPMissingDataset(
        csv_path=Path(config["dataset_root"]) / config[f"{split}_csv"],
        image_root=config["image_root"],
        tokenizer=tokenizer,
        split=split,
        missing_rate=float(missing_rate),
        missing_seed=int(config["missing_mask_seed"]),
        image_size=int(config["image_res"]),
        caption_max_length=int(config["caption_max_length"]),
        caption_column=str(config["caption_column"]),
        augment=bool(split == "train" and config["augment"]),
        expected_label_names=expected_label_names,
    )


def make_loader(dataset, config, split):
    workers = int(config.get("num_workers", 0))
    if os.name == "nt" and workers > 0:
        print("Warning: num_workers=0 is recommended on Windows.")
    return DataLoader(
        dataset,
        batch_size=int(config["batch_size"] if split == "train" else config["test_batch_size"]),
        shuffle=split == "train",
        drop_last=split == "train",
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )


def summarize_dataset(dataset, name):
    summary = {"split": name, "samples": len(dataset), "image_present": dataset.paired_count,
               "image_missing": len(dataset) - dataset.paired_count, "labels": len(dataset.label_names)}
    if hasattr(dataset, "actual_missing_rate"):
        summary["actual_missing_rate"] = dataset.actual_missing_rate
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def save_mask_manifest(dataset, path):
    """Record the deterministic SkinCAP image-availability mask."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["skincap_file_path", "image_available", "requested_missing_rate", "missing_mask_seed"])
        for relative_path, available in zip(dataset.relative_paths, dataset.image_available.tolist()):
            writer.writerow([relative_path, int(bool(available)), dataset.missing_rate, int(dataset.missing_seed)])


def rate_name(rate):
    percent = int(round(float(rate) * 100))
    return f"s{percent}"


def build_model(config, bert_name, local_files_only, device):
    embed_dim, num_classes = int(config["embed_dim"]), int(config["num_classes"])
    image_encoder = ImageEncoder(embed_dim=embed_dim, pretrained=bool(config["image_pretrained"]))
    if is_mimic(config):
        clinical_encoder = StructuredEHREncoder(
            input_dim=int(config["ehr_input_dim"]),
            embed_dim=embed_dim,
            hidden_dim=int(config["ehr_hidden_dim"]),
            num_classes=num_classes,
            sequence_length=int(config["ehr_length"]),
            num_layers=int(config["ehr_num_layers"]),
            num_heads=int(config["ehr_num_heads"]),
            feedforward_dim=int(config["ehr_feedforward_dim"]),
            dropout=float(config["ehr_dropout"]),
            input_clip=float(config["ehr_input_clip"]),
        )
    else:
        clinical_encoder = ClinicalTextSequenceEncoder(
            model_name=bert_name,
            num_classes=num_classes,
            embed_dim=embed_dim,
            dropout=float(config["clinical_text_dropout"]),
            freeze_backbone=True,
            local_files_only=local_files_only,
        )
    model = GCPSCModel(
        image_encoder=image_encoder,
        ehr_encoder=clinical_encoder,
        label_decoder=KnowledgeQueryDecoder(embed_dim=embed_dim),
        guideline_decoder=KnowledgeQueryDecoder(embed_dim=embed_dim),
        embed_dim=embed_dim,
        num_classes=num_classes,
        completion_hidden_dim=int(config["completion_hidden_dim"]),
        completion_logvar_min=float(config["completion_logvar_min"]),
        completion_logvar_max=float(config["completion_logvar_max"]),
        router_context_dim=int(config["router_context_dim"]),
        router_hidden_dim=int(config["router_hidden_dim"]),
        router_dropout=float(config["router_dropout"]),
        router_uncertainty_penalty=float(config["router_uncertainty_penalty"]),
        router_temperature=float(config["router_temperature"]),
        uncertainty_hidden_dim=int(config["uncertainty_hidden_dim"]),
        ehr_linear_logit_weight=float(config["ehr_linear_logit_weight"]),
        reliability_temperature=float(config["completion_reliability_temperature"]),
        reliability_uncertainty_scale=float(config["completion_reliability_uncertainty_scale"]),
        reliability_strength=float(config["completion_reliability_strength"]),
        reliability_fusion_power=float(config["completion_reliability_fusion_power"]),
        reliability_floor=float(config["completion_reliability_floor"]),
        reliability_eps=float(config["completion_reliability_eps"]),
        completion_guideline_conditioning=bool(config.get("completion_guideline_conditioning", True)),
        router_mode=str(config.get("router_mode", "learned")),
    )
    return model.to(device)


class ModelEMA:

    def __init__(self, model, decay):
        self.module = copy.deepcopy(model).eval()
        self.module.requires_grad_(False)
        self.decay = float(decay)
        self.num_updates = 0

    @torch.no_grad()
    def copy_from(self, model):
        self.module.load_state_dict(model.state_dict())
        self.module.eval()
        self.num_updates = 0

    @torch.no_grad()
    def update(self, model):
        online = model.state_dict()
        for name, value in self.module.state_dict().items():
            if value.is_floating_point():
                value.mul_(self.decay).add_(online[name].detach().to(dtype=value.dtype), alpha=1.0 - self.decay)
            else:
                value.copy_(online[name])
        self.num_updates += 1

    def state_dict(self):
        return {"model": self.module.state_dict(), "decay": self.decay, "num_updates": self.num_updates}

    def load_state_dict(self, state):
        self.module.load_state_dict(strip_legacy_keys(state["model"]))
        self.module.eval()
        self.num_updates = int(state.get("num_updates", 0))


def strip_legacy_keys(state_dict):
    return {k: v for k, v in state_dict.items() if not k.startswith(LEGACY_PREFIXES)}


def check_objective(checkpoint, config, path):
    old = checkpoint.get("config") or {}
    different = [k for k in OBJECTIVE_KEYS if k in old and float(old[k]) != float(config[k])]
    different += [k for k, default in ABLATION_DEFAULTS.items() if old.get(k, default) != config.get(k, default)]
    if different:
        raise RuntimeError(
            f"{path} was trained with different settings for {different}. Use its own config or a fresh output folder."
        )


def save_checkpoint(path, model, text_encoder, optimizer, scheduler, epoch, best_score, bad_epochs, config, ema, weights_kind):
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "model": model.state_dict(),
            "text_encoder": text_encoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": int(epoch),
            "best_score": float(best_score),
            "early_stopping_bad_epochs": int(bad_epochs),
            "ema_model": ema.state_dict() if ema is not None else None,
            "evaluation_weights": weights_kind,
            "selection_metric": "validation_macro_auprc",
            "config": config,
        },
        path,
    )


def load_evaluation_weights(path, model, text_encoder, ema, config, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    check_objective(checkpoint, config, path)
    model.load_state_dict(strip_legacy_keys(checkpoint["model"]))
    text_encoder.load_state_dict(checkpoint["text_encoder"])
    kind = checkpoint.get("evaluation_weights", "online")
    if kind == "ema":
        ema.load_state_dict(checkpoint["ema_model"])
        model.load_state_dict(ema.module.state_dict())
    print(f"Loaded {path} (epoch {int(checkpoint['epoch']) + 1}, {kind} weights)", flush=True)
    return int(checkpoint["epoch"]), kind


def run(args):
    overrides = parse_overrides(getattr(args, "set", None))
    config = load_config(args.dataset, args.config, args.seed, overrides)
    if overrides:
        print(f"Config overrides: {json.dumps(overrides)}", flush=True)
    if args.bootstrap:
        config["bootstrap"]["enabled"] = True
    seed_everything(int(config["seed"]))
    output_dir = Path(args.output_dir or ROOT / "outputs" / f"{args.dataset}_seed{config['seed']}").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "effective_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)

    bert_name = args.bert_model_name or config["bert_model_name"]
    local_files_only = bool(config["bert_local_files_only"])
    if Path(bert_name).is_absolute() and not Path(bert_name).is_dir() and args.mode != "data-check":
        raise FileNotFoundError(f"ClinicalBERT directory not found: {bert_name}")
    tokenizer = None
    if not is_mimic(config) or args.mode != "data-check":
        tokenizer = AutoTokenizer.from_pretrained(bert_name, do_lower_case=True, local_files_only=local_files_only)

    if is_mimic(config):
        train_dataset, val_dataset = make_mimic_dataset(config, "train"), make_mimic_dataset(config, "val")
    else:
        train_dataset = make_skincap_dataset(config, tokenizer, "train", config["train_missing_rate"])
        val_dataset = make_skincap_dataset(config, tokenizer, "val", config["val_missing_rate"],
                                           expected_label_names=train_dataset.label_names)
        save_mask_manifest(train_dataset, output_dir / "mask_train.csv")
        save_mask_manifest(val_dataset, output_dir / "mask_val.csv")
    if train_dataset.label_names != val_dataset.label_names:
        raise ValueError("Train and validation label orders differ")
    train_summary = summarize_dataset(train_dataset, "train")
    val_summary = summarize_dataset(val_dataset, "val")
    if args.mode == "data-check":
        for test_dataset, name in iter_test_sets(config, tokenizer, train_dataset.label_names):
            summarize_dataset(test_dataset, name)
        sample = train_dataset[0]
        print(json.dumps({"clinical_input_shape": list(sample["ehr"].shape), "image_shape": list(sample["image"].shape),
                          "label_shape": list(sample["label"].shape)}), flush=True)
        return

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    train_loader, val_loader = make_loader(train_dataset, config, "train"), make_loader(val_dataset, config, "val")
    text_encoder = KnowledgeTextEncoder(bert_name, embed_dim=int(config["embed_dim"]),
                                        local_files_only=local_files_only).to(device)
    text_encoder.requires_grad_(False)
    model = build_model(config, bert_name, local_files_only, device)
    guidelines, radiology = load_guidelines(config["guideline_path"], train_dataset.label_names)
    text_bundle = {"labels": train_dataset.label_names, "guidelines": guidelines, "radiology": radiology}

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=float(config["optimizer"]["lr"]),
                                  weight_decay=float(config["optimizer"]["weight_decay"]))
    epochs = int(config["schedular"]["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs,
                                                           eta_min=float(config["schedular"]["min_lr"]))
    scaler = torch.amp.GradScaler("cuda", enabled=bool(config["amp"] and device.type == "cuda"))
    ema = ModelEMA(model, decay=float(config["ema_decay"]))
    ema_start_epoch = int(config["ema_start_epoch"])
    joint_start_epoch = int(config["classification_warmup_epochs"]) + int(config["guidance_warmup_epochs"])
    select_start_epoch = joint_start_epoch if bool(config["select_from_joint_stage"]) else 0
    patience = int(config["early_stopping_patience"])
    writer = SummaryWriter(str(output_dir)) if (SummaryWriter is not None and args.mode == "train") else None

    if args.mode == "train":
        start_epoch, best_score, bad_epochs = 0, -float("inf"), 0
        resume_path = Path(args.resume) if args.resume else output_dir / "checkpoint_last.pt"
        if resume_path.is_file():
            checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
            check_objective(checkpoint, config, resume_path)
            model.load_state_dict(strip_legacy_keys(checkpoint["model"]))
            text_encoder.load_state_dict(checkpoint["text_encoder"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            ema.load_state_dict(checkpoint["ema_model"])
            start_epoch = int(checkpoint["epoch"]) + 1
            best_score = float(checkpoint["best_score"])
            bad_epochs = int(checkpoint["early_stopping_bad_epochs"])
            print(f"Resumed from {resume_path} at epoch {start_epoch + 1}; best={best_score:.6f}", flush=True)
        elif args.resume:
            raise FileNotFoundError(resume_path)

        for epoch in range(start_epoch, epochs):
            if bad_epochs >= patience:
                print("Early stopping already reached; skipping further training.", flush=True)
                break
            if epoch == ema_start_epoch and ema.num_updates == 0:
                ema.copy_from(model)
            train_metrics = train_one_epoch(
                model, text_encoder, tokenizer, train_loader, optimizer, device, epoch, config, text_bundle,
                scaler=scaler, writer=writer, ema=ema if epoch >= ema_start_epoch else None,
            )
            use_ema = epoch >= ema_start_epoch and ema.num_updates > 0
            weights_kind = "ema" if use_ema else "online"
            val_metrics = evaluate(ema.module if use_ema else model, text_encoder, tokenizer, val_loader, device,
                                   config, text_bundle, output_dir / f"val_epoch_{epoch}")
            score = float(val_metrics["all"]["macro_auprc"])
            scheduler.step()
            if epoch >= select_start_epoch and score > best_score:
                best_score, bad_epochs = score, 0
                save_checkpoint(output_dir / "checkpoint_best.pt", model, text_encoder, optimizer, scheduler, epoch,
                                best_score, bad_epochs, config, ema, weights_kind)
            elif epoch >= joint_start_epoch:
                bad_epochs += 1
            else:
                bad_epochs = 0  # objective still changing in stages 1-2
            save_checkpoint(output_dir / "checkpoint_last.pt", model, text_encoder, optimizer, scheduler, epoch,
                            best_score, bad_epochs, config, ema, weights_kind)
            print(json.dumps({"epoch": epoch + 1, "train": train_metrics, "val_macro_auprc": score,
                              "val_macro_auroc": val_metrics["all"]["macro_auroc"], "best_val_macro_auprc": best_score,
                              "bad_epochs": bad_epochs, "patience": patience, "weights": weights_kind}), flush=True)
            if bad_epochs >= patience:
                print(f"Early stopping at epoch {epoch + 1}; best validation Macro AUPRC={best_score:.6f}", flush=True)
                break

    best_path = Path(args.checkpoint) if args.checkpoint else output_dir / "checkpoint_best.pt"
    if not best_path.is_file():
        raise FileNotFoundError(f"No best checkpoint at {best_path}")
    best_epoch, weights_kind = load_evaluation_weights(best_path, model, text_encoder, ema, config, device)
    metadata = {"dataset": config["dataset"], "checkpoint": str(best_path), "best_epoch": best_epoch + 1,
                "evaluation_weights": weights_kind, "train_samples": train_summary["samples"],
                "val_samples": val_summary["samples"]}
    evaluate(model, text_encoder, tokenizer, val_loader, device, config, text_bundle,
             output_dir / "best_validation", experiment_metadata=metadata)
    if args.skip_test:
        return
    del train_loader, val_loader, train_dataset, val_dataset
    gc.collect()
    results = {}
    for test_dataset, name in iter_test_sets(config, tokenizer, text_bundle["labels"]):
        summary = summarize_dataset(test_dataset, name)
        if not is_mimic(config):
            save_mask_manifest(test_dataset, output_dir / f"mask_{name}.csv")
        results[name] = evaluate(
            model, text_encoder, tokenizer, make_loader(test_dataset, config, "test"), device, config, text_bundle,
            output_dir / name, compute_bootstrap=bool(config["bootstrap"]["enabled"]),
            experiment_metadata={**metadata, "test_samples": summary["samples"],
                                 "test_image_missing": summary["image_missing"]},
        )
        headline = {k: results[name][k]["macro_auprc"] for k in results[name] if isinstance(results[name][k], dict)
                    and "macro_auprc" in results[name][k]}
        print(json.dumps({name: {"macro_auprc": headline}}), flush=True)
    if len(results) > 1:
        with (output_dir / "test_all_missing_rates.json").open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2, allow_nan=True)
    if writer is not None:
        writer.close()


def iter_test_sets(config, tokenizer, label_names):
    if is_mimic(config):
        yield make_mimic_dataset(config, "test"), "test"
    else:
        for rate in config["test_missing_rates"]:
            yield make_skincap_dataset(config, tokenizer, "test", rate, expected_label_names=label_names), f"test_{rate_name(rate)}"


def run_seeds(args, script):
    """Sequential fresh training runs, one process per seed, into <output>/seed_<N>."""
    seeds = list(args.seeds)
    if len(set(seeds)) != len(seeds):
        raise ValueError("--seeds must be distinct")
    if args.mode != "train" or args.seed is not None or args.resume or args.checkpoint:
        raise ValueError("--seeds supports fresh training only (no --seed, --resume or --checkpoint)")
    root = Path(args.output_dir or ROOT / "outputs" / f"{args.dataset}_multiseed").resolve()
    destinations = [root / f"seed_{seed}" for seed in seeds]
    for destination in destinations:
        if destination.exists() and any(destination.iterdir()):
            raise ValueError(f"Fresh training requires an empty folder: {destination}")
    root.mkdir(parents=True, exist_ok=True)
    status_path = root / "multi_seed_status.json"
    runs = []
    for seed, destination in zip(seeds, destinations):
        command = [sys.executable, str(script), "--dataset", args.dataset, "--mode", "train", "--seed", str(seed),
                   "--output-dir", str(destination), "--device", args.device]
        for flag, value in (("--config", args.config), ("--bert-model-name", args.bert_model_name)):
            if value:
                command += [flag, str(value)]
        for item in getattr(args, "set", None) or ():
            command += ["--set", str(item)]
        command += ["--skip-test"] if args.skip_test else []
        command += ["--bootstrap"] if args.bootstrap else []
        runs.append({"seed": seed, "output_dir": str(destination), "command": command, "status": "pending"})
    for index, run_info in enumerate(runs, 1):
        print(f"[{index}/{len(runs)}] seed={run_info['seed']} -> {run_info['output_dir']}", flush=True)
        run_info["status"] = "running"
        status_path.write_text(json.dumps(runs, indent=2), encoding="utf-8")
        returncode = subprocess.run(run_info["command"], cwd=ROOT).returncode
        run_info["status"] = "completed" if returncode == 0 else f"failed ({returncode})"
        status_path.write_text(json.dumps(runs, indent=2), encoding="utf-8")
        if returncode:
            raise RuntimeError(f"Seed {run_info['seed']} failed; remaining seeds were not started")



from __future__ import annotations

import json
import time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm.auto import tqdm

from gcpsc.bootstrap import bootstrap_confidence_intervals


LOSS_WEIGHT_NAMES = (
    "guideline_classification_weight",
    "expert_classification_weight",
    "uncertainty_classification_weight",
    "completion_nll_weight",
    "masked_classification_weight",
)

def training_stage_weights(config, epoch):

    weights = {name: float(config[name]) for name in LOSS_WEIGHT_NAMES}
    classification_epochs = int(config["classification_warmup_epochs"])
    guidance_epochs = int(config["guidance_warmup_epochs"])
    if epoch < classification_epochs:
        return "classification", {n: (w if n == "expert_classification_weight" else 0.0) for n, w in weights.items()}
    if epoch < classification_epochs + guidance_epochs:
        delayed = {"completion_nll_weight", "masked_classification_weight"}
        return "guidance_router", {n: (0.0 if n in delayed else w) for n, w in weights.items()}
    return "joint", weights


def expert_losses(output, labels, include_guideline_experts):
    expert_logits = output["label_expert_logits"]
    expanded = labels.unsqueeze(-1).expand_as(expert_logits)
    flat_mean = lambda values: values.flatten().contiguous().mean()
    loss_expert = flat_mean(F.binary_cross_entropy_with_logits(expert_logits, expanded, reduction="none"))
    if include_guideline_experts:
        loss_guideline_expert = flat_mean(F.binary_cross_entropy_with_logits(
            output["guideline_expert_logits"], expanded, reduction="none"
        ))
        loss_expert = 0.5 * (loss_expert + loss_guideline_expert)
    logvars = output["expert_logvars"]
    elementwise = F.binary_cross_entropy_with_logits(expert_logits, expanded, reduction="none")
    loss_uncertainty = flat_mean(torch.exp(-logvars) * elementwise + 0.5 * logvars)
    return loss_expert, loss_uncertainty


def completion_nll(output, image_available):
    paired = image_available.bool()
    if not paired.any():
        return output["label_logits"].sum() * 0.0
    target = output["observed_cxr_evidence"][paired].detach().float()
    mean = output["generated_cxr_mean"][paired].float()
    logvar = output["generated_cxr_logvar"][paired].float()
    nll = 0.5 * (torch.exp(-logvar) * (target - mean).square() + logvar)
    return nll.flatten(start_dim=1).mean(dim=1).mean()


# --------------------------------------------------------------------------- knowledge features
def encode_texts(text_encoder, tokenizer, texts, device, max_length):
    encoded = tokenizer(list(texts), padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    return text_encoder({key: value.to(device) for key, value in encoded.items()})


def knowledge_features(text_encoder, tokenizer, text_bundle, device, config):
    if "_cached_features" in text_bundle:
        return text_bundle["_cached_features"]
    label_pool, _ = encode_texts(text_encoder, tokenizer, text_bundle["labels"], device, config["label_max_length"])
    guideline_pool, _ = encode_texts(text_encoder, tokenizer, text_bundle["guidelines"], device, config["guideline_max_length"])
    radiology_pool, _ = encode_texts(text_encoder, tokenizer, text_bundle["radiology"], device, config["radiology_max_length"])
    result = {
        "label_tokens": label_pool.unsqueeze(1),
        "guideline_tokens": guideline_pool.unsqueeze(1),
        "radiology_pool": radiology_pool,
    }
    result = {key: value.detach().float() for key, value in result.items()}
    text_bundle["_cached_features"] = result
    return result


def _move_batch(batch, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def _forward(model, batch, queries, image_available=None, rows=None):
    select = (lambda x: x) if rows is None else (lambda x: x[rows])
    return model(
        select(batch["image"]),
        select(batch["image_available"]) if image_available is None else image_available,
        select(batch["ehr"]),
        select(batch["ehr_mask"]),
        queries["label_tokens"],
        queries["guideline_tokens"],
        queries["radiology_pool"],
    )


def train_one_epoch(model, text_encoder, tokenizer, loader, optimizer, device, epoch, config, text_bundle,
                    scaler=None, writer=None, ema=None):
    model.train()
    text_encoder.eval()
    optimizer.zero_grad(set_to_none=True)
    stage_name, weights = training_stage_weights(config, epoch)
    use_amp = bool(config.get("amp", True) and device.type == "cuda")
    amp_context = (lambda: torch.amp.autocast("cuda")) if use_amp else nullcontext
    use_tqdm = bool(config.get("use_tqdm", True))
    totals = {}
    progress = tqdm(loader, total=len(loader), dynamic_ncols=True, mininterval=0.5, leave=True, disable=not use_tqdm,
                    desc=f"Epoch {epoch + 1}/{config['schedular']['epochs']} train [{stage_name}]")
    epoch_start = time.perf_counter()

    for step, raw_batch in enumerate(progress):
        batch = _move_batch(raw_batch, device)
        labels = batch["label"].float()
        with amp_context():
            queries = knowledge_features(text_encoder, tokenizer, text_bundle, device, config)
            output = _forward(model, batch, queries)
            zero = output["label_logits"].sum() * 0.0
            loss_label = F.binary_cross_entropy_with_logits(output["label_logits"], labels)
            loss_guideline = zero
            if weights["guideline_classification_weight"] > 0:
                loss_guideline = F.binary_cross_entropy_with_logits(output["guideline_logits"], labels)
            loss_expert = loss_uncertainty = zero
            if weights["expert_classification_weight"] > 0 or weights["uncertainty_classification_weight"] > 0:
                loss_expert, loss_uncertainty = expert_losses(
                    output, labels, include_guideline_experts=weights["guideline_classification_weight"] > 0
                )
            loss_nll = zero
            if weights["completion_nll_weight"] > 0:
                loss_nll = completion_nll(output, batch["image_available"])
            loss_masked = zero
            paired = batch["image_available"].bool()
            if weights["masked_classification_weight"] > 0 and paired.any():
                masked_output = _forward(
                    model, batch, queries, image_available=torch.zeros_like(batch["image_available"][paired]), rows=paired
                )
                loss_masked = F.binary_cross_entropy_with_logits(masked_output["label_logits"], labels[paired])

            terms = {
                "guideline_classification_weight": loss_guideline,
                "expert_classification_weight": loss_expert,
                "uncertainty_classification_weight": loss_uncertainty,
                "completion_nll_weight": loss_nll,
                "masked_classification_weight": loss_masked,
            }
            loss = loss_label
            for name, term in terms.items():
                if weights[name] > 0:
                    loss = loss + weights[name] * term

        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at epoch {epoch + 1}, batch {step + 1}, stage={stage_name}")
        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if ema is not None:
            ema.update(model)
        optimizer.zero_grad(set_to_none=True)

        values = {
            "loss": loss, "loss_label": loss_label, "loss_guideline": loss_guideline, "loss_expert": loss_expert,
            "loss_uncertainty": loss_uncertainty, "loss_completion_nll": loss_nll, "loss_masked_classification": loss_masked,
        }
        for name, value in values.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach())
        if writer is not None:
            global_step = epoch * len(loader) + step
            for name, value in values.items():
                writer.add_scalar(f"train/{name}", float(value.detach()), global_step)
        if use_tqdm and (step == 0 or (step + 1) % 10 == 0 or step + 1 == len(loader)):
            progress.set_postfix(loss=f"{float(loss.detach()):.4f}", paired=f"{int(paired.sum())}/{len(labels)}",
                                 lr=f"{optimizer.param_groups[0]['lr']:.2e}")
        elif not use_tqdm and (step == 0 or (step + 1) % int(config.get("log_interval", 50)) == 0):
            per_batch = (time.perf_counter() - epoch_start) / (step + 1)
            eta = timedelta(seconds=int(per_batch * (len(loader) - step - 1)))
            print(f"Epoch {epoch + 1} [{step + 1}/{len(loader)}] loss={float(loss.detach()):.4f} ETA={eta}", flush=True)

    result = {name: value / max(len(loader), 1) for name, value in totals.items()}
    result["stage"] = stage_name
    return result


def _per_label(metric, labels, predictions):
    values = []
    for index in range(labels.shape[1]):
        try:
            values.append(float(metric(labels[:, index], predictions[:, index])))
        except ValueError:
            values.append(float("nan"))
    return values


def compute_metrics(labels, predictions):
    per_label_auc = _per_label(roc_auc_score, labels, predictions)
    per_label_ap = _per_label(average_precision_score, labels, predictions)
    metrics = {
        "macro_auroc": float(np.nanmean(per_label_auc)),
        "macro_auprc": float(np.nanmean(per_label_ap)),
        "per_label_auroc": per_label_auc,
        "per_label_auprc": per_label_ap,
    }
    try:
        metrics["micro_auroc"] = float(roc_auc_score(labels.ravel(), predictions.ravel()))
    except ValueError:
        metrics["micro_auroc"] = float("nan")
    metrics["micro_auprc"] = float(average_precision_score(labels.ravel(), predictions.ravel()))
    return metrics


def subgroup_metrics(labels, predictions, availability, present_name, missing_name):
    result = {"all": compute_metrics(labels, predictions)}
    if availability.any():
        result[present_name] = compute_metrics(labels[availability], predictions[availability])
    if (~availability).any():
        result[missing_name] = compute_metrics(labels[~availability], predictions[~availability])
    return result


def _quantiles(values):
    values = np.asarray(values, dtype=np.float64).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {k: float("nan") for k in ("min", "q05", "q25", "median", "q75", "q95", "max", "mean")}
    q = np.quantile(values, [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
    return {"min": float(q[0]), "q05": float(q[1]), "q25": float(q[2]), "median": float(q[3]),
            "q75": float(q[4]), "q95": float(q[5]), "max": float(q[6]), "mean": float(values.mean())}


def _spearman(left, right):
    left = np.asarray(left, dtype=np.float64).ravel()
    right = np.asarray(right, dtype=np.float64).ravel()
    keep = np.isfinite(left) & np.isfinite(right)
    if keep.sum() < 3:
        return float("nan")
    rank_left = np.argsort(np.argsort(left[keep])).astype(np.float64)
    rank_right = np.argsort(np.argsort(right[keep])).astype(np.float64)
    if rank_left.std() == 0 or rank_right.std() == 0:
        return float("nan")
    return float(np.corrcoef(rank_left, rank_right)[0, 1])


def summarize_router(router_weights, expert_logvars, availability, names, present_name, missing_name):
    def summarize(values, mask=None):
        selected = values if mask is None else values[mask]
        if selected.shape[0] == 0:
            return None
        overall, per_disease = selected.mean(axis=(0, 1)), selected.mean(axis=0)
        return {"overall": {n: float(overall[i]) for i, n in enumerate(names)},
                "per_disease": {n: per_disease[:, i].tolist() for i, n in enumerate(names)}}

    def shape(mask=None):
        selected = router_weights if mask is None else router_weights[mask]
        if selected.shape[0] == 0:
            return None
        safe = np.clip(selected, 1e-12, 1.0)
        entropy = -(safe * np.log(safe)).sum(axis=-1)
        maximum = selected.max(axis=-1)
        return {"normalized_mean_entropy": float(entropy.mean() / np.log(3.0)), "mean_max_weight": float(maximum.mean()),
                "collapse_fraction_max_weight_gt_0_9": float((maximum > 0.9).mean())}

    result = {}
    for suffix, mask in (("all", None), (present_name, availability), (missing_name, ~availability)):
        result[f"weights_{suffix}"] = summarize(router_weights, mask)
        result[f"logvars_{suffix}"] = summarize(expert_logvars, mask)
        result[f"shape_{suffix}"] = shape(mask)
    return result



def guideline_logits_used(config):
    return float(config["guideline_classification_weight"]) > 0


def combine_logits(label_logits, guideline_logits, use_guideline_logits):
    return 0.5 * (label_logits + guideline_logits) if use_guideline_logits else label_logits


@torch.no_grad()
def evaluate(model, text_encoder, tokenizer, loader, device, config, text_bundle, output_path=None,
             compute_bootstrap=False, experiment_metadata=None):

    model.eval()
    text_encoder.eval()
    groups = config.get("availability_group_names", {"present": "paired", "missing": "ehr_only"})
    present_name, missing_name = str(groups["present"]), str(groups["missing"])
    expert_names = tuple(config.get("expert_display_names", ("ehr", "cxr", "fusion")))
    store = {k: [] for k in ("labels", "predictions", "availability", "subject_ids", "expert_predictions",
                             "router_weights", "expert_logvars", "variances", "cosines", "errors", "reliabilities",
                             "adjustments")}
    stay_names = []
    use_guideline_logits = guideline_logits_used(config)
    for raw_batch in tqdm(loader, total=len(loader), desc="Evaluation", dynamic_ncols=True,
                          disable=not bool(config.get("use_tqdm", True))):
        batch = _move_batch(raw_batch, device)
        queries = knowledge_features(text_encoder, tokenizer, text_bundle, device, config)
        output = _forward(model, batch, queries)
        logits = combine_logits(output["label_logits"], output["guideline_logits"], use_guideline_logits)
        store["labels"].append(batch["label"].cpu().numpy())
        store["predictions"].append(torch.sigmoid(logits).float().cpu().numpy())
        store["availability"].append(batch["image_available"].cpu().numpy())
        store["subject_ids"].append(batch["subject_id"].cpu().numpy())
        stay_names.extend(str(s) for s in raw_batch["stay"])
        expert_logits = combine_logits(output["label_expert_logits"], output["guideline_expert_logits"],
                                       use_guideline_logits)
        store["expert_predictions"].append(torch.sigmoid(expert_logits).float().cpu().numpy())
        store["router_weights"].append(output["router_weights"].float().cpu().numpy())
        store["expert_logvars"].append(output["expert_logvars"].float().cpu().numpy())

        mean = output["generated_cxr_mean"].float()
        evidence = output["observed_cxr_evidence"].float()
        paired_mask = batch["image_available"].bool().unsqueeze(-1)
        nan = lambda t: torch.where(paired_mask, t, torch.full_like(t, float("nan")))
        store["variances"].append(torch.exp(output["generated_cxr_logvar"].float()).mean(dim=-1).cpu().numpy())
        store["cosines"].append(nan(F.cosine_similarity(mean, evidence, dim=-1)).cpu().numpy())
        store["errors"].append(nan((mean - evidence).square().mean(dim=-1)).cpu().numpy())
        store["reliabilities"].append(output["completion_reliability"].float().cpu().numpy())
        store["adjustments"].append(output["completion_reliability_logit_adjustment"].float().cpu().numpy())

    arrays = {k: np.concatenate(v) for k, v in store.items()}
    labels, predictions = arrays["labels"], arrays["predictions"]
    availability = arrays["availability"].astype(bool)
    result = subgroup_metrics(labels, predictions, availability, present_name, missing_name)
    result["prediction_logits"] = "0.5*(label+guideline)" if use_guideline_logits else "label"
    result["experts"] = {
        name: subgroup_metrics(labels, arrays["expert_predictions"][:, :, i], availability, present_name, missing_name)
        for i, name in enumerate(expert_names)
    }
    result["router"] = summarize_router(arrays["router_weights"], arrays["expert_logvars"], availability,
                                        expert_names, present_name, missing_name)
    variances, reliabilities = arrays["variances"], arrays["reliabilities"]
    result["completion"] = {
        "mean_variance_all": float(variances.mean()),
        f"mean_variance_{present_name}": float(variances[availability].mean()) if availability.any() else float("nan"),
        f"mean_variance_{missing_name}": float(variances[~availability].mean()) if (~availability).any() else float("nan"),
        f"{present_name}_evidence_cosine": float(np.nanmean(arrays["cosines"][availability])) if availability.any() else float("nan"),
        f"uncertainty_error_spearman_{present_name}": _spearman(variances[availability], arrays["errors"][availability]),
        f"reliability_quantiles_{missing_name}": _quantiles(reliabilities[~availability]),
        "reliability_effective_temperature": float(config["completion_reliability_temperature"])
        * float(config["completion_reliability_uncertainty_scale"]),
    }

    bootstrap_summary = bootstrap_distributions = None
    if compute_bootstrap:
        bootstrap_summary, bootstrap_distributions = bootstrap_confidence_intervals(
            labels, predictions, availability, arrays["subject_ids"].astype(np.int64), config["bootstrap"]
        )
        level = float(bootstrap_summary["settings"]["confidence_level"])
        ci_name = "bootstrap_ci95" if np.isclose(level, 0.95) else "bootstrap_ci"
        for subgroup, summary in bootstrap_summary["subgroups"].items():
            if subgroup in result:
                result[subgroup][ci_name] = summary["metrics"]
        result["bootstrap"] = bootstrap_summary["settings"]
    if experiment_metadata is not None:
        result["experiment"] = dict(experiment_metadata)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path.with_suffix(".npz"),
            labels=labels, predictions=predictions, image_available=availability,
            subject_ids=arrays["subject_ids"].astype(np.int64), stay_names=np.asarray(stay_names, dtype=str),
            expert_predictions=arrays["expert_predictions"], router_weights=arrays["router_weights"],
            expert_logvars=arrays["expert_logvars"], completion_variances=variances,
            completion_cosines=arrays["cosines"], completion_squared_errors=arrays["errors"],
            completion_reliabilities=reliabilities, completion_reliability_logit_adjustments=arrays["adjustments"],
        )
        with output_path.with_suffix(".json").open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, allow_nan=True)
        if bootstrap_summary is not None:
            with output_path.with_name(f"{output_path.name}_bootstrap.json").open("w", encoding="utf-8") as handle:
                json.dump(bootstrap_summary, handle, indent=2)
            np.savez_compressed(output_path.with_name(f"{output_path.name}_bootstrap_scores.npz"), **bootstrap_distributions)
    return result

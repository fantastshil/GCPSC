

from __future__ import annotations

import math

import torch
from torch import nn


class ProbabilisticCXRCompletion(nn.Module):


    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.ehr_projection = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, hidden_dim))
        self.guideline_projection = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, hidden_dim))
        self.shared = nn.Sequential(
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.mean_head = nn.Linear(hidden_dim, embed_dim)
        self.logvar_head = nn.Linear(hidden_dim, embed_dim)
        for head in (self.mean_head, self.logvar_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, ehr_pool, prototypes, logvar_min: float, logvar_max: float):
        hidden = self.ehr_projection(ehr_pool).unsqueeze(1) + self.guideline_projection(prototypes).unsqueeze(0)
        hidden = self.shared(hidden)  # [B, C, H]
        mean_delta = self.mean_head(hidden)
        mean = prototypes.to(dtype=mean_delta.dtype).unsqueeze(0) + mean_delta
        logvar = self.logvar_head(hidden).clamp(min=logvar_min, max=logvar_max)
        return mean, logvar


class DiseaseUncertaintyHead(nn.Module):

    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.context_projection = nn.Linear(embed_dim, hidden_dim)
        self.guideline_projection = nn.Linear(embed_dim, hidden_dim)
        self.output = nn.Sequential(nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, context, guideline_prototypes):
        hidden = self.context_projection(context).unsqueeze(1) + self.guideline_projection(guideline_prototypes).unsqueeze(0)
        return self.output(hidden).squeeze(-1)


class GuidelineConditionedRouter(nn.Module):

    def __init__(self, embed_dim, context_dim, hidden_dim, dropout, uncertainty_penalty, temperature=1.0):
        super().__init__()
        self.uncertainty_penalty = float(uncertainty_penalty)
        self.temperature = float(temperature)
        if self.temperature <= 0.0:
            raise ValueError("router_temperature must be positive")
        self.ehr_projection = nn.Linear(embed_dim, context_dim)
        self.cxr_projection = nn.Linear(embed_dim, context_dim)
        self.fusion_projection = nn.Linear(embed_dim, context_dim)
        self.guideline_projection = nn.Linear(embed_dim, context_dim)
        input_dim = context_dim * 4 + 7
        self.router = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

    def routing_logits(self, ehr_pool, cxr_pool, fusion_pool, guideline_prototypes, expert_logits, expert_logvars, image_available):
        batch_size, class_count = expert_logits.shape[:2]

        def expand(projection, value):
            return projection(value).unsqueeze(1).expand(-1, class_count, -1)

        guideline_context = self.guideline_projection(guideline_prototypes).unsqueeze(0).expand(batch_size, -1, -1)
        availability = image_available.to(dtype=expert_logits.dtype)[:, None, None].expand(-1, class_count, -1)
        router_input = torch.cat(
            [
                expand(self.ehr_projection, ehr_pool),
                expand(self.cxr_projection, cxr_pool),
                expand(self.fusion_projection, fusion_pool),
                guideline_context,
                expert_logits,
                expert_logvars,
                availability,
            ],
            dim=-1,
        )
        return self.router(router_input) - self.uncertainty_penalty * expert_logvars

    def normalize(self, routing_logits):
        return torch.softmax(routing_logits / self.temperature, dim=-1)


class GCPSCModel(nn.Module):
    expert_names = ("ehr", "cxr", "fusion")

    def __init__(
        self,
        image_encoder,
        ehr_encoder,
        label_decoder,
        guideline_decoder,
        embed_dim: int = 768,
        num_classes: int = 25,
        completion_hidden_dim: int = 512,
        completion_logvar_min: float = -6.0,
        completion_logvar_max: float = 2.0,
        router_context_dim: int = 64,
        router_hidden_dim: int = 256,
        router_dropout: float = 0.1,
        router_uncertainty_penalty: float = 0.5,
        router_temperature: float = 1.0,
        uncertainty_hidden_dim: int = 128,
        ehr_linear_logit_weight: float = 0.5,
        reliability_temperature: float = 1.0,
        reliability_uncertainty_scale: float = 1.0,
        reliability_strength: float = 1.0,
        reliability_fusion_power: float = 0.5,
        reliability_floor: float = 0.5,
        reliability_eps: float = 1e-6,
        completion_guideline_conditioning: bool = True,
        router_mode: str = "learned",
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.ehr_encoder = ehr_encoder
        self.label_decoder = label_decoder
        self.guideline_decoder = guideline_decoder
        self.embed_dim = int(embed_dim)
        self.num_classes = int(num_classes)
        self.completion_logvar_min = float(completion_logvar_min)
        self.completion_logvar_max = float(completion_logvar_max)
        self.ehr_linear_logit_weight = float(ehr_linear_logit_weight)
        self.reliability_temperature = float(reliability_temperature)
        self.reliability_uncertainty_scale = float(reliability_uncertainty_scale)
        self.reliability_strength = float(reliability_strength)
        self.reliability_fusion_power = float(reliability_fusion_power)
        self.reliability_floor = float(reliability_floor)
        self.reliability_eps = float(reliability_eps)
        if self.reliability_temperature <= 0 or self.reliability_uncertainty_scale <= 0:
            raise ValueError("reliability temperature and uncertainty scale must be positive")
        if not 0.0 <= self.reliability_floor < 1.0:
            raise ValueError("reliability floor must be in [0, 1)")
        # Ablation switches (defaults = the full GCPSC model).
        self.completion_guideline_conditioning = bool(completion_guideline_conditioning)
        if router_mode not in ("learned", "uniform"):
            raise ValueError("router_mode must be 'learned' or 'uniform'")
        self.router_mode = router_mode


        self.missing_image_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.missing_image_token, std=0.02)
        self.cxr_completion = ProbabilisticCXRCompletion(embed_dim, completion_hidden_dim, router_dropout)
        self.fusion_pool_norm = nn.LayerNorm(embed_dim)
        self.uncertainty_heads = nn.ModuleList(
            [DiseaseUncertaintyHead(embed_dim, uncertainty_hidden_dim, router_dropout) for _ in self.expert_names]
        )
        self.router = GuidelineConditionedRouter(
            embed_dim=embed_dim,
            context_dim=router_context_dim,
            hidden_dim=router_hidden_dim,
            dropout=router_dropout,
            uncertainty_penalty=router_uncertainty_penalty,
            temperature=router_temperature,
        )
        if not self.completion_guideline_conditioning:
            self.disease_embeddings = nn.Parameter(torch.zeros(num_classes, embed_dim))
            nn.init.normal_(self.disease_embeddings, std=0.02)

    @staticmethod
    def disease_evidence(tokens, padding_mask, prototypes):
        prototypes = prototypes.to(dtype=tokens.dtype)
        scores = torch.einsum("bld,cd->bcl", tokens, prototypes) / math.sqrt(tokens.shape[-1])
        scores = scores.masked_fill(padding_mask.unsqueeze(1), float("-inf"))
        return torch.einsum("bcl,bld->bcd", torch.softmax(scores, dim=-1), tokens)

    def _encode_image_and_complete(self, image, image_available, ehr_pool, prototypes):
        batch_size = image.shape[0]
        available = image_available.bool()
        available_indices = torch.nonzero(available, as_tuple=False).flatten()
        missing_indices = torch.nonzero(~available, as_tuple=False).flatten()

        generated_mean, generated_logvar = self.cxr_completion(
            ehr_pool, prototypes, self.completion_logvar_min, self.completion_logvar_max
        )
        if self.training:
            generated_tokens = generated_mean + torch.randn_like(generated_mean) * torch.exp(0.5 * generated_logvar)
        else:
            generated_tokens = generated_mean

        observed_tokens = observed_pool = None
        observed_count = 0
        if available_indices.numel() > 0:
            observed_tokens, observed_pool = self.image_encoder(image[available_indices])
            observed_count = observed_tokens.shape[1]
        generated_count = generated_tokens.shape[1]
        token_count = max(observed_count, generated_count)
        dtype = observed_tokens.dtype if observed_tokens is not None else ehr_pool.dtype

        missing_token = self.missing_image_token.to(dtype=dtype)
        cxr_tokens = missing_token.expand(batch_size, token_count, -1).clone()
        cxr_pool = missing_token[:, 0].expand(batch_size, -1).clone()
        cxr_padding_mask = torch.ones(batch_size, token_count, dtype=torch.bool, device=image.device)
        if observed_tokens is not None:
            cxr_tokens[available_indices, :observed_count] = observed_tokens
            cxr_pool[available_indices] = observed_pool
            cxr_padding_mask[available_indices, :observed_count] = False
        if missing_indices.numel() > 0:
            generated = generated_tokens.to(dtype=dtype)
            cxr_tokens[missing_indices, :generated_count] = generated[missing_indices]
            cxr_pool[missing_indices] = generated[missing_indices].mean(dim=1)
            cxr_padding_mask[missing_indices, :generated_count] = False

        return {
            "cxr_tokens": cxr_tokens,
            "cxr_padding_mask": cxr_padding_mask,
            "cxr_pool": cxr_pool,
            "cxr_evidence": self.disease_evidence(cxr_tokens, cxr_padding_mask, prototypes),
            "generated_cxr_mean": generated_mean,
            "generated_cxr_logvar": generated_logvar,
        }

    def forward(self, image, image_available, ehr, ehr_mask, label_features, guideline_features, radiology_features):
        ehr_tokens, ehr_pool, ehr_padding_mask, ehr_linear_logits = self.ehr_encoder(ehr, ehr_mask)
        guideline_prototypes = guideline_features.mean(dim=1)
        radiology_prototypes = radiology_features.mean(dim=1) if radiology_features.ndim == 3 else radiology_features
        if not self.completion_guideline_conditioning:
            radiology_prototypes = self.disease_embeddings
        encoded = self._encode_image_and_complete(image, image_available, ehr_pool, radiology_prototypes)

        fusion_tokens = torch.cat([encoded["cxr_tokens"], ehr_tokens], dim=1)
        fusion_padding_mask = torch.cat([encoded["cxr_padding_mask"], ehr_padding_mask], dim=1)
        fusion_pool = self.fusion_pool_norm(0.5 * (ehr_pool + encoded["cxr_pool"]))
        expert_inputs = (
            (ehr_tokens, ehr_padding_mask),
            (encoded["cxr_tokens"], encoded["cxr_padding_mask"]),
            (fusion_tokens, fusion_padding_mask),
        )

        label_logits_per_expert = [self.label_decoder(label_features, t, fusion_padding_mask=m) for t, m in expert_inputs]
        label_logits_per_expert[0] = label_logits_per_expert[0] + self.ehr_linear_logit_weight * ehr_linear_logits.to(
            dtype=label_logits_per_expert[0].dtype
        )
        label_expert_logits = torch.stack(label_logits_per_expert, dim=-1)  # [B, C, 3]
        guideline_expert_logits = torch.stack(
            [self.guideline_decoder(guideline_features, t, fusion_padding_mask=m) for t, m in expert_inputs], dim=-1
        )

        contexts = (ehr_pool, encoded["cxr_pool"], fusion_pool)
        raw_logvars = [head(context, guideline_prototypes) for head, context in zip(self.uncertainty_heads, contexts)]
        expert_logvars = torch.stack(raw_logvars, dim=-1).clamp(min=self.completion_logvar_min, max=self.completion_logvar_max)

        generated_logvar = encoded["generated_cxr_logvar"]
        completion_uncertainty = torch.exp(generated_logvar.detach().float()).mean(dim=-1).to(dtype=raw_logvars[0].dtype)
        effective_temperature = self.reliability_temperature * self.reliability_uncertainty_scale
        raw_reliability = torch.exp(-completion_uncertainty / effective_temperature)
        missing_reliability = (self.reliability_floor + (1.0 - self.reliability_floor) * raw_reliability).clamp(
            min=max(self.reliability_eps, self.reliability_floor), max=1.0
        )
        available = image_available.bool().unsqueeze(-1)
        log_reliability = torch.where(
            available, torch.zeros_like(missing_reliability), torch.log(missing_reliability)
        ).detach()
        reliability = torch.exp(log_reliability).detach()
        adjustment = torch.stack(
            [
                torch.zeros_like(log_reliability),
                self.reliability_strength * log_reliability,
                self.reliability_fusion_power * self.reliability_strength * log_reliability,
            ],
            dim=-1,
        ).detach()

        router_logits = self.router.routing_logits(
            ehr_pool, encoded["cxr_pool"], fusion_pool, guideline_prototypes, label_expert_logits, expert_logvars, image_available
        ) + adjustment
        router_weights = self.router.normalize(router_logits)
        if self.router_mode == "uniform":
            # Ablation "uniform routing": every expert gets weight 1/3 for every disease.
            router_weights = torch.full_like(router_weights, 1.0 / router_weights.shape[-1])

        return {
            "label_logits": (router_weights * label_expert_logits).sum(dim=-1),
            "guideline_logits": (router_weights * guideline_expert_logits).sum(dim=-1),
            "label_expert_logits": label_expert_logits,
            "guideline_expert_logits": guideline_expert_logits,
            "expert_logvars": expert_logvars,
            "router_weights": router_weights,
            "router_logits": router_logits,
            "generated_cxr_mean": encoded["generated_cxr_mean"],
            "generated_cxr_logvar": generated_logvar,
            "observed_cxr_evidence": encoded["cxr_evidence"],
            "completion_uncertainty": completion_uncertainty,
            "completion_reliability": reliability,
            "completion_reliability_logit_adjustment": adjustment,
            "ehr_linear_logits": ehr_linear_logits,
            "image_pool": encoded["cxr_pool"],
            "ehr_pool": ehr_pool,
        }

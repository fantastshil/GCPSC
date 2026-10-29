from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import torch
from einops import rearrange
from torch import nn
import torchvision.models as tv_models
from transformers import AutoConfig, AutoModel


class KnowledgeTextEncoder(nn.Module):


    def __init__(self, bert_model_name: str, embed_dim: int = 768, local_files_only: bool = True):
        super().__init__()
        config = AutoConfig.from_pretrained(
            bert_model_name, output_hidden_states=True, local_files_only=local_files_only
        )
        self.bert_model = AutoModel.from_pretrained(
            bert_model_name, config=config, local_files_only=local_files_only
        )
        self.mlp_embed = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim)
        )
        self.embed_dim = embed_dim
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        nn.init.constant_(self.logit_scale, np.log(1 / 0.07))
        for module in self.mlp_embed:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=self.embed_dim ** -0.5)

    def forward(self, text):
        output = self.bert_model(input_ids=text["input_ids"], attention_mask=text["attention_mask"])
        last_hidden_state = output.last_hidden_state
        return self.mlp_embed(last_hidden_state[:, 0, :]), last_hidden_state


class ImageEncoder(nn.Module):


    def __init__(self, embed_dim: int, pretrained: bool = True):
        super().__init__()
        weights = tv_models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        # ``resnet`` stays registered (incl. its unused fc) for key compatibility.
        self.resnet = tv_models.resnet50(weights=weights)
        num_ftrs = int(self.resnet.fc.in_features)
        self.res_features = nn.Sequential(*list(self.resnet.children())[:-2])
        self.res_l1 = nn.Linear(num_ftrs, num_ftrs)
        self.res_l2 = nn.Linear(num_ftrs, embed_dim)

    def forward(self, img):
        batch_size = img.shape[0]
        features = rearrange(self.res_features(img), "b d n1 n2 -> b (n1 n2) d")
        hidden = rearrange(features, "b n d -> (b n) d")
        hidden = self.res_l2(torch.relu(self.res_l1(hidden)))
        tokens = rearrange(hidden, "(b n) d -> b n d", b=batch_size)
        return tokens, tokens.mean(dim=1)


class StructuredEHREncoder(nn.Module):


    def __init__(
        self,
        input_dim: int = 76,
        embed_dim: int = 768,
        hidden_dim: int = 256,
        num_classes: int = 25,
        sequence_length: int = 48,
        num_layers: int = 1,
        num_heads: int = 4,
        feedforward_dim: int = 1024,
        dropout: float = 0.3,
        input_clip: float = 10.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.input_clip = float(input_clip)
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.position_embedding = nn.Parameter(torch.empty(1, sequence_length, hidden_dim))
        self.input_dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output_projection = nn.Identity() if hidden_dim == embed_dim else nn.Linear(hidden_dim, embed_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        nn.init.uniform_(self.position_embedding, -0.1, 0.1)

    def forward(self, ehr: torch.Tensor, valid_mask: torch.Tensor):
        if ehr.ndim != 3 or ehr.shape[-2:] != (self.sequence_length, self.input_dim):
            raise ValueError(f"Expected EHR [B, {self.sequence_length}, {self.input_dim}], got {tuple(ehr.shape)}")
        # Clamp rare extreme outliers so FP16 projection cannot overflow.
        ehr = torch.nan_to_num(ehr, nan=0.0, posinf=self.input_clip, neginf=-self.input_clip)
        ehr = ehr.clamp(min=-self.input_clip, max=self.input_clip)
        padding_mask = ~valid_mask.bool()
        hidden = self.input_dropout(self.input_projection(ehr) + self.position_embedding[:, : ehr.shape[1]])
        hidden = self.encoder(hidden, src_key_padding_mask=padding_mask)
        valid = valid_mask.to(dtype=hidden.dtype).unsqueeze(-1)
        pooled = (hidden * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        logits = self.classifier(pooled)
        return self.output_projection(hidden), self.output_projection(pooled), padding_mask, logits


class ClinicalTextSequenceEncoder(nn.Module):


    def __init__(
        self,
        model_name: str,
        num_classes: int,
        embed_dim: int = 768,
        dropout: float = 0.1,
        freeze_backbone: bool = True,
        local_files_only: bool = True,
    ) -> None:
        super().__init__()
        config = AutoConfig.from_pretrained(model_name, local_files_only=local_files_only)
        if int(config.hidden_size) != int(embed_dim):
            raise ValueError(f"Text hidden size {config.hidden_size} != embed_dim {embed_dim}")
        self.backbone = AutoModel.from_pretrained(model_name, config=config, local_files_only=local_files_only)
        self.freeze_backbone = bool(freeze_backbone)
        if self.freeze_backbone:
            self.backbone.requires_grad_(False)
        self.token_norm = nn.LayerNorm(embed_dim)
        self.pool_projection = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )
        self.classifier = nn.Linear(embed_dim, int(num_classes))

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, input_ids, attention_mask):
        if self.freeze_backbone:
            self.backbone.eval()
        context = torch.no_grad() if self.freeze_backbone else nullcontext()
        with context:
            output = self.backbone(input_ids=input_ids.long(), attention_mask=attention_mask.long())
        tokens = self.token_norm(output.last_hidden_state)
        padding_mask = ~attention_mask.bool()
        valid = attention_mask.to(dtype=tokens.dtype).unsqueeze(-1)
        mean_pool = (tokens * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        pool = mean_pool + self.pool_projection(mean_pool)
        return tokens, pool, padding_mask, self.classifier(pool)

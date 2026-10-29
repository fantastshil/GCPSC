from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class DecoderLayer(nn.Module):


    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, queries, memory, memory_key_padding_mask=None):
        normed = self.norm1(queries)
        attended, _ = self.self_attn(normed, normed, value=normed)
        queries = queries + self.dropout1(attended)
        normed = self.norm2(queries)
        attended, _ = self.multihead_attn(
            query=normed, key=memory, value=memory, key_padding_mask=memory_key_padding_mask
        )
        queries = queries + self.dropout2(attended)
        normed = self.norm3(queries)
        queries = queries + self.dropout3(self.linear2(self.dropout(F.relu(self.linear1(normed)))))
        return queries


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer: nn.Module, num_layers: int, norm: nn.Module):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, queries, memory, memory_key_padding_mask=None):
        for layer in self.layers:
            queries = layer(queries, memory, memory_key_padding_mask)
        return self.norm(queries)


class KnowledgeQueryDecoder(nn.Module):


    def __init__(self, embed_dim: int = 768, class_num: int = 1):
        super().__init__()
        self.d_model = embed_dim
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        layer = DecoderLayer(self.d_model, 4, 1024, 0.1)
        self.decoder_norm = nn.LayerNorm(self.d_model)
        self.decoder = TransformerDecoder(layer, 4, self.decoder_norm)
        self.dropout_feas = nn.Dropout(0.1)
        self.mlp_head = nn.Sequential(nn.Linear(embed_dim, class_num))
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.MultiheadAttention):
            module.in_proj_weight.data.normal_(mean=0.0, std=0.02)
            module.out_proj.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, queries, tokens, fusion_padding_mask=None):
        batch_size = tokens.shape[0]
        queries = queries.repeat(1, batch_size // queries.size(1), 1)  # [C, B, d]
        memory = self.decoder_norm(tokens.transpose(0, 1))  # [L, B, d]
        queries = self.decoder_norm(queries)
        features = self.decoder(queries, memory, memory_key_padding_mask=fusion_padding_mask)
        features = self.dropout_feas(features).transpose(0, 1)  # [B, C, d]
        return self.mlp_head(features).squeeze(-1)  # [B, C]

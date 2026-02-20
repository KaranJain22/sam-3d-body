# Copyright (c) Meta Platforms, Inc. and affiliates.

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PerceptualWeightHead(nn.Module):
    """Predict per-vertex perceptual importance omega(v)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_parts: int = 8,
        part_embed_dim: int = 16,
        mlp_depth: int = 2,
        min_omega: float = 1e-3,
    ):
        super().__init__()
        self.part_embedding = nn.Embedding(num_parts, part_embed_dim)
        self.min_omega = min_omega

        feature_dim = input_dim + part_embed_dim + 4
        layers = []
        curr_dim = feature_dim
        for _ in range(max(1, mlp_depth)):
            layers.append(nn.Linear(curr_dim, hidden_dim))
            layers.append(nn.GELU())
            curr_dim = hidden_dim
        layers.append(nn.Linear(curr_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        vertex_features: torch.Tensor,
        semantic_part_ids: torch.Tensor,
        curvature_scale_features: torch.Tensor,
        differentiable_visibility: torch.Tensor,
        projected_screen_size: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        part_emb = self.part_embedding(semantic_part_ids.long())
        x = torch.cat(
            [
                vertex_features,
                part_emb,
                curvature_scale_features,
                differentiable_visibility,
                projected_screen_size,
            ],
            dim=-1,
        )
        omega_logits = self.mlp(x).squeeze(-1)
        omega = F.softplus(omega_logits) + self.min_omega
        return {"omega": omega, "omega_logits": omega_logits}


def bradley_terry_pairwise_loss(
    score_a: torch.Tensor,
    score_b: torch.Tensor,
    preferred_a: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Logistic pairwise ranking loss (Bradley-Terry)."""
    logits = score_a - score_b
    targets = preferred_a.to(logits.dtype)
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    raise ValueError(f"Unsupported reduction: {reduction}")

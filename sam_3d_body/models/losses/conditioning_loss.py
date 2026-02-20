# Copyright (c) Meta Platforms, Inc. and affiliates.

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class ConditioningLoss(nn.Module):
    """Conditioning-aware vertex reconstruction loss.

    Per-vertex weights follow:
        alpha(v) ∝ 1 / (s(v) + tau)

    Optionally, perceptual weights modulate alpha:
        alpha(v) <- alpha(v) * omega(v)
    """

    def __init__(self, tau: float = 1e-6, eps: float = 1e-12):
        super().__init__()
        self.tau = float(tau)
        self.eps = float(eps)

    def vertex_weights(
        self,
        local_scale: torch.Tensor,
        perceptual_weights: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Build normalized per-vertex weights with shape (B, V)."""
        alpha = 1.0 / (local_scale.clamp_min(0.0) + self.tau)

        if perceptual_weights is not None:
            alpha = alpha * perceptual_weights

        if valid_mask is not None:
            alpha = alpha * valid_mask.to(alpha.dtype)

        normalizer = alpha.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        alpha = alpha / normalizer
        return alpha

    def forward(
        self,
        pred_vertices: torch.Tensor,
        target_vertices: torch.Tensor,
        local_scale: torch.Tensor,
        perceptual_weights: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if pred_vertices.shape != target_vertices.shape:
            raise ValueError(
                f"pred/target shape mismatch: {pred_vertices.shape} vs {target_vertices.shape}"
            )
        if pred_vertices.ndim != 3 or pred_vertices.shape[-1] != 3:
            raise ValueError(
                f"pred_vertices must have shape [B, V, 3], got {pred_vertices.shape}"
            )

        alpha = self.vertex_weights(
            local_scale=local_scale,
            perceptual_weights=perceptual_weights,
            valid_mask=valid_mask,
        )
        vertex_err = torch.linalg.norm(pred_vertices - target_vertices, dim=-1)
        return (alpha * vertex_err).sum(dim=-1).mean()

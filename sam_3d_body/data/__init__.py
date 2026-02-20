# Copyright (c) Meta Platforms, Inc. and affiliates.

from .perceptual_preference_dataset import (
    PairwisePerceptualPreferenceDataset,
    build_pairwise_ranking_targets,
)

__all__ = [
    "PairwisePerceptualPreferenceDataset",
    "build_pairwise_ranking_targets",
]

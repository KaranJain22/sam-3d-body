# Copyright (c) Meta Platforms, Inc. and affiliates.

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset


class PairwisePerceptualPreferenceDataset(Dataset):
    """JSONL loader for pairwise human preference annotations."""

    def __init__(self, annotation_file: str | Path):
        self.annotation_file = Path(annotation_file)
        self.samples: List[Dict] = []
        with self.annotation_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.samples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.samples)

    def _to_tensor_dict(self, candidate: Dict) -> Dict[str, torch.Tensor]:
        out = {
            "semantic_part_ids": torch.as_tensor(candidate["semantic_part_ids"], dtype=torch.long),
            "curvature_scale_features": torch.as_tensor(
                candidate["curvature_scale_features"], dtype=torch.float32
            ),
            "visibility": torch.as_tensor(candidate["visibility"], dtype=torch.float32).unsqueeze(-1),
            "screen_size": torch.as_tensor(candidate["screen_size"], dtype=torch.float32).unsqueeze(-1),
        }
        if "vertex_error" in candidate:
            out["vertex_error"] = torch.as_tensor(candidate["vertex_error"], dtype=torch.float32)
        return out

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        preferred = sample["preferred"]
        preferred_a = torch.tensor(1.0 if preferred == "a" else 0.0, dtype=torch.float32)
        return {
            "sample_id": sample["sample_id"],
            "image_id": sample["image_id"],
            "person_id": sample["person_id"],
            "candidate_a": self._to_tensor_dict(sample["candidate_a"]),
            "candidate_b": self._to_tensor_dict(sample["candidate_b"]),
            "preferred_a": preferred_a,
            "confidence": torch.tensor(sample.get("confidence", 1.0), dtype=torch.float32),
        }


def build_pairwise_ranking_targets(
    omega: torch.Tensor,
    candidate_a_vertex_error: torch.Tensor,
    candidate_b_vertex_error: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Compute pairwise Bradley-Terry scores from human preference samples."""
    score_a = -(omega * candidate_a_vertex_error).sum(dim=-1)
    score_b = -(omega * candidate_b_vertex_error).sum(dim=-1)
    return {"score_a": score_a, "score_b": score_b}

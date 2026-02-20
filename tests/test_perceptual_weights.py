import json

import torch

from sam_3d_body.data.perceptual_preference_dataset import (
    PairwisePerceptualPreferenceDataset,
    build_pairwise_ranking_targets,
)
from sam_3d_body.models.heads.perceptual_weight_head import (
    PerceptualWeightHead,
    bradley_terry_pairwise_loss,
)


def test_perceptual_weight_head_shapes():
    head = PerceptualWeightHead(input_dim=8, hidden_dim=16, num_parts=4)
    b, v = 2, 5
    out = head(
        vertex_features=torch.randn(b, v, 8),
        semantic_part_ids=torch.randint(0, 4, (b, v)),
        curvature_scale_features=torch.randn(b, v, 2),
        differentiable_visibility=torch.rand(b, v, 1),
        projected_screen_size=torch.rand(b, v, 1),
    )
    assert out["omega"].shape == (b, v)
    assert torch.all(out["omega"] > 0)


def test_pairwise_dataset_and_ranking_target(tmp_path):
    sample = {
        "sample_id": "s0",
        "image_id": "img0",
        "person_id": 0,
        "preferred": "a",
        "candidate_a": {
            "mesh_id": "m0",
            "semantic_part_ids": [0, 1],
            "curvature_scale_features": [[0.1, 1.0], [0.2, 1.1]],
            "visibility": [1.0, 0.5],
            "screen_size": [0.2, 0.1],
            "vertex_error": [0.1, 0.4],
        },
        "candidate_b": {
            "mesh_id": "m1",
            "semantic_part_ids": [0, 1],
            "curvature_scale_features": [[0.1, 1.0], [0.2, 1.1]],
            "visibility": [1.0, 0.5],
            "screen_size": [0.2, 0.1],
            "vertex_error": [0.3, 0.7],
        },
    }
    ann = tmp_path / "prefs.jsonl"
    ann.write_text(json.dumps(sample) + "\n", encoding="utf-8")

    ds = PairwisePerceptualPreferenceDataset(ann)
    item = ds[0]
    omega = torch.ones(1, 2)
    targets = build_pairwise_ranking_targets(
        omega,
        item["candidate_a"]["vertex_error"].unsqueeze(0),
        item["candidate_b"]["vertex_error"].unsqueeze(0),
    )
    loss = bradley_terry_pairwise_loss(
        targets["score_a"], targets["score_b"], item["preferred_a"].unsqueeze(0)
    )
    assert loss.ndim == 0

import torch

from sam_3d_body.conditioning.precompute import build_mesh_operators, get_mesh_operators


def test_build_mesh_operators_normalized_fallback(tmp_path):
    faces = torch.tensor([[0, 1, 2], [2, 1, 3]], dtype=torch.long)

    ops = build_mesh_operators(faces, cache_dir=str(tmp_path))

    assert ops["laplacian"]["type"] == "normalized_graph"
    assert ops["vertex_adjacency"]["indices"].shape[0] == 2
    assert ops["mass_matrix"]["diag"].shape[0] == 4


def test_build_mesh_operators_cotangent_and_cache(tmp_path):
    vertices = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    faces = torch.tensor([[0, 1, 2], [2, 1, 3]], dtype=torch.long)

    first = build_mesh_operators(faces, vertices=vertices, cache_dir=str(tmp_path))
    second = build_mesh_operators(faces, cache_dir=str(tmp_path))

    assert first["laplacian"]["type"] == "cotangent"
    assert second["laplacian"]["type"] == "cotangent"


def test_get_mesh_operators_device_transfer(tmp_path):
    faces = torch.tensor([[0, 1, 2]], dtype=torch.long)
    ops = build_mesh_operators(faces, cache_dir=str(tmp_path))

    moved = get_mesh_operators(ops, "cpu")
    assert moved["laplacian"]["values"].device.type == "cpu"

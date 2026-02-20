# Copyright (c) Meta Platforms, Inc. and affiliates.

from __future__ import annotations

from typing import Tuple

import numpy as np


def as_numpy_array(x) -> np.ndarray:
    """Convert tensors/arrays/lists to numpy arrays without forcing torch dependency."""
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x)


def triangle_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)


def vertex_areas(vertices: np.ndarray, faces: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    tri_areas = triangle_areas(vertices, faces)
    areas = np.zeros(len(vertices), dtype=np.float64)
    np.add.at(areas, faces[:, 0], tri_areas / 3.0)
    np.add.at(areas, faces[:, 1], tri_areas / 3.0)
    np.add.at(areas, faces[:, 2], tri_areas / 3.0)
    return np.maximum(areas, eps)


def cotangent_laplacian(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Return symmetric positive semi-definite cotan Laplacian."""
    n_verts = len(vertices)
    lap = np.zeros((n_verts, n_verts), dtype=np.float64)

    i = faces[:, 0]
    j = faces[:, 1]
    k = faces[:, 2]

    vi = vertices[i]
    vj = vertices[j]
    vk = vertices[k]

    e_ij, e_ik = vj - vi, vk - vi
    e_ji, e_jk = vi - vj, vk - vj
    e_ki, e_kj = vi - vk, vj - vk

    cot_i = np.einsum("ij,ij->i", e_ij, e_ik) / np.maximum(
        np.linalg.norm(np.cross(e_ij, e_ik), axis=1), 1e-12
    )
    cot_j = np.einsum("ij,ij->i", e_ji, e_jk) / np.maximum(
        np.linalg.norm(np.cross(e_ji, e_jk), axis=1), 1e-12
    )
    cot_k = np.einsum("ij,ij->i", e_ki, e_kj) / np.maximum(
        np.linalg.norm(np.cross(e_ki, e_kj), axis=1), 1e-12
    )

    w_ij = 0.5 * cot_k
    w_ik = 0.5 * cot_j
    w_jk = 0.5 * cot_i

    # Off-diagonals.
    np.add.at(lap, (i, j), -w_ij)
    np.add.at(lap, (j, i), -w_ij)
    np.add.at(lap, (i, k), -w_ik)
    np.add.at(lap, (k, i), -w_ik)
    np.add.at(lap, (j, k), -w_jk)
    np.add.at(lap, (k, j), -w_jk)

    # Diagonals.
    np.add.at(lap, (i, i), w_ij + w_ik)
    np.add.at(lap, (j, j), w_ij + w_jk)
    np.add.at(lap, (k, k), w_ik + w_jk)

    return lap


def mass_matrix(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    return np.diag(vertex_areas(vertices, faces))


def build_vertex_adjacency(faces: np.ndarray, n_vertices: int) -> Tuple[np.ndarray, np.ndarray]:
    adjacency = np.zeros((n_vertices, n_vertices), dtype=bool)
    for i, j, k in faces:
        adjacency[i, j] = adjacency[j, i] = True
        adjacency[i, k] = adjacency[k, i] = True
        adjacency[j, k] = adjacency[k, j] = True
    np.fill_diagonal(adjacency, True)
    degree = adjacency.sum(axis=1)
    return adjacency, degree

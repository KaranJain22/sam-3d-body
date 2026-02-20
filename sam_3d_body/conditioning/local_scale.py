from __future__ import annotations

import numpy as np

from sam_3d_body.models.modules.mesh_geometry_utils import (
    as_numpy_array,
    build_vertex_adjacency,
    cotangent_laplacian,
    vertex_areas,
)


def mean_curvature_magnitude(vertices, faces, eps: float = 1e-12) -> np.ndarray:
    vertices = as_numpy_array(vertices).astype(np.float64)
    faces = as_numpy_array(faces).astype(np.int64)

    lap = cotangent_laplacian(vertices, faces)
    v_areas = vertex_areas(vertices, faces, eps=eps)
    h_vec = lap @ vertices / (2.0 * v_areas[:, None])
    return np.linalg.norm(h_vec, axis=1)


def medial_axis_distance(vertices, faces) -> np.ndarray:
    """Approximate medial-axis distance via nearest non-1-ring vertex distance."""
    vertices = as_numpy_array(vertices).astype(np.float64)
    faces = as_numpy_array(faces).astype(np.int64)
    n = len(vertices)
    adjacency, _ = build_vertex_adjacency(faces, n)

    dists = np.linalg.norm(vertices[:, None, :] - vertices[None, :, :], axis=2)
    dists[adjacency] = np.inf
    nearest_non_local = np.min(dists, axis=1)

    # Fallback for very small meshes where every vertex is adjacent.
    fallback = np.partition(
        np.linalg.norm(vertices[:, None, :] - vertices[None, :, :], axis=2),
        kth=min(2, max(1, n - 1)),
        axis=1,
    )[:, min(2, max(1, n - 1))]
    nearest_non_local[~np.isfinite(nearest_non_local)] = fallback[
        ~np.isfinite(nearest_non_local)
    ]
    return nearest_non_local


def compute_local_scale(vertices, faces, eps: float = 1e-8) -> np.ndarray:
    h_abs = mean_curvature_magnitude(vertices, faces, eps=eps)
    curvature_scale = 1.0 / (h_abs + eps)
    medial_dist = medial_axis_distance(vertices, faces)
    return np.minimum(curvature_scale, medial_dist)

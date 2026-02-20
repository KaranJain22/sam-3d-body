from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from .local_scale import compute_local_scale
from .robust_stats import robust_log_dynamic_range
from .spectral import spectral_condition_number
from sam_3d_body.models.modules.mesh_geometry_utils import as_numpy_array


def _extract_vertices_faces(mesh: Any) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(mesh, dict):
        vertices = mesh.get("vertices")
        faces = mesh.get("faces")
    else:
        vertices = getattr(mesh, "vertices", None)
        faces = getattr(mesh, "faces", None)

    if vertices is None and isinstance(mesh, (tuple, list)) and len(mesh) == 2:
        vertices, faces = mesh

    if vertices is None or faces is None:
        raise ValueError("mesh must provide vertices and faces")

    return as_numpy_array(vertices), as_numpy_array(faces)


def compute_conditioning(
    mesh,
    visibility: Optional[np.ndarray] = None,
    semantics: Optional[np.ndarray] = None,
    cam: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    vertices, faces = _extract_vertices_faces(mesh)

    s = compute_local_scale(vertices, faces)
    kappa_geom = robust_log_dynamic_range(s)
    kappa_spec = spectral_condition_number(vertices, faces)

    output = {
        "s": s,
        "kappa_geom": float(kappa_geom),
        "kappa_spec": float(kappa_spec),
    }

    perceptual_terms: Dict[str, float] = {}
    if visibility is not None:
        vis = np.asarray(visibility).astype(bool)
        if vis.shape[0] == s.shape[0] and np.any(vis):
            perceptual_terms["kappa_visible"] = robust_log_dynamic_range(s[vis])
    if semantics is not None:
        sem = np.asarray(semantics)
        if sem.shape[0] == s.shape[0]:
            per_class = []
            for label in np.unique(sem):
                mask = sem == label
                if np.any(mask):
                    per_class.append(np.median(s[mask]))
            if per_class:
                per_class = np.asarray(per_class)
                perceptual_terms["semantic_spread"] = float(
                    np.max(per_class) / max(np.min(per_class), 1e-12)
                )
    if cam is not None and isinstance(cam, dict) and "depth" in cam:
        depth = np.asarray(cam["depth"], dtype=np.float64)
        perceptual_terms["depth_scale"] = float(1.0 / np.maximum(depth.mean(), 1e-12))

    if perceptual_terms:
        output["perceptual_terms"] = perceptual_terms

    return output

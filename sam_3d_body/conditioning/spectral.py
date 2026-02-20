from __future__ import annotations

import numpy as np

try:
    from scipy.sparse.linalg import eigsh  # type: ignore
except Exception:  # pragma: no cover
    eigsh = None

from sam_3d_body.models.modules.mesh_geometry_utils import (
    as_numpy_array,
    cotangent_laplacian,
    mass_matrix,
)


def laplace_beltrami_spectrum(vertices, faces, n_eigs: int = 16) -> np.ndarray:
    vertices = as_numpy_array(vertices).astype(np.float64)
    faces = as_numpy_array(faces).astype(np.int64)

    lap = cotangent_laplacian(vertices, faces)
    mass = mass_matrix(vertices, faces)

    mass_diag = np.maximum(np.diag(mass), 1e-12)
    inv_sqrt_mass = 1.0 / np.sqrt(mass_diag)
    normalized = (lap * inv_sqrt_mass[:, None]) * inv_sqrt_mass[None, :]

    n = normalized.shape[0]
    k = min(n_eigs, n)
    if k <= 0:
        return np.empty(0, dtype=np.float64)

    # Fast path for partial spectrum if scipy is available.
    if eigsh is not None and k < n:
        try:
            vals = eigsh(normalized, k=k, which="SM", return_eigenvectors=False)
            eigvals = np.sort(np.maximum(vals, 0.0))
            return eigvals
        except Exception:
            # Fallback to dense eigendecomposition.
            pass

    eigvals = np.linalg.eigvalsh(normalized)
    eigvals = np.sort(np.maximum(eigvals, 0.0))
    return eigvals[:k]


def spectral_condition_number(vertices, faces, low_rank: int = 1, high_rank: int = -1) -> float:
    eigvals = laplace_beltrami_spectrum(vertices, faces)
    nz = eigvals[eigvals > 1e-10]
    if len(nz) == 0:
        return 1.0

    low_idx = min(max(low_rank, 0), len(nz) - 1)
    high_idx = len(nz) + high_rank if high_rank < 0 else min(high_rank, len(nz) - 1)
    high_idx = max(high_idx, low_idx)

    lambda_low = nz[low_idx]
    lambda_high = nz[high_idx]
    return float(lambda_high / max(lambda_low, 1e-12))

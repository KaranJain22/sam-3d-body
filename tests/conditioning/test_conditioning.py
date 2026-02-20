import numpy as np

from sam_3d_body.conditioning.local_scale import compute_local_scale
from sam_3d_body.conditioning.robust_stats import robust_log_dynamic_range
from sam_3d_body.conditioning.spectral import spectral_condition_number


def make_grid(n=8, noise=0.0, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    xs = np.linspace(-1.0, 1.0, n)
    ys = np.linspace(-1.0, 1.0, n)
    xv, yv = np.meshgrid(xs, ys)
    zv = np.zeros_like(xv)
    if noise > 0:
        zv = zv + noise * rng.normal(size=zv.shape)
    vertices = np.stack([xv.ravel(), yv.ravel(), zv.ravel()], axis=1)

    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j
            b = a + 1
            c = a + n
            d = c + 1
            faces.append([a, b, c])
            faces.append([b, d, c])
    return vertices, np.asarray(faces, dtype=np.int64)


def laplacian_smooth(vertices, faces, iterations=8):
    vertices = vertices.copy()
    adjacency = [set() for _ in range(len(vertices))]
    for i, j, k in faces:
        adjacency[i].update([j, k])
        adjacency[j].update([i, k])
        adjacency[k].update([i, j])

    for _ in range(iterations):
        nxt = vertices.copy()
        for i, nbs in enumerate(adjacency):
            if nbs:
                nxt[i] = 0.5 * vertices[i] + 0.5 * np.mean(vertices[list(nbs)], axis=0)
        vertices = nxt
    return vertices


def test_percentile_robustness_to_outliers():
    base = np.ones(100)
    with_outliers = np.concatenate([base, [1e-12, 1e12]])
    kappa = robust_log_dynamic_range(with_outliers)
    assert kappa < 1.5


def test_local_scale_stability_near_zero_curvature():
    vertices, faces = make_grid(n=10, noise=0.0)
    s = compute_local_scale(vertices, faces)
    assert np.all(np.isfinite(s))
    assert np.all(s > 0)


def test_kappa_spec_monotonic_smoothed_vs_noisy():
    n = 20
    xs = np.linspace(-1.0, 1.0, n)
    ys = np.linspace(-1.0, 1.0, n)
    xv, yv = np.meshgrid(xs, ys)

    smooth_vertices = np.stack(
        [xv.ravel(), yv.ravel(), 0.1 * np.sin(np.pi * xv).ravel() * np.sin(np.pi * yv).ravel()],
        axis=1,
    )
    noisy_vertices = np.stack(
        [
            xv.ravel(),
            yv.ravel(),
            0.1 * np.sin(8 * np.pi * xv).ravel() * np.sin(8 * np.pi * yv).ravel(),
        ],
        axis=1,
    )

    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j
            b = a + 1
            c = a + n
            d = c + 1
            faces.append([a, b, c])
            faces.append([b, d, c])
    faces = np.asarray(faces, dtype=np.int64)

    kappa_smooth = spectral_condition_number(smooth_vertices, faces)
    kappa_noisy = spectral_condition_number(noisy_vertices, faces)

    assert kappa_smooth >= kappa_noisy

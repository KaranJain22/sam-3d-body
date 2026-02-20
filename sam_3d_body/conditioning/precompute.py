from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def _faces_to_tensor(faces: Any) -> torch.Tensor:
    faces_t = torch.as_tensor(faces, dtype=torch.long)
    if faces_t.ndim != 2 or faces_t.shape[1] != 3:
        raise ValueError("faces must have shape [F, 3]")
    return faces_t.cpu().contiguous()


def _vertices_to_tensor(vertices: Any) -> torch.Tensor:
    vertices_t = torch.as_tensor(vertices, dtype=torch.float64)
    if vertices_t.ndim != 2 or vertices_t.shape[1] != 3:
        raise ValueError("vertices must have shape [V, 3]")
    return vertices_t.cpu().contiguous()


def _face_cache_key(faces: torch.Tensor) -> str:
    payload = faces.to(torch.int64).numpy().tobytes()
    digest = hashlib.sha256(payload).hexdigest()
    return f"faces-{faces.shape[0]}-{digest[:16]}"


def _build_undirected_edges(faces: torch.Tensor) -> torch.Tensor:
    edges = torch.cat(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]],
        dim=0,
    )
    edges = torch.sort(edges, dim=1).values
    edges = torch.unique(edges, dim=0)
    return edges


def _build_vertex_adjacency(n_vertices: int, edges: torch.Tensor) -> Dict[str, torch.Tensor]:
    row = torch.cat([edges[:, 0], edges[:, 1]], dim=0)
    col = torch.cat([edges[:, 1], edges[:, 0]], dim=0)
    values = torch.ones(row.shape[0], dtype=torch.float32)
    adjacency = torch.sparse_coo_tensor(
        torch.stack([row, col], dim=0), values, (n_vertices, n_vertices)
    ).coalesce()
    degree = torch.sparse.sum(adjacency, dim=1).to_dense().to(torch.float32)
    return {
        "indices": adjacency.indices().cpu(),
        "values": adjacency.values().cpu(),
        "shape": torch.tensor(adjacency.shape, dtype=torch.long),
        "degree": degree.cpu(),
    }


def _cotangent_laplacian(
    vertices: torch.Tensor, faces: torch.Tensor, n_vertices: int
) -> Dict[str, torch.Tensor]:
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    eps = 1e-12
    cot0 = ((v1 - v0) * (v2 - v0)).sum(dim=1) / torch.clamp(
        torch.linalg.norm(torch.cross(v1 - v0, v2 - v0, dim=1), dim=1), min=eps
    )
    cot1 = ((v0 - v1) * (v2 - v1)).sum(dim=1) / torch.clamp(
        torch.linalg.norm(torch.cross(v0 - v1, v2 - v1, dim=1), dim=1), min=eps
    )
    cot2 = ((v0 - v2) * (v1 - v2)).sum(dim=1) / torch.clamp(
        torch.linalg.norm(torch.cross(v0 - v2, v1 - v2, dim=1), dim=1), min=eps
    )

    # edge opposite vertex 0 -> (1, 2), etc.
    edge_pairs = torch.cat(
        [faces[:, [1, 2]], faces[:, [0, 2]], faces[:, [0, 1]]],
        dim=0,
    )
    weights = 0.5 * torch.cat([cot0, cot1, cot2], dim=0)

    edge_pairs = torch.sort(edge_pairs, dim=1).values
    unique_edges, inv = torch.unique(edge_pairs, dim=0, return_inverse=True)
    edge_weights = torch.zeros(unique_edges.shape[0], dtype=torch.float64)
    edge_weights.index_add_(0, inv, weights)

    row_off = torch.cat([unique_edges[:, 0], unique_edges[:, 1]], dim=0)
    col_off = torch.cat([unique_edges[:, 1], unique_edges[:, 0]], dim=0)
    val_off = torch.cat([-edge_weights, -edge_weights], dim=0)

    diag = torch.zeros(n_vertices, dtype=torch.float64)
    diag.index_add_(0, unique_edges[:, 0], edge_weights)
    diag.index_add_(0, unique_edges[:, 1], edge_weights)

    row = torch.cat([row_off, torch.arange(n_vertices)])
    col = torch.cat([col_off, torch.arange(n_vertices)])
    values = torch.cat([val_off, diag])

    lap = torch.sparse_coo_tensor(
        torch.stack([row, col], dim=0), values, (n_vertices, n_vertices)
    ).coalesce()

    tri_areas = 0.5 * torch.linalg.norm(torch.cross(v1 - v0, v2 - v0, dim=1), dim=1)
    mass_diag = torch.zeros(n_vertices, dtype=torch.float64)
    for i in range(3):
        mass_diag.index_add_(0, faces[:, i], tri_areas / 3.0)
    mass_diag = torch.clamp(mass_diag, min=1e-12)

    return {
        "indices": lap.indices().cpu(),
        "values": lap.values().to(torch.float32).cpu(),
        "shape": torch.tensor(lap.shape, dtype=torch.long),
        "type": "cotangent",
        "mass_diag": mass_diag.to(torch.float32).cpu(),
    }


def _normalized_graph_laplacian(
    adjacency: Dict[str, torch.Tensor], n_vertices: int
) -> Dict[str, torch.Tensor]:
    idx = adjacency["indices"]
    deg = torch.clamp(adjacency["degree"], min=1.0)

    norm = 1.0 / torch.sqrt(deg[idx[0]] * deg[idx[1]])
    row = torch.cat([idx[0], torch.arange(n_vertices)])
    col = torch.cat([idx[1], torch.arange(n_vertices)])
    values = torch.cat([-norm, torch.ones(n_vertices, dtype=norm.dtype)])

    lap = torch.sparse_coo_tensor(
        torch.stack([row, col], dim=0), values, (n_vertices, n_vertices)
    ).coalesce()

    return {
        "indices": lap.indices().cpu(),
        "values": lap.values().to(torch.float32).cpu(),
        "shape": torch.tensor(lap.shape, dtype=torch.long),
        "type": "normalized_graph",
        "mass_diag": torch.ones(n_vertices, dtype=torch.float32),
    }


def _build_geodesic_neighbors(
    n_vertices: int, edges: torch.Tensor, max_hops: int
) -> Dict[str, torch.Tensor]:
    neighbors = [set() for _ in range(n_vertices)]
    for u, v in edges.tolist():
        neighbors[u].add(v)
        neighbors[v].add(u)

    all_indices = []
    offsets = [0]
    for root in range(n_vertices):
        visited = {root}
        frontier = {root}
        reached = set()
        for _ in range(max_hops):
            nxt = set()
            for node in frontier:
                nxt.update(neighbors[node])
            nxt -= visited
            if not nxt:
                break
            reached.update(nxt)
            visited.update(nxt)
            frontier = nxt

        row = [root] * len(reached)
        if row:
            all_indices.extend(zip(row, sorted(reached)))
        offsets.append(len(all_indices))

    if all_indices:
        indices = torch.tensor(all_indices, dtype=torch.long)
    else:
        indices = torch.empty((0, 2), dtype=torch.long)

    return {
        "indices": indices,
        "offsets": torch.tensor(offsets, dtype=torch.long),
        "max_hops": torch.tensor(max_hops, dtype=torch.long),
    }


def _operators_to_device(data: Any, device: torch.device) -> Any:
    if torch.is_tensor(data):
        return data.to(device)
    if isinstance(data, dict):
        return {k: _operators_to_device(v, device) for k, v in data.items()}
    return data


def _cache_path(cache_dir: Path, key: str) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"mesh_operators_{key}.pt"


def build_mesh_operators(
    faces: Any,
    vertices: Optional[Any] = None,
    *,
    cache_dir: Optional[str] = None,
    include_geodesic: bool = False,
    geodesic_max_hops: int = 2,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    faces_t = _faces_to_tensor(faces)
    key = _face_cache_key(faces_t)

    cache_dir = Path(cache_dir or ".cache/sam_3d_body")
    out_path = _cache_path(cache_dir, key)

    if out_path.exists() and not force_recompute:
        cached = torch.load(out_path, map_location="cpu")
        cached["cache_path"] = str(out_path)
        return cached

    n_vertices = int(faces_t.max().item()) + 1
    edges = _build_undirected_edges(faces_t)

    adjacency = _build_vertex_adjacency(n_vertices, edges)

    if vertices is not None:
        laplacian = _cotangent_laplacian(_vertices_to_tensor(vertices), faces_t, n_vertices)
    else:
        laplacian = _normalized_graph_laplacian(adjacency, n_vertices)

    operators: Dict[str, Any] = {
        "cache_key": key,
        "num_vertices": n_vertices,
        "vertex_adjacency": adjacency,
        "laplacian": laplacian,
        "mass_matrix": {"diag": laplacian["mass_diag"]},
        "face_index": faces_t,
    }

    if include_geodesic:
        operators["geodesic"] = _build_geodesic_neighbors(
            n_vertices, edges, max_hops=max(1, geodesic_max_hops)
        )

    torch.save(operators, out_path)
    operators["cache_path"] = str(out_path)
    return operators


def get_mesh_operators(
    operators_cpu: Dict[str, Any], device: torch.device | str
) -> Dict[str, Any]:
    return _operators_to_device(operators_cpu, torch.device(device))

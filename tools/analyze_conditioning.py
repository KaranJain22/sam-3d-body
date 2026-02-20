#!/usr/bin/env python3
"""Analyze conditioning metrics and save reproducible tables/plots.

This script reads one or more tabular files containing conditioning and evaluation metrics,
then writes:
- scatter plots for required metric pairs,
- correlation summaries (Pearson/Spearman),
- monotonic trend summaries,
- a cleaned analysis CSV used to generate the outputs.

Default input discovery follows data/README.md conventions via SAM3D_BODY_ANN_DIR.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Required analyses from the task request.
REQUIRED_PAIRS: Sequence[Tuple[str, str]] = (
    ("kappa_geom", "hand_error"),
    ("kappa_geom", "face_error"),
    ("kappa_spec", "mpjpe"),
    ("kappa_spec", "pve"),
    ("kappa_spec", "cne"),
)

# Candidate column aliases seen in practice.
COLUMN_ALIASES: Dict[str, Sequence[str]] = {
    "kappa_geom": ("kappa_geom",),
    "kappa_spec": ("kappa_spec",),
    "hand_error": (
        "hand_error",
        "hands_error",
        "hand_mpjpe",
        "lhand_error",
        "rhand_error",
    ),
    "face_error": ("face_error", "faces_error", "face_mpjpe"),
    "mpjpe": ("mpjpe", "pa_mpjpe", "body_mpjpe"),
    "pve": ("pve",),
    "cne": ("cne", "cne_mean", "cne_global"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-files",
        nargs="+",
        default=None,
        help=(
            "Input files (.csv/.parquet/.json/.jsonl). If omitted, discovers files under "
            "$SAM3D_BODY_ANN_DIR, as described in data/README.md."
        ),
    )
    parser.add_argument(
        "--glob-pattern",
        default="**/*metrics*.csv",
        help="Glob used when --input-files is not set. Relative to $SAM3D_BODY_ANN_DIR.",
    )
    parser.add_argument(
        "--output-dir",
        default="tools/analysis_outputs/conditioning",
        help="Directory for output CSV and figure artifacts.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=10,
        help="Number of quantile bins for monotonic trend summaries.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=40000,
        help="Subsample scatter points above this size for plotting speed.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for plotting subsampling.",
    )
    return parser.parse_args()


def _discover_input_files(explicit: Optional[Sequence[str]], glob_pattern: str) -> List[Path]:
    if explicit:
        return [Path(p) for p in explicit]

    ann_dir = os.environ.get("SAM3D_BODY_ANN_DIR")
    if not ann_dir:
        raise ValueError(
            "No --input-files were provided and SAM3D_BODY_ANN_DIR is not set. "
            "Set SAM3D_BODY_ANN_DIR as described in data/README.md or pass --input-files."
        )

    root = Path(ann_dir)
    if not root.exists():
        raise ValueError(f"SAM3D_BODY_ANN_DIR does not exist: {root}")

    files = sorted(root.glob(glob_pattern))
    if not files:
        raise ValueError(f"No input files matched {glob_pattern!r} under {root}")
    return files


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".json":
        return pd.read_json(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    raise ValueError(f"Unsupported file type: {path}")


def _load_inputs(paths: Sequence[Path]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in paths:
        frame = _read_table(path)
        frame["source_file"] = str(path)
        frames.append(frame)
    if not frames:
        raise ValueError("No valid input tables were loaded")
    return pd.concat(frames, ignore_index=True)


def _resolve_columns(df: pd.DataFrame) -> Dict[str, str]:
    resolved: Dict[str, str] = {}
    lowered = {c.lower(): c for c in df.columns}
    for canonical, aliases in COLUMN_ALIASES.items():
        found = None
        for alias in aliases:
            if alias in df.columns:
                found = alias
                break
            if alias.lower() in lowered:
                found = lowered[alias.lower()]
                break
        if found is not None:
            resolved[canonical] = found
    return resolved


def _coerce_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x * x).sum() * (y * y).sum())
    if denom <= 0:
        return float("nan")
    return float((x * y).sum() / denom)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    xr = pd.Series(x).rank(method="average").to_numpy(dtype=np.float64)
    yr = pd.Series(y).rank(method="average").to_numpy(dtype=np.float64)
    return _pearson(xr, yr)


def _trend_summary(x: np.ndarray, y: np.ndarray, bins: int) -> Dict[str, float | str]:
    work = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(work) < max(4, bins):
        return {
            "n_bins": 0,
            "n_diffs": 0,
            "frac_non_decreasing": float("nan"),
            "frac_non_increasing": float("nan"),
            "trend_direction": "insufficient_data",
            "trend_strength": float("nan"),
        }

    work["bin"] = pd.qcut(work["x"], q=min(bins, len(work)), duplicates="drop")
    grouped = work.groupby("bin", observed=True)["y"].median().reset_index(drop=True)
    diffs = grouped.diff().dropna().to_numpy(dtype=np.float64)
    if diffs.size == 0:
        return {
            "n_bins": int(len(grouped)),
            "n_diffs": 0,
            "frac_non_decreasing": float("nan"),
            "frac_non_increasing": float("nan"),
            "trend_direction": "flat",
            "trend_strength": 1.0,
        }

    non_dec = float(np.mean(diffs >= 0.0))
    non_inc = float(np.mean(diffs <= 0.0))
    if non_dec > non_inc:
        direction = "non_decreasing"
        strength = non_dec
    elif non_inc > non_dec:
        direction = "non_increasing"
        strength = non_inc
    else:
        direction = "mixed"
        strength = non_dec

    return {
        "n_bins": int(len(grouped)),
        "n_diffs": int(diffs.size),
        "frac_non_decreasing": non_dec,
        "frac_non_increasing": non_inc,
        "trend_direction": direction,
        "trend_strength": float(strength),
    }


def _scatter_plot(
    x: np.ndarray,
    y: np.ndarray,
    x_name: str,
    y_name: str,
    out_path: Path,
    max_points: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    if n > max_points:
        idx = rng.choice(n, size=max_points, replace=False)
        x = x[idx]
        y = y[idx]

    fig, ax = plt.subplots(figsize=(6, 5), dpi=160)
    ax.scatter(x, y, s=8, alpha=0.35, edgecolors="none")

    if x.size >= 2 and np.unique(x).size > 1:
        slope, intercept = np.polyfit(x, y, deg=1)
        xs = np.linspace(np.min(x), np.max(x), 200)
        ax.plot(xs, slope * xs + intercept, linewidth=2)

    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_title(f"{x_name} vs {y_name}")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_files = _discover_input_files(args.input_files, args.glob_pattern)
    out_dir = Path(args.output_dir)
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    raw = _load_inputs(input_files)
    resolved = _resolve_columns(raw)

    needed = sorted({name for pair in REQUIRED_PAIRS for name in pair})
    missing = [n for n in needed if n not in resolved]
    if missing:
        raise ValueError(
            "Missing required columns after alias resolution: "
            f"{missing}. Available columns: {list(raw.columns)}"
        )

    working = raw[[resolved[n] for n in needed] + ["source_file"]].copy()
    rename_map = {resolved[n]: n for n in needed}
    working = working.rename(columns=rename_map)
    working = _coerce_numeric(working, needed)

    cleaned = working.dropna(subset=needed).reset_index(drop=True)
    cleaned.to_csv(out_dir / "analysis_table.csv", index=False)

    corr_rows = []
    trend_rows = []
    for x_name, y_name in REQUIRED_PAIRS:
        pair = cleaned[[x_name, y_name]].dropna()
        x = pair[x_name].to_numpy(dtype=np.float64)
        y = pair[y_name].to_numpy(dtype=np.float64)

        pearson = _pearson(x, y)
        spearman = _spearman(x, y)
        corr_rows.append(
            {
                "x": x_name,
                "y": y_name,
                "n": int(len(pair)),
                "pearson": pearson,
                "spearman": spearman,
            }
        )

        trend = _trend_summary(x, y, bins=args.bins)
        trend_rows.append({"x": x_name, "y": y_name, **trend})

        fig_name = f"scatter_{x_name}_vs_{y_name}.png"
        _scatter_plot(
            x=x,
            y=y,
            x_name=x_name,
            y_name=y_name,
            out_path=fig_dir / fig_name,
            max_points=args.max_points,
            seed=args.seed,
        )

    pd.DataFrame(corr_rows).to_csv(out_dir / "correlations.csv", index=False)
    pd.DataFrame(trend_rows).to_csv(out_dir / "monotonic_trends.csv", index=False)

    manifest = pd.DataFrame({"input_file": [str(p) for p in input_files]})
    manifest.to_csv(out_dir / "input_manifest.csv", index=False)

    print(f"Wrote analysis outputs to: {out_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate pairwise realism judgments against geometric/perceptual metrics.

Computes Pearson and Spearman correlations between aggregated human preference
scores and pairwise metric deltas for MPJPE, PVE, CNE, and CNE-log.

Also compares static semantic weights vs learned omega(v) variants and reports
paired bootstrap significance tests for correlation differences.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", required=True, help="Aggregated ratings CSV/JSONL")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        return pd.read_json(path)
    raise ValueError(f"Unsupported file type: {path}")


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


def _pairwise_delta(df: pd.DataFrame, metric_col_a: str, metric_col_b: str) -> np.ndarray:
    # Higher score means candidate A is more realistic to align with preference_score.
    a = pd.to_numeric(df[metric_col_a], errors="coerce").to_numpy(dtype=np.float64)
    b = pd.to_numeric(df[metric_col_b], errors="coerce").to_numpy(dtype=np.float64)
    return b - a


def _available_metric_pairs(df: pd.DataFrame) -> List[Tuple[str, str, str]]:
    candidates = [
        ("mpjpe", "candidate_a_mpjpe", "candidate_b_mpjpe"),
        ("pve", "candidate_a_pve", "candidate_b_pve"),
        ("cne", "candidate_a_cne", "candidate_b_cne"),
        ("cne_log", "candidate_a_cne_log", "candidate_b_cne_log"),
        ("cne_static", "candidate_a_cne_static", "candidate_b_cne_static"),
        ("cne_log_static", "candidate_a_cne_log_static", "candidate_b_cne_log_static"),
        ("cne_learned", "candidate_a_cne_learned", "candidate_b_cne_learned"),
        ("cne_log_learned", "candidate_a_cne_log_learned", "candidate_b_cne_log_learned"),
    ]
    return [(name, a, b) for (name, a, b) in candidates if a in df.columns and b in df.columns]


def compute_correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    target = pd.to_numeric(df["preference_score"], errors="coerce").to_numpy(dtype=np.float64)
    rows = []
    for metric_name, col_a, col_b in _available_metric_pairs(df):
        score = _pairwise_delta(df, col_a, col_b)
        mask = np.isfinite(target) & np.isfinite(score)
        x = target[mask]
        y = score[mask]
        if x.size < 3:
            continue
        rows.append(
            {
                "metric": metric_name,
                "n": int(x.size),
                "pearson_r": _pearson(x, y),
                "spearman_r": _spearman(x, y),
            }
        )
    return pd.DataFrame(rows).sort_values("metric").reset_index(drop=True)


def _bootstrap_corr_diff(
    target: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(target)
    deltas = np.zeros(n_bootstrap, dtype=np.float64)

    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        t = target[idx]
        a = score_a[idx]
        b = score_b[idx]
        deltas[i] = _spearman(t, a) - _spearman(t, b)

    obs = _spearman(target, score_a) - _spearman(target, score_b)
    ci_low, ci_high = np.quantile(deltas, [0.025, 0.975])
    p_two_sided = min(1.0, 2.0 * min(np.mean(deltas <= 0.0), np.mean(deltas >= 0.0)))
    return {
        "delta_spearman": float(obs),
        "bootstrap_ci_low": float(ci_low),
        "bootstrap_ci_high": float(ci_high),
        "bootstrap_p": float(p_two_sided),
    }


def compute_significance(df: pd.DataFrame, n_bootstrap: int, seed: int) -> pd.DataFrame:
    target = pd.to_numeric(df["preference_score"], errors="coerce").to_numpy(dtype=np.float64)
    comparisons = [
        ("cne", "candidate_a_cne_learned", "candidate_b_cne_learned", "candidate_a_cne_static", "candidate_b_cne_static"),
        (
            "cne_log",
            "candidate_a_cne_log_learned",
            "candidate_b_cne_log_learned",
            "candidate_a_cne_log_static",
            "candidate_b_cne_log_static",
        ),
    ]

    rows = []
    for name, la, lb, sa, sb in comparisons:
        if not all(c in df.columns for c in (la, lb, sa, sb)):
            continue
        learned = _pairwise_delta(df, la, lb)
        static = _pairwise_delta(df, sa, sb)
        mask = np.isfinite(target) & np.isfinite(learned) & np.isfinite(static)
        if mask.sum() < 8:
            continue
        stats = _bootstrap_corr_diff(
            target=target[mask],
            score_a=learned[mask],
            score_b=static[mask],
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        rows.append(
            {
                "comparison": f"{name}: learned_omega_vs_static_semantic",
                "n": int(mask.sum()),
                **stats,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = _read_table(Path(args.input_file))
    if "preference_score" not in df.columns:
        raise ValueError("Input must contain preference_score (use preprocess_pairwise_ratings.py)")

    corr = compute_correlation_table(df)
    sig = compute_significance(df, n_bootstrap=args.bootstrap_samples, seed=args.seed)

    corr_path = output_dir / "pairwise_metric_correlations.csv"
    sig_path = output_dir / "pairwise_significance_tests.csv"
    corr.to_csv(corr_path, index=False)
    sig.to_csv(sig_path, index=False)

    print(f"Wrote {corr_path}")
    print(f"Wrote {sig_path}")


if __name__ == "__main__":
    main()

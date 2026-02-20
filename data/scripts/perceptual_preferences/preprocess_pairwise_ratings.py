#!/usr/bin/env python3
"""Preprocess raw pairwise realism ratings into canonical tables.

This script accepts CSV/JSON/JSONL files exported from annotation tools,
normalizes column names, aggregates repeated ratings per pair, and writes:

1) `pairwise_realism_aggregated.csv` for analysis/evaluation.
2) `pairwise_realism_aggregated.jsonl` in canonical pairwise format.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd


COLUMN_ALIASES: Dict[str, Sequence[str]] = {
    "sample_id": ("sample_id", "comparison_id", "pair_id", "id"),
    "image_id": ("image_id", "frame_id"),
    "person_id": ("person_id", "track_id", "subject_id"),
    "candidate_a_id": ("candidate_a_id", "mesh_a", "mesh_id_a", "candidate_a_mesh_id"),
    "candidate_b_id": ("candidate_b_id", "mesh_b", "mesh_id_b", "candidate_b_mesh_id"),
    "rater_id": ("rater_id", "worker_id", "annotator_id"),
    "winner": ("winner", "preferred", "label", "choice"),
    "confidence": ("confidence", "rating_confidence", "score"),
}

CANDIDATE_METRIC_PREFIXES = (
    "mpjpe",
    "pve",
    "cne",
    "cne_log",
    "cne_static",
    "cne_log_static",
    "cne_learned",
    "cne_log_learned",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-files", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
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


def _resolve_col(df: pd.DataFrame, aliases: Iterable[str]) -> str | None:
    lowered = {c.lower(): c for c in df.columns}
    for alias in aliases:
        if alias in df.columns:
            return alias
        if alias.lower() in lowered:
            return lowered[alias.lower()]
    return None


def _canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for canonical, aliases in COLUMN_ALIASES.items():
        found = _resolve_col(out, aliases)
        if found is not None and found != canonical:
            out = out.rename(columns={found: canonical})

    if "sample_id" not in out.columns:
        out["sample_id"] = out.index.map(lambda i: f"sample_{i}")
    if "winner" not in out.columns:
        raise ValueError("Input must include winner/preferred label column")

    out["winner"] = (
        out["winner"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"left": "a", "right": "b", "0": "a", "1": "b", "tie": "t"})
    )
    out.loc[~out["winner"].isin(["a", "b", "t"]), "winner"] = "t"

    if "confidence" in out.columns:
        out["confidence"] = pd.to_numeric(out["confidence"], errors="coerce")

    return out


def _load_inputs(paths: Sequence[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = _canonicalize_columns(_read_table(path))
        frame["source_file"] = str(path)
        frames.append(frame)
    if not frames:
        raise ValueError("No rating files were loaded")
    return pd.concat(frames, ignore_index=True)


def aggregate_pairwise_ratings(raw: pd.DataFrame) -> pd.DataFrame:
    group_cols = [c for c in ["sample_id", "image_id", "person_id", "candidate_a_id", "candidate_b_id"] if c in raw.columns]

    metric_cols = []
    for prefix in CANDIDATE_METRIC_PREFIXES:
        for side in ("a", "b"):
            col = f"candidate_{side}_{prefix}"
            if col in raw.columns:
                metric_cols.append(col)

    rows: List[Dict[str, object]] = []
    for _, grp in raw.groupby(group_cols, dropna=False):
        votes_a = int((grp["winner"] == "a").sum())
        votes_b = int((grp["winner"] == "b").sum())
        votes_t = int((grp["winner"] == "t").sum())
        n = len(grp)
        if n <= 0:
            continue

        preference_score = (votes_a - votes_b) / float(n)
        preferred = "a" if votes_a > votes_b else "b" if votes_b > votes_a else "t"

        row: Dict[str, object] = {
            "n_ratings": int(n),
            "votes_a": votes_a,
            "votes_b": votes_b,
            "votes_tie": votes_t,
            "preference_score": preference_score,
            "preferred": preferred,
        }
        if "confidence" in grp.columns:
            row["confidence"] = float(pd.to_numeric(grp["confidence"], errors="coerce").mean())

        for c in group_cols:
            row[c] = grp.iloc[0][c]
        for c in metric_cols:
            row[c] = pd.to_numeric(grp[c], errors="coerce").mean()
        rows.append(row)

    return pd.DataFrame(rows)


def to_canonical_jsonl_rows(aggregated: pd.DataFrame) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for _, r in aggregated.iterrows():
        candidate_a = {"mesh_id": str(r.get("candidate_a_id", ""))}
        candidate_b = {"mesh_id": str(r.get("candidate_b_id", ""))}
        for col in aggregated.columns:
            if col.startswith("candidate_a_") and col not in {"candidate_a_id"}:
                candidate_a[col.replace("candidate_a_", "")] = r[col]
            if col.startswith("candidate_b_") and col not in {"candidate_b_id"}:
                candidate_b[col.replace("candidate_b_", "")] = r[col]

        rows.append(
            {
                "sample_id": str(r["sample_id"]),
                "image_id": str(r.get("image_id", "")),
                "person_id": r.get("person_id", ""),
                "candidate_a": candidate_a,
                "candidate_b": candidate_b,
                "preferred": r["preferred"],
                "preference_score": float(r["preference_score"]),
                "n_ratings": int(r["n_ratings"]),
                "confidence": float(r.get("confidence", 1.0)),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = _load_inputs([Path(p) for p in args.input_files])
    aggregated = aggregate_pairwise_ratings(raw).sort_values("sample_id").reset_index(drop=True)

    csv_path = output_dir / "pairwise_realism_aggregated.csv"
    jsonl_path = output_dir / "pairwise_realism_aggregated.jsonl"
    aggregated.to_csv(csv_path, index=False)

    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in to_canonical_jsonl_rows(aggregated):
            f.write(json.dumps(row) + "\n")

    print(f"Wrote {csv_path}")
    print(f"Wrote {jsonl_path}")


if __name__ == "__main__":
    main()

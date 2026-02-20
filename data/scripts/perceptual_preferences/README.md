# Pairwise Realism Judgments: Format + Evaluation

This folder defines a review-ready workflow for pairwise human realism judgments:

1. **Annotation format** (`schema.json`) for pairwise ratings.
2. **Preprocessing** (`preprocess_pairwise_ratings.py`) to aggregate raw ratings.
3. **Evaluation** (`evaluate_pairwise_realism.py`) to correlate human judgments with MPJPE/PVE/CNE/CNE-log and run significance tests for learned `omega(v)` vs static semantic weighting.

## Annotation format

Each JSONL row is one pairwise comparison for the same `(image_id, person_id)`.

Required top-level fields:

- `sample_id`: unique pair id.
- `image_id`: frame/image id.
- `person_id`: person track id.
- `candidate_a`, `candidate_b`: candidate metadata.
- `preferred`: majority label (`"a"`, `"b"`, or `"t"` for tie).

Recommended aggregate fields:

- `preference_score` in `[-1, 1]` where +1 means unanimous for A, -1 unanimous for B.
- `n_ratings`, `confidence`.

Candidate entries can be either:
- **full vertex payload** (`semantic_part_ids`, `curvature_scale_features`, `visibility`, `screen_size`, optional `vertex_error`), or
- **scalar metric payload** (`mpjpe`, `pve`, `cne`, `cne_log`, `cne_static`, `cne_log_static`, `cne_learned`, `cne_log_learned`).

## Preprocess raw ratings

Convert annotation exports (CSV/JSON/JSONL) into canonical aggregate outputs:

```bash
python data/scripts/perceptual_preferences/preprocess_pairwise_ratings.py \
  --input-files /path/raw_batch1.csv /path/raw_batch2.csv \
  --output-dir /path/processed
```

Outputs:

- `pairwise_realism_aggregated.csv`
- `pairwise_realism_aggregated.jsonl`

## Evaluate metric alignment with human judgments

```bash
python data/scripts/perceptual_preferences/evaluate_pairwise_realism.py \
  --input-file /path/processed/pairwise_realism_aggregated.csv \
  --output-dir /path/processed/reports \
  --bootstrap-samples 2000
```

Outputs:

- `pairwise_metric_correlations.csv` with Pearson/Spearman for MPJPE, PVE, CNE, CNE-log (and static/learned variants when present).
- `pairwise_significance_tests.csv` with paired bootstrap tests of correlation improvement for:
  - `cne`: learned `omega(v)` vs static semantic weights
  - `cne_log`: learned `omega(v)` vs static semantic weights

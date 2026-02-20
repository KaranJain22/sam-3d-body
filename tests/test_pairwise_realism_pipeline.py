import pandas as pd

from data.scripts.perceptual_preferences.evaluate_pairwise_realism import (
    compute_correlation_table,
    compute_significance,
)
from data.scripts.perceptual_preferences.preprocess_pairwise_ratings import (
    aggregate_pairwise_ratings,
)


def test_aggregate_pairwise_ratings_majority_vote():
    raw = pd.DataFrame(
        [
            {
                "sample_id": "s0",
                "image_id": "img0",
                "person_id": 1,
                "candidate_a_id": "a0",
                "candidate_b_id": "b0",
                "winner": "a",
            },
            {
                "sample_id": "s0",
                "image_id": "img0",
                "person_id": 1,
                "candidate_a_id": "a0",
                "candidate_b_id": "b0",
                "winner": "b",
            },
            {
                "sample_id": "s0",
                "image_id": "img0",
                "person_id": 1,
                "candidate_a_id": "a0",
                "candidate_b_id": "b0",
                "winner": "a",
            },
        ]
    )
    agg = aggregate_pairwise_ratings(raw)
    assert len(agg) == 1
    assert agg.loc[0, "preferred"] == "a"
    assert agg.loc[0, "preference_score"] == (2 - 1) / 3


def test_correlation_and_significance_outputs():
    # Construct data where learned omega tracks human preference better than static.
    pref = [-1, -0.8, -0.5, -0.2, 0.2, 0.5, 0.8, 1.0, 0.6, -0.6]
    df = pd.DataFrame(
        {
            "preference_score": pref,
            "candidate_a_mpjpe": [2.0] * 10,
            "candidate_b_mpjpe": [2.0 - x for x in pref],
            "candidate_a_pve": [3.0] * 10,
            "candidate_b_pve": [3.0 - x for x in pref],
            "candidate_a_cne_learned": [1.0] * 10,
            "candidate_b_cne_learned": [1.0 - x for x in pref],
            "candidate_a_cne_static": [1.0] * 10,
            "candidate_b_cne_static": [1.0 - 0.2 * x for x in pref],
            "candidate_a_cne_log_learned": [1.0] * 10,
            "candidate_b_cne_log_learned": [1.0 - x for x in pref],
            "candidate_a_cne_log_static": [1.0] * 10,
            "candidate_b_cne_log_static": [1.0 - 0.2 * x for x in pref],
        }
    )

    corr = compute_correlation_table(df)
    assert {"mpjpe", "pve", "cne_learned", "cne_static"}.issubset(set(corr["metric"]))

    sig = compute_significance(df, n_bootstrap=200, seed=0)
    assert len(sig) >= 1
    assert (sig["delta_spearman"] > 0).all()

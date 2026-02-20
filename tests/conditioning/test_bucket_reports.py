import pandas as pd

from tools.analyze_conditioning import _assign_quantile_bucket, _bucket_metric_report


def test_assign_quantile_bucket_labels_all_three_ranges():
    values = pd.Series([0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 2.0, 2.1, 2.2])
    buckets = _assign_quantile_bucket(values)

    assert set(buckets.dropna().unique()) == {"low", "med", "high"}
    assert (buckets.iloc[:3] == "low").all()
    assert (buckets.iloc[3:6] == "med").all()
    assert (buckets.iloc[6:] == "high").all()


def test_bucket_metric_report_contains_required_metrics():
    df = pd.DataFrame(
        {
            "kappa_geom": [0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 2.0, 2.1, 2.2],
            "kappa_spec": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            "mpjpe": [10] * 9,
            "pve": [20] * 9,
            "cne": [30] * 9,
            "cne_log": [3.4] * 9,
            "hand_error": [40] * 9,
            "face_error": [50] * 9,
        }
    )

    report = _bucket_metric_report(df, "kappa_geom")

    assert list(report["bucket"]) == ["low", "med", "high"]
    assert all(report["n"] == 3)
    for col in ["mpjpe", "pve", "cne", "cne_log", "hand_error", "face_error"]:
        assert col in report.columns

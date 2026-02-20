import numpy as np

from sam_3d_body.metrics.conditioning_metrics import cne, cne_log, summarize_cne_metrics


def test_cne_and_cne_log_match_expected_for_simple_case():
    pred = np.array([[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]])
    gt = np.zeros_like(pred)
    scale = np.array([[2.0, 4.0]])

    got_cne = cne(pred, gt, scale, tau=0.0)
    got_cne_log = cne_log(pred, gt, scale, eps=1e-12)

    assert np.isclose(got_cne, 0.5)
    assert np.isclose(got_cne_log, np.mean(np.abs(np.log([0.5, 0.5]))))


def test_summarize_cne_metrics_with_perceptual_and_regions():
    pred = np.array([[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]])
    gt = np.zeros_like(pred)
    scale = np.ones((1, 3))
    omega = np.array([[1.0, 3.0, 6.0]])

    region_masks = {
        "hands": np.array([True, False, False]),
        "face": np.array([False, True, False]),
        "body": np.array([False, False, True]),
    }

    summary = summarize_cne_metrics(
        pred,
        gt,
        scale,
        tau=0.0,
        eps=1e-12,
        omega=omega,
        region_masks=region_masks,
    )

    assert "cne" in summary
    assert "cne_perceptual" in summary
    assert np.isclose(summary["hands/cne"], 1.0)
    assert np.isclose(summary["face/cne"], 2.0)
    assert np.isclose(summary["body/cne"], 3.0)
    assert np.isclose(summary["cne_perceptual"], (1 * 1 + 2 * 3 + 3 * 6) / 10)

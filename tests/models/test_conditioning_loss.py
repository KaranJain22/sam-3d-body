import torch

from sam_3d_body.models.losses import ConditioningLoss


def test_conditioning_loss_inverse_scale_weights():
    loss_fn = ConditioningLoss(tau=0.1)
    local_scale = torch.tensor([[0.0, 0.9]], dtype=torch.float32)

    alpha = loss_fn.vertex_weights(local_scale)
    expected_raw = torch.tensor([[10.0, 1.0]], dtype=torch.float32)
    expected = expected_raw / expected_raw.sum(dim=-1, keepdim=True)
    assert torch.allclose(alpha, expected, atol=1e-6)


def test_conditioning_loss_perceptual_modulation():
    loss_fn = ConditioningLoss(tau=0.0)

    pred = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]])
    target = torch.zeros_like(pred)
    local_scale = torch.tensor([[1.0, 1.0]])
    omega = torch.tensor([[2.0, 0.5]])

    # weighted L2 norms: [1.0, 2.0] with normalized alpha [0.8, 0.2]
    loss = loss_fn(pred, target, local_scale, perceptual_weights=omega)
    assert torch.allclose(loss, torch.tensor(1.2), atol=1e-6)

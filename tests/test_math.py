import pytest
import torch

from vita.math import TemporalDCT, dct_matrix, progressive_mask, ssim_loss, straight_through_lookup
from vita.quantizer import SharedVectorQuantizer


@pytest.mark.parametrize("horizon", [1, 4, 12, 32])
def test_dct_is_orthonormal_and_invertible(horizon):
    matrix = dct_matrix(horizon, dtype=torch.float64)
    torch.testing.assert_close(matrix @ matrix.T, torch.eye(horizon, dtype=torch.float64), atol=1e-12, rtol=1e-12)
    transform = TemporalDCT(horizon)
    values = torch.randn(2, horizon, 7, requires_grad=True)
    restored = transform.inverse(transform(values))
    torch.testing.assert_close(restored, values, atol=1e-5, rtol=1e-5)
    restored.square().sum().backward()
    assert torch.isfinite(values.grad).all()
    assert values.grad.abs().sum() > 0


def test_dct_constant_signal_has_only_dc():
    transform = TemporalDCT(12)
    values = torch.ones(2, 12, 7)
    coefficients = transform(values)
    torch.testing.assert_close(coefficients[:, 1:], torch.zeros_like(coefficients[:, 1:]), atol=1e-5, rtol=0)


def test_progressive_attention_direction():
    mask = progressive_mask(3, 2, 4)
    assert not mask[:3, :3].any()
    assert mask[:3, 3:].all()
    assert not mask[3:5, :5].any()
    assert mask[3:5, 5:].all()
    assert not mask[5:, :5].any()
    assert torch.equal(mask[5:, 5:], torch.ones(4, 4, dtype=torch.bool).triu(1))


@pytest.mark.parametrize("counts", [(0, 2, 3), (2, -1, 3), (2, 1, -1)])
def test_invalid_mask(counts):
    with pytest.raises(ValueError):
        progressive_mask(*counts)


def test_straight_through_tokens_are_hard_but_differentiable():
    logits = torch.randn(2, 3, 8, requires_grad=True)
    weights = torch.randn(8, 5)
    values, indices = straight_through_lookup(logits, weights, 1.0)
    torch.testing.assert_close(values, weights[indices], atol=0, rtol=0)
    values.square().sum().backward()
    assert logits.grad.abs().sum() > 0


@pytest.mark.parametrize("update", [True, False])
def test_quantizer_nearest_neighbor_and_gradient_ownership(update):
    quantizer = SharedVectorQuantizer(3, 2)
    with torch.no_grad():
        quantizer.embedding.weight.copy_(torch.tensor([[0., 0.], [1., 0.], [0., 1.]]))
    values = torch.tensor([[[0.1, 0.2], [0.9, 0.1]]], requires_grad=True)
    quantized = quantizer(values, update_codebook=update)
    assert quantized.indices.tolist() == [[0, 1]]
    torch.testing.assert_close(quantized.values, quantizer.embedding.weight[quantized.indices])
    (quantized.values.sum() + quantized.loss).backward()
    assert values.grad.abs().sum() > 0
    grad = quantizer.embedding.weight.grad
    assert (grad is not None and grad.abs().sum() > 0) if update else grad is None


def test_ssim_identical_and_perturbed_images():
    image = torch.rand(2, 3, 3, 16, 16)
    assert abs(float(ssim_loss(image, image))) < 1e-6
    prediction = (image * 0.8).requires_grad_()
    loss = ssim_loss(prediction, image)
    assert loss > 0
    loss.backward()
    assert torch.isfinite(prediction.grad).all()

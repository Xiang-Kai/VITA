from __future__ import annotations

import math
import torch
from torch import Tensor, nn
from torch.nn import functional as F


def dct_matrix(size: int, *, device=None, dtype=torch.float32) -> Tensor:
    if size < 1:
        raise ValueError("DCT size must be positive")
    n = torch.arange(size, device=device, dtype=dtype)
    k = n[:, None]
    matrix = torch.cos(math.pi * (n + 0.5) * k / size)
    matrix[0] *= math.sqrt(1.0 / size)
    matrix[1:] *= math.sqrt(2.0 / size)
    return matrix


class TemporalDCT(nn.Module):
    """Orthonormal DCT-II on the time axis, never on action dimensions."""

    def __init__(self, horizon: int):
        super().__init__()
        self.register_buffer("matrix", dct_matrix(horizon))

    def forward(self, actions: Tensor) -> Tensor:
        return torch.einsum("kt,...td->...kd", self.matrix.to(actions), actions)

    def inverse(self, coefficients: Tensor) -> Tensor:
        return torch.einsum("kt,...kd->...td", self.matrix.to(coefficients), coefficients)


def progressive_mask(context: int, subtasks: int, hybrid: int, *, device=None) -> Tensor:
    """True blocks attention; rows are queries, columns are keys."""
    if min(context, subtasks, hybrid) < 0 or context == 0:
        raise ValueError("Need context and nonnegative group lengths")
    length = context + subtasks + hybrid
    allowed = torch.zeros(length, length, dtype=torch.bool, device=device)
    allowed[:context, :context] = True
    prefix = context + subtasks
    allowed[context:prefix, :prefix] = True
    allowed[prefix:, :prefix] = True
    allowed[prefix:, prefix:] = torch.ones(
        hybrid, hybrid, dtype=torch.bool, device=device
    ).tril()
    return ~allowed


def straight_through_lookup(logits: Tensor, weights: Tensor, temperature: float) -> tuple[Tensor, Tensor]:
    probabilities = (logits.float() / temperature).softmax(dim=-1).to(weights.dtype)
    indices = probabilities.argmax(dim=-1)
    hard = F.one_hot(indices, weights.shape[0]).to(probabilities)
    # Hard tokens in the forward pass; explicitly chosen softmax gradient estimator.
    distribution = hard + (probabilities - probabilities.detach())
    return distribution @ weights, indices


def ssim_loss(prediction: Tensor, target: Tensor) -> Tensor:
    """Local 3x3 SSIM for [0, 1] RGB; window size is a reproduction assumption."""
    if prediction.shape != target.shape or prediction.ndim != 5:
        raise ValueError("SSIM expects matching [B,T,C,H,W] tensors")
    x = prediction.float().flatten(0, 1)
    y = target.float().flatten(0, 1)
    pool = lambda z: F.avg_pool2d(z, 3, stride=1)
    mx, my = pool(x), pool(y)
    vx = (pool(x * x) - mx.square()).clamp_min(0)
    vy = (pool(y * y) - my.square()).clamp_min(0)
    covariance = pool(x * y) - mx * my
    similarity = ((2 * mx * my + 0.01**2) * (2 * covariance + 0.03**2)) / (
        (mx.square() + my.square() + 0.01**2) * (vx + vy + 0.03**2)
    )
    return 1 - similarity.mean()

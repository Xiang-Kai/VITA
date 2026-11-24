from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class Quantized:
    values: Tensor
    indices: Tensor
    loss: Tensor
    perplexity: Tensor


class SharedVectorQuantizer(nn.Module):
    def __init__(self, size: int, dimension: int, commitment_weight: float = 0.25):
        super().__init__()
        self.embedding = nn.Embedding(size, dimension)
        nn.init.normal_(self.embedding.weight, std=dimension**-0.5)
        self.commitment_weight = commitment_weight

    @torch.no_grad()
    def initialize_from_data(self, latents: Tensor, *, seed: int = 0, iterations: int = 20) -> dict:
        """Explicit optional k-means initialization using training latents only."""
        if not self.embedding.weight.requires_grad:
            raise RuntimeError("Cannot initialize a frozen codebook")
        points = latents.detach().reshape(-1, self.embedding.embedding_dim).float()
        size = self.embedding.num_embeddings
        if len(points) < size or not torch.isfinite(points).all() or iterations < 1:
            raise ValueError("Need finite training latents, at least K points, and positive iterations")
        generator = torch.Generator(device=points.device).manual_seed(seed)
        centers = points[torch.randperm(len(points), generator=generator, device=points.device)[:size]].clone()
        for _ in range(iterations):
            labels = self.nearest(points, centers)
            sums = torch.zeros_like(centers).index_add_(0, labels, points)
            counts = torch.bincount(labels, minlength=size)
            nonempty = counts > 0
            centers[nonempty] = sums[nonempty] / counts[nonempty, None]
            missing = (~nonempty).sum().item()
            if missing:
                indices = torch.randperm(len(points), generator=generator, device=points.device)[:missing]
                centers[~nonempty] = points[indices]
        self.embedding.weight.copy_(centers.to(self.embedding.weight))
        used = self.nearest(points, centers).unique().numel()
        return {"method": "training_latent_kmeans", "points": len(points), "iterations": iterations,
                "occupied_codes_after_initialization": used, "seed": seed}

    @staticmethod
    @torch.no_grad()
    def nearest(inputs: Tensor, weights: Tensor, chunk_size: int = 1024) -> Tensor:
        outputs = []
        # .float() alone does not stop autocast from lowering the distance matmul.
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            c = weights.float()
            for part in inputs.reshape(-1, inputs.shape[-1]).split(chunk_size):
                x = part.float()
                distances = x.square().sum(-1, keepdim=True) + c.square().sum(-1) - 2 * x @ c.T
                outputs.append(distances.argmin(-1))
        return torch.cat(outputs).reshape(inputs.shape[:-1])

    def forward(self, inputs: Tensor, *, update_codebook: bool) -> Quantized:
        flat = inputs.reshape(-1, inputs.shape[-1])
        weights = self.embedding.weight
        indices = self.nearest(flat, weights).view(inputs.shape[:-1])
        codes = self.embedding(indices)
        commitment = F.mse_loss(inputs.float(), codes.detach().float())
        loss = self.commitment_weight * commitment
        if update_codebook:
            loss = loss + F.mse_loss(codes.float(), inputs.detach().float())
        quantized = inputs + (codes - inputs).detach()
        usage = torch.bincount(indices.flatten(), minlength=weights.shape[0]).float()
        probabilities = usage / usage.sum()
        perplexity = (-(probabilities * probabilities.clamp_min(1e-10).log()).sum()).exp()
        return Quantized(quantized, indices, loss, perplexity)

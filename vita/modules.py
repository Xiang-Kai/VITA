from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .config import VITAConfig
from .math import TemporalDCT


def encoder(width: int, heads: int, layers: int) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        width, heads, width * 4, dropout=0.0, activation="gelu",
        batch_first=True, norm_first=True,
    )
    return nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(width), enable_nested_tensor=False)


def decoder(width: int, heads: int, layers: int) -> nn.TransformerDecoder:
    layer = nn.TransformerDecoderLayer(
        width, heads, width * 4, dropout=0.0, activation="gelu",
        batch_first=True, norm_first=True,
    )
    return nn.TransformerDecoder(layer, layers, norm=nn.LayerNorm(width))


def decode(net, queries, memory, checkpoint_layers=False):
    if not checkpoint_layers or not torch.is_grad_enabled() or not net.training:
        return net(queries, memory)
    values = queries
    for layer in net.layers:
        values = checkpoint(layer, values, memory, use_reentrant=False)
    return net.norm(values) if net.norm is not None else values


class FrozenImageEncoder(nn.Module):
    def __init__(self, config: VITAConfig):
        super().__init__()
        self.backend = config.vision_backend
        self.batch_size = config.feature_batch_size
        if self.backend == "dinov2":
            from transformers import AutoConfig, AutoModel
            foundation_config = AutoConfig.from_pretrained(config.dinov2_path, local_files_only=True)
            if foundation_config.model_type not in {"dinov2", "dinov2_with_registers"}:
                raise ValueError("Expected DINOv2 or explicitly declared DINOv2-with-registers checkpoint")
            self.model, loading = AutoModel.from_pretrained(
                config.dinov2_path, local_files_only=True, output_loading_info=True)
            if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
                raise ValueError(f"Incomplete DINOv2 checkpoint: {loading}")
            self.feature_dim = self.model.config.hidden_size
            self.prefix_tokens = 1 + getattr(self.model.config, "num_register_tokens", 0)
        else:
            self.model = nn.Conv2d(3, config.feature_dim, config.patch_size, config.patch_size)
            self.feature_dim = config.feature_dim
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, images: Tensor) -> Tensor:
        if self.batch_size and len(images) > self.batch_size:
            return torch.cat([self.forward(part) for part in images.split(self.batch_size)], dim=0)
        if self.backend == "dinov2":
            normalized = (images - self.mean) / self.std
            return self.model(pixel_values=normalized).last_hidden_state[:, self.prefix_tokens:]
        return self.model(images).flatten(2).transpose(1, 2)


class MotionFormer(nn.Module):
    """Frame-pair queries with optional recurrent memory and L-token resampling."""

    def __init__(self, config: VITAConfig, feature_dim: int):
        super().__init__()
        self.project = nn.Linear(2 * feature_dim, config.latent_dim)
        self.query = nn.Parameter(torch.randn(1, 1, config.latent_dim) * 0.02)
        self.net = decoder(config.latent_dim, config.mformer_heads, config.mformer_layers)
        self.checkpoint_layers = config.gradient_checkpointing
        self.memory = (nn.Parameter(torch.randn(1, config.mformer_memory_tokens, config.latent_dim) * 0.02)
                       if config.mformer_memory_tokens else None)
        self.output_queries = (
            nn.Parameter(torch.randn(1, config.token_count, config.latent_dim) * 0.02)
            if config.token_count != config.future_frames else None
        )

    def forward(self, current: Tensor, future: Tensor) -> Tensor:
        batch, time, patches, dim = future.shape
        current = current[:, None].expand(-1, time, -1, -1)
        memory = self.project(torch.cat([current, future], dim=-1)).flatten(0, 1)
        # A deterministic 2-D position encoding retains spatial layout without a fixed DINO grid.
        side = int(patches**0.5)
        if side * side != patches:
            raise ValueError("Expected a square image patch grid")
        axis = torch.linspace(-1, 1, side, device=memory.device, dtype=memory.dtype)
        y, x = torch.meshgrid(axis, axis, indexing="ij")
        coordinates = torch.stack([x.flatten(), y.flatten()], dim=-1)
        repeats = (memory.shape[-1] + 1) // 2
        positions = coordinates.repeat(1, repeats)[:, :memory.shape[-1]]
        memory = memory + positions[None]
        if self.memory is None:
            motion = decode(self.net, self.query.expand(batch * time, -1, -1), memory,
                            self.checkpoint_layers).reshape(batch, time, -1)
        else:
            # Memory is reset per example, carried only within its future-frame sequence.
            carry = self.memory.expand(batch, -1, -1)
            motion = []
            for frame in memory.unflatten(0, (batch, time)).unbind(1):
                queries = torch.cat([self.query.expand(batch, -1, -1), carry], dim=1)
                hidden = decode(self.net, queries, torch.cat([frame, carry], dim=1),
                                self.checkpoint_layers)
                motion.append(hidden[:, :1])
                carry = hidden[:, 1:]
            motion = torch.cat(motion, dim=1)
        if self.output_queries is not None:
            motion = decode(self.net, self.output_queries.expand(batch, -1, -1), motion,
                            self.checkpoint_layers)
        return motion


class VisualDecoder(nn.Module):
    def __init__(self, config: VITAConfig, feature_dim: int):
        super().__init__()
        self.config = config
        self.grid = config.image_size // config.patch_size
        self.context = nn.Linear(feature_dim, config.visual_dim)
        self.latents = nn.Linear(config.latent_dim, config.visual_dim)
        self.latent_positions = (
            nn.Parameter(torch.randn(1, config.token_count, config.visual_dim) * 0.02)
            if config.latent_positions else None
        )
        self.positions = nn.Parameter(torch.randn(1, self.grid**2, config.visual_dim) * 0.02)
        self.times = nn.Parameter(torch.randn(1, config.future_frames, 1, config.visual_dim) * 0.02)
        self.net = decoder(config.visual_dim, config.visual_heads, config.visual_layers)
        self.checkpoint_layers = config.gradient_checkpointing
        self.pixels = nn.Linear(config.visual_dim, 3 * config.patch_size**2)

    def forward(self, current: Tensor, latents: Tensor) -> Tensor:
        batch = latents.shape[0]
        context = self.context(current)
        source_grid = int(context.shape[1]**0.5)
        if source_grid**2 != context.shape[1]:
            raise ValueError("Expected square context patches")
        if source_grid != self.grid:
            context = context.transpose(1, 2).reshape(batch, -1, source_grid, source_grid)
            context = F.interpolate(context, (self.grid, self.grid), mode="bilinear", align_corners=False)
            context = context.flatten(2).transpose(1, 2)
        queries = context[:, None] + self.positions[:, None] + self.times
        time = self.config.future_frames
        memory = self.latents(latents)
        if self.latent_positions is not None:
            memory = memory + self.latent_positions
        memory = memory[:, None].expand(-1, time, -1, -1).flatten(0, 1)
        values = self.pixels(decode(self.net, queries.flatten(0, 1), memory,
                                   self.checkpoint_layers)).sigmoid()
        patch = self.config.patch_size
        values = values.reshape(batch, time, self.grid, self.grid, 3, patch, patch)
        return values.permute(0, 1, 4, 2, 5, 3, 6).reshape(
            batch, time, 3, self.config.image_size, self.config.image_size
        )


class ActionEncoder(nn.Module):
    def __init__(self, config: VITAConfig):
        super().__init__()
        self.config = config
        self.dct = TemporalDCT(config.action_horizon)
        self.net = nn.Sequential(
            nn.Linear(config.action_horizon * config.action_dim, config.action_hidden_dim),
            nn.GELU(),
            nn.Linear(config.action_hidden_dim, config.token_count * config.latent_dim),
        )
        self.latent_norm = nn.LayerNorm(config.latent_dim) if config.action_latent_norm else nn.Identity()

    def forward(self, actions: Tensor) -> Tensor:
        c = self.config
        if actions.shape[1:] != (c.action_horizon, c.action_dim):
            raise ValueError("Action shape does not match the configured horizon/dimension")
        latents = self.net(self.dct(actions).flatten(1)).reshape(-1, c.token_count, c.latent_dim)
        return self.latent_norm(latents)


class ActionDecoder(nn.Module):
    def __init__(self, config: VITAConfig):
        super().__init__()
        self.project = nn.Linear(config.latent_dim, config.action_hidden_dim)
        self.latent_positions = (
            nn.Parameter(torch.randn(1, config.token_count, config.action_hidden_dim) * 0.02)
            if config.latent_positions else None
        )
        self.queries = nn.Parameter(torch.randn(1, config.action_horizon, config.action_hidden_dim) * 0.02)
        self.net = decoder(config.action_hidden_dim, config.action_heads, config.action_layers)
        self.checkpoint_layers = config.gradient_checkpointing
        self.coefficients = nn.Linear(config.action_hidden_dim, config.action_dim)
        self.dct = TemporalDCT(config.action_horizon)
        self.output_mlp = (nn.Sequential(nn.Linear(config.action_dim, config.action_hidden_dim), nn.GELU(),
                                         nn.Linear(config.action_hidden_dim, config.action_dim))
                           if config.action_decode_order == "project_idct_mlp" else nn.Identity())

    def forward(self, latents: Tensor) -> Tensor:
        memory = self.project(latents)
        if self.latent_positions is not None:
            memory = memory + self.latent_positions
        coefficients = self.coefficients(decode(
            self.net, self.queries.expand(len(latents), -1, -1), memory, self.checkpoint_layers))
        return self.output_mlp(self.dct.inverse(coefficients))


class FutureFrameEncoder(nn.Module):
    """Explicit visual CoT: actions depend on decoded pixels, not shared codes."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch = nn.Conv2d(3, config.latent_dim, config.patch_size, config.patch_size)
        count = config.future_frames * config.num_views * (config.image_size // config.patch_size)**2
        self.positions = nn.Parameter(torch.randn(1, count, config.latent_dim) * 0.02)
        self.queries = nn.Parameter(torch.randn(1, config.token_count, config.latent_dim) * 0.02)
        self.net = decoder(config.latent_dim, config.mformer_heads, config.mformer_layers)

    def forward(self, images, view_mask=None):
        if images.ndim == 5:
            images = images[:, :, None]
        batch, time, views = images.shape[:3]
        if (time, views) != (self.config.future_frames, self.config.num_views):
            raise ValueError("Future frame sequence does not match explicit visual CoT")
        if view_mask is None:
            view_mask = torch.ones(batch, views, dtype=torch.bool, device=images.device)
        if view_mask.shape != (batch, views) or view_mask.dtype != torch.bool or not view_mask.any(1).all():
            raise ValueError("Invalid future-image view mask")
        memory = self.patch(images.flatten(0, 2)).flatten(2).transpose(1, 2)
        patches = memory.shape[1]
        memory = memory.reshape(batch, time * views * patches, -1) + self.positions
        valid = view_mask[:, None, :, None].expand(-1, time, -1, patches).reshape(batch, -1)
        return self.net(self.queries.expand(batch, -1, -1), memory, memory_key_padding_mask=~valid)

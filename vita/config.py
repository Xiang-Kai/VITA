from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path


COT_VARIANTS = ("none", "textual", "visual", "textual_visual", "internal", "textual_internal")


def validate_numeric_fields(config, *, allow_zero=()):
    """Validate declared types, not supplied types: bool is not an integer count."""
    for field in fields(config):
        value = getattr(config, field.name)
        if field.type == "bool":
            if type(value) is not bool:
                raise ValueError(f"{field.name} must be a boolean")
            continue
        if field.type not in {"int", "float"}:
            continue
        if field.type == "int":
            valid = type(value) is int
        else:
            valid = type(value) in {int, float}
            if valid:
                try:
                    valid = math.isfinite(value)
                except OverflowError:
                    valid = False
        if not valid:
            expected = "an integer" if field.type == "int" else "a finite number"
            raise ValueError(f"{field.name} must be {expected}")
        if value < 0 or (value == 0 and field.name not in allow_zero):
            bound = "nonnegative" if field.name in allow_zero else "positive"
            raise ValueError(f"{field.name} must be {bound}")


@dataclass(frozen=True)
class VITAConfig:
    image_size: int = 32
    patch_size: int = 8
    feature_dim: int = 32
    latent_dim: int = 32
    codebook_size: int = 64
    future_frames: int = 2
    action_horizon: int = 4
    action_dim: int = 7
    state_dim: int = 8
    subtask_slots: int = 2
    text_length: int = 48
    policy_dim: int = 64
    policy_layers: int = 2
    policy_heads: int = 4
    visual_dim: int = 32
    visual_layers: int = 1
    visual_heads: int = 4
    action_hidden_dim: int = 64
    action_layers: int = 2
    action_heads: int = 4
    mformer_layers: int = 1
    mformer_heads: int = 4
    commitment_weight: float = 0.25
    visual_l1_weight: float = 1.0
    ssim_weight: float = 1.0
    action_weight: float = 1.0
    temperature: float = 1.0
    codebook_update: str = "visual_only"
    vision_backend: str = "tiny"
    dinov2_path: str | None = None
    policy_backend: str = "tiny"
    paligemma_path: str | None = None
    num_views: int = 1
    latent_positions: bool = True
    auxiliary_token_weight: float = 0.0
    action_latent_norm: bool = False
    gradient_checkpointing: bool = False
    feature_batch_size: int = 0
    paligemma_resolution: str = "strict"
    cot_variant: str = "textual_internal"
    modality_alignment: str = "shared"
    latent_tokens: int = 0
    mformer_memory_tokens: int = 0
    subtask_embedding: str = "learned"
    finetune_scope: str = "action_only"
    action_decode_order: str = "project_idct"

    def __post_init__(self):
        validate_numeric_fields(self, allow_zero={
            "auxiliary_token_weight", "feature_batch_size", "latent_tokens", "mformer_memory_tokens",
        })
        if self.image_size % self.patch_size:
            raise ValueError("image_size must be divisible by decoder patch_size")
        for width, heads in (
            (self.policy_dim, self.policy_heads),
            (self.visual_dim, self.visual_heads),
            (self.action_hidden_dim, self.action_heads),
            (self.latent_dim, self.mformer_heads),
        ):
            if width % heads:
                raise ValueError("Transformer width must be divisible by head count")
        if self.codebook_update not in {"visual_only", "both"}:
            raise ValueError("codebook_update must be visual_only or both")
        if self.vision_backend not in {"tiny", "dinov2"}:
            raise ValueError("Unsupported vision backend")
        if self.policy_backend not in {"tiny", "paligemma"}:
            raise ValueError("Unsupported policy backend")
        if self.vision_backend == "dinov2" and not self.dinov2_path:
            raise ValueError("DINOv2 requires an explicit local checkpoint")
        if self.policy_backend == "paligemma" and not self.paligemma_path:
            raise ValueError("PaliGemma requires an explicit local checkpoint")
        if self.paligemma_resolution not in {"strict", "interpolate_positions", "resize_input"}:
            raise ValueError("Unknown PaliGemma resolution adaptation")
        if self.cot_variant not in COT_VARIANTS:
            raise ValueError("Unknown CoT variant")
        if self.modality_alignment not in {"shared", "separate"}:
            raise ValueError("Unknown modality alignment")
        if self.modality_alignment == "separate" and not self.internal_cot:
            raise ValueError("Separate codebooks require an internal CoT variant")
        if self.subtask_embedding not in {"learned", "tokenizer"}:
            raise ValueError("Unknown subtask embedding")
        if self.auxiliary_token_weight and not self.internal_cot:
            raise ValueError("Token CE requires internal CoT")
        if self.finetune_scope not in {"action_only", "joint"}:
            raise ValueError("Unknown fine-tuning scope")
        if self.action_decode_order not in {"project_idct", "project_idct_mlp"}:
            raise ValueError("Unknown action decoding order")

    @property
    def token_count(self):
        return self.latent_tokens or self.future_frames

    @property
    def textual_cot(self):
        return self.cot_variant in {"textual", "textual_visual", "textual_internal"}

    @property
    def internal_cot(self):
        return self.cot_variant in {"internal", "textual_internal"}

    @property
    def explicit_visual_cot(self):
        return self.cot_variant in {"visual", "textual_visual"}

    @property
    def predicts_images(self):
        return self.internal_cot or self.explicit_visual_cot

    @classmethod
    def load(cls, path: str | Path) -> VITAConfig:
        return cls(**json.loads(Path(path).read_text()))

    def to_dict(self) -> dict:
        return asdict(self)

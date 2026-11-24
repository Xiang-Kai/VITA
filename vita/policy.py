from __future__ import annotations

import math
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .config import VITAConfig
from .math import progressive_mask, straight_through_lookup
from .modules import encoder


SUBTASKS = ("GRASP", "LIFT", "MOVE", "PLACE", "ROTATE", "OPEN", "CLOSE", "END_SUBTASK")


def views_and_mask(images: Tensor, mask: Tensor | None, expected: int) -> tuple[Tensor, Tensor]:
    if images.ndim == 4:
        images = images[:, None]
    if images.ndim != 5 or images.shape[1] != expected:
        raise ValueError("Expected [B,V,3,H,W] with the configured number of views")
    if mask is None:
        mask = torch.ones(images.shape[:2], dtype=torch.bool, device=images.device)
    if mask.shape != images.shape[:2] or mask.dtype != torch.bool or not mask.any(-1).all():
        raise ValueError("view_mask must be boolean [B,V] with at least one valid view per sample")
    images = images.masked_fill(~mask[:, :, None, None, None], 0)
    return images, mask


class TinyContext(nn.Module):
    """Byte text + patch images, deliberately NOT a pretrained VLM."""

    def __init__(self, config: VITAConfig):
        super().__init__()
        self.config = config
        self.width = config.policy_dim
        self.text = nn.Embedding(257, self.width, padding_idx=0)
        self.vision = nn.Conv2d(3, self.width, config.patch_size, config.patch_size)
        self.view_embeddings = nn.Parameter(torch.randn(1, config.num_views, 1, self.width) * 0.02)
        self.net = encoder(self.width, config.policy_heads, config.policy_layers)
        max_length = config.text_length + config.num_views * (config.image_size // config.patch_size)**2 + 1
        max_length += config.subtask_slots + config.token_count + 1
        self.positions = nn.Parameter(torch.randn(1, max_length, self.width) * 0.02)

    def context(self, instructions: list[str], images: Tensor, view_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        images, view_mask = views_and_mask(images, view_mask, self.config.num_views)
        ids = torch.zeros(len(instructions), self.config.text_length, device=images.device, dtype=torch.long)
        for index, text in enumerate(instructions):
            values = [value + 1 for value in text.encode("utf-8")[:self.config.text_length]]
            if not values:
                raise ValueError("Instructions must not be empty")
            ids[index, :len(values)] = torch.tensor(values, device=ids.device)
        visual = self.vision(images.flatten(0, 1)).flatten(2).transpose(1, 2)
        visual = visual.unflatten(0, images.shape[:2]) + self.view_embeddings
        visual_valid = view_mask[:, :, None].expand(-1, -1, visual.shape[2]).flatten(1)
        visual = visual.flatten(1, 2)
        context = torch.cat([self.text(ids), visual], dim=1)
        valid = torch.cat([ids.ne(0), visual_valid], dim=1)
        return context, valid

    def forward(self, values: Tensor, mask: Tensor, valid: Tensor) -> Tensor:
        values = values + self.positions[:, :values.shape[1]]
        if self.config.gradient_checkpointing and self.training and torch.is_grad_enabled():
            for layer in self.net.layers:
                values = checkpoint(layer, values, src_mask=mask,
                                    src_key_padding_mask=~valid, use_reentrant=False)
            return self.net.norm(values)
        return self.net(values, mask=mask, src_key_padding_mask=~valid)

    def symbolic_embeddings(self):
        return torch.stack([self.text.weight[
            torch.tensor([v + 1 for v in word.encode("utf-8")], device=self.text.weight.device)
        ].mean(0) for word in SUBTASKS])


class PaliGemmaContext(nn.Module):
    """Local-only pretrained adapter with explicit group attention, no hub fallback."""

    def __init__(self, config: VITAConfig):
        super().__init__()
        from transformers import AutoTokenizer, PaliGemmaForConditionalGeneration

        foundation, loading = PaliGemmaForConditionalGeneration.from_pretrained(
            config.paligemma_path, local_files_only=True, attn_implementation="eager",
            output_loading_info=True,
        )
        if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError(f"Incomplete PaliGemma checkpoint: {loading}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.paligemma_path, local_files_only=True)
        # Older Transformers wrap Gemma in a causal-LM head; newer versions expose it directly.
        backbone = getattr(foundation, "model", foundation)
        language = backbone.language_model
        self.language = language if hasattr(language, "embed_tokens") else language.model
        self.vision = backbone.vision_tower
        self.projector = backbone.multi_modal_projector
        self.width = foundation.config.text_config.hidden_size
        self.text_length = config.text_length
        self.image_size = foundation.config.vision_config.image_size
        self.num_views = config.num_views
        self.resolution = config.paligemma_resolution
        self.view_embeddings = nn.Parameter(torch.randn(1, config.num_views, 1, self.width) * 0.02)
        if config.image_size != self.image_size and self.resolution == "strict":
            raise ValueError(
                f"Checkpoint resolution is {self.image_size}, not {config.image_size}. "
                "Resolve the paper/checkpoint resolution discrepancy explicitly."
            )
        if config.gradient_checkpointing:
            self.language.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.vision.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def context(self, instructions: list[str], images: Tensor, view_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        images, view_mask = views_and_mask(images, view_mask, self.num_views)
        tokens = self.tokenizer(
            instructions, return_tensors="pt", padding="max_length",
            truncation=True, max_length=self.text_length,
        ).to(images.device)
        text = self.language.embed_tokens(tokens.input_ids)
        pixels = images.flatten(0, 1) * 2 - 1
        if self.resolution == "resize_input" and pixels.shape[-1] != self.image_size:
            pixels = F.interpolate(pixels, self.image_size, mode="bicubic", align_corners=False, antialias=True)
        vision = self.vision(pixel_values=pixels,
                             interpolate_pos_encoding=self.resolution == "interpolate_positions").last_hidden_state
        vision = self.projector(vision) / math.sqrt(self.width)
        vision = vision.unflatten(0, images.shape[:2]) + self.view_embeddings
        vision_valid = view_mask[:, :, None].expand(-1, -1, vision.shape[2]).flatten(1)
        vision = vision.flatten(1, 2)
        valid = torch.cat([
            tokens.attention_mask.bool(),
            vision_valid,
        ], dim=1)
        return torch.cat([text, vision], dim=1), valid

    def forward(self, values: Tensor, mask: Tensor, valid: Tensor) -> Tensor:
        blocked = mask[None, None] | ~valid[:, None, None, :]
        additive = torch.zeros(blocked.shape, device=values.device, dtype=values.dtype)
        additive.masked_fill_(blocked, torch.finfo(values.dtype).min)
        return self.language(
            inputs_embeds=values, attention_mask=additive,
            use_cache=False, return_dict=True,
        ).last_hidden_state

    def symbolic_embeddings(self):
        tokens = self.tokenizer(list(SUBTASKS), return_tensors="pt", padding=True,
                                add_special_tokens=False).to(self.language.embed_tokens.weight.device)
        embeddings = self.language.embed_tokens(tokens.input_ids)
        mask = tokens.attention_mask.to(embeddings.dtype)
        return (embeddings * mask[:, :, None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)


class ProgressivePolicy(nn.Module):
    def __init__(self, config: VITAConfig):
        super().__init__()
        self.config = config
        self.trunk = TinyContext(config) if config.policy_backend == "tiny" else PaliGemmaContext(config)
        width = self.trunk.width
        self.state = nn.Linear(config.state_dim, width)
        self.subtask_queries = nn.Parameter(torch.randn(1, config.subtask_slots, width) * 0.02)
        self.subtask_embeddings = nn.Embedding(len(SUBTASKS), width)
        self.subtask_head = nn.Linear(width, len(SUBTASKS))
        self.code_to_policy = nn.Linear(config.latent_dim, width)
        self.code_head = nn.Linear(width, config.codebook_size)
        self.bos = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        if not config.internal_cot:
            self.output_queries = nn.Parameter(torch.randn(1, config.token_count, width) * 0.02)
            self.continuous_head = nn.Linear(width, config.latent_dim)
        if config.modality_alignment == "separate":
            self.action_code_head = nn.Linear(width, config.codebook_size)
            self.action_code_to_policy = nn.Linear(config.latent_dim, width)
            self.action_bos = nn.Parameter(torch.randn(1, 1, width) * 0.02)

    def forward(self, instructions: list[str], images: Tensor, states: Tensor, codebook: Tensor,
                view_mask: Tensor | None = None, action_codebook: Tensor | None = None) -> dict[str, Tensor]:
        if len(instructions) != len(images) or states.shape != (len(images), self.config.state_dim):
            raise ValueError("Instruction/image/state batch mismatch")
        context, valid = self.trunk.context(instructions, images, view_mask)
        context = torch.cat([context, self.state(states)[:, None]], dim=1)
        valid = torch.cat([valid, torch.ones(len(images), 1, device=images.device, dtype=torch.bool)], dim=1)
        context_length = context.shape[1]
        batch = len(images)
        slots = self.config.subtask_slots if self.config.textual_cot else 0
        prefix_valid = torch.cat([valid, valid.new_ones(batch, slots)], dim=1)
        if slots:
            values = torch.cat([context, self.subtask_queries.expand(batch, -1, -1)], dim=1)
            hidden = self.trunk(
                values, progressive_mask(context_length, slots, 0, device=images.device), prefix_valid,
            )
            subtask_logits = self.subtask_head(hidden[:, -slots:])
            weights = (self.trunk.symbolic_embeddings() if self.config.subtask_embedding == "tokenizer"
                       else self.subtask_embeddings.weight)
            subtask_values, subtask_ids = straight_through_lookup(
                subtask_logits, weights, self.config.temperature,
            )
        else:
            subtask_values = context[:, :0]
            subtask_ids = torch.empty(batch, 0, device=images.device, dtype=torch.long)
            subtask_logits = context.new_empty(batch, 0, len(SUBTASKS))
        prefix = torch.cat([context, subtask_values], dim=1)
        result = {"subtask_ids": subtask_ids, "subtask_logits": subtask_logits}
        if not self.config.internal_cot:
            # Parallel continuous readout, with no discrete internal reasoning stream.
            values = torch.cat([prefix, self.output_queries.expand(batch, -1, -1)], dim=1)
            active = torch.cat([prefix_valid, valid.new_ones(batch, self.config.token_count)], dim=1)
            hidden = self.trunk(values, progressive_mask(context_length, slots, self.config.token_count,
                                                       device=images.device), active)
            return {**result, "latents": self.continuous_head(hidden[:, -self.config.token_count:]),
                    "token_ids": subtask_ids[:, :0],
                    "token_logits": context.new_empty(batch, 0, self.config.codebook_size)}
        visual = self.generate_codes(prefix, prefix_valid, context_length, slots, codebook,
                                     self.bos, self.code_head, self.code_to_policy)
        if self.config.modality_alignment == "separate":
            if action_codebook is None:
                raise ValueError("Decoupled policy needs a separate action codebook")
            action = self.generate_codes(prefix, prefix_valid, context_length, slots, action_codebook,
                                         self.action_bos, self.action_code_head, self.action_code_to_policy)
            return {**result, **action, "visual_latents": visual["latents"],
                    "visual_token_ids": visual["token_ids"], "visual_token_logits": visual["token_logits"]}
        return {**result, **visual}

    def generate_codes(self, prefix, prefix_valid, context_length, slots, codebook, bos, head, project):
        batch = len(prefix)
        previous = bos.expand(batch, -1, -1)
        generated, ids, logits_list = [], [], []
        for _ in range(self.config.token_count):
            values = torch.cat([prefix, previous], dim=1)
            mask = progressive_mask(context_length, slots, previous.shape[1], device=prefix.device)
            active = torch.cat([prefix_valid, prefix_valid.new_ones(batch, previous.shape[1])], dim=1)
            hidden = self.trunk(values, mask, active)
            logits = head(hidden[:, -1])
            latent, index = straight_through_lookup(logits, codebook, self.config.temperature)
            generated.append(latent)
            ids.append(index)
            logits_list.append(logits)
            previous = torch.cat([previous, project(latent)[:, None]], dim=1)
        return {
            "latents": torch.stack(generated, dim=1),
            "token_ids": torch.stack(ids, dim=1),
            "token_logits": torch.stack(logits_list, dim=1),
        }

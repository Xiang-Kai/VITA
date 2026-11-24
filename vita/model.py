from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import VITAConfig
from .math import ssim_loss
from .modules import (ActionDecoder, ActionEncoder, FrozenImageEncoder, FutureFrameEncoder,
                      MotionFormer, VisualDecoder)
from .policy import ProgressivePolicy, views_and_mask
from .quantizer import SharedVectorQuantizer


def masked_mse(prediction: Tensor, target: Tensor, mask: Tensor | None = None) -> Tensor:
    if mask is None:
        return F.mse_loss(prediction, target)
    if mask.shape != target.shape or mask.dtype != torch.bool or not mask.flatten(1).any(1).all():
        raise ValueError("action_mask must be boolean, match actions and have valid values in each sample")
    return (prediction.float() - target.float()).square()[mask].mean()


class VITA(nn.Module):
    def __init__(self, config: VITAConfig):
        super().__init__()
        self.config = config
        self.visual_encoder = FrozenImageEncoder(config)
        self.motion_encoder = MotionFormer(config, self.visual_encoder.feature_dim)
        self.visual_decoder = VisualDecoder(config, self.visual_encoder.feature_dim)
        self.action_encoder = ActionEncoder(config)
        self.action_decoder = ActionDecoder(config)
        self.quantizer = SharedVectorQuantizer(config.codebook_size, config.latent_dim, config.commitment_weight)
        self.policy = ProgressivePolicy(config)
        if config.modality_alignment == "separate":
            self.action_quantizer = SharedVectorQuantizer(
                config.codebook_size, config.latent_dim, config.commitment_weight)
        if config.explicit_visual_cot:
            self.future_encoder = FutureFrameEncoder(config)
        self.stage = ""
        self.set_stage("warmup")

    def set_stage(self, stage: str) -> None:
        if stage not in {"warmup", "cotrain", "finetune"}:
            raise ValueError(f"Unknown stage: {stage}")
        self.requires_grad_(False)
        trainable = {
            "warmup": ("motion_encoder", "visual_decoder", "action_encoder", "action_decoder", "quantizer"),
            "cotrain": ("policy", "visual_decoder", "action_decoder"),
            "finetune": ("action_decoder",),
        }
        if self.config.finetune_scope == "joint":
            trainable["finetune"] = trainable["cotrain"]
        for name in trainable[stage]:
            getattr(self, name).requires_grad_(True)
        if self.config.modality_alignment == "separate":
            self.action_quantizer.requires_grad_(stage == "warmup")
        if self.config.explicit_visual_cot:
            self.future_encoder.requires_grad_(stage in {"cotrain", "finetune"})
        # Clear gradients from an earlier stage even before a new optimizer is built.
        for parameter in self.parameters():
            parameter.grad = None
        self.stage = stage
        self.train(self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        for child in self.children():
            if not any(parameter.requires_grad for parameter in child.parameters()):
                child.eval()
        return self

    def visual_latents(self, current: Tensor, future: Tensor, view_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        self._check_images(current, future)
        if current.ndim == 4:
            current_features = self.visual_encoder(current)
            future_features = self.visual_encoder(future.flatten(0, 1)).unflatten(
                0, (len(current), self.config.future_frames)
            )
            return self.motion_encoder(current_features, future_features), current_features
        images, valid = views_and_mask(current, view_mask, self.config.num_views)
        batch, views = images.shape[:2]
        features = self.visual_encoder(images.flatten(0, 1))
        targets = future.transpose(1, 2).reshape(batch * views, self.config.future_frames, *future.shape[3:])
        future_features = self.visual_encoder(targets.flatten(0, 1)).unflatten(
            0, (batch * views, self.config.future_frames)
        )
        motion = self.motion_encoder(features, future_features).unflatten(0, (batch, views))
        latents = (motion * valid[:, :, None, None]).sum(1) / valid.sum(1)[:, None, None]
        return latents, features.unflatten(0, (batch, views))

    def decode_visual(self, features: Tensor, latents: Tensor) -> Tensor:
        if features.ndim == 3:
            return self.visual_decoder(features, latents)
        batch, views = features.shape[:2]
        copies = latents[:, None].expand(-1, views, -1, -1).flatten(0, 1)
        prediction = self.visual_decoder(features.flatten(0, 1), copies)
        return prediction.unflatten(0, (batch, views)).transpose(1, 2)

    def predict_images(self, current, latents, view_mask=None):
        if current.ndim == 5:
            images, _ = views_and_mask(current, view_mask, self.config.num_views)
            features = self.visual_encoder(images.flatten(0, 1)).unflatten(0, images.shape[:2])
        else:
            features = self.visual_encoder(current)
        return self.decode_visual(features, latents)

    @property
    def action_codebook(self):
        return self.action_quantizer if self.config.modality_alignment == "separate" else self.quantizer

    @staticmethod
    def valid_visual_pairs(prediction: Tensor, target: Tensor, view_mask: Tensor | None) -> tuple[Tensor, Tensor]:
        if prediction.ndim == 5:
            return prediction, target
        prediction = prediction.transpose(1, 2).flatten(0, 1)
        target = target.transpose(1, 2).flatten(0, 1)
        if view_mask is not None:
            prediction, target = prediction[view_mask.flatten()], target[view_mask.flatten()]
        return prediction, target

    def warmup_video(self, current: Tensor, future: Tensor, view_mask: Tensor | None = None) -> dict[str, Tensor]:
        if self.stage != "warmup":
            raise RuntimeError("warmup_video requires warmup stage")
        latents, current_features = self.visual_latents(current, future, view_mask)
        quantized = self.quantizer(latents, update_codebook=True)
        prediction = self.decode_visual(current_features, quantized.values)
        valid_prediction, valid_target = self.valid_visual_pairs(prediction, future, view_mask)
        l1 = F.l1_loss(valid_prediction, valid_target)
        structural = ssim_loss(valid_prediction, valid_target)
        loss = self.config.visual_l1_weight * l1 + self.config.ssim_weight * structural + quantized.loss
        return {"loss": loss, "visual_l1": l1, "ssim": structural, "vq": quantized.loss,
                "perplexity": quantized.perplexity, "prediction": prediction}

    def warmup_action(self, actions: Tensor, action_mask: Tensor | None = None) -> dict[str, Tensor]:
        if self.stage != "warmup":
            raise RuntimeError("warmup_action requires warmup stage")
        if action_mask is not None:
            masked_mse(actions, actions, action_mask)
            actions = actions.masked_fill(~action_mask, 0)
        latents = self.action_encoder(actions)
        quantized = self.action_codebook(latents, update_codebook=(
            self.config.codebook_update == "both" or self.config.modality_alignment == "separate"))
        prediction = self.action_decoder(quantized.values)
        mse = masked_mse(prediction, actions, action_mask)
        return {"loss": mse + quantized.loss, "action_mse": mse, "vq": quantized.loss,
                "perplexity": quantized.perplexity, "prediction": prediction}

    def generate(self, batch: dict) -> dict[str, Tensor]:
        # Only these three inputs cross the policy boundary; targets cannot leak.
        states = batch["states"]
        if "state_mask" in batch:
            mask = batch["state_mask"]
            if mask.shape != states.shape or mask.dtype != torch.bool:
                raise ValueError("state_mask must be boolean and match states")
            states = states.masked_fill(~mask, 0)
        return self.policy(batch["instructions"], batch["current"], states,
                           self.quantizer.embedding.weight.detach(), batch.get("view_mask"),
                           self.action_codebook.embedding.weight.detach())

    def cotrain(self, batch: dict) -> dict[str, Tensor]:
        if self.stage != "cotrain" and not (self.stage == "finetune" and self.config.finetune_scope == "joint"):
            raise RuntimeError("cotrain requires cotrain stage")
        if not self.config.predicts_images:
            if "actions" not in batch:
                raise ValueError("Without visual reasoning, co-training requires paired action targets")
            generated = self.generate(batch)
            actions = self.action_decoder(generated["latents"])
            mse = masked_mse(actions, batch["actions"], batch.get("action_mask"))
            return {"loss": self.config.action_weight * mse, "action_mse": mse,
                    "actions": actions, **generated}
        self._check_images(batch["current"], batch["future"])
        generated = self.generate(batch)
        latents = generated["latents"]
        current = batch["current"]
        prediction = self.predict_images(current, generated.get("visual_latents", latents), batch.get("view_mask"))
        valid_prediction, valid_target = self.valid_visual_pairs(prediction, batch["future"], batch.get("view_mask"))
        l1 = F.l1_loss(valid_prediction, valid_target)
        result = {"visual_l1": l1, **generated}
        if "actions" in batch:
            if self.config.explicit_visual_cot:
                latents = self.future_encoder(prediction, batch.get("view_mask"))
            actions = self.action_decoder(latents)
            mse = masked_mse(actions, batch["actions"], batch.get("action_mask"))
            loss = self.config.visual_l1_weight * l1 + self.config.action_weight * mse
            result.update(action_mse=mse, actions=actions)
        else:
            structural = ssim_loss(valid_prediction, valid_target)
            loss = self.config.visual_l1_weight * l1 + self.config.ssim_weight * structural
            result["ssim"] = structural
        if self.config.auxiliary_token_weight:
            # Optional engineering extension: reconstruction targets supervise logits, never inputs.
            with torch.no_grad():
                if "actions" in batch:
                    target_actions = batch["actions"]
                    if "action_mask" in batch:
                        target_actions = target_actions.masked_fill(~batch["action_mask"], 0)
                    teacher = self.action_encoder(target_actions)
                else:
                    teacher, _ = self.visual_latents(current, batch["future"], batch.get("view_mask"))
                quantizer = self.action_codebook if "actions" in batch else self.quantizer
                targets = quantizer.nearest(teacher, quantizer.embedding.weight)
            logits = (generated["token_logits"] if "actions" in batch else
                      generated.get("visual_token_logits", generated["token_logits"]))
            token_loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
            loss = loss + self.config.auxiliary_token_weight * token_loss
            result["auxiliary_token_ce"] = token_loss
        result.update(loss=loss, images=prediction)
        return result

    def finetune(self, batch: dict) -> dict[str, Tensor]:
        if self.stage != "finetune":
            raise RuntimeError("finetune requires finetune stage")
        if self.config.finetune_scope == "joint":
            return self.cotrain(batch)
        with torch.no_grad():
            generated = self.generate(batch)
            if self.config.explicit_visual_cot:
                images = self.predict_images(batch["current"], generated["latents"], batch.get("view_mask"))
        latents = (self.future_encoder(images, batch.get("view_mask")) if self.config.explicit_visual_cot
                   else generated["latents"])
        actions = self.action_decoder(latents)
        mse = masked_mse(actions, batch["actions"], batch.get("action_mask"))
        return {"loss": mse, "action_mse": mse, "actions": actions, **generated}

    @torch.no_grad()
    def act(self, instructions: list[str], current: Tensor, states: Tensor, view_mask: Tensor | None = None) -> Tensor:
        generated = self.generate({"instructions": instructions, "current": current,
                                   "states": states, "view_mask": view_mask})
        latents = generated["latents"]
        if self.config.explicit_visual_cot:
            latents = self.future_encoder(self.predict_images(current, latents, view_mask), view_mask)
        return self.action_decoder(latents)

    def _check_images(self, current: Tensor, future: Tensor) -> None:
        c = self.config
        current_shape = (len(current), 3, c.image_size, c.image_size)
        future_shape = (len(current), c.future_frames, 3, c.image_size, c.image_size)
        if current.ndim == 5:
            current_shape = (len(current), c.num_views, 3, c.image_size, c.image_size)
            future_shape = (len(current), c.future_frames, c.num_views, 3, c.image_size, c.image_size)
        elif c.num_views != 1:
            raise ValueError("Multi-view model requires explicit view dimension")
        if current.shape != current_shape:
            raise ValueError("Invalid current image shape")
        if future.shape != future_shape:
            raise ValueError("Invalid future image shape")

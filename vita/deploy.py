"""Action-only deployment: omit DINO, motion/action encoders and visual decoder."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch
from torch import nn

from .config import VITAConfig
from .modules import ActionDecoder, FrozenImageEncoder, FutureFrameEncoder, VisualDecoder
from .model import VITA
from .policy import ProgressivePolicy
from .train import atomic_checkpoint, core_hashes
from .runtime import require_checkpoint_source


ACTION_PREFIXES = ("policy.", "action_decoder.", "quantizer.embedding.")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def action_prefixes(config):
    prefixes = ACTION_PREFIXES
    if config.modality_alignment == "separate":
        prefixes += ("action_quantizer.embedding.",)
    if config.explicit_visual_cot:
        prefixes += ("visual_encoder.", "visual_decoder.", "future_encoder.")
    return prefixes


def data_contract(checkpoint, source_name=None):
    manifest = checkpoint.get("data_manifest")
    if not manifest or manifest.get("kind") != "mixture":
        if source_name is not None:
            raise ValueError("data_source is only valid for a mixture checkpoint")
        return manifest or {"normalization": checkpoint["normalization"], "cameras": checkpoint["cameras"]}
    candidates = [s for s in manifest["sources"] if s["mode"] == "paired" and
                  (s["name"] == source_name if source_name is not None else "finetune" in s["weights"])]
    if len(candidates) != 1:
        raise ValueError("Specify exactly one paired deployment data source; normalization cannot be guessed")
    return candidates[0]["manifest"]


class ActionPolicy(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.policy = ProgressivePolicy(config)
        self.action_decoder = ActionDecoder(config)
        self.quantizer = nn.Module()
        self.quantizer.embedding = nn.Embedding(config.codebook_size, config.latent_dim)
        if config.modality_alignment == "separate":
            self.action_quantizer = nn.Module()
            self.action_quantizer.embedding = nn.Embedding(config.codebook_size, config.latent_dim)
        if config.explicit_visual_cot:
            self.visual_encoder = FrozenImageEncoder(config)
            self.visual_decoder = VisualDecoder(config, self.visual_encoder.feature_dim)
            self.future_encoder = FutureFrameEncoder(config)

    decode_visual = VITA.decode_visual
    predict_images = VITA.predict_images

    @torch.no_grad()
    def act(self, instructions, current, states, view_mask=None):
        action_book = getattr(self, "action_quantizer", self.quantizer).embedding.weight
        generated = self.policy(instructions, current, states, self.quantizer.embedding.weight, view_mask, action_book)
        latents = generated["latents"]
        if self.config.explicit_visual_cot:
            latents = self.future_encoder(self.predict_images(current, latents, view_mask), view_mask)
        return self.action_decoder(latents)


def load_action_policy(path, device="cpu", *, require_source_match=False):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("schema_version") not in {2, 3}:
        raise ValueError("Expected a training or action-only checkpoint")
    if saved.get("schema_version") == 3 and saved.get("kind") != "vita_action_policy":
        raise ValueError("Unknown deployment checkpoint type")
    if require_source_match:
        require_checkpoint_source(saved, core_hashes())
    model = ActionPolicy(VITAConfig(**saved["model_config"]))
    selected = {k: v for k, v in saved["model"].items() if k.startswith(action_prefixes(model.config))}
    model.load_state_dict(selected, strict=True)
    model.requires_grad_(False).eval().to(device)
    metadata = {k: v for k, v in saved.items() if k not in {
        "model", "optimizer", "rng_cpu", "rng_cuda", "rng_python", "sampler"}}
    return model, metadata


def export_action_policy(source, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    saved = torch.load(source, map_location="cpu", weights_only=True)
    if saved.get("schema_version") != 2 or saved.get("stage") != "finetune":
        raise ValueError("Export requires a fine-tuned schema-v2 training checkpoint")
    config = VITAConfig(**saved["model_config"])
    weights = {k: v for k, v in saved["model"].items() if k.startswith(action_prefixes(config))}
    model = ActionPolicy(config)
    model.load_state_dict(weights, strict=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {k: saved.get(k) for k in
                ("model_config", "normalization", "cameras", "data_manifest", "manifest_sha256", "source_sha256")}
    atomic_checkpoint(output, {
        **metadata, "schema_version": 3, "kind": "vita_action_policy", "model": weights,
        "stage": "finetune", "source_checkpoint": str(Path(source).resolve()),
        "source_checkpoint_sha256": file_sha256(source),
        "benchmark_reproduced": False,
        "deployment_dependencies": (
            "Explicit visual ablation retains visual reconstruction and its foundation dependency"
            if config.explicit_visual_cot else
            "PaliGemma config/tokenizer/checkpoint path still required when selected; DINO not required"),
    })
    return {"output": str(output.resolve()), "retained_parameters": sum(v.numel() for v in weights.values()),
            "original_state_elements": sum(v.numel() for v in saved["model"].values()),
            "bytes": output.stat().st_size, "benchmark_reproduced": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(export_action_policy(args.checkpoint, args.output))


if __name__ == "__main__":
    main()

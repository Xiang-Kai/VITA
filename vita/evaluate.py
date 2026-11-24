"""Offline split evaluation; deliberately distinct from simulation success."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .config import VITAConfig
from .dataset import atomic_json
from .deploy import file_sha256
from .mixture import load_data_manifest, make_dataset
from .model import VITA
from .train import TrainConfig, core_hashes, evaluate, prepare_device
from .runtime import dependency_versions, require_checkpoint_source


def load_policy(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema_version") != 2:
        raise ValueError("Expected a schema-v2 training checkpoint")
    require_checkpoint_source(checkpoint, core_hashes())
    model = VITA(VITAConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.set_stage(checkpoint["stage"])
    model.eval()
    metadata = {key: value for key, value in checkpoint.items() if key not in {
        "model", "optimizer", "sampler", "rng_cpu", "rng_cuda", "rng_python",
    }}
    return model, metadata


def main():
    parser = argparse.ArgumentParser(description="Offline reconstruction / policy MSE, NOT task success")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="validation")
    parser.add_argument("--samples", type=int, help="Omit to evaluate every window of the split")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--memory-gib", type=float, default=4.0, help="CUDA allocator cap; allow room for the model")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    if args.memory_gib <= 0:
        raise ValueError("memory-gib must be positive")
    prepare_device(args.device, args.memory_gib)
    model, checkpoint = load_policy(args.checkpoint, args.device)
    manifest = load_data_manifest(args.manifest)
    if manifest["manifest_sha256"] != checkpoint["manifest_sha256"]:
        raise ValueError("Evaluation data identity differs from the checkpoint")
    dataset = make_dataset(manifest, args.split)
    if args.samples is not None and args.samples < 1:
        raise ValueError("samples must be positive")
    try:
        result = evaluate(model, dataset, TrainConfig(**checkpoint["train_config"]), args.device,
                          limit=args.samples or len(dataset))
        result.update(checkpoint=str(Path(args.checkpoint).resolve()), status="offline_evaluation_completed",
                      checkpoint_sha256=file_sha256(args.checkpoint), benchmark_reproduced=False,
                      manifest_sha256=manifest["manifest_sha256"],
                      runtime_source_sha256=core_hashes(), dependency_versions=dependency_versions())
        output.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(output, result)
        print(result)
    finally:
        dataset.close()


if __name__ == "__main__":
    main()

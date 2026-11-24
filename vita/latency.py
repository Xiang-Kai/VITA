"""Synchronized wall-clock profiling of existing, unmodified policy checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import torch

from .dataset import atomic_json
from .deploy import load_action_policy
from .train import prepare_device
from .runtime import machine_metadata


@torch.no_grad()
def profile_policy(model, *, device="cpu", warmup=5, repeats=30, batch_size=1):
    if min(warmup, repeats, batch_size) < 1:
        raise ValueError("Need positive warmup, repeats and batch size")
    config = model.config
    dtype = next(model.parameters()).dtype
    generator = torch.Generator(device=device).manual_seed(17)
    images = torch.rand(batch_size, config.num_views, 3, config.image_size, config.image_size,
                        generator=generator, device=device, dtype=dtype)
    states = torch.zeros(batch_size, config.state_dim, device=device, dtype=dtype)
    instructions = ["move the synthetic profiling object"] * batch_size
    model.eval()

    def sync():
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)

    for _ in range(warmup):
        model.act(instructions, images, states)
    sync()
    measured = {"backbone_ms": [], "action_decoder_ms": [], "end_to_end_ms": []}
    handles, starts = [], {}

    def pre(name):
        def hook(_m, _args):
            sync()
            starts[name] = time.perf_counter_ns()
        return hook

    def post(name):
        def hook(_m, _args, _output):
            sync()
            measured[name].append((time.perf_counter_ns() - starts[name]) / 1e6)
        return hook

    for module, name in ((model.policy, "backbone_ms"), (model.action_decoder, "action_decoder_ms")):
        handles.extend([module.register_forward_pre_hook(pre(name)), module.register_forward_hook(post(name))])
    try:
        for _ in range(repeats):
            sync()
            start = time.perf_counter_ns()
            actions = model.act(instructions, images, states)
            sync()
            measured["end_to_end_ms"].append((time.perf_counter_ns() - start) / 1e6)
            if actions.shape != (batch_size, config.action_horizon, config.action_dim) or not torch.isfinite(actions).all():
                raise ValueError("Invalid policy output during latency measurement")
    finally:
        for handle in handles:
            handle.remove()
    summaries = {key: {"mean": float(np.mean(values)), "median": float(np.median(values)),
                       "p95": float(np.percentile(values, 95)), "samples": values}
                 for key, values in measured.items()}
    return {
        "measurements_ms": summaries, "warmup": warmup, "repeats": repeats, "batch_size": batch_size,
        "future_frames": config.future_frames, "latent_tokens": config.token_count,
        "action_horizon": config.action_horizon, "cot_variant": config.cot_variant,
        "predicted_actions_per_second": batch_size * config.action_horizon * 1000 / summaries["end_to_end_ms"]["mean"],
        "chunk_calls_per_second": batch_size * 1000 / summaries["end_to_end_ms"]["mean"],
        **machine_metadata(), "platform": platform.platform(), "torch": str(torch.__version__),
        "device": device, "hardware": torch.cuda.get_device_name(device) if str(device).startswith("cuda") else platform.processor(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "dtype": str(dtype),
        "torch_threads": torch.get_num_threads(), "synthetic_inputs": True,
        "scope": "synchronized inference incl. tokenization; excludes camera/robot I/O and action execution",
        "timing_note": "Hooks introduce measurement overhead; total includes explicit visual path when selected.",
        "benchmark_reproduced": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", required=True, help="Future-horizon=checkpoint, e.g. 12=run12.pt")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--memory-gib", type=float, default=6)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    prepare_device(args.device, args.memory_gib)
    results = {}
    for item in args.checkpoints:
        horizon, path = item.split("=", 1)
        if horizon in results:
            raise ValueError("Duplicate horizon")
        model, _ = load_action_policy(path, args.device)
        if model.config.future_frames != int(horizon):
            raise ValueError("Checkpoint future horizon mismatch; no decoder or token resizing is allowed")
        result = profile_policy(model, device=args.device, warmup=args.warmup, repeats=args.repeats)
        digest = hashlib.sha256()
        with open(path, "rb") as stream:
            for block in iter(lambda: stream.read(2**20), b""):
                digest.update(block)
        result.update(checkpoint=str(Path(path).resolve()), checkpoint_sha256=digest.hexdigest())
        results[horizon] = result
        del model
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, {"horizons": results, "paper_horizons_present": set(results) == {"12", "24", "48"}})


if __name__ == "__main__":
    main()

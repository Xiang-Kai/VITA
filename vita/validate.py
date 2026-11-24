from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import torch

from .config import VITAConfig
from .data import load_libero_windows, synthetic_batch, tensor_fingerprint
from .model import VITA
from .runtime import check_device_scope, machine_metadata


def module_hashes(model: VITA) -> dict[str, str]:
    result = {}
    for name, module in model.named_children():
        digest = hashlib.sha256()
        for key, value in module.state_dict().items():
            digest.update(key.encode())
            digest.update(value.detach().float().cpu().contiguous().numpy().tobytes())
        result[name] = digest.hexdigest()
    return result


def source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent.parent
    files = sorted(root.glob("vita/*.py")) + sorted(root.glob("tests/*.py"))
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def scalars(result: dict) -> dict:
    return {name: float(value.detach()) for name, value in result.items()
            if isinstance(value, torch.Tensor) and value.numel() == 1}


def validate(args) -> dict:
    if args.steps < 1 or args.batch_size < 1:
        raise ValueError("Steps and batch size must be positive")
    check_device_scope(args.device)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        total_memory = torch.cuda.get_device_properties(args.device).total_memory
        torch.cuda.set_per_process_memory_fraction(min(0.10, 2 * 2**30 / total_memory), device=args.device)
        torch.cuda.reset_peak_memory_stats(args.device)
    config = VITAConfig.load(args.config) if args.config else VITAConfig()
    if args.libero_hdf5:
        batch, normalizer, data_record = load_libero_windows(
            args.libero_hdf5, config, batch_size=args.batch_size, frame_stride=args.frame_stride,
        )
    else:
        batch = synthetic_batch(config, args.batch_size, args.seed)
        normalizer = None
        data_record = {"kind": "synthetic_only", "sample_sha256": tensor_fingerprint(batch)}
    batch = {key: value.to(args.device) if isinstance(value, torch.Tensor) else value
             for key, value in batch.items()}
    model = VITA(config).to(args.device)
    run = {
        "status": "running", "benchmark_reproduced": False,
        "purpose": "bounded numerical and training-window validation; no physical actuation",
        "runtime_clock_asia_shanghai": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        **machine_metadata(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device": args.device, "torch": torch.__version__, "python": platform.python_version(),
        "seed": args.seed, "config": config.to_dict(), "data": data_record,
        "validation_training": {
            "steps_per_branch": args.steps, "batch_size": args.batch_size,
            "learning_rate": args.learning_rate, "optimizer": "AdamW",
            "betas": [0.9, 0.95], "weight_decay": 0.01, "gradient_clip": 1.0,
            "precision": "float32", "autocast": False,
            "gpu_allocator_cap_mib": 2048 if args.device.startswith("cuda") else None,
        },
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "source_sha256": source_hashes(), "stages": {},
    }
    started = time.monotonic()
    for stage in ("warmup", "cotrain", "finetune"):
        model.set_stage(stage)
        model.train()
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.01,
        )
        before = module_hashes(model)

        def evaluate():
            if stage == "warmup":
                return {"video": scalars(model.warmup_video(batch["current"], batch["future"])),
                        "action": scalars(model.warmup_action(batch["actions"]))}
            return scalars(getattr(model, stage)(batch))

        with torch.no_grad():
            initial = evaluate()
        history = []
        for step in range(args.steps):
            branches = ("video", "action") if stage == "warmup" else (stage,)
            for branch in branches:
                optimizer.zero_grad(set_to_none=True)
                if branch == "video":
                    result = model.warmup_video(batch["current"], batch["future"])
                elif branch == "action":
                    result = model.warmup_action(batch["actions"])
                else:
                    result = getattr(model, stage)(batch)
                if not torch.isfinite(result["loss"]):
                    raise FloatingPointError(f"Nonfinite loss at {stage}/{branch}/{step}")
                result["loss"].backward()
                for name, parameter in model.named_parameters():
                    if not parameter.requires_grad and parameter.grad is not None:
                        raise AssertionError(f"Frozen parameter has a gradient: {name}")
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                        raise FloatingPointError(f"Nonfinite gradient: {name}")
                if stage == "cotrain" and step == 0:
                    for name in ("code_head", "subtask_head"):
                        grad = getattr(model.policy, name).weight.grad
                        if grad is None or grad.abs().sum() == 0:
                            raise AssertionError(f"No gradient reaches policy.{name}")
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                history.append({"step": step, "branch": branch, **scalars(result)})
        with torch.no_grad():
            final = evaluate()
        after = module_hashes(model)
        changed = [name for name in before if before[name] != after[name]]
        allowed = {
            "warmup": {"motion_encoder", "visual_decoder", "action_encoder", "action_decoder", "quantizer"},
            "cotrain": {"policy", "visual_decoder", "action_decoder"},
            "finetune": {"action_decoder"},
        }[stage]
        if not set(changed).issubset(allowed):
            raise AssertionError(f"Frozen module changed in {stage}: {changed}")
        record = {"initial": initial, "final": final, "optimizer_steps": len(history),
                  "changed_modules": changed, "frozen_modules_unchanged": True, "history": history}
        run["stages"][stage] = record
        print(json.dumps({"stage": stage, "initial": initial, "final": final, "changed": changed}), flush=True)
    model.eval()
    output = model.act(batch["instructions"], batch["current"], batch["states"])
    if not torch.isfinite(output).all():
        raise FloatingPointError("Nonfinite action output")
    normalized_mse = float(torch.mean((output - batch["actions"])**2))
    run["inference"] = {"shape": list(output.shape), "finite": True, "normalized_training_mse": normalized_mse}
    with torch.no_grad():
        generated = model.generate(batch)
    shared_chunk = batch["actions"].mean(dim=0, keepdim=True)
    mean_chunk_mse = float((shared_chunk - batch["actions"]).square().mean())
    token_ids = generated["token_ids"]
    run["learning_diagnostics"] = {
        "unique_token_ids": int(token_ids.unique().numel()),
        "unique_sequences_in_batch": int(token_ids.unique(dim=0).shape[0]),
        "token_ids": token_ids.tolist(),
        "constant_per_timestep_training_baseline_mse": mean_chunk_mse,
        "beats_constant_training_baseline": normalized_mse < mean_chunk_mse,
        "caveat": "Training-set diagnostics only; neither generalization nor task success.",
    }
    if normalizer:
        raw_mse = (normalizer.inverse(output) - normalizer.inverse(batch["actions"])).square().mean()
        run["inference"]["raw_training_mse"] = float(raw_mse)
    run["elapsed_seconds"] = time.monotonic() - started
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    torch.save({
        "config": config.to_dict(), "model": model.state_dict(), "stage": model.stage,
        "normalization": normalizer.to_dict() if normalizer else None,
        "data": data_record, "benchmark_reproduced": False,
    }, destination / "checkpoint.pt")
    restored = VITA(config).to(args.device)
    checkpoint = torch.load(destination / "checkpoint.pt", map_location=args.device, weights_only=True)
    restored.load_state_dict(checkpoint["model"])
    restored.set_stage(checkpoint["stage"])
    restored.eval()
    reloaded = restored.act(batch["instructions"], batch["current"], batch["states"])
    torch.testing.assert_close(output, reloaded, atol=0, rtol=0)
    run["checkpoint_roundtrip_exact"] = True
    if args.device.startswith("cuda"):
        run["gpu_peak_allocated_mib"] = torch.cuda.max_memory_allocated(args.device) / 2**20
        run["gpu_peak_reserved_mib"] = torch.cuda.max_memory_reserved(args.device) / 2**20
    run["status"] = "numerical_validation_passed"
    (destination / "metrics.json").write_text(json.dumps(run, indent=2) + "\n")
    print(json.dumps({"status": run["status"], "elapsed_seconds": run["elapsed_seconds"],
                      "metrics": str(destination / "metrics.json")}), flush=True)
    return run


def main():
    parser = argparse.ArgumentParser(description="VITA minimal validation, not a benchmark evaluation")
    parser.add_argument("--config")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--libero-hdf5")
    parser.add_argument("--frame-stride", type=int, default=20)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise FileExistsError(f"Refusing to mix validation runs in {destination}")
    try:
        validate(args)
    except Exception as error:
        (destination / "failure.json").write_text(json.dumps({
            "status": "failed", "type": type(error).__name__, "message": str(error),
            "benchmark_reproduced": False,
        }, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()

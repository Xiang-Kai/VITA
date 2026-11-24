"""Foreground, resumable single-device training with explicit offline evaluation."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from zoneinfo import ZoneInfo

import torch
from torch.utils.data import default_collate

from .config import VITAConfig, validate_numeric_fields
from .dataset import StatefulBatchSampler, atomic_json
from .mixture import MixtureDataset, MixtureBatchSampler, load_data_manifest, make_dataset
from .model import VITA, masked_mse
from .runtime import check_device_scope, machine_metadata
from .validate import module_hashes, scalars


@dataclass(frozen=True)
class TrainConfig:
    warmup_steps: int = 200
    cotrain_steps: int = 200
    finetune_steps: int = 100
    batch_size: int = 8
    accumulation_steps: int = 1
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    lr_schedule: str = "constant"
    precision: str = "float32"
    seed: int = 17
    eval_every: int = 100
    eval_samples: int = 64
    checkpoint_every: int = 100
    codebook_init_batches: int = 0
    memory_gib: float = 4.0

    def __post_init__(self):
        validate_numeric_fields(self, allow_zero={
            "codebook_init_batches", "seed", "weight_decay",
            "warmup_steps", "cotrain_steps", "finetune_steps",
        })
        if self.total_steps < 1:
            raise ValueError("At least one training stage must contain updates")
        if self.precision not in {"float32", "bfloat16"} or self.lr_schedule not in {"constant", "cosine"}:
            raise ValueError("Unknown precision or LR schedule")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))

    @property
    def total_steps(self):
        return self.warmup_steps + self.cotrain_steps + self.finetune_steps

    def stage_at(self, step):
        if step < self.warmup_steps:
            return "warmup", step, self.warmup_steps
        if step < self.warmup_steps + self.cotrain_steps:
            return "cotrain", step - self.warmup_steps, self.cotrain_steps
        return "finetune", step - self.warmup_steps - self.cotrain_steps, self.finetune_steps


def core_hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob("*.py"))}


def prepare_device(device: str, memory_gib: float):
    torch.set_num_threads(2)
    check_device_scope(device)
    if device.startswith("cuda"):
        available, total = torch.cuda.mem_get_info(device)
        cap = int(memory_gib * 2**30)
        if available < cap + 2 * 2**30:
            raise RuntimeError("Insufficient free memory for the requested allocator cap plus safety margin")
        torch.cuda.set_per_process_memory_fraction(cap / total, device)
        torch.cuda.reset_peak_memory_stats(device)


def batch_from(dataset, indices, device):
    batch = default_collate([dataset[i] for i in indices])
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def autocast(training: TrainConfig, device: str):
    if training.precision == "float32":
        return nullcontext()
    return torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16)


def training_objective(model, batch, stage_step=0):
    if model.stage != "warmup":
        return getattr(model, model.stage)(batch)
    if "current" not in batch or ("actions" in batch and stage_step % 2):
        return model.warmup_action(batch["actions"], batch.get("action_mask"))
    return model.warmup_video(batch["current"], batch["future"], batch.get("view_mask"))


@torch.no_grad()
def evaluate_source(model, dataset, training, device, *, mode, limit=None):
    """Per-source metrics; never average incompatible embodiment units together."""
    was_training = model.training
    model.eval()
    count = min(len(dataset), limit if limit is not None else training.eval_samples)
    indices = torch.linspace(0, len(dataset) - 1, count).long().tolist()
    totals, denominators = {}, {}
    token_counts = torch.zeros(model.config.codebook_size, dtype=torch.long)
    sequences = set()
    for offset in range(0, count, training.batch_size):
        batch = batch_from(dataset, indices[offset:offset + training.batch_size], device)
        if mode == "video":
            batch.pop("actions", None)
            batch.pop("action_mask", None)
        with autocast(training, device):
            if model.stage == "warmup":
                result = {}
                if mode != "action":
                    video = model.warmup_video(batch["current"], batch["future"], batch.get("view_mask"))
                    result.update(visual_l1=video["visual_l1"], ssim=video["ssim"])
                if mode != "video":
                    action = model.warmup_action(batch["actions"], batch.get("action_mask"))
                    result["action_mse"] = action["action_mse"]
                    prediction = action["prediction"]
            else:
                result = getattr(model, model.stage)(batch)
                prediction = result.get("actions")
        values = scalars(result)
        if "token_ids" in result:
            ids = result["token_ids"].detach().cpu()
            token_counts += torch.bincount(ids.flatten(), minlength=len(token_counts))
            if ids.numel():
                sequences.update(tuple(row) for row in ids.tolist())
        samples = len(batch["sample_id"])
        if mode != "video":
            values["train_mean_baseline_mse"] = float(masked_mse(
                torch.zeros_like(batch["actions"]), batch["actions"], batch.get("action_mask")))
            values["raw_action_mse"] = float(masked_mse(
                dataset.normalizer.inverse(prediction.float()),
                dataset.normalizer.inverse(batch["actions"]), batch.get("action_mask")))
        for key, value in values.items():
            if not math.isfinite(value):
                raise FloatingPointError("Nonfinite evaluation metric")
            # Masked MSE reduces over valid scalar controls, not over padded samples.
            weight = (int(batch["action_mask"].sum()) if "action_mask" in batch
                      else batch["actions"].numel()) if key in {
                          "action_mse", "raw_action_mse", "train_mean_baseline_mse"} else samples
            totals[key] = totals.get(key, 0.0) + value * weight
            denominators[key] = denominators.get(key, 0) + weight
    model.train(was_training)
    result = {"split": dataset.split, "mode": mode, "samples": count,
            "metrics": {k: v / denominators[k] for k, v in totals.items()},
            "action_source": ("not_applicable" if mode == "video" else
                              "action_autoencoder_with_targets" if model.stage == "warmup"
                              else "observation_only_policy"), "benchmark_success_rate": None}
    if token_counts.sum():
        probabilities = token_counts.double() / token_counts.sum()
        result.update(occupied_codes=int((token_counts > 0).sum()), distinct_sequences=len(sequences),
                      code_perplexity=float((-(probabilities * probabilities.clamp_min(1e-20).log()).sum()).exp()))
    return result


@torch.no_grad()
def evaluate(model, dataset, training, device, *, limit=None):
    if isinstance(dataset, MixtureDataset):
        results = {}
        for source, child in zip(dataset.manifest["sources"], dataset.sources):
            if model.stage in source["weights"]:
                results[source["name"]] = evaluate_source(
                    model, child, training, device, mode=source["mode"], limit=limit)
        return {"split": dataset.split, "per_source": results, "benchmark_success_rate": None,
                "scope": "source-specific offline metrics, not a mixed-unit success metric"}
    if dataset.manifest.get("kind") == "canonical_episodes":
        return evaluate_source(model, dataset, training, device, mode=dataset.manifest["mode"], limit=limit)
    was_training = model.training
    model.eval()
    count = min(len(dataset), limit if limit is not None else training.eval_samples)
    # Fixed, evenly spaced windows give task coverage without consuming the training sampler/RNG.
    indices = torch.linspace(0, len(dataset) - 1, count).long().tolist()
    totals, samples = {}, 0
    token_counts = torch.zeros(model.config.codebook_size, dtype=torch.long)
    sequences = set()
    normalizer = dataset.normalizer
    for offset in range(0, count, training.batch_size):
        batch = batch_from(dataset, indices[offset:offset + training.batch_size], device)
        with autocast(training, device):
            if model.stage == "warmup":
                video = model.warmup_video(batch["current"], batch["future"], batch["view_mask"])
                action = model.warmup_action(batch["actions"])
                result = {"loss": video["loss"] + action["loss"],
                          "visual_l1": video["visual_l1"], "action_mse": action["action_mse"]}
                latents, _ = model.visual_latents(batch["current"], batch["future"], batch["view_mask"])
                ids = model.quantizer.nearest(latents, model.quantizer.embedding.weight)
                predicted = action["prediction"]
            else:
                result = getattr(model, model.stage)(batch)
                ids = result["token_ids"]
                predicted = result["actions"]
        batch_size = len(batch["actions"])
        values = scalars(result)
        values["raw_action_mse"] = float(
            (normalizer.inverse(predicted.float()) - normalizer.inverse(batch["actions"])).square().mean()
        )
        # Zero normalized actions are the constant mean fitted ONLY on training demonstrations.
        values["train_mean_baseline_mse"] = float(batch["actions"].square().mean())
        if not all(math.isfinite(value) for value in values.values()):
            raise FloatingPointError("Nonfinite evaluation metric")
        for key, value in values.items():
            totals[key] = totals.get(key, 0.0) + value * batch_size
        samples += batch_size
        ids = ids.detach().cpu()
        token_counts += torch.bincount(ids.flatten(), minlength=len(token_counts))
        if ids.numel():
            sequences.update(tuple(row) for row in ids.tolist())
    probabilities = token_counts.double() / token_counts.sum().clamp_min(1)
    perplexity = (float((-(probabilities * probabilities.clamp_min(1e-20).log()).sum()).exp())
                  if token_counts.sum() else None)
    model.train(was_training)
    means = {key: value / samples for key, value in totals.items()}
    return {"split": dataset.split, "samples": samples, "total_available": len(dataset),
            "metrics": means, "code_perplexity": perplexity,
            "occupied_codes": int((token_counts > 0).sum()), "distinct_sequences": len(sequences),
            "beats_train_mean": means["action_mse"] < means["train_mean_baseline_mse"],
            "action_source": "action_autoencoder_with_targets" if model.stage == "warmup" else "observation_only_policy",
            "benchmark_success_rate": None}


def atomic_checkpoint(path, state):
    temporary = path.with_name(path.name + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def initialize_components(model, prior, selection):
    """Table 7 transfers, without optimizer/RNG restoration or silent shape filtering."""
    choices = {
        "all": None,
        "codebook": ("quantizer.", "action_quantizer."),
        "codebook_visual": ("quantizer.", "action_quantizer.", "visual_decoder."),
        "codebook_decoders": ("quantizer.", "action_quantizer.", "visual_decoder.", "action_decoder."),
    }
    if selection not in choices:
        raise ValueError("Unknown initialization component selection")
    if prior.get("schema_version") != 2:
        raise ValueError("Stage initialization requires a schema-v2 checkpoint")
    old = VITAConfig(**prior["model_config"])
    allowed = {"gradient_checkpointing", "feature_batch_size", "dinov2_path", "paligemma_path"}
    if any(getattr(old, key) != value for key, value in model.config.to_dict().items() if key not in allowed):
        raise ValueError("Stage initialization architecture/semantics must match")
    if selection != "all" and prior.get("stage") != "warmup":
        raise ValueError("Selective initialization requires a warmup checkpoint")
    prefixes = choices[selection]
    target = model.state_dict()
    keys = [k for k in target if prefixes is None or k.startswith(prefixes)]
    for key in keys:
        if key not in prior["model"] or prior["model"][key].shape != target[key].shape:
            raise ValueError(f"Missing or incompatible transferred parameter: {key}")
    target.update({key: prior["model"][key] for key in keys})
    model.load_state_dict(target, strict=True)
    return sorted({key.split(".")[0] for key in keys})


def run_training(model_config: VITAConfig, training: TrainConfig, manifest_path, output, *,
                 device="cpu", resume=None, max_updates=None, initialize_from=None,
                 initialize_components_selection="all") -> dict:
    if resume and initialize_from:
        raise ValueError("Exact resume and weight-only stage initialization are mutually exclusive")
    if initialize_components_selection != "all" and not initialize_from:
        raise ValueError("Selective initialization needs --initialize-from")
    prepare_device(device, training.memory_gib)
    manifest = load_data_manifest(manifest_path)
    for key in ("future_frames", "action_horizon", "action_dim", "state_dim", "image_size"):
        if manifest[key] != getattr(model_config, key):
            raise ValueError(f"Model and manifest disagree on {key}")
    if manifest.get("num_views", len(manifest.get("cameras", []))) != model_config.num_views:
        raise ValueError("Camera count differs from model")
    output = Path(output)
    if output.exists() and any(output.iterdir()) and resume is None:
        raise FileExistsError("Refusing to overwrite a nonempty training run")
    output.mkdir(parents=True, exist_ok=True)
    if max_updates is not None and max_updates < 1:
        raise ValueError("max_updates must be positive")
    torch.manual_seed(training.seed)
    random.seed(training.seed)
    model = VITA(model_config).to(device)
    dataset = make_dataset(manifest, "train")
    validation = make_dataset(manifest, "validation")
    sampler = (MixtureBatchSampler(dataset, training.batch_size, training.seed)
               if isinstance(dataset, MixtureDataset)
               else StatefulBatchSampler(len(dataset), training.batch_size, training.seed))
    if isinstance(dataset, MixtureDataset):
        if not model_config.predicts_images and training.cotrain_steps and any(
                "cotrain" in source["weights"] and source["mode"] == "video" for source in manifest["sources"]):
            raise ValueError("No-visual CoT ablations require a paired-only co-training mixture")
        for stage in ("warmup", "cotrain", "finetune"):
            if getattr(training, f"{stage}_steps") and not any(
                stage in source["weights"] for source in manifest["sources"]
            ):
                raise ValueError(f"No eligible mixture sources for {stage}")
    elif manifest.get("kind") == "canonical_episodes":
        from .mixture import ALLOWED
        for stage in ALLOWED:
            if getattr(training, f"{stage}_steps") and manifest["mode"] not in ALLOWED[stage]:
                raise ValueError(f"Source modality is not eligible for {stage}")
    hashes = core_hashes()
    state = None
    start = 0
    initialization = {"method": "random", "engineering_extension": False}
    if initialize_from:
        from .deploy import file_sha256

        if training.codebook_init_batches:
            raise ValueError("Do not reinitialize a transferred codebook")
        prior = torch.load(initialize_from, map_location="cpu", weights_only=True)
        transferred = initialize_components(model, prior, initialize_components_selection)
        initialization = {"method": "weight_only_stage_transfer", "source": str(Path(initialize_from).resolve()),
                          "source_checkpoint_sha256": file_sha256(initialize_from),
                          "source_stage": prior["stage"], "source_manifest_sha256": prior["manifest_sha256"],
                          "selection": initialize_components_selection, "transferred_modules": transferred,
                          "optimizer_restored": False, "engineering_extension": False}
        del prior
    if resume:
        state = torch.load(resume, map_location="cpu", weights_only=True)
        if state["schema_version"] != 2:
            raise ValueError("Only production schema-v2 checkpoints can resume optimizer state")
        for key, actual in (("model_config", model_config.to_dict()), ("train_config", asdict(training)),
                            ("manifest_sha256", manifest["manifest_sha256"]), ("source_sha256", hashes)):
            if state[key] != actual:
                raise ValueError(f"Resume mismatch: {key}")
        model.load_state_dict(state["model"])
        sampler.load_state_dict(state["sampler"])
        start = state["next_step"]
        initialization = state["initialization"]
        torch.set_rng_state(state["rng_cpu"])
        random.setstate(state["rng_python"])
        if device.startswith("cuda") and state["rng_cuda"] is not None:
            torch.cuda.set_rng_state_all(state["rng_cuda"])
    elif not initialize_from and training.codebook_init_batches:
        if isinstance(dataset, MixtureDataset) or manifest.get("mode") == "action":
            raise ValueError("Optional visual k-means initialization requires a single video/paired source")
        points = []
        with torch.no_grad():
            for _ in range(training.codebook_init_batches):
                batch = batch_from(dataset, sampler.next(), device)
                latent, _ = model.visual_latents(batch["current"], batch["future"], batch["view_mask"])
                points.append(latent.detach())
        initialization = model.quantizer.initialize_from_data(torch.cat(points), seed=training.seed)
        initialization["engineering_extension"] = True

    optimizer = None
    last_stage = None
    stage_frozen = {}
    history = []
    next_step = start
    start_time = time.monotonic()
    end = min(training.total_steps, start + max_updates) if max_updates else training.total_steps
    if start >= end:
        raise ValueError("No updates remain")
    atomic_json(output / "run_config.json", {
        "model": model_config.to_dict(), "training": asdict(training), "manifest_sha256": manifest["manifest_sha256"],
        **machine_metadata(), "device": device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "started_asia_shanghai": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "initialization": initialization, "resume": str(resume) if resume else None,
        "source_sha256": hashes, "benchmark_reproduced": False,
        "foundation_initialization": {"vision_backend": model_config.vision_backend,
                                      "vision_checkpoint": model_config.dinov2_path,
                                      "policy_backend": model_config.policy_backend,
                                      "policy_checkpoint": model_config.paligemma_path},
    })

    def save_checkpoint(name):
        atomic_checkpoint(output / name, {
            "schema_version": 2, "model_config": model_config.to_dict(), "train_config": asdict(training),
            "manifest_sha256": manifest["manifest_sha256"], "source_sha256": hashes,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "stage": model.stage,
            "next_step": next_step, "sampler": sampler.state_dict(),
            "rng_cpu": torch.get_rng_state(), "rng_python": random.getstate(),
            "rng_cuda": torch.cuda.get_rng_state_all() if device.startswith("cuda") else None,
            "normalization": manifest.get("normalization"), "cameras": manifest.get("cameras"),
            "data_manifest": manifest if manifest.get("kind") in {"mixture", "canonical_episodes"} else None,
            "initialization": initialization, "benchmark_reproduced": False,
        })

    try:
        for step in range(start, end):
            stage, stage_step, stage_length = training.stage_at(step)
            if stage != last_stage:
                model.set_stage(stage)
                optimizer = torch.optim.AdamW(
                    [p for p in model.parameters() if p.requires_grad], lr=training.learning_rate,
                    betas=(0.9, 0.95), weight_decay=training.weight_decay,
                )
                if state is not None and step == start and state["stage"] == stage:
                    optimizer.load_state_dict(state["optimizer"])
                # Model/RNG/sampler are already restored; do not retain duplicate CPU weights.
                state = None
                stage_frozen = {name: value for name, value in module_hashes(model).items()
                                if not any(p.requires_grad for p in getattr(model, name).parameters())}
                last_stage = stage
            model.train()
            lr_scale = 1.0 if training.lr_schedule == "constant" else 0.5 * (1 + math.cos(math.pi * stage_step / stage_length))
            for group in optimizer.param_groups:
                group["lr"] = training.learning_rate * lr_scale
            optimizer.zero_grad(set_to_none=True)
            train_metrics = {}
            source_batches = {}
            for _ in range(training.accumulation_steps):
                batch = batch_from(dataset, sampler.next(stage), device)
                if "source_name" in batch:
                    name = batch["source_name"][0]
                    source_batches[name] = source_batches.get(name, 0) + 1
                with autocast(training, device):
                    result = training_objective(model, batch, stage_step)
                    loss = result["loss"] / training.accumulation_steps
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss at update {step}")
                loss.backward()
                for key, value in scalars(result).items():
                    train_metrics[key] = train_metrics.get(key, 0.0) + value / training.accumulation_steps
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    if not parameter.requires_grad or not torch.isfinite(parameter.grad).all():
                        raise FloatingPointError(f"Invalid gradient: {name}")
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            next_step = step + 1
            boundary = stage_step + 1 == stage_length
            if next_step % training.eval_every == 0 or boundary or next_step == end:
                current_hashes = module_hashes(model)
                if any(current_hashes[name] != value for name, value in stage_frozen.items()):
                    raise AssertionError("A frozen module changed")
                record = {
                    "update": next_step, "stage": stage, "stage_update": stage_step + 1,
                    "gradient_norm_before_clip": float(norm), "lr": optimizer.param_groups[0]["lr"],
                    "training_batch": train_metrics, "validation": evaluate(model, validation, training, device),
                    "frozen_modules_unchanged": True, "sampler_epoch": sampler.epoch,
                    "source_microbatches_this_update": source_batches,
                    "source_microbatches_cumulative": getattr(sampler, "draws", None),
                }
                history.append(record)
                with (output / "history.jsonl").open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
                print(json.dumps(record), flush=True)
            if next_step % training.checkpoint_every == 0 or boundary or next_step == end:
                save_checkpoint("latest.pt")
            if boundary:
                save_checkpoint(f"{stage}_final.pt")
        final = evaluate(model, validation, training, device)
        result = {
            "status": "training_schedule_completed" if next_step == training.total_steps else "bounded_run_stopped",
            "next_step": next_step, "total_steps": training.total_steps, "last_stage": model.stage,
            "elapsed_seconds": time.monotonic() - start_time, "final_validation": final,
            "parameters": sum(p.numel() for p in model.parameters()), "initialization": initialization,
            "source_sha256": hashes, "manifest_sha256": manifest["manifest_sha256"],
            "benchmark_reproduced": False, "test_split_evaluated": False,
            "initialization_scope": "codebook/stage-transfer; foundation initialization is in run_config.json",
        }
        if device.startswith("cuda"):
            result["gpu_peak_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 2**20
            result["gpu_peak_reserved_mib"] = torch.cuda.max_memory_reserved(device) / 2**20
        atomic_json(output / "summary.json", result)
        return result
    except Exception as error:
        atomic_json(output / "failure.json", {
            "status": "failed", "next_step": next_step, "error": type(error).__name__,
            "message": str(error), "benchmark_reproduced": False,
        })
        raise
    finally:
        dataset.close()
        validation.close()


def main():
    parser = argparse.ArgumentParser(description="Resumable three-stage VITA training, not a success-rate evaluator")
    parser.add_argument("--config", required=True)
    parser.add_argument("--training", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume")
    parser.add_argument("--initialize-from", help="Weights only, for a new stage/dataset; not exact resume")
    parser.add_argument("--initialize-components", default="all",
                        choices=["all", "codebook", "codebook_visual", "codebook_decoders"])
    parser.add_argument("--max-updates", type=int)
    args = parser.parse_args()
    result = run_training(VITAConfig.load(args.config), TrainConfig.load(args.training), args.manifest,
                          args.output, device=args.device, resume=args.resume, max_updates=args.max_updates,
                          initialize_from=args.initialize_from,
                          initialize_components_selection=args.initialize_components)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

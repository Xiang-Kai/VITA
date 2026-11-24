"""Generate and run explicit VITA ablation jobs; generation never launches training."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

from .config import COT_VARIANTS, VITAConfig
from .dataset import atomic_json
from .mixture import load_data_manifest, seal
from .subset import training_subset
from .train import TrainConfig, run_training


FINETUNING = {
    "calvin": (40000, 192, 1e-4, 1),
    "libero": (60000, 192, 1e-4, 1),
    "fractal": (50000, 128, 1e-4, 5),
    "bridge": (80000, 128, 1e-4, 5),
    "ur5e": (40000, 128, 5e-5, 15),
}


def paired_cotrain_manifest(manifest):
    result = json.loads(json.dumps(manifest))
    result.pop("manifest_sha256", None)
    if result.get("kind") == "mixture":
        for source in result["sources"]:
            if source["mode"] != "paired":
                source["weights"].pop("cotrain", None)
    elif result.get("mode") in {"video", "action"}:
        raise ValueError("CoT comparison requires paired co-training data")
    return seal(result)


def generate_suite(config, manifest_path, output, *, suite="all", benchmark="calvin", smoke=False):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if suite not in {"all", "cot", "initialization", "efficiency", "finetune"}:
        raise ValueError("Unknown suite")
    if benchmark not in FINETUNING:
        raise ValueError("Unknown fine-tuning protocol")
    manifest_path = Path(manifest_path).resolve()
    manifest = load_data_manifest(manifest_path)
    output.mkdir(parents=True)
    ft_steps, batch, lr, chunk = FINETUNING[benchmark]
    # Accumulation preserves the declared effective batch without assuming a multi-node allocation.
    base = TrainConfig(warmup_steps=1 if smoke else 200000, cotrain_steps=1 if smoke else 100000,
                       finetune_steps=1 if smoke else ft_steps, batch_size=1,
                       accumulation_steps=1 if smoke else 1024,
                       learning_rate=1e-4, precision="float32" if smoke else "bfloat16",
                       eval_every=1 if smoke else 1000, eval_samples=2 if smoke else 64,
                       checkpoint_every=1 if smoke else 1000, memory_gib=4 if smoke else 64)
    jobs = {}

    def add(name, model_config, training, data=manifest_path, parent=None, selection="all"):
        atomic_json(output / f"{name}.model.json", model_config.to_dict())
        atomic_json(output / f"{name}.train.json", asdict(training))
        jobs[name] = {"model": f"{name}.model.json", "training": f"{name}.train.json",
                      "manifest": str(data), "depends_on": parent, "initialize_components": selection,
                      "output": f"runs/{name}"}

    finetuning = replace(base, warmup_steps=0, cotrain_steps=0,
                         accumulation_steps=1 if smoke else batch, learning_rate=lr)
    def add_staged(name, model_config, data=manifest_path, parent=None, selection="all", warmup=True):
        pretrain = name + "_pretrain"
        add(pretrain, model_config, replace(base, warmup_steps=base.warmup_steps if warmup else 0,
                                            finetune_steps=0), data, parent, selection)
        add(name, model_config, finetuning, data, pretrain)

    if suite in {"all", "cot"}:
        # Same paired supervision for all six variants, rather than changing their data budget.
        paired_path = output / "paired_cotrain.json"
        atomic_json(paired_path, paired_cotrain_manifest(manifest))
        for variant in COT_VARIANTS:
            add_staged(f"cot_{variant}", replace(config, cot_variant=variant, modality_alignment="shared",
                                                auxiliary_token_weight=0), paired_path)
    if suite in {"all", "initialization"}:
        add("initialization_warmup", config, replace(base, cotrain_steps=0, finetune_steps=0))
        for name, selection in (("random", None), ("codebook", "codebook"),
                                ("visual", "codebook_visual"), ("full", "codebook_decoders")):
            add_staged(f"initialization_{name}", config,
                       parent="initialization_warmup" if selection else None, selection=selection or "all",
                       warmup=False)
        add_staged("no_alignment", replace(config, modality_alignment="separate"), warmup=False)
    if suite in {"all", "efficiency"}:
        small_path = output / "ten_percent.json"
        atomic_json(small_path, training_subset(manifest, 0.1, base.seed))
        for warmed in (False, True):
            name = "warmup" if warmed else "no_warmup"
            parent = f"efficiency_{name}_pretrain"
            add(parent, config, replace(base, warmup_steps=base.warmup_steps if warmed else 0, finetune_steps=0))
            for budget in (500, 1000, 2000):
                add(f"efficiency_{name}_{budget}", config,
                    replace(base, warmup_steps=0, cotrain_steps=0, finetune_steps=1 if smoke else budget,
                            accumulation_steps=1 if smoke else batch, learning_rate=lr), parent=parent)
            for label, data in (("ten_percent", small_path), ("full_data", manifest_path)):
                add(f"efficiency_{name}_{label}", config,
                    replace(base, warmup_steps=0, cotrain_steps=0,
                            accumulation_steps=1 if smoke else batch, learning_rate=lr),
                    data=data, parent=parent)
    if suite == "finetune":
        add(f"finetune_{benchmark}", config,
            replace(base, warmup_steps=0, cotrain_steps=0,
                    accumulation_steps=1 if smoke else batch, learning_rate=lr))
    plan = {"schema_version": 1, "suite": suite, "benchmark": benchmark, "jobs": jobs,
            "smoke_only": smoke, "executed": False, "benchmark_reproduced": False,
            "executed_action_chunk": chunk, "seed": base.seed,
            "notes": ["VITA variants only; cited third-party baselines are not reimplemented.",
                      "Underspecified variant architectures are independent choices.",
                      "Fine-tune-only plans need --initialize-from a matching co-trained checkpoint.",
                      "Held-out task evaluation is separate from training completion."]}
    atomic_json(output / "suite.json", plan)
    return plan


def run_job(path, name, *, device="cpu", max_updates=None, initialize_from=None):
    path = Path(path).resolve()
    plan = json.loads(path.read_text())
    root, job = path.parent, plan["jobs"][name]
    parent = job["depends_on"]
    if parent:
        if initialize_from:
            raise ValueError("Cannot override a declared experiment dependency")
        prior = root / plan["jobs"][parent]["output"]
        summary = json.loads((prior / "summary.json").read_text())
        if summary["status"] != "training_schedule_completed":
            raise ValueError("Dependency did not finish its declared training schedule")
        initialize_from = prior / "latest.pt"
    if plan["suite"] == "finetune" and not initialize_from:
        raise ValueError("Fine-tune-only protocol requires a co-trained checkpoint")
    return run_training(VITAConfig.load(root / job["model"]), TrainConfig.load(root / job["training"]),
                        job["manifest"], root / job["output"], device=device, max_updates=max_updates,
                        initialize_from=initialize_from, initialize_components_selection=job["initialize_components"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate")
    generate.add_argument("--config", required=True)
    generate.add_argument("--manifest", required=True)
    generate.add_argument("--output", required=True)
    generate.add_argument("--suite", choices=["all", "cot", "initialization", "efficiency", "finetune"], default="all")
    generate.add_argument("--benchmark", choices=FINETUNING, default="calvin")
    generate.add_argument("--smoke", action="store_true")
    run = sub.add_parser("run")
    run.add_argument("--plan", required=True)
    run.add_argument("--job", required=True)
    run.add_argument("--device", default="cpu")
    run.add_argument("--max-updates", type=int)
    run.add_argument("--initialize-from")
    args = parser.parse_args()
    if args.command == "generate":
        result = generate_suite(VITAConfig.load(args.config), args.manifest, args.output,
                                suite=args.suite, benchmark=args.benchmark, smoke=args.smoke)
        print(json.dumps({"jobs": list(result["jobs"]), "executed": False}))
    else:
        print(json.dumps(run_job(args.plan, args.job, device=args.device, max_updates=args.max_updates,
                                 initialize_from=args.initialize_from)))


if __name__ == "__main__":
    main()

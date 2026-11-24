"""Inspect LIBERO recipe budgets without creating a run or reading any data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Inspect the checked-out implementation, not a different installed revision.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vita.dataset import SUITES
from vita.libero_recipe import make_parser, training_configs


MICROBATCHES = (1, 2, 4, 8, 16, 32, 64)
TASKS_PER_SUITE = 10
STAGES = ("warmup", "cotrain", "finetune")


def describe_budget(microbatch=16, *, smoke=False, suites=SUITES):
    """Counts are configured work, never evidence of completed training."""
    if type(microbatch) is not int or microbatch not in MICROBATCHES:
        raise ValueError("Choose a supported microbatch: 1, 2, 4, 8, 16, 32 or 64")
    if (isinstance(suites, str) or not suites or
            len(set(suites)) != len(suites) or not set(suites) <= set(SUITES)):
        raise ValueError("Choose distinct supported LIBERO suites")
    defaults = make_parser().parse_args(["prepare", "--output", "unused"])
    pretrain, finetune = training_configs(microbatch, smoke=smoke, memory_gib=defaults.memory_gib)
    stages = []
    for name, training in zip(STAGES, (pretrain, pretrain, finetune)):
        updates = getattr(training, f"{name}_steps")
        stages.append({
            "name": name, "updates": updates,
            "effectiveBatch": training.batch_size * training.accumulation_steps,
            "accumulation": training.accumulation_steps,
            "microsteps": updates * training.accumulation_steps,
        })
    episodes = 1 if smoke else defaults.episodes_per_task
    max_tasks = 1 if smoke else defaults.max_tasks
    tasks = min(max_tasks or TASKS_PER_SUITE, TASKS_PER_SUITE)
    output = f"runs/libero-{'smoke' if smoke else 'paper-budget'}"
    if tuple(suites) != tuple(SUITES):
        output += "-" + "-".join(suite.removeprefix("libero_") for suite in suites)
    command = [
        "python -m vita.libero_recipe prepare \\",
        f"  --output {output}{' --smoke' if smoke else ''} \\",
        (f"  --microbatch {microbatch} --episodes-per-task {episodes} --max-tasks {max_tasks}"
         if smoke else f"  --microbatch {microbatch} --memory-gib {defaults.memory_gib:g}"),
    ]
    if tuple(suites) != tuple(SUITES):
        command[-1] += " \\"
        command.append("  --suites " + " ".join(suites))
    return {
        "mode": "smoke" if smoke else "paper",
        "microbatch": microbatch, "suites": list(suites),
        "memoryGiB": defaults.memory_gib,
        "updates": sum(stage["updates"] for stage in stages),
        "microsteps": sum(stage["microsteps"] for stage in stages),
        "stages": stages, "precision": pretrain.precision,
        "evaluation": {
            "tasksPerSuite": tasks, "episodesPerTask": episodes,
            "maxTasks": max_tasks, "trials": len(suites) * tasks * episodes,
        },
        "command": "\n".join(command),
    }


def web_contract():
    """Small generated data module; the website does not import Python."""
    return {
        "schemaVersion": 1, "microbatches": list(MICROBATCHES),
        "suites": list(SUITES), "tasksPerSuite": TASKS_PER_SUITE,
        "plans": {mode: describe_budget(1, smoke=mode == "smoke")
                  for mode in ("smoke", "paper")},
    }


def verify_web_plans(plans):
    """Compare every browser choice with real Python configuration defaults."""
    expected = [
        describe_budget(batch, smoke=smoke, suites=suites)
        for smoke in (True, False) for batch in MICROBATCHES
        for suites in (SUITES, *((suite,) for suite in SUITES))
    ]
    if not isinstance(plans, list) or len(plans) != len(expected):
        raise ValueError("Expected all 70 website plan combinations")
    keys = expected[0].keys()
    fingerprint = lambda plan: json.dumps({key: plan.get(key) for key in keys}, sort_keys=True)
    remaining = {fingerprint(plan) for plan in expected}
    for index, plan in enumerate(plans):
        if not isinstance(plan, dict) or fingerprint(plan) not in remaining:
            raise ValueError(f"Website plan {index + 1} is invalid, stale or duplicated")
        remaining.remove(fingerprint(plan))
    return len(plans)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--microbatch", type=int, choices=MICROBATCHES, default=16)
    parser.add_argument("--suites", nargs="+", choices=SUITES, default=list(SUITES))
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--json", action="store_true", help="Machine-readable budget")
    output.add_argument("--export-web", action="store_true", help="Export all website budget defaults")
    output.add_argument("--check-web-plans", type=Path, help="Check the website's exported 70 plans, without executing them")
    args = parser.parse_args(argv)
    if args.export_web or args.check_web_plans:
        if args.smoke or args.microbatch != 16 or args.suites != list(SUITES):
            parser.error("Website export/check cannot be combined with a custom plan")
    if args.check_web_plans:
        try:
            count = verify_web_plans(json.loads(args.check_web_plans.read_text(encoding="utf-8")))
        except (OSError, ValueError) as error:
            parser.error(str(error))
        print(f"Verified {count} website plans against Python recipe defaults; no commands executed")
        return
    if args.export_web:
        print("// Generated by Xiang-Kai/VITA scripts/recipe_budget.py --export-web.")
        print("// Configuration only; no measured results or private runtime metadata.")
        print("export const recipeContract = " + json.dumps(web_contract(), indent=2) + ";")
        return
    try:
        plan = describe_budget(args.microbatch, smoke=args.smoke, suites=args.suites)
    except ValueError as error:
        parser.error(str(error))
    if args.json:
        print(json.dumps(plan, indent=2))
        return
    print("CONFIGURATION ONLY / single device / no files written")
    print(f"Plan: {plan['mode']} | precision: {plan['precision']}")
    print(f"Suites: {', '.join(plan['suites'])}")
    print(f"{'Stage':<12} {'Updates':>12} {'Batch':>8} {'Accum.':>8} {'Microsteps':>14}")
    for stage in plan["stages"]:
        print(f"{stage['name']:<12} {stage['updates']:>12,} {stage['effectiveBatch']:>8,} "
              f"{stage['accumulation']:>8,} {stage['microsteps']:>14,}")
    print(f"Total: {plan['updates']:,} optimizer updates; {plan['microsteps']:,} training microsteps")
    evaluation = plan["evaluation"]
    print(f"Simulation: {len(plan['suites'])} suites x {evaluation['tasksPerSuite']} tasks/suite x "
          f"{evaluation['episodesPerTask']} initial states/task = {evaluation['trials']:,} requested trials")
    print("Task limits affect simulation only, not training data or offline evaluation.")
    print("Microsteps exclude evaluation/export; counts are not runtime or VRAM estimates.")
    print("Default model: compact, random initialization; no paper-performance equivalence.")
    print("Prepare separately in a fresh output directory:\n\n" + plan["command"])


if __name__ == "__main__":
    main()

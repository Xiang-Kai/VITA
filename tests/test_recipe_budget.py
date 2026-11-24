"""Offline budget inspection; no dataset, model, simulator or GPU execution."""
import importlib.util
import json
from pathlib import Path
import shlex

import pytest

from vita.libero_recipe import make_parser, prepare_recipe, training_configs


spec = importlib.util.spec_from_file_location(
    "recipe_budget", Path(__file__).resolve().parents[1] / "scripts/recipe_budget.py")
budget = importlib.util.module_from_spec(spec)
spec.loader.exec_module(budget)


@pytest.mark.parametrize("microbatch", budget.MICROBATCHES)
@pytest.mark.parametrize("smoke", (False, True))
def test_budget_matches_real_configs_and_prepare_command(tmp_path, microbatch, smoke):
    suites = ["libero_object", "libero_goal"]
    plan = budget.describe_budget(microbatch, smoke=smoke, suites=suites)
    pre, fine = training_configs(microbatch, smoke=smoke)
    assert plan["microsteps"] == (
        pre.total_steps * pre.accumulation_steps + fine.total_steps * fine.accumulation_steps)
    assert plan["updates"] == pre.total_steps + fine.total_steps
    assert plan["evaluation"]["trials"] == (2 if smoke else 1000)
    tokens = shlex.split(plan["command"].replace("\\\n", ""))
    assert tokens[:3] == ["python", "-m", "vita.libero_recipe"]
    args = make_parser().parse_args(tokens[3:])
    args.output = tmp_path / "config-only"
    recipe = prepare_recipe(**{key: value for key, value in vars(args).items() if key != "command"})
    assert recipe["suites"] == suites
    assert recipe["total_optimizer_updates"] == plan["updates"]
    assert recipe["episodes_per_task"] == plan["evaluation"]["episodesPerTask"]
    assert recipe["max_tasks"] == plan["evaluation"]["maxTasks"]
    assert set(path.name for path in args.output.iterdir()) == {
        "model.json", "pretrain.json", "finetune.json", "recipe.json",
    }
    for stage, training in zip(plan["stages"], (pre, pre, fine)):
        assert stage["effectiveBatch"] == training.batch_size * training.accumulation_steps
        assert stage["accumulation"] == training.accumulation_steps
        assert stage["microsteps"] == stage["updates"] * stage["accumulation"]


@pytest.mark.parametrize("microbatch", (0, -1, 3, 128, 1.5, True, "2"))
def test_budget_rejects_unsupported_microbatch(microbatch):
    with pytest.raises(ValueError, match="microbatch"):
        budget.describe_budget(microbatch)


@pytest.mark.parametrize("suites", ([], ["unknown"], ["libero_goal", "libero_goal"], "libero_goal"))
def test_budget_rejects_invalid_suites(suites):
    with pytest.raises(ValueError, match="suites"):
        budget.describe_budget(suites=suites)


def test_inspection_is_read_only_and_totals_are_unambiguous(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    budget.main(["--microbatch", "16"])
    text = capsys.readouterr().out
    assert "360,000 optimizer updates; 19,920,000 training microsteps" in text
    assert "2,000 requested trials" in text
    assert "not training data or offline evaluation" in text
    assert not list(tmp_path.iterdir())
    budget.main(["--smoke", "--microbatch", "2", "--json"])
    plan = json.loads(capsys.readouterr().out)
    assert plan["updates"] == plan["microsteps"] == 6
    assert plan["evaluation"]["trials"] == 4
    assert not list(tmp_path.iterdir())


def test_web_contract_is_deterministic_and_contains_no_runtime_identity(capsys):
    budget.main(["--export-web"])
    first = capsys.readouterr().out
    budget.main(["--export-web"])
    assert first == capsys.readouterr().out
    exported = json.loads(first.split("export const recipeContract = ", 1)[1].removesuffix(";\n"))
    assert exported == budget.web_contract()
    assert exported["plans"]["paper"]["stages"][0]["effectiveBatch"] == 1024
    assert exported["plans"]["paper"]["stages"][2]["effectiveBatch"] == 192
    assert exported["plans"]["smoke"]["evaluation"]["maxTasks"] == 1
    assert "source_sha256" not in first and str(Path.cwd()) not in first
    with pytest.raises(SystemExit):
        budget.main(["--export-web", "--smoke"])


def test_complete_web_matrix_and_stale_commands_are_checked():
    plans = [
        budget.describe_budget(batch, smoke=smoke, suites=suites)
        for smoke in (True, False) for batch in budget.MICROBATCHES
        for suites in (budget.SUITES, *((suite,) for suite in budget.SUITES))
    ]
    assert budget.verify_web_plans(plans) == 70
    with pytest.raises(ValueError, match="70"):
        budget.verify_web_plans(plans[:-1])
    duplicate = [*plans[:-1], plans[0]]
    with pytest.raises(ValueError, match="duplicated"):
        budget.verify_web_plans(duplicate)
    plans[0]["command"] += "\necho synthetic-invalid-command"
    with pytest.raises(ValueError, match="invalid"):
        budget.verify_web_plans(plans)

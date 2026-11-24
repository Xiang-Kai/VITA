from dataclasses import replace
import json

import h5py
import numpy as np
import pytest
import torch

from vita.dataset import atomic_json
from vita.deploy import ActionPolicy
from vita.experiments import generate_suite, run_job
from vita.latency import profile_policy
from vita.mixture import load_data_manifest
from vita.subset import training_subset
from test_dataset import small_manifest
from test_mixture import mixture_fixture, source_fixture


@pytest.mark.parametrize("kind", ["canonical", "libero", "mixture"])
def test_recording_subsampling_refits_stats_and_preserves_heldout(config, tmp_path, kind):
    config = replace(config, num_views=2)
    manifest = (small_manifest(tmp_path, config) if kind == "libero" else
                mixture_fixture(tmp_path, config) if kind == "mixture" else source_fixture(tmp_path, config, "paired"))
    subset = training_subset(manifest, 0.1, 5)
    assert subset == training_subset(manifest, 0.1, 5)
    assert subset["counts"]["train"] < manifest["counts"]["train"]
    for split in ("validation", "test"):
        assert subset["counts"][split] == manifest["counts"][split]
    if kind != "mixture":
        assert [w for w in subset["windows"] if w["split"] != "train"] == [
            w for w in manifest["windows"] if w["split"] != "train"]
        rows = []
        infos = subset["files"] if kind == "libero" else [subset]
        for info in infos:
            with h5py.File(info["path"], "r") as handle:
                for name, ep in info["episodes"].items():
                    if ep["split"] == "train":
                        rows.append(handle[f"{'data' if kind == 'libero' else 'episodes'}/{name}/actions"][:])
        columns = list(range(config.action_dim)) if kind == "libero" else subset["action_columns"]
        np.testing.assert_allclose(np.array(subset["normalization"]["mean"])[columns],
                                   np.concatenate(rows).mean(0), atol=1e-5)
    atomic_json(tmp_path / "subset.json", subset)
    assert load_data_manifest(tmp_path / "subset.json") == subset


def test_suite_contains_all_variants_and_executes_dependencies(config, tmp_path):
    manifest = source_fixture(tmp_path / "data", config, "paired")
    path = tmp_path / "data.json"
    atomic_json(path, manifest)
    plan = generate_suite(config, path, tmp_path / "suite", smoke=True)
    jobs = plan["jobs"]
    for name in ("cot_none", "cot_textual", "cot_visual", "cot_textual_visual",
                 "cot_internal", "cot_textual_internal", "no_alignment",
                 "initialization_codebook", "initialization_visual", "initialization_full"):
        assert name in jobs
    for budget in (500, 1000, 2000):
        assert f"efficiency_no_warmup_{budget}" in jobs
        assert f"efficiency_warmup_{budget}" in jobs
    suite = tmp_path / "suite/suite.json"
    run_job(suite, "initialization_warmup")
    result = run_job(suite, "initialization_codebook_pretrain")
    assert result["initialization"]["transferred_modules"] == ["quantizer"]
    run_job(suite, "initialization_codebook")
    run_job(suite, "cot_none_pretrain")
    run_job(suite, "cot_none")
    assert plan["executed"] is False


def test_paper_suite_preserves_effective_stage_batches(config, tmp_path):
    path = tmp_path / "data.json"
    atomic_json(path, source_fixture(tmp_path / "data", config, "paired"))
    generate_suite(config, path, tmp_path / "suite", suite="cot", benchmark="libero")
    root = tmp_path / "suite"
    train = json.loads((root / "cot_internal_pretrain.train.json").read_text())
    ft = json.loads((root / "cot_internal.train.json").read_text())
    assert train["warmup_steps"] == 200000 and train["cotrain_steps"] == 100000
    assert train["batch_size"] * train["accumulation_steps"] == 1024
    assert ft["batch_size"] * ft["accumulation_steps"] == 192
    assert ft["finetune_steps"] == 60000


@pytest.mark.parametrize("future", [12, 24, 48])
def test_latency_measures_real_unmodified_model(config, future):
    # A tiny synthetic performance fixture, never the paper's measured latency.
    config = replace(config, future_frames=future, latent_tokens=2, action_horizon=3)
    model = ActionPolicy(config).requires_grad_(False)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    result = profile_policy(model, warmup=1, repeats=2)
    assert result["future_frames"] == future and result["latent_tokens"] == 2
    assert result["action_horizon"] == 3
    assert result["synthetic_inputs"] is True and result["benchmark_reproduced"] is False
    for metric in result["measurements_ms"].values():
        assert len(metric["samples"]) == 2 and metric["mean"] > 0
    for key, value in before.items():
        torch.testing.assert_close(value, model.state_dict()[key], atol=0, rtol=0)

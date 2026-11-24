from dataclasses import replace

import pytest
import torch

from vita import VITA
from vita.data import synthetic_batch
from vita.deploy import ActionPolicy, data_contract, export_action_policy, load_action_policy
from vita.evaluate import load_policy


def test_action_export_preserves_predictions_and_omits_visual_modules(config, tmp_path):
    model = VITA(config).eval()
    model.set_stage("finetune")
    source = tmp_path / "training.pt"
    torch.save({"schema_version": 2, "stage": "finetune", "model": model.state_dict(),
                "model_config": config.to_dict()}, source)
    result = export_action_policy(source, tmp_path / "actions.pt")
    assert result["retained_parameters"] < result["original_state_elements"]
    restored, metadata = load_action_policy(tmp_path / "actions.pt")
    assert not hasattr(restored, "visual_encoder") and not hasattr(restored, "visual_decoder")
    batch = synthetic_batch(config)
    before = model.act(batch["instructions"], batch["current"], batch["states"])
    # PyTorch decoder numerics differ slightly when parameter requires_grad flags differ,
    # even under no_grad. Match the deployment flags for the bit-exact check.
    model.requires_grad_(False)
    frozen = model.act(batch["instructions"], batch["current"], batch["states"])
    deployed = restored.act(batch["instructions"], batch["current"], batch["states"])
    torch.testing.assert_close(before, deployed, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(frozen, deployed, rtol=0, atol=0)
    for name, value in restored.state_dict().items():
        torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)
    assert metadata["schema_version"] == 3
    assert "model" not in metadata
    # A DINO path is irrelevant for action-only deployment.
    ActionPolicy(replace(config, vision_backend="dinov2", dinov2_path="/nonexistent/unused/dino"))


def test_source_normalization_is_not_guessed():
    checkpoint = {"data_manifest": {"kind": "mixture", "sources": [
        {"name": name, "mode": "paired", "weights": {"finetune": 1}, "manifest": {"identity": name}}
        for name in ("first", "second")
    ]}}
    with pytest.raises(ValueError, match="normalization"):
        data_contract(checkpoint)
    assert data_contract(checkpoint, "second")["identity"] == "second"


@pytest.mark.parametrize("loader", [
    load_policy,
    lambda path, device: load_action_policy(path, device, require_source_match=True),
])
def test_evaluators_reject_other_source_before_model_construction(tmp_path, loader):
    source = tmp_path / "synthetic-mismatched-source.pt"
    # No model config/weights: reaching model construction would fail with a different error.
    torch.save({"schema_version": 2, "source_sha256": {"fixture.py": "different"}}, source)
    with pytest.raises(ValueError, match="current evaluator"):
        loader(source, "cpu")

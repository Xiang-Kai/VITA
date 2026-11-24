"""Offline synthetic tensors only: no simulator, hardware, provider or network calls."""
from dataclasses import replace

import pytest
import torch

from vita import VITA
from vita.config import COT_VARIANTS
from vita.data import synthetic_batch
from vita.deploy import export_action_policy, load_action_policy
from vita.train import initialize_components


@pytest.mark.parametrize("variant", COT_VARIANTS)
def test_distinct_cot_graphs_and_export(config, tmp_path, variant):
    config = replace(config, cot_variant=variant)
    model = VITA(config).eval()
    batch = synthetic_batch(config)
    model.set_stage("cotrain")
    calls = []
    handles = [getattr(model, name).register_forward_hook(
        lambda _m, _a, _o, name=name: calls.append(name))
        for name in ("visual_decoder", "action_decoder", "quantizer")]
    result = model.cotrain(batch)
    result["loss"].backward()
    assert ("visual_decoder" in calls) == config.predicts_images
    assert result["subtask_ids"].shape[1] == (config.subtask_slots if config.textual_cot else 0)
    assert result["token_ids"].shape[1] == (config.token_count if config.internal_cot else 0)
    assert (model.policy.subtask_head.weight.grad is not None) == config.textual_cot
    assert (model.policy.code_head.weight.grad is not None) == config.internal_cot
    if config.explicit_visual_cot:
        assert model.future_encoder.patch.weight.grad.abs().sum() > 0
        # Action loss alone must traverse the decoded image branch.
        model.zero_grad(set_to_none=True)
        model.cotrain(batch)["action_mse"].backward()
        assert model.visual_decoder.pixels.weight.grad.abs().sum() > 0
    for handle in handles:
        handle.remove()
    model.set_stage("finetune")
    model.finetune(batch)["loss"].backward()
    assert all(p.grad is None for name, p in model.named_parameters()
               if not name.startswith(("action_decoder.", "future_encoder.")))
    path = tmp_path / "training.pt"
    torch.save({"schema_version": 2, "stage": "finetune", "model": model.state_dict(),
                "model_config": config.to_dict()}, path)
    export_action_policy(path, tmp_path / "export.pt")
    deployed, _ = load_action_policy(tmp_path / "export.pt")
    model.requires_grad_(False)
    args = (batch["instructions"], batch["current"], batch["states"])
    torch.testing.assert_close(model.act(*args), deployed.act(*args), atol=0, rtol=0)
    changed = dict(batch, future=torch.randn_like(batch["future"]), actions=torch.randn_like(batch["actions"]))
    torch.testing.assert_close(model.generate(batch)["latents"], model.generate(changed)["latents"],
                               atol=0, rtol=0)


def test_separate_codebooks_and_policy_heads_are_really_independent(config, tmp_path):
    config = replace(config, modality_alignment="separate")
    model = VITA(config).eval()
    batch = synthetic_batch(config)
    model.warmup_action(batch["actions"])["loss"].backward()
    assert model.quantizer.embedding.weight.grad is None
    assert model.action_quantizer.embedding.weight.grad.abs().sum() > 0
    model.set_stage("cotrain")
    result = model.cotrain(batch)
    result["loss"].backward()
    assert model.policy.action_code_head.weight.grad.abs().sum() > 0
    assert model.policy.code_head.weight.grad.abs().sum() > 0
    with torch.no_grad():
        before = model.generate(batch)
        model.quantizer.embedding.weight.add_(100)
        after = model.generate(batch)
    torch.testing.assert_close(before["latents"], after["latents"], atol=0, rtol=0)
    assert not torch.equal(before["visual_latents"], after["visual_latents"])
    model.set_stage("finetune")
    path = tmp_path / "training.pt"
    torch.save({"schema_version": 2, "stage": "finetune", "model": model.state_dict(),
                "model_config": config.to_dict()}, path)
    export_action_policy(path, tmp_path / "export.pt")
    deployed, _ = load_action_policy(tmp_path / "export.pt")
    model.requires_grad_(False)
    args = (batch["instructions"], batch["current"], batch["states"])
    torch.testing.assert_close(model.act(*args), deployed.act(*args), atol=0, rtol=0)


def test_independent_lengths_memory_and_tokenized_symbols(config):
    config = replace(config, latent_tokens=3, future_frames=2, action_horizon=5,
                     mformer_memory_tokens=2, subtask_embedding="tokenizer")
    model = VITA(config)
    batch = synthetic_batch(config)
    latent, _ = model.visual_latents(batch["current"], batch["future"])
    assert latent.shape == (2, 3, config.latent_dim)
    model.warmup_video(batch["current"], batch["future"])["loss"].backward()
    assert model.motion_encoder.memory.grad.abs().sum() > 0
    assert model.action_encoder(batch["actions"]).shape == latent.shape
    model.set_stage("cotrain")
    result = model.cotrain(batch)
    result["loss"].backward()
    assert result["images"].shape == batch["future"].shape
    assert result["actions"].shape == batch["actions"].shape
    assert result["token_ids"].shape == (2, 3)
    assert model.policy.subtask_embeddings.weight.grad is None
    assert model.policy.trunk.text.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("selection,expected", [
    ("codebook", {"quantizer"}),
    ("codebook_visual", {"quantizer", "visual_decoder"}),
    ("codebook_decoders", {"quantizer", "visual_decoder", "action_decoder"}),
])
def test_selective_initialization_does_not_copy_other_modules(config, selection, expected):
    prior, model = VITA(config), VITA(config)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    saved = {"schema_version": 2, "stage": "warmup", "model_config": config.to_dict(),
             "model": prior.state_dict()}
    assert set(initialize_components(model, saved, selection)) == expected
    for key, value in model.state_dict().items():
        wanted = saved["model"][key] if key.split(".")[0] in expected else before[key]
        torch.testing.assert_close(value, wanted, atol=0, rtol=0)
    saved["stage"] = "finetune"
    with pytest.raises(ValueError, match="warmup"):
        initialize_components(model, saved, selection)


def test_no_visual_cot_rejects_unpaired_video(config):
    model = VITA(replace(config, cot_variant="none"))
    model.set_stage("cotrain")
    batch = synthetic_batch(config)
    batch.pop("actions")
    with pytest.raises(ValueError, match="paired"):
        model.cotrain(batch)


def test_explicit_alternative_appendix_readings(config):
    config = replace(config, finetune_scope="joint", action_decode_order="project_idct_mlp")
    model = VITA(config)
    model.set_stage("finetune")
    result = model.finetune(synthetic_batch(config))
    result["loss"].backward()
    assert model.policy.code_head.weight.grad.abs().sum() > 0
    assert model.visual_decoder.pixels.weight.grad.abs().sum() > 0
    assert model.action_decoder.output_mlp[0].weight.grad.abs().sum() > 0
    assert model.quantizer.embedding.weight.grad is None

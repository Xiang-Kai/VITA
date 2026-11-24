from dataclasses import replace

import pytest
import torch

from vita import VITA
from vita.data import synthetic_batch
from vita.math import progressive_mask


@pytest.mark.parametrize("field,value", [
    ("image_size", 32.0),
    ("num_views", True),
    ("latent_tokens", 1.5),
    ("feature_batch_size", "4"),
    ("temperature", float("nan")),
    ("action_weight", float("inf")),
    ("auxiliary_token_weight", False),
    ("gradient_checkpointing", "false"),
])
def test_model_config_rejects_invalid_numeric_and_boolean_fields(config, field, value):
    with pytest.raises(ValueError, match=field):
        replace(config, **{field: value})


@pytest.mark.parametrize("stage,expected", [
    ("warmup", {"motion_encoder", "visual_decoder", "action_encoder", "action_decoder", "quantizer"}),
    ("cotrain", {"policy", "visual_decoder", "action_decoder"}),
    ("finetune", {"action_decoder"}),
])
def test_stage_freezes_exact_modules(config, stage, expected):
    model = VITA(config)
    model.set_stage(stage)
    model.train()
    trainable = {name.split(".")[0] for name, p in model.named_parameters() if p.requires_grad}
    assert trainable == expected
    assert not model.visual_encoder.training
    for name, module in model.named_children():
        if name not in expected:
            assert not module.training


@pytest.mark.parametrize("mode", ["visual_only", "both"])
def test_action_warmup_codebook_policy(config, mode):
    model = VITA(replace(config, codebook_update=mode))
    batch = synthetic_batch(config)
    result = model.warmup_action(batch["actions"])
    assert result["prediction"].shape == batch["actions"].shape
    result["loss"].backward()
    grad = model.quantizer.embedding.weight.grad
    assert (grad is not None) == (mode == "both")
    assert any(p.grad is not None for p in model.action_encoder.parameters())
    assert all(p.grad is None for p in model.visual_decoder.parameters())


def test_visual_warmup_trains_codebook_not_frozen_image_encoder(config):
    model = VITA(config)
    batch = synthetic_batch(config)
    result = model.warmup_video(batch["current"], batch["future"])
    assert result["prediction"].shape == batch["future"].shape
    result["loss"].backward()
    assert model.quantizer.embedding.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.visual_encoder.parameters())


@pytest.mark.parametrize("paired", [True, False])
def test_cotrain_gradients_reach_policy_and_correct_decoders(config, paired):
    model = VITA(config)
    model.set_stage("cotrain")
    batch = synthetic_batch(config)
    if not paired:
        batch.pop("actions")
    result = model.cotrain(batch)
    result["loss"].backward()
    for head in (model.policy.code_head, model.policy.subtask_head):
        assert head.weight.grad is not None
        assert head.weight.grad.abs().sum() > 0
    assert model.quantizer.embedding.weight.grad is None
    assert any(p.grad is not None for p in model.visual_decoder.parameters())
    assert any(p.grad is not None for p in model.action_decoder.parameters()) == paired


def test_both_decoders_receive_the_identical_hard_latent_stream(config):
    model = VITA(config)
    model.set_stage("cotrain")
    captured = {}
    first = model.visual_decoder.register_forward_pre_hook(lambda _m, args: captured.update(visual=args[1]))
    second = model.action_decoder.register_forward_pre_hook(lambda _m, args: captured.update(action=args[0]))
    result = model.cotrain(synthetic_batch(config))
    first.remove()
    second.remove()
    assert captured["visual"] is captured["action"]
    torch.testing.assert_close(result["latents"], model.quantizer.embedding.weight[result["token_ids"]], rtol=0, atol=0)


def test_generation_does_not_read_future_or_actions(config):
    model = VITA(config).eval()
    batch = synthetic_batch(config)
    original = model.generate(batch)["latents"]
    batch["future"] = torch.randn_like(batch["future"])
    batch["actions"] = torch.randn_like(batch["actions"])
    changed = model.generate(batch)["latents"]
    torch.testing.assert_close(original, changed, rtol=0, atol=0)


def test_inference_never_calls_warmup_encoder_or_visual_decoder(config):
    model = VITA(config).eval()
    batch = synthetic_batch(config)

    def forbidden(*_):
        raise AssertionError("Visual reconstruction must be absent from action inference")

    handles = [module.register_forward_pre_hook(forbidden)
               for module in (model.visual_encoder, model.motion_encoder, model.visual_decoder, model.action_encoder)]
    actions = model.act(batch["instructions"], batch["current"], batch["states"])
    for handle in handles:
        handle.remove()
    assert actions.shape == batch["actions"].shape


def test_attention_prevents_future_leakage(config):
    model = VITA(config)
    trunk = model.policy.trunk
    values = torch.randn(2, 9, config.policy_dim)
    valid = torch.ones(2, 9, dtype=torch.bool)
    mask = progressive_mask(3, 2, 4)
    original = trunk(values, mask, valid)
    changed = values.clone()
    changed[:, -1] += torch.randn_like(changed[:, -1]) * 100
    modified = trunk(changed, mask, valid)
    torch.testing.assert_close(original[:, :-1], modified[:, :-1], atol=1e-6, rtol=1e-5)


def test_stage_transition_clears_old_gradients(config):
    model = VITA(config)
    batch = synthetic_batch(config)
    model.warmup_video(batch["current"], batch["future"])["loss"].backward()
    model.set_stage("finetune")
    model.finetune(batch)["loss"].backward()
    assert all(p.grad is None for name, p in model.named_parameters() if not name.startswith("action_decoder."))


def test_wrong_stage_is_rejected(config):
    model = VITA(config)
    with pytest.raises(RuntimeError):
        model.cotrain(synthetic_batch(config))


def test_checkpoint_preserves_predictions(config, tmp_path):
    model = VITA(config).eval()
    batch = synthetic_batch(config)
    path = tmp_path / "model.pt"
    torch.save(model.state_dict(), path)
    other = VITA(config).eval()
    other.load_state_dict(torch.load(path, weights_only=True))
    arguments = (batch["instructions"], batch["current"], batch["states"])
    torch.testing.assert_close(model.act(*arguments), other.act(*arguments), atol=0, rtol=0)

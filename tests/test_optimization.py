from dataclasses import replace

import pytest
import torch

from vita import VITA
from vita.data import synthetic_batch
from vita.modules import ActionDecoder, ActionEncoder, VisualDecoder
from vita.quantizer import SharedVectorQuantizer


def test_decoder_latent_order_is_observable(config):
    latents = torch.randn(2, config.future_frames, config.latent_dim)
    action = ActionDecoder(config).eval()
    assert not torch.allclose(action(latents), action(latents.flip(1)), atol=1e-7, rtol=1e-7)
    visual = VisualDecoder(config, config.feature_dim).eval()
    context = torch.randn(2, 16, config.feature_dim)
    assert not torch.allclose(visual(context, latents), visual(context, latents.flip(1)), atol=1e-7, rtol=1e-7)


def test_kmeans_initialization_is_repeatable_and_does_not_update_frozen_codebook():
    points = torch.randn(100, 8)
    first, second = SharedVectorQuantizer(16, 8), SharedVectorQuantizer(16, 8)
    result = first.initialize_from_data(points, seed=3)
    second.initialize_from_data(points, seed=3)
    torch.testing.assert_close(first.embedding.weight, second.embedding.weight, atol=0, rtol=0)
    assert result["occupied_codes_after_initialization"] > 1
    first.requires_grad_(False)
    with pytest.raises(RuntimeError):
        first.initialize_from_data(points)


def test_kmeans_rejects_insufficient_data():
    with pytest.raises(ValueError):
        SharedVectorQuantizer(16, 8).initialize_from_data(torch.randn(2, 8))


def multi_batch(config):
    batch = synthetic_batch(config)
    batch["current"] = torch.stack([batch["current"], batch["current"].flip(-1)], dim=1)
    batch["future"] = torch.stack([batch["future"], batch["future"].flip(-1)], dim=2)
    batch["view_mask"] = torch.tensor([[True, False], [True, True]])
    return batch


def test_multiview_all_stages_and_masked_view_invariance(config):
    config = replace(config, num_views=2)
    model = VITA(config)
    batch = multi_batch(config)
    video = model.warmup_video(batch["current"], batch["future"], batch["view_mask"])
    assert video["prediction"].shape == batch["future"].shape
    video["loss"].backward()
    model.set_stage("cotrain")
    result = model.cotrain(batch)
    result["loss"].backward()
    assert torch.isfinite(result["loss"])
    original = model.generate(batch)["token_logits"]
    batch["current"][0, 1] = torch.randn_like(batch["current"][0, 1]) * 100
    changed = model.generate(batch)["token_logits"]
    torch.testing.assert_close(original, changed, rtol=0, atol=0)
    model.set_stage("finetune")
    model.finetune(batch)["loss"].backward()
    assert model.act(batch["instructions"], batch["current"], batch["states"], batch["view_mask"]).shape == (2, 4, 7)


def test_no_valid_view_is_rejected(config):
    model = VITA(replace(config, num_views=2))
    batch = multi_batch(config)
    batch["view_mask"][0] = False
    with pytest.raises(ValueError, match="view_mask"):
        model.generate(batch)


def test_auxiliary_labels_only_affect_loss_not_generation(config):
    model = VITA(replace(config, auxiliary_token_weight=0.5))
    model.set_stage("cotrain")
    batch = synthetic_batch(config)
    first = model.cotrain(batch)
    assert first["auxiliary_token_ce"] > 0
    batch["actions"] = torch.randn_like(batch["actions"])
    second = model.cotrain(batch)
    torch.testing.assert_close(first["token_logits"], second["token_logits"], rtol=0, atol=0)
    second["loss"].backward()
    assert model.policy.code_head.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.action_encoder.parameters())


def test_optional_action_latent_normalization_matches_visual_scale(config):
    encoder = ActionEncoder(replace(config, action_latent_norm=True))
    values = encoder(synthetic_batch(config)["actions"])
    torch.testing.assert_close(values.mean(-1), torch.zeros_like(values.mean(-1)), atol=1e-6, rtol=0)
    torch.testing.assert_close(values.var(-1, unbiased=False), torch.ones_like(values.var(-1, unbiased=False)),
                               atol=0.003, rtol=0)


def test_quantizer_assignment_remains_float32_under_autocast():
    weights = torch.randn(32, 32)
    points = weights + torch.randn_like(weights) * 0.002
    expected = SharedVectorQuantizer.nearest(points, weights)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = SharedVectorQuantizer.nearest(points, weights)
    assert torch.equal(actual, expected)
    assert torch.equal(expected, torch.arange(32))


@pytest.mark.parametrize("stage", ["warmup", "cotrain", "finetune"])
def test_gradient_checkpointing_preserves_values_and_gradients(config, stage):
    regular = VITA(config)
    checked = VITA(replace(config, gradient_checkpointing=True, feature_batch_size=1))
    checked.load_state_dict(regular.state_dict())
    batch = synthetic_batch(config)
    for model in (regular, checked):
        model.set_stage(stage)
        result = (model.warmup_video(batch["current"], batch["future"])
                  if stage == "warmup" else getattr(model, stage)(batch))
        result["loss"].backward()
    for (name, first), (_, second) in zip(regular.named_parameters(), checked.named_parameters()):
        if first.grad is not None:
            torch.testing.assert_close(first.grad, second.grad, rtol=3e-5, atol=1e-6, msg=name)

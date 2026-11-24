"""Offline random mini-checkpoints test adapter APIs, NOT pretrained quality."""

from dataclasses import replace

import pytest
import torch

from vita.data import synthetic_batch
from vita.modules import FrozenImageEncoder
from vita.policy import ProgressivePolicy


@pytest.mark.parametrize("registers", [False, True])
def test_local_dinov2_adapter(config, tmp_path, registers):
    transformers = pytest.importorskip("transformers")
    config_class = transformers.Dinov2WithRegistersConfig if registers else transformers.Dinov2Config
    model_class = transformers.Dinov2WithRegistersModel if registers else transformers.Dinov2Model
    dino_config = config_class(
        image_size=32, patch_size=8, hidden_size=32, num_hidden_layers=1,
        num_attention_heads=4, intermediate_size=64,
    )
    model_class(dino_config).save_pretrained(tmp_path)
    encoder = FrozenImageEncoder(replace(config, vision_backend="dinov2", dinov2_path=str(tmp_path)))
    features = encoder(torch.rand(2, 3, 32, 32))
    assert features.shape == (2, 16, 32)
    assert not features.requires_grad


@pytest.mark.parametrize("resolution", ["strict", "interpolate_positions", "resize_input"])
@pytest.mark.parametrize("embedding", ["learned", "tokenizer"])
def test_local_paligemma_progressive_adapter(config, tmp_path, resolution, embedding):
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel(
        {"[PAD]": 0, "[UNK]": 1, "[BOS]": 2, "move": 3, "the": 4, "synthetic": 5, "object": 6},
        unk_token="[UNK]",
    ))
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token="[PAD]", unk_token="[UNK]", bos_token="[BOS]",
    ).save_pretrained(tmp_path)
    foundation_config = transformers.PaliGemmaConfig(
        vision_config={"hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 1,
                       "num_attention_heads": 4, "image_size": 32 if resolution == "strict" else 24,
                       "patch_size": 8},
        text_config={"hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 1,
                     "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
                     "vocab_size": 16, "pad_token_id": 0},
        image_token_index=15, projection_dim=32, hidden_size=32,
    )
    transformers.PaliGemmaForConditionalGeneration(foundation_config).save_pretrained(tmp_path)
    policy = ProgressivePolicy(replace(config, policy_backend="paligemma", paligemma_path=str(tmp_path),
                                       paligemma_resolution=resolution, gradient_checkpointing=True,
                                       subtask_embedding=embedding))
    assert policy.trunk.language.get_input_embeddings() is policy.trunk.language.embed_tokens
    assert not hasattr(policy.trunk.language, "lm_head")
    batch = synthetic_batch(config)
    codebook = torch.randn(config.codebook_size, config.latent_dim)
    result = policy(batch["instructions"], batch["current"], batch["states"], codebook)
    assert result["latents"].shape == (2, config.future_frames, config.latent_dim)
    result["latents"].square().mean().backward()
    assert policy.code_head.weight.grad.abs().sum() > 0
    assert policy.trunk.vision.vision_model.embeddings.patch_embedding.weight.grad.abs().sum() > 0
    if resolution != "strict":
        with pytest.raises(ValueError, match="Checkpoint resolution"):
            ProgressivePolicy(replace(config, policy_backend="paligemma", paligemma_path=str(tmp_path)))


def test_incomplete_pretrained_weights_fail_instead_of_random_initialization(config, tmp_path):
    transformers = pytest.importorskip("transformers")
    safetensors = pytest.importorskip("safetensors.torch")
    dino_config = transformers.Dinov2Config(
        image_size=32, patch_size=8, hidden_size=32, num_hidden_layers=1, num_attention_heads=4)
    transformers.Dinov2Model(dino_config).save_pretrained(tmp_path)
    weights = safetensors.load_file(tmp_path / "model.safetensors")
    weights.pop(next(iter(weights)))
    safetensors.save_file(weights, tmp_path / "model.safetensors", metadata={"format": "pt"})
    with pytest.raises(ValueError, match="Incomplete"):
        FrozenImageEncoder(replace(config, vision_backend="dinov2", dinov2_path=str(tmp_path)))

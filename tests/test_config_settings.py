from __future__ import annotations

from micv.utils import load_config


def test_load_config_accepts_token_backbone_and_augmentation_settings(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model:
  stream1_backbones: 4
  stream2_backbones: 2
  latent_dim: 1024
  mlp_hidden_dims: [1024, 256]
  dropout: 0.2
  token_pooling: attention
  backbone:
    model_name_or_path: facebook/dinov3-vitb16-pretrain-lvd1689m
    trust_remote_code: true
    gradient_checkpointing: true
    freeze: false
    output_mode: patch_tokens
    include_cls_token: false
    include_register_tokens: false
augmentation:
  train_difficulty: mixed
  static_val_augmentation: true
  clean_prob: 0.30
  max_ops: 5
data:
  delete_invalid_images: true
""".lstrip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.model.latent_dim == 1024
    assert config.model.mlp_hidden_dims == [1024, 256]
    assert config.model.token_pooling == "attention"
    assert config.model.backbone.output_mode == "patch_tokens"
    assert config.model.backbone.include_cls_token is False
    assert config.model.backbone.include_register_tokens is False
    assert config.augmentation.static_val_augmentation is True
    assert config.augmentation.clean_prob == 0.30
    assert config.augmentation.max_ops == 5
    assert config.data.delete_invalid_images is True
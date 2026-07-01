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


def test_load_config_accepts_stream_backbone_settings(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model:
  latent_dim: 768
  token_pooling: attention
  stream_fusion: token_concat_attention
  streams:
    - name: stream1
      backbones:
        - model_name_or_path: facebook/dinov3-vits16-pretrain-lvd1689m
          freeze: false
        - model_name_or_path: facebook/dinov3-vitb16-pretrain-lvd1689m
          freeze: true
    - name: stream2
      repeat: 2
      backbone:
        model_name_or_path: facebook/dinov3-vitb16-pretrain-lvd1689m
        freeze: false
""".lstrip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.model.stream_fusion == "token_concat_attention"
    assert len(config.model.streams) == 2
    assert config.model.streams[0].name == "stream1"
    assert config.model.streams[0].backbones[0].model_name_or_path.endswith("vits16-pretrain-lvd1689m")
    assert config.model.streams[0].backbones[1].freeze is True
    assert config.model.streams[1].repeat == 2
    assert config.model.streams[1].backbone is not None
    assert config.model.streams[1].backbone.model_name_or_path.endswith("vitb16-pretrain-lvd1689m")

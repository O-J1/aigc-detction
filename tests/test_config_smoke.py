from __future__ import annotations

import importlib.util
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from PIL import Image

from micv.models import MICVDualStreamEnsemble
from micv.utils import load_config


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("config_path", sorted(CONFIG_DIR.glob("*.yaml")))
def test_checked_in_config_builds_dummy_model_and_runs_forward(config_path: Path) -> None:
    config = load_config(config_path)
    config.model.use_dummy_backbone = True
    config.model.dummy_feature_dim = 8
    config.model.latent_dim = 16
    config.model.mlp_hidden_dims = [8]
    config.model.dropout = 0.0

    model = MICVDualStreamEnsemble.from_config(asdict(config.model))
    outputs = model(torch.randn(2, 3, 32, 32))

    assert outputs["stream1_logits"].shape == (2,)
    assert outputs["stream2_logits"].shape == (2,)
    assert outputs["fused_prob"].shape == (2,)


def _load_script_module(name: str, script_name: str):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / "scripts" / script_name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_images(image_dir: Path, count: int, fake: bool, seed: int) -> list[Path]:
    generator = torch.Generator().manual_seed(seed)
    image_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(count):
        if fake:
            pixels = torch.randint(0, 256, (64, 64, 3), generator=generator, dtype=torch.uint8)
            image = Image.fromarray(pixels.numpy())
        else:
            image = Image.new("RGB", (64, 64), color=(180, 40 + 10 * index, 40))
        path = image_dir / f"img_{index}.png"
        image.save(path)
        paths.append(path)
    return paths


def test_train_and_predict_end_to_end(tmp_path, monkeypatch) -> None:
    real_paths = _write_images(tmp_path / "images" / "real", 6, fake=False, seed=1)
    fake_paths = _write_images(tmp_path / "images" / "fake", 6, fake=True, seed=2)

    manifest_path = tmp_path / "manifest.csv"
    lines = ["path,label,split"]
    for split, start, stop in (("train", 0, 4), ("val", 4, 6)):
        for label, paths in ((0, real_paths), (1, fake_paths)):
            for path in paths[start:stop]:
                lines.append(f"{path.as_posix()},{label},{split}")
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    output_dir = tmp_path / "outputs"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
model:
  use_dummy_backbone: true
  dummy_feature_dim: 8
  latent_dim: 16
  mlp_hidden_dims: [8]
  dropout: 0.0
  per_backbone_views: true
data:
  train_manifest: {manifest_path.as_posix()}
  val_manifest: {manifest_path.as_posix()}
  image_size: 32
  train_image_sizes: [24, 32]
  batch_size: 2
  num_workers: 0
  persistent_workers: false
  pin_memory: false
augmentation:
  static_val_augmentation: true
  static_val_severity: val
training:
  epochs: 2
  amp: false
  output_dir: {output_dir.as_posix()}
  log_every_steps: 1
swa:
  enabled: true
  start_epoch: 2
distributed:
  enabled: false
""",
        encoding="utf-8",
    )

    train_module = _load_script_module("micv_train_script", "train.py")
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", str(config_path)])
    train_module.main()

    for checkpoint_name in ("latest.pt", "best.pt", "swa_model.pt"):
        assert (output_dir / checkpoint_name).exists(), checkpoint_name

    best_checkpoint = torch.load(output_dir / "best.pt")
    assert best_checkpoint["metrics"] is not None
    assert "roc_auc" in best_checkpoint["metrics"]
    assert best_checkpoint["degraded_metrics"] is not None

    swa_checkpoint = torch.load(output_dir / "swa_model.pt")
    assert swa_checkpoint["metrics"] is not None
    assert "model" in swa_checkpoint

    predict_module = _load_script_module("micv_predict_script", "predict.py")
    predictions_path = tmp_path / "predictions.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "predict.py",
            "--config",
            str(config_path),
            "--checkpoint",
            str(output_dir / "swa_model.pt"),
            "--input",
            str(tmp_path / "images"),
            "--output",
            str(predictions_path),
            "--tta",
            "hflip",
        ],
    )
    predict_module.main()

    prediction_lines = predictions_path.read_text(encoding="utf-8").strip().splitlines()
    assert prediction_lines[0].startswith("path,prob_ai,prediction")
    assert len(prediction_lines) == 1 + len(real_paths) + len(fake_paths)


def test_train_uses_one_view_per_committee_slot() -> None:
    train_module = _load_script_module("micv_train_view_count", "train.py")
    config = load_config(CONFIG_DIR / "cluster.yaml")

    assert train_module._resolve_num_views(config.model) == 6


def test_standalone_evaluation_uses_configured_static_severity(monkeypatch) -> None:
    evaluate_module = _load_script_module("micv_evaluate_transform", "evaluate.py")
    config = load_config(CONFIG_DIR / "smoke.yaml")
    captured: dict[str, object] = {}

    def fake_build_eval_transform(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(evaluate_module, "build_eval_transform", fake_build_eval_transform)

    evaluate_module._build_configured_eval_transform(config, static_augmentation=True)

    assert captured["static_augmentation"] is True
    assert captured["static_severity"] == config.augmentation.static_val_severity

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from micv.data import AIGCManifestDataset, build_eval_transform
from micv.models import MICVDualStreamEnsemble
from micv.training.losses import CombinedMICVLoss
from micv.training.trainer import Trainer, load_checkpoint
from micv.utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a MICV checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--static-augmentation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    manifest_path = args.manifest or config.data.val_manifest
    if manifest_path is None:
        raise ValueError("Provide --manifest or set data.val_manifest in the config.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_transform = build_eval_transform(
        image_size=config.data.image_size,
        mean=config.augmentation.mean,
        std=config.augmentation.std,
        static_augmentation=args.static_augmentation or config.augmentation.static_val_augmentation,
    )
    dataset = AIGCManifestDataset(
        manifest_path,
        split=config.data.val_split,
        root_dir=config.data.root_dir,
        transform=eval_transform,
        bad_image_policy=config.data.bad_image_policy,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )
    model = MICVDualStreamEnsemble.from_config(asdict(config.model)).to(device)
    load_checkpoint(args.checkpoint, model=model, map_location=device)

    trainer = Trainer(
        model=model,
        train_loader=data_loader,
        val_loader=data_loader,
        loss_fn=CombinedMICVLoss(),
        optimizer=torch.optim.AdamW(model.parameters(), lr=1.0e-5),
        scheduler=None,
        device=device,
        output_dir=config.training.output_dir,
        amp=config.training.amp,
        amp_dtype=config.training.amp_dtype,
        swa_enabled=False,
    )
    metrics = trainer.evaluate(data_loader)
    print(metrics)


if __name__ == "__main__":
    main()
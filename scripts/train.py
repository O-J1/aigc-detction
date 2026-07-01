from __future__ import annotations

import argparse
import math
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from micv.data import (  # noqa: E402
    AIGCManifestDataset,
    DistributedWeightedSampler,
    build_eval_transform,
    build_train_transform,
    make_weighted_sampler,
)
from micv.models import MICVDualStreamEnsemble  # noqa: E402
from micv.training.distributed import get_rank, init_distributed_from_env, is_distributed  # noqa: E402
from micv.training.losses import BinaryFocalLossWithLogits, BinaryFocalLossWithProbabilities, CombinedMICVLoss  # noqa: E402
from micv.training.scheduler import build_param_groups, build_warmup_cosine_scheduler  # noqa: E402
from micv.training.trainer import Trainer, load_checkpoint  # noqa: E402
from micv.utils import load_config, seed_everything  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the MICV DINOv3 ensemble detector.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument("--resume", default=None, help="Optional checkpoint path to resume from.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    distributed_started = False
    if config.distributed.enabled:
        distributed_started = init_distributed_from_env(config.distributed.backend)

    seed_everything(config.training.seed + get_rank(), deterministic=config.training.deterministic)
    device = torch.device("cuda", int(torch.cuda.current_device())) if torch.cuda.is_available() else torch.device("cpu")

    if config.data.train_manifest is None:
        raise ValueError("data.train_manifest must be set before training.")

    train_transform = build_train_transform(
        image_size=config.data.image_size,
        difficulty=config.augmentation.train_difficulty,
        clean_prob=config.augmentation.clean_prob,
        max_ops=config.augmentation.max_ops,
        mean=config.augmentation.mean,
        std=config.augmentation.std,
    )
    eval_transform = build_eval_transform(
        image_size=config.data.image_size,
        mean=config.augmentation.mean,
        std=config.augmentation.std,
        static_augmentation=config.augmentation.static_val_augmentation,
    )

    train_dataset = AIGCManifestDataset(
        config.data.train_manifest,
        split=config.data.train_split,
        root_dir=config.data.root_dir,
        transform=train_transform,
    )
    val_dataset = None
    if config.data.val_manifest is not None:
        val_dataset = AIGCManifestDataset(
            config.data.val_manifest,
            split=config.data.val_split,
            root_dir=config.data.root_dir,
            transform=eval_transform,
        )

    if is_distributed() and config.data.balanced_sampling:
        train_sampler = DistributedWeightedSampler(train_dataset, replacement=True)
    elif is_distributed():
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
    elif config.data.balanced_sampling:
        train_sampler = make_weighted_sampler(train_dataset)
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        persistent_workers=config.data.persistent_workers and config.data.num_workers > 0,
    )
    val_loader = None
    if val_dataset is not None:
        val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_distributed() else None
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.data.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=config.data.num_workers,
            pin_memory=config.data.pin_memory,
            persistent_workers=config.data.persistent_workers and config.data.num_workers > 0,
        )

    if get_rank() == 0:
        print(
            f"train records={len(train_dataset)} steps_per_epoch={len(train_loader)} "
            f"batch_size={config.data.batch_size}"
        )
        if val_dataset is not None and val_loader is not None:
            print(f"val records={len(val_dataset)} steps={len(val_loader)} batch_size={config.data.batch_size}")

    model = MICVDualStreamEnsemble.from_config(asdict(config.model)).to(device)
    if distributed_started:
        device_ids = [torch.cuda.current_device()] if torch.cuda.is_available() else None
        model = DistributedDataParallel(model, device_ids=device_ids)

    optimizer = torch.optim.AdamW(
        build_param_groups(
            model,
            backbone_lr=config.training.learning_rate,
            head_lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
    )
    steps_per_epoch = math.ceil(len(train_loader) / config.training.gradient_accumulation_steps)
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        epochs=config.training.epochs,
        steps_per_epoch=max(1, steps_per_epoch),
        warmup_epochs=config.training.warmup_epochs,
        min_learning_rate=config.training.min_learning_rate,
    )
    loss_fn = CombinedMICVLoss(
        fused_loss=BinaryFocalLossWithProbabilities(alpha=0.5, gamma=2.0),
        stream_loss=BinaryFocalLossWithLogits(alpha=0.5, gamma=2.0),
        fused_weight=1.0,
        stream_weight=config.training.auxiliary_stream_loss_weight,
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=config.training.output_dir,
        amp=config.training.amp,
        amp_dtype=config.training.amp_dtype,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        clip_grad_norm=config.training.clip_grad_norm,
        log_every_steps=config.training.log_every_steps,
        validate_every_epochs=config.training.validate_every_epochs,
        swa_enabled=config.swa.enabled,
        swa_start_epoch=config.swa.start_epoch,
        swa_learning_rate=config.swa.learning_rate,
    )

    resume_path = args.resume or config.training.resume_from
    start_epoch = 0
    if resume_path is not None:
        start_epoch = load_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=trainer.scaler,
            map_location=device,
        )
    trainer.fit(config.training.epochs, start_epoch=start_epoch)


if __name__ == "__main__":
    main()
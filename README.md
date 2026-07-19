# AIGC Detection

An experimental PyTorch implementation of the architecture described in MICV_framework.md.

The original MICV authors have not released code or model weights. This implementation follows the "NTIRE 2026: Challenge on Robust AI-Generated Image Detection in the Wild" paper where possible, including:

* Two DINOv3 committee streams
* Four backbones in stream 1 and two in stream 2
* Per-stream projections and classifier heads
* Late probability averaging
* Focal loss, AdamW, warmup with cosine decay, and SWA
* ROC AUC validation

<img src="Arch.png" width="850">

## Installation

The project targets Python 3.13. Direct dependencies are exactly pinned in
`pyproject.toml` and mirrored in `constraints/py313-cu130.txt`. The PyTorch pair is
pinned to the CUDA 13.0 wheels configured under `tool.uv.sources`.

Using `uv`:

```powershell
uv sync --all-extras
```

Using `pip`:

```powershell
python -m pip install --upgrade pip
python -m pip install torch==2.12.1 torchvision==0.27.1 `
  --index-url https://download.pytorch.org/whl/cu130
python -m pip install -e ".[large-data,dev]" `
  -c constraints/py313-cu130.txt
```

Update the project metadata and constraint file together when changing a
dependency; a regression test rejects unpinned or divergent entries.

Dataset construction is handled by the separate
[`databuilder`](https://github.com/O-J1/databuilder) repository. Install it from
source in the same environment:

```powershell
git clone https://github.com/O-J1/databuilder.git ..\databuilder
python -m pip install -e "..\databuilder[embed,viz,dev]"
```

## Prepare the Data

Use `databuilder` to download or scan sources, filter invalid images, deduplicate,
embed, cluster, balance, and assign splits. Start from its example TOML, edit the
dataset entries and paths, then run a dry run before the real build:

```powershell
Copy-Item ..\databuilder\examples\build.example.toml .\build.toml
databuilder run --config .\build.toml --dry-run
databuilder run --config .\build.toml
```

The final manifest is written to
`<work_dir>/artifacts/manifest/manifest.parquet` (and optionally CSV). See the
[`databuilder` documentation](https://github.com/O-J1/databuilder#readme) for local
sources, Hugging Face sources, filtering, balancing, distributed runs, and the
viewer.

The detector accepts CSV and Parquet manifests. Training and evaluation require
`path`, `label`, and `split` columns. Labels may be `0`/`1` or names such as
`real`, `fake`, `ai`, and `generated`. Relative image paths resolve against
`data.root_dir`, or against the manifest's directory when `root_dir` is unset.
Parquet loading uses a PyArrow scanner with split pushdown and column projection;
`data.manifest_metadata_columns` controls the optional columns retained in memory.
For corrupted inputs, `data.bad_image_policy: raise` is the safe default;
`exclude` pre-validates and reports skipped records, while legacy `zero` mode emits
warnings and exposes corruption counts instead of silently substituting images.

For a quick folder-to-CSV conversion without the full data pipeline, the legacy
helper remains available:

```powershell
python scripts/build_manifest.py --root D:\datasets\micv --output manifests\train.csv
```

## Configure and Train

The checked-in configs are templates; their manifest paths are not bundled data.
Set the following fields before training:

```yaml
data:
  train_manifest: D:/datasets/micv/work/artifacts/manifest/manifest.parquet
  val_manifest: D:/datasets/micv/work/artifacts/manifest/manifest.parquet
  root_dir: null
  train_split: train
  val_split: val
```

Train or resume from a checkpoint:

```powershell
python scripts/train.py --config configs/local_train.yaml
python scripts/train.py --config configs/local_train.yaml --resume outputs/local_train/latest.pt
```

`latest.pt` contains the primary model, optimizer, scheduler, scaler, best ROC AUC,
SWA averaged parameters and scheduler, per-rank random-number-generator states,
and resolved backbone revisions. Resume therefore continues model selection and
SWA averaging rather than reinitializing them.

`configs/local_train.yaml` creates four and two independent copies of the same
DINOv3 backbone. The checked-in Hugging Face models use immutable commit revisions
and `trust_remote_code: false`; the resolved revision is recorded in checkpoints
and checked when loading them. Enable remote code only for a repository that
requires it, and keep an immutable revision in that experiment config. For a
trainer-only check without downloading DINOv3, set `model.use_dummy_backbone: true`.

### Model choices

- Use explicit `backbones` lists for heterogeneous committees, as in
  `configs/local_2gpu.yaml`.
- `token_concat_attention` keeps token-level fusion and requires matching token
  counts. `pooled_concat_mlp` pools each slot first and supports different token
  layouts and hidden sizes.
- The final prediction is the mean of the two stream probabilities.

### Distributed training

The two-GPU profile uses DDP and NCCL on Linux. Each process holds a complete model
replica, and `data.batch_size` is per process.

```bash
torchrun --standalone --nproc_per_node=2 scripts/train.py --config configs/local_2gpu.yaml
```

For a cluster, use `configs/cluster.yaml` as a base and launch through `torchrun`
or a wrapper that sets `RANK`, `WORLD_SIZE`, and `LOCAL_RANK`. The profile does not
choose the number of GPUs; the launcher does.

## Evaluate and Predict

The evaluation manifest defaults to `data.val_manifest`; override it with
`--manifest` when needed:

```powershell
python scripts/evaluate.py `
  --config configs/local_train.yaml `
  --checkpoint outputs/local_train/best.pt
```

Prediction accepts an image, an image directory, or a CSV/Parquet manifest and
writes CSV results. The config must describe the architecture used by the
checkpoint.

```powershell
python scripts/predict.py `
  --config configs/local_train.yaml `
  --checkpoint outputs/local_train/best.pt `
  --input D:\datasets\images `
  --output outputs\predictions.csv
```

## Decisions where paper left unspecified

- Feature aggregation: concatenate slot tokens by default, with an optional pool-first concatenation mode for heterogeneous committees.
- Stream fusion: average sigmoid probabilities, matching the diagram.
- Training loss: fused focal loss on the averaged probability (computed in fp32 for AMP safety) plus per-stream focal-with-logits auxiliary losses (`training.auxiliary_stream_loss_weight`, default 0.5) so each committee stream is directly supervised.
- Projection latent dimension: 768 in the repeated-copy local baseline, 1024 in the heterogeneous 2-GPU and cluster profiles.
- SWA: final 2-3 epochs by default, annealed to `swa.learning_rate` at the start of the SWA phase. The averaged model is evaluated on the validation set after training and saved to `swa_model.pt` in the standard checkpoint format, so `scripts/predict.py --checkpoint outputs/.../swa_model.pt` works directly.
- Validation: two passes per epoch — a clean resize-only pass that drives `best.pt` selection, and an optional static degraded pass (`augmentation.static_val_augmentation`, severity via `augmentation.static_val_severity`) reported alongside for robustness tracking.

## Optional Training Extras

- Multi-resolution training: set `data.train_image_sizes` (e.g., `[384, 448, 512]`) to resize each training batch to a randomly sampled square resolution. Evaluation always uses `data.image_size`.
- Per-backbone augmented views: set `model.per_backbone_views: true` to create one distinct augmentation per committee slot. The default four-slot and two-slot streams therefore receive six non-overlapping views. Evaluation still uses a single shared view.
- Inference TTA: `scripts/predict.py --tta hflip` averages the fused probability over the original and horizontally flipped input.

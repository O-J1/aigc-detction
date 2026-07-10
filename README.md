# AIGC Detection

An experimental PyTorch implementation of the architecture described in MICV_framework.md.

The original authors have not released code or model weights. This implementation follows the paper where possible, including:

* Two DINOv3 committee streams
* Four backbones in stream 1 and two in stream 2
* Per-stream projections and classifier heads
* Late probability averaging
* Focal loss, AdamW, warmup with cosine decay, and SWA
* ROC AUC validation

<img src="Arch.png" width="850">

## Installation

Clone this repository, then create a virtual environment and activate it. Then install `databuilder` in editable mode (its needed for balancing and filtering data):

```py
python -m pip install --upgrade pip
python -m pip install -e "databuilder[embed,viz] @ git+https://github.com/O-J1/databuilder.git"
```


## Data Manifest

Use Parquet for large data, CSV is supported for testing reasons. Required columns:

```text
path,label,split
```

Recommended columns for the larger corpus:

```text
source_tier,source_dataset,generator,task_type,width,height,md5,group_id
```

Labels may be numeric (`0`, `1`) or text (`real`, `fake`, `ai`, `generated`). Relative paths are resolved against `data.root_dir` when set, otherwise against the manifest directory.

Manifest construction preflights each discovered image by loading it with Pillow. Invalid or inaccessible images are skipped by default, and each emitted row includes an `md5` content hash used as the stable identity for static validation augmentation. Pass `--no-verify-images` only when you intentionally want to skip build-time image validation.

Folder layouts can be converted with:

```powershell
python scripts/build_manifest.py --root D:\datasets\micv --output manifests/train.csv
```

The builder auto-detects these common layouts:

```text
root/train/real        root/real/train       root/real
root/train/fake        root/fake/train       root/fake
root/val/real          root/real/val
root/val/fake          root/fake/val
```

You can also point at explicit directories:

```powershell
python scripts/build_manifest.py `
	--root D:\datasets\micv `
	--real-dirs D:\data\photos_real `
	--fake-dirs D:\data\generated_a D:\data\generated_b `
	--binary-split train `
	--output manifests/train.csv
```

Use `--layout split-class`, `--layout class-split`, or `--layout binary-dirs` when auto-detection is not specific enough. Use `--real-names` and `--fake-names` if your folders are named differently.

For mixed sources where some folders must stay in a specific split and other folders should be randomly split, use a hybrid YAML spec:

```yaml
root: D:/datasets/micv
seed: 42
random_train_fraction: 0.8
random_split_unit: leaf-folder
sources:
  - path: heldout_real
    label: real
    split: val
    source_dataset: heldout_real
  - path: heldout_fake
    label: fake
    split: val
    generator: heldout_generator
    source_dataset: heldout_fake
  - path: training_real
    label: real
    split: train
    keep_percent: 100
    min_per_leaf_folder: 1
  - path: pooled_real
    label: real
    split: random
    keep_percent: 25
    min_per_leaf_folder: 1
  - path: pooled_fake
    label: fake
    split: random
    generator: mixed_local_aigc
    keep_percent: 50
    min_per_leaf_folder: 1
```

Build the combined manifest with:

```powershell
python scripts/build_manifest.py --spec manifests/hybrid.yaml --output manifests/hybrid.csv
```

The emitted `split` column remains the source of truth. Forced train and validation sources are written directly to those splits; random sources are shuffled with the configured seed and assigned to `train` or `val`. Use `random_split_unit: leaf-folder` when folders contain related images that should not cross the train/validation boundary.

## Local Smoke Training

The local config uses repeated independent copies of `facebook/dinov3-vitb16-pretrain-lvd1689m`. Set `data.train_manifest` and `data.val_manifest` before running against real data.

```powershell
python scripts/train.py --config ./configs/local_train.yaml
```

For a trainer-only check without Hugging Face access, set `model.use_dummy_backbone: true` in the local config.

## Model Configuration

Backbone slots are configured per stream. Use explicit `backbones` entries when each slot should load a different DINOv3 variant:

```yaml
model:
  latent_dim: 1024
  token_pooling: attention
  stream_fusion: token_concat_attention
  streams:
    - name: stream1
      backbones:
        - model_name_or_path: facebook/dinov3-vits16-pretrain-lvd1689m
          freeze: false
        - model_name_or_path: facebook/dinov3-vits16plus-pretrain-lvd1689m
          freeze: false
        - model_name_or_path: facebook/dinov3-vitb16-pretrain-lvd1689m
          freeze: false
        - model_name_or_path: facebook/dinov3-vitl16-pretrain-lvd1689m
          freeze: false
    - name: stream2
      backbones:
        - model_name_or_path: facebook/dinov3-vitb16-pretrain-lvd1689m
          freeze: false
        - model_name_or_path: facebook/dinov3-vitl16-pretrain-lvd1689m
          freeze: false
```

Use `repeat` plus `backbone` for the repeated-copy MICV interpretation:

```yaml
model:
  latent_dim: 768
  token_pooling: attention
  stream_fusion: token_concat_attention
  streams:
    - name: stream1
      repeat: 4
      backbone:
        model_name_or_path: facebook/dinov3-vitb16-pretrain-lvd1689m
        freeze: false
    - name: stream2
      repeat: 2
      backbone:
        model_name_or_path: facebook/dinov3-vitb16-pretrain-lvd1689m
        freeze: false
```

`stream_fusion: token_concat_attention` preserves the original token fusion path and requires matching token counts across slots. `stream_fusion: pooled_concat_mlp` pools each slot first, then concatenates pooled vectors, which is useful for mixed hidden sizes, different token layouts, or ConvNeXt-style experiments.

## 2 GPU Trial

```powershell
torchrun --nproc_per_node=2 scripts/train.py --config configs/local_2gpu.yaml
```

## Cluster Run

Use `configs/cluster.yaml` as the base profile. The launcher command depends on the environment, but the script is designed for `torchrun` or a SLURM wrapper that sets `RANK`, `WORLD_SIZE`, and `LOCAL_RANK`.
The cluster profile uses `training.amp_dtype: bf16` for A100 mixed precision without gradient scaling.

## Guesses Chosen Where Unspecified

- Feature aggregation: concatenate slot tokens by default, with an optional pool-first concatenation mode for heterogeneous committees.
- Stream fusion: average sigmoid probabilities, matching the diagram.
- Training loss: fused focal loss on the averaged probability (computed in fp32 for AMP safety) plus per-stream focal-with-logits auxiliary losses (`training.auxiliary_stream_loss_weight`, default 0.5) so each committee stream is directly supervised.
- Projection latent dimension: 768 in the repeated-copy local baseline, 1024 in the heterogeneous 2-GPU and cluster profiles.
- SWA: final 2-3 epochs by default, annealed to `swa.learning_rate` at the start of the SWA phase. The averaged model is evaluated on the validation set after training and saved to `swa_model.pt` in the standard checkpoint format, so `scripts/predict.py --checkpoint outputs/.../swa_model.pt` works directly.
- Validation: two passes per epoch — a clean resize-only pass that drives `best.pt` selection, and an optional static degraded pass (`augmentation.static_val_augmentation`, severity via `augmentation.static_val_severity`) reported alongside for robustness tracking.

## Optional Training Extras

- Multi-resolution training: set `data.train_image_sizes` (e.g., `[384, 448, 512]`) to resize each training batch to a randomly sampled square resolution. Evaluation always uses `data.image_size`.
- Per-backbone augmented views: set `model.per_backbone_views: true` to feed each committee slot a differently-augmented view of the same image, forcing ensemble diversity for repeated-copy committees. Evaluation still uses a single shared view.
- Inference TTA: `scripts/predict.py --tta hflip` averages the fused probability over the original and horizontally flipped input.

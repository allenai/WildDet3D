# WildDet3D Training Guide

WildDet3D uses a 3-stage training pipeline that progressively scales from indoor/outdoor benchmarks to diverse in-the-wild data with all prompt types.

## Prerequisites

- Python 3.11+
- PyTorch 2.0+
- [vis4d](https://github.com/SysCV/vis4d) framework
- 8 GPUs per node (A100 80GB recommended)

Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Preparation

All per-dataset prep (download, frame extraction scripts, HDF5 packing, mask annotations, pretrained checkpoints) lives in **[TRAINING_DATA.md](TRAINING_DATA.md)**. Follow that doc first; everything below assumes your `data/` and `pretrained/` dirs match its expected layout.

## 3-Stage Training Pipeline

### Overview

```
Stage 1: Omni3D Canonical (12 epochs)
   |  SAM3 pretrained -> train on Omni3D with canonical rotation
   |  Collator: 5-mode (text + box geometry prompts)
   v
Stage 2: All-Data Dense Finetune (12 epochs)
   |  Stage 1 ckpt -> 8 datasets (human-only ITW + V3Det)
   |  Collator: 5-mode (text + box geometry prompts; no mask)
   v
Stage 3: High-Quality Mix-Prompt Finetune (3 epochs)
   |  Stage 2 ckpt -> Omni3D 90% + ITW human 10%
   |  Collator: 5-mode mask_pt (text + box + point; points sampled from masks), lr=1e-4
   v
Final WildDet3D Model
```

### Stage 1: Omni3D Canonical

Train on Omni3D only with canonical rotation (dims W<=L, yaw [0, pi)):

```bash
# Single node (8 GPUs)
MIXED_PRECISION=bf16 vis4d fit \
    --config configs/training/stage1_omni3d.py \
    --gpus 8

# 2 nodes (16 GPUs)
MIXED_PRECISION=bf16 vis4d fit \
    --config configs/training/stage1_omni3d.py \
    --gpus 8 --num_nodes 2
```

**Output:** `vis4d-workspace/wilddet3d_stage1_omni3d/checkpoints/epoch=11-step=XXXX.ckpt`

> **Skip Stage 1 — download the released checkpoint:** to start from a pretrained Stage 1 ckpt instead of running 12 epochs, grab it from HuggingFace:
> ```bash
> huggingface-cli download allenai/WildDet3D wilddet3d_stage1_omni3d_12e_v1.0.pt --local-dir ckpt/
> ```
> Same format as the released `wilddet3d_alldata_all_prompt_v1.0.pt` (state_dict + epoch/step + hparams, ~4.7 GB, optimizer state stripped). Pass it as `--ckpt ckpt/wilddet3d_stage1_omni3d_12e_v1.0.pt` to the Stage 2 command below.

### Stage 2: All-Data Dense Finetune

Load Stage 1 checkpoint, train on 8 datasets with the plain 5-mode collator (text + box prompts, no mask):

```bash
MIXED_PRECISION=bf16 vis4d fit \
    --config configs/training/stage2_alldata.py \
    --gpus 8 --num_nodes 4 \
    --ckpt vis4d-workspace/wilddet3d_stage1_omni3d/checkpoints/epoch=11-step=XXXX.ckpt
```

8 datasets (human-only) with proportions:
| Dataset | Proportion |
|---------|-----------|
| Omni3D | 80% |
| CA-1M | 3% |
| Waymo | 2% |
| 3EED-det | 1% |
| 3EED-ref | 1% |
| FoundationPose | 6% |
| ITW human | 5% |
| V3Det human | 2% |

Uses the plain `5mode` collator (text + box geometry prompts); point prompts sampled from masks are introduced in Stage 3.

**Output:** `vis4d-workspace/wilddet3d_stage2_alldata/checkpoints/epoch=11-step=XXXX.ckpt`

> **Skip Stage 2 — download the released checkpoint:** to start the Stage 3 finetune from a pretrained Stage 2 ckpt instead of running 12 epochs on 4 nodes, grab it from HuggingFace:
> ```bash
> huggingface-cli download allenai/WildDet3D wilddet3d_stage2_alldata_12e_v1.0.pt --local-dir ckpt/
> ```
> Same format as the released `wilddet3d_alldata_all_prompt_v1.0.pt` (state_dict + epoch/step + hparams, ~4.7 GB, optimizer state stripped). Pass it as `--ckpt ckpt/wilddet3d_stage2_alldata_12e_v1.0.pt` to the Stage 3 command below.

### Stage 3: High-Quality Mix-Prompt Finetune

Load Stage 2 checkpoint, short finetune with only Omni3D + ITW human:

```bash
MIXED_PRECISION=bf16 vis4d fit \
    --config configs/training/stage3_mix_box_point_text_ft.py \
    --gpus 8 --num_nodes 4 \
    --ckpt vis4d-workspace/wilddet3d_stage2_alldata/checkpoints/epoch=11-step=XXXX.ckpt
```

2 datasets (simplified, high-quality):
| Dataset | Proportion |
|---------|-----------|
| Omni3D | 90% |
| ITW human | 10% |

Key differences from Stage 2: only 2 high-quality datasets, short 3 epochs with step_1=1, step_2=2.

**Output:** Final WildDet3D checkpoint.

## Configuration Options

### Override via Command Line

All config parameters can be overridden:
```bash
# Change batch size
vis4d fit --config configs/training/stage1_omni3d.py \
    --config.params.samples_per_gpu=2

# Change learning rate (4x for 4 nodes)
vis4d fit --config configs/training/stage2_alldata.py \
    --config.params.base_lr=4e-4

# Enable mixed precision
MIXED_PRECISION=bf16 vis4d fit --config configs/training/stage1_omni3d.py
```

### Key Settings

| Parameter | Stage 1 | Stage 2 | Stage 3 | Description |
|-----------|---------|---------|---------|-------------|
| `num_epochs` | 12 | 12 | 3 | Training epochs |
| `base_lr` | 1e-4 | 1e-4 | 1e-4 | Base learning rate |
| `step_1/step_2` | 8/10 | 6/9 | 1/2 | LR schedule milestones |
| `backbone_freeze_blocks` | 28 | 28 | 28 | SAM3 ViT blocks to freeze (of 32) |
| `canonical_rotation` | True | True | True | Normalize dims W<=L, yaw [0, pi) |
| `collator` | 5mode | 5mode | mask_pt | Prompt training strategy |
| `presence_loss_weight` | 20.0 | 5.0 | 5.0 | Presence prediction loss weight |
| `datasets` | 1 (Omni3D) | 8 (all, human) | 2 (Omni3D + ITW) | Number of training datasets |

### Multi-Node Training

For multi-node distributed training:
```bash
# Node 0 (master)
torchrun --nnodes=4 --node-rank=0 --nproc-per-node=8 \
    --rdzv_backend=static --rdzv_endpoint=<MASTER_IP>:29401 \
    -m vis4d.engine.run fit \
    --config configs/training/stage2_alldata.py \
    --gpus 8 --nodes 4 --ckpt <checkpoint>

# Node 1
torchrun --nnodes=4 --node-rank=1 --nproc-per-node=8 \
    --rdzv_backend=static --rdzv_endpoint=<MASTER_IP>:29401 \
    -m vis4d.engine.run fit \
    --config configs/training/stage2_alldata.py \
    --gpus 8 --nodes 4 --ckpt <checkpoint>
```

## Evaluation

After training, evaluate with:
```bash
# InTheWild text evaluation
vis4d test --config configs/eval/in_the_wild/text.py \
    --gpus 1 --ckpt <path_to_checkpoint>

# InTheWild box prompt evaluation
vis4d test --config configs/eval/in_the_wild/box_prompt.py \
    --gpus 1 --ckpt <path_to_checkpoint>
```

See [EVALUATION.md](EVALUATION.md) for detailed evaluation instructions.

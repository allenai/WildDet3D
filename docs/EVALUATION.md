# WildDet3D Evaluation Guide

Evaluation uses the [vis4d](https://github.com/SysCV/vis4d) framework. Follow the [vis4d documentation](https://vis4d.readthedocs.io/) for general setup.

The Omni3D / ScanNet / Argoverse 2 evaluation protocol — dataset
splits, ODS metric, Base/Novel groupings — follows the prior work
**3D-MOOD** ([cvg/3D-MOOD](https://github.com/cvg/3D-MOOD)) so the
numbers reported here are directly comparable to theirs.

## Quick Start

```bash
# General format
vis4d test --config configs/eval/<benchmark>/<mode>.py \
    --gpus 1 --ckpt ckpt/wilddet3d.pt
```

## Metrics

| Metric | Description |
|---|---|
| **AP** | Average Precision (COCO-style 3D IoU). For ScanNet / Argoverse2 / In-the-Wild / DROID the evaluator switches to center-distance ("dist" / "prox") matching and additionally reports the ODS family below. |
| **ATE / ASE / AOE** | Mean translation / scale / orientation error of matched detections. |
| **ODS / ODS_Sym / ODS_Canonical** | nuScenes-style Open Detection Score: `(3*mAP + (1 - mATE) + (1 - mASE) + (1 - mAOE)) / 6`. The `_Canonical` variant uses canonical-rotation AOE (matches the model's training target) and is the primary ranking metric for open-vocabulary benchmarks. |
| **AP_\<Subset>** | Omni3D reports a separate AP per Omni3D sub-dataset (KITTI / nuScenes / SUNRGBD / Hypersim / ARKitScenes / Objectron). |
| **Base / Novel** | Most evaluators also produce a Base / Novel split. The "Base" set is the canonical-category list of each benchmark (e.g. 15 SUNRGBD-style indoor categories for ScanNet, 11 AV2 driving categories for Argoverse2, the 45 single-word frequent targets for DROID); everything else is Novel. |

## Benchmarks

### Omni3D

Training benchmark. Evaluates on KITTI, nuScenes, SUNRGBD, Hypersim, ARKitScenes, Objectron.

| Mode | Config | Description |
|---|---|---|
| Text | `configs/eval/omni3d/text.py` | Text prompt, monocular depth |
| Text + Depth | `configs/eval/omni3d/text_with_depth.py` | Text prompt, with GT depth |
| Box Prompt | `configs/eval/omni3d/box_prompt.py` | GT 2D box as prompt |
| Box Prompt + Depth | `configs/eval/omni3d/box_prompt_with_depth.py` | GT 2D box + GT depth |

```bash
vis4d test --config configs/eval/omni3d/text.py --gpus 1 --ckpt ckpt/wilddet3d.pt
```

### ScanNet (Zero-Shot)

Indoor 3D scene understanding. Zero-shot evaluation (not in training set).

| Mode | Config | Description |
|---|---|---|
| Text | `configs/eval/scannet/text.py` | Text prompt, monocular depth |
| Text + Depth | `configs/eval/scannet/text_with_depth.py` | Text prompt, with GT depth |
| Box Prompt | `configs/eval/scannet/box_prompt.py` | GT 2D box as prompt |
| Box Prompt + Depth | `configs/eval/scannet/box_prompt_with_depth.py` | GT 2D box + GT depth |

```bash
vis4d test --config configs/eval/scannet/text.py --gpus 1 --ckpt ckpt/wilddet3d.pt
```

### Argoverse2 (Zero-Shot)

Outdoor autonomous driving. Zero-shot evaluation.

| Mode | Config | Description |
|---|---|---|
| Text | `configs/eval/argoverse/text.py` | Text prompt, monocular depth |
| Text + Depth | `configs/eval/argoverse/text_with_depth.py` | Text prompt, with GT depth |
| Box Prompt | `configs/eval/argoverse/box_prompt.py` | GT 2D box as prompt |
| Box Prompt + Depth | `configs/eval/argoverse/box_prompt_with_depth.py` | GT 2D box + GT depth |

```bash
vis4d test --config configs/eval/argoverse/text.py --gpus 1 --ckpt ckpt/wilddet3d.pt
```

### Stereo4D (Zero-Shot)

Dynamic 3D object detection from stereo video.

| Mode | Config | Description |
|---|---|---|
| Text | `configs/eval/stereo4d/text.py` | Text prompt, monocular depth |
| Text + Depth | `configs/eval/stereo4d/text_with_depth.py` | Text prompt, with GT depth |
| Box Prompt | `configs/eval/stereo4d/box_prompt.py` | GT 2D box as prompt |
| Box Prompt + Depth | `configs/eval/stereo4d/box_prompt_with_depth.py` | GT 2D box + GT depth |

```bash
vis4d test --config configs/eval/stereo4d/text.py --gpus 1 --ckpt ckpt/wilddet3d.pt
```

### In-the-Wild

Large-scale in-the-wild evaluation with diverse categories.

| Mode | Config | Description |
|---|---|---|
| Text | `configs/eval/in_the_wild/text.py` | Text prompt, monocular depth |
| Text + Depth | `configs/eval/in_the_wild/text_with_depth.py` | Text prompt, with GT depth |
| Box Prompt | `configs/eval/in_the_wild/box_prompt.py` | GT 2D box as prompt |
| Box Prompt + Depth | `configs/eval/in_the_wild/box_prompt_with_depth.py` | GT 2D box + GT depth |

```bash
vis4d test --config configs/eval/in_the_wild/text.py --gpus 1 --ckpt ckpt/wilddet3d.pt
```

## Config Naming Convention

```
configs/eval/<benchmark>/<prompt>[_with_depth].py
```

| Component | Options | Meaning |
|---|---|---|
| `<benchmark>` | `omni3d`, `scannet`, `argoverse`, `stereo4d`, `in_the_wild` | Evaluation dataset |
| `<prompt>` | `text`, `box_prompt` | Text prompt vs. GT 2D box as prompt |
| `with_depth` | present or absent | Whether GT depth is provided to the model |

## Data Setup

Evaluation datasets should follow the [vis4d data format](https://vis4d.readthedocs.io/). Default data paths:

```
data/
├── omni3d/                    # Omni3D (KITTI, nuScenes, SUNRGBD, etc.)
├── scannet/                   # ScanNet val
├── argoverse/                 # Argoverse2
├── stereo4d/                  # Stereo4D
└── in_the_wild/               # WildDet3D in-the-wild annotations
```

### In-the-Wild / Stereo4D eval annotations

These aren't checked into the repo; grab them from HuggingFace before running the corresponding eval config:

```bash
mkdir -p data/in_the_wild/annotations
# In-the-Wild v3 val
wget https://huggingface.co/datasets/allenai/WildDet3D-Data/resolve/main/annotations/InTheWild_v3_val.json \
    -O data/in_the_wild/annotations/InTheWild_v3_val.json

# Stereo4D val + test
wget https://huggingface.co/datasets/allenai/WildDet3D-Stereo4D-Bench-Images/resolve/main/annotations/Stereo4D_val.json \
    -O data/in_the_wild/annotations/Stereo4D_val.json
wget https://huggingface.co/datasets/allenai/WildDet3D-Stereo4D-Bench-Images/resolve/main/annotations/Stereo4D_test.json \
    -O data/in_the_wild/annotations/Stereo4D_test.json
```

The small `*_class_map.json` files are already in the repo. See [TRAINING_DATA.md](TRAINING_DATA.md) for Omni3D / ScanNet / Argoverse 2 eval data (via 3D-MOOD's `docs/DATA.md`).

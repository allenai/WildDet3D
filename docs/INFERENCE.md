# WildDet3D Inference Guide

## Overview

WildDet3D supports 5 prompt modes for 3D object detection:

| Mode | Prompt Input | Behavior | Use Case |
|---|---|---|---|
| **Text** | `input_texts=["car", "person"]` | Detect all instances of given categories | Open-vocabulary detection |
| **Visual** | `input_boxes` + `prompt_text="visual"` | Use box as visual example, find similar objects | One-to-many matching |
| **Visual+Label** | `input_boxes` + `prompt_text="visual: car"` | Visual example with category constraint | Filtered one-to-many |
| **Geometric** | `input_boxes` + `prompt_text="geometric"` | Lift the given 2D box to 3D | **One-to-one**, box prompt |
| **Geometric+Label** | `input_boxes` + `prompt_text="geometric: car"` | Lift 2D box to 3D with category label | **One-to-one** with label |

Point prompts (`input_points`) work with any `prompt_text`. In `geometric` mode they are also one-to-one (the single output is the proposal whose box contains the most positive points, tie-broken by score).

## Setup

```python
from wilddet3d import build_model, preprocess
import numpy as np
from PIL import Image

# Build model
model = build_model(
    checkpoint="ckpt/wilddet3d.pt",
    score_threshold=0.3,
    score_3d_threshold=0.1,
    skip_pretrained=True,       # faster loading from full checkpoint
    # use_depth_input_test=True,  # enable ONLY if you will pass `depth_gt`
)

# Load and preprocess image
image = np.array(Image.open("image.jpg")).astype(np.float32)

# Option A: unknown intrinsics (model uses its predicted K internally)
data = preprocess(image, intrinsics=None)

# Option B: known camera intrinsics
# intrinsics = np.load("intrinsics.npy")  # (3, 3)
# data = preprocess(image, intrinsics)

# Option C: known intrinsics AND a measured depth map
# depth = np.load("depth.npy")  # (H, W) float32 meters, same resolution as image
# data = preprocess(image, intrinsics, depth=depth)
```

## `build_model()` Parameters

| Parameter | Default | Description |
|---|---|---|
| `checkpoint` | (required) | Path to WildDet3D `.ckpt` file |
| `sam3_checkpoint` | `"pretrained/sam3/sam3_detector.pt"` | SAM3 pretrained weights (ignored when `skip_pretrained=True`) |
| `score_threshold` | `0.3` | 2D score floor for text / visual prompts (see [Score Filtering](#score-filtering)) |
| `score_3d_threshold` | `0.1` | 3D score floor for text / visual prompts |
| `nms` | `True` | Apply NMS on 2D boxes before the score filter |
| `iou_threshold` | `0.6` | IoU threshold for NMS |
| `device` | `"cuda"` | Device to load model on |
| `backbone_freeze_blocks` | `28` | SAM3 ViT blocks to freeze (not relevant for inference) |
| `lingbot_encoder_freeze_blocks` | `21` | LingBot-Depth encoder blocks frozen (not relevant for inference) |
| `ambiguous_rotation` | `False` | Use ambiguous (sin/cos) rotation head |
| `canonical_rotation` | `False` | Use canonical rotation representation |
| `use_depth_input_test` | `False` | If `True`, the geometry backend consumes a measured depth input in addition to the image |
| `use_predicted_intrinsics` | `False` | Use predicted K (ignore the passed intrinsics) |
| `skip_pretrained` | `False` | Build model without loading SAM3 / LingBot weights; use this when the full checkpoint already contains them |

## `preprocess()` Parameters

| Parameter | Default | Description |
|---|---|---|
| `image` | (required) | RGB image as numpy array `(H, W, 3)`. Any numeric dtype; cast to `float32` internally. |
| `intrinsics` | `None` | Camera intrinsics `(3, 3)`. `None` creates a placeholder; the model then uses its predicted K. |
| `depth` | `None` | Optional depth map `(H, W)`, values in **metres**, same resolution as `image`. When provided, the depth is resized + center-padded to the model's `input_hw` using the same `ResizeDepthMaps(nearest)` + `CenterPadDepthMaps` transforms eval uses, and the result is returned under `data["depth_gt"]`. |

Returned dict keys:

| Key | Shape / type | Always? |
|---|---|---|
| `images` | `Tensor (1, 3, 1008, 1008)` | yes |
| `intrinsics` | `Tensor (3, 3)` — K for `input_hw` space | yes |
| `original_intrinsics` | `Tensor (3, 3)` — K for the original image | yes |
| `input_hw` | `(1008, 1008)` | yes |
| `original_hw` | `(H, W)` of the input image | yes |
| `padding` | `(left, right, top, bottom)` | yes |
| `depth_gt` | `Tensor (1, 1, 1008, 1008)` metres | **only when `depth=` passed** |

## Input coordinates

User-supplied `input_boxes` and `input_points` are in **original-image pixel coordinates**, not `input_hw`. The wrapper applies the same resize + pad transform preprocess applied to the image, so the prompt lines up with the model's view. The predicted 2D boxes returned in the output are likewise in original-image pixels.

## Prompt Modes

All four examples below use the same `data` dict from [Setup](#setup). Add `depth_gt=data["depth_gt"].cuda()` to any call if you preprocessed with `depth=`.

### 1. Text Prompt (one-to-many per class)

```python
results = model(
    images=data["images"].cuda(),
    intrinsics=data["intrinsics"].cuda()[None],
    input_hw=[data["input_hw"]],
    original_hw=[data["original_hw"]],
    padding=[data["padding"]],
    input_texts=["car", "person", "bicycle"],
    # depth_gt=data["depth_gt"].cuda(),  # include only if preprocess was called with depth=...
)
boxes, boxes3d, scores, scores_2d, scores_3d, class_ids, depth_maps = results
```

### 2. Visual Prompt — Box as exemplar, one-to-many

```python
results = model(
    images=data["images"].cuda(),
    intrinsics=data["intrinsics"].cuda()[None],
    input_hw=[data["input_hw"]],
    original_hw=[data["original_hw"]],
    padding=[data["padding"]],
    input_boxes=[[100, 200, 300, 400]],  # pixel xyxy in the original image
    prompt_text="visual",                 # or "visual: car" for a label-constrained version
    # depth_gt=data["depth_gt"].cuda(),
)
```

### 3. Geometric Prompt — Box, one-to-one

Returns **exactly one** prediction per input box: the proposal with the highest 2D score among the top-10 by IoU with the prompt box. Falls back to overall argmax when no proposal overlaps the prompt.

```python
results = model(
    images=data["images"].cuda(),
    intrinsics=data["intrinsics"].cuda()[None],
    input_hw=[data["input_hw"]],
    original_hw=[data["original_hw"]],
    padding=[data["padding"]],
    input_boxes=[[100, 200, 300, 400]],
    prompt_text="geometric",              # or "geometric: car"
    # depth_gt=data["depth_gt"].cuda(),
)
```

### 4. Point Prompt

Each point is `(x, y, label)` where `label=1` is positive, `label=0` is negative. In `prompt_text="geometric"` mode, point prompts are also one-to-one — the output is the proposal whose 2D box contains the most positive points, tie-broken by score.

```python
results = model(
    images=data["images"].cuda(),
    intrinsics=data["intrinsics"].cuda()[None],
    input_hw=[data["input_hw"]],
    original_hw=[data["original_hw"]],
    padding=[data["padding"]],
    input_points=[[(150, 250, 1), (200, 300, 0)]],
    prompt_text="geometric",
    # depth_gt=data["depth_gt"].cuda(),
)
```

## Using a measured depth map

Providing a GT depth (LiDAR / stereo / etc.) needs **three things set together**:

```python
# 1. Preprocess the image with the depth map (meters, (H, W), same res as image).
data = preprocess(image, intrinsics, depth=depth_np)

# 2. Build the model with the depth input path enabled.
model = build_model(
    checkpoint="ckpt/wilddet3d.pt",
    skip_pretrained=True,
    use_depth_input_test=True,   # <-- required when passing depth_gt below
)

# 3. Pass depth_gt to every model(...) call.
results = model(
    images=data["images"].cuda(),
    intrinsics=data["intrinsics"].cuda()[None],
    input_hw=[data["input_hw"]],
    original_hw=[data["original_hw"]],
    padding=[data["padding"]],
    input_texts=["car", "person"],
    depth_gt=data["depth_gt"].cuda(),
)
```

Omit all three for monocular inference (the model predicts depth internally via LingBot-Depth).

## Score Filtering

Two thresholds apply for text / visual prompts (geometric returns 1 pred per prompt regardless):

| Threshold | Applied to | Default |
|---|---|---|
| `score_threshold` | combined 2D × 3D score | `0.3` |
| `score_3d_threshold` | standalone 3D confidence | `0.1` |

Both are set via `build_model(...)`. Pass `0.0` to disable a floor.

## Output Format

All outputs are **per-image lists** (outer list has length `B = number of prompts / batch elements`):

| Output | Shape | Description |
|---|---|---|
| `boxes` | `list[Tensor[N, 4]]` | 2D boxes in pixel xyxy (**original image space**) |
| `boxes3d` | `list[Tensor[N, 10]]` | 3D boxes (center_x, center_y, center_z, w, h, l, q_w, q_x, q_y, q_z) in camera coords, metres |
| `scores` | `list[Tensor[N]]` | Combined confidence |
| `scores_2d` | `list[Tensor[N]]` | 2D detection confidence |
| `scores_3d` | `list[Tensor[N]]` | 3D detection confidence |
| `class_ids` | `list[Tensor[N]]` | Class indices (into `input_texts`, or prompt order for box / point prompts) |
| `depth_maps` | `list[Tensor[1, H, W]]` or `None` | Predicted metric depth in metres |

`N` varies per image: in text / visual modes it reflects score-threshold filtering + NMS; in geometric modes `N == 1` per prompt.

### Getting Predicted Intrinsics

```python
results = model(
    ...,
    input_texts=["car"],
    return_predicted_intrinsics=True,
)
boxes, boxes3d, scores, scores_2d, scores_3d, class_ids, depth_maps, predicted_K, confidence_maps = results
# predicted_K: Tensor[B, 3, 3], confidence_maps: Tensor[B, 1, H, W]
```

## Visualization

`draw_3d_boxes` renders 3D wireframes on the **original image** and projects 3D corners via the original-image intrinsics. Pass `data["original_intrinsics"].numpy()` (or the same `intrinsics` you fed to `preprocess`).

```python
from wilddet3d.vis.visualize import draw_3d_boxes

boxes, boxes3d, scores, scores_2d, scores_3d, class_ids, depth_maps = results

draw_3d_boxes(
    image=image.astype(np.uint8),          # original RGB (H, W, 3)
    boxes3d=boxes3d[0],
    intrinsics=data["original_intrinsics"].numpy(),
    scores_2d=scores_2d[0],
    scores_3d=scores_3d[0],
    class_ids=class_ids[0],
    class_names=["car", "person", "bicycle"],
    score_2d_threshold=0.3,                # drop rows below this 2D score (0 disables)
    score_3d_threshold=0.1,                # drop rows below this 3D score (0 disables)
    save_path="output.png",
    # Optional debug overlays (both default off):
    #   predicted 2D boxes (green):
    # boxes_2d=boxes[0],
    # draw_predicted_2d_boxes=True,
    #   user prompt boxes (red) / points (red pos, gray neg):
    # input_boxes=[[100, 200, 300, 400]],
    # input_points=[[(150, 250, 1)]],
    # draw_prompt=True,
)
```

### `draw_3d_boxes()` Parameters

| Parameter | Default | Description |
|---|---|---|
| `image` | (required) | RGB `(H, W, 3)` uint8 — the original image |
| `boxes3d` | (required) | `(N, 10)` tensor/array — camera-frame 3D boxes |
| `intrinsics` | (required) | `(3, 3)` — original-image K (i.e. `data["original_intrinsics"]`) |
| `scores_2d`, `scores_3d`, `class_ids` | `None` | `(N,)` — used for labels and the score filter |
| `class_names` | `None` | List of class names |
| `line_width`, `font_size`, `n_colors`, `score_format`, `near_clip` | sane defaults | 3D wireframe / label rendering knobs |
| `score_2d_threshold` | `0.3` | Drop rows below this 2D score (`0` disables) |
| `score_3d_threshold` | `0.1` | Drop rows below this 3D score (`0` disables) |
| `save_path` | `None` | If given, save the rendered image |
| `boxes_2d` | `None` | `(N, 4)` predicted 2D boxes for the green overlay |
| `draw_predicted_2d_boxes` | `False` | Enable the predicted-2D overlay |
| `input_boxes` / `input_points` | `None` | User prompt geometry for the red overlay |
| `draw_prompt` | `False` | Enable the prompt overlay |

## Faster Inference: BF16 Autocast + `torch.compile`

`optimize_for_inference()` wraps the predictor with `torch.autocast` and (optionally) compiles it with `torch.compile`. On an H100 80GB this gives a **3.0x** speedup over FP32 eager with no detectable change in detection outputs.

```python
from wilddet3d import build_model, optimize_for_inference, preprocess

model = build_model(
    checkpoint="ckpt/wilddet3d_alldata_all_prompt_v1.0.pt",
    skip_pretrained=True,
)
# BF16 autocast + compile (default: max-autotune-no-cudagraphs, ~3.0x speedup)
model = optimize_for_inference(model)
# Equivalent to:
# model = optimize_for_inference(model, dtype="bf16", compile_mode="max-autotune-no-cudagraphs")
# First call triggers compile (~17 min); subsequent calls hit the inductor cache.
results = model(images=..., input_texts=["chair", "table"], ...)
```

### Latency (1008x1008 input, H100 80GB)

| Config | Median | Speedup | Cos sim vs FP32 | Rel L2 vs FP32 |
|---|---|---|---|---|
| FP32 eager (baseline) | 219 ms | 1.00x | — | — |
| BF16 autocast | 132 ms | **1.66x** | 1.0000 | 5e-4 |
| BF16 + `torch.compile("default")` | 83 ms | **2.64x** | 1.0000 | 7e-4 |
| BF16 + `torch.compile("max-autotune-no-cudagraphs")` **(default)** | 73 ms | **3.01x** | 1.0000 | 7e-4 |

Reproduce with `python scripts/benchmark_inference.py`.

### `optimize_for_inference()` Parameters

| Parameter | Default | Description |
|---|---|---|
| `model` | (required) | A `WildDet3DPredictor` (the result of `build_model(...)`). |
| `dtype` | `"bf16"` | Autocast dtype. `"bf16"` (recommended, Ampere+), `"fp16"` (older GPUs, slightly less stable on the depth head), `"fp32"` (no autocast). |
| `compile_mode` | `"max-autotune-no-cudagraphs"` | `torch.compile` mode: `"max-autotune-no-cudagraphs"` (3.0x, ~17 min compile), `"default"` (2.6x, ~2 min compile), or `None` (no compile). |

### Caveats

- **PyTorch autocast preserves op-level precision policy** — `LayerNorm`, `softmax`, `log`, `exp`, `pow`, `reduce_sum` stay in FP32 automatically. This is why detection outputs are bit-equivalent to FP32 in practice. A static `model.half()` cast or naive ONNX FP16 export does **not** preserve this policy and will produce numerically wrong results.
- **CUDA-graph modes are unsupported.** `compile_mode="reduce-overhead"` and `"max-autotune"` (which enable CUDA graph capture) crash during graph capture because the head produces dynamic shapes (NMS output count, boolean-mask indexing in canonical-rotation normalization). Use `"default"` or `"max-autotune-no-cudagraphs"` instead.
- **First-time compile is slow.** Expect ~2 min for `"default"` and ~17 min for `"max-autotune-no-cudagraphs"`. Subsequent runs in the same process are near-instant; across processes the inductor cache speeds up re-compile.
- **One graph break inside the model.** SAM3's geometry encoder calls `pin_memory()` which dynamo can't trace, so we set `torch._dynamo.config.suppress_errors = True` and let that fragment fall back to eager. The surrounding graph still compiles. This adds <1ms vs a hypothetical no-break version.
- **Backwards / training is not the target.** `optimize_for_inference()` assumes `model.eval()` + `torch.no_grad()` and is only validated for inference.

## Batch Inference

```python
data_list = [preprocess(img) for img in images]

import torch
batch_images = torch.stack([d["images"] for d in data_list]).cuda()
batch_intrinsics = torch.stack([d["intrinsics"] for d in data_list]).cuda()

results = model(
    images=batch_images,
    intrinsics=batch_intrinsics,
    input_hw=[d["input_hw"] for d in data_list],
    original_hw=[d["original_hw"] for d in data_list],
    padding=[d["padding"] for d in data_list],
    input_texts=["car", "person"],
    # depth_gt=torch.stack([d["depth_gt"][0] for d in data_list]).cuda(),  # include if depth was preprocessed
)
```

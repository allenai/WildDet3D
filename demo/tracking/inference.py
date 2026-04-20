"""Per-frame WildDet3D inference for tracking pipeline.

Uses the WildDet3D build_model/preprocess API for inference with
geometric box prompts derived from object masks.
"""

import os
import sys
from pathlib import Path

# Ensure project root is in path for wilddet3d imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "third_party" / "sam3"))
sys.path.insert(0, str(project_root / "third_party" / "lingbot_depth"))

from wilddet3d import build_model, preprocess

from . import config


def _resolve_checkpoint():
    """Resolve checkpoint: local if exists, else download from HF Hub."""
    for path in config.LOCAL_CHECKPOINTS:
        if os.path.exists(path):
            return path
    from huggingface_hub import hf_hub_download
    return hf_hub_download(
        repo_id=config.HF_MODEL_REPO,
        filename=config.HF_CKPT_NAME,
    )


def load_model(checkpoint=None, device="cuda"):
    """Load WildDet3D model for tracking inference.

    Args:
        checkpoint: Path to model checkpoint. Auto-resolves if None.
        device: Device string.

    Returns:
        WildDet3DPredictor model
    """
    if checkpoint is None:
        checkpoint = _resolve_checkpoint()

    model = build_model(
        checkpoint=checkpoint,
        score_threshold=config.SCORE_THRESHOLD,
        nms=True,
        iou_threshold=config.NMS_IOU_THRESHOLD,
        device=device,
        skip_pretrained=True,
    )
    return model


def get_mask_bbox(rle):
    """Get bounding box [x1, y1, x2, y2] from RLE mask.

    Args:
        rle: COCO RLE dict with 'counts' and 'size' keys.

    Returns:
        [x1, y1, x2, y2] in pixel coordinates, or None if too small.
    """
    from pycocotools import mask as mask_util

    area = mask_util.area(rle)
    if area < config.MIN_MASK_AREA:
        return None

    bbox = mask_util.toBbox(rle)
    x, y, w, h = bbox
    return [float(x), float(y), float(x + w), float(y + h)]


def run_inference_single_frame(
    model,
    frame_rgb,
    intrinsics,
    obj_masks,
    categories,
    device="cuda",
):
    """Run WildDet3D on a single frame using mask bboxes as geometric prompts.

    For each object with a visible mask, uses its mask bbox as a geometric
    box prompt to get a 3D detection.

    Args:
        model: WildDet3DPredictor (from load_model)
        frame_rgb: RGB image numpy array (H, W, 3), uint8
        intrinsics: Camera intrinsics numpy array (3, 3)
        obj_masks: Dict mapping obj_id -> RLE dict for this frame
        categories: Dict mapping obj_id -> category string
        device: Device string

    Returns:
        List of dicts with keys:
            track_id, box_2d (4,), box_3d (10,), score, category
    """
    valid_objects = []
    for obj_id, category in categories.items():
        rle = obj_masks.get(obj_id)
        if rle is None:
            continue
        bbox = get_mask_bbox(rle)
        if bbox is None:
            continue
        valid_objects.append((obj_id, bbox, category))

    if not valid_objects:
        return []

    # Preprocess frame using WildDet3D API
    data = preprocess(frame_rgb, intrinsics)

    images = data["images"].to(device)
    K = data["intrinsics"].to(device)[None]
    input_hw = [data["input_hw"]]
    original_hw = [data["original_hw"]]
    padding = [data["padding"]]

    results = []

    for obj_id, bbox, category in valid_objects:
        # WildDet3DPredictor expects box prompts in ORIGINAL image pixel space.
        # It will internally convert to the model's input_hw space (resize + pad).
        x1, y1, x2, y2 = bbox

        prompt_text = "geometric"
        if category is not None and str(category).strip():
            prompt_text = f"geometric: {str(category).strip()}"

        boxes_2d, boxes_3d, scores, scores_2d, scores_3d, class_ids, _ = model(
            images=images,
            intrinsics=K,
            input_hw=input_hw,
            original_hw=original_hw,
            padding=padding,
            input_boxes=[[x1, y1, x2, y2]],
            prompt_text=prompt_text,
        )

        img_boxes_2d = boxes_2d[0]
        img_boxes_3d = boxes_3d[0]
        img_scores = scores[0]

        if len(img_scores) == 0:
            continue

        best_idx = img_scores.argmax()
        results.append({
            "track_id": obj_id,
            "box_2d": img_boxes_2d[best_idx].cpu().numpy(),
            "box_3d": img_boxes_3d[best_idx].cpu().numpy(),
            "score": img_scores[best_idx].item(),
            "category": category,
        })

    return results

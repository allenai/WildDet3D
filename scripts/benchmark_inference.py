"""Benchmark WildDet3D inference: FP32 / BF16 autocast / + torch.compile.

Reproduces the latency table in the README. Loads the public paper
checkpoint and a 1008x1008 demo image, then times the forward pass under
four configs:

  A) FP32 eager                            -- baseline
  B) BF16 autocast (no compile)            -- ~1.6x speedup
  C) BF16 autocast + torch.compile (default)
                                           -- ~2.7x speedup, ~2 min compile
  D) BF16 autocast + torch.compile (max-autotune-no-cudagraphs)
                                           -- ~2.9x speedup, ~17 min compile

For each config we report median/mean/min/std latency over 20 trials
after 5 warmup runs, plus the cosine similarity of the 2D box output vs
the FP32 reference (sanity check that autocast/compile didn't change the
detections).

Usage::

    python scripts/benchmark_inference.py \\
        --ckpt ckpt/wilddet3d_alldata_all_prompt_v1.0.pt \\
        --image assets/demo/rgb.png \\
        --intrinsics assets/demo/intrinsics.npy
"""

import argparse
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn

from wilddet3d import build_model, optimize_for_inference, preprocess


N_WARMUP = 5
N_TRIALS = 20


def run_once(model: nn.Module, data: dict, input_texts: list) -> tuple:
    """One forward pass. Returns (boxes, scores) on CPU."""
    boxes, boxes3d, scores, scores_2d, scores_3d, class_ids, depth_maps = model(
        images=data["images"].cuda(),
        intrinsics=data["intrinsics"].cuda()[None],
        input_hw=[data["input_hw"]],
        original_hw=[data["original_hw"]],
        padding=[data["padding"]],
        input_texts=input_texts,
    )
    boxes_cat = (
        boxes[0].detach().cpu()
        if len(boxes) > 0 and boxes[0].numel() > 0
        else torch.zeros(0, 4)
    )
    scores_cat = (
        scores[0].detach().cpu()
        if len(scores) > 0 and scores[0].numel() > 0
        else torch.zeros(0)
    )
    return boxes_cat, scores_cat


def time_config(
    name: str,
    model: nn.Module,
    data: dict,
    input_texts: list,
    ref_boxes: Optional[Tensor] = None,
) -> dict:
    print(f"\n[{name}] warmup x{N_WARMUP}...")
    with torch.no_grad():
        for i in range(N_WARMUP):
            t0 = time.time()
            boxes, _ = run_once(model, data, input_texts)
            torch.cuda.synchronize()
            dt = (time.time() - t0) * 1000
            print(f"  warmup {i}: {dt:.1f}ms (n_det={len(boxes)})")

    times = []
    with torch.no_grad():
        for _ in range(N_TRIALS):
            torch.cuda.synchronize()
            t0 = time.time()
            boxes, _ = run_once(model, data, input_texts)
            torch.cuda.synchronize()
            times.append((time.time() - t0) * 1000)
    times = np.array(times)

    result = {
        "name": name,
        "median_ms": float(np.median(times)),
        "mean_ms": float(times.mean()),
        "min_ms": float(times.min()),
        "std_ms": float(times.std()),
        "n_det": int(len(boxes)),
        "boxes": boxes,
    }
    if ref_boxes is not None and len(boxes) == len(ref_boxes) and len(boxes) > 0:
        a, b = boxes.flatten().float(), ref_boxes.flatten().float()
        result["cos_sim"] = float((a * b).sum() / (a.norm() * b.norm() + 1e-12))
        result["rel_l2"] = float((a - b).norm() / (b.norm() + 1e-12))
    elif ref_boxes is not None:
        result["cos_sim"] = float("nan")
        result["rel_l2"] = float("nan")

    line = (
        f"[{name}] median={result['median_ms']:6.1f}ms "
        f"mean={result['mean_ms']:6.1f}ms "
        f"min={result['min_ms']:6.1f}ms "
        f"std={result['std_ms']:.1f}ms "
        f"n_det={result['n_det']}"
    )
    if "cos_sim" in result:
        line += f"  cos_sim={result['cos_sim']:.4f} rel_l2={result['rel_l2']:.4f}"
    print(line)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        default="ckpt/wilddet3d_alldata_all_prompt_v1.0.pt",
        help="Path to WildDet3D checkpoint",
    )
    parser.add_argument(
        "--image", default="assets/demo/rgb.png", help="Demo RGB image"
    )
    parser.add_argument(
        "--intrinsics",
        default="assets/demo/intrinsics.npy",
        help="Demo intrinsics .npy (3x3)",
    )
    parser.add_argument(
        "--texts",
        nargs="+",
        default=["chair", "table"],
        help="Text prompts to use",
    )
    parser.add_argument(
        "--skip-max-autotune",
        action="store_true",
        help="Skip config D (saves ~15 min on first run)",
    )
    args = parser.parse_args()

    assert Path(args.ckpt).exists(), f"checkpoint not found: {args.ckpt}"
    assert Path(args.image).exists(), f"image not found: {args.image}"

    print(f"Loading {args.image}")
    image = np.array(Image.open(args.image)).astype(np.float32)
    intrinsics = np.load(args.intrinsics)
    data = preprocess(image, intrinsics)
    input_texts = args.texts

    print(f"Building model from {args.ckpt}")
    base = build_model(
        checkpoint=args.ckpt,
        score_threshold=0.3,
        canonical_rotation=True,
        skip_pretrained=True,
    )

    # A) FP32 eager (reference)
    res_fp32 = time_config("A_fp32_eager", base, data, input_texts)
    ref_boxes = res_fp32["boxes"]

    # B) BF16 autocast, no compile
    model_bf16 = optimize_for_inference(base, dtype="bf16", compile_mode=None)
    res_bf16 = time_config(
        "B_bf16_autocast", model_bf16, data, input_texts, ref_boxes=ref_boxes
    )

    # C) BF16 autocast + compile default
    torch._dynamo.reset()
    model_compile = optimize_for_inference(
        base, dtype="bf16", compile_mode="default"
    )
    res_compile = time_config(
        "C_bf16_compile_default",
        model_compile,
        data,
        input_texts,
        ref_boxes=ref_boxes,
    )

    results = [res_fp32, res_bf16, res_compile]

    # D) BF16 autocast + max-autotune (no cudagraphs, optional)
    if not args.skip_max_autotune:
        torch._dynamo.reset()
        model_mat = optimize_for_inference(
            base, dtype="bf16", compile_mode="max-autotune-no-cudagraphs"
        )
        res_mat = time_config(
            "D_bf16_compile_max-autotune-no-cudagraphs",
            model_mat,
            data,
            input_texts,
            ref_boxes=ref_boxes,
        )
        results.append(res_mat)

    print("\n" + "=" * 76)
    print("SUMMARY  (median latency, lower is better)")
    print("=" * 76)
    fp32_median = res_fp32["median_ms"]
    for r in results:
        speedup = fp32_median / r["median_ms"]
        cos = r.get("cos_sim", float("nan"))
        l2 = r.get("rel_l2", float("nan"))
        print(
            f"  {r['name']:42s} "
            f"median={r['median_ms']:6.1f}ms  "
            f"speedup={speedup:4.2f}x  "
            f"cos_sim={cos:.4f}  rel_l2={l2:.4f}"
        )
    print("=" * 76)


if __name__ == "__main__":
    main()

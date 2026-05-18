"""Render Boxer-format fused-OBB CSV back onto the Aria video frames.

Pairs with `run_wilddet3d.py --fuse`: that command writes
`<seq>/wilddet3d_3dbbs_fused.csv` (one set of static instances at ts=0)
in Boxer's exact CSV schema. This script re-loads the AriaLoader, draws
those static instances onto each frame, and writes a clean mp4.
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="nym10_gen1")
    p.add_argument("--write_name", default="wilddet3d")
    p.add_argument("--max_n", type=int, default=90)
    p.add_argument("--skip_n", type=int, default=1)
    p.add_argument(
        "--output_dir",
        default=str(Path(__file__).parent / "output"),
    )
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--boxer_path",
        default=os.environ.get("BOXER_PATH", "boxer"),
        help="Path to a local clone of facebookresearch/boxer.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    boxer_path = Path(args.boxer_path).expanduser().resolve()
    if not boxer_path.exists():
        raise FileNotFoundError(
            f"Boxer clone not found at {boxer_path}."
        )
    sys.path.insert(0, str(boxer_path))

    from loaders.aria_loader import AriaLoader
    from utils.demo_utils import SAMPLE_DATA_PATH
    from utils.file_io import read_obb_csv
    from utils.image import draw_bb3s, put_text

    log_dir = os.path.join(args.output_dir, args.input)
    fused_csv = os.path.join(log_dir, f"{args.write_name}_3dbbs_fused.csv")
    viz_dir = os.path.join(log_dir, "viz_frames_fused")
    os.makedirs(viz_dir, exist_ok=True)

    print(f"==> Loading fused CSV: {fused_csv}")
    timed_obbs = read_obb_csv(fused_csv)
    fused_obbs = list(timed_obbs.values())[0].to(args.device)
    print(f"==> {len(fused_obbs)} fused instances")

    seq_root = os.path.join(SAMPLE_DATA_PATH, args.input)
    loader = AriaLoader(
        seq_root,
        camera="rgb",
        with_traj=True,
        with_sdp=False,
        max_n=args.max_n,
        skip_n=args.skip_n,
        pinhole=True,
        unrotate=True,
    )

    for ii, datum in enumerate(tqdm(loader, total=args.max_n)):
        img_tensor = datum["img0"]
        cam = datum["cam0"].float().to(args.device)
        T_wr = datum["T_world_rig0"].float().to(args.device)
        img_np = (
            (img_tensor.squeeze(0).permute(1, 2, 0).numpy() * 255)
            .astype(np.uint8)
            .copy()
        )
        viz = draw_bb3s(
            viz=img_np.copy(),
            T_world_rig=T_wr,
            cam=cam,
            obbs=fused_obbs,
            already_rotated=False,
            rotate_label=False,
        )
        put_text(
            viz,
            f"WildDet3D + Boxer fusion  frame {ii}  "
            f"({len(fused_obbs)} instances)",
            scale=0.6,
            line=0,
        )
        cv2.imwrite(
            os.path.join(viz_dir, f"{args.write_name}_fused_viz_{ii:05d}.jpg"),
            cv2.cvtColor(viz, cv2.COLOR_RGB2BGR),
        )

    print(f"\n==> Wrote {viz_dir}")


if __name__ == "__main__":
    main()

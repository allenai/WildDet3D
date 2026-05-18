"""Drop-in replacement for Boxer's run_boxer.py using WildDet3D.

WildDet3D replaces Boxer's OWL (2D detection) + BoxerNet (2D->3D lifting)
stages with a single open-vocabulary monocular 3D detector. All other
components - data loader (AriaLoader), pose math (PoseTW / ObbTW),
visualization (draw_bb3s / make_mp4), offline fusion (fuse_obbs_from_csv),
and online tracker (BoundingBox3DTracker) - are imported from Boxer
without modification, and the output CSV uses Boxer's exact schema, so
Boxer's view_fusion.py / view_tracker.py work on our outputs unchanged.

Required external dependency: a local clone of Boxer
(https://github.com/facebookresearch/boxer), exposed via --boxer_path or
the BOXER_PATH environment variable. Boxer is released under
CC-BY-NC 4.0 - see demo/boxer/README.md.

Usage:
    python -m demo.boxer.run_wilddet3d \\
        --input nym10_gen1 \\
        --max_n 90 \\
        --fuse \\
        --ckpt ckpt/wilddet3d_alldata_all_prompt_v1.0.pt \\
        --boxer_path /path/to/boxer
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

DEFAULT_TEXT_PROMPTS = [
    "chair", "table", "desk", "sofa", "cabinet", "shelf", "lamp",
    "monitor", "pillow", "curtain", "nightstand", "dresser", "mirror",
    "box", "book", "bottle", "cup", "plant", "vase", "bag",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="WildDet3D as drop-in replacement for Boxer's "
        "OWL + BoxerNet pipeline. Outputs Boxer-format CSV so "
        "view_fusion.py / view_tracker.py work unchanged."
    )
    parser.add_argument(
        "--input",
        default="nym10_gen1",
        help="Boxer sequence name (e.g., nym10_gen1, hohen_gen1).",
    )
    parser.add_argument(
        "--write_name",
        default="wilddet3d",
        help="CSV/video file-name prefix (default: wilddet3d).",
    )
    parser.add_argument(
        "--max_n", type=int, default=90, help="Max frames to process."
    )
    parser.add_argument(
        "--skip_n", type=int, default=1, help="Frame stride."
    )
    parser.add_argument(
        "--ckpt",
        default="ckpt/wilddet3d_alldata_all_prompt_v1.0.pt",
        help="WildDet3D checkpoint path.",
    )
    parser.add_argument(
        "--labels",
        default=",".join(DEFAULT_TEXT_PROMPTS),
        help="Comma-separated text prompts.",
    )
    parser.add_argument(
        "--thresh3d", type=float, default=0.5, help="3D score threshold."
    )
    parser.add_argument(
        "--output_dir",
        default=str(Path(__file__).parent / "output"),
        help="Output directory for CSVs and viz.",
    )
    parser.add_argument(
        "--device", default="cuda", help="Inference device (cuda or cpu)."
    )
    parser.add_argument(
        "--boxer_path",
        default=os.environ.get("BOXER_PATH", "boxer"),
        help="Path to a local clone of facebookresearch/boxer. "
        "Required - we import AriaLoader, ObbTW, draw_bb3s etc. from there.",
    )
    parser.add_argument(
        "--use_depth",
        action="store_true",
        help="Feed Aria's semi-dense points (SDP, from on-device SLAM) as "
        "sparse depth into WildDet3D's geometry backend, matching Boxer's "
        "BoxerNet which consumes the same SDP. Without this flag, "
        "WildDet3D runs pure monocular.",
    )
    parser.add_argument(
        "--fuse",
        action="store_true",
        help="Run Boxer's offline 3D box fusion (single-frame 3D NMS + "
        "cross-frame instance fusion) on the output CSV after processing.",
    )
    parser.add_argument(
        "--track",
        action="store_true",
        help="Run Boxer's online 3D box tracker per frame and write a "
        "*_tracked.csv with persistent instance IDs. Mutually exclusive "
        "with --fuse.",
    )
    args = parser.parse_args()
    if args.fuse and args.track:
        parser.error("--fuse and --track are mutually exclusive")
    return args


def aria_to_cam_K(cam_data):
    """Pull (W, H, fx, fy, cx, cy) out of Aria's CameraTW packed data."""
    return (
        float(cam_data[0]),
        float(cam_data[1]),
        float(cam_data[2]),
        float(cam_data[3]),
        float(cam_data[4]),
        float(cam_data[5]),
    )


def load_boxer_aria_loader(sequence_name, max_n, skip_n, with_sdp=False):
    """Boxer AriaLoader with unrotate=True so the image is upright and
    cam.T_camera_rig already encodes the corresponding rotation."""
    from loaders.aria_loader import AriaLoader
    from utils.demo_utils import SAMPLE_DATA_PATH

    seq_root = os.path.join(SAMPLE_DATA_PATH, sequence_name)
    return AriaLoader(
        seq_root,
        camera="rgb",
        with_traj=True,
        with_sdp=with_sdp,
        max_n=max_n,
        skip_n=skip_n,
        pinhole=True,
        unrotate=True,
    )


def sdp_world_to_sparse_depth_map(sdp_w, T_world_cam, K, image_hw):
    """Project Aria SDP (world-frame N,3 points) into the camera and
    rasterize into a sparse (H, W) depth map in meters.

    Pixels with no SDP point get 0.0 (treated as "no observation" by
    the geometry backend). When multiple points fall on the same pixel,
    the nearest one wins.
    """
    H, W = image_hw
    if sdp_w is None or sdp_w.numel() == 0:
        return np.zeros((H, W), dtype=np.float32)
    pts_w = sdp_w.float().cpu().numpy()
    # Drop NaN padding
    valid = np.isfinite(pts_w).all(axis=1)
    pts_w = pts_w[valid]
    if len(pts_w) == 0:
        return np.zeros((H, W), dtype=np.float32)
    # World -> camera (we have T_world_cam, need its inverse)
    T_cw = T_world_cam.inverse()
    R = T_cw.R.cpu().numpy().reshape(3, 3)
    t = T_cw.t.cpu().numpy().reshape(3)
    pts_c = pts_w @ R.T + t
    z = pts_c[:, 2]
    in_front = z > 0.05
    pts_c = pts_c[in_front]
    z = z[in_front]
    if len(z) == 0:
        return np.zeros((H, W), dtype=np.float32)
    # Project with K
    uv = (K @ (pts_c / z[:, None]).T).T[:, :2]
    u = np.round(uv[:, 0]).astype(np.int32)
    v = np.round(uv[:, 1]).astype(np.int32)
    mask = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, z = u[mask], v[mask], z[mask]
    depth = np.zeros((H, W), dtype=np.float32)
    # Z-buffer: keep nearest depth at each pixel
    order = np.argsort(-z)  # furthest first
    depth[v[order], u[order]] = z[order]
    nearest = depth[v, u]
    keep = z < nearest + 1e-6
    if keep.any():
        depth[v[keep], u[keep]] = z[keep]
    return depth


def wilddet3d_to_obb(boxes3d, scores, class_ids, text_prompts, device):
    """Convert WildDet3D's per-frame predictions to Boxer's ObbTW
    (in camera frame; caller transforms to world frame).

    WildDet3D RoI2Det3D output: (N, 10) = [center(3), dims(3), quat_wxyz(4)]
    where dims = (W, L, H). The model's local box frame is x=L, y=H, z=W
    (see opendet3d coder.py canonical_rotation docstring); Boxer's
    bb3_object keeps the vertical axis on z so we rotate 90 deg around
    the vertical and use bb3 = (L=x, W=y, H=z).
    """
    from utils.tw.obb import ObbTW
    from utils.tw.pose import PoseTW

    if boxes3d is None or len(boxes3d) == 0:
        return None

    boxes3d = boxes3d.cpu()
    centers = boxes3d[:, :3]
    dims = boxes3d[:, 3:6]
    q = torch.nn.functional.normalize(boxes3d[:, 6:10], dim=1)
    qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.stack(
        [
            1 - 2 * (qy * qy + qz * qz),
            2 * (qx * qy - qz * qw),
            2 * (qx * qz + qy * qw),
            2 * (qx * qy + qz * qw),
            1 - 2 * (qx * qx + qz * qz),
            2 * (qy * qz - qx * qw),
            2 * (qx * qz - qy * qw),
            2 * (qy * qz + qx * qw),
            1 - 2 * (qx * qx + qy * qy),
        ],
        dim=1,
    ).reshape(-1, 3, 3)

    N = len(boxes3d)
    text_array = np.zeros((N, 128), dtype=np.uint8)
    for i, cid in enumerate(class_ids.cpu().numpy()):
        name = text_prompts[int(cid)] if int(cid) < len(text_prompts) else "object"
        encoded = name.encode("ascii", errors="replace")[:128]
        text_array[i, : len(encoded)] = np.frombuffer(encoded, dtype=np.uint8)
    text_tensor = torch.from_numpy(text_array).float()

    w_, l_, h_ = dims[:, 0], dims[:, 1], dims[:, 2]
    bb3_object = torch.stack(
        [-l_ / 2, l_ / 2, -w_ / 2, w_ / 2, -h_ / 2, h_ / 2], dim=1
    )

    T_co = PoseTW.from_Rt(R, centers).to(device)
    inst_id = torch.arange(N).reshape(N, 1).float()
    sem_id = 32 + class_ids.cpu().reshape(N, 1).float()
    prob = scores.cpu().reshape(N, 1)

    return ObbTW.from_lmc(
        bb3_object=bb3_object.to(device),
        T_world_object=T_co,
        sem_id=sem_id.to(device),
        inst_id=inst_id.to(device),
        prob=prob.to(device),
        text=text_tensor.to(device),
    )


def main():
    args = parse_args()
    boxer_path = Path(args.boxer_path).expanduser().resolve()
    if not boxer_path.exists():
        raise FileNotFoundError(
            f"Boxer clone not found at {boxer_path}. "
            "Clone https://github.com/facebookresearch/boxer and pass "
            "--boxer_path or set BOXER_PATH."
        )
    sys.path.insert(0, str(boxer_path))

    text_prompts = [s.strip() for s in args.labels.split(",") if s.strip()]
    sem_id_to_name = {32 + i: t for i, t in enumerate(text_prompts)}

    log_dir = os.path.join(args.output_dir, args.input)
    viz_dir = os.path.join(log_dir, "viz_frames")
    os.makedirs(viz_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, f"{args.write_name}_3dbbs.csv")

    from utils.file_io import ObbCsvWriter2
    from utils.image import draw_bb3s, put_text
    from utils.video import make_mp4
    from wilddet3d import build_model, preprocess

    writer = ObbCsvWriter2(csv_path)

    print(f"==> Loading AriaLoader: {args.input} (with_sdp={args.use_depth})")
    loader = load_boxer_aria_loader(
        args.input, args.max_n, args.skip_n, with_sdp=args.use_depth
    )

    print(f"==> Loading WildDet3D from {args.ckpt}")
    model = build_model(
        checkpoint=args.ckpt,
        score_threshold=0.5,
        nms=True,
        device=args.device,
        canonical_rotation=True,
        skip_pretrained=True,
        use_depth_input_test=args.use_depth,
    )

    tracker = None
    if args.track:
        from utils.track_3d_boxes import BoundingBox3DTracker
        tracker = BoundingBox3DTracker(
            iou_threshold=0.25,
            min_hits=8,
            conf_threshold=args.thresh3d,
            samp_per_dim=8,
            max_missed=90,
            force_cpu=False,
            verbose=False,
        )

    print(f"==> Running inference, writing to {log_dir}")
    for ii, datum in enumerate(tqdm(loader, total=args.max_n)):
        img_tensor = datum["img0"]
        cam = datum["cam0"].float()
        T_wr = datum["T_world_rig0"].float()
        time_ns = int(datum["time_ns0"])

        img_np = (
            (img_tensor.squeeze(0).permute(1, 2, 0).numpy() * 255)
            .astype(np.uint8)
            .copy()
        )
        cam_data = cam._data if hasattr(cam, "_data") else cam.data
        cam_data_np = cam_data.cpu().numpy() if torch.is_tensor(cam_data) else cam_data
        _W, _H, fx, fy, cx_p, cy_p = aria_to_cam_K(cam_data_np)
        K = np.array(
            [[fx, 0, cx_p], [0, fy, cy_p], [0, 0, 1]], dtype=np.float32
        )

        # T_world_camera = T_world_rig @ T_camera_rig^-1 (Aria convention)
        T_world_cam = T_wr @ cam.T_camera_rig.inverse()

        depth_input = None
        if args.use_depth and "sdp_w" in datum:
            depth_input = sdp_world_to_sparse_depth_map(
                datum["sdp_w"],
                T_world_cam,
                K,
                (img_np.shape[0], img_np.shape[1]),
            )

        data = preprocess(img_np.astype(np.float32), K, depth=depth_input)
        imgs = data["images"].to(args.device)
        if imgs.dim() == 3:
            imgs = imgs.unsqueeze(0)
        K_tensor = data["intrinsics"].to(args.device)
        if K_tensor.dim() == 2:
            K_tensor = K_tensor.unsqueeze(0)

        depth_gt_kw = {}
        if args.use_depth and "depth_gt" in data:
            depth_gt_kw["depth_gt"] = data["depth_gt"].to(args.device)

        with torch.no_grad():
            boxes, boxes3d, scores, class_ids, _ = model(
                images=imgs,
                intrinsics=K_tensor,
                input_hw=[data["input_hw"]],
                original_hw=[data["original_hw"]],
                padding=[data["padding"]],
                input_texts=text_prompts,
                **depth_gt_kw,
            )

        obb_cam = wilddet3d_to_obb(
            boxes3d[0], scores[0], class_ids[0], text_prompts, args.device
        )
        if obb_cam is not None and len(obb_cam) > 0:
            mask = obb_cam.prob.squeeze(-1) >= args.thresh3d
            obb_cam = obb_cam[mask]
        if obb_cam is None or len(obb_cam) == 0:
            continue

        obb_world = obb_cam.transform(T_world_cam.to(args.device))

        writer.write(obb_world, time_ns, sem_id_to_name=sem_id_to_name)

        if tracker is not None:
            tracker.update(
                obb_world,
                ii,
                cam=cam.to(args.device),
                T_world_rig=T_wr.to(args.device),
                observed_points=None,
            )

        viz_3d = draw_bb3s(
            viz=img_np.copy(),
            T_world_rig=T_wr.to(args.device),
            cam=cam.to(args.device),
            obbs=obb_world.to(args.device),
            already_rotated=False,
            rotate_label=False,
        )
        put_text(
            viz_3d,
            f"WildDet3D  frame {ii}  ({len(obb_world)} dets)",
            scale=0.6,
            line=0,
        )
        cv2.imwrite(
            os.path.join(viz_dir, f"{args.write_name}_viz_{ii:05d}.jpg"),
            cv2.cvtColor(viz_3d, cv2.COLOR_RGB2BGR),
        )

    print(f"\n==> CSV: {csv_path}")
    print("==> Making mp4...")
    make_mp4(
        viz_dir,
        10,
        output_dir=log_dir,
        image_glob=f"{args.write_name}_viz_*.jpg",
        output_name=f"{args.write_name}_viz_final.mp4",
    )

    if args.fuse:
        from utils.fuse_3d_boxes import fuse_obbs_from_csv

        print(f"\n==> Running Boxer fusion on {csv_path}")
        fuse_obbs_from_csv(csv_path)

    if tracker is not None:
        from utils.file_io import unpad_string, tensor2string

        active_tracks = tracker._get_active_tracks()
        print(f"==> {len(active_tracks)} active tracks from inline tracker")
        if len(active_tracks) > 0:
            base, ext = os.path.splitext(csv_path)
            track_output_path = f"{base}_tracked{ext}"
            tracked_obbs = torch.stack([t.obb for t in active_tracks])
            ids = torch.tensor(
                [t.track_id for t in active_tracks], dtype=torch.int32
            )
            tracked_obbs.set_inst_id(ids)
            rounded_prob = torch.round(tracked_obbs.prob * 100) / 100
            tracked_obbs.set_prob(rounded_prob.squeeze(-1), use_mask=False)
            track_sem = {}
            for obb in tracked_obbs:
                sid = int(obb.sem_id.item())
                if sid not in track_sem:
                    track_sem[sid] = unpad_string(
                        tensor2string(obb.text.int())
                    )
            track_writer = ObbCsvWriter2(track_output_path)
            track_writer.write(
                tracked_obbs, timestamps_ns=0, sem_id_to_name=track_sem
            )
            track_writer.close()
            print(
                f"==> Saved {len(active_tracks)} tracked OBBs to "
                f"{track_output_path}"
            )


if __name__ == "__main__":
    main()

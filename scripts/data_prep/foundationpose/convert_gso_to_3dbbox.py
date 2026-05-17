#!/usr/bin/env python3
"""
Convert FoundationPose GSO dataset to 3D bounding box format.

Produces one JSON per image matching the reference format used by
COCO/Objects365/LVIS 3D bbox datasets (step30_result_{image_id}.json).

Output format per image:
{
  "image_id": str,
  "boxes3d": [[{"box3d": [cx, cy, cz, w, h, l, qw, qx, qy, qz]}], ...],
  "boxes2d": [[x1, y1, x2, y2], ...],
  "categories": [str, ...],
  "category_certain": [bool, ...],
  "num_candidates": [int, ...]
}

box3d conventions (matching box3d_utils.py):
  (cx, cy, cz): center in OpenCV camera space (X-right, Y-down, Z-forward), meters
  (w, h, l): width(X), height(Y), length(Z) in box's local frame, meters
  (qw, qx, qy, qz): scalar-first quaternion, rotation from box local -> camera frame

Coordinate pipeline:
  Pixel + depth -> OpenGL camera (Isaac Sim native)
  OpenGL camera -> World (via V_inv)
  World -> Scaled object local (via pure rotation inverse)
  AABB in scaled local -> metric (w, h, l)
  Rotation: scaled local -> OpenCV camera (via R_pure, V, flip)
"""

import argparse
import json
import os
import sys
import numpy as np
from PIL import Image
from multiprocessing import Pool
from pathlib import Path
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def rotation_matrix_to_quaternion(R):
    """Convert 3x3 rotation matrix to [qw, qx, qy, qz] quaternion (scalar-first).

    Uses Shepperd's method, matching box3d_utils.py.
    """
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return np.array([qw, qx, qy, qz])


def load_qwen_classifications(base_dir, qwen_path=None):
    """Load Qwen v2 classifications into {asset_name: (category, certain)} dict.

    Resolution order:
    1. Explicit path passed via --qwen_classifications.
    2. {base_dir}/qwen_classifications_v2/gso_qwen_classifications_v2.json
    3. gso_qwen_classifications_v2.json co-located with this script
       (released alongside the data-prep scripts; covers all 928 GSO assets).
    """
    candidates = []
    if qwen_path is not None:
        candidates.append(qwen_path)
    candidates.append(
        os.path.join(
            base_dir, "qwen_classifications_v2", "gso_qwen_classifications_v2.json"
        )
    )
    candidates.append(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "gso_qwen_classifications_v2.json",
        )
    )
    for path in candidates:
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            lookup = {}
            for result in data["results"]:
                asset_name = result["asset_name"]
                raw_class = result["qwen_classification"]
                certain = "(uncertain)" not in raw_class
                category = raw_class.replace(" (uncertain)", "")
                lookup[asset_name] = (category, certain)
            logger.info(f"Loaded Qwen classifications from {path}")
            return lookup
    raise FileNotFoundError(
        "Could not locate gso_qwen_classifications_v2.json. Tried: "
        + ", ".join(candidates)
    )


# ---------------------------------------------------------------------------
# Worker state (initialized once per process)
# ---------------------------------------------------------------------------
_worker_state = {}


def init_worker(base_dir, output_dir, min_mask_pixels, qwen_path=None):
    """Initialize worker process with shared data."""
    _worker_state["base_dir"] = base_dir
    _worker_state["output_dir"] = output_dir
    _worker_state["min_mask_pixels"] = min_mask_pixels
    _worker_state["qwen"] = load_qwen_classifications(base_dir, qwen_path)
    _worker_state["scene_cache"] = {}


def get_scene_data(scene_stem):
    """Load scene data with per-worker caching."""
    cache = _worker_state["scene_cache"]
    if scene_stem not in cache:
        path = os.path.join(
            _worker_state["base_dir"],
            "scene_data",
            "gso",
            f"{scene_stem}_states.json",
        )
        if os.path.exists(path):
            with open(path) as f:
                cache[scene_stem] = json.load(f)
        else:
            cache[scene_stem] = None
    return cache[scene_stem]


# OpenGL -> OpenCV flip: negate Y and Z
_F = np.diag([1.0, -1.0, -1.0])


def process_image(stem):
    """Process a single image and produce a step30_result JSON.

    Returns a dict with status and metadata for aggregation.
    """
    base_dir = _worker_state["base_dir"]
    output_dir = _worker_state["output_dir"]
    min_mask_pixels = _worker_state["min_mask_pixels"]
    qwen = _worker_state["qwen"]

    # Resumability: skip if output already exists
    out_path = os.path.join(output_dir, f"step30_result_{stem}.json")
    if os.path.exists(out_path):
        return {"status": "skipped", "stem": stem}

    try:
        # --- Load all inputs ---
        mask_path = os.path.join(base_dir, "masks", "gso", f"{stem}.png")
        depth_path = os.path.join(base_dir, "depth", "gso", f"{stem}.npy")
        cam_path = os.path.join(base_dir, "camera_params", "gso", f"{stem}.json")
        mapping_path = os.path.join(
            base_dir,
            "annotations",
            "instance_mappings",
            "gso",
            f"{stem}_mapping.json",
        )

        for p in [mask_path, depth_path, cam_path, mapping_path]:
            if not os.path.exists(p):
                return {"status": "missing_file", "stem": stem, "file": p}

        mask = np.array(Image.open(mask_path))  # uint16 (480, 640)
        depth = np.load(depth_path)  # float32 (480, 640)
        with open(cam_path) as f:
            cam = json.load(f)
        with open(mapping_path) as f:
            mapping = json.load(f)

        # Scene data (shared across cam_0 / cam_1)
        # stem = "{group}_scene_{id}_cam_{c}"
        parts = stem.rsplit("_", 2)  # ["{group}_scene_{id}", "cam", "{c}"]
        scene_stem = parts[0]
        scene_data = get_scene_data(scene_stem)
        if scene_data is None:
            return {"status": "no_scene_data", "stem": stem}

        # --- Camera params ---
        V = np.array(cam["cameraViewTransform"]).reshape(4, 4)
        P = np.array(cam["cameraProjection"]).reshape(4, 4)
        w_img, h_img = cam["renderProductResolution"]

        fx = P[0, 0] * w_img / 2.0
        fy = P[1, 1] * h_img / 2.0
        cx = w_img / 2.0
        cy = h_img / 2.0

        V_3x3 = V[:3, :3]
        V_t = V[3, :3]  # translation in last row (row-vector convention)
        V_inv = np.linalg.inv(V)

        # --- Process each object ---
        boxes3d = []
        boxes2d = []
        categories = []
        category_certain = []

        for sem_id_str, prim_path in mapping.items():
            sem_id = int(sem_id_str)

            # Skip non-object entries
            if sem_id <= 1:
                continue
            if "collision_box" in prim_path:
                continue
            if not prim_path.startswith("/World/objects/"):
                continue

            # Extract asset name and scene_data key
            path_parts = prim_path.split("/")
            if len(path_parts) < 4:
                continue
            asset_name = path_parts[3]  # e.g. "gso_ASICS_GEL..."
            obj_name = asset_name[4:] if asset_name.startswith("gso_") else asset_name

            # Object transform from scene data
            if obj_name not in scene_data.get("objects", {}):
                continue
            obj_data = scene_data["objects"][obj_name]
            M = np.array(obj_data["transform_matrix_world"])  # 4x4

            # --- Mask ---
            obj_mask = mask == sem_id
            ys, xs = np.where(obj_mask)
            n_pixels = len(ys)
            if n_pixels < min_mask_pixels:
                continue

            # 2D bbox from mask
            x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

            # --- Depth at mask pixels ---
            d_vals = depth[ys, xs]
            valid = np.isfinite(d_vals) & (d_vals > 0)
            if np.sum(valid) < 10:
                continue

            px = xs[valid].astype(np.float64)
            py = ys[valid].astype(np.float64)
            d = d_vals[valid].astype(np.float64)

            # --- Unproject to OpenGL camera space ---
            X_gl = (px - cx) * d / fx
            Y_gl = -(py - cy) * d / fy
            Z_gl = -d
            pts_cam_gl = np.stack([X_gl, Y_gl, Z_gl], axis=-1)  # (N, 3)

            # --- Camera -> World (row-vector: p_cam = p_world @ V) ---
            pts_cam_hom = np.hstack([pts_cam_gl, np.ones((len(pts_cam_gl), 1))])
            pts_world = (pts_cam_hom @ V_inv)[:, :3]

            # --- World -> Scaled local ---
            # M is row-vector: p_world = p_scaled_local @ R_pure + t
            # where M_3x3 = diag(scale) @ R_pure, scale = row norms
            M_3x3 = M[:3, :3]
            t_obj = M[3, :3]

            row_norms = np.linalg.norm(M_3x3, axis=1)
            R_pure = M_3x3 / row_norms[:, None]

            # p_scaled = (p_world - t) @ R_pure^T
            pts_scaled = (pts_world - t_obj) @ R_pure.T

            # --- AABB in scaled local space -> metric dimensions ---
            aabb_min = pts_scaled.min(axis=0)
            aabb_max = pts_scaled.max(axis=0)
            extents = aabb_max - aabb_min
            w_box, h_box, l_box = extents[0], extents[1], extents[2]

            # Skip degenerate boxes
            if w_box < 1e-6 or h_box < 1e-6 or l_box < 1e-6:
                continue

            # --- Center in OpenCV camera space ---
            center_scaled = (aabb_min + aabb_max) / 2.0
            center_world = center_scaled @ R_pure + t_obj  # row-vector
            center_cam_gl = center_world @ V_3x3 + V_t  # row-vector
            center_cam_cv = np.array(
                [center_cam_gl[0], -center_cam_gl[1], -center_cam_gl[2]]
            )

            # --- Rotation: box local -> OpenCV camera (column-vector) ---
            # Chain: scaled_local --(R_pure^T)--> world --(V_3x3^T)--> cam_gl --(F)--> cam_cv
            R_box_to_cam_cv = _F @ V_3x3.T @ R_pure.T
            quat = rotation_matrix_to_quaternion(R_box_to_cam_cv)

            # --- Category from Qwen ---
            if asset_name in qwen:
                cat, certain = qwen[asset_name]
            else:
                cat, certain = "unknown", False

            # --- Assemble box3d ---
            box3d = [
                float(center_cam_cv[0]),
                float(center_cam_cv[1]),
                float(center_cam_cv[2]),
                float(w_box),
                float(h_box),
                float(l_box),
                float(quat[0]),
                float(quat[1]),
                float(quat[2]),
                float(quat[3]),
            ]

            boxes3d.append([{"box3d": box3d}])
            boxes2d.append([float(x1), float(y1), float(x2), float(y2)])
            categories.append(cat)
            category_certain.append(certain)

        if not boxes3d:
            return {"status": "no_objects", "stem": stem}

        result = {
            "image_id": stem,
            "boxes3d": boxes3d,
            "boxes2d": boxes2d,
            "categories": categories,
            "category_certain": category_certain,
            "num_candidates": [1] * len(boxes3d),
        }

        # Atomic write: write to .tmp then rename
        tmp_path = out_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(result, f, indent=2)
        os.rename(tmp_path, out_path)

        return {"status": "ok", "stem": stem, "n_objects": len(boxes3d)}

    except Exception as e:
        return {"status": "error", "stem": stem, "error": str(e)}


def discover_stems(base_dir):
    """Discover all image stems from the masks directory."""
    masks_dir = os.path.join(base_dir, "masks", "gso")
    stems = []
    for fname in os.listdir(masks_dir):
        if fname.endswith(".png"):
            stems.append(fname[:-4])  # strip .png
    stems.sort()
    return stems


def main():
    parser = argparse.ArgumentParser(
        description="Convert FoundationPose GSO to 3D bbox format"
    )
    parser.add_argument(
        "--base_dir",
        required=True,
        help="Base directory of FoundationPose extracted data",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for step30_result JSON files",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--min_mask_pixels",
        type=int,
        default=100,
        help="Minimum mask pixels to process an object",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only first N images (0 = all, for testing)",
    )
    parser.add_argument(
        "--qwen_classifications",
        default=None,
        help=(
            "Path to gso_qwen_classifications_v2.json. "
            "If omitted, falls back to "
            "{base_dir}/qwen_classifications_v2/gso_qwen_classifications_v2.json "
            "and then to the JSON shipped next to this script."
        ),
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Discovering image stems from masks directory...")
    stems = discover_stems(args.base_dir)
    logger.info(f"Found {len(stems)} images")

    if args.limit > 0:
        stems = stems[: args.limit]
        logger.info(f"Limited to {len(stems)} images")

    t0 = time.time()

    # Counters
    stats = {"ok": 0, "skipped": 0, "no_objects": 0, "error": 0, "missing_file": 0, "no_scene_data": 0}
    total_objects = 0
    log_interval = 5000

    if args.workers <= 1:
        # Single-process mode (easier to debug)
        init_worker(args.base_dir, args.output_dir, args.min_mask_pixels, args.qwen_classifications)
        for i, stem in enumerate(stems):
            result = process_image(stem)
            status = result["status"]
            stats[status] = stats.get(status, 0) + 1
            if status == "ok":
                total_objects += result["n_objects"]
            elif status == "error":
                logger.warning(f"Error on {stem}: {result['error']}")
            if (i + 1) % log_interval == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                logger.info(
                    f"Progress: {i+1}/{len(stems)} ({rate:.0f} img/s) "
                    f"ok={stats['ok']} skip={stats['skipped']} "
                    f"no_obj={stats['no_objects']} err={stats['error']}"
                )
    else:
        with Pool(
            args.workers,
            initializer=init_worker,
            initargs=(args.base_dir, args.output_dir, args.min_mask_pixels, args.qwen_classifications),
        ) as pool:
            for i, result in enumerate(
                pool.imap_unordered(process_image, stems, chunksize=64)
            ):
                status = result["status"]
                stats[status] = stats.get(status, 0) + 1
                if status == "ok":
                    total_objects += result["n_objects"]
                elif status == "error":
                    logger.warning(f"Error on {result['stem']}: {result['error']}")
                if (i + 1) % log_interval == 0:
                    elapsed = time.time() - t0
                    rate = (i + 1) / elapsed
                    logger.info(
                        f"Progress: {i+1}/{len(stems)} ({rate:.0f} img/s) "
                        f"ok={stats['ok']} skip={stats['skipped']} "
                        f"no_obj={stats['no_objects']} err={stats['error']}"
                    )

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info(f"Done in {elapsed:.1f}s ({len(stems)/elapsed:.0f} img/s)")
    logger.info(f"  OK:            {stats['ok']}")
    logger.info(f"  Skipped:       {stats['skipped']}")
    logger.info(f"  No objects:    {stats['no_objects']}")
    logger.info(f"  Missing file:  {stats['missing_file']}")
    logger.info(f"  No scene data: {stats['no_scene_data']}")
    logger.info(f"  Errors:        {stats['error']}")
    logger.info(f"  Total objects: {total_objects}")
    if stats["ok"] > 0:
        logger.info(f"  Avg obj/image: {total_objects / stats['ok']:.1f}")


if __name__ == "__main__":
    main()

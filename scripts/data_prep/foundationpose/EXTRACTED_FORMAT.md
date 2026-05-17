# FoundationPose GSO Dataset - Data Format Documentation

Base directory: `/weka/oe-training-default/weikaih/molmo_detection_curation/data/foundationpose_extracted/`

---

## Naming Convention

All per-view files follow the pattern:
```
{group_id}_scene_{scene_id:08d}_cam_{cam_id}.{ext}
```
- `group_id`: Numeric scene group identifier (e.g., `1004491151`)
- `scene_id`: 8-digit zero-padded scene index (e.g., `00000412`)
- `cam_id`: Camera index (`0` or `1` — two cameras per scene)

Example: `4280099961_scene_00000412_cam_1.png`

Per-scene files (no camera dimension):
```
{group_id}_scene_{scene_id:08d}_states.json
```

---

## 1. Images

**Location:** `images/gso/`
**Count:** 446,228 files
**Format:** PNG (RGBA), 640 x 480 pixels

```
4280099961_scene_00000412_cam_1.png
```

---

## 2. Depth Maps

**Location:** `depth/gso/`
**Count:** 446,232 files
**Format:** NumPy `.npy`, float32, shape `(480, 640)`

- **Units:** Meters (scene unit = 1 meter, see `metersPerSceneUnit` in camera params)
- **Invalid pixels:** `inf` (background / out-of-range)
- **Typical range:** ~1.8 - 3.5 meters

```python
depth = np.load("1004491151_scene_00000000_cam_0.npy")
# dtype: float32, shape: (480, 640)
# valid range: 1.805 - 3.212 meters
# background: inf
```

> **Note:** The reference datasets (COCO/Objects365/LVIS) at
> `v4_depth/` store depth as float32 `.npy` in **millimeters** with
> filename pattern `{image_id:012d}_sr_1024_long.npy` and variable
> resolution (up to 1024 on the long edge). FoundationPose stores depth
> in **meters** at a fixed 640x480.

---

## 3. Camera Parameters

**Location:** `camera_params/gso/`
**Count:** 446,232 files
**Format:** JSON

```json
{
  "cameraAperture": [22.53, 15.29],
  "cameraApertureOffset": [0.0, 0.0],
  "cameraFisheyeLensP": [],
  "cameraFisheyeLensS": [],
  "cameraFisheyeMaxFOV": 0.0,
  "cameraFisheyeNominalHeight": 0,
  "cameraFisheyeNominalWidth": 0,
  "cameraFisheyeOpticalCentre": [0.0, 0.0],
  "cameraFisheyePolynomial": [0.0, 0.0, 0.0, 0.0, 0.0],
  "cameraFocalLength": 34.54,
  "cameraFocusDistance": 321.90,
  "cameraFStop": 0.0,
  "cameraModel": "pinhole",
  "cameraNearFar": [0.01, 1000000.0],
  "cameraProjection": [
    3.066, 0.0, 0.0, 0.0,
    0.0, 4.089, 0.0, 0.0,
    0.0, 0.0, 1e-08, -1.0,
    0.0, 0.0, 0.01, 0.0
  ],
  "cameraViewTransform": [
    -0.177, -0.711, 0.681, 0.0,
    0.984, -0.128, 0.122, 0.0,
    0.0, 0.692, 0.722, 0.0,
    0.0, 0.0, -2.461, 1.0
  ],
  "metersPerSceneUnit": 1.0,
  "renderProductResolution": [640, 480]
}
```

### Key fields

| Field | Description |
|-------|-------------|
| `cameraModel` | Always `"pinhole"` |
| `cameraProjection` | 4x4 projection matrix, stored as flat 16-element array (row-major). `P[0]=fx_ndc`, `P[5]=fy_ndc`. To get pixel focal lengths: `fx = P[0] * width/2`, `fy = P[5] * height/2` |
| `cameraViewTransform` | 4x4 world-to-camera view matrix, flat 16-element array (row-major, row-vector convention). Translation is in the last row. Multiply as `p_cam = p_world @ V` |
| `renderProductResolution` | `[width, height]` = `[640, 480]` |
| `metersPerSceneUnit` | Scale factor, always `1.0` (depth values are already in meters) |
| `cameraFocalLength` | Physical focal length in scene units (mm-equivalent) |

### Camera space convention (OpenGL)

Isaac Sim uses **OpenGL** camera space, NOT OpenCV:
- **X**: right
- **Y**: up (opposite to OpenCV)
- **Z**: into the screen is negative (camera looks along **-Z**)

This affects unprojection and any conversions to/from OpenCV camera space (X-right, Y-down, Z-forward).

### Deriving intrinsics

```python
P = np.array(cam["cameraProjection"]).reshape(4, 4)
w, h = cam["renderProductResolution"]
fx = P[0, 0] * w / 2   # ~981 pixels
fy = P[1, 1] * h / 2   # ~981 pixels
cx = w / 2              # 320
cy = h / 2              # 240
```

### Unprojection (pixel + depth → 3D)

Because the camera is in OpenGL convention, unprojection from pixel coordinates
`(px, py)` with metric depth `d` (in meters) is:

```python
X_gl = (px - cx) * d / fx
Y_gl = -(py - cy) * d / fy   # note negation (Y-up vs pixel Y-down)
Z_gl = -d                     # camera looks along -Z
```

To convert to **OpenCV camera space** (X-right, Y-down, Z-forward):
```python
X_cv = X_gl
Y_cv = -Y_gl
Z_cv = -Z_gl
```

### Projection (world → pixel)

```python
p_clip = p_world @ V @ P          # row-vector convention
ndc = p_clip[:2] / p_clip[3]      # perspective divide (clip_w = -z_eye for OpenGL)
px = (ndc[0] + 1) / 2 * width
py = (1 - ndc[1]) / 2 * height    # flip Y: NDC Y-up → pixel Y-down
```

> **Reference format (COCO/Objects365/LVIS):** Camera params are stored
> as JSON at `v4_depth/{dataset}/{split}/camera_params/{image_id:012d}.json`:
> ```json
> {
>   "intrinsics": [[fx, s, cx], [0, fy, cy], [0, 0, 1]],
>   "depth_size": [H, W],
>   "image_size": [H, W],
>   "rotation": [[3x3]],
>   "translation": [x, y, z]
> }
> ```
> Intrinsics are given directly as a 3x3 matrix in pixel coordinates.

---

## 4. Instance Segmentation Masks

**Location:** `masks/gso/`
**Count:** 446,232 files
**Format:** PNG, 16-bit grayscale (`I;16`), 640 x 480 pixels

Pixel values are semantic instance IDs:
- `0` = Background
- `1` = Unlabelled
- `2+` = Object instances (mapped via instance mappings)

```python
mask = np.array(Image.open("...cam_0.png"))
# dtype: uint16, shape: (480, 640)
# unique values example: [0, 1, 4, 10, 11, 12, ..., 27]
```

---

## 5. 2D Bounding Boxes

**Location:** `bounding_boxes/gso/{tight,loose}/`
**Count:** 163,594 (tight), 446,232 (loose)
**Format:** NumPy structured array `.npy`

```python
dtype = np.dtype([
    ('semanticId', '<u4'),       # Instance ID (matches mask pixel values)
    ('x_min',      '<i4'),       # Left edge (pixels)
    ('y_min',      '<i4'),       # Top edge (pixels)
    ('x_max',      '<i4'),       # Right edge (pixels, inclusive)
    ('y_max',      '<i4'),       # Bottom edge (pixels, inclusive)
    ('occlusionRatio', '<f4'),   # Fraction of object occluded [0.0, 1.0]
])
```

Sample:
```python
bbox = np.load("1004491151_scene_00000000_cam_0.npy")
# shape: (19,)
# bbox[0] = (semanticId=0, x_min=0, y_min=0, x_max=639, y_max=479, occlusionRatio=0.348)
# bbox[2] = (semanticId=2, x_min=348, y_min=239, x_max=423, y_max=323, occlusionRatio=0.285)
```

- **semanticId=0** is background (spans full image), skip when processing objects
- **tight** boxes fit closely around the visible mask; **loose** boxes include padding
- Not all images have tight boxes (163K vs 446K)

> **Warning:** The `semanticId` values in the bbox `.npy` files use a **different**
> numbering than the mask PNG pixel values. The bbox `semanticId` is an index into
> the `bounding_box_paths/` annotation arrays, NOT the mask semantic ID. Always
> derive object identity from **mask + instance_mapping** files, which are the
> authoritative source for semantic IDs.

---

## 6. 3D Object Poses (Scene Data)

**Location:** `scene_data/gso/`
**Count:** 449,469 files
**Format:** JSON, one per scene (shared across both cameras)

```json
{
  "collision_box": {
    "prim_path": "/World/collision_box",
    "name": "collision_box",
    "position": [0.0, 0.0, 1.305],
    "width": 1.042,
    "height": 1.968,
    "depth": 2.610,
    "visible": true,
    "distribution": "normal"
  },
  "objects": {
    "ASICS_GELBlur33_20_GS_BlackWhiteSafety_Orange": {
      "prim_path": "/World/objects/gso_ASICS_GELBlur33_20_GS_...",
      "scale": [0.124, 0.124, 0.124],
      "translation": [-0.036, -0.009, 0.054],
      "rotation_matrix": [
        [-0.043, 0.028, 0.113],
        [-0.032, -0.118, 0.018],
        [0.112, -0.023, 0.048]
      ],
      "transform_matrix_world": [
        [-0.043, -0.032, 0.112, 0.0],
        [0.028, -0.118, -0.023, 0.0],
        [0.113, 0.018, 0.048, 0.0],
        [-0.036, -0.009, 0.054, 1.0]
      ]
    }
  }
}
```

### Object fields

| Field | Description |
|-------|-------------|
| `prim_path` | USD primitive path. Format: `/World/objects/gso_{AssetName}` |
| `scale` | `[sx, sy, sz]` — scale factors (often uniform) |
| `translation` | `[tx, ty, tz]` — position in world coordinates (meters) |
| `rotation_matrix` | 3x3 rotation matrix (includes scale: column norms = scale) |
| `transform_matrix_world` | 4x4 homogeneous transform. Row-vector convention: `p_world = p_local * M`. Upper-left 3x3 = rotation (transposed relative to `rotation_matrix` columns). Last row = `[tx, ty, tz, 1.0]` |

### collision_box

Defines the table/surface where objects are placed:
- `position`: Center of the box in world coordinates
- `width`, `height`, `depth`: Dimensions of the placement surface

---

## 7. Instance Mappings

**Location:** `annotations/instance_mappings/gso/`

### mapping.json
Maps semantic IDs (from masks) to USD prim paths.

```json
{
  "0": "BACKGROUND",
  "1": "UNLABELLED",
  "4": "/World/collision_box/collision_box_floor",
  "10": "/World/objects/gso_Perricone_MD_Nutritive_Cleanser/model/mesh",
  "11": "/World/objects/gso_Phillips_Caplets_Size_24/model/mesh"
}
```

Asset name extraction: `prim_path.split('/')[3]` gives `gso_Perricone_MD_Nutritive_Cleanser`

### semantics.json
Maps semantic IDs to lowercased class names.

```json
{
  "0": {"class": "BACKGROUND"},
  "1": {"class": "UNLABELLED"},
  "10": {"class": "world_objects_gso_perricone_md_nutritive_cleanser_model_mesh"}
}
```

---

## 8. Bounding Box Path Annotations

**Location:** `annotations/bounding_box_paths/gso/`

### labels.json / tight_labels.json
Maps bbox array index to class name.

```json
{
  "0": {"class": "world_collision_box_collision_box_floor"},
  "1": {"class": "world_objects_gso_heavyduty_flashlight_model_mesh"},
  "2": {"class": "world_objects_gso_wilton_easy_layers_cake_pan_set_model_mesh"}
}
```

### tight_paths.json / loose_paths.json
Array of prim paths in bbox order.

```json
[
  "/World/collision_box/collision_box_floor",
  "/World/objects/gso_BREAKFAST_MENU/model/mesh",
  "/World/objects/gso_Travel_Mate_P_series_Notebook/model/mesh"
]
```

Index `i` in this array corresponds to `bbox_data[i]` in the `.npy` file.

---

## 9. Occlusion Data

**Location:** `occlusion/gso/`
**Count:** 163,594 files
**Format:** NumPy structured array `.npy`

```python
dtype = np.dtype([
    ('instanceId',     '<u4'),  # Unique instance ID
    ('semanticId',     '<u4'),  # Semantic class ID (from mask)
    ('occlusionRatio', '<f4'),  # Occlusion ratio [0.0, 1.0]
])
```

---

## 10. Qwen VLM Classifications

**Location:** `qwen_classifications_v2/gso_qwen_classifications_v2.json`
**Format:** JSON

```json
{
  "dataset": "gso",
  "status": "completed",
  "total_assets": 928,
  "successful": 866,
  "uncertain": 62,
  "failed": 0,
  "results": [
    {
      "asset_name": "gso_AMBERLIGHT_UP_W",
      "dataset": "gso",
      "num_crops_available": 10,
      "num_crops_used": 1,
      "selected_crops": ["crop_005.png"],
      "total_views_available": 8395,
      "qwen_classification": "running shoe",
      "qwen_reasoning": "The object is a high-top athletic shoe with a blue upper..."
    }
  ]
}
```

- 928 unique GSO assets classified
- `qwen_classification`: category label, or `"category (uncertain)"` for low-confidence
- Maps to objects via `asset_name` matching the prim path component (e.g., `gso_AMBERLIGHT_UP_W`)

---

## Cross-referencing Data

To connect all annotations for a single image:

```python
stem = "1004491151_scene_00000000_cam_0"

image     = Image.open(f"images/gso/{stem}.png")
depth     = np.load(f"depth/gso/{stem}.npy")
mask      = np.array(Image.open(f"masks/gso/{stem}.png"))
bbox      = np.load(f"bounding_boxes/gso/tight/{stem}.npy")
cam       = json.load(open(f"camera_params/gso/{stem}.json"))
mapping   = json.load(open(f"annotations/instance_mappings/gso/{stem}_mapping.json"))

# Scene data (shared across both cameras)
scene_id  = "_".join(stem.split("_")[:-2])  # "1004491151_scene_00000000"
scene     = json.load(open(f"scene_data/gso/{scene_id}_states.json"))

# For each bbox entry:
for row in bbox:
    sem_id = int(row["semanticId"])
    if sem_id == 0:  # skip background
        continue
    prim_path = mapping.get(str(sem_id), "")
    asset_name = prim_path.split("/")[3] if len(prim_path.split("/")) >= 4 else None

    # 2D bbox
    x_min, y_min, x_max, y_max = row["x_min"], row["y_min"], row["x_max"], row["y_max"]

    # 3D pose (from scene data, using object name without gso_ prefix)
    obj_name = asset_name.replace("gso_", "") if asset_name else None
    if obj_name and obj_name in scene["objects"]:
        obj = scene["objects"][obj_name]
        transform = np.array(obj["transform_matrix_world"])  # 4x4

    # Qwen label
    qwen_label = cls_lookup.get(asset_name, "unknown")

    # Mask pixels
    obj_pixels = (mask == sem_id)
```

---

## Comparison with Reference Datasets (COCO / Objects365 / LVIS)

| Aspect | FoundationPose GSO | COCO / Objects365 / LVIS |
|--------|-------------------|--------------------------|
| **Depth format** | float32 `.npy`, meters, (480, 640) fixed | float32 `.npy`, millimeters, variable resolution (up to 1024 long edge) |
| **Depth naming** | `{group}_scene_{id}_cam_{c}.npy` | `{image_id:012d}_sr_1024_long.npy` |
| **Camera params** | Full Isaac Sim camera JSON (projection matrix, view transform, aperture, etc.) | Minimal JSON: `{intrinsics, depth_size, image_size, rotation, translation}` |
| **Intrinsics** | Derived from `cameraProjection` matrix | Given directly as 3x3 `intrinsics` matrix |
| **2D boxes** | NumPy structured array `.npy` per image | COCO-format JSON annotation file (shared for all images) |
| **Segmentation** | 16-bit PNG masks (per-pixel instance IDs) | Polygon or RLE in annotation JSON |
| **3D poses** | Full 4x4 transform per object in `scene_data` JSON | 3D bboxes from monocular estimation pipeline |
| **Categories** | 928 unique asset names, Qwen-classified | Fixed category sets (80/365/1203) |
| **Image resolution** | Fixed 640x480 | Variable |

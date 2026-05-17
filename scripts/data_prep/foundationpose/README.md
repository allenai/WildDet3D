# FoundationPose data preparation

The WildDet3D-Data synthetic split includes a portion derived from the
FoundationPose GSO renderings. Preparing that split from the official
FoundationPose Google Drive release requires three steps:

```
raw FoundationPose .zip   ──Step 1──>   foundationpose_extracted/    ──Step 2──>   foundationpose_3dbbox/gso/   ──Step 3──>   Omni3D-style annotations
(Google Drive download)                 images/, depth/, masks/,                   step30_result_{img_id}.json                 used by the WildDet3D training pipeline
                                        camera_params/, metadata/, ...
```

This directory contains one script per step.

## Step 1 — Extract the raw FoundationPose archive

Script: `extract_foundationpose.py`

Takes the raw zip archives downloaded from the official FoundationPose Google
Drive release and lays them out into the structured directory used by Step 2.

```
python extract_foundationpose.py \
    --source <dir-with-FoundationPose-zips> \
    --output <foundationpose_extracted> \
    --dataset gso \
    --workers 32
```

The resulting `<foundationpose_extracted>/` directory follows the format
documented in `EXTRACTED_FORMAT.md` (per-frame `images/`, `depth/`,
`camera_params/`, `masks/`, `metadata/`, plus per-scene `scene_data/`,
`bounding_boxes/`, etc.).

## Step 2 — Generate per-image 3D bbox JSONs

Script: `convert_gso_to_3dbbox.py`

Reads the extracted directory and produces one `step30_result_{image_id}.json`
per image, matching the per-image 3D bbox format used by the COCO / Objects365
/ LVIS preprocessing pipelines.

```
python convert_gso_to_3dbbox.py \
    --base_dir <foundationpose_extracted> \
    --output_dir <foundationpose_3dbbox/gso> \
    --workers 32
```

Each output JSON contains `boxes3d`, `boxes2d`, `categories`, and per-object
metadata. See the header of the script for the coordinate convention.

This step assigns each GSO asset a natural-language category using a Qwen
VLM. To avoid forcing every user to re-run the VLM, the pre-computed
classifications for all 928 GSO assets are shipped alongside this script as
`gso_qwen_classifications_v2.json` and are picked up automatically. You can
override with `--qwen_classifications <path>` if you regenerate them.

## Step 3 — Convert to Omni3D-style annotations

Script: `convert_foundationpose_fast.py`

Combines the Step 1 output (images + camera params) with the Step 2 output
(3D bbox JSONs) to produce the Omni3D-style annotation JSON consumed by the
WildDet3D training pipeline.

```
python convert_foundationpose_fast.py \
    --bbox3d_dir <foundationpose_3dbbox/gso> \
    --extracted_dir <foundationpose_extracted> \
    --output_dir <data/foundationpose/annotations>
```

After this step the FoundationPose subset is in the same format as the other
synthetic sources in WildDet3D-Data and can be loaded by the standard
WildDet3D dataset class.

## Reference layout after all three steps

```
data/foundationpose/
├── images/            # symlinked or copied from <foundationpose_extracted>/images/gso
├── depth/             # from <foundationpose_extracted>/depth/gso
├── camera_params/     # from <foundationpose_extracted>/camera_params/gso
└── annotations/       # Step 3 output (Omni3D-style)
```

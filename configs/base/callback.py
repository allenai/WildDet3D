"""Callbacks for 3D-MOOD."""

from __future__ import annotations

import os

from ml_collections import ConfigDict, FieldReference
from vis4d.config import class_config
from vis4d.data.const import AxisMode
from vis4d.engine.callbacks import EvaluatorCallback, VisualizerCallback
from vis4d.engine.connectors import CallbackConnector
from vis4d.vis.image.bbox3d_visualizer import BoundingBox3DVisualizer
from vis4d.vis.image.canvas import PillowCanvasBackend
from vis4d.zoo.base import get_default_callbacks_cfg

from wilddet3d.data.datasets.argoverse import (
    av2_class_map,
    av2_det_map,
)
from wilddet3d.data.datasets.labelany3d_coco import (
    labelany3d_coco_class_map,
    labelany3d_coco_det_map,
)
from wilddet3d.data.datasets.scannet import (
    scannet200_class_map,
    scannet200_det_map,
    scannet_class_map,
    scannet_det_map,
)
from wilddet3d.eval.detect3d import Detect3DEvaluator
from wilddet3d.eval.omni3d import Omni3DEvaluator
from wilddet3d.eval.open import OpenDetect3DEvaluator
from wilddet3d.vis.image.depth_visualizer import DepthVisualizer
from configs.base.base_connector import (
    CONN_BBOX_3D_VIS,
    CONN_COCO_DET3D_EVAL,
    CONN_DEPTH_VIS,
    CONN_OMNI3D_DET3D_EVAL,
    CONN_POSTPROCESS_CACHE_EXPORT,
)


def get_callback_cfg(
    output_dir: str | FieldReference,
    open_test_datasets: list[str] | None,
    omni3d_evaluator: ConfigDict | None = None,
    visualize_depth: bool = True,
    export_postprocess_cache: bool = False,
    postprocess_cache_root: str | None = None,
) -> list[ConfigDict]:
    """Get callbacks for 3D-MOOD."""
    # Logger
    callbacks = get_default_callbacks_cfg()

    # Add profiling callback if enabled via environment variable
    if os.environ.get("PROFILE_WILDDET3D", "0") == "1":
        from wilddet3d.ops.profiler_callback import EnhancedProfilingCallback
        print("[Profiling] EnhancedProfilingCallback will be added to callbacks")
        callbacks.append(class_config(EnhancedProfilingCallback))

    # Evaluator
    if "ScanNet200_val" in open_test_datasets:
        assert (
            len(open_test_datasets) == 1 and omni3d_evaluator is None
        ), "ScanNet200_val should be evaluated alone."

        callbacks.append(
            class_config(
                EvaluatorCallback,
                evaluator=get_scannet_evaluator_cfg(scannet200=True),
                metrics_to_eval=["3D"],
                save_predictions=True,
                output_dir=output_dir,
                save_prefix="detection",
                test_connector=class_config(
                    CallbackConnector, key_mapping=CONN_COCO_DET3D_EVAL
                ),
            )
        )
    elif len(open_test_datasets) > 0:
        evaluators = []
        for dataset in open_test_datasets:
            if dataset == "Argoverse_val":
                evaluators.append(get_av2_evaluator_cfg())
            elif dataset == "ScanNet_val":
                evaluators.append(get_scannet_evaluator_cfg())
            elif dataset == "LabelAny3D_COCO_val":
                evaluators.append(get_labelany3d_coco_evaluator_cfg())
            else:
                raise ValueError(
                    f"Unknown dataset {dataset} for open evaluation."
                )

        callbacks.append(
            class_config(
                EvaluatorCallback,
                evaluator=class_config(
                    OpenDetect3DEvaluator,
                    datasets=open_test_datasets,
                    evaluators=evaluators,
                    omni3d_evaluator=omni3d_evaluator,
                ),
                metrics_to_eval=["3D"],
                save_predictions=True,
                output_dir=output_dir,
                save_prefix="detection",
                test_connector=class_config(
                    CallbackConnector, key_mapping=CONN_OMNI3D_DET3D_EVAL
                ),
            )
        )
    else:
        assert omni3d_evaluator is not None, "No evaluator provided."
        callbacks.append(
            class_config(
                EvaluatorCallback,
                evaluator=omni3d_evaluator,
                metrics_to_eval=["3D"],
                save_predictions=True,
                output_dir=output_dir,
                save_prefix="detection",
                test_connector=class_config(
                    CallbackConnector, key_mapping=CONN_OMNI3D_DET3D_EVAL
                ),
            )
        )

    # Visualizer
    callbacks.extend(
        get_visualizer_callback_cfg(
            output_dir, visualize_depth=visualize_depth
        )
    )

    # Optional: export per-image cache for depth-based post-processing
    if export_postprocess_cache:
        # output_dir may be a FieldReference; resolve it to avoid creating
        # directories like "<FieldReference object at ...>/postprocess_cache".
        resolved_output_dir = (
            output_dir.get() if hasattr(output_dir, "get") else str(output_dir)
        )
        cache_root = (
            postprocess_cache_root
            if postprocess_cache_root is not None
            else os.path.join(str(resolved_output_dir), "postprocess_cache")
        )
        callbacks.append(
            class_config(
                EvaluatorCallback,
                evaluator=class_config(
                    PostprocessCacheExporter,
                    cache_root=cache_root,
                    compress=True,
                    overwrite=False,
                    depth_dtype="float32",
                ),
                metrics_to_eval=[],
                save_predictions=False,
                output_dir=output_dir,
                save_prefix="postprocess_cache",
                test_connector=class_config(
                    CallbackConnector, key_mapping=CONN_POSTPROCESS_CACHE_EXPORT
                ),
            )
        )

    return callbacks


def get_in_the_wild_evaluator_cfg(
    data_root: str = "data/in_the_wild",
    iou_type: str = "bbox",
    human_filtered: bool = True,
    enable_aprel3d: bool = False,
    annotation_name: str | None = None,
) -> ConfigDict:
    """Get InTheWild evaluator config.

    Uses Detect3DEvaluator for the 800+ category human-annotated
    in-the-wild dataset (COCO/LVIS/Objects365).

    Args:
        data_root: Root directory for in-the-wild data.
        iou_type: "bbox" for 3D IoU matching, "dist" for center distance.
        human_filtered: If True, use human-filtered annotations
            (valid3D + 2-pass human review). If False, use original.
        annotation_name: Override annotation file name (without .json).
            If None, uses human_filtered flag to pick.

    Returns:
        ConfigDict: Evaluator configuration.
    """
    from wilddet3d.data.datasets.in_the_wild import load_in_the_wild_class_map

    if annotation_name is not None:
        ann_name = annotation_name
    else:
        ann_name = (
            "InTheWild_val_final" if human_filtered else "InTheWild_val"
        )
    annotation = os.path.join(data_root, f"annotations/{ann_name}.json")
    class_map = load_in_the_wild_class_map(annotation)
    det_map = {name: i for i, name in enumerate(sorted(class_map.keys()))}

    return class_config(
        Detect3DEvaluator,
        det_map=det_map,
        cat_map=class_map,
        eval_prox=False,
        iou_type=iou_type,
        num_columns=4,
        annotation=annotation,
        freq_rare_thresh=5,
        freq_freq_thresh=20,
        enable_aprel3d=enable_aprel3d,
    )


def get_in_the_wild_eval_callbacks(
    data_root: str = "data/in_the_wild",
    output_dir: FieldReference | str = "",
    eval_connector_mapping: dict | None = None,
    test_connector: ConfigDict | None = None,
    enable_aprel3d: bool = False,
    annotation_name: str | None = None,
) -> list[ConfigDict]:
    """Get InTheWild evaluation callbacks: 2D AP, 3D AP (bbox), 3D AP (dist).

    Args:
        data_root: Root directory for in-the-wild data.
        output_dir: Output directory for saving predictions.
        eval_connector_mapping: Key mapping for CallbackConnector (GDino3D).
            Ignored if test_connector is provided.
        test_connector: Pre-built connector config (WildDet3D). If None,
            builds CallbackConnector from eval_connector_mapping.
        enable_aprel3d: Whether to enable APRel3D (per-image scale alignment).
        annotation_name: Override annotation file name (without .json).

    Returns:
        List of EvaluatorCallback configs.
    """
    if test_connector is None:
        from configs.base.base_connector import CONN_COCO_DET3D_EVAL

        if eval_connector_mapping is None:
            eval_connector_mapping = CONN_COCO_DET3D_EVAL
        test_connector = class_config(
            CallbackConnector, key_mapping=eval_connector_mapping
        )

    cbs = []

    # 2D + 3D AP (bbox mode: 3D IoU matching)
    cbs.append(
        class_config(
            EvaluatorCallback,
            evaluator=get_in_the_wild_evaluator_cfg(
                data_root=data_root,
                iou_type="bbox",
                enable_aprel3d=enable_aprel3d,
                annotation_name=annotation_name,
            ),
            metrics_to_eval=["2D", "3D"],
            save_predictions=True,
            output_dir=output_dir,
            save_prefix="detection_bbox",
            test_connector=test_connector,
        )
    )

    # 3D AP (dist mode: center distance matching)
    cbs.append(
        class_config(
            EvaluatorCallback,
            evaluator=get_in_the_wild_evaluator_cfg(
                data_root=data_root,
                iou_type="dist",
                enable_aprel3d=enable_aprel3d,
                annotation_name=annotation_name,
            ),
            metrics_to_eval=["3D"],
            save_predictions=True,
            output_dir=output_dir,
            save_prefix="detection_dist",
            test_connector=test_connector,
        )
    )

    return cbs


def get_stereo4d_evaluator_cfg(
    data_root: str = "data/in_the_wild",
    iou_type: str = "bbox",
    enable_aprel3d: bool = False,
) -> ConfigDict:
    """Get Stereo4D val evaluator config.

    Uses Detect3DEvaluator for the Stereo4D val benchmark:
    500 images with real stereo depth and human-reviewed 3D boxes.

    APr/APc/APf thresholds: rare<5 images, common 5-10, frequent>=10.

    Args:
        data_root: Root directory for data.
        iou_type: "bbox" for 3D IoU matching, "dist" for center distance.

    Returns:
        ConfigDict: Evaluator configuration.
    """
    from wilddet3d.data.datasets.stereo4d import load_stereo4d_class_map

    annotation = os.path.join(
        data_root, "annotations/Stereo4D_val.json"
    )
    class_map = load_stereo4d_class_map(annotation)
    det_map = {name: i for i, name in enumerate(sorted(class_map.keys()))}

    return class_config(
        Detect3DEvaluator,
        det_map=det_map,
        cat_map=class_map,
        eval_prox=False,
        iou_type=iou_type,
        num_columns=4,
        annotation=annotation,
        freq_rare_thresh=5,
        freq_freq_thresh=10,
        enable_aprel3d=enable_aprel3d,
    )


def get_stereo4d_eval_callbacks(
    data_root: str = "data/in_the_wild",
    output_dir: FieldReference | str = "",
    eval_connector_mapping: dict | None = None,
    test_connector: ConfigDict | None = None,
    enable_aprel3d: bool = False,
) -> list[ConfigDict]:
    """Get Stereo4D val evaluation callbacks: 2D AP, 3D AP (bbox), 3D AP (dist).

    Args:
        data_root: Root directory for data.
        output_dir: Output directory for saving predictions.
        eval_connector_mapping: Key mapping for CallbackConnector (GDino3D).
            Ignored if test_connector is provided.
        test_connector: Pre-built connector config (WildDet3D). If None,
            builds CallbackConnector from eval_connector_mapping.
        enable_aprel3d: Whether to enable APRel3D (per-image scale alignment).

    Returns:
        List of EvaluatorCallback configs.
    """
    if test_connector is None:
        from configs.base.base_connector import CONN_COCO_DET3D_EVAL

        if eval_connector_mapping is None:
            eval_connector_mapping = CONN_COCO_DET3D_EVAL
        test_connector = class_config(
            CallbackConnector, key_mapping=eval_connector_mapping
        )

    cbs = []

    # 2D + 3D AP (bbox mode: 3D IoU matching)
    cbs.append(
        class_config(
            EvaluatorCallback,
            evaluator=get_stereo4d_evaluator_cfg(
                data_root=data_root,
                iou_type="bbox",
                enable_aprel3d=enable_aprel3d,
            ),
            metrics_to_eval=["2D", "3D"],
            save_predictions=True,
            output_dir=output_dir,
            save_prefix="detection_bbox",
            test_connector=test_connector,
        )
    )

    # 3D AP (dist mode: center distance matching)
    cbs.append(
        class_config(
            EvaluatorCallback,
            evaluator=get_stereo4d_evaluator_cfg(
                data_root=data_root,
                iou_type="dist",
                enable_aprel3d=enable_aprel3d,
            ),
            metrics_to_eval=["3D"],
            save_predictions=True,
            output_dir=output_dir,
            save_prefix="detection_dist",
            test_connector=test_connector,
        )
    )

    return cbs


def get_omni3d_evaluator_cfg(
    data_root: str,
    omni3d50: bool,
    test_datasets: list[str],
    use_mini_dataset: bool = False,
) -> ConfigDict:
    """Get Omni3D evaluator config.

    Args:
        data_root: Root directory for Omni3D data.
        omni3d50: Whether to use Omni3D-50 class mapping.
        test_datasets: List of test dataset names.
        use_mini_dataset: If True, use mini100 annotations for evaluation.
    """
    return class_config(
        Omni3DEvaluator,
        data_root=data_root,
        omni3d50=omni3d50,
        datasets=test_datasets,
        use_mini_dataset=use_mini_dataset,
    )


def get_av2_evaluator_cfg(data_root: str = "data/argoverse") -> ConfigDict:
    """Get Argoverse 2 evaluator config."""
    return class_config(
        Detect3DEvaluator,
        det_map=av2_det_map,
        cat_map=av2_class_map,
        eval_prox=True,
        iou_type="dist",
        num_columns=2,
        annotation=os.path.join(data_root, "annotations/Argoverse_val.json"),
        base_classes=[
            "regular vehicle",
            "pedestrian",
            "bicyclist",
            "construction cone",
            "construction barrel",
            "large vehicle",
            "bus",
            "truck",
            "vehicular trailer",
            "bicycle",
            "motorcycle",
        ],
    )


def get_scannet_evaluator_cfg(
    data_root: str = "data/scannet", scannet200: bool = False
) -> ConfigDict:
    """Get ScanNet evaluator config."""
    if scannet200:
        s_det_map = scannet200_det_map
        s_class_map = scannet200_class_map
        annotation = os.path.join(data_root, "annotations/ScanNet200_val.json")
        base_classes = None
    else:
        s_det_map = scannet_det_map
        s_class_map = scannet_class_map
        annotation = os.path.join(data_root, "annotations/ScanNet_val.json")
        base_classes = [
            "cabinet",
            "bed",
            "chair",
            "sofa",
            "table",
            "door",
            "window",
            "picture",
            "counter",
            "desk",
            "curtain",
            "refrigerator",
            "toilet",
            "sink",
            "bathtub",
        ]

    return class_config(
        Detect3DEvaluator,
        det_map=s_det_map,
        cat_map=s_class_map,
        iou_type="dist",
        num_columns=2,
        annotation=annotation,
        base_classes=base_classes,
    )


def get_labelany3d_coco_evaluator_cfg(
    data_root: str = "data/labelany3d_coco",
) -> ConfigDict:
    """Get LabelAny3D COCO evaluator config.

    This evaluator is configured for the LabelAny3D COCO dataset which contains
    COCO val2017 images with 3D bounding box annotations. Uses 3D IoU-based
    evaluation metrics.

    Args:
        data_root: Root directory for the LabelAny3D COCO data.

    Returns:
        ConfigDict: Evaluator configuration.
    """
    # Select a subset of commonly-annotated COCO categories as base classes
    # These are categories that typically have reliable 3D annotations
    base_classes = [
        "person",
        "car",
        "truck",
        "bus",
        "motorcycle",
        "bicycle",
        "chair",
        "couch",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "refrigerator",
        "oven",
        "sink",
        "bottle",
        "cup",
        "bowl",
        "potted plant",
    ]

    return class_config(
        Detect3DEvaluator,
        det_map=labelany3d_coco_det_map,
        cat_map=labelany3d_coco_class_map,
        eval_prox=False,  # Not a proximity-based dataset
        iou_type="bbox",  # Use 3D bounding box IoU
        num_columns=4,  # Display 4 columns in results table (80 categories)
        annotation=os.path.join(
            data_root, "annotations/LabelAny3D_COCO_val.json"
        ),
        base_classes=base_classes,
    )


def get_visualizer_callback_cfg(
    output_dir: str | FieldReference,
    visualize_depth: bool = False,
    vis_freq: int = 50,
    width: int = 4,
    font_size: int = 16,
    save_boxes3d: bool = True,
) -> list[ConfigDict]:
    """Get basic callbacks."""
    callbacks = []

    callbacks.append(
        class_config(
            VisualizerCallback,
            visualizer=class_config(
                BoundingBox3DVisualizer,
                axis_mode=AxisMode.OPENCV,
                width=width,
                camera_near_clip=0.01,
                plot_heading=False,
                vis_freq=vis_freq,
                plot_trajectory=False,
                canvas=class_config(PillowCanvasBackend, font_size=font_size),
                save_boxes3d=save_boxes3d,
            ),
            output_dir=output_dir,
            save_prefix="box3d",
            test_connector=class_config(
                CallbackConnector, key_mapping=CONN_BBOX_3D_VIS
            ),
        )
    )

    if visualize_depth:
        callbacks.append(
            class_config(
                VisualizerCallback,
                visualizer=class_config(
                    DepthVisualizer,
                    plot_error=False,
                    lift=True,
                    vis_freq=vis_freq,
                ),
                output_dir=output_dir,
                save_prefix="depth",
                test_connector=class_config(
                    CallbackConnector, key_mapping=CONN_DEPTH_VIS
                ),
            )
        )

    return callbacks

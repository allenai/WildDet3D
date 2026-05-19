"""WildDet3D + LingBot-Depth on ScanNet - Text + GT depth zero-shot eval.

Same as text.py but feeds the FoundationStereo / sensor GT depth map to
LingBot at inference time (use_depth_input_test=True).
"""

from __future__ import annotations

from vis4d.config import class_config
from vis4d.config.typing import ExperimentConfig
from vis4d.data.io.hdf5 import HDF5Backend
from vis4d.engine.callbacks import EvaluatorCallback
from vis4d.zoo.base import get_default_callbacks_cfg, get_default_cfg

from configs.base.callback import (
    get_scannet_evaluator_cfg,
    get_visualizer_callback_cfg,
)
from configs.base.connector import (
    WildDet3DEvalConnector,
    get_wilddet3d_data_connector_cfg,
)
from configs.base.data import get_wilddet3d_data_cfg
from configs.base.dataset.omni3d import get_omni3d_train_cfg
from configs.base.dataset.open import get_scannet_data_cfg
from configs.base.loss import get_wilddet3d_loss_cfg
from configs.base.model import (
    get_wilddet3d_cfg,
    get_wilddet3d_hyperparams_cfg,
)
from configs.base.optim import get_wilddet3d_optim_cfg
from configs.base.pl import get_pl_cfg

from wilddet3d.eval.open import OpenDetect3DEvaluator


def get_config() -> ExperimentConfig:
    """WildDet3D + LingBot text + GT-depth eval on ScanNet."""
    config = get_default_cfg(
        exp_name="wilddet3d_lingbot_depth_freeze21_scannet_with_depth"
    )

    config.use_checkpoint = True

    params = get_wilddet3d_hyperparams_cfg(
        num_epochs=12,
        samples_per_gpu=4,
        workers_per_gpu=4,
        base_lr=1e-4,
    )

    data_backend = class_config(HDF5Backend)
    sam3_image_shape = (1008, 1008)
    omni3d_data_root = "data/omni3d"

    omni3d_train_data_cfg = get_omni3d_train_cfg(
        data_root=omni3d_data_root,
        data_backend=data_backend,
        shape=sam3_image_shape,
    )
    scannet_test_data_cfg = get_scannet_data_cfg(
        data_backend=data_backend,
        shape=sam3_image_shape,
        with_depth=True,
    )

    config.data = get_wilddet3d_data_cfg(
        train_datasets=omni3d_train_data_cfg,
        test_datasets=[scannet_test_data_cfg],
        samples_per_gpu=params.samples_per_gpu,
        workers_per_gpu=params.workers_per_gpu,
        max_prompts_per_image=50,
        use_text_prompts=True,
        use_geometry_prompts=True,
    )

    config.model, box_coder = get_wilddet3d_cfg(
        params=params,
        sam3_checkpoint="pretrained/sam3/sam3_detector.pt",
        geometry_backend_type="lingbot_depth",
        lingbot_encoder_freeze_blocks=21,
        backbone_freeze_blocks=28,
        canonical_rotation=True,
        use_depth_input_test=True,
    )

    config.loss = get_wilddet3d_loss_cfg(params, box_coder)

    config.optimizers = get_wilddet3d_optim_cfg(
        params,
        freeze_backbone=params.freeze_backbone,
        freeze_all_pretrained=params.freeze_all_pretrained,
    )

    config.train_data_connector, config.test_data_connector = (
        get_wilddet3d_data_connector_cfg()
    )

    callbacks = get_default_callbacks_cfg()

    open_test_datasets = ["ScanNet_val"]
    evaluators = [get_scannet_evaluator_cfg()]

    callbacks.append(
        class_config(
            EvaluatorCallback,
            evaluator=class_config(
                OpenDetect3DEvaluator,
                datasets=open_test_datasets,
                evaluators=evaluators,
                omni3d_evaluator=None,
            ),
            metrics_to_eval=["3D"],
            save_predictions=True,
            output_dir=config.output_dir,
            save_prefix="detection",
            test_connector=class_config(WildDet3DEvalConnector),
        )
    )

    callbacks.extend(
        get_visualizer_callback_cfg(
            output_dir=config.output_dir,
            visualize_depth=False,
        )
    )

    config.callbacks = callbacks
    config.pl_trainer = get_pl_cfg(config, params)

    return config.value_mode()

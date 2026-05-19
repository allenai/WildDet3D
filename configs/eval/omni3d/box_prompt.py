"""WildDet3D + LingBot-Depth on Omni3D - Box prompt (oracle) eval.

GT 2D boxes as geometry prompts (oracle_eval=True). Isolates 3D
regression quality from 2D detection.

Evaluates WildDet3D on the standard 6-subset Omni3D test split (KITTI,
nuScenes, SUNRGBD, Hypersim, ARKitScenes, Objectron). Omni3D is the
primary training distribution; this is the in-domain eval reported in
the paper. Uses `Omni3DEvaluator` which reports `AP_<Subset>` per
sub-dataset plus a macro AP.

Mode: text prompts, monocular depth (no GT depth input).
"""

from __future__ import annotations

from vis4d.config import class_config
from vis4d.config.typing import ExperimentConfig
from vis4d.data.io.hdf5 import HDF5Backend
from vis4d.zoo.base import get_default_cfg

from configs.base.callback import (
    get_callback_cfg,
    get_omni3d_evaluator_cfg,
)
from configs.base.connector import get_wilddet3d_data_connector_cfg
from configs.base.data import get_wilddet3d_data_cfg
from configs.base.dataset.omni3d import (
    get_omni3d_test_cfg,
    get_omni3d_train_cfg,
)
from configs.base.loss import get_wilddet3d_loss_cfg
from configs.base.model import (
    get_wilddet3d_cfg,
    get_wilddet3d_hyperparams_cfg,
)
from configs.base.optim import get_wilddet3d_optim_cfg
from configs.base.pl import get_pl_cfg


OMNI3D_TEST_DATASETS = (
    "KITTI_test",
    "nuScenes_test",
    "SUNRGBD_test",
    "Hypersim_test",
    "ARKitScenes_test",
    "Objectron_test",
)


def get_config() -> ExperimentConfig:
    """WildDet3D + LingBot box-prompt eval on Omni3D."""
    config = get_default_cfg(
        exp_name="wilddet3d_lingbot_depth_freeze21_omni3d_box_prompt"
    )

    config.use_checkpoint = True

    params = get_wilddet3d_hyperparams_cfg(
        num_epochs=12,
        samples_per_gpu=4,
        workers_per_gpu=4,
        base_lr=1e-4,
    )

    ######################################################
    ##                     DATA                         ##
    ######################################################
    data_backend = class_config(HDF5Backend)
    sam3_image_shape = (1008, 1008)
    omni3d_data_root = "data/omni3d"

    omni3d_train_data_cfg = get_omni3d_train_cfg(
        data_root=omni3d_data_root,
        data_backend=data_backend,
        shape=sam3_image_shape,
    )
    omni3d_test_data_cfg = get_omni3d_test_cfg(
        data_root=omni3d_data_root,
        test_datasets=OMNI3D_TEST_DATASETS,
        data_backend=data_backend,
        shape=sam3_image_shape,
    )

    config.data = get_wilddet3d_data_cfg(
        train_datasets=omni3d_train_data_cfg,
        test_datasets=[omni3d_test_data_cfg],
        samples_per_gpu=params.samples_per_gpu,
        workers_per_gpu=params.workers_per_gpu,
        max_prompts_per_image=50,
        use_text_prompts=True,
        use_geometry_prompts=True,
        oracle_eval=True,
    )

    ######################################################
    ##                  MODEL & LOSS                    ##
    ######################################################
    config.model, box_coder = get_wilddet3d_cfg(
        params=params,
        sam3_checkpoint="pretrained/sam3/sam3_detector.pt",
        geometry_backend_type="lingbot_depth",
        lingbot_encoder_freeze_blocks=21,
        backbone_freeze_blocks=28,
        canonical_rotation=True,
        oracle_eval=True,
    )

    config.loss = get_wilddet3d_loss_cfg(params, box_coder)

    ######################################################
    ##                    OPTIMIZERS                    ##
    ######################################################
    config.optimizers = get_wilddet3d_optim_cfg(
        params,
        freeze_backbone=params.freeze_backbone,
        freeze_all_pretrained=params.freeze_all_pretrained,
    )

    ######################################################
    ##                  DATA CONNECTOR                  ##
    ######################################################
    config.train_data_connector, config.test_data_connector = (
        get_wilddet3d_data_connector_cfg()
    )

    ######################################################
    ##                     CALLBACKS                    ##
    ######################################################
    config.callbacks = get_callback_cfg(
        output_dir=config.output_dir,
        open_test_datasets=[],
        omni3d_evaluator=get_omni3d_evaluator_cfg(
            data_root=omni3d_data_root,
            omni3d50=True,
            test_datasets=OMNI3D_TEST_DATASETS,
        ),
        visualize_depth=False,
    )

    ######################################################
    ##                     PL CLI                       ##
    ######################################################
    config.pl_trainer = get_pl_cfg(config, params)

    return config.value_mode()

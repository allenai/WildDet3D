"""WildDet3D: Open-Vocabulary Monocular 3D Object Detection in the Wild."""

import sys
from pathlib import Path

# Add third_party submodules to Python path
_third_party = Path(__file__).parent.parent / "third_party"
_sam3_path = str(_third_party / "sam3")
_lingbot_path = str(_third_party / "lingbot_depth")
_moge_path = str(_third_party / "moge")

if _sam3_path not in sys.path:
    sys.path.insert(0, _sam3_path)
if _lingbot_path not in sys.path:
    sys.path.insert(0, _lingbot_path)
if _moge_path not in sys.path:
    sys.path.insert(0, _moge_path)

from .data_types import Det3DOut, WildDet3DInput, WildDet3DOut
from .inference import WildDet3DPredictor, build_model
from .model import WildDet3D
from .optimize import optimize_for_inference
from .preprocessing import preprocess

__all__ = [
    "WildDet3D",
    "WildDet3DPredictor",
    "WildDet3DInput",
    "WildDet3DOut",
    "Det3DOut",
    "build_model",
    "optimize_for_inference",
    "preprocess",
]

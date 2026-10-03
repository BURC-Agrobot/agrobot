# Detector backends package.
# Import all three detector classes so ROS 2 nodes and eval scripts can do:
#   from agrobot_perception.detectors import SAM2AMGDetector
# All three implement the same interface: detect(preprocessed_chw) → list[dict].
from agrobot_perception.detectors.dino_sam2_detector import DINOv2SAM2Detector
from agrobot_perception.detectors.sam2_amg_detector import SAM2AMGDetector
from agrobot_perception.detectors.sam2_semantic_detector import SAM2SemanticDetector

__all__ = ["DINOv2SAM2Detector", "SAM2AMGDetector", "SAM2SemanticDetector"]

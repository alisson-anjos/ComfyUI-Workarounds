"""
ComfyUI-Workaround
A collection of utility nodes for ComfyUI
"""

from .nodes.face.geometric_align import GeometricFaceAlign
from .nodes.face.face_mask import FaceMaskCreator
from .nodes.face.face_detect import FaceLandmarkDetector
from .nodes.face.face_3d_projection import Face3DProjection
from .nodes.face.planar_overlay import PlanarFaceOverlay
from .nodes.face.face_region_options import FaceRegionOptions
from .nodes.face.skin_tone import SkinToneDetector, SkinToneColorMatch
from .nodes.image.transform import ImageTransform
from .nodes.image.composite import ImageComposite

from .nodes.flowmatch_scheduler import (
    FlowMatchScheduler,
    FlowMatchSchedulerPresets,
    FlowMatchAutoConfig,
    FlowMatchGuide
)

__version__ = "1.1.0"

NODE_CLASS_MAPPINGS = {
    # Face nodes
    "WA_GeometricFaceAlign": GeometricFaceAlign,
    "WA_FaceMaskCreator": FaceMaskCreator,
    "WA_FaceLandmarkDetector": FaceLandmarkDetector,
    "WA_Face3DProjection": Face3DProjection,
    "WA_PlanarFaceOverlay": PlanarFaceOverlay,
    "WA_FaceRegionOptions": FaceRegionOptions,
    "WA_SkinToneDetector": SkinToneDetector,
    "WA_SkinToneColorMatch": SkinToneColorMatch,
    
    # Image nodes
    "WA_ImageTransform": ImageTransform,
    "WA_ImageComposite": ImageComposite,
    
    "FlowMatchScheduler": FlowMatchScheduler,
    "FlowMatchSchedulerPresets": FlowMatchSchedulerPresets,
    "FlowMatchAutoConfig": FlowMatchAutoConfig,
    "FlowMatchGuide": FlowMatchGuide,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WA_GeometricFaceAlign": "Geometric Face Align 🔧",
    "WA_FaceMaskCreator": "Face Mask Creator 🔧",
    "WA_FaceLandmarkDetector": "Face Landmark Detector 🔧",
    "WA_Face3DProjection": "3D Face Projection 🔧",
    "WA_PlanarFaceOverlay": "Planar Face Overlay 🔧",
    "WA_FaceRegionOptions": "Face Region Options 🔧",
    "WA_SkinToneDetector": "Skin Tone Detector 🔧",
    "WA_SkinToneColorMatch": "Skin Tone Color Match 🔧",
    "WA_ImageTransform": "Image Transform 🔧",
    "WA_ImageComposite": "Image Composite 🔧",
    "FlowMatchScheduler": "FlowMatch Scheduler (Advanced)",
    "FlowMatchSchedulerPresets": "FlowMatch Scheduler Presets",
    "FlowMatchAutoConfig": "FlowMatch Auto Config",
    "FlowMatchGuide": "FlowMatch Settings Guide",
}

WEB_DIRECTORY = "./web"
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
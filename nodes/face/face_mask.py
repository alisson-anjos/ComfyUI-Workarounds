"""
Face Mask Creator Node
Creates precise face masks using facial landmarks
"""

import torch
import cv2
import numpy as np
from ...utils.image_utils import tensor_to_numpy, numpy_to_tensor
from ...utils.landmark_utils import detect_face_landmarks

class FaceMaskCreator:
    """
    Creates a mask of the face region using landmark detection
    """
    
    CATEGORY = "Workaround/Face"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask_type": (["face_oval", "face_with_forehead", "full_head"], {
                    "default": "face_oval"
                }),
                "feather": ("INT", {
                    "default": 5,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "display": "slider"
                }),
                "expand": ("INT", {
                    "default": 0,
                    "min": -50,
                    "max": 100,
                    "step": 1,
                    "display": "slider"
                }),
            }
        }
    
    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("mask", "preview")
    FUNCTION = "create_mask"
    
    def create_mask(self, image, mask_type, feather, expand):
        """
        Creates a face mask
        
        Args:
            image: Input image
            mask_type: Type of mask to create
            feather: Edge feathering amount
            expand: Mask expansion/contraction
        
        Returns:
            tuple: (mask, preview image)
        """
        
        # Convert to numpy
        image_np = tensor_to_numpy(image)
        h, w = image_np.shape[:2]
        
        # Detect landmarks
        landmarks = detect_face_landmarks(image_np)
        
        if landmarks is None:
            print("[FaceMaskCreator] Failed to detect face")
            # Return empty mask
            empty_mask = torch.zeros((1, h, w))
            return (empty_mask, image)
        
        # Create mask based on type
        mask = self._create_mask_by_type(landmarks, (h, w), mask_type)
        
        # Expand/contract mask
        if expand != 0:
            kernel_size = abs(expand) * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            
            if expand > 0:
                mask = cv2.dilate(mask, kernel, iterations=1)
            else:
                mask = cv2.erode(mask, kernel, iterations=1)
        
        # Apply feathering
        if feather > 0:
            kernel_size = feather * 2 + 1
            mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
        
        # Create preview (overlay mask on image)
        preview = self._create_preview(image_np, mask)
        
        # Convert to tensors
        mask_tensor = torch.from_numpy(mask.astype(np.float32) / 255.0).unsqueeze(0)
        preview_tensor = numpy_to_tensor(preview)
        
        return (mask_tensor, preview_tensor)
    
    def _create_mask_by_type(self, landmarks, shape, mask_type):
        """Creates different types of face masks"""
        mask = np.zeros(shape[:2], dtype=np.uint8)
        
        if mask_type == "face_oval":
            # Face oval (excludes forehead)
            face_oval_idx = [
                10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
            ]
            points = landmarks[face_oval_idx].astype(np.int32)
            
        elif mask_type == "face_with_forehead":
            # Face with forehead
            face_contour_idx = [
                10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
                10, 109, 67, 103, 54, 21, 162, 127, 234, 93, 132, 58, 172,
                136, 150, 149, 176, 148, 152  # Top of head
            ]
            points = landmarks[face_contour_idx].astype(np.int32)
            
        else:  # full_head
            # Full head using convex hull
            points = cv2.convexHull(landmarks.astype(np.int32))
        
        cv2.fillPoly(mask, [points], 255)
        
        return mask
    
    def _create_preview(self, image, mask):
        """Creates preview with mask overlay"""
        preview = image.copy()
        
        # Create colored overlay
        overlay = preview.copy()
        overlay[mask > 0] = [0, 255, 0]  # Green
        
        # Blend
        mask_3ch = np.stack([mask] * 3, axis=-1) / 255.0 * 0.5
        preview = (preview * (1 - mask_3ch) + overlay * mask_3ch).astype(np.uint8)
        
        return preview
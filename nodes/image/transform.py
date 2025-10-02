"""
Image Transform Node
Advanced image transformations (affine, perspective, etc.)
"""

import torch
import cv2
import numpy as np
from ...utils.image_utils import tensor_to_numpy, numpy_to_tensor

class ImageTransform:
    """
    Applies various geometric transformations to images
    """
    
    CATEGORY = "Workaround/Image"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "transform_type": (["rotate", "scale", "translate", "shear", "flip"], {
                    "default": "rotate"
                }),
                "angle": ("FLOAT", {
                    "default": 0.0,
                    "min": -180.0,
                    "max": 180.0,
                    "step": 0.1,
                    "display": "slider"
                }),
                "scale_x": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1
                }),
                "scale_y": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1
                }),
                "translate_x": ("INT", {
                    "default": 0,
                    "min": -2000,
                    "max": 2000,
                    "step": 1
                }),
                "translate_y": ("INT", {
                    "default": 0,
                    "min": -2000,
                    "max": 2000,
                    "step": 1
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "transform"
    
    def transform(self, image, transform_type, angle, scale_x, scale_y, translate_x, translate_y):
        """
        Applies geometric transformation to image
        
        Args:
            image: Input image
            transform_type: Type of transformation
            angle: Rotation angle in degrees
            scale_x: Horizontal scale factor
            scale_y: Vertical scale factor
            translate_x: Horizontal translation in pixels
            translate_y: Vertical translation in pixels
        
        Returns:
            Transformed image
        """
        
        # Convert to numpy
        image_np = tensor_to_numpy(image)
        h, w = image_np.shape[:2]
        
        if transform_type == "rotate":
            result = self._rotate(image_np, angle)
            
        elif transform_type == "scale":
            result = self._scale(image_np, scale_x, scale_y)
            
        elif transform_type == "translate":
            result = self._translate(image_np, translate_x, translate_y)
            
        elif transform_type == "shear":
            result = self._shear(image_np, angle)
            
        elif transform_type == "flip":
            if angle > 0:
                result = cv2.flip(image_np, 1)  # Horizontal
            else:
                result = cv2.flip(image_np, 0)  # Vertical
        else:
            result = image_np
        
        # Convert back to tensor
        result_tensor = numpy_to_tensor(result)
        
        return (result_tensor,)
    
    def _rotate(self, image, angle):
        """Rotates image around center"""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        
        return rotated
    
    def _scale(self, image, scale_x, scale_y):
        """Scales image"""
        h, w = image.shape[:2]
        new_w = int(w * scale_x)
        new_h = int(h * scale_y)
        
        scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Pad or crop to original size
        if new_w < w or new_h < h:
            # Pad
            result = np.zeros_like(image)
            y_offset = (h - new_h) // 2
            x_offset = (w - new_w) // 2
            result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = scaled
        else:
            # Crop
            y_offset = (new_h - h) // 2
            x_offset = (new_w - w) // 2
            result = scaled[y_offset:y_offset+h, x_offset:x_offset+w]
        
        return result
    
    def _translate(self, image, tx, ty):
        """Translates image"""
        h, w = image.shape[:2]
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        translated = cv2.warpAffine(image, M, (w, h))
        
        return translated
    
    def _shear(self, image, angle):
        """Applies shear transformation"""
        h, w = image.shape[:2]
        shear_factor = np.tan(np.radians(angle))
        
        M = np.float32([
            [1, shear_factor, 0],
            [0, 1, 0]
        ])
        
        sheared = cv2.warpAffine(image, M, (w, h))
        
        return sheared
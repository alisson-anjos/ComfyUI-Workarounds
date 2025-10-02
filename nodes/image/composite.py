"""
Image Composite Node
Smart image composition with masks
"""

import torch
import cv2
import numpy as np
from ...utils.image_utils import tensor_to_numpy, numpy_to_tensor

class ImageComposite:
    """
    Composites two images using a mask with various blending modes
    """
    
    CATEGORY = "Workaround/Image"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "background": ("IMAGE",),
                "foreground": ("IMAGE",),
                "mask": ("MASK",),
                "blend_mode": (["normal", "multiply", "screen", "overlay", "add"], {
                    "default": "normal"
                }),
                "opacity": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider"
                }),
                "feather": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "step": 1
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "composite"
    
    def composite(self, background, foreground, mask, blend_mode, opacity, feather):
        """
        Composites foreground onto background using mask
        
        Args:
            background: Background image
            foreground: Foreground image
            mask: Mask (white = foreground, black = background)
            blend_mode: Blending mode
            opacity: Overall opacity of foreground
            feather: Edge feathering amount
        
        Returns:
            Composited image
        """
        
        # Convert to numpy
        bg_np = tensor_to_numpy(background)
        fg_np = tensor_to_numpy(foreground)
        mask_np = (mask[0].cpu().numpy() * 255).astype(np.uint8)
        
        # Resize foreground and mask to match background
        h, w = bg_np.shape[:2]
        if fg_np.shape[:2] != (h, w):
            fg_np = cv2.resize(fg_np, (w, h))
        if mask_np.shape[:2] != (h, w):
            mask_np = cv2.resize(mask_np, (w, h))
        
        # Apply feathering
        if feather > 0:
            kernel_size = feather * 2 + 1
            mask_np = cv2.GaussianBlur(mask_np, (kernel_size, kernel_size), 0)
        
        # Apply opacity
        mask_np = (mask_np * opacity).astype(np.uint8)
        
        # Apply blending mode
        if blend_mode == "normal":
            result = self._blend_normal(bg_np, fg_np, mask_np)
        elif blend_mode == "multiply":
            result = self._blend_multiply(bg_np, fg_np, mask_np)
        elif blend_mode == "screen":
            result = self._blend_screen(bg_np, fg_np, mask_np)
        elif blend_mode == "overlay":
            result = self._blend_overlay(bg_np, fg_np, mask_np)
        elif blend_mode == "add":
            result = self._blend_add(bg_np, fg_np, mask_np)
        else:
            result = self._blend_normal(bg_np, fg_np, mask_np)
        
        # Convert back to tensor
        result_tensor = numpy_to_tensor(result)
        
        return (result_tensor,)
    
    def _blend_normal(self, bg, fg, mask):
        """Normal blending"""
        mask_3ch = np.stack([mask] * 3, axis=-1) / 255.0
        result = bg * (1 - mask_3ch) + fg * mask_3ch
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _blend_multiply(self, bg, fg, mask):
        """Multiply blending"""
        mask_3ch = np.stack([mask] * 3, axis=-1) / 255.0
        blended = (bg.astype(float) * fg.astype(float)) / 255.0
        result = bg * (1 - mask_3ch) + blended * mask_3ch
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _blend_screen(self, bg, fg, mask):
        """Screen blending"""
        mask_3ch = np.stack([mask] * 3, axis=-1) / 255.0
        blended = 255 - ((255 - bg.astype(float)) * (255 - fg.astype(float))) / 255.0
        result = bg * (1 - mask_3ch) + blended * mask_3ch
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _blend_overlay(self, bg, fg, mask):
        """Overlay blending"""
        mask_3ch = np.stack([mask] * 3, axis=-1) / 255.0
        
        # Overlay formula
        bg_norm = bg.astype(float) / 255.0
        fg_norm = fg.astype(float) / 255.0
        
        blended = np.where(
            bg_norm < 0.5,
            2 * bg_norm * fg_norm,
            1 - 2 * (1 - bg_norm) * (1 - fg_norm)
        ) * 255.0
        
        result = bg * (1 - mask_3ch) + blended * mask_3ch
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _blend_add(self, bg, fg, mask):
        """Additive blending"""
        mask_3ch = np.stack([mask] * 3, axis=-1) / 255.0
        blended = bg.astype(float) + fg.astype(float) * mask_3ch
        return np.clip(blended, 0, 255).astype(np.uint8)
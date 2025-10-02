"""
Image utility functions
"""

import torch
import numpy as np
import cv2

def tensor_to_numpy(tensor):
    """
    Converts ComfyUI tensor to numpy array (OpenCV format)
    
    Args:
        tensor: Tensor in ComfyUI format (B, H, W, C) with values [0, 1]
    
    Returns:
        numpy array in format (H, W, C) with values [0, 255]
    """
    if isinstance(tensor, torch.Tensor):
        image = tensor[0].cpu().numpy()
    else:
        image = tensor[0]
    
    # Convert from [0, 1] to [0, 255]
    image = (image * 255).astype(np.uint8)
    
    # Convert RGB to BGR (OpenCV)
    if image.shape[-1] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    return image

def numpy_to_tensor(array):
    """
    Converts numpy array (OpenCV format) to ComfyUI tensor
    
    Args:
        array: Numpy array in format (H, W, C) with values [0, 255]
    
    Returns:
        Tensor in ComfyUI format (1, H, W, C) with values [0, 1]
    """
    # Convert BGR to RGB (if needed)
    if array.shape[-1] == 3:
        array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
    
    # Convert from [0, 255] to [0, 1]
    image = array.astype(np.float32) / 255.0
    
    # Add batch dimension
    tensor = torch.from_numpy(image).unsqueeze(0)
    
    return tensor

def resize_image(image, width=None, height=None, keep_aspect=True):
    """Resizes image maintaining aspect ratio"""
    h, w = image.shape[:2]
    
    if width is None and height is None:
        return image
    
    if keep_aspect:
        if width is not None:
            height = int(h * (width / w))
        elif height is not None:
            width = int(w * (height / h))
    
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

def pad_image(image, target_width, target_height, color=(0, 0, 0)):
    """Pads image to target size"""
    h, w = image.shape[:2]
    
    if w == target_width and h == target_height:
        return image
    
    # Calculate padding
    pad_top = (target_height - h) // 2
    pad_bottom = target_height - h - pad_top
    pad_left = (target_width - w) // 2
    pad_right = target_width - w - pad_left
    
    return cv2.copyMakeBorder(
        image, 
        pad_top, pad_bottom, 
        pad_left, pad_right,
        cv2.BORDER_CONSTANT, 
        value=color
    )
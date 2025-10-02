"""
Face Landmark Detector Node
Detects and visualizes facial landmarks
"""

import torch
import cv2
import numpy as np
from ...utils.image_utils import tensor_to_numpy, numpy_to_tensor
from ...utils.landmark_utils import detect_face_landmarks, get_face_bbox

class FaceLandmarkDetector:
    """
    Detects facial landmarks and visualizes them
    """
    
    CATEGORY = "Workaround/Face"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "draw_landmarks": ("BOOLEAN", {"default": True}),
                "draw_connections": ("BOOLEAN", {"default": True}),
                "draw_bbox": ("BOOLEAN", {"default": True}),
                "point_size": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 10,
                    "step": 1
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("annotated_image", "landmark_info")
    FUNCTION = "detect_landmarks"
    
    def detect_landmarks(self, image, draw_landmarks, draw_connections, draw_bbox, point_size):
        """
        Detects and visualizes facial landmarks
        
        Args:
            image: Input image
            draw_landmarks: Whether to draw landmark points
            draw_connections: Whether to draw connections between landmarks
            draw_bbox: Whether to draw bounding box
            point_size: Size of landmark points
        
        Returns:
            tuple: (annotated image, info string)
        """
        
        # Convert to numpy
        image_np = tensor_to_numpy(image)
        
        # Detect landmarks
        landmarks = detect_face_landmarks(image_np)
        
        if landmarks is None:
            info = "No face detected"
            return (image, info)
        
        # Create annotated image
        annotated = image_np.copy()
        
        # Draw bounding box
        if draw_bbox:
            x_min, y_min, x_max, y_max = get_face_bbox(landmarks)
            cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        # Draw connections
        if draw_connections:
            self._draw_connections(annotated, landmarks)
        
        # Draw landmarks
        if draw_landmarks:
            for point in landmarks:
                x, y = int(point[0]), int(point[1])
                cv2.circle(annotated, (x, y), point_size, (255, 0, 0), -1)
        
        # Create info string
        num_landmarks = len(landmarks)
        bbox = get_face_bbox(landmarks)
        face_width = bbox[2] - bbox[0]
        face_height = bbox[3] - bbox[1]
        
        info = f"Detected {num_landmarks} landmarks\n"
        info += f"Face size: {face_width}x{face_height}px\n"
        info += f"BBox: ({bbox[0]}, {bbox[1]}) to ({bbox[2]}, {bbox[3]})"
        
        # Convert back to tensor
        annotated_tensor = numpy_to_tensor(annotated)
        
        return (annotated_tensor, info)
    
    def _draw_connections(self, image, landmarks):
        """Draws connections between facial landmarks"""
        try:
            from mediapipe.python.solutions.face_mesh_connections import (
                FACEMESH_TESSELATION,
                FACEMESH_CONTOURS,
                FACEMESH_IRISES
            )
            
            # Draw face contours
            for connection in FACEMESH_CONTOURS:
                start_idx, end_idx = connection
                start_point = tuple(landmarks[start_idx].astype(int))
                end_point = tuple(landmarks[end_idx].astype(int))
                cv2.line(image, start_point, end_point, (0, 255, 0), 1)
            
        except ImportError:
            # Fallback: draw simple connections
            pass
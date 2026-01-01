"""
Face Landmark Detector Node (Advanced)
Detects and visualizes facial landmarks with region filtering
"""

import torch
import cv2
import numpy as np
import json
from ...utils.image_utils import tensor_to_numpy, numpy_to_tensor
from ...utils.landmark_utils import detect_face_landmarks, get_face_bbox

class FaceLandmarkDetector:
    """
    Detects facial landmarks with granular control over which facial features to visualize.
    Great for expression transfer without enforcing head shape.
    """
    
    CATEGORY = "Workaround/Face"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "output_mode": (["annotated_image", "landmarks_only_image", "json_data"], {"default": "landmarks_only_image"}),
                
                "draw_landmarks_points": ("BOOLEAN", {"default": False}),
                "draw_connections_lines": ("BOOLEAN", {"default": True}),
                "draw_bbox": ("BOOLEAN", {"default": False}),
                
                "include_face_oval": ("BOOLEAN", {"default": False}), 
                "include_eyebrows": ("BOOLEAN", {"default": True}),
                "include_eyes": ("BOOLEAN", {"default": True}),
                "include_nose": ("BOOLEAN", {"default": True}),
                "include_lips": ("BOOLEAN", {"default": True}),
                
                "point_size": ("INT", {"default": 2, "min": 1, "max": 10, "step": 1}),
                "line_thickness": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "stroke_color": ("INT", {"default": 255, "min": 0, "max": 255}),
                "background_color": ("INT", {"default": 0, "min": 0, "max": 255}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("output_image", "info_string")
    FUNCTION = "detect_landmarks"
    
    def detect_landmarks(self, image, output_mode, 
                        draw_landmarks_points, draw_connections_lines, draw_bbox,
                        include_face_oval, include_eyebrows, include_eyes, include_nose, include_lips,
                        point_size, line_thickness, stroke_color, background_color):
        
        image_np = tensor_to_numpy(image)
        height, width, _ = image_np.shape
        
        landmarks = detect_face_landmarks(image_np)
        
        if landmarks is None:
            info = "No face detected"
            if output_mode == "landmarks_only_image":
                blank = np.full((height, width, 3), background_color, dtype=np.uint8)
                return (numpy_to_tensor(blank), info)
            return (image, info)

        if output_mode == "landmarks_only_image":
            canvas = np.full((height, width, 3), background_color, dtype=np.uint8)
        elif output_mode == "annotated_image":
            canvas = image_np.copy()
        else: # JSON Mode
            canvas = np.full((height, width, 3), 0, dtype=np.uint8)

        landmark_info_str = ""

        if output_mode != "json_data":
            color_tuple = (stroke_color, stroke_color, stroke_color)
            
            try:
                from mediapipe.python.solutions.face_mesh_connections import (
                    FACEMESH_FACE_OVAL,
                    FACEMESH_LIPS,
                    FACEMESH_LEFT_EYE, FACEMESH_RIGHT_EYE,
                    FACEMESH_LEFT_EYEBROW, FACEMESH_RIGHT_EYEBROW,
                    FACEMESH_NOSE,
                    FACEMESH_IRISES
                )
                
                connections_to_draw = []
                
                if include_face_oval:
                    connections_to_draw.extend(FACEMESH_FACE_OVAL)
                if include_lips:
                    connections_to_draw.extend(FACEMESH_LIPS)
                if include_eyes:
                    connections_to_draw.extend(FACEMESH_LEFT_EYE)
                    connections_to_draw.extend(FACEMESH_RIGHT_EYE)
                    connections_to_draw.extend(FACEMESH_IRISES) # Opcional, adiciona detalhe
                if include_eyebrows:
                    connections_to_draw.extend(FACEMESH_LEFT_EYEBROW)
                    connections_to_draw.extend(FACEMESH_RIGHT_EYEBROW)
                if include_nose:
                    connections_to_draw.extend(FACEMESH_NOSE)

                if draw_connections_lines:
                    for connection in connections_to_draw:
                        start_idx = connection[0]
                        end_idx = connection[1]
                        
                        pt1 = (int(landmarks[start_idx][0]), int(landmarks[start_idx][1]))
                        pt2 = (int(landmarks[end_idx][0]), int(landmarks[end_idx][1]))
                        
                        cv2.line(canvas, pt1, pt2, color_tuple, line_thickness)

                if draw_landmarks_points:
                    active_indices = set()
                    for connection in connections_to_draw:
                        active_indices.add(connection[0])
                        active_indices.add(connection[1])
                    
                    for idx in active_indices:
                        pt = (int(landmarks[idx][0]), int(landmarks[idx][1]))
                        cv2.circle(canvas, pt, point_size, color_tuple, -1)

            except ImportError:
                landmark_info_str = "Error: MediaPipe dependencies missing for granular drawing."
                if draw_landmarks_points:
                    for point in landmarks:
                        x, y = int(point[0]), int(point[1])
                        cv2.circle(canvas, (x, y), point_size, color_tuple, -1)

            if draw_bbox:
                x_min, y_min, x_max, y_max = get_face_bbox(landmarks)
                cv2.rectangle(canvas, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        if output_mode == "json_data":
            normalized_list = [[p[0]/width, p[1]/height] for p in landmarks]
            landmark_info_str = json.dumps(normalized_list, indent=None)
        elif not landmark_info_str:
            landmark_info_str = f"Landmarks detected. Mode: {output_mode}"

        return (numpy_to_tensor(canvas), landmark_info_str)
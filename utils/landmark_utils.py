"""
Face landmark detection utilities
"""

import numpy as np
import cv2

# Cache for MediaPipe model
_face_mesh_detector = None

def detect_face_landmarks(image, refine=True):
    """
    Detects facial landmarks using MediaPipe
    
    Args:
        image: Numpy array image (H, W, C)
        refine: If True, uses refined landmarks (more accurate)
    
    Returns:
        numpy array (N, 2) with landmark coordinates or None
    """
    global _face_mesh_detector
    
    try:
        import mediapipe as mp
        
        # Initialize detector (lazy loading)
        if _face_mesh_detector is None:
            mp_face_mesh = mp.solutions.face_mesh
            _face_mesh_detector = mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=refine,
                min_detection_confidence=0.5
            )
        
        # Detect
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = _face_mesh_detector.process(rgb_image)
        
        if results.multi_face_landmarks:
            h, w = image.shape[:2]
            landmarks = []
            
            for lm in results.multi_face_landmarks[0].landmark:
                landmarks.append([lm.x * w, lm.y * h])
            
            return np.array(landmarks, dtype=np.float32)
        
    except ImportError:
        print("[LandmarkUtils] MediaPipe not installed. Run: pip install mediapipe")
    except Exception as e:
        print(f"[LandmarkUtils] Error detecting landmarks: {e}")
    
    return None

def get_alignment_points(source_landmarks, target_landmarks, num_points=5):
    """
    Returns key points for facial alignment
    
    Args:
        source_landmarks: Source image landmarks
        target_landmarks: Target image landmarks
        num_points: Number of points to use (5 or 8)
    
    Returns:
        tuple: (source_points, target_points)
    """
    if num_points == 5:
        # Basic points: eyes, nose, mouth corners
        key_idx = [33, 263, 1, 61, 291]
    else:
        # Extended points for perspective transform
        key_idx = [33, 263, 1, 61, 291, 10, 152, 234]
    
    src_pts = source_landmarks[key_idx]
    dst_pts = target_landmarks[key_idx]
    
    return src_pts, dst_pts

def get_face_bbox(landmarks, padding=0.2):
    """
    Returns face bounding box with optional padding
    
    Args:
        landmarks: Face landmarks array
        padding: Padding percentage (0.2 = 20%)
    
    Returns:
        tuple: (x_min, y_min, x_max, y_max)
    """
    x_min = np.min(landmarks[:, 0])
    x_max = np.max(landmarks[:, 0])
    y_min = np.min(landmarks[:, 1])
    y_max = np.max(landmarks[:, 1])
    
    # Add padding
    width = x_max - x_min
    height = y_max - y_min
    
    x_min = int(x_min - width * padding)
    x_max = int(x_max + width * padding)
    y_min = int(y_min - height * padding)
    y_max = int(y_max + height * padding)
    
    return x_min, y_min, x_max, y_max

def calculate_face_angle(landmarks):
    """
    Calculates face rotation angles (yaw, pitch, roll)
    
    Args:
        landmarks: Face landmarks array
    
    Returns:
        dict: {'yaw': float, 'pitch': float, 'roll': float} in degrees
    """
    # Get key points
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    nose = landmarks[1]
    
    # Calculate roll (rotation around z-axis)
    eye_center = (left_eye + right_eye) / 2
    eye_vector = right_eye - left_eye
    roll = np.degrees(np.arctan2(eye_vector[1], eye_vector[0]))
    
    # Simplified yaw and pitch estimation
    # (For more accurate results, use 3D landmarks)
    yaw = 0  # Would need 3D landmarks for accurate calculation
    pitch = 0  # Would need 3D landmarks for accurate calculation
    
    return {
        'yaw': yaw,
        'pitch': pitch,
        'roll': roll
    }
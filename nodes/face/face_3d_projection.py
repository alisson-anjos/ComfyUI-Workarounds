"""
3D Face Projection Node - Complete Fixed Edition
Projects face onto 3D model, rotates it, and renders back to 2D
"""

import torch
import cv2
import numpy as np
from ...utils.image_utils import tensor_to_numpy, numpy_to_tensor
from ...utils.landmark_utils import detect_face_landmarks

class Face3DProjection:
    """
    Projects 2D face onto 3D model for accurate pose matching
    """
    
    CATEGORY = "Workaround/Face"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_face": ("IMAGE",),
                "target_body": ("IMAGE",),
                
                # 3D method
                "projection_method": (["simple_affine", "cylindrical", "spherical", "perspective", "tps"], {
                    "default": "simple_affine"
                }),
                
                # Alignment settings
                "alignment_mode": (["landmarks", "center", "eyes", "auto"], {
                    "default": "landmarks"
                }),
                
                "vertical_align": (["none", "nose_to_center", "eyes_level", "auto"], {
                    "default": "none"
                }),
                
                "horizontal_align": (["none", "center", "nose_vertical", "auto"], {
                    "default": "none"
                }),
                
                "face_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.5,
                    "max": 2.0,
                    "step": 0.05,
                    "display": "slider"
                }),
                
                "rotation_adjust": ("FLOAT", {
                    "default": 0.0,
                    "min": -45.0,
                    "max": 45.0,
                    "step": 0.5,
                    "display": "slider"
                }),
                
                # Mask settings
                "mask_mode": (["face_only", "face_with_forehead", "full_head"], {
                    "default": "face_only"
                }),
                
                "mask_expand": ("INT", {
                    "default": 0,
                    "min": -50,
                    "max": 100,
                    "step": 1,
                    "display": "slider"
                }),
                
                "feather_edges": ("INT", {
                    "default": 20,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "display": "slider"
                }),
                
                # Color matching
                "color_match": ("BOOLEAN", {"default": True}),
                
                "color_match_method": (["mean", "histogram", "lab"], {
                    "default": "lab"
                }),
                
                "color_match_strength": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider"
                }),
                
                # Output options
                "output_landmarks": ("BOOLEAN", {"default": True}),
                "show_alignment_grid": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("result", "face_mask", "debug_view", "landmark_view", "projection_only")
    FUNCTION = "project_face"
    
    def project_face(self, source_face, target_body, projection_method, 
                    alignment_mode, vertical_align, horizontal_align,
                    face_scale, rotation_adjust,
                    mask_mode, mask_expand, feather_edges,
                    color_match, color_match_method, color_match_strength,
                    output_landmarks, show_alignment_grid):
        
        # Convert to numpy
        source_np = tensor_to_numpy(source_face)
        target_np = tensor_to_numpy(target_body)
        h, w = target_np.shape[:2]
        
        print(f"\n{'='*60}")
        print(f"[Face3DProjection] Starting projection")
        print(f"[Face3DProjection] Source: {source_np.shape}, Target: {target_np.shape}")
        print(f"[Face3DProjection] Method: {projection_method}")
        print(f"{'='*60}\n")
        
        # Detect landmarks
        print("[Face3DProjection] Detecting landmarks...")
        source_landmarks = detect_face_landmarks(source_np)
        target_landmarks = detect_face_landmarks(target_np)
        
        if source_landmarks is None or target_landmarks is None:
            print("[Face3DProjection] ❌ Failed to detect landmarks")
            empty = torch.zeros((1, h, w))
            return (target_body, empty, target_body, target_body, target_body)
        
        print(f"[Face3DProjection] ✓ Detected landmarks")
        
        # Calculate poses
        source_pose = self._estimate_pose(source_landmarks, source_np.shape[:2])
        target_pose = self._estimate_pose(target_landmarks, target_np.shape[:2])
        
        print(f"\n[Pose Analysis]")
        print(f"  Source: yaw={source_pose['yaw']:>6.1f}° pitch={source_pose['pitch']:>6.1f}° roll={source_pose['roll']:>6.1f}°")
        print(f"  Target: yaw={target_pose['yaw']:>6.1f}° pitch={target_pose['pitch']:>6.1f}° roll={target_pose['roll']:>6.1f}°")
        
        # Auto-detect best alignment if set to auto
        if alignment_mode == "auto":
            angle_diff = abs(target_pose['yaw'] - source_pose['yaw'])
            if angle_diff < 10:
                alignment_mode = "landmarks"
            elif angle_diff < 30:
                alignment_mode = "eyes"
            else:
                alignment_mode = "center"
        
        # Project face based on method
        print(f"\n[Face3DProjection] Projecting with {projection_method}...")
        
        if projection_method == "cylindrical":
            projected = self._cylindrical_projection(
                source_np, source_landmarks, target_landmarks,
                source_pose, target_pose, face_scale
            )
        elif projection_method == "spherical":
            projected = self._spherical_projection(
                source_np, source_landmarks, target_landmarks,
                source_pose, target_pose, face_scale
            )
        else:
            # Simple affine, perspective, tps
            projected = self._project_with_alignment(
                source_np, source_landmarks, target_landmarks,
                source_pose, target_pose,
                projection_method, alignment_mode,
                vertical_align, horizontal_align,
                face_scale, rotation_adjust
            )
        
        # Verify projection
        non_zero = np.count_nonzero(projected)
        print(f"[Face3DProjection] Projection: {non_zero:,} non-zero pixels")
        
        # Store projection-only
        projection_only = projected.copy()
        
        # Create mask
        print(f"[Face3DProjection] Creating {mask_mode} mask...")
        mask = self._create_face_mask_advanced(target_landmarks, (h, w), mask_mode)
        
        # Expand/contract mask
        if mask_expand != 0:
            kernel_size = abs(mask_expand) * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            if mask_expand > 0:
                mask = cv2.dilate(mask, kernel, iterations=1)
            else:
                mask = cv2.erode(mask, kernel, iterations=1)
        
        # Color matching
        if color_match:
            print(f"[Face3DProjection] Color matching ({color_match_method})...")
            projected = self._match_color(projected, target_np, mask, color_match_method, color_match_strength)
        
        # Feather mask
        if feather_edges > 0:
            kernel_size = feather_edges * 2 + 1
            mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
        
        # Composite
        print("[Face3DProjection] Compositing...")
        mask_3ch = np.stack([mask] * 3, axis=-1) / 255.0
        result = target_np * (1 - mask_3ch) + projected * mask_3ch
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        # Create outputs
        if output_landmarks:
            landmark_view = self._draw_landmarks_detailed(
                target_np.copy(), source_landmarks, target_landmarks,
                source_pose, target_pose, projection_method
            )
        else:
            landmark_view = target_np.copy()
        
        if show_alignment_grid:
            debug = self._draw_alignment_grid(
                result.copy(), target_landmarks, 
                vertical_align, horizontal_align, mask
            )
        else:
            debug = self._draw_pose_info(result.copy(), source_pose, target_pose, mask)
        
        # Convert to tensors
        result_tensor = numpy_to_tensor(result)
        mask_tensor = torch.from_numpy(mask.astype(np.float32) / 255.0).unsqueeze(0)
        debug_tensor = numpy_to_tensor(debug)
        landmark_tensor = numpy_to_tensor(landmark_view)
        projection_tensor = numpy_to_tensor(projection_only)
        
        print(f"{'='*60}")
        print("[Face3DProjection] ✓ Complete!")
        print(f"{'='*60}\n")
        
        return (result_tensor, mask_tensor, debug_tensor, landmark_tensor, projection_tensor)
    
    def _project_with_alignment(self, source, source_lm, target_lm,
                                source_pose, target_pose,
                                method, align_mode, v_align, h_align,
                                scale, rotation):
        """
        Affine/Perspective/TPS projection with alignment
        """
        src_h, src_w = source.shape[:2]
        
        # Get alignment points - ALWAYS use 5+ points for robust estimation
        if align_mode == "eyes":
            indices = [33, 263, 1, 61, 291]  # Eyes, nose, mouth
        elif align_mode == "center":
            indices = [1, 33, 263, 61, 291, 152]  # Nose, eyes, mouth, chin
        else:  # landmarks
            indices = [33, 263, 1, 61, 291, 152, 10, 234]  # More points for stability
        
        src_pts = source_lm[indices].copy().astype(np.float32)
        dst_pts = target_lm[indices].copy().astype(np.float32)
        
        # Apply vertical alignment
        if v_align != "none":
            if v_align == "auto":
                v_align = "eyes_level"
            
            if v_align == "eyes_level" and len(src_pts) >= 2:
                # Align eye level
                src_eye_y = (src_pts[0, 1] + src_pts[1, 1]) / 2
                dst_eye_y = (dst_pts[0, 1] + dst_pts[1, 1]) / 2
                
                # Adjust all destination points vertically
                offset_y = dst_eye_y - src_eye_y
                for i in range(len(src_pts)):
                    rel_y = src_pts[i, 1] - src_eye_y
                    dst_pts[i, 1] = dst_eye_y + rel_y
        
        # Apply horizontal alignment
        if h_align != "none":
            if h_align == "auto":
                h_align = "nose_vertical"
            
            if h_align == "nose_vertical" and len(src_pts) >= 3:
                # Align nose vertical
                src_nose_x = src_pts[2, 0]  # Nose
                dst_nose_x = dst_pts[2, 0]
                offset_x = dst_nose_x - src_nose_x
                src_pts[:, 0] += offset_x
        
        # Apply scale
        if scale != 1.0:
            dst_center = np.mean(dst_pts, axis=0)
            dst_pts = (dst_pts - dst_center) * scale + dst_center
        
        # Apply rotation
        if rotation != 0.0:
            dst_center = np.mean(dst_pts, axis=0)
            angle_rad = np.radians(rotation)
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)
            
            for i in range(len(dst_pts)):
                rel = dst_pts[i] - dst_center
                dst_pts[i] = dst_center + np.array([
                    rel[0] * cos_a - rel[1] * sin_a,
                    rel[0] * sin_a + rel[1] * cos_a
                ])
        
        # Apply transformation
        if method == "simple_affine":
            # Use estimateAffinePartial2D for robust estimation with RANSAC
            M, inliers = cv2.estimateAffinePartial2D(
                src_pts, dst_pts, 
                method=cv2.RANSAC,
                ransacReprojThreshold=5.0,
                maxIters=2000,
                confidence=0.99
            )
            
            if M is None:
                print("[Affine] ⚠️  Estimation failed, trying full affine")
                M = cv2.getAffineTransform(src_pts[:3], dst_pts[:3])
            
            result = cv2.warpAffine(source, M, (src_w, src_h), 
                                   flags=cv2.INTER_CUBIC,
                                   borderMode=cv2.BORDER_CONSTANT)
            
        elif method == "perspective":
            if len(src_pts) >= 4:
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if M is not None:
                    result = cv2.warpPerspective(source, M, (src_w, src_h),
                                                 flags=cv2.INTER_CUBIC,
                                                 borderMode=cv2.BORDER_CONSTANT)
                else:
                    print("[Perspective] Failed, using affine")
                    M = cv2.estimateAffinePartial2D(src_pts, dst_pts)[0]
                    result = cv2.warpAffine(source, M, (src_w, src_h))
            else:
                print("[Perspective] Not enough points")
                result = source
                
        elif method == "tps":
            result = self._tps_warp(source, src_pts, dst_pts, (src_w, src_h))
            
        else:
            result = source
        
        return result
    
    def _cylindrical_projection(self, source, source_lm, target_lm,
                               source_pose, target_pose, scale):
        """
        Cylindrical projection - FIXED VERSION
        """
        print("[Cylindrical] Starting projection")
        
        src_h, src_w = source.shape[:2]
        
        # Get face crop from source
        src_x_min = max(0, int(source_lm[:, 0].min() * 0.8))
        src_x_max = min(src_w, int(source_lm[:, 0].max() * 1.2))
        src_y_min = max(0, int(source_lm[:, 1].min() * 0.8))
        src_y_max = min(src_h, int(source_lm[:, 1].max() * 1.2))
        
        face_crop = source[src_y_min:src_y_max, src_x_min:src_x_max].copy()
        
        if face_crop.size == 0:
            print("[Cylindrical] Empty crop, using affine")
            return self._project_with_alignment(
                source, source_lm, target_lm,
                source_pose, target_pose,
                "simple_affine", "landmarks", "none", "none", scale, 0
            )
        
        crop_h, crop_w = face_crop.shape[:2]
        
        # Calculate angle difference
        yaw_diff = target_pose['yaw'] - source_pose['yaw']
        pitch_diff = target_pose['pitch'] - source_pose['pitch']
        
        print(f"[Cylindrical] Angles: yaw={yaw_diff:.1f}° pitch={pitch_diff:.1f}°")
        
        # For small angles, use affine (faster and more stable)
        if abs(yaw_diff) < 15 and abs(pitch_diff) < 15:
            print("[Cylindrical] Small angles, using affine")
            return self._project_with_alignment(
                source, source_lm, target_lm,
                source_pose, target_pose,
                "simple_affine", "landmarks", "none", "none", scale, 0
            )
        
        # Cylindrical warp for larger angles
        print("[Cylindrical] Large angles, using cylindrical warp")
        
        # Cylinder parameters
        radius = crop_w / (np.pi * 2)
        yaw_rad = np.radians(yaw_diff)
        pitch_factor = pitch_diff / 45.0
        
        # Create warped face
        warped_w = int(crop_w * 1.2)
        warped_h = int(crop_h * 1.2)
        warped = np.zeros((warped_h, warped_w, 3), dtype=np.uint8)
        
        # Warp pixels
        for y in range(warped_h):
            for x in range(warped_w):
                # Relative coordinates
                rel_x = x - warped_w / 2
                rel_y = y - warped_h / 2
                
                # Inverse cylindrical mapping
                theta = np.arctan2(rel_x, radius) - yaw_rad
                
                # Source coordinates
                src_x = radius * np.sin(theta) + crop_w / 2
                src_y = rel_y + crop_h / 2 - (pitch_factor * crop_h * 0.15)
                
                # Sample with bilinear interpolation
                if 0 <= src_x < crop_w - 1 and 0 <= src_y < crop_h - 1:
                    x0, y0 = int(src_x), int(src_y)
                    x1, y1 = min(x0 + 1, crop_w - 1), min(y0 + 1, crop_h - 1)
                    
                    dx = src_x - x0
                    dy = src_y - y0
                    
                    pixel = (
                        face_crop[y0, x0] * (1 - dx) * (1 - dy) +
                        face_crop[y0, x1] * dx * (1 - dy) +
                        face_crop[y1, x0] * (1 - dx) * dy +
                        face_crop[y1, x1] * dx * dy
                    )
                    
                    warped[y, x] = pixel.astype(np.uint8)
        
        # Now place warped face in target position using affine
        # Use target landmarks to position the warped face
        result = np.zeros_like(source)
        
        tgt_center_x = int(target_lm[:, 0].mean())
        tgt_center_y = int(target_lm[:, 1].mean())
        
        tgt_width = int((target_lm[:, 0].max() - target_lm[:, 0].min()) * scale * 1.1)
        tgt_height = int((target_lm[:, 1].max() - target_lm[:, 1].min()) * scale * 1.1)
        
        # Resize warped to target size
        if warped.shape[:2] != (tgt_height, tgt_width):
            warped = cv2.resize(warped, (tgt_width, tgt_height), interpolation=cv2.INTER_CUBIC)
        
        # Place in result
        y1 = max(0, tgt_center_y - tgt_height // 2)
        y2 = min(src_h, y1 + tgt_height)
        x1 = max(0, tgt_center_x - tgt_width // 2)
        x2 = min(src_w, x1 + tgt_width)
        
        wy1 = 0 if y1 >= 0 else abs(y1)
        wy2 = wy1 + (y2 - y1)
        wx1 = 0 if x1 >= 0 else abs(x1)
        wx2 = wx1 + (x2 - x1)
        
        if wy2 > wy1 and wx2 > wx1:
            result[y1:y2, x1:x2] = warped[wy1:wy2, wx1:wx2]
        
        print(f"[Cylindrical] Placed at ({x1},{y1}) size {tgt_width}x{tgt_height}")
        
        return result
    
    def _spherical_projection(self, source, source_lm, target_lm,
                             source_pose, target_pose, scale):
        """
        Spherical projection - FIXED VERSION
        Similar to cylindrical but handles pitch better
        """
        print("[Spherical] Starting projection")
        
        # For now, use cylindrical with adjusted parameters
        # TODO: Implement true spherical projection
        return self._cylindrical_projection(
            source, source_lm, target_lm,
            source_pose, target_pose, scale
        )
    
    def _tps_warp(self, source, src_pts, dst_pts, size):
        """Thin Plate Spline warping"""
        w, h = size
        
        try:
            tps = cv2.createThinPlateSplineShapeTransformer()
            
            src_shape = src_pts.reshape(1, -1, 2).astype(np.float32)
            dst_shape = dst_pts.reshape(1, -1, 2).astype(np.float32)
            
            matches = [cv2.DMatch(i, i, 0) for i in range(len(src_pts))]
            tps.estimateTransformation(dst_shape, src_shape, matches)
            
            result = tps.warpImage(source)
            
            if result.shape[:2] != (h, w):
                result = cv2.resize(result, (w, h))
            
            return result
            
        except Exception as e:
            print(f"[TPS] Failed: {e}")
            M = cv2.estimateAffinePartial2D(src_pts, dst_pts)[0]
            if M is not None:
                return cv2.warpAffine(source, M, (w, h))
            return np.zeros((h, w, 3), dtype=np.uint8)
    
    def _estimate_pose(self, landmarks, image_shape):
        """Estimate head pose"""
        h, w = image_shape
        
        model_points = np.array([
            (0.0, 0.0, 0.0),
            (0.0, -330.0, -65.0),
            (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0),
            (-150.0, -150.0, -125.0),
            (150.0, -150.0, -125.0)
        ], dtype=np.float64)
        
        image_points = np.array([
            landmarks[1], landmarks[152], landmarks[33],
            landmarks[263], landmarks[61], landmarks[291]
        ], dtype=np.float64)
        
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        
        dist_coeffs = np.zeros((4, 1))
        
        success, rotation_vec, translation_vec = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return {'yaw': 0, 'pitch': 0, 'roll': 0}
        
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        sy = np.sqrt(rotation_mat[0, 0] ** 2 + rotation_mat[1, 0] ** 2)
        
        if sy > 1e-6:
            pitch = np.arctan2(rotation_mat[2, 1], rotation_mat[2, 2])
            yaw = np.arctan2(-rotation_mat[2, 0], sy)
            roll = np.arctan2(rotation_mat[1, 0], rotation_mat[0, 0])
        else:
            pitch = np.arctan2(-rotation_mat[1, 2], rotation_mat[1, 1])
            yaw = np.arctan2(-rotation_mat[2, 0], sy)
            roll = 0
        
        return {
            'yaw': np.degrees(yaw),
            'pitch': np.degrees(pitch),
            'roll': np.degrees(roll)
        }
    
    def _create_face_mask_advanced(self, landmarks, shape, mask_mode):
        """Create face mask"""
        mask = np.zeros(shape[:2], dtype=np.uint8)
        
        if mask_mode == "full_head":
            points = cv2.convexHull(landmarks.astype(np.int32))
            cv2.fillPoly(mask, [points], 255)
        elif mask_mode == "face_with_forehead":
            indices = [
                10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
                10, 109, 67, 103, 54, 21
            ]
            max_idx = len(landmarks)
            indices = [i for i in indices if i < max_idx]
            points = landmarks[indices].astype(np.int32)
            cv2.fillPoly(mask, [points], 255)
        else:  # face_only
            face_oval = [
                10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
            ]
            max_idx = len(landmarks)
            face_oval = [i for i in face_oval if i < max_idx]
            points = landmarks[face_oval].astype(np.int32)
            cv2.fillPoly(mask, [points], 255)
        
        return mask
    
    def _match_color(self, source, target, mask, method, strength):
        """Color matching"""
        mask_bool = mask > 127
        
        if not np.any(mask_bool):
            return source
        
        if method == "lab":
            source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
            target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)
            result_lab = source_lab.copy()
            
            for c in range(3):
                src_pixels = source_lab[:, :, c][mask_bool]
                tgt_pixels = target_lab[:, :, c][mask_bool]
                
                if len(src_pixels) > 0 and len(tgt_pixels) > 0:
                    src_mean, src_std = np.mean(src_pixels), np.std(src_pixels)
                    tgt_mean, tgt_std = np.mean(tgt_pixels), np.std(tgt_pixels)
                    
                    if src_std > 0:
                        result_lab[:, :, c] = (result_lab[:, :, c] - src_mean) / src_std
                        result_lab[:, :, c] = result_lab[:, :, c] * tgt_std + tgt_mean
                    
                    result_lab[:, :, c] = (
                        result_lab[:, :, c] * strength +
                        source_lab[:, :, c] * (1 - strength)
                    )
            
            result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)
            return cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)
        
        elif method == "histogram":
            result = source.copy()
            for c in range(3):
                src_channel = source[:, :, c]
                tgt_pixels = target[:, :, c][mask_bool]
                
                if len(tgt_pixels) > 0:
                    matched = self._match_histogram(src_channel, tgt_pixels)
                    result[:, :, c] = (matched * strength + src_channel * (1 - strength)).astype(np.uint8)
            return result
        
        else:  # mean
            result = source.copy().astype(np.float32)
            for c in range(3):
                src_mean = np.mean(source[:, :, c][mask_bool])
                tgt_mean = np.mean(target[:, :, c][mask_bool])
                shift = (tgt_mean - src_mean) * strength
                result[:, :, c] += shift
            return np.clip(result, 0, 255).astype(np.uint8)
    
    def _match_histogram(self, source_img, target_pixels):
        """Histogram matching"""
        source_hist, _ = np.histogram(source_img.flatten(), 256, [0, 256])
        target_hist, _ = np.histogram(target_pixels.flatten(), 256, [0, 256])
        
        source_cdf = source_hist.cumsum()
        target_cdf = target_hist.cumsum()
        
        source_cdf = source_cdf / source_cdf[-1]
        target_cdf = target_cdf / target_cdf[-1]
        
        lookup = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            j = np.searchsorted(target_cdf, source_cdf[i])
            lookup[i] = min(j, 255)
        
        return lookup[source_img]
    
    def _draw_alignment_grid(self, image, landmarks, v_align, h_align, mask):
        """Draw alignment grid"""
        h, w = image.shape[:2]
        
        # Draw vertical center line
        center_x = int(landmarks[:, 0].mean())
        cv2.line(image, (center_x, 0), (center_x, h), (0, 255, 0), 1)
        cv2.putText(image, "Center", (center_x + 5, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw nose vertical
        nose_x = int(landmarks[1, 0])
        cv2.line(image, (nose_x, 0), (nose_x, h), (255, 0, 0), 1)
        cv2.putText(image, "Nose", (nose_x + 5, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Draw eyes level
        left_eye_y = int(landmarks[33, 1])
        right_eye_y = int(landmarks[263, 1])
        eyes_y = (left_eye_y + right_eye_y) // 2
        cv2.line(image, (0, eyes_y), (w, eyes_y), (0, 255, 255), 1)
        cv2.putText(image, "Eyes", (10, eyes_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Draw face center horizontal
        center_y = int(landmarks[:, 1].mean())
        cv2.line(image, (0, center_y), (w, center_y), (255, 255, 0), 1)
        
        # Draw key points
        key_points = [1, 33, 263, 61, 291, 152]  # Nose, eyes, mouth, chin
        for idx in key_points:
            if idx < len(landmarks):
                pt = tuple(landmarks[idx].astype(int))
                cv2.circle(image, pt, 5, (255, 255, 255), -1)
                cv2.circle(image, pt, 6, (0, 0, 255), 2)
        
        # Draw info
        y = h - 100
        cv2.rectangle(image, (5, y - 30), (250, h - 10), (0, 0, 0), -1)
        
        cv2.putText(image, f"V-Align: {v_align}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(image, f"V-Align: {v_align}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        y += 25
        cv2.putText(image, f"H-Align: {h_align}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(image, f"H-Align: {h_align}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        return image
    
    def _draw_landmarks_detailed(self, image, source_lm, target_lm, 
                                 source_pose, target_pose, method):
        """Draw detailed landmarks"""
        # Draw all target landmarks
        for i, point in enumerate(target_lm):
            x, y = int(point[0]), int(point[1])
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                if i in [1, 33, 263, 61, 291, 152]:  # Key points
                    cv2.circle(image, (x, y), 4, (0, 255, 0), -1)
                    cv2.circle(image, (x, y), 6, (255, 255, 255), 1)
                else:
                    cv2.circle(image, (x, y), 1, (0, 255, 0), -1)
        
        # Draw face contour
        face_oval = [
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
            397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
            172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
        ]
        
        max_idx = len(target_lm)
        for i in range(len(face_oval)):
            if face_oval[i] < max_idx and face_oval[(i+1)%len(face_oval)] < max_idx:
                pt1 = tuple(target_lm[face_oval[i]].astype(int))
                pt2 = tuple(target_lm[face_oval[(i+1)%len(face_oval)]].astype(int))
                cv2.line(image, pt1, pt2, (0, 255, 0), 1)
        
        # Draw eyes
        for eye_idx in [[33, 133, 160, 159, 158, 157, 173, 33], 
                        [263, 362, 385, 386, 387, 388, 466, 263]]:
            for i in range(len(eye_idx) - 1):
                if eye_idx[i] < max_idx and eye_idx[i+1] < max_idx:
                    pt1 = tuple(target_lm[eye_idx[i]].astype(int))
                    pt2 = tuple(target_lm[eye_idx[i+1]].astype(int))
                    cv2.line(image, pt1, pt2, (0, 255, 255), 1)
        
        # Draw mouth
        mouth = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61]
        for i in range(len(mouth) - 1):
            if mouth[i] < max_idx and mouth[i+1] < max_idx:
                pt1 = tuple(target_lm[mouth[i]].astype(int))
                pt2 = tuple(target_lm[mouth[i+1]].astype(int))
                cv2.line(image, pt1, pt2, (0, 255, 255), 1)
        
        # Add info box
        y = 30
        cv2.rectangle(image, (5, 5), (350, 90), (0, 0, 0), -1)
        
        cv2.putText(image, f"Method: {method}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, f"Method: {method}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
        
        y += 30
        cv2.putText(image, f"Landmarks: {len(target_lm)}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(image, f"Landmarks: {len(target_lm)}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 1)
        
        # Pose info at bottom
        y = image.shape[0] - 80
        cv2.rectangle(image, (5, y - 25), (350, image.shape[0] - 5), (0, 0, 0), -1)
        
        cv2.putText(image, f"Src: Y{source_pose['yaw']:.0f} P{source_pose['pitch']:.0f} R{source_pose['roll']:.0f}",
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)
        y += 20
        cv2.putText(image, f"Tgt: Y{target_pose['yaw']:.0f} P{target_pose['pitch']:.0f} R{target_pose['roll']:.0f}",
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y += 20
        angle_diff = abs(target_pose['yaw'] - source_pose['yaw'])
        cv2.putText(image, f"Diff: {angle_diff:.0f}deg", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        return image
    
    def _draw_pose_info(self, image, source_pose, target_pose, mask):
        """Draw pose info overlay"""
        h, w = image.shape[:2]
        
        # Create semi-transparent overlay
        overlay = image.copy()
        cv2.rectangle(overlay, (10, 10), (310, 190), (0, 0, 0), -1)
        image = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)
        
        y = 35
        line_height = 25
        
        # Title
        cv2.putText(image, "Pose Analysis", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y += line_height + 5
        
        # Source pose
        cv2.putText(image, "Source:", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 2)
        y += line_height
        
        cv2.putText(image, f"  Yaw:   {source_pose['yaw']:>6.1f}deg", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y += 20
        
        cv2.putText(image, f"  Pitch: {source_pose['pitch']:>6.1f}deg", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y += 20
        
        cv2.putText(image, f"  Roll:  {source_pose['roll']:>6.1f}deg", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y += line_height
        
        # Target pose
        cv2.putText(image, "Target:", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y += line_height
        
        cv2.putText(image, f"  Yaw:   {target_pose['yaw']:>6.1f}deg", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y += 20
        
        cv2.putText(image, f"  Pitch: {target_pose['pitch']:>6.1f}deg", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y += 20
        
        cv2.putText(image, f"  Roll:  {target_pose['roll']:>6.1f}deg", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Mask info
        if mask is not None:
            mask_area = np.count_nonzero(mask)
            total_area = mask.shape[0] * mask.shape[1]
            percentage = (mask_area / total_area) * 100
            
            cv2.putText(image, f"Mask: {percentage:.1f}%", (w - 150, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(image, f"Mask: {percentage:.1f}%", (w - 150, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return image
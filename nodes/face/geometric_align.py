"""
Geometric Face Alignment Node - With Triangular Warp
Uses Delaunay triangulation for accurate face warping
"""

import torch
import cv2
import numpy as np
from scipy.spatial import Delaunay
from ...utils.image_utils import tensor_to_numpy, numpy_to_tensor
from ...utils.landmark_utils import detect_face_landmarks

class GeometricFaceAlign:
    """
    Advanced geometric face alignment using triangular warping
    """
    
    CATEGORY = "Workaround/Face"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_face": ("IMAGE",),
                "target_body": ("IMAGE",),
                
                # Warp method
                "warp_method": (["triangular", "affine", "perspective", "tps"], {
                    "default": "triangular"
                }),
                
                # Landmark settings
                "num_landmarks": (["5_points", "8_points", "15_points", "68_points", "all_mediapipe"], {
                    "default": "68_points"
                }),
                
                # Mask settings
                "feather_edges": ("INT", {
                    "default": 15,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "display": "slider"
                }),
                "mask_expand": ("INT", {
                    "default": 0,
                    "min": -50,
                    "max": 100,
                    "step": 1,
                    "display": "slider"
                }),
                
                # Color matching
                "color_match": ("BOOLEAN", {"default": False}),
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
                
                # Debug
                "show_triangulation": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("result", "face_mask", "debug_preview")
    FUNCTION = "align_face"
    
    def align_face(self, source_face, target_body, warp_method, num_landmarks,
                   feather_edges, mask_expand, color_match, color_match_method,
                   color_match_strength, show_triangulation):
        
        # Convert to numpy
        source_np = tensor_to_numpy(source_face)
        target_np = tensor_to_numpy(target_body)
        h, w = target_np.shape[:2]
        
        # Detect landmarks
        print("[GeometricFaceAlign] Detecting landmarks...")
        source_landmarks = detect_face_landmarks(source_np)
        target_landmarks = detect_face_landmarks(target_np)
        
        if source_landmarks is None or target_landmarks is None:
            print("[GeometricFaceAlign] Failed to detect landmarks")
            empty = torch.zeros((1, h, w))
            return (target_body, empty, target_body)
        
        # Get landmarks based on selection
        src_pts, dst_pts = self._get_landmark_points(
            source_landmarks, target_landmarks, num_landmarks
        )
        
        print(f"[GeometricFaceAlign] Using {len(src_pts)} landmarks for {warp_method} warp")
        
        # Apply warp based on method
        if warp_method == "triangular":
            warped = self._triangular_warp(source_np, src_pts, dst_pts, (h, w))
        elif warp_method == "tps":
            warped = self._tps_warp(source_np, src_pts, dst_pts, (h, w))
        elif warp_method == "perspective":
            M = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)[0]
            if M is None:
                return (target_body, torch.zeros((1, h, w)), target_body)
            warped = cv2.warpPerspective(source_np, M, (w, h))
        else:  # affine
            M = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)[0]
            if M is None:
                return (target_body, torch.zeros((1, h, w)), target_body)
            warped = cv2.warpAffine(source_np, M, (w, h))
        
        # Create mask
        mask = self._create_face_mask(target_landmarks, (h, w))
        
        # Expand/contract mask
        if mask_expand != 0:
            kernel_size = abs(mask_expand) * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            if mask_expand > 0:
                mask = cv2.dilate(mask, kernel)
            else:
                mask = cv2.erode(mask, kernel)
        
        # Color matching
        if color_match:
            warped = self._match_color(warped, target_np, mask, 
                                      color_match_method, color_match_strength)
        
        # Feather mask
        if feather_edges > 0:
            kernel_size = feather_edges * 2 + 1
            mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
        
        # Composite
        mask_3ch = np.stack([mask] * 3, axis=-1) / 255.0
        result = target_np * (1 - mask_3ch) + warped * mask_3ch
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        # Debug preview
        if show_triangulation and warp_method == "triangular":
            debug = self._draw_triangulation(target_np.copy(), dst_pts)
        else:
            debug = self._draw_landmarks(target_np.copy(), src_pts, dst_pts, warp_method)
        
        # Convert to tensors
        result_tensor = numpy_to_tensor(result)
        mask_tensor = torch.from_numpy(mask.astype(np.float32) / 255.0).unsqueeze(0)
        debug_tensor = numpy_to_tensor(debug)
        
        return (result_tensor, mask_tensor, debug_tensor)
    
    def _triangular_warp(self, source, src_pts, dst_pts, size):
        """
        Warps source image to match target using Delaunay triangulation
        """
        h, w = size
        src_h, src_w = source.shape[:2]
        warped = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Validate input points
        valid_mask = (
            (dst_pts[:, 0] >= 0) & (dst_pts[:, 0] < w) &
            (dst_pts[:, 1] >= 0) & (dst_pts[:, 1] < h) &
            (src_pts[:, 0] >= 0) & (src_pts[:, 0] < src_w) &
            (src_pts[:, 1] >= 0) & (src_pts[:, 1] < src_h) &
            np.isfinite(src_pts).all(axis=1) &
            np.isfinite(dst_pts).all(axis=1)
        )
        
        src_pts = src_pts[valid_mask]
        dst_pts = dst_pts[valid_mask]
        
        print(f"[TriangularWarp] Using {len(dst_pts)} valid landmarks")
        
        if len(dst_pts) < 3:
            print("[TriangularWarp] Not enough valid points")
            return source
        
        # Add corner points to avoid edge artifacts
        corners_src = np.array([
            [0, 0], [src_w-1, 0], [0, src_h-1], [src_w-1, src_h-1],
            [src_w//2, 0], [0, src_h//2], [src_w-1, src_h//2], [src_w//2, src_h-1]
        ], dtype=np.float32)
        
        corners_dst = np.array([
            [0, 0], [w-1, 0], [0, h-1], [w-1, h-1],
            [w//2, 0], [0, h//2], [w-1, h//2], [w//2, h-1]
        ], dtype=np.float32)
        
        src_pts_full = np.vstack([src_pts, corners_src])
        dst_pts_full = np.vstack([dst_pts, corners_dst])
        
        # Compute Delaunay triangulation on destination points
        try:
            tri = Delaunay(dst_pts_full)
        except Exception as e:
            print(f"[TriangularWarp] Delaunay failed: {e}")
            # Fallback to affine
            M = cv2.estimateAffinePartial2D(src_pts[:5], dst_pts[:5])[0]
            if M is not None:
                return cv2.warpAffine(source, M, (w, h))
            return source
        
        # For each triangle, warp the corresponding region
        print(f"[TriangularWarp] Warping {len(tri.simplices)} triangles...")
        
        successful_warps = 0
        failed_warps = 0
        
        for triangle_indices in tri.simplices:
            # Get triangle vertices
            src_tri = src_pts_full[triangle_indices].astype(np.float32)
            dst_tri = dst_pts_full[triangle_indices].astype(np.float32)
            
            # Warp this triangle
            try:
                self._warp_triangle(source, warped, src_tri, dst_tri)
                successful_warps += 1
            except:
                failed_warps += 1
                continue
        
        print(f"[TriangularWarp] Success: {successful_warps}, Failed: {failed_warps}")
        
        # If too many failures, fallback to affine
        if successful_warps < len(tri.simplices) * 0.3:
            print("[TriangularWarp] Too many failures, using affine fallback")
            M = cv2.estimateAffinePartial2D(src_pts[:min(5, len(src_pts))], 
                                            dst_pts[:min(5, len(dst_pts))])[0]
            if M is not None:
                return cv2.warpAffine(source, M, (w, h))
        
        return warped
    
    def _warp_triangle(self, src_img, dst_img, src_tri, dst_tri):
        """
        Warps a single triangle from source to destination
        """
        # Validate triangle points
        if len(src_tri) != 3 or len(dst_tri) != 3:
            return
        
        # Check if points are valid (not NaN or Inf)
        if not (np.isfinite(src_tri).all() and np.isfinite(dst_tri).all()):
            return
        
        # Get bounding boxes
        src_rect = cv2.boundingRect(src_tri)
        dst_rect = cv2.boundingRect(dst_tri)
        
        # Validate bounding rects
        if src_rect[2] <= 0 or src_rect[3] <= 0 or dst_rect[2] <= 0 or dst_rect[3] <= 0:
            return
        
        # Check if source rect is within image bounds
        src_h, src_w = src_img.shape[:2]
        if (src_rect[0] < 0 or src_rect[1] < 0 or 
            src_rect[0] + src_rect[2] > src_w or 
            src_rect[1] + src_rect[3] > src_h):
            # Clip to valid region
            src_rect = (
                max(0, src_rect[0]),
                max(0, src_rect[1]),
                min(src_rect[2], src_w - src_rect[0]),
                min(src_rect[3], src_h - src_rect[1])
            )
            if src_rect[2] <= 0 or src_rect[3] <= 0:
                return
        
        # Check if dest rect is within image bounds
        dst_h, dst_w = dst_img.shape[:2]
        if (dst_rect[0] < 0 or dst_rect[1] < 0 or 
            dst_rect[0] + dst_rect[2] > dst_w or 
            dst_rect[1] + dst_rect[3] > dst_h):
            # Clip to valid region
            dst_rect = (
                max(0, dst_rect[0]),
                max(0, dst_rect[1]),
                min(dst_rect[2], dst_w - dst_rect[0]),
                min(dst_rect[3], dst_h - dst_rect[1])
            )
            if dst_rect[2] <= 0 or dst_rect[3] <= 0:
                return
        
        # Offset points by top-left corner of bounding box
        src_tri_cropped = np.array([
            [src_tri[0][0] - src_rect[0], src_tri[0][1] - src_rect[1]],
            [src_tri[1][0] - src_rect[0], src_tri[1][1] - src_rect[1]],
            [src_tri[2][0] - src_rect[0], src_tri[2][1] - src_rect[1]]
        ], dtype=np.float32)
        
        dst_tri_cropped = np.array([
            [dst_tri[0][0] - dst_rect[0], dst_tri[0][1] - dst_rect[1]],
            [dst_tri[1][0] - dst_rect[0], dst_tri[1][1] - dst_rect[1]],
            [dst_tri[2][0] - dst_rect[0], dst_tri[2][1] - dst_rect[1]]
        ], dtype=np.float32)
        
        # Verify we have exactly 3 valid points
        if src_tri_cropped.shape != (3, 2) or dst_tri_cropped.shape != (3, 2):
            return
        
        # Check for degenerate triangles (collinear points)
        src_area = cv2.contourArea(src_tri_cropped)
        dst_area = cv2.contourArea(dst_tri_cropped)
        if abs(src_area) < 1.0 or abs(dst_area) < 1.0:
            return
        
        try:
            # Crop source rectangle
            src_cropped = src_img[
                src_rect[1]:src_rect[1]+src_rect[3],
                src_rect[0]:src_rect[0]+src_rect[2]
            ]
            
            if src_cropped.size == 0 or src_cropped.shape[0] == 0 or src_cropped.shape[1] == 0:
                return
            
            # Calculate affine transform for this triangle
            warp_mat = cv2.getAffineTransform(src_tri_cropped, dst_tri_cropped)
            
            # Warp cropped source
            dst_cropped = cv2.warpAffine(
                src_cropped,
                warp_mat,
                (dst_rect[2], dst_rect[3]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101
            )
            
            # Create mask for triangle
            mask = np.zeros((dst_rect[3], dst_rect[2], 3), dtype=np.float32)
            cv2.fillConvexPoly(mask, dst_tri_cropped.astype(np.int32), (1.0, 1.0, 1.0), cv2.LINE_AA)
            
            # Get destination ROI
            dst_roi = dst_img[
                dst_rect[1]:dst_rect[1]+dst_rect[3],
                dst_rect[0]:dst_rect[0]+dst_rect[2]
            ]
            
            # Check dimensions match
            if dst_roi.shape[:2] == dst_cropped.shape[:2] == mask.shape[:2]:
                # Blend
                dst_roi[:] = dst_roi * (1 - mask) + dst_cropped * mask
                
        except Exception as e:
            # Silently skip problematic triangles
            pass
    
    def _tps_warp(self, source, src_pts, dst_pts, size):
        """
        Thin Plate Spline warping - smooth deformation
        Alternative to triangular warp
        """
        h, w = size
        
        try:
            # Create TPS transformer
            tps = cv2.createThinPlateSplineShapeTransformer()
            
            # Reshape points for OpenCV
            src_shape = src_pts.reshape(1, -1, 2).astype(np.float32)
            dst_shape = dst_pts.reshape(1, -1, 2).astype(np.float32)
            
            # Estimate transformation
            matches = [cv2.DMatch(i, i, 0) for i in range(len(src_pts))]
            tps.estimateTransformation(dst_shape, src_shape, matches)
            
            # Apply transformation
            warped = tps.warpImage(source)
            
            # Resize if needed
            if warped.shape[:2] != (h, w):
                warped = cv2.resize(warped, (w, h))
            
            return warped
            
        except Exception as e:
            print(f"[TPSWarp] Failed: {e}, falling back to affine")
            M = cv2.estimateAffinePartial2D(src_pts, dst_pts)[0]
            return cv2.warpAffine(source, M, (w, h))
    
    def _get_landmark_points(self, source_lm, target_lm, num_landmarks):
        """Get landmark points based on selection"""
        
        max_idx = min(len(source_lm), len(target_lm))
        
        if num_landmarks == "5_points":
            indices = [33, 263, 1, 61, 291]
        elif num_landmarks == "8_points":
            indices = [33, 263, 1, 61, 291, 10, 152, 234]
        elif num_landmarks == "15_points":
            indices = [33, 133, 263, 362, 1, 4, 61, 291, 0, 17,
                    172, 136, 150, 149, 176]
        elif num_landmarks == "68_points":
            # Comprehensive 68 points for good triangulation
            # Using MediaPipe face mesh indices
            indices = [
                # Contour (17 points)
                10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                397, 365, 379, 378, 152,
                # Eyes (12 points)
                33, 133, 160, 159, 158, 157, 263, 362, 385, 386, 387, 388,
                # Eyebrows (8 points)
                70, 63, 105, 66, 300, 293, 334, 296,
                # Nose (9 points)
                1, 2, 98, 327, 4, 5, 195, 197, 6,
                # Mouth (12 points)
                61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375,
                # Inner face (10 points)
                168, 6, 197, 195, 5, 4, 1, 19, 94, 2
            ]
        else:  # all_mediapipe
            # Use subset of all points (every 3rd point to avoid too many)
            indices = list(range(0, max_idx, 3))
        
        # Filter valid indices
        indices = [i for i in indices if i < max_idx]
        
        # Remove duplicates while preserving order
        seen = set()
        indices = [i for i in indices if not (i in seen or seen.add(i))]
        
        if len(indices) < 3:
            print(f"[LandmarkPoints] Warning: Only {len(indices)} valid points, using first 3")
            indices = list(range(min(3, max_idx)))
        
        return source_lm[indices], target_lm[indices]
    
    def _create_face_mask(self, landmarks, shape):
        """Create face mask"""
        mask = np.zeros(shape[:2], dtype=np.uint8)
        
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
        
        # Simple mean matching as fallback
        result = source.copy().astype(np.float32)
        for c in range(3):
            src_mean = np.mean(source[:, :, c][mask_bool])
            tgt_mean = np.mean(target[:, :, c][mask_bool])
            shift = (tgt_mean - src_mean) * strength
            result[:, :, c] += shift
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _draw_triangulation(self, image, points):
        """Draw Delaunay triangulation for debugging"""
        h, w = image.shape[:2]
        
        # Add corners
        corners = np.array([
            [0, 0], [w-1, 0], [0, h-1], [w-1, h-1],
            [w//2, 0], [0, h//2], [w-1, h//2], [w//2, h-1]
        ], dtype=np.float32)
        
        all_points = np.vstack([points, corners])
        
        try:
            tri = Delaunay(all_points)
            
            for simplex in tri.simplices:
                pts = all_points[simplex].astype(np.int32)
                cv2.polylines(image, [pts], True, (0, 255, 0), 1)
            
            # Draw points
            for pt in all_points:
                cv2.circle(image, tuple(pt.astype(int)), 2, (255, 0, 0), -1)
                
        except Exception as e:
            print(f"[Debug] Triangulation draw failed: {e}")
        
        return image
    
    def _draw_landmarks(self, image, src_pts, dst_pts, method):
        """Draw landmarks for debugging"""
        # Draw target landmarks
        for pt in dst_pts:
            cv2.circle(image, tuple(pt.astype(int)), 3, (0, 255, 0), -1)
        
        # Add text
        cv2.putText(image, f"Method: {method}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image, f"Points: {len(dst_pts)}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return image
"""
Planar Face Overlay Node
Aligns and overlays a 2D face onto a body using only flip/rotation/scale (planar)
"""

import torch
import cv2
import numpy as np
import json
from ...utils.image_utils import tensor_to_numpy, numpy_to_tensor
from ...utils.landmark_utils import detect_face_landmarks, get_face_bbox

class PlanarFaceOverlay:
    """
    Overlays a face image onto a body using rotation, scale and optional mirroring only.
    No non-linear warping; strictly planar transforms.
    """

    CATEGORY = "Workaround/Face"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_face": ("IMAGE",),
                "target_body": ("IMAGE",),
                "auto_flip": ("BOOLEAN", {"default": True}),
                "align_rotation": ("BOOLEAN", {"default": True}),
                "pre_normalize_camera": ("BOOLEAN", {"default": True}),
                "alignment_axis": (["nose", "eyes"], {"default": "nose"}),
                "anchor_point": (["nose_tip", "eye_center", "bbox_center"], {"default": "nose_tip"}),
                "scale_method": (["interocular", "bbox_width", "bbox_height", "nose_to_chin"], {"default": "interocular"}),
                "scale_adjust": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.01, "display": "slider"}),
                "offset_x": ("INT", {"default": 0, "min": -2000, "max": 2000, "step": 1}),
                "offset_y": ("INT", {"default": 0, "min": -2000, "max": 2000, "step": 1}),
                "feather": ("INT", {"default": 10, "min": 0, "max": 100, "step": 1, "display": "slider"}),
                "mask_expand": ("INT", {"default": 0, "min": -50, "max": 100, "step": 1, "display": "slider"}),
                "color_match": ("BOOLEAN", {"default": False}),
                "color_match_method": (["lab", "mean"], {"default": "lab"}),
                "color_match_strength": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05, "display": "slider"}),
                "full_head_mask": ("BOOLEAN", {"default": False}),
                "use_region_mask": ("BOOLEAN", {"default": False}),
                "draw_pose_map": ("BOOLEAN", {"default": False}),
                "pose_style": (["contours", "points", "tesselation", "iris"], {"default": "contours"}),
                "pose_thickness": ("INT", {"default": 2, "min": 1, "max": 8, "step": 1}),
                "pose_alpha": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05, "display": "slider"}),
                "pose_color": (["cyan", "magenta", "green", "red", "white", "yellow"], {"default": "cyan"}),
                "show_box": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "region_mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "STRING", "STRING", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("result", "face_mask", "debug_preview", "src_landmarks_in_target_json", "metrics_json", "face_only_target", "face_new_pose_source", "pose_map")
    FUNCTION = "overlay"

    def overlay(self, source_face, target_body, auto_flip, align_rotation, pre_normalize_camera,
                alignment_axis, anchor_point, scale_method, scale_adjust, offset_x, offset_y,
                feather, mask_expand, color_match, color_match_method, color_match_strength,
                full_head_mask, use_region_mask, draw_pose_map, pose_style, pose_thickness, pose_alpha, pose_color,
                show_box, region_mask=None):
        # Convert to numpy
        src_np = tensor_to_numpy(source_face)
        tgt_np = tensor_to_numpy(target_body)
        th, tw = tgt_np.shape[:2]

        # Detect landmarks
        src_lm = detect_face_landmarks(src_np)
        tgt_lm = detect_face_landmarks(tgt_np)

        if src_lm is None or tgt_lm is None:
            print("[PlanarFaceOverlay] Failed to detect landmarks")
            empty = torch.zeros((1, th, tw))
            return (target_body, empty, target_body)

        # MediaPipe indices
        L_EYE, R_EYE = 33, 263

        # Flip automático baseado em yaw (nariz vs centro dos olhos) com fallback para ordem dos olhos
        flipped = False
        sh, sw = src_np.shape[:2]
        if auto_flip:
            s_yaw = self._yaw_sign(src_lm)
            t_yaw = self._yaw_sign(tgt_lm)
            need_flip = False
            if s_yaw != 0 and t_yaw != 0 and s_yaw * t_yaw < 0:
                need_flip = True
            else:
                # Fallback: detectar imagens espelhadas (ordem dos olhos invertida)
                s_order = np.sign(src_lm[L_EYE, 0] - src_lm[R_EYE, 0])
                t_order = np.sign(tgt_lm[L_EYE, 0] - tgt_lm[R_EYE, 0])
                if s_order * t_order < 0:
                    need_flip = True
            if need_flip:
                src_np = cv2.flip(src_np, 1)
                src_lm[:, 0] = (sw - 1) - src_lm[:, 0]
                flipped = True

        # Build source mask (can use external, or auto oval/full-head)
        if use_region_mask and region_mask is not None:
            src_mask = (region_mask[0].cpu().numpy() * 255).astype(np.uint8)
            if src_mask.shape[:2] != (sh, sw):
                src_mask = cv2.resize(src_mask, (sw, sh), interpolation=cv2.INTER_NEAREST)
        else:
            if full_head_mask:
                src_mask = self._create_full_head_mask(src_lm, (sh, sw))
            else:
                src_mask = self._create_face_mask(src_lm, (sh, sw))

        # Ponto-âncora para centralizar transformações
        src_anchor = self._get_anchor_point(src_lm, anchor_point)
        tgt_anchor = self._get_anchor_point(tgt_lm, anchor_point)

        # Normalização prévia ao ângulo da câmara (planificar)
        # - "eyes": zera a inclinação dos olhos (roll horizontal)
        # - "nose": torna o eixo nariz-queixo vertical
        src_axis_angle = self._axis_angle(src_lm, alignment_axis)
        if pre_normalize_camera:
            # girar a face fonte para que o eixo escolhido fique "plano"
            M_pre = cv2.getRotationMatrix2D((float(src_anchor[0]), float(src_anchor[1])), -float(src_axis_angle), 1.0)
            src_np = cv2.warpAffine(src_np, M_pre, (sw, sh), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            src_mask = cv2.warpAffine(src_mask, M_pre, (sw, sh), flags=cv2.INTER_NEAREST,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            # atualizar landmarks e âncora
            src_lm = self._apply_affine_to_points(src_lm, M_pre)
            src_anchor = self._get_anchor_point(src_lm, anchor_point)
            # após normalizar, o ângulo da fonte é zero no eixo escolhido
            src_axis_angle = 0.0

        # Ângulo do alvo (eixo da câmara do corpo)
        tgt_axis_angle = self._axis_angle(tgt_lm, alignment_axis)

        # Ângulo final a aplicar
        angle = 0.0
        if align_rotation:
            if pre_normalize_camera:
                angle = float(tgt_axis_angle)
            else:
                angle = float(tgt_axis_angle - src_axis_angle)

        # Escala
        def _interocular(lm):
            return float(np.linalg.norm(lm[R_EYE] - lm[L_EYE]) + 1e-6)

        def _nose_to_chin(lm):
            NOSE, CHIN = 1, 152
            return float(np.linalg.norm(lm[CHIN] - lm[NOSE]) + 1e-6)

        if scale_method == "interocular":
            s_scale = _interocular(tgt_lm) / _interocular(src_lm)
        elif scale_method == "nose_to_chin":
            s_scale = _nose_to_chin(tgt_lm) / _nose_to_chin(src_lm)
        else:
            x0_s, y0_s, x1_s, y1_s = get_face_bbox(src_lm, padding=0.0)
            x0_t, y0_t, x1_t, y1_t = get_face_bbox(tgt_lm, padding=0.0)
            if scale_method == "bbox_width":
                s_scale = max(1.0, (x1_t - x0_t)) / max(1.0, (x1_s - x0_s))
            else:
                s_scale = max(1.0, (y1_t - y0_t)) / max(1.0, (y1_s - y0_s))

        s_scale *= float(scale_adjust)

        # Transformação final: rotação+escala ao redor da âncora da fonte, depois translada para âncora do alvo
        M = cv2.getRotationMatrix2D((float(src_anchor[0]), float(src_anchor[1])), angle, s_scale)
        M[0, 2] += float(tgt_anchor[0] - src_anchor[0] + offset_x)
        M[1, 2] += float(tgt_anchor[1] - src_anchor[1] + offset_y)

        # Warp da imagem e máscara para o tamanho do alvo
        warped = cv2.warpAffine(src_np, M, (tw, th), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        mask = cv2.warpAffine(src_mask, M, (tw, th), flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # Ajuste morfológico da máscara
        if mask_expand != 0:
            k = abs(int(mask_expand)) * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            if mask_expand > 0:
                mask = cv2.dilate(mask, kernel, iterations=1)
            else:
                mask = cv2.erode(mask, kernel, iterations=1)

        # Feather nas bordas
        if feather > 0:
            k = int(feather) * 2 + 1
            mask = cv2.GaussianBlur(mask, (k, k), 0)

        # Color match opcional
        if color_match:
            warped_cm = self._match_color(warped, tgt_np, mask, color_match_method, color_match_strength)
        else:
            warped_cm = warped

        # Composite
        mask_f = (mask.astype(np.float32) / 255.0)
        mask_3 = np.stack([mask_f] * 3, axis=-1)
        result = tgt_np * (1 - mask_3) + warped_cm * mask_3
        result = np.clip(result, 0, 255).astype(np.uint8)

        # Face-only (target-sized): warped face over black background using transformed mask
        face_only_target = np.clip(warped_cm.astype(np.float32) * mask_3, 0, 255).astype(np.uint8)

        # Face-only in source resolution with the new pose (no translation to target)
        M_src = cv2.getRotationMatrix2D((float(src_anchor[0]), float(src_anchor[1])), angle, s_scale)
        src_pose = cv2.warpAffine(src_np, M_src, (sw, sh), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        mask_src_pose = cv2.warpAffine(src_mask, M_src, (sw, sh), flags=cv2.INTER_NEAREST,
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        # Apply same mask adjustments in source space
        if mask_expand != 0:
            k_src = abs(int(mask_expand)) * 2 + 1
            kernel_src = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_src, k_src))
            if mask_expand > 0:
                mask_src_pose = cv2.dilate(mask_src_pose, kernel_src, iterations=1)
            else:
                mask_src_pose = cv2.erode(mask_src_pose, kernel_src, iterations=1)
        if feather > 0:
            kf_src = int(feather) * 2 + 1
            mask_src_pose = cv2.GaussianBlur(mask_src_pose, (kf_src, kf_src), 0)
        mask_src_pose_f = (mask_src_pose.astype(np.float32) / 255.0)
        mask_src_pose_3 = np.stack([mask_src_pose_f] * 3, axis=-1)
        face_new_pose_source = np.clip(src_pose.astype(np.float32) * mask_src_pose_3, 0, 255).astype(np.uint8)

        # Separate pose map (target-sized) for external use
        pose_map = np.zeros_like(tgt_np)
        self._draw_face_pose(pose_map, tgt_lm, pose_style, pose_color, int(pose_thickness))

        # Optional face pose overlay (DW-Pose-like) using target face landmarks as guide
        if draw_pose_map:
            pose_layer = result.copy()
            self._draw_face_pose(pose_layer, tgt_lm, pose_style, pose_color, int(pose_thickness))
            result = np.clip(result.astype(np.float32) * (1.0 - float(pose_alpha)) +
                             pose_layer.astype(np.float32) * float(pose_alpha), 0, 255).astype(np.uint8)

        # Debug preview
        debug = tgt_np.copy()
        if show_box:
            ys, xs = np.where(mask > 0)
            if xs.size > 0 and ys.size > 0:
                x_min, x_max = int(xs.min()), int(xs.max())
                y_min, y_max = int(ys.min()), int(ys.max())
                cv2.rectangle(debug, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        info = f"axis={alignment_axis}, anchor={anchor_point}, angle={angle:.1f}°, scale={s_scale:.2f}, flipped={flipped}"
        cv2.putText(debug, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        # Landmarks da fonte transformados para o espaço do alvo
        src_lm_in_tgt = self._apply_affine_to_points(src_lm, M)
        landmarks_json = json.dumps(src_lm_in_tgt.tolist())

        # Métricas/diagnósticos
        # - âncora da fonte transformada para o alvo
        anchor_src_in_tgt = self._apply_affine_to_points(np.array([src_anchor], dtype=np.float32), M)[0]
        # - bbox da região aplicada (a partir da máscara)
        ys, xs = np.where(mask > 0)
        bbox = None
        if xs.size > 0 and ys.size > 0:
            bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

        yaw_source = self._yaw_sign(src_lm)
        yaw_target = self._yaw_sign(tgt_lm)

        metrics = {
            "alignment_axis": alignment_axis,
            "anchor_point": anchor_point,
            "angle_deg": float(angle),
            "scale": float(s_scale),
            "flipped": bool(flipped),
            "yaw_sign": {"source": int(yaw_source), "target": int(yaw_target)},
            "anchor_src_xy": {"x": float(src_anchor[0]), "y": float(src_anchor[1])},
            "anchor_tgt_xy": {"x": float(tgt_anchor[0]), "y": float(tgt_anchor[1])},
            "anchor_src_in_target_xy": {"x": float(anchor_src_in_tgt[0]), "y": float(anchor_src_in_tgt[1])},
            "applied_color_match": bool(color_match),
            "color_match_method": str(color_match_method) if color_match else None,
            "color_match_strength": float(color_match_strength) if color_match else None,
            "applied_bbox_xyxy": bbox,
            "target_size_hw": {"h": int(th), "w": int(tw)},
        }
        metrics_json = json.dumps(metrics)

        # Tensors
        result_t = numpy_to_tensor(result)
        mask_t = torch.from_numpy(mask.astype(np.float32) / 255.0).unsqueeze(0)
        debug_t = numpy_to_tensor(debug)
        face_only_target_t = numpy_to_tensor(face_only_target)
        face_new_pose_source_t = numpy_to_tensor(face_new_pose_source)
        pose_map_t = numpy_to_tensor(pose_map)
        
        return (result_t, mask_t, debug_t, landmarks_json, metrics_json,
                face_only_target_t, face_new_pose_source_t, pose_map_t)

    def _create_face_mask(self, landmarks, shape):
        """Creates an oval face mask from landmarks"""
        h, w = shape
        mask = np.zeros((h, w), dtype=np.uint8)
        face_oval = [
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
            397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
            172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
        ]
        max_idx = len(landmarks)
        idx = [i for i in face_oval if i < max_idx]
        if len(idx) < 3:
            return mask
        pts = landmarks[idx].astype(np.int32)
        cv2.fillPoly(mask, [pts], 255)
        return mask

    def _create_full_head_mask(self, landmarks, shape):
        """Creates a full-head mask (face + hair region proxy) using the convex hull of all landmarks"""
        h, w = shape
        mask = np.zeros((h, w), dtype=np.uint8)
        try:
            pts = landmarks.astype(np.int32)
            if pts.ndim == 2 and pts.shape[1] == 2 and len(pts) >= 3:
                hull = cv2.convexHull(pts)
                if hull is not None and len(hull) >= 3:
                    cv2.fillConvexPoly(mask, hull, 255)
        except Exception:
            # Safe fallback: return empty mask if hull fails
            pass
        return mask

    def _match_color(self, source, target, mask, method, strength):
        """
        Performs color matching from 'source' to 'target' using the mask region.
        method: "lab" (default) or "mean"
        strength: 0..1
        """
        mask_bool = mask > 127
        if not np.any(mask_bool):
            return source

        if method == "lab":
            src_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
            tgt_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)

            result_lab = src_lab.copy()
            for c in range(3):
                src_pixels = src_lab[:, :, c][mask_bool]
                tgt_pixels = tgt_lab[:, :, c][mask_bool]
                if len(src_pixels) > 0 and len(tgt_pixels) > 0:
                    src_mean, src_std = float(np.mean(src_pixels)), float(np.std(src_pixels))
                    tgt_mean, tgt_std = float(np.mean(tgt_pixels)), float(np.std(tgt_pixels))
                    if src_std > 1e-6:
                        result_lab[:, :, c] = (result_lab[:, :, c] - src_mean) / max(src_std, 1e-6)
                        result_lab[:, :, c] = result_lab[:, :, c] * tgt_std + tgt_mean
                    # blend by strength
                    result_lab[:, :, c] = (
                        result_lab[:, :, c] * strength +
                        src_lab[:, :, c] * (1.0 - strength)
                    )
            result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)
            return cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)

        # Fallback: adjust only mean per channel (BGR)
        result = source.copy().astype(np.float32)
        for c in range(3):
            src_mean = float(np.mean(source[:, :, c][mask_bool]))
            tgt_mean = float(np.mean(target[:, :, c][mask_bool]))
            shift = (tgt_mean - src_mean) * strength
            result[:, :, c] += shift
        return np.clip(result, 0, 255).astype(np.uint8)

    def _yaw_sign(self, landmarks):
        """
        Estimates yaw direction using the nose (1) relative to the eyes' center.
        Returns:
          -1 if the nose is left of the eye center (looks to viewer's left),
           1 if right,
           0 if ambiguous.
        """
        L_EYE, R_EYE, NOSE = 33, 263, 1
        eye_center = (landmarks[L_EYE] + landmarks[R_EYE]) / 2.0
        dx = float(landmarks[NOSE][0] - eye_center[0])
        thr = 0.5  # tolerance
        if dx > thr:
            return 1
        elif dx < -thr:
            return -1
        else:
            return 0

    def _axis_angle(self, landmarks, axis):
        """
        Computes the angle for the chosen alignment axis:
        - eyes: angle of the line between eyes relative to the horizontal (0° = leveled eyes)
        - nose: angle of the nose→chin vector relative to the vertical (0° = vertical nose)
        Returns angle in degrees.
        """
        L_EYE, R_EYE, NOSE, CHIN = 33, 263, 1, 152
        if axis == "eyes":
            v = landmarks[R_EYE] - landmarks[L_EYE]
            return float(np.degrees(np.arctan2(v[1], v[0])))
        else:
            v = landmarks[CHIN] - landmarks[NOSE]
            # vertical-axis angle: use dx/dy
            return float(np.degrees(np.arctan2(v[0], v[1])))

    def _apply_affine_to_points(self, pts, M):
        """Applies a 2x3 affine matrix to an Nx2 array of points"""
        ones = np.ones((pts.shape[0], 1), dtype=np.float32)
        hom = np.hstack([pts.astype(np.float32), ones])  # (N,3)
        out = hom @ M.T  # (N,2)
        return out

    def _get_anchor_point(self, landmarks, name):
        """Returns the anchor point according to selection"""
        L_EYE, R_EYE = 33, 263
        if name == "nose_tip":
            return landmarks[1]
        elif name == "eye_center":
            return (landmarks[L_EYE] + landmarks[R_EYE]) / 2.0
        elif name == "bbox_center":
            x0, y0, x1, y1 = get_face_bbox(landmarks, padding=0.0)
            return np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0], dtype=np.float32)
        else:
            return (landmarks[L_EYE] + landmarks[R_EYE]) / 2.0

    def _draw_face_pose(self, image, landmarks, style, color_name, thickness):
        """
        Draws a DW-Pose-like facial overlay on the given image using provided landmarks.
        style: "contours" | "points" | "tesselation" | "iris"
        color_name: named color mapped to BGR
        thickness: line/circle thickness
        """
        color = self._color_to_bgr(color_name)
        h, w = image.shape[:2]
        n = len(landmarks)

        # Try to use MediaPipe face mesh connections if available
        mp_contours = mp_tesselation = mp_irises = None
        try:
            from mediapipe.python.solutions.face_mesh_connections import (
                FACEMESH_TESSELATION,
                FACEMESH_CONTOURS,
                FACEMESH_IRISES
            )
            mp_contours = FACEMESH_CONTOURS
            mp_tesselation = FACEMESH_TESSELATION
            mp_irises = FACEMESH_IRISES
        except Exception:
            pass

        def _pt_ok(idx):
            return 0 <= idx < n and np.isfinite(landmarks[idx]).all()

        if style == "points":
            # Draw keypoints
            for i in range(n):
                if _pt_ok(i):
                    x, y = int(landmarks[i][0]), int(landmarks[i][1])
                    cv2.circle(image, (x, y), max(1, thickness), color, -1)
            return

        if style == "contours" and mp_contours is not None:
            for (a, b) in mp_contours:
                if _pt_ok(a) and _pt_ok(b):
                    xa, ya = int(landmarks[a][0]), int(landmarks[a][1])
                    xb, yb = int(landmarks[b][0]), int(landmarks[b][1])
                    cv2.line(image, (xa, ya), (xb, yb), color, thickness, cv2.LINE_AA)
            return

        if style == "tesselation" and mp_tesselation is not None:
            for (a, b) in mp_tesselation:
                if _pt_ok(a) and _pt_ok(b):
                    xa, ya = int(landmarks[a][0]), int(landmarks[a][1])
                    xb, yb = int(landmarks[b][0]), int(landmarks[b][1])
                    cv2.line(image, (xa, ya), (xb, yb), color, max(1, thickness - 1), cv2.LINE_AA)
            return

        if style == "iris" and mp_irises is not None:
            for (a, b) in mp_irises:
                if _pt_ok(a) and _pt_ok(b):
                    xa, ya = int(landmarks[a][0]), int(landmarks[a][1])
                    xb, yb = int(landmarks[b][0]), int(landmarks[b][1])
                    cv2.line(image, (xa, ya), (xb, yb), color, thickness, cv2.LINE_AA)
            return

        # Fallback: draw simple contours if possible; otherwise draw points
        if mp_contours is not None:
            for (a, b) in mp_contours:
                if _pt_ok(a) and _pt_ok(b):
                    xa, ya = int(landmarks[a][0]), int(landmarks[a][1])
                    xb, yb = int(landmarks[b][0]), int(landmarks[b][1])
                    cv2.line(image, (xa, ya), (xb, yb), color, thickness, cv2.LINE_AA)
        else:
            for i in range(n):
                if _pt_ok(i):
                    x, y = int(landmarks[i][0]), int(landmarks[i][1])
                    cv2.circle(image, (x, y), max(1, thickness), color, -1)

    def _color_to_bgr(self, name):
        """Maps a color name to BGR tuple"""
        table = {
            "cyan": (255, 255, 0),
            "magenta": (255, 0, 255),
            "green": (0, 255, 0),
            "red": (0, 0, 255),
            "white": (255, 255, 255),
            "yellow": (0, 255, 255),
        }
        return table.get(str(name).lower(), (255, 255, 0))
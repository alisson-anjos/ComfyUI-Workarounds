"""
Skin tone detection and color match nodes for ComfyUI Workarounds
"""

import cv2
import numpy as np
import torch

from ...utils.image_utils import tensor_to_numpy, numpy_to_tensor
from ...utils.landmark_utils import detect_face_landmarks, get_face_bbox

# Reference emoji-like skin tone categories and LAB refs (CIE L*a*b*)
SKIN_TONES = {
    "NOT_DETECTED": 0,
    "LIGHT": 1,
    "MEDIUM_LIGHT": 2,
    "MEDIUM": 3,
    "MEDIUM_DARK": 4,
    "DARK": 5,
}

REFERENCE_TONES_LAB = {
    "LIGHT": (85.0, 5.0, 15.0),
    "MEDIUM_LIGHT": (75.0, 10.0, 25.0),
    "MEDIUM": (65.0, 15.0, 30.0),
    "MEDIUM_DARK": (45.0, 20.0, 35.0),
    "DARK": (30.0, 15.0, 30.0),
}

def _clip_bbox(x0, y0, x1, y1, w, h):
    x0 = max(0, min(x0, w-1))
    y0 = max(0, min(y0, h-1))
    x1 = max(0, min(x1, w-1))
    y1 = max(0, min(y1, h-1))
    if x1 <= x0: x1 = min(w-1, x0+1)
    if y1 <= y0: y1 = min(h-1, y0+1)
    return x0, y0, x1, y1

def _apply_clahe_on_l(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.merge([l, a, b])

def _lab_cv_to_cie(lab_cv):
    # lab_cv uint8 or float range: L 0..255 (~0..100), a,b 0..255 (~-128..127 + 128)
    lab_cv = lab_cv.astype(np.float32)
    L = lab_cv[..., 0] * (100.0 / 255.0)
    a = lab_cv[..., 1] - 128.0
    b = lab_cv[..., 2] - 128.0
    return np.stack([L, a, b], axis=-1)

def _cie_to_lab_cv(cie_lab):
    L = np.clip(cie_lab[..., 0] * (255.0 / 100.0), 0, 255)
    a = np.clip(cie_lab[..., 1] + 128.0, 0, 255)
    b = np.clip(cie_lab[..., 2] + 128.0, 0, 255)
    lab_cv = np.stack([L, a, b], axis=-1).astype(np.uint8)
    return lab_cv

def _cie_lab_to_hex(cie_lab):
    lab_cv = _cie_to_lab_cv(cie_lab.reshape(1,1,3))
    bgr = cv2.cvtColor(lab_cv, cv2.COLOR_Lab2BGR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).reshape(-1,3)[0]
    return "#{:02X}{:02X}{:02X}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

def _skin_mask(bgr, face_bbox=None):
    h, w = bgr.shape[:2]
    roi = bgr
    x0=y0=0
    if face_bbox is not None:
        x0, y0, x1, y1 = _clip_bbox(face_bbox[0], face_bbox[1], face_bbox[2], face_bbox[3], w, h)
        roi = bgr[y0:y1, x0:x1]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)

    # Broad HSV range for skin
    lower_hsv1 = np.array([0, 40, 20], dtype=np.uint8)
    upper_hsv1 = np.array([25, 255, 255], dtype=np.uint8)

    # YCrCb skin range
    lower_ycrcb = np.array([0, 135, 85], dtype=np.uint8)
    upper_ycrcb = np.array([255, 180, 135], dtype=np.uint8)

    mask_hsv = cv2.inRange(hsv, lower_hsv1, upper_hsv1)
    mask_ycc = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)

    mask = cv2.bitwise_and(mask_hsv, mask_ycc)

    # Morphological cleanup
    kernel3 = np.ones((3,3), np.uint8)
    kernel5 = np.ones((5,5), np.uint8)
    mask = cv2.medianBlur(mask, 3)
    mask = cv2.erode(mask, kernel3, iterations=1)
    mask = cv2.dilate(mask, kernel5, iterations=2)
    mask = cv2.GaussianBlur(mask, (3,3), 0)

    full_mask = np.zeros((h, w), dtype=np.uint8)
    if face_bbox is not None:
        full_mask[y0: y0+mask.shape[0], x0: x0+mask.shape[1]] = mask
    else:
        full_mask = mask

    return full_mask

def _exclude_shadows(cie_lab, percentile=30.0):
    L = cie_lab[..., 0]
    if L.size == 0:
        return np.zeros((0,3), dtype=np.float32)
    thr = np.percentile(L, percentile)
    sel = L > thr
    return cie_lab[sel]

def _euclidean(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return np.sqrt(np.sum((a - b) ** 2, axis=-1))

class SkinToneDetector:
    CATEGORY = "Workaround/Face"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "use_face_roi": ("BOOLEAN", {"default": True}),
                "min_skin_pixels": ("INT", {"default": 500, "min": 0, "max": 100000, "step": 50}),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("tone_label", "palette_index", "info")
    FUNCTION = "detect"

    def detect(self, image, use_face_roi, min_skin_pixels):
        bgr = tensor_to_numpy(image)  # (H,W,C) uint8 BGR
        h, w = bgr.shape[:2]

        bbox = None
        if use_face_roi:
            lms = detect_face_landmarks(bgr)
            if lms is not None:
                x0, y0, x1, y1 = get_face_bbox(lms, padding=0.2)
                x0, y0, x1, y1 = _clip_bbox(x0, y0, x1, y1, w, h)
                bbox = (x0, y0, x1, y1)

        mask = _skin_mask(bgr, face_bbox=bbox)
        num_skin = int((mask > 0).sum())
        if num_skin < min_skin_pixels:
            info = f"NOT_DETECTED | skin_pixels={num_skin}"
            return ("NOT_DETECTED", SKIN_TONES["NOT_DETECTED"], info)

        # Normalize illumination on ROI before LAB
        if bbox is not None:
            roi = bgr[bbox[1]:bbox[3], bbox[0]:bbox[2]].copy()
            roi_lab = _apply_clahe_on_l(roi)
            bgr_norm = bgr.copy()
            bgr_norm[bbox[1]:bbox[3], bbox[0]:bbox[2]] = cv2.cvtColor(roi_lab, cv2.COLOR_Lab2BGR)
        else:
            lab = _apply_clahe_on_l(bgr)
            bgr_norm = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)

        lab_cv = cv2.cvtColor(bgr_norm, cv2.COLOR_BGR2Lab)
        cie_lab = _lab_cv_to_cie(lab_cv)

        skin_lab = cie_lab[mask > 0]
        skin_lab = _exclude_shadows(skin_lab, percentile=30.0)
        if skin_lab.shape[0] == 0:
            info = f"NOT_DETECTED | no non-shadow pixels"
            return ("NOT_DETECTED", SKIN_TONES["NOT_DETECTED"], info)

        med = np.median(skin_lab, axis=0)

        # Find closest reference
        best_label = "NOT_DETECTED"
        best_dist = 1e9
        for label, ref in REFERENCE_TONES_LAB.items():
            d = _euclidean(med, ref)
            if d < best_dist:
                best_dist = d
                best_label = label

        palette_index = SKIN_TONES[best_label]
        hex_color = _cie_lab_to_hex(med)
        info = f"label={best_label} idx={palette_index} lab=({med[0]:.1f},{med[1]:.1f},{med[2]:.1f}) hex={hex_color} skin_pixels={num_skin}"
        return (best_label, int(palette_index), info)

class SkinToneColorMatch:
    CATEGORY = "Workaround/Face"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_image": ("IMAGE",),
                "target_image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "preserve_luminance": ("BOOLEAN", {"default": True}),
                "use_face_roi": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("matched_image", "info")
    FUNCTION = "match"

    def match(self, source_image, target_image, strength, preserve_luminance, use_face_roi):
        src_bgr = tensor_to_numpy(source_image)
        tgt_bgr = tensor_to_numpy(target_image)

        # Determine face bbox
        src_bbox = None
        tgt_bbox = None
        if use_face_roi:
            lms_s = detect_face_landmarks(src_bgr)
            if lms_s is not None:
                sbox = get_face_bbox(lms_s, padding=0.2)
                src_bbox = _clip_bbox(sbox[0], sbox[1], sbox[2], sbox[3], src_bgr.shape[1], src_bgr.shape[0])
            lms_t = detect_face_landmarks(tgt_bgr)
            if lms_t is not None:
                tbox = get_face_bbox(lms_t, padding=0.2)
                tgt_bbox = _clip_bbox(tbox[0], tbox[1], tbox[2], tbox[3], tgt_bgr.shape[1], tgt_bgr.shape[0])

        src_mask = _skin_mask(src_bgr, face_bbox=src_bbox)
        tgt_mask = _skin_mask(tgt_bgr, face_bbox=tgt_bbox)

        if (src_mask > 0).sum() < 200 or (tgt_mask > 0).sum() < 200:
            info = "Insufficient skin pixels for transfer"
            return (numpy_to_tensor(tgt_bgr), info)

        # LAB conversions
        src_lab_cv = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2Lab)
        tgt_lab_cv = cv2.cvtColor(tgt_bgr, cv2.COLOR_BGR2Lab)

        src_lab = _lab_cv_to_cie(src_lab_cv)
        tgt_lab = _lab_cv_to_cie(tgt_lab_cv)

        s_vals = src_lab[src_mask > 0]
        t_vals = tgt_lab[tgt_mask > 0]

        s_mean = s_vals.mean(axis=0)
        t_mean = t_vals.mean(axis=0)
        s_std = s_vals.std(axis=0) + 1e-5
        t_std = t_vals.std(axis=0) + 1e-5

        # Prepare output
        out_lab = tgt_lab.copy()

        # Adjust channels
        if preserve_luminance:
            # Adjust a,b only
            for ch in [1, 2]:
                out_ch = out_lab[..., ch]
                adj = (out_ch - t_mean[ch]) * (s_std[ch] / t_std[ch]) + s_mean[ch]
                out_lab[..., ch] = np.where(tgt_mask > 0, (1 - strength) * out_ch + strength * adj, out_ch)
        else:
            for ch in [0, 1, 2]:
                out_ch = out_lab[..., ch]
                adj = (out_ch - t_mean[ch]) * (s_std[ch] / t_std[ch]) + s_mean[ch]
                out_lab[..., ch] = np.where(tgt_mask > 0, (1 - strength) * out_ch + strength * adj, out_ch)

        # Convert back to image
        out_lab_cv = _cie_to_lab_cv(out_lab)
        out_bgr = cv2.cvtColor(out_lab_cv, cv2.COLOR_Lab2BGR)

        info = "Color transfer applied"
        return (numpy_to_tensor(out_bgr), info)

# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "SkinToneDetector": SkinToneDetector,
    "SkinToneColorMatch": SkinToneColorMatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SkinToneDetector": "Skin Tone Detector",
    "SkinToneColorMatch": "Skin Tone Color Match",
}
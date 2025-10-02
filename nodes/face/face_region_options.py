"""
Face Region Options Node
Generates combinable masks for head/face transfer: skin, forehead, eyebrows, eyes, nose, mouth, full head.
"""

import torch
import cv2
import numpy as np
from ...utils.image_utils import tensor_to_numpy, numpy_to_tensor
from ...utils.landmark_utils import detect_face_landmarks

class FaceRegionOptions:
    """
    Builds facial region masks from MediaPipe landmarks and allows combining regions.
    Useful as input mask for PlanarFaceOverlay (transfer only the desired parts).
    """

    CATEGORY = "Workaround/Face"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "preset": (["custom", "face_oval", "face_with_forehead", "full_head"], {"default": "custom"}),
                "include_face_skin": ("BOOLEAN", {"default": True}),
                "include_forehead": ("BOOLEAN", {"default": False}),
                "include_eyebrows": ("BOOLEAN", {"default": False}),
                "include_eyes": ("BOOLEAN", {"default": False}),
                "include_nose": ("BOOLEAN", {"default": False}),
                "include_mouth": ("BOOLEAN", {"default": False}),
                "expand": ("INT", {"default": 0, "min": -50, "max": 100, "step": 1, "display": "slider"}),
                "feather": ("INT", {"default": 10, "min": 0, "max": 100, "step": 1, "display": "slider"}),
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("mask", "preview")
    FUNCTION = "build_mask"

    def build_mask(self, image, preset,
                   include_face_skin, include_forehead, include_eyebrows,
                   include_eyes, include_nose, include_mouth,
                   expand, feather):
        # Convert
        img_np = tensor_to_numpy(image)
        h, w = img_np.shape[:2]

        # Landmarks
        lm = detect_face_landmarks(img_np)
        if lm is None:
            print("[FaceRegionOptions] Failed to detect face")
            empty = torch.zeros((1, h, w))
            return (empty, image)

        # Seleção por preset
        if preset == "face_oval":
            mask = self._mask_face_oval(lm, (h, w))
        elif preset == "face_with_forehead":
            mask = self._mask_face_with_forehead(lm, (h, w))
        elif preset == "full_head":
            mask = self._mask_full_head(lm, (h, w))
        else:
            # custom: combinar regiões
            mask = np.zeros((h, w), dtype=np.uint8)
            if include_face_skin:
                mask = cv2.bitwise_or(mask, self._mask_face_oval(lm, (h, w)))
            if include_forehead:
                mask = cv2.bitwise_or(mask, self._mask_face_with_forehead(lm, (h, w)))
            if include_eyebrows:
                mask = cv2.bitwise_or(mask, self._mask_eyebrows(lm, (h, w)))
            if include_eyes:
                mask = cv2.bitwise_or(mask, self._mask_eyes(lm, (h, w)))
            if include_nose:
                mask = cv2.bitwise_or(mask, self._mask_nose(lm, (h, w)))
            if include_mouth:
                mask = cv2.bitwise_or(mask, self._mask_mouth(lm, (h, w)))

        # Expand/erode
        if int(expand) != 0:
            k = abs(int(expand)) * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            if expand > 0:
                mask = cv2.dilate(mask, kernel, iterations=1)
            else:
                mask = cv2.erode(mask, kernel, iterations=1)

        # Feather
        if int(feather) > 0:
            k = int(feather) * 2 + 1
            mask = cv2.GaussianBlur(mask, (k, k), 0)

        # Preview
        preview = self._preview(img_np, mask)

        # To tensors
        mask_t = torch.from_numpy(mask.astype(np.float32) / 255.0).unsqueeze(0)
        preview_t = numpy_to_tensor(preview)

        return (mask_t, preview_t)

    # ----------------- Region helpers -----------------

    def _mask_face_oval(self, lm, shape):
        h, w = shape
        mask = np.zeros((h, w), dtype=np.uint8)
        face_oval = [
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
            397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
            172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
        ]
        idx = [i for i in face_oval if i < len(lm)]
        if len(idx) >= 3:
            pts = lm[idx].astype(np.int32)
            cv2.fillPoly(mask, [pts], 255)
        return mask

    def _mask_face_with_forehead(self, lm, shape):
        h, w = shape
        mask = np.zeros((h, w), dtype=np.uint8)
        face_contour_idx = [
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
            397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
            172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
            10, 109, 67, 103, 54, 21, 162, 127, 234, 93, 132, 58, 172,
            136, 150, 149, 176, 148, 152
        ]
        idx = [i for i in face_contour_idx if i < len(lm)]
        if len(idx) >= 3:
            pts = lm[idx].astype(np.int32)
            cv2.fillPoly(mask, [pts], 255)
        return mask

    def _mask_full_head(self, lm, shape):
        h, w = shape
        mask = np.zeros((h, w), dtype=np.uint8)
        pts = cv2.convexHull(lm.astype(np.int32))
        cv2.fillConvexPoly(mask, pts, 255)
        return mask

    def _mask_eyebrows(self, lm, shape):
        h, w = shape
        mask = np.zeros((h, w), dtype=np.uint8)
        # approximate groups (MediaPipe)
        left = [70, 63, 105, 66, 107, 55, 65, 52]   # extra points to ensure area
        right = [300, 293, 334, 296, 336, 285, 295, 282]
        left = [i for i in left if i < len(lm)]
        right = [i for i in right if i < len(lm)]
        if len(left) >= 3:
            hull = cv2.convexHull(lm[left].astype(np.int32))
            cv2.fillConvexPoly(mask, hull, 255)
        if len(right) >= 3:
            hull = cv2.convexHull(lm[right].astype(np.int32))
            cv2.fillConvexPoly(mask, hull, 255)
        return mask

    def _mask_eyes(self, lm, shape):
        h, w = shape
        mask = np.zeros((h, w), dtype=np.uint8)
        # eye rings from MediaPipe (approximation)
        left_eye = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
        right_eye = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
        le = [i for i in left_eye if i < len(lm)]
        re = [i for i in right_eye if i < len(lm)]
        if len(le) >= 3:
            hull = cv2.convexHull(lm[le].astype(np.int32))
            cv2.fillConvexPoly(mask, hull, 255)
        if len(re) >= 3:
            hull = cv2.convexHull(lm[re].astype(np.int32))
            cv2.fillConvexPoly(mask, hull, 255)
        return mask

    def _mask_nose(self, lm, shape):
        h, w = shape
        mask = np.zeros((h, w), dtype=np.uint8)
        nose_idx = [1, 2, 98, 327, 4, 5, 195, 197, 6]
        idx = [i for i in nose_idx if i < len(lm)]
        if len(idx) >= 3:
            hull = cv2.convexHull(lm[idx].astype(np.int32))
            cv2.fillConvexPoly(mask, hull, 255)
        return mask

    def _mask_mouth(self, lm, shape):
        h, w = shape
        mask = np.zeros((h, w), dtype=np.uint8)
        mouth_idx = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375]
        idx = [i for i in mouth_idx if i < len(lm)]
        if len(idx) >= 3:
            hull = cv2.convexHull(lm[idx].astype(np.int32))
            cv2.fillConvexPoly(mask, hull, 255)
        return mask

    def _preview(self, image, mask):
        overlay = image.copy()
        colored = overlay.copy()
        colored[mask > 0] = [0, 255, 0]  # green
        alpha = 0.4
        mask_3 = (np.stack([mask] * 3, axis=-1) / 255.0) * alpha
        preview = (overlay * (1 - mask_3) + colored * mask_3).astype(np.uint8)
        return preview
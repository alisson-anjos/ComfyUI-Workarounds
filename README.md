# ComfyUI-Workarounds — Planar Face Overlay and Region Options

This repository provides practical ComfyUI nodes for fast, planar face overlay without non-linear warping, plus flexible region masks to transfer the full head (face + hair + eyebrows + eyes + nose + mouth) or any combination of regions. It also includes optional color matching and JSON diagnostics to integrate with downstream logic.

- Node keys:
  - WA_PlanarFaceOverlay
  - WA_FaceRegionOptions
- Python: 3.10+
- Dependencies: see requirements.txt (OpenCV, NumPy, MediaPipe, SciPy, Torch)

## Installation

1) Ensure you have the dependencies:
- pip install -r requirements.txt

2) Copy or symlink this folder into ComfyUI/custom_nodes.

3) Restart ComfyUI.

If MediaPipe fails, ensure the correct version for your Python and environment. On some Linux setups, you may need system-level packages for video libs.

## Nodes Overview

### 1) Planar Face Overlay (WA_PlanarFaceOverlay)

Overlays a source face image onto a target body image using only planar transforms (flip/mirror, rotation, translation, uniform scale). It does not use non-linear warp (no perspective/TPS). It supports:
- Automatic orientation matching (mirror/flip) based on yaw heuristics
- Camera normalization (pre-normalize) aligning either eye-line (horizontal) or nose–chin axis (vertical)
- Anchor-based alignment (nose tip, eye center, or bbox center)
- Optional color match (LAB or per-channel mean)
- Optional external region mask (combine with Face Region Options)
- JSON outputs for landmarks (transformed to target space) and metrics/diagnostics

Inputs (required):
- source_face (IMAGE): face image to transfer
- target_body (IMAGE): destination/body image
- auto_flip (BOOLEAN): auto mirror the source to match the target orientation; default True
- align_rotation (BOOLEAN): align rotation to target; default True
- pre_normalize_camera (BOOLEAN): pre-rotate the source so the chosen axis is normalized; default True
- alignment_axis (["nose", "eyes"]): which axis to normalize/align
  - nose: normalize nose→chin to vertical (0° = perfectly vertical nose)
  - eyes: normalize eye-line to horizontal (0° = leveled eyes)
- anchor_point (["nose_tip", "eye_center", "bbox_center"]): pivot used for rotation/scale/translation
- scale_method (["interocular", "bbox_width", "bbox_height", "nose_to_chin"]): how to estimate scale
  - interocular: distance between eyes
  - bbox_width/height: face bbox size
  - nose_to_chin: nose tip to chin distance
- scale_adjust (FLOAT): fine-tune scale multiplier (default 1.0)
- offset_x (INT), offset_y (INT): manual pixel offsets
- feather (INT): Gaussian blur radius for the mask (edge feathering)
- mask_expand (INT): morphology grow/shrink the mask region
- color_match (BOOLEAN): enable color match; default False
- color_match_method (["lab", "mean"]): LAB-based or simple mean shift
- color_match_strength (FLOAT 0..1): blend strength of color match
- show_box (BOOLEAN): show rectangle of the applied region in the debug output

Inputs (optional):
- use_region_mask (BOOLEAN): use a custom region mask instead of the default face oval
- region_mask (MASK): mask input; only used when use_region_mask = True

Outputs:
- result (IMAGE): final composited image
- face_mask (MASK): mask used after transformation
- debug_preview (IMAGE): target with overlay region box and info text (angle/scale/flip)
- src_landmarks_in_target_json (STRING): JSON of the source landmarks transformed into target space; shape (N,2)
- metrics_json (STRING): JSON diagnostics:
  - alignment_axis: "nose" or "eyes"
  - anchor_point: "nose_tip" | "eye_center" | "bbox_center"
  - angle_deg: applied rotation in degrees
  - scale: applied scale factor
  - flipped: whether mirroring was applied
  - yaw_sign: {"source": -1|0|1, "target": -1|0|1} (nose vs eyes-center heuristic)
  - anchor_src_xy: original source anchor (pixels in source)
  - anchor_tgt_xy: target anchor (pixels in target)
  - anchor_src_in_target_xy: source anchor transformed into target space
  - applied_color_match: boolean
  - color_match_method: "lab"|"mean"|null
  - color_match_strength: float|null
  - applied_bbox_xyxy: [x_min, y_min, x_max, y_max] bounding box of the applied mask area in the target
  - target_size_hw: {"h": H, "w": W}

Algorithm summary:
1) Detect landmarks in source and target using MediaPipe.
2) Auto flip (mirror) the source:
   - First, compare yaw direction using nose relative to eye center (left/right facing)
   - If ambiguous, fallback to eye ordering (left vs right eye X positions)
3) Build the source mask:
   - Either the face oval (default) or an external region mask if provided
4) Pre-normalize camera:
   - If alignment_axis = "eyes": rotate source so eyes are horizontal
   - If alignment_axis = "nose": rotate source so nose→chin is vertical
5) Compute final transform:
   - Angle is the target axis angle (or difference) depending on pre_normalize_camera
   - Scale via configured method
   - Rotate+scale around the chosen anchor; then translate to target anchor and apply user offsets
6) Optionally color match (LAB or mean) inside the mask region
7) Composite onto the target using the (feathered/expanded) mask
8) Output diagnostics JSONs

Notes on color match:
- LAB method: matches mean and standard deviation in LAB channels within the masked region and blends by color_match_strength
- mean method: quick per-channel BGR mean shift inside the masked region
- The result is clipped to 0..255 and converted back to RGB for ComfyUI

Troubleshooting:
- If no face is detected, the node returns the original target and an empty mask
- If the overlay is offset, adjust offsets and verify anchor_point and alignment_axis
- If color match looks too strong, reduce color_match_strength or try "mean" method
- Auto flip might flip unexpectedly for extreme profiles or noisy landmarks; disable auto_flip if needed and flip manually upstream

### 2) Face Region Options (WA_FaceRegionOptions)

Builds region masks from landmark geometry. You can quickly generate masks to transfer:
- face skin (oval)
- forehead (broader top area)
- eyebrows
- eyes
- nose
- mouth
- full head (convex hull)

Use cases:
- Transfer full head (face + hair/eyebrows/eyes/nose/mouth)
- Transfer only specific parts (e.g., eyes + eyebrows + mouth over skin)
- Use as region_mask in PlanarFaceOverlay to control which areas blend

Inputs:
- image (IMAGE): image whose landmarks define the regions (usually the source face)
- preset (["custom","face_oval","face_with_forehead","full_head"]):
  - face_oval: standard facial oval (no forehead)
  - face_with_forehead: broader coverage including forehead
  - full_head: convex hull of all landmarks (widest coverage)
  - custom: combine individual regions below
- include_face_skin (BOOLEAN): include face oval in custom mode
- include_forehead (BOOLEAN)
- include_eyebrows (BOOLEAN)
- include_eyes (BOOLEAN)
- include_nose (BOOLEAN)
- include_mouth (BOOLEAN)
- expand (INT): morphology grow/shrink mask size
- feather (INT): feather the edges of the final composite mask

Outputs:
- mask (MASK): region mask (white = include)
- preview (IMAGE): green overlay preview for visual validation

Tips:
- For full head transfer with the overlay node, choose preset=full_head and feed this mask as region_mask with use_region_mask=True.
- For custom mixes (e.g., skin + eyebrows + hair), enable the respective booleans in custom mode.

## Recommended Pipelines

A) Full Head Transfer with Color Match
1) Face Region Options:
   - image = source_face
   - preset = full_head
   - expand = 4..12 (optional), feather = 8..20 (optional)
2) Planar Face Overlay:
   - source_face = same as above
   - target_body = body image
   - auto_flip = True
   - pre_normalize_camera = True, alignment_axis = "nose", anchor_point = "nose_tip"
   - scale_method = "nose_to_chin" (or "interocular"), scale_adjust = 0.9..1.2
   - use_region_mask = True, region_mask = mask from step 1
   - color_match = True, method = "lab", strength = 0.6..0.8
   - Check debug_preview and tweak offsets/scale/feather if needed

B) Face-Only Replacement
1) Face Region Options:
   - image = source_face
   - preset = face_oval
   - expand/feather as needed
2) Planar Face Overlay:
   - use_region_mask = True and provide the mask
   - Or leave use_region_mask = False to use the default oval
   - color_match optional

C) Eyes + Eyebrows + Mouth Only
1) Face Region Options:
   - preset = custom
   - include_eyes = True, include_eyebrows = True, include_mouth = True
2) Planar Face Overlay with use_region_mask = True

## JSON Outputs

src_landmarks_in_target_json:
- A JSON string encoding float coordinates of the source landmarks transformed to target space
- Shape: N x 2
- Usage: downstream UI overlays, debugging, or to drive other geometric operations

metrics_json:
- Contains final alignment info and anchor positions
- Can be used by automation scripts to track quality or consistency

Example metrics_json:
{
  "alignment_axis": "nose",
  "anchor_point": "nose_tip",
  "angle_deg": 12.3,
  "scale": 0.98,
  "flipped": true,
  "yaw_sign": {"source": 1, "target": 1},
  "anchor_src_xy": {"x": 123.0, "y": 210.5},
  "anchor_tgt_xy": {"x": 340.0, "y": 400.0},
  "anchor_src_in_target_xy": {"x": 338.2, "y": 398.7},
  "applied_color_match": true,
  "color_match_method": "lab",
  "color_match_strength": 0.7,
  "applied_bbox_xyxy": [300, 250, 480, 520],
  "target_size_hw": {"h": 1024, "w": 768}
}

## Limitations

- Planar only: no perspective or non-linear warps. For strong yaw/pitch mismatches, geometry may not perfectly match.
- Landmark dependency: quality relies on MediaPipe detection stability.
- Extreme profiles or occlusions can reduce orientation detection accuracy. Consider disabling auto_flip or using a manual flip upstream in those cases.

## Development Notes

- The nodes operate on BGR images (OpenCV). Conversion to/from ComfyUI tensors is handled.
- Color match uses only the masked region to avoid contaminating background statistics.
- For best results, keep source face and target body at similar resolution and crop regions appropriately.

## License

MIT

## Acknowledgements

- MediaPipe Face Mesh (Google/MediaPipe)
- OpenCV and NumPy communities

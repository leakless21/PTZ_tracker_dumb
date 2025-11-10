// Revised plan with references

# Robust Motion Detection under PTZ Motion — Revised Plan with References

This document refines the earlier plan to make motion detection robust against dynamic backgrounds (trees, clouds, parallax) when the virtual PTZ view moves. It is incremental, OpenCV‑only, and config‑driven. It maps cleanly to the current SRP layout: CV logic in `tracker.py`, PTZ in `ptz.py`, visualization in `debug_view.py`, I/O + state in `main.py`.

The plan is organized by implementation phases with rationale and citations.

---

## Summary Pipeline (High Level)

- Quick wins (no BS freeze):
  - Enforce contour solidity and extent
  - Apply an edge/border mask
  - ROI‑limited detection in reinforcement/redetection
- Global Motion Compensation (GMC) during PTZ motion:
  - Estimate inter‑frame global warp (ECC)
  - Build residual motion mask and intersect with BS mask
- Optical Flow Deviation Gate:
  - Compute dominant camera flow; keep outlier motion only
- Selection and validation filters:
  - Unify shape and motion gates across detect/reinforce/redetect

---

## Config Additions (safe defaults)

```yaml
camera_motion:
  enabled: false
  method: "ECC" # Global ECC alignment for dominant camera motion
  warp_mode: "affine" # Allowed: translation | euclidean | affine
  downscale: 0.5 # Compute alignment at reduced resolution
  max_iter: 50
  eps: 1.0e-4
  border_mask: 12 # px to ignore near image borders
  residual_thresh: 15 # 8-bit threshold for residual diff
  min_shift_px: 3.0 # trigger ECC only for significant motion
  min_rotation_deg: 0.5
  min_scale_delta: 0.01
  model: "affine" # for optional feature+RANSAC validation
  ransac_reproj_thresh: 3.0
  min_inlier_ratio: 0.4
  ptz_motion_window: 5
  ptz_motion_ratio_on: 0.6
  ptz_motion_ratio_off: 0.2

flow_gating:
  enabled: false
  method: "sparse" # sparse (KLT) | farneback
  max_features: 400
  quality: 0.01
  min_distance: 7
  residual_sigma: 2.5 # MAD multiplier for outlier rejection
  min_cluster_area: 30 # px; remove tiny outlier specks
  min_tracks_for_gating: 20
  min_inlier_ratio: 0.5

object_detection:
  # keep existing fields; solidity/extent will be enforced in code
  min_solidity: 0.3
  min_extent: 0.2

tracking:
  reinforcement:
    search_margin: 40 # px margin around bbox for detection assist
    base_search_margin: 40
    ptz_motion_multiplier: 1.5
  recovery:
    search_radius: 150 # existing key; used for redetection search
    base_search_margin: 60
    ptz_motion_multiplier: 2.0

detection:
  border_margin: 12 # px; fallback to camera_motion.border_mask if omitted
  warmup_frames: 30
  stable_min_fps: 15
  bs_learning_rate_static: 0.001
  bs_learning_rate_during_ptz: 0.0
  motion_on_threshold: 0.005 # fraction of foreground pixels
  motion_off_threshold: 0.002
  min_motion_duration_ms: 200
  min_foreground_fraction_for_gmc: 0.0005
  max_foreground_fraction_for_gmc: 0.25

selection:
  min_persist_frames: 3
  max_allowed_gap_frames: 2

debug:
  show_gmc_quality: true
  show_flow_inliers: false
  show_ptz_state: true
```

Rationale: These knobs are compact and stable; defaults are opt-in (`camera_motion.enabled` and `flow_gating.enabled` are `false`) so current runs behave identically. When enabled, they favor robustness and FPS at 720p/1080p and align with PTZ/moving-camera best practices.

---

## Phase 1 — Quick Wins (excluding BS learning freeze)

### 1. Enforce solidity and extent in contour filtering

- Where: `tracker.py` → `detect_contours(mask)`
- Tasks:
  - Add `_contour_shape_metrics(contour) -> Tuple[float, float]`:
    - `solidity = contourArea / convexHullArea`
    - `extent = contourArea / (w*h)` from the contour’s bounding rect
  - After area/aspect checks, require:
    - `solidity >= object_detection.min_solidity`
    - `extent >= object_detection.min_extent`
- Rationale: Wind‑driven foliage and cloud texture produce irregular, low‑solidity blobs; extent rejects ribbon‑like shapes. These gates preserve compact, cohesive moving objects.
- References:
  - OpenCV contour features and convex hull usage: OpenCV Docs — Contours, hierarchy, and convex hull.

### 2. Edge/border mask to suppress warp/resize artifacts

- Where: `tracker.py`
- Tasks:
  - Add `_apply_border_mask(mask: np.ndarray, margin: int) -> np.ndarray` that zeros a `margin`‑pixel border.
  - Use `margin = detection.border_margin` (fallback to `camera_motion.border_mask`).
  - Apply before every contour search: detection, reinforcement, redetection.
- Rationale: Interpolation and warping leak gradients on borders; masking avoids false motion.
- References:
  - Image warping and valid region handling: OpenCV Docs — Geometric Image Transformations.

### 3. ROI‑limited detection for reinforcement/redetection

- Where: `tracker.py`
- Tasks:
  - In `reinforce_with_detection(frame, tracked_bbox)` and `attempt_redetection(frame)`:
    - Build an expanded rectangular ROI mask around the bbox using `search_margin`.
    - Intersect with the cleaned mask before contouring.
- Rationale: Reduces background hits and accelerates search by constraining plausible locations.
- References:
  - Tracking‑by‑detection practical tips; simple ROI gating is common in single‑object tracking pipelines.

---

## Phase 2 — Global Motion Compensation (GMC) during PTZ motion

We compensate camera motion by aligning the previous grayscale frame to the current grayscale frame using ECC at reduced resolution, warping the previous frame, computing an absolute-difference residual mask in the valid overlap region, and intersecting this residual mask with the cleaned BS mask to retain only motion inconsistent with the estimated camera motion.

### Implementation details

- Fields in `ObjectTracker`:
  - `self.prev_gray: Optional[np.ndarray] = None`
  - `self.ecc_cfg = config.get("camera_motion", {})`
- Helpers:
  1. `_to_gray_downscaled(img, scale) -> gray_small`
  2. `_estimate_global_motion(prev_small, curr_small) -> (warp, cc, success)`
     - Use `cv2.findTransformECC` with `warp_mode` (`MOTION_TRANSLATION`, `MOTION_EUCLIDEAN`, or `MOTION_AFFINE`) [Evangelidis & Psarakis 2008; OpenCV 4.x].
     - Initialize warp to identity; termination criteria `(COUNT=max_iter, EPS=eps)`.
     - Only attempt ECC when a cheap sparse-flow proxy indicates significant camera motion (`min_shift_px`, `min_rotation_deg`, `min_scale_delta`).
     - On failure (non-convergence or exception), retry once with a simpler mode (e.g., translation). Do not use homography for this MVP.
     - Optionally validate the warp using a similarity/affine model estimated from KLT tracks with RANSAC; require inlier ratio ≥ `camera_motion.min_inlier_ratio`.
  3. `_warp_image(img_gray, warp, mode) -> (warped_prev, valid_mask)`
     - Use `cv2.warpAffine` for translation/euclidean/affine (2×3); derive `valid_mask` by warping an all-ones image.
  4. `_residual_mask(prev_gray, curr_gray, warp, valid_mask) -> mask`
     - Warp `prev_gray` to current, compute `absdiff`, threshold by `residual_thresh`, morphologically clean, then `AND valid_mask`, and apply `_apply_border_mask`.

### Integration points

- `reinforce_with_detection(...)` and `attempt_redetection(...)`:
  - If `camera_motion.enabled` and `self.prev_gray` exists and GMC quality checks pass:
    - Compute residual mask via ECC (or validated affine) and intersect with the cleaned BS mask.
    - Apply ROI mask and border mask before contouring.
  - If overlap area, ECC correlation `cc`, or RANSAC inlier ratio are below thresholds, skip GMC for that frame.
  - Update `self.prev_gray = curr_gray` at the end; reset in `reset_to_detection_mode()`.
- `update_detection_mode(...)`:
  - Keep baseline detection as default; allow residual blending behind a config flag for tuning.

### Rationale & performance

- ECC directly maximizes a correlation-like objective for parametric warps and is reasonably tolerant to mild photometric changes when images are normalized, but it is not invariant to strong local illumination or appearance changes. Half-scale affine typically meets 720p ≥30 FPS.
- References:
  - Evangelidis, G., Psarakis, E. “Parametric Image Alignment using Enhanced Correlation Coefficient (ECC) Maximization.” IEEE TPAMI, 2008. DOI: 10.1109/TPAMI.2008.113
  - OpenCV: `cv2.findTransformECC` (Video tracking/optical flow module docs).

---

## Phase 3 — Optical Flow Deviation Gate

We suppress motion consistent with the camera by removing inliers to the dominant flow and keep only outliers (object motion).

### Sparse KLT method (default)

- Detect corners via `cv2.goodFeaturesToTrack` (Shi‑Tomasi) on `prev_gray`.
- Track with `cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, pts_prev)` [Lucas & Kanade 1981].
- Compute flow vectors `(u, v)` from good pairs; estimate robust medians `(u_med, v_med)`.
- Require at least `flow_gating.min_tracks_for_gating` valid tracks before applying gating; otherwise, skip flow gating for this frame.
- Residual per point: `r = hypot(u - u_med, v - v_med)`; compute MAD of `r` and threshold outliers with `r > residual_sigma * MAD(r)`, using MAD as a robust scale estimate.
- If fewer than `flow_gating.min_inlier_ratio` of points support the dominant flow, treat the global flow estimate as unreliable and skip flow gating for that frame.
- Create a binary mask by painting small discs at outlier points; dilate and remove tiny components `< min_cluster_area`.

### Dense Farnebäck (optional)

- If `flow_gating.method: farneback`, compute dense flow via `cv2.calcOpticalFlowFarneback` [Farnebäck 2003] on downscaled grayscale frames, subtract the median flow vector, threshold by magnitude, and morphologically clean. Use this mode only when profiling confirms FPS is acceptable.

### Integration and rationale

- After creating the final mask (BS + GMC + ROI + border), intersect with the flow residual mask when `flow_gating.enabled` and `self.prev_gray` exists and flow quality checks pass.
- Rationale: Background that follows the camera shares the dominant flow; outliers correspond to objects moving independently.
- References:
  - Lucas, B., Kanade, T. “An Iterative Image Registration Technique with an Application to Stereo Vision.” 1981.
  - Farnebäck, G. “Two‑Frame Motion Estimation Based on Polynomial Expansion.” SCIA 2003.
  - OpenCV: `goodFeaturesToTrack`, `calcOpticalFlowPyrLK`, `calcOpticalFlowFarneback`.
  - Robust stats (MAD): Rousseeuw & Croux, 1993.

---

## Selection and Validation Filters

Unify gating across detection and tracker reinforcement.

- `detect_contours(mask)`
  - Enforce area, aspect, solidity, extent (same thresholds used everywhere).
  - Apply a motion/no-motion hysteresis state derived from the combined mask (BS + GMC + flow) using `detection.motion_on_threshold`, `motion_off_threshold`, and `min_motion_duration_ms`.
- `reinforce_with_detection(frame, tracked_bbox)`
  - Constrain search to an ROI:
    - Use `tracking.reinforcement.search_margin` / `base_search_margin` scaled by PTZ/zoom state.
  - After final mask, apply the same contour validation and temporal persistence:
    - Require support in `selection.min_persist_frames` within `max_allowed_gap_frames` before accepting large changes.
  - Keep your acceptance gates:
    - IoU ≥ `reinforcement.iou_threshold` preferred; else
    - center distance ≤ `reinforcement.max_center_distance` AND size similarity ≥ `recovery.size_similarity_threshold`.
- `attempt_redetection(frame)`
  - Same validation; use `tracking.recovery.search_radius` (and optional PTZ-scaled margin) and size similarity.
- Optional: tighten distance/IoU thresholds when PTZ-active or when GMC/flow indicate strong camera motion.

References:

- OpenCV contour analysis; IoU is a standard metric in detection/tracking literature.

---

## File‑level Implementation Tasks

- tracker.py
  - Add `_contour_shape_metrics`; enforce solidity/extent in `detect_contours`.
  - Add `_apply_border_mask` and apply before every `findContours`.
  - Implement background model warm-up and conservative adaptation:
    - Use `detection.warmup_frames`, `bs_learning_rate_static`, `bs_learning_rate_during_ptz`.
  - Add hysteresis-based motion state and temporal consistency:
    - Use `detection.motion_on_threshold`, `motion_off_threshold`, `min_motion_duration_ms`.
    - Use `selection.min_persist_frames`, `max_allowed_gap_frames` at selection stage.
  - Add ECC helpers: `_to_gray_downscaled`, `_estimate_global_motion`, `_warp_image`, `_residual_mask`.
    - Gate ECC calls with `camera_motion.min_shift_px`, `min_rotation_deg`, `min_scale_delta`.
    - Apply ECC quality checks (overlap, `cc` threshold).
    - Optionally validate with feature+RANSAC when configured.
  - Add flow helper: `_compute_flow_residual_mask` (sparse by default; optional Farnebäck mode).
    - Enforce `flow_gating.min_tracks_for_gating`, `min_inlier_ratio`.
  - Implement PTZ-active segment logic using `camera_motion.ptz_motion_window`, `ptz_motion_ratio_on`, `ptz_motion_ratio_off` to:
    - Control when GMC/flow gating are applied.
    - Adjust ROI margins via `tracking.*.ptz_motion_multiplier`.
  - Integrate GMC + flow gating in `reinforce_with_detection` and `attempt_redetection`; keep their use optional in `update_detection_mode`.
  - Add mask sanity checks using `detection.min_foreground_fraction_for_gmc` / `max_foreground_fraction_for_gmc` before expensive steps.
  - Reset `self.prev_gray` and any PTZ/flow state in `reset_to_detection_mode`.
- debug_view.py
  - Add optional tiles: `stabilized_prev`, `residual`, `flow_residual` for tuning.
  - Add lightweight overlays for:
    - ECC correlation coefficient and GMC trust state,
    - flow inlier ratios,
    - PTZ-active vs stable state,
    - guarded by `debug.show_*` flags.
- config.yaml
  - Add the new blocks and fields shown above. Keep all advanced features disabled by default to avoid changing existing behavior until explicitly enabled.

---

## Performance Guardrails

- ECC at half‑scale, affine: target 720p ≥30 FPS; 1080p ≥20 FPS.
- Sparse KLT: limit to ≤400 features; skip flow gating every other frame if needed.
- Use existing morphological kernel and avoid extra copies; compute on grayscale.

---

## Debugging & Observability

- Show masks in the debug mosaic (when enabled): BS cleaned, residual, flow residual, contours overlay.
- Log:
  - ECC correlation coefficient, warp mode, and success/fallback
  - Flow inliers/outliers, MAD threshold
  - Contour counts before/after shape gating and after GMC/flow intersections

---

## Acceptance Criteria

- Functional:
  - With PTZ motion in scenes with foliage/clouds, false positives drop by ≥60% vs baseline.
  - Reinforcement/redetection jitter and spurious jumps are significantly reduced.
- Performance:
  - Meets FPS guardrails with defaults (`downscale=0.5`, sparse flow).
- Observability:
  - Debug tiles clearly show alignment residuals and flow outliers.

---

## References (selected)

1. Evangelidis, G., Psarakis, E. (2008). Parametric Image Alignment using Enhanced Correlation Coefficient (ECC) Maximization. IEEE TPAMI. DOI: 10.1109/TPAMI.2008.113

2. OpenCV Documentation — `cv2.findTransformECC`, Geometric Image Transformations, and Video Tracking/Optical Flow modules: https://docs.opencv.org/4.x/

3. Lucas, B., Kanade, T. (1981). An Iterative Image Registration Technique with an Application to Stereo Vision.

4. Farnebäck, G. (2003). Two‑Frame Motion Estimation Based on Polynomial Expansion. SCIA 2003.

5. Zivkovic, Z. (2004, 2006). Improved Adaptive Gaussian Mixture Model for Background Subtraction; Efficient Adaptive Density Estimation per Image Pixel for Background Subtraction. IEEE ICPR 2004; Pattern Recognition Letters 2006.

6. Fischler, M.A., Bolles, R.C. (1981). Random Sample Consensus (RANSAC): A paradigm for model fitting with applications to image analysis and automated cartography. CACM.

7. OpenCV Background Subtraction (MOG2/KNN): https://docs.opencv.org/4.x/d1/dc5/tutorial_background_subtraction.html

8. Norfair multi‑object tracking library: https://github.com/tryolabs/norfair

9. Rousseeuw, P.J., Croux, C. (1993). Alternatives to the Median Absolute Deviation. Journal of the American Statistical Association.

10. Piccardi, M. (2004). Background subtraction techniques: a review. IEEE SMC.

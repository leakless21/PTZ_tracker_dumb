# Task 4: PTZ Control System

**Phase:** 2 - PTZ Control & Visualization
**Duration:** 2 days
**Priority:** Critical
**Dependencies:** Task 3 (Single-Object Tracking)

---

## Overview

Implement virtual Pan-Tilt-Zoom (PTZ) control to keep tracked object centered and appropriately sized in the frame. This simulates a physical PTZ camera by extracting and scaling regions of interest from the video.

---

## Implementation Details

### 4.1 PTZ Controller Class Structure

**Objective:** Maintain PTZ state and provide update/extraction methods

**Core State Variables:**
- `pan`: Horizontal angle in degrees (-90 to +90, 0 = center)
- `tilt`: Vertical angle in degrees (-90 to +90, 0 = center)
- `zoom`: Zoom factor (1.0 to max_zoom)
- `frame_shape`: Reference frame dimensions (height, width)

**Class Methods:**
- `__init__(frame_shape, config)`: Initialize with frame size and config
- `update(bbox)`: Calculate new PTZ values from object bbox
- `extract_roi(frame)`: Extract and scale ROI based on PTZ state
- `reset()`: Return PTZ to neutral position

### 4.2 Pan Calculation

**Objective:** Calculate horizontal adjustment to center object

**Algorithm:**
1. Calculate object center X: `cx = bbox[0] + bbox[2] / 2`
2. Calculate frame center X: `frame_cx = frame_width / 2`
3. Calculate horizontal error: `error_x = (cx - frame_cx) / frame_width`
4. Apply deadband threshold
5. Update pan: `pan += error_x * pan_sensitivity`
6. Clamp pan to limits: `pan = clip(pan, -90, +90)`

**Configuration Parameters:**
- `ptz.pan_sensitivity`: Degrees per unit error (default: 45.0)
- `ptz.deadband`: Center zone with no adjustment (default: 0.05 = 5%)
- `ptz.pan_max`: Maximum pan angle (default: 90 degrees)

**Deadband Zone:**
- If `|error_x| <= deadband`, do not adjust pan
- Prevents jitter when object near center
- Typical deadband: 5-10% of frame width

### 4.3 Tilt Calculation

**Objective:** Calculate vertical adjustment to center object

**Algorithm:**
1. Calculate object center Y: `cy = bbox[1] + bbox[3] / 2`
2. Calculate frame center Y: `frame_cy = frame_height / 2`
3. Calculate vertical error: `error_y = (cy - frame_cy) / frame_height`
4. Apply deadband threshold
5. Update tilt: `tilt += error_y * tilt_sensitivity`
6. Clamp tilt to limits: `tilt = clip(tilt, -90, +90)`

**Configuration Parameters:**
- `ptz.tilt_sensitivity`: Degrees per unit error (default: 30.0)
- `ptz.deadband`: Center zone with no adjustment (shared with pan)
- `ptz.tilt_max`: Maximum tilt angle (default: 90 degrees)

**Note:** Tilt sensitivity typically lower than pan (30 vs 45) because vertical motion less common

### 4.4 Zoom Calculation

**Objective:** Adjust zoom to maintain target object size

**Algorithm:**
1. Calculate object area: `obj_area = bbox[2] * bbox[3]`
2. Calculate frame area: `frame_area = frame_width * frame_height`
3. Calculate target area: `target_area = frame_area * target_object_size`
4. Calculate required zoom: `zoom = sqrt(target_area / obj_area)`
5. Clamp zoom to limits: `zoom = clip(zoom, zoom_min, zoom_max)`

**Configuration Parameters:**
- `ptz.target_object_size`: Fraction of frame for object (default: 0.3 = 30%)
- `ptz.zoom_min`: Minimum zoom factor (default: 1.0)
- `ptz.zoom_max`: Maximum zoom factor (default: 5.0)

**Rationale:**
- Uses square root because zoom affects both width and height
- Zoom 2.0 makes object appear 4x larger in area
- Target size 0.3 means object occupies ~30% of frame

### 4.5 ROI Extraction

**Objective:** Extract region of interest based on PTZ state

**Algorithm:**
1. Calculate ROI dimensions:
   - `roi_width = frame_width / zoom`
   - `roi_height = frame_height / zoom`
2. Calculate ROI center based on pan/tilt:
   - `center_x = frame_width/2 + (pan * frame_width / 90)`
   - `center_y = frame_height/2 + (tilt * frame_height / 90)`
3. Calculate ROI top-left corner:
   - `roi_x = center_x - roi_width / 2`
   - `roi_y = center_y - roi_height / 2`
4. Clamp ROI to frame bounds:
   - `roi_x = clamp(roi_x, 0, frame_width - roi_width)`
   - `roi_y = clamp(roi_y, 0, frame_height - roi_height)`
5. Extract ROI: `roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]`
6. Resize to original dimensions: `output = resize(roi, (frame_width, frame_height))`

**Interpolation:**
- Use INTER_LINEAR for upscaling (zoom > 1)
- Use INTER_AREA for downscaling (zoom < 1)

### 4.6 Smoothing (Optional)

**Objective:** Reduce jitter in PTZ motion

**Exponential Moving Average:**
```
alpha = smoothing_factor  # 0.2 to 0.5
pan_smoothed = alpha * pan_new + (1 - alpha) * pan_previous
tilt_smoothed = alpha * tilt_new + (1 - alpha) * tilt_previous
zoom_smoothed = alpha * zoom_new + (1 - alpha) * zoom_previous
```

**Configuration Parameters:**
- `ptz.smoothing_enabled`: Boolean (default: false for MVP)
- `ptz.smoothing_factor`: 0.0 to 1.0 (default: 0.3)

**Trade-offs:**
- Pro: Smoother, more natural motion
- Con: Introduces lag in response
- Recommendation: Add post-MVP if needed

### 4.7 Coordinate Transformation

**Objective:** Transform coordinates between original and PTZ frames

**Original to PTZ Coordinates:**
```
ptz_x = (orig_x - roi_x) * (frame_width / roi_width)
ptz_y = (orig_y - roi_y) * (frame_height / roi_height)
```

**PTZ to Original Coordinates:**
```
orig_x = (ptz_x / (frame_width / roi_width)) + roi_x
orig_y = (ptz_y / (frame_height / roi_height)) + roi_y
```

**Use Cases:**
- Updating tracker bbox after PTZ transformation
- Drawing overlays in correct coordinate space
- Debug visualization

---

## Test Scenarios

### Test 4.1: Pan Right
- **Scenario:** Object moves to right side of frame
- **Expected Result:** Pan increases (positive), ROI center shifts right
- **Validation:** Object returns toward frame center

### Test 4.2: Pan Left
- **Scenario:** Object moves to left side of frame
- **Expected Result:** Pan decreases (negative), ROI center shifts left
- **Validation:** Object returns toward frame center

### Test 4.3: Tilt Up
- **Scenario:** Object moves to top of frame
- **Expected Result:** Tilt increases (positive), ROI center shifts up
- **Validation:** Object returns toward frame center

### Test 4.4: Tilt Down
- **Scenario:** Object moves to bottom of frame
- **Expected Result:** Tilt decreases (negative), ROI center shifts down
- **Validation:** Object returns toward frame center

### Test 4.5: Zoom In
- **Scenario:** Small object tracked (area < target)
- **Expected Result:** Zoom > 1, object appears larger
- **Validation:** Object occupies ~30% of frame

### Test 4.6: Zoom Out
- **Scenario:** Large object tracked (area > target)
- **Expected Result:** Zoom approaches 1.0, object appears smaller
- **Validation:** Object occupies ~30% of frame

### Test 4.7: Deadband Zone
- **Scenario:** Object within 5% of center
- **Expected Result:** No pan/tilt adjustment
- **Validation:** PTZ values remain constant

### Test 4.8: Combined Motion
- **Scenario:** Object moves diagonally toward corner
- **Expected Result:** Both pan and tilt adjust simultaneously
- **Validation:** Object re-centered smoothly

### Test 4.9: Zoom Limits
- **Scenario:** Very small object (area << target)
- **Expected Result:** Zoom clamped to max (e.g., 5.0)
- **Validation:** Zoom doesn't exceed max_zoom

### Test 4.10: Frame Boundary Handling
- **Scenario:** PTZ attempts to pan beyond frame edge
- **Expected Result:** ROI clamped to frame bounds
- **Validation:** No out-of-bounds indexing errors

### Test 4.11: ROI Extraction Quality
- **Scenario:** Zoom to 2.0x, extract and resize
- **Expected Result:** Image scaled up cleanly
- **Validation:** No pixelation artifacts, smooth interpolation

### Test 4.12: Reset to Neutral
- **Scenario:** Call ptz.reset()
- **Expected Result:** Pan=0, Tilt=0, Zoom=1.0
- **Validation:** Full frame visible, no transformation

### Test 4.13: Coordinate Transformation
- **Scenario:** Transform bbox from original to PTZ space
- **Expected Result:** Coordinates correct in transformed frame
- **Validation:** Drawn box aligns with object in PTZ view

### Test 4.14: Performance Impact
- **Scenario:** Run with PTZ enabled vs disabled
- **Expected Result:** Minimal FPS drop (<10%)
- **Validation:** ROI extraction and resize are fast operations

---

## Caveats

### Mathematical Considerations
- **Gimbal Lock**: Extreme tilt angles (near ±90°) may cause instability
- **Zoom Non-linearity**: Area scales with zoom squared, not linearly
- **Coordinate Precision**: Integer pixel coordinates may introduce rounding errors

### Performance Trade-offs
- **High Zoom**: Upscaling (zoom > 1) may introduce blur or pixelation
- **Interpolation Cost**: INTER_CUBIC is higher quality but slower than INTER_LINEAR
- **Resolution**: Higher input resolution provides better quality at high zoom

### Control Stability
- **Oscillation**: High sensitivity may cause oscillation around center
- **Overshoot**: Without smoothing, may overshoot and need to correct
- **Deadband Too Large**: Object may not be well-centered
- **Deadband Too Small**: Jitter and constant micro-adjustments

### Edge Cases
- **Object at Frame Edge**: May not be able to center (ROI limited by frame bounds)
- **Very Large Object**: Zoom may hit minimum (1.0) and object still too large
- **Very Small Object**: Zoom may hit maximum and object still too small
- **Aspect Ratio Mismatch**: Non-square objects may not fit target size optimally

### Configuration Pitfalls
- **Pan/Tilt Sensitivity Too High**: Jerky, unstable motion
- **Pan/Tilt Sensitivity Too Low**: Sluggish response, object drifts
- **Target Size Too Large**: Excessive zoom, image quality degradation
- **Target Size Too Small**: Insufficient zoom, object appears tiny
- **Zoom Max Too High**: Extreme pixelation at high zoom levels
- **Zoom Max Too Low**: Cannot adequately frame small distant objects

### Coordinate System Confusion
- **Frame Origin**: OpenCV uses top-left origin, Y increases downward
- **Pan Direction**: Positive pan = right, negative = left
- **Tilt Direction**: Positive tilt = down, negative = up (screen coordinates)
- **Bbox Updates**: Must update tracker bbox in correct coordinate system after PTZ

---

## Success Criteria

✅ PTZ controller initialized with frame dimensions
✅ Pan adjustment keeps object horizontally centered
✅ Tilt adjustment keeps object vertically centered
✅ Zoom adjustment maintains target object size (~30% of frame)
✅ Deadband zone prevents jitter near center
✅ PTZ values clamped to configured limits
✅ ROI extracted correctly based on PTZ state
✅ ROI resized to original frame dimensions smoothly
✅ No out-of-bounds errors at frame edges
✅ Minimal performance impact (<10% FPS reduction)
✅ Reset functionality returns to neutral position
✅ Coordinate transformations mathematically correct

---

## Dependencies

**Python Libraries:**
- opencv-python >= 4.5.0
- numpy >= 1.19.0

**Previous Tasks:**
- Task 3: Provides tracked object bbox for PTZ calculation

**Configuration File:**
- config.yaml (ptz section)

---

## Integration Notes

**Inputs:**
- Video frame (from Task 1)
- Tracked object bbox (from Task 3)
- Frame shape (height, width)

**Outputs:**
- PTZ-transformed frame (ROI extracted and scaled)
- Current PTZ state (pan, tilt, zoom values)
- ROI coordinates (for debug overlay)

**Used By:**
- Main loop: Displays PTZ-transformed output
- Task 5 (Debug Mosaic): Shows PTZ ROI overlay visualization
- Task 3 (Tracking): May need coordinate transformation for bbox updates

**State Dependencies:**
- Only active in TRACKING state
- Reset when transitioning to DETECTION or LOST

---

## Estimated Effort

- PTZ Controller Class: 4-5 hours
- Pan/Tilt Calculation: 2-3 hours
- Zoom Calculation: 2-3 hours
- ROI Extraction: 3-4 hours
- Coordinate Transformation: 2-3 hours
- Testing: 3-4 hours
- Tuning & Debugging: 2-3 hours
- **Total: 1.5-2 days**

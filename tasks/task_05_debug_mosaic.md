# Task 5: Debug Mosaic Visualization

**Phase:** 2 - PTZ Control & Visualization
**Duration:** 2 days
**Priority:** Medium
**Dependencies:** Tasks 1-4 (All pipeline stages)

---

## Overview

Create a 2×4 debug mosaic grid that displays all pipeline stages simultaneously for debugging and demonstration. Users can toggle this view on/off with the 'D' key to understand how each processing step transforms the video.

---

## Implementation Details

### 5.1 Mosaic Layout Design

**Objective:** Organize 8 pipeline stages into a clear 2×4 grid

**Grid Structure:**
```
┌─────────────┬─────────────┬─────────────┬─────────────┐
│ 1. Original │ 2. FG Mask  │ 3. Cleaned  │ 4. Contours │
│    Frame    │    (Raw)    │    Mask     │  Overlay    │
├─────────────┼─────────────┼─────────────┼─────────────┤
│ 5. Norfair  │ 6. CSRT     │ 7. PTZ ROI  │ 8. Final    │
│  Detection  │  Tracking   │   Overlay   │   Output    │
└─────────────┴─────────────┴─────────────┴─────────────┘
```

**Tile Dimensions:**
- Default: 320×240 per tile
- Configurable via `display.debug_mosaic.tile_width/height`
- Total mosaic: 1280×480 at default settings

### 5.2 Pipeline Stage Definitions

**Stage 1: Original Frame**
- Source: Raw input frame from video
- Format: Color (BGR)
- Purpose: Reference for comparison

**Stage 2: Foreground Mask (Raw)**
- Source: Output of bg_subtractor.apply()
- Format: Grayscale (0-255)
- Purpose: Show raw background subtraction results
- Note: May include shadows (gray values)

**Stage 3: Cleaned Mask**
- Source: After morphological operations (opening, closing, threshold)
- Format: Binary (0 or 255)
- Purpose: Show noise reduction effect

**Stage 4: Contours Overlay**
- Source: Original frame with detected contours drawn
- Format: Color (BGR)
- Purpose: Visualize which contours passed filtering
- Draw: Green contours for valid objects, red for filtered out

**Stage 5: Norfair Detection**
- Source: Frame with cyan boxes and IDs
- Format: Color (BGR)
- Purpose: Show multi-object tracking with persistent IDs
- Active: Only in DETECTION state

**Stage 6: CSRT Tracking**
- Source: Frame with green tracking box
- Format: Color (BGR)
- Purpose: Show single-object tracking
- Active: Only in TRACKING and LOST states

**Stage 7: PTZ ROI Overlay**
- Source: Original frame with ROI rectangle drawn
- Format: Color (BGR)
- Purpose: Show which region will be extracted by PTZ
- Draw: Yellow rectangle showing ROI bounds

**Stage 8: Final Output**
- Source: PTZ-transformed output frame (or original if no PTZ)
- Format: Color (BGR)
- Purpose: Show final result with all visualizations

### 5.3 Tile Preparation

**Objective:** Normalize all stages to same size and format for grid

**Processing Steps:**
1. **Resize**: Scale to tile dimensions (e.g., 320×240)
2. **Convert**: Grayscale → BGR if needed (for uniform grid)
3. **Label**: Add text label at top (e.g., "1. Original")
4. **Border** (Optional): Add border around each tile

**Implementation:**
```
def prepare_tile(frame, label, tile_size):
    # Resize
    resized = cv2.resize(frame, tile_size, interpolation=INTER_AREA)

    # Convert grayscale to BGR if needed
    if len(resized.shape) == 2:
        resized = cv2.cvtColor(resized, COLOR_GRAY2BGR)

    # Add label
    cv2.putText(resized, label, (10, 30),
                FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    return resized
```

### 5.4 Grid Assembly

**Objective:** Combine 8 tiles into single mosaic image

**Algorithm:**
1. Prepare all 8 tiles (ensure all are same size and format)
2. Create empty mosaic canvas (4×tile_width, 2×tile_height)
3. Place tiles in grid positions:
   - Row 0: Tiles 0-3
   - Row 1: Tiles 4-7
4. Return complete mosaic

**Implementation:**
```
def create_mosaic(tiles, tile_size):
    tile_w, tile_h = tile_size
    mosaic_w = tile_w * 4
    mosaic_h = tile_h * 2
    mosaic = np.zeros((mosaic_h, mosaic_w, 3), dtype=np.uint8)

    for i, tile in enumerate(tiles):
        row = i // 4
        col = i % 4
        y = row * tile_h
        x = col * tile_w
        mosaic[y:y+tile_h, x:x+tile_w] = tile

    return mosaic
```

### 5.5 State-Dependent Display

**Objective:** Show relevant stages based on current state

**DETECTION State:**
- Original, Raw Mask, Cleaned Mask, Contours: Active
- Norfair Detection: Active (cyan boxes)
- CSRT Tracking: Blank or "Not Active"
- PTZ ROI: Blank or "Not Active"
- Final: Original frame with detections

**TRACKING State:**
- Original, Raw Mask, Cleaned Mask, Contours: Blank or last frame
- Norfair Detection: Blank or "Not Active"
- CSRT Tracking: Active (green box)
- PTZ ROI: Active (yellow rectangle)
- Final: PTZ-transformed output

**LOST State:**
- Original, Raw Mask, Cleaned Mask, Contours: Active (redetection)
- Norfair Detection: Blank
- CSRT Tracking: Last known position (red circle)
- PTZ ROI: Last known ROI
- Final: Frame with search visualization

### 5.6 Toggle Functionality

**Objective:** Allow user to show/hide mosaic

**Implementation:**
- Global or class variable: `show_debug_mosaic` (boolean)
- Keyboard handler: 'D' key toggles boolean
- Display logic:
  - If True: cv2.imshow("Debug Pipeline", mosaic)
  - If False: cv2.destroyWindow("Debug Pipeline")

**Configuration:**
- `display.show_debug_mosaic`: Default state (true/false)

### 5.7 Helper Visualizations

**Objective:** Create specialized overlays for certain stages

**Contour Overlay:**
```
def draw_contours_on_frame(frame, contours, valid_indices):
    overlay = frame.copy()
    for i, contour in enumerate(contours):
        color = (0, 255, 0) if i in valid_indices else (0, 0, 255)
        cv2.drawContours(overlay, [contour], -1, color, 2)
    return overlay
```

**PTZ ROI Overlay:**
```
def draw_ptz_roi_overlay(frame, ptz_controller):
    overlay = frame.copy()
    # Calculate ROI rectangle from PTZ state
    roi_rect = ptz_controller.get_roi_rect()
    cv2.rectangle(overlay, roi_rect, (0, 255, 255), 3)
    # Add PTZ info text
    info = f"Pan:{ptz.pan:.1f} Tilt:{ptz.tilt:.1f} Zoom:{ptz.zoom:.1f}x"
    cv2.putText(overlay, info, (10, 30), ...)
    return overlay
```

---

## Test Scenarios

### Test 5.1: Mosaic Creation
- **Scenario:** Press 'D' to enable debug mosaic
- **Expected Result:** Window appears with 2×4 grid
- **Validation:** All 8 tiles visible, labeled correctly

### Test 5.2: Tile Sizing
- **Scenario:** Change tile dimensions in config
- **Expected Result:** Mosaic scales accordingly
- **Validation:** All tiles same size, no distortion

### Test 5.3: Grayscale Conversion
- **Scenario:** Display grayscale masks in color grid
- **Expected Result:** Masks converted to BGR, white = white, black = black
- **Validation:** No format errors, uniform appearance

### Test 5.4: Label Visibility
- **Scenario:** Check labels on all tiles
- **Expected Result:** Each tile has clear label (e.g., "1. Original")
- **Validation:** Text readable, yellow color for visibility

### Test 5.5: State Transitions - DETECTION
- **Scenario:** View mosaic in DETECTION state
- **Expected Result:** Stages 1-5 active, 6-7 blank or inactive
- **Validation:** Cyan boxes visible in stage 5

### Test 5.6: State Transitions - TRACKING
- **Scenario:** Lock onto object, view mosaic
- **Expected Result:** Stages 1, 6, 7, 8 active
- **Validation:** Green box in stage 6, yellow ROI in stage 7

### Test 5.7: State Transitions - LOST
- **Scenario:** Lose track, view mosaic
- **Expected Result:** Stages 1-4 active (redetection), stage 6 shows red circle
- **Validation:** Search area visualized

### Test 5.8: Toggle On/Off
- **Scenario:** Press 'D' multiple times
- **Expected Result:** Window appears and disappears
- **Validation:** Clean toggle, no errors

### Test 5.9: Performance Impact
- **Scenario:** Compare FPS with mosaic on vs off
- **Expected Result:** <20% FPS drop with mosaic enabled
- **Validation:** Mosaic is debugging tool, some overhead acceptable

### Test 5.10: Contour Visualization
- **Scenario:** View stage 4 with multiple contours
- **Expected Result:** Valid contours green, filtered contours red
- **Validation:** Clearly see which objects passed filtering

### Test 5.11: PTZ ROI Visualization
- **Scenario:** View stage 7 during tracking
- **Expected Result:** Yellow rectangle shows ROI bounds, text shows pan/tilt/zoom
- **Validation:** ROI matches PTZ-transformed output in stage 8

### Test 5.12: Missing Stages
- **Scenario:** Stage data unavailable (e.g., CSRT inactive)
- **Expected Result:** Tile shows black screen with "Not Active" text
- **Validation:** No crashes, graceful handling

---

## Caveats

### Performance Considerations
- **Multiple Resize Operations**: Each tile resized independently, may be slow
- **Extra Memory**: Stores 8 separate frames, increases RAM usage
- **Display Overhead**: Additional cv2.imshow() window costs CPU
- **Recommendation**: Use for debugging only, disable for production

### Visual Quality
- **Downscaling Artifacts**: Small tiles may lose detail
- **Text Legibility**: Labels may be hard to read at small tile sizes
- **Color Consistency**: Grayscale-to-BGR conversion may look odd

### State Synchronization
- **Frame Lag**: Some stages may be from previous frames if not updated
- **State Mismatch**: Pipeline stages must match current state accurately
- **Missing Data**: Must handle cases where certain stages not available

### Configuration Trade-offs
- **Large Tiles**: Better visibility but mosaic window very large
- **Small Tiles**: Compact but hard to see detail
- **Recommendation**: 320×240 good balance for 1080p content

### Implementation Challenges
- **Pipeline Data Collection**: Must store intermediate results from all stages
- **Memory Management**: Must clean up old frames to prevent memory leak
- **Coordinate Systems**: PTZ ROI overlay must account for coordinate transformations

### Edge Cases
- **First Frame**: Some stages may not have data yet (e.g., masks, tracks)
- **Empty Detections**: No contours/tracks, some tiles may be blank
- **Window Management**: Multiple windows may clutter screen
- **Display Resolution**: Large mosaic may not fit on small screens

---

## Success Criteria

✅ Mosaic displays 2×4 grid with 8 labeled tiles
✅ All tiles uniform size and format (BGR)
✅ Labels clearly visible on each tile
✅ Toggle with 'D' key works reliably
✅ Original frame shown in stage 1
✅ Raw and cleaned masks shown in stages 2-3
✅ Contours overlay in stage 4 (green=valid, red=filtered)
✅ Norfair detection shown in stage 5 (DETECTION state)
✅ CSRT tracking shown in stage 6 (TRACKING state)
✅ PTZ ROI overlay shown in stage 7 (TRACKING state)
✅ Final output shown in stage 8
✅ Graceful handling of unavailable stages
✅ Performance impact acceptable (<20% FPS drop)

---

## Dependencies

**Python Libraries:**
- opencv-python >= 4.5.0
- numpy >= 1.19.0

**Previous Tasks:**
- Task 1: Provides original frame, masks
- Task 2: Provides contours, Norfair detections
- Task 3: Provides CSRT tracking visualization
- Task 4: Provides PTZ ROI overlay

**Configuration File:**
- config.yaml (display.debug_mosaic section)

---

## Integration Notes

**Inputs:**
- All pipeline stage frames (8 total)
- Current state (DETECTION, TRACKING, LOST)
- Configuration (tile size)

**Outputs:**
- Complete mosaic image for display
- Window toggle state

**Used By:**
- Main loop: Displays mosaic when enabled
- Debugging: Understanding pipeline behavior
- Demonstrations: Showing how system works

**State Dependencies:**
- Works in all states, but shows different active stages
- Some stages only populated in certain states

---

## Estimated Effort

- Tile Preparation: 3-4 hours
- Grid Assembly: 2-3 hours
- State-Dependent Display: 2-3 hours
- Helper Visualizations: 3-4 hours
- Toggle Functionality: 1-2 hours
- Testing: 2-3 hours
- Polish & Debugging: 2-3 hours
- **Total: 1.5-2 days**

# Task 3: Single-Object Tracking with CSRT/KCF

**Phase:** 1 - Foundation & Core Tracking
**Duration:** 3 days
**Priority:** Critical
**Dependencies:** Task 2 (Multi-Object Detection)

---

## Overview

Implement high-accuracy single-object tracking using OpenCV's CSRT or KCF trackers. When user selects an object by ID, system transitions from multi-object detection mode to locked tracking mode with improved accuracy and stability.

---

## Implementation Details

### 3.1 Tracker Selection

**Objective:** Choose appropriate tracker based on speed/accuracy requirements

**KCF (Kernelized Correlation Filters) - Default for MVP**
- Speed: Fast (30+ FPS)
- Accuracy: Good
- Best for: Real-time applications, general purpose
- Limitations: Less robust to occlusions and scale changes

**CSRT (Discriminative Correlation Filter with Channel and Spatial Reliability) - Alternative**
- Speed: Moderate (15-25 FPS)
- Accuracy: Excellent
- Best for: When accuracy matters more than speed
- Limitations: Higher computational cost

**Configuration Parameter:**
- `tracking.tracker`: "KCF" or "CSRT"

### 3.2 Object Selection

**Objective:** Allow user to select tracked object by pressing number key

**Key Components:**
- Listen for keyboard input (0-9 keys)
- Map key press to Norfair track ID
- Validate track exists and is stable
- Extract bounding box from selected track
- Transition to TRACKING state

**Selection Criteria:**
- Track must exist in current frame
- Track age >= min_track_age (default: 5 frames)
- Must have recent detection (not predicted)
- Bounding box must be valid

**Configuration Parameters:**
- `tracking.selection.min_track_age`: Minimum frames before track selectable

### 3.3 Tracker Initialization

**Objective:** Initialize OpenCV tracker with selected object's bounding box

**Key Components:**
- Create tracker instance (KCF or CSRT)
- Call tracker.init(frame, bbox)
- Validate initialization success
- Store tracker reference for updates
- Store initial bbox for recovery reference

**Bounding Box Format:**
- OpenCV format: (x, y, width, height) as floats or ints
- Origin at top-left corner
- Coordinates relative to frame

**Error Handling:**
- Check tracker creation success
- Validate bbox is within frame bounds
- Handle initialization failure gracefully

### 3.4 Tracking Update Loop

**Objective:** Update tracker each frame and extract new bounding box

**Key Components:**
- Call tracker.update(frame) each frame
- Receive success flag and new bbox
- Validate bbox is reasonable
- Handle tracking failures
- Maintain tracking confidence

**Update Flow:**
1. Get new frame
2. Call tracker.update(frame)
3. Check success flag
4. Validate bbox (bounds, size, aspect ratio)
5. If valid: continue tracking, draw box
6. If invalid: transition to LOST state

### 3.5 Bounding Box Validation

**Objective:** Detect tracking failures and invalid results

**Validation Checks:**
1. **Bounds Check**: Bbox fully within frame
2. **Size Check**: Area between min and max thresholds
3. **Aspect Ratio**: Not extremely elongated
4. **Position Jump**: Not moved impossibly far
5. **Success Flag**: Tracker reported success

**Configuration Parameters:**
- `tracking.min_bbox_area`: Minimum valid area (default: 100 pixels)
- `tracking.max_bbox_area_fraction`: Maximum as fraction of frame (default: 0.8)
- `tracking.max_position_jump`: Maximum centroid movement per frame (optional)

**Failure Actions:**
- Log failure reason for debugging
- Store last known good bbox
- Transition to LOST state for recovery attempt

### 3.6 State Machine Integration

**Objective:** Implement three-state system for tracking lifecycle

**States:**
1. **DETECTION**: Multi-object mode (Norfair active)
2. **TRACKING**: Single-object mode (CSRT/KCF active)
3. **LOST**: Recovery mode (attempting redetection)

**State Transitions:**
- DETECTION → TRACKING: User presses number key
- TRACKING → LOST: Tracker.update() fails or bbox invalid
- LOST → TRACKING: Recovery successful
- LOST → DETECTION: Recovery timeout
- Any state → DETECTION: User presses 'R' (reset)

**State-Specific Behavior:**
- DETECTION: Background subtraction ON, Norfair ON, CSRT OFF
- TRACKING: Background subtraction OFF, Norfair OFF, CSRT ON
- LOST: Background subtraction ON, Norfair OFF, CSRT suspended

### 3.7 Visualization

**Objective:** Provide clear visual feedback for tracking state

**TRACKING State Display:**
- Green bounding box (0, 255, 0) around locked object
- Thicker line (3 pixels) than detection boxes
- ID label in green
- Status overlay: "TRACKING - ID: X"
- Optional: Tracking confidence percentage

**User Controls:**
- R key: Release lock, return to DETECTION
- D key: Toggle debug mosaic
- Space: Pause/resume
- Q/ESC: Quit

---

## Test Scenarios

### Test 3.1: Object Selection
- **Scenario:** User presses '0' key in DETECTION mode
- **Expected Result:** System locks onto object with ID 0, transitions to TRACKING
- **Validation:** Green box appears, status shows "TRACKING - ID: 0"

### Test 3.2: Invalid ID Selection
- **Scenario:** User presses '5' but no object with ID 5 exists
- **Expected Result:** No action, remain in DETECTION mode
- **Validation:** No state change, cyan boxes still visible

### Test 3.3: Stable Tracking
- **Scenario:** Track object moving smoothly across frame
- **Expected Result:** Green box follows object accurately throughout
- **Validation:** Box centered on object, no jitter or lag

### Test 3.4: Tracking Through Partial Occlusion
- **Scenario:** Object partially hidden behind obstacle briefly
- **Expected Result:** Tracker maintains lock, box stays on object
- **Validation:** Tracking continues, no transition to LOST state

### Test 3.5: Complete Occlusion
- **Scenario:** Object fully hidden behind obstacle
- **Expected Result:** Tracker loses lock, transitions to LOST
- **Validation:** State changes to LOST, red circle appears

### Test 3.6: Object Leaves Frame
- **Scenario:** Object exits frame boundary
- **Expected Result:** Bbox validation fails, transition to LOST
- **Validation:** Last known position marked with red circle

### Test 3.7: Scale Change
- **Scenario:** Object moves closer/farther from camera
- **Expected Result:** CSRT handles scale change, maintains lock
- **Validation:** Box size adjusts with object size

### Test 3.8: Fast Motion
- **Scenario:** Object accelerates rapidly
- **Expected Result:** KCF handles fast motion, maintains lock
- **Validation:** Tracking continuous, no frame skips

### Test 3.9: Reset to Detection
- **Scenario:** User presses 'R' during tracking
- **Expected Result:** Release lock, return to DETECTION mode
- **Validation:** Cyan boxes reappear, multiple objects visible

### Test 3.10: KCF vs CSRT Performance
- **Scenario:** Track same object with both trackers
- **Expected Result:** CSRT more accurate, KCF faster
- **Validation:** Measure FPS and tracking accuracy for both

### Test 3.11: Invalid Bbox Detection
- **Scenario:** Tracker returns bbox outside frame bounds
- **Expected Result:** Validation fails, transition to LOST
- **Validation:** Last valid bbox stored for recovery

### Test 3.12: Bbox Too Small
- **Scenario:** Tracker reports bbox area < min_bbox_area
- **Expected Result:** Validation fails, transition to LOST
- **Validation:** Prevents tracking of noise or artifacts

---

## Caveats

### Tracker Limitations
- **Drift**: Long-term tracking may drift away from object center
- **No Re-detection**: Once lost, cannot reacquire automatically (handled by Task 6)
- **Appearance Changes**: Drastic appearance changes may cause tracking failure
- **Background Similarity**: Object blending into similar background may lose track

### KCF Specific Issues
- **Scale Insensitivity**: Less robust to object size changes
- **Occlusion Vulnerability**: More likely to fail during occlusions
- **Fast Rotation**: May lose track if object rotates quickly

### CSRT Specific Issues
- **Computational Cost**: 2x slower than KCF
- **Real-time Challenges**: May drop below 30 FPS on high-res video
- **Initialization Sensitivity**: Poor initialization bbox affects accuracy

### State Transition Challenges
- **Thrashing**: Rapid TRACKING ↔ LOST transitions if recovery unstable
- **Premature Loss**: Overly strict validation may declare loss too early
- **Delayed Loss Detection**: Overly lenient validation may track incorrect object

### Configuration Trade-offs
- **Strict Validation**: Fewer false positives but more false losses
- **Lenient Validation**: Continues tracking longer but may track wrong object
- **Min Bbox Area**: Too high rejects small distant objects, too low allows noise
- **Max Position Jump**: Too low causes false losses on fast motion, too high misses actual losses

### Performance Considerations
- **Resolution Impact**: Higher resolution improves accuracy but reduces FPS
- **Tracker Choice**: CSRT may require resolution reduction to maintain real-time
- **Frame Skip**: Skipping frames reduces accuracy but improves throughput

### Edge Cases
- **Initialization at Frame Edge**: Object at boundary may cause immediate loss
- **Multiple Similar Objects**: May switch to tracking wrong object after occlusion
- **Lighting Changes**: Sudden lighting changes may affect tracker confidence
- **Motion Blur**: Fast camera movement creates blur, affecting feature matching

---

## Success Criteria

✅ User can select object by pressing number key (0-9)
✅ Tracker initializes successfully with selected bbox
✅ Green bounding box drawn around tracked object
✅ Tracking continues smoothly through normal motion
✅ Handles partial occlusions gracefully
✅ Detects tracking failures reliably
✅ Transitions to LOST state when tracking fails
✅ Bbox validation prevents invalid results
✅ State machine operates correctly (DETECTION ↔ TRACKING ↔ LOST)
✅ 'R' key releases lock and returns to DETECTION
✅ Achieves target performance (KCF ≥30 FPS, CSRT ≥20 FPS)

---

## Dependencies

**Python Libraries:**
- opencv-python >= 4.5.0 (with contrib modules for CSRT/KCF)
- numpy >= 1.19.0

**Previous Tasks:**
- Task 2: Provides tracked objects with IDs for selection

**Configuration File:**
- config.yaml (tracking section)

---

## Integration Notes

**Inputs:**
- Video frame (from Task 1)
- User keyboard input (number keys 0-9)
- Selected object bbox (from Task 2 tracked objects)

**Outputs:**
- Tracked object bbox (updated each frame)
- Tracking success status
- Frame with green tracking box

**Used By:**
- Task 4 (PTZ Control): Uses tracked bbox to calculate pan/tilt/zoom
- Task 6 (Recovery): Uses last known bbox for redetection
- Task 5 (Debug Mosaic): Displays tracking stage visualization

**State Dependencies:**
- Only active in TRACKING state
- Triggers transition to LOST on failure
- Can be interrupted to return to DETECTION

---

## Estimated Effort

- Tracker Implementation: 6-8 hours
- Object Selection Logic: 2-3 hours
- Bbox Validation: 3-4 hours
- State Machine: 4-5 hours
- Visualization: 2-3 hours
- Testing: 4-5 hours
- Debugging: 3-4 hours
- **Total: 2.5-3 days**

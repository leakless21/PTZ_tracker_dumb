# Task 2: Multi-Object Detection with Norfair

**Phase:** 1 - Foundation & Core Tracking
**Duration:** 2 days
**Priority:** Critical
**Dependencies:** Task 1 (Video I/O & Background Subtraction)

---

## Overview

Implement multi-object detection using contour detection and Norfair tracking library to assign persistent IDs to detected objects. Users will be able to see all moving objects with unique identifiers for selection.

---

## Implementation Details

### 2.1 Contour Detection

**Objective:** Extract object bounding boxes from cleaned foreground mask

**Key Components:**
- Use cv2.findContours with RETR_EXTERNAL mode (only outer contours)
- Use CHAIN_APPROX_SIMPLE for memory efficiency
- Process each contour to extract bounding rectangle
- Calculate contour area for filtering

**Processing Flow:**
1. Receive cleaned binary mask from Task 1
2. Find all external contours
3. For each contour, compute area and bounding rectangle
4. Apply filtering criteria (next section)
5. Store valid contours for tracking

### 2.2 Contour Filtering

**Objective:** Remove noise and invalid detections

**Filtering Criteria:**
1. **Minimum Area**: Reject small noise blobs (default: 500 pixels)
2. **Maximum Area**: Reject objects too large (default: 50% of frame)
3. **Aspect Ratio** (Optional): Reject extremely elongated shapes
4. **Position**: Reject objects at frame edges (optional)

**Key Components:**
- Calculate contour area using cv2.contourArea()
- Extract bounding box with cv2.boundingRect()
- Compute aspect ratio (width/height)
- Compare against configured thresholds

**Configuration Parameters:**
- `object_detection.min_area`: Minimum valid object area in pixels
- `object_detection.max_area_fraction`: Maximum as fraction of frame (0.0-1.0)
- `object_detection.min_aspect_ratio`: Optional minimum width/height ratio
- `object_detection.max_aspect_ratio`: Optional maximum width/height ratio

### 2.3 Norfair Integration

**Objective:** Assign persistent IDs to detected objects across frames

**Why Norfair:**
- Automatic ID management (no manual tracking)
- Handles temporary occlusions gracefully
- Maintains ID consistency through brief disappearances
- Mature library with proven track record
- Saves ~150 lines of custom tracking code

**Key Components:**
- Convert contours to Norfair Detection objects
- Initialize Norfair Tracker with tuned parameters
- Update tracker each frame with new detections
- Extract tracked objects with persistent IDs
- Handle track initialization delay

**Norfair Detection Format:**
```
Detection(
    points=[[center_x, center_y]],  # Object centroid
    data={'bbox': (x, y, w, h)}     # Store full bounding box
)
```

**Configuration Parameters:**
- `tracking.norfair.distance_threshold`: Max pixel distance for association (default: 50)
- `tracking.norfair.hit_counter_max`: Frames to keep track without detection (default: 10)
- `tracking.norfair.initialization_delay`: Frames before confirming new track (default: 3)

### 2.4 ID Display and Management

**Objective:** Visualize detected objects with their IDs for user selection

**Key Components:**
- Draw cyan bounding boxes around all tracked objects
- Display ID number above each box
- Only show objects with active detections
- Filter by track age (avoid showing unstable new tracks)
- Limit displayed IDs to 0-9 for keyboard selection

**Visual Design:**
- Box Color: Cyan (255, 255, 0) for high visibility
- Box Thickness: 2 pixels
- Font: cv2.FONT_HERSHEY_SIMPLEX
- Font Scale: 0.6
- ID Position: 10 pixels above top-left corner of box

**Selection Readiness:**
- Track must exist for minimum age (default: 5 frames)
- Must have recent detection (within last 2 frames)
- ID must be 0-9 for keyboard mapping

---

## Test Scenarios

### Test 2.1: Single Object Detection
- **Scenario:** Video with one moving object, static background
- **Expected Result:** One contour detected, one Norfair track with ID 0
- **Validation:** Track persists across all frames, same ID maintained

### Test 2.2: Multiple Object Detection
- **Scenario:** Video with 3 moving objects
- **Expected Result:** Three separate tracks with IDs 0, 1, 2
- **Validation:** Each object gets unique ID, IDs don't swap between objects

### Test 2.3: Noise Filtering
- **Scenario:** Noisy mask with small artifacts
- **Expected Result:** Small noise blobs ignored, only real objects tracked
- **Validation:** Number of tracks matches number of visible objects

### Test 2.4: ID Persistence Through Occlusion
- **Scenario:** Object temporarily hidden behind obstacle
- **Expected Result:** ID maintained when object reappears
- **Validation:** Same ID before and after occlusion (within hit_counter_max frames)

### Test 2.5: New Object Entry
- **Scenario:** New object enters frame during video
- **Expected Result:** New track created with next available ID
- **Validation:** New ID assigned after initialization_delay frames

### Test 2.6: Object Exit
- **Scenario:** Object leaves frame
- **Expected Result:** Track removed after hit_counter_max frames
- **Validation:** ID freed and available for reuse

### Test 2.7: Track Age Filtering
- **Scenario:** New detection appears briefly then disappears
- **Expected Result:** Not displayed if age < min_track_age
- **Validation:** Only stable tracks shown to user

### Test 2.8: Maximum Objects
- **Scenario:** More than 10 objects in scene
- **Expected Result:** All tracked but only IDs 0-9 selectable
- **Validation:** System handles gracefully, displays first 10 IDs

### Test 2.9: Fast Motion
- **Scenario:** Object moves quickly across frame
- **Expected Result:** ID maintained despite large frame-to-frame displacement
- **Validation:** Track continuous, no ID switches

### Test 2.10: Similar Objects
- **Scenario:** Two identical objects in close proximity
- **Expected Result:** Both tracked with separate IDs
- **Validation:** IDs don't swap when objects pass near each other

---

## Caveats

### Norfair Limitations
- **Centroid-Based Tracking**: Uses object center only, not appearance features
- **ID Reuse**: IDs can be reassigned after track deletion (not globally unique)
- **Initialization Delay**: Brief delay before new tracks confirmed (prevents false positives)
- **No Re-identification**: Once track lost, cannot recognize same object later

### Contour Detection Issues
- **Merged Objects**: Close objects may create single contour, resulting in one track for two objects
- **Split Objects**: Partial occlusion may split single object into multiple contours/tracks
- **Shape Changes**: Significant shape changes may cause track loss and new ID assignment
- **Lighting Artifacts**: Reflections or shadows may create false contours

### Configuration Challenges
- **Distance Threshold Too High**: May associate wrong detections (ID swaps)
- **Distance Threshold Too Low**: May create duplicate tracks for same object
- **Hit Counter Too High**: Dead tracks linger, wasting IDs
- **Hit Counter Too Low**: Brief occlusions cause track loss
- **Init Delay Too Long**: Slow response to new objects
- **Init Delay Too Short**: Noise creates spurious tracks

### Performance Considerations
- **Many Small Objects**: Hundreds of small detections may slow down processing
- **Frequent Track Creation/Deletion**: High object turnover increases computational load
- **Memory Usage**: Each track stores history, many long-lived tracks increase memory

### Edge Cases
- **Static Objects**: Objects that stop moving will disappear from detections and lose track
- **Identical Trajectories**: Objects moving in exact same path may cause ID confusion
- **Camera Shake**: Small camera motion may cause background to be detected as moving objects
- **First Frame**: No prior frame for comparison, may miss objects present from start

---

## Success Criteria

✅ Contours successfully extracted from foreground mask
✅ Noise filtered out, only valid objects detected
✅ Norfair tracker initialized and updating correctly
✅ Persistent IDs assigned to detected objects
✅ IDs maintained across frames (no random swapping)
✅ IDs persist through brief occlusions (up to hit_counter_max frames)
✅ New objects get new IDs after initialization_delay
✅ Cyan bounding boxes drawn around all tracked objects
✅ ID numbers displayed clearly above each box
✅ Only stable tracks (age >= min_track_age) shown
✅ System handles 1-10 objects gracefully

---

## Dependencies

**Python Libraries:**
- opencv-python >= 4.5.0
- numpy >= 1.19.0
- norfair >= 2.0.0

**Previous Tasks:**
- Task 1: Provides cleaned foreground mask

**Configuration File:**
- config.yaml (object_detection and tracking.norfair sections)

---

## Integration Notes

**Inputs:**
- Cleaned binary mask from Task 1 (background subtraction)
- Original frame for drawing visualizations

**Outputs:**
- List of tracked objects with IDs and bounding boxes
- Frame with cyan boxes and ID labels (detection mode visualization)

**Used By:**
- Task 3 (Single-Object Tracking): User selects ID to lock onto
- Task 5 (Debug Mosaic): Displays detection stage visualization
- main.py: Displays in DETECTION state

**State Dependencies:**
- Only active in DETECTION state
- Inactive in TRACKING and LOST states

---

## Estimated Effort

- Norfair Integration: 4-6 hours
- Contour Detection & Filtering: 3-4 hours
- Visualization & ID Display: 2-3 hours
- Testing: 3-4 hours
- Debugging: 2-3 hours
- **Total: 1.5-2 days**

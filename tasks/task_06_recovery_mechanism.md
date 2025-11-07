# Task 6: Recovery Mechanism

**Phase:** 3 - Recovery & Configuration
**Duration:** 2 days
**Priority:** High
**Dependencies:** Tasks 1-3 (Background Subtraction, Detection, Tracking)

---

## Overview

Implement automatic recovery when tracking is lost. The LOST state uses background subtraction to search for the lost object near its last known position, matching by size and location. If reacquired, tracking resumes; if timeout expires, system returns to multi-object detection.

---

## Implementation Details

### 6.1 Lost State Trigger

**Objective:** Detect when tracking has failed and transition to recovery

**Trigger Conditions:**
- CSRT tracker.update() returns False
- Bounding box validation fails (out of bounds, invalid size)
- Tracking confidence below threshold (if available)

**Transition Actions:**
1. Store last known valid bbox
2. Store last known object center
3. Record loss timestamp
4. Transition to LOST state
5. Initialize recovery parameters

**Stored Recovery Data:**
```
last_bbox = (x, y, w, h)
last_center = (cx, cy)
loss_timestamp = time.time()
search_radius = config['recovery']['search_radius']
timeout = config['recovery']['timeout']
```

### 6.2 Search Area Visualization

**Objective:** Show user where system is searching for lost object

**Visual Elements:**
- Red circle centered on last known position
- Radius = search_radius (default: 150 pixels)
- Semi-transparent fill (optional)
- Text: "SEARCHING..." with timer

**Implementation:**
```
def draw_search_area(frame, center, radius, elapsed_time):
    # Draw red circle
    cv2.circle(frame, center, radius, (0, 0, 255), 3)

    # Optional: semi-transparent fill
    overlay = frame.copy()
    cv2.circle(overlay, center, radius, (0, 0, 255), -1)
    cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)

    # Draw status text
    time_remaining = timeout - elapsed_time
    text = f"SEARCHING... {time_remaining:.1f}s"
    cv2.putText(frame, text, (center[0]-50, center[1]),
                FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
```

### 6.3 Redetection Algorithm

**Objective:** Find lost object using background subtraction and matching

**Algorithm:**
1. Apply background subtraction to current frame
2. Clean mask with morphological operations
3. Find contours in cleaned mask
4. Filter contours by:
   - Distance from last known position
   - Size similarity to last known size
5. Select best matching contour
6. Return bounding box if match found, else None

**Distance Matching:**
```
def is_within_search_radius(contour_center, last_center, search_radius):
    distance = sqrt((cx - last_cx)² + (cy - last_cy)²)
    return distance <= search_radius
```

**Size Matching:**
```
def is_size_similar(contour_area, last_area, threshold):
    size_ratio = min(contour_area, last_area) / max(contour_area, last_area)
    return size_ratio >= threshold  # default: 0.5
```

**Best Match Selection:**
- If multiple candidates, choose closest to last position
- Prefer size match over distance if tie-breaking needed

### 6.4 Recovery Logic

**Objective:** Attempt redetection each frame until success or timeout

**Per-Frame Process:**
1. Calculate elapsed time since loss
2. Check if timeout exceeded:
   - If yes: Return to DETECTION state
   - If no: Continue
3. Run redetection algorithm
4. If match found:
   - Reinitialize CSRT tracker with new bbox
   - Transition to TRACKING state
   - Reset PTZ if needed
5. If no match:
   - Continue searching next frame

**Implementation:**
```
def update_lost_state(frame, last_bbox, loss_time, config):
    elapsed = time.time() - loss_time

    # Check timeout
    if elapsed > config['recovery']['timeout']:
        return None, 'timeout'

    # Attempt redetection
    new_bbox = attempt_redetection(frame, last_bbox, config)

    if new_bbox is not None:
        # Success!
        return new_bbox, 'recovered'
    else:
        # Keep searching
        return None, 'searching'
```

### 6.5 Tracker Reinitialization

**Objective:** Restart CSRT tracking with recovered object

**Process:**
1. Validate recovered bbox
2. Create new tracker instance (KCF or CSRT)
3. Initialize with current frame and bbox
4. Verify initialization success
5. Transition to TRACKING state
6. Resume PTZ control

**Implementation:**
```
def reinitialize_tracker(frame, bbox, config):
    # Create new tracker
    if config['tracking']['tracker'] == 'KCF':
        tracker = cv2.TrackerKCF_create()
    else:
        tracker = cv2.TrackerCSRT_create()

    # Initialize
    success = tracker.init(frame, bbox)

    if success:
        return tracker
    else:
        # Initialization failed, return to DETECTION
        return None
```

### 6.6 Timeout Handling

**Objective:** Gracefully abandon recovery after timeout

**Configuration:**
- `tracking.recovery.timeout`: Seconds to search (default: 3.0)

**Actions on Timeout:**
1. Log recovery failure (optional)
2. Clear stored last_bbox and tracker
3. Reset PTZ to neutral position
4. Transition to DETECTION state
5. Show "RECOVERY FAILED" message briefly

**Rationale:**
- 3 seconds = ~90 frames at 30 FPS
- Long enough for brief occlusions
- Short enough to not frustrate user
- Configurable for different use cases

### 6.7 User Override

**Objective:** Allow user to manually abort recovery

**Implementation:**
- 'R' key: Immediately return to DETECTION
- Works in LOST state same as TRACKING state
- Clears recovery data
- Resets PTZ

---

## Test Scenarios

### Test 6.1: Tracking Loss Detection
- **Scenario:** Object leaves frame during tracking
- **Expected Result:** Transition to LOST state, red circle appears
- **Validation:** State changes, timer starts, search area displayed

### Test 6.2: Successful Redetection
- **Scenario:** Object briefly occluded then reappears
- **Expected Result:** Object redetected, tracking resumes with green box
- **Validation:** State changes to TRACKING, same object tracked

### Test 6.3: Search Radius Constraint
- **Scenario:** Object reappears far from last position (>150 pixels)
- **Expected Result:** Not redetected (outside search radius)
- **Validation:** Search continues, timeout eventual

### Test 6.4: Size Similarity Constraint
- **Scenario:** Different object appears near last position
- **Expected Result:** Not redetected (size mismatch)
- **Validation:** Wrong object not locked onto

### Test 6.5: Multiple Candidates
- **Scenario:** Multiple objects in search area
- **Expected Result:** Closest match by distance selected
- **Validation:** Correct object reacquired

### Test 6.6: Recovery Timeout
- **Scenario:** Object never reappears, wait 3 seconds
- **Expected Result:** Return to DETECTION state, cyan boxes appear
- **Validation:** State changes after timeout, multi-object mode restored

### Test 6.7: Rapid Loss/Recovery
- **Scenario:** Object repeatedly occluded (flickering)
- **Expected Result:** System recovers each time
- **Validation:** No state thrashing, smooth recovery

### Test 6.8: Manual Abort
- **Scenario:** User presses 'R' during recovery
- **Expected Result:** Immediately return to DETECTION
- **Validation:** Recovery aborted, no delay

### Test 6.9: Recovery Visualization
- **Scenario:** View recovery in debug mosaic
- **Expected Result:** Background subtraction active, contours visible
- **Validation:** Stage 6 shows red circle, stages 1-4 show redetection process

### Test 6.10: Reinitialization Failure
- **Scenario:** Recovered bbox invalid for CSRT init
- **Expected Result:** Return to DETECTION instead of crashing
- **Validation:** Graceful handling, clear error message

### Test 6.11: Partial Reappearance
- **Scenario:** Only part of object visible after occlusion
- **Expected Result:** Redetected if size ratio ≥ 0.5
- **Validation:** Tracking resumes even with partial visibility

### Test 6.12: Background Motion
- **Scenario:** Background moves (camera shake) during recovery
- **Expected Result:** False detections filtered by size/distance
- **Validation:** No false recovery, correct timeout behavior

---

## Caveats

### Recovery Limitations
- **Appearance Changes**: Cannot redetect if object appearance drastically changed
- **Crowded Scenes**: May redetect wrong object if multiple similar objects
- **Fast Motion**: Object may move outside search radius before recovery
- **Long Occlusions**: 3-second timeout may be too short for long occlusions

### Configuration Trade-offs
- **Large Search Radius**: More permissive but higher false positive rate
- **Small Search Radius**: More restrictive but may miss fast-moving objects
- **High Size Threshold**: Strict matching but fails if object size changes
- **Low Size Threshold**: Permissive but may match wrong objects
- **Long Timeout**: More recovery chances but frustrating wait
- **Short Timeout**: Quick fallback but may give up too early

### Background Subtraction Dependencies
- **Static Background Required**: Recovery assumes stationary camera
- **Lighting Consistency**: Sudden lighting changes may break redetection
- **Model Staleness**: If object was stationary before loss, background model may exclude it

### Performance Considerations
- **Recovery Overhead**: Background subtraction + contour detection every frame in LOST state
- **State Thrashing**: Rapid TRACKING ↔ LOST transitions waste CPU
- **Timeout Accumulation**: Multiple recovery attempts add latency

### Edge Cases
- **Loss at Frame Edge**: Last position near boundary, search area clipped
- **Immediate Re-loss**: Object recovered but immediately lost again
- **Multiple Simultaneous Losses**: Design assumes single tracked object
- **Scale Changes**: Significant zoom change between loss and recovery

### User Experience
- **Unclear Failure**: User may not understand why recovery failed
- **Search Area Obscures View**: Red circle may block visibility
- **Timer Anxiety**: Countdown creates pressure
- **False Hope**: Recovery attempt may give impression system will definitely recover

---

## Success Criteria

✅ System detects tracking loss reliably
✅ Transitions to LOST state with stored last bbox
✅ Red circle shows search area clearly
✅ Timer displays time remaining
✅ Background subtraction reactivated in LOST state
✅ Redetection algorithm finds matching objects
✅ Distance constraint enforced (within search_radius)
✅ Size similarity constraint enforced (ratio ≥ threshold)
✅ Best match selected when multiple candidates
✅ CSRT tracker reinitialized on successful recovery
✅ Tracking resumes smoothly after recovery
✅ Timeout causes return to DETECTION after 3 seconds
✅ 'R' key aborts recovery immediately
✅ No crashes or errors during recovery process

---

## Dependencies

**Python Libraries:**
- opencv-python >= 4.5.0
- numpy >= 1.19.0

**Previous Tasks:**
- Task 1: Background subtraction for redetection
- Task 2: Contour detection and filtering
- Task 3: Tracker initialization and validation

**Configuration File:**
- config.yaml (tracking.recovery section)

---

## Integration Notes

**Inputs:**
- Last known bbox and center (from Task 3)
- Current frame (from Task 1)
- Loss timestamp
- Recovery configuration

**Outputs:**
- Recovered bbox (if successful)
- Recovery status (searching, recovered, timeout)
- Search area visualization

**Used By:**
- Main loop: Handles state transitions
- Task 3: Reinitializes tracker on recovery
- Task 5: Visualizes recovery process in mosaic

**State Dependencies:**
- Only active in LOST state
- Transitions to TRACKING (success) or DETECTION (timeout/abort)

---

## Estimated Effort

- State Transition Logic: 3-4 hours
- Redetection Algorithm: 4-5 hours
- Distance & Size Matching: 2-3 hours
- Tracker Reinitialization: 2-3 hours
- Timeout Handling: 2-3 hours
- Visualization: 2-3 hours
- Testing: 3-4 hours
- Debugging: 2-3 hours
- **Total: 1.5-2 days**

# PTZ Tracker - REVISED Recommendations (Keeping Multi-Object Workflow)

**Date:** November 7, 2025 (Revised)
**Status:** Updated based on user requirement
**Goal:** Clean, minimal implementation that **keeps multi-object → ID selection → single-object workflow**

---

## Important Update

**User Requirement:** "I still want the multi object tracking -> ID selection -> single object tracking behavior"

This is a **valid and user-friendly workflow**! The revised recommendations below maintain this feature while still simplifying the overall architecture.

---

## Revised Architecture: Supporting the Desired Workflow

### The Workflow (Keep This!)

```
1. DETECTION_MODE: Show all moving objects with IDs (cyan boxes)
   ↓
2. User presses number key (0-9) to select object by ID
   ↓
3. LOCKED_MODE: Track selected object with high accuracy (green box)
   ↓
4. If tracking fails → LOST state → attempt recovery
   ↓
5. If recovery fails or user presses 'R' → back to DETECTION_MODE
```

**This is good UX!** The user can see all options before committing to track one.

---

## Question: How to Implement Multi-Object Tracking?

You have **two approaches**:

### Option A: Use Norfair (Simpler, Recommended)

**Pros:**
- ✅ Handles ID assignment automatically
- ✅ Maintains ID persistence across frames
- ✅ Robust to occlusions
- ✅ Proven library, well-maintained
- ✅ ~50 lines of integration code

**Cons:**
- ❌ External dependency (but lightweight)
- ❌ Slightly more complex than needed

**Code Example:**
```python
from norfair import Detection, Tracker

# Initialize once
tracker = Tracker(distance_threshold=50, hit_counter_max=10)

# In detection loop
detections = []
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    centroid = np.array([[x + w/2, y + h/2]])
    detections.append(Detection(points=centroid, data={'bbox': (x,y,w,h)}))

# Update tracker
tracked_objects = tracker.update(detections)

# Display with IDs
for obj in tracked_objects:
    bbox = obj.last_detection.data['bbox']
    obj_id = obj.id
    cv2.rectangle(frame, bbox, (255, 255, 0), 2)  # Cyan
    cv2.putText(frame, f"ID: {obj_id}", ...)
```

**Total additional code: ~50 lines**

---

### Option B: Simple Centroid Tracking (More Minimal)

Implement basic centroid tracking yourself without external library.

**Pros:**
- ✅ No external dependency
- ✅ Full control
- ✅ Very lightweight

**Cons:**
- ❌ You have to implement ID assignment logic
- ❌ Less robust to occlusions
- ❌ More code to maintain

**Code Example:**
```python
class CentroidTracker:
    def __init__(self, max_disappeared=10):
        self.next_id = 0
        self.objects = {}  # ID -> centroid
        self.disappeared = {}  # ID -> frames disappeared
        self.max_disappeared = max_disappeared

    def update(self, detected_centroids):
        # If no objects detected
        if len(detected_centroids) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    del self.objects[obj_id]
                    del self.disappeared[obj_id]
            return self.objects

        # If no existing objects, register all as new
        if len(self.objects) == 0:
            for centroid in detected_centroids:
                self._register(centroid)
        else:
            # Match existing objects to new detections
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            # Compute distances between existing and new centroids
            distances = np.linalg.norm(
                np.array(object_centroids)[:, None] - np.array(detected_centroids),
                axis=2
            )

            # Hungarian algorithm or simple nearest neighbor
            matched = self._match_objects(distances, object_ids, detected_centroids)

        return self.objects

    def _register(self, centroid):
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def _match_objects(self, distances, object_ids, detected_centroids):
        # Implementation of matching logic
        # Use scipy.optimize.linear_sum_assignment or simple nearest neighbor
        pass
```

**Total code: ~150 lines**

---

## Recommendation: Use Norfair

**For your use case, I recommend Norfair because:**

1. **It's designed exactly for this** - multi-object tracking with ID persistence
2. **Saves you 100+ lines of code** you'd have to write and debug
3. **More robust** - handles edge cases you might miss
4. **Still minimal** - it's a focused library, not a framework
5. **Easy to remove later** if you want to replace it

**The complexity I was worried about** isn't Norfair itself - it's all the other stuff (debug mosaic, extensive logging, BGSLibrary conflicts, etc.). Norfair is actually the right tool for the job.

---

## Revised Simplified Architecture

### File Structure (4-5 Files!)

```
PTZ_tracker_dumb/
├── main.py              # ~250 lines: Video I/O, main loop, UI, state machine
├── tracker.py           # ~300 lines: Background sub, Norfair multi-tracking, CSRT single-tracking
├── ptz.py              # ~100 lines: PTZ calculations
├── debug_view.py       # ~100 lines: Debug mosaic visualization
├── config.yaml         # ~50 lines: Config (with Norfair + debug mosaic params)
├── requirements.txt    # opencv-python, numpy, norfair, pyyaml
└── README.md
```

**Total: ~750 lines of actual code, 5 files**

Still clean and minimal!

---

## Revised State Machine (3 States, Simpler)

```python
class State(Enum):
    DETECTION = 0    # Multi-object mode with Norfair
    TRACKING = 1     # Single-object mode with CSRT
    LOST = 2         # Tracking failed, attempting recovery

# State transitions
def handle_frame(frame, state):
    if state == State.DETECTION:
        # Run background subtraction
        # Detect objects
        # Update Norfair tracker
        # Display all objects with IDs (cyan boxes)
        # Wait for user to press number key

        if user_pressed_number_key:
            selected_id = int(key)
            if selected_id in tracked_objects:
                bbox = get_bbox_for_id(selected_id)
                initialize_csrt(bbox)
                state = State.TRACKING

    elif state == State.TRACKING:
        # Update CSRT tracker
        success, bbox = csrt_tracker.update(frame)

        if success:
            # Display green box
            # Update PTZ to center object
        else:
            # Tracking lost
            state = State.LOST
            recovery_start_time = time.time()

    elif state == State.LOST:
        # Attempt to redetect object near last position
        # using background subtraction

        if object_redetected:
            reinitialize_csrt(new_bbox)
            state = State.TRACKING
        elif time.time() - recovery_start_time > TIMEOUT:
            # Give up, return to detection mode
            state = State.DETECTION

    if user_pressed_reset_key:
        state = State.DETECTION

    return state
```

**Simple, clear, supports your desired workflow.**

---

## Revised tracker.py Structure

```python
"""
Object detection and tracking module with dual-mode support
"""

import cv2
import numpy as np
from norfair import Detection, Tracker

class ObjectTracker:
    def __init__(self, config):
        self.config = config

        # Background subtractor
        self.bg_subtractor = self._create_bg_subtractor()

        # Multi-object tracker (Norfair)
        self.norfair_tracker = Tracker(
            distance_threshold=config['tracking']['norfair']['distance_threshold'],
            hit_counter_max=config['tracking']['norfair']['hit_counter_max'],
            initialization_delay=config['tracking']['norfair']['initialization_delay']
        )

        # Single-object tracker (CSRT)
        self.csrt_tracker = None

        # State
        self.state = "DETECTION"
        self.tracked_objects = []
        self.selected_id = None
        self.last_bbox = None

    def _create_bg_subtractor(self):
        """Create OpenCV background subtractor"""
        algo = self.config['background_subtraction']['algorithm']
        if algo == "MOG2":
            return cv2.createBackgroundSubtractorMOG2(
                history=self.config['background_subtraction']['history'],
                detectShadows=True
            )
        elif algo == "KNN":
            return cv2.createBackgroundSubtractorKNN()

    def update_detection_mode(self, frame):
        """Update in DETECTION mode - track all objects with Norfair"""
        # Apply background subtraction
        mask = self.bg_subtractor.apply(frame)

        # Clean mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours and create Norfair detections
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.config['object_detection']['min_area']:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            centroid = np.array([[x + w/2, y + h/2]])

            detection = Detection(
                points=centroid,
                data={'bbox': (x, y, w, h), 'area': area}
            )
            detections.append(detection)

        # Update Norfair tracker
        self.tracked_objects = self.norfair_tracker.update(detections)

        return self.tracked_objects

    def lock_onto_object(self, frame, object_id):
        """Lock onto specific object by ID - transition to TRACKING mode"""
        # Find object with this ID
        selected_obj = None
        for obj in self.tracked_objects:
            if obj.id == object_id:
                selected_obj = obj
                break

        if selected_obj is None:
            return False

        # Check if object is stable enough
        min_age = self.config['tracking']['selection']['min_track_age']
        if selected_obj.age < min_age:
            print(f"Object {object_id} not stable yet (age: {selected_obj.age})")
            return False

        # Get bbox
        if selected_obj.last_detection is None:
            return False

        bbox = selected_obj.last_detection.data['bbox']

        # Initialize CSRT tracker
        tracker_type = self.config['tracking']['tracker']
        if tracker_type == "CSRT":
            self.csrt_tracker = cv2.TrackerCSRT_create()
        elif tracker_type == "KCF":
            self.csrt_tracker = cv2.TrackerKCF_create()
        else:
            self.csrt_tracker = cv2.TrackerKCF_create()  # Default

        success = self.csrt_tracker.init(frame, bbox)

        if success:
            self.state = "TRACKING"
            self.selected_id = object_id
            self.last_bbox = bbox
            print(f"Locked onto object {object_id}")
            return True

        return False

    def update_tracking_mode(self, frame):
        """Update in TRACKING mode - track single object with CSRT"""
        if self.csrt_tracker is None:
            return False, None

        success, bbox = self.csrt_tracker.update(frame)

        if success and self._is_valid_bbox(bbox, frame.shape):
            self.last_bbox = tuple(map(int, bbox))
            return True, self.last_bbox
        else:
            # Tracking failed
            self.state = "LOST"
            return False, None

    def update_lost_mode(self, frame):
        """Attempt to redetect object in LOST mode"""
        # Run background subtraction
        mask = self.bg_subtractor.apply(frame)

        # Clean mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find contour nearest to last known position
        if self.last_bbox is None:
            return False, None

        last_center_x = self.last_bbox[0] + self.last_bbox[2] / 2
        last_center_y = self.last_bbox[1] + self.last_bbox[3] / 2

        search_radius = self.config['tracking']['recovery']['search_radius']
        best_contour = None
        best_distance = float('inf')

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.config['object_detection']['min_area']:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w / 2
            center_y = y + h / 2

            distance = np.sqrt((center_x - last_center_x)**2 + (center_y - last_center_y)**2)

            if distance < search_radius and distance < best_distance:
                # Check size similarity
                last_area = self.last_bbox[2] * self.last_bbox[3]
                size_ratio = min(area, last_area) / max(area, last_area)

                threshold = self.config['tracking']['recovery']['size_similarity_threshold']
                if size_ratio >= threshold:
                    best_contour = contour
                    best_distance = distance

        if best_contour is not None:
            # Redetected! Reinitialize CSRT
            bbox = cv2.boundingRect(best_contour)

            tracker_type = self.config['tracking']['tracker']
            if tracker_type == "CSRT":
                self.csrt_tracker = cv2.TrackerCSRT_create()
            else:
                self.csrt_tracker = cv2.TrackerKCF_create()

            success = self.csrt_tracker.init(frame, bbox)

            if success:
                self.state = "TRACKING"
                self.last_bbox = bbox
                print(f"Reacquired object {self.selected_id}")
                return True, bbox

        return False, None

    def _is_valid_bbox(self, bbox, frame_shape):
        """Check if bbox is reasonable"""
        x, y, w, h = map(int, bbox)

        # Check bounds
        if x < 0 or y < 0 or x + w > frame_shape[1] or y + h > frame_shape[0]:
            return False

        # Check area
        area = w * h
        min_area = self.config['tracking']['csrt'].get('min_bbox_area', 100)
        max_area = frame_shape[0] * frame_shape[1] * \
                   self.config['tracking']['csrt'].get('max_bbox_area_fraction', 0.8)

        return min_area <= area <= max_area

    def reset_to_detection_mode(self):
        """Reset tracker to DETECTION mode"""
        self.state = "DETECTION"
        self.csrt_tracker = None
        self.selected_id = None
        self.last_bbox = None
        print("Reset to DETECTION mode")
```

**~300 lines, handles both modes cleanly.**

---

## Revised main.py Structure

```python
"""
PTZ Object Tracker - Main Entry Point
Supports multi-object detection → ID selection → single-object tracking workflow
"""

import cv2
import yaml
import time
from tracker import ObjectTracker
from ptz import PTZController

def load_config(path='config.yaml'):
    """Load and validate configuration"""
    with open(path) as f:
        return yaml.safe_load(f)

def draw_detection_mode(frame, tracked_objects):
    """Draw all tracked objects with IDs in cyan"""
    for obj in tracked_objects:
        if obj.last_detection is None:
            continue

        bbox = obj.last_detection.data['bbox']
        x, y, w, h = bbox

        # Cyan box for detected objects
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)

        # Display ID
        cv2.putText(frame, f"ID: {obj.id}", (x, y-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    return frame

def draw_tracking_mode(frame, bbox, obj_id):
    """Draw locked object with green box"""
    x, y, w, h = bbox

    # Green box for locked object
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

    # Display "LOCKED" status
    cv2.putText(frame, f"LOCKED: ID {obj_id}", (x, y-10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return frame

def draw_lost_mode(frame, last_bbox, search_radius):
    """Draw search area in red"""
    if last_bbox is None:
        return frame

    x, y, w, h = last_bbox
    center_x = x + w // 2
    center_y = y + h // 2

    # Red circle showing search area
    cv2.circle(frame, (center_x, center_y), search_radius, (0, 0, 255), 2)

    # Display "SEARCHING" status
    cv2.putText(frame, "SEARCHING...", (center_x - 50, center_y - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return frame

def draw_status_overlay(frame, state, ptz, fps=0):
    """Draw status information overlay"""
    y = 30
    color = (255, 255, 255)

    # State
    cv2.putText(frame, f"Mode: {state}", (10, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    y += 30

    # PTZ info
    cv2.putText(frame, f"Pan: {ptz.pan:.1f}° Tilt: {ptz.tilt:.1f}° Zoom: {ptz.zoom:.1f}x",
               (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    y += 30

    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame

def main():
    # Load config
    config = load_config()

    # Initialize video
    cap = cv2.VideoCapture(config['video']['input'])
    if not cap.isOpened():
        print(f"Error: Could not open video {config['video']['input']}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {width}x{height} @ {fps} fps")

    # Initialize components
    tracker = ObjectTracker(config)
    ptz = PTZController((height, width), config['ptz'])

    # Output video
    if config['video']['save_output']:
        fourcc = cv2.VideoWriter_fourcc(*config['video']['output_codec'])
        out = cv2.VideoWriter(config['video']['output_path'], fourcc, fps, (width, height))
    else:
        out = None

    # State variables
    recovery_start_time = None
    recovery_timeout = config['tracking']['csrt']['recovery_timeout']
    frame_time = 0

    print("\n=== PTZ Tracker Started ===")
    print("DETECTION MODE: Press 0-9 to select object by ID")
    print("TRACKING MODE: Press 'R' to release and return to detection")
    print("Press 'Q' to quit\n")

    # Main loop
    while cap.isOpened():
        loop_start = time.time()

        ret, frame = cap.read()
        if not ret:
            if config['video'].get('loop_playback', False):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                break

        # Process based on state
        if tracker.state == "DETECTION":
            # Multi-object detection with Norfair
            tracked_objects = tracker.update_detection_mode(frame)

            # Draw all objects with IDs
            display_frame = frame.copy()
            draw_detection_mode(display_frame, tracked_objects)

            # PTZ follows largest object (optional)
            if tracked_objects:
                largest = max(tracked_objects,
                            key=lambda obj: obj.last_detection.data.get('area', 0)
                            if obj.last_detection else 0)
                if largest.last_detection:
                    bbox = largest.last_detection.data['bbox']
                    ptz.update(bbox)

        elif tracker.state == "TRACKING":
            # Single-object tracking with CSRT
            success, bbox = tracker.update_tracking_mode(frame)

            display_frame = frame.copy()

            if success:
                # Draw green box
                draw_tracking_mode(display_frame, bbox, tracker.selected_id)

                # Update PTZ to center object
                ptz.update(bbox)
            else:
                # Transitioned to LOST
                recovery_start_time = time.time()

        elif tracker.state == "LOST":
            # Attempting recovery
            display_frame = frame.copy()

            # Draw search area
            search_radius = config['tracking']['recovery']['search_radius']
            draw_lost_mode(display_frame, tracker.last_bbox, search_radius)

            # Attempt redetection
            success, bbox = tracker.update_lost_mode(frame)

            if success:
                # Reacquired! Back to TRACKING
                recovery_start_time = None
            elif time.time() - recovery_start_time > recovery_timeout:
                # Timeout - give up, return to DETECTION
                print("Recovery timeout - returning to DETECTION mode")
                tracker.reset_to_detection_mode()
                recovery_start_time = None

        # Apply PTZ transformation
        ptz_frame = ptz.extract_roi(display_frame)

        # Draw status overlay
        frame_time = 0.9 * frame_time + 0.1 * (time.time() - loop_start)
        current_fps = 1.0 / frame_time if frame_time > 0 else 0
        draw_status_overlay(ptz_frame, tracker.state, ptz, current_fps)

        # Display
        cv2.imshow("PTZ Tracker", ptz_frame)

        # Write output
        if out:
            out.write(ptz_frame)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:  # Q or ESC
            break

        elif key == ord('r'):  # Reset to DETECTION
            tracker.reset_to_detection_mode()
            ptz.reset()
            recovery_start_time = None

        elif ord('0') <= key <= ord('9'):  # Select object by ID
            if tracker.state == "DETECTION":
                selected_id = key - ord('0')
                success = tracker.lock_onto_object(frame, selected_id)
                if not success:
                    print(f"Could not lock onto object {selected_id}")

        elif key == ord(' '):  # Pause
            cv2.waitKey(0)

    # Cleanup
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

    print("\n=== PTZ Tracker Stopped ===")

if __name__ == "__main__":
    main()
```

**~250 lines, complete application with proper workflow support.**

---

## Revised Configuration

```yaml
# PTZ Tracker Configuration
# Supports multi-object detection → ID selection → single-object tracking

video:
  input: "input.mp4"
  output: "output.mp4"
  output_codec: "mp4v"
  save_output: true
  loop_playback: false

background_subtraction:
  algorithm: "MOG2"  # MOG2 or KNN
  history: 500
  learning_rate: -1  # -1 for automatic

object_detection:
  min_area: 500
  max_area_fraction: 0.5

tracking:
  # Tracker for single-object mode
  tracker: "KCF"  # KCF (fast) or CSRT (accurate)

  # Norfair settings for multi-object detection mode
  norfair:
    distance_threshold: 50
    hit_counter_max: 10
    initialization_delay: 3

  # Object selection
  selection:
    min_track_age: 5  # Minimum frames before object is selectable

  # CSRT/KCF settings
  csrt:
    min_bbox_area: 100
    max_bbox_area_fraction: 0.8
    recovery_timeout: 3.0

  # Recovery settings when tracking is lost
  recovery:
    search_radius: 150
    size_similarity_threshold: 0.5

ptz:
  pan_sensitivity: 45.0
  tilt_sensitivity: 30.0
  zoom_min: 1.0
  zoom_max: 5.0
  deadband: 0.05
  target_object_size: 0.3

display:
  show_window: true
  show_info: true
  show_debug_mosaic: true  # 2×4 grid of pipeline stages

  debug_mosaic:
    tile_width: 320
    tile_height: 240
```

**~45 lines, includes Norfair params.**

---

## Summary of Revised Approach

### What Changed
✅ **Kept:** Multi-object tracking → ID selection → single-object workflow
✅ **Kept:** Norfair for multi-object ID management (it's the right tool)
✅ **Kept:** CSRT/KCF for single-object tracking
✅ **Kept:** 3-state machine (DETECTION, TRACKING, LOST)

### What's Still Simplified
✅ **4 core files** instead of 34+
✅ **~750 lines** of focused code (including debug mosaic)
✅ **Debug mosaic included** - useful for debugging pipeline stages
✅ **No extensive logging infrastructure** in MVP (use basic logging)
✅ **OpenCV-only** background subtraction
✅ **Simple config** (~50 lines vs 131)
✅ **Clear, readable code structure**

### Why This Works
1. **Norfair is appropriate here** - it solves the exact problem (multi-object ID persistence)
2. **Still minimal** - only 4 files, ~650 lines total
3. **Clean separation** - each module has clear purpose
4. **Supports desired UX** - user can see all options before selecting
5. **Easy to understand** - state machine is simple and clear

---

## Dependencies

```txt
# requirements.txt
opencv-python>=4.5.0
numpy>=1.19.0
norfair>=2.0.0
pyyaml>=5.0
```

**Or with Pixi:**
```toml
[dependencies]
python = ">=3.8,<3.12"
opencv = ">=4.5"
numpy = ">=1.19"
pyyaml = ">=5.0"

[pypi-dependencies]
norfair = ">=2.0.0"
```

---

## Implementation Timeline

### Week 1: Core Detection and Tracking
- Day 1-2: Background subtraction + Norfair integration
- Day 3-4: CSRT single-object tracking
- Day 5: State transitions and keyboard input

### Week 2: PTZ and Polish
- Day 1-2: PTZ controller implementation
- Day 3: Drawing functions and UI
- Day 4-5: Testing, bug fixes, edge cases

### Week 3: Optional Enhancements
- Debug view
- Better recovery logic
- Telemetry
- Additional features as needed

---

## Conclusion

**The multi-object → ID selection → single-object workflow is good UX and worth keeping!**

The key is to implement it **cleanly** with:
- ✅ Focused modules (4 files)
- ✅ Clear responsibilities
- ✅ Simple state management
- ✅ Using the right tool (Norfair) for the right job

Norfair adds **~50 lines of integration code** but saves you **~150 lines of DIY tracking code** while being more robust. It's a good trade-off.

**You can still have a clean, minimal, easy-to-use system while supporting the workflow you want.**

---

**Document Version:** 2.0 (Revised)
**Status:** Updated to support multi-object → ID selection → single-object workflow
**Next Step:** Review and approve, then start implementation!

# PTZ Camera Object Tracking System - Technical Specifications

**Version:** 2.0 (Simplified)
**Date:** November 7, 2025
**Status:** Implementation Ready

---

## 1. PROJECT OVERVIEW

### 1.1 Purpose

A clean, minimal PTZ object tracking system that:
- Detects multiple moving objects using background subtraction
- Allows user to select an object by ID
- Tracks selected object with high accuracy
- Applies virtual PTZ control to keep object centered

### 1.2 Scope

**MVP Features:**
- Multi-object detection with ID assignment
- Manual object selection (press 0-9 for ID)
- Single-object high-accuracy tracking
- Virtual PTZ (pan, tilt, zoom) control
- Debug mosaic visualization (2×4 grid)
- Video input/output
- Keyboard controls

**Out of Scope for MVP:**
- Deep learning detection
- Physical PTZ camera control
- Real-time camera streams (file input only)
- Extensive telemetry and analytics

### 1.3 Design Principles

- **YAGNI**: Only implement what's needed now
- **Simplicity**: 5 files, ~750 lines of code
- **Clarity**: Each module has single responsibility
- **Proven patterns**: Use established CV techniques

---

## 2. SYSTEM ARCHITECTURE

### 2.1 Simplified Architecture

```
┌─────────────────────────────────────────────┐
│                  main.py                    │
│  (Video I/O, Main Loop, State Machine, UI)  │
└─────────────────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┬──────────┐
        │             │             │          │
┌───────▼────────┐ ┌──▼────────┐ ┌─▼──────┐ ┌─▼──────────┐
│   tracker.py   │ │  ptz.py   │ │config  │ │debug_view  │
│ (Detection &   │ │ (PTZ      │ │.yaml   │ │.py         │
│  Tracking)     │ │  Control) │ │        │ │(Mosaic)    │
└────────────────┘ └───────────┘ └────────┘ └────────────┘
```

**5 Files, ~750 Lines Total**

### 2.2 Module Responsibilities

| Module | Lines | Purpose |
|--------|-------|---------|
| **main.py** | ~250 | Video I/O, main loop, state machine, keyboard input, UI coordination |
| **tracker.py** | ~300 | Background subtraction, Norfair multi-tracking, CSRT single-tracking |
| **ptz.py** | ~100 | PTZ calculations, ROI extraction, coordinate transformations |
| **debug_view.py** | ~100 | Debug mosaic creation (2×4 grid of pipeline stages) |
| **config.yaml** | ~50 | Configuration parameters |

### 2.3 Data Flow

```
Video Frame
    │
    ├──> [DETECTION STATE]
    │    │
    │    ├─> Background Subtraction (OpenCV MOG2/KNN)
    │    ├─> Contour Detection & Filtering
    │    ├─> Norfair Multi-Object Tracking
    │    ├─> Display all objects with IDs (cyan boxes)
    │    └─> Wait for user to press number key (0-9)
    │
    ├──> [TRACKING STATE]
    │    │
    │    ├─> CSRT/KCF Single-Object Tracking
    │    ├─> PTZ Control (center object)
    │    ├─> Display locked object (green box)
    │    └─> Check tracking success
    │
    └──> [LOST STATE]
         │
         ├─> Background Subtraction near last position
         ├─> Attempt redetection (size + position match)
         ├─> Reinitialize CSRT if found
         └─> Timeout → return to DETECTION
```

---

## 3. STATE MANAGEMENT

### 3.1 Three-State System

```python
class State(Enum):
    DETECTION = 0   # Multi-object mode (Norfair)
    TRACKING = 1    # Single-object mode (CSRT/KCF)
    LOST = 2        # Recovery mode
```

### 3.2 State Descriptions

#### DETECTION State

**Purpose:** Show all moving objects with persistent IDs

**Active Components:**
- Background Subtraction: ✅ Active
- Norfair Tracker: ✅ Active
- CSRT Tracker: ❌ Inactive

**Visual:**
- Cyan boxes around all detected objects
- ID numbers above each box
- Status: "DETECTION MODE"

**User Actions:**
- Press 0-9: Lock onto object with that ID → TRACKING
- Press R: Reset PTZ to center
- Press Q: Quit

#### TRACKING State

**Purpose:** Track selected object with high accuracy

**Active Components:**
- Background Subtraction: ❌ Inactive
- Norfair Tracker: ❌ Inactive
- CSRT Tracker: ✅ Active

**Visual:**
- Green box around locked object
- ID number
- Status: "TRACKING - ID: X"

**User Actions:**
- Press R: Release lock → DETECTION
- Press Q: Quit

**Automatic Transitions:**
- CSRT fails → LOST

#### LOST State

**Purpose:** Attempt to reacquire lost object

**Active Components:**
- Background Subtraction: ✅ Active (for redetection)
- Norfair Tracker: ❌ Inactive
- CSRT Tracker: ⏸️ Suspended

**Visual:**
- Red circle showing search area
- Status: "SEARCHING..."
- Timer showing time remaining

**Recovery Logic:**
1. Run background subtraction
2. Find contours near last known position (within search_radius)
3. Match by size similarity
4. If found: Reinitialize CSRT → TRACKING
5. If timeout (3s): Give up → DETECTION

**User Actions:**
- Press R: Abort recovery → DETECTION
- Press Q: Quit

### 3.3 State Transition Diagram

```
        ┌──────────────┐
   ┌───▶│  DETECTION   │◀────────┐
   │    │ (Multi-obj)  │         │
   │    └──────┬───────┘         │
   │           │                 │
   │    Press number key         │
   │    (0-9)                    │
   │           │                 │
   │           ▼                 │
   │    ┌──────────────┐         │
   │    │   TRACKING   │         │
   │    │ (Single-obj) │         │
   │    └──────┬───────┘         │
   │           │                 │
   │    CSRT fails               │ Timeout
   │           │                 │ or press R
   │           ▼                 │
   │    ┌──────────────┐         │
   └────│     LOST     │─────────┘
        │  (Recovery)  │
        └──────────────┘
             ▲    │
             └────┘
           Redetected
```

---

## 4. BACKGROUND SUBTRACTION

### 4.1 Library: OpenCV Only

**Decision:** Use OpenCV exclusively (no BGSLibrary)

**Rationale:**
- Already a dependency
- MOG2 and KNN are sufficient for most cases
- Simpler installation and deployment
- Fewer dependencies to manage

### 4.2 Algorithms

#### MOG2 (Default - Recommended)

```python
bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=16,
    detectShadows=True
)
```

**Pros:**
- Best balance of speed and accuracy
- Good shadow detection
- Adaptive to lighting changes
- Works well outdoors

**Use for:** General purpose, outdoor scenes, dynamic lighting

#### KNN (Alternative)

```python
bg_subtractor = cv2.createBackgroundSubtractorKNN(
    history=500,
    dist2Threshold=400.0,
    detectShadows=True
)
```

**Pros:**
- Better for rapid changes
- Less memory usage
- Good for indoor scenes

**Use for:** Indoor scenes, static lighting, rapid motion

### 4.3 Mask Post-Processing (Simplified)

**Pipeline: 3 Steps (not 6)**

```python
# Step 1: Opening (removes noise, preserves objects)
kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# Step 2: Closing (fills holes)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# Step 3: Binary threshold
_, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
```

**Configuration:**
- `kernel_size`: 3, 5, or 7 (default: 5)
- `threshold_value`: 100-150 (default: 127)

---

## 5. OBJECT DETECTION

### 5.1 Contour Detection

```python
contours, _ = cv2.findContours(
    mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)
```

### 5.2 Filtering Criteria

```python
for contour in contours:
    area = cv2.contourArea(contour)

    # Filter by area
    if area < min_area:  # Default: 500
        continue

    if area > frame_area * max_area_fraction:  # Default: 0.5
        continue

    # Optional: Filter by aspect ratio
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / h
    if not (min_aspect_ratio <= aspect_ratio <= max_aspect_ratio):
        continue

    # Accept this contour
    valid_contours.append(contour)
```

### 5.3 Convert to Norfair Detections

```python
from norfair import Detection

detections = []
for contour in valid_contours:
    x, y, w, h = cv2.boundingRect(contour)
    centroid = np.array([[x + w/2, y + h/2]])

    detection = Detection(
        points=centroid,
        data={'bbox': (x, y, w, h)}
    )
    detections.append(detection)
```

---

## 6. MULTI-OBJECT TRACKING (Norfair)

### 6.1 Why Norfair?

- Handles ID assignment automatically
- Maintains ID persistence across frames
- Robust to temporary occlusions
- Only ~50 lines of integration code
- Saves ~150 lines of DIY tracking

### 6.2 Initialization

```python
from norfair import Tracker

tracker = Tracker(
    distance_threshold=50,      # Max pixels for association
    hit_counter_max=10,         # Frames to keep track without detection
    initialization_delay=3      # Frames before confirming new track
)
```

### 6.3 Update and Display

```python
# Update tracker
tracked_objects = tracker.update(detections)

# Display with IDs
for obj in tracked_objects:
    if obj.last_detection is None:
        continue

    bbox = obj.last_detection.data['bbox']
    obj_id = obj.id

    # Draw cyan box
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)

    # Draw ID
    cv2.putText(frame, f"ID: {obj_id}", (x, y-10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
```

### 6.4 Object Selection

```python
# User presses number key (0-9)
if ord('0') <= key <= ord('9'):
    selected_id = key - ord('0')

    # Find object with this ID
    for obj in tracked_objects:
        if obj.id == selected_id:
            # Check minimum age
            if obj.age >= min_track_age:  # Default: 5 frames
                bbox = obj.last_detection.data['bbox']
                # Initialize CSRT with this bbox
                initialize_csrt_tracker(frame, bbox)
                state = State.TRACKING
```

---

## 7. SINGLE-OBJECT TRACKING (CSRT/KCF)

### 7.1 Tracker Selection

**KCF (Default - Recommended for MVP)**
```python
tracker = cv2.TrackerKCF_create()
```
- **Speed:** ⚡⚡⚡ Fast (30+ FPS)
- **Accuracy:** ⭐⭐⭐ Good
- **Use for:** Real-time applications, MVP

**CSRT (Alternative - Higher Accuracy)**
```python
tracker = cv2.TrackerCSRT_create()
```
- **Speed:** ⚡⚡ Moderate (15-25 FPS)
- **Accuracy:** ⭐⭐⭐⭐ Excellent
- **Use for:** When accuracy > speed

### 7.2 Initialization

```python
def initialize_csrt(frame, bbox):
    tracker = cv2.TrackerKCF_create()  # or TrackerCSRT_create()
    success = tracker.init(frame, bbox)
    return tracker if success else None
```

### 7.3 Update Loop

```python
success, bbox = tracker.update(frame)

if success and is_valid_bbox(bbox, frame.shape):
    # Continue tracking
    x, y, w, h = map(int, bbox)
    # Draw green box
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
else:
    # Tracking failed
    state = State.LOST
```

### 7.4 Validation

```python
def is_valid_bbox(bbox, frame_shape):
    x, y, w, h = map(int, bbox)

    # Check bounds
    if x < 0 or y < 0:
        return False
    if x + w > frame_shape[1] or y + h > frame_shape[0]:
        return False

    # Check reasonable size
    area = w * h
    if area < min_bbox_area:  # Default: 100
        return False
    if area > frame_shape[0] * frame_shape[1] * 0.8:
        return False

    return True
```

---

## 8. RECOVERY MECHANISM

### 8.1 Lost Object Redetection

```python
def attempt_redetection(frame, last_bbox, config):
    # Run background subtraction
    mask = bg_subtractor.apply(frame)
    mask = clean_mask(mask)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Last known position
    last_cx = last_bbox[0] + last_bbox[2] / 2
    last_cy = last_bbox[1] + last_bbox[3] / 2
    last_area = last_bbox[2] * last_bbox[3]

    # Find nearest matching contour
    search_radius = config['recovery']['search_radius']  # Default: 150
    size_threshold = config['recovery']['size_similarity_threshold']  # Default: 0.5

    best_match = None
    best_distance = float('inf')

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        cx = x + w / 2
        cy = y + h / 2

        # Check distance
        distance = np.sqrt((cx - last_cx)**2 + (cy - last_cy)**2)
        if distance > search_radius:
            continue

        # Check size similarity
        size_ratio = min(area, last_area) / max(area, last_area)
        if size_ratio < size_threshold:
            continue

        # Better match found
        if distance < best_distance:
            best_match = (x, y, w, h)
            best_distance = distance

    return best_match
```

### 8.2 Timeout Handling

```python
recovery_start_time = time.time()
recovery_timeout = 3.0  # seconds

while state == State.LOST:
    bbox = attempt_redetection(frame, last_bbox, config)

    if bbox is not None:
        # Reacquired!
        tracker = initialize_csrt(frame, bbox)
        state = State.TRACKING
        break

    if time.time() - recovery_start_time > recovery_timeout:
        # Give up
        state = State.DETECTION
        break
```

---

## 9. PTZ CONTROL

### 9.1 Controller Class Structure

```python
class PTZController:
    def __init__(self, frame_shape, config):
        self.frame_shape = frame_shape  # (height, width)
        self.pan = 0.0
        self.tilt = 0.0
        self.zoom = 1.0
        self.config = config

    def update(self, bbox):
        """Update PTZ based on object bbox"""
        # Calculate object center
        cx = bbox[0] + bbox[2] / 2
        cy = bbox[1] + bbox[3] / 2

        # Calculate error from frame center
        error_x = (cx - self.frame_shape[1]/2) / self.frame_shape[1]
        error_y = (cy - self.frame_shape[0]/2) / self.frame_shape[0]

        # Apply deadband
        deadband = self.config['deadband']
        if abs(error_x) > deadband:
            self.pan += error_x * self.config['pan_sensitivity']
        if abs(error_y) > deadband:
            self.tilt += error_y * self.config['tilt_sensitivity']

        # Calculate zoom
        object_area = bbox[2] * bbox[3]
        frame_area = self.frame_shape[0] * self.frame_shape[1]
        target_fraction = self.config['target_object_size']
        target_area = frame_area * target_fraction

        self.zoom = np.sqrt(target_area / object_area)
        self.zoom = np.clip(self.zoom,
                           self.config['zoom_min'],
                           self.config['zoom_max'])

    def extract_roi(self, frame):
        """Extract ROI based on current PTZ state"""
        h, w = self.frame_shape

        # ROI size based on zoom
        roi_w = int(w / self.zoom)
        roi_h = int(h / self.zoom)

        # ROI center based on pan/tilt
        center_x = w // 2 + int(self.pan * w / 90)
        center_y = h // 2 + int(self.tilt * h / 90)

        # ROI top-left
        roi_x = max(0, min(center_x - roi_w // 2, w - roi_w))
        roi_y = max(0, min(center_y - roi_h // 2, h - roi_h))

        # Extract and resize
        roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        output = cv2.resize(roi, (w, h), interpolation=cv2.INTER_LINEAR)

        return output
```

### 9.2 Deadband Zone

Prevents jitter when object is near center:

```
        ┌────────────────────────┐
        │                        │
        │    ┌──────────┐        │
        │    │          │        │
        │    │ Deadband │        │  ← No adjustment
        │    │   Zone   │        │     in this area
        │    │          │        │
        │    └──────────┘        │
        │                        │
        └────────────────────────┘
```

Default: ±5% of frame dimension

### 9.3 Smoothing

Optional exponential smoothing:

```python
alpha = 0.3  # Smoothing factor
pan = alpha * new_pan + (1 - alpha) * previous_pan
tilt = alpha * new_tilt + (1 - alpha) * previous_tilt
zoom = alpha * new_zoom + (1 - alpha) * previous_zoom
```

---

## 10. DEBUG MOSAIC

### 10.1 Layout (2×4 Grid)

```
┌─────────────┬─────────────┬─────────────┬─────────────┐
│ 1. Original │ 2. FG Mask  │ 3. Cleaned  │ 4. Contours │
│             │    (Raw)    │    Mask     │             │
├─────────────┼─────────────┼─────────────┼─────────────┤
│ 5. Norfair  │ 6. CSRT     │ 7. PTZ ROI  │ 8. Final    │
│  Detection  │  Tracking   │   Overlay   │   Output    │
└─────────────┴─────────────┴─────────────┴─────────────┘
```

### 10.2 Implementation

See `debug_view.py`:

```python
from debug_view import DebugMosaic

# Initialize
debug = DebugMosaic(config)

# In main loop
pipeline_stages = {
    'original': frame.copy(),
    'fg_mask_raw': fg_mask_raw,
    'fg_mask_clean': fg_mask_clean,
    'contours': contours_frame,
    'detection': detection_frame,
    'tracking': tracking_frame,
    'ptz_roi': ptz_roi_overlay,
    'final': final_output
}

mosaic = debug.create_mosaic(pipeline_stages)
cv2.imshow("Debug Pipeline", mosaic)
```

### 10.3 Toggle During Runtime

Press **'D'** key to toggle on/off

---

## 11. CONFIGURATION

### 11.1 config.yaml Structure

```yaml
# Video I/O
video:
  input: "input.mp4"
  output: "output.mp4"
  output_codec: "mp4v"
  save_output: true
  loop_playback: false

# Background Subtraction (OpenCV only)
background_subtraction:
  algorithm: "MOG2"  # MOG2 or KNN
  history: 500
  learning_rate: -1  # -1 for automatic

# Object Detection
object_detection:
  min_area: 500
  max_area_fraction: 0.5

# Tracking
tracking:
  tracker: "KCF"  # KCF (fast) or CSRT (accurate)

  # Norfair multi-object settings
  norfair:
    distance_threshold: 50
    hit_counter_max: 10
    initialization_delay: 3

  # Selection
  selection:
    min_track_age: 5

  # Recovery
  recovery:
    search_radius: 150
    size_similarity_threshold: 0.5
    timeout: 3.0

# PTZ Control
ptz:
  pan_sensitivity: 45.0
  tilt_sensitivity: 30.0
  zoom_min: 1.0
  zoom_max: 5.0
  deadband: 0.05
  target_object_size: 0.3

# Display
display:
  show_window: true
  show_info: true
  show_debug_mosaic: true

  debug_mosaic:
    tile_width: 320
    tile_height: 240
```

---

## 12. KEYBOARD CONTROLS

| Key | Action | Available In |
|-----|--------|--------------|
| **0-9** | Select object by ID | DETECTION |
| **R** | Reset to DETECTION mode | All states |
| **D** | Toggle debug mosaic | All states |
| **Space** | Pause/Resume | All states |
| **Q / ESC** | Quit application | All states |

---

## 13. DEPENDENCIES

### 13.1 Requirements

```txt
opencv-python>=4.5.0
numpy>=1.19.0
norfair>=2.0.0
pyyaml>=5.0
```

### 13.2 Installation

**With pip:**
```bash
pip install -r requirements.txt
```

**With Pixi (recommended):**
```toml
[dependencies]
python = ">=3.8,<3.12"
opencv = ">=4.5"
numpy = ">=1.19"
pyyaml = ">=5.0"

[pypi-dependencies]
norfair = ">=2.0.0"
```

```bash
pixi install
pixi run python main.py
```

---

## 14. PERFORMANCE TARGETS

### 14.1 Frame Rate

- **720p (1280×720)**: ≥30 FPS
- **1080p (1920×1080)**: ≥20 FPS

### 14.2 Latency

- Frame to display: <50ms

### 14.3 Resource Usage

- **Memory**: <500 MB
- **CPU**: <80% of single core

---

## 15. FILE STRUCTURE

```
PTZ_tracker_dumb/
├── main.py              # 250 lines: Main application
├── tracker.py           # 300 lines: Detection & tracking
├── ptz.py              # 100 lines: PTZ control
├── debug_view.py       # 100 lines: Debug mosaic
├── config.yaml         # 50 lines: Configuration
├── requirements.txt    # Dependencies
├── README.md           # Quick start guide
├── PROJECT_PLAN.md     # Implementation plan
└── .gitignore

Total: ~750 lines of implementation code
```

---

## 16. IMPLEMENTATION CHECKLIST

### Week 1: Core Functionality
- [ ] Video I/O with OpenCV
- [ ] Background subtraction (MOG2)
- [ ] Contour detection and filtering
- [ ] Norfair integration for multi-object tracking
- [ ] CSRT/KCF single-object tracking
- [ ] State machine (3 states)
- [ ] Keyboard input handling

### Week 2: PTZ and Visualization
- [ ] PTZ controller implementation
- [ ] ROI extraction and transformation
- [ ] Debug mosaic creation
- [ ] Drawing functions (boxes, IDs, status)
- [ ] Recovery mechanism

### Week 3: Polish and Testing
- [ ] Configuration system
- [ ] Edge case handling
- [ ] Performance optimization
- [ ] Documentation
- [ ] Testing with various videos

---

## 17. SUCCESS CRITERIA

✅ Loads and processes video files
✅ Detects multiple moving objects
✅ Assigns persistent IDs to objects
✅ User can select object by pressing number key
✅ Tracks selected object with CSRT/KCF
✅ Applies virtual PTZ to keep object centered
✅ Debug mosaic shows pipeline stages
✅ Handles tracking loss gracefully
✅ Processes at target frame rate
✅ Clean, maintainable codebase

---

## 18. FUTURE ENHANCEMENTS (Post-MVP)

- Deep learning detection (YOLO)
- Physical PTZ camera support (ONVIF)
- Real-time camera streams (RTSP)
- Multiple object simultaneous tracking
- Trajectory analysis and heatmaps
- Configuration UI
- Performance profiling and telemetry
- Advanced recovery strategies

---

**Document Version:** 2.0 (Simplified Implementation-Ready Version)
**Previous Version:** 1.0 (Comprehensive Research Document)
**Changes:** Simplified from 2700+ lines to 800 lines, removed over-engineering, focused on MVP implementation

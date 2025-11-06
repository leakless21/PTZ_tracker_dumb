# PTZ Camera Object Tracking System - Technical Specifications

## 1. PROJECT OVERVIEW

### 1.1 Purpose
A real-time object tracking system that uses background subtraction for object detection and simulated PTZ (Pan-Tilt-Zoom) camera control to keep detected objects centered in the video frame.

### 1.2 Scope
- Process pre-recorded video files with simulated PTZ control
- Detect moving objects using background subtraction
- Track detected objects and adjust virtual camera position
- Simulate pan, tilt, and zoom operations on video
- Provide real-time visualization of tracking performance

### 1.3 Key Constraints
- Use background subtraction (not deep learning) for object detection
- Simulate PTZ control on existing video (no physical camera)
- Real-time or near real-time processing capability
- Single object tracking (extensible to multiple objects)

---

## 2. SYSTEM ARCHITECTURE

### 2.1 High-Level Architecture
The system follows a pipeline architecture with the following stages:
1. Video Input Stage
2. Frame Preprocessing Stage
3. Background Subtraction Stage
4. Object Detection & Filtering Stage
5. Tracking Decision Stage
6. Virtual PTZ Control Stage
7. Frame Rendering Stage
8. Display & Recording Stage

### 2.2 Component Diagram
```
[Video Source]
    ↓
[Frame Capture Module]
    ↓
[Preprocessing Module] → [Background Model Manager]
    ↓
[Background Subtractor]
    ↓
[Object Detector] → [Morphological Processor]
    ↓
[Object Filter & Selector]
    ↓
[Tracking Controller] ← [PTZ State Manager]
    ↓
[Virtual PTZ Engine]
    ↓
[Frame Renderer] → [Overlay Generator]
    ↓
[Display Manager] + [Video Writer]
```

### 2.3 Data Flow
- Unidirectional flow from video input to display output
- State feedback loop between Tracking Controller and PTZ State Manager
- Background model continuously updated by Background Model Manager

---

## 3. COORDINATE SYSTEMS

### 3.1 Frame Coordinate System
- Origin: Top-left corner of the original video frame
- X-axis: Horizontal, increasing right (0 to frame_width)
- Y-axis: Vertical, increasing downward (0 to frame_height)
- Units: Pixels

### 3.2 Virtual PTZ Coordinate System
- Pan: Horizontal rotation in degrees (-180° to +180°, 0° = center)
- Tilt: Vertical rotation in degrees (-90° to +90°, 0° = center)
- Zoom: Magnification factor (1.0 = no zoom, >1.0 = zoomed in)

### 3.3 Normalized Coordinate System
- Used for tracking calculations
- Range: (0.0, 0.0) to (1.0, 1.0)
- Independent of actual frame resolution
- Center point: (0.5, 0.5)

### 3.4 Region of Interest (ROI) Coordinate System
- Defines the current virtual camera viewport
- Expressed as rectangle in frame coordinates
- Calculated from pan, tilt, and zoom parameters
- Bounds: Must remain within original frame dimensions

---

## 4. BACKGROUND SUBTRACTION MODULE

### 4.1 Algorithm Selection
Support for multiple background subtraction algorithms:
- **MOG2** (Mixture of Gaussians 2): Primary algorithm
- **KNN** (K-Nearest Neighbors): Alternative option
- **GMG** (Geometric Multigrid): For static camera scenarios
- **Simple Frame Differencing**: Lightweight baseline

### 4.2 MOG2 Configuration Parameters
- **History**: Number of last frames affecting background model (default: 500)
- **varThreshold**: Threshold on squared Mahalanobis distance (default: 16)
- **detectShadows**: Enable/disable shadow detection (default: true)
- **shadowValue**: Value used to mark shadows in output (default: 127)
- **shadowThreshold**: Shadow detection threshold (default: 0.5)
- **learningRate**: Background model update rate (default: -1 for automatic)

### 4.3 KNN Configuration Parameters
- **History**: Number of frames for background model (default: 500)
- **dist2Threshold**: Distance threshold (default: 400.0)
- **detectShadows**: Enable shadow detection (default: true)

### 4.4 Background Model Management
- **Initialization Phase**: First N frames used to build initial model (N = 30-120)
- **Update Strategy**: Continuous online update or periodic reset
- **Learning Rate Adaptation**:
  - Higher rate (0.01-0.05) for dynamic scenes
  - Lower rate (0.001-0.005) for static scenes
  - Automatic adjustment based on scene change detection
- **Reset Conditions**:
  - Manual reset trigger
  - Automatic reset when tracking is lost for >T seconds
  - Scene change detection (average pixel change > threshold)

### 4.5 Foreground Mask Post-Processing
- **Binary Threshold**: Convert grayscale mask to binary (threshold: 128)
- **Morphological Operations Sequence**:
  1. Opening (erosion followed by dilation) - removes noise
  2. Closing (dilation followed by erosion) - fills holes
- **Kernel Configurations**:
  - Small kernel (3×3 or 5×5) for fine details
  - Medium kernel (7×7 or 9×9) for standard processing
  - Large kernel (11×11 or 15×15) for aggressive filtering
- **Kernel Shapes**: Rectangular, elliptical, or cross-shaped

---

## 5. OBJECT DETECTION MODULE

### 5.1 Contour Detection
- **Method**: Find external contours in binary foreground mask
- **Mode**: External contours only (RETR_EXTERNAL)
- **Approximation**: Chain approximation for efficiency (CHAIN_APPROX_SIMPLE)

### 5.2 Contour Filtering Criteria
- **Minimum Area**: Minimum contour area in pixels (default: 500)
- **Maximum Area**: Maximum contour area as fraction of frame (default: 0.3)
- **Aspect Ratio**: Width/height ratio limits (default: 0.2 to 5.0)
- **Solidity**: Contour area / convex hull area (default: >0.3)
- **Extent**: Contour area / bounding rectangle area (default: >0.2)

### 5.3 Bounding Box Calculation
- **Method**: Minimum area rectangle or axis-aligned rectangle
- **Representation**: (x, y, width, height) in frame coordinates
- **Padding**: Optional expansion by N pixels in all directions

### 5.4 Object Properties Extraction
For each detected object, calculate:
- **Centroid**: Center point of bounding box or contour moments
- **Area**: Contour area in square pixels
- **Perimeter**: Contour perimeter in pixels
- **Bounding Box**: Axis-aligned rectangle
- **Orientation**: Angle of minimum area rectangle
- **Aspect Ratio**: Width / height
- **Solidity**: Area / convex hull area
- **Extent**: Area / bounding rectangle area

### 5.5 Object Selection Strategy
When multiple objects detected, prioritize by:
1. **Largest Area**: Select object with maximum area
2. **Closest to Center**: Select object nearest to frame center
3. **Previously Tracked**: Continue tracking same object (with association)
4. **Highest Confidence**: Composite score from multiple features
5. **User-defined Priority Zones**: Prefer objects in specific regions

---

## 6. TRACKING CONTROLLER

### 6.1 Tracking States
- **IDLE**: No tracking active, searching for objects
- **ACQUIRING**: Object detected, initializing tracking
- **TRACKING**: Actively tracking object
- **LOST**: Tracking lost, attempting recovery
- **RECOVERING**: Attempting to reacquire lost object

### 6.2 State Transitions
- IDLE → ACQUIRING: Object detected for N consecutive frames
- ACQUIRING → TRACKING: Object position stable for M frames
- TRACKING → LOST: Object not detected for K frames
- LOST → RECOVERING: Initiate search pattern
- RECOVERING → TRACKING: Object redetected
- RECOVERING → IDLE: Recovery timeout exceeded
- Any → IDLE: Manual reset

### 6.3 Tracking Parameters
- **Acquisition Frames**: Consecutive detections required (default: 3)
- **Stability Frames**: Frames for stable tracking (default: 5)
- **Lost Frames Threshold**: Frames before declaring lost (default: 10)
- **Recovery Timeout**: Maximum recovery duration in seconds (default: 5)
- **Position Stability Threshold**: Maximum centroid movement in pixels (default: 5)

### 6.4 Object Association
When reacquiring or handling occlusions:
- **Distance-based**: Associate to nearest object within threshold
- **Appearance-based**: Compare color histograms or basic features
- **Motion prediction**: Predict position using velocity estimate
- **IoU (Intersection over Union)**: Overlap with predicted bounding box

### 6.5 Tracking Quality Metrics
- **Confidence Score**: 0.0 to 1.0, based on:
  - Object size consistency
  - Position smoothness
  - Detection consistency
  - Time since last detection
- **Update Confidence Calculation**:
  - Decay over time when object not detected
  - Increase when object consistently detected
  - Use exponential moving average

---

## 7. VIRTUAL PTZ CONTROL ENGINE

### 7.1 Control Loop Architecture
- **Input**: Target object centroid in frame coordinates
- **Output**: Pan, tilt, zoom commands
- **Update Rate**: Match video frame rate (e.g., 30 Hz)
- **Control Strategy**: PID controller or proportional control

### 7.2 Pan Control
- **Input**: Horizontal offset from frame center
- **Calculation**:
  - Error = (object_x - frame_center_x) / frame_width
  - Pan_angle += error × pan_sensitivity
- **Pan Sensitivity**: Degrees per normalized unit (default: 45°)
- **Pan Limits**: Calculated based on zoom level and frame dimensions
- **Pan Speed Limits**: Maximum change per frame (default: 2° per frame)

### 7.3 Tilt Control
- **Input**: Vertical offset from frame center
- **Calculation**:
  - Error = (object_y - frame_center_y) / frame_height
  - Tilt_angle += error × tilt_sensitivity
- **Tilt Sensitivity**: Degrees per normalized unit (default: 30°)
- **Tilt Limits**: Calculated based on zoom level and frame dimensions
- **Tilt Speed Limits**: Maximum change per frame (default: 1.5° per frame)

### 7.4 Zoom Control
- **Trigger Conditions**:
  - Object too small: zoom in
  - Object too large: zoom out
  - Object near frame edge: zoom out for wider view
- **Zoom Target Calculation**:
  - Desired object size: 20-40% of frame dimension
  - Zoom = frame_dimension / (desired_fraction × object_dimension)
- **Zoom Limits**:
  - Minimum: 1.0 (no zoom)
  - Maximum: Configurable (default: 5.0)
- **Zoom Speed**: Exponential change rate (default: 0.05 per frame)

### 7.5 Deadband Zones
- **Center Deadband**: No adjustment if object within central region
  - Horizontal deadband: ±5% of frame width
  - Vertical deadband: ±5% of frame height
- **Purpose**: Reduce jitter and unnecessary micro-adjustments
- **Configurable**: Adjust based on tracking precision requirements

### 7.6 Smoothing and Damping
- **Exponential Moving Average**:
  - Smooth pan/tilt commands: value = alpha × new + (1-alpha) × old
  - Alpha (smoothing factor): 0.1 to 0.5
- **Velocity Limiting**: Cap maximum change per frame
- **Acceleration Limiting**: Cap rate of velocity change

### 7.7 ROI Calculation from PTZ Parameters
Given pan, tilt, and zoom:
1. Calculate virtual viewport size: (width/zoom, height/zoom)
2. Calculate viewport center from pan/tilt angles
3. Calculate ROI top-left: (center_x - width/(2×zoom), center_y - height/(2×zoom))
4. Clamp ROI to stay within original frame bounds
5. Return ROI as (x, y, w, h)

---

## 8. FRAME RENDERING MODULE

### 8.1 Virtual Camera Transformation
- **Input**: Original frame + ROI parameters
- **Process**:
  1. Extract ROI from original frame
  2. Resize ROI to output frame dimensions
  3. Apply interpolation for quality
- **Interpolation Methods**:
  - INTER_LINEAR: Fast, good quality (default)
  - INTER_CUBIC: Slower, better quality
  - INTER_LANCZOS4: Slowest, best quality
  - INTER_NEAREST: Fastest, lowest quality

### 8.2 Overlay Rendering
Render the following overlays on output frame:

#### 8.2.1 Bounding Box Overlay
- **Rectangle**: Around detected object
- **Color**: Green (tracking), Yellow (acquiring), Red (lost)
- **Thickness**: 2-3 pixels
- **Style**: Solid or dashed line

#### 8.2.2 Crosshair Overlay
- **Position**: Frame center
- **Color**: White or cyan
- **Size**: 20-40 pixels
- **Thickness**: 1-2 pixels
- **Style**: Cross or circle

#### 8.2.3 Information Overlay
Display text information:
- **PTZ State**: Current pan, tilt, zoom values
- **Tracking State**: IDLE, TRACKING, LOST, etc.
- **Object Info**: Size, position, confidence
- **Frame Counter**: Current frame number
- **FPS**: Processing frame rate
- **Position**: Top-left corner or custom
- **Font**: Monospace, scalable
- **Background**: Semi-transparent rectangle for readability

#### 8.2.4 Tracking Visualization
- **Trajectory Trail**: Show past N positions of tracked object
- **Velocity Vector**: Arrow showing object movement direction
- **Zoom Indicator**: Visual representation of current zoom level
- **Deadband Zone**: Rectangle showing center deadband area

### 8.3 Multi-View Display
Optional side-by-side or picture-in-picture views:
- **Original Frame**: Full original video with detections
- **Virtual PTZ View**: Current zoomed/panned view
- **Foreground Mask**: Binary detection mask
- **Background Model**: Current background estimate (optional)

### 8.4 Color Space Handling
- **Internal Processing**: BGR (OpenCV default)
- **Display Output**: BGR for video, RGB for GUI frameworks
- **Conversion**: Automatic when required

---

## 9. VIDEO INPUT/OUTPUT MODULE

### 9.1 Video Input Specifications
- **Supported Formats**:
  - Container: MP4, AVI, MOV, MKV
  - Codecs: H.264, H.265, MJPEG, MPEG-4
- **Resolution**: Arbitrary (tested from 640×480 to 1920×1080)
- **Frame Rate**: Arbitrary (common: 24, 25, 30, 60 fps)
- **Color Space**: RGB, BGR, or grayscale

### 9.2 Video Input Handler
- **Frame Capture**: Sequential frame-by-frame reading
- **Frame Buffering**: Optional buffer for smooth playback
- **Frame Skipping**: Option to process every Nth frame
- **Loop Playback**: Restart video when end reached
- **Seek Support**: Jump to specific frame or timestamp

### 9.3 Video Output Specifications
- **Output Format**: MP4 (H.264) or AVI (MJPEG)
- **Resolution**: Same as input or configurable
- **Frame Rate**: Match input frame rate
- **Quality**: Configurable compression level
- **Audio**: Passthrough or discard (not processed)

### 9.4 Output Recording
- **Dual Recording**:
  - Virtual PTZ output view
  - Original frame with overlays (optional)
- **Filename Convention**: timestamp-based or sequential
- **Metadata**: Include PTZ parameters in separate log file

---

## 10. CONFIGURATION SYSTEM

### 10.1 Configuration File Format
- **Format**: JSON or YAML
- **Location**: Configuration file in project directory
- **Structure**: Hierarchical sections matching modules

### 10.2 Configuration Parameters Structure

#### Background Subtraction Section
```
background_subtraction:
  algorithm: "MOG2" | "KNN" | "GMG"
  history: integer (100-1000)
  var_threshold: float (4.0-50.0)
  detect_shadows: boolean
  learning_rate: float (-1.0 for auto, 0.0001-0.1)
```

#### Morphological Processing Section
```
morphology:
  enable: boolean
  open_kernel_size: integer (3, 5, 7, ...)
  close_kernel_size: integer (3, 5, 7, ...)
  kernel_shape: "rect" | "ellipse" | "cross"
  iterations: integer (1-3)
```

#### Object Detection Section
```
object_detection:
  min_area: integer (pixels)
  max_area: float (0.0-1.0, fraction of frame)
  min_aspect_ratio: float
  max_aspect_ratio: float
  min_solidity: float (0.0-1.0)
  selection_strategy: "largest" | "center" | "previous"
```

#### Tracking Section
```
tracking:
  acquisition_frames: integer (1-10)
  lost_frames_threshold: integer (5-30)
  recovery_timeout: float (seconds)
  deadband_x: float (0.0-0.2, fraction)
  deadband_y: float (0.0-0.2, fraction)
```

#### PTZ Control Section
```
ptz_control:
  pan_sensitivity: float (10.0-90.0 degrees)
  tilt_sensitivity: float (10.0-90.0 degrees)
  max_pan_speed: float (degrees per frame)
  max_tilt_speed: float (degrees per frame)
  min_zoom: float (1.0)
  max_zoom: float (1.0-10.0)
  zoom_speed: float (0.01-0.2)
  smoothing_factor: float (0.0-1.0)
```

#### Display Section
```
display:
  show_bounding_box: boolean
  show_crosshair: boolean
  show_info_overlay: boolean
  show_trajectory: boolean
  show_original: boolean
  show_mask: boolean
  output_width: integer
  output_height: integer
```

#### Video I/O Section
```
video:
  input_path: string
  output_path: string
  output_codec: "H264" | "MJPEG"
  save_output: boolean
  loop_playback: boolean
  process_every_n_frames: integer (1 = all frames)
```

### 10.3 Runtime Configuration
- **Hot Reload**: Reload configuration without restart (optional)
- **Command-line Override**: Override config file with CLI arguments
- **Validation**: Schema validation on load
- **Defaults**: Fallback values for all parameters

---

## 11. PERFORMANCE REQUIREMENTS

### 11.1 Processing Speed
- **Target Frame Rate**:
  - 30 fps for 720p video (real-time)
  - 15 fps minimum for 1080p video
- **Latency**: Maximum 100ms from frame capture to display
- **Startup Time**: <3 seconds to begin processing

### 11.2 Resource Utilization
- **CPU Usage**: <80% of single core for 720p
- **Memory**: <500 MB for 1080p processing
- **Disk I/O**: Sufficient for video read/write without dropping frames

### 11.3 Scalability
- **Resolution Independence**: Work with any resolution (within memory limits)
- **Frame Rate Independence**: Adapt to input video frame rate
- **Multi-threading**: Optional parallel processing for performance boost

---

## 12. ERROR HANDLING AND RECOVERY

### 12.1 Input Validation
- **Video File Existence**: Check before processing
- **Video Format Support**: Verify codec compatibility
- **Configuration Validation**: Check for invalid parameter values
- **Error Messages**: Clear, actionable error descriptions

### 12.2 Runtime Error Handling
- **Frame Read Failure**:
  - Skip corrupted frame
  - Log error and continue
  - Abort if consecutive failures exceed threshold
- **Object Detection Failure**:
  - Return to IDLE state
  - Reset background model if persistent
- **PTZ Calculation Errors**:
  - Clamp to valid ranges
  - Log warning
  - Continue with constrained values

### 12.3 Recovery Mechanisms
- **Tracking Loss Recovery**:
  - Expand search area
  - Reduce background subtraction threshold
  - Reset to wide view (zoom out)
- **Background Model Reset**:
  - Automatic reset on scene change
  - Manual reset command
- **Graceful Degradation**:
  - Reduce frame rate if processing falls behind
  - Disable non-essential overlays for performance

### 12.4 Logging
- **Log Levels**: DEBUG, INFO, WARNING, ERROR
- **Log Format**: Timestamp, level, module, message
- **Log Output**: Console and file
- **Rotation**: Time-based or size-based log rotation

---

## 13. DATA STRUCTURES

### 13.1 Object Detection Result
```
DetectedObject:
  - centroid: (x: float, y: float)
  - bounding_box: (x: int, y: int, width: int, height: int)
  - area: float
  - contour: array of points
  - confidence: float (0.0-1.0)
  - timestamp: float
  - id: integer (optional, for multi-object tracking)
```

### 13.2 PTZ State
```
PTZState:
  - pan: float (degrees)
  - tilt: float (degrees)
  - zoom: float (magnification)
  - pan_velocity: float (degrees per second)
  - tilt_velocity: float (degrees per second)
  - zoom_velocity: float (zoom units per second)
  - roi: (x: int, y: int, width: int, height: int)
  - timestamp: float
```

### 13.3 Tracking State
```
TrackingState:
  - status: enum (IDLE, ACQUIRING, TRACKING, LOST, RECOVERING)
  - target_object: DetectedObject or null
  - frames_since_detection: integer
  - confidence: float (0.0-1.0)
  - trajectory: array of (x, y, timestamp) tuples
  - acquisition_counter: integer
  - lost_counter: integer
```

### 13.4 Frame Metadata
```
FrameMetadata:
  - frame_number: integer
  - timestamp: float
  - frame_shape: (height: int, width: int, channels: int)
  - ptz_state: PTZState
  - tracking_state: TrackingState
  - detections: array of DetectedObject
  - processing_time: float (milliseconds)
```

---

## 14. PROCESSING PIPELINE SEQUENCE

### 14.1 Initialization Phase
1. Load configuration from file
2. Validate configuration parameters
3. Open video input source
4. Read first frame to get dimensions
5. Initialize background subtraction model
6. Initialize PTZ state (pan=0, tilt=0, zoom=1.0)
7. Initialize tracking state (status=IDLE)
8. Create video writer for output (if enabled)
9. Initialize display window

### 14.2 Main Processing Loop
For each frame:
1. **Capture Frame**
   - Read next frame from video
   - If end of video: handle based on loop setting
   - Extract timestamp or calculate from frame number

2. **Preprocessing**
   - Resize if needed (optional)
   - Convert to appropriate color space
   - Apply noise reduction (optional, e.g., Gaussian blur)

3. **Background Subtraction**
   - Apply background subtraction algorithm
   - Get foreground mask
   - Update background model with current frame

4. **Foreground Mask Processing**
   - Apply binary threshold
   - Perform morphological operations (opening, closing)
   - Remove shadows (if enabled)

5. **Object Detection**
   - Find contours in processed mask
   - Filter contours by size, shape criteria
   - Calculate bounding boxes and centroids
   - Generate DetectedObject instances

6. **Object Selection**
   - Apply selection strategy to choose target
   - If multiple objects: prioritize based on criteria
   - If no objects: return null

7. **Tracking Update**
   - Update tracking state based on detection
   - Perform state transitions
   - Update confidence score
   - Associate detection with previous target (if applicable)

8. **PTZ Control Calculation**
   - If tracking: calculate error from frame center
   - Compute pan, tilt, zoom adjustments
   - Apply smoothing and limiting
   - Update PTZ state
   - Calculate new ROI

9. **Frame Rendering**
   - Extract ROI from original frame
   - Resize to output dimensions
   - Apply interpolation

10. **Overlay Rendering**
    - Draw bounding boxes
    - Draw crosshair
    - Draw information text
    - Draw trajectory (if enabled)

11. **Output**
    - Display frame in window
    - Write frame to output video (if enabled)
    - Log frame metadata (if enabled)

12. **User Input Handling**
    - Check for keyboard input
    - Handle pause, reset, quit commands
    - Adjust parameters if interactive mode enabled

13. **Performance Monitoring**
    - Calculate frame processing time
    - Update FPS counter
    - Adjust processing if falling behind (optional)

### 14.3 Cleanup Phase
1. Release video input source
2. Release video output writer
3. Close display windows
4. Save final logs and statistics
5. Export PTZ trajectory data (optional)

---

## 15. USER INTERFACE

### 15.1 Display Window
- **Main Window**: Shows virtual PTZ output with overlays
- **Size**: Configurable, default 800×600
- **Title**: Application name and current status
- **Refresh Rate**: Match video frame rate

### 15.2 Keyboard Controls
- **Space**: Pause/resume playback
- **R**: Reset tracking and PTZ to defaults
- **B**: Reset background model
- **Q/Esc**: Quit application
- **S**: Save current frame as image
- **+/-**: Manually adjust zoom level
- **Arrow Keys**: Manually adjust pan/tilt (when in manual mode)
- **M**: Toggle between auto and manual PTZ mode
- **O**: Toggle overlays on/off
- **F**: Toggle fullscreen mode

### 15.3 Mouse Controls (Optional)
- **Left Click**: Select object to track manually
- **Right Click**: Reset PTZ to clicked position
- **Mouse Wheel**: Adjust zoom level

### 15.4 Status Display
Real-time information shown in overlay:
- Current tracking state
- Pan angle (degrees)
- Tilt angle (degrees)
- Zoom level (magnification)
- Target object size (pixels)
- Processing FPS
- Frame number / total frames

---

## 16. OUTPUT DATA AND LOGGING

### 16.1 Video Output
- **Primary Output**: Virtual PTZ view video file
- **Debug Output** (optional): Side-by-side comparison video
- **Mask Output** (optional): Foreground mask video

### 16.2 Telemetry Log
CSV or JSON file containing per-frame data:
- Frame number
- Timestamp
- Pan angle
- Tilt angle
- Zoom level
- Tracking state
- Object centroid (x, y)
- Object bounding box (x, y, w, h)
- Object area
- Confidence score
- Processing time (ms)

### 16.3 Event Log
Text log file containing:
- Application start/stop events
- Configuration changes
- State transitions (IDLE→TRACKING, etc.)
- Errors and warnings
- Performance metrics summary

### 16.4 Statistics Summary
Final statistics file (JSON) containing:
- Total frames processed
- Average FPS
- Tracking success rate (% of frames with active tracking)
- Average pan, tilt, zoom values
- Number of tracking losses and recoveries
- Processing time distribution (min, max, mean, median)

---

## 17. TESTING REQUIREMENTS

### 17.1 Unit Testing
Test individual components:
- Background subtraction with synthetic sequences
- Object detection with known objects
- PTZ calculations with known inputs
- Coordinate transformations
- Configuration loading and validation

### 17.2 Integration Testing
Test component interactions:
- Full pipeline with sample videos
- State transitions under various scenarios
- Error handling and recovery
- Configuration changes during runtime

### 17.3 Performance Testing
- Benchmark processing speed on various resolutions
- Memory usage monitoring
- CPU utilization profiling
- Frame drop detection

### 17.4 Test Scenarios
Provide test videos covering:
- **Simple Scene**: Single person walking, static background
- **Complex Scene**: Multiple moving objects, dynamic background
- **Challenging Conditions**:
  - Low contrast
  - Rapid lighting changes
  - Camera shake (in original video)
  - Occlusions
  - Objects entering/leaving frame

### 17.5 Validation Criteria
- **Tracking Accuracy**: Object stays centered within ±10% of frame
- **Tracking Stability**: No excessive jitter (measured by pan/tilt variance)
- **Recovery Time**: Reacquire tracking within 2 seconds of loss
- **False Positive Rate**: <5% false detections on test set

---

## 18. FUTURE ENHANCEMENTS (Out of Scope for Initial Version)

### 18.1 Advanced Object Detection
- Deep learning-based detection (YOLO, SSD)
- Multi-object tracking
- Object classification (person, vehicle, animal, etc.)
- Specific object filtering

### 18.2 Advanced Tracking
- Kalman filter for motion prediction
- Optical flow for motion estimation
- Appearance-based tracking (color histograms, HOG)
- Object persistence across occlusions

### 18.3 Physical Camera Integration
- PTZ camera protocol support (ONVIF, Visca, PELCO-D)
- Network camera streaming (RTSP, HTTP)
- Real-time camera control commands
- Camera preset positions

### 18.4 Advanced PTZ Control
- Trajectory prediction and anticipation
- Multi-zone tracking
- Auto-patrol patterns when no object detected
- Smooth cinematic camera movements

### 18.5 User Interface Enhancements
- GUI configuration editor
- Real-time parameter adjustment sliders
- Multiple camera view support
- Playback controls (seek, speed adjustment)

### 18.6 Analytics and Reporting
- Heatmaps of object movement
- Dwell time analysis
- Event detection (object entering/leaving zones)
- Automated highlights extraction

---

## 19. DEVELOPMENT GUIDELINES

### 19.1 Programming Language
- **Primary**: Python 3.8+
- **Rationale**: Rich ecosystem for computer vision (OpenCV, NumPy)

### 19.2 Key Dependencies
- **OpenCV**: Video I/O, image processing, background subtraction
- **NumPy**: Numerical computations, array operations
- **Configuration**: JSON (built-in) or PyYAML
- **Logging**: Built-in logging module
- **Optional**: Matplotlib for visualization, SciPy for advanced filtering

### 19.3 Code Structure
Modular architecture with separate modules:
- `video_io.py`: Video input/output handling
- `background_subtraction.py`: Background subtraction algorithms
- `object_detection.py`: Object detection and filtering
- `tracking.py`: Tracking logic and state management
- `ptz_control.py`: Virtual PTZ calculations
- `rendering.py`: Frame rendering and overlays
- `config.py`: Configuration loading and validation
- `main.py`: Main application loop and orchestration
- `utils.py`: Utility functions and helpers

### 19.4 Coding Standards
- Follow PEP 8 style guidelines
- Type hints for function signatures
- Docstrings for all classes and functions
- Comprehensive error handling
- Avoid global variables (use class-based state)

### 19.5 Documentation
- README with setup and usage instructions
- Architecture diagram
- Configuration parameter reference
- API documentation for key functions
- Example videos and expected outputs

---

## 20. ACCEPTANCE CRITERIA

The system is considered complete when it can:

1. ✓ Load and process a video file without errors
2. ✓ Detect moving objects using background subtraction
3. ✓ Track a single object and maintain tracking state
4. ✓ Simulate PTZ camera movements (pan, tilt, zoom)
5. ✓ Keep tracked object centered in virtual camera view
6. ✓ Display real-time output with overlays
7. ✓ Save processed video to file
8. ✓ Handle tracking loss and recovery gracefully
9. ✓ Process video at acceptable frame rate (>15 fps for 720p)
10. ✓ Provide configurable parameters via config file
11. ✓ Respond to keyboard controls during playback
12. ✓ Generate telemetry logs with PTZ data
13. ✓ Work with various video formats and resolutions
14. ✓ Run without crashes on test video suite
15. ✓ Meet tracking accuracy criteria on validation set

---

## 21. GLOSSARY

- **PTZ**: Pan-Tilt-Zoom camera system
- **ROI**: Region of Interest
- **Background Subtraction**: Technique to identify moving objects by comparing frames to background model
- **MOG2**: Mixture of Gaussians 2, adaptive background subtraction algorithm
- **Foreground Mask**: Binary image showing detected moving regions
- **Morphological Operations**: Image processing operations (erosion, dilation, opening, closing)
- **Contour**: Curve joining continuous points with same intensity
- **Centroid**: Geometric center of a contour or bounding box
- **Deadband**: Range of values where no action is taken to prevent oscillation
- **IoU**: Intersection over Union, metric for object overlap
- **Interpolation**: Method for estimating pixel values when resizing images
- **Telemetry**: Automated recording of data for analysis

---

## END OF SPECIFICATIONS

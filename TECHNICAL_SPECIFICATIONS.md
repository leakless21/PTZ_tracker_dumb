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

### 4.1 Library Selection

#### 4.1.1 OpenCV Built-in Algorithms (Primary)
OpenCV provides efficient built-in background subtraction classes:
- **cv2.createBackgroundSubtractorMOG2()**: Gaussian Mixture Model (recommended)
- **cv2.createBackgroundSubtractorKNN()**: K-Nearest Neighbors based
- **cv2.bgsegm.createBackgroundSubtractorMOG()**: Original MOG (legacy)
- **cv2.bgsegm.createBackgroundSubtractorGMG()**: Geometric multigrid
- **cv2.bgsegm.createBackgroundSubtractorCNT()**: Counting-based

#### 4.1.2 BGSLibrary (Advanced Option)
Install via: `pip install pybgs`

BGSLibrary provides 43+ advanced algorithms including:
- **FrameDifference**: Simple frame differencing baseline
- **StaticFrameDifference**: For static camera scenarios
- **AdaptiveBackgroundLearning**: Adaptive learning rate
- **CodeBook**: Codebook-based algorithm
- **KDE**: Kernel Density Estimation
- **MixtureOfGaussianV2**: Enhanced MOG2 variant
- **PAWCS**: Pixel-based Adaptive Word Consensus Segmenter
- **SigmaDelta**: Sigma-delta filter based
- **ViBe**: Visual Background Extractor
- **SuBSENSE**: Self-Balanced Sensitivity Segmenter (state-of-the-art)
- **LBSP**: Local Binary Similarity Pattern
- **MultiLayer**: Multi-layer background subtraction
- **T2FGMM_UM**: Type-2 Fuzzy Gaussian Mixture Model
- **T2FGMM_UV**: Type-2 Fuzzy with UV adaptation

### 4.2 Algorithm Selection Strategy
- **Default**: OpenCV's MOG2 (best balance of speed and accuracy)
- **High Accuracy**: bgslibrary's SuBSENSE or PAWCS
- **High Speed**: FrameDifference or simple MOG
- **Outdoor/Dynamic Lighting**: PAWCS, SigmaDelta
- **Indoor/Static Lighting**: MOG2, KNN
- **Minimal Resources**: FrameDifference, StaticFrameDifference

### 4.3 OpenCV MOG2 Implementation Details

#### 4.3.1 Initialization
Function: `cv2.createBackgroundSubtractorMOG2(history, varThreshold, detectShadows)`

#### 4.3.2 Parameters
- **history**: Number of last frames affecting background model (default: 500)
- **detectShadows**: Enable/disable shadow detection (default: true)
- **shadowValue**: Value used to mark shadows in output (default: 127)
- **shadowThreshold**: Shadow detection threshold (default: 0.5)
- **learningRate**: Background model update rate (default: -1 for automatic)
  - Range: -1 (automatic), 0.0 (no learning) to 1.0 (complete replacement)
  - Recommended: 0.001 to 0.01 for slow learning, 0.05 to 0.1 for fast adaptation
- **varThreshold**: Threshold on squared Mahalanobis distance (default: 16)
  - Lower values: More sensitive (more foreground pixels)
  - Higher values: Less sensitive (fewer foreground pixels)
  - Recommended range: 9 to 25

#### 4.3.3 Application Method
Function: `foreground_mask = background_subtractor.apply(frame, learningRate)`
- **Input**: Current frame (BGR or grayscale)
- **Output**: Foreground mask (grayscale image)
  - 0 (black): Background
  - 255 (white): Foreground
  - 127 (gray): Shadow (if detectShadows=True)
- **learningRate parameter**: Can be overridden per frame

#### 4.3.4 Additional Methods
- `getBackgroundImage()`: Returns current background model image
- `setHistory(frames)`: Update history parameter
- `setVarThreshold(threshold)`: Update variance threshold
- `setDetectShadows(boolean)`: Enable/disable shadow detection
- `setShadowValue(value)`: Set shadow pixel value
- `setShadowThreshold(threshold)`: Set shadow detection sensitivity

### 4.4 OpenCV KNN Implementation Details

#### 4.4.1 Initialization
Function: `cv2.createBackgroundSubtractorKNN(history, dist2Threshold, detectShadows)`

#### 4.4.2 Parameters
- **history**: Number of frames for background model (default: 500)
- **dist2Threshold**: Squared Euclidean distance threshold (default: 400.0)
  - Lower values: More foreground pixels
  - Higher values: Fewer foreground pixels
  - Recommended range: 200 to 800
- **detectShadows**: Enable shadow detection (default: True)

#### 4.4.3 Application Method
Same as MOG2: `foreground_mask = background_subtractor.apply(frame, learningRate)`

#### 4.4.4 KNN vs MOG2 Comparison
- **KNN Advantages**: Better for scenes with rapid changes, less memory
- **MOG2 Advantages**: Better shadow detection, more stable in static scenes
- **Performance**: KNN typically faster than MOG2

### 4.5 BGSLibrary Implementation Details

#### 4.5.1 Installation and Import
Installation: `pip install pybgs`
Import: `import pybgs`

#### 4.5.2 Initialization
General pattern: `algorithm = pybgs.AlgorithmName()`

Examples:
- `bgs = pybgs.FrameDifference()`
- `bgs = pybgs.MixtureOfGaussianV2()`
- `bgs = pybgs.SuBSENSE()`
- `bgs = pybgs.PAWCS()`
- `bgs = pybgs.ViBe()`
- `bgs = pybgs.SigmaDelta()`

#### 4.5.3 Application Method
Function: `foreground_mask = algorithm.apply(frame)`
- **Input**: Frame as NumPy array (BGR format)
- **Output**: Binary foreground mask (0=background, 255=foreground)

#### 4.5.4 Available Algorithms List
**Simple Algorithms:**
- FrameDifference
- StaticFrameDifference
- WeightedMovingMean
- WeightedMovingVariance

**Statistical Algorithms:**
- AdaptiveBackgroundLearning
- AdaptiveSelectiveBackgroundLearning
- MixtureOfGaussianV1
- MixtureOfGaussianV2
- KNN

**Advanced Algorithms:**
- CodeBook
- ViBe (Visual Background Extractor)
- PAWCS (Pixel-based Adaptive Word Consensus Segmenter)
- SuBSENSE (Self-Balanced Sensitivity Segmenter)
- LOBSTER (LOcal Binary SimiLarity segmenTER)
- SigmaDelta
- T2FGMM (Type-2 Fuzzy Gaussian Mixture Model)

**Special Purpose:**
- MultiLayer (for multi-layer scenes)
- KDE (Kernel Density Estimation)
- LBP_MRF (Local Binary Pattern with Markov Random Field)
- FuzzySugenoIntegral
- FuzzyChoquetIntegral

#### 4.5.5 Algorithm Selection Guidelines
- **FrameDifference**: Fastest, minimal memory, good for testing
- **ViBe**: Good balance of speed and accuracy
- **SuBSENSE**: Best accuracy, slower, good for challenging scenes
- **PAWCS**: Excellent for dynamic backgrounds
- **SigmaDelta**: Good for camouflage and gradual changes

### 4.6 Background Model Management
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

### 4.7 Foreground Mask Post-Processing (OpenCV)

This is the recommended pipeline for cleaning the foreground mask:

**Complete mask cleanup workflow:**

```
Sequence:
1. Erosion - Remove small noise
2. Dilation - Restore object size
3. Gaussian Blur - Smooth edges and reduce noise
4. Morphological Closing - Fill remaining holes
5. Binary Threshold - Final cleanup
```

**Detailed Steps:**

1. **Create Kernel**
   ```
   kernel = np.ones((5, 5), np.uint8)
   # Creates a 5×5 rectangular kernel of ones
   ```

2. **Erosion** (Remove small noise pixels)
   ```
   Function: cv2.erode(mask, kernel, iterations)
   fgMask = cv2.erode(fgMask, kernel, iterations=1)
   ```
   - Removes small white noise
   - Shrinks foreground objects slightly

3. **Dilation** (Restore object size)
   ```
   Function: cv2.dilate(mask, kernel, iterations)
   fgMask = cv2.dilate(fgMask, kernel, iterations=1)
   ```
   - Restores object size after erosion
   - Connects nearby components
   - Together with erosion forms an "opening" operation

4. **Gaussian Blur** (Smooth and reduce noise)
   ```
   Function: cv2.GaussianBlur(mask, ksize, sigmaX)
   fgMask = cv2.GaussianBlur(fgMask, (3, 3), 0)
   ```
   - **ksize**: Kernel size (3, 3) - must be odd
   - **sigmaX**: Standard deviation (0 = auto-calculate from kernel size)
   - **Purpose**: Smooths edges, reduces high-frequency noise
   - **Effect**: Softens mask before final operations

5. **Morphological Closing** (Fill holes)
   ```
   Function: cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
   fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernel)
   ```
   - Fills small holes inside objects
   - Connects broken parts of objects
   - Dilation followed by erosion

6. **Binary Threshold** (Final cleanup)
   ```
   Function: cv2.threshold(mask, threshold, maxval, type)
   _, fgMask = cv2.threshold(fgMask, 130, 255, cv2.THRESH_BINARY)
   ```
   - **threshold**: 130 (higher than default 127)
   - **maxval**: 255 (white)
   - **type**: cv2.THRESH_BINARY
   - **Purpose**: Convert to pure binary after blur
   - **Returns**: (threshold_value, binary_mask) - use _ to discard first value

**Complete Function Chain:**
```
kernel = np.ones((5, 5), np.uint8)
fgMask = cv2.erode(fgMask, kernel, iterations=1)
fgMask = cv2.dilate(fgMask, kernel, iterations=1)
fgMask = cv2.GaussianBlur(fgMask, (3, 3), 0)
fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernel)
_, fgMask = cv2.threshold(fgMask, 130, 255, cv2.THRESH_BINARY)
```

**Kernel Size Configuration:**
- **Small kernel** (3×3): For fine details and small objects
- **Default kernel** (5×5): Good general-purpose size (recommended)
- **Large kernel** (7×7 or 9×9): For aggressive filtering, merge nearby objects

**Gaussian Blur Kernel:**
- Use small odd values: (3, 3) or (5, 5)
- Larger values create more smoothing

**Threshold Value:**
- Default: 130 (stricter than standard 127)
- Lower values (100-127): More permissive, more foreground pixels
- Higher values (130-150): Stricter, fewer false positives

**Advantages of this pipeline:**
- Gaussian blur adds noise reduction beyond morphological operations
- Sequence is optimized: removes noise first, then smooths, then finalizes
- Threshold value of 130 is stricter, reducing false positives
- Works well with both OpenCV and bgslibrary outputs
- Simple and effective for most scenarios

---

## 5. OBJECT DETECTION MODULE

### 5.1 Contour Detection (OpenCV)

#### 5.1.1 Find Contours
Function: `cv2.findContours(binary_mask, mode, method)`
- **binary_mask**: Binary foreground mask (8-bit single-channel)
- **mode**: `cv2.RETR_EXTERNAL` (only external contours)
  - Alternative: `cv2.RETR_LIST` (all contours without hierarchy)
- **method**: `cv2.CHAIN_APPROX_SIMPLE` (compress contours)
  - Alternative: `cv2.CHAIN_APPROX_NONE` (all points, higher memory)
- **Returns**: Tuple (contours, hierarchy)
  - **contours**: List of contour arrays
  - **hierarchy**: Hierarchy information (can be ignored)

**Usage Note**: OpenCV 4.x returns (contours, hierarchy), OpenCV 3.x returns (image, contours, hierarchy)

### 5.2 Contour Filtering (OpenCV)

#### 5.2.1 Area Filtering
Function: `cv2.contourArea(contour)`
- **Returns**: Area in square pixels
- **Filtering Logic**:
  - Minimum area: `area >= min_area` (default: 500 pixels)
  - Maximum area: `area <= frame_width * frame_height * max_fraction` (default: 0.3)
- **Purpose**: Remove tiny noise and overly large regions

#### 5.2.2 Aspect Ratio Filtering
Steps:
1. Get bounding box: `x, y, w, h = cv2.boundingRect(contour)`
2. Calculate aspect ratio: `aspect_ratio = w / h`
3. Filter: `min_ratio <= aspect_ratio <= max_ratio` (default: 0.2 to 5.0)
- **Purpose**: Remove elongated objects (likely noise or shadows)

#### 5.2.3 Solidity Filtering
Steps:
1. Get contour area: `area = cv2.contourArea(contour)`
2. Get convex hull: `hull = cv2.convexHull(contour)`
3. Get hull area: `hull_area = cv2.contourArea(hull)`
4. Calculate solidity: `solidity = area / hull_area`
5. Filter: `solidity >= min_solidity` (default: 0.3)
- **Purpose**: Remove irregular shapes (likely fragmented detections)

#### 5.2.4 Extent Filtering
Steps:
1. Get contour area: `area = cv2.contourArea(contour)`
2. Get bounding rect: `x, y, w, h = cv2.boundingRect(contour)`
3. Calculate extent: `extent = area / (w * h)`
4. Filter: `extent >= min_extent` (default: 0.2)
- **Purpose**: Remove sparse or scattered detections

### 5.3 Bounding Box Calculation (OpenCV)

#### 5.3.1 Axis-Aligned Bounding Rectangle
Function: `cv2.boundingRect(contour)`
- **Returns**: (x, y, width, height)
- **Coordinate**: (x, y) is top-left corner
- **Usage**: Fast, simple, good for most cases

#### 5.3.2 Minimum Area Rectangle (Rotated)
Function: `cv2.minAreaRect(contour)`
- **Returns**: ((center_x, center_y), (width, height), angle)
- **Usage**: For rotated objects
- **Box Points**: `cv2.boxPoints(rect)` to get 4 corner points

#### 5.3.3 Bounding Box Padding
Add padding to bounding box:
- `x_padded = max(0, x - padding)`
- `y_padded = max(0, y - padding)`
- `w_padded = min(frame_width - x_padded, w + 2*padding)`
- `h_padded = min(frame_height - y_padded, h + 2*padding)`

### 5.4 Object Properties Extraction (OpenCV)

#### 5.4.1 Centroid Calculation
Method 1 (from bounding box):
- `centroid_x = x + w / 2`
- `centroid_y = y + h / 2`

Method 2 (from contour moments):
Function: `cv2.moments(contour)`
- **Returns**: Dictionary of moment values
- **Centroid calculation**:
  - `M = cv2.moments(contour)`
  - `centroid_x = M['m10'] / M['m00']` (if M['m00'] != 0)
  - `centroid_y = M['m01'] / M['m00']` (if M['m00'] != 0)
- **Advantage**: More accurate, weighted by pixel distribution

#### 5.4.2 Area and Perimeter
- **Area**: `cv2.contourArea(contour)` - returns float
- **Perimeter**: `cv2.arcLength(contour, closed=True)` - returns float
- **closed**: True for closed contours

#### 5.4.3 Complete Property Set
For each detected object, extract:
- **Centroid**: (cx, cy) - from moments or bounding box
- **Area**: Square pixels - from contourArea()
- **Perimeter**: Pixels - from arcLength()
- **Bounding Box**: (x, y, w, h) - from boundingRect()
- **Aspect Ratio**: w / h
- **Solidity**: area / hull_area
- **Extent**: area / (w × h)
- **Convex Hull**: cv2.convexHull(contour) - for advanced processing

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

### 8.1 Virtual Camera Transformation (OpenCV)

#### 8.1.1 ROI Extraction
Direct array slicing (NumPy):
- `roi = frame[y:y+h, x:x+w]`
- Where (x, y, w, h) is the ROI rectangle
- **Bounds checking**: Ensure x, y, w, h are within frame dimensions

Alternative using OpenCV:
- `roi = frame[y1:y2, x1:x2].copy()`

#### 8.1.2 ROI Resizing
Function: `cv2.resize(roi, dsize, interpolation)`
- **roi**: Extracted region of interest
- **dsize**: Output size as (width, height) tuple
- **interpolation**: Resampling method
  - `cv2.INTER_LINEAR`: Bilinear (default, fast, good quality)
  - `cv2.INTER_CUBIC`: Bicubic (slower, better quality for zooming in)
  - `cv2.INTER_LANCZOS4`: Lanczos (slowest, best quality)
  - `cv2.INTER_NEAREST`: Nearest neighbor (fastest, lowest quality)
  - `cv2.INTER_AREA`: Pixel area relation (best for shrinking)
- **Returns**: Resized image
- **Recommended**:
  - Zooming in (enlarging): `INTER_CUBIC` or `INTER_LANCZOS4`
  - Zooming out (shrinking): `INTER_AREA`

#### 8.1.3 Complete Transform Pipeline
Process:
1. Calculate ROI from PTZ parameters (pan, tilt, zoom)
2. Clamp ROI to frame boundaries
3. Extract ROI: `roi = frame[y:y+h, x:x+w]`
4. Resize to output dimensions: `output = cv2.resize(roi, (out_w, out_h), interpolation)`
5. Return transformed frame

### 8.2 Overlay Rendering (OpenCV)

#### 8.2.1 Bounding Box Overlay
Function: `cv2.rectangle(image, pt1, pt2, color, thickness)`
- **image**: Frame to draw on (modified in-place)
- **pt1**: Top-left corner (x, y)
- **pt2**: Bottom-right corner (x+w, y+h)
- **color**: BGR tuple
  - Green: (0, 255, 0) - active tracking
  - Yellow: (0, 255, 255) - acquiring
  - Red: (0, 0, 255) - lost
  - Cyan: (255, 255, 0) - recovering
- **thickness**: 2 or 3 pixels (use -1 for filled rectangle)

Alternative with corners:
- `pt2 = (x+w, y+h)` where (x, y, w, h) is bounding box

#### 8.2.2 Crosshair Overlay
**Method 1**: Draw lines
Function: `cv2.line(image, pt1, pt2, color, thickness)`
```
Center crosshair at (cx, cy) with size s:
- Horizontal: cv2.line(image, (cx-s, cy), (cx+s, cy), (255, 255, 255), 2)
- Vertical: cv2.line(image, (cx, cy-s), (cx, cy+s), (255, 255, 255), 2)
```

**Method 2**: Draw circle
Function: `cv2.circle(image, center, radius, color, thickness)`
- **center**: (cx, cy) frame center
- **radius**: 10-20 pixels
- **color**: (255, 255, 255) white or (255, 255, 0) cyan
- **thickness**: 2 (outline) or -1 (filled)

#### 8.2.3 Information Overlay (Text)
Function: `cv2.putText(image, text, org, fontFace, fontScale, color, thickness, lineType)`
- **image**: Frame to draw on
- **text**: String to display
- **org**: Bottom-left corner of text (x, y)
- **fontFace**: Font type
  - `cv2.FONT_HERSHEY_SIMPLEX`: Clean, readable
  - `cv2.FONT_HERSHEY_COMPLEX`: More formal
  - `cv2.FONT_HERSHEY_DUPLEX`: Monospace-like
- **fontScale**: Size multiplier (0.5 to 2.0)
- **color**: BGR tuple (255, 255, 255) for white
- **thickness**: 1 or 2
- **lineType**: `cv2.LINE_AA` for anti-aliased

**Text Background** (for readability):
Draw filled rectangle behind text:
1. Calculate text size: `(w, h), baseline = cv2.getTextSize(text, font, scale, thickness)`
2. Draw rectangle: `cv2.rectangle(image, (x, y-h-5), (x+w, y+5), (0,0,0), -1)`
3. Draw text: `cv2.putText(image, text, (x, y), ...)`

**Multi-line text example**:
```
Position text at (10, 30) with 25-pixel line spacing:
cv2.putText(image, "State: TRACKING", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
cv2.putText(image, "Pan: 15.2°", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
cv2.putText(image, "Tilt: -3.5°", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
cv2.putText(image, "Zoom: 2.5x", (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
```

#### 8.2.4 Trajectory Trail Overlay
Function: `cv2.polylines(image, pts, isClosed, color, thickness)`
- **pts**: List of trajectory points [(x1,y1), (x2,y2), ...]
- **isClosed**: False for trajectory trail
- **color**: (0, 255, 0) green
- **thickness**: 2

Alternative (draw circles at each point):
```
For each point in trajectory:
  cv2.circle(image, point, 3, (0, 255, 0), -1)
```

#### 8.2.5 Velocity Vector Overlay
Draw arrow from current position in movement direction:
Function: `cv2.arrowedLine(image, pt1, pt2, color, thickness, tipLength)`
- **pt1**: Current object centroid
- **pt2**: Predicted next position (centroid + velocity_vector)
- **color**: (255, 0, 0) blue
- **thickness**: 2
- **tipLength**: 0.3 (arrow tip size ratio)

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

## 9. VIDEO INPUT/OUTPUT MODULE (OpenCV)

### 9.1 Video Input Implementation

#### 9.1.1 Video Capture Initialization
Function: `cv2.VideoCapture(filename)`
- **filename**: Path to video file
- **Returns**: VideoCapture object
- **Check success**: `if cap.isOpened():`

Example:
```
cap = cv2.VideoCapture("input_video.mp4")
if not cap.isOpened():
    # Handle error
```

#### 9.1.2 Get Video Properties
Functions:
- `cap.get(cv2.CAP_PROP_FRAME_WIDTH)` - frame width
- `cap.get(cv2.CAP_PROP_FRAME_HEIGHT)` - frame height
- `cap.get(cv2.CAP_PROP_FPS)` - frames per second
- `cap.get(cv2.CAP_PROP_FRAME_COUNT)` - total frame count
- `cap.get(cv2.CAP_PROP_FOURCC)` - codec fourcc code
- `cap.get(cv2.CAP_PROP_POS_FRAMES)` - current frame position

Convert to integers where needed:
```
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
```

#### 9.1.3 Frame Reading
Function: `cap.read()`
- **Returns**: Tuple (ret, frame)
  - **ret**: Boolean (True if frame read successfully)
  - **frame**: NumPy array (BGR format) or None
- **Usage pattern**:
  ```
  ret, frame = cap.read()
  if not ret:
      # End of video or error
      break
  # Process frame
  ```

#### 9.1.4 Advanced Input Operations
- **Seek to frame**: `cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)`
- **Seek to millisecond**: `cap.set(cv2.CAP_PROP_POS_MSEC, milliseconds)`
- **Release**: `cap.release()` - close video file

#### 9.1.5 Loop Playback Implementation
```
When ret == False (end of video):
  cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
  continue
```

#### 9.1.6 Frame Skipping
Process every Nth frame:
```
frame_counter = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    if frame_counter % N == 0:
        # Process this frame
    frame_counter += 1
```

### 9.2 Video Output Implementation

#### 9.2.1 Video Writer Initialization
Function: `cv2.VideoWriter(filename, fourcc, fps, frameSize, isColor)`
- **filename**: Output file path
- **fourcc**: Codec code from `cv2.VideoWriter_fourcc()`
- **fps**: Frame rate (match input or custom)
- **frameSize**: (width, height) tuple
- **isColor**: True for color, False for grayscale
- **Returns**: VideoWriter object

#### 9.2.2 Codec Selection (FourCC)
Function: `cv2.VideoWriter_fourcc(*'XXXX')`

Common codecs:
- `cv2.VideoWriter_fourcc(*'mp4v')` - MPEG-4 (.mp4)
- `cv2.VideoWriter_fourcc(*'X264')` - H.264 (.mp4)
- `cv2.VideoWriter_fourcc(*'XVID')` - XVID (.avi)
- `cv2.VideoWriter_fourcc(*'MJPG')` - Motion JPEG (.avi)
- `cv2.VideoWriter_fourcc(*'H264')` - H.264 variant

**Recommended**:
- **Best quality**: 'X264' or 'H264' for .mp4
- **Best compatibility**: 'MJPG' for .avi
- **Fast writing**: 'MJPG'

Example:
```
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (640, 480))
```

#### 9.2.3 Writing Frames
Function: `out.write(frame)`
- **frame**: NumPy array (must match frameSize and isColor)
- **No return value**
- Frame must be correct size and type (BGR if isColor=True)

Example:
```
for each frame:
    # Process frame
    out.write(processed_frame)
```

#### 9.2.4 Finalizing Output
Function: `out.release()`
- Finalizes and closes video file
- **Must call** before program ends or file may be corrupted

#### 9.2.5 Complete Video I/O Pattern
```
# Input
cap = cv2.VideoCapture("input.mp4")
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

# Process loop
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Process frame
    processed = process_frame(frame)
    out.write(processed)

# Cleanup
cap.release()
out.release()
```

### 9.3 Display Window (OpenCV)

#### 9.3.1 Create Window
Function: `cv2.namedWindow(window_name, flags)`
- **window_name**: String identifier
- **flags**:
  - `cv2.WINDOW_NORMAL` - resizable
  - `cv2.WINDOW_AUTOSIZE` - fixed size (default)
  - `cv2.WINDOW_FULLSCREEN` - fullscreen mode

#### 9.3.2 Display Frame
Function: `cv2.imshow(window_name, frame)`
- **window_name**: Window identifier (creates if doesn't exist)
- **frame**: Image to display (NumPy array)

#### 9.3.3 Wait for Key Input
Function: `cv2.waitKey(delay)`
- **delay**: Milliseconds to wait (1 for ~real-time, 0 for infinite)
- **Returns**: ASCII code of key pressed, or -1 if no key
- **Usage**: Must call to update window display

Example:
```
key = cv2.waitKey(1) & 0xFF
if key == ord('q'):
    break  # Quit
elif key == ord(' '):
    paused = not paused  # Toggle pause
```

#### 9.3.4 Close Windows
- `cv2.destroyWindow(window_name)` - close specific window
- `cv2.destroyAllWindows()` - close all OpenCV windows

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
  library: "opencv" | "bgslibrary"  # Choose library
  algorithm: string  # Algorithm name depends on library

  # OpenCV algorithms: "MOG2", "KNN", "MOG", "GMG", "CNT"
  # BGSLibrary algorithms: "FrameDifference", "ViBe", "SuBSENSE", "PAWCS", "SigmaDelta", etc.

  # OpenCV-specific parameters (used when library="opencv")
  history: integer (100-1000, default: 500)
  var_threshold: float (4.0-50.0, default: 16.0)  # For MOG2
  dist2_threshold: float (200-800, default: 400.0)  # For KNN
  detect_shadows: boolean (default: true)
  learning_rate: float (-1.0 for auto, 0.0001-0.1, default: -1)

  # BGSLibrary note: Uses internal defaults, no configuration needed
```

#### Mask Post-Processing Section
```
mask_processing:
  enable: boolean (default: true)
  kernel_size: integer (3, 5, 7, 9, default: 5)
  erosion_iterations: integer (default: 1)
  dilation_iterations: integer (default: 1)
  gaussian_blur_kernel: integer (3, 5, 7, must be odd, default: 3)
  threshold_value: integer (100-150, default: 130)
```

Pipeline sequence: Erosion → Dilation → Gaussian Blur → Closing → Threshold

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

## 19. OPENCV/BGSLIBRARY IMPLEMENTATION WORKFLOW

### 19.1 Complete Pipeline with OpenCV

This section provides the detailed step-by-step workflow for implementing the system using OpenCV.

#### 19.1.1 Initialization Phase

**Step 1: Import Libraries**
```
Required imports:
- import cv2
- import numpy as np
- import time (for FPS calculation)
- import json or yaml (for configuration)
- Optional: import pybgs (if using bgslibrary)
```

**Step 2: Load Configuration**
- Read JSON/YAML config file
- Set default values for missing parameters
- Validate parameter ranges

**Step 3: Initialize Video Capture**
```
cap = cv2.VideoCapture(input_video_path)
Verify: if not cap.isOpened(): handle error
Get properties:
  - frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  - frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  - fps = cap.get(cv2.CAP_PROP_FPS)
  - total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
```

**Step 4: Initialize Background Subtractor**

*Option A: OpenCV MOG2*
```
bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=16,
    detectShadows=True
)
```

*Option B: OpenCV KNN*
```
bg_subtractor = cv2.createBackgroundSubtractorKNN(
    history=500,
    dist2Threshold=400.0,
    detectShadows=True
)
```

*Option C: BGSLibrary*
```
import pybgs
bg_subtractor = pybgs.FrameDifference()
# or pybgs.SuBSENSE(), pybgs.PAWCS(), etc.
```

**Step 5: Initialize PTZ State**
```
ptz_state = {
    'pan': 0.0,
    'tilt': 0.0,
    'zoom': 1.0
}
```

**Step 6: Initialize Tracking State**
```
tracking_state = {
    'status': 'IDLE',  # IDLE, ACQUIRING, TRACKING, LOST, RECOVERING
    'target': None,
    'frames_tracked': 0,
    'frames_lost': 0,
    'confidence': 0.0
}
```

**Step 7: Initialize Video Writer** (if saving output)
```
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(
    output_video_path,
    fourcc,
    fps,
    (frame_width, frame_height)
)
```

**Step 8: Create Display Window**
```
cv2.namedWindow('PTZ Tracking', cv2.WINDOW_NORMAL)
```

#### 19.1.2 Main Processing Loop

**Frame-by-frame processing structure:**

```
frame_count = 0

while True:
    # STEP 1: Capture Frame
    ret, frame = cap.read()
    if not ret:
        if loop_playback:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        else:
            break

    frame_count += 1
    original_frame = frame.copy()  # Keep original for display

    # STEP 2: Background Subtraction
    # For OpenCV:
    fg_mask = bg_subtractor.apply(frame, learningRate=0.01)
    # For bgslibrary:
    # fg_mask = bg_subtractor.apply(frame)

    # STEP 3: Post-process Foreground Mask
    # Recommended cleanup sequence:
    kernel = np.ones((5, 5), np.uint8)
    fg_mask = cv2.erode(fg_mask, kernel, iterations=1)
    fg_mask = cv2.dilate(fg_mask, kernel, iterations=1)
    fg_mask = cv2.GaussianBlur(fg_mask, (3, 3), 0)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    _, fg_mask = cv2.threshold(fg_mask, 130, 255, cv2.THRESH_BINARY)

    # STEP 4: Find Contours
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # STEP 5: Filter and Select Object
    valid_objects = []

    for contour in contours:
        # Area filtering
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area_fraction * frame_width * frame_height:
            continue

        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)

        # Aspect ratio filtering
        aspect_ratio = w / h if h > 0 else 0
        if aspect_ratio < min_aspect or aspect_ratio > max_aspect:
            continue

        # Solidity filtering
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        if solidity < min_solidity:
            continue

        # Calculate centroid
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        else:
            cx = x + w // 2
            cy = y + h // 2

        # Store valid object
        valid_objects.append({
            'contour': contour,
            'bbox': (x, y, w, h),
            'centroid': (cx, cy),
            'area': area
        })

    # Select best object (largest area strategy)
    target_object = None
    if valid_objects:
        target_object = max(valid_objects, key=lambda obj: obj['area'])

    # STEP 6: Update Tracking State
    if target_object is not None:
        if tracking_state['status'] == 'IDLE':
            tracking_state['status'] = 'ACQUIRING'
            tracking_state['frames_tracked'] = 1
        elif tracking_state['status'] == 'ACQUIRING':
            tracking_state['frames_tracked'] += 1
            if tracking_state['frames_tracked'] >= acquisition_threshold:
                tracking_state['status'] = 'TRACKING'
        elif tracking_state['status'] in ['TRACKING', 'LOST']:
            tracking_state['status'] = 'TRACKING'
            tracking_state['frames_lost'] = 0

        tracking_state['target'] = target_object
        tracking_state['confidence'] = min(1.0, tracking_state['frames_tracked'] / 30.0)
    else:
        if tracking_state['status'] == 'TRACKING':
            tracking_state['frames_lost'] += 1
            if tracking_state['frames_lost'] >= lost_threshold:
                tracking_state['status'] = 'LOST'
        elif tracking_state['status'] in ['ACQUIRING', 'LOST']:
            tracking_state['status'] = 'IDLE'

    # STEP 7: Calculate PTZ Adjustments
    if tracking_state['status'] == 'TRACKING' and target_object:
        cx, cy = target_object['centroid']

        # Calculate error from center
        error_x = (cx - frame_width / 2) / frame_width
        error_y = (cy - frame_height / 2) / frame_height

        # Apply deadband
        if abs(error_x) > deadband_x:
            ptz_state['pan'] += error_x * pan_sensitivity
        if abs(error_y) > deadband_y:
            ptz_state['tilt'] += error_y * tilt_sensitivity

        # Zoom control based on object size
        x, y, w, h = target_object['bbox']
        object_size = max(w, h)
        desired_size = frame_width * 0.3  # Target: object fills 30% of frame

        if object_size < desired_size * 0.8:
            # Object too small, zoom in
            ptz_state['zoom'] += zoom_speed
        elif object_size > desired_size * 1.2:
            # Object too large, zoom out
            ptz_state['zoom'] -= zoom_speed

        # Clamp zoom
        ptz_state['zoom'] = np.clip(ptz_state['zoom'], min_zoom, max_zoom)

        # Apply smoothing (exponential moving average)
        # ptz_state values would be smoothed here

    # STEP 8: Calculate ROI from PTZ State
    zoom = ptz_state['zoom']
    roi_w = int(frame_width / zoom)
    roi_h = int(frame_height / zoom)

    # Center based on pan/tilt (simplified - pan/tilt affect center position)
    # For simulation, pan/tilt can shift the ROI center
    center_x = frame_width // 2 + int(ptz_state['pan'])
    center_y = frame_height // 2 + int(ptz_state['tilt'])

    roi_x = center_x - roi_w // 2
    roi_y = center_y - roi_h // 2

    # Clamp ROI to frame boundaries
    roi_x = max(0, min(roi_x, frame_width - roi_w))
    roi_y = max(0, min(roi_y, frame_height - roi_h))

    # STEP 9: Extract and Resize ROI (Virtual PTZ)
    roi = original_frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
    ptz_view = cv2.resize(roi, (frame_width, frame_height), interpolation=cv2.INTER_LINEAR)

    # STEP 10: Draw Overlays
    display_frame = ptz_view.copy()

    # Draw bounding box (if tracking)
    if tracking_state['status'] == 'TRACKING' and target_object:
        # Transform bbox coordinates to ROI space
        x, y, w, h = target_object['bbox']
        # Adjust coordinates relative to ROI
        x_roi = int((x - roi_x) * frame_width / roi_w)
        y_roi = int((y - roi_y) * frame_height / roi_h)
        w_roi = int(w * frame_width / roi_w)
        h_roi = int(h * frame_height / roi_h)

        # Color based on state
        color = (0, 255, 0)  # Green for tracking
        cv2.rectangle(display_frame, (x_roi, y_roi), (x_roi+w_roi, y_roi+h_roi), color, 2)

    # Draw crosshair at center
    center = (frame_width // 2, frame_height // 2)
    cv2.line(display_frame, (center[0]-20, center[1]), (center[0]+20, center[1]), (255,255,255), 2)
    cv2.line(display_frame, (center[0], center[1]-20), (center[0], center[1]+20), (255,255,255), 2)

    # Draw info text
    cv2.putText(display_frame, f"State: {tracking_state['status']}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    cv2.putText(display_frame, f"Pan: {ptz_state['pan']:.1f}", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    cv2.putText(display_frame, f"Tilt: {ptz_state['tilt']:.1f}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    cv2.putText(display_frame, f"Zoom: {ptz_state['zoom']:.2f}x", (10, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    # STEP 11: Display and Save
    cv2.imshow('PTZ Tracking', display_frame)

    if save_output:
        out.write(display_frame)

    # STEP 12: Handle User Input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        cv2.waitKey(0)  # Pause
    elif key == ord('r'):
        # Reset tracking and PTZ
        tracking_state['status'] = 'IDLE'
        ptz_state = {'pan': 0.0, 'tilt': 0.0, 'zoom': 1.0}
```

#### 19.1.3 Cleanup Phase

```
# Release resources
cap.release()
if save_output:
    out.release()
cv2.destroyAllWindows()

# Save telemetry/logs
# Export statistics
```

### 19.2 BGSLibrary-Specific Implementation Notes

When using bgslibrary instead of OpenCV's built-in subtractors:

**Differences:**
1. **Import**: `import pybgs`
2. **Initialization**: `bg_subtractor = pybgs.AlgorithmName()`
3. **Application**: `fg_mask = bg_subtractor.apply(frame)` (no learningRate parameter)
4. **Output**: Returns binary mask directly (0 and 255)

**Recommended Algorithms by Use Case:**

- **Default/Testing**: `pybgs.FrameDifference()`
- **Good Balance**: `pybgs.ViBe()`
- **Best Accuracy**: `pybgs.SuBSENSE()`
- **Dynamic Scenes**: `pybgs.PAWCS()`
- **Static Scenes**: `pybgs.MixtureOfGaussianV2()`

**No Additional Configuration Required**: BGSLibrary algorithms use internal default parameters.

### 19.3 Key Implementation Tips

1. **Frame Copy**: Always copy original frame before processing for overlay rendering
2. **ROI Bounds**: Always clamp ROI coordinates to prevent out-of-bounds errors
3. **Division by Zero**: Check denominators before division (moments, areas)
4. **Coordinate Transformation**: When zoomed, transform object coordinates to display space
5. **Color Space**: OpenCV uses BGR by default, not RGB
6. **Mask Type**: Ensure masks are 8-bit single-channel (CV_8UC1) for findContours
7. **Wait Key**: Must call cv2.waitKey() for imshow() to work
8. **Resource Cleanup**: Always release VideoCapture and VideoWriter

### 19.4 Performance Optimization

1. **Resize Input**: Process smaller frames for speed: `frame = cv2.resize(frame, (width//2, height//2))`
2. **Skip Frames**: Process every Nth frame for near-real-time on slow hardware
3. **Reduce Morphology**: Use smaller kernels and fewer iterations
4. **Simplify Filtering**: Remove extent/solidity checks if not needed
5. **Algorithm Choice**: Use faster algorithms (FrameDifference, MOG) over slower (SuBSENSE)
6. **Disable Shadows**: Set detectShadows=False to skip shadow processing
7. **Reduce History**: Lower history parameter (e.g., 100-200) for faster adaptation

---

## 20. DEVELOPMENT GUIDELINES

### 20.1 Programming Language
- **Primary**: Python 3.8+
- **Rationale**: Rich ecosystem for computer vision (OpenCV, NumPy)

### 20.2 Environment Management with Pixi

#### 20.2.1 Why Pixi?
- **Modern package manager**: Fast, reproducible environments
- **Cross-platform**: Works on Windows, Linux, macOS
- **Lock files**: Ensures reproducible builds
- **Conda + PyPI**: Access to both conda-forge and PyPI packages
- **Task runner**: Built-in task execution
- **No separate venv**: Automatic environment isolation

#### 20.2.2 Pixi Installation
Visit: https://pixi.sh or install via:
```bash
# Linux/macOS
curl -fsSL https://pixi.sh/install.sh | bash

# Windows (PowerShell)
iwr -useb https://pixi.sh/install.ps1 | iex

# Verify installation
pixi --version
```

#### 20.2.3 Project Dependencies

**Required Libraries:**
- **opencv**: Computer vision library (from conda-forge)
  - Provides: cv2 module, video I/O, image processing
  - Version: 4.5+
- **numpy**: Numerical computing
  - Version: 1.19+
  - Used for: Array operations, numerical computations
- **python**: Python interpreter
  - Version: 3.8-3.11 recommended

**Optional Libraries:**
- **pybgs**: BGSLibrary Python bindings (from PyPI)
  - Installation: Via pip in Pixi environment
  - Use when: Need advanced background subtraction algorithms (43+ algorithms)
- **pyyaml**: YAML configuration support
  - Alternative: Use JSON (built-in)
- **matplotlib**: Visualization and plotting
  - For: Offline analysis, trajectory visualization
- **pandas**: Data analysis and telemetry processing
  - For: CSV/JSON log analysis

**Development Tools:**
- **ipython**: Enhanced interactive shell
- **pytest**: Testing framework (for unit tests)

#### 20.2.4 Pixi Configuration File (pyproject.toml)

Create `pyproject.toml` in project root:

```toml
[project]
name = "ptz-tracker"
version = "0.1.0"
description = "PTZ Camera Object Tracking System with Background Subtraction"
authors = [{name = "Your Name", email = "your.email@example.com"}]
requires-python = ">=3.8,<3.12"
readme = "README.md"
license = {text = "MIT"}

[tool.pixi.project]
channels = ["conda-forge"]
platforms = ["linux-64", "osx-arm64", "osx-64", "win-64"]

[tool.pixi.pypi-dependencies]
pybgs = "*"  # BGSLibrary - only available on PyPI

[tool.pixi.dependencies]
python = ">=3.8,<3.12"
opencv = ">=4.5"
numpy = ">=1.19"
pyyaml = ">=5.0"

[tool.pixi.feature.viz.dependencies]
matplotlib = ">=3.5"
pandas = ">=1.3"

[tool.pixi.feature.dev.dependencies]
ipython = "*"
pytest = ">=7.0"

# Tasks for common operations
[tool.pixi.tasks]
# Run the main tracking application
track = "python main.py"

# Run with specific config
track-config = "python main.py --config config.yaml"

# Run tests
test = "pytest tests/"

# Interactive Python shell
shell = "ipython"

# Clean generated files
clean = "rm -rf output/*.mp4 logs/*.log"

[tool.pixi.environments]
default = {solve-group = "default"}
viz = {features = ["viz"], solve-group = "default"}
dev = {features = ["dev", "viz"], solve-group = "default"}
```

#### 20.2.5 Alternative: Minimal pixi.toml

For simpler projects, use `pixi.toml`:

```toml
[project]
name = "ptz-tracker"
version = "0.1.0"
channels = ["conda-forge"]
platforms = ["linux-64", "osx-arm64", "osx-64", "win-64"]

[dependencies]
python = ">=3.8,<3.12"
opencv = ">=4.5"
numpy = ">=1.19"

[pypi-dependencies]
pybgs = "*"

[tasks]
track = "python main.py"
```

#### 20.2.6 Project Setup with Pixi

**Initial Setup:**
```bash
# Navigate to project directory
cd PTZ_tracker_dumb

# Initialize Pixi project (if not already done)
pixi init

# Or use the pyproject.toml above and install
pixi install

# Install with visualization features
pixi install --environment viz

# Install development environment
pixi install --environment dev
```

**Running the Application:**
```bash
# Run main tracking application
pixi run track

# Run with specific config file
pixi run track-config

# Or run Python directly in Pixi environment
pixi run python main.py

# Run with arguments
pixi run python main.py --input video.mp4 --output result.mp4
```

**Development Workflow:**
```bash
# Enter Pixi shell (activates environment)
pixi shell

# Now you're in the environment, run commands normally:
python main.py
ipython
pytest tests/

# Exit shell
exit
```

**Managing Dependencies:**
```bash
# Add a new dependency
pixi add scipy

# Add PyPI package
pixi add --pypi scikit-image

# Remove dependency
pixi remove scipy

# Update all dependencies
pixi update

# Show installed packages
pixi list
```

#### 20.2.7 Environment Isolation

Pixi automatically handles environment isolation:
- **No manual venv activation needed**
- **Reproducible across machines** via `pixi.lock` file
- **Automatic environment selection** based on tasks/commands
- **Per-project environments** (not system-wide)

#### 20.2.8 Cross-Platform Compatibility

Pixi ensures the project works across platforms:
```bash
# On Linux
pixi run track

# On macOS (ARM or Intel)
pixi run track

# On Windows
pixi run track
```

Same commands, same results!

#### 20.2.9 Quick Start Commands

```bash
# Clone and setup
git clone <repository-url>
cd PTZ_tracker_dumb
pixi install

# Run application
pixi run track

# Run with custom video
pixi run python main.py --input path/to/video.mp4

# Development mode
pixi shell
python main.py  # Run inside Pixi shell

# Run tests
pixi run test
```

#### 20.2.10 Legacy pip Installation (Alternative)

If Pixi is not available, fall back to traditional pip:
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install opencv-python numpy pybgs pyyaml

# Run application
python main.py
```

### 20.3 Project Structure

```
PTZ_tracker_dumb/
├── pyproject.toml                 # Pixi configuration (recommended)
├── pixi.toml                      # Alternative minimal Pixi config
├── pixi.lock                      # Pixi lock file (auto-generated)
├── config.yaml                    # Application configuration
├── .gitignore                     # Git ignore rules
├── README.md                      # Project documentation
├── TECHNICAL_SPECIFICATIONS.md    # This file
├── main.py                        # Main application entry point
├── src/                           # Source code directory
│   ├── __init__.py
│   ├── video_io.py               # Video input/output handling
│   ├── background_subtraction.py # Background subtraction (OpenCV/bgslibrary)
│   ├── object_detection.py       # Object detection and filtering
│   ├── tracking.py               # Tracking logic and state management
│   ├── ptz_control.py            # Virtual PTZ calculations
│   ├── rendering.py              # Frame rendering and overlays
│   ├── config.py                 # Configuration loading and validation
│   └── utils.py                  # Utility functions and helpers
├── tests/                         # Unit tests (pytest)
│   ├── __init__.py
│   ├── test_background_subtraction.py
│   ├── test_object_detection.py
│   └── test_ptz_control.py
├── logs/                          # Log files (auto-generated)
│   ├── tracking.log
│   └── telemetry.csv
├── output/                        # Output videos (auto-generated)
│   └── output.mp4
└── input/                         # Input videos
    └── example.mp4
```

### 20.4 Code Module Architecture

**Core Modules:**

1. **video_io.py**: Video input/output handling
   - VideoCapture wrapper
   - VideoWriter wrapper
   - Frame buffering

2. **background_subtraction.py**: Background subtraction algorithms
   - OpenCV subtractor wrapper (MOG2, KNN, etc.)
   - BGSLibrary wrapper (SuBSENSE, PAWCS, etc.)
   - Mask post-processing pipeline
   - Unified interface for both libraries

3. **object_detection.py**: Object detection and filtering
   - Contour detection
   - Filtering (area, aspect ratio, solidity)
   - Bounding box calculation
   - Centroid calculation

4. **tracking.py**: Tracking logic and state management
   - State machine (IDLE, ACQUIRING, TRACKING, LOST)
   - Object association
   - Confidence scoring
   - Trajectory recording

5. **ptz_control.py**: Virtual PTZ calculations
   - Pan/tilt calculation from centroid
   - Zoom control from object size
   - Smoothing and deadband
   - ROI calculation

6. **rendering.py**: Frame rendering and overlays
   - ROI extraction and resizing
   - Bounding box overlay
   - Crosshair overlay
   - Text information overlay
   - Trajectory visualization

7. **config.py**: Configuration loading and validation
   - YAML/JSON loading
   - Parameter validation
   - Default values

8. **main.py**: Main application loop and orchestration
   - Initialize all components
   - Main processing loop
   - User input handling
   - Cleanup

### 20.5 Coding Standards
- Follow PEP 8 style guidelines
- Type hints for function signatures
- Docstrings for all classes and functions
- Comprehensive error handling
- Avoid global variables (use class-based state)

### 20.6 Documentation
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

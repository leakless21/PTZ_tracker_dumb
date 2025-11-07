# Task 1: Video I/O & Background Subtraction

**Phase:** 1 - Foundation & Core Tracking
**Duration:** 2 days
**Priority:** Critical
**Dependencies:** None

---

## Overview

Implement video input/output handling and background subtraction to detect moving objects in video frames. This forms the foundation for all tracking functionality.

---

## Implementation Details

### 1.1 Video Capture Setup

**Objective:** Initialize video capture from file with proper error handling

**Key Components:**
- Use OpenCV VideoCapture to load input video
- Extract frame dimensions, FPS, and total frame count
- Validate video file exists and is readable
- Support looping playback for testing

**Configuration Parameters:**
- `video.input`: Path to input video file
- `video.loop_playback`: Boolean for continuous playback

### 1.2 Video Writer Setup

**Objective:** Configure video output writer for saving results

**Key Components:**
- Initialize VideoWriter with matching resolution
- Use configurable codec (default: mp4v)
- Match input FPS for smooth playback
- Create output directory if needed

**Configuration Parameters:**
- `video.output`: Path to output video file
- `video.output_codec`: FourCC codec identifier
- `video.save_output`: Boolean to enable/disable saving

### 1.3 Background Subtraction Implementation

**Objective:** Detect moving objects by separating foreground from background

**Algorithm Selection:**
- **MOG2 (Default)**: Best for general use, handles lighting changes, detects shadows
- **KNN (Alternative)**: Faster, better for static lighting

**Key Components:**
- Initialize background subtractor with tuned parameters
- Apply subtractor to each frame to generate foreground mask
- Handle learning rate (-1 for automatic adaptation)
- Store history frames for background model

**Configuration Parameters:**
- `background_subtraction.algorithm`: "MOG2" or "KNN"
- `background_subtraction.history`: Number of frames for background model (default: 500)
- `background_subtraction.learning_rate`: -1 for automatic, or 0.001-0.1 for manual

### 1.4 Mask Post-Processing

**Objective:** Clean up noisy foreground mask to improve detection quality

**Pipeline Steps:**
1. **Opening**: Remove small noise pixels while preserving object shapes
2. **Closing**: Fill holes inside objects
3. **Binary Threshold**: Convert to pure black/white mask

**Key Components:**
- Create morphological kernel (square, 5×5 default)
- Apply opening operation to remove noise
- Apply closing operation to fill gaps
- Threshold to ensure binary mask (0 or 255 only)

**Configuration Parameters:**
- `object_detection.kernel_size`: 3, 5, or 7 (default: 5)
- `object_detection.threshold_value`: 100-150 (default: 127)

---

## Test Scenarios

### Test 1.1: Video Loading
- **Scenario:** Load various video formats (mp4, avi, mov)
- **Expected Result:** Successfully open and read first frame
- **Validation:** Check frame is not None, dimensions are positive

### Test 1.2: Invalid Video Path
- **Scenario:** Attempt to load non-existent file
- **Expected Result:** Clear error message, graceful exit
- **Validation:** Exception caught, user-friendly error displayed

### Test 1.3: Video Writer Creation
- **Scenario:** Create output video with same specs as input
- **Expected Result:** Output file created successfully
- **Validation:** File exists, size increases as frames written

### Test 1.4: Background Subtraction - Stationary Camera
- **Scenario:** Process video with fixed camera, moving objects
- **Expected Result:** Moving objects appear white in mask, background black
- **Validation:** Visual inspection shows clear object separation

### Test 1.5: Background Subtraction - Lighting Changes
- **Scenario:** Process video with gradual lighting changes
- **Expected Result:** MOG2 adapts, continues detecting objects
- **Validation:** Objects remain visible through lighting transition

### Test 1.6: Mask Noise Reduction
- **Scenario:** Compare raw mask vs. post-processed mask
- **Expected Result:** Post-processed mask has fewer small noise spots
- **Validation:** Count connected components, should decrease after processing

### Test 1.7: Shadow Handling
- **Scenario:** Process video with strong shadows
- **Expected Result:** MOG2 with detectShadows=True marks shadows gray, not white
- **Validation:** Shadow pixels have value ~127, object pixels are 255

### Test 1.8: Frame Rate Consistency
- **Scenario:** Process 100 frames, measure FPS
- **Expected Result:** Maintains ≥20 FPS on 1080p video
- **Validation:** Time 100 frames, calculate average FPS

---

## Caveats

### Performance Considerations
- **High Resolution Impact**: 4K videos may require downscaling to maintain real-time performance
- **History Buffer Memory**: Large history values (>1000) increase memory usage significantly
- **Learning Rate Trade-off**: Fast learning (high rate) adapts quickly but may lose stationary objects; slow learning (low rate) is stable but slow to adapt

### Algorithm Limitations
- **MOG2 Shadow Detection**: Not perfect, may miss shadows in low contrast scenes or mark them as foreground
- **Static Objects**: Objects that stop moving will eventually become part of background model
- **Camera Motion**: Background subtraction assumes stationary camera; any camera movement breaks the model

### Edge Cases
- **First N Frames**: Background model is unstable during initialization (first ~100 frames)
- **Sudden Scene Changes**: Cuts or transitions cause entire frame to be marked as foreground temporarily
- **Empty Scenes**: If no motion for extended period, all foreground detection stops

### Configuration Pitfalls
- **Threshold Too High**: May lose small objects or thin appendages
- **Threshold Too Low**: Increases noise in mask
- **Kernel Too Large**: Can merge nearby objects or erode object boundaries
- **Kernel Too Small**: May not remove enough noise

### File I/O Issues
- **Codec Availability**: Not all codecs available on all systems (mp4v is safest)
- **Write Permissions**: Ensure output directory is writable
- **Disk Space**: Long videos require significant storage
- **Frame Drop**: If processing slower than real-time, may need to skip frames

---

## Success Criteria

✅ Video loads successfully from file path
✅ Frame dimensions and FPS extracted correctly
✅ Background subtraction produces clean foreground masks
✅ Moving objects clearly visible as white regions in mask
✅ Noise effectively reduced through morphological operations
✅ Output video writer configured and functional
✅ Processing achieves target frame rate (≥20 FPS for 1080p)
✅ Shadow detection working (gray pixels for shadows in MOG2)

---

## Dependencies

**Python Libraries:**
- opencv-python >= 4.5.0
- numpy >= 1.19.0
- pyyaml >= 5.0

**Configuration File:**
- config.yaml (video and background_subtraction sections)

---

## Integration Notes

**Outputs Used By:**
- Task 2 (Multi-Object Detection): Uses cleaned foreground mask for contour detection
- Task 6 (Recovery Mechanism): Uses background subtraction for redetecting lost objects
- Task 5 (Debug Mosaic): Displays raw mask, cleaned mask, and original frame

**State Dependencies:**
- Active in DETECTION state
- Active in LOST state (for recovery)
- Inactive in TRACKING state (not needed when CSRT is tracking)

---

## Estimated Effort

- Implementation: 6-8 hours
- Testing: 3-4 hours
- Debugging: 2-3 hours
- **Total: 1.5-2 days**

# PTZ Camera Object Tracking System - Project Plan

## Executive Summary

This document outlines the complete implementation plan for a real-time PTZ camera object tracking system using background subtraction, dual-mode tracking (Norfair + CSRT), and virtual camera control. The system processes pre-recorded video and simulates pan-tilt-zoom operations to keep tracked objects centered.

---

## Table of Contents

1. [Development Phases & Todo Lists](#development-phases--todo-lists)
2. [Project File Structure](#project-file-structure)
3. [Module Contracts & Responsibilities](#module-contracts--responsibilities)
4. [Data Flow & Integration Points](#data-flow--integration-points)
5. [Testing Strategy](#testing-strategy)
6. [Deployment & Maintenance](#deployment--maintenance)

---

## Development Phases & Todo Lists

### Phase 1: Foundation & Core Infrastructure

#### Configuration Management
- Create configuration loader module that parses YAML configuration files
- Implement configuration validation with schema checking
- Add configuration hot-reload capability for development
- Create default configuration fallback mechanism
- Implement configuration override via command-line arguments

#### Logging & Telemetry Infrastructure
- Set up structured logging system with multiple log levels
- Implement file-based logging with rotation
- Create telemetry data collection framework
- Design CSV export format for tracking metrics
- Add performance profiling decorators for critical functions

#### Project Structure Setup
- Create all required directories and subdirectories
- Set up package initialization files
- Configure import paths and module discovery
- Add development environment setup scripts
- Create requirements files for different installation methods

### Phase 2: Video Input & Preprocessing

#### Video Capture Module
- Implement video file reader with OpenCV VideoCapture
- Add video metadata extraction for resolution, FPS, codec information
- Create frame buffer management system
- Implement frame skip functionality for performance optimization
- Add loop playback support for continuous testing

#### Frame Preprocessing
- Implement frame resizing with configurable interpolation methods
- Add color space conversion utilities
- Create frame normalization functions
- Implement timestamp extraction and synchronization
- Add frame validation and error handling

### Phase 3: Background Subtraction Engine

#### OpenCV Background Subtraction Implementation
- Implement OpenCV MOG2 background subtractor
- Add KNN-based background subtraction option
- Create parameter configuration for OpenCV algorithms
- Implement learning rate control
- Add shadow detection toggle

#### Mask Post-Processing Pipeline
- Implement morphological erosion operation
- Add morphological dilation operation
- Create Gaussian blur filter for noise reduction
- Implement morphological closing operation
- Add binary thresholding with configurable values
- Create pipeline chaining mechanism for sequential operations

### Phase 4: Object Detection & Analysis

#### Contour Detection
- Implement contour finding with hierarchy analysis
- Add contour approximation for shape simplification
- Create contour filtering by area thresholds
- Implement bounding box extraction from contours
- Add contour moment calculation for centroids

#### Object Property Analysis
- Calculate object area in pixels
- Compute aspect ratio for shape filtering
- Calculate solidity for object density analysis
- Compute extent for bounding box fill ratio
- Implement convex hull calculation
- Add perimeter calculation

#### Object Filtering & Selection
- Implement multi-criteria filtering system based on size, shape, density
- Create largest object selector
- Add center-weighted object selector
- Implement previous position proximity selector
- Create object ranking system for selection priority

### Phase 5: State Management System

#### System State Controller
- Implement finite state machine with four states
- Create state transition validation logic
- Add state change event callbacks
- Implement state persistence for recovery
- Create state history tracking for debugging

#### State-Specific Logic
- Implement DETECTION_MODE processing pipeline
- Create LOCKED_MODE tracking loop
- Add LOST state recovery mechanism
- Implement IDLE state handling
- Create state-specific rendering logic

#### User Input Handler
- Implement keyboard event listener
- Create state-specific key mapping
- Add numeric key handler for object ID selection
- Implement release and reset key handlers
- Add debug toggle key handler
- Create pause and quit key handlers

### Phase 6: Multi-Object Tracking with Norfair

#### Norfair Integration
- Install and configure Norfair library
- Create detection-to-Norfair format converter
- Implement Norfair tracker initialization
- Configure distance function for object association
- Set up hit counter and initialization delay parameters

#### Track Management
- Implement track creation from detections
- Add track update mechanism with new detections
- Create track termination logic for lost objects
- Implement track ID assignment and management
- Add track history for trajectory visualization

#### Detection Integration
- Convert contour detections to Norfair detection format
- Implement centroid-based tracking
- Add bounding box information to tracked objects
- Create track confidence scoring
- Implement track age filtering for selection eligibility

### Phase 7: Single-Object Tracking with CSRT

#### CSRT Tracker Initialization
- Implement OpenCV CSRT tracker creation
- Add bounding box initialization from selected object
- Create tracker validation after initialization
- Implement failure handling during initialization
- Add tracker reset mechanism

#### CSRT Update & Monitoring
- Implement frame-by-frame tracker update
- Add bounding box validation for reasonable dimensions
- Create loss detection based on tracking quality
- Implement lost frame counter
- Add automatic transition to LOST state on failure

#### Recovery Mechanism
- Implement search area definition around last known position
- Create candidate object detection in search area
- Add size and position similarity scoring
- Implement best match selection algorithm
- Create CSRT re-initialization with recovered object
- Add recovery timeout with transition to DETECTION_MODE

### Phase 8: Virtual PTZ Control System

#### PTZ State Manager
- Implement pan angle state with bounds checking
- Add tilt angle state with bounds checking
- Create zoom factor state with min/max limits
- Implement PTZ state smoothing with exponential filtering
- Add velocity limiting for realistic camera movement

#### PTZ Control Logic
- Calculate object offset from frame center in normalized coordinates
- Implement deadband zone to prevent jitter
- Create proportional control law for pan adjustment
- Add proportional control law for tilt adjustment
- Implement zoom control based on object size
- Add maximum speed clamping for smooth movement

#### ROI Calculation
- Convert pan/tilt/zoom to Region of Interest in pixel coordinates
- Implement bounds checking to prevent ROI outside frame
- Add aspect ratio preservation
- Create ROI centering logic
- Implement ROI coordinate transformation utilities

#### Virtual Camera Rendering
- Extract ROI from original frame
- Implement ROI resizing to output dimensions
- Add interpolation method selection
- Create viewport transformation matrix
- Implement coordinate mapping from ROI to output frame

### Phase 9: Visualization & Rendering

#### Bounding Box Rendering
- Draw multi-object bounding boxes in cyan for DETECTION_MODE
- Render locked object bounding box in green for LOCKED_MODE
- Add search area circle in red for LOST state
- Implement variable line thickness
- Add anti-aliased drawing option

#### Overlay Information
- Display current system state text
- Show tracking mode indicator
- Add object ID labels above bounding boxes
- Display PTZ parameters on screen
- Show frame counter and timestamp
- Add FPS counter display
- Create track age display for locked objects

#### Debug Mosaic View
- Design 2x4 grid layout for eight pipeline stages
- Implement original frame tile
- Add foreground mask tile
- Create processed mask tile
- Add contours visualization tile
- Implement detection results tile
- Add tracking visualization tile
- Create PTZ ROI visualization tile
- Implement final output tile
- Add resize logic for uniform tile dimensions
- Create mosaic composition and border drawing

#### Trajectory Visualization
- Store object position history
- Implement trajectory line drawing
- Add trajectory decay for visual clarity
- Create color-coded trajectory by track ID
- Implement trajectory pruning for performance

### Phase 10: Output & Recording

#### Video Writer Setup
- Initialize OpenCV VideoWriter with codec selection
- Configure output resolution and frame rate
- Implement codec fallback mechanism
- Add file path validation and directory creation
- Create writer error handling

#### Frame Recording
- Write processed frames to output video
- Add optional mosaic recording
- Implement conditional recording based on configuration
- Create frame queue for asynchronous writing
- Add recording statistics tracking

#### Display Manager
- Create OpenCV window with proper naming
- Implement window resize and positioning
- Add fullscreen mode toggle
- Create window close event handler
- Implement frame display with refresh rate control

### Phase 11: Main Application Loop

#### Application Initialization
- Load and validate configuration
- Initialize logging and telemetry systems
- Create video capture instance
- Initialize background subtractor
- Create Norfair tracker instance
- Initialize system state machine
- Set up PTZ state
- Create video writer if output enabled
- Initialize display window if visualization enabled

#### Main Processing Loop
- Read frame from video source
- Increment frame counter and update timestamp
- Branch processing based on current system state
- Execute state-specific pipeline
- Update PTZ control based on tracked object
- Calculate and extract ROI
- Render visualization overlays
- Compose debug mosaic if enabled
- Write frame to output video if enabled
- Display frame to window if enabled
- Handle keyboard input and state transitions
- Log telemetry data
- Check for loop termination conditions

#### Cleanup & Finalization
- Release video capture resources
- Release video writer and flush buffers
- Destroy OpenCV windows
- Save final telemetry data to CSV
- Close log files
- Print summary statistics
- Exit gracefully

### Phase 12: Testing & Validation

#### Unit Tests
- Write tests for configuration loader
- Create tests for preprocessing functions
- Add tests for object detection filters
- Implement tests for PTZ control calculations
- Create tests for state transitions
- Add tests for coordinate transformations

#### Integration Tests
- Test video input to preprocessing pipeline
- Validate background subtraction to detection flow
- Test detection to tracking integration
- Validate tracking to PTZ control flow
- Test state machine transitions with real scenarios
- Validate end-to-end pipeline with sample videos

#### Performance Tests
- Benchmark frame processing speed
- Test memory usage under extended runtime
- Validate real-time capability at different resolutions
- Test algorithm performance comparisons
- Benchmark PTZ calculation overhead

#### Edge Case Testing
- Test with no objects detected
- Validate behavior with multiple simultaneous objects
- Test object entering and leaving frame
- Validate occlusion handling
- Test rapid camera motion scenarios
- Validate extreme lighting conditions

### Phase 13: Documentation & Polish

#### Code Documentation
- Add docstrings to all public functions
- Create type hints for function signatures
- Write module-level documentation
- Add inline comments for complex logic
- Create API reference documentation

#### User Documentation
- Update README with complete usage instructions
- Document all configuration parameters
- Create troubleshooting guide
- Add example configurations for common scenarios
- Write algorithm selection guide

#### Developer Documentation
- Document system architecture
- Create data flow diagrams
- Write module interaction guide
- Document state machine behavior
- Add extension and customization guide

---

## Project File Structure

PTZ_tracker_dumb/
│
├── config.yaml                      # Main configuration file
├── README.md                        # User-facing documentation
├── TECHNICAL_SPECIFICATIONS.md      # Complete technical specifications
├── PROJECT_PLAN.md                  # This file - project planning document
├── pixi.toml                        # Pixi project configuration and dependencies
├── .gitignore                       # Git ignore patterns
│
├── main.py                          # Application entry point
│
├── src/                             # Source code root
│   │
│   ├── __init__.py                  # Package initialization
│   │
│   ├── config/                      # Configuration management
│   │   ├── __init__.py
│   │   ├── loader.py                # YAML configuration loader
│   │   ├── validator.py             # Configuration schema validation
│   │   └── defaults.py              # Default configuration values
│   │
│   ├── core/                        # Core application logic
│   │   ├── __init__.py
│   │   ├── application.py           # Main application class
│   │   ├── state_machine.py         # System state management
│   │   └── pipeline.py              # Processing pipeline orchestration
│   │
│   ├── video/                       # Video input/output
│   │   ├── __init__.py
│   │   ├── capture.py               # Video file reading and frame extraction
│   │   ├── writer.py                # Video file writing
│   │   └── preprocessing.py         # Frame preprocessing utilities
│   │
│   ├── background_subtraction/      # Background subtraction module
│   │   ├── __init__.py
│   │   ├── base.py                  # Abstract base class for subtractors
│   │   ├── opencv_subtractor.py     # OpenCV algorithms implementation
│   │   ├── factory.py               # Algorithm factory pattern
│   │   └── postprocessing.py        # Mask morphological operations
│   │
│   ├── detection/                   # Object detection module
│   │   ├── __init__.py
│   │   ├── contour_detector.py      # Contour-based object detection
│   │   ├── object_analyzer.py       # Object property calculation
│   │   ├── filters.py               # Object filtering by criteria
│   │   └── selector.py              # Object selection strategies
│   │
│   ├── tracking/                    # Object tracking module
│   │   ├── __init__.py
│   │   ├── base.py                  # Abstract tracker interface
│   │   ├── norfair_tracker.py       # Norfair multi-object tracking
│   │   ├── csrt_tracker.py          # CSRT single-object tracking
│   │   ├── track_manager.py         # Track lifecycle management
│   │   └── recovery.py              # Lost object recovery logic
│   │
│   ├── ptz/                         # PTZ control module
│   │   ├── __init__.py
│   │   ├── controller.py            # PTZ control logic
│   │   ├── state.py                 # PTZ state management (pan/tilt/zoom)
│   │   ├── roi_calculator.py        # ROI calculation from PTZ parameters
│   │   └── virtual_camera.py        # Virtual camera rendering
│   │
│   ├── rendering/                   # Visualization and rendering
│   │   ├── __init__.py
│   │   ├── bbox_drawer.py           # Bounding box drawing utilities
│   │   ├── overlay.py               # Information overlay rendering
│   │   ├── mosaic.py                # Debug mosaic composition
│   │   ├── trajectory.py            # Object trajectory visualization
│   │   └── colors.py                # Color definitions and utilities
│   │
│   ├── input/                       # User input handling
│   │   ├── __init__.py
│   │   ├── keyboard.py              # Keyboard event handling
│   │   └── commands.py              # Command interpretation and dispatch
│   │
│   └── utils/                       # Utility functions
│       ├── __init__.py
│       ├── logging.py               # Logging configuration and utilities
│       ├── telemetry.py             # Telemetry data collection and export
│       ├── coordinates.py           # Coordinate system transformations
│       ├── geometry.py              # Geometric calculations
│       └── performance.py           # Performance profiling utilities
│
├── tests/                           # Test suite
│   ├── __init__.py
│   ├── conftest.py                  # Pytest configuration and fixtures
│   │
│   ├── unit/                        # Unit tests
│   │   ├── __init__.py
│   │   ├── test_config.py           # Configuration loading tests
│   │   ├── test_detection.py        # Detection logic tests
│   │   ├── test_ptz.py              # PTZ calculation tests
│   │   ├── test_coordinates.py      # Coordinate transformation tests
│   │   └── test_state_machine.py    # State transition tests
│   │
│   ├── integration/                 # Integration tests
│   │   ├── __init__.py
│   │   ├── test_pipeline.py         # End-to-end pipeline tests
│   │   ├── test_tracking.py         # Tracking integration tests
│   │   └── test_video_io.py         # Video input/output tests
│   │
│   └── fixtures/                    # Test fixtures and data
│       ├── sample_videos/           # Sample video files for testing
│       ├── sample_configs/          # Sample configuration files
│       └── expected_outputs/        # Expected test outputs
│
├── logs/                            # Runtime logs (generated)
│   ├── tracking.log                 # Application log file
│   └── telemetry.csv                # Telemetry data export
│
├── docs/                            # Additional documentation
│   ├── architecture.md              # System architecture documentation
│   ├── algorithms.md                # Algorithm selection guide
│   ├── api_reference.md             # API documentation
│   └── troubleshooting.md           # Common issues and solutions
│
└── examples/                        # Example scripts and configurations
    ├── configs/                     # Example configurations
    │   ├── outdoor_tracking.yaml    # Configuration for outdoor scenarios
    │   ├── indoor_tracking.yaml     # Configuration for indoor scenarios
    │   └── high_performance.yaml    # Performance-optimized configuration
    └── scripts/                     # Utility scripts
        ├── benchmark.py             # Performance benchmarking script
        ├── compare_algorithms.py    # BGS algorithm comparison tool
        └── analyze_telemetry.py     # Telemetry data analysis script

---

## Module Contracts & Responsibilities

### Configuration Module (src/config/)

**Purpose**: Centralized configuration management with validation

**Responsibilities**:
- Load YAML configuration files
- Validate configuration against schema
- Provide default values for missing parameters
- Support command-line overrides
- Expose configuration as structured objects

**Key Functions**:
- load_config: Parse YAML file and return configuration object
- validate_config: Check configuration for required fields and valid ranges
- merge_configs: Combine multiple configuration sources with priority
- get_default_config: Return default configuration

**Inputs**:
- YAML configuration file path
- Optional command-line argument dictionary

**Outputs**:
- Validated configuration object with all parameters
- Configuration error exceptions if validation fails

**Dependencies**:
- PyYAML library
- Python dataclasses or pydantic for structure

---

### Video Capture Module (src/video/capture.py)

**Purpose**: Handle video file input and frame extraction

**Responsibilities**:
- Open video files using OpenCV VideoCapture
- Extract frame metadata (resolution, FPS, codec)
- Provide frame-by-frame iteration
- Support frame skipping for performance
- Handle loop playback
- Manage video resources and cleanup

**Key Functions**:
- open_video: Initialize video capture from file path
- read_frame: Read next frame and return as numpy array
- get_metadata: Return video properties
- reset: Restart video from beginning
- close: Release video resources

**Inputs**:
- Video file path
- Configuration parameters for frame skipping and looping

**Outputs**:
- RGB/BGR frames as numpy arrays
- Frame metadata (timestamp, frame number)
- End-of-stream flag

**Dependencies**:
- OpenCV (cv2.VideoCapture)
- NumPy

---

### Video Writer Module (src/video/writer.py)

**Purpose**: Handle video file output and recording

**Responsibilities**:
- Initialize OpenCV VideoWriter with codec
- Write frames to output video file
- Handle codec fallback if primary fails
- Manage output file creation and directory structure
- Release resources on completion

**Key Functions**:
- initialize_writer: Create VideoWriter with specified parameters
- write_frame: Write single frame to video
- finalize: Close and flush video file
- is_opened: Check writer status

**Inputs**:
- Output file path
- Frame dimensions and FPS
- Codec specification
- Frames as numpy arrays

**Outputs**:
- Video file on disk
- Write success status

**Dependencies**:
- OpenCV (cv2.VideoWriter)
- NumPy

---

### Preprocessing Module (src/video/preprocessing.py)

**Purpose**: Frame preprocessing operations

**Responsibilities**:
- Resize frames with configurable interpolation
- Convert color spaces
- Normalize frame values
- Apply brightness/contrast adjustments if needed

**Key Functions**:
- resize_frame: Resize frame to target dimensions
- convert_color: Convert between color spaces
- normalize_frame: Scale pixel values to standard range

**Inputs**:
- Input frame as numpy array
- Target dimensions or color space

**Outputs**:
- Processed frame as numpy array

**Dependencies**:
- OpenCV
- NumPy

---

### Background Subtraction Module (src/background_subtraction/)

**Purpose**: Separate foreground objects from background using various algorithms

**Responsibilities**:
- Provide unified interface for multiple BGS algorithms
- Support OpenCV implementations
- Manage background model updates
- Apply mask post-processing pipeline

**Key Classes**:
- BackgroundSubtractor (abstract base class)
- OpenCVSubtractor (OpenCV implementation)
- SubtractorFactory (algorithm selection)
- MaskPostprocessor (morphological operations)

**Key Functions**:
- apply: Process frame and return foreground mask
- postprocess_mask: Apply erosion, dilation, blur, threshold
- reset: Reinitialize background model

**Inputs**:
- Current frame as numpy array
- Configuration parameters for algorithm and post-processing

**Outputs**:
- Binary foreground mask (uint8 with 0/255 values)

**Dependencies**:
- OpenCV
- NumPy

---

### Detection Module (src/detection/)

**Purpose**: Detect and analyze objects from foreground mask

**Responsibilities**:
- Find contours in foreground mask
- Calculate object properties (area, aspect ratio, solidity, extent)
- Filter objects based on size and shape criteria
- Select best object for tracking based on strategy
- Convert detections to standardized format

**Key Classes**:
- ContourDetector: Extract contours from mask
- ObjectAnalyzer: Calculate object properties
- ObjectFilter: Apply filtering criteria
- ObjectSelector: Select object based on strategy

**Key Functions**:
- detect_objects: Find all objects in mask
- analyze_object: Compute properties for single object
- filter_objects: Remove objects not meeting criteria
- select_object: Choose best object for tracking
- compute_centroid: Calculate object center point

**Inputs**:
- Binary foreground mask
- Configuration for filtering criteria
- Previous frame state for selection

**Outputs**:
- List of detected objects with bounding boxes and properties
- Selected object for tracking

**Dependencies**:
- OpenCV (findContours, moments)
- NumPy

---

### Tracking Module (src/tracking/)

**Purpose**: Track objects across frames in dual-mode system

**Responsibilities**:
- Multi-object tracking using Norfair in DETECTION_MODE
- Single-object tracking using CSRT in LOCKED_MODE
- Manage track lifecycle (creation, update, termination)
- Implement lost object recovery
- Assign and maintain track IDs

**Key Classes**:
- NorfairTracker: Multi-object tracking wrapper
- CSRTTracker: Single-object CSRT tracking wrapper
- TrackManager: Track lifecycle and ID management
- RecoverySystem: Lost object reacquisition

**Key Functions**:
- update_tracks: Update Norfair with new detections
- initialize_csrt: Start CSRT tracking on selected object
- update_csrt: Update CSRT tracker with new frame
- check_csrt_lost: Detect CSRT tracking failure
- recover_object: Search and reacquire lost object

**Inputs**:
- Current frame
- Detected objects in DETECTION_MODE
- Previous tracking state
- Current system mode

**Outputs**:
- Updated track list with IDs and bounding boxes
- Tracking success/failure status
- Recovered object if found

**Dependencies**:
- Norfair library
- OpenCV (cv2.TrackerCSRT)
- NumPy

---

### PTZ Control Module (src/ptz/)

**Purpose**: Simulate PTZ camera control to center tracked objects

**Responsibilities**:
- Maintain PTZ state (pan, tilt, zoom)
- Calculate control commands from object position
- Implement proportional control with deadband
- Apply smoothing and velocity limits
- Calculate ROI from PTZ parameters
- Render virtual camera view

**Key Classes**:
- PTZController: Main control logic
- PTZState: State management (pan/tilt/zoom)
- ROICalculator: Convert PTZ to region of interest
- VirtualCamera: Render ROI as output frame

**Key Functions**:
- update_ptz: Calculate new PTZ values from object position
- calculate_roi: Compute ROI bounds from PTZ state
- extract_roi: Extract and resize ROI from frame
- apply_deadband: Prevent jitter near center
- smooth_motion: Apply exponential smoothing

**Inputs**:
- Object centroid in frame coordinates
- Object bounding box dimensions
- Current PTZ state
- Configuration for sensitivity and limits

**Outputs**:
- Updated PTZ state
- ROI coordinates
- Rendered output frame

**Dependencies**:
- NumPy
- OpenCV (for ROI extraction and resizing)

---

### State Machine Module (src/core/state_machine.py)

**Purpose**: Manage system-wide state transitions and behavior

**Responsibilities**:
- Maintain current system state (DETECTION, LOCKED, LOST, IDLE)
- Validate and execute state transitions
- Provide state-specific processing branches
- Track state history for debugging
- Manage timers and counters for state conditions

**Key Functions**:
- transition_to_detection_mode: Switch to multi-object detection
- transition_to_locked_mode: Switch to single-object tracking
- transition_to_lost: Enter recovery mode
- transition_to_idle: Pause processing
- can_transition: Validate state change
- get_current_state: Return current state
- handle_keyboard_input: Process input based on state

**Inputs**:
- State transition requests
- User input events
- Tracking success/failure flags
- Timing information

**Outputs**:
- Current system state
- State change notifications
- State-specific processing flags

**Dependencies**:
- Python enum for state definitions
- Time module for timeouts

---

### Rendering Module (src/rendering/)

**Purpose**: Visualize tracking results and pipeline stages

**Responsibilities**:
- Draw bounding boxes with state-specific colors
- Render information overlay text
- Display PTZ parameters and tracking status
- Compose debug mosaic with pipeline stages
- Draw object trajectories
- Render track IDs and ages

**Key Classes**:
- BoundingBoxDrawer: Draw boxes and labels
- OverlayRenderer: Text and information display
- MosaicComposer: Debug view composition
- TrajectoryDrawer: Path visualization

**Key Functions**:
- draw_detection_boxes: Draw cyan boxes for all tracks
- draw_locked_box: Draw green box for locked object
- draw_search_area: Draw red circle for lost state
- render_overlay: Draw text information
- compose_mosaic: Create 2x4 debug grid
- draw_trajectory: Render object path

**Inputs**:
- Frame to draw on
- Tracking results (boxes, IDs)
- System state information
- PTZ state
- Configuration for colors and styles

**Outputs**:
- Annotated frame with visualizations
- Debug mosaic frame

**Dependencies**:
- OpenCV (drawing functions)
- NumPy

---

### Application Module (src/core/application.py)

**Purpose**: Main application orchestration and lifecycle

**Responsibilities**:
- Initialize all subsystems
- Run main processing loop
- Coordinate between modules
- Handle errors and exceptions
- Manage resource cleanup
- Implement pause and quit functionality

**Key Functions**:
- initialize: Set up all components
- run: Execute main loop
- process_frame: State-based frame processing
- cleanup: Release resources
- handle_error: Error recovery

**Inputs**:
- Configuration object
- Command-line arguments

**Outputs**:
- Processed video output
- Telemetry data
- Log files
- Exit status

**Dependencies**:
- All other modules

---

### Input Handler Module (src/input/)

**Purpose**: Process user input and keyboard commands

**Responsibilities**:
- Listen for keyboard events
- Map keys to commands based on state
- Dispatch commands to state machine
- Handle special keys (pause, debug toggle, quit)

**Key Functions**:
- get_key_press: Poll for keyboard input
- interpret_command: Map key to action based on state
- execute_command: Dispatch to appropriate handler

**Inputs**:
- Keyboard events from OpenCV
- Current system state

**Outputs**:
- Command objects for state machine
- System control signals

**Dependencies**:
- OpenCV (cv2.waitKey)

---

### Utilities Module (src/utils/)

**Purpose**: Shared utility functions for common operations

**Responsibilities**:
- Logging setup and management
- Telemetry data collection and export
- Coordinate system transformations
- Geometric calculations
- Performance profiling

**Key Functions**:
- setup_logging: Configure logging system
- log_telemetry: Record tracking metrics
- export_telemetry_csv: Save telemetry to file
- transform_coordinates: Convert between coordinate systems
- calculate_distance: Euclidean distance
- calculate_iou: Intersection over union
- profile_function: Performance measurement decorator

**Inputs**:
- Various data types depending on function
- Configuration parameters

**Outputs**:
- Log entries
- CSV telemetry files
- Transformed coordinates
- Calculated metrics

**Dependencies**:
- Python logging
- CSV module
- NumPy
- Time and datetime modules

---

## Data Flow & Integration Points

### Frame Processing Pipeline

Video File → Capture → Preprocess → State Machine Decision:

**DETECTION_MODE Branch**:
Frame → Background Subtraction → Mask Post-processing → Contour Detection → Object Analysis → Object Filtering → Object Selection → Norfair Update → PTZ Control → ROI Calculation → Virtual Camera → Rendering → Display/Recording

**LOCKED_MODE Branch**:
Frame → CSRT Update → Check Success → PTZ Control → ROI Calculation → Virtual Camera → Rendering → Display/Recording

**LOST Branch**:
Frame → Background Subtraction → Detection → Search Area Filtering → Match Scoring → Recovery Decision → (Transition to LOCKED or DETECTION) → PTZ Hold → Rendering → Display/Recording

### State Transition Triggers

**DETECTION → LOCKED**:
- User presses numeric key matching track ID
- Track must be valid and age > threshold
- CSRT initialization succeeds

**LOCKED → LOST**:
- CSRT update fails for threshold frames
- Bounding box dimensions invalid

**LOST → LOCKED**:
- Recovery finds matching object
- CSRT reinitialization succeeds

**LOST → DETECTION**:
- Recovery timeout expires
- User presses reset key

**Any → DETECTION**:
- User presses reset key

### Configuration Flow

YAML File → Config Loader → Validator → Config Object → Distributed to modules during initialization

### Telemetry Flow

Module Events → Telemetry Collector → In-Memory Buffer → Periodic CSV Write → Log File

---

## Testing Strategy

### Unit Testing Approach

**Test each module in isolation**:
- Mock dependencies using pytest fixtures
- Test boundary conditions and edge cases
- Validate error handling
- Test configuration variations

**Coverage Goals**:
- Minimum 80% code coverage
- 100% coverage for critical paths (state transitions, PTZ calculations)

### Integration Testing Approach

**Test module interactions**:
- Use real video files from fixtures
- Test full pipeline with different configurations
- Validate state transitions with real scenarios
- Test error propagation between modules

### Performance Testing

**Benchmarks to validate**:
- Frame processing time < 33ms for 30 FPS real-time
- Memory usage stable over 1000+ frames
- Algorithm comparison for speed vs accuracy tradeoff

### Test Data Requirements

**Video fixtures needed**:
- Single object in center
- Multiple objects
- Object entering/leaving frame
- Occlusion scenarios
- Different lighting conditions
- Different resolutions

---

## Deployment & Maintenance

### Installation Methods

**Primary: Pixi**
- Use pixi.toml for environment management
- Cross-platform compatibility
- Automatic dependency resolution

**Secondary: pip**
- Provide requirements.txt
- Support virtual environments

### Configuration Management

**Default configuration**:
- Embedded defaults in code
- config.yaml for user customization
- Command-line overrides for testing

**Profile-based configurations**:
- Provide example configs for common scenarios
- Outdoor vs indoor
- High accuracy vs high performance
- Different camera types

### Logging and Debugging

**Log levels**:
- DEBUG: Frame-by-frame state changes
- INFO: Significant events (state transitions, object selection)
- WARNING: Recoverable errors (CSRT loss)
- ERROR: Critical failures

**Debug tools**:
- Mosaic view for pipeline visualization
- Telemetry export for post-analysis
- Performance profiling decorators

### Maintenance Considerations

**Code organization**:
- Modular design for easy updates
- Abstract interfaces for algorithm swapping
- Configuration-driven behavior

**Extension points**:
- Additional BGS algorithms
- Alternative tracking methods
- Custom object selection strategies
- New visualization modes

**Version control**:
- Git for source control
- Semantic versioning
- Changelog maintenance
- Branch strategy for features

### Performance Optimization Opportunities

**Potential bottlenecks**:
- Background subtraction (algorithm choice critical)
- Morphological operations (kernel size matters)
- ROI extraction and resizing (interpolation method)
- Debug mosaic composition (disable in production)

**Optimization strategies**:
- Frame skipping for non-real-time use
- Reduce processing resolution
- Optimize contour detection parameters
- Use faster BGS algorithms
- Disable unnecessary visualizations

---

## Risk Assessment & Mitigation

### Technical Risks

**Risk: CSRT tracking failures in challenging conditions**
- Mitigation: Implement robust recovery mechanism, tunable timeout

**Risk: Performance below real-time on target hardware**
- Mitigation: Configurable frame skipping, algorithm selection guide

**Risk: Memory leaks in long-running sessions**
- Mitigation: Proper resource management, periodic profiling

### Usability Risks

**Risk: Complex configuration overwhelming users**
- Mitigation: Provide sensible defaults, example configs, validation

**Risk: Unclear system state during operation**
- Mitigation: Clear visual indicators, status overlay, logging

**Risk: Difficult debugging when tracking fails**
- Mitigation: Debug mosaic, telemetry export, comprehensive logging

---

## Success Criteria

### Functional Requirements

- System successfully processes video files with background subtraction
- Multi-object tracking in DETECTION_MODE with ID assignment
- Single-object CSRT tracking in LOCKED_MODE with keyboard selection
- Automatic recovery from tracking loss within 3 seconds
- Virtual PTZ control centers objects with smooth motion
- Debug mosaic displays all pipeline stages correctly
- Configuration file controls all major parameters

### Performance Requirements

- Process 30 FPS video at 1920x1080 resolution
- Frame latency under 50ms on modern hardware
- Memory usage under 500MB for 5-minute video
- No memory leaks over extended runtime

### Quality Requirements

- Code coverage above 80%
- All tests passing in CI/CD pipeline
- Documentation complete for all public APIs
- User guide with examples and troubleshooting

---

## Timeline Estimation

**Phase 1-2 (Foundation & Video I/O)**: 3-5 days
**Phase 3 (Background Subtraction)**: 4-6 days
**Phase 4 (Object Detection)**: 3-4 days
**Phase 5 (State Machine)**: 2-3 days
**Phase 6 (Norfair Tracking)**: 3-4 days
**Phase 7 (CSRT & Recovery)**: 4-5 days
**Phase 8 (PTZ Control)**: 4-6 days
**Phase 9 (Rendering)**: 3-4 days
**Phase 10 (Output)**: 2-3 days
**Phase 11 (Main Loop)**: 3-4 days
**Phase 12 (Testing)**: 5-7 days
**Phase 13 (Documentation)**: 2-3 days

**Total Estimated Duration**: 38-54 days (7-10 weeks)

**Note**: Timeline assumes single developer working full-time. Adjust for team size and part-time work.

---

## Conclusion

This project plan provides a comprehensive roadmap for implementing the PTZ Camera Object Tracking System. The modular architecture ensures maintainability and extensibility, while the dual-mode tracking approach balances multi-object awareness with high-accuracy single-object tracking.

Key success factors:
- Rigorous testing at each phase
- Clear module contracts preventing coupling
- Configuration-driven behavior for flexibility
- Robust state management for reliability
- Comprehensive documentation for usability

The phased approach allows for incremental development and validation, with each phase building on the previous foundation. Regular testing and integration checkpoints will ensure the system meets performance and quality requirements.
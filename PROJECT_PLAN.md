# PTZ Camera Object Tracking System - Project Plan

**Version:** 2.0 (Simplified)
**Date:** November 7, 2025
**Estimated Duration:** 2-3 weeks
**Target:** Clean, minimal MVP implementation

---

## Executive Summary

This plan outlines a **simplified, focused approach** to implementing a PTZ object tracking system. The goal is to deliver a **working MVP in 2-3 weeks** with clean, maintainable code (~750 lines across 5 files).

**Key Changes from Previous Plan:**
- Reduced from 13 phases to **4 phases**
- Reduced from 7-10 weeks to **2-3 weeks**
- Reduced from 34+ files to **5 files**
- Reduced complexity while keeping desired workflow

**Core Principle:** Build the MVP first, then iterate based on real needs.

---

## Development Approach

### YAGNI Principle

**"You Aren't Gonna Need It"** - Only implement what's needed now.

**Included in MVP:**
- ✅ Multi-object detection with IDs (Norfair)
- ✅ Manual ID selection (0-9 keys)
- ✅ Single-object tracking (CSRT/KCF)
- ✅ Virtual PTZ control
- ✅ Debug mosaic (2×4 grid)
- ✅ Basic configuration
- ✅ Video I/O

**Deferred to Post-MVP:**
- ❌ Extensive logging infrastructure (use basic logging)
- ❌ Telemetry CSV export
- ❌ Performance profiling decorators
- ❌ Complex recovery strategies (keep it simple)
- ❌ Multiple background subtraction libraries

---

## Project Structure

```
PTZ_tracker_dumb/
├── main.py              # ~250 lines
├── tracker.py           # ~300 lines
├── ptz.py              # ~100 lines
├── debug_view.py       # ~100 lines
├── config.yaml         # ~50 lines
├── requirements.txt
├── README.md
└── .gitignore

Total: ~750 lines of implementation code
```

---

## Development Phases

### Phase 1: Foundation & Core Tracking (Week 1)

**Goal:** Get object detection and tracking working

**Duration:** 5-7 days

#### Tasks

**Day 1-2: Video I/O & Background Subtraction**
- [ ] Set up project structure (5 files)
- [ ] Implement video capture with OpenCV
- [ ] Implement background subtraction (MOG2)
- [ ] Add mask post-processing (opening, closing, threshold)
- [ ] Test with sample video

**Day 3-4: Multi-Object Detection with Norfair**
- [ ] Implement contour detection and filtering
- [ ] Convert contours to Norfair detections
- [ ] Initialize Norfair tracker
- [ ] Display all objects with IDs (cyan boxes)
- [ ] Test ID persistence across frames

**Day 5-7: Single-Object Tracking with CSRT/KCF**
- [ ] Implement KCF tracker initialization
- [ ] Add object selection by ID (keyboard 0-9)
- [ ] Implement CSRT update loop
- [ ] Add bbox validation
- [ ] Implement simple state machine (3 states)
- [ ] Test transition from DETECTION → TRACKING

**Deliverable:** Video showing multi-object detection with IDs, and ability to lock onto selected object

---

### Phase 2: PTZ Control & Visualization (Week 1-2)

**Goal:** Add virtual PTZ and debug visualization

**Duration:** 5-7 days

#### Tasks

**Day 1-2: PTZ Controller**
- [ ] Implement PTZController class
- [ ] Calculate pan/tilt from object position
- [ ] Calculate zoom from object size
- [ ] Add deadband zone (prevent jitter)
- [ ] Implement ROI extraction
- [ ] Test PTZ keeps object centered

**Day 3-4: Debug Mosaic**
- [ ] Implement DebugMosaic class in debug_view.py
- [ ] Create 2×4 grid layout
- [ ] Add pipeline stage visualization:
  - Original frame
  - Raw mask
  - Cleaned mask
  - Contours
  - Norfair detection
  - CSRT tracking
  - PTZ ROI overlay
  - Final output
- [ ] Add toggle with 'D' key
- [ ] Test mosaic visualization

**Day 5-7: Drawing & UI**
- [ ] Implement drawing functions:
  - Cyan boxes for detection mode
  - Green box for tracking mode
  - Red circle for lost mode
  - ID labels
  - Status overlay
- [ ] Add keyboard controls (R, D, Space, Q)
- [ ] Implement video output writer
- [ ] Test complete UI workflow

**Deliverable:** Full PTZ tracking with debug mosaic and clean UI

---

### Phase 3: Recovery & Configuration (Week 2)

**Goal:** Handle edge cases and add configuration

**Duration:** 4-5 days

#### Tasks

**Day 1-2: Recovery Mechanism**
- [ ] Implement LOST state handling
- [ ] Add redetection logic:
  - Background subtraction near last position
  - Match by size and distance
  - Reinitialize CSRT if found
- [ ] Add recovery timeout (3 seconds)
- [ ] Draw search area (red circle)
- [ ] Test recovery with occlusions

**Day 3-4: Configuration System**
- [ ] Create config.yaml structure
- [ ] Implement YAML loading
- [ ] Add basic validation
- [ ] Test different parameter combinations
- [ ] Document all parameters

**Day 5: Polish**
- [ ] Handle edge cases:
  - No objects detected
  - Object leaves frame
  - Invalid bbox
  - Video end/loop
- [ ] Add error messages
- [ ] Test with various videos

**Deliverable:** Robust system with configuration and error handling

---

### Phase 4: Testing & Documentation (Week 3)

**Goal:** Ensure quality and usability

**Duration:** 3-5 days

#### Tasks

**Day 1-2: Testing**
- [ ] Test with different video types:
  - Single object
  - Multiple objects
  - Fast motion
  - Occlusions
  - Different lighting
- [ ] Performance testing:
  - Measure FPS at 720p and 1080p
  - Check memory usage
  - Identify bottlenecks
- [ ] Fix bugs found during testing

**Day 3-4: Documentation**
- [ ] Update README with:
  - Quick start guide
  - Installation instructions
  - Usage examples
  - Keyboard controls
- [ ] Add code comments
- [ ] Create example configuration files
- [ ] Record demo video

**Day 5: Final Review**
- [ ] Code review and cleanup
- [ ] Verify all success criteria met
- [ ] Create release notes
- [ ] Tag version 1.0

**Deliverable:** Production-ready MVP with documentation

---

## Detailed Module Implementation

### Module 1: main.py (~250 lines)

**Purpose:** Application entry point, main loop, state machine

**Components:**
- Video I/O setup
- Main processing loop
- State machine (3 states)
- Keyboard input handling
- UI coordination
- Drawing functions

**Key Functions:**
```python
def load_config(path)
def draw_detection_mode(frame, tracked_objects)
def draw_tracking_mode(frame, bbox, obj_id)
def draw_lost_mode(frame, last_bbox, search_radius)
def draw_status_overlay(frame, state, ptz, fps)
def main()
```

**Estimated Time:** 3-4 days

---

### Module 2: tracker.py (~300 lines)

**Purpose:** Object detection and tracking logic

**Components:**
- Background subtraction (OpenCV MOG2/KNN)
- Mask post-processing
- Contour detection
- Norfair multi-object tracking
- CSRT/KCF single-object tracking
- Recovery logic

**Key Class:**
```python
class ObjectTracker:
    def __init__(self, config)
    def update_detection_mode(self, frame)
    def lock_onto_object(self, frame, object_id)
    def update_tracking_mode(self, frame)
    def update_lost_mode(self, frame)
    def reset_to_detection_mode()
```

**Estimated Time:** 4-5 days

---

### Module 3: ptz.py (~100 lines)

**Purpose:** Virtual PTZ calculations and ROI extraction

**Components:**
- PTZ state (pan, tilt, zoom)
- Proportional control
- Deadband logic
- ROI calculation
- ROI extraction and resizing

**Key Class:**
```python
class PTZController:
    def __init__(self, frame_shape, config)
    def update(self, bbox)
    def extract_roi(self, frame)
    def reset()
```

**Estimated Time:** 2 days

---

### Module 4: debug_view.py (~100 lines)

**Purpose:** Debug mosaic visualization

**Components:**
- 2×4 grid layout
- Tile preparation
- Label overlay
- Grayscale to color conversion

**Key Class:**
```python
class DebugMosaic:
    def __init__(self, config)
    def create_mosaic(self, pipeline_stages)
    def _prepare_tile(self, frame, label)
```

**Helper Functions:**
```python
def draw_contours_on_frame(frame, mask)
def draw_ptz_roi_overlay(frame, ptz_state)
```

**Estimated Time:** 2 days

---

### Module 5: config.yaml (~50 lines)

**Purpose:** Configuration parameters

**Sections:**
- video (input/output)
- background_subtraction (algorithm, parameters)
- object_detection (filtering criteria)
- tracking (Norfair, CSRT, recovery)
- ptz (sensitivity, limits)
- display (windows, debug mosaic)

**Estimated Time:** 1 day (spread across phases)

---

## Dependencies

### Required Libraries

```txt
opencv-python>=4.5.0  # Computer vision
numpy>=1.19.0         # Numerical operations
norfair>=2.0.0        # Multi-object tracking
pyyaml>=5.0           # Configuration
```

### Installation

**Option 1: pip**
```bash
pip install -r requirements.txt
```

**Option 2: Pixi (recommended)**
```bash
pixi install
pixi run python main.py
```

---

## Testing Strategy

### Unit Testing (Optional for MVP)

Focus on critical functions:
- PTZ calculations (pan, tilt, zoom, ROI)
- Bbox validation
- State transitions

### Integration Testing

Test complete workflows:
1. Video load → Detection → Selection → Tracking → Output
2. Tracking loss → Recovery → Reacquisition
3. Multiple state transitions
4. Debug mosaic display

### Performance Testing

Measure and optimize:
- FPS at different resolutions
- Memory usage over time
- Processing latency

### Test Videos

Prepare test cases:
- Single object, slow motion
- Multiple objects
- Fast motion
- Occlusions
- Lighting changes
- Object entering/leaving frame

---

## Success Criteria

### Functional Requirements

✅ **Video Processing**
- Loads video files
- Processes frames in real-time (≥20 FPS)
- Saves output video

✅ **Detection**
- Detects multiple moving objects
- Assigns persistent IDs (0-9)
- Filters noise effectively

✅ **Tracking**
- User can select object by ID
- Tracks selected object accurately
- Handles tracking loss gracefully

✅ **PTZ Control**
- Keeps object centered
- Smooth pan/tilt/zoom
- Respects deadband zone

✅ **Visualization**
- Debug mosaic shows pipeline stages
- Clear visual indicators for states
- Status overlay with info

✅ **Usability**
- Keyboard controls work
- Configuration is simple
- Easy to understand and modify

### Code Quality Requirements

✅ **Clean Code**
- ~750 lines total
- Single responsibility per module
- Clear function names
- Minimal comments needed (self-documenting)

✅ **Maintainability**
- Easy to understand
- Easy to debug
- Easy to extend

✅ **Documentation**
- README with quick start
- Code comments where needed
- Example configurations

---

## Risk Management

### Technical Risks

| Risk | Mitigation |
|------|------------|
| **Norfair ID persistence issues** | Test with various scenarios, adjust parameters |
| **CSRT tracking failures** | Implement robust recovery, test with difficult videos |
| **Performance below target** | Use KCF instead of CSRT, reduce resolution |
| **PTZ jitter** | Tune deadband and smoothing parameters |

### Schedule Risks

| Risk | Mitigation |
|------|------------|
| **Feature creep** | Stick to MVP scope, defer non-essentials |
| **Unexpected complexity** | Time-boxed tasks, ask for help if blocked |
| **Testing takes longer** | Start testing early, test incrementally |

---

## Timeline Summary

### Week 1 (Days 1-7)
- Video I/O & background subtraction
- Norfair multi-object tracking
- CSRT single-object tracking
- Basic state machine

### Week 2 (Days 8-14)
- PTZ controller
- Debug mosaic
- UI and drawing functions
- Recovery mechanism
- Configuration system

### Week 3 (Days 15-21)
- Testing with various videos
- Bug fixes
- Documentation
- Polish and release

**Total: 15-21 days (3 weeks)**

---

## Post-MVP Roadmap

### Version 1.1 (Optional Enhancements)
- Better logging (loguru with rotation)
- Telemetry export (CSV)
- More background subtraction algorithms
- Performance profiling

### Version 2.0 (Future Features)
- Deep learning detection (YOLO)
- Physical PTZ camera support
- Real-time camera streams
- Multiple object simultaneous tracking
- Trajectory analysis

### Version 3.0 (Advanced Features)
- Configuration UI
- Cloud integration
- Multi-camera support
- Event detection and alerts

---

## Development Best Practices

### Code Organization

- **One class per file** (except helper functions)
- **Clear module boundaries**
- **Minimal inter-module dependencies**

### Coding Standards

- **PEP 8** style guide
- **Type hints** for function signatures
- **Docstrings** for public functions
- **Descriptive variable names**

### Git Workflow

- **Feature branches** for each phase
- **Frequent commits** with clear messages
- **Test before committing**
- **Code review** before merging

### Development Tools

- **VSCode** or **PyCharm** for IDE
- **Git** for version control
- **Pixi** for environment management
- **pytest** for testing (if time allows)

---

## Resource Requirements

### Hardware

- **CPU**: Multi-core processor (i5/Ryzen 5 or better)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 10GB for videos and outputs

### Software

- **Python**: 3.8-3.11
- **OS**: Linux, macOS, or Windows
- **Git**: Version control
- **Pixi**: Package management

### Sample Videos

Prepare test videos:
- Resolution: 720p to 1080p
- Duration: 30 seconds to 2 minutes
- Various scenarios (indoor, outdoor, single/multi object)

---

## Conclusion

This simplified project plan focuses on delivering a **working MVP in 2-3 weeks** with **clean, maintainable code**. By following the YAGNI principle and deferring non-essential features, we can create a solid foundation that's easy to understand, use, and extend.

**Key Success Factors:**
1. **Focus on MVP** - Don't add features not in the plan
2. **Test incrementally** - Verify each component works before moving on
3. **Keep it simple** - If something seems complex, simplify it
4. **Document as you go** - Don't leave docs for the end

**Next Steps:**
1. Review and approve this plan
2. Set up development environment
3. Begin Phase 1: Foundation & Core Tracking
4. Iterate based on feedback

---

**Document Version:** 2.0 (Simplified 3-Week Plan)
**Previous Version:** 1.0 (13-Phase 7-10 Week Plan)
**Changes:** Reduced to 4 phases, 2-3 weeks, focused on MVP, removed over-engineering

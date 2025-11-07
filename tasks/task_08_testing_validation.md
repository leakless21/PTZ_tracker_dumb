# Task 8: Testing & Validation

**Phase:** 4 - Testing & Documentation
**Duration:** 3-5 days
**Priority:** Critical
**Dependencies:** All previous tasks

---

## Overview

Comprehensive testing strategy to ensure system reliability, performance, and usability. Covers functional testing, performance benchmarking, edge case handling, and user acceptance testing.

---

## Implementation Details

### 8.1 Test Video Preparation

**Objective:** Prepare diverse test videos covering various scenarios

**Required Test Videos:**

1. **Basic Tracking (easy.mp4)**
   - Single object
   - Slow, smooth motion
   - Good lighting
   - No occlusions
   - Duration: 30-60 seconds

2. **Multiple Objects (multi.mp4)**
   - 3-5 moving objects
   - Different sizes
   - Crossing paths
   - Duration: 30-60 seconds

3. **Fast Motion (fast.mp4)**
   - Rapid object movement
   - Quick direction changes
   - Tests tracker responsiveness
   - Duration: 20-40 seconds

4. **Occlusions (occlusion.mp4)**
   - Object passes behind obstacles
   - Partial and complete occlusions
   - Tests recovery mechanism
   - Duration: 30-60 seconds

5. **Lighting Changes (lighting.mp4)**
   - Gradual lighting transitions
   - Shadows moving
   - Tests background adaptation
   - Duration: 40-60 seconds

6. **Crowded Scene (crowded.mp4)**
   - Many moving objects (10+)
   - Similar looking objects
   - Tests ID persistence
   - Duration: 30-60 seconds

**Video Specifications:**
- Resolution: 720p or 1080p
- Format: MP4 (H.264)
- Frame rate: 25-30 FPS
- Color: Full color (not grayscale)

### 8.2 Functional Testing

**Objective:** Verify all features work as specified

**Test Categories:**

#### Video I/O Tests
- [ ] Load video successfully
- [ ] Extract correct frame dimensions and FPS
- [ ] Write output video with same specs
- [ ] Handle end of video (loop or exit)
- [ ] Handle invalid video path gracefully

#### Background Subtraction Tests
- [ ] MOG2 detects moving objects
- [ ] KNN detects moving objects
- [ ] Mask post-processing removes noise
- [ ] Shadow detection works (MOG2)
- [ ] Adapts to lighting changes

#### Object Detection Tests
- [ ] Contours detected from mask
- [ ] Small noise filtered out (min_area)
- [ ] Large blobs filtered out (max_area_fraction)
- [ ] Valid objects passed to Norfair

#### Multi-Object Tracking Tests
- [ ] Norfair assigns persistent IDs
- [ ] IDs maintained across frames
- [ ] IDs persist through brief occlusions
- [ ] New objects get new IDs
- [ ] Exiting objects release IDs
- [ ] Cyan boxes drawn correctly
- [ ] ID labels visible

#### Single-Object Tracking Tests
- [ ] User can select object by pressing 0-9
- [ ] CSRT/KCF tracker initializes
- [ ] Green box follows object
- [ ] Tracking works through partial occlusions
- [ ] Detects tracking failure
- [ ] Transitions to LOST on failure

#### PTZ Control Tests
- [ ] Pan adjusts to center object horizontally
- [ ] Tilt adjusts to center object vertically
- [ ] Zoom maintains target object size
- [ ] Deadband prevents jitter
- [ ] ROI extracted correctly
- [ ] ROI resized smoothly

#### Recovery Tests
- [ ] Enters LOST state on tracking failure
- [ ] Red circle shows search area
- [ ] Timer displays correctly
- [ ] Redetects object if within radius
- [ ] Size matching works
- [ ] Recovers and resumes tracking
- [ ] Times out after 3 seconds
- [ ] Returns to DETECTION on timeout

#### Debug Mosaic Tests
- [ ] Mosaic displays 2×4 grid
- [ ] All tiles labeled correctly
- [ ] Toggle with 'D' key works
- [ ] Shows correct stages per state
- [ ] Grayscale masks converted to BGR

#### Configuration Tests
- [ ] Config loads from YAML
- [ ] Defaults used if config missing
- [ ] Validation catches invalid values
- [ ] Command-line args override config

#### Keyboard Controls Tests
- [ ] 0-9 keys select objects
- [ ] 'R' resets to DETECTION
- [ ] 'D' toggles debug mosaic
- [ ] Space pauses/resumes
- [ ] 'Q' and ESC quit

### 8.3 Performance Testing

**Objective:** Measure and optimize system performance

**Metrics to Measure:**

1. **Frame Rate**
   - Target: ≥30 FPS (720p with KCF)
   - Target: ≥20 FPS (1080p with CSRT)
   - Measure: Average FPS over 1000 frames
   - Measure: Minimum FPS (detect stutters)

2. **Processing Latency**
   - Target: <50ms per frame
   - Measure: Time from capture to display
   - Critical for real-time applications

3. **Memory Usage**
   - Target: <500 MB
   - Measure: Peak memory consumption
   - Monitor for memory leaks (long runs)

4. **CPU Usage**
   - Target: <80% single core
   - Measure: Average CPU percentage
   - Check multi-threading opportunities

**Performance Test Procedure:**
```python
import time
import psutil

def benchmark_performance(video_path, num_frames=1000):
    process = psutil.Process()
    start_time = time.time()
    start_memory = process.memory_info().rss / 1024 / 1024  # MB

    # Process num_frames
    for i in range(num_frames):
        # ... processing ...
        pass

    end_time = time.time()
    end_memory = process.memory_info().rss / 1024 / 1024  # MB

    elapsed = end_time - start_time
    fps = num_frames / elapsed
    memory_used = end_memory - start_memory

    print(f"FPS: {fps:.1f}")
    print(f"Memory: {memory_used:.1f} MB")
    print(f"CPU: {psutil.cpu_percent()}%")
```

**Performance Optimization:**
- Profile code to find bottlenecks
- Optimize hot loops
- Reduce unnecessary copying
- Consider resolution reduction if needed

### 8.4 Edge Case Testing

**Objective:** Verify system handles unusual conditions gracefully

**Edge Cases to Test:**

1. **No Objects Detected**
   - Scenario: Empty scene or all objects filtered
   - Expected: System continues, no crashes

2. **All Objects Leave Frame**
   - Scenario: All tracked objects exit
   - Expected: Empty detection, ready for new objects

3. **Object at Frame Edge**
   - Scenario: Select object partially off-screen
   - Expected: Tracking handles, bbox clipped to bounds

4. **Very Small Object**
   - Scenario: Object only a few pixels
   - Expected: Filtered by min_area or tracks with high zoom

5. **Very Large Object**
   - Scenario: Object fills 80% of frame
   - Expected: Filtered or tracks with minimum zoom

6. **Rapid State Transitions**
   - Scenario: Quickly press 'R', select, lose track, repeat
   - Expected: No crashes, state machine stable

7. **Camera Shake**
   - Scenario: Video with camera movement
   - Expected: Background subtraction fails gracefully

8. **Sudden Scene Cut**
   - Scenario: Video with hard cut to different scene
   - Expected: Entire frame detected as foreground temporarily

9. **Stationary Object**
   - Scenario: Object stops moving
   - Expected: Becomes part of background, track lost

10. **Zero Division**
    - Scenario: Bbox with zero width or height
    - Expected: Validation catches, no division errors

11. **Negative Coordinates**
    - Scenario: Tracker returns invalid bbox
    - Expected: Validation rejects, transition to LOST

12. **Maximum Zoom**
    - Scenario: Zoom reaches max (5.0x)
    - Expected: Clamped, no further increase

### 8.5 Integration Testing

**Objective:** Test complete workflows end-to-end

**Test Workflows:**

**Workflow 1: Happy Path**
1. Start application with valid video
2. Wait for objects to appear
3. Press '0' to select first object
4. Verify tracking follows object
5. Object stays in frame for 30 seconds
6. Press 'Q' to quit
7. Verify output video saved

**Workflow 2: Recovery Path**
1. Start and select object
2. Wait for object to be occluded
3. Verify transition to LOST state
4. Wait for object to reappear
5. Verify recovery to TRACKING
6. Continue tracking

**Workflow 3: Timeout Path**
1. Start and select object
2. Wait for object to leave frame
3. Verify transition to LOST
4. Wait 3 seconds (no reappearance)
5. Verify transition to DETECTION
6. See all objects with IDs again

**Workflow 4: Manual Reset**
1. Start and select object
2. Track for 10 seconds
3. Press 'R' to release
4. Verify return to DETECTION
5. Select different object
6. Verify new tracking session

**Workflow 5: Debug Visualization**
1. Start with debug mosaic enabled
2. Observe all 8 tiles
3. Select object
4. Observe state changes in mosaic
5. Toggle mosaic off and on
6. Verify synchronized display

### 8.6 User Acceptance Testing

**Objective:** Ensure system meets user needs and expectations

**User Testing Tasks:**

1. **Task: Basic Tracking**
   - "Track the person walking across the frame"
   - Measure: Success rate, time to complete
   - Feedback: Was it intuitive?

2. **Task: Multiple Object Selection**
   - "Track the car with ID 2"
   - Measure: Selection accuracy
   - Feedback: Were IDs clear?

3. **Task: Recovery Understanding**
   - "What happens when tracking is lost?"
   - Measure: User comprehension
   - Feedback: Is red circle clear?

4. **Task: PTZ Effect**
   - "Does the tracking keep object centered?"
   - Measure: User satisfaction
   - Feedback: Is zoom appropriate?

5. **Task: Debug Mosaic**
   - "Toggle debug view and describe what you see"
   - Measure: Understanding of stages
   - Feedback: Is layout clear?

**Usability Criteria:**
- User can start tracking within 30 seconds
- User understands state transitions
- User can interpret visualizations
- Keyboard shortcuts are memorable

### 8.7 Regression Testing

**Objective:** Ensure changes don't break existing functionality

**Regression Test Suite:**
- Run all functional tests after each change
- Compare performance metrics with baseline
- Automated testing (if time allows)

**Regression Checklist:**
```
Phase 1 Tests:
[ ] Video I/O still works
[ ] Background subtraction unchanged
[ ] Multi-object detection stable
[ ] Single-object tracking reliable

Phase 2 Tests:
[ ] PTZ control accurate
[ ] Debug mosaic displays correctly

Phase 3 Tests:
[ ] Recovery mechanism functional
[ ] Configuration loads properly
```

### 8.8 Bug Tracking and Resolution

**Objective:** Systematically identify and fix issues

**Bug Report Template:**
```
**Bug ID:** BUG-001
**Severity:** Critical / High / Medium / Low
**Component:** Tracking / PTZ / Recovery / etc.
**Description:** [What went wrong]
**Steps to Reproduce:**
1. [Step 1]
2. [Step 2]
3. [Step 3]
**Expected Behavior:** [What should happen]
**Actual Behavior:** [What actually happens]
**Screenshots/Logs:** [If applicable]
**Status:** Open / In Progress / Resolved
**Resolution:** [How it was fixed]
```

**Bug Prioritization:**
- Critical: System crashes, data loss
- High: Major feature broken
- Medium: Minor feature issue
- Low: Cosmetic or rare edge case

---

## Test Scenarios

### Test 8.1: Complete System Test
- **Scenario:** Run all functional tests on basic_tracking.mp4
- **Expected Result:** All tests pass
- **Validation:** Checklist 100% complete

### Test 8.2: Performance Baseline
- **Scenario:** Benchmark with 1080p video, KCF tracker
- **Expected Result:** ≥20 FPS average
- **Validation:** Meets performance target

### Test 8.3: Stress Test
- **Scenario:** Process 10-minute video continuously
- **Expected Result:** No memory leaks, stable FPS
- **Validation:** Memory usage constant, no slowdown

### Test 8.4: Multi-Video Test
- **Scenario:** Test all 6 prepared test videos
- **Expected Result:** All videos process successfully
- **Validation:** No crashes, reasonable results

### Test 8.5: Configuration Variations
- **Scenario:** Test with different config presets
- **Expected Result:** Each preset works as intended
- **Validation:** Fast preset faster, accurate preset more accurate

### Test 8.6: Edge Case Suite
- **Scenario:** Run all 12 edge case tests
- **Expected Result:** Graceful handling, no crashes
- **Validation:** Clear error messages, state recovery

### Test 8.7: User Testing Session
- **Scenario:** 3 users complete 5 testing tasks
- **Expected Result:** >80% success rate, positive feedback
- **Validation:** User feedback forms, task completion times

---

## Caveats

### Testing Limitations
- **Automated Testing**: MVP may not have unit tests (time constraint)
- **Coverage**: Cannot test every possible scenario
- **Test Data**: Limited to available videos
- **Subjectivity**: Some metrics (usability) are subjective

### Performance Variability
- **Hardware Dependent**: FPS varies by CPU/GPU
- **Video Dependent**: Complex scenes slower than simple ones
- **State Dependent**: DETECTION slower than TRACKING

### Bug Discovery
- **Late Discovery**: Some bugs only appear in production use
- **Rare Edge Cases**: Difficult to anticipate all scenarios
- **User Behavior**: Real usage may differ from test scenarios

### Time Constraints
- **Comprehensive Testing**: Takes significant time
- **Bug Fixing**: May discover issues requiring rework
- **Trade-offs**: May need to defer non-critical bugs

---

## Success Criteria

✅ All functional tests pass on basic video
✅ Performance meets targets (≥20 FPS 1080p)
✅ No crashes during 1000-frame stress test
✅ All 6 test videos process successfully
✅ Edge cases handled gracefully (no crashes)
✅ Memory usage stable (<500 MB)
✅ User testing tasks completed with >80% success
✅ Configuration presets work as expected
✅ Debug mosaic displays correctly in all states
✅ Keyboard controls all functional
✅ Recovery mechanism works in occlusion test
✅ PTZ keeps object centered in all tests

---

## Dependencies

**Python Libraries:**
- opencv-python >= 4.5.0
- numpy >= 1.19.0
- norfair >= 2.0.0
- pyyaml >= 5.0
- psutil (for performance monitoring)

**Previous Tasks:**
- All tasks (this tests complete system)

**Test Assets:**
- 6 test videos (various scenarios)
- Configuration presets

---

## Integration Notes

**Inputs:**
- Test videos (6 scenarios)
- Configuration presets
- User testing scripts

**Outputs:**
- Test results report
- Performance metrics
- Bug list
- User feedback

**Deliverables:**
- Test results document
- Performance benchmark report
- Known issues list
- User acceptance report

---

## Estimated Effort

- Test Video Preparation: 4-6 hours
- Functional Testing: 8-10 hours
- Performance Testing: 4-6 hours
- Edge Case Testing: 4-6 hours
- Integration Testing: 4-6 hours
- User Acceptance Testing: 4-6 hours
- Bug Fixing: 8-12 hours
- Documentation: 4-6 hours
- **Total: 3-5 days**

---

## Testing Checklist

### Pre-Testing Setup
- [ ] Install all dependencies
- [ ] Prepare 6 test videos
- [ ] Create configuration presets
- [ ] Set up performance monitoring tools

### Functional Tests
- [ ] Video I/O (5 tests)
- [ ] Background Subtraction (5 tests)
- [ ] Object Detection (4 tests)
- [ ] Multi-Object Tracking (7 tests)
- [ ] Single-Object Tracking (6 tests)
- [ ] PTZ Control (6 tests)
- [ ] Recovery (7 tests)
- [ ] Debug Mosaic (5 tests)
- [ ] Configuration (4 tests)
- [ ] Keyboard Controls (5 tests)

### Performance Tests
- [ ] FPS measurement (720p and 1080p)
- [ ] Latency measurement
- [ ] Memory usage monitoring
- [ ] CPU usage monitoring
- [ ] Long-run stability test

### Edge Cases
- [ ] No objects detected
- [ ] All objects leave frame
- [ ] Object at frame edge
- [ ] Very small/large objects
- [ ] Rapid state transitions
- [ ] Camera shake
- [ ] Scene cuts
- [ ] Zero division cases
- [ ] Invalid bbox handling

### Integration Tests
- [ ] Happy path workflow
- [ ] Recovery workflow
- [ ] Timeout workflow
- [ ] Manual reset workflow
- [ ] Debug visualization workflow

### User Acceptance
- [ ] Basic tracking task
- [ ] Multiple object selection
- [ ] Recovery understanding
- [ ] PTZ effect evaluation
- [ ] Debug mosaic comprehension

### Documentation
- [ ] Test results documented
- [ ] Performance metrics recorded
- [ ] Bugs logged and prioritized
- [ ] User feedback collected
- [ ] Final report written

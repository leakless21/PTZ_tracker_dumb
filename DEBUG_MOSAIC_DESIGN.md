# Debug Mosaic Design for PTZ Tracker

**Purpose:** Visualize all pipeline stages side-by-side for debugging and understanding

---

## Mosaic Layout (2×4 Grid)

```
┌─────────────┬─────────────┬─────────────┬─────────────┐
│ 1. Original │ 2. FG Mask  │ 3. Cleaned  │ 4. Contours │
│    Frame    │   (Raw)     │    Mask     │   Detected  │
├─────────────┼─────────────┼─────────────┼─────────────┤
│ 5. Norfair  │ 6. CSRT     │ 7. PTZ ROI  │ 8. Final    │
│  Detection  │  Tracking   │   Overlay   │   Output    │
└─────────────┴─────────────┴─────────────┴─────────────┘
```

**8 tiles showing:**
1. **Original Frame**: Input frame with no processing
2. **FG Mask (Raw)**: Background subtraction output (before cleaning)
3. **Cleaned Mask**: After morphological operations
4. **Contours Detected**: Mask with detected contours outlined
5. **Norfair Detection**: Objects tracked with Norfair (IDs shown)
6. **CSRT Tracking**: Single object being tracked (when in TRACKING mode)
7. **PTZ ROI Overlay**: Shows current PTZ viewport on original frame
8. **Final Output**: Result with overlays and PTZ transformation

---

## Implementation: debug_view.py

```python
"""
Debug mosaic visualization for PTZ tracker
Shows all pipeline stages in 2×4 grid
"""

import cv2
import numpy as np

class DebugMosaic:
    def __init__(self, config):
        self.config = config
        self.enabled = config['display'].get('show_debug_mosaic', False)
        self.tile_width = 320  # Each tile width
        self.tile_height = 240  # Each tile height

    def create_mosaic(self, pipeline_stages):
        """
        Create 2×4 mosaic from pipeline stages

        Args:
            pipeline_stages: dict with keys:
                - 'original': Original frame
                - 'fg_mask_raw': Raw foreground mask
                - 'fg_mask_clean': Cleaned mask
                - 'contours': Frame with contours
                - 'detection': Norfair detection visualization
                - 'tracking': CSRT tracking visualization
                - 'ptz_roi': PTZ ROI overlay
                - 'final': Final output

        Returns:
            mosaic: 2×4 grid image
        """
        if not self.enabled:
            return None

        # Prepare tiles
        tiles = []

        # Row 1
        tiles.append(self._prepare_tile(pipeline_stages.get('original'), "1. Original"))
        tiles.append(self._prepare_tile(pipeline_stages.get('fg_mask_raw'), "2. FG Mask Raw"))
        tiles.append(self._prepare_tile(pipeline_stages.get('fg_mask_clean'), "3. Cleaned Mask"))
        tiles.append(self._prepare_tile(pipeline_stages.get('contours'), "4. Contours"))

        # Row 2
        tiles.append(self._prepare_tile(pipeline_stages.get('detection'), "5. Detection"))
        tiles.append(self._prepare_tile(pipeline_stages.get('tracking'), "6. Tracking"))
        tiles.append(self._prepare_tile(pipeline_stages.get('ptz_roi'), "7. PTZ ROI"))
        tiles.append(self._prepare_tile(pipeline_stages.get('final'), "8. Final"))

        # Create rows
        row1 = cv2.hconcat([tiles[0], tiles[1], tiles[2], tiles[3]])
        row2 = cv2.hconcat([tiles[4], tiles[5], tiles[6], tiles[7]])

        # Stack rows
        mosaic = cv2.vconcat([row1, row2])

        return mosaic

    def _prepare_tile(self, frame, label):
        """Prepare a single tile with label"""
        if frame is None:
            # Create blank tile if frame is missing
            tile = np.zeros((self.tile_height, self.tile_width, 3), dtype=np.uint8)
        else:
            # Resize to tile size
            tile = cv2.resize(frame, (self.tile_width, self.tile_height))

            # Convert grayscale to BGR if needed
            if len(tile.shape) == 2:
                tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)

        # Add label
        cv2.putText(tile, label, (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Add border
        cv2.rectangle(tile, (0, 0), (self.tile_width-1, self.tile_height-1),
                     (100, 100, 100), 2)

        return tile

    def save_mosaic(self, mosaic, output_path):
        """Save mosaic to file"""
        if mosaic is not None:
            cv2.imwrite(output_path, mosaic)


def draw_contours_on_frame(frame, mask):
    """Helper: Draw contours from mask onto frame"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output = frame.copy()
    cv2.drawContours(output, contours, -1, (0, 255, 0), 2)

    return output


def draw_ptz_roi_overlay(frame, ptz_state):
    """Helper: Draw PTZ ROI rectangle on original frame"""
    output = frame.copy()

    # Calculate ROI bounds from PTZ state
    h, w = frame.shape[:2]
    roi_w = int(w / ptz_state.zoom)
    roi_h = int(h / ptz_state.zoom)

    center_x = w // 2 + int(ptz_state.pan * w / 90)
    center_y = h // 2 + int(ptz_state.tilt * h / 90)

    roi_x = max(0, center_x - roi_w // 2)
    roi_y = max(0, center_y - roi_h // 2)
    roi_x = min(roi_x, w - roi_w)
    roi_y = min(roi_y, h - roi_h)

    # Draw ROI rectangle
    cv2.rectangle(output, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h),
                 (0, 255, 255), 3)  # Yellow

    # Add label
    cv2.putText(output, "PTZ ROI", (roi_x + 10, roi_y + 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    return output
```

---

## Integration into main.py

```python
# In main.py, add debug mosaic support

from debug_view import DebugMosaic, draw_contours_on_frame, draw_ptz_roi_overlay

def main():
    # ... existing initialization ...

    # Initialize debug mosaic
    debug_mosaic = DebugMosaic(config)

    # Main loop
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Initialize pipeline stages dict
        pipeline_stages = {}

        # Store original
        pipeline_stages['original'] = frame.copy()

        # Process based on state
        if tracker.state == "DETECTION":
            # Get mask stages from tracker
            fg_mask_raw = tracker.bg_subtractor.apply(frame)
            pipeline_stages['fg_mask_raw'] = fg_mask_raw

            # Cleaned mask
            kernel = np.ones((5, 5), np.uint8)
            fg_mask_clean = cv2.morphologyEx(fg_mask_raw, cv2.MORPH_OPEN, kernel)
            pipeline_stages['fg_mask_clean'] = fg_mask_clean

            # Contours visualization
            contours_frame = draw_contours_on_frame(frame, fg_mask_clean)
            pipeline_stages['contours'] = contours_frame

            # Update tracker (this returns tracked_objects)
            tracked_objects = tracker.update_detection_mode(frame)

            # Detection visualization
            detection_frame = frame.copy()
            draw_detection_mode(detection_frame, tracked_objects)
            pipeline_stages['detection'] = detection_frame

            # No tracking in this mode
            pipeline_stages['tracking'] = None

        elif tracker.state == "TRACKING":
            # ... tracking code ...

            # Store stages
            pipeline_stages['fg_mask_raw'] = None
            pipeline_stages['fg_mask_clean'] = None
            pipeline_stages['contours'] = None
            pipeline_stages['detection'] = None

            # Tracking visualization
            tracking_frame = frame.copy()
            if success:
                draw_tracking_mode(tracking_frame, bbox, tracker.selected_id)
            pipeline_stages['tracking'] = tracking_frame

        # PTZ ROI overlay
        ptz_roi_frame = draw_ptz_roi_overlay(frame, ptz)
        pipeline_stages['ptz_roi'] = ptz_roi_frame

        # Apply PTZ transformation
        ptz_frame = ptz.extract_roi(display_frame)
        draw_status_overlay(ptz_frame, tracker.state, ptz, current_fps)
        pipeline_stages['final'] = ptz_frame

        # Display main output
        cv2.imshow("PTZ Tracker", ptz_frame)

        # Create and display debug mosaic
        if debug_mosaic.enabled:
            mosaic = debug_mosaic.create_mosaic(pipeline_stages)
            if mosaic is not None:
                cv2.imshow("Debug Pipeline", mosaic)

        # Handle keyboard
        key = cv2.waitKey(1) & 0xFF

        if key == ord('d'):  # Toggle debug mosaic
            debug_mosaic.enabled = not debug_mosaic.enabled
            if not debug_mosaic.enabled:
                cv2.destroyWindow("Debug Pipeline")
```

---

## Updated Configuration

```yaml
display:
  show_window: true
  show_info: true
  show_debug_mosaic: true  # Enable debug mosaic by default

  # Debug mosaic settings
  debug_mosaic:
    tile_width: 320
    tile_height: 240
    save_to_file: false
    output_path: "output/debug_mosaic.mp4"
```

---

## Keyboard Controls

- **'D' key**: Toggle debug mosaic on/off during runtime
- **'Q' or ESC**: Quit application (closes both windows)

---

## Example Output

When enabled, you'll see two windows:

**Window 1: "PTZ Tracker"**
- Main output with PTZ transformation
- Status overlay
- Bounding boxes

**Window 2: "Debug Pipeline"**
- 2×4 grid showing all 8 stages
- Each tile labeled
- Updated in real-time

---

## Benefits

1. **Understanding**: See exactly what's happening at each stage
2. **Debugging**: Quickly identify where pipeline fails
3. **Tuning**: Adjust parameters and see immediate effect
4. **Presentation**: Great for demos and documentation
5. **Learning**: Excellent educational tool

---

## Performance Impact

- **Minimal**: Mosaic creation adds ~5-10ms per frame
- **Toggleable**: Can be turned off with 'D' key
- **Efficient**: Uses cv2.resize and cv2.hconcat (fast operations)

---

## Optional: Save Debug Mosaic to Video

```python
# In main.py initialization
if config['display']['debug_mosaic'].get('save_to_file', False):
    mosaic_path = config['display']['debug_mosaic']['output_path']
    mosaic_width = debug_mosaic.tile_width * 4
    mosaic_height = debug_mosaic.tile_height * 2

    mosaic_writer = cv2.VideoWriter(
        mosaic_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (mosaic_width, mosaic_height)
    )
else:
    mosaic_writer = None

# In main loop
if mosaic_writer and mosaic is not None:
    mosaic_writer.write(mosaic)

# Cleanup
if mosaic_writer:
    mosaic_writer.release()
```

---

## Summary

The debug mosaic is a **powerful debugging tool** that adds:
- ~100 lines of code in `debug_view.py`
- ~20 lines of integration in `main.py`
- Minimal performance impact
- Huge value for development and debugging

**Definitely worth including in MVP!**

---

**Document Version:** 1.0
**Part of:** Revised recommendations with debug mosaic support

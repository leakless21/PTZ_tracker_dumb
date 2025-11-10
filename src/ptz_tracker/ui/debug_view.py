"""
Debug visualization helpers for the PTZ Camera Object Tracking System.

Provides visualization utilities for pipeline stages, overlays, and debug mosaic.
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from loguru import logger


class DebugMosaic:
    """Creates a 2x4 grid mosaic of debug pipeline stages."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize debug mosaic with configuration.

        Args:
            config: Configuration dictionary containing display settings
        """
        self.config = config
        debug_config = config["display"].get("debug_mosaic", {})
        self.tile_width = debug_config.get("tile_width", 320)
        self.tile_height = debug_config.get("tile_height", 240)
        logger.info(
            "DebugMosaic initialized: {w}x{h} tiles",
            w=self.tile_width,
            h=self.tile_height,
        )

    def create_mosaic(
        self, stages: Dict[str, np.ndarray], info: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """Create a 3x3 grid mosaic from pipeline stages with detailed labels.

        Args:
            stages: Dictionary with stage names as keys and frames as values.
                   Expected keys (some optional): original, frame_diff_raw, fg_mask,
                   fg_mask_clean, contours, detection, tracking, ptz_roi, final
            info: Optional dictionary with program state info:
                  - state: Current tracking state (DETECTION/TRACKING/LOST)
                  - tracker_type: Tracker type (CSRT/KCF)
                  - selected_id: Selected object ID
                  - detection_method: Detection method (background_subtraction/frame_difference)
                  - fps: Current FPS

        Returns:
            Mosaic frame of all stages in 3x3 grid (720x960 for 240x320 tiles)
        """
        if info is None:
            info = {}

        # Define layout: 3 rows × 3 columns
        layout = [
            ["original", "frame_diff_raw", "fg_mask"],
            ["fg_mask_clean", "contours", "detection"],
            ["tracking", "ptz_roi", "final"],
        ]

        rows = []
        for row in layout:
            row_frames = []
            for stage_name in row:
                if stage_name in stages:
                    frame = stages[stage_name]
                else:
                    # Create placeholder
                    frame = np.zeros(
                        (self.tile_height, self.tile_width, 3), dtype=np.uint8
                    )

                # Handle grayscale frames
                if len(frame.shape) == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

                # Resize to tile size
                resized = cv2.resize(frame, (self.tile_width, self.tile_height))

                # Add detailed label with background
                label = stage_name
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                font_color = (0, 255, 0)
                text_thickness = 1

                # Get text size for background
                text_size = cv2.getTextSize(label, font, font_scale, text_thickness)[0]
                text_width = text_size[0]
                text_height = text_size[1]
                padding = 4

                # Draw semi-transparent background for label
                cv2.rectangle(
                    resized,
                    (2, 2),
                    (2 + text_width + 2 * padding, 2 + text_height + 2 * padding),
                    (0, 0, 0),
                    -1,
                )

                # Draw label text
                cv2.putText(
                    resized,
                    label,
                    (2 + padding, 2 + text_height + padding),
                    font,
                    font_scale,
                    font_color,
                    text_thickness,
                )

                row_frames.append(resized)

            # Concatenate horizontally
            row_mosaic = np.hstack(row_frames)
            rows.append(row_mosaic)

        # Concatenate vertically
        mosaic = np.vstack(rows)

        # Add header with program state information
        header_height = 60
        header = np.zeros((header_height, mosaic.shape[1], 3), dtype=np.uint8)

        # Build state info text
        state = info.get("state", "UNKNOWN")
        tracker_type = info.get("tracker_type", "UNKNOWN")
        selected_id = info.get("selected_id", None)
        detection_method = info.get("detection_method", "background_subtraction")
        fps = info.get("fps", 0.0)

        # Line 1: State, Tracker, ID
        line1 = f"State: {state} | Tracker: {tracker_type} | ID: {selected_id if selected_id is not None else 'None'}"

        # Line 2: Detection method and FPS
        line2 = f"Method: {detection_method} | FPS: {fps:.1f}"

        # Draw header background
        cv2.rectangle(
            header, (0, 0), (header.shape[1], header_height), (30, 30, 30), -1
        )
        cv2.rectangle(
            header, (0, 0), (header.shape[1], header_height), (100, 100, 100), 1
        )

        # Draw text
        cv2.putText(
            header,
            line1,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            1,
        )
        cv2.putText(
            header,
            line2,
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 200),
            1,
        )

        # Combine header and mosaic
        final_mosaic = np.vstack([header, mosaic])

        return final_mosaic


def draw_contours_overlay(
    frame: np.ndarray,
    contours: List[np.ndarray],
    indices: List[int],
    color: Tuple[int, int, int] = (0, 255, 255),
    thickness: int = 2,
) -> np.ndarray:
    """Draw contours and their indices on a frame.

    Args:
        frame: Input frame to draw on
        contours: List of contour arrays
        indices: List of corresponding contour indices
        color: BGR color tuple for drawing (default: cyan)
        thickness: Line thickness for contour drawing

    Returns:
        Frame with contours drawn
    """
    output = frame.copy()
    for contour, idx in zip(contours, indices):
        cv2.drawContours(output, [contour], 0, color, thickness)

        # Draw index label at centroid
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(
                output,
                str(idx),
                (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
            )

    return output


def draw_ptz_roi_overlay(
    frame: np.ndarray,
    roi_rect: Optional[Tuple[int, int, int, int]],
    ptz_state: Dict[str, Any],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw PTZ ROI rectangle and state information on a frame.

    Args:
        frame: Input frame to draw on
        roi_rect: ROI rectangle (x, y, w, h) or None
        ptz_state: PTZ state dictionary with keys: pan, tilt, zoom
        color: BGR color tuple for drawing (default: green)
        thickness: Line thickness for rectangle

    Returns:
        Frame with PTZ ROI and state drawn
    """
    output = frame.copy()

    # Draw ROI rectangle if available
    if roi_rect is not None:
        x, y, w, h = roi_rect
        cv2.rectangle(output, (x, y), (x + w, y + h), color, thickness)

    # Draw PTZ state text
    pan = ptz_state.get("pan", 0.0)
    tilt = ptz_state.get("tilt", 0.0)
    zoom = ptz_state.get("zoom", 1.0)

    text = f"Pan: {pan:.1f}° Tilt: {tilt:.1f}° Zoom: {zoom:.2f}x"
    cv2.putText(
        output,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        1,
    )

    return output


def draw_search_area(
    frame: np.ndarray,
    center: Tuple[int, int],
    search_radius: int,
    elapsed_time: float,
    timeout: float,
    color: Tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2,
) -> np.ndarray:
    """Draw search area and recovery progress on a frame.

    Args:
        frame: Input frame to draw on
        center: Search center point (x, y)
        search_radius: Search radius in pixels
        elapsed_time: Elapsed time in recovery mode (seconds)
        timeout: Total timeout for recovery (seconds)
        color: BGR color tuple for drawing (default: red)
        thickness: Line thickness for circle

    Returns:
        Frame with search area and recovery progress drawn
    """
    output = frame.copy()

    # Draw search circle
    cv2.circle(output, center, search_radius, color, thickness)

    # Draw progress bar
    progress = min(elapsed_time / timeout, 1.0) if timeout > 0 else 0.0
    bar_width = 200
    bar_height = 30
    bar_x = frame.shape[1] // 2 - bar_width // 2
    bar_y = 10

    # Background
    cv2.rectangle(
        output,
        (bar_x, bar_y),
        (bar_x + bar_width, bar_y + bar_height),
        (50, 50, 50),
        -1,
    )

    # Progress fill
    fill_width = int(bar_width * progress)
    cv2.rectangle(
        output, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), color, -1
    )

    # Border
    cv2.rectangle(
        output,
        (bar_x, bar_y),
        (bar_x + bar_width, bar_y + bar_height),
        color,
        thickness,
    )

    # Time text
    time_text = f"Recovery: {elapsed_time:.1f}s / {timeout:.1f}s"
    cv2.putText(
        output,
        time_text,
        (bar_x, bar_y + bar_height + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        1,
    )

    return output

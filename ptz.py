"""
Pan-Tilt-Zoom controller for virtual PTZ simulation.

Maintains PTZ state and extracts regions of interest based on object position and size.
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple
from loguru import logger


class PTZController:
    """Virtual PTZ (Pan-Tilt-Zoom) controller."""

    def __init__(self, frame_shape: Tuple[int, int], config: Dict[str, Any]):
        """Initialize PTZ controller.

        Args:
            frame_shape: Frame shape (height, width)
            config: Configuration dictionary
        """
        self.frame_shape = frame_shape
        self.config = config['ptz_control']

        # PTZ state
        self.pan = 0.0
        self.tilt = 0.0
        self.zoom = 1.0

        logger.info("Initialized PTZ controller: {h}x{w}", h=frame_shape[0], w=frame_shape[1])

    def update(self, bbox: Tuple[int, int, int, int]) -> None:
        """Update PTZ based on object bounding box.

        Args:
            bbox: Object bounding box (x, y, w, h)
        """
        x, y, w, h = bbox
        cx = x + w / 2
        cy = y + h / 2

        height, width = self.frame_shape

        # Calculate errors (normalized, -1 to 1)
        error_x = (cx - width / 2) / width
        error_y = (cy - height / 2) / height

        # Apply deadband
        if abs(error_x) > 0.05:
            self.pan += error_x * self.config['pan_sensitivity']
        if abs(error_y) > 0.05:
            self.tilt += error_y * self.config['tilt_sensitivity']

        # Clamp pan/tilt
        pan_max = 90.0
        self.pan = np.clip(self.pan, -pan_max, pan_max)
        self.tilt = np.clip(self.tilt, -pan_max, pan_max)

        # Calculate zoom from object size
        object_area = w * h
        frame_area = height * width
        target_fraction = self.config['target_object_size']
        target_area = frame_area * target_fraction

        if target_area > 0 and object_area > 0:
            self.zoom = np.sqrt(target_area / object_area)

        # Clamp zoom
        self.zoom = np.clip(self.zoom, self.config['min_zoom'], self.config['max_zoom'])

    def extract_roi(self, frame: np.ndarray) -> np.ndarray:
        """Extract and scale ROI based on PTZ state.

        Args:
            frame: Input frame (BGR)

        Returns:
            PTZ-transformed frame (same size as input)
        """
        height, width = frame.shape[:2]

        # ROI size based on zoom
        roi_w = int(width / self.zoom)
        roi_h = int(height / self.zoom)

        # ROI center based on pan/tilt
        center_x = width // 2 + int(self.pan * width / 180)
        center_y = height // 2 + int(self.tilt * height / 180)

        # ROI top-left
        roi_x = max(0, min(center_x - roi_w // 2, width - roi_w))
        roi_y = max(0, min(center_y - roi_h // 2, height - roi_h))

        # Extract ROI
        roi = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

        # Resize to original frame size
        if roi.shape[0] > 0 and roi.shape[1] > 0:
            output = cv2.resize(roi, (width, height), interpolation=cv2.INTER_LINEAR)
        else:
            output = frame.copy()

        return output

    def get_roi_rect(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Get ROI rectangle coordinates for visualization.

        Returns:
            Tuple of ((x1, y1), (x2, y2)) for cv2.rectangle()
        """
        height, width = self.frame_shape

        # ROI size based on zoom
        roi_w = int(width / self.zoom)
        roi_h = int(height / self.zoom)

        # ROI center based on pan/tilt
        center_x = width // 2 + int(self.pan * width / 180)
        center_y = height // 2 + int(self.tilt * height / 180)

        # ROI corners
        roi_x = max(0, min(center_x - roi_w // 2, width - roi_w))
        roi_y = max(0, min(center_y - roi_h // 2, height - roi_h))

        return (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h)

    def reset(self) -> None:
        """Reset PTZ to neutral position."""
        self.pan = 0.0
        self.tilt = 0.0
        self.zoom = 1.0
        logger.info("Reset PTZ to neutral position")

    def get_state(self) -> Dict[str, float]:
        """Get current PTZ state.

        Returns:
            Dictionary with pan, tilt, zoom values
        """
        return {
            'pan': self.pan,
            'tilt': self.tilt,
            'zoom': self.zoom
        }

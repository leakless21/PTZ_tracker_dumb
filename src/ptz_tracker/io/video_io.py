"""
Video I/O module for PTZ Camera Object Tracking System.

Handles video capture, output writing, and frame processing operations.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from loguru import logger


class VideoIO:
    """Handles video input/output operations and frame processing."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize video I/O with configuration.

        Args:
            config: Video configuration dictionary
        """
        self.config = config
        self.cap: Optional[cv2.VideoCapture] = None
        self.out: Optional[cv2.VideoWriter] = None
        self.frame_width = 0
        self.frame_height = 0
        self.fps_in = 0.0
        self.frame_count = 0
        self.frame_idx = 0

    def open_input(self, input_path: str) -> bool:
        """Open video input file.

        Args:
            input_path: Path to input video file

        Returns:
            True if successful, False otherwise
        """
        if not Path(input_path).exists():
            logger.error("Input video file not found: {path}", path=input_path)
            return False

        self.cap = cv2.VideoCapture(input_path)
        if not self.cap.isOpened():
            logger.error("Failed to open video: {path}", path=input_path)
            return False

        # Get video properties
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps_in = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(
            "Opened video: {w}x{h} @ {fps:.1f} FPS, {count} frames",
            w=self.frame_width,
            h=self.frame_height,
            fps=self.fps_in,
            count=self.frame_count,
        )
        return True

    def setup_output(self, output_path: str) -> bool:
        """Setup video output writer.

        Args:
            output_path: Path for output video file

        Returns:
            True if successful, False otherwise
        """
        if not self.config.get("save_output", True):
            logger.info("Output saving disabled")
            return True

        try:
            codec = cv2.VideoWriter_fourcc(*self.config["output_codec"])
            self.out = cv2.VideoWriter(
                output_path, codec, self.fps_in, (self.frame_width, self.frame_height)
            )

            if not self.out.isOpened():
                logger.error("Failed to create output video writer")
                return False

            logger.info("Setup output video: {path}", path=output_path)
            return True

        except Exception as e:
            logger.error("Error setting up output: {err}", err=e)
            return False

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read next frame from video.

        Returns:
            Tuple of (success, frame) where frame is None if failed
        """
        if self.cap is None:
            return False, None

        ret, frame = self.cap.read()
        if ret:
            self.frame_idx += 1
        return ret, frame

    def write_frame(self, frame: np.ndarray) -> None:
        """Write frame to output video.

        Args:
            frame: Frame to write
        """
        if self.out is not None and self.config.get("save_output", True):
            self.out.write(frame)

    def loop_video(self) -> None:
        """Reset video to beginning for looping."""
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.frame_idx = 0
            logger.info("Looped video playback")

    def get_frame_info(self) -> Dict[str, Any]:
        """Get current frame information.

        Returns:
            Dictionary with frame info
        """
        return {
            "frame_idx": self.frame_idx,
            "frame_count": self.frame_count,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "fps": self.fps_in,
        }

    def close(self) -> None:
        """Close video streams."""
        if self.cap is not None:
            self.cap.release()
        if self.out is not None:
            self.out.release()
        logger.info("Closed video streams")
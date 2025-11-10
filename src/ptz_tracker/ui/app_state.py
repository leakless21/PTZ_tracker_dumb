"""
Application state management module for PTZ Camera Object Tracking System.

Manages the main application state, transitions, and orchestration.
"""

import time
from typing import Optional, Dict, Any
from loguru import logger

from ptz_tracker.core.tracker import State


class AppState:
    """Manages overall application state and coordination."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize application state.

        Args:
            config: Application configuration
        """
        self.config = config
        self.tracking_state = State.DETECTION
        self.selected_id: Optional[int] = None
        self.show_debug = config["display"]["show_debug_mosaic"]
        self.paused = False
        self.fps = 0.0
        self.fps_clock = 0.0
        self.last_frame_time = 0.0

    def transition_to_detection(self) -> None:
        """Transition to detection state."""
        self.tracking_state = State.DETECTION
        self.selected_id = None
        logger.info("Transitioned to DETECTION state")

    def transition_to_tracking(self, object_id: int) -> None:
        """Transition to tracking state.

        Args:
            object_id: ID of object to track
        """
        self.tracking_state = State.TRACKING
        self.selected_id = object_id
        logger.info("Transitioned to TRACKING state for object {id}", id=object_id)

    def transition_to_lost(self) -> None:
        """Transition to lost state."""
        self.tracking_state = State.LOST
        logger.info("Transitioned to LOST state")

    def reset(self) -> None:
        """Reset application state."""
        self.transition_to_detection()
        self.show_debug = self.config["display"]["show_debug_mosaic"]
        self.paused = False
        logger.info("Application state reset")

    def toggle_debug(self) -> None:
        """Toggle debug display."""
        self.show_debug = not self.show_debug
        logger.info("Debug display: {state}", state="enabled" if self.show_debug else "disabled")

    def toggle_pause(self) -> None:
        """Toggle pause state."""
        self.paused = not self.paused
        logger.info("Paused: {state}", state=self.paused)

    def update_fps(self) -> None:
        """Update FPS calculation."""
        current_time = time.time()
        if self.fps_clock > 0:
            elapsed = current_time - self.fps_clock
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / (elapsed + 1e-6))
        self.fps_clock = current_time

    def should_process_frame(self) -> bool:
        """Check if frame should be processed (not paused).

        Returns:
            True if should process frame
        """
        return not self.paused

    def get_display_info(self) -> Dict[str, Any]:
        """Get information for display overlay.

        Returns:
            Dictionary with display information
        """
        return {
            "state": self.tracking_state,
            "selected_id": self.selected_id,
            "fps": self.fps,
            "show_debug": self.show_debug,
            "paused": self.paused,
        }
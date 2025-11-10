"""
Input handling module for PTZ Camera Object Tracking System.

Manages keyboard input, ID buffer, and user interaction.
"""

import time
from typing import Optional, Tuple
from loguru import logger


class InputHandler:
    """Handles keyboard input and user interaction."""

    def __init__(self):
        """Initialize input handler."""
        self.id_input_buffer = ""  # Buffer for multi-digit tracker ID input
        self.last_input_time = 0.0  # Time of last keyboard input

    def process_key(self, key: int) -> Tuple[Optional[str], Optional[int], bool]:
        """Process a keyboard key press.

        Args:
            key: Key code from cv2.waitKey()

        Returns:
            Tuple of (action, object_id, should_quit)
            - action: 'select', 'reset', 'toggle_debug', 'pause', None
            - object_id: Selected object ID if selecting, None otherwise
            - should_quit: True if should quit application
        """
        action = None
        object_id = None
        should_quit = False

        if key == ord("q") or key == 27:  # 'Q' or ESC
            should_quit = True
            logger.info("Quit requested")

        elif key == ord("r") or key == ord("R"):  # 'R'
            action = "reset"
            self._clear_buffer()
            logger.info("Reset requested")

        elif key == ord("d") or key == ord("D"):  # 'D'
            action = "toggle_debug"
            logger.info("Debug toggle requested")

        elif key == ord(" "):  # Space
            action = "pause"
            logger.info("Pause toggle requested")

        elif key == 8 or key == 255:  # Backspace (8 is common, 255 is fallback)
            # Clear the input buffer
            if key == 8:  # Only on actual backspace, not on idle frames
                self.id_input_buffer = self.id_input_buffer[:-1] if self.id_input_buffer else ""
                self.last_input_time = time.time()
                logger.debug("Input buffer: '{buffer}'", buffer=self.id_input_buffer)

        elif key == 13 or key == 10:  # Enter / Return
            # Confirm the ID selection
            if self.id_input_buffer:
                try:
                    object_id = int(self.id_input_buffer)
                    action = "select"
                    self._clear_buffer()
                    logger.info("ID selection confirmed: {id}", id=object_id)
                except ValueError:
                    logger.warning("Invalid ID input: {buffer}", buffer=self.id_input_buffer)
                    self._clear_buffer()
            else:
                logger.debug("Enter pressed with empty buffer")

        elif ord("0") <= key <= ord("9"):  # Number keys - add to buffer
            self.id_input_buffer += chr(key)
            self.last_input_time = time.time()
            logger.debug("ID input buffer: '{buffer}'", buffer=self.id_input_buffer)

        return action, object_id, should_quit

    def _clear_buffer(self) -> None:
        """Clear the ID input buffer."""
        self.id_input_buffer = ""
        self.last_input_time = 0.0

    def get_buffer_display(self) -> str:
        """Get current buffer content for display.

        Returns:
            String to display current buffer state
        """
        if self.id_input_buffer:
            return f"Enter ID: {self.id_input_buffer} (Press ENTER to confirm)"
        return ""

    def has_recent_input(self, timeout: float = 5.0) -> bool:
        """Check if there was recent input activity.

        Args:
            timeout: Time in seconds to consider as recent

        Returns:
            True if there was recent input
        """
        return (time.time() - self.last_input_time) < timeout
"""
PTZ Camera Object Tracking System - Main Application

Detects multiple moving objects, allows user selection, tracks selected object,
and applies virtual PTZ control to keep it centered.
"""

import cv2
import numpy as np
import yaml
import argparse
import os
import time
from pathlib import Path
from loguru import logger
from typing import Optional, Dict, Any

from tracker import ObjectTracker, State
from ptz import PTZController
from debug_view import DebugMosaic, draw_ptz_roi_overlay, draw_search_area, draw_contours_overlay


def setup_logging() -> None:
    """Configure Loguru logging."""
    os.makedirs("logs", exist_ok=True)
    logger.remove()  # Remove default handler
    logger.add(
        "logs/ptz_tracker.log",
        rotation="10 MB",
        retention=5,
        enqueue=True,
        backtrace=True,
        diagnose=False,
    )
    logger.add(lambda msg: print(msg, end=""), colorize=True)
    logger.info("Logging initialized")


def load_config(config_path: str = "config.yaml") -> Optional[Dict[str, Any]]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary, or None if error
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info("Loaded config from {path}", path=config_path)
        return config
    except FileNotFoundError:
        logger.error("Config file not found: {path}", path=config_path)
        return None
    except yaml.YAMLError as e:
        logger.error("Error parsing YAML: {err}", err=e)
        return None


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration values.

    Args:
        config: Configuration dictionary

    Returns:
        True if valid, False otherwise
    """
    errors = []

    # Check video input
    input_path = config['video']['input_path']
    if not Path(input_path).exists():
        errors.append(f"Video input file not found: {input_path}")

    # Check algorithm
    algo = config['background_subtraction']['algorithm']
    if algo not in ['MOG2', 'KNN']:
        errors.append(f"background_subtraction.algorithm must be 'MOG2' or 'KNN', got: {algo}")

    # Check tracker
    tracker = config['tracking']['tracker']
    if tracker not in ['KCF', 'CSRT']:
        errors.append(f"tracking.tracker must be 'KCF' or 'CSRT', got: {tracker}")

    # Check numeric ranges
    if config['ptz_control']['target_object_size'] < 0.1 or config['ptz_control']['target_object_size'] > 0.9:
        errors.append("ptz_control.target_object_size should be between 0.1 and 0.9")

    if errors:
        for error in errors:
            logger.error(error)
        return False

    return True


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='PTZ Object Tracker')
    parser.add_argument('--config', default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--input', help='Input video file')
    parser.add_argument('--output', help='Output video file')
    parser.add_argument('--tracker', choices=['KCF', 'CSRT'],
                       help='Tracker type')
    parser.add_argument('--no-display', action='store_true',
                       help='Disable display window')
    return parser.parse_args()


def draw_status_overlay(frame: np.ndarray, state: State, ptz_state: Dict[str, float],
                       fps: float, selected_id: Optional[int] = None) -> np.ndarray:
    """Draw status overlay on frame.

    Args:
        frame: Input frame
        state: Current tracking state
        ptz_state: PTZ state
        fps: Current FPS
        selected_id: Selected object ID

    Returns:
        Frame with overlay
    """
    overlay = frame.copy()

    # State text
    state_text = state.name
    if selected_id is not None:
        state_text += f" (ID: {selected_id})"

    cv2.putText(overlay, f"State: {state_text}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # FPS
    cv2.putText(overlay, f"FPS: {fps:.1f}", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Controls hint
    hint = "0-9:Select  R:Reset  D:Debug  Space:Pause  Q:Quit"
    cv2.putText(overlay, hint, (10, frame.shape[0] - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return overlay


def main() -> None:
    """Main application loop."""
    # Setup
    setup_logging()
    args = parse_arguments()

    # Load config
    config = load_config(args.config)
    if config is None:
        logger.error("Failed to load configuration")
        return

    # Override with command-line args
    if args.input:
        config['video']['input_path'] = args.input
    if args.output:
        config['video']['output_path'] = args.output
    if args.tracker:
        config['tracking']['tracker'] = args.tracker
    if args.no_display:
        config['display']['show_window'] = False

    # Validate config
    if not validate_config(config):
        logger.error("Configuration validation failed")
        return

    # Open video
    input_path = config['video']['input_path']
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        logger.error("Failed to open video: {path}", path=input_path)
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info("Video: {w}x{h} @ {fps:.1f} FPS, {count} frames",
                w=frame_width, h=frame_height, fps=fps_in, count=frame_count)

    # Setup video writer
    output_path = config['video']['output_path']
    codec = cv2.VideoWriter_fourcc(*config['video']['output_codec'])
    out = cv2.VideoWriter(output_path, codec, fps_in, (frame_width, frame_height))

    # Initialize modules
    tracker = ObjectTracker(config, (frame_height, frame_width))
    ptz = PTZController((frame_height, frame_width), config)
    debug_mosaic = DebugMosaic(config)

    # State variables
    state = State.DETECTION
    selected_id = None
    show_debug = config['display']['show_debug_mosaic']
    paused = False
    frame_idx = 0
    fps_clock = 0
    fps = 0.0
    frame = None
    output_frame = None
    fg_mask = None
    cleaned_mask = None
    detection_frame = None
    tracking_frame = None
    ptz_roi_overlay_frame = None
    contours_frame = None

    logger.info("Starting main loop")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                if config['video']['loop_playback']:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    logger.info("Looping video")
                    continue
                else:
                    logger.info("End of video")
                    break

            frame_idx += 1

        # FPS calculation
        fps_clock = time.time()

        # Process frame
        if state == State.DETECTION:
            output_frame, cleaned_mask, fg_mask = tracker.update_detection_mode(frame)
            ptz.reset()

            # Generate contours overlay for debug mosaic
            valid_contours, valid_indices = tracker.detect_contours(cleaned_mask)
            contours_frame = draw_contours_overlay(frame.copy(), valid_contours, valid_indices)

            # Store for debug mosaic
            detection_frame = output_frame.copy()
            tracking_frame = None
            ptz_roi_overlay_frame = None

        elif state == State.TRACKING:
            if tracker.single_tracker is None:
                logger.warning("Tracker not initialized, returning to DETECTION")
                state = State.DETECTION
                selected_id = None
                continue

            success, bbox = tracker.update_tracking_mode(frame)

            if success and bbox:
                # Update PTZ
                ptz.update(bbox)

                # Draw tracking visualization
                output_frame = frame.copy()
                x, y, w, h = bbox
                cv2.rectangle(output_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(output_frame, f"ID: {selected_id}", (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Extract PTZ ROI (for actual tracking - this is the final output)
                tracking_with_roi = ptz.extract_roi(output_frame)

                # Keep pre-ROI version for tracking_frame in mosaic (to show what was tracked)
                tracking_frame = output_frame.copy()

                # Update output_frame to the ROI version for final display
                output_frame = tracking_with_roi
            else:
                # Transition to LOST
                logger.info("Tracking lost, entering recovery mode")
                state = State.LOST
                tracker.loss_timestamp = time.time()
                continue

            # Prepare debug frames (show actual pipeline state, not None)
            detection_frame = None  # Not in tracking mode
            fg_mask = None  # Not needed in pure tracking mode
            cleaned_mask = None  # Not needed in pure tracking mode
            contours_frame = None  # Not needed in pure tracking mode
            ptz_roi_overlay_frame = draw_ptz_roi_overlay(frame, ptz.get_roi_rect(), ptz.get_state())

        elif state == State.LOST:
            # Attempt recovery
            elapsed_time = time.time() - tracker.loss_timestamp
            timeout = config['tracking']['recovery']['timeout']

            if elapsed_time > timeout:
                logger.warning("Recovery timeout, returning to DETECTION")
                state = State.DETECTION
                selected_id = None
                tracker.reset_to_detection_mode()
                continue

            # Try to redetect
            new_bbox = tracker.attempt_redetection(frame)

            if new_bbox:
                # Recovery successful
                logger.info("Object redetected!")
                if tracker.initialize_single_tracker(frame, new_bbox):
                    state = State.TRACKING
                    selected_id = tracker.tracked_object_id
                else:
                    logger.warning("Failed to initialize tracker, returning to DETECTION")
                    state = State.DETECTION
                    selected_id = None
                continue
            else:
                # Still searching
                output_frame = frame.copy()
                last_center = (tracker.last_bbox[0] + tracker.last_bbox[2] // 2,
                              tracker.last_bbox[1] + tracker.last_bbox[3] // 2)
                output_frame = draw_search_area(output_frame, last_center,
                                              config['tracking']['recovery']['search_radius'],
                                              elapsed_time, timeout)

                # Prepare debug frames
                tracking_frame = output_frame.copy()
                detection_frame = None
                ptz_roi_overlay_frame = None
                contours_frame = None

                # Continue searching
                fg_mask = tracker.update_background_subtraction(frame)
                cleaned_mask = tracker.clean_mask(fg_mask)

        # Create debug mosaic if enabled
        if show_debug:
            stages = {
                'original': frame,
                'fg_mask_raw': fg_mask if fg_mask is not None else np.zeros_like(frame[:, :, 0]),
                'fg_mask_clean': cleaned_mask if cleaned_mask is not None else np.zeros_like(frame[:, :, 0]),
                'contours': contours_frame if contours_frame is not None else frame,
                'detection': detection_frame if detection_frame is not None else frame,
                'tracking': tracking_frame if tracking_frame is not None else frame,
                'ptz_roi': ptz_roi_overlay_frame if ptz_roi_overlay_frame is not None else frame,
                'final': output_frame
            }
            debug_frame = debug_mosaic.create_mosaic(stages)
            cv2.imshow("Debug Pipeline", debug_frame)

        # Add status overlay
        output_frame = draw_status_overlay(output_frame, state, ptz.get_state(), fps, selected_id)

        # Display
        if config['display']['show_window']:
            cv2.imshow(config['display'].get('window_name', 'PTZ Tracker'), output_frame)

        # Save output
        if config['video']['save_output']:
            out.write(output_frame)

        # Calculate FPS
        elapsed = time.time() - fps_clock
        fps = 0.9 * fps + 0.1 * (1.0 / (elapsed + 1e-6))

        # Keyboard input
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:  # 'Q' or ESC
            logger.info("Quit requested")
            break

        elif key == ord('r') or key == ord('R'):  # 'R'
            logger.info("Reset requested")
            state = State.DETECTION
            selected_id = None
            tracker.reset_to_detection_mode()
            ptz.reset()

        elif key == ord('d') or key == ord('D'):  # 'D'
            show_debug = not show_debug
            logger.info("Debug mosaic: {state}", state="enabled" if show_debug else "disabled")
            if not show_debug:
                cv2.destroyWindow("Debug Pipeline")

        elif key == ord(' '):  # Space
            paused = not paused
            logger.info("Paused: {state}", state=paused)

        elif ord('0') <= key <= ord('9'):  # Number keys
            selected_id = key - ord('0')
            bbox = tracker.lock_onto_object(selected_id)

            if bbox:
                if tracker.initialize_single_tracker(frame, bbox):
                    state = State.TRACKING
                    logger.info("Locked onto object ID: {id}", id=selected_id)
                else:
                    logger.warning("Failed to initialize tracker")
                    selected_id = None
            else:
                logger.warning("Object ID {id} not found", id=selected_id)
                selected_id = None

        # Periodic logging
        if frame_idx % 300 == 0:  # Every 10 seconds at 30 FPS
            logger.debug("Frame {idx}/{total}, FPS: {fps:.1f}, State: {state}",
                        idx=frame_idx, total=frame_count, fps=fps, state=state.name)

    # Cleanup
    logger.info("Closing application")
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    logger.info("Application finished - {total} frames processed", total=frame_idx)


if __name__ == '__main__':
    # Add numpy to imports at top level for type hints
    import numpy as np
    main()

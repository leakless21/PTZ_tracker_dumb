"""
PTZ Camera Object Tracking System - Main Application

Detects multiple moving objects, allows user selection, tracks selected object,
and applies virtual PTZ control to keep it centered.
"""

import cv2
import numpy as np
import os
import time
from loguru import logger
from typing import Optional

from ptz_tracker.core.tracker import ObjectTracker, State
from ptz_tracker.core.ptz import PTZController
from ptz_tracker.ui.debug_view import (
    DebugMosaic,
    draw_ptz_roi_overlay,
    draw_search_area,
    draw_contours_overlay,
)
from ptz_tracker.io.video_io import VideoIO
from ptz_tracker.ui.input_handler import InputHandler
from ptz_tracker.io.config_manager import ConfigManager
from ptz_tracker.ui.app_state import AppState


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


def draw_status_overlay(
    frame: np.ndarray,
    state: State,
    ptz_state: dict,
    fps: float,
    selected_id: Optional[int] = None,
    id_input_buffer: str = "",
) -> np.ndarray:
    """Draw status overlay on frame.

    Args:
        frame: Input frame
        state: Current tracking state
        ptz_state: PTZ state
        fps: Current FPS
        selected_id: Selected object ID
        id_input_buffer: Current ID input buffer

    Returns:
        Frame with overlay
    """
    overlay = frame.copy()

    # State text
    state_text = state.name
    if selected_id is not None:
        state_text += f" (ID: {selected_id})"

    cv2.putText(
        overlay,
        f"State: {state_text}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    # FPS
    cv2.putText(
        overlay,
        f"FPS: {fps:.1f}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    # ID input buffer display (if actively typing)
    if id_input_buffer:
        cv2.putText(
            overlay,
            f"Enter ID: {id_input_buffer} (Press ENTER to confirm)",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

    # Controls hint
    hint = "Digits:Enter ID  ENTER:Confirm  BACKSPACE:Clear  R:Reset  D:Debug  Space:Pause  Q:Quit"
    cv2.putText(
        overlay,
        hint,
        (10, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
    )

    return overlay


def main() -> None:
    """Main application loop."""
    # Setup
    setup_logging()

    # Initialize managers
    config_manager = ConfigManager()
    args = config_manager.parse_arguments()

    # Load and validate config
    config = config_manager.load_config(args.config)
    if config is None:
        logger.error("Failed to load configuration")
        return

    config_manager.apply_arguments(args)

    if not config_manager.validate_config():
        logger.error("Configuration validation failed")
        return

    # Initialize video I/O
    video_io = VideoIO(config["video"])
    if not video_io.open_input(config["video"]["input_path"]):
        return

    if not video_io.setup_output(config["video"]["output_path"]):
        return

    # Initialize components
    frame_info = video_io.get_frame_info()
    tracker = ObjectTracker(
        config, (frame_info["frame_height"], frame_info["frame_width"])
    )
    ptz = PTZController((frame_info["frame_height"], frame_info["frame_width"]), config)
    debug_mosaic = DebugMosaic(config)

    # Initialize state management
    app_state = AppState(config)
    input_handler = InputHandler()

    # Processing variables
    frame = None
    output_frame = None
    fg_mask = None
    cleaned_mask = None
    detection_frame = None
    tracking_frame = None
    ptz_roi_overlay_frame = None
    contours_frame = None
    frame_diff_raw = None

    logger.info("Starting main loop")

    while True:
        # Read frame if not paused
        if app_state.should_process_frame():
            ret, frame = video_io.read_frame()
            if not ret:
                if config["video"]["loop_playback"]:
                    video_io.loop_video()
                    continue
                else:
                    logger.info("End of video")
                    break

        if frame is None:
            continue

        # Update FPS
        app_state.update_fps()

        # Process frame based on state
        if app_state.tracking_state == State.DETECTION:
            output_frame, cleaned_mask, fg_mask = tracker.update_detection_mode(frame)
            ptz.reset()

            # Generate contours overlay from main pipeline
            valid_contours, valid_indices = tracker.detect_contours(cleaned_mask)
            contours_frame = draw_contours_overlay(
                frame.copy(), valid_contours, list(valid_indices)
            )

            # Detection frame is the output with detections drawn
            detection_frame = output_frame.copy()
            tracking_frame = frame.copy()  # Use clean frame placeholder
            ptz_roi_overlay_frame = frame.copy()  # No PTZ in detection mode

        elif app_state.tracking_state == State.TRACKING:
            if tracker.single_tracker is None:
                logger.warning("Tracker not initialized, returning to DETECTION")
                app_state.transition_to_detection()
                tracker.reset_to_detection_mode()
                ptz.reset()
                continue

            # Extract PTZ view FIRST (simulates what the PTZ camera sees)
            ptz_frame = ptz.extract_roi(frame)

            # Track on the PTZ output (what the camera actually sees)
            success, bbox = tracker.update_tracking_mode(ptz_frame)

            if success and bbox:
                # Optional: reinforce tracker using detection at interval
                reinf_cfg = config["tracking"].get("reinforcement", {})
                if reinf_cfg.get("enabled", False):
                    interval = int(reinf_cfg.get("interval", 5))
                    if interval > 0 and (frame_info["frame_idx"] % interval == 0):
                        corrected = tracker.reinforce_with_detection(ptz_frame, bbox)
                        if corrected is not None:
                            if reinf_cfg.get("reinitialize", True):
                                # Reinitialize the tracker to the corrected bbox for robustness
                                tracker.initialize_single_tracker(ptz_frame, corrected)
                            bbox = corrected

                # Update PTZ based on object position in PTZ view
                ptz.update(bbox)

                # Draw tracking visualization on PTZ frame (this becomes output_frame)
                output_frame = ptz_frame.copy()
                x, y, w, h = bbox
                cv2.rectangle(output_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(
                    output_frame,
                    f"ID: {app_state.selected_id}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

                # Generate debug frames from main pipeline (reuse tracker internals)
                # These are generated by the reinforcement or tracking operations
                frame_diff_raw = None
                if tracker.use_frame_diff_in_tracking:
                    fd_threshold = config["tracking"].get(
                        "frame_difference_threshold", 30
                    )
                    # Compute frame difference mask and extract raw diff
                    result = tracker.compute_frame_difference_mask(
                        ptz_frame, threshold=fd_threshold, return_raw_diff=True
                    )
                    cleaned_mask, frame_diff_raw = result
                    fg_mask = cleaned_mask  # Frame difference is the mask
                else:
                    fg_mask = tracker.update_background_subtraction(ptz_frame)
                    cleaned_mask = tracker.clean_mask(fg_mask)

                # Generate contours and detection overlay from pipeline
                vc, vi = tracker.detect_contours(cleaned_mask)
                contours_frame = draw_contours_overlay(ptz_frame.copy(), vc, list(vi))

                detection_frame = ptz_frame.copy()
                for contour in vc:
                    dx, dy, dw, dh = cv2.boundingRect(contour)
                    cv2.rectangle(
                        detection_frame, (dx, dy), (dx + dw, dy + dh), (255, 255, 0), 2
                    )

                # Tracking frame is the output with tracker bbox
                tracking_frame = output_frame.copy()
            else:
                # Transition to LOST
                logger.info("Tracking lost, entering recovery mode")
                app_state.transition_to_lost()
                continue

            # Prepare PTZ ROI overlay on the original frame
            roi_rect_tuple = ptz.get_roi_rect()
            (x1, y1), (x2, y2) = roi_rect_tuple
            roi_rect = (x1, y1, x2 - x1, y2 - y1)
            ptz_roi_overlay_frame = draw_ptz_roi_overlay(
                frame, roi_rect, ptz.get_state()
            )

        elif app_state.tracking_state == State.LOST:
            # Attempt recovery
            elapsed_time = time.time() - tracker.loss_timestamp
            timeout = config["tracking"]["recovery"]["timeout"]

            if elapsed_time > timeout:
                logger.warning("Recovery timeout, returning to DETECTION")
                app_state.transition_to_detection()
                tracker.reset_to_detection_mode()
                ptz.reset()
                continue

            # Extract PTZ view (continue simulating PTZ camera)
            ptz_frame = ptz.extract_roi(frame)

            # Try to redetect on PTZ frame - this internally computes mask and contours
            new_bbox = tracker.attempt_redetection(ptz_frame)

            if new_bbox:
                # Recovery successful
                logger.info("Object redetected!")
                if tracker.initialize_single_tracker(ptz_frame, new_bbox):
                    if tracker.tracked_object_id is not None:
                        app_state.transition_to_tracking(tracker.tracked_object_id)
                    else:
                        logger.warning("Tracker initialized but no object ID set")
                        app_state.transition_to_detection()
                else:
                    logger.warning(
                        "Failed to initialize tracker, returning to DETECTION"
                    )
                    app_state.transition_to_detection()
                    ptz.reset()
                continue
            else:
                # Still searching - show PTZ output with search indicator
                output_frame = ptz_frame.copy()
                if tracker.last_bbox is not None:
                    last_center = (
                        tracker.last_bbox[0] + tracker.last_bbox[2] // 2,
                        tracker.last_bbox[1] + tracker.last_bbox[3] // 2,
                    )
                    output_frame = draw_search_area(
                        output_frame,
                        last_center,
                        config["tracking"]["recovery"]["search_radius"],
                        elapsed_time,
                        timeout,
                    )

                # Generate debug frames from main pipeline (reuse from redetection attempt)
                if tracker.use_frame_diff_in_tracking:
                    fd_threshold = config["tracking"].get(
                        "frame_difference_threshold", 30
                    )
                    cleaned_mask = tracker.compute_frame_difference_mask(
                        ptz_frame, threshold=fd_threshold
                    )
                    fg_mask = cleaned_mask
                else:
                    fg_mask = tracker.update_background_subtraction(ptz_frame)
                    cleaned_mask = tracker.clean_mask(fg_mask)

                vc, vi = tracker.detect_contours(cleaned_mask)
                contours_frame = draw_contours_overlay(ptz_frame.copy(), vc, list(vi))

                detection_frame = ptz_frame.copy()
                for contour in vc:
                    dx, dy, dw, dh = cv2.boundingRect(contour)
                    cv2.rectangle(
                        detection_frame, (dx, dy), (dx + dw, dy + dh), (255, 255, 0), 2
                    )

                tracking_frame = output_frame.copy()

                roi_rect_tuple = ptz.get_roi_rect()
                (x1, y1), (x2, y2) = roi_rect_tuple
                roi_rect = (x1, y1, x2 - x1, y2 - y1)
                ptz_roi_overlay_frame = draw_ptz_roi_overlay(
                    frame, roi_rect, ptz.get_state()
                )

        # Create debug mosaic if enabled
        if app_state.show_debug and config["display"].get("show_debug_mosaic", True):
            # Prepare display info
            display_info = app_state.get_display_info()

            # Determine detection method
            detection_method = (
                "frame_difference"
                if tracker.use_frame_diff_in_tracking
                else "background_subtraction"
            )

            # Build info dict for debug mosaic
            mosaic_info = {
                "state": app_state.tracking_state.name,
                "tracker_type": tracker.tracker_type,
                "selected_id": display_info["selected_id"],
                "detection_method": detection_method,
                "fps": display_info["fps"],
            }

            # Build stages dict from main pipeline frames
            stages = {
                "original": frame,
                "frame_diff_raw": frame_diff_raw
                if frame_diff_raw is not None
                else np.zeros_like(frame[:, :, 0]),
                "fg_mask": fg_mask
                if fg_mask is not None
                else np.zeros_like(frame[:, :, 0]),
                "fg_mask_clean": cleaned_mask
                if cleaned_mask is not None
                else np.zeros_like(frame[:, :, 0]),
                "contours": contours_frame if contours_frame is not None else frame,
                "detection": detection_frame if detection_frame is not None else frame,
                "tracking": tracking_frame if tracking_frame is not None else frame,
                "ptz_roi": ptz_roi_overlay_frame
                if ptz_roi_overlay_frame is not None
                else frame,
                "final": output_frame if output_frame is not None else frame,
            }
            debug_frame = debug_mosaic.create_mosaic(stages, mosaic_info)
            cv2.imshow("Debug Pipeline", debug_frame)

        # Add status overlay
        display_info = app_state.get_display_info()
        if output_frame is not None:
            output_frame = draw_status_overlay(
                output_frame,
                app_state.tracking_state,
                ptz.get_state(),
                display_info["fps"],
                display_info["selected_id"],
                input_handler.get_buffer_display(),
            )

        # Display
        if config["display"]["show_window"] and output_frame is not None:
            cv2.imshow(
                config["display"].get("window_name", "PTZ Tracker"), output_frame
            )

        # Save output
        if output_frame is not None:
            video_io.write_frame(output_frame)

        # Keyboard input (waitKey delay in milliseconds)
        key = cv2.waitKey(30) & 0xFF

        action, object_id, should_quit = input_handler.process_key(key)

        if should_quit:
            break

        elif action == "reset":
            app_state.reset()
            tracker.reset_to_detection_mode()
            ptz.reset()
            logger.info("System reset to DETECTION mode")

        elif action == "toggle_debug":
            app_state.toggle_debug()
            if not app_state.show_debug:
                try:
                    cv2.destroyWindow("Debug Pipeline")
                except Exception:
                    pass  # Window may not exist

        elif action == "pause":
            app_state.toggle_pause()

        elif action == "select" and object_id is not None:
            bbox = tracker.lock_onto_object(object_id)
            if bbox:
                if tracker.initialize_single_tracker(frame, bbox):
                    app_state.transition_to_tracking(object_id)
                    logger.info("Locked onto object ID: {id}", id=object_id)
                else:
                    logger.warning("Failed to initialize tracker")
            else:
                logger.warning("Object ID {id} not found", id=object_id)

        # Periodic logging
        frame_info = video_io.get_frame_info()
        if frame_info["frame_idx"] % 300 == 0:  # Every 10 seconds at 30 FPS
            logger.debug(
                "Frame {idx}/{total}, FPS: {fps:.1f}, State: {state}",
                idx=frame_info["frame_idx"],
                total=frame_info["frame_count"],
                fps=display_info["fps"],
                state=app_state.tracking_state.name,
            )

    # Cleanup
    logger.info("Closing application")
    video_io.close()
    cv2.destroyAllWindows()

    frame_info = video_io.get_frame_info()
    logger.info(
        "Application finished - {total} frames processed", total=frame_info["frame_idx"]
    )


if __name__ == "__main__":
    # Add numpy to imports at top level for type hints
    import numpy as np

    main()

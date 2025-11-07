"""
Object detection and tracking module.

Handles background subtraction, contour detection, Norfair multi-object tracking,
and CSRT/KCF single-object tracking.
"""

import cv2
import numpy as np
from norfair import Detection, Tracker
from loguru import logger
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum


class State(Enum):
    """Tracking state machine."""
    DETECTION = 0   # Multi-object mode (Norfair)
    TRACKING = 1    # Single-object mode (CSRT/KCF)
    LOST = 2        # Recovery mode


class ObjectTracker:
    """Multi-object detection and single-object tracking controller."""

    def __init__(self, config: Dict[str, Any], frame_shape: Tuple[int, int]):
        """Initialize tracker with configuration.

        Args:
            config: Configuration dictionary
            frame_shape: Frame shape (height, width)
        """
        self.config = config
        self.frame_shape = frame_shape
        self.state = State.DETECTION

        # Background subtraction setup
        bg_config = config['background_subtraction']
        algorithm = bg_config['algorithm']

        if algorithm == 'MOG2':
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=bg_config['history'],
                varThreshold=bg_config['var_threshold'],
                detectShadows=bg_config['detect_shadows']
            )
        elif algorithm == 'KNN':
            self.bg_subtractor = cv2.createBackgroundSubtractorKNN(
                history=bg_config['history'],
                dist2Threshold=bg_config['dist2_threshold'],
                detectShadows=bg_config['detect_shadows']
            )
        else:
            raise ValueError(f"Unknown background subtraction algorithm: {algorithm}")

        logger.info("Initialized background subtractor: {algo}", algo=algorithm)

        # Morphological kernel setup
        kernel_size = config['mask_processing']['kernel_size']
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        # Norfair tracker for multi-object detection
        norfair_cfg = config['tracking']['norfair']
        self.norfair_tracker = Tracker(
            distance_function="mean_euclidean",
            distance_threshold=norfair_cfg['distance_threshold'],
            hit_counter_max=norfair_cfg['hit_counter_max'],
            initialization_delay=norfair_cfg['initialization_delay']
        )

        # Single-object tracker (CSRT/KCF)
        self.single_tracker = None
        self.tracker_type = config['tracking']['tracker']
        logger.info("Using {tracker} tracker", tracker=self.tracker_type)

        # Tracking state
        self.last_bbox = None
        self.tracked_object_id = None
        self.loss_timestamp = 0.0
        self.tracked_objects = []

    def update_background_subtraction(self, frame: np.ndarray) -> np.ndarray:
        """Apply background subtraction to frame.

        Args:
            frame: Input frame (BGR)

        Returns:
            Foreground mask (binary)
        """
        fg_mask = self.bg_subtractor.apply(frame)
        return fg_mask

    def clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """Post-process foreground mask with morphological operations.

        Args:
            mask: Raw foreground mask

        Returns:
            Cleaned binary mask
        """
        # Opening: remove noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=1)

        # Closing: fill holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=1)

        # Binary threshold
        threshold_value = self.config['mask_processing']['threshold_value']
        _, mask = cv2.threshold(mask, threshold_value, 255, cv2.THRESH_BINARY)

        return mask

    def detect_contours(self, mask: np.ndarray) -> Tuple[List[np.ndarray], set]:
        """Detect and filter contours from mask.

        Args:
            mask: Binary foreground mask

        Returns:
            Tuple of (valid_contours, valid_indices_set)
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        frame_shape = self.frame_shape
        frame_area = frame_shape[0] * frame_shape[1]

        cfg = self.config['object_detection']
        min_area = cfg['min_area']
        max_area = frame_area * cfg['max_area_fraction']
        min_aspect = cfg.get('min_aspect_ratio', 0.0)
        max_aspect = cfg.get('max_aspect_ratio', float('inf'))

        valid_contours = []
        valid_indices = set()

        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)

            # Area filtering
            if area < min_area or area > max_area:
                continue

            # Aspect ratio filtering (optional)
            x, y, w, h = cv2.boundingRect(contour)
            if w == 0 or h == 0:
                continue

            aspect_ratio = w / h
            if not (min_aspect <= aspect_ratio <= max_aspect):
                continue

            valid_contours.append(contour)
            valid_indices.add(i)

        logger.debug("Detected {total} contours, {valid} valid", total=len(contours), valid=len(valid_contours))

        return valid_contours, valid_indices

    def update_detection_mode(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Update detection mode (multi-object tracking with Norfair).

        Args:
            frame: Input frame (BGR)

        Returns:
            Tuple of (frame_with_detections, cleaned_mask, raw_fg_mask)
        """
        # Background subtraction
        fg_mask = self.update_background_subtraction(frame)
        cleaned_mask = self.clean_mask(fg_mask)

        # Contour detection
        valid_contours, _ = self.detect_contours(cleaned_mask)

        # Convert to Norfair detections
        detections = []
        for contour in valid_contours:
            x, y, w, h = cv2.boundingRect(contour)
            cx = x + w / 2
            cy = y + h / 2
            detection = Detection(
                points=np.array([[cx, cy]]),
                data={'bbox': (x, y, w, h)}
            )
            detections.append(detection)

        # Update Norfair tracker
        self.tracked_objects = self.norfair_tracker.update(detections)

        # Draw detection visualization
        output_frame = frame.copy()
        min_track_age = self.config['tracking']['selection']['min_track_age']

        for obj in self.tracked_objects:
            if obj.last_detection is None:
                continue

            bbox = obj.last_detection.data['bbox']
            x, y, w, h = bbox

            # Only show objects that are old enough to select
            if obj.age >= min_track_age:
                # Cyan box for detection
                cv2.rectangle(output_frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

                # ID label
                cv2.putText(output_frame, f"ID: {obj.id}", (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        logger.debug("Detection mode: {count} objects tracked", count=len(self.tracked_objects))

        return output_frame, cleaned_mask, fg_mask

    def lock_onto_object(self, object_id: int) -> Optional[Tuple[int, int, int, int]]:
        """Lock tracking onto specific object ID.

        Args:
            object_id: Target object ID (0-9)

        Returns:
            Bounding box (x, y, w, h) if found, None otherwise
        """
        for obj in self.tracked_objects:
            if obj.id == object_id and obj.age >= self.config['tracking']['selection']['min_track_age']:
                bbox = obj.last_detection.data['bbox']
                self.tracked_object_id = object_id
                logger.info("Locked onto object ID: {id}", id=object_id)
                return bbox

        logger.warning("Object ID {id} not found or not old enough", id=object_id)
        return None

    def _create_opencv_tracker(self):
        """Create an OpenCV tracker instance with compatibility fallbacks."""
        tracker_name = self.tracker_type.upper()

        # Prefer non-legacy API if available
        if tracker_name == "CSRT":
            if hasattr(cv2, "TrackerCSRT_create"):
                logger.info("Using cv2.TrackerCSRT_create for CSRT")
                return cv2.TrackerCSRT_create()
            if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
                logger.info("Using cv2.legacy.TrackerCSRT_create for CSRT")
                return cv2.legacy.TrackerCSRT_create()
            logger.error("CSRT tracker not available in this OpenCV build")
            return None

        if tracker_name == "KCF":
            if hasattr(cv2, "TrackerKCF_create"):
                logger.info("Using cv2.TrackerKCF_create for KCF")
                return cv2.TrackerKCF_create()
            if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerKCF_create"):
                logger.info("Using cv2.legacy.TrackerKCF_create for KCF")
                return cv2.legacy.TrackerKCF_create()
            logger.error("KCF tracker not available in this OpenCV build")
            return None

        logger.error("Unknown tracker type: {type}", type=self.tracker_type)
        return None

    def initialize_single_tracker(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> bool:
        """Initialize single-object tracker.

        Args:
            frame: Current frame
            bbox: Bounding box (x, y, w, h)

        Returns:
            True if initialization successful
        """
        logger.info(
            "Initializing single-object tracker type={type} with bbox={bbox}",
            type=self.tracker_type,
            bbox=bbox,
        )

        tracker = self._create_opencv_tracker()
        if tracker is None:
            logger.error("Failed to create tracker instance; TRACKING will not start")
            return False

        # Normalize bbox to float tuple for OpenCV (x, y, w, h)
        try:
            x, y, w, h = bbox
            bbox_float = (float(x), float(y), float(w), float(h))
        except Exception as e:
            logger.error("Invalid bbox format for tracker initialization: {err}", err=e)
            return False

        try:
            success = tracker.init(frame, bbox_float)
        except Exception as e:
            logger.error("Tracker.init threw exception: {err}", err=e)
            return False

        if success:
            self.single_tracker = tracker
            self.last_bbox = (int(x), int(y), int(w), int(h))
            logger.info("Initialized {tracker} tracker successfully", tracker=self.tracker_type)
            return True

        logger.error("Tracker initialization failed (init() returned False)")
        return False

    def update_tracking_mode(self, frame: np.ndarray) -> Tuple[bool, Optional[Tuple[int, int, int, int]]]:
        """Update single-object tracking mode.

        Args:
            frame: Input frame (BGR)

        Returns:
            Tuple of (success, bbox) - bbox is None if tracking failed
        """
        if self.single_tracker is None:
            return False, None

        success, bbox = self.single_tracker.update(frame)

        if success and self._is_valid_bbox(bbox, frame.shape):
            bbox_int: Tuple[int, int, int, int] = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
            self.last_bbox = bbox_int
            return True, bbox_int
        else:
            logger.warning("Tracking failed or invalid bbox")
            return False, None

    def _is_valid_bbox(self, bbox: Tuple[float, float, float, float], frame_shape: Tuple[int, int]) -> bool:
        """Validate bounding box.

        Args:
            bbox: Bounding box (x, y, w, h)
            frame_shape: Frame shape (height, width, channels)

        Returns:
            True if bbox is valid
        """
        x, y, w, h = map(int, bbox)
        height, width = frame_shape[0], frame_shape[1]

        # Check bounds
        if x < 0 or y < 0 or x + w > width or y + h > height:
            return False

        # Check reasonable size
        area = w * h
        min_area = self.config['tracking']['validation']['min_bbox_area']
        max_area = height * width * self.config['tracking']['validation']['max_bbox_area_fraction']

        if area < min_area or area > max_area:
            return False

        return True

    def attempt_redetection(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Attempt to redetect lost object.

        Args:
            frame: Current frame

        Returns:
            Bounding box (x, y, w, h) if found, None otherwise
        """
        if self.last_bbox is None:
            return None

        # Background subtraction for redetection
        fg_mask = self.update_background_subtraction(frame)
        cleaned_mask = self.clean_mask(fg_mask)

        # Find contours
        valid_contours, _ = self.detect_contours(cleaned_mask)

        if not valid_contours:
            return None

        last_x, last_y, last_w, last_h = self.last_bbox
        last_cx = last_x + last_w / 2
        last_cy = last_y + last_h / 2
        last_area = last_w * last_h

        cfg = self.config['tracking']['recovery']
        search_radius = cfg['search_radius']
        size_threshold = cfg['size_similarity_threshold']

        best_match = None
        best_distance = float('inf')

        for contour in valid_contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            cx = x + w / 2
            cy = y + h / 2

            # Distance check
            distance = np.sqrt((cx - last_cx) ** 2 + (cy - last_cy) ** 2)
            if distance > search_radius:
                continue

            # Size similarity check
            size_ratio = min(area, last_area) / max(area, last_area) if max(area, last_area) > 0 else 0
            if size_ratio < size_threshold:
                continue

            # Better match found
            if distance < best_distance:
                best_match = (x, y, w, h)
                best_distance = distance

        if best_match:
            logger.info("Redetected object at distance {dist:.1f}", dist=best_distance)

        return best_match

    def reset_to_detection_mode(self):
        """Reset to detection mode."""
        self.state = State.DETECTION
        self.single_tracker = None
        self.tracked_object_id = None
        self.last_bbox = None
        self.loss_timestamp = 0.0
        logger.info("Reset to DETECTION mode")

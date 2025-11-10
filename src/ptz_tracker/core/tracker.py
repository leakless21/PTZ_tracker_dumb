"""
Object detection and tracking module.

Handles background subtraction, contour detection, Norfair multi-object tracking,
and CSRT/KCF single-object tracking.
"""

import cv2
import numpy as np
from norfair import Detection, Tracker
from loguru import logger
from typing import List, Tuple, Optional, Dict, Any, Union
from enum import Enum


class State(Enum):
    """Tracking state machine."""

    DETECTION = 0  # Multi-object mode (Norfair)
    TRACKING = 1  # Single-object mode (CSRT/KCF)
    LOST = 2  # Recovery mode


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
        bg_config = config["background_subtraction"]
        algorithm = bg_config["algorithm"]

        if algorithm == "MOG2":
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=bg_config["history"],
                varThreshold=bg_config["var_threshold"],
                detectShadows=bg_config["detect_shadows"],
            )
        elif algorithm == "KNN":
            self.bg_subtractor = cv2.createBackgroundSubtractorKNN(
                history=bg_config["history"],
                dist2Threshold=bg_config["dist2_threshold"],
                detectShadows=bg_config["detect_shadows"],
            )
        else:
            raise ValueError(f"Unknown background subtraction algorithm: {algorithm}")

        logger.info("Initialized background subtractor: {algo}", algo=algorithm)

        # Morphological kernel setup
        kernel_size = config["mask_processing"]["kernel_size"]
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )

        # Norfair tracker for multi-object detection
        norfair_cfg = config["tracking"]["norfair"]
        self.norfair_tracker = Tracker(
            distance_function="mean_euclidean",
            distance_threshold=norfair_cfg["distance_threshold"],
            hit_counter_max=norfair_cfg["hit_counter_max"],
            initialization_delay=norfair_cfg["initialization_delay"],
        )

        # Single-object tracker (CSRT/KCF)
        self.single_tracker = None
        self.tracker_type = config["tracking"]["tracker"]
        logger.info("Using {tracker} tracker", tracker=self.tracker_type)

        # Tracking state
        self.last_bbox = None
        self.tracked_object_id = None
        self.loss_timestamp = 0.0
        self.tracked_objects = []
        # Internal counter for optional reinforcement cadence can be managed by caller

        # Frame difference for tracking mode
        self.prev_frame_gray = None
        self.use_frame_diff_in_tracking = config["tracking"].get(
            "use_frame_difference_in_tracking", False
        )

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
        # Binary threshold first
        threshold_value = self.config["mask_processing"]["threshold_value"]
        _, mask = cv2.threshold(mask, threshold_value, 255, cv2.THRESH_BINARY)

        # Apply advanced motion mask cleaning
        mask = self.clean_motion_mask(mask)

        return mask

    def clean_motion_mask(
        self, mask: np.ndarray, min_area: Optional[int] = None
    ) -> np.ndarray:
        """Advanced motion mask cleaning for both frame difference and background subtraction.

        Removes noise, fills holes, and filters small disconnected components.

        Args:
            mask: Raw motion mask (binary, 0-255)
            min_area: Minimum component area in pixels (uses config if None)

        Returns:
            Cleaned motion mask with noise removed
        """
        # Determine minimum area threshold
        if min_area is None:
            min_area_val: int = int(self.config["object_detection"].get("min_area", 20))
        else:
            min_area_val = int(min_area)

        # Step 1: Morphological opening to remove small noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=1)

        # Step 2: Morphological closing to fill small holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=1)

        # Step 3: Additional dilation to connect nearby components
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(mask, kernel_dilate, iterations=1)

        # Step 4: Remove small connected components (noise filtering)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        clean_mask = np.zeros_like(mask)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area_val:
                cv2.drawContours(clean_mask, [contour], 0, 255, -1)

        # Step 5: Final erosion to avoid over-dilation artifacts
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        clean_mask = cv2.erode(clean_mask, kernel_erode, iterations=1)

        return clean_mask

    def compute_frame_difference_mask(
        self, curr_frame: np.ndarray, threshold: int = 30, return_raw_diff: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Compute motion mask from frame difference.

        Args:
            curr_frame: Current frame (BGR)
            threshold: Difference threshold (0-255)
            return_raw_diff: If True, return tuple (mask, raw_diff), else just mask

        Returns:
            If return_raw_diff=False: Binary motion mask from frame difference
            If return_raw_diff=True: Tuple of (binary_mask, raw_diff_before_threshold)
        """
        if self.prev_frame_gray is None:
            # No previous frame, return empty
            empty = np.zeros(self.frame_shape, dtype=np.uint8)
            if return_raw_diff:
                return empty, empty
            return empty

        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(self.prev_frame_gray, curr_gray)
        raw_diff = diff.copy()  # Store raw diff before thresholding

        _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

        # Update previous frame for next iteration
        self.prev_frame_gray = curr_gray

        # Apply advanced motion mask cleaning
        mask = self.clean_motion_mask(mask)

        if return_raw_diff:
            return mask, raw_diff
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

        cfg = self.config["object_detection"]
        min_area = cfg["min_area"]
        max_area = frame_area * cfg["max_area_fraction"]
        min_aspect = cfg.get("min_aspect_ratio", 0.0)
        max_aspect = cfg.get("max_aspect_ratio", float("inf"))

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

        logger.debug(
            "Detected {total} contours, {valid} valid",
            total=len(contours),
            valid=len(valid_contours),
        )

        return valid_contours, valid_indices

    def update_detection_mode(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
            # Convert numpy int64 to Python int for OpenCV compatibility
            bbox_tuple = (int(x), int(y), int(w), int(h))
            detection = Detection(
                points=np.array([[cx, cy]]), data={"bbox": bbox_tuple}
            )
            detections.append(detection)

        # Update Norfair tracker
        self.tracked_objects = self.norfair_tracker.update(detections)

        # Draw detection visualization
        output_frame = frame.copy()
        min_track_age = self.config["tracking"]["selection"]["min_track_age"]

        for obj in self.tracked_objects:
            if obj.last_detection is None:
                continue

            bbox = obj.last_detection.data["bbox"]
            x, y, w, h = bbox

            # Only show objects that are old enough to select
            if obj.age >= min_track_age:
                # Cyan box for detection
                cv2.rectangle(output_frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

                # ID label
                cv2.putText(
                    output_frame,
                    f"ID: {obj.id}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2,
                )

        logger.debug(
            "Detection mode: {count} objects tracked", count=len(self.tracked_objects)
        )

        return output_frame, cleaned_mask, fg_mask

    def lock_onto_object(self, object_id: int) -> Optional[Tuple[int, int, int, int]]:
        """Lock tracking onto specific object ID.

        Args:
            object_id: Target object ID (any non-negative integer)

        Returns:
            Bounding box (x, y, w, h) if found, None otherwise
        """
        for obj in self.tracked_objects:
            if (
                obj.id == object_id
                and obj.age >= self.config["tracking"]["selection"]["min_track_age"]
            ):
                bbox = obj.last_detection.data["bbox"]
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

    def initialize_single_tracker(
        self, frame: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> bool:
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

        # Normalize bbox to pure Python int tuple for OpenCV (x, y, w, h)
        # OpenCV requires pure Python numeric types, not numpy types
        try:
            x, y, w, h = bbox
            # Convert explicitly to Python int to avoid numpy dtype issues
            bbox_int = (int(x), int(y), int(w), int(h))
            # Ensure all components are reasonable
            if any(v <= 0 for v in bbox_int[2:]):  # width and height must be positive
                logger.error(
                    "Invalid bbox dimensions (w,h must be positive): {bbox}",
                    bbox=bbox_int,
                )
                return False

            # Additional validation: ensure bbox is within frame bounds
            frame_h, frame_w = frame.shape[:2]
            x, y, w, h = bbox_int
            if x < 0 or y < 0 or x + w > frame_w or y + h > frame_h:
                logger.error(
                    "Bbox out of frame bounds: bbox={bbox}, frame=({fw}x{fh})",
                    bbox=bbox_int,
                    fw=frame_w,
                    fh=frame_h,
                )
                return False

        except Exception as e:
            logger.error("Invalid bbox format for tracker initialization: {err}", err=e)
            return False

        try:
            tracker.init(frame, bbox_int)
            # Note: tracker.init() returns None on success (not a bool!)
            # If it raises an exception above, we catch it. Otherwise assume success.
        except Exception as e:
            logger.error("Tracker.init threw exception: {err}", err=e)
            return False

        self.single_tracker = tracker
        self.last_bbox = bbox_int
        logger.info(
            "Initialized {tracker} tracker successfully", tracker=self.tracker_type
        )
        return True

    def update_tracking_mode(
        self, frame: np.ndarray
    ) -> Tuple[bool, Optional[Tuple[int, int, int, int]]]:
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
            bbox_int: Tuple[int, int, int, int] = (
                int(bbox[0]),
                int(bbox[1]),
                int(bbox[2]),
                int(bbox[3]),
            )
            self.last_bbox = bbox_int
            return True, bbox_int
        else:
            logger.warning("Tracking failed or invalid bbox")
            return False, None

    def _is_valid_bbox(
        self, bbox: Tuple[float, float, float, float], frame_shape: Tuple[int, int]
    ) -> bool:
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
        min_area = self.config["tracking"]["validation"]["min_bbox_area"]
        max_area = (
            height
            * width
            * self.config["tracking"]["validation"]["max_bbox_area_fraction"]
        )

        if area < min_area or area > max_area:
            return False

        return True

    def attempt_redetection(
        self, frame: np.ndarray
    ) -> Optional[Tuple[int, int, int, int]]:
        """Attempt to redetect lost object.

        Uses frame difference if in tracking mode (and configured), otherwise
        falls back to background subtraction.

        Args:
            frame: Current frame

        Returns:
            Bounding box (x, y, w, h) if found, None otherwise
        """
        if self.last_bbox is None:
            return None

        # Use motion detection (frame difference or background subtraction)
        if self.use_frame_diff_in_tracking:
            fd_threshold = self.config["tracking"].get("frame_difference_threshold", 30)
            cleaned_mask = self.compute_frame_difference_mask(
                frame, threshold=fd_threshold
            )
        else:
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

        cfg = self.config["tracking"]["recovery"]
        search_radius = cfg["search_radius"]
        size_threshold = cfg["size_similarity_threshold"]

        best_match = None
        best_distance = float("inf")

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
            size_ratio = (
                min(area, last_area) / max(area, last_area)
                if max(area, last_area) > 0
                else 0
            )
            if size_ratio < size_threshold:
                continue

            # Better match found
            if distance < best_distance:
                best_match = (x, y, w, h)
                best_distance = distance

        if best_match:
            logger.info("Redetected object at distance {dist:.1f}", dist=best_distance)

        return best_match

    # --- Detection-assisted reinforcement helpers ---
    def _bbox_iou(
        self, a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]
    ) -> float:
        """Compute IoU between two bboxes (x, y, w, h)."""
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        a_x2, a_y2 = ax + aw, ay + ah
        b_x2, b_y2 = bx + bw, by + bh

        inter_x1 = max(ax, bx)
        inter_y1 = max(ay, by)
        inter_x2 = min(a_x2, b_x2)
        inter_y2 = min(a_y2, b_y2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = max(0, aw) * max(0, ah)
        area_b = max(0, bw) * max(0, bh)
        denom = area_a + area_b - inter_area + 1e-6
        return float(inter_area / denom) if denom > 0 else 0.0

    def _bbox_center(self, b: Tuple[int, int, int, int]) -> Tuple[float, float]:
        x, y, w, h = b
        return (x + w / 2.0, y + h / 2.0)

    def _euclidean_distance(
        self, p1: Tuple[float, float], p2: Tuple[float, float]
    ) -> float:
        return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))

    def reinforce_with_detection(
        self, frame: np.ndarray, tracked_bbox: Tuple[int, int, int, int]
    ) -> Optional[Tuple[int, int, int, int]]:
        """Use current detection to reinforce or correct the tracker bbox.

        Runs motion detection (background subtraction or frame difference,
        depending on config) and contour detection on the given frame,
        selects the best matching detection by IoU (or nearest center with
        size similarity as fallback), and returns a corrected bbox if it
        passes configured gates.

        Args:
            frame: Current frame (typically the PTZ ROI frame)
            tracked_bbox: Bbox from the tracker (x, y, w, h)

        Returns:
            New bbox to use if accepted, otherwise None.
        """
        # Validate incoming bbox
        if not self._is_valid_bbox(tracked_bbox, frame.shape):
            return None

        # Config
        reinf_cfg = self.config["tracking"].get("reinforcement", {})
        iou_thresh = float(reinf_cfg.get("iou_threshold", 0.3))
        max_center_dist = float(reinf_cfg.get("max_center_distance", 50))
        size_sim_thresh = float(
            self.config["tracking"]["recovery"].get("size_similarity_threshold", 0.5)
        )

        # Perform detection on this frame using frame difference or background subtraction
        if self.use_frame_diff_in_tracking:
            # Use frame difference for motion detection in tracking mode
            fd_threshold = self.config["tracking"].get("frame_difference_threshold", 30)
            cleaned_mask = self.compute_frame_difference_mask(
                frame, threshold=fd_threshold
            )
        else:
            # Use background subtraction (default)
            fg_mask = self.update_background_subtraction(frame)
            cleaned_mask = self.clean_mask(fg_mask)

        valid_contours, _ = self.detect_contours(cleaned_mask)
        if not valid_contours:
            return None

        # Find best IoU candidate
        best_iou = -1.0
        best_iou_bbox: Optional[Tuple[int, int, int, int]] = None
        for contour in valid_contours:
            x, y, w, h = cv2.boundingRect(contour)
            cand_bbox = (int(x), int(y), int(w), int(h))
            iou = self._bbox_iou(tracked_bbox, cand_bbox)
            if iou > best_iou:
                best_iou = iou
                best_iou_bbox = cand_bbox

        if best_iou_bbox is not None and best_iou >= iou_thresh:
            logger.debug(
                "Reinforcement accepted by IoU: {iou:.2f} >= {thr}",
                iou=best_iou,
                thr=iou_thresh,
            )
            return best_iou_bbox

        # Fallback: nearest center with size similarity gate
        tracked_center = self._bbox_center(tracked_bbox)
        tracked_area = max(1, tracked_bbox[2] * tracked_bbox[3])

        best_dist = float("inf")
        best_dist_bbox: Optional[Tuple[int, int, int, int]] = None
        for contour in valid_contours:
            x, y, w, h = cv2.boundingRect(contour)
            cand_bbox = (int(x), int(y), int(w), int(h))
            cand_center = self._bbox_center(cand_bbox)
            dist = self._euclidean_distance(tracked_center, cand_center)
            if dist > max_center_dist:
                continue
            # Size similarity check
            cand_area = max(1, w * h)
            size_ratio = min(cand_area, tracked_area) / max(cand_area, tracked_area)
            if size_ratio < size_sim_thresh:
                continue
            if dist < best_dist:
                best_dist = dist
                best_dist_bbox = cand_bbox

        if best_dist_bbox is not None:
            logger.debug(
                "Reinforcement accepted by distance: {dist:.1f} <= {maxd}",
                dist=best_dist,
                maxd=max_center_dist,
            )
            return best_dist_bbox

        return None

    def reset_to_detection_mode(self):
        """Reset to detection mode."""
        self.state = State.DETECTION
        self.single_tracker = None
        self.tracked_object_id = None
        self.last_bbox = None
        self.loss_timestamp = 0.0
        self.prev_frame_gray = None  # Reset frame difference history
        logger.info("Reset to DETECTION mode")

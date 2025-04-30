"""
This module implements the DeepSORT tracking algorithm using the BoxMOT library.

DeepSORT (Deep Simple Online and Realtime Tracking) is an extension of the SORT
(Simple Online and Realtime Tracking) algorithm that incorporates appearance
information to improve tracking performance. This module provides a class,
DeepSORTTracker, which initializes the tracker with specified parameters and
updates the tracking state with new detections for each video frame.

Classes:
    DeepSORTTracker: A class that encapsulates the DeepSORT tracking logic,
    allowing for initialization with custom parameters and updating with
    detections from a YOLO model.

Example usage:
    from detection.yolo import YOLODetector

    # Initialize detector and tracker
    detector = YOLODetector()
    tracker = DeepSORTTracker()

    # Get detections and update tracker
    detections = detector.detect(frame)
    tracked_objects = tracker.update(frame, detections)
"""

import pathlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from boxmot import DeepOcSort


class DeepSORTTracker:
    """
    A class to encapsulate the DeepSORT tracking algorithm using the BoxMOT library.

    The DeepSORTTracker class initializes the DeepSORT tracker with specified parameters
    and updates the tracking state with new
    detections for each video frame. It leverages
    appearance information to improve tracking
    performance over the basic SORT algorithm.

    Attributes:
        model_path (str | Path): Path to the ReID model weights.
        max_dist (float): Maximum cosine distance for matching.
        min_confidence (float): Minimum detection confidence threshold.
        nms_max_overlap (float): Non-maxima suppression threshold.
        max_iou_distance (float): Maximum IOU distance for matching.
        max_age (int): Maximum number of missed frames before a track is deleted.
        n_init (int): Number of consecutive frames a track
        must be detected to be initialized.
        nn_budget (int): Maximum size of the appearance descriptor gallery.
        half (bool): Whether to use half-precision
        floating point (FP16) for faster inference.

    Methods:
        update(frame, detections):
            Updates the tracker with new detections for a given frame
            and returns tracked objects.
    """

    def __init__(
        self,
        model_path: str | Path = "osnet_x0_25_msmt17.pt",
        max_dist: float = 0.2,
        min_confidence: float = 0.3,
        nms_max_overlap: float = 1.0,
        max_iou_distance: float = 0.7,
        max_age: int = 70,
        n_init: int = 3,
        nn_budget: int = 100,
        half: bool = False,
    ) -> None:
        """Initializes the DeepSORT tracker.

        Args:
            model_path (str): Path to the ReID model weights
            (e.g., 'osnet_x0_25_msmt17.pt').
                                BoxMOT will download it if not found.
            max_dist (float): Maximum cosine distance for matching.
            min_confidence (float): Minimum detection confidence threshold.
            nms_max_overlap (float): Non-maxima suppression threshold.
            max_iou_distance (float): Maximum IOU distance for matching.
            max_age (int): Maximum number of missed frames before a track is deleted.
            n_init (int): Number of consecutive frames a track
            must be detected to be initialized.
            nn_budget (int): Maximum size of the appearance descriptor gallery.
            half (bool): Whether to use half-precision floating point
            (FP16) for faster inference.
        """
        print(f"Initializing DeepSORT model with ReID weights: {model_path}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        try:
            model_path = (
                pathlib.Path(str(model_path))
                if isinstance(model_path, str)
                else model_path
            )

            self.tracker = DeepOcSort(
                reid_weights=model_path,
                device=self.device,
                half=half,
                fp16=half,  # Set to True if using GPU and want faster inference
                max_dist=max_dist,
                min_confidence=min_confidence,
                nms_max_overlap=nms_max_overlap,
                max_iou_distance=max_iou_distance,
                max_age=max_age,
                n_init=n_init,
                nn_budget=nn_budget,
            )
            print("DeepSORT tracker initialized successfully.")
        except Exception as e:
            print(f"Error initializing DeepSORT tracker: {e}")
            raise

    def update(self, frame: np.ndarray, detections: np.ndarray) -> Any:
        """Updates the tracker with new detections for a given frame.

        Args:
            frame (numpy.ndarray): The current video frame.
            detections (numpy.ndarray): Detections from the YOLO model in the format
                                        [x1, y1, x2, y2, confidence, class_id].

        Returns:
            numpy.ndarray: Tracked objects in the format
            [x1, y1, x2, y2, track_id, confidence, class_id].
                           Returns an empty array if no objects are tracked.
        """
        if detections is None or len(detections) == 0:
            # If no detections, return empty array with correct shape
            # The tracker state will be updated in the next frame with detections
            return np.empty((0, 7))

        # Ensure detections has at least 6 columns (x1,y1,x2,y2,conf,cls)
        if detections.shape[1] < 6:
            return np.empty((0, 7))

        # Additional validation to prevent boxmot Kalman filter error
        if len(detections.shape) == 1 or detections.shape[0] == 0:
            return np.empty((0, 7))

        # BoxMOT DeepSORT expects detections in shape
        # (N, 6) -> [x1, y1, x2, y2, conf, cls]
        # It returns tracks in shape (N, 7) -> [x1, y1, x2, y2, track_id, conf, cls]
        try:
            tracked_objects = self.tracker.update(detections, frame)
        except IndexError:
            # Handle boxmot Kalman filter error gracefully
            tracked_objects = np.empty((0, 7))

        if tracked_objects is not None and len(tracked_objects) > 0:
            return tracked_objects

        return np.empty((0, 7))

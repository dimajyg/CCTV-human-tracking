# tracking/deepsort.py

import torch
import numpy as np
from boxmot import DeepOcSort

class DeepSORTTracker:
    def __init__(self, 
                 model_path='osnet_x0_25_msmt17.pt', 
                 max_dist=0.2,
                 min_confidence=0.3,
                 nms_max_overlap=1.0,
                 max_iou_distance=0.7,
                 max_age=70,
                 n_init=3,
                 nn_budget=100,
                 half=False):
        """Initializes the DeepSORT tracker.

        Args:
            model_path (str): Path to the ReID model weights (e.g., 'osnet_x0_25_msmt17.pt').
                                BoxMOT will download it if not found.
            max_dist (float): Maximum cosine distance for matching.
            min_confidence (float): Minimum detection confidence threshold.
            nms_max_overlap (float): Non-maxima suppression threshold.
            max_iou_distance (float): Maximum IOU distance for matching.
            max_age (int): Maximum number of missed frames before a track is deleted.
            n_init (int): Number of consecutive frames a track must be detected to be initialized.
            nn_budget (int): Maximum size of the appearance descriptor gallery.
            half (bool): Whether to use half-precision floating point (FP16) for faster inference.
        """
        print(f"Initializing DeepSORT model with ReID weights: {model_path}")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        try:
            # Convert model_path to Path object if it's a string
            import pathlib
            model_path = pathlib.Path(model_path) if isinstance(model_path, str) else model_path
            
            self.tracker = DeepOcSort(
                reid_weights=model_path,
                device=self.device,
                half=half,
                fp16=half, # Set to True if using GPU and want faster inference
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

    def update(self, frame, detections):
        """Updates the tracker with new detections for a given frame.

        Args:
            frame (numpy.ndarray): The current video frame.
            detections (numpy.ndarray): Detections from the YOLO model in the format 
                                        [x1, y1, x2, y2, confidence, class_id].

        Returns:
            numpy.ndarray: Tracked objects in the format [x1, y1, x2, y2, track_id, confidence, class_id].
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

        # BoxMOT DeepSORT expects detections in shape (N, 6) -> [x1, y1, x2, y2, conf, cls]
        # It returns tracks in shape (N, 7) -> [x1, y1, x2, y2, track_id, conf, cls]
        try:
            tracked_objects = self.tracker.update(detections, frame)
        except IndexError:
            # Handle boxmot Kalman filter error gracefully
            tracked_objects = np.empty((0, 7))

        # Ensure output is a numpy array even if empty
        if not isinstance(tracked_objects, np.ndarray):
             tracked_objects = np.empty((0, 7))
        elif tracked_objects.size == 0:
            tracked_objects = np.empty((0, 7))

        return tracked_objects

# Example usage (optional, for testing)
if __name__ == '__main__':
    import cv2
    import numpy as np
    from detection.yolo import YOLODetector # Assuming yolo.py is in detection folder

    # Create a dummy black image
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    try:
        # Initialize detector and tracker
        detector = YOLODetector() # Use default yolov8n.pt
        tracker = DeepSORTTracker() # Use default osnet_x0_25_msmt17.pt
        print("Detector and Tracker initialized.")

        # Get dummy detections (or use a real image/video)
        # For this test, let's create a fake detection
        # Format: [x1, y1, x2, y2, confidence, class_id]
        # Class ID 0 represents 'person'
        dummy_detections = np.array([[100, 100, 200, 300, 0.9, 0]]) 
        # detections = detector.detect(dummy_frame) # Use this for real detection
        print(f"Dummy detections: {dummy_detections}")

        # Update tracker
        tracked_objects = tracker.update(dummy_frame, dummy_detections)
        print(f"Tracked objects in dummy frame:")
        print(tracked_objects)

    except Exception as e:
        print(f"An error occurred during example usage: {e}")
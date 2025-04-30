"""YOLO-based human detection module for CCTV tracking system.

This module provides a YOLODetector class that uses the YOLO (You Only Look Once)
model from the ultralytics package to detect humans in video frames. It's optimized
for person detection and supports both CPU and CUDA acceleration.

Typical usage:
    detector = YOLODetector(model_path='yolov8n.pt')
    detections = detector.detect(frame)

The detector returns bounding boxes in the format [x1, y1, x2, y2, confidence, class_id]
where class_id 0 represents the 'person' class in the COCO dataset.

Dependencies:
    - numpy
    - torch
    - ultralytics
"""

import numpy as np
import torch
from ultralytics import YOLO


class YOLODetector:
    """YOLO-based human detection class."""

    def __init__(self, model_path: str = "yolov8n.pt") -> None:
        """Initializes the YOLO detector.

        Args:
            model_path (str): Path to the YOLO model weights file (e.g., 'yolov8n.pt').
            Defaults to 'yolov8n.pt'. BoxMOT will download it if not found.
        """
        print(f"Initializing YOLO model with weights: {model_path}")
        # Check for CUDA availability
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Load the YOLO model using boxmot
        try:
            self.model = YOLO(model_path)
            self.model.to(self.device)
            print("YOLO model loaded successfully.")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            raise

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """Detects objects (specifically humans) in a given frame.

        Args:
            frame (numpy.ndarray): The input video frame.

        Returns:
            numpy.ndarray: Detections in the format
            [x1, y1, x2, y2, confidence, class_id].
            Returns an empty array if no humans are detected.
        """
        detections: np.ndarray = self.model(frame)[0].boxes.data.cpu().numpy()
        # bboxes: np.ndarray = detections[:, :4]
        confidences: np.ndarray = detections[:, 4]
        # Perform inference
        results = self.model.predict(
            frame, device=self.device, classes=[0], verbose=False
        )  # class 0 is 'person' in COCO

        # Extract bounding boxes, confidences, and class IDs for detected persons
        if results and results[0].boxes is not None:
            boxes = (
                results[0].boxes.xyxy.cpu().numpy()
            )  # Bounding boxes (x1, y1, x2, y2)
            confidences = results[0].boxes.conf.cpu().numpy()  # Confidence scores
            class_ids = results[0].boxes.cls.cpu().numpy()  # Class IDs

            # Combine into the desired format [x1, y1, x2, y2, confidence, class_id]
            detections = np.hstack(
                (boxes, confidences[:, np.newaxis], class_ids[:, np.newaxis])
            )

        # Ensure detections is a numpy array even if empty
        else:
            detections = np.empty((0, 6))  # Ensure shape (0, 6)

        return detections.astype(np.float32)

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        """Calls the detect method to perform object detection on the frame."""
        return self.detect(frame)


# Example usage (optional, for testing)
if __name__ == "__main__":
    # Create a dummy black image
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    detector = YOLODetector()
    print("Detector initialized.")
    detects = detector.detect(dummy_frame)
    print(f"Detected {len(detects)} objects in dummy frame:")
    print(detects)

# detection/yolo.py

import torch
from ultralytics import YOLO

import numpy as np

class YOLODetector:
    def __init__(self, model_path='yolov8n.pt'):
        """Initializes the YOLO detector.

        Args:
            model_path (str): Path to the YOLO model weights file (e.g., 'yolov8n.pt').
                                Defaults to 'yolov8n.pt'. BoxMOT will download it if not found.
        """
        print(f"Initializing YOLO model with weights: {model_path}")
        # Check for CUDA availability
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        # Load the YOLO model using boxmot
        try:
            self.model = YOLO(model_path)
            self.model.to(self.device)
            print("YOLO model loaded successfully.")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            raise

    def detect(self, frame):
        """Detects objects (specifically humans) in a given frame.

        Args:
            frame (numpy.ndarray): The input video frame.

        Returns:
            numpy.ndarray: Detections in the format [x1, y1, x2, y2, confidence, class_id].
                           Returns an empty array if no humans are detected.
        """
        # Perform inference
        results = self.model.predict(frame, device=self.device, classes=[0], verbose=False) # class 0 is 'person' in COCO

        # Extract bounding boxes, confidences, and class IDs for detected persons
        detections = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes (x1, y1, x2, y2)
            confidences = results[0].boxes.conf.cpu().numpy()  # Confidence scores
            class_ids = results[0].boxes.cls.cpu().numpy()    # Class IDs

            # Combine into the desired format [x1, y1, x2, y2, confidence, class_id]
            detections = np.hstack((boxes, confidences[:, np.newaxis], class_ids[:, np.newaxis]))

        # Ensure detections is a numpy array even if empty
        if not isinstance(detections, np.ndarray):
            detections = np.empty((0, 6)) # Ensure shape (0, 6)

        return detections

# Example usage (optional, for testing)
if __name__ == '__main__':
    import cv2
    import numpy as np

    # Create a dummy black image
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    try:
        detector = YOLODetector()
        print("Detector initialized.")
        detections = detector.detect(dummy_frame)
        print(f"Detected {len(detections)} objects in dummy frame:")
        print(detections)
    except Exception as e:
        print(f"An error occurred during example usage: {e}")
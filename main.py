"""CCTV Human Tracking System - Main Application Module.

This module provides the main functionality for the CCTV human tracking system,
integrating YOLO-based detection with DeepSORT tracking. It supports both real-time
visualization and video file processing with configurable parameters.

Key Features:
    - Human detection using YOLO
    - Multi-object tracking using DeepSORT
    - Real-time visualization
    - Video file processing and saving
    - Configurable processing parameters

Typical usage:
    python main.py --input video.mp4 --output result.avi
"""

import argparse
import time
from typing import Callable, Optional

import cv2
import numpy as np

from detection.yolo import YOLODetector
from tracking.deepsort import DeepSORTTracker

# --- Configuration ---
DEFAULT_INPUT_VIDEO = "input.mp4"  # Default input video file
DEFAULT_OUTPUT_VIDEO = "output.avi"  # Default output video file
YOLO_MODEL = "weights/yolo/yolo11n.pt"  # Path to YOLO weights
REID_MODEL = "weights/reid/osnet_x0_25_msmt17.pt"  # Path to ReID weights
SHOW_VIDEO = True  # Display the processed video in a window
SAVE_VIDEO = True  # Save the processed video to a file

# --- Helper Functions ---


def draw_boxes(frame: np.ndarray, tracked_objects: np.ndarray) -> np.ndarray:
    """Draws bounding boxes and track IDs on the frame."""
    for track in tracked_objects:
        if len(track) == 7:  # Expected format: [x1, y1, x2, y2, track_id, conf, cls_id]
            x1, y1, x2, y2, track_id, _, _ = map(int, track)
        else:  # Handle different format if needed
            x1, y1, x2, y2, track_id = map(int, track[:5])  # First 5 elements
            # conf = track[5] if len(track) > 5 else 1.0
            # cls_id = track[6] if len(track) > 6 else 0

        label = f"ID:{track_id}"
        color = (0, 255, 0)  # Green for bounding boxes

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Draw label background
        label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y1_label = max(y1, label_size[1] + 10)
        cv2.rectangle(
            frame,
            (x1, y1_label - label_size[1] - 10),
            (x1 + label_size[0], y1_label - base_line - 10),
            color,
            cv2.FILLED,
        )

        # Draw label text
        cv2.putText(
            frame,
            label,
            (x1, y1_label - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
        )
    return frame


# --- Main Processing Function ---


def process_video(
    input_path: str,
    output_path: str,
    yolo_model_path: str,
    reid_model_path: str,
    show: bool = True,
    save: bool = True,
    stride: int = 1,
    frame_callback: Optional[Callable] = None,
) -> None:
    """Processes the video for human tracking."""
    print("Initializing detector and tracker...")
    detector = YOLODetector(model_path=yolo_model_path)
    tracker = DeepSORTTracker(model_path=reid_model_path)

    print(f"Opening video source: {input_path}")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video source {input_path}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize video writer if saving
    writer = None
    if save:
        print(f"Initializing video writer for: {output_path}")
        fourcc = cv2.VideoWriter.fourcc(*"XVID")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        if not writer.isOpened():
            print(f"Error: Could not open video writer for {output_path}")
            save = False  # Disable saving if writer fails

    frame_count = 0
    start_time = time.time()

    print("Starting video processing...")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream.")
            break

        frame_count += 1

        # Skip frames based on stride
        if frame_count % stride != 0:
            continue

        # 1. Detect humans
        detections = detector.detect(frame)

        # 2. Update tracker
        if detections.size > 0:
            tracked_objects = tracker.update(frame, detections)
        else:
            # If no detections, still update tracker's internal state (e.g., age)
            tracked_objects = tracker.update(frame, np.empty((0, 6)))

        # 3. Draw results
        frame_with_boxes = draw_boxes(frame.copy(), tracked_objects)

        # 4. Display frame (optional)
        if show:
            cv2.imshow("CCTV Human Tracking", frame_with_boxes)
            if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to quit
                print("Processing stopped by user.")
                break

        # 5. Save frame (optional)
        if save and writer:
            writer.write(frame_with_boxes)

        # 6. Call frame callback if provided
        if frame_callback:
            frame_callback(frame_with_boxes, frame_count, total_frames)

    end_time = time.time()
    processing_time = end_time - start_time
    avg_fps = frame_count / processing_time if processing_time > 0 else 0

    print("\n--- Processing Summary ---")
    print(f"Total frames processed: {frame_count}")
    print(f"Total processing time: {processing_time:.2f} seconds")
    print(f"Average FPS: {avg_fps:.2f}")

    # Release resources
    cap.release()
    if writer:
        writer.release()
        print(f"Output video saved to: {output_path}")
    if show:
        cv2.destroyAllWindows()
    print("Resources released.")


# --- Entry Point ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CCTV Human Tracking using YOLO and DeepSORT"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT_VIDEO,
        help=f"Path to the input video file (default: {DEFAULT_INPUT_VIDEO})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_VIDEO,
        help=f"Path to save the output video file (default: {DEFAULT_OUTPUT_VIDEO})",
    )
    parser.add_argument(
        "--yolo_model",
        type=str,
        default=YOLO_MODEL,
        help=f"Path to YOLO model weights (default: {YOLO_MODEL})",
    )
    parser.add_argument(
        "--reid_model",
        type=str,
        default=REID_MODEL,
        help=f"Path to ReID model weights (default: {REID_MODEL})",
    )
    parser.add_argument(
        "--hide_video",
        action="store_true",
        help="Do not display the processed video window.",
    )
    parser.add_argument(
        "--no_save", action="store_true", help="Do not save the processed video."
    )
    parser.add_argument(
        "--stride", type=int, default=1, help="Process every nth frame (default: 1)"
    )

    args = parser.parse_args()

    print("--- CCTV Human Tracking Initialized ---")
    print(f"Input Video: {args.input}")
    print(f"Output Video: {args.output}")
    print(f"YOLO Model: {args.yolo_model}")
    print(f"ReID Model: {args.reid_model}")
    print(f"Show Video: {not args.hide_video}")
    print(f"Save Video: {not args.no_save}")
    print("---------------------------------------")

    process_video(
        input_path=args.input,
        output_path=args.output,
        yolo_model_path=args.yolo_model,
        reid_model_path=args.reid_model,
        show=not args.hide_video,
        save=not args.no_save,
        stride=args.stride,
    )

    print("--- CCTV Human Tracking Finished ---")

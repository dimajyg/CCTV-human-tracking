"""CCTV Human Tracking System - Streamlit Web Interface.

This module provides a web-based interface for the CCTV human tracking system using
Streamlit. It allows users to upload videos, configure tracking parameters, and
visualize the tracking results in real-time.

Features:
    - Video upload interface
    - Model selection (YOLO and ReID)
    - Processing parameter configuration
    - Real-time tracking visualization
    - Progress tracking
    - Result download

Typical usage:
    streamlit run app.py
"""

import os
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from main import process_video

# Set page config
st.set_page_config(page_title="CCTV Human Tracking", layout="wide")

# Title and description
st.title("CCTV Human Tracking System")
st.write(
    """Upload a video and configure tracking parameters
     to detect and track humans in the footage."""
)

# Sidebar controls
with st.sidebar:
    st.header("Configuration")

    # Video upload
    video_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

    # Output directory selection
    output_dir = st.text_input(
        "Output Directory",
        value="output",
        help="Directory where processed videos will be saved",
    )

    # Get available model weights
    yolo_weights_dir = Path("weights/yolo")
    reid_weights_dir = Path("weights/reid")
    yolo_weights = list(yolo_weights_dir.glob("*.pt"))
    reid_weights = list(reid_weights_dir.glob("*.pt"))

    # Model selection
    selected_yolo = st.selectbox(
        "Select YOLO Model",
        options=yolo_weights,
        format_func=lambda x: x.name,
        help="Choose the YOLO model weights for detection",
    )

    selected_reid = st.selectbox(
        "Select ReID Model",
        options=reid_weights,
        format_func=lambda x: x.name,
        help="Choose the ReID model weights for tracking",
    )

    # Video processing parameters
    stride = st.slider(
        "Frame Stride",
        min_value=1,
        max_value=10,
        value=1,
        help="""Process every nth frame
         (higher values = faster processing but lower smoothness)""",
    )

    # Process button
    process_button = st.button("Start Processing")

# Main content area
if video_file is not None:
    # Save uploaded video temporarily
    temp_path = Path("temp_video.mp4")
    with open(temp_path, "wb") as f:
        f.write(video_file.read())

    if process_button:
        st.write("Processing video...")

        # Create output directory if it doesn't exist
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(exist_ok=True)

        # Generate unique output filename
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_path = output_dir_path / f"processed_{timestamp}.avi"

        # Create placeholders for the video display and progress
        frame_placeholder = st.empty()
        progress_text = st.empty()
        progress_bar = st.progress(0)

        # Custom frame callback function
        def frame_callback(
            frame: np.ndarray, frame_count: int, total_frames: int
        ) -> None:
            """
            Callback function to be called after each frame is processed.
            """
            # Convert frame from BGR to RGB for Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Update progress:
            # Convert frame from BGR to RGB for Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Update progress
            progress = int((frame_count / total_frames) * 100)
            progress_bar.progress(progress)
            progress_text.text(
                f"Processing frame {frame_count} of {total_frames} ({progress}%)"
            )
            # Display the frame
            frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

        # Process the video with frame callback
        process_video(
            input_path=str(temp_path),
            output_path=str(output_path),
            yolo_model_path=str(selected_yolo),
            reid_model_path=str(selected_reid),
            show=False,  # Don't show in OpenCV window
            save=True,
            stride=stride,  # Pass the stride parameter
            frame_callback=frame_callback,  # Add frame callback
        )

        # Display success message and download button
        st.success(f"Video processed successfully! Saved to: {output_path}")

        # Cleanup temporary file
        os.remove(str(temp_path))

else:
    st.info("Please upload a video to begin")

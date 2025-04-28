# CCTV Human Tracking

This project implements human tracking in CCTV footage using YOLO for object detection and DeepSORT for tracking.

## Features

- Real-time human detection using YOLO.
- Robust human tracking across frames using DeepSORT.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd cctv-human-tracking
    ```

2.  **Install dependencies using Poetry:**
    ```bash
    poetry install
    ```

3.  **Download Model Weights:**
    - Place the YOLO weights file (e.g., `yolovX.pt`) in the `./weights/` directory.
    - Place the DeepSORT model file (e.g., `deepsort_model.pb`) in the `./weights/` directory.
    *(Note: You'll need to find and download appropriate pre-trained weights for YOLO and DeepSORT)*

## Usage

Run the main script:

```bash
poetry run python main.py --video <path_to_video_file> --yolo_weights <path_to_yolo_weights> --deepsort_model <path_to_deepsort_model>
```

*(Command-line arguments might need adjustment based on implementation)*

## Docker Deployment

1.  **Build the Docker image:**
    ```bash
    docker build -t cctv-human-tracking .
    ```

2.  **Run the Docker container:**
    ```bash
    docker run -it --rm \
        -v $(pwd)/videos:/app/videos \
        -v $(pwd)/weights:/app/weights \
        cctv-human-tracking \
        python main.py --video /app/videos/<your_video.mp4> --yolo_weights /app/weights/<yolo_weights> --deepsort_model /app/weights/<deepsort_model>
    ```
    *(Adjust volume mounts and command arguments as needed)*
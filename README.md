# CCTV Human Tracking

A human detection and tracking system using YOLO and DeepOCSORT, with both command-line and Streamlit interfaces.

## Features

- Human detection using YOLO (You Only Look Once)
- Multi-object tracking with DeepOCSORT
- Support for both video files and live camera feeds
- Streamlit web interface for easy interaction
- Docker support for containerized deployment
- Configurable detection and tracking parameters

## Training and Datasets

### Dataset
- MOT15 / MOT17 (Multiple Object Tracking) dataset used for training and evaluation
- Dataset structure includes both training and test sets
- Each sequence contains frame-by-frame images with ground truth annotations

### Performance Metrics

Our implementation shows significant improvements over the baseline model across key metrics:

| Metric | Baseline Weights | Our Weights |
|---------|-----------------|-------------|
| Precision | 0.709 | 0.821 |
| Recall | 0.451 | 0.651 |
| mAP50 | 0.527 | 0.758 |
| mAP50-95 | 0.242 | 0.434 |

### Data Augmentation
To improve model robustness and prevent overfitting, we use the Albumentations library to apply various augmentation techniques during training. These augmentations are designed to simulate different real-world conditions:

#### Time of Day Simulation
- RandomBrightnessContrast: Adjusts image brightness and contrast to simulate different lighting conditions
- RandomGamma: Modifies image gamma to simulate different exposure levels

#### Poor Camera Quality Simulation
- GaussianBlur: Adds blur to simulate unfocused or low-quality cameras
- Downscale: Reduces image quality to simulate low-resolution cameras
- ImageCompression: Simulates JPEG compression artifacts

#### Weather Condition Simulation
- RandomRain: Adds rain effects with configurable intensity
- RandomFog: Simulates foggy conditions with variable density
- RandomShadow: Adds random shadows to simulate partial occlusions

Additional Basic Augmentations:
- Random horizontal flips
- Random rotations (±15 degrees)
- Random scaling (0.8 to 1.2)
- Random translations

## Setup

1. **Clone the repository:**
    ```bash
    git clone https://github.com/dimajyg/CCTV-human-tracking.git
    cd cctv-human-tracking
    ```

2. **Install dependencies using Poetry (via Makefile):**
    ```bash
    make install
    ```
    *This will create a virtual environment and install all dependencies*

3. **Download Model Weights:**
    - Place the YOLO weights file (e.g., `yolov8n.pt`) in the `./weights/` directory
    - Place the DeepSORT model file (e.g., `osnet_x0_25_msmt17.pt`) in the `./weights/` directory

## Usage

### Streamlit Interface (Recommended)

1. **Start the Streamlit app:**
    ```bash
    poetry run streamlit run app.py
    ```

2. **Interface Options:**
    - Adjust detection confidence threshold
    - Configure tracking parameters
    - Real-time visualization of detection and tracking results
    - Export tracking data and annotated video

### Command Line Interface

```bash
# Run with default settings
poetry run python main.py --video <path_to_video>

# Run with custom settings
poetry run python main.py \
    --video <path_to_video> \
    --yolo_weights weights/yolov8n.pt \
    --reid_model weights/osnet_x0_25_msmt17.pt \
    --conf_thres 0.5 \
    --output_path output/
```

## Docker Deployment

1. **Build the Docker image:**
    ```bash
    docker build -t cctv-human-tracking .
    ```

2. **Run with Streamlit interface:**
    ```bash
    docker run -it --rm \
        -p 8501:8501 \
        -v $(pwd)/videos:/app/videos \
        -v $(pwd)/weights:/app/weights \
        cctv-human-tracking \
        streamlit run app.py
    ```

3. **Run with CLI:**
    ```bash
    docker run -it --rm \
        -v $(pwd)/videos:/app/videos \
        -v $(pwd)/weights:/app/weights \
        cctv-human-tracking \
        python main.py --video /app/videos/<your_video.mp4>
    ```

## Development

- **Format code:**
    ```bash
    make format
    ```

- **Run linting:**
    ```bash
    make lint
    ```

- **Clean environment:**
    ```bash
    make clean
    ```

- **Run tests:**
    ```bash
    poetry run pytest
    ```

## Project Structure

```
cctv-human-tracking/
├── detection/       # YOLO detection implementation
├── tracking/        # DeepSORT tracking implementation
├── weights/         # Model weights directory
├── app.py           # Streamlit interface
├── main.py          # CLI application
├── Makefile         # Project automation
├── pyproject.toml   # Poetry configuration
└── Dockerfile       # Container configuration
```

## Requirements

- Python 3.10
- Poetry for dependency management
- CUDA-compatible GPU (optional, for faster processing)
- Docker (optional, for containerized deployment)

## Contributers
- [Tikhanovskii Dmitrii](https://github.com/dimajyg) - Repository / Deployment / UI
- [Stasilovich Rudolf](https://github.com/rudiandradi) - Dataset Research / Model Training

## License

This project is licensed under the MIT License.
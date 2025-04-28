# CCTV Human Tracking

## Setup

1. **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd cctv-human-tracking
    ```

2. **Install dependencies using Poetry (via Makefile):**
    ```bash
    make install
    ```
    *This will create a virtual environment and install all dependencies*

3. **Download Model Weights:**
    - Place the YOLO weights file (e.g., `yolovX.pt`) in the `./weights/` directory
    - Place the DeepSORT model file (e.g., `osnet_x0_25_msmt17.pt`) in the `./weights/` directory

## Usage

### Using Makefile (recommended)
```bash
# Install dependencies (if not already done)
make install

# Run the tracking system
poetry run python main.py --video <path_to_video> --yolo_weights <path_to_yolo_weights> --reid_model <path_to_reid_model>
```

### Direct Poetry Commands
```bash
# Install dependencies
poetry install

# Run the tracking system
poetry run python main.py --video <path_to_video> --yolo_weights <path_to_yolo_weights> --reid_model <path_to_reid_model>
```

## Docker Deployment

1. **Build the Docker image:**
    ```bash
    docker build -t cctv-human-tracking .
    ```

2. **Run the Docker container:**
    ```bash
    docker run -it --rm \
        -v $(pwd)/videos:/app/videos \
        -v $(pwd)/weights:/app/weights \
        cctv-human-tracking \
        python main.py --video /app/videos/<your_video.mp4> --yolo_weights /app/weights/<yolo_weights> --reid_model /app/weights/<reid_model>
    ```

## Development

- **Clean environment:**
    ```bash
    make clean
    ```

- **Run tests:**
    ```bash
    poetry run pytest
    ```

## CI/CD

The project includes GitHub Actions workflow for continuous integration:
- Runs tests on push/pull requests
- Verifies YOLO and DeepSORT imports

## Project Structure

```
cctv-human-tracking/
├── detection/       # YOLO detection implementation
├── tracking/        # DeepSORT tracking implementation
├── weights/         # Model weights directory
├── main.py          # Main application script
├── Makefile         # Project automation
├── pyproject.toml   # Poetry configuration
└── Dockerfile       # Container configuration
```
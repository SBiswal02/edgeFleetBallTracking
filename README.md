# Cricket Ball Detection & Tracking

A Python project for training YOLOv8 models object detection model to detect and track cricket balls in videos

## Features

**Model Training**: Train YOLOv8 models on custom cricket ball dataset  
**Inference**: Run predictions on single images or directories  
**Ball Tracking**: Track cricket balls in videos with bounce detection  
**Interactive ROI Selection**: Define active play areas and start zones for tracking  
**CSV Annotations**: Export tracking data as frame-by-frame CSV files  
**Video Visualization**: Generate tracked videos with ball trails and positions  

## Project Structure

```
ball_tracking/
├── code/
│   ├── training.py         # Model training
│   ├── inference.py        # Image inference
│   ├── tracking.py         # Video ball tracking
│   ├── utilities.py        # Helpers/utilities
│   ├── cricket_ball_data/
│   |   ├── dataset.yaml        # config
│   |   ├── train/              # Training images/labels
│   |   ├── valid/              # Validation images/labels
│   |   └── test/               # Test images/labels
│   └── model/                 # Saved models
├── results/               # Detection outputs
├── annotations/           # CSV tracking data
├── requirements.txt       # Dependencies
├── Report.pdf     # Design report
├── ballTracking _ Ultralytics – Weights & Biases.pdf     ## wandb report since it can't be shared publicly
└── README.md              # Project readme
```

## Installation

### 1. Clone/Setup Repository

```bash
cd c:\ball_tracking
```

### 2. Create Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Setup Weights & Biases (Optional)

Create a `.env` file in the project root:

```env
WANDB_API_KEY=your_api_key_here
```

Get your API key from [Weights & Biases](https://wandb.ai/)

## Usage

### Training

Train a YOLOv8 model on your cricket ball dataset:

```bash
python code/training.py --model yolov8s --yaml cricket_ball_data/dataset.yaml --epochs 50 --batch 8
```

**Available Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `yolov8s` | Model size (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x) |
| `--yaml` | `cricket_ball_data/dataset.yaml` | Path to dataset YAML file |
| `--epochs` | `50` | Number of training epochs |
| `--batch` | `8` | Batch size |
| `--imgsz` | `640` | Image size for training |
| `--freeze` | `15` | Number of backbone layers to freeze |
| `--device` | `auto` | Device (cuda, cpu, auto) |
| `--no-visualize` | - | Skip visualizing training samples |
| `--no-validate` | - | Skip validation after training |

**Example with Custom Parameters:**

```bash
python code/training.py --model yolov8m --yaml cricket_ball_data/dataset.yaml --epochs 100 --batch 16 --imgsz 800 --freeze 20 --no-visualize
```

### Inference (Detection)

Run predictions on images and save annotated results:

```bash
# Single image
python code/inference.py --image test_image.jpg --model model/cricket_ball_detector_model_v12s.pt --conf 0.2

# All images in directory
python code/inference.py --image cricket_ball_data/test/images --model model/cricket_ball_detector_model_v12s.pt --conf 0.3

# Default (with visualization enabled)
python code/inference.py --image test_image.jpg --model model/cricket_ball_detector_model_v12s.pt
```

**Available Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--image` | Required | Path to image or directory |
| `--model` | `model/cricket_ball_detector_model_v12s.pt` | Path to trained model |
| `--imgsz` | `640` | Image size for inference |
| `--conf` | `0.2` | Confidence threshold |
| `--device` | `auto` | Device (cuda, cpu, auto) |
| `--output` | `results` | Output directory for results |
| `--no-visualize` | - | Skip displaying results |

**Available Models:**
- `model/cricket_ball_detector_model_v12s.pt` - Fine-tuned for cricket ball detection (recommended for inference)
- `model/cricket_ball_detector_model_v8m.pt` - Public YOLOv8m model, not fine-tuned but performs better for tracking

**Output:** Annotated images are saved to `results/` folder as `detected_<original_name>.jpg`

### Tracking (Ball Tracking in Video)

Track cricket balls in video files with simple interpolation:

```bash
# Basic tracking with interactive ROI setup
python code/tracking.py --video test_vids/15.mov --model model/cricket_ball_detector_model_v8m.pt

# Skip ROI setup (use defaults)
python code/tracking.py --video test_vids/15.mov --model model/cricket_ball_detector_model_v8m.pt --no-roi

# Custom confidence and image size
python code/tracking.py --video test_vids/15.mov --model model/cricket_ball_detector_model_v8m.pt --conf 0.0001 --imgsz 2048

# Skip video visualization (only save CSV)
python code/tracking.py --video test_vids/15.mov --skip-viz

# Custom output directory
python code/tracking.py --video test_vids/15.mov --output-dir my_tracking_results
```

**Available Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--video` | Required | Path to input video file |
| `--model` | `model/cricket_ball_detector_model_v8m.pt` | Path to trained model |
| `--conf` | `0.005` | Confidence threshold for detections |
| `--imgsz` | `1920` | Image size for model inference |
| `--device` | `auto` | Device (cuda, cpu, auto) |
| `--no-roi` | - | Skip interactive ROI setup |
| `--skip-viz` | - | Skip visualization video creation |
| `--output-dir` | `results` | Output directory for videos |

**Interactive ROI Setup:**

When running without `--no-roi`, you'll see two interactive windows:

1. **STEP 1**: Draw the active play area (everything outside is ignored)
2. **STEP 2**: Draw the start zone (where ball must first appear)

This is a recommended step to improve tracking accuracy since the videos may contain graphic overlays that can confuse the model and affect performance.

Step 2 ensures tracking starts only when the ball enters the defined start zone to avoid false positives from before the ball is released.

Instructions in each window:
- **Draw**: Click and drag to select area
- **Confirm**: Press SPACE or ENTER
- **Cancel**: Press C

**Troubleshooting Tracking:**

If tracking fails or performs poorly, try adjusting `--imgsz` and `--conf` parameters:

- **Low detection rate**: Reduce `--conf` (e.g., 0.0001 to 0.001) to catch more detections
- **Too many false positives**: Increase `--conf` (e.g., 0.01 to 0.05)
- **Poor ball localization**: Increase `--imgsz` (e.g., 2048 or 2560) for higher resolution detection
- **Out of memory**: Reduce `--imgsz` (e.g., 640 or 1024)
- **Slow inference**: Reduce `--imgsz` or use GPU acceleration with `--device cuda`

Example: `python code/tracking.py --video test_vids/15.mov --conf 0.0001 --imgsz 2048`

**Output Structure:**

```
results/
  └── 8_tracked.mp4          # Video with ball trail overlay

annotations/
  └── .csv                  # Frame-by-frame ball coordinates
```

**CSV Format:**

```csv
frame,x,y,visible
0,1024.50,512.75,1
1,1025.30,515.20,1
2,1026.10,518.60,1
...
150,-1,-1,0
```

- `frame`: Frame number (0-indexed)
- `x`, `y`: Ball center coordinates
- `visible`: 1 if ball is visible in the overlay, 0 if missing

## Output Folders

### `results/`
- Detected images from inference
- Tracked videos with ball visualization

### `annotations/`
- CSV files with ball coordinates per frame

### `model/`
- Trained model weights (.pt files)

## Help & Documentation

Get detailed help for any script:

```bash
python code/training.py --help
python code/inference.py --help
python code/tracking.py --help
```

## Troubleshooting

### Model Not Found Error

Ensure model file exists:

```bash
ls model/
```

Common model paths:
- `model/cricket_ball_detector_model_v8.pt` (yolov8s)
- `model/cricket_ball_detector_model_v8m.pt` (yolov8m)
- `model/cricket_ball_detector_model_v12s.pt` (yolov12s)

### CUDA Out of Memory

Reduce `--imgsz` or `--batch`:

```bash
python code/training.py --imgsz 416 --batch 4
```

### Video Codec Issues

If tracked videos don't play, try different codec. Edit the fourcc line in `code/tracking.py`:

```python
# Change: fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# To: fourcc = cv2.VideoWriter_fourcc(*'XVID')
```

### Slow Inference

- Use GPU: `--device cuda`
- Reduce image size: `--imgsz 640`
- Use smaller model: `--model yolov8n`

## Library Versions

All required versions are specified in `requirements.txt`:

```bash
# View installed versions
pip list

# Update all packages
pip install --upgrade -r requirements.txt
```

## Dataset Preparation

Your dataset should follow this structure with YOLO format labels:

```
cricket_ball_data/
├── dataset.yaml
├── train/
│   ├── images/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── labels/
│       ├── img1.txt
│       └── img2.txt
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

Each `.txt` label file should contain:
```
<class_id> <x_center> <y_center> <width> <height>
```

Where coordinates are normalized (0-1).

## License & Credits

This project uses:
- [YOLOv8](https://github.com/ultralytics/ultralytics) - Ultralytics
- [PyTorch](https://pytorch.org/) - Meta
- [OpenCV](https://opencv.org/) - OpenCV Community

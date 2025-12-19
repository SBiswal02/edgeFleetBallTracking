# Cricket Ball Detection & Tracking: Design Report

## Executive Summary

This project implements a comprehensive cricket ball detection and tracking system using YOLOv8 object detection models combined with physics-aware interpolation and intelligent post-processing. The system is designed to handle the challenging task of tracking a small, fast-moving cricket ball in video sequences with high accuracy and robustness.

---

## 1. Project Overview

### 1.1 Problem Statement

Tracking a cricket ball in video presents several unique challenges:
- **Small object size**: Cricket balls are small relative to frame resolution
- **High velocity**: Balls move quickly, creating motion blur
- **Occlusion**: Balls can be occluded by players, equipment, or camera angles
- **Variable lighting**: Outdoor conditions create varying illumination
- **Complex backgrounds**: Stadium environments with crowds, graphics overlays, and field markings
- **Physics constraints**: Ball motion follows physical laws (gravity, bounce)

### 1.2 Solution Architecture

The solution employs a three-stage pipeline:
1. **Detection**: YOLOv8-based object detection for ball localization
2. **Tracking**: State machine with candidate selection and filtering
3. **Post-processing**: Physics-aware interpolation with outlier removal

---

## 2. Model Selection & Training Decisions

### 2.1 YOLOv8 Architecture Choice

**Decision**: Use YOLOv8 (Ultralytics) as the base detection model.

**Rationale**:
- **State-of-the-art performance**: YOLOv8 provides excellent speed-accuracy tradeoff
- **Small object detection**: Improved anchor-free design handles small objects better than earlier YOLO versions
- **Transfer learning**: Pre-trained weights on COCO dataset provide strong feature representations
- **Flexibility**: Multiple model sizes (nano, small, medium, large, xlarge) allow balancing accuracy vs. speed
- **Active development**: Well-maintained framework with regular updates

### 2.2 Model Variants Tested

The project experimented with multiple YOLOv8 variants:

| Model | Size | Use Case | Rationale |
|-------|------|----------|-----------|
| YOLOv8s | Small | General detection | Balanced accuracy/speed, good for inference |
| YOLOv8m | Medium | Video tracking | Better accuracy for tracking pipeline |
| YOLOv12s | Small | Fine-tuned detection | Latest architecture with improved features |

**Key Finding**: YOLOv8m (`cricket_ball_detector_model_v8m.pt`) performs better for tracking despite not being fine-tuned, likely due to:
- Better feature extraction for small objects
- More robust to motion blur
- Better generalization across video frames

### 2.3 Training Configuration

**Hyperparameters**:
```python
- Model: YOLOv8s (default) / YOLOv8m / YOLOv12s
- Epochs: 50 (default, configurable)
- Batch size: 8 (default, configurable)
- Image size: 640 (training), 1920 (inference)
- Freeze layers: 15 (transfer learning)
- Patience: 5 (early stopping)
```

**Design Decisions**:

1. **Transfer Learning with Layer Freezing**:
   - Freeze first 15 backbone layers to preserve pre-trained features
   - Only fine-tune detection head and final layers
   - **Rationale**: Cricket balls share visual features with other spherical objects in COCO dataset

2. **Image Size Strategy**:
   - Training: 640×640 (standard YOLO size, faster training)
   - Inference: 1920×1920 (higher resolution for small ball detection)
   - **Rationale**: Higher resolution at inference improves small object detection without slowing training

3. **Early Stopping**:
   - Patience of 5 epochs prevents overfitting
   - **Rationale**: Small dataset size (1778 training images) requires regularization

4. **Dataset Split**:
   - Train: 1778 images
   - Validation: 63 images
   - Test: 71 images
   - **Rationale**: Standard 80/10/10 split ensures sufficient training data

### 2.4 Confidence Threshold Strategy

**Different thresholds for different tasks**:
- **Inference (images)**: `conf=0.2` (higher threshold, fewer false positives)
- **Tracking (video)**: `conf=0.005` (very low threshold, catch all detections)
- **Rationale**: 
  - Video tracking can filter false positives through temporal consistency
  - Low threshold ensures no ball detections are missed
  - Post-processing removes noise

---

## 3. Tracking Pipeline Design

### 3.1 Two-Phase Architecture

The tracking system operates in two distinct phases:

#### Phase 1: Detection & State Machine
- Frame-by-frame detection using YOLO
- State machine with two modes: **Scanning** and **Tracking**
- Candidate selection based on confidence and distance

#### Phase 2: Post-Processing
- Streak noise removal
- Outlier detection and removal
- Physics-aware interpolation
- Gap filling

### 3.2 State Machine Design

**States**:
1. **Scanning Mode** (before lock-on):
   - Waiting for ball to appear in start zone
   - Validates detections against start zone ROI
   - Requires valid start position to transition

2. **Tracking Mode** (after lock-on):
   - Accepts detections based on distance/confidence
   - Tracks ball through frames
   - Handles temporary occlusions

**State Transition Logic**:
```python
if not has_locked_on:
    # Scanning: Only accept if in start zone
    if detection_in_start_zone:
        has_locked_on = True
        enter_tracking_mode()
else:
    # Tracking: Accept based on distance/confidence
    if candidate_within_dynamic_limit:
        accept_detection()
```

**Rationale**:
- Prevents false positives from graphics/overlays before ball release
- Ensures tracking starts at correct moment
- Reduces computational overhead in early frames

### 3.3 Candidate Selection Algorithm

**Function**: `select_best_candidate()` in `utilities.py`

**Strategy**:
1. Sort detections by confidence (highest first)
2. Calculate distance from last known position
3. Apply dynamic distance limits:
   - **High confidence** (>0.4): Hard limit of 250 pixels
   - **Low confidence**: Dynamic limit = 100 + 5 × frames_since_last_detection
4. Select first detection within limit

**Rationale**:
- Prioritizes high-confidence detections
- Allows larger search radius when ball hasn't been seen recently
- Prevents tracking from jumping to distant false positives
- Adapts to occlusion scenarios

---

## 4. Physics-Aware Interpolation

### 4.1 Separate Physics Models

**Key Innovation**: Different interpolation models for X and Y axes.

**X-Axis (Horizontal Motion)**:
- **Model**: Linear interpolation (`kind='linear'`)
- **Rationale**: Horizontal motion is approximately constant velocity (neglecting air resistance)
- **Implementation**: `scipy.interpolate.interp1d` with linear kernel

**Y-Axis (Vertical Motion)**:
- **Model**: Quadratic interpolation (`kind='quadratic'`)
- **Rationale**: Vertical motion follows gravity (parabolic trajectory)
- **Fallback**: Linear if insufficient points (<3)
- **Implementation**: `scipy.interpolate.interp1d` with quadratic kernel

**Code Implementation** (`utilities.py:interpolate_segment`):
```python
# X-axis: Linear (constant horizontal velocity)
fx = interpolate.interp1d(x_local, y_x_local, kind='linear', fill_value="extrapolate")

# Y-axis: Quadratic (gravity) or linear fallback
kind_y = 'quadratic' if len(indices) > 3 else 'linear'
fy = interpolate.interp1d(x_local, y_y_local, kind=kind_y, fill_value="extrapolate")
```

### 4.2 Bounce Detection

**Algorithm**:
1. Find maximum Y-coordinate (lowest point in frame)
2. Split trajectory at bounce point
3. Interpolate pre-bounce and post-bounce segments separately

**Rationale**:
- Ball trajectory changes at bounce (velocity reversal)
- Separate interpolation prevents smoothing across bounce discontinuity
- More accurate trajectory reconstruction

**Implementation** (`tracking.py:post_process_detections`):
```python
# Detect bounce (maximum Y value)
valid_y_values = pre_detections_np[valid_indices, 1]
argmax_y = np.argmax(valid_y_values)
bounce_idx_global = valid_indices[argmax_y]

# Split into two segments
mask_s1 = (valid_indices <= bounce_idx_global)  # Pre-bounce
mask_s2 = (valid_indices >= bounce_idx_global)  # Post-bounce
```

---

## 5. Post-Processing & Filtering

### 5.1 Multi-Stage Filtering Pipeline

The post-processing pipeline consists of four stages:

#### Stage 1: Streak Noise Removal
**Function**: `remove_streak_noise()`

**Purpose**: Remove isolated false positive detections before consistent tracking begins.

**Algorithm**:
- Scan from start of video
- Count consecutive detections (streak)
- Only keep detections after minimum streak length (default: 2 frames)

**Rationale**: 
- Prevents false positives from graphics/overlays in early frames
- Ensures tracking starts with consistent detections
- Reduces noise before main processing

#### Stage 2: Jump Outlier Removal
**Function**: `remove_jump_outliers()`

**Purpose**: Remove detections that jump unrealistically far between frames.

**Algorithm**:
- Calculate distance between consecutive detections
- Calculate gap size (frames between detections)
- Remove if `distance > max_jump_ratio × gap_size` (default: 100 pixels/frame)

**Rationale**:
- Balls cannot move faster than physical limits
- Prevents tracking from jumping to distant false positives
- Maintains temporal consistency

#### Stage 3: Physics-Aware Interpolation
- Separate X/Y interpolation as described in Section 4
- Fill gaps between valid detections
- Extrapolate beyond valid range if needed

#### Stage 4: Gap Cutting
**Purpose**: Remove interpolated segments across large gaps.

**Algorithm**:
- Identify gaps > `max_gap` frames (default: 15)
- Set interpolated values in large gaps to (-1, -1) (invalid)

**Rationale**:
- Large gaps likely indicate ball is out of frame or occluded
- Interpolation across large gaps is unreliable
- Better to mark as missing than provide inaccurate positions

### 5.2 Detection Filtering

**Function**: `filter_detections()` in `utilities.py`

**Filters Applied**:
1. **Size filtering**: 
   - Minimum area: 40 pixels²
   - Maximum area: 500 pixels²
   - **Rationale**: Cricket balls have predictable size range

2. **Aspect ratio filtering**:
   - Minimum: 0.7 (width/height)
   - Maximum: 1.6 (width/height)
   - **Rationale**: Balls are approximately circular (aspect ratio ~1.0)

**Rationale**: Removes detections that are clearly not balls (too large/small, wrong shape)

---

## 6. ROI Selection System

### 6.1 Interactive ROI Selection

**Purpose**: Allow users to define regions of interest to improve tracking accuracy.

**Two ROI Types**:

1. **Active Play Area**:
   - Defines where ball can appear
   - Everything outside is masked (set to black)
   - **Rationale**: Excludes graphics overlays, scoreboards, crowd areas

2. **Start Zone**:
   - Defines where ball must first appear
   - Tracking only begins when ball detected in this zone
   - **Rationale**: Prevents false positives before ball release

### 6.2 Scaled ROI Selection

**Function**: `select_roi_scaled()` in `utilities.py`

**Implementation**:
- Display frame scaled to 1280px width (for usability)
- User selects ROI on scaled frame
- Coordinates mapped back to original resolution

**Rationale**:
- High-resolution videos (1920×1080+) make ROI selection difficult
- Scaled display improves user experience
- Coordinate mapping ensures accuracy

### 6.3 ROI Application

**Active Area Masking**:
```python
def apply_active_area_mask(frame, roi_active):
    mask = np.zeros_like(frame)
    cv2.rectangle(mask, (ax, ay), (ax + aw, ay + ah), (255, 255, 255), -1)
    return cv2.bitwise_and(frame, mask)
```

**Rationale**: 
- Black masking prevents model from detecting objects outside play area
- Reduces false positives from graphics/overlays
- Improves tracking accuracy

---

## 7. Technical Implementation Details

### 7.1 Technology Stack

**Core Libraries**:
- **PyTorch 2.9.1**: Deep learning framework
- **Ultralytics 8.2.26**: YOLOv8 implementation
- **OpenCV 4.12.0**: Video processing, ROI selection, visualization
- **NumPy 2.2.6**: Numerical operations
- **SciPy 1.16.3**: Interpolation functions
- **Matplotlib 3.9.2**: Visualization

**Rationale**:
- Industry-standard libraries with active maintenance
- Good performance and GPU acceleration support
- Well-documented APIs

### 7.2 Code Organization

**Modular Design**:
```
code/
├── training.py      # Model training pipeline
├── inference.py    # Image detection
├── tracking.py     # Video tracking pipeline
└── utilities.py    # Shared helper functions
```

**Design Principles**:
- **Separation of concerns**: Each script handles one task
- **Reusability**: Utilities shared across modules
- **Command-line interface**: All scripts support CLI arguments
- **Configurability**: Extensive parameter options

### 7.3 Data Flow

**Tracking Pipeline Flow**:
```
Video Input
    ↓
ROI Selection (optional)
    ↓
Frame-by-Frame Detection (YOLO)
    ↓
State Machine Filtering
    ↓
Candidate Selection
    ↓
Raw Detections Array
    ↓
Post-Processing:
  - Streak removal
  - Outlier removal
  - Physics interpolation
  - Gap cutting
    ↓
Post-Processed Detections
    ↓
CSV Export + Video Visualization
```

### 7.4 Output Formats

**CSV Annotations**:
```csv
frame,x,y,visible
0,1024.50,512.75,1
1,1025.30,515.20,1
...
150,-1,-1,0
```

**Format Rationale**:
- Simple, human-readable format
- Easy to import into analysis tools
- Standard format for tracking data
- `-1,-1` indicates missing/occluded ball

**Video Visualization**:
- White translucent trail showing ball path
- Red circle at current ball position
- **Rationale**: Visual feedback for tracking quality

---

## 8. Performance Considerations

### 8.1 Inference Speed Optimization

**Strategies**:
1. **GPU Acceleration**: Automatic CUDA detection and usage
2. **Batch Processing**: Process multiple images simultaneously (inference.py)
3. **Configurable Image Size**: Balance accuracy vs. speed
4. **Model Selection**: Smaller models (YOLOv8s) for faster inference

**Performance Trade-offs**:
- **High resolution** (1920px): Better accuracy, slower inference
- **Low resolution** (640px): Faster inference, lower accuracy
- **Default**: 1920px for tracking (accuracy priority), 640px for training (speed priority)

### 8.2 Memory Management

**Strategies**:
1. **Frame-by-frame processing**: Don't load entire video into memory
2. **Configurable batch size**: Adjust for available GPU memory
3. **Efficient data structures**: NumPy arrays for detections

**Memory Considerations**:
- Large videos processed incrementally
- Detection arrays stored as lists (sparse) then converted to NumPy
- Video writing uses OpenCV's efficient codec

### 8.3 Scalability

**Design for Scalability**:
- **Modular pipeline**: Easy to parallelize frame processing
- **Configurable parameters**: Adapt to different hardware
- **Efficient algorithms**: O(n) complexity for most operations

**Limitations**:
- Single-threaded frame processing (could be parallelized)
- No distributed training support (single GPU/CPU)

---

## 9. Design Trade-offs & Decisions

### 9.1 Accuracy vs. Speed

**Decision**: Prioritize accuracy for tracking, speed for training.

**Rationale**:
- Tracking is offline/post-processing task (speed less critical)
- Training benefits from faster iteration
- Users can adjust image size based on needs

### 9.2 False Positives vs. False Negatives

**Decision**: Accept more false positives, filter in post-processing.

**Rationale**:
- Low confidence threshold (0.005) catches all balls
- Post-processing removes false positives through temporal consistency
- Better to have false positives than miss ball detections

### 9.3 Manual ROI vs. Automatic

**Decision**: Provide interactive ROI selection with option to skip.

**Rationale**:
- Manual ROI improves accuracy (excludes graphics)
- Optional for flexibility (users can skip if not needed)
- Default zones work for most cases

### 9.4 Physics Models: Simple vs. Complex

**Decision**: Use simple linear/quadratic models rather than complex physics simulation.

**Rationale**:
- Simple models are sufficient for interpolation
- Complex models would require more parameters (air resistance, spin, etc.)
- Quadratic interpolation captures gravity effect adequately
- Faster computation

---

## 10. Limitations & Future Improvements

### 10.1 Current Limitations

1. **Single Ball Tracking**: System tracks one ball at a time
2. **No Multi-Camera Support**: Single video input only
3. **Fixed Physics Models**: Doesn't account for air resistance, spin
4. **Manual ROI**: Requires user interaction (though optional)
5. **No Real-Time Processing**: Designed for offline video analysis

### 10.2 Potential Improvements

1. **Multi-Ball Tracking**:
   - Extend state machine to track multiple objects
   - Use object tracking algorithms (DeepSORT, ByteTrack)

2. **Advanced Physics Models**:
   - Air resistance modeling
   - Spin effects
   - More accurate trajectory prediction

3. **Automatic ROI Detection**:
   - Use semantic segmentation to detect field boundaries
   - Automatic start zone detection from player positions

4. **Real-Time Processing**:
   - Optimize for lower latency
   - Frame skipping strategies
   - Model quantization

5. **Multi-Camera Fusion**:
   - Combine detections from multiple camera angles
   - 3D trajectory reconstruction

6. **Deep Learning Tracking**:
   - Replace state machine with learned tracking model
   - End-to-end trainable tracking

7. **Data Augmentation**:
   - More aggressive augmentation during training
   - Synthetic data generation

8. **Model Ensembling**:
   - Combine multiple model predictions
   - Improve robustness

---

## 11. Conclusion

This cricket ball tracking system demonstrates a well-engineered approach to a challenging computer vision problem. Key strengths include:

1. **Robust Detection**: YOLOv8 provides strong baseline detection
2. **Intelligent Tracking**: State machine and candidate selection handle occlusions
3. **Physics-Aware Processing**: Separate X/Y interpolation models respect ball physics
4. **Comprehensive Post-Processing**: Multi-stage filtering ensures high-quality output
5. **User-Friendly**: Interactive ROI selection and extensive configuration options

The system successfully balances accuracy, robustness, and usability while maintaining a clean, modular codebase that can be extended for future improvements.

---

## Appendix: Key Design Patterns

### A.1 State Machine Pattern
- Used for tracking mode transitions (scanning → tracking)
- Clear state boundaries and transition logic

### A.2 Pipeline Pattern
- Sequential processing stages (detection → filtering → interpolation → output)
- Each stage is independent and testable

### A.3 Strategy Pattern
- Different interpolation strategies for X/Y axes
- Configurable confidence thresholds for different use cases

### A.4 Template Method Pattern
- Common structure across training/inference/tracking scripts
- Shared setup and device detection logic

---

**Report Generated**: 2025
**Project Version**: Based on codebase analysis
**Author**: Design Documentation


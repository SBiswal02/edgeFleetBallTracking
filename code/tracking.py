"""
Ball tracking and interpolation script for cricket video analysis.
Includes detection, state machine tracking, physics-aware interpolation, and visualization.
Can be run from command line with customizable parameters.

Example:
    python tracking.py --video test_vids/15.mov --model model/cricket_ball_detector_model_v8m.pt --conf 0.0002 --imgsz 1920
"""

import os
import argparse
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from utilities import (
    select_roi_scaled, apply_active_area_mask, filter_detections, select_best_candidate,
    interpolate_segment, remove_streak_noise, remove_jump_outliers
)


def setup_device():
    """Check and setup CUDA device."""
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {DEVICE}')
    return DEVICE


def setup_paths():
    """Setup dataset and model paths."""
    DATASET_PATH = os.path.join(os.getcwd(), 'cricket_ball_data')
    MODEL_PATH = os.path.join(os.getcwd(), 'model')
    TEST_VIDS_PATH = os.path.join(os.getcwd(), 'test_vids')
    OUTPUT_PATH = os.path.join(os.getcwd(), 'output_videos')
    ANNOTATION_PATH = os.path.join(os.getcwd(), 'annotations')
    RESULTS_PATH = os.path.join(os.getcwd(), 'results')

    for path in [MODEL_PATH, OUTPUT_PATH, ANNOTATION_PATH, RESULTS_PATH]:
        if not os.path.exists(path):
            os.makedirs(path)

    return {
        'dataset': DATASET_PATH,
        'model': MODEL_PATH,
        'test_vids': TEST_VIDS_PATH,
        'output': OUTPUT_PATH,
        'annotations': ANNOTATION_PATH,
        'results': RESULTS_PATH,
    }


def detect_ball_in_video(video_path, model, device='cuda', conf=0.0002, imgsz=1920, 
                         roi_active=None, roi_start=None):
    """
    Detect ball in video frames with state machine tracking.
    
    Args:
        video_path (str): Path to input video
        model: YOLO model instance
        device (str): Device to use
        conf (float): Confidence threshold
        imgsz (int): Image size for detection
        roi_active (tuple): Active play area ROI (x, y, w, h)
        roi_start (tuple): Start zone ROI (x, y, w, h)
    
    Returns:
        tuple: (pre_detections, fps, frame_width, frame_height)
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f'Cannot open video: {video_path}')
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    pre_detections = []
    frame_id = 0
    has_locked_on = False
    frames_since_last_valid = 0
    
    # Unpack ROI coordinates
    ax, ay, aw, ah = roi_active if roi_active else (0, 0, 0, 0)
    has_active_area = aw > 0 and ah > 0
    
    sx, sy, sw, sh = roi_start if roi_start else (0, 0, 0, 0)
    has_start_zone = sw > 0 and sh > 0
    
    print(f'Processing video: {os.path.basename(video_path)}')
    print(f'Resolution: {frame_width}x{frame_height}, FPS: {fps}')
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply active area mask
        if has_active_area:
            frame = apply_active_area_mask(frame, roi_active)
        
        # Predict
        results = model.predict(source=frame, imgsz=imgsz, conf=conf, device=device, verbose=False)
        
        # Filter detections
        detections = filter_detections(results)
        
        # Select best candidate
        last_detection = None
        if pre_detections:
            for det in reversed(pre_detections):
                if det is not None:
                    last_detection = det
                    break
        
        best_candidate = select_best_candidate(
            detections, 
            last_detection=last_detection,
            dynamic_gap=frames_since_last_valid
        )
        
        # State machine
        if not has_locked_on:
            # Scanning mode
            if best_candidate:
                center_x, center_y = best_candidate
                is_valid_start = False
                
                if has_start_zone:
                    if (sx < center_x < sx + sw) and (sy < center_y < sy + sh):
                        is_valid_start = True
                else:
                    if (0.2 * frame_width < center_x < 0.8 * frame_width) and \
                       (0.3 * frame_height < center_y < 0.85 * frame_height):
                        is_valid_start = True
                
                if is_valid_start:
                    pre_detections.append(best_candidate)
                    has_locked_on = True
                else:
                    pre_detections.append(None)
            else:
                pre_detections.append(None)
        else:
            # Tracking mode
            pre_detections.append(best_candidate)
        
        if best_candidate:
            frames_since_last_valid = 0
        else:
            frames_since_last_valid += 1
        
        frame_id += 1
    
    cap.release()
    print(f'Detection phase complete. Processed {frame_id} frames.')
    
    return pre_detections, fps, frame_width, frame_height


def post_process_detections(pre_detections, min_streak=2, max_gap=15, max_jump_ratio=100):
    """
    Post-process raw detections with interpolation and outlier removal.
    
    Args:
        pre_detections (list): Raw detections from video
        min_streak (int): Minimum consecutive detections
        max_gap (int): Maximum gap to interpolate
        max_jump_ratio (int): Maximum pixels per frame of gap
    
    Returns:
        list: Post-processed detections
    """
    print('\nPost-processing detections...')
    
    # Phase 1: Streak filtering
    cleaned_detections = remove_streak_noise(pre_detections, min_streak=min_streak)
    
    # Phase 2: Convert to numpy
    pre_detections_np = np.array(
        [det if det is not None else (np.nan, np.nan) for det in cleaned_detections]
    )
    
    # Phase 3: Remove jump outliers
    pre_detections_np = remove_jump_outliers(pre_detections_np, max_jump_ratio=max_jump_ratio)
    
    # Phase 4: Piecewise interpolation
    post_processed_detections = []
    valid_mask = ~np.isnan(pre_detections_np[:, 0])
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) > 2:
        first_valid_idx = valid_indices[0]
        last_valid_idx = valid_indices[-1]
        
        # Detect bounce (maximum Y value)
        valid_y_values = pre_detections_np[valid_indices, 1]
        argmax_y = np.argmax(valid_y_values)
        bounce_idx_global = valid_indices[argmax_y]
        
        # Split into two segments
        mask_s1 = (valid_indices <= bounce_idx_global)
        indices_s1 = valid_indices[mask_s1]
        
        mask_s2 = (valid_indices >= bounce_idx_global)
        indices_s2 = valid_indices[mask_s2]
        
        # Interpolate both segments
        funcs_s1 = interpolate_segment(indices_s1, len(pre_detections_np), pre_detections_np)
        funcs_s2 = interpolate_segment(indices_s2, len(pre_detections_np), pre_detections_np)
        
        # Fill data
        for i in range(len(pre_detections_np)):
            if i < first_valid_idx or i > last_valid_idx:
                post_processed_detections.append((0.0, 0.0))
                continue
            
            if i <= bounce_idx_global and funcs_s1:
                fx, fy = funcs_s1
                post_processed_detections.append((float(fx(i)), float(fy(i))))
            elif i > bounce_idx_global and funcs_s2:
                fx, fy = funcs_s2
                post_processed_detections.append((float(fx(i)), float(fy(i))))
            else:
                # Fallback to linear
                fx = np.interp(i, valid_indices, pre_detections_np[valid_indices, 0])
                fy = np.interp(i, valid_indices, pre_detections_np[valid_indices, 1])
                post_processed_detections.append((float(fx), float(fy)))
        
        # Gap cutting
        for k in range(len(valid_indices) - 1):
            idx_current = valid_indices[k]
            idx_next = valid_indices[k + 1]
            gap_len = idx_next - idx_current
            
            if gap_len > max_gap:
                for g in range(idx_current + 1, idx_next):
                    post_processed_detections[g] = (0.0, 0.0)
        
    else:
        post_processed_detections = [(0.0, 0.0)] * len(pre_detections_np)
    
    # Replace (0.0, 0.0) with (-1, -1)
    for i in range(len(post_processed_detections)):
        if post_processed_detections[i] == (0.0, 0.0):
            post_processed_detections[i] = (-1, -1)
    
    return post_processed_detections


def save_annotations(post_processed_detections, video_filename, output_path):
    """
    Save detections as CSV.
    
    Args:
        post_processed_detections (list): Detections to save
        video_filename (str): Name of video file
        output_path (str): Output directory path
    """
    csv_file = os.path.join(output_path, video_filename.split('.')[0] + '.csv')
    
    with open(csv_file, 'w') as f:
        f.write("frame,x,y,visible\n")
        for i, det in enumerate(post_processed_detections):
            x, y = det
            visible = 1 if (x != -1 and y != -1) else 0
            if visible:
                f.write(f"{i},{x:.2f},{y:.2f},{visible}\n")
            else:
                f.write(f"{i},-1,-1,{visible}\n")
    
    print(f'Annotations saved to {csv_file}')


def visualize_tracking(video_path, post_processed_detections, fps, frame_width, frame_height, output_path):
    """
    Create output video with tracking visualization.
    
    Args:
        video_path (str): Path to input video
        post_processed_detections (list): Detections for visualization
        fps (float): Frames per second
        frame_width (int): Frame width
        frame_height (int): Frame height
        output_path (str): Output video path
    """
    print(f'Creating visualization video...')
    
    cap = cv2.VideoCapture(video_path)
    
    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    frame_id = 0
    trace_points = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_id >= len(post_processed_detections):
            break
        
        detection = post_processed_detections[frame_id]
        cx, cy = map(int, detection)
        
        is_valid_point = (cx > 5 and cy > 5)
        
        if is_valid_point:
            trace_points.append((cx, cy))
        
        # Draw translucent white trail
        if len(trace_points) > 1:
            overlay = frame.copy()
            pts_np = np.array(trace_points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(overlay, [pts_np], isClosed=False, color=(255, 255, 255), 
                         thickness=5, lineType=cv2.LINE_AA)
            alpha = 0.4
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        # Draw red current ball
        if is_valid_point:
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        
        out.write(frame)
        frame_id += 1
    
    cap.release()
    out.release()
    print(f' Visualization video saved to {output_path}')
    return output_path


def track_video(video_path, model, paths, device='cuda', roi_active=None, roi_start=None):
    """
    Full pipeline: detect, post-process, and visualize ball tracking.
    
    Args:
        video_path (str): Path to input video
        model: YOLO model instance
        paths (dict): Dictionary of paths
        device (str): Device to use
        roi_active (tuple): Active play area ROI
        roi_start (tuple): Start zone ROI
    """
    # Detection
    pre_detections, fps, frame_width, frame_height = detect_ball_in_video(
        video_path, model, device=device, roi_active=roi_active, roi_start=roi_start
    )
    
    # Post-processing
    post_processed_detections = post_process_detections(pre_detections)
    
    # Save annotations
    video_filename = os.path.basename(video_path)
    save_annotations(post_processed_detections, video_filename, paths['annotations'])
    
    # Visualize
    output_filename = video_filename.split('.')[0] + '_tracked.mp4'
    output_path = os.path.join(paths['output'], output_filename)
    visualize_tracking(video_path, post_processed_detections, fps, frame_width, frame_height, output_path)


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Track cricket ball in video with physics-aware interpolation')
    parser.add_argument('--video', type=str, required=True,
                        help='Path to input video file')
    parser.add_argument('--model', type=str, default='model/cricket_ball_detector_model_v8m.pt',
                        help='Path to trained YOLO model')
    parser.add_argument('--conf', type=float, default=0.05,
                        help='Confidence threshold for detections')
    parser.add_argument('--imgsz', type=int, default=1920,
                        help='Image size for model inference')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cuda, cpu, or auto)')
    parser.add_argument('--no-roi', action='store_true',
                        help='Skip interactive ROI setup (use defaults)')
    parser.add_argument('--skip-viz', action='store_true',
                        help='Skip visualization video creation')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Base output directory for results and annotations')
    
    args = parser.parse_args()
    
    # Setup
    if args.device == 'auto':
        device = setup_device()
    else:
        device = args.device
        print(f'Using device: {device}')
    
    paths = setup_paths()
    
    # Override output directories if provided
    if args.output_dir != 'results':
        # Keep results and annotations as separate folders
        paths['results'] = args.output_dir
        paths['annotations'] = os.path.join(os.path.dirname(args.output_dir), 'annotations')
        for path in [paths['results'], paths['annotations']]:
            if not os.path.exists(path):
                os.makedirs(path)
    
    # Load model
    if not os.path.exists(args.model):
        print(f'Error: Model not found: {args.model}')
        exit(1)
    
    model = YOLO(args.model)
    print(f' Model loaded: {args.model}')
    
    # Verify video exists
    if not os.path.exists(args.video):
        print(f'Error: Video not found: {args.video}')
        exit(1)
    
    print('===========================')
    print('Ball Tracking Configuration')
    print('===========================') 
    print(f'Video: {args.video}')
    print(f'Model: {args.model}')
    print(f'Confidence threshold: {args.conf}')
    print(f'Image size: {args.imgsz}')
    print(f'Device: {device}')
    print(f'Output directory: {paths["results"]}')
    print(f'Annotations directory: {paths["annotations"]}')
    print('===========================\n')
    
    # Load first frame for interactive ROI setup
    cap = cv2.VideoCapture(args.video)
    ret, first_frame = cap.read()
    cap.release()
    
    if not ret:
        print(f'Error: Cannot read video: {args.video}')
        exit(1)
    
    # Interactive ROI setup
    roi_active = None
    roi_start = None
    
    if not args.no_roi:
        roi_active = select_roi_scaled(
            'STEP 1: Draw ACTIVE PLAY AREA (Ignore everything else), Press SPACE or ENTER to confirm selection, C to cancel',
            first_frame
        )
        roi_start = select_roi_scaled(
            'STEP 2: Draw START ZONE (Where ball appears first / release point), Press SPACE or ENTER to confirm selection, C to cancel',
            first_frame
        )
        print(f'Active Play Area: {roi_active}')
        print(f'Start Zone: {roi_start}\n')
    else:
        print('ROI setup skipped. Using default detection zones.\n')
    
    # Run tracking pipeline
    print('Starting detection phase...')
    pre_detections, fps, frame_width, frame_height = detect_ball_in_video(
        args.video, model, device=device, conf=args.conf, imgsz=args.imgsz,
        roi_active=roi_active, roi_start=roi_start
    )
    
    print('\nPost-processing detections...')
    post_processed_detections = post_process_detections(pre_detections)
    
    # Save annotations
    video_filename = os.path.basename(args.video)
    save_annotations(post_processed_detections, video_filename, paths['annotations'])
    
    # Save visualization video
    if not args.skip_viz:
        output_filename = video_filename.split('.')[0] + '_tracked.mp4'
        output_path = os.path.join(paths['results'], output_filename)
        visualize_tracking(args.video, post_processed_detections, fps, frame_width, frame_height, output_path)
    else:
        print('Video visualization skipped.')
    
    print('\n===========================')
    print(' Tracking complete!')
    print('===========================') 
    print(f'CSV annotations: {os.path.join(paths["annotations"], video_filename.split(".")[0] + ".csv")}')
    if not args.skip_viz:
        print(f'Tracked video: {os.path.join(paths["results"], output_filename)}')
    print('===========================')

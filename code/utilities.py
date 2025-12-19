"""
Utility functions for cricket ball tracking and detection.
"""

import cv2
import numpy as np
import scipy.interpolate as interpolate


def select_roi_scaled(window_name, frame, target_width=1280):
    """
    Opens a scaled-down version of the frame for ROI selection,
    then maps the coordinates back to the original resolution.
    
    Args:
        window_name (str): Name of the display window
        frame (np.ndarray): Input frame for ROI selection
        target_width (int): Target width for scaled display
    
    Returns:
        tuple: (x, y, w, h) coordinates in original frame resolution
    """
    h, w = frame.shape[:2]
    scale = target_width / w

    # Resize for display
    small_frame = cv2.resize(frame, (target_width, int(h * scale)))

    # Select ROI on the small frame
    roi_small = cv2.selectROI(window_name, small_frame, showCrosshair=True)
    cv2.destroyAllWindows()

    # Map back to original coordinates
    if roi_small[2] > 0 and roi_small[3] > 0:  # If width and height > 0
        x = int(roi_small[0] / scale)
        y = int(roi_small[1] / scale)
        roi_w = int(roi_small[2] / scale)
        roi_h = int(roi_small[3] / scale)
        return (x, y, roi_w, roi_h)
    return (0, 0, 0, 0)


def apply_active_area_mask(frame, roi_active):
    """
    Apply a mask to make areas outside the active ROI black.
    
    Args:
        frame (np.ndarray): Input frame
        roi_active (tuple): (x, y, w, h) of active area
    
    Returns:
        np.ndarray: Frame with masked areas
    """
    if roi_active[2] <= 0 or roi_active[3] <= 0:
        return frame

    ax, ay, aw, ah = roi_active
    mask = np.zeros_like(frame)
    cv2.rectangle(mask, (ax, ay), (ax + aw, ay + ah), (255, 255, 255), -1)
    return cv2.bitwise_and(frame, mask)


def filter_detections(results, min_area=40, max_area=500, min_aspect=0.7, max_aspect=1.6):
    """
    Filter detections based on size and aspect ratio.
    
    Args:
        results: YOLOv8 prediction results
        min_area (int): Minimum bounding box area
        max_area (int): Maximum bounding box area
        min_aspect (float): Minimum width/height ratio
        max_aspect (float): Maximum width/height ratio
    
    Returns:
        list: Filtered detections as (center_x, center_y, confidence) tuples
    """
    closest_detections = []

    for result in results:
        boxes = result.boxes
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = box.conf[0].item()

            box_w, box_h = x2 - x1, y2 - y1
            box_area = box_w * box_h
            aspect_ratio = box_w / box_h if box_h != 0 else 0

            if min_area < box_area < max_area and min_aspect < aspect_ratio < max_aspect:
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                closest_detections.append((center_x, center_y, confidence))

    return closest_detections


def select_best_candidate(detections, last_detection=None, dynamic_gap=0, 
                          high_conf_threshold=0.4, hard_limit=250):
    """
    Select the best detection candidate based on confidence and distance.
    
    Args:
        detections (list): List of (x, y, confidence) detections
        last_detection (tuple): Last known detection (x, y)
        dynamic_gap (int): Number of frames since last detection
        high_conf_threshold (float): Confidence threshold for high confidence
        hard_limit (int): Maximum distance allowed for high confidence detections
    
    Returns:
        tuple: (x, y) of selected detection or None
    """
    if not detections:
        return None

    # Sort by confidence (highest first)
    detections.sort(key=lambda x: x[2], reverse=True)

    if last_detection:
        last_x, last_y = last_detection
        dynamic_limit = 100 + 5 * dynamic_gap

        for det in detections:
            center_x, center_y, conf = det
            dist = np.sqrt((center_x - last_x) ** 2 + (center_y - last_y) ** 2)

            is_high_confidence = conf > high_conf_threshold
            limit_to_use = hard_limit if is_high_confidence else dynamic_limit

            if dist < limit_to_use:
                return (center_x, center_y)

        return None
    else:
        # First detection
        return (detections[0][0], detections[0][1])


def interpolate_segment(indices, full_len, detections_np):
    """
    Interpolate a segment of detections with separate physics for X and Y axes.
    
    Args:
        indices (np.ndarray): Frame indices for this segment
        full_len (int): Total number of frames
        detections_np (np.ndarray): Full detection array
    
    Returns:
        tuple: (fx, fy) interpolation functions or None
    """
    if len(indices) < 2:
        return None

    x_local = indices
    y_x_local = detections_np[indices, 0]
    y_y_local = detections_np[indices, 1]

    # X-axis: Linear (constant horizontal velocity)
    try:
        fx = interpolate.interp1d(x_local, y_x_local, kind='linear', fill_value="extrapolate")
    except:
        return None

    # Y-axis: Quadratic (gravity) or linear fallback
    try:
        kind_y = 'quadratic' if len(indices) > 3 else 'linear'
        fy = interpolate.interp1d(x_local, y_y_local, kind=kind_y, fill_value="extrapolate")
    except:
        fy = interpolate.interp1d(x_local, y_y_local, kind='linear', fill_value="extrapolate")

    return fx, fy


def remove_streak_noise(detections, min_streak=2):
    """
    Remove short noise streaks before consistent ball detection.
    
    Args:
        detections (list): List of detections (can be None for gaps)
        min_streak (int): Minimum consecutive detections for valid streak
    
    Returns:
        list: Cleaned detections
    """
    cleaned = detections.copy()
    streak_count = 0
    start_index = -1

    for i, det in enumerate(cleaned):
        if det is not None:
            if streak_count == 0:
                start_index = i
            streak_count += 1
            if streak_count >= min_streak:
                break
        else:
            streak_count = 0

    if streak_count >= min_streak:
        for k in range(start_index):
            cleaned[k] = None

    return cleaned


def remove_jump_outliers(detections_np, max_jump_ratio=100):
    """
    Remove outlier detections that jump too far.
    
    Args:
        detections_np (np.ndarray): Array of detections with NaN for gaps
        max_jump_ratio (int): Maximum pixels per frame of gap
    
    Returns:
        np.ndarray: Cleaned detection array
    """
    valid_indices = np.where(~np.isnan(detections_np[:, 0]))[0]

    if len(valid_indices) > 0:
        last_idx = valid_indices[0]
        for idx in valid_indices[1:]:
            curr_pt = detections_np[idx]
            last_pt = detections_np[last_idx]
            dist = np.linalg.norm(curr_pt - last_pt)
            gap_size = idx - last_idx

            if dist > (max_jump_ratio * gap_size):
                detections_np[idx] = (np.nan, np.nan)
            else:
                last_idx = idx

    return detections_np

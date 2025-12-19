"""
Inference script for cricket ball detection on test images.
Can be run from command line with customizable parameters.

Example:
    python inference.py --image test_image.jpg --model model/cricket_ball_detector_model_v12s.pt --imgsz 800 --conf 0.2
    python inference.py --image test_imgs/ --model model/cricket_ball_detector_model_v12s.pt
"""

import os
import argparse
import cv2
import torch
import matplotlib.pyplot as plt
import random
from ultralytics import YOLO


def setup_device():
    """Check and setup CUDA device."""
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {DEVICE}')
    return DEVICE


def setup_paths():
    """Setup dataset and model paths."""
    DATASET_PATH = os.path.join(os.getcwd(), 'cricket_ball_data')
    TEST_PATH = os.path.join(DATASET_PATH, 'test')
    TEST_IMAGES_PATH = os.path.join(TEST_PATH, 'images')
    MODEL_PATH = os.path.join(os.getcwd(), 'model')

    return {
        'dataset': DATASET_PATH,
        'test': TEST_PATH,
        'test_images': TEST_IMAGES_PATH,
        'model': MODEL_PATH,
    }


def load_model(model_path):
    """
    Load trained model.
    
    Args:
        model_path (str): Path to model file
    
    Returns:
        YOLO: Loaded model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model not found: {model_path}')
    
    model = YOLO(model_path)
    print(f'Model loaded from {model_path}')
    return model


def predict_image(model, image_path, imgsz=640, conf=0.2, device='cuda', save=False, save_dir=None):
    """
    Run inference on a single image.
    
    Args:
        model: YOLO model instance
        image_path (str): Path to input image
        imgsz (int): Image size for inference
        conf (float): Confidence threshold
        device (str): Device to use
        save (bool): Whether to save annotated image
        save_dir (str): Directory to save results
    
    Returns:
        Results object
    """
    results = model.predict(
        source=image_path,
        imgsz=imgsz,
        conf=conf,
        device=device,
        save=save,
        project=save_dir,
        name='predictions',
        exist_ok=True,
        verbose=False
    )
    return results


def save_detection_image(result, output_path):
    """
    Save annotated detection image.
    
    Args:
        result: Prediction result object
        output_path (str): Path to save image
    
    Returns:
        str: Path to saved image
    """
    result_image = result.plot()
    result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(output_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
    return output_path


def predict_batch(model, image_dir, imgsz=640, conf=0.2, device='cuda'):
    """
    Run inference on multiple images from a directory.
    
    Args:
        model: YOLO model instance
        image_dir (str): Directory containing images
        imgsz (int): Image size for inference
        conf (float): Confidence threshold
        device (str): Device to use
    
    Returns:
        list: List of results
    """
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    sample_files = image_files
    
    results_list = []
    for image_file in sample_files:
        image_path = os.path.join(image_dir, image_file)
        results = predict_image(model, image_path, imgsz=imgsz, conf=conf, device=device)
        results_list.append(results)
        print(f'Processed: {image_file}')
    
    return results_list


def visualize_predictions(results):
    """
    Visualize detection results.
    
    Args:
        results (list): List of prediction results
    """
    plt.figure(figsize=(16, 8))
    
    for i, result in enumerate(results[:8]):
        result_image = result.plot()
        result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        
        plt.subplot(2, 4, i + 1)
        plt.imshow(result_image)
        plt.axis('off')
        plt.title(f'Image {i+1}')
    
    plt.tight_layout()
    plt.show()


def print_detections(results):
    """
    Print detection statistics.
    
    Args:
        results (list or Results): Prediction results
    """
    if isinstance(results, list):
        for i, result in enumerate(results):
            print(f'\nImage {i}:')
            print(f'  Detections: {len(result.boxes)}')
            for box in result.boxes:
                conf = box.conf[0].item()
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                print(f'    Box: ({x1:.1f}, {y1:.1f}) - ({x2:.1f}, {y2:.1f}), Conf: {conf:.3f}')
    else:
        result = results
        print(f'Detections: {len(result.boxes)}')
        for box in result.boxes:
            conf = box.conf[0].item()
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            print(f'  Box: ({x1:.1f}, {y1:.1f}) - ({x2:.1f}, {y2:.1f}), Conf: {conf:.3f}')


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run inference on cricket ball detection')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image or directory of images')
    parser.add_argument('--model', type=str, default='model/cricket_ball_detector_model_v12s.pt',
                        help='Path to trained model')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Image size for inference')
    parser.add_argument('--conf', type=float, default=0.2,
                        help='Confidence threshold')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cuda, cpu, or auto)')
    parser.add_argument('--no-visualize', action='store_true',
                        help='Skip displaying results')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    # Setup
    if args.device == 'auto':
        device = setup_device()
    else:
        device = args.device
        print(f'Using device: {device}')
    
    # Create output directory
    if not os.path.exists(args.output):
        os.makedirs(args.output)
        print(f'Created output directory: {args.output}')
    
    # Load model
    if not os.path.exists(args.model):
        print(f'Error: Model not found: {args.model}')
        exit(1)
    
    model = load_model(args.model)
    
    # Check if input is file or directory
    if os.path.isfile(args.image):
        # Single image inference
        print('\n====================================================')
        print('Single Image Inference')
        print('====================================================') 
        print(f'Image: {args.image}')
        print(f'Model: {args.model}')
        print(f'Image size: {args.imgsz}')
        print(f'Confidence threshold: {args.conf}')
        print(f'Device: {device}')
        print('====================================================\n')
        
        results = predict_image(
            model,
            args.image,
            imgsz=args.imgsz,
            conf=args.conf,
            device=device
        )
        
        # Save annotated image
        image_basename = os.path.basename(args.image)
        output_filename = f'detected_{image_basename}'
        output_path = os.path.join(args.output, output_filename)
        
        save_detection_image(results[0], output_path)
        print(f' Annotated image saved to: {output_path}')
        
        # Print detections
        print_detections(results[0])
        
        # Optional: Visualize
        if not args.no_visualize:
            print('Displaying result...')
            result_image = results[0].plot()
            result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(12, 8))
            plt.imshow(result_image)
            plt.axis('off')
            plt.title(f'Detections - {image_basename}')
            plt.tight_layout()
            plt.show()
    
    elif os.path.isdir(args.image):
        # Batch image inference
        print('\n====================================================')
        print('Batch Image Inference')
        print('====================================================') 
        print(f'Directory: {args.image}')
        print(f'Model: {args.model}')
        print(f'Image size: {args.imgsz}')
        print(f'Confidence threshold: {args.conf}')
        print(f'Device: {device}')
        print(f'{"="*60}\n')
        
        results = predict_batch(
            model,
            args.image,
            imgsz=args.imgsz,
            conf=args.conf,
            device=device
        )
        
        # Save annotated images
        image_files = [f for f in os.listdir(args.image) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        sample_files = image_files
        
        for i, result in enumerate(results):
            image_basename = os.path.basename(sample_files[i])
            output_filename = f'detected_{image_basename}'
            output_path = os.path.join(args.output, output_filename)
            save_detection_image(result, output_path)
            print(f' Saved: {output_path}')
        
        print(f'\nAll {len(results)} images processed and saved to: {args.output}')
        
        # Optional: Visualize
        if not args.no_visualize:
            print('Displaying results...')
            visualize_predictions(results)
    
    else:
        print(f'Error: File or directory not found: {args.image}')
        exit(1)
    
    print('\n Inference complete!')

"""
Training script for YOLOv8 cricket ball detector model.
To run with customizable parameters
E.g.: python training.py --model yolov8s --yaml cricket_ball_data/dataset.yaml --epochs 50 --batch 8 --imgsz 640
"""

import os
import argparse
import torch
import cv2
import matplotlib.pyplot as plt
import random
from ultralytics import YOLO


def setup_device():
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {DEVICE}')
    print(f'Torch version: {torch.__version__}')
    if DEVICE == 'cuda':
        print(f'CUDA available: {torch.cuda.is_available()}')
    return DEVICE


def setup_paths():
    DATASET_PATH = "/home/smrutibiswal/Projects/edgeFleetAI/cricket_ball_data"
    TEST_PATH = os.path.join(DATASET_PATH, 'test')
    TEST_IMAGES_PATH = os.path.join(TEST_PATH, 'images')
    TEST_LABELS_PATH = os.path.join(TEST_PATH, 'labels')
    TRAIN_PATH = os.path.join(DATASET_PATH, 'train')
    TRAIN_IMAGES_PATH = os.path.join(TRAIN_PATH, 'images')
    TRAIN_LABELS_PATH = os.path.join(TRAIN_PATH, 'labels')
    VALIDATION_PATH = os.path.join(DATASET_PATH, 'valid')
    VALIDATION_IMAGES_PATH = os.path.join(VALIDATION_PATH, 'images')
    VALIDATION_LABELS_PATH = os.path.join(VALIDATION_PATH, 'labels')
    MODEL_PATH = os.path.join(os.getcwd(), 'model')

    return {
        'dataset': DATASET_PATH,
        'train': TRAIN_PATH,
        'train_images': TRAIN_IMAGES_PATH,
        'train_labels': TRAIN_LABELS_PATH,
        'valid': VALIDATION_PATH,
        'valid_images': VALIDATION_IMAGES_PATH,
        'valid_labels': VALIDATION_LABELS_PATH,
        'test': TEST_PATH,
        'test_images': TEST_IMAGES_PATH,
        'test_labels': TEST_LABELS_PATH,
        'model': MODEL_PATH,
    }


def visualize_training_samples(paths, num_samples=8):
    """Visualize random training images with bounding boxes."""
    plt.figure(figsize=(16, 8))

    for i in range(num_samples):
        image_files = os.listdir(paths['train_images'])
        random_image_file = random.choice(image_files)
        image_path = os.path.join(paths['train_images'], random_image_file)

        # Handle different image extensions
        if random_image_file.endswith('.jpg'):
            label_file = random_image_file.replace('.jpg', '.txt')
        elif random_image_file.endswith('.png'):
            label_file = random_image_file.replace('.png', '.txt')
        else:
            label_file = random_image_file.split('.')[0] + '.txt'

        label_path = os.path.join(paths['train_labels'], label_file)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()

            for line in lines:
                class_id, x_center, y_center, bbox_width, bbox_height = map(float, line.strip().split())
                x_center *= width
                y_center *= height
                bbox_width *= width
                bbox_height *= height

                x1 = int(x_center - bbox_width / 2)
                y1 = int(y_center - bbox_height / 2)
                x2 = int(x_center + bbox_width / 2)
                y2 = int(y_center + bbox_height / 2)

                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

        plt.subplot(2, 4, i + 1)
        plt.imshow(image)
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def train_model(model_name='yolov8s', dataset_yaml='/home/smrutibiswal/Projects/edgeFleetAI/cricket_ball_data/dataset.yaml', epochs=20, 
                batch_size=8, imgsz=640, device='cuda', freeze_layers=15):
    """
    Train YOLOv8 model on cricket ball dataset.
    
    Args:
        model_name (str): Base model name (e.g., 'yolov8s', 'yolov8n')
        dataset_yaml (str): Path to dataset YAML file
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        imgsz (int): Image size for training
        device (str): Device to use ('cuda' or 'cpu')
        freeze_layers (int): Number of layers to freeze
    
    Returns:
        YOLO: Trained model
    """
    # Load base model
    model = YOLO(f'{model_name}.pt')

    # Train model
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        freeze=freeze_layers,
        device=device,
        name='cricket_ball_detector',
        patience=5,
        save=True,
        verbose=True
    )

    return model


def validate_model(model, dataset_yaml='/home/smrutibiswal/Projects/edgeFleetAI/cricket_ball_data/dataset.yaml', imgsz=640, batch_size=8, device='cuda'):
    """
    Validate trained model.
    
    Args:
        model: YOLO model instance
        dataset_yaml (str): Path to dataset YAML
        imgsz (int): Image size for validation
        batch_size (int): Batch size
        device (str): Device to use
    
    Returns:
        Results object
    """
    results = model.val(
        data=dataset_yaml,
        imgsz=imgsz,
        batch=batch_size,
        device=device
    )
    return results


def save_model(model, model_path):
    """Save trained model."""
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # Save PyTorch format
    pt_path = os.path.join(model_path, 'cricket_ball_detector.pt')
    model.save(pt_path)
    print(f'Model saved to {pt_path}')

    # Optional: Export to other formats
    try:
        onnx_path = os.path.join(model_path, 'cricket_ball_detector.onnx')
        model.export(format='onnx', imgsz=640)
        print(f'ONNX model saved to {onnx_path}')
    except Exception as e:
        print(f'ONNX export failed: {e}')


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train YOLOv8 cricket ball detector')
    parser.add_argument('--model', type=str, default='yolov8s',
                        help='Model name (yolov8n, yolov8s, yolov8m, etc.)')
    parser.add_argument('--yaml', type=str, default='/home/smrutibiswal/Projects/edgeFleetAI/cricket_ball_data/dataset.yaml',
                        help='Path to dataset YAML file')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Image size for training')
    parser.add_argument('--freeze', type=int, default=15,
                        help='Number of layers to freeze')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cuda, cpu, or auto)')
    parser.add_argument('--no-visualize', action='store_true',
                        help='Skip visualizing training samples')
    parser.add_argument('--no-validate', action='store_true',
                        help='Skip validation after training')
    
    args = parser.parse_args()
    
    # Setup
    if args.device == 'auto':
        device = setup_device()
    else:
        device = args.device
        print(f'Using device: {device}')
    
    paths = setup_paths()
    
    # Verify dataset YAML exists
    if not os.path.exists(args.yaml):
        print(f'Error: Dataset YAML not found: {args.yaml}')
        print(f'Looking in current directory and common locations...')
        exit(1)
    
    # Optional: Visualize training data
    if not args.no_visualize:
        print('Visualizing training samples...')
        visualize_training_samples(paths, num_samples=8)
    
    # Train model
    print('\n====================================================')
    print('Training Configuration:')
    print('====================================================') 
    print(f'Model: {args.model}')
    print(f'Dataset YAML: {args.yaml}')
    print(f'Epochs: {args.epochs}')
    print(f'Batch size: {args.batch}')
    print(f'Image size: {args.imgsz}')
    print(f'Freeze layers: {args.freeze}')
    print(f'Device: {device}')
    print('====================================================\n')
    
    model = train_model(
        model_name=args.model,
        dataset_yaml=args.yaml,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        device=device,
        freeze_layers=args.freeze
    )
    
    # Validate model
    if not args.no_validate:
        print('\nValidating model...')
        validate_model(model, dataset_yaml=args.yaml, device=device)
    
    # Save model
    save_model(model, paths['model'])
    
    print('\n Training complete!')
    print(f'Model saved to: {paths["model"]}')

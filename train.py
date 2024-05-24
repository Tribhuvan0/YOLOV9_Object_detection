import argparse
from ultralytics import YOLO

def train_model(model_path, data_yaml, epochs, img_size, device, classes):
    """
    Train a YOLOv8 model with the specified parameters.

    Parameters:
    - model_path: Path to the pre-trained model.
    - data_yaml: Path to the dataset configuration file.
    - epochs: Number of training epochs.
    - img_size: Image size for training.
    - device: Device to perform the training on (e.g., 'cpu', 'cuda', 'mps').
    - classes: Classes to focus on during training (can be an int or list of ints).
    """
    model = YOLO(model_path)
    results = model.train(data=data_yaml, epochs=epochs, imgsz=img_size, save=True, device=device, classes=classes)
    return results

def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 models for person and PPE detection.")
    parser.add_argument('--person_model_path', type=str, default='yolov8n.pt', help='Path to the pre-trained person detection model')
    parser.add_argument('--ppe_model_path', type=str, default='yolov8n.pt', help='Path to the pre-trained PPE detection model')
    parser.add_argument('--data_yaml_person', type=str, required=True, help='Dataset configuration file for person detection')
    parser.add_argument('--data_yaml_ppe', type=str, required=True, help='Dataset configuration file for PPE detection')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train')
    parser.add_argument('--img_size_person', type=int, default=512, help='Image size for person detection training')
    parser.add_argument('--img_size_ppe', type=int, default=320, help='Image size for PPE detection training')
    parser.add_argument('--device', type=str, default='mps', help='Device to use for training')

    args = parser.parse_args()

    # Train person detection model
    print("Training person detection model...")
    train_model(args.person_model_path, args.data_yaml_person, args.epochs, args.img_size_person, args.device, 0)

    # Train PPE detection model
    print("Training PPE detection model...")
    train_model(args.ppe_model_path, args.data_yaml_ppe, args.epochs, args.img_size_ppe, args.device, [1, 2, 3, 4, 5, 6, 7, 8, 9])

    print("Training completed for both models.")

if __name__ == '__main__':
    main()

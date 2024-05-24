import cv2
from ultralytics import YOLO
from PIL import Image
from pathlib import Path
import argparse
import logging

# Hardcoded path to the model weights file
MODEL_PATH = '/Users/tribhuvan/Documents/yololo/runs_person/detect/train/weights/last.pt'

def load_model(model_path):
    """
    Load the YOLO model from the specified file.
    
    Parameters:
    - model_path: Path to the model weights file.
    
    Returns:
    - Loaded YOLO model object.
    """
    try:
        model = YOLO(model_path)
        logging.info(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise

def draw_bounding_boxes(img, boxes, class_names):
    """
    Draw bounding boxes and labels on the image.
    
    Parameters:
    - img: The image to draw on.
    - boxes: Detected bounding boxes.
    - class_names: List of class names.
    
    Returns:
    - Image with drawn bounding boxes and labels.
    """
    for box in boxes:
        cls = int(box.cls)  # Class ID of the detected object
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
        class_name = class_names[cls]  # Class name of the detected object

        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Position the text label within image boundaries
        text_x = max(x1, 0)
        text_y = max(y1 - 10, 10)  

        # Draw the class label
        cv2.putText(img, class_name, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    return img

def process_image(image_path, model, output_dir):
    """
    Process an image for object detection, draw bounding boxes, and save results.
    
    Parameters:
    - image_path: Path to the image file.
    - model: YOLO model object for inference.
    - output_dir: Directory to save processed images.
    """
    base_name = image_path.stem
    logging.info(f"Processing image: {base_name}")

    # Load the image
    img = cv2.imread(str(image_path))
    if img is None:
        logging.error(f"Failed to load image: {image_path}")
        return

    # Perform object detection
    results = model.predict(img, imgsz=512)

    for result in results:
        boxes = result.boxes  # Detected bounding boxes

        # Draw bounding boxes and labels on the image
        img_with_boxes = draw_bounding_boxes(img.copy(), boxes, result.names)

        # Save the modified image with bounding boxes and labels
        detected_image_path = output_dir / f"{base_name}_detected.jpg"
        cv2.imwrite(str(detected_image_path), img_with_boxes)
        logging.info(f"Detection results saved to {detected_image_path}")

def main(args):
    """
    Main function to handle the inference pipeline.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Load the model
    model = load_model(MODEL_PATH)

    # Ensure the output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each image in the directory
    image_dir = Path(args.image_dir)
    if image_dir.is_dir():
        for image_path in image_dir.glob('*.jpg'):
            process_image(image_path, model, output_dir)
    else:
        logging.error(f"Image directory is not valid: {image_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object Detection and Image Cropping using YOLO")
    parser.add_argument('--image_dir', type=str, required=True, help='Path to the directory containing input images')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the processed images')

    args = parser.parse_args()
    main(args)

import cv2
from pathlib import Path

def parse_annotation(line):
    """
    Parse a line from an annotation file and return bounding box details.
    
    Parameters:
    - line: A string representing one line from an annotation file.

    Returns:
    - Tuple containing class ID and bounding box coordinates (normalized).
    """
    parts = line.split()
    cls, cx, cy, w, h = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
    return cls, cx, cy, w, h

def convert_to_absolute(cx, cy, w, h, img_width, img_height):
    """
    Convert normalized bounding box coordinates to absolute pixel values.

    Parameters:
    - cx, cy, w, h: Normalized bounding box coordinates (center x, center y, width, height).
    - img_width, img_height: Dimensions of the image.

    Returns:
    - Tuple of (x1, y1, x2, y2) representing the absolute pixel coordinates of the bounding box.
    """
    x_center = cx * img_width
    y_center = cy * img_height
    width = w * img_width
    height = h * img_height
    x1 = int(x_center - width / 2)
    y1 = int(y_center - height / 2)
    x2 = int(x_center + width / 2)
    y2 = int(y_center + height / 2)
    return x1, y1, x2, y2

def crop_and_save_images(image_dir, annotation_dir, output_dir):
    """
    Crop images based on annotations and save them to a specified directory.

    Parameters:
    - image_dir: Path to the directory containing images.
    - annotation_dir: Path to the directory containing annotation files.
    - output_dir: Path where cropped images will be saved.
    """
    image_dir = Path(image_dir)
    annotation_dir = Path(annotation_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)  

    for annotation_path in annotation_dir.glob('*.txt'):
        image_path = image_dir / f"{annotation_path.stem}.jpg"
        if not image_path.exists():
            print(f"No corresponding image found for {annotation_path}")
            continue

        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Failed to load image {image_path}")
            continue

        with open(annotation_path, 'r') as file:
            person_count = 0
            for line in file:
                cls, cx, cy, w, h = parse_annotation(line)
                if cls == 0:  # Process only class '0'
                    x1, y1, x2, y2 = convert_to_absolute(cx, cy, w, h, img.shape[1], img.shape[0])
                    cropped_img = img[y1:y2, x1:x2]
                    crop_filename = f"{annotation_path.stem}_{person_count}.jpg"
                    cv2.imwrite(str(output_dir / crop_filename), cropped_img)
                    person_count += 1
                    print(f"Cropped image saved as {crop_filename}")


image_dir = 'path/to/images_crop'  # Path to the folder containing images
annotation_dir = 'path/to/labels'  # Path to the folder containing annotations
output_dir = 'path/to/cropped_imgs'  # Path where cropped images will be saved

crop_and_save_images(image_dir, annotation_dir, output_dir)

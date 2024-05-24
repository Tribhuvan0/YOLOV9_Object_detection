import os
from pathlib import Path

# Constants for scaling
SCALE_FACTOR = 640
CENTER_OFFSET = SCALE_FACTOR / 2

def parse_annotation(line):
    """
    Parse a single line from an annotation file to extract class ID and bounding box details.

    Parameters:
    - line: A string representing one line of an annotation.

    Returns:
    - Tuple containing class ID and normalized bounding box coordinates (cx, cy, w, h).
    """
    parts = line.split()
    cls, cx, cy, w, h = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
    return cls, cx, cy, w, h

def calculate_absolute_bbox(cx, cy, w, h):
    """
    Calculate absolute bounding box coordinates from normalized values.

    Parameters:
    - cx, cy, w, h: Normalized center coordinates and dimensions of the bounding box.

    Returns:
    - Tuple of (x1, y1, x2, y2) representing the absolute pixel coordinates of the bounding box.
    """
    img_width = w * SCALE_FACTOR
    img_height = h * SCALE_FACTOR
    x1 = cx * SCALE_FACTOR - w * CENTER_OFFSET
    y1 = cy * SCALE_FACTOR - h * CENTER_OFFSET
    x2 = cx * SCALE_FACTOR + w * CENTER_OFFSET
    y2 = cy * SCALE_FACTOR + h * CENTER_OFFSET
    return int(x1), int(y1), int(x2), int(y2)

def check_overlap(box1, box2):
    """
    Check if two bounding boxes overlap.

    Parameters:
    - box1, box2: Tuples of (x1, y1, x2, y2) representing the absolute pixel coordinates of two bounding boxes.

    Returns:
    - Boolean indicating if there is any overlap.
    """
    return not (box2[2] < box1[0] or box2[0] > box1[2] or box2[3] < box1[1] or box2[1] > box1[3])

def process_annotations(annotations_path, output_dir):
    """
    Process each annotation file to find overlaps and adjust bounding box coordinates relative to 'person' class bounding boxes.

    Parameters:
    - annotations_path: Path object pointing to the annotation file.
    - output_dir: Path object pointing to the directory where processed files will be saved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)  # Ensure the output directory exists

    with open(annotations_path, 'r') as file:
        lines = file.readlines()

    persons = []
    others = []

    # Classify annotations
    for line in lines:
        cls, cx, cy, w, h = parse_annotation(line)
        if cls == 0:
            persons.append((cls, cx, cy, w, h))
        elif cls != 0 and cls != 6 and cls != 3:
            others.append((cls, cx, cy, w, h))

    # Process each 'person' annotation
    for index, person in enumerate(persons):
        cls, pcx, pcy, pw, ph = person
        person_bbox = calculate_absolute_bbox(pcx, pcy, pw, ph)

        contained_annotations = [f"{cls} {pcx:.6f} {pcy:.6f} {pw:.6f} {ph:.6f}\n"]
        for other in others:
            ocls, ocx, ocy, ow, oh = other
            other_bbox = calculate_absolute_bbox(ocx, ocy, ow, oh)
            if check_overlap(person_bbox, other_bbox):
                # Calculate relative positions
                norm_x, norm_y, norm_w, norm_h = adjust_bbox(other_bbox, person_bbox)
                contained_annotations.append(f"{ocls} {norm_x:.6f} {norm_y:.6f} {norm_w:.6f} {norm_h:.6f}\n")

        
        output_path = output_dir / f"{annotations_path.stem}_{index}.txt"
        with open(output_path, 'w') as f:
            f.writelines(contained_annotations)

def adjust_bbox(bbox, reference_bbox):
    """
    Adjust the bounding box coordinates relative to a reference bounding box.

    Parameters:
    - bbox: Tuple (x1, y1, x2, y2) of the bounding box to adjust.
    - reference_bbox: Tuple (x1, y1, x2, y2) of the reference bounding box.

    Returns:
    - Tuple of (norm_x, norm_y, norm_w, norm_h) normalized coordinates relative to the reference.
    """
    norm_x = (bbox[0] - reference_bbox[0]) / (reference_bbox[2] - reference_bbox[0])
    norm_y = (bbox[1] - reference_bbox[1]) / (reference_bbox[3] - reference_bbox[1])
    norm_w = (bbox[2] - bbox[0]) / (reference_bbox[2] - reference_bbox[0])
    norm_h = (bbox[3] - bbox[1]) / (reference_bbox[3] - reference_bbox[1])
    return max(0, min(1, norm_x)), max(0, min(1, norm_y)), max(0, min(1, norm_w)), max(0, min(1, norm_h))


if __name__ == "__main__":
    annotations_dir = Path('updated_anno/labels') #
    output_dir = Path('updated_anno/cropped_anno')
    for annotations_file in annotations_dir.glob('*.txt'):
        process_annotations(annotations_file, output_dir)

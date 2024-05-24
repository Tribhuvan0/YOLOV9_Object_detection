import os
import xml.etree.ElementTree as ET
import argparse

def convert(size, box):
    """
    Convert Pascal VOC bounding box format to YOLO format.

    Parameters:
    - size: Tuple of (width, height) of the image.
    - box: Tuple of (xmin, xmax, ymin, ymax) for the bounding box.

    Returns:
    - Tuple of (x_center, y_center, width, height) normalized by image size.
    """
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0 - box[0]
    y = (box[2] + box[3]) / 2.0 - box[2]
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def convert_annotation(input_dir, output_dir, image_id, classes):
    """
    Convert an XML annotation file to YOLO format and save it.

    Parameters:
    - input_dir: Directory containing the XML files.
    - output_dir: Directory to save the converted YOLO annotations.
    - image_id: The base name of the image file to process.
    - classes: List of class labels.
    """
    in_file = open(os.path.join(input_dir, f'{image_id}.xml'), 'r')
    out_file = open(os.path.join(output_dir, f'{image_id}.txt'), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls not in classes:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), 
             float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

    in_file.close()
    out_file.close()

def main():
    """
    Main function to process XML files and convert them to YOLO format.
    """
    parser = argparse.ArgumentParser(description="Convert Pascal VOC annotation XMLs to YOLO format text files.")
    parser.add_argument('input_dir', type=str, help='Base directory for dataset containing XML annotations')
    parser.add_argument('output_dir', type=str, help='Directory to save converted YOLO format annotations')
    args = parser.parse_args()

    # Hard-coded class labels
    classes = ['person', 'hard-hat', 'gloves', 'mask', 'glasses', 'boots', 'vest', 'ppe-suit', 'ear-protector', 'safety-harness']

    # Check and create output directory if not exist
    if not os.path.exists(args.output_dir):
        try:
            os.makedirs(args.output_dir)
        except OSError as e:
            print(f"Error creating directory {args.output_dir}: {e}")
            return

    # Process each XML file
    try:
        image_ids = [os.path.splitext(file)[0] for file in os.listdir(args.input_dir) if file.endswith('.xml')]
        for image_id in image_ids:
            convert_annotation(args.input_dir, args.output_dir, image_id, classes)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()

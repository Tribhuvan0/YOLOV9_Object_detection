import os
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt

def count_classes_in_annotations(annotation_dir):
    """
    Count how many times each class appears in the annotation files in the given directory.

    Parameters:
    - annotation_dir: The path to the directory containing annotation files.

    Returns:
    - A Counter object with the count of each class.
    """
    class_counts = Counter()
    files = Path(annotation_dir).glob('*.txt')  # Get all .txt files in the directory

    for file_path in files:
        with open(file_path, 'r') as file: 
            for line in file:
                if line.strip():  
                    class_id = int(line.split()[0])  # Get the class ID from the line
                    class_counts[class_id] += 1  # Add to the count for this class

    return class_counts

def plot_class_distribution(class_counts):
    """
    Make a bar graph showing how often each class appears.

    Parameters:
    - class_counts: A Counter object with the count of each class.
    """
    # Get the data ready for plotting
    classes = list(class_counts.keys())
    counts = [class_counts[cls] for cls in classes]

    # Create the bar plot
    plt.figure(figsize=(10, 5))
    plt.bar(classes, counts, color='blue')
    plt.xlabel('Class ID')
    plt.ylabel('Frequency')
    plt.title('Class Distribution in Annotations')
    plt.xticks(classes)  
    plt.grid(True)  
    plt.show()

# Path to the directory with annotation files
annotation_dir = 'path/to/annotations'  
# Count the classes in the annotation files
class_counts = count_classes_in_annotations(annotation_dir)
print(class_counts)
# Plot the class distribution
plot_class_distribution(class_counts)

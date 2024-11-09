# dataset_prep.py

import os
import cv2
import numpy as np
from pathlib import Path

class FlowchartDatasetPrep:
    def __init__(self, data_dir: str = "dataset"):
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images"
        self.labels_dir = self.data_dir / "labels"
        self.classes = ['oval', 'rectangle', 'diamond']
        
        # Create directories
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        
    def create_yolo_label(self, image_size, boxes, classes):
        """
        Convert bounding boxes to YOLO format
        Format: <class> <x_center> <y_center> <width> <height>
        All values are normalized to [0, 1]
        """
        labels = []
        img_h, img_w = image_size
        
        for box, cls in zip(boxes, classes):
            # Convert box coordinates (x1, y1, x2, y2) to YOLO format
            x1, y1, x2, y2 = box
            x_center = (x1 + x2) / (2 * img_w)
            y_center = (y1 + y2) / (2 * img_h)
            width = (x2 - x1) / img_w
            height = (y2 - y1) / img_h
            
            # Get class index
            class_idx = self.classes.index(cls)
            
            labels.append(f"{class_idx} {x_center} {y_center} {width} {height}")
            
        return labels
    
    def save_dataset_file(self, image, filename, boxes, classes):
        """
        Save image and its corresponding label file
        """
        # Save image
        img_path = self.images_dir / f"{filename}.png"
        cv2.imwrite(str(img_path), image)
        
        # Create and save label
        labels = self.create_yolo_label(image.shape[:2], boxes, classes)
        label_path = self.labels_dir / f"{filename}.txt"
        with open(label_path, 'w') as f:
            f.write('\n'.join(labels))
            
    def create_data_yaml(self):
        """
        Create data.yaml file for YOLO training
        """
        yaml_content = f"""
train: {str(self.images_dir)}/train
val: {str(self.images_dir)}/val
test: {str(self.images_dir)}/test

nc: {len(self.classes)}
names: {self.classes}
        """
        
        with open(self.data_dir / "data.yaml", 'w') as f:
            f.write(yaml_content.strip())

# Example usage
def main():
    prep = FlowchartDatasetPrep()
    
    # Example of labeling a single image
    image = cv2.imread("flowchart.png")
    boxes = [
        [0, 0, 100, 50],    # Start (oval)
        [0, 100, 200, 150], # Process (rectangle)
        [100, 200, 200, 300] # Decision (diamond)
    ]
    classes = ['oval', 'rectangle', 'diamond']
    
    prep.save_dataset_file(image, "example", boxes, classes)
    prep.create_data_yaml()

if __name__ == "__main__":
    main()
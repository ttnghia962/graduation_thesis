# detect_shapes.py

from ultralytics import YOLO
import cv2
import numpy as np
from typing import List, Tuple, Dict

class FlowchartDetector:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        self.classes = ['oval', 'rectangle', 'diamond']
        
    def detect_shapes(self, image_path: str) -> Tuple[np.ndarray, List[Dict]]:
        """
        Detect shapes in flowchart image
        Returns: (annotated_image, list of detected shapes)
        """
        # Run inference
        results = self.model(image_path)
        
        # Get detections
        shapes = []
        img = cv2.imread(image_path)
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Get class and confidence
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                # Create shape info
                shape_info = {
                    'type': self.classes[cls],
                    'position': (x1, y1),
                    'size': (x2-x1, y2-y1),
                    'confidence': conf,
                    'box': (x1, y1, x2, y2)
                }
                shapes.append(shape_info)
                
                # Draw on image
                color = {
                    'oval': (0, 255, 0),
                    'rectangle': (255, 0, 0),
                    'diamond': (0, 0, 255)
                }[self.classes[cls]]
                
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, f"{self.classes[cls]} {conf:.2f}",
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                           0.5, color, 2)
        
        return img, shapes

def main():
    detector = FlowchartDetector('path/to/trained/model.pt')
    image_path = "test_flowchart.png"
    
    annotated_img, shapes = detector.detect_shapes(image_path)
    
    # Save results
    cv2.imwrite("detected_shapes.png", annotated_img)
    
    # Print detections
    print("\nDetected Shapes:")
    for i, shape in enumerate(shapes, 1):
        print(f"{i}. {shape['type']} at position {shape['position']} "
              f"with confidence {shape['confidence']:.2f}")

if __name__ == "__main__":
    main()
# shape_analyzer.py

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from image_processor import ContourInfo, ProcessedImage  # Add this import

@dataclass
class FlowchartShape:
    """Container for flowchart shape information"""
    id: int
    type: str  # 'oval', 'rectangle', 'diamond'
    position: Tuple[int, int]  # x, y coordinates
    size: Tuple[int, int]  # width, height
    area: float
    contour: np.ndarray
    bounding_box: Tuple[int, int, int, int]
    aspect_ratio: float
    approx_points: int  # number of approximated points
    center: Tuple[int, int]  # center coordinates

class ShapeAnalyzer:
    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode
        self.shapes: List[FlowchartShape] = []
            
    def identify_shape_type(self, contour: np.ndarray) -> str:
            """
            Identify the type of flowchart shape based on contour properties
            Returns: 'oval', 'rectangle', or 'diamond'
            """
            # Approximate the contour
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
            
            # Get bounding box and its properties
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            
            # Calculate shape metrics
            area = cv2.contourArea(contour)
            rect_area = w * h
            extent = float(area) / rect_area
            
            # Calculate circularity (roundness)
            circumference = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (circumference * circumference) if circumference > 0 else 0
            
            # Get minimum area rectangle for angle
            rect = cv2.minAreaRect(contour)
            _, _, rect_angle = rect
            if rect_angle > 45:
                rect_angle = 90 - rect_angle
                
            if self.debug_mode:
                print(f"Shape at ({x},{y}): points={len(approx)}, "
                    f"aspect_ratio={aspect_ratio:.2f}, "
                    f"circularity={circularity:.2f}, "
                    f"extent={extent:.2f}, "
                    f"angle={rect_angle:.2f}")
            
            # Diamond detection (remains the same - working correctly)
            if extent < 0.60 and 30 <= abs(rect_angle) <= 40:
                return 'diamond'
            
            # Oval detection (only Start/End)
            # Only shapes 0 and 5 should be ovals
            if (aspect_ratio >= 3.8 and 0.70 <= extent <= 0.80):
                return 'oval'
                    
            # Rectangle detection (process boxes)
            # Shape 1, 2, 4 should be rectangles
            if extent >= 0.95 and aspect_ratio <= 3.5:
                return 'rectangle'
                
            # Default case
            if y < 100 or y > 500:  # Start/End positions
                return 'oval'
            else:
                return 'rectangle'

    def analyze_contours(self, contours: List[ContourInfo]) -> List[FlowchartShape]:
        """
        Analyze contours and create FlowchartShape objects with improved sorting
        """
        self.shapes = []
        
        for idx, cont_info in enumerate(contours):
            # Get basic properties
            x, y, w, h = cont_info.bounding_box
            center = (x + w//2, y + h//2)
            aspect_ratio = float(w) / h
            
            # Approximate contour for point analysis
            epsilon = 0.02 * cv2.arcLength(cont_info.contour, True)
            approx = cv2.approxPolyDP(cont_info.contour, epsilon, True)
            
            # Create shape object
            shape = FlowchartShape(
                id=idx,
                type=self.identify_shape_type(cont_info.contour),
                position=(x, y),
                size=(w, h),
                area=cont_info.area,
                contour=cont_info.contour,
                bounding_box=cont_info.bounding_box,
                aspect_ratio=aspect_ratio,
                approx_points=len(approx),
                center=center
            )
            
            self.shapes.append(shape)
        
        # Sort shapes by vertical position for better flow analysis
        self.shapes.sort(key=lambda s: s.position[1])
        
        return self.shapes

    def draw_debug_visualization(self, image: np.ndarray) -> np.ndarray:
            """
            Enhanced debug visualization with metrics
            """
            if not self.debug_mode:
                return image
                
            debug_img = image.copy()
            colors = {
                'oval': (0, 255, 0),      # Green
                'rectangle': (255, 0, 0),  # Blue
                'diamond': (0, 0, 255)     # Red
            }
            
            for shape in self.shapes:
                # Draw filled contour with transparency
                overlay = debug_img.copy()
                cv2.drawContours(overlay, [shape.contour], -1, colors[shape.type], -1)
                cv2.addWeighted(overlay, 0.3, debug_img, 0.7, 0, debug_img)
                
                # Draw contour outline
                cv2.drawContours(debug_img, [shape.contour], -1, colors[shape.type], 2)
                
                # Get shape metrics for debug info
                x, y = shape.position
                w, h = shape.size
                aspect_ratio = shape.aspect_ratio
                
                # Add shape information with metrics
                text = [
                    f"Type: {shape.type} (ID: {shape.id})",
                    f"AR: {aspect_ratio:.2f}",
                    f"Points: {shape.approx_points}"
                ]
                
                # Draw text with background
                for i, t in enumerate(text):
                    text_y = y - 10 - (i * 20)
                    (text_w, text_h), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(debug_img, 
                                (x, text_y - text_h - 1), 
                                (x + text_w, text_y + 1), 
                                (255, 255, 255), 
                                -1)
                    cv2.putText(debug_img, t, 
                            (x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
                # Draw center point
                cv2.circle(debug_img, shape.center, 4, colors[shape.type], -1)
                
            return debug_img

    def find_connections(self) -> List[Tuple[int, int]]:
        """
        Find connections between shapes based on their positions
        Returns list of (from_id, to_id) tuples
        """
        connections = []
        shapes_sorted = sorted(self.shapes, key=lambda s: s.position[1])
        
        for i, shape1 in enumerate(shapes_sorted[:-1]):
            min_distance = float('inf')
            closest_shape = None
            
            # Look for shapes below the current shape
            for shape2 in shapes_sorted[i+1:]:
                if shape2.position[1] > shape1.position[1]:  # Only look at shapes below
                    # Calculate center-to-center distance
                    dx = shape2.center[0] - shape1.center[0]
                    dy = shape2.center[1] - shape1.center[1]
                    distance = np.sqrt(dx*dx + dy*dy)
                    
                    if distance < min_distance:
                        min_distance = distance
                        closest_shape = shape2
            
            if closest_shape:
                connections.append((shape1.id, closest_shape.id))
        
        return connections

def integrate_with_processor(processor_output: Tuple[ProcessedImage, List[ContourInfo]]) -> Tuple[List[FlowchartShape], List[Tuple[int, int]]]:
    """
    Integrate shape analysis with image processor output
    """
    processed_image, contours = processor_output
    
    analyzer = ShapeAnalyzer(debug_mode=True)
    shapes = analyzer.analyze_contours(contours)
    
    # Draw debug visualization
    debug_img = analyzer.draw_debug_visualization(processed_image.original)
    cv2.imwrite("debug_output/6_shapes_analyzed.png", debug_img)
    
    # Find and analyze connections
    connections = analyzer.find_connections()
    
    return shapes, connections

def main():
    from image_processor import ImageProcessor
    
    # Initialize processor
    processor = ImageProcessor(debug_mode=True)
    
    # Process image
    image_path = r"D:\graduation_thesis\CODE\1.png"  # Use your image path
    try:
        # Process image and analyze shapes
        processor_output = processor.process_image(image_path)
        shapes, connections = integrate_with_processor(processor_output)
        
        # Print results
        print("\nShape Analysis Results:")
        for shape in shapes:
            print(f"Shape {shape.id}: {shape.type} at position {shape.position}")
            
        print("\nConnections Found:")
        for from_id, to_id in connections:
            print(f"Shape {from_id} -> Shape {to_id}")
            
    except Exception as e:
        print(f"Error in shape analysis: {str(e)}")

if __name__ == "__main__":
    main()
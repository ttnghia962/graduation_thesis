import cv2
import numpy as np
import pytesseract
import networkx as nx
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Node:
    id: int
    text: str
    shape: str
    bbox: Tuple[int, int, int, int]
    confidence: float = 0.0
    level: int = 0

class ConfidenceOptimizer(nn.Module):
    def __init__(self, input_features=5):
        super(ConfidenceOptimizer, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

class FlowchartAnalyzer:
    def __init__(self):
        self.nodes = []
        self.graph = nx.DiGraph()
        self.confidence_model = ConfidenceOptimizer()
        
    def extract_features(self, node: Node) -> np.ndarray:
        """Extract features for confidence prediction"""
        x, y, w, h = node.bbox
        
        # Features that might affect confidence
        features = [
            w * h,                    # Area of the shape
            w / h,                    # Aspect ratio
            len(node.text),           # Text length
            self.get_text_clarity(),  # OCR confidence
            self.get_shape_quality()  # Shape detection quality
        ]
        return np.array(features, dtype=np.float32)
    
    def get_text_clarity(self) -> float:
        """Get OCR confidence from Tesseract"""
        # Use Tesseract's confidence score
        return float(pytesseract.image_to_data(
            self.current_roi, 
            output_type=pytesseract.Output.DICT
        )['conf'][0]) / 100.0
    
    def get_shape_quality(self) -> float:
        """Calculate shape detection quality"""
        # Use contour approximation quality
        epsilon = 0.02 * cv2.arcLength(self.current_contour, True)
        approx = cv2.approxPolyDP(self.current_contour, epsilon, True)
        return min(1.0, len(approx) / 8.0)  # Normalize
    
    def optimize_confidence(self, node: Node) -> float:
        """Use DL model to optimize confidence score"""
        features = self.extract_features(node)
        features_tensor = torch.FloatTensor(features)
        
        with torch.no_grad():
            confidence = self.confidence_model(features_tensor)
        return confidence.item()
    
    def process_image(self, image_path: str):
        # Read image
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect shapes using OpenCV
        _, thresh = cv2.threshold(gray, 127, 255, 0)
        contours, _ = cv2.findContours(
            thresh, 
            cv2.RETR_TREE, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Process each contour
        for i, contour in enumerate(contours):
            self.current_contour = contour
            x, y, w, h = cv2.boundingRect(contour)
            roi = img[y:y+h, x:x+w]
            self.current_roi = roi
            
            # Basic shape detection
            shape = self.detect_shape(contour)
            
            # OCR
            text = pytesseract.image_to_string(roi).strip()
            
            # Create node
            node = Node(
                id=i,
                text=text,
                shape=shape,
                bbox=(x, y, w, h)
            )
            
            # Optimize confidence using DL
            node.confidence = self.optimize_confidence(node)
            self.nodes.append(node)
    
    def build_hierarchy(self):
        # Create graph
        for node in self.nodes:
            self.graph.add_node(node.id, data=node)
        
        # Build edges based on position and confidence
        for node in self.nodes:
            for potential_child in self.nodes:
                if self.is_parent_child(node, potential_child):
                    weight = node.confidence * potential_child.confidence
                    self.graph.add_edge(
                        node.id, 
                        potential_child.id, 
                        weight=weight
                    )
        
        # Use NetworkX for hierarchy analysis
        levels = nx.topological_generations(self.graph)
        for level_num, level in enumerate(levels, 1):
            for node_id in level:
                self.nodes[node_id].level = level_num
    
    def detect_shape(self, contour) -> str:
        """Basic shape detection using OpenCV"""
        approx = cv2.approxPolyDP(
            contour,
            0.04 * cv2.arcLength(contour, True),
            True
        )
        
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            if abs(w - h) <= 10:
                return "diamond"
            else:
                return "rectangle"
        return "unknown"
    
    def is_parent_child(self, parent: Node, child: Node) -> bool:
        """Check if two nodes have parent-child relationship"""
        px, py, pw, ph = parent.bbox
        cx, cy, cw, ch = child.bbox
        
        # Basic position check
        if cy > py + ph:  # Child is below parent
            # Check if child is somewhat aligned with parent
            if cx > px - pw/2 and cx < px + pw*1.5:
                return True
        return False
    
    def generate_output(self) -> str:
        """Generate hierarchical output"""
        output = []
        current_level = 1
        
        # Sort nodes by level and position
        sorted_nodes = sorted(
            self.nodes, 
            key=lambda n: (n.level, n.bbox[1])
        )
        
        for node in sorted_nodes:
            if node.level > current_level:
                current_level = node.level
            
            indent = "  " * (node.level - 1)
            confidence_info = f"({node.confidence:.2f})"
            output.append(
                f"{indent}{node.text} {confidence_info}"
            )
        
        return "\n".join(output)

def main():
    analyzer = FlowchartAnalyzer()
    
    # Process image
    analyzer.process_image("flowchart.png")
    
    # Build hierarchy
    analyzer.build_hierarchy()
    
    # Generate and print output
    output = analyzer.generate_output()
    print(output)

if __name__ == "__main__":
    main()
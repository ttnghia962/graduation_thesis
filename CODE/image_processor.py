# image_processor.py

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List
import os

@dataclass
class ProcessedImage:
    """Container for processed image outputs"""
    original: np.ndarray
    grayscale: np.ndarray
    binary: np.ndarray
    enhanced: np.ndarray
    debug_path: str = None

@dataclass
class ContourInfo:
    """Container for contour information"""
    contour: np.ndarray
    position: Tuple[int, int]  # x, y
    area: float
    bounding_box: Tuple[int, int, int, int]  # x, y, w, h

class ImageProcessor:
    def __init__(self, debug_mode: bool = False, debug_dir: str = "debug_output"):
        """
        Initialize the image processor
        
        Args:
            debug_mode: If True, save intermediate steps
            debug_dir: Directory to save debug images
        """
        self.debug_mode = debug_mode
        self.debug_dir = debug_dir
        if debug_mode:
            os.makedirs(debug_dir, exist_ok=True)

    def load_image(self, image_path: str) -> np.ndarray:
        """Load image and verify it exists"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")
        return image

    def convert_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale if it's not already"""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def enhance_contrast(self, gray_image: np.ndarray) -> np.ndarray:
        """Enhance image contrast using CLAHE"""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        return clahe.apply(gray_image)

    def denoise_image(self, image: np.ndarray) -> np.ndarray:
        """Remove noise while preserving edges"""
        return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

    def create_binary_image(self, gray_image: np.ndarray) -> np.ndarray:
        """Create binary image using adaptive thresholding"""
        binary = cv2.adaptiveThreshold(
            gray_image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=11,
            C=2
        )
        
        # Clean up binary image
        kernel = np.ones((3,3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        return binary

    def enhance_text(self, gray_image: np.ndarray) -> np.ndarray:
        """Enhance text for better OCR"""
        # Sharpen
        kernel = np.array([[-1,-1,-1],
                         [-1, 9,-1],
                         [-1,-1,-1]])
        sharpened = cv2.filter2D(gray_image, -1, kernel)
        
        # Denoise while preserving text
        return cv2.fastNlMeansDenoising(
            sharpened,
            None,
            h=10,
            templateWindowSize=7,
            searchWindowSize=21
        )

    def find_contours(self, binary_image: np.ndarray, 
                     min_area: int = 1000) -> List[ContourInfo]:
        """Find and filter contours"""
        contours, _ = cv2.findContours(
            binary_image,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        filtered_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                filtered_contours.append(ContourInfo(
                    contour=contour,
                    position=(x, y),
                    area=area,
                    bounding_box=(x, y, w, h)
                ))
        
        # Sort by vertical position
        filtered_contours.sort(key=lambda x: x.position[1])
        return filtered_contours

    def save_debug_image(self, name: str, image: np.ndarray):
        """Save intermediate results for debugging"""
        if self.debug_mode:
            path = os.path.join(self.debug_dir, f"{name}.png")
            cv2.imwrite(path, image)

    def process_image(self, image_path: str) -> Tuple[ProcessedImage, List[ContourInfo]]:
        """
        Main processing pipeline
        
        Args:
            image_path: Path to input image
            
        Returns:
            Tuple containing:
                - ProcessedImage object with all processed versions
                - List of ContourInfo objects
        """
        # Load image
        original = self.load_image(image_path)
        self.save_debug_image("1_original", original)

        # Convert to grayscale
        gray = self.convert_to_grayscale(original)
        self.save_debug_image("2_grayscale", gray)

        # Create binary image for shape detection
        denoised = self.denoise_image(gray)
        binary = self.create_binary_image(denoised)
        self.save_debug_image("3_binary", binary)

        # Create enhanced image for text recognition
        contrast_enhanced = self.enhance_contrast(gray)
        text_enhanced = self.enhance_text(contrast_enhanced)
        self.save_debug_image("4_enhanced", text_enhanced)

        # Find contours
        contours = self.find_contours(binary)
        
        # Draw contours on debug image if needed
        if self.debug_mode:
            debug_img = original.copy()
            for cont_info in contours:
                cv2.drawContours(debug_img, [cont_info.contour], -1, (0, 255, 0), 2)
                x, y, w, h = cont_info.bounding_box
                cv2.rectangle(debug_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            self.save_debug_image("5_contours", debug_img)

        processed_image = ProcessedImage(
            original=original,
            grayscale=gray,
            binary=binary,
            enhanced=text_enhanced,
            debug_path=self.debug_dir if self.debug_mode else None
        )

        return processed_image, contours

# Example usage
def main():
    # Initialize processor in debug mode
    processor = ImageProcessor(debug_mode=True)
    
    # Process image
    image_path = r"D:\graduation_thesis\CODE\1.png"  # <-- Input image path goes here
    try:
        processed_image, contours = processor.process_image(image_path)
        
        # Print some information about found contours
        print(f"Found {len(contours)} shapes in the image")
        for i, cont_info in enumerate(contours, 1):
            print(f"Shape {i}: Position {cont_info.position}, Area: {cont_info.area:.2f}")
            
    except Exception as e:
        print(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main()
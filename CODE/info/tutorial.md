Tôi sẽ giúp bạn tối ưu hóa việc xử lý ảnh flowchart để chuẩn bị cho OCR. Đây là những bước xử lý ảnh quan trọng:



```python
import cv2
import numpy as np

class FlowchartPreprocessor:
    @staticmethod
    def preprocess_image(image_path: str) -> tuple:
        """
        Preprocess flowchart image to enhance text and shape recognition.
        Returns tuple of processed images: (binary, text_enhanced)
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Could not read image")
            
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 1. Create binary image for shape detection
        # Remove noise with bilateral filter (preserves edges better than Gaussian)
        denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
        
        # Adaptive thresholding for better handling of uneven lighting
        binary = cv2.adaptiveThreshold(
            denoised,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=11,
            C=2
        )
        
        # 2. Create enhanced image for text recognition
        # Increase contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        contrast_enhanced = clahe.apply(gray)
        
        # Sharpen image
        kernel = np.array([[-1,-1,-1],
                         [-1, 9,-1],
                         [-1,-1,-1]])
        sharpened = cv2.filter2D(contrast_enhanced, -1, kernel)
        
        # Denoise again but preserve text
        text_enhanced = cv2.fastNlMeansDenoising(
            sharpened,
            None,
            h=10,
            templateWindowSize=7,
            searchWindowSize=21
        )
        
        # Optional: Morphological operations to clean up binary image
        kernel = np.ones((3,3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        return binary, text_enhanced
    
    @staticmethod
    def debug_save_steps(image_path: str, output_dir: str):
        """
        Save intermediate processing steps for debugging and visualization
        """
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Get processed images
        binary, text_enhanced = FlowchartPreprocessor.preprocess_image(image_path)
        
        # Save all steps
        cv2.imwrite(f"{output_dir}/1_original.png", image)
        cv2.imwrite(f"{output_dir}/2_grayscale.png", gray)
        cv2.imwrite(f"{output_dir}/3_binary.png", binary)
        cv2.imwrite(f"{output_dir}/4_text_enhanced.png", text_enhanced)

    @staticmethod
    def get_contours(binary_image: np.ndarray, min_area: int = 1000) -> list:
        """
        Extract and filter contours from binary image
        """
        # Find contours
        contours, hierarchy = cv2.findContours(
            binary_image,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter by area and sort by position
        filtered_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                filtered_contours.append({
                    'contour': contour,
                    'position': (x, y),
                    'area': area,
                    'bbox': (x, y, w, h)
                })
        
        # Sort by vertical position
        filtered_contours.sort(key=lambda x: x['position'][1])
        
        return filtered_contours

```

Tôi đã tạo một class `FlowchartPreprocessor` với các cải tiến quan trọng để xử lý ảnh flowchart. Đây là những điểm chính:

1. **Xử lý nhiễu và làm sắc nét ảnh**:
   - Sử dụng `bilateralFilter` thay vì `GaussianBlur` để giữ lại các cạnh tốt hơn
   - Áp dụng CLAHE (Contrast Limited Adaptive Histogram Equalization) để cải thiện độ tương phản
   - Thêm bước làm sắc nét ảnh (sharpening)

2. **Tạo hai phiên bản ảnh riêng biệt**:
   - Ảnh nhị phân (binary) để nhận dạng hình dạng
   - Ảnh đã được tăng cường (text_enhanced) để OCR text

3. **Cải thiện phát hiện contour**:
   - Sử dụng ngưỡng thích ứng (adaptive thresholding)
   - Thêm các phép toán morphological để làm sạch nhiễu
   - Lọc và sắp xếp contour theo vị trí

4. **Chức năng debug**:
   - Thêm method `debug_save_steps()` để lưu từng bước xử lý
   - Giúp kiểm tra và tinh chỉnh các thông số

Để sử dụng trong code hiện tại của bạn, bạn có thể thay đổi phương thức `detect_shapes` như sau:

```python
def detect_shapes(self, image_path: str):
    # Preprocess image
    preprocessor = FlowchartPreprocessor()
    binary, text_enhanced = preprocessor.preprocess_image(image_path)
    
    # Get filtered contours
    filtered_contours = preprocessor.get_contours(binary)
    
    # Rest of your shape detection code...
```

Bạn có muốn tôi giải thích thêm về bất kỳ bước xử lý nào không? Hoặc bạn muốn tôi điều chỉnh thêm các thông số nào không?
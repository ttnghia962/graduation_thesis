import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_path
from typing import List, Tuple, Dict
import math

class FlowchartRecognition:
    def __init__(self):
        # Khởi tạo các tham số
        self.min_contour_area = 100
        self.max_contour_area = 50000
        
    def read_image(self, file_path: str) -> np.ndarray:
        """Đọc ảnh từ nhiều định dạng khác nhau."""
        if file_path.lower().endswith('.pdf'):
            # Xử lý file PDF
            pages = convert_from_path(file_path)
            # Lấy trang đầu tiên và chuyển sang numpy array
            return cv2.cvtColor(np.array(pages[0]), cv2.COLOR_RGB2BGR)
        else:
            # Đọc ảnh thông thường
            return cv2.imread(file_path)
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Tiền xử lý ảnh."""
        # Chuyển sang ảnh xám
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Perspective correction
        # Tìm các góc của ảnh
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
        if lines is not None:
            # Tính toán góc nghiêng
            angle = self._calculate_skew_angle(lines)
            # Xoay ảnh nếu cần
            if abs(angle) > 0.5:
                rows, cols = gray.shape
                M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
                gray = cv2.warpAffine(gray, M, (cols, rows))
        
        # Giảm nhiễu
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Điều chỉnh độ tương phản
        gray = cv2.equalizeHist(gray)
        
        # Ngưỡng hóa
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
    
    def detect_shapes(self, binary_image: np.ndarray) -> Dict:
        """Phát hiện các hình dạng trong ảnh."""
        shapes = {
            'rectangles': [],
            'diamonds': [],
            'circles': [],
            'arrows': []
        }
        
        # Tìm contours
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_contour_area < area < self.max_contour_area:
                # Xấp xỉ đa giác
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Phân loại hình dạng
                if len(approx) == 4:
                    # Kiểm tra hình chữ nhật hay hình thoi
                    x, y, w, h = cv2.boundingRect(approx)
                    ar = float(w)/h
                    if 0.85 <= ar <= 1.15:  # Gần vuông
                        shapes['diamonds'].append(contour)
                    else:
                        shapes['rectangles'].append(contour)
                elif len(approx) > 6:
                    # Kiểm tra hình tròn/oval
                    circularity = 4 * np.pi * area / (cv2.arcLength(contour, True) ** 2)
                    if circularity > 0.8:
                        shapes['circles'].append(contour)
                else:
                    # Kiểm tra mũi tên
                    if self._is_arrow(contour):
                        shapes['arrows'].append(contour)
        
        return shapes
    
    def extract_text(self, image: np.ndarray, shapes: Dict) -> Dict:
        """Trích xuất văn bản từ các vùng đã phát hiện."""
        text_results = {
            'rectangles': [],
            'diamonds': [],
            'circles': [],
            'outside': []
        }
        
        # Tạo mask cho các shape đã phát hiện
        shape_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for shape_type, contours in shapes.items():
            if shape_type != 'arrows':  # Bỏ qua arrows
                for contour in contours:
                    cv2.drawContours(shape_mask, [contour], -1, 255, -1)
        
        # Xử lý text trong từng shape
        for shape_type, contours in shapes.items():
            if shape_type != 'arrows':
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    roi = image[y:y+h, x:x+w]
                    
                    # Sử dụng Tesseract OCR
                    text = pytesseract.image_to_string(roi, lang='vie+eng')
                    text = text.strip()
                    
                    if text:
                        text_results[shape_type].append({
                            'text': text,
                            'position': (x, y, w, h)
                        })
        
        # Xử lý text bên ngoài shapes
        inv_mask = cv2.bitwise_not(shape_mask)
        outside_text = pytesseract.image_to_string(
            cv2.bitwise_and(image, image, mask=inv_mask),
            lang='vie+eng'
        )
        if outside_text.strip():
            text_results['outside'].append(outside_text.strip())
        
        return text_results
    
    def _calculate_skew_angle(self, lines: np.ndarray) -> float:
        """Tính góc nghiêng của ảnh."""
        angles = []
        for rho, theta in lines[:, 0]:
            angle = theta * 180 / np.pi
            if angle < 45:
                angles.append(angle)
            elif angle > 135:
                angles.append(angle - 180)
        return np.median(angles) if angles else 0
    
    def _is_arrow(self, contour: np.ndarray) -> bool:
        """Kiểm tra xem contour có phải là mũi tên không."""
        # Tính tỷ lệ chiều dài/chiều rộng
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w)/h
        
        # Tính độ phức tạp của contour
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        contour_area = cv2.contourArea(contour)
        solidity = float(contour_area)/hull_area
        
        # Điều kiện để là mũi tên
        return aspect_ratio > 2 and solidity < 0.9

    def process_flowchart(self, file_path: str) -> Dict:
        """Xử lý flowchart hoàn chỉnh."""
        # Đọc ảnh
        image = self.read_image(file_path)
        
        # Tiền xử lý
        binary = self.preprocess_image(image)
        
        # Phát hiện hình dạng
        shapes = self.detect_shapes(binary)
        
        # Trích xuất văn bản
        text_results = self.extract_text(image, shapes)
        
        return {
            'shapes': shapes,
            'text': text_results
        }
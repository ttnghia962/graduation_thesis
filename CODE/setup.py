# test_environment.py

import sys
import cv2
import numpy as np
import pytesseract
from PIL import Image
import logging
from pathlib import Path
import os

def test_environment():
    """Test if all required packages are installed and working"""
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('environment_test')
    
    # Set Tesseract path
    tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
    
    test_results = {}
    
    # 1. Test Python version
    python_version = sys.version_info
    logger.info(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    test_results['python'] = python_version.major == 3 and python_version.minor >= 8
    
    # 2. Test OpenCV
    opencv_version = cv2.__version__
    logger.info(f"OpenCV version: {opencv_version}")
    test_results['opencv'] = True
    
    # 3. Test NumPy
    numpy_version = np.__version__
    logger.info(f"NumPy version: {numpy_version}")
    test_results['numpy'] = True
    
    # 4. Test PIL/Pillow
    pil_version = Image.__version__
    logger.info(f"Pillow version: {pil_version}")
    test_results['pillow'] = True
    
    # 5. Test Tesseract
    try:
        # First check if tesseract executable exists
        if not os.path.exists(tesseract_path):
            raise FileNotFoundError(f"Tesseract not found at: {tesseract_path}")
            
        # Try to get version
        tesseract_version = pytesseract.get_tesseract_version()
        logger.info(f"Tesseract version: {tesseract_version}")
        
        # Try simple OCR test
        test_image = np.full((100, 300), 255, dtype=np.uint8)
        cv2.putText(test_image, "Test123", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        test_text = pytesseract.image_to_string(test_image).strip()
        logger.info(f"OCR Test Result: {test_text}")
        
        test_results['tesseract'] = True
        
    except Exception as e:
        logger.error(f"Tesseract error: {str(e)}")
        logger.error("Detailed error information:")
        logger.error(f"1. Tesseract path exists: {os.path.exists(tesseract_path)}")
        logger.error(f"2. Tesseract path: {tesseract_path}")
        logger.error(f"3. PATH environment: {os.environ.get('PATH', '')}")
        test_results['tesseract'] = False
    
    # Create a simple test image
    try:
        # Create a white image with black text
        img = np.full((100, 300), 255, dtype=np.uint8)
        cv2.putText(img, "Test Image", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Save test image
        test_dir = Path("test_output")
        test_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(test_dir / "test_image.png"), img)
        
        logger.info("Successfully created and saved test image")
        test_results['image_processing'] = True
    except Exception as e:
        logger.error(f"Image processing error: {str(e)}")
        test_results['image_processing'] = False
    
    # Print summary
    logger.info("\nTest Summary:")
    all_passed = True
    for test, passed in test_results.items():
        status = "PASSED" if passed else "FAILED"
        logger.info(f"{test:15} : {status}")
        all_passed = all_passed and passed
    
    if not test_results['tesseract']:
        logger.info("\nTesseract Installation Guide:")
        logger.info("1. Download Tesseract installer from: https://github.com/UB-Mannheim/tesseract/wiki")
        logger.info("2. Install to the default location: C:\\Program Files\\Tesseract-OCR")
        logger.info("3. Add to PATH: C:\\Program Files\\Tesseract-OCR")
        logger.info("4. Restart your terminal/IDE")
        logger.info("5. Run this test again")
    
    return all_passed

if __name__ == "__main__":
    passed = test_environment()
    if not passed:
        print("\nSome tests failed. Please check the output above.")
        sys.exit(1)
    print("\nAll tests passed! Environment is ready.")
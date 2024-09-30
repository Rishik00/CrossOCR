## BaseOCR.py
import cv2
from PIL import Image

class OCRBaseUtil:
    def __init__(self, image_path: str = '', scale_factor: float = 3.5):
        self.image_path = image_path
        self.scale_factor = scale_factor
        self.image = cv2.imread(self.image_path)
    
        if self.image is None:
            raise FileNotFoundError(f"Image not found at path: {self.image_path}")
    
    def image_preprocess(self):

        original_height, original_width = self.image.shape[:2]
        new_size = (int(original_width * self.scale_factor), int(original_height * self.scale_factor))
        larger_image = cv2.resize(self.image, new_size, interpolation=cv2.INTER_LINEAR)
        
        # Convert the larger image to grayscale
        gray_image = cv2.cvtColor(larger_image, cv2.COLOR_BGR2GRAY)
        # _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
        denoised_image = cv2.fastNlMeansDenoising(gray_image, None, 30, 7, 21)
        
        return denoised_image
    
    def ocr(self):
        raise NotImplementedError('Subclasses should implement this!')
    
    def display_ocr_result(self):
        raise NotImplementedError('Subclasses should implement this!')

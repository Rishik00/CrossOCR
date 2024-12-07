import cv2

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

    def enhance_image(self):
        output_path = 'enh_' + self.image_path.split("\\")[-1]

        # Convert image to grayscale directly
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to smooth the image
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
        
        # Threshold the grayscale image using Otsu's method
        _, binary_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Perform morphological transformation (opening) to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
        enhanced_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

        # Save the enhanced image
        cv2.imwrite(output_path, enhanced_image)
        print('Output path is: ', output_path)

        return output_path
    
    def ocr(self):
        raise NotImplementedError('Subclasses should implement this!')
    
    def display_ocr_result(self):
        raise NotImplementedError('Subclasses should implement this!')

# if __name__ == "__main__":
#     # Ensure the path is correctly formatted
#     image_path = r'C:\Users\sridh\OneDrive\Desktop\webdev\CrossOCR\src\ocr\sample_image-3.jpg'
    
#     ocr = OCRBaseUtil(image_path=image_path)
#     print('opened it yay')
    
#     output_file_path = ocr.enhance_image()
    
#     # Read and display the image
#     image = cv2.imread(output_file_path)
    
#     # Check if the image was successfully loaded
#     if image is not None:
#         cv2.imshow('Enhanced Image', image)
#         cv2.waitKey(0)  # Wait for a key press
#         cv2.destroyAllWindows()  # Close the image window
#     else:
#         print('Error: Image not found or unable to load.')

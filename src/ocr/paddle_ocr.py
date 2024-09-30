## OCRUtils.py
from paddleocr import PaddleOCR
from matplotlib import pyplot as plt
import numpy as np

# Local imports
from ocr.base_ocr import OCRBaseUtil

class PaddleOCRutil(OCRBaseUtil):
    def __init__(self, image_path: str = '', lang: str = 'hi', confidence_score: float = 0.7):
        super().__init__(image_path)
        self.lang = lang
        self.paddle_ocr = PaddleOCR(
            use_angle_cls=True,
            lang=self.lang
        )
        self.threshold = confidence_score

    def ocr(self):

        denoised_image = self.image_preprocess()
        res = self.paddle_ocr.ocr(self.image_path)  # PaddleOCR expects the file path, not the image array
        return res

    def parse_ocr_results(self, high_confidence: bool = True):

        final_text_results = {}
        ocr_res = self.ocr()

        for line in ocr_res:
            for word_info in line:
                
                text = word_info[1][0]
                confidence_score = word_info[1][1]

                if confidence_score >= self.threshold:
                    final_text_results.update({text: confidence_score})

        print('Parsed results from OCR!')
        return final_text_results, ocr_res
    
    def display_ocr_result(self):
        print(f'Displaying OCR results: {self.image_path}')
        ocr_results, _ = self.parse_ocr_results()

        for key in ocr_results.keys():
            print(key, ocr_results[key])
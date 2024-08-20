import os
import cv2
import easyocr

class TextExtractionModel:
    def __init__(self):
        # Initialize the OCR reader
        self.reader = easyocr.Reader(['en'])  

    def extract_text(self, image_path):
        try:
            # Read the image
            img = cv2.imread(image_path)
            if img is None or img.size == 0:
                print(f"Warning: Empty or invalid image at {image_path}")
                return ""

            # Check if the image is too small
            if img.shape[0] < 10 or img.shape[1] < 10:
                print(f"Warning: Image too small at {image_path}")
                return ""

            # Perform OCR on the image
            results = self.reader.readtext(img)
            
            # Extract the text from the results
            extracted_text = ' '.join([result[1] for result in results])
            
            return extracted_text
        except Exception as e:
            print(f"Error processing image at {image_path}: {str(e)}")
            return ""

    def process_objects(self, object_dir):
        text_data = {}
        for filename in os.listdir(object_dir):
            if filename.endswith('.png') or filename.endswith('.jpg'):
                object_path = os.path.join(object_dir, filename)
                object_id = os.path.splitext(filename)[0]
                extracted_text = self.extract_text(object_path)
                text_data[object_id] = extracted_text
        return text_data
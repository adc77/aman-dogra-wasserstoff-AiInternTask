import easyocr
import os

class TextExtractionModel:
    def __init__(self):
        # Initialize the OCR reader
        self.reader = easyocr.Reader(['en'])  # For English text

    def extract_text(self, image_path):
        # Perform OCR on the image
        results = self.reader.readtext(image_path)
        
        # Extract the text from the results
        extracted_text = ' '.join([result[1] for result in results])
        
        return extracted_text

    def process_objects(self, object_dir):
        text_data = {}
        for filename in os.listdir(object_dir):
            if filename.endswith('.png') or filename.endswith('.jpg'):
                object_path = os.path.join(object_dir, filename)
                object_id = os.path.splitext(filename)[0]
                extracted_text = self.extract_text(object_path)
                text_data[object_id] = extracted_text
        return text_data
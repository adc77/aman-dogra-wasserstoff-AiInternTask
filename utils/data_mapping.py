import json
import os
import uuid
import numpy as np
import cv2  # Added this import

class DataMapper:
    def __init__(self, output_dir='data/mapped_data'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.master_data = {}

    def map_data(self, input_image_path, segmentation_data, analysis_data):
        master_id = str(uuid.uuid4())
        self.master_data[master_id] = {'objects': {}}
        
        for obj in segmentation_data:
            object_id = obj['id']
            bbox = obj.get('bbox', [])
            segmentation = obj.get('segmentation', [])
            
            # Convert to list if they're NumPy arrays
            if isinstance(bbox, np.ndarray):
                bbox = bbox.tolist()
            if isinstance(segmentation, np.ndarray):
                segmentation = segmentation.tolist()
            
            self.master_data[master_id]['objects'][object_id] = {
                'bbox': bbox,
                'segmentation': segmentation,
                'identification': analysis_data.get(object_id, {}).get('identification', ''),
                'extracted_text': analysis_data.get(object_id, {}).get('extracted_text', ''),
                'summary': analysis_data.get(object_id, {}).get('summary', '')
            }
        
        return master_id

    def convert_to_json_serializable(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def save_mapping(self, filename='mapped_data.json'):
        mapping_path = os.path.join(self.output_dir, filename)
        
        serializable_data = json.loads(
            json.dumps(self.master_data, default=self.convert_to_json_serializable)
        )
        
        with open(mapping_path, 'w') as f:
            json.dump(serializable_data, f, indent=2)

    def load_mapping(self, filename='mapped_data.json'):
        filepath = os.path.join(self.output_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                self.master_data = json.load(f)
        return self.master_data

class OutputGenerator:
    def __init__(self, output_dir='data/output'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_output(self, master_id, mapped_data, image):
        # Create a copy of the image for drawing
        output_image = image.copy()

        # Prepare data for CSV
        csv_data = []

        # Check if 'objects' key exists
        if 'objects' not in mapped_data[master_id]:
            print(f"Warning: 'objects' key not found in mapped_data for master_id {master_id}")
            return None, None

        for obj_id, obj_info in mapped_data[master_id]['objects'].items():
            # Draw bounding box
            bbox = obj_info.get('bbox', [])
            if bbox:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(output_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # Prepare CSV data
            csv_data.append({
                'Object ID': obj_id,
                'Identification': obj_info.get('identification', ''),
                'Extracted Text': obj_info.get('extracted_text', ''),
                'Summary': obj_info.get('summary', '')
            })

        # Save output image
        output_image_path = os.path.join(self.output_dir, f'{master_id}_output.jpg')
        cv2.imwrite(output_image_path, output_image)

        # Save CSV
        output_csv_path = os.path.join(self.output_dir, f'{master_id}_output.csv')
        with open(output_csv_path, 'w', newline='') as csvfile:
            fieldnames = ['Object ID', 'Identification', 'Extracted Text', 'Summary']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in csv_data:
                writer.writerow(row)

        return output_image_path, output_csv_path
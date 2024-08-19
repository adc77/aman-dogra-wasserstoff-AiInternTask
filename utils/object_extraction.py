import cv2
import numpy as np
import os
import uuid
import json
from PIL import Image

class ObjectExtractor:
    def __init__(self, output_dir='data/segmented_objects'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.metadata = []

    def extract_objects(self, image, masks, max_objects=100):
        object_ids = []
        
        print(f"Image shape: {image.shape}")
        
        for i, mask in enumerate(masks[:max_objects]):  # Limit to max_objects
            print(f"Processing mask {i}, shape: {mask.shape}")
            
            if mask.shape[:2] != image.shape[:2]:
                print(f"Mask shape mismatch for object {i}. Resizing mask.")
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            try:
                object_mask = mask.astype(bool)
                object_image = image.copy()
                object_image[~object_mask] = 0

                # Find bounding box
                y, x = np.where(object_mask)
                if len(y) == 0 or len(x) == 0:
                    print(f"Skipping empty mask for object {i}")
                    continue
                top, bottom, left, right = y.min(), y.max(), x.min(), x.max()

                # Crop the object
                cropped_object = object_image[top:bottom+1, left:right+1]

                # Save the object
                object_id = f"object_{i}"
                object_path = os.path.join(self.output_dir, f"{object_id}.jpg")
                cv2.imwrite(object_path, cropped_object)

                # Store metadata
                self.metadata.append({
                    'id': object_id,
                    'bbox': [left, top, right, bottom],
                    'path': object_path
                })

                object_ids.append(object_id)
                print(f"Extracted object {i}")
            except Exception as e:
                print(f"Error processing object {i}: {str(e)}")

        return object_ids

    def save_metadata(self):
        metadata_path = os.path.join(self.output_dir, 'metadata.json')
        
        def convert_to_json_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        serializable_metadata = json.loads(
            json.dumps(self.metadata, default=convert_to_json_serializable)
        )
        
        with open(metadata_path, 'w') as f:
            json.dump(serializable_metadata, f)
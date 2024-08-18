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

    def extract_objects(self, image, masks):
        object_ids = []
        for i, mask in enumerate(masks):
            # Check if mask is empty
            if not np.any(mask):
                print(f"Skipping empty mask for object {i}")
                continue

            # Find contours
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                print(f"No contours found for object {i}")
                continue

            # Find bounding box
            x, y, w, h = cv2.boundingRect(contours[0])
            
            # Extract object
            object_image = image[y:y+h, x:x+w].copy()
            object_mask = mask[y:y+h, x:x+w]
            object_image[~object_mask] = 0

            # Save object
            object_id = f"object_{i}"
            object_path = os.path.join(self.output_dir, f"{object_id}.png")
            cv2.imwrite(object_path, cv2.cvtColor(object_image, cv2.COLOR_RGB2BGR))

            # Save metadata
            self.metadata.append({
                'id': object_id,
                'bbox': [x, y, w, h],
                'path': object_path
            })

            object_ids.append(object_id)

        return object_ids

    def save_metadata(self):
        metadata_path = os.path.join(self.output_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
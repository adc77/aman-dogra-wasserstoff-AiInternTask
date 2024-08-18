
"""
from segmentation_sam import SegmentationModel, EnsembleSegmentationModel
import cv2
import os
import numpy as np

def main():
    # Initialize the ensemble segmentation model
    seg_model = EnsembleSegmentationModel()

    # Path to input image
    input_image_path = "data/input_images/test_image5.jpg"

    # Perform segmentation
    image, masks = seg_model.segment_image(input_image_path)

    # Visualize segmentation
    segmented_image = seg_model.models[0].visualize_segmentation(image, masks)

    # Save the segmented image
    output_path = "data/output/segmented_image.jpg"
    cv2.imwrite(output_path, cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))

    print(f"Segmented image saved to {output_path}")

    # Extract and save individual objects
    for i, mask in enumerate(masks):
        object_image = image.copy()
        object_image[~mask] = 0

        object_path = f"data/segmented_objects/object_{i}.jpg"
        cv2.imwrite(object_path, cv2.cvtColor(object_image, cv2.COLOR_RGB2BGR))
        print(f"Object {i} saved to {object_path}")

if __name__ == "__main__":
    main()
"""


from models.segmentation_sam import SegmentationModel, EnsembleSegmentationModel
from utils.object_extraction import ObjectExtractor
from models.identification_model import IdentificationModel
import cv2
import os
import json

def main():
    # Initialize models
    seg_model = SegmentationModel()
    extractor = ObjectExtractor()
    id_model = IdentificationModel()

    # Path to input image
    input_image_path = "data/input_images/test_image5.jpg"

    # statement to see the full path being used
    print("Attempting to read image from:", os.path.abspath(input_image_path))

    # Perform segmentation
    image, masks = seg_model.segment_image(input_image_path)

    # Extract objects
    object_ids = extractor.extract_objects(image, masks)

    # Save metadata
    extractor.save_metadata()

    # Identify and describe objects
    descriptions = id_model.process_objects(extractor.output_dir)

    # Save descriptions
    with open('data/object_descriptions.json', 'w') as f:
        json.dump(descriptions, f, indent=2)

    print("Segmentation, extraction, and identification complete.")
    print(f"Extracted {len(object_ids)} objects.")
    print("Object descriptions saved to data/object_descriptions.json")

if __name__ == "__main__":
    main()
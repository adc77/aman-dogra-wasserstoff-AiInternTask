from models.segmentation_yolo import SegmentationModel
# from models.segmentation_sam import SegmentationModel, EnsembleSegmentationModel
from utils.object_extraction import ObjectExtractor
from models.identification_model import IdentificationModel
from models.text_extraction_model import TextExtractionModel
from models.summarization_model import SummarizationModel
from utils.data_mapping import DataMapper, OutputGenerator
from utils.preprocessing import preprocess_image
import cv2
import os
import json

def main(input_image_path):
    # Initialize models
    seg_model = SegmentationModel()
    extractor = ObjectExtractor()
    id_model = IdentificationModel()
    text_model = TextExtractionModel()
    sum_model = SummarizationModel()
    data_mapper = DataMapper()
    output_gen = OutputGenerator()

    # Path to input image
    #input_image_path = "data/input_images/test_image2.jpg"

    # Perform segmentation
    image, masks = seg_model.segment_image(input_image_path)

    # adding to check
    # Visualize segmentation
    segmented_image = seg_model.visualize_segmentation(image, masks)

    # Save the segmented image
    output_path = "data/output/segmented_image.jpg"
    #cv2.imwrite(output_path, cv2.cvtColor(segment_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(output_path, segmented_image)

    print(f"Segmented image saved to {output_path}")
    # check

    # Extract objects
    object_ids = extractor.extract_objects(image, masks, max_objects=15)

    # Save metadata
    extractor.save_metadata()

    # Identify objects
    identifications = id_model.process_objects(extractor.output_dir)

    # Extract text from objects
    # text_data = text_model.process_objects(extractor.output_dir)
    try:
        text_data = text_model.process_objects(extractor.output_dir)
    except Exception as e:
        print(f"Error during text extraction: {str(e)}")
        text_data = {}  
        # Use an empty dict if text extraction fails


    # Summarize objects
    summaries = sum_model.process_objects(identifications, text_data)

    # Combine all data
    final_data = {}
    for object_id in object_ids:
        final_data[object_id] = {
            "identification": identifications.get(object_id, ""),
            "extracted_text": text_data.get(object_id, ""),
            "summary": summaries.get(object_id, "")
        }

    # Map data
    master_id = data_mapper.map_data(input_image_path, extractor.metadata, final_data)
    data_mapper.save_mapping()

    # output
    output_image_path, output_csv_path = output_gen.generate_output(master_id, data_mapper.master_data, image)

    print("Segmentation, extraction, identification, text extraction, summarization, and output generation complete.")
    print(f"Analyzed {len(object_ids)} objects.")
    print(f"Final output image saved to {output_image_path}")
    print(f"Final output table saved to {output_csv_path}")

    return output_image_path, output_csv_path

if __name__ == "__main__":
    input_image_path = "data/input_images/test_image2.jpg"  
    main(input_image_path)  
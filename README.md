# Image Processing Pipeline

## Overview
This project implements an image processing pipeline that performs segmentation, object extraction, identification, text extraction, summarization, and output generation. It utilizes various models to analyze input images and generate structured output.

## Features
- **Segmentation**: Tried using Mask R-CNN (segmentation_model.py) & Segment Anything Model (SAM)(segmentation_sam.py) initially, then finally used YOLOv8 (segmentation_yolo.py) for segmentation.
- **Object Extraction**: Extracts identified objects from the segmented image and stores them with unique object IDs (used pillow and open cv for extraction)
- **Identification**: Identifies the extracted objects using open ai's CLIP model (tells what the object is)
- **Text Extraction**: Extracts text from the identified objects using easyOCR.
- **Summarization**: Summarizes the information related to the identified objects using sumy python library.
- **Data Mapping**: Maps the extracted data into a structured JSON format.
- **Output Generation**: Generates an output image and an output CSV file containing containing all object id’s, identification, extracted text & summary.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/adc77/aman-dogra-wasserstoff-AiInternTask.git
   cd aman-dogra-wasserstoff-AiInternTask
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Place your input images in the `data/input_images` directory.
2. Update the `input_image_path` in `main.py` to point to your image.
3. Run the main script:
   ```bash
   python -m models.main
   ```

## Output
- The segmented image will be saved in the `data/output` directory.
- A CSV file containing the analysis results will also be generated in the output directory.


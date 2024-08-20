# Image Processing Pipeline

## Overview
This project implements an image processing pipeline that performs segmentation, object extraction, identification, text extraction, summarization, and output generation. It utilizes various models to analyze input images and generate structured output.

## Features
- **Segmentation**: Uses a segmentation model to identify and segment objects in images.
- **Object Extraction**: Extracts identified objects from the segmented image.
- **Identification**: Identifies the extracted objects using a dedicated identification model.
- **Text Extraction**: Extracts text from the identified objects.
- **Summarization**: Summarizes the information related to the identified objects.
- **Output Generation**: Generates an output image and a CSV file containing the results.

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
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


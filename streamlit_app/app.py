import streamlit as st
import cv2
import numpy as np
from segmentation_model import SegmentationModel

def main():
    st.title("Instance Segmentation")
    st.write("Upload an image.")

    # File uploader for image input
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the image file
        image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        # Display the uploaded image
        st.image(image, channels="BGR", caption="Uploaded Image", use_column_width=True)

        # Create an instance of the segmentation model
        segmentation_model = SegmentationModel(output_dir='output')

        # Perform segmentation
        output_image_path = segmentation_model.segment_image(uploaded_file)

        # Read and display the segmented image
        segmented_image = cv2.imread(output_image_path)
        st.image(segmented_image, channels="BGR", caption="Segmented Image", use_column_width=True)

if __name__ == "__main__":
    main()
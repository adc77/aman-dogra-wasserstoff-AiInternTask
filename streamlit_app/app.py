import sys
import streamlit as st
import os

# Add the project root directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(project_root)

from models.main import main



def app():
    st.title("Object Analysis App")

    # upload image 
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Save the uploaded file to the "data/input_images" directory
        input_image_path = os.path.join("data", "input_images", uploaded_file.name)
        with open(input_image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.write(f"Analyzing image: {uploaded_file.name}")

        # Run the main function from main.py
        output_image_path, output_csv_path = main(input_image_path)

        # Display the output image
        st.image(output_image_path, caption="Analyzed Image")

        # Download the output CSV file
        with open(output_csv_path, "r") as f:
            csv_data = f.read()
        st.download_button(
            label="Download output CSV",
            data=csv_data,
            file_name="output_csv_path.csv",
            mime="text/csv",
        )

        st.write("Analysis complete!")

if __name__ == "__main__":
    app()
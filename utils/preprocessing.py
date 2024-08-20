import cv2
import numpy as np

def preprocess_image(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Convert to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize the image to a fixed size (e.g. 1024x768)
    image = cv2.resize(image, (1024, 768))

    # normalization
    image = image / 255.0

    return image
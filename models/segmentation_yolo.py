from ultralytics import YOLO
import cv2
import numpy as np

class SegmentationModel:
    def __init__(self, model_path=r'C:\Users\amand\OneDrive\Desktop\aman-dogra-wasserstoff-AiInternTask\yolov8x-seg.pt', conf_threshold=0.5):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def segment_image(self, image_path):
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image from {image_path}")

        # Perform segmentation
        results = self.model(image_path, conf=self.conf_threshold, stream=True)
        masks = []

        for result in results:
            if result.masks is not None:
                for i, mask_data in enumerate(result.masks.xy):
                    try:
                        mask = np.zeros(image.shape[:2], dtype=np.uint8)
                        pts = np.array(mask_data, dtype=np.int32)
                        
                        if pts.size == 0:
                            print(f"Skipping empty mask for object {i}")
                            continue
                        
                        if pts.ndim != 2 or pts.shape[1] != 2:
                            print(f"Invalid points shape for object {i}: {pts.shape}")
                            continue
                        
                        cv2.fillPoly(mask, [pts], 1)
                        masks.append(mask)
                    except Exception as e:
                        print(f"Error processing mask for object {i}: {str(e)}")
                        print(f"Mask data: {mask_data}")
            else:
                print("No masks detected in the image.")

        if not masks:
            print("No valid masks were created.")
            return image, None

        return image, masks

    def visualize_segmentation(self, image, masks):
        segmented_image = image.copy()
        for mask in masks:
            color = np.random.randint(0, 255, (3,), dtype=np.uint8)
            segmented_image[mask > 0] = segmented_image[mask > 0] * 0.5 + color * 0.5
        return segmented_image
import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor

class SegmentationModel:
    def __init__(self):
        # Download the SAM checkpoint from https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
        # and place it in the models/ directory
        sam_checkpoint = "models/sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        
        self.predictor = SamPredictor(sam)


    def post_process_mask(self, mask, min_size=1000):
        # Apply morphological operations
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Remove small objects
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_size:
                mask[labels == i] = 0

        return mask.astype(bool)

    def segment_image(self, image_path, points_per_side=32, pred_iou_thresh=0.88, stability_score_thresh=0.95):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        self.predictor.set_image(image)
        
        # Generate automatic masks with adjusted parameters
        masks, _, _ = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            multimask_output=True
        )
        
        processed_masks = [self.post_process_mask(mask) for mask in masks]
        return image, processed_masks

    def visualize_segmentation(self, image, masks):
        segmented_image = image.copy()
        for i, mask in enumerate(masks):
            color = np.random.randint(0, 255, (3,), dtype=np.uint8)
            segmented_image[mask] = segmented_image[mask] * 0.5 + color * 0.5
        return segmented_image



class EnsembleSegmentationModel:
    def __init__(self):
        self.models = [
            SegmentationModel(),
            SegmentationModel(),
            SegmentationModel()
        ]

    def segment_image(self, image_path):
        all_masks = []
        for model in self.models:
            _, masks = model.segment_image(image_path)
            all_masks.extend(masks)

        # Combine masks using majority voting
        combined_mask = np.zeros_like(all_masks[0], dtype=int)
        for mask in all_masks:
            combined_mask += mask.astype(int)
        
        final_mask = (combined_mask > len(self.models) / 2).astype(bool)
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image, [final_mask]
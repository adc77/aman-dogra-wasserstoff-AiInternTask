import torch
import torchvision
from torchvision.transforms import functional as F
from torchvision.ops import nms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

class SegmentationModel:
    def __init__(self):
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.coco_names = self.load_coco_names() # Load COCO class names

    def load_coco_names(self):
        # COCO class names
        return [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
            'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

    def preprocess_image(self, image):
        # Normalize the image
        image = F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return image

    def post_process(self, boxes, labels, scores, masks):
        # Remove duplicate detections
        unique_labels = []
        unique_boxes = []
        unique_scores = []
        unique_masks = []
        for box, label, score, mask in zip(boxes, labels, scores, masks):
            if label not in unique_labels:
                unique_labels.append(label)
                unique_boxes.append(box)
                unique_scores.append(score)
                unique_masks.append(mask)
        
        if not unique_boxes:  # If no objects detected
            return torch.empty(0, 4), torch.empty(0, dtype=torch.long), torch.empty(0), torch.empty(0, 1, 1, 1)
        
        return torch.stack(unique_boxes), torch.tensor(unique_labels), torch.tensor(unique_scores), torch.stack(unique_masks)

    def segment_image(self, image_path, confidence_threshold=0.3, nms_threshold=0.3):
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.preprocess_image(F.to_tensor(image).unsqueeze(0).to(self.device))

        # inference
        with torch.no_grad():
            prediction = self.model(image_tensor)[0]

        # Filter predictions based on confidence threshold
        mask = prediction['scores'] > confidence_threshold
        boxes = prediction['boxes'][mask]
        labels = prediction['labels'][mask]
        scores = prediction['scores'][mask]
        masks = prediction['masks'][mask]

        # If no objects detected after filtering
        if len(boxes) == 0:  
            return image_tensor.squeeze().permute(1, 2, 0).cpu().numpy(), torch.empty(0, 1, 1, 1), torch.empty(0, 4), torch.empty(0, dtype=torch.long), torch.empty(0)

        # Apply non-maximum suppression
        keep = nms(boxes, scores, nms_threshold)
        boxes = boxes[keep]
        labels = labels[keep]
        scores = scores[keep]
        masks = masks[keep]

        # Post-processing
        boxes, labels, scores, masks = self.post_process(boxes, labels, scores, masks)

        # Convert image back to numpy for visualization
        image_np = image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()

        return image_np, masks, boxes, labels, scores

    def visualize_segmentation(self, image_np, masks, boxes, labels, scores):
        plt.figure(figsize=(12, 8))
        plt.imshow(image_np)

        if len(masks) > 0:
            masks = masks.squeeze().cpu().numpy()
            h, w = image_np.shape[:2]
            cmap = plt.colormaps['tab20']  # Updated to use new colormap syntax
            for i, (mask, box, label, score) in enumerate(zip(masks, boxes, labels, scores)):
                color = cmap(i % 20)[:3]
                # Reshape the mask to match the image dimensions
                if mask.ndim == 1:
                    mask = mask.reshape(1, -1)  # Reshape to 2D
                if mask.shape[-1] == h * w:
                    mask = mask.reshape(h, w)
                elif mask.shape[-1] == w:
                    mask = mask.reshape(-1, w)
                else:
                    print(f"Warning: Mask shape {mask.shape} doesn't match image shape {image_np.shape}")
                    continue  
                    # Skip this mask

                plt.imshow(mask, alpha=0.3, cmap=plt.colormaps.get_cmap('tab20'))
                x, y, w, h = box.cpu().numpy()
                plt.gca().add_patch(plt.Rectangle((x, y), w - x, h - y, fill=False, edgecolor=color, linewidth=2))
                class_name = self.coco_names[label] if label < len(self.coco_names) else f"Class {label}"
                plt.text(x, y, f"{class_name}: {score:.2f}", bbox=dict(facecolor='yellow', alpha=0.5))

        plt.axis('off')
        plt.tight_layout()
        plt.show()

def main(image_path):
    model = SegmentationModel()
    image_np, masks, boxes, labels, scores = model.segment_image(image_path)
    model.visualize_segmentation(image_np, masks, boxes, labels, scores)
    

if __name__ == "__main__":
    image_path = r'data\input_images\test_image1.jpg'
    main(image_path)

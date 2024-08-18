import torch
from PIL import Image
import clip
import os

class IdentificationModel:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.categories = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
            "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
            "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
            "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
        ]

    def identify_object(self, image_path):
        image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
        text = clip.tokenize(self.categories).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(text)
            
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            values, indices = similarity[0].topk(5)

        results = [
            {'category': self.categories[idx], 'confidence': val.item()}
            for val, idx in zip(values, indices)
        ]

        return results

    def describe_object(self, image_path):
        top_categories = self.identify_object(image_path)
        description = f"This image likely contains a {top_categories[0]['category']} "
        description += f"(confidence: {top_categories[0]['confidence']:.2f}). "
        description += "Other possibilities include " + ", ".join([f"{cat['category']} ({cat['confidence']:.2f})" for cat in top_categories[1:4]])
        return description

    def process_objects(self, object_dir):
        descriptions = {}
        for filename in os.listdir(object_dir):
            if filename.endswith('.png') or filename.endswith('.jpg'):
                object_path = os.path.join(object_dir, filename)
                object_id = os.path.splitext(filename)[0]
                descriptions[object_id] = self.describe_object(object_path)
        return descriptions
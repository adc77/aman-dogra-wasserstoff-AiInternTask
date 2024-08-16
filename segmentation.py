import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F

# Load a pre-trained Mask R-CNN model
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

image = Image.open('path_to_your_image.jpg')
image_tensor = F.to_tensor(image).unsqueeze(0)  # Add batch dimension

with torch.no_grad():
    predictions = model(image_tensor)

# Get masks, labels, and bounding boxes from predictions
masks = predictions[0]['masks']
labels = predictions[0]['labels']
boxes = predictions[0]['boxes']

plt.imshow(image)
for i in range(len(masks)):
    mask = masks[i, 0].numpy()
    plt.imshow(mask, alpha=0.5)
plt.show()

for i, mask in enumerate(masks):
    mask_image = Image.fromarray((mask[0].numpy() * 255).astype('uint8'))
    mask_image.save(f'segmented_object_{i}.png')

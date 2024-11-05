from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image

# Load the pre-trained model and processor
model_name = "ufdatastudio/vit-orientation"
model = ViTForImageClassification.from_pretrained(model_name)
processor = ViTImageProcessor.from_pretrained(model_name)

# Load and preprocess the image
image_path = "D:/images/10-19-2020/100_6189.JPG"
image = Image.open(image_path)

# Apply the image processor
inputs = processor(images=image, 
                    size={"height": 224, "width": 224},
                    rescale_factor=0.00392156862745098,
                    do_normalize=True, 
                    do_rescale=True, 
                    do_resize=True,
                    resample=2,
                    return_tensors="pt")

# Perform inference
outputs = model(**inputs)
logits = outputs.logits

# Print the logits or any specific information
print(logits)
import torch
import math

# Assume logits are x and y components
x, y = logits[0]

# Calculate the angle in radians
angle_radians = torch.atan2(y, x).item()  # .item() to get the Python number from tensor

# Convert radians to degrees
angle_degrees = angle_radians * (180 / math.pi)

print(f"Predicted Angle: {angle_degrees:.2f} degrees")
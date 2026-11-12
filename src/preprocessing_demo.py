
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

# Locate dataset
project_root = Path(__file__).resolve().parent.parent
image_path = project_root / "data" / "bottle" / "train" / "good" / "000.png"

# Load image
image = Image.open(image_path).convert("RGB")

print("Original image size:", image.size)

# Convert image into tensor
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

image_tensor = transform(image)

print("\n========== TENSOR INFORMATION ==========")
print("Tensor shape:", image_tensor.shape)
print("Data type:", image_tensor.dtype)
print("Minimum pixel value:", image_tensor.min().item())
print("Maximum pixel value:", image_tensor.max().item())

# Display tensor as image
display_image = image_tensor.permute(1, 2, 0).numpy()

import matplotlib.pyplot as plt

plt.imshow(display_image)
plt.title("Resized Bottle Image")
plt.axis("off")
plt.show()


from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "bottle"

# Dataset directories
train_path = DATASET_PATH / "train"
test_path = DATASET_PATH / "test"
mask_path = DATASET_PATH / "ground_truth"


# Function to count images
def count_images(folder):
    return len(list(folder.glob("*.png")))


# 1. Training image information
print("\n========== TRAINING DATA ==========")

for category in sorted(train_path.iterdir()):
    if category.is_dir():
        print(f"{category.name}: {count_images(category)} images")


# 2. Testing image information
print("\n========== TESTING DATA ==========")

for category in sorted(test_path.iterdir()):
    if category.is_dir():
        print(f"{category.name}: {count_images(category)} images")


# 3. Ground truth mask information
print("\n========== GROUND TRUTH MASKS ==========")

for category in sorted(mask_path.iterdir()):
    if category.is_dir():
        print(f"{category.name}: {count_images(category)} masks")


# 4. Visualize normal and defective samples
normal_images = sorted((test_path / "good").glob("*.png"))

defect_categories = [
    folder for folder in sorted(test_path.iterdir())
    if folder.is_dir() and folder.name != "good"
]

if not normal_images or not defect_categories:
    raise RuntimeError("Could not find normal or defective test images.")

normal_image = normal_images[0]

defect_category = defect_categories[0].name
defect_image = sorted((test_path / defect_category).glob("*.png"))[0]

# Find matching ground-truth mask
mask_file = mask_path / defect_category / f"{defect_image.stem}_mask.png"

# Load images
normal = Image.open(normal_image).convert("RGB")
defective = Image.open(defect_image).convert("RGB")
mask = Image.open(mask_file).convert("L")

# Display images
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(normal)
axes[0].set_title("Normal Bottle")

axes[1].imshow(defective)
axes[1].set_title(f"Defective Bottle: {defect_category}")

axes[2].imshow(mask, cmap="gray")
axes[2].set_title("Ground Truth Mask")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()



# 5. Inspect image dimensions

print("\n========== IMAGE DIMENSIONS ==========")

for image_path in [
    normal_image,
    defect_image,
    mask_file
]:
    with Image.open(image_path) as img:
        print(
            f"{image_path.name}: "
            f"Size={img.size}, "
            f"Mode={img.mode}"
        )


# 6. Inspect ground-truth mask values

mask_array = np.array(mask)
unique_values = np.unique(mask_array)

print("Unique pixel values:", unique_values)


# Convert images into NumPy arrays
defect_array = np.array(defective)
mask_array = np.array(mask)

# Create a copy of the defective image
overlay = defect_array.copy()

# Highlight anomalous pixels in red
overlay[mask_array == 255] = [255, 0, 0]

# Blend original image and highlighted image
alpha = 0.4

result = (
    (1 - alpha) * defect_array +
    alpha * overlay
).astype(np.uint8)

# Display comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(defect_array)
axes[0].set_title("Original Defective Image")

axes[1].imshow(mask_array, cmap="gray")
axes[1].set_title("Ground Truth Mask")

axes[2].imshow(result)
axes[2].set_title("Defect Overlay")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()

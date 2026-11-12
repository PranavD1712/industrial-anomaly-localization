
from pathlib import Path
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import torch

from PIL import Image
from torch.utils.data import Dataset, DataLoader

from anomalib.models import Patchcore
from anomalib.engine import Engine
from anomalib.data import ImageItem, ImageBatch




class SingleImageDataset(Dataset):

    def __init__(self, image_path):
        self.image_path = Path(image_path)

    def __len__(self):
        return 1

    def __getitem__(self, index):

        # Load image
        image = Image.open(self.image_path).convert("RGB")

        # Convert PIL image into NumPy array
        image = np.array(image).copy()

        # Convert NumPy array into PyTorch tensor
        image = torch.from_numpy(image)

        # Change shape from HWC to CHW
        image = image.permute(2, 0, 1).float() / 255.0

        # Return one Anomalib ImageItem
        return ImageItem(
            image=image,
            image_path=str(self.image_path)
        )




def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        required=True,
        help="Path of the image to predict"
    )

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent

    image_path = Path(args.image)

    if not image_path.is_absolute():
        image_path = project_root / image_path

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Load trained PatchCore checkpoint
    checkpoint_path = (
        project_root / "results" / "Patchcore" / "MVTecAD"
        / "bottle" / "latest" / "weights" / "lightning" / "model.ckpt"
    )

    model = Patchcore.load_from_checkpoint(
        checkpoint_path,
        map_location="cpu",
        weights_only=False
    )

    model.eval()

    print("PatchCore checkpoint loaded!")

    # Load threshold configuration
    config_path = project_root / "reports" / "threshold_analysis.json"

    with open(config_path, "r") as file:
        config = json.load(file)

    threshold = float(config["threshold"])

    print(f"Classification threshold: {threshold:.4f}")

    # Create single-image dataset
    dataset = SingleImageDataset(
    image_path=image_path
    )

    dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    collate_fn=ImageBatch.collate
    )
    
    # Run inference
    engine = Engine()

    predictions = engine.predict(
        model=model,
        dataloaders=dataloader
    )

    if not predictions:
        raise RuntimeError("No prediction was returned.")

    prediction = predictions[0]

    # Extract anomaly score
    anomaly_score = float(prediction.pred_score[0].item())

    # Apply saved image-level threshold
    predicted_class = (
        "Defective" if anomaly_score > threshold else "Normal"
    )

    print("\n========== Prediction Result ==========")
    print("Image:", image_path.name)
    print(f"Anomaly Score: {anomaly_score:.4f}")
    print(f"Threshold: {threshold:.4f}")
    print("Predicted Class:", predicted_class)

    # Load original image for visualization
    original_image = Image.open(image_path).convert("RGB")
    original_image = original_image.resize((256, 256))

    # Extract anomaly map and predicted mask
    anomaly_map = (
        prediction.anomaly_map[0].detach().cpu().numpy()
    )

    predicted_mask = (
        prediction.pred_mask[0].detach().cpu().numpy()
    )

    anomaly_map = np.squeeze(anomaly_map)
    predicted_mask = np.squeeze(predicted_mask)

    # Visualize results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")

    axes[1].imshow(anomaly_map, cmap="jet")
    axes[1].set_title("Anomaly Heatmap")

    axes[2].imshow(predicted_mask, cmap="gray")
    axes[2].set_title("Predicted Defect Mask")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        f"{predicted_class} | Anomaly Score: {anomaly_score:.4f}"
    )

    plt.tight_layout()

    # Save visualization
    output_path = (
        project_root / "reports" / "single_image_prediction.png"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=150)

    print("\nVisualization saved at:", output_path)

    plt.show()


if __name__ == "__main__":
    main()

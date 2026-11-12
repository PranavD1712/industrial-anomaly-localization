
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

from predictor import AnomalyPredictor


def main():

    project_root = Path(__file__).resolve().parent.parent

    predictor = AnomalyPredictor(project_root)

    image_path = (
        project_root
        / "data"
        / "bottle"
        / "test"
        / "broken_large"
        / "000.png"
    )

    result = predictor.predict(image_path)

    print("\n========== PREDICTION RESULT ==========")
    print("Image:", Path(result["image_path"]).name)
    print(f"Anomaly Score: {result['anomaly_score']:.4f}")
    print(f"Threshold: {result['threshold']:.4f}")
    print("Predicted Class:", result["predicted_class"])

    # Load original image
    original_image = Image.open(image_path).convert("RGB")
    original_image = original_image.resize((256, 256))

    # Get model outputs
    anomaly_map = result["anomaly_map"]
    defect_mask = result["defect_mask"]

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    fig.suptitle(
        f"{result['predicted_class']} | "
        f"Anomaly Score: {result['anomaly_score']:.4f}",
        fontsize=15
    )

    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")

    axes[1].imshow(anomaly_map, cmap="jet")
    axes[1].set_title("Anomaly Heatmap")

    axes[2].imshow(defect_mask, cmap="gray")
    axes[2].set_title("Predicted Defect Mask")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()

    # Save visualization
    output_path = (
        project_root
        / "reports"
        / "single_image_prediction.png"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print("\nVisualization saved at:", output_path)

    plt.show()


if __name__ == "__main__":
    main()


import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path

import torch
from anomalib.data import MVTecAD
from anomalib.models import Patchcore
from anomalib.engine import Engine

from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def main():

    # ==========================================
    # 1. PROJECT PATHS
    # ==========================================

    project_root = Path(__file__).resolve().parent.parent

    checkpoint_path = (
        project_root
        / "results"
        / "Patchcore"
        / "MVTecAD"
        / "bottle"
        / "latest"
        / "weights"
        / "lightning"
        / "model.ckpt"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    # ==========================================
    # 2. LOAD TRAINED PATCHCORE MODEL
    # ==========================================

    model = Patchcore.load_from_checkpoint(
        checkpoint_path,
        map_location="cpu",
        weights_only=False
    )

    model.eval()

    print("PatchCore checkpoint loaded successfully!")

    # ==========================================
    # 3. EXPERIMENTAL SENSITIVITY SETTINGS
    # ==========================================

    model.post_processor.image_sensitivity = 0.6
    model.post_processor.pixel_sensitivity = 0.5

    print("\n========== SENSITIVITY SETTINGS ==========")

    print(
        "Image sensitivity:",
        model.post_processor.image_sensitivity
    )

    print(
        "Pixel sensitivity:",
        model.post_processor.pixel_sensitivity
    )

    print(
        "Normalized image threshold:",
        model.post_processor.normalized_image_threshold.item()
    )

    print(
        "Normalized pixel threshold:",
        model.post_processor.normalized_pixel_threshold.item()
    )

    # ==========================================
    # 4. INSPECT MODEL THRESHOLDS
    # ==========================================

    print("\n========== MODEL THRESHOLDS ==========")

    post_processor = model.post_processor

    print("Image-level threshold:", post_processor.image_threshold)
    print("Pixel-level threshold:", post_processor.pixel_threshold)

    print("\n========== POST-PROCESSOR STATE ==========")

    for key, value in post_processor.state_dict().items():
        print(f"{key}: {value}")

    # ==========================================
    # 5. LOAD TEST DATASET
    # ==========================================

    datamodule = MVTecAD(
        root=project_root / "data",
        category="bottle",
        num_workers=0
    )

    datamodule.setup()

    # ==========================================
    # 6. RUN INFERENCE
    # ==========================================

    engine = Engine()

    predictions = engine.predict(
        model=model,
        datamodule=datamodule
    )

    if not predictions:
        raise RuntimeError("No predictions were returned.")

    print("\nInference completed successfully!")

    # ==========================================
    # 7. FIND AND VISUALIZE MISCLASSIFIED IMAGES
    # ==========================================

    print("\n========== MISCLASSIFIED IMAGES ==========")

    misclassified_count = 0

    for batch in predictions:

        for i in range(len(batch.gt_label)):

            actual = batch.gt_label[i].item()
            predicted = batch.pred_label[i].item()

            if actual != predicted:

                misclassified_count += 1

                image_path = Path(batch.image_path[i])

                print("\nIncorrect prediction found!")
                print("Image:", image_path)
                print("Ground truth:", actual)
                print("Prediction:", predicted)
                print("Anomaly score:", batch.pred_score[i].item())

                print("\n========== SCORE INSPECTION ==========")

                print("Image threshold:",
                      model.post_processor.normalized_image_threshold.item())

                print("Prediction score:",
                      batch.pred_score[i].item())

                print("Anomaly map minimum:",
                      batch.anomaly_map[i].min().item())

                print("Anomaly map maximum:",
                      batch.anomaly_map[i].max().item())

                print("Predicted mask unique values:",
                      torch.unique(batch.pred_mask[i]))

                # Load original image
                original_image = Image.open(image_path).convert("RGB")

                # Extract ground truth mask
                gt_mask = (
                    batch.gt_mask[i]
                    .detach()
                    .cpu()
                    .squeeze()
                    .numpy()
                )

                # Extract anomaly map
                anomaly_map = (
                    batch.anomaly_map[i]
                    .detach()
                    .cpu()
                    .squeeze()
                    .numpy()
                )

                # Extract predicted mask
                pred_mask = (
                    batch.pred_mask[i]
                    .detach()
                    .cpu()
                    .squeeze()
                    .numpy()
                )

                # Resize original image
                original_image = original_image.resize(
                    (anomaly_map.shape[1], anomaly_map.shape[0])
                )

                # Create comparison visualization
                fig, axes = plt.subplots(
                    1, 4, figsize=(18, 5)
                )

                axes[0].imshow(original_image)
                axes[0].set_title("Original Image")

                axes[1].imshow(gt_mask, cmap="gray")
                axes[1].set_title("Ground Truth Mask")

                axes[2].imshow(anomaly_map, cmap="jet")
                axes[2].set_title("Anomaly Heatmap")

                axes[3].imshow(pred_mask, cmap="gray")
                axes[3].set_title("Predicted Mask")

                for ax in axes:
                    ax.axis("off")

                plt.tight_layout()

                output_path = (
                    project_root
                    / "reports"
                    / f"misclassified_{image_path.stem}.png"
                )

                output_path.parent.mkdir(
                    parents=True,
                    exist_ok=True
                )

                plt.savefig(output_path, dpi=150)
                plt.close(fig)

                print("Visualization saved at:", output_path)

    if misclassified_count == 0:
        print("No misclassified images found!")
    else:
        print(
            f"\nTotal misclassified images: {misclassified_count}"
        )

    # ==========================================
    # 8. COMBINE PREDICTION BATCHES
    # ==========================================

    gt_labels = torch.cat([
        batch.gt_label.detach().cpu().reshape(-1)
        for batch in predictions
    ])

    pred_labels = torch.cat([
        batch.pred_label.detach().cpu().reshape(-1)
        for batch in predictions
    ])

    pred_scores = torch.cat([
        batch.pred_score.detach().cpu().reshape(-1)
        for batch in predictions
    ])

    gt_masks = torch.cat([
        batch.gt_mask.detach().cpu()
        for batch in predictions
    ])

    pred_masks = torch.cat([
        batch.pred_mask.detach().cpu()
        for batch in predictions
    ])

    anomaly_maps = torch.cat([
        batch.anomaly_map.detach().cpu()
        for batch in predictions
    ])

    # ==========================================
    # 9. PREPARE IMAGE-LEVEL DATA
    # ==========================================

    gt_labels = gt_labels.int().numpy()
    pred_labels = pred_labels.int().numpy()
    pred_scores = pred_scores.numpy()

    # ==========================================
    # 10. PREPARE PIXEL-LEVEL DATA
    # ==========================================

    gt_masks = gt_masks.reshape(-1).int().numpy()
    pred_masks = pred_masks.reshape(-1).int().numpy()
    anomaly_maps = anomaly_maps.reshape(-1).numpy()

    # ==========================================
    # 11. IMAGE-LEVEL EVALUATION
    # ==========================================

    print("\n========== IMAGE-LEVEL EVALUATION ==========")

    image_auc = roc_auc_score(
        gt_labels,
        pred_scores
    )

    image_precision = precision_score(
        gt_labels,
        pred_labels,
        zero_division=0
    )

    image_recall = recall_score(
        gt_labels,
        pred_labels,
        zero_division=0
    )

    image_f1 = f1_score(
        gt_labels,
        pred_labels,
        zero_division=0
    )

    print(f"Image-level AUROC : {image_auc:.4f}")
    print(f"Image Precision   : {image_precision:.4f}")
    print(f"Image Recall      : {image_recall:.4f}")
    print(f"Image F1-score    : {image_f1:.4f}")

    print("\nImage-level Confusion Matrix:")
    print(confusion_matrix(gt_labels, pred_labels))

    # ==========================================
    # 12. PIXEL-LEVEL EVALUATION
    # ==========================================

    print("\n========== PIXEL-LEVEL EVALUATION ==========")

    pixel_auc = roc_auc_score(
        gt_masks,
        anomaly_maps
    )

    pixel_precision = precision_score(
        gt_masks,
        pred_masks,
        zero_division=0
    )

    pixel_recall = recall_score(
        gt_masks,
        pred_masks,
        zero_division=0
    )

    pixel_f1 = f1_score(
        gt_masks,
        pred_masks,
        zero_division=0
    )

    print(f"Pixel-level AUROC : {pixel_auc:.4f}")
    print(f"Pixel Precision   : {pixel_precision:.4f}")
    print(f"Pixel Recall      : {pixel_recall:.4f}")
    print(f"Pixel F1-score    : {pixel_f1:.4f}")

    print("\nPixel-level Confusion Matrix:")
    print(confusion_matrix(gt_masks, pred_masks))

    print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    main()

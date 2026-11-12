
import json
import matplotlib.pyplot as plt
from pathlib import Path

import numpy as np
import torch

from anomalib.data import MVTecAD
from anomalib.models import Patchcore
from anomalib.engine import Engine

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
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

    # Create reports directory
    reports_path = project_root / "reports"
    reports_path.mkdir(parents=True, exist_ok=True)

    # ==========================================
    # 2. LOAD PATCHCORE MODEL
    # ==========================================

    model = Patchcore.load_from_checkpoint(
        checkpoint_path,
        map_location="cpu",
        weights_only=False
    )

    model.eval()

    print("PatchCore checkpoint loaded successfully!")

    # ==========================================
    # 3. LOAD DATASET
    # ==========================================

    datamodule = MVTecAD(
        root=project_root / "data",
        category="bottle",
        num_workers=0
    )

    datamodule.setup()

    # ==========================================
    # 4. RUN INFERENCE
    # ==========================================

    engine = Engine()

    predictions = engine.predict(
        model=model,
        datamodule=datamodule
    )

    if not predictions:
        raise RuntimeError("No predictions returned.")

    print("Inference completed successfully!")

    # ==========================================
    # 5. COLLECT IMAGE LABELS AND SCORES
    # ==========================================

    gt_labels = torch.cat([
        batch.gt_label.detach().cpu().reshape(-1)
        for batch in predictions
    ]).numpy().astype(int)

    pred_scores = torch.cat([
        batch.pred_score.detach().cpu().reshape(-1)
        for batch in predictions
    ]).numpy()

    print("\n========== DATASET INFORMATION ==========")

    print("Total images:", len(gt_labels))
    print("Normal images:", np.sum(gt_labels == 0))
    print("Defective images:", np.sum(gt_labels == 1))

    # ==========================================
    # 6. SPLIT INTO VALIDATION AND HOLDOUT
    # ==========================================

    val_scores, holdout_scores, val_labels, holdout_labels = (
        train_test_split(
            pred_scores,
            gt_labels,
            test_size=0.30,
            random_state=42,
            stratify=gt_labels
        )
    )

    print("\n========== DATA SPLIT ==========")

    print("Validation images:", len(val_labels))
    print("Holdout images:", len(holdout_labels))

    # ==========================================
    # 7. FIND BEST THRESHOLD ON VALIDATION SET
    # ==========================================

    thresholds = np.arange(0.05, 0.96, 0.01)

    best_threshold = None
    best_f1 = -1

    # Store metric values for visualization
    precision_values = []
    recall_values = []
    f1_values = []

    print("\n========== THRESHOLD SEARCH ==========")

    for threshold in thresholds:

        val_predictions = (
            val_scores > threshold
        ).astype(int)

        # Calculate precision
        current_precision = precision_score(
            val_labels,
            val_predictions,
            zero_division=0
        )

        # Calculate recall
        current_recall = recall_score(
            val_labels,
            val_predictions,
            zero_division=0
        )

        # Calculate F1-score
        current_f1 = f1_score(
            val_labels,
            val_predictions,
            zero_division=0
        )

        precision_values.append(current_precision)
        recall_values.append(current_recall)
        f1_values.append(current_f1)

        # Find threshold with the highest F1-score
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = threshold

    print("Best threshold:", round(best_threshold, 3))
    print("Validation F1-score:", round(best_f1, 4))

    # ==========================================
    # 8. VISUALIZE THRESHOLD PERFORMANCE
    # ==========================================

    plt.figure(figsize=(10, 6))

    plt.plot(
        thresholds,
        precision_values,
        label="Precision"
    )

    plt.plot(
        thresholds,
        recall_values,
        label="Recall"
    )

    plt.plot(
        thresholds,
        f1_values,
        label="F1-score"
    )

    # Mark selected threshold
    plt.axvline(
        x=best_threshold,
        linestyle="--",
        label=f"Best Threshold: {best_threshold:.2f}"
    )

    plt.xlabel("Anomaly Threshold")
    plt.ylabel("Score")

    plt.title(
        "Precision, Recall and F1-score vs Threshold"
    )

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    graph_path = reports_path / "threshold_analysis.png"

    plt.savefig(
        graph_path,
        dpi=150
    )

    plt.close()

    print("\nThreshold graph saved at:", graph_path)

    # ==========================================
    # 9. SAVE SELECTED THRESHOLD AS JSON
    # ==========================================

    threshold_config = {
        "model": "PatchCore",
        "dataset": "MVTecAD",
        "category": "bottle",
        "threshold": float(best_threshold),
        "selection_metric": "F1-score",
        "validation_f1": float(best_f1),
        "random_state": 42
    }

    json_path = reports_path / "threshold_analysis.json"

    with open(json_path, "w") as file:
        json.dump(
            threshold_config,
            file,
            indent=4
        )

    print("Threshold configuration saved at:", json_path)

    # ==========================================
    # 10. EVALUATE ON HOLDOUT SET
    # ==========================================

    holdout_predictions = (
        holdout_scores > best_threshold
    ).astype(int)

    holdout_precision = precision_score(
        holdout_labels,
        holdout_predictions,
        zero_division=0
    )

    holdout_recall = recall_score(
        holdout_labels,
        holdout_predictions,
        zero_division=0
    )

    holdout_f1 = f1_score(
        holdout_labels,
        holdout_predictions,
        zero_division=0
    )

    print("\n========== HOLDOUT EVALUATION ==========")

    print(f"Selected threshold: {best_threshold:.3f}")
    print(f"Precision:          {holdout_precision:.4f}")
    print(f"Recall:             {holdout_recall:.4f}")
    print(f"F1-score:           {holdout_f1:.4f}")

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            holdout_labels,
            holdout_predictions
        )
    )

    print("\nClassification Report:")

    print(
        classification_report(
            holdout_labels,
            holdout_predictions,
            zero_division=0
        )
    )

    print("\nThreshold analysis completed successfully!")


if __name__ == "__main__":
    main()

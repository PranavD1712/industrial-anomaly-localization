
from pathlib import Path
import json

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from anomalib.data import ImageItem, ImageBatch
from huggingface_hub import hf_hub_download
from anomalib.models import Patchcore
from anomalib.engine import Engine


class SingleImageDataset(Dataset):

    def __init__(self, image_path):
        self.image_path = Path(image_path)

    def __len__(self):
        return 1

    def __getitem__(self, index):

        image = Image.open(self.image_path).convert("RGB")
        image = np.array(image).copy()

        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1).float() / 255.0

        return ImageItem(
            image=image,
            image_path=str(self.image_path)
        )


class AnomalyPredictor:

    def __init__(self, project_root):

        self.project_root = Path(project_root)
        
        checkpoint_path = hf_hub_download(
            repo_id="PranavD1712/industrial-anomaly-patchcore-bottle",
            filename="patchcore_bottle.ckpt"
        )
        

        self.model = Patchcore.load_from_checkpoint(
            checkpoint_path,
            map_location="cpu",
            weights_only=False
        )

        self.model.eval()

        # Load classification threshold
        threshold_path = (
            self.project_root
            / "reports"
            / "threshold_analysis.json"
        )

        with open(threshold_path, "r") as file:
            threshold_config = json.load(file)

        self.threshold = float(threshold_config["threshold"])

        if not 0 <= self.threshold <= 1:
            raise ValueError(
                f"Invalid classification threshold: {self.threshold}"
                )
        
        self.engine = Engine()

        print("PatchCore predictor initialized successfully!")
        print(f"Checkpoint loaded from: {checkpoint_path}")
        print(f"Classification threshold: {self.threshold:.4f}")

    def predict(self, image_path):

        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        # Create single-image dataset
        dataset = SingleImageDataset(image_path)

        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=ImageBatch.collate
        )

        # Run inference
        predictions = self.engine.predict(
            model=self.model,
            dataloaders=dataloader
        )

        prediction = predictions[0]

        # Extract anomaly score
        anomaly_score = float(
            prediction.pred_score[0].cpu().item()
        )

        # Apply our selected image classification threshold
        is_defective = anomaly_score > self.threshold

        # Extract localization outputs
        anomaly_map = (
            prediction.anomaly_map[0]
            .detach()
            .cpu()
            .numpy()
        )

        defect_mask = (
            prediction.pred_mask[0]
            .detach()
            .cpu()
            .numpy()
            .squeeze()
        )

        # Return structured prediction results
        return {
            "image_path": str(image_path),
            "anomaly_score": anomaly_score,
            "threshold": self.threshold,
            "predicted_class": (
                "Defective" if is_defective else "Normal"
            ),
            "anomaly_map": anomaly_map,
            "defect_mask": defect_mask
        }

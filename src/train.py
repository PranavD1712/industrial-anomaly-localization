
from pathlib import Path

from anomalib.data import MVTecAD
from anomalib.models import Patchcore
from anomalib.engine import Engine


def main():

    # 1. Project directory
    project_root = Path(__file__).resolve().parent.parent

    # 2. Dataset directory
    dataset_root = project_root / "data"

    # 3. Load dataset
    datamodule = MVTecAD(
        root=dataset_root,
        category="bottle",
        num_workers=0
    )

    # 4. Initialize PatchCore
    model = Patchcore()

    # 5. Initialize engine
    engine = Engine()

    # 6. Start training
    print("Starting PatchCore training...")

    engine.fit(
        model=model,
        datamodule=datamodule
    )

    print("PatchCore training completed!")


# Important for Windows multiprocessing
if __name__ == "__main__":
    main()

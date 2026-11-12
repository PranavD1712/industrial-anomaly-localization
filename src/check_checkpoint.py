from pathlib import Path

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

if checkpoint_path.exists():
    print("Checkpoint found!")
    print("File size:", round(checkpoint_path.stat().st_size / (1024 * 1024), 2), "MB")
else:
    print("Checkpoint not found. Check the folder path.")
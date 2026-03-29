import subprocess
from pathlib import Path

from ultralytics import YOLO

DATASET_YAML = Path(__file__).parent.parent / "dataset.yaml"
RUNS_DIR = Path(__file__).parent.parent / "runs" / "detect"


def train(model_size: str, epochs: int, imgsz: int) -> None:
    model = YOLO(f"yolo26{model_size}.pt")
    model.train(
        data=str(DATASET_YAML),
        epochs=epochs,
        imgsz=imgsz,
        device="mps",
        batch=16,
        patience=10,
        seed=0,
        optimizer="AdamW",
        project=str(RUNS_DIR),
        name="yolo26s_aerial",
    )


if __name__ == "__main__":
    caffeinate = subprocess.Popen(["caffeinate", "-s"])
    try:
        train(model_size="s", epochs=18, imgsz=544)
    finally:
        caffeinate.terminate()

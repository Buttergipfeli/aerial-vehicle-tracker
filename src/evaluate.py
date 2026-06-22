import gc
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATASET_YAML = ROOT / "dataset.yaml"
RUNS_DIR = ROOT / "runs" / "detect"
MODEL_NAMES = ("yolo26s_aerial", "yolo26m_aerial")

DEVICE = "mps"
IMGSZ = 544
SPLIT = "test"
CACHE = False


def evaluate(model_name: str) -> None:
    weights = RUNS_DIR / model_name / "weights" / "best.pt"

    if not DATASET_YAML.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {DATASET_YAML}")
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    model = YOLO(str(weights))
    metrics = model.val(
        data=str(DATASET_YAML),
        split=SPLIT,
        device=DEVICE,
        imgsz=IMGSZ,
        cache=CACHE,
        project=str(RUNS_DIR),
        name=f"eval_{model_name}_{SPLIT}",
        exist_ok=False,
    )
    precision, recall, map50, map50_95 = metrics.mean_results()

    print()
    print(f"{model_name} ({SPLIT})")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"mAP50: {map50:.4f}")
    print(f"mAP50-95: {map50_95:.4f}")

    del metrics
    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


if __name__ == "__main__":
    for name in MODEL_NAMES:
        evaluate(name)

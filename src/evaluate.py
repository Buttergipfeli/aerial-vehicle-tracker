import gc
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATASET_YAML = ROOT / "dataset.yaml"
DATASET_BINARY_YAML = ROOT / "dataset_binary.yaml"
RUNS_DIR = ROOT / "runs" / "detect"
MODELS = (
    ("yolo26s_aerial", DATASET_YAML),
    ("yolo26m_aerial", DATASET_YAML),
    ("yolo26s_binary", DATASET_BINARY_YAML),
    ("yolo26m_binary", DATASET_BINARY_YAML),
)

DEVICE = "mps"
IMGSZ = 544
SPLIT = "test"
CACHE = False


def evaluate(model_name: str, dataset_yaml: Path) -> None:
    weights = RUNS_DIR / model_name / "weights" / "best.pt"

    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {dataset_yaml}")
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    model = YOLO(str(weights))
    metrics = model.val(
        data=str(dataset_yaml),
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
    for name, data in MODELS:
        evaluate(name, data)

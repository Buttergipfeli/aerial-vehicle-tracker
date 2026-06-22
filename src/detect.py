from pathlib import Path

from ultralytics import YOLO

FINETUNED_S_WEIGHTS = Path(__file__).parent.parent / "runs" / "detect" / "yolo26s_aerial" / "weights" / "best.pt"
FINETUNED_M_WEIGHTS = Path(__file__).parent.parent / "runs" / "detect" / "yolo26m_aerial" / "weights" / "best.pt"
PRETRAINED_S_WEIGHTS = "yolo26s.pt"
PRETRAINED_M_WEIGHTS = "yolo26m.pt"
INPUT_DIR = Path(__file__).parent.parent / "assets" / "detect" / "input"
OUTPUT_DIR = Path(__file__).parent.parent / "assets" / "detect" / "output"


def detect(image_path: Path, weights: str | Path, output_dir: Path) -> None:
    model = YOLO(str(weights))
    results = model(str(image_path), device="mps")
    output_path = output_dir / image_path.name
    results[0].save(str(output_path))
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    output_dirs = {
        "finetuned_s": FINETUNED_S_WEIGHTS,
        "finetuned_m": FINETUNED_M_WEIGHTS,
        "pretrained_s": PRETRAINED_S_WEIGHTS,
        "pretrained_m": PRETRAINED_M_WEIGHTS,
    }
    for name, weights in output_dirs.items():
        out = OUTPUT_DIR / name
        out.mkdir(parents=True, exist_ok=True)
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for image in INPUT_DIR.glob(ext):
                detect(image, weights, out)

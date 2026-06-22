from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).parent.parent
INPUT_DIR = ROOT / "assets" / "track" / "input"
OUTPUT_DIR = ROOT / "assets" / "track" / "output"
WEIGHTS = ROOT / "runs" / "detect" / "yolo26s_aerial" / "weights" / "best.pt"
CROP_SIZE = (64, 64)


class VehicleTracker:
    def __init__(self):
        self.model = YOLO(str(WEIGHTS))
        images = []
        for ext in (".jpg", ".jpeg"):
            images.extend(INPUT_DIR.glob(f"track*{ext}"))
        for img in images:
            digits = "".join(c for c in img.stem if c.isdigit())
            if not digits:
                raise ValueError(f"Invalid filename: {img.name} (expected track1.jpg, track2.jpg, ...)")
        self.images = sorted(images, key=lambda p: int("".join(c for c in p.stem if c.isdigit())))
        self.reference_crop = None

    def detect(self, image_path):
        results = self.model(str(image_path), device="mps")
        return results[0]

    def draw_boxes(self, image, boxes, highlight_idx=None):
        draw = ImageDraw.Draw(image)
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_name = self.model.names[int(box.cls[0])]
            conf = box.conf[0]
            is_highlight = i == highlight_idx
            color = "red" if is_highlight else "lime"
            width = 3 if is_highlight else 2
            draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
            label = f"[{i}] {cls_name} {conf:.2f}"
            if is_highlight:
                label = f">> {label}"
            draw.text((x1, y1 - 15), label, fill=color)
        return image

    def crop_box(self, image, box):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        return image.crop((x1, y1, x2, y2))

    def find_match(self, boxes, image):
        if self.reference_crop is None or not boxes:
            return None

        ref = np.array(self.reference_crop.resize(CROP_SIZE), dtype=np.float32).flatten()
        ref_norm = np.linalg.norm(ref)
        if ref_norm == 0:
            return None

        best_sim, best_idx = -1, None
        for i, box in enumerate(boxes):
            crop = self.crop_box(image, box)
            if crop.size[0] < 4 or crop.size[1] < 4:
                continue
            cand = np.array(crop.resize(CROP_SIZE), dtype=np.float32).flatten()
            cand_norm = np.linalg.norm(cand)
            if cand_norm == 0:
                continue
            sim = np.dot(ref, cand) / (ref_norm * cand_norm)
            if sim > best_sim:
                best_sim, best_idx = sim, i
        return best_idx

    def track(self):
        if not self.images:
            print(f"No images found in {INPUT_DIR}")
            return

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        result = self.detect(self.images[0])
        boxes = result.boxes
        first_img = Image.open(self.images[0])

        numbered = self.draw_boxes(first_img.copy(), boxes)
        preview_path = OUTPUT_DIR / "preview.jpg"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        numbered.save(preview_path)
        numbered.show()
        print(f"Detected {len(boxes)} vehicles. Preview opened.")
        print("Select a vehicle by entering its number:")

        while True:
            try:
                choice = int(input("> "))
                if 0 <= choice < len(boxes):
                    break
                print(f"Invalid. Enter 0-{len(boxes) - 1}")
            except ValueError:
                print("Enter a number.")

        selected = boxes[choice]
        cls_name = self.model.names[int(selected.cls[0])]
        self.reference_crop = self.crop_box(first_img, selected)
        print(f"Selected [{choice}] {cls_name} - tracking {len(self.images) - 1} images...")

        for img_path in self.images[1:]:
            result = self.detect(img_path)
            img = Image.open(img_path)
            match_idx = self.find_match(result.boxes, img)
            img = self.draw_boxes(img, result.boxes, highlight_idx=match_idx)
            img.save(OUTPUT_DIR / img_path.name)
            status = f"matched [{match_idx}]" if match_idx is not None else "no match"
            print(f"  {img_path.name}: {status}")

        print(f"Done. Results in {OUTPUT_DIR}")


if __name__ == "__main__":
    VehicleTracker().track()

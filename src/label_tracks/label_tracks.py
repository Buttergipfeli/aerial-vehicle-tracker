import argparse
import json
import mimetypes
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = ROOT / "data" / "data" / "7053442" / "images"
LABELS_FILE = ROOT / "assets" / "track" / "labels" / "tracks.json"
WEIGHTS = ROOT / "runs" / "detect" / "yolo26m_binary" / "weights" / "best.pt"

HOST = "127.0.0.1"
PORT = 8765
DEVICE = "mps"
CONFIDENCE = 0.25
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SPLIT_ORDER = {"train": 0, "valid": 1, "test": 2}
DEFAULT_IOU_THRESHOLD = 0.1
DEFAULT_MAX_MISSING = 2
DEFAULT_MAX_FRAME_GAP = 60
DEFAULT_MIN_FRAMES = 2


def frame_number(path: Path) -> int:
    digits = ""
    for char in reversed(path.stem):
        if not char.isdigit():
            break
        digits = char + digits
    return int(digits) if digits else 0


def sequence_key(path: Path) -> str:
    stem = path.stem
    index = len(stem)
    while index > 0 and stem[index - 1].isdigit():
        index -= 1
    return stem[:index]


def image_sort_key(path: Path) -> tuple[str, int, int, str]:
    split = next((part for part in path.parts if part in SPLIT_ORDER), "")
    return sequence_key(path), frame_number(path), SPLIT_ORDER.get(split, 99), path.name


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def image_key(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def load_images(input_dir: Path) -> list[Path]:
    return sorted(
        [path for path in input_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS],
        key=image_sort_key,
    )


def load_tracks() -> list[dict]:
    if not LABELS_FILE.exists():
        return []
    with LABELS_FILE.open(encoding="utf-8") as file:
        return json.load(file).get("tracks", [])


def save_tracks(tracks: list[dict]) -> None:
    LABELS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LABELS_FILE.open("w", encoding="utf-8") as file:
        json.dump({"tracks": tracks}, file, indent=2)


def optional_int(value) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def optional_float(value, default: float) -> float:
    if value in (None, ""):
        return default
    return float(value)


def grouped_images(images: list[Path]) -> list[list[Path]]:
    groups = {}
    for image in images:
        groups.setdefault(sequence_key(image), []).append(image)
    return [sorted(group, key=frame_number) for _key, group in sorted(groups.items())]


def iou(first: list[float], second: list[float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def detect(model: YOLO, image: Path, device: str, confidence: float) -> list[dict]:
    result = model(str(image), device=device, conf=confidence, verbose=False)[0]
    detections = []
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append({"bbox": [x1, y1, x2, y2], "confidence": float(box.conf[0])})
    return detections


def match_tracks(active_tracks: list[dict], detections: list[dict], threshold: float) -> list[tuple[int, int]]:
    pairs = []
    for track_index, track in enumerate(active_tracks):
        for detection_index, detection in enumerate(detections):
            score = iou(track["bbox"], detection["bbox"])
            if score >= threshold:
                pairs.append((score, track_index, detection_index))

    matches = []
    used_tracks = set()
    used_detections = set()
    for _score, track_index, detection_index in sorted(pairs, reverse=True):
        if track_index in used_tracks or detection_index in used_detections:
            continue
        matches.append((track_index, detection_index))
        used_tracks.add(track_index)
        used_detections.add(detection_index)
    return matches


def append_prelabel_frame(track: dict, image: Path, detection: dict) -> None:
    track["bbox"] = detection["bbox"]
    track["missing"] = 0
    track["frames"].append(
        {
            "image": image_key(image),
            "bbox": [round(value, 2) for value in detection["bbox"]],
            "confidence": round(detection["confidence"], 4),
        }
    )


def prelabel_group(
    model: YOLO,
    images: list[Path],
    next_track_id: int,
    device: str,
    confidence: float,
    iou_threshold: float,
    max_missing: int,
    max_frame_gap: int,
) -> tuple[list[dict], int]:
    tracks = []
    active_tracks = []
    previous_frame = None

    for image in images:
        current_frame = frame_number(image)
        if previous_frame is not None and current_frame - previous_frame > max_frame_gap:
            active_tracks = []
        previous_frame = current_frame

        detections = detect(model, image, device, confidence)
        matches = match_tracks(active_tracks, detections, iou_threshold)
        matched_tracks = {track_index for track_index, _detection_index in matches}
        matched_detections = {detection_index for _track_index, detection_index in matches}

        for track_index, detection_index in matches:
            append_prelabel_frame(active_tracks[track_index], image, detections[detection_index])

        for track_index, track in enumerate(active_tracks):
            if track_index not in matched_tracks:
                track["missing"] += 1

        active_tracks = [track for track in active_tracks if track["missing"] <= max_missing]

        for detection_index, detection in enumerate(detections):
            if detection_index in matched_detections:
                continue
            track = {"id": next_track_id, "bbox": detection["bbox"], "missing": 0, "frames": []}
            next_track_id += 1
            append_prelabel_frame(track, image, detection)
            tracks.append(track)
            active_tracks.append(track)

    return tracks, next_track_id


def prelabel_tracks(
    input_dir: Path,
    output: Path,
    device: str,
    confidence: float,
    iou_threshold: float,
    max_missing: int,
    max_frame_gap: int,
    min_frames: int,
    limit_images: int | None,
    overwrite: bool,
) -> dict:
    if output.exists() and not overwrite:
        print(f"Prelabel skipped because labels already exist: {output}")
        return {"skipped": True, "tracks": 0, "frames": 0}

    images = load_images(input_dir)
    if limit_images:
        images = images[:limit_images]
    if not images:
        raise FileNotFoundError(f"No images found in {input_dir}")

    model = YOLO(str(WEIGHTS))
    tracks = []
    next_track_id = 1
    for group_index, group in enumerate(grouped_images(images), start=1):
        group_tracks, next_track_id = prelabel_group(
            model,
            group,
            next_track_id,
            device,
            confidence,
            iou_threshold,
            max_missing,
            max_frame_gap,
        )
        tracks.extend(group_tracks)
        print(f"Prelabel group {group_index}: {len(group)} images, {len(group_tracks)} raw tracks")

    tracks = [{"id": track["id"], "frames": track["frames"]} for track in tracks if len(track["frames"]) >= min_frames]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump({"tracks": tracks}, file, indent=2)

    frame_count = sum(len(track["frames"]) for track in tracks)
    print(f"Prelabel saved {len(tracks)} tracks with {frame_count} frames to {output}")
    return {"skipped": False, "tracks": len(tracks), "frames": frame_count}


class TrackLabelService:
    def __init__(self, input_dir: Path, limit_images: int | None = None) -> None:
        self.model = None
        self.lock = threading.Lock()
        self.configure(input_dir, limit_images)

    def configure(self, input_dir: Path, limit_images: int | None = None) -> None:
        self.input_dir = input_dir
        self.limit_images = limit_images
        self.images = load_images(input_dir)
        if limit_images:
            self.images = self.images[:limit_images]
        if not self.images:
            raise FileNotFoundError(f"No images found in {input_dir}")

        self.tracks = load_tracks()
        self.active_track = []
        self.next_track_id = max((track["id"] for track in self.tracks), default=0) + 1
        self.image_index = 0
        self.detection_cache = {}
        self.status = "Click a vehicle, then Next. Use Save when the vehicle is no longer visible."

    def model_instance(self) -> YOLO:
        if not WEIGHTS.exists():
            raise FileNotFoundError(f"Weights not found: {WEIGHTS}")
        if self.model is None:
            self.model = YOLO(str(WEIGHTS))
        return self.model

    def config_payload(self) -> dict:
        labels_exist = LABELS_FILE.exists()
        tracks = load_tracks()
        return {
            "input_dir": image_key(self.input_dir),
            "images": len(self.images),
            "sequences": self.sequence_count(),
            "labels_exist": labels_exist,
            "track_count": len(tracks),
            "labels_file": image_key(LABELS_FILE),
            "defaults": {
                "confidence": CONFIDENCE,
                "iou_threshold": DEFAULT_IOU_THRESHOLD,
                "max_missing": DEFAULT_MAX_MISSING,
                "max_frame_gap": DEFAULT_MAX_FRAME_GAP,
                "min_frames": DEFAULT_MIN_FRAMES,
                "limit_images": self.limit_images,
            },
        }

    def start(self, data: dict) -> dict:
        with self.lock:
            input_dir = resolve_path(Path(data.get("input_dir") or DEFAULT_INPUT_DIR)).resolve()
            limit_images = optional_int(data.get("limit_images"))
            if data.get("prelabel"):
                summary = prelabel_tracks(
                    input_dir,
                    LABELS_FILE,
                    DEVICE,
                    optional_float(data.get("confidence"), CONFIDENCE),
                    optional_float(data.get("iou_threshold"), DEFAULT_IOU_THRESHOLD),
                    optional_int(data.get("max_missing")) or DEFAULT_MAX_MISSING,
                    optional_int(data.get("max_frame_gap")) or DEFAULT_MAX_FRAME_GAP,
                    optional_int(data.get("min_frames")) or DEFAULT_MIN_FRAMES,
                    limit_images,
                    bool(data.get("overwrite_labels")),
                )
                self.configure(input_dir, limit_images)
                if summary["skipped"]:
                    self.status = "Existing labels kept. Review started."
                else:
                    self.status = f"Prelabel complete: {summary['tracks']} tracks, {summary['frames']} frames."
            else:
                self.configure(input_dir, limit_images)
                self.status = "Review started."
            return self.payload()

    def payload(self) -> dict:
        image = self.images[self.image_index]
        current_sequence = sequence_key(image)
        return {
            "index": self.image_index,
            "total": len(self.images),
            "image": {"name": image.name, "path": image_key(image), "url": f"/image/{self.image_index}"},
            "sequence": {
                "name": current_sequence,
                "index": self.sequence_index(current_sequence),
                "total": self.sequence_count(),
            },
            "boxes": self.detect_boxes(self.image_index),
            "saved": self.saved_boxes(self.image_index),
            "active_count": len(self.active_track),
            "saved_count": len(self.tracks),
            "status": self.status,
        }

    def set_index(self, index: int) -> dict:
        with self.lock:
            self.image_index = max(0, min(index, len(self.images) - 1))
            return self.payload()

    def detect_boxes(self, index: int) -> list[dict]:
        if index in self.detection_cache:
            return self.detection_cache[index]

        result = self.model_instance()(str(self.images[index]), device=DEVICE, conf=CONFIDENCE, verbose=False)[0]
        boxes = []
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            boxes.append({"bbox": [x1, y1, x2, y2], "confidence": float(box.conf[0])})

        self.detection_cache[index] = boxes
        return boxes

    def saved_boxes(self, index: int) -> list[dict]:
        current_image_key = image_key(self.images[index])
        saved = []
        for track in self.tracks:
            for frame in track["frames"]:
                if frame["image"] == current_image_key:
                    saved.append({"track_id": track["id"], "bbox": frame["bbox"], "active": False})
        for frame in self.active_track:
            if frame["image"] == current_image_key:
                saved.append({"track_id": "active", "bbox": frame["bbox"], "active": True})
        return saved

    def action(self, data: dict) -> dict:
        with self.lock:
            action = data.get("action")
            if action == "next":
                self.next_image(data.get("box_index"))
            elif action == "next_sequence":
                self.next_sequence()
            elif action == "delete_frame":
                self.delete_frame(data.get("track_id"))
            elif action == "delete_track":
                self.delete_track(data.get("track_id"))
            elif action == "previous":
                self.previous_image()
            elif action == "skip":
                self.skip_image()
            elif action == "clear":
                self.clear_current()
            elif action == "save":
                self.save_active_track()
            elif action == "stop":
                self.status = "Server stopped. You can close this browser tab."
            else:
                self.status = f"Unknown action: {action}"
            return self.payload()

    def next_image(self, box_index: int | None) -> None:
        boxes = self.detect_boxes(self.image_index)
        if not isinstance(box_index, int) or box_index < 0 or box_index >= len(boxes):
            self.status = "Select a detected vehicle or use Skip."
            return

        box = boxes[box_index]
        current_image_key = image_key(self.images[self.image_index])
        self.active_track = [frame for frame in self.active_track if frame["image"] != current_image_key]
        self.active_track.append(
            {
                "image": current_image_key,
                "bbox": [round(value, 2) for value in box["bbox"]],
                "confidence": round(box["confidence"], 4),
            }
        )
        self.status = "Label added."
        self.move_to_next_image()

    def previous_image(self) -> None:
        if self.image_index > 0:
            self.image_index -= 1
            self.status = "Previous image."
        else:
            self.status = "Already at first image."

    def skip_image(self) -> None:
        self.status = "Image skipped."
        self.move_to_next_image()

    def move_to_next_image(self) -> None:
        current_sequence = self.current_sequence()
        if self.image_index < len(self.images) - 1:
            self.image_index += 1
            if self.current_sequence() != current_sequence:
                saved = self.save_active_track_if_needed()
                self.status = f"New sequence: {self.current_sequence()}." if not saved else f"New sequence: {self.current_sequence()}. Active track saved."
        else:
            self.status = "End reached."

    def clear_current(self) -> None:
        current_image_key = image_key(self.images[self.image_index])
        before = len(self.active_track)
        self.active_track = [frame for frame in self.active_track if frame["image"] != current_image_key]
        removed = before - len(self.active_track)
        self.status = f"Cleared {removed} active labels for current image." if removed else "Current selection cleared."

    def delete_frame(self, track_id: int | None) -> None:
        if not isinstance(track_id, int):
            self.status = "Select a saved track first."
            return

        current_image_key = image_key(self.images[self.image_index])
        removed = 0
        for track in self.tracks:
            if track["id"] != track_id:
                continue
            before = len(track["frames"])
            track["frames"] = [frame for frame in track["frames"] if frame["image"] != current_image_key]
            removed = before - len(track["frames"])
            break

        self.tracks = [track for track in self.tracks if track["frames"]]
        save_tracks(self.tracks)
        self.status = f"Deleted frame from track {track_id}." if removed else "No frame deleted."

    def delete_track(self, track_id: int | None) -> None:
        if not isinstance(track_id, int):
            self.status = "Select a saved track first."
            return

        before = len(self.tracks)
        self.tracks = [track for track in self.tracks if track["id"] != track_id]
        save_tracks(self.tracks)
        self.status = f"Deleted track {track_id}." if len(self.tracks) < before else "No track deleted."

    def current_sequence(self) -> str:
        return sequence_key(self.images[self.image_index])

    def sequence_count(self) -> int:
        return len({sequence_key(image) for image in self.images})

    def sequence_index(self, sequence: str) -> int:
        sequences = []
        for image in self.images:
            current_sequence = sequence_key(image)
            if not sequences or sequences[-1] != current_sequence:
                sequences.append(current_sequence)
        return sequences.index(sequence) + 1

    def next_sequence(self) -> None:
        current_sequence = self.current_sequence()
        for index in range(self.image_index + 1, len(self.images)):
            if sequence_key(self.images[index]) == current_sequence:
                continue
            self.save_active_track_if_needed()
            self.image_index = index
            self.status = f"Moved to sequence {sequence_key(self.images[index])}."
            return
        self.status = "Already at last sequence."

    def save_active_track_if_needed(self) -> bool:
        if not self.active_track:
            return False

        self.tracks.append({"id": self.next_track_id, "frames": self.active_track})
        self.next_track_id += 1
        self.active_track = []
        save_tracks(self.tracks)
        return True

    def save_active_track(self) -> None:
        if not self.save_active_track_if_needed():
            self.status = "No vehicle labels selected for this track."
            return

        self.status = f"Saved labels to {LABELS_FILE}"


class TrackLabelHandler(BaseHTTPRequestHandler):
    service: TrackLabelService

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_static("index.html")
            return
        if parsed.path in ("/styles.css", "/app.js"):
            self.send_static(parsed.path.removeprefix("/"))
            return
        if parsed.path == "/api/config":
            self.send_json(self.service.config_payload())
            return
        if parsed.path == "/api/state":
            query = parse_qs(parsed.query)
            index = int(query["index"][0]) if "index" in query else self.service.image_index
            self.send_json(self.service.set_index(index))
            return
        if parsed.path.startswith("/image/"):
            self.send_image(parsed.path)
            return
        self.send_error(404)

    def do_POST(self) -> None:
        parsed_path = urlparse(self.path).path
        if parsed_path not in ("/api/action", "/api/start"):
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8") if length else "{}"
        data = json.loads(body)
        response = self.service.start(data) if parsed_path == "/api/start" else self.service.action(data)
        self.send_json(response)
        if data.get("action") == "stop":
            threading.Thread(target=self.server.shutdown, daemon=True).start()

    def send_static(self, name: str) -> None:
        path = STATIC_DIR / name
        content = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def send_json(self, data: dict) -> None:
        content = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def send_image(self, path: str) -> None:
        index = int(path.rsplit("/", 1)[1])
        image_path = self.service.images[index]
        content = image_path.read_bytes()
        content_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, _format: str, *_args) -> None:
        return


def create_server(service: TrackLabelService) -> ThreadingHTTPServer:
    TrackLabelHandler.service = service
    for port in range(PORT, PORT + 20):
        try:
            return ThreadingHTTPServer((HOST, port), TrackLabelHandler)
        except OSError:
            continue
    raise OSError(f"No free port found between {PORT} and {PORT + 19}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = resolve_path(args.input_dir).resolve()
    service = TrackLabelService(input_dir)
    server = create_server(service)
    url = f"http://{server.server_address[0]}:{server.server_address[1]}"
    print(f"Track label UI: {url}")
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

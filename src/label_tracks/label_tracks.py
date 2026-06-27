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
INPUT_DIR = ROOT / "assets" / "track" / "input"
LABELS_FILE = ROOT / "assets" / "track" / "labels" / "tracks.json"
WEIGHTS = ROOT / "runs" / "detect" / "yolo26m_binary" / "weights" / "best.pt"

HOST = "127.0.0.1"
PORT = 8765
DEVICE = "mps"
CONFIDENCE = 0.25
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def image_sort_key(path: Path) -> tuple[int, str]:
    digits = "".join(char for char in path.stem if char.isdigit())
    return int(digits) if digits else 0, path.name


def load_images() -> list[Path]:
    return sorted(
        [path for path in INPUT_DIR.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS],
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


class TrackLabelService:
    def __init__(self) -> None:
        if not WEIGHTS.exists():
            raise FileNotFoundError(f"Weights not found: {WEIGHTS}")

        self.images = load_images()
        if not self.images:
            raise FileNotFoundError(f"No images found in {INPUT_DIR}")

        self.model = YOLO(str(WEIGHTS))
        self.tracks = load_tracks()
        self.active_track = []
        self.next_track_id = max((track["id"] for track in self.tracks), default=0) + 1
        self.image_index = 0
        self.detection_cache = {}
        self.status = "Click a vehicle, then Next. Use Save when the vehicle is no longer visible."
        self.lock = threading.Lock()

    def payload(self) -> dict:
        image = self.images[self.image_index]
        return {
            "index": self.image_index,
            "total": len(self.images),
            "image": {"name": image.name, "url": f"/image/{self.image_index}"},
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

        result = self.model(str(self.images[index]), device=DEVICE, conf=CONFIDENCE, verbose=False)[0]
        boxes = []
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            boxes.append({"bbox": [x1, y1, x2, y2], "confidence": float(box.conf[0])})

        self.detection_cache[index] = boxes
        return boxes

    def saved_boxes(self, index: int) -> list[dict]:
        image_name = self.images[index].name
        saved = []
        for track in self.tracks:
            for frame in track["frames"]:
                if frame["image"] == image_name:
                    saved.append({"track_id": track["id"], "bbox": frame["bbox"], "active": False})
        for frame in self.active_track:
            if frame["image"] == image_name:
                saved.append({"track_id": "active", "bbox": frame["bbox"], "active": True})
        return saved

    def action(self, data: dict) -> dict:
        with self.lock:
            action = data.get("action")
            if action == "next":
                self.next_image(data.get("box_index"))
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
        image_name = self.images[self.image_index].name
        self.active_track = [frame for frame in self.active_track if frame["image"] != image_name]
        self.active_track.append(
            {
                "image": image_name,
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
        if self.image_index < len(self.images) - 1:
            self.image_index += 1
        else:
            self.status = "End reached."

    def clear_current(self) -> None:
        image_name = self.images[self.image_index].name
        before = len(self.active_track)
        self.active_track = [frame for frame in self.active_track if frame["image"] != image_name]
        removed = before - len(self.active_track)
        self.status = f"Cleared {removed} active labels for current image." if removed else "Current selection cleared."

    def save_active_track(self) -> None:
        if not self.active_track:
            self.status = "No vehicle labels selected for this track."
            return

        self.tracks.append({"id": self.next_track_id, "frames": self.active_track})
        self.next_track_id += 1
        self.active_track = []
        save_tracks(self.tracks)
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
        if urlparse(self.path).path != "/api/action":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8") if length else "{}"
        data = json.loads(body)
        response = self.service.action(data)
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


def main() -> None:
    service = TrackLabelService()
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

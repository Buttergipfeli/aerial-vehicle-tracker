const canvas = document.getElementById("canvas");
const context = canvas.getContext("2d");
const stage = document.getElementById("stage");
const statusEl = document.getElementById("status");
const metaEl = document.getElementById("meta");
const image = new Image();
const startup = document.getElementById("startup");
const startupStatus = document.getElementById("startup-status");
const header = document.querySelector("header");
const main = document.querySelector("main");

let state = null;
let selectedBox = null;
let selectedTrackId = null;
let scale = 1;

async function requestJson(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return response.json();
}

async function loadConfig() {
  const config = await requestJson("/api/config");
  document.getElementById("startup-input-dir").value = config.input_dir;
  document.getElementById("startup-confidence").value = config.defaults.confidence;
  document.getElementById("startup-iou").value = config.defaults.iou_threshold;
  document.getElementById("startup-min-frames").value = config.defaults.min_frames;
  document.getElementById("startup-limit").value = config.defaults.limit_images || "";
  const labels = config.labels_exist ? `${config.track_count} existing tracks` : "no labels yet";
  startupStatus.textContent = `${config.images} images, ${config.sequences} sequences, ${labels}.`;
}

function numberValue(id) {
  const value = document.getElementById(id).value;
  return value === "" ? null : Number(value);
}

function startOptions(prelabel, overwriteLabels) {
  return {
    input_dir: document.getElementById("startup-input-dir").value,
    prelabel: prelabel,
    overwrite_labels: overwriteLabels,
    confidence: numberValue("startup-confidence"),
    iou_threshold: numberValue("startup-iou"),
    min_frames: numberValue("startup-min-frames"),
    limit_images: numberValue("startup-limit"),
  };
}

async function startApp(prelabel, overwriteLabels) {
  setStartupBusy(true);
  startupStatus.textContent = prelabel ? "Prelabeling..." : "Starting...";
  try {
    state = await requestJson("/api/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(startOptions(prelabel, overwriteLabels)),
    });
    selectedBox = activeSelection();
    selectedTrackId = null;
    await loadImage(state.image.url);
    startup.hidden = true;
    header.hidden = false;
    main.hidden = false;
    draw();
  } catch (error) {
    startupStatus.textContent = error.message;
    setStartupBusy(false);
  }
}

function setStartupBusy(busy) {
  document.getElementById("start-review").disabled = busy;
  document.getElementById("start-prelabel").disabled = busy;
  document.getElementById("start-overwrite").disabled = busy;
}

async function loadState(index = null) {
  const suffix = index === null ? "" : `?index=${index}`;
  state = await requestJson(`/api/state${suffix}`);
  selectedBox = activeSelection();
  selectedTrackId = null;
  await loadImage(state.image.url);
  draw();
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    image.onload = resolve;
    image.onerror = reject;
    image.src = `${src}?t=${Date.now()}`;
  });
}

function activeSelection() {
  const active = state.saved.find((item) => item.active);
  if (!active) {
    return null;
  }

  let bestIndex = null;
  let bestScore = 0;
  state.boxes.forEach((box, index) => {
    const score = iou(active.bbox, box.bbox);
    if (score > bestScore) {
      bestScore = score;
      bestIndex = index;
    }
  });

  return bestScore > 0.95 ? bestIndex : null;
}

function iou(a, b) {
  const left = Math.max(a[0], b[0]);
  const top = Math.max(a[1], b[1]);
  const right = Math.min(a[2], b[2]);
  const bottom = Math.min(a[3], b[3]);
  const intersection = Math.max(0, right - left) * Math.max(0, bottom - top);
  const areaA = Math.max(0, a[2] - a[0]) * Math.max(0, a[3] - a[1]);
  const areaB = Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
  const union = areaA + areaB - intersection;
  return union > 0 ? intersection / union : 0;
}

function draw() {
  const maxWidth = Math.max(320, stage.clientWidth - 24);
  const maxHeight = Math.max(240, stage.clientHeight - 24);
  scale = Math.min(maxWidth / image.naturalWidth, maxHeight / image.naturalHeight, 1);
  const width = Math.round(image.naturalWidth * scale);
  const height = Math.round(image.naturalHeight * scale);
  const ratio = window.devicePixelRatio || 1;

  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  context.setTransform(ratio * scale, 0, 0, ratio * scale, 0, 0);
  context.clearRect(0, 0, image.naturalWidth, image.naturalHeight);
  context.drawImage(image, 0, 0);

  state.boxes.forEach((box, index) => {
    drawBox(box.bbox, index === selectedBox ? "#ff4d4d" : "#33d17a", `${index} ${box.confidence.toFixed(2)}`, index === selectedBox ? 4 : 2);
  });
  state.saved.forEach((item) => {
    const selected = item.track_id === selectedTrackId;
    const color = selected ? "#ff4dff" : item.active ? "#ffd84d" : "#4db6ff";
    const label = item.active ? "active" : `track ${item.track_id}`;
    drawBox(item.bbox, color, label, selected ? 5 : 4);
  });

  metaEl.textContent = `${state.index + 1}/${state.total} | sequence ${state.sequence.index}/${state.sequence.total}: ${state.sequence.name} | ${state.image.path}`;
  statusEl.textContent = `${state.status} Detections: ${state.boxes.length}. Active labels: ${state.active_count}. Saved tracks: ${state.saved_count}.`;
}

function drawBox(bbox, color, label, width) {
  const [x1, y1, x2, y2] = bbox;
  context.strokeStyle = color;
  context.fillStyle = color;
  context.lineWidth = width / scale;
  context.strokeRect(x1, y1, x2 - x1, y2 - y1);
  context.font = `${14 / scale}px system-ui`;
  context.fillText(label, x1, Math.max(16 / scale, y1 - 6 / scale));
}

canvas.addEventListener("click", (event) => {
  const rect = canvas.getBoundingClientRect();
  const x = (event.clientX - rect.left) / scale;
  const y = (event.clientY - rect.top) / scale;
  selectedTrackId = pickSavedTrack(x, y);
  selectedBox = selectedTrackId === null ? pickBox(x, y) : null;
  draw();
});

function pickSavedTrack(x, y) {
  const containing = [];
  state.saved.forEach((item) => {
    if (item.active) {
      return;
    }
    const [x1, y1, x2, y2] = item.bbox;
    if (x >= x1 && x <= x2 && y >= y1 && y <= y2) {
      containing.push([(x2 - x1) * (y2 - y1), item.track_id]);
    }
  });
  return containing.length ? containing.sort((a, b) => a[0] - b[0])[0][1] : null;
}

function pickBox(x, y) {
  const containing = [];
  const nearest = [];
  state.boxes.forEach((box, index) => {
    const [x1, y1, x2, y2] = box.bbox;
    if (x >= x1 && x <= x2 && y >= y1 && y <= y2) {
      containing.push([(x2 - x1) * (y2 - y1), index]);
      return;
    }
    const dx = Math.max(x1 - x, 0, x - x2);
    const dy = Math.max(y1 - y, 0, y - y2);
    nearest.push([Math.sqrt(dx * dx + dy * dy) * scale, index]);
  });

  if (containing.length) {
    return containing.sort((a, b) => a[0] - b[0])[0][1];
  }

  nearest.sort((a, b) => a[0] - b[0]);
  return nearest.length && nearest[0][0] <= 48 ? nearest[0][1] : null;
}

async function action(name, extra = {}) {
  const body = JSON.stringify({ action: name, ...extra });
  state = await requestJson("/api/action", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body,
  });
  selectedBox = activeSelection();
  selectedTrackId = null;
  await loadImage(state.image.url);
  draw();
}

async function selectedTrackAction(name) {
  if (selectedTrackId === null) {
    statusEl.textContent = "Select a saved track first.";
    return;
  }
  await action(name, { track_id: selectedTrackId });
}

async function stopApp() {
  try {
    state = await requestJson("/api/action", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action: "stop" }),
    });
    statusEl.textContent = state.status;
  } finally {
    window.close();
    setTimeout(() => {
      statusEl.textContent = "Server stopped. You can close this tab.";
    }, 300);
  }
}

document.getElementById("previous").onclick = () => action("previous");
document.getElementById("start-review").onclick = () => startApp(false, false);
document.getElementById("start-prelabel").onclick = () => startApp(true, false);
document.getElementById("start-overwrite").onclick = () => startApp(true, true);
document.getElementById("skip").onclick = () => action("skip");
document.getElementById("next-sequence").onclick = () => action("next_sequence");
document.getElementById("clear").onclick = () => action("clear");
document.getElementById("delete-frame").onclick = () => selectedTrackAction("delete_frame");
document.getElementById("delete-track").onclick = () => selectedTrackAction("delete_track");
document.getElementById("save").onclick = () => action("save");
document.getElementById("stop").onclick = () => stopApp();
document.getElementById("next").onclick = () => {
  if (selectedBox === null) {
    statusEl.textContent = "Select a vehicle first, or use Skip.";
    return;
  }
  action("next", { box_index: selectedBox });
};

window.addEventListener("resize", () => {
  if (state) {
    draw();
  }
});

loadConfig().catch((error) => {
  startupStatus.textContent = error.message;
});

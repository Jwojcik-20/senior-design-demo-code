# file: lbph_face_recognition_dual_display.py
import argparse, json, os, time, uuid
import http.server
import socketserver
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import cv2.data as cvdata
import asyncio
from queue import Empty, Full, Queue

from unitree_webrtc_connect.webrtc_driver import UnitreeWebRTCConnection, WebRTCConnectionMethod
from unitree_webrtc_connect.constants import RTC_TOPIC, SPORT_CMD
from aiortc import MediaStreamTrack

# ---------- Paths ----------
DATASET_DIR = Path("dataset")           # dataset/<person>/*.jpg
MODEL_DIR   = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH  = MODEL_DIR / "lbph_face.yml"
LABELS_PATH = MODEL_DIR / "labels.json"

GO2_IP = os.environ.get("GO2_IP", "192.168.123.161")
ROBOT_HOST = f"unitree@{GO2_IP}"
GO2_SIT_ON_DETECT = os.environ.get("GO2_SIT_ON_DETECT", "0").lower() in {"1", "true", "yes", "on"}
GO2_DETECT_EVERY = max(1, int(os.environ.get("GO2_DETECT_EVERY", "2")))
GO2_SIT_HUMAN_FRAMES = max(1, int(os.environ.get("GO2_SIT_HUMAN_FRAMES", "3")))
GO2_SIT_MIN_FACE_RATIO = max(0.0, float(os.environ.get("GO2_SIT_MIN_FACE_RATIO", "0.22")))
GO2_SIT_MIN_BODY_RATIO = max(0.0, float(os.environ.get("GO2_SIT_MIN_BODY_RATIO", "0.58")))
GO2_SEARCH_LIGHT_ON_SIT = os.environ.get("GO2_SEARCH_LIGHT_ON_SIT", "0").lower() in {"1", "true", "yes", "on"}
GO2_SEARCH_LIGHT_BRIGHTNESS = max(0, min(10, int(os.environ.get("GO2_SEARCH_LIGHT_BRIGHTNESS", "10"))))
REMOTE_GLOB = "~/unitree_sdk2/build/bin/*.jpg"
LOCAL_DIR = Path("/tmp/go2_cam")
LOCAL_FILE = LOCAL_DIR / "latest.jpg"
PORT = 9000

HTML = f"""<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta http-equiv="Cache-Control" content="no-store" />
    <title>Go2 Camera</title>
    <style>
      body {{ margin: 0; background: #111; color: #ddd; font-family: Arial, sans-serif; }}
      .wrap {{ display: flex; flex-direction: column; align-items: center; gap: 8px; padding: 10px; }}
      img {{ max-width: 95vw; max-height: 85vh; border: 1px solid #333; }}
      .meta {{ font-size: 12px; opacity: 0.7; }}
    </style>
  </head>
  <body>
    <div class="wrap">
      <div class="meta">Auto-refreshing every second</div>
      <img id="frame" src="/latest.jpg" />
    </div>
    <script>
      const img = document.getElementById('frame');
      setInterval(() => {{
        img.src = "/latest.jpg?ts=" + Date.now();
      }}, 1000);
    </script>
  </body>
</html>
"""

# ---------- Face detector (Haar) ----------
CASCADE_PATH = os.path.join(cvdata.haarcascades, "haarcascade_frontalface_default.xml")
PROFILE_PATH = os.path.join(cvdata.haarcascades, "haarcascade_profileface.xml")
FACE_CASCADE = cv2.CascadeClassifier(CASCADE_PATH)
PROFILE_CASCADE = cv2.CascadeClassifier(PROFILE_PATH)
if FACE_CASCADE.empty() or PROFILE_CASCADE.empty():
    raise RuntimeError("Could not load Haar cascades. Check your OpenCV install.")

# ---------- UI helpers ----------
def put_label(img, text, org, scale=0.6, color=(255, 255, 255), bg=(0, 0, 0), thickness=1):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = org
    cv2.rectangle(img, (x, y - h - 6), (x + w + 6, y + 4), bg, -1)
    cv2.putText(img, text, (x + 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

def fit_to_square(img: np.ndarray, size: int = 480) -> np.ndarray:
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((size, size, 3), dtype=np.uint8)
    scale = min(size / w, size / h)
    resized = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))))
    out = np.zeros((size, size, 3), dtype=np.uint8)
    y_off = (size - resized.shape[0]) // 2
    x_off = (size - resized.shape[1]) // 2
    out[y_off:y_off + resized.shape[0], x_off:x_off + resized.shape[1]] = resized
    return out


def get_latest_frame(frame_queue: Queue) -> Optional[np.ndarray]:
    latest = None
    while not frame_queue.empty():
        latest = frame_queue.get()
    return latest


def replace_queued_frame(frame_queue: Queue, frame: np.ndarray):
    try:
        frame_queue.put_nowait(frame)
        return
    except Full:
        pass


def proximity_from_boxes(
    faces: List[Tuple[int, int, int, int]],
    body_boxes: List[Tuple[int, int, int, int]],
    frame_shape: Tuple[int, int, int],
) -> Tuple[bool, float, float]:
    frame_h = max(1, frame_shape[0])
    max_face_ratio = max((h / frame_h for (_, _, _, h) in faces), default=0.0)
    max_body_ratio = max((h / frame_h for (_, _, _, h) in body_boxes), default=0.0)
    close_enough = max_face_ratio >= GO2_SIT_MIN_FACE_RATIO or max_body_ratio >= GO2_SIT_MIN_BODY_RATIO
    return close_enough, max_face_ratio, max_body_ratio

    try:
        frame_queue.get_nowait()
    except Empty:
        pass

    try:
        frame_queue.put_nowait(frame)
    except Full:
        pass


async def send_sport_command(conn: UnitreeWebRTCConnection, api_id: int, parameter: Optional[dict] = None):
    payload = {"api_id": api_id}
    if parameter is not None:
        payload["parameter"] = parameter
    return await conn.datachannel.pub_sub.publish_request_new(
        RTC_TOPIC["SPORT_MOD"],
        payload,
    )


async def send_robot_sit(conn: UnitreeWebRTCConnection):
    return await send_sport_command(conn, SPORT_CMD["Sit"])


async def send_robot_stand_up(conn: UnitreeWebRTCConnection):
    return await send_sport_command(conn, SPORT_CMD["StandUp"])


async def set_vui_brightness(conn: UnitreeWebRTCConnection, brightness: int):
    brightness = max(0, min(10, int(brightness)))
    return await conn.datachannel.pub_sub.publish_request_new(
        RTC_TOPIC["VUI"],
        {
            "api_id": 1005,
            "parameter": {"brightness": brightness},
        },
    )


async def ensure_normal_motion_mode(conn: UnitreeWebRTCConnection):
    try:
        response = await conn.datachannel.pub_sub.publish_request_new(
            RTC_TOPIC["MOTION_SWITCHER"],
            {"api_id": 1001},
        )
        data = json.loads(response["data"]["data"])
        current_mode = data.get("name")
        print(f"[Motion] Current mode: {current_mode}")
        if current_mode != "normal":
            await conn.datachannel.pub_sub.publish_request_new(
                RTC_TOPIC["MOTION_SWITCHER"],
                {
                    "api_id": 1002,
                    "parameter": {"name": "normal"},
                },
            )
            print("[Motion] Switched to normal mode")
            await asyncio.sleep(2.0)
    except Exception as exc:
        print(f"[Motion] Could not verify/switch mode: {exc}")


CLAHE = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
VALID_ROLES = {"student", "staff"}


def adjust_lighting(gray: np.ndarray, mode: str = "clahe") -> np.ndarray:
    mode = (mode or "none").lower()
    if mode == "hist":
        return cv2.equalizeHist(gray)
    if mode == "clahe":
        return CLAHE.apply(gray)
    return gray


def estimate_face_rotation(gray_face: np.ndarray) -> Optional[Tuple[Tuple[float, float], Tuple[float, float], float]]:
    if gray_face.size == 0:
        return None
    edges = cv2.Canny(gray_face, 40, 140)
    points = cv2.findNonZero(edges)
    if points is None or len(points) < 5:
        return None
    return cv2.minAreaRect(points)


def draw_oriented_box(
    display: np.ndarray,
    rect: Tuple[Tuple[float, float], Tuple[float, float], float],
    offset: Tuple[int, int] = (0, 0),
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    box = cv2.boxPoints(rect)
    box += np.array(offset, dtype=np.float32)
    pts = np.int32(box)
    cv2.drawContours(display, [pts], 0, color, thickness)
    return pts


def annotate_face(
    display: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    gray_face: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
) -> Tuple[int, int]:
    rect = estimate_face_rotation(gray_face)
    if rect is None:
        cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
        return (x, max(22, y))
    box = draw_oriented_box(display, rect, (x, y), color)
    top_idx = int(np.argmin(box[:, 1]))
    x_label = max(0, box[top_idx][0])
    y_label = max(22, box[top_idx][1])
    return (x_label, y_label)


def augment_head_pose(img: np.ndarray, angles: Tuple[int, ...] = (-90, -70, -50, -30, -10, 0, 10, 30, 50, 70, 90)) -> List[np.ndarray]:
    """Return a small set of rotated variants so the recognizer sees tilted heads."""
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    variants = []
    for angle in angles:
        if angle == 0:
            variants.append(img)
            continue
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            img,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        variants.append(rotated)
    return variants

# ---------- Detection ----------
ROTATION_ANGLES = (-30.0, -15.0, 15.0, 30.0)
DISTANCE_SCALE_FACTORS = (1.2, 1.5, 2.2)
MIN_FACE_SIZE = 60
HUMAN_PERSISTENCE_FRAMES = 6
PERSON_DETECTION_SCALE = 0.6
BODY_WIDTH_MULTIPLIER = 1.25
BODY_HEIGHT_MULTIPLIER = 2.8
BODY_VERTICAL_BIAS = 0.12
FULLBODY_MIN_SIZE = 220
FULLBODY_SCALE = 1.1
FULLBODY_MIN_NEIGHBORS = 6
HOG_HIT_THRESHOLD = 0.2
HOG_MIN_CONFIDENCE = 0.45
BODY_MIN_HEIGHT_RATIO = 0.22
BODY_MAX_HEIGHT_RATIO = 0.98
BODY_MIN_ASPECT_RATIO = 0.22
BODY_MAX_ASPECT_RATIO = 0.75
BODY_MIN_AREA_RATIO = 0.03
BODY_SMOOTHING_ALPHA = 0.08
BODY_TRACK_IOU = 0.10
BODY_MISS_TTL = 10
HOG_PERSON_DETECTOR = cv2.HOGDescriptor()
HOG_PERSON_DETECTOR.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
BODY_DEDUPE_IOU = 0.85
FULLBODY_CASCADE_PATH = os.path.join(cvdata.haarcascades, "haarcascade_fullbody.xml")
FULLBODY_CASCADE = cv2.CascadeClassifier(FULLBODY_CASCADE_PATH)
if FULLBODY_CASCADE.empty():
    raise RuntimeError(f"Could not load full body cascade from {FULLBODY_CASCADE_PATH}")


def _make_min_size(size: int) -> Tuple[int, int]:
    clamped = max(1, int(size))
    return (clamped, clamped)


def _cascade_detect(gray: np.ndarray, min_size: Tuple[int, int]) -> List[Tuple[int, int, int, int]]:
    faces = list(FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=min_size))
    if faces:
        return faces
    profiles = list(PROFILE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=min_size))
    return profiles


def _rotate_gray(gray: np.ndarray, angle: float) -> Tuple[np.ndarray, np.ndarray]:
    h, w = gray.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated, M


def _map_rotated_box(
    det: Tuple[int, int, int, int],
    invM: np.ndarray,
    shape: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    x, y, w, h = det
    corners = np.array([[x, y], [x + w, y], [x, y + h], [x + w, y + h]], dtype=np.float32).reshape(-1, 1, 2)
    mapped = cv2.transform(corners, invM).reshape(-1, 2)
    min_x = int(np.clip(mapped[:, 0].min(), 0, shape[1] - 1))
    max_x = int(np.clip(mapped[:, 0].max(), 0, shape[1] - 1))
    min_y = int(np.clip(mapped[:, 1].min(), 0, shape[0] - 1))
    max_y = int(np.clip(mapped[:, 1].max(), 0, shape[0] - 1))
    width = max(1, max_x - min_x)
    height = max(1, max_y - min_y)
    return (min_x, min_y, width, height)


def detect_with_rotation(gray: np.ndarray, min_face_size: int) -> List[Tuple[int, int, int, int]]:
    min_size = _make_min_size(min_face_size)
    detections = _cascade_detect(gray, min_size)
    if detections:
        return detections
    for angle in ROTATION_ANGLES:
        rotated, M = _rotate_gray(gray, angle)
        rotated_dets = _cascade_detect(rotated, min_size)
        if rotated_dets:
            invM = cv2.invertAffineTransform(M)
            return [_map_rotated_box(det, invM, gray.shape[:2]) for det in rotated_dets]
    mirrored = cv2.flip(gray, 1)
    mirrored_profiles = PROFILE_CASCADE.detectMultiScale(
        mirrored, scaleFactor=1.1, minNeighbors=5, minSize=min_size
    )
    if len(mirrored_profiles) > 0:
        width = gray.shape[1]
        return [(width - x - w, y, w, h) for (x, y, w, h) in mirrored_profiles]
    return []


def _map_scaled_box(
    det: Tuple[int, int, int, int],
    scale: float,
    shape: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    inv_scale = 1.0 / scale
    x, y, w, h = det
    min_x = int(np.clip(round(x * inv_scale), 0, shape[1] - 1))
    min_y = int(np.clip(round(y * inv_scale), 0, shape[0] - 1))
    width = max(1, int(round(w * inv_scale)))
    height = max(1, int(round(h * inv_scale)))
    return (min_x, min_y, width, height)


def detect_faces(gray: np.ndarray, min_face_size: int = MIN_FACE_SIZE) -> List[Tuple[int, int, int, int]]:
    detections = detect_with_rotation(gray, min_face_size)
    if detections:
        return detections
    original_shape = gray.shape[:2]
    for scale in DISTANCE_SCALE_FACTORS:
        scaled = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        scaled_size = int(round(min_face_size * scale))
        far = detect_with_rotation(scaled, scaled_size)
        if far:
            return [_map_scaled_box(det, scale, original_shape) for det in far]
    return []


def _box_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    xi1 = max(ax, bx)
    yi1 = max(ay, by)
    xi2 = min(ax + aw, bx + bw)
    yi2 = min(ay + ah, by + bh)
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height
    if inter_area == 0:
        return 0.0
    area_a = aw * ah
    area_b = bw * bh
    union = area_a + area_b - inter_area
    if union == 0:
        return 0.0
    return inter_area / union


def expand_body_box(
    box: Tuple[int, int, int, int],
    frame_shape: Tuple[int, int],
    width_mul: float = BODY_WIDTH_MULTIPLIER,
    height_mul: float = BODY_HEIGHT_MULTIPLIER,
    vertical_bias: float = BODY_VERTICAL_BIAS,
) -> Tuple[int, int, int, int]:
    fx, fy, fw, fh = box
    frame_h, frame_w = frame_shape
    new_w = min(frame_w, max(1, int(round(fw * width_mul))))
    new_h = min(frame_h, max(1, int(round(fh * height_mul))))
    new_x = max(0, min(frame_w - new_w, int(round(fx - (new_w - fw) / 2))))
    new_y = max(0, min(frame_h - new_h, int(round(fy - fh * vertical_bias))))
    return (new_x, new_y, new_w, new_h)


def is_plausible_body_box(box: Tuple[int, int, int, int], frame_shape: Tuple[int, int]) -> bool:
    _, _, w, h = box
    frame_h, frame_w = frame_shape
    if w <= 0 or h <= 0:
        return False
    height_ratio = h / max(1, frame_h)
    width_ratio = w / max(1, frame_w)
    area_ratio = (w * h) / max(1, frame_h * frame_w)
    aspect_ratio = w / max(1, h)
    if height_ratio < BODY_MIN_HEIGHT_RATIO or height_ratio > BODY_MAX_HEIGHT_RATIO:
        return False
    if width_ratio > BODY_MAX_ASPECT_RATIO:
        return False
    if area_ratio < BODY_MIN_AREA_RATIO:
        return False
    if aspect_ratio < BODY_MIN_ASPECT_RATIO or aspect_ratio > BODY_MAX_ASPECT_RATIO:
        return False
    return True


def detect_people(frame: np.ndarray, scale: float = PERSON_DETECTION_SCALE) -> List[Tuple[int, int, int, int]]:
    if scale <= 0 or scale > 1:
        scale = 1.0
    resized = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    rects, weights = HOG_PERSON_DETECTOR.detectMultiScale(
        resized,
        hitThreshold=HOG_HIT_THRESHOLD,
        winStride=(8, 8),
        padding=(8, 8),
        scale=1.05,
    )
    inv_scale = 1.0 / scale
    boxes = []
    frame_shape = frame.shape[:2]
    for (x, y, w, h), weight in zip(rects, weights):
        if float(weight) < HOG_MIN_CONFIDENCE:
            continue
        base_box = (
            int(round(x * inv_scale)),
            int(round(y * inv_scale)),
            int(round(w * inv_scale)),
            int(round(h * inv_scale)),
        )
        expanded = expand_body_box(base_box, frame_shape)
        if is_plausible_body_box(expanded, frame_shape):
            boxes.append(expanded)
    return boxes


def dedupe_body_boxes(boxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
    unique: List[Tuple[int, int, int, int]] = []
    for box in boxes:
        if not any(_box_iou(box, existing) > BODY_DEDUPE_IOU for existing in unique):
            unique.append(box)
    return unique


def smooth_body_box(
    previous: Tuple[int, int, int, int],
    current: Tuple[int, int, int, int],
    alpha: float = BODY_SMOOTHING_ALPHA,
) -> Tuple[int, int, int, int]:
    px, py, pw, ph = previous
    cx, cy, cw, ch = current
    beta = max(0.0, min(1.0, alpha))
    return (
        int(round(px * (1.0 - beta) + cx * beta)),
        int(round(py * (1.0 - beta) + cy * beta)),
        int(round(pw * (1.0 - beta) + cw * beta)),
        int(round(ph * (1.0 - beta) + ch * beta)),
    )


def stabilize_body_boxes(
    current_boxes: List[Tuple[int, int, int, int]],
    tracked_boxes: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    updated_tracks: List[Dict[str, object]] = []
    used_tracks = set()
    for box in current_boxes:
        best_index = -1
        best_iou = 0.0
        for index, track in enumerate(tracked_boxes):
            if index in used_tracks:
                continue
            iou = _box_iou(box, track["box"])
            if iou > best_iou:
                best_iou = iou
                best_index = index
        if best_index >= 0 and best_iou >= BODY_TRACK_IOU:
            track = tracked_boxes[best_index]
            smoothed = smooth_body_box(track["box"], box)
            updated_tracks.append({"box": smoothed, "misses": 0})
            used_tracks.add(best_index)
        else:
            updated_tracks.append({"box": box, "misses": 0})
    for index, track in enumerate(tracked_boxes):
        if index in used_tracks:
            continue
        misses = int(track["misses"]) + 1
        if misses <= BODY_MISS_TTL:
            updated_tracks.append({"box": track["box"], "misses": misses})
    return updated_tracks


def detect_full_body(gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
    min_size = _make_min_size(FULLBODY_MIN_SIZE)
    bodies = list(
        FULLBODY_CASCADE.detectMultiScale(
            gray,
            scaleFactor=FULLBODY_SCALE,
            minNeighbors=FULLBODY_MIN_NEIGHBORS,
            minSize=min_size,
        )
    )
    frame_shape = gray.shape[:2]
    filtered: List[Tuple[int, int, int, int]] = []
    for box in bodies:
        expanded = expand_body_box(box, frame_shape)
        if is_plausible_body_box(expanded, frame_shape):
            filtered.append(expanded)
    return filtered


def infer_body_boxes_from_faces(faces: List[Tuple[int, int, int, int]], frame_shape: Tuple[int, int]) -> List[Tuple[int, int, int, int]]:
    boxes: List[Tuple[int, int, int, int]] = []
    frame_h, frame_w = frame_shape
    for (x, y, w, h) in faces:
        target_w = max(1, int(round(w * 1.25)))
        target_h = max(1, int(round(h * 2.6)))
        target_w = min(target_w, frame_w)
        target_h = min(target_h, frame_h - max(0, y - int(h * 0.1)))
        body_x = max(0, min(frame_w - target_w, x - (target_w - w) // 2))
        body_y = max(0, y - int(h * 0.1))
        boxes.append(expand_body_box((body_x, body_y, target_w, target_h), frame_shape))
    return boxes


def normalize_role(role: str) -> str:
    normalized = (role or "").strip().lower()
    if normalized not in VALID_ROLES:
        raise ValueError(f"Role must be one of: {', '.join(sorted(VALID_ROLES))}")
    return normalized


def format_role(role: str) -> str:
    return (role or "unknown").strip().title()


def format_identity(name: str, role: str) -> str:
    return f"{name} ({format_role(role)})"


def decode_identity(value) -> Dict[str, str]:
    if isinstance(value, dict):
        return {
            "name": str(value.get("name", "Unknown")),
            "role": format_role(str(value.get("role", "unknown"))),
        }
    return {"name": str(value), "role": "Unknown"}

# ---------- Collect ----------
def collect(
    role: str,
    label: str,
    cam_index: int = 0,
    shots: int = 50,
    lighting_mode: str = "clahe",
    min_face_size: int = MIN_FACE_SIZE,
):
    role = normalize_role(role)
    save_dir = DATASET_DIR / role / label
    save_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {cam_index}")
    count, last_save = 0, 0.0
    print(f"[Collect] Capturing {shots} images for '{label}' ({format_role(role)}). Press 'q' to stop early.")
    while count < shots:
        ok, frame = cap.read()
        if not ok: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        proc = adjust_lighting(gray, lighting_mode)
        faces = detect_faces(proc, min_face_size=min_face_size)
        display = frame.copy()
        for (x, y, w, h) in faces:
            roi = proc[y:y+h, x:x+w]
            label_org = annotate_face(display, x, y, w, h, roi)
            t = time.time()
            if t - last_save > 0.2:
                fname = save_dir / f"{uuid.uuid4().hex}.jpg"
                cv2.imwrite(str(fname), roi)
                last_save = t
                count += 1
                put_label(display, f"Saved {count}/{shots}", label_org)
        cv2.imshow("Collect", display)
        if (cv2.waitKey(1) & 0xFF) in (27, ord("q")): break
    cap.release(); cv2.destroyAllWindows()
    print(f"[Collect] Saved {count} images in {save_dir}")

# ---------- Train ----------
def load_dataset() -> Tuple[List[np.ndarray], List[int], Dict[int, Dict[str, str]]]:
    images, labels, id2identity = [], [], {}
    next_id = 0
    for role_dir in sorted(DATASET_DIR.glob("*")):
        if not role_dir.is_dir():
            continue
        role = role_dir.name.strip().lower()
        if role not in VALID_ROLES:
            continue
        for person_dir in sorted(role_dir.glob("*")):
            if not person_dir.is_dir():
                continue
            name = person_dir.name
            pid = next_id
            next_id += 1
            id2identity[pid] = {"name": name, "role": format_role(role)}
            for img_path in person_dir.glob("*.jpg"):
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                for variant in augment_head_pose(img):
                    images.append(variant)
                    labels.append(pid)
    return images, labels, id2identity

def train():
    images, labels, id2identity = load_dataset()
    if not images:
        raise RuntimeError("No training images found. Use dataset/<student|staff>/<name> or run collect with --role first.")
    recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
    recognizer.train(images, np.array(labels))
    recognizer.save(str(MODEL_PATH))
    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(id2identity, f, indent=2)
    print(f"[Train] Saved model to {MODEL_PATH}")
    print(f"[Train] Saved labels to {LABELS_PATH}")

# ---------- Run (recognition) ----------
def run(
    cam_index: int = 0,
    conf_threshold: float = 70.0,
    lighting_mode: str = "clahe",
    min_face_size: int = MIN_FACE_SIZE,
):
    if not MODEL_PATH.is_file() or not LABELS_PATH.is_file():
        raise RuntimeError("Model not trained. Run with --train after collecting images.")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(MODEL_PATH))
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        id2identity = {int(k): decode_identity(v) for k, v in json.load(f).items()}

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {cam_index}")

    cv2.namedWindow("All Faces", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Top Face", cv2.WINDOW_NORMAL)

    fps, prev = 0.0, time.time()
    human_persistence = 0
    tracked_body_boxes: List[Dict[str, object]] = []

    while True:
        ok, frame = cap.read()
        if not ok: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        proc = adjust_lighting(gray, lighting_mode)
        faces = detect_faces(proc, min_face_size=min_face_size)
        body_candidates = detect_full_body(gray)
        body_candidates.extend(detect_people(frame))
        body_boxes_real = dedupe_body_boxes(body_candidates)
        tracked_body_boxes = stabilize_body_boxes(body_boxes_real, tracked_body_boxes)
        body_boxes = [track["box"] for track in tracked_body_boxes]
        if not body_boxes:
            body_boxes = infer_body_boxes_from_faces(faces, frame.shape[:2])
        if body_boxes_real:
            human_persistence = min(human_persistence + 1, HUMAN_PERSISTENCE_FRAMES)
        else:
            human_persistence = max(0, human_persistence - 1)
        scene_is_human = human_persistence > 0

        now = time.time()
        dt = now - prev
        prev = now
        fps = 0.9*fps + 0.1*(1.0/dt if dt>0 else 0.0)

        display = frame.copy()
        top_face, top_score, top_name = None, -1e9, "?"
        top_nature = None

        for (px, py, pw, ph) in body_boxes:
            cv2.rectangle(display, (px, py), (px + pw, py + ph), (0, 150, 255), 2)
            put_label(display, "Body", (px, py + ph + 18), scale=0.5, bg=(0, 120, 200))

        face_known_present = False
        for (x, y, w, h) in faces:
            roi = proc[y:y+h, x:x+w]
            label_id, distance = recognizer.predict(roi)
            identity = id2identity.get(label_id, {"name": "Unknown", "role": "Unknown"})
            conf = max(0.0, min(100.0, 100.0 - distance))
            if conf < conf_threshold:
                display_name = "Unknown"
            else:
                face_known_present = True
                display_name = format_identity(identity["name"], identity["role"])

            label_org = annotate_face(display, x, y, w, h, roi)
            face_box = (x, y, w, h)
            human_overlap = any(_box_iou(face_box, person) >= 0.05 for person in body_boxes)
            nature = "Human" if human_overlap or scene_is_human or display_name != "Unknown" else "Non-human"
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 1)
            put_label(display, f"{display_name} ({conf:.1f}%) {nature}", label_org)

            if conf > top_score:
                top_score, top_name = conf, display_name
                top_face = frame[y:y+h, x:x+w].copy()
                top_nature = nature

        scene_status = "Human" if scene_is_human or face_known_present else "Non-human"
        put_label(display, f"Scene: {scene_status}", (10, 60), scale=0.6, bg=(40,40,40))
        if top_face is not None:
            top_view = fit_to_square(top_face, 480)
            top_label = f"Top: {top_name} ({top_score:.1f}%)"
            if top_nature:
                top_label = f"{top_label} [{top_nature}]"
            put_label(top_view, top_label, (10, 30), scale=0.8, bg=(40,40,40))
        else:
            top_view = np.zeros((480, 480, 3), dtype=np.uint8)
            put_label(top_view, "No face", (10, 30), scale=0.8, bg=(40,40,40))

        put_label(display, f"FPS: {fps:.1f}", (10, 30), scale=0.7, bg=(40,40,40))
        cv2.imshow("All Faces", display)
        cv2.imshow("Top Face", top_view)

        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord('q')): break

    cap.release(); cv2.destroyAllWindows()

# ---------- Run (WebRTC camera) ----------
def run_webrtc(
    conf_threshold: float = 70.0,
    lighting_mode: str = "clahe",
    min_face_size: int = MIN_FACE_SIZE,
):
    if not MODEL_PATH.is_file() or not LABELS_PATH.is_file():
        raise RuntimeError("Model not trained. Run with train after collecting images.")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(MODEL_PATH))
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        id2identity = {int(k): decode_identity(v) for k, v in json.load(f).items()}

    frame_queue: Queue = Queue(maxsize=1)

    conn = UnitreeWebRTCConnection(
        WebRTCConnectionMethod.LocalSTA,
        ip=GO2_IP,
    )

    async def recv_camera_stream(track: MediaStreamTrack):
        while True:
            frame = await track.recv()
            img = frame.to_ndarray(format="bgr24")
            replace_queued_frame(frame_queue, img)

    def run_asyncio_loop(loop):
        asyncio.set_event_loop(loop)
        async def setup():
            await conn.connect()
            await ensure_normal_motion_mode(conn)
            conn.video.switchVideoChannel(True)
            conn.video.add_track_callback(recv_camera_stream)
        loop.run_until_complete(setup())
        loop.run_forever()

    loop = asyncio.new_event_loop()
    asyncio_thread = threading.Thread(target=run_asyncio_loop, args=(loop,), daemon=True)
    asyncio_thread.start()

    cv2.namedWindow("All Faces", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Top Face", cv2.WINDOW_NORMAL)
    fps, prev = 0.0, time.time()
    human_persistence = 0
    sit_triggered = False
    stand_triggered = False
    light_on = False
    sit_future = None
    light_on_future = None
    stand_future = None
    light_off_future = None
    frame_index = 0
    body_boxes: List[Tuple[int, int, int, int]] = []
    tracked_body_boxes: List[Dict[str, object]] = []
    faces: List[Tuple[int, int, int, int]] = []
    scene_is_human = False
    close_enough = False
    max_face_ratio = 0.0
    max_body_ratio = 0.0

    try:
        while True:
            try:
                frame = get_latest_frame(frame_queue)
                if frame is None:
                    time.sleep(0.005)
                    continue
                frame_index += 1
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                proc = adjust_lighting(gray, lighting_mode)
                if frame_index % GO2_DETECT_EVERY == 1:
                    faces = detect_faces(proc, min_face_size=min_face_size)
                    body_candidates = detect_full_body(gray)
                    body_candidates.extend(detect_people(frame))
                    body_boxes_real = dedupe_body_boxes(body_candidates)
                    tracked_body_boxes = stabilize_body_boxes(body_boxes_real, tracked_body_boxes)
                    body_boxes = [track["box"] for track in tracked_body_boxes]
                    if not body_boxes:
                        body_boxes = infer_body_boxes_from_faces(faces, frame.shape[:2])
                    if body_boxes_real:
                        human_persistence = min(human_persistence + 1, HUMAN_PERSISTENCE_FRAMES)
                    else:
                        human_persistence = max(0, human_persistence - 1)
                    scene_is_human = human_persistence > 0
                    close_enough, max_face_ratio, max_body_ratio = proximity_from_boxes(
                        faces,
                        body_boxes,
                        frame.shape,
                    )

                now = time.time()
                dt = now - prev
                prev = now
                fps = 0.9*fps + 0.1*(1.0/dt if dt>0 else 0.0)

                display = frame.copy()
                top_face, top_score, top_name = None, -1e9, "?"
                top_nature = None

                for (px, py, pw, ph) in body_boxes:
                    cv2.rectangle(display, (px, py), (px + pw, py + ph), (0, 150, 255), 2)
                    put_label(display, "Body", (px, py + ph + 18), scale=0.5, bg=(0, 120, 200))

                face_known_present = False
                recognized_names: List[str] = []
                for (x, y, w, h) in faces:
                    roi = proc[y:y+h, x:x+w]
                    label_id, distance = recognizer.predict(roi)
                    identity = id2identity.get(label_id, {"name": "Unknown", "role": "Unknown"})
                    conf = max(0.0, min(100.0, 100.0 - distance))
                    if conf < conf_threshold:
                        display_name = "Unknown"
                    else:
                        face_known_present = True
                        display_name = format_identity(identity["name"], identity["role"])
                        recognized_names.append(identity["name"])

                    label_org = annotate_face(display, x, y, w, h, roi)
                    face_box = (x, y, w, h)
                    human_overlap = any(_box_iou(face_box, person) >= 0.05 for person in body_boxes)
                    nature = "Human" if human_overlap or scene_is_human or display_name != "Unknown" else "Non-human"
                    cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 1)
                    put_label(display, f"{display_name} ({conf:.1f}%) {nature}", label_org)

                    if conf > top_score:
                        top_score, top_name = conf, display_name
                        top_face = frame[y:y+h, x:x+w].copy()
                        top_nature = nature

                if sit_future is not None and sit_future.done():
                    try:
                        response = sit_future.result()
                        print(
                            f"[Detect] Stable close human detected, sent Sit: {response} "
                            f"(face_ratio={max_face_ratio:.2f}, body_ratio={max_body_ratio:.2f})"
                        )
                        sit_triggered = True
                    except Exception as exc:
                        print(f"[Detect] Failed to send Sit: {exc}")
                    sit_future = None

                if light_on_future is not None and light_on_future.done():
                    try:
                        light_response = light_on_future.result()
                        print(f"[Detect] Search light on: {light_response}")
                        light_on = True
                    except Exception as exc:
                        print(f"[Detect] Failed to turn on search light: {exc}")
                    light_on_future = None

                if stand_future is not None and stand_future.done():
                    try:
                        response = stand_future.result()
                        trigger_name = recognized_names[0] if recognized_names else "known ID"
                        print(f"[Detect] Recognized {trigger_name}, sent StandUp: {response}")
                        stand_triggered = True
                    except Exception as exc:
                        print(f"[Detect] Failed to send StandUp: {exc}")
                    stand_future = None

                if light_off_future is not None and light_off_future.done():
                    try:
                        light_response = light_off_future.result()
                        print(f"[Detect] Search light off: {light_response}")
                        light_on = False
                    except Exception as exc:
                        print(f"[Detect] Failed to turn off search light: {exc}")
                    light_off_future = None

                if (
                    GO2_SIT_ON_DETECT
                    and scene_is_human
                    and close_enough
                    and human_persistence >= GO2_SIT_HUMAN_FRAMES
                    and not sit_triggered
                    and sit_future is None
                ):
                    try:
                        sit_future = asyncio.run_coroutine_threadsafe(send_robot_sit(conn), loop)
                        if GO2_SEARCH_LIGHT_ON_SIT and not light_on and light_on_future is None:
                            light_on_future = asyncio.run_coroutine_threadsafe(
                                set_vui_brightness(conn, GO2_SEARCH_LIGHT_BRIGHTNESS), loop
                            )
                    except Exception as exc:
                        print(f"[Detect] Failed to send Sit: {exc}")

                if sit_triggered and not stand_triggered and recognized_names and stand_future is None:
                    try:
                        stand_future = asyncio.run_coroutine_threadsafe(send_robot_stand_up(conn), loop)
                        if light_on and light_off_future is None:
                            light_off_future = asyncio.run_coroutine_threadsafe(set_vui_brightness(conn, 0), loop)
                    except Exception as exc:
                        print(f"[Detect] Failed to send StandUp: {exc}")

                if top_face is not None:
                    top_view = fit_to_square(top_face, 480)
                else:
                    top_view = np.zeros((480, 480, 3), dtype=np.uint8)

                put_label(display, f"FPS: {fps:.1f}", (10, 30), scale=0.7, bg=(40,40,40))
                cv2.imshow("All Faces", display)
                cv2.imshow("Top Face", top_view)

                k = cv2.waitKey(1) & 0xFF
                if k in (27, ord('q')):
                    break
            except Exception as exc:
                print(f"[Run] Frame processing error: {exc}")
                time.sleep(0.05)
                continue
    finally:
        cv2.destroyAllWindows()
        loop.call_soon_threadsafe(loop.stop)
        asyncio_thread.join()

# ---------- Collect (WebRTC camera) ----------
def collect_webrtc(
    role: str,
    label: str,
    shots: int = 60,
    lighting_mode: str = "clahe",
    min_face_size: int = MIN_FACE_SIZE,
):
    role = normalize_role(role)
    save_dir = DATASET_DIR / role / label
    save_dir.mkdir(parents=True, exist_ok=True)

    frame_queue: Queue = Queue(maxsize=1)

    conn = UnitreeWebRTCConnection(
        WebRTCConnectionMethod.LocalSTA,
        ip=GO2_IP,
    )

    async def recv_camera_stream(track: MediaStreamTrack):
        while True:
            frame = await track.recv()
            img = frame.to_ndarray(format="bgr24")
            replace_queued_frame(frame_queue, img)

    def run_asyncio_loop(loop):
        asyncio.set_event_loop(loop)
        async def setup():
            await conn.connect()
            await ensure_normal_motion_mode(conn)
            conn.video.switchVideoChannel(True)
            conn.video.add_track_callback(recv_camera_stream)
        loop.run_until_complete(setup())
        loop.run_forever()

    loop = asyncio.new_event_loop()
    asyncio_thread = threading.Thread(target=run_asyncio_loop, args=(loop,), daemon=True)
    asyncio_thread.start()

    count, last_save = 0, 0.0
    print(f"[Collect/WebRTC] Capturing {shots} images for '{label}' ({format_role(role)}). Press 'q' to stop early.")

    try:
        while count < shots:
            frame = get_latest_frame(frame_queue)
            if frame is None:
                time.sleep(0.005)
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            proc = adjust_lighting(gray, lighting_mode)
            faces = detect_faces(proc, min_face_size=min_face_size)
            display = frame.copy()
            for (x, y, w, h) in faces:
                roi = proc[y:y+h, x:x+w]
                label_org = annotate_face(display, x, y, w, h, roi)
                t = time.time()
                if t - last_save > 0.2:
                    fname = save_dir / f"{uuid.uuid4().hex}.jpg"
                    cv2.imwrite(str(fname), roi)
                    last_save = t
                    count += 1
                    put_label(display, f"Saved {count}/{shots}", label_org)
            cv2.imshow("Collect (WebRTC)", display)
            if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                break
    finally:
        cv2.destroyAllWindows()
        loop.call_soon_threadsafe(loop.stop)
        asyncio_thread.join()
    print(f"[Collect/WebRTC] Saved {count} images in {save_dir}")

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/latest.jpg"):
            if LOCAL_FILE.exists():
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                with LOCAL_FILE.open("rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.end_headers()
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML.encode("utf-8"))


def pull_latest_loop():
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    last_remote = ""
    while True:
        try:
            latest = subprocess.run(
                [
                    "ssh",
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "ConnectTimeout=3",
                    ROBOT_HOST,
                    f"ls -t {REMOTE_GLOB} 2>/dev/null | head -n 1",
                ],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()
            if latest and latest != last_remote:
                local_name = LOCAL_DIR / os.path.basename(latest)
                subprocess.run(
                    ["scp", "-q", f"{ROBOT_HOST}:{latest}", str(local_name)],
                    check=False,
                )
                try:
                    if LOCAL_FILE.exists() or LOCAL_FILE.is_symlink():
                        LOCAL_FILE.unlink()
                    LOCAL_FILE.symlink_to(local_name)
                except Exception:
                    shutil.copy2(local_name, LOCAL_FILE)
                last_remote = latest
        except Exception:
            pass
        time.sleep(0.2)


def serve_remote(port: int = PORT, host: str = ""):
    t = threading.Thread(target=pull_latest_loop, daemon=True)
    t.start()
    with socketserver.TCPServer((host, port), Handler) as httpd:
        print(f"Serving Go2 camera preview on http://{host or 'localhost'}:{port}")
        httpd.serve_forever()

# ---------- Simple Remote Viewer ----------
def view_remote():
    t = threading.Thread(target=pull_latest_loop, daemon=True)
    t.start()
    cv2.namedWindow("Go2 Remote", cv2.WINDOW_NORMAL)
    while True:
        if LOCAL_FILE.exists():
            frame = cv2.imread(str(LOCAL_FILE))
            if frame is not None:
                cv2.imshow("Go2 Remote", frame)
        if (cv2.waitKey(1) & 0xFF) in (27, ord('q')):
            break
        time.sleep(0.01)
    cv2.destroyAllWindows()

# ---------- CLI ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="LBPH Face Recognition (collect/train/run) with dual display.")
    sub = ap.add_subparsers(dest="cmd")

    s1 = sub.add_parser("collect", help="Collect images for one person")
    s1.add_argument("--role", required=True, choices=sorted(VALID_ROLES), help="Person role")
    s1.add_argument("--label", required=True, help="Person's name (folder name)")
    s1.add_argument("--shots", type=int, default=60)
    s1.add_argument("--cam", type=int, default=0)
    s1.add_argument("--lighting", choices=["none", "hist", "clahe"], default="clahe")
    s1.add_argument("--min-face-size", type=int, default=MIN_FACE_SIZE, help="Smallest face size (pixels) to detect")

    s2 = sub.add_parser("train", help="Train LBPH model on dataset folders")

    s3 = sub.add_parser("run", help="Run live recognition")
    s3.add_argument("--cam", type=int, default=0)
    s3.add_argument("--threshold", type=float, default=55.0, help="Confidence cutoff for 'Unknown' (0-100)")
    s3.add_argument("--lighting", choices=["none", "hist", "clahe"], default="clahe")
    s3.add_argument("--min-face-size", type=int, default=MIN_FACE_SIZE, help="Smallest face size (pixels) to detect")

    s4 = sub.add_parser("serve-remote", help="Serve Go2 robot camera preview via HTTP")
    s4.add_argument("--port", type=int, default=PORT)
    s4.add_argument("--host", default="")

    s5 = sub.add_parser("view-remote", help="View Go2 robot camera in an OpenCV window")

    s6 = sub.add_parser("run-webrtc", help="Run live recognition using Go2 WebRTC camera")
    s6.add_argument("--threshold", type=float, default=55.0, help="Confidence cutoff for 'Unknown' (0-100)")
    s6.add_argument("--lighting", choices=["none", "hist", "clahe"], default="clahe")
    s6.add_argument("--min-face-size", type=int, default=MIN_FACE_SIZE, help="Smallest face size (pixels) to detect")

    s7 = sub.add_parser("collect-webrtc", help="Collect images using Go2 WebRTC camera")
    s7.add_argument("--role", required=True, choices=sorted(VALID_ROLES), help="Person role")
    s7.add_argument("--label", required=True, help="Person's name (folder name)")
    s7.add_argument("--shots", type=int, default=60)
    s7.add_argument("--lighting", choices=["none", "hist", "clahe"], default="clahe")
    s7.add_argument("--min-face-size", type=int, default=MIN_FACE_SIZE, help="Smallest face size (pixels) to detect")

    args = ap.parse_args()
    if args.cmd == "collect":
        collect(
            args.role,
            args.label,
            cam_index=args.cam,
            shots=args.shots,
            lighting_mode=args.lighting,
            min_face_size=args.min_face_size,
        )
    elif args.cmd == "train":
        train()
    elif args.cmd == "run":
        run(
            cam_index=args.cam,
            conf_threshold=args.threshold,
            lighting_mode=args.lighting,
            min_face_size=args.min_face_size,
        )
    elif args.cmd == "serve-remote":
        serve_remote(port=args.port, host=args.host)
    elif args.cmd == "view-remote":
        view_remote()
    elif args.cmd == "run-webrtc":
        run_webrtc(
            conf_threshold=args.threshold,
            lighting_mode=args.lighting,
            min_face_size=args.min_face_size,
        )
    elif args.cmd == "collect-webrtc":
        collect_webrtc(
            role=args.role,
            label=args.label,
            shots=args.shots,
            lighting_mode=args.lighting,
            min_face_size=args.min_face_size,
        )
    else:
        ap.print_help()

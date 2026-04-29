import cv2
import numpy as np
import os
import json
import mimetypes
import smtplib
import subprocess
import time
import wave
from email.message import EmailMessage
from pathlib import Path
from typing import List, Tuple

import asyncio
import logging
import threading
from queue import Empty, Full, Queue
from unitree_webrtc_connect.webrtc_driver import UnitreeWebRTCConnection
from unitree_webrtc_connect.constants import RTC_TOPIC, SPORT_CMD
from aiortc import MediaStreamTrack
from go2_connection import GO2_IP, connect_best_go2, patch_unitree_local_signaling, start_go2_video_stream

try:
    import pyaudio
except Exception:
    pyaudio = None

logging.basicConfig(level=logging.FATAL)
patch_unitree_local_signaling()

BASE_DIR = Path(__file__).resolve().parent
FACE_BACKEND = os.getenv("FACE_BACKEND", "embedding").strip().lower()
MODEL_PATH = os.getenv("FACE_MODEL_PATH", str(BASE_DIR / "models" / "lbph_face.yml"))
LABELS_PATH = os.getenv("FACE_LABELS_PATH", str(BASE_DIR / "models" / "labels.json"))
EMBEDDINGS_PATH = os.getenv("FACE_EMBEDDINGS_PATH", str(BASE_DIR / "models" / "opencv_face_embeddings.npz"))
EMBEDDING_LABELS_PATH = os.getenv("FACE_EMBEDDING_LABELS_PATH", str(BASE_DIR / "models" / "opencv_face_labels.json"))
EMAIL_CONFIG_PATH = Path(os.getenv("EMAIL_CONFIG_PATH", str(BASE_DIR / "email_config.json")))
ALERT_CLIP_DIR = Path(os.getenv("ALERT_CLIP_DIR", str(BASE_DIR / "alert_clips")))
ALERT_CLIP_SECONDS = float(os.getenv("ALERT_CLIP_SECONDS", "5"))
ALERT_CLIP_FPS = float(os.getenv("ALERT_CLIP_FPS", "10"))
ALERT_CLIP_MAX_WIDTH = int(os.getenv("ALERT_CLIP_MAX_WIDTH", "960"))
ALERT_CLIP_INCLUDE_AUDIO = os.getenv("ALERT_CLIP_INCLUDE_AUDIO", "1").lower() in {"1", "true", "yes"}
FFMPEG_BIN = os.getenv("FFMPEG_BIN", "/opt/homebrew/bin/ffmpeg")
CONF_THRESHOLD = float(os.getenv("FACE_CONF_THRESHOLD", "55"))
DETECT_FPS = max(1.0, float(os.getenv("DETECT_FPS", "12")))
EMAIL_SENT = set()
EVENT_LOG = []
LOGGED_IDENTITIES = set()
LOGIN_USER = os.getenv("LIVEINTERFACE_USER", "demo")
LOGIN_PASS = os.getenv("LIVEINTERFACE_PASS", "demo")
LOGO_PATH = BASE_DIR / "images" / "Logo.png"
SMALL_LOGO_PATH = BASE_DIR / "images" / "logononame.png"
VALID_ROLES = {"student", "staff", "visitor", "faculty", "guest"}
GO2_AUDIO_ENABLED = os.getenv("GO2_AUDIO_ENABLED", "1").lower() in {"1", "true", "yes"}
GO2_AUDIO_SAMPLERATE = int(os.getenv("GO2_AUDIO_SAMPLERATE", "48000"))
GO2_AUDIO_CHANNELS = int(os.getenv("GO2_AUDIO_CHANNELS", "2"))
GO2_AUDIO_BUFFER = int(os.getenv("GO2_AUDIO_BUFFER", "8192"))
LIVEINTERFACE_SOURCE = os.getenv("LIVEINTERFACE_SOURCE", "go2").strip().lower()
LIVEINTERFACE_CAM_INDEX = int(os.getenv("LIVEINTERFACE_CAM_INDEX", "0"))
GO2_REACT_ENABLED = os.getenv("GO2_REACT_ENABLED", "0").lower() in {"1", "true", "yes", "on"}
GO2_STAND_ON_RECOGNITION = os.getenv("GO2_STAND_ON_RECOGNITION", "1").lower() in {"1", "true", "yes", "on"}
GO2_SIT_ON_DETECT = os.getenv("GO2_SIT_ON_DETECT", "0").lower() in {"1", "true", "yes", "on"}
GO2_DETECT_EVERY = max(1, int(os.getenv("GO2_DETECT_EVERY", "2")))
GO2_SIT_HUMAN_FRAMES = max(1, int(os.getenv("GO2_SIT_HUMAN_FRAMES", "3")))
GO2_SIT_MIN_FACE_RATIO = max(0.0, float(os.getenv("GO2_SIT_MIN_FACE_RATIO", "0.22")))
GO2_SIT_MIN_BODY_RATIO = max(0.0, float(os.getenv("GO2_SIT_MIN_BODY_RATIO", "0.58")))
GO2_SEARCH_LIGHT_ON_SIT = os.getenv("GO2_SEARCH_LIGHT_ON_SIT", "0").lower() in {"1", "true", "yes", "on"}
GO2_SEARCH_LIGHT_BRIGHTNESS = max(0, min(10, int(os.getenv("GO2_SEARCH_LIGHT_BRIGHTNESS", "10"))))
EMBEDDING_TEMPORAL_WINDOW = max(1, int(os.getenv("EMBEDDING_TEMPORAL_WINDOW", "5")))
EMBEDDING_TRACK_TTL = max(1, int(os.getenv("EMBEDDING_TRACK_TTL", "10")))
EMBEDDING_TRACK_IOU = max(0.0, min(1.0, float(os.getenv("EMBEDDING_TRACK_IOU", "0.12"))))

CASCADE_PATH = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
PROFILE_PATH = os.path.join(cv2.data.haarcascades, "haarcascade_profileface.xml")
FACE_CASCADE = cv2.CascadeClassifier(CASCADE_PATH)
PROFILE_CASCADE = cv2.CascadeClassifier(PROFILE_PATH)
FACE_LIGHTING_MODE = os.getenv("FACE_LIGHTING_MODE", "clahe")
FACE_MIN_SIZE = max(20, int(os.getenv("FACE_MIN_SIZE", "60")))
ROTATION_ANGLES = (-30.0, -15.0, 15.0, 30.0)
DISTANCE_SCALE_FACTORS = (1.2, 1.5, 2.2)
CLAHE = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
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
BODY_DEDUPE_IOU = 0.85
HOG_PERSON_DETECTOR = cv2.HOGDescriptor()
HOG_PERSON_DETECTOR.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
FULLBODY_CASCADE_PATH = os.path.join(cv2.data.haarcascades, "haarcascade_fullbody.xml")
FULLBODY_CASCADE = cv2.CascadeClassifier(FULLBODY_CASCADE_PATH)
EMBEDDING_HELPERS = None


def format_role(role: str) -> str:
    return (role or "Unknown").strip().title()


def format_identity(name: str, role: str) -> str:
    role_label = format_role(role)
    if role_label and role_label != "Unknown":
        return f"{name} ({role_label})"
    return name


def parse_identity(value):
    if isinstance(value, dict):
        return {
            "name": str(value.get("name") or "Unknown").strip() or "Unknown",
            "role": format_role(str(value.get("role") or "Unknown")),
        }
    return {"name": str(value or "Unknown"), "role": "Unknown"}


def put_label(img, text, org, scale=0.6, color=(255, 255, 255), bg=(0, 0, 0), thickness=1):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = org
    cv2.rectangle(img, (x, y - h - 6), (x + w + 6, y + 4), bg, -1)
    cv2.putText(img, text, (x + 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def replace_queued_frame(frame_queue: Queue, frame: np.ndarray):
    try:
        frame_queue.put_nowait(frame)
        return
    except Full:
        pass


def load_embedding_helpers():
    global EMBEDDING_HELPERS
    if EMBEDDING_HELPERS is not None:
        return EMBEDDING_HELPERS
    import embedding_face_recognition_dual_display as embedding_helpers

    embedding_helpers.ensure_models(download=False)
    EMBEDDING_HELPERS = embedding_helpers
    return EMBEDDING_HELPERS


def load_embedding_runtime():
    helpers = load_embedding_helpers()
    if not Path(EMBEDDINGS_PATH).is_file() or not Path(EMBEDDING_LABELS_PATH).is_file():
        raise RuntimeError(
            "Embedding model not trained. Run:\n"
            'python "embedding_face_recognition_dual_display.py" train'
        )
    payload = np.load(EMBEDDINGS_PATH)
    embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
    labels = np.asarray(payload["labels"], dtype=np.int32)
    with open(EMBEDDING_LABELS_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
        id2identity = {int(k): parse_identity(v) for k, v in raw.items()}
    return {
        "helpers": helpers,
        "recognizer": helpers.create_recognizer(),
        "detector": None,
        "embeddings": embeddings,
        "labels": labels,
        "id2identity": id2identity,
    }


def proximity_from_boxes(
    faces: List[dict],
    body_boxes: List[Tuple[int, int, int, int]],
    frame_shape: Tuple[int, int, int],
) -> Tuple[bool, float, float]:
    frame_h = max(1, frame_shape[0])
    max_face_ratio = max((face["h"] / frame_h for face in faces), default=0.0)
    max_body_ratio = max((h / frame_h for (_, _, _, h) in body_boxes), default=0.0)
    close_enough = max_face_ratio >= GO2_SIT_MIN_FACE_RATIO or max_body_ratio >= GO2_SIT_MIN_BODY_RATIO
    return close_enough, max_face_ratio, max_body_ratio


async def send_sport_command(conn: UnitreeWebRTCConnection, api_id: int, parameter: dict | None = None):
    payload = {"api_id": api_id}
    if parameter is not None:
        payload["parameter"] = parameter
    return await conn.datachannel.pub_sub.publish_request_new(RTC_TOPIC["SPORT_MOD"], payload)


async def send_robot_sit(conn: UnitreeWebRTCConnection):
    return await send_sport_command(conn, SPORT_CMD["Sit"])


async def send_robot_stand_up(conn: UnitreeWebRTCConnection):
    return await send_sport_command(conn, SPORT_CMD["StandUp"])


async def set_vui_brightness(conn: UnitreeWebRTCConnection, brightness: int):
    brightness = max(0, min(10, int(brightness)))
    return await conn.datachannel.pub_sub.publish_request_new(
        RTC_TOPIC["VUI"],
        {"api_id": 1005, "parameter": {"brightness": brightness}},
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
                {"api_id": 1002, "parameter": {"name": "normal"}},
            )
            print("[Motion] Switched to normal mode")
            await asyncio.sleep(2.0)
    except Exception as exc:
        print(f"[Motion] Could not verify/switch mode: {exc}")

    try:
        frame_queue.get_nowait()
    except Empty:
        pass

    try:
        frame_queue.put_nowait(frame)
    except Full:
        pass


def load_email_settings() -> dict:
    settings = {
        "sender": "",
        "password": "",
        "recipients": [],
        "smtp_host": "smtp.gmail.com",
        "smtp_port": 465,
    }

    if EMAIL_CONFIG_PATH.is_file():
        try:
            with open(EMAIL_CONFIG_PATH, "r", encoding="utf-8") as f:
                raw = json.load(f)
            settings["sender"] = str(raw.get("sender") or "").strip()
            settings["password"] = str(raw.get("password") or "").strip()
            recipients = raw.get("recipients") or []
            if isinstance(recipients, str):
                recipients = [r.strip() for r in recipients.split(",") if r.strip()]
            settings["recipients"] = [str(r).strip() for r in recipients if str(r).strip()]
            settings["smtp_host"] = str(raw.get("smtp_host") or settings["smtp_host"]).strip() or settings["smtp_host"]
            settings["smtp_port"] = int(raw.get("smtp_port") or settings["smtp_port"])
            return settings
        except Exception as exc:
            print(f"[Email] Could not read {EMAIL_CONFIG_PATH}: {exc}")

    sender = os.getenv("ALERT_EMAIL_SENDER", "").strip()
    password = os.getenv("ALERT_EMAIL_PASSWORD", "").strip()
    recipients_raw = os.getenv("ALERT_EMAIL_RECIPIENTS", "").strip()
    recipients = [r.strip() for r in recipients_raw.split(",") if r.strip()]
    if sender:
        settings["sender"] = sender
    if password:
        settings["password"] = password
    if recipients:
        settings["recipients"] = recipients
    return settings


def send_alert_email(name: str, attachment_path: Path | None = None, detected_at: str | None = None) -> bool:
    settings = load_email_settings()
    sender = settings["sender"]
    password = settings["password"]
    recipients = settings["recipients"]
    smtp_host = settings["smtp_host"]
    smtp_port = settings["smtp_port"]
    if not sender or not password or not recipients:
        return False

    msg = EmailMessage()
    msg["Subject"] = f"Patrol Robot Alert: Threat Level: None | ID: {name}"
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    body = f"Patrol Robot Alert\nThreat Level: None\nDetected ID: {name}"
    if detected_at:
        body += f"\nDetected at: {detected_at}"
    msg.set_content(body)

    if attachment_path is not None and attachment_path.is_file():
        mime_type, _ = mimetypes.guess_type(str(attachment_path))
        maintype, subtype = ("application", "octet-stream")
        if mime_type:
            maintype, subtype = mime_type.split("/", 1)
        with open(attachment_path, "rb") as f:
            msg.add_attachment(
                f.read(),
                maintype=maintype,
                subtype=subtype,
                filename=attachment_path.name,
            )

    try:
        with smtplib.SMTP_SSL(smtp_host, smtp_port) as smtp:
            smtp.login(sender, password)
            smtp.send_message(msg)
        print(f"[Email] Sent alert for '{name}'.")
        return True
    except Exception as exc:
        print(f"[Email] SMTP Error: {exc}")
        return False


def prepare_clip_frame(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= ALERT_CLIP_MAX_WIDTH:
        return frame.copy()
    scale = ALERT_CLIP_MAX_WIDTH / max(1, w)
    resized_h = max(1, int(h * scale))
    return cv2.resize(frame, (ALERT_CLIP_MAX_WIDTH, resized_h), interpolation=cv2.INTER_AREA)


def alert_clip_stem(name: str, detected_at_epoch: float) -> str:
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(detected_at_epoch))
    safe_name = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name).strip("_") or "unknown"
    return f"{timestamp}_{safe_name}"


def write_alert_clip(frames: list[np.ndarray], name: str, detected_at_epoch: float) -> Path | None:
    if not frames:
        return None
    ALERT_CLIP_DIR.mkdir(parents=True, exist_ok=True)
    clip_path = ALERT_CLIP_DIR / f"{alert_clip_stem(name, detected_at_epoch)}_video.mp4"
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(clip_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(1.0, ALERT_CLIP_FPS),
        (width, height),
    )
    if not writer.isOpened():
        return None
    try:
        for frame in frames:
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(frame)
    finally:
        writer.release()
    return clip_path


def write_alert_audio(audio_chunks: list[bytes], name: str, detected_at_epoch: float) -> Path | None:
    if not audio_chunks:
        return None
    ALERT_CLIP_DIR.mkdir(parents=True, exist_ok=True)
    audio_path = ALERT_CLIP_DIR / f"{alert_clip_stem(name, detected_at_epoch)}_audio.wav"
    try:
        with wave.open(str(audio_path), "wb") as wav_file:
            wav_file.setnchannels(GO2_AUDIO_CHANNELS)
            wav_file.setsampwidth(2)
            wav_file.setframerate(GO2_AUDIO_SAMPLERATE)
            for chunk in audio_chunks:
                wav_file.writeframes(chunk)
        return audio_path
    except Exception as exc:
        print(f"[Email] Could not write alert audio clip: {exc}")
        return None


def mux_alert_media(video_path: Path | None, audio_path: Path | None, name: str, detected_at_epoch: float) -> Path | None:
    if video_path is None:
        return None
    if audio_path is None or not ALERT_CLIP_INCLUDE_AUDIO:
        return video_path
    if not Path(FFMPEG_BIN).exists():
        print(f"[Email] ffmpeg not found at {FFMPEG_BIN}; sending video-only clip.")
        return video_path

    output_path = ALERT_CLIP_DIR / f"{alert_clip_stem(name, detected_at_epoch)}.mp4"
    cmd = [
        FFMPEG_BIN,
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(output_path),
    ]
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            print(f"[Email] ffmpeg mux failed: {completed.stderr.strip()}")
            return video_path
        try:
            video_path.unlink(missing_ok=True)
            audio_path.unlink(missing_ok=True)
        except Exception:
            pass
        return output_path
    except Exception as exc:
        print(f"[Email] Could not combine alert media: {exc}")
        return video_path


def load_recognizer(model_path: str, labels_path: str):
    if not Path(model_path).is_file() or not Path(labels_path).is_file():
        return None, {}
    if not hasattr(cv2, "face"):
        return None, {}

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(model_path)
    with open(labels_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
        id2identity = {int(k): parse_identity(v) for k, v in raw.items()}
    return recognizer, id2identity


def load_or_train_recognizer(model_path: str, labels_path: str):
    recognizer, id2name = load_recognizer(model_path, labels_path)
    if recognizer is not None:
        return recognizer, id2name

    dataset_dir = BASE_DIR / "dataset"
    if not dataset_dir.is_dir():
        return None, {}
    if not hasattr(cv2, "face"):
        return None, {}

    images = []
    labels = []
    id2identity = {}
    next_id = 0

    role_dirs = [p for p in sorted(dataset_dir.iterdir()) if p.is_dir() and p.name.lower() in VALID_ROLES]
    if role_dirs:
        for role_dir in role_dirs:
            role = role_dir.name.lower()
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
                    images.append(img)
                    labels.append(pid)
    else:
        for person_dir in sorted(dataset_dir.glob("*")):
            if not person_dir.is_dir():
                continue
            name = person_dir.name
            pid = next_id
            next_id += 1
            id2identity[pid] = {"name": name, "role": "Unknown"}
            for img_path in person_dir.glob("*.jpg"):
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                images.append(img)
                labels.append(pid)

    if not images:
        return None, {}

    recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
    recognizer.train(images, np.array(labels))
    try:
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        recognizer.save(str(model_path))
        Path(labels_path).parent.mkdir(parents=True, exist_ok=True)
        with open(labels_path, "w", encoding="utf-8") as f:
            json.dump(id2identity, f, indent=2)
    except Exception:
        pass
    return recognizer, id2identity


def detect_faces(gray: np.ndarray):
    return detect_faces_advanced(gray, min_face_size=FACE_MIN_SIZE)


def adjust_lighting(gray: np.ndarray, mode: str = "clahe") -> np.ndarray:
    mode = (mode or "none").lower()
    if mode == "hist":
        return cv2.equalizeHist(gray)
    if mode == "clahe":
        return CLAHE.apply(gray)
    return gray


def _make_min_size(size: int):
    clamped = max(1, int(size))
    return (clamped, clamped)


def _cascade_detect(gray: np.ndarray, min_size):
    faces = list(FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=min_size))
    if faces:
        return faces
    profiles = list(PROFILE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=min_size))
    return profiles


def _rotate_gray(gray: np.ndarray, angle: float):
    h, w = gray.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated, M


def _map_rotated_box(det, invM, shape):
    x, y, w, h = det
    corners = np.array([[x, y], [x + w, y], [x, y + h], [x + w, y + h]], dtype=np.float32).reshape(-1, 1, 2)
    mapped = cv2.transform(corners, invM).reshape(-1, 2)
    min_x = int(np.clip(mapped[:, 0].min(), 0, shape[1] - 1))
    max_x = int(np.clip(mapped[:, 0].max(), 0, shape[1] - 1))
    min_y = int(np.clip(mapped[:, 1].min(), 0, shape[0] - 1))
    max_y = int(np.clip(mapped[:, 1].max(), 0, shape[0] - 1))
    return (min_x, min_y, max(1, max_x - min_x), max(1, max_y - min_y))


def detect_with_rotation(gray: np.ndarray, min_face_size: int):
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
    mirrored_profiles = PROFILE_CASCADE.detectMultiScale(mirrored, scaleFactor=1.1, minNeighbors=5, minSize=min_size)
    if len(mirrored_profiles) > 0:
        width = gray.shape[1]
        return [(width - x - w, y, w, h) for (x, y, w, h) in mirrored_profiles]
    return []


def _map_scaled_box(det, scale: float, shape):
    inv_scale = 1.0 / scale
    x, y, w, h = det
    return (
        int(np.clip(round(x * inv_scale), 0, shape[1] - 1)),
        int(np.clip(round(y * inv_scale), 0, shape[0] - 1)),
        max(1, int(round(w * inv_scale))),
        max(1, int(round(h * inv_scale))),
    )


def detect_faces_advanced(gray: np.ndarray, min_face_size: int = 60):
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


def _box_iou(a, b):
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


def expand_body_box(box, frame_shape, width_mul=BODY_WIDTH_MULTIPLIER, height_mul=BODY_HEIGHT_MULTIPLIER, vertical_bias=BODY_VERTICAL_BIAS):
    fx, fy, fw, fh = box
    frame_h, frame_w = frame_shape
    new_w = min(frame_w, max(1, int(round(fw * width_mul))))
    new_h = min(frame_h, max(1, int(round(fh * height_mul))))
    new_x = max(0, min(frame_w - new_w, int(round(fx - (new_w - fw) / 2))))
    new_y = max(0, min(frame_h - new_h, int(round(fy - fh * vertical_bias))))
    return (new_x, new_y, new_w, new_h)


def is_plausible_body_box(box, frame_shape):
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


def detect_people(frame: np.ndarray, scale: float = PERSON_DETECTION_SCALE):
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


def dedupe_body_boxes(boxes):
    unique = []
    for box in boxes:
        if not any(_box_iou(box, existing) > BODY_DEDUPE_IOU for existing in unique):
            unique.append(box)
    return unique


def smooth_body_box(previous, current, alpha: float = BODY_SMOOTHING_ALPHA):
    px, py, pw, ph = previous
    cx, cy, cw, ch = current
    beta = max(0.0, min(1.0, alpha))
    return (
        int(round(px * (1.0 - beta) + cx * beta)),
        int(round(py * (1.0 - beta) + cy * beta)),
        int(round(pw * (1.0 - beta) + cw * beta)),
        int(round(ph * (1.0 - beta) + ch * beta)),
    )


def stabilize_body_boxes(current_boxes, tracked_boxes):
    updated_tracks = []
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


def detect_full_body(gray: np.ndarray):
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
    filtered = []
    for box in bodies:
        expanded = expand_body_box(box, frame_shape)
        if is_plausible_body_box(expanded, frame_shape):
            filtered.append(expanded)
    return filtered


def infer_body_boxes_from_faces(faces, frame_shape):
    boxes = []
    frame_h, frame_w = frame_shape
    for face in faces:
        x, y, w, h = face["x"], face["y"], face["w"], face["h"]
        target_w = max(1, int(round(w * 1.25)))
        target_h = max(1, int(round(h * 2.6)))
        target_w = min(target_w, frame_w)
        target_h = min(target_h, frame_h - max(0, y - int(h * 0.1)))
        body_x = max(0, min(frame_w - target_w, x - (target_w - w) // 2))
        body_y = max(0, y - int(h * 0.1))
        boxes.append(expand_body_box((body_x, body_y, target_w, target_h), frame_shape))
    return boxes


def average_embedding(vectors):
    if not vectors:
        return None
    stacked = np.vstack(vectors).astype(np.float32)
    mean_vec = np.mean(stacked, axis=0)
    norm = float(np.linalg.norm(mean_vec))
    if norm <= 1e-8:
        return None
    return mean_vec / norm


def update_embedding_tracks(face_dicts, face_tracks):
    updated_tracks = []
    used_tracks = set()
    for face in face_dicts:
        box = (face["x"], face["y"], face["w"], face["h"])
        best_index = -1
        best_iou = 0.0
        for index, track in enumerate(face_tracks):
            if index in used_tracks:
                continue
            iou = _box_iou(box, track["box"])
            if iou > best_iou:
                best_iou = iou
                best_index = index
        if best_index >= 0 and best_iou >= EMBEDDING_TRACK_IOU:
            track = face_tracks[best_index]
            history = list(track["history"])
            history.append(face.get("feature"))
            history = [item for item in history[-EMBEDDING_TEMPORAL_WINDOW:] if item is not None]
            updated_tracks.append({"box": box, "history": history, "misses": 0})
            face["feature_avg"] = average_embedding(history)
            used_tracks.add(best_index)
        else:
            history = [face.get("feature")] if face.get("feature") is not None else []
            updated_tracks.append({"box": box, "history": history, "misses": 0})
            face["feature_avg"] = average_embedding(history)
    for index, track in enumerate(face_tracks):
        if index in used_tracks:
            continue
        misses = int(track["misses"]) + 1
        if misses <= EMBEDDING_TRACK_TTL:
            updated_tracks.append({"box": track["box"], "history": list(track["history"]), "misses": misses})
    return updated_tracks


def recognize_faces_embedding(frame: np.ndarray, runtime: dict, conf_threshold: float):
    helpers = runtime["helpers"]
    if runtime["detector"] is None:
        runtime["detector"] = helpers.create_detector((frame.shape[1], frame.shape[0]))
    detected_faces = helpers.detect_faces(frame, runtime["detector"], min_face_size=FACE_MIN_SIZE)
    if not detected_faces:
        return False, "Unknown", 0.0, []

    top_name = "Unknown"
    top_conf = 0.0
    out_faces = []
    for face in detected_faces:
        x, y, w, h = helpers.face_box(face)
        feature = helpers.encode_face(frame, face, runtime["recognizer"])
        out_faces.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h), "feature": feature})

    runtime["face_tracks"] = update_embedding_tracks(out_faces, runtime.get("face_tracks", []))

    for face in out_faces:
        feature_for_match = face.get("feature_avg")
        if feature_for_match is None:
            feature_for_match = face.get("feature")
        if feature_for_match is None:
            display_name, conf = "Unknown", 0.0
        else:
            display_name, conf = helpers.classify_feature(
                feature_for_match,
                runtime["embeddings"],
                runtime["labels"],
                runtime["id2identity"],
                conf_threshold,
            )
        face["name"] = display_name
        face["conf"] = conf
        face.pop("feature", None)
        face.pop("feature_avg", None)
        if conf > top_conf:
            top_conf = conf
            top_name = display_name
    return True, top_name, top_conf, out_faces


def recognize_faces(frame: np.ndarray, recognizer, id2identity: dict, conf_threshold: float):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    proc = adjust_lighting(gray, FACE_LIGHTING_MODE)
    faces = detect_faces_advanced(proc, min_face_size=FACE_MIN_SIZE)
    if len(faces) == 0:
        return False, "Unknown", 0.0, []

    top_name = "Unknown"
    top_conf = 0.0
    out_faces = []
    for (x, y, w, h) in faces:
        if recognizer is None:
            out_faces.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h), "name": "Unknown", "conf": 0.0})
            continue
        roi = proc[y:y + h, x:x + w]
        label_id, distance = recognizer.predict(roi)
        identity = id2identity.get(label_id, {"name": "Unknown", "role": "Unknown"})
        raw_name = format_identity(identity["name"], identity["role"])
        conf = max(0.0, min(100.0, 100.0 - distance))
        if raw_name != "Unknown" and conf >= conf_threshold:
            name = raw_name
        else:
            name = "Unknown"
        out_faces.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h), "name": name, "conf": conf})
        if conf > top_conf:
            top_conf = conf
            top_name = name
    return True, top_name, top_conf, out_faces


def draw_viewer(frame: np.ndarray, faces: list, name: str, conf: float, fps_smoothed: float, muted: bool, body_boxes: list | None = None, scene_status: str = "Unknown"):
    canvas_h = 1120
    canvas_w = 1920
    sidebar_w = 420
    header_h = 82
    margin = 18

    out = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

    content_x1 = margin
    content_y1 = margin
    content_x2 = canvas_w - sidebar_w - margin
    content_y2 = canvas_h - margin

    video_x1 = content_x1
    video_y1 = content_y1 + header_h
    video_x2 = content_x2
    video_y2 = content_y2

    video_w = video_x2 - video_x1
    video_h = video_y2 - video_y1
    fh, fw = frame.shape[:2]
    scale = min(video_w / fw, video_h / fh)
    resized_w = max(1, int(fw * scale))
    resized_h = max(1, int(fh * scale))
    resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    offset_x = video_x1 + (video_w - resized_w) // 2
    offset_y = video_y1 + (video_h - resized_h) // 2
    out[offset_y:offset_y + resized_h, offset_x:offset_x + resized_w] = resized
    scale_x = resized_w / fw
    scale_y = resized_h / fh
    for (x, y, w, h) in body_boxes or []:
        sx1 = offset_x + int(x * scale_x)
        sy1 = offset_y + int(y * scale_y)
        sx2 = offset_x + int((x + w) * scale_x)
        sy2 = offset_y + int((y + h) * scale_y)
        cv2.rectangle(out, (sx1, sy1), (sx2, sy2), (0, 150, 255), 2)
        put_label(out, "Body", (sx1, min(video_y2 - 8, sy2 + 20)), scale=0.5, bg=(0, 120, 200))
    for face in faces:
        x, y, w, h = face["x"], face["y"], face["w"], face["h"]
        face_name = face.get("name", "Unknown")
        face_conf = face.get("conf", 0.0)
        face_nature = face.get("nature", "")
        sx1 = offset_x + int(x * scale_x)
        sy1 = offset_y + int(y * scale_y)
        sx2 = offset_x + int((x + w) * scale_x)
        sy2 = offset_y + int((y + h) * scale_y)
        cv2.rectangle(out, (sx1, sy1), (sx2, sy2), (0, 0, 255), 2)
        suffix = f" {face_nature}" if face_nature else ""
        put_label(out, f"{face_name} ({face_conf:.1f}%){suffix}", (sx1, max(video_y1 + 24, sy1)))

    small_logo = cv2.imread(str(SMALL_LOGO_PATH))
    if small_logo is not None:
        target_h = 40
        scale = target_h / small_logo.shape[0]
        target_w = int(small_logo.shape[1] * scale)
        small_logo = cv2.resize(small_logo, (target_w, target_h), interpolation=cv2.INTER_AREA)
        out[content_y1 + 8:content_y1 + 8 + target_h, content_x1 + 6:content_x1 + 6 + target_w] = small_logo

    mute_rect = (content_x2 - 168, content_y1 + 10, content_x2 - 90, content_y1 + 42)
    cv2.rectangle(out, (mute_rect[0], mute_rect[1]), (mute_rect[2], mute_rect[3]), (255, 255, 255), -1)
    cv2.rectangle(out, (mute_rect[0], mute_rect[1]), (mute_rect[2], mute_rect[3]), (215, 215, 215), 1)
    cv2.putText(out, "Unmute" if muted else "Mute", (mute_rect[0] + 8, mute_rect[1] + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (17, 17, 17), 1, cv2.LINE_AA)

    logout_rect = (content_x2 - 82, content_y1 + 10, content_x2 - 6, content_y1 + 42)
    cv2.rectangle(out, (logout_rect[0], logout_rect[1]), (logout_rect[2], logout_rect[3]), (255, 255, 255), -1)
    cv2.rectangle(out, (logout_rect[0], logout_rect[1]), (logout_rect[2], logout_rect[3]), (215, 215, 215), 1)
    cv2.putText(out, "Logout", (logout_rect[0] + 10, logout_rect[1] + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (17, 17, 17), 1, cv2.LINE_AA)

    cv2.rectangle(out, (video_x1, video_y1), (video_x2, video_y2), (215, 215, 215), 1)
    cv2.rectangle(out, (canvas_w - sidebar_w, 0), (canvas_w, canvas_h), (255, 255, 255), -1)
    cv2.line(out, (canvas_w - sidebar_w, 0), (canvas_w - sidebar_w, canvas_h), (215, 215, 215), 1)
    cv2.putText(out, "Detection Log", (canvas_w - sidebar_w + 18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 20, 20), 2, cv2.LINE_AA)
    put_label(out, f"FPS: {fps_smoothed:.1f}", (video_x1 + 10, video_y1 + 28), scale=0.58, bg=(40, 40, 40))
    put_label(out, f"Scene: {scene_status}", (video_x1 + 10, video_y1 + 58), scale=0.55, bg=(40, 40, 40))

    y = 60
    for item in EVENT_LOG[-12:][::-1]:
        cv2.putText(out, item["name"], (canvas_w - sidebar_w + 18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(out, item["ts"], (canvas_w - sidebar_w + 18, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (95, 95, 95), 1, cv2.LINE_AA)
        y += 42
    return out, {"logout": logout_rect, "mute": mute_rect}


def draw_login_screen(canvas: np.ndarray, username: str, password: str, focus: str, message: str):
    canvas[:] = 248
    h, w = canvas.shape[:2]

    logo = cv2.imread(str(LOGO_PATH))
    if logo is not None:
        target_w = min(760, w - 160)
        scale = target_w / logo.shape[1]
        target_h = int(logo.shape[0] * scale)
        logo = cv2.resize(logo, (target_w, target_h), interpolation=cv2.INTER_AREA)
        x = (w - target_w) // 2
        y = max(28, h // 18)
        canvas[y:y + target_h, x:x + target_w] = logo
        top = y + target_h + max(28, h // 40)
    else:
        top = max(90, h // 9)

    card_w = min(620, w - 140)
    card_h = min(420, h - top - 80)
    x1 = (w - card_w) // 2
    y1 = top
    x2 = x1 + card_w
    y2 = y1 + card_h

    cv2.rectangle(canvas, (x1 + 10, y1 + 14), (x2 + 10, y2 + 14), (232, 232, 232), -1)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 255), -1)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (215, 215, 215), 1)

    cv2.putText(canvas, "Campus Security Robot Live Interface Login", (x1 + 28, y1 + 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.95, (17, 17, 17), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Access to live system of the Patrol Robot (Demo)", (x1 + 28, y1 + 84),
                cv2.FONT_HERSHEY_SIMPLEX, 0.66, (68, 68, 68), 1, cv2.LINE_AA)

    field_w = card_w - 56
    user_rect = (x1 + 28, y1 + 142, x1 + 28 + field_w, y1 + 194)
    pass_rect = (x1 + 28, y1 + 238, x1 + 28 + field_w, y1 + 290)
    button_rect = (x1 + 28, y1 + 330, x1 + 28 + field_w, y1 + 384)

    cv2.putText(canvas, "USERNAME", (user_rect[0], user_rect[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (50, 50, 50), 1, cv2.LINE_AA)
    cv2.putText(canvas, "PASSWORD", (pass_rect[0], pass_rect[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (50, 50, 50), 1, cv2.LINE_AA)

    for rect, active in ((user_rect, focus == "user"), (pass_rect, focus == "pass")):
        color = (215, 25, 32) if active else (207, 207, 207)
        cv2.rectangle(canvas, (rect[0], rect[1]), (rect[2], rect[3]), (255, 255, 255), -1)
        cv2.rectangle(canvas, (rect[0], rect[1]), (rect[2], rect[3]), color, 2 if active else 1)

    cv2.putText(canvas, username, (user_rect[0] + 14, user_rect[1] + 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.78, (17, 17, 17), 1, cv2.LINE_AA)
    masked = "*" * len(password)
    cv2.putText(canvas, masked, (pass_rect[0] + 14, pass_rect[1] + 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.78, (17, 17, 17), 1, cv2.LINE_AA)

    active_rect = user_rect if focus == "user" else pass_rect
    active_text = username if focus == "user" else masked
    caret_x = active_rect[0] + 14
    if active_text:
        (text_w, _), _ = cv2.getTextSize(active_text, cv2.FONT_HERSHEY_SIMPLEX, 0.78, 1)
        caret_x += text_w + 2
    cv2.line(canvas, (caret_x, active_rect[1] + 10), (caret_x, active_rect[1] + 40), (215, 25, 32), 2)

    cv2.rectangle(canvas, (button_rect[0], button_rect[1]), (button_rect[2], button_rect[3]), (215, 25, 32), -1)
    cv2.putText(canvas, "Login", (button_rect[0] + field_w // 2 - 34, button_rect[1] + 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.82, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(canvas, "Demo version 1.1.2 on 4/17/2026.", (x1 + 130, y2 - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (102, 102, 102), 1, cv2.LINE_AA)
    if message:
        cv2.putText(canvas, message, (x1 + 28, y2 + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (215, 25, 32), 1, cv2.LINE_AA)
    return {"user": user_rect, "pass": pass_rect, "button": button_rect}


def point_in_rect(x: int, y: int, rect):
    return rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]


def login_screen():
    window = "LiveInterface Login"
    screen_w = 1920
    screen_h = 1120
    canvas = np.full((screen_h, screen_w, 3), 248, dtype=np.uint8)
    state = {
        "focus": "user",
        "username": "",
        "password": "",
        "message": "",
        "submit": False,
    }
    rects = {}

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if point_in_rect(x, y, rects["user"]):
            state["focus"] = "user"
        elif point_in_rect(x, y, rects["pass"]):
            state["focus"] = "pass"
        elif point_in_rect(x, y, rects["button"]):
            state["submit"] = True

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1880, 1080)
    cv2.setMouseCallback(window, on_mouse)

    while True:
        rects = draw_login_screen(canvas, state["username"], state["password"], state["focus"], state["message"])
        cv2.imshow(window, canvas)
        key = cv2.waitKey(30) & 0xFF

        if state["submit"]:
            state["submit"] = False
            if state["username"] == LOGIN_USER and state["password"] == LOGIN_PASS:
                cv2.destroyWindow(window)
                return True
            state["message"] = "Invalid username or password."
            continue
        if key in (10, 13):
            if state["focus"] == "user":
                state["focus"] = "pass"
            else:
                if state["username"] == LOGIN_USER and state["password"] == LOGIN_PASS:
                    cv2.destroyWindow(window)
                    return True
                state["message"] = "Invalid username or password."
            continue
        if key == 9:
            state["focus"] = "pass" if state["focus"] == "user" else "user"
        elif key in (8, 127):
            if state["focus"] == "user":
                state["username"] = state["username"][:-1]
            else:
                state["password"] = state["password"][:-1]
        elif key == 27:
            cv2.destroyWindow(window)
            return False
        elif 32 <= key <= 126:
            state["message"] = ""
            if state["focus"] == "user":
                state["username"] += chr(key)
            else:
                state["password"] += chr(key)


def run_viewer():
    global EVENT_LOG, EMAIL_SENT, LOGGED_IDENTITIES
    height, width = 1120, 1920
    img = np.zeros((height, width, 3), dtype=np.uint8)
    EVENT_LOG = []
    EMAIL_SENT = set()
    LOGGED_IDENTITIES = set()

    frame_queue = Queue(maxsize=1)
    face_runtime = load_embedding_runtime()
    print(f"[Faces] Using embedding backend with {len(face_runtime['id2identity'])} identities.")
    last_detect_ts = 0.0
    cached_faces = []
    cached_name = "Unknown"
    cached_conf = 0.0
    cached_scene_status = "Unknown"
    cached_body_boxes = []
    fps_smoothed = 0.0
    last_frame_ts = time.time()
    pending_email_names = []
    recording_state = {
        "active": False,
        "name": "",
        "detected_at_epoch": 0.0,
        "detected_at_label": "",
        "end_ts": 0.0,
        "next_frame_ts": 0.0,
        "frames": [],
        "audio_chunks": [],
    }
    audio_stream = None
    audio_pyaudio = None
    audio_status = "Audio off"
    audio_muted = False
    local_capture = None
    human_persistence = 0
    tracked_body_boxes = []
    sit_triggered = False
    stand_triggered = False
    light_on = False
    sit_future = None
    stand_future = None
    light_on_future = None
    light_off_future = None
    frame_index = 0

    user = os.getenv("UNITREE_EMAIL")
    pw = os.getenv("UNITREE_PASS")
    conn = None
    connection_state = {"error": None}

    viewer_state = {"logout": False, "mute": False, "rects": {"logout": (0, 0, 0, 0), "mute": (0, 0, 0, 0)}}

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if point_in_rect(x, y, viewer_state["rects"]["logout"]):
            viewer_state["logout"] = True
        elif point_in_rect(x, y, viewer_state["rects"]["mute"]):
            viewer_state["mute"] = True

    cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Video", 1880, 1080)
    cv2.moveWindow("Video", 20, 20)
    cv2.imshow("Video", img)
    cv2.waitKey(1)
    cv2.setMouseCallback("Video", on_mouse)

    async def recv_camera_stream(track: MediaStreamTrack):
        while True:
            frame = await track.recv()
            replace_queued_frame(frame_queue, frame.to_ndarray(format="bgr24"))

    async def recv_audio_stream(frame):
        audio_data = np.frombuffer(frame.to_ndarray(), dtype=np.int16)
        if not audio_data.size:
            return
        audio_bytes = audio_data.tobytes()
        if recording_state["active"] and ALERT_CLIP_INCLUDE_AUDIO:
            recording_state["audio_chunks"].append(audio_bytes)
        if audio_stream is None:
            return
        if not audio_muted:
            audio_stream.write(audio_bytes)

    def start_clip_recording(name: str, detected_at_epoch: float, detected_at_label: str):
        recording_state["active"] = True
        recording_state["name"] = name
        recording_state["detected_at_epoch"] = detected_at_epoch
        recording_state["detected_at_label"] = detected_at_label
        recording_state["end_ts"] = detected_at_epoch + ALERT_CLIP_SECONDS
        recording_state["next_frame_ts"] = detected_at_epoch
        recording_state["frames"] = []
        recording_state["audio_chunks"] = []
        print(f"[Email] Recording alert clip for '{name}' for {ALERT_CLIP_SECONDS:.1f}s")

    def finalize_clip_recording():
        name = recording_state["name"]
        detected_at_epoch = recording_state["detected_at_epoch"]
        detected_at_label = recording_state["detected_at_label"]
        frames = list(recording_state["frames"])
        audio_chunks = list(recording_state["audio_chunks"])
        recording_state["active"] = False
        recording_state["name"] = ""
        recording_state["detected_at_epoch"] = 0.0
        recording_state["detected_at_label"] = ""
        recording_state["end_ts"] = 0.0
        recording_state["next_frame_ts"] = 0.0
        recording_state["frames"] = []
        recording_state["audio_chunks"] = []

        video_path = write_alert_clip(frames, name, detected_at_epoch)
        audio_path = write_alert_audio(audio_chunks, name, detected_at_epoch) if LIVEINTERFACE_SOURCE == "go2" else None
        clip_path = mux_alert_media(video_path, audio_path, name, detected_at_epoch)

        def send_job():
            if send_alert_email(name, clip_path, detected_at_label):
                EMAIL_SENT.add(name)

        threading.Thread(target=send_job, daemon=True).start()

    if LIVEINTERFACE_SOURCE == "go2" and GO2_AUDIO_ENABLED and pyaudio is not None:
        try:
            audio_pyaudio = pyaudio.PyAudio()
            audio_stream = audio_pyaudio.open(
                format=pyaudio.paInt16,
                channels=GO2_AUDIO_CHANNELS,
                rate=GO2_AUDIO_SAMPLERATE,
                output=True,
                frames_per_buffer=GO2_AUDIO_BUFFER,
            )
            audio_status = "Audio ready"
        except Exception as exc:
            audio_stream = None
            audio_pyaudio = None
            audio_status = f"Audio unavailable: {exc}"
            print(f"[Audio] {audio_status}")
    elif LIVEINTERFACE_SOURCE == "go2" and GO2_AUDIO_ENABLED and pyaudio is None:
        audio_status = "Audio unavailable: PyAudio not installed"
        print(f"[Audio] {audio_status}")
    elif LIVEINTERFACE_SOURCE != "go2":
        audio_status = "Audio off"

    def run_asyncio_loop(loop):
        asyncio.set_event_loop(loop)

        async def setup():
            nonlocal conn
            try:
                conn, _ = await connect_best_go2(ip=GO2_IP, username=user, password=pw)
                if GO2_REACT_ENABLED:
                    await ensure_normal_motion_mode(conn)
                await start_go2_video_stream(conn, recv_camera_stream)
                if audio_stream is not None:
                    conn.audio.switchAudioChannel(True)
                    conn.audio.add_track_callback(recv_audio_stream)
            except Exception as exc:
                connection_state["error"] = exc
                logging.error(f"Error in WebRTC connection: {exc}")

        loop.run_until_complete(setup())
        loop.run_forever()

    loop = asyncio.new_event_loop()
    asyncio_thread = None
    if LIVEINTERFACE_SOURCE == "go2":
        asyncio_thread = threading.Thread(target=run_asyncio_loop, args=(loop,))
        asyncio_thread.start()
    else:
        local_capture = cv2.VideoCapture(LIVEINTERFACE_CAM_INDEX)
        if not local_capture.isOpened():
            raise RuntimeError(f"Could not open local camera index {LIVEINTERFACE_CAM_INDEX}")

    try:
        while True:
            if LIVEINTERFACE_SOURCE == "go2":
                has_frame = not frame_queue.empty()
                if has_frame:
                    img = frame_queue.get()
                else:
                    if connection_state["error"] is not None:
                        raise RuntimeError(f"Go2 video connection failed: {connection_state['error']}")
                    time.sleep(0.01)
                    continue
            else:
                ok, local_frame = local_capture.read()
                if not ok or local_frame is None:
                    time.sleep(0.01)
                    continue
                img = local_frame

            frame_index += 1

            now = time.time()
            if recording_state["active"]:
                if now >= recording_state["next_frame_ts"]:
                    recording_state["frames"].append(prepare_clip_frame(img))
                    recording_state["next_frame_ts"] = now + (1.0 / max(1.0, ALERT_CLIP_FPS))
                if now >= recording_state["end_ts"]:
                    finalize_clip_recording()
            elif pending_email_names:
                next_item = pending_email_names.pop(0)
                start_clip_recording(next_item["name"], next_item["detected_at_epoch"], next_item["detected_at_label"])

            if now - last_detect_ts >= 1.0 / max(1.0, DETECT_FPS):
                detected, cached_name, cached_conf, cached_faces = recognize_faces_embedding(
                    img, face_runtime, CONF_THRESHOLD / 100.0
                )
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                body_candidates = detect_full_body(gray)
                body_candidates.extend(detect_people(img))
                body_boxes_real = dedupe_body_boxes(body_candidates)
                tracked_body_boxes = stabilize_body_boxes(body_boxes_real, tracked_body_boxes)
                cached_body_boxes = [track["box"] for track in tracked_body_boxes]
                if not cached_body_boxes:
                    cached_body_boxes = infer_body_boxes_from_faces(cached_faces, img.shape[:2])
                if body_boxes_real:
                    human_persistence = min(human_persistence + 1, HUMAN_PERSISTENCE_FRAMES)
                else:
                    human_persistence = max(0, human_persistence - 1)
                scene_is_human = human_persistence > 0
                face_known_present = False
                for face in cached_faces:
                    face_name = str(face.get("name", "")).strip()
                    if face_name and face_name != "Unknown":
                        face_known_present = True
                    face_box = (face["x"], face["y"], face["w"], face["h"])
                    human_overlap = any(_box_iou(face_box, person) >= 0.05 for person in cached_body_boxes)
                    face["nature"] = "Human" if human_overlap or scene_is_human or face_name != "Unknown" else "Non-human"
                cached_scene_status = "Human" if scene_is_human or face_known_present else "Non-human"
                print(f"[Faces] detected={detected} name={cached_name} conf={cached_conf:.1f} faces={len(cached_faces)}")
                labeled_faces = []
                for face in cached_faces:
                    face_name = str(face.get("name", "")).strip()
                    if face_name and face_name != "Unknown" and face_name not in labeled_faces:
                        labeled_faces.append(face_name)

                if detected and labeled_faces:
                    detected_at_label = time.strftime("%Y-%m-%d %I:%M:%S %p", time.localtime(now))
                    for normalized_name in labeled_faces:
                        if normalized_name in LOGGED_IDENTITIES:
                            continue
                        EVENT_LOG.append({
                            "name": normalized_name,
                            "ts": detected_at_label,
                        })
                        LOGGED_IDENTITIES.add(normalized_name)
                        del EVENT_LOG[:-20]
                        already_pending = any(item["name"] == normalized_name for item in pending_email_names)
                        is_currently_recording = recording_state["active"] and recording_state["name"] == normalized_name
                        if normalized_name not in EMAIL_SENT and not already_pending and not is_currently_recording:
                            pending_email_names.append({
                                "name": normalized_name,
                                "detected_at_epoch": now,
                                "detected_at_label": detected_at_label,
                            })
                elif detected:
                    unknown_name = "Unknown"
                    detected_at_label = time.strftime("%Y-%m-%d %I:%M:%S %p", time.localtime(now))
                    already_pending = any(item["name"] == unknown_name for item in pending_email_names)
                    is_currently_recording = recording_state["active"] and recording_state["name"] == unknown_name
                    if unknown_name not in EMAIL_SENT and not already_pending and not is_currently_recording:
                            pending_email_names.append({
                                "name": unknown_name,
                                "detected_at_epoch": now,
                                "detected_at_label": detected_at_label,
                            })
                if LIVEINTERFACE_SOURCE == "go2" and GO2_REACT_ENABLED and conn is not None:
                    close_enough, max_face_ratio, max_body_ratio = proximity_from_boxes(cached_faces, cached_body_boxes, img.shape)
                    recognized_names = [face["name"] for face in cached_faces if face.get("name") and face.get("name") != "Unknown"]
                    if sit_future is not None and sit_future.done():
                        try:
                            sit_future.result()
                            sit_triggered = True
                            print(
                                f"[Detect] Stable close human detected, sent Sit "
                                f"(face_ratio={max_face_ratio:.2f}, body_ratio={max_body_ratio:.2f})"
                            )
                        except Exception as exc:
                            print(f"[Detect] Failed to send Sit: {exc}")
                        sit_future = None
                    if light_on_future is not None and light_on_future.done():
                        try:
                            light_on_future.result()
                            light_on = True
                        except Exception as exc:
                            print(f"[Detect] Failed to turn on search light: {exc}")
                        light_on_future = None
                    if stand_future is not None and stand_future.done():
                        try:
                            stand_future.result()
                            stand_triggered = True
                            print(f"[Detect] Recognized {recognized_names[0] if recognized_names else 'known ID'}, sent StandUp")
                        except Exception as exc:
                            print(f"[Detect] Failed to send StandUp: {exc}")
                        stand_future = None
                    if light_off_future is not None and light_off_future.done():
                        try:
                            light_off_future.result()
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
                        sit_future = asyncio.run_coroutine_threadsafe(send_robot_sit(conn), loop)
                        if GO2_SEARCH_LIGHT_ON_SIT and not light_on and light_on_future is None:
                            light_on_future = asyncio.run_coroutine_threadsafe(
                                set_vui_brightness(conn, GO2_SEARCH_LIGHT_BRIGHTNESS), loop
                            )
                    if GO2_STAND_ON_RECOGNITION and sit_triggered and not stand_triggered and recognized_names and stand_future is None:
                        stand_future = asyncio.run_coroutine_threadsafe(send_robot_stand_up(conn), loop)
                        if light_on and light_off_future is None:
                            light_off_future = asyncio.run_coroutine_threadsafe(set_vui_brightness(conn, 0), loop)
                last_detect_ts = now

            dt = max(1e-6, now - last_frame_ts)
            fps = 1.0 / dt
            fps_smoothed = fps if fps_smoothed == 0.0 else fps_smoothed * 0.9 + fps * 0.1
            last_frame_ts = now
            display, viewer_state["rects"] = draw_viewer(
                img,
                cached_faces,
                cached_name,
                cached_conf,
                fps_smoothed,
                audio_muted,
                cached_body_boxes,
                cached_scene_status,
            )
            cv2.imshow("Video", display)
            key = cv2.waitKey(1) & 0xFF
            if viewer_state["mute"]:
                audio_muted = not audio_muted
                viewer_state["mute"] = False
            if key == ord("q") or viewer_state["logout"]:
                break
    finally:
        if recording_state["active"] and recording_state["frames"]:
            finalize_clip_recording()
        cv2.destroyAllWindows()
        if asyncio_thread is not None:
            loop.call_soon_threadsafe(loop.stop)
            asyncio_thread.join()
        if local_capture is not None:
            local_capture.release()
        if audio_stream is not None:
            try:
                audio_stream.stop_stream()
                audio_stream.close()
            except Exception:
                pass
        if audio_pyaudio is not None:
            try:
                audio_pyaudio.terminate()
            except Exception:
                pass


def main():
    if login_screen():
        run_viewer()


if __name__ == "__main__":
    main()

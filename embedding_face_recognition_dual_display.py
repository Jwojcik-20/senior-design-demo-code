import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Dict, List, Optional, Tuple

import cv2
import cv2.data as cvdata
import numpy as np
from aiortc import MediaStreamTrack
from unitree_webrtc_connect.constants import RTC_TOPIC
from unitree_webrtc_connect.webrtc_driver import UnitreeWebRTCConnection, WebRTCConnectionMethod
from go2_connection import (
    GO2_IP,
    connect_best_go2,
    connect_go2_control_only,
    connect_go2_media_only,
    connect_go2_single_peer_camera,
    connect_go2_video_only,
    patch_unitree_local_signaling,
    start_go2_video_stream,
)


DATASET_DIR = Path("dataset")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDINGS_PATH = MODEL_DIR / "opencv_face_embeddings.npz"
LABELS_PATH = MODEL_DIR / "opencv_face_labels.json"
ASSET_DIR = MODEL_DIR / "opencv_zoo"
ASSET_DIR.mkdir(parents=True, exist_ok=True)
YUNET_MODEL_PATH = ASSET_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_MODEL_PATH = ASSET_DIR / "face_recognition_sface_2021dec.onnx"

YUNET_MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
ROBOT_USER = os.environ.get("GO2_SSH_USER", "unitree").strip()
ROBOT_HOST = os.environ.get("GO2_SSH_HOST", "192.168.123.18").strip() or "192.168.123.18"
ROBOT_PASS = os.environ.get("GO2_SSH_PASS", "123")
ROBOT_TARGET = f"{ROBOT_USER}@{ROBOT_HOST}"
REMOTE_GLOB = os.environ.get("GO2_SSH_REMOTE_GLOB", "/tmp/*.jpg")
SSH_CACHE_DIR = MODEL_DIR / "go2_ssh_cache"
SSH_CACHE_FILE = SSH_CACHE_DIR / "latest.jpg"
REMOTE_VIDEO_CLIENT = os.environ.get("GO2_SSH_VIDEO_CLIENT", "/home/unitree/unitree_sdk2/build/bin/go2_video_client")
_SSH_CLIENT = None
_SSH_SFTP = None
_SSH_VIDEO_CHANNEL = None
GO2_DIRECT_CONNECT_TIMEOUT = max(3.0, float(os.environ.get("GO2_DIRECT_CONNECT_TIMEOUT", "6")))
GO2_HYBRID_STARTUP_TIMEOUT = max(5.0, float(os.environ.get("GO2_HYBRID_STARTUP_TIMEOUT", "12")))
GO2_HYBRID_DROPOUT_TIMEOUT = max(2.0, float(os.environ.get("GO2_HYBRID_DROPOUT_TIMEOUT", "3")))
WEBRTC_BRIDGE_DIR = MODEL_DIR / "go2_webrtc_bridge"
WEBRTC_BRIDGE_FILE = WEBRTC_BRIDGE_DIR / "latest.jpg"
WEBRTC_BRIDGE_READY = WEBRTC_BRIDGE_DIR / "ready.txt"
WEBRTC_BRIDGE_ERROR = WEBRTC_BRIDGE_DIR / "error.txt"
WEBRTC_BRIDGE_SCRIPT = Path(__file__).resolve().parent / "go2_webrtc_bridge.py"

CASCADE_PATH = os.path.join(cvdata.haarcascades, "haarcascade_frontalface_default.xml")
PROFILE_PATH = os.path.join(cvdata.haarcascades, "haarcascade_profileface.xml")
FACE_CASCADE = cv2.CascadeClassifier(CASCADE_PATH)
PROFILE_CASCADE = cv2.CascadeClassifier(PROFILE_PATH)
if FACE_CASCADE.empty() or PROFILE_CASCADE.empty():
    raise RuntimeError("Could not load Haar cascades. Check your OpenCV install.")

CLAHE = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
VALID_ROLES = {"student", "staff", "guest"}
MIN_FACE_SIZE = 50
ROTATION_ANGLES = (-30.0, -15.0, 15.0, 30.0)
DISTANCE_SCALE_FACTORS = (1.2, 1.5, 2.0, 2.6)
MATCH_THRESHOLD = 0.36
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
FULLBODY_CASCADE_PATH = os.path.join(cvdata.haarcascades, "haarcascade_fullbody.xml")
FULLBODY_CASCADE = cv2.CascadeClassifier(FULLBODY_CASCADE_PATH)
if FULLBODY_CASCADE.empty():
    raise RuntimeError(f"Could not load full body cascade from {FULLBODY_CASCADE_PATH}")

patch_unitree_local_signaling("full")


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


def make_waiting_view(message: str, size: tuple[int, int] = (960, 540)) -> np.ndarray:
    w, h = size
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (w, h), (18, 28, 42), -1)
    put_label(canvas, message, (30, h // 2), scale=0.9, bg=(30, 45, 65))
    return canvas


def adjust_lighting(gray: np.ndarray, mode: str = "clahe") -> np.ndarray:
    mode = (mode or "none").lower()
    if mode == "hist":
        return cv2.equalizeHist(gray)
    if mode == "clahe":
        return CLAHE.apply(gray)
    return gray


def get_latest_frame(frame_queue: Queue) -> Optional[np.ndarray]:
    latest = None
    while True:
        try:
            latest = frame_queue.get_nowait()
        except Empty:
            return latest


def replace_queued_frame(frame_queue: Queue, frame: np.ndarray):
    try:
        frame_queue.put_nowait(frame)
        return
    except Full:
        pass


def pull_latest_ssh_frame(last_remote: str) -> tuple[Optional[np.ndarray], str]:
    SSH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    sftp = get_ssh_sftp()
    remote_dir = os.path.dirname(REMOTE_GLOB)
    entries = []
    for attr in sftp.listdir_attr(remote_dir):
        if attr.filename.lower().endswith(".jpg"):
            entries.append((attr.st_mtime, f"{remote_dir}/{attr.filename}"))
    entries.sort(reverse=True)
    latest = entries[0][1] if entries else ""
    if not latest:
        return None, last_remote
    if latest != last_remote:
        local_name = SSH_CACHE_DIR / os.path.basename(latest)
        sftp.get(latest, str(local_name))
        if not local_name.exists():
            return None, last_remote
        try:
            if SSH_CACHE_FILE.exists() or SSH_CACHE_FILE.is_symlink():
                SSH_CACHE_FILE.unlink()
            SSH_CACHE_FILE.symlink_to(local_name)
        except Exception:
            shutil.copy2(local_name, SSH_CACHE_FILE)
        last_remote = latest
    frame = cv2.imread(str(SSH_CACHE_FILE)) if SSH_CACHE_FILE.exists() else None
    return frame, last_remote


def run_ssh_command(remote_cmd: str, timeout: int = 8) -> subprocess.CompletedProcess:
    client = get_ssh_client(timeout=timeout)
    stdin, stdout, stderr = client.exec_command(remote_cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", "ignore")
    err = stderr.read().decode("utf-8", "ignore")
    status = stdout.channel.recv_exit_status()
    return subprocess.CompletedProcess(["ssh", ROBOT_TARGET, remote_cmd], status, out, err)


def get_ssh_client(timeout: int = 8):
    global _SSH_CLIENT
    if _SSH_CLIENT is not None:
        transport = _SSH_CLIENT.get_transport()
        if transport is not None and transport.is_active():
            return _SSH_CLIENT
    import paramiko

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=ROBOT_HOST,
        username=ROBOT_USER,
        password=ROBOT_PASS,
        timeout=timeout,
        banner_timeout=timeout,
        auth_timeout=timeout,
    )
    _SSH_CLIENT = client
    return client


def get_ssh_sftp(timeout: int = 8):
    global _SSH_SFTP
    if _SSH_SFTP is not None:
        return _SSH_SFTP
    _SSH_SFTP = get_ssh_client(timeout=timeout).open_sftp()
    return _SSH_SFTP


def ensure_remote_go2_video_client():
    global _SSH_VIDEO_CHANNEL
    if _SSH_VIDEO_CHANNEL is not None and not _SSH_VIDEO_CHANNEL.closed:
        return
    run_ssh_command("pkill -f go2_video_client >/dev/null 2>&1 || true; rm -f /tmp/*.jpg >/dev/null 2>&1 || true", timeout=12)
    transport = get_ssh_client(timeout=12).get_transport()
    if transport is None or not transport.is_active():
        raise RuntimeError("SSH transport is not active.")
    channel = transport.open_session(timeout=12)
    channel.exec_command(f"cd /tmp && {REMOTE_VIDEO_CLIENT}")
    _SSH_VIDEO_CHANNEL = channel
    time.sleep(4.0)


def stop_remote_go2_video_client():
    global _SSH_VIDEO_CHANNEL, _SSH_SFTP, _SSH_CLIENT
    try:
        run_ssh_command("pkill -f go2_video_client >/dev/null 2>&1 || true", timeout=6)
    except Exception:
        pass
    try:
        if _SSH_VIDEO_CHANNEL is not None:
            _SSH_VIDEO_CHANNEL.close()
    except Exception:
        pass
    _SSH_VIDEO_CHANNEL = None
    try:
        if _SSH_SFTP is not None:
            _SSH_SFTP.close()
    except Exception:
        pass
    _SSH_SFTP = None
    try:
        if _SSH_CLIENT is not None:
            _SSH_CLIENT.close()
    except Exception:
        pass
    _SSH_CLIENT = None


def start_go2_webrtc_camera(frame_queue: Queue):
    connection_state: Dict[str, object] = {"error": None, "mode": None, "attempts": []}
    session: Dict[str, object | None] = {"conn": None, "pc": None, "control_pc": None}

    async def recv_camera_stream(track: MediaStreamTrack):
        while True:
            try:
                frame = await track.recv()
            except Exception:
                break
            img = frame.to_ndarray(format="bgr24")
            replace_queued_frame(frame_queue, img)

    def run_asyncio_loop(loop):
        asyncio.set_event_loop(loop)

        async def setup():
            attempts: list[dict[str, str]] = connection_state["attempts"]  # type: ignore[assignment]
            try:
                try:
                    patch_unitree_local_signaling("signal")
                    conn = await connect_go2_single_peer_camera(
                        recv_camera_stream,
                        ip=GO2_IP,
                        timeout=15,
                        patch_level="signal",
                    )
                    session["conn"] = conn
                    connection_state["mode"] = "single-peer-camera"
                    return
                except Exception as single_peer_exc:
                    attempts.append({"mode": "single-peer-camera", "error": str(single_peer_exc)})

                try:
                    patch_unitree_local_signaling("signal")
                    conn = UnitreeWebRTCConnection(
                        WebRTCConnectionMethod.LocalSTA,
                        ip=GO2_IP,
                    )
                    session["conn"] = conn
                    await asyncio.wait_for(conn.connect(), timeout=GO2_DIRECT_CONNECT_TIMEOUT)
                    await conn.datachannel.disableTrafficSaving(True)
                    conn.video.switchVideoChannel(True)
                    conn.video.add_track_callback(recv_camera_stream)
                    connection_state["mode"] = "raw-localsta"
                    return
                except SystemExit as raw_exit:
                    attempts.append({"mode": "raw-localsta", "error": f"connect aborted ({raw_exit})"})
                except asyncio.TimeoutError:
                    attempts.append({"mode": "raw-localsta", "error": f"connect timed out after {GO2_DIRECT_CONNECT_TIMEOUT:.1f}s"})
                except Exception as raw_exc:
                    attempts.append({"mode": "raw-localsta", "error": str(raw_exc)})

                try:
                    patch_unitree_local_signaling("full")
                    conn = UnitreeWebRTCConnection(
                        WebRTCConnectionMethod.LocalSTA,
                        ip=GO2_IP,
                    )
                    session["conn"] = conn
                    await asyncio.wait_for(conn.connect(), timeout=GO2_DIRECT_CONNECT_TIMEOUT)
                    await ensure_normal_motion_mode(conn)
                    try:
                        await conn.datachannel.disableTrafficSaving(True)
                    except Exception:
                        pass
                    conn.video.switchVideoChannel(True)
                    conn.video.add_track_callback(recv_camera_stream)
                    connection_state["mode"] = "lbph-direct"
                    return
                except Exception as lbph_direct_exc:
                    if isinstance(lbph_direct_exc, asyncio.TimeoutError):
                        attempts.append({"mode": "lbph-direct", "error": f"connect timed out after {GO2_DIRECT_CONNECT_TIMEOUT:.1f}s"})
                    else:
                        attempts.append({"mode": "lbph-direct", "error": str(lbph_direct_exc)})
                    try:
                        patch_unitree_local_signaling("full")
                        control_pc, control_dc = await connect_go2_control_only(ip=GO2_IP, timeout=15)
                        session["control_pc"] = control_pc
                        try:
                            await control_dc.disableTrafficSaving(True)
                        except Exception:
                            pass
                        control_dc.switchVideoChannel(True)
                        session["pc"] = await connect_go2_media_only(recv_camera_stream, ip=GO2_IP, timeout=15)
                        connection_state["mode"] = "split-webrtc"
                        return
                    except Exception as split_exc:
                        attempts.append({"mode": "split-webrtc", "error": str(split_exc)})
                        conn, _ = await connect_best_go2(ip=GO2_IP)
                        session["conn"] = conn
                        connection_state["mode"] = "full-session"
                        try:
                            try:
                                await conn.datachannel.disableTrafficSaving(True)
                            except Exception:
                                pass
                            await start_go2_video_stream(conn, recv_camera_stream)
                            return
                        except Exception as full_session_exc:
                            attempts.append({"mode": "full-session", "error": str(full_session_exc)})
                            raise RuntimeError(
                                "Go2 camera startup failed. "
                                f"Raw LocalSTA path error: {attempts[0]['error'] if attempts else 'unknown'}. "
                                f"LBPH-direct path error: {lbph_direct_exc}. "
                                f"Split camera path error: {split_exc}. "
                                f"Full-session path error: {full_session_exc}."
                            ) from full_session_exc
            except Exception as exc:
                connection_state["error"] = exc

        try:
            loop.run_until_complete(setup())
            loop.run_forever()
        except RuntimeError as exc:
            if "Event loop stopped before Future completed" not in str(exc):
                raise
        finally:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                try:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                except Exception:
                    pass
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception:
                pass
            loop.close()

    loop = asyncio.new_event_loop()
    asyncio_thread = threading.Thread(target=run_asyncio_loop, args=(loop,), daemon=True)
    asyncio_thread.start()
    return loop, asyncio_thread, session, connection_state


def stop_go2_webrtc_camera(loop, session: Dict[str, object | None]):
    conn = session.get("conn")
    pc = session.get("pc")
    control_pc = session.get("control_pc")
    try:
        if conn is not None:
            asyncio.run_coroutine_threadsafe(conn.disconnect(), loop).result(timeout=3)
        else:
            if pc is not None:
                asyncio.run_coroutine_threadsafe(pc.close(), loop).result(timeout=3)
            if control_pc is not None:
                asyncio.run_coroutine_threadsafe(control_pc.close(), loop).result(timeout=3)
    except Exception:
        pass
    loop.call_soon_threadsafe(loop.stop)


def start_go2_hybrid_camera(frame_queue: Queue):
    return {
        "source": "webrtc",
        "frame_queue": frame_queue,
        "loop": None,
        "thread": None,
        "session": None,
        "connection_state": None,
        "last_remote": "",
        "last_frame_ts": 0.0,
        "using_ssh": False,
    }


def _hybrid_switch_to_ssh(hybrid_state: Dict[str, object], reason: str):
    if hybrid_state.get("using_ssh"):
        return
    loop = hybrid_state.get("loop")
    session = hybrid_state.get("session")
    if loop is not None and session is not None:
        try:
            stop_go2_webrtc_camera(loop, session)  # type: ignore[arg-type]
        except Exception:
            pass
    thread = hybrid_state.get("thread")
    if thread is not None:
        try:
            thread.join(timeout=3.0)  # type: ignore[union-attr]
        except Exception:
            pass
    ensure_remote_go2_video_client()
    hybrid_state["source"] = "ssh"
    hybrid_state["using_ssh"] = True
    hybrid_state["loop"] = None
    hybrid_state["thread"] = None
    hybrid_state["session"] = None
    hybrid_state["connection_state"] = None
    print(f"[Go2Hybrid] Switched to SSH fallback: {reason}", flush=True)


def begin_go2_hybrid_camera(hybrid_state: Dict[str, object]):
    loop, asyncio_thread, session, connection_state = start_go2_webrtc_camera(hybrid_state["frame_queue"])  # type: ignore[arg-type]
    hybrid_state["loop"] = loop
    hybrid_state["thread"] = asyncio_thread
    hybrid_state["session"] = session
    hybrid_state["connection_state"] = connection_state
    hybrid_state["source"] = "webrtc"
    hybrid_state["using_ssh"] = False
    hybrid_state["last_frame_ts"] = 0.0


def stop_go2_hybrid_camera(hybrid_state: Dict[str, object]):
    if hybrid_state.get("using_ssh"):
        stop_remote_go2_video_client()
        return
    loop = hybrid_state.get("loop")
    session = hybrid_state.get("session")
    if loop is not None and session is not None:
        try:
            stop_go2_webrtc_camera(loop, session)  # type: ignore[arg-type]
        except Exception:
            pass
    thread = hybrid_state.get("thread")
    if thread is not None:
        try:
            thread.join(timeout=3.0)  # type: ignore[union-attr]
        except Exception:
            pass


def get_latest_go2_hybrid_frame(hybrid_state: Dict[str, object]) -> Optional[np.ndarray]:
    if hybrid_state.get("using_ssh"):
        frame, last_remote = pull_latest_ssh_frame(str(hybrid_state.get("last_remote", "")))
        hybrid_state["last_remote"] = last_remote
        if frame is not None:
            hybrid_state["last_frame_ts"] = time.time()
        return frame

    frame = get_latest_frame(hybrid_state["frame_queue"])  # type: ignore[arg-type]
    now = time.time()
    connection_state = hybrid_state.get("connection_state")
    if frame is not None:
        hybrid_state["last_frame_ts"] = now
        return frame

    if connection_state is None:
        return None

    mode = connection_state.get("mode") if isinstance(connection_state, dict) else None
    err = connection_state.get("error") if isinstance(connection_state, dict) else None
    last_frame_ts = float(hybrid_state.get("last_frame_ts") or 0.0)
    started = last_frame_ts <= 0.0
    if err is not None:
        _hybrid_switch_to_ssh(hybrid_state, f"WebRTC error in {mode or 'startup'}: {err}")
        return get_latest_go2_hybrid_frame(hybrid_state)
    if started:
        startup_deadline = float(hybrid_state.setdefault("_startup_deadline", now + GO2_HYBRID_STARTUP_TIMEOUT))
        if now >= startup_deadline:
            _hybrid_switch_to_ssh(hybrid_state, f"no WebRTC frame after {GO2_HYBRID_STARTUP_TIMEOUT:.1f}s")
            return get_latest_go2_hybrid_frame(hybrid_state)
    elif now - last_frame_ts >= GO2_HYBRID_DROPOUT_TIMEOUT:
        _hybrid_switch_to_ssh(hybrid_state, f"WebRTC frame dropout > {GO2_HYBRID_DROPOUT_TIMEOUT:.1f}s")
        return get_latest_go2_hybrid_frame(hybrid_state)
    return None


def go2_hybrid_status(hybrid_state: Dict[str, object]) -> str:
    if hybrid_state.get("using_ssh"):
        return "ssh-fallback"
    connection_state = hybrid_state.get("connection_state")
    if isinstance(connection_state, dict):
        return str(connection_state.get("mode") or "webrtc-starting")
    return "webrtc"


def start_webrtc_bridge() -> subprocess.Popen:
    WEBRTC_BRIDGE_DIR.mkdir(parents=True, exist_ok=True)
    if WEBRTC_BRIDGE_READY.exists():
        WEBRTC_BRIDGE_READY.unlink()
    if WEBRTC_BRIDGE_ERROR.exists():
        WEBRTC_BRIDGE_ERROR.unlink()
    return subprocess.Popen(
        [sys.executable, str(WEBRTC_BRIDGE_SCRIPT)],
        cwd=str(Path(__file__).resolve().parent),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=os.environ.copy(),
    )


def wait_for_webrtc_bridge(process: subprocess.Popen, timeout: float = 25.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            error = WEBRTC_BRIDGE_ERROR.read_text(encoding="utf-8").strip() if WEBRTC_BRIDGE_ERROR.exists() else f"bridge exited with code {process.returncode}"
            raise RuntimeError(f"Go2 bridge failed: {error}")
        if WEBRTC_BRIDGE_READY.exists():
            status = WEBRTC_BRIDGE_READY.read_text(encoding="utf-8").strip()
            if status.startswith("streaming:"):
                return
        if WEBRTC_BRIDGE_ERROR.exists():
            raise RuntimeError(f"Go2 bridge failed: {WEBRTC_BRIDGE_ERROR.read_text(encoding='utf-8').strip()}")
        time.sleep(0.1)
    raise RuntimeError("Go2 bridge did not become ready in time.")
    try:
        frame_queue.get_nowait()
    except Empty:
        pass
    try:
        frame_queue.put_nowait(frame)
    except Full:
        pass


def normalize_role(role: str) -> str:
    normalized = (role or "").strip().lower()
    if normalized not in VALID_ROLES:
        raise ValueError(f"Role must be one of: {', '.join(sorted(VALID_ROLES))}")
    return normalized


async def ensure_normal_motion_mode(conn):
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


def download_file(url: str, target_path: Path):
    target_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[Download] {url}")
    urllib.request.urlretrieve(url, target_path)
    print(f"[Download] Saved {target_path}")


def ensure_models(download: bool = False):
    missing = []
    if not YUNET_MODEL_PATH.is_file():
        missing.append((YUNET_MODEL_PATH, YUNET_MODEL_URL))
    if not SFACE_MODEL_PATH.is_file():
        missing.append((SFACE_MODEL_PATH, SFACE_MODEL_URL))
    if not missing:
        return
    if download:
        for path, url in missing:
            download_file(url, path)
        return
    missing_lines = "\n".join(f"- {path}\n  {url}" for path, url in missing)
    raise RuntimeError(
        "Missing OpenCV model files. Run:\n"
        'python "embedding_face_recognition_dual_display.py" download-models\n'
        "Or download these files manually:\n"
        f"{missing_lines}"
    )


def create_detector(frame_size: Tuple[int, int]):
    ensure_models(download=False)
    width, height = frame_size
    return cv2.FaceDetectorYN.create(
        str(YUNET_MODEL_PATH),
        "",
        (max(1, width), max(1, height)),
        0.85,
        0.3,
        5000,
    )


def create_recognizer():
    ensure_models(download=False)
    return cv2.FaceRecognizerSF.create(str(SFACE_MODEL_PATH), "")


def _make_min_size(size: int) -> Tuple[int, int]:
    clamped = max(1, int(size))
    return (clamped, clamped)


def _cascade_detect(gray: np.ndarray, min_size: Tuple[int, int]) -> List[Tuple[int, int, int, int]]:
    faces = list(FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=min_size))
    if faces:
        return faces
    return list(PROFILE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=min_size))


def _rotate_gray(gray: np.ndarray, angle: float) -> Tuple[np.ndarray, np.ndarray]:
    h, w = gray.shape[:2]
    center = (w / 2, h / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(gray, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated, matrix


def _map_rotated_box(
    det: Tuple[int, int, int, int],
    inv_matrix: np.ndarray,
    shape: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    x, y, w, h = det
    corners = np.array([[x, y], [x + w, y], [x, y + h], [x + w, y + h]], dtype=np.float32).reshape(-1, 1, 2)
    mapped = cv2.transform(corners, inv_matrix).reshape(-1, 2)
    min_x = int(np.clip(mapped[:, 0].min(), 0, shape[1] - 1))
    max_x = int(np.clip(mapped[:, 0].max(), 0, shape[1] - 1))
    min_y = int(np.clip(mapped[:, 1].min(), 0, shape[0] - 1))
    max_y = int(np.clip(mapped[:, 1].max(), 0, shape[0] - 1))
    return (min_x, min_y, max(1, max_x - min_x), max(1, max_y - min_y))


def detect_with_rotation(gray: np.ndarray, min_face_size: int) -> List[Tuple[int, int, int, int]]:
    min_size = _make_min_size(min_face_size)
    detections = _cascade_detect(gray, min_size)
    if detections:
        return detections
    for angle in ROTATION_ANGLES:
        rotated, matrix = _rotate_gray(gray, angle)
        rotated_dets = _cascade_detect(rotated, min_size)
        if rotated_dets:
            inv_matrix = cv2.invertAffineTransform(matrix)
            return [_map_rotated_box(det, inv_matrix, gray.shape[:2]) for det in rotated_dets]
    mirrored = cv2.flip(gray, 1)
    mirrored_profiles = PROFILE_CASCADE.detectMultiScale(mirrored, scaleFactor=1.1, minNeighbors=5, minSize=min_size)
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
    return (
        int(np.clip(round(x * inv_scale), 0, shape[1] - 1)),
        int(np.clip(round(y * inv_scale), 0, shape[0] - 1)),
        max(1, int(round(w * inv_scale))),
        max(1, int(round(h * inv_scale))),
    )


def detect_faces_fallback(gray: np.ndarray, min_face_size: int = MIN_FACE_SIZE) -> List[Tuple[int, int, int, int]]:
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
    boxes: List[Tuple[int, int, int, int]] = []
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


def infer_body_boxes_from_faces(
    faces: List[np.ndarray],
    frame_shape: Tuple[int, int],
) -> List[Tuple[int, int, int, int]]:
    boxes: List[Tuple[int, int, int, int]] = []
    frame_h, frame_w = frame_shape
    for face in faces:
        x, y, w, h = face_box(face)
        target_w = max(1, int(round(w * 1.25)))
        target_h = max(1, int(round(h * 2.6)))
        target_w = min(target_w, frame_w)
        target_h = min(target_h, frame_h - max(0, y - int(h * 0.1)))
        body_x = max(0, min(frame_w - target_w, x - (target_w - w) // 2))
        body_y = max(0, y - int(h * 0.1))
        boxes.append(expand_body_box((body_x, body_y, target_w, target_h), frame_shape))
    return boxes


def detect_faces(frame: np.ndarray, detector, min_face_size: int = MIN_FACE_SIZE) -> List[np.ndarray]:
    h, w = frame.shape[:2]
    detector.setInputSize((w, h))
    _, detections = detector.detect(frame)
    faces: List[np.ndarray] = []
    if detections is not None:
        for face in detections:
            x, y, fw, fh = face[:4]
            if min(fw, fh) >= min_face_size:
                faces.append(face.astype(np.float32))
    if faces:
        return faces
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fallback = detect_faces_fallback(gray, min_face_size=min_face_size)
    converted: List[np.ndarray] = []
    for x, y, fw, fh in fallback:
        converted.append(
            np.array(
                [
                    float(x), float(y), float(fw), float(fh),
                    float(x), float(y + fh * 0.38),
                    float(x + fw), float(y + fh * 0.38),
                    float(x + fw * 0.5), float(y + fh * 0.58),
                    float(x + fw * 0.32), float(y + fh * 0.82),
                    float(x + fw * 0.68), float(y + fh * 0.82),
                    0.0,
                ],
                dtype=np.float32,
            )
        )
    return converted


def face_box(face: np.ndarray) -> Tuple[int, int, int, int]:
    x, y, w, h = face[:4]
    return (int(round(x)), int(round(y)), int(round(w)), int(round(h)))


def encode_face(frame: np.ndarray, face: np.ndarray, recognizer) -> Optional[np.ndarray]:
    try:
        aligned = recognizer.alignCrop(frame, face)
        feature = recognizer.feature(aligned)
    except cv2.error:
        return None
    if feature is None:
        return None
    vector = np.asarray(feature, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        return None
    return vector / norm


def collect(
    role: str,
    label: str,
    cam_index: int = 0,
    shots: int = 60,
    min_face_size: int = MIN_FACE_SIZE,
):
    ensure_models(download=False)
    role = normalize_role(role)
    save_dir = DATASET_DIR / role / label
    save_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {cam_index}")
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Could not read initial frame from camera index {cam_index}")
    detector = create_detector((frame.shape[1], frame.shape[0]))
    count = 0
    last_save = 0.0
    print(f"[Collect] Capturing {shots} images for '{label}' ({format_role(role)}). Press 'q' to stop early.")
    while count < shots:
        ok, frame = cap.read()
        if not ok:
            break
        faces = detect_faces(frame, detector, min_face_size=min_face_size)
        display = frame.copy()
        for face in faces:
            x, y, w, h = face_box(face)
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
            t = time.time()
            if t - last_save > 0.20:
                crop = frame[max(0, y):max(0, y) + h, max(0, x):max(0, x) + w]
                if crop.size == 0:
                    continue
                fname = save_dir / f"{uuid.uuid4().hex}.jpg"
                cv2.imwrite(str(fname), crop)
                count += 1
                last_save = t
                put_label(display, f"Saved {count}/{shots}", (x, max(22, y)))
        cv2.imshow("Collect", display)
        if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
            break
    cap.release()
    cv2.destroyAllWindows()
    print(f"[Collect] Saved {count} images in {save_dir}")


def collect_webrtc(
    role: str,
    label: str,
    shots: int = 60,
    min_face_size: int = MIN_FACE_SIZE,
):
    ensure_models(download=False)
    role = normalize_role(role)
    save_dir = DATASET_DIR / role / label
    save_dir.mkdir(parents=True, exist_ok=True)
    frame_queue: Queue = Queue(maxsize=1)
    loop, asyncio_thread, session, connection_state = start_go2_webrtc_camera(frame_queue)

    detector = None
    count = 0
    last_save = 0.0
    cv2.namedWindow("Collect", cv2.WINDOW_NORMAL)
    try:
        while count < shots:
            frame = get_latest_frame(frame_queue)
            if frame is None:
                if connection_state["error"] is not None:
                    raise RuntimeError(f"Go2 video connection failed: {connection_state['error']}")
                mode = connection_state.get("mode") or "starting"
                wait_view = make_waiting_view(f"Waiting for Go2 camera... ({mode})", size=(960, 540))
                cv2.imshow("Collect", wait_view)
                if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                    break
                time.sleep(0.01)
                continue
            if connection_state.get("mode"):
                cv2.setWindowTitle("Collect", f"Collect ({connection_state['mode']})")
            if detector is None:
                detector = create_detector((frame.shape[1], frame.shape[0]))
            faces = detect_faces(frame, detector, min_face_size=min_face_size)
            display = frame.copy()
            for face in faces:
                x, y, w, h = face_box(face)
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                t = time.time()
                if t - last_save > 0.20:
                    crop = frame[max(0, y):max(0, y) + h, max(0, x):max(0, x) + w]
                    if crop.size == 0:
                        continue
                    fname = save_dir / f"{uuid.uuid4().hex}.jpg"
                    cv2.imwrite(str(fname), crop)
                    count += 1
                    last_save = t
                    put_label(display, f"Saved {count}/{shots}", (x, max(22, y)))
            cv2.imshow("Collect", display)
            if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                break
    finally:
        stop_go2_webrtc_camera(loop, session)
        cv2.destroyAllWindows()
    print(f"[Collect] Saved {count} images in {save_dir}")


def collect_ssh(
    role: str,
    label: str,
    shots: int = 60,
    min_face_size: int = MIN_FACE_SIZE,
):
    ensure_models(download=False)
    ensure_remote_go2_video_client()
    role = normalize_role(role)
    save_dir = DATASET_DIR / role / label
    save_dir.mkdir(parents=True, exist_ok=True)
    last_remote = ""
    detector = None
    count = 0
    last_save = 0.0
    cv2.namedWindow("Collect", cv2.WINDOW_NORMAL)
    try:
        while count < shots:
            frame, last_remote = pull_latest_ssh_frame(last_remote)
            if frame is None:
                time.sleep(0.08)
                continue
            if detector is None:
                detector = create_detector((frame.shape[1], frame.shape[0]))
            faces = detect_faces(frame, detector, min_face_size=min_face_size)
            display = frame.copy()
            for face in faces:
                x, y, w, h = face_box(face)
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                t = time.time()
                if t - last_save > 0.20:
                    crop = frame[max(0, y):max(0, y) + h, max(0, x):max(0, x) + w]
                    if crop.size == 0:
                        continue
                    fname = save_dir / f"{uuid.uuid4().hex}.jpg"
                    cv2.imwrite(str(fname), crop)
                    count += 1
                    last_save = t
                    put_label(display, f"Saved {count}/{shots}", (x, max(22, y)))
            cv2.imshow("Collect", display)
            if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                break
    finally:
        stop_remote_go2_video_client()
        cv2.destroyAllWindows()
    print(f"[Collect] Saved {count} images in {save_dir}")


def collect_go2(
    role: str,
    label: str,
    shots: int = 60,
    min_face_size: int = MIN_FACE_SIZE,
):
    ensure_models(download=False)
    role = normalize_role(role)
    save_dir = DATASET_DIR / role / label
    save_dir.mkdir(parents=True, exist_ok=True)
    frame_queue: Queue = Queue(maxsize=1)
    hybrid_state = start_go2_hybrid_camera(frame_queue)
    begin_go2_hybrid_camera(hybrid_state)

    detector = None
    count = 0
    last_save = 0.0
    cv2.namedWindow("Collect", cv2.WINDOW_NORMAL)
    try:
        while count < shots:
            frame = get_latest_go2_hybrid_frame(hybrid_state)
            if frame is None:
                mode = go2_hybrid_status(hybrid_state)
                wait_view = make_waiting_view(f"Waiting for Go2 camera... ({mode})", size=(960, 540))
                cv2.imshow("Collect", wait_view)
                if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                    break
                time.sleep(0.01 if not hybrid_state.get("using_ssh") else 0.08)
                continue
            cv2.setWindowTitle("Collect", f"Collect ({go2_hybrid_status(hybrid_state)})")
            if detector is None:
                detector = create_detector((frame.shape[1], frame.shape[0]))
            faces = detect_faces(frame, detector, min_face_size=min_face_size)
            display = frame.copy()
            for face in faces:
                x, y, w, h = face_box(face)
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                t = time.time()
                if t - last_save > 0.20:
                    crop = frame[max(0, y):max(0, y) + h, max(0, x):max(0, x) + w]
                    if crop.size == 0:
                        continue
                    fname = save_dir / f"{uuid.uuid4().hex}.jpg"
                    cv2.imwrite(str(fname), crop)
                    count += 1
                    last_save = t
                    put_label(display, f"Saved {count}/{shots}", (x, max(22, y)))
            cv2.imshow("Collect", display)
            if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                break
    finally:
        stop_go2_hybrid_camera(hybrid_state)
        cv2.destroyAllWindows()
    print(f"[Collect] Saved {count} images in {save_dir}")


def load_dataset() -> Tuple[np.ndarray, np.ndarray, Dict[int, Dict[str, str]]]:
    ensure_models(download=False)
    recognizer = create_recognizer()
    detector = None
    embeddings: List[np.ndarray] = []
    labels: List[int] = []
    id2identity: Dict[int, Dict[str, str]] = {}
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
            person_id = next_id
            next_id += 1
            id2identity[person_id] = {"name": person_dir.name, "role": format_role(role)}
            for img_path in sorted(person_dir.glob("*.jpg")):
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                if detector is None:
                    detector = create_detector((img.shape[1], img.shape[0]))
                faces = detect_faces(img, detector, min_face_size=MIN_FACE_SIZE)
                if not faces:
                    continue
                best_face = max(faces, key=lambda item: float(item[2] * item[3]))
                feature = encode_face(img, best_face, recognizer)
                if feature is None:
                    continue
                embeddings.append(feature)
                labels.append(person_id)

    if not embeddings:
        raise RuntimeError("No encodable training images found. Collect more images before training.")
    return np.vstack(embeddings).astype(np.float32), np.asarray(labels, dtype=np.int32), id2identity


def train():
    embeddings, labels, id2identity = load_dataset()
    np.savez_compressed(EMBEDDINGS_PATH, embeddings=embeddings, labels=labels)
    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(id2identity, f, indent=2)
    print(f"[Train] Saved embeddings to {EMBEDDINGS_PATH}")
    print(f"[Train] Saved labels to {LABELS_PATH}")


def load_model():
    if not EMBEDDINGS_PATH.is_file() or not LABELS_PATH.is_file():
        raise RuntimeError("Embedding model not trained. Run train after collecting images.")
    payload = np.load(EMBEDDINGS_PATH)
    embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
    labels = np.asarray(payload["labels"], dtype=np.int32)
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        id2identity = {int(k): decode_identity(v) for k, v in json.load(f).items()}
    return embeddings, labels, id2identity


def classify_feature(
    feature: np.ndarray,
    embeddings: np.ndarray,
    labels: np.ndarray,
    id2identity: Dict[int, Dict[str, str]],
    threshold: float,
) -> Tuple[str, float]:
    scores = embeddings @ feature
    best_score_by_id: Dict[int, float] = {}
    for index, label_id in enumerate(labels):
        score = float(scores[index])
        current = best_score_by_id.get(int(label_id))
        if current is None or score > current:
            best_score_by_id[int(label_id)] = score
    if not best_score_by_id:
        return ("Unknown", 0.0)
    best_id, best_score = max(best_score_by_id.items(), key=lambda item: item[1])
    confidence = max(0.0, min(100.0, 100.0 * best_score))
    if best_score < threshold:
        return ("Unknown", confidence)
    identity = id2identity.get(best_id, {"name": "Unknown", "role": "Unknown"})
    return (format_identity(identity["name"], identity["role"]), confidence)


def run(
    cam_index: int = 0,
    threshold: float = MATCH_THRESHOLD,
    min_face_size: int = MIN_FACE_SIZE,
):
    ensure_models(download=False)
    embeddings, labels, id2identity = load_model()
    recognizer = create_recognizer()
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {cam_index}")
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Could not read initial frame from camera index {cam_index}")
    detector = create_detector((frame.shape[1], frame.shape[0]))

    cv2.namedWindow("All Faces", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Top Face", cv2.WINDOW_NORMAL)
    fps = 0.0
    prev = time.time()
    human_persistence = 0
    tracked_body_boxes: List[Dict[str, object]] = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        faces = detect_faces(frame, detector, min_face_size=min_face_size)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
        fps = 0.9 * fps + 0.1 * (1.0 / dt if dt > 0 else 0.0)

        display = frame.copy()
        top_face = None
        top_score = -1.0
        top_name = "?"
        top_nature = None

        for (px, py, pw, ph) in body_boxes:
            cv2.rectangle(display, (px, py), (px + pw, py + ph), (0, 150, 255), 2)
            put_label(display, "Body", (px, py + ph + 18), scale=0.5, bg=(0, 120, 200))

        face_known_present = False
        for face in faces:
            x, y, w, h = face_box(face)
            feature = encode_face(frame, face, recognizer)
            if feature is None:
                display_name, conf = "Unknown", 0.0
            else:
                display_name, conf = classify_feature(feature, embeddings, labels, id2identity, threshold)
            if display_name != "Unknown":
                face_known_present = True
            person_face_box = (x, y, w, h)
            human_overlap = any(_box_iou(person_face_box, person) >= 0.05 for person in body_boxes)
            nature = "Human" if human_overlap or scene_is_human or display_name != "Unknown" else "Non-human"
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
            put_label(display, f"{display_name} ({conf:.1f}%) {nature}", (x, max(22, y)))
            if conf > top_score:
                top_score = conf
                top_name = display_name
                top_face = frame[y:y + h, x:x + w].copy()
                top_nature = nature

        scene_status = "Human" if scene_is_human or face_known_present else "Non-human"
        put_label(display, f"Scene: {scene_status}", (10, 60), scale=0.6, bg=(40, 40, 40))
        if top_face is not None:
            top_view = fit_to_square(top_face, 480)
            top_label = f"Top: {top_name} ({top_score:.1f}%)"
            if top_nature:
                top_label = f"{top_label} [{top_nature}]"
            put_label(top_view, top_label, (10, 30), scale=0.8, bg=(40, 40, 40))
        else:
            top_view = np.zeros((480, 480, 3), dtype=np.uint8)
            put_label(top_view, "No face", (10, 30), scale=0.8, bg=(40, 40, 40))

        put_label(display, f"FPS: {fps:.1f}", (10, 30), scale=0.7, bg=(40, 40, 40))
        cv2.imshow("All Faces", display)
        cv2.imshow("Top Face", top_view)

        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()


def run_webrtc(
    threshold: float = MATCH_THRESHOLD,
    min_face_size: int = MIN_FACE_SIZE,
):
    ensure_models(download=False)
    embeddings, labels, id2identity = load_model()
    recognizer = create_recognizer()
    frame_queue: Queue = Queue(maxsize=1)
    loop, asyncio_thread, session, connection_state = start_go2_webrtc_camera(frame_queue)

    detector = None
    cv2.namedWindow("All Faces", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Top Face", cv2.WINDOW_NORMAL)
    fps = 0.0
    prev = time.time()
    human_persistence = 0
    tracked_body_boxes: List[Dict[str, object]] = []

    try:
        while True:
            frame = get_latest_frame(frame_queue)
            if frame is None:
                if connection_state["error"] is not None:
                    raise RuntimeError(f"Go2 video connection failed: {connection_state['error']}")
                mode = connection_state.get("mode") or "starting"
                wait_main = make_waiting_view(f"Waiting for Go2 camera... ({mode})", size=(1280, 720))
                wait_top = make_waiting_view("No Go2 frame yet", size=(480, 480))
                cv2.imshow("All Faces", wait_main)
                cv2.imshow("Top Face", wait_top)
                if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                    break
                time.sleep(0.01)
                continue
            if connection_state.get("mode"):
                cv2.setWindowTitle("All Faces", f"All Faces ({connection_state['mode']})")
            if detector is None:
                detector = create_detector((frame.shape[1], frame.shape[0]))
            faces = detect_faces(frame, detector, min_face_size=min_face_size)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
            fps = 0.9 * fps + 0.1 * (1.0 / dt if dt > 0 else 0.0)

            display = frame.copy()
            top_face = None
            top_score = -1.0
            top_name = "?"
            top_nature = None

            for (px, py, pw, ph) in body_boxes:
                cv2.rectangle(display, (px, py), (px + pw, py + ph), (0, 150, 255), 2)
                put_label(display, "Body", (px, py + ph + 18), scale=0.5, bg=(0, 120, 200))

            face_known_present = False
            for face in faces:
                x, y, w, h = face_box(face)
                feature = encode_face(frame, face, recognizer)
                if feature is None:
                    display_name, conf = "Unknown", 0.0
                else:
                    display_name, conf = classify_feature(feature, embeddings, labels, id2identity, threshold)
                if display_name != "Unknown":
                    face_known_present = True
                person_face_box = (x, y, w, h)
                human_overlap = any(_box_iou(person_face_box, person) >= 0.05 for person in body_boxes)
                nature = "Human" if human_overlap or scene_is_human or display_name != "Unknown" else "Non-human"
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                put_label(display, f"{display_name} ({conf:.1f}%) {nature}", (x, max(22, y)))
                if conf > top_score:
                    top_score = conf
                    top_name = display_name
                    top_face = frame[y:y + h, x:x + w].copy()
                    top_nature = nature

            scene_status = "Human" if scene_is_human or face_known_present else "Non-human"
            put_label(display, f"Scene: {scene_status}", (10, 60), scale=0.6, bg=(40, 40, 40))
            if top_face is not None:
                top_view = fit_to_square(top_face, 480)
                top_label = f"Top: {top_name} ({top_score:.1f}%)"
                if top_nature:
                    top_label = f"{top_label} [{top_nature}]"
                put_label(top_view, top_label, (10, 30), scale=0.8, bg=(40, 40, 40))
            else:
                top_view = np.zeros((480, 480, 3), dtype=np.uint8)
                put_label(top_view, "No face", (10, 30), scale=0.8, bg=(40, 40, 40))

            put_label(display, f"FPS: {fps:.1f}", (10, 30), scale=0.7, bg=(40, 40, 40))
            cv2.imshow("All Faces", display)
            cv2.imshow("Top Face", top_view)

            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord("q")):
                break
    finally:
        stop_go2_webrtc_camera(loop, session)
        cv2.destroyAllWindows()


def run_ssh(
    threshold: float = MATCH_THRESHOLD,
    min_face_size: int = MIN_FACE_SIZE,
):
    ensure_models(download=False)
    ensure_remote_go2_video_client()
    embeddings, labels, id2identity = load_model()
    recognizer = create_recognizer()
    detector = None
    cv2.namedWindow("All Faces", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Top Face", cv2.WINDOW_NORMAL)
    fps = 0.0
    prev = time.time()
    human_persistence = 0
    tracked_body_boxes: List[Dict[str, object]] = []
    last_remote = ""

    try:
        while True:
            frame, last_remote = pull_latest_ssh_frame(last_remote)
            if frame is None:
                time.sleep(0.08)
                continue
            if detector is None:
                detector = create_detector((frame.shape[1], frame.shape[0]))
            faces = detect_faces(frame, detector, min_face_size=min_face_size)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
            fps = 0.9 * fps + 0.1 * (1.0 / dt if dt > 0 else 0.0)

            display = frame.copy()
            top_face = None
            top_score = -1.0
            top_name = "?"
            top_nature = None

            for (px, py, pw, ph) in body_boxes:
                cv2.rectangle(display, (px, py), (px + pw, py + ph), (0, 150, 255), 2)
                put_label(display, "Body", (px, py + ph + 18), scale=0.5, bg=(0, 120, 200))

            face_known_present = False
            for face in faces:
                x, y, w, h = face_box(face)
                feature = encode_face(frame, face, recognizer)
                if feature is None:
                    display_name, conf = "Unknown", 0.0
                else:
                    display_name, conf = classify_feature(feature, embeddings, labels, id2identity, threshold)
                if display_name != "Unknown":
                    face_known_present = True
                person_face_box = (x, y, w, h)
                human_overlap = any(_box_iou(person_face_box, person) >= 0.05 for person in body_boxes)
                nature = "Human" if human_overlap or scene_is_human or display_name != "Unknown" else "Non-human"
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                put_label(display, f"{display_name} ({conf:.1f}%) {nature}", (x, max(22, y)))
                if conf > top_score:
                    top_score = conf
                    top_name = display_name
                    top_face = frame[y:y + h, x:x + w].copy()
                    top_nature = nature

            scene_status = "Human" if scene_is_human or face_known_present else "Non-human"
            put_label(display, f"Scene: {scene_status}", (10, 60), scale=0.6, bg=(40, 40, 40))
            if top_face is not None:
                top_view = fit_to_square(top_face, 480)
                top_label = f"Top: {top_name} ({top_score:.1f}%)"
                if top_nature:
                    top_label = f"{top_label} [{top_nature}]"
                put_label(top_view, top_label, (10, 30), scale=0.8, bg=(40, 40, 40))
            else:
                top_view = np.zeros((480, 480, 3), dtype=np.uint8)
                put_label(top_view, "No face", (10, 30), scale=0.8, bg=(40, 40, 40))

            put_label(display, f"FPS: {fps:.1f}", (10, 30), scale=0.7, bg=(40, 40, 40))
            cv2.imshow("All Faces", display)
            cv2.imshow("Top Face", top_view)

            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord("q")):
                break
    finally:
        stop_remote_go2_video_client()
        cv2.destroyAllWindows()


def run_go2(
    threshold: float = MATCH_THRESHOLD,
    min_face_size: int = MIN_FACE_SIZE,
):
    ensure_models(download=False)
    embeddings, labels, id2identity = load_model()
    recognizer = create_recognizer()
    detector = None
    frame_queue: Queue = Queue(maxsize=1)
    hybrid_state = start_go2_hybrid_camera(frame_queue)
    begin_go2_hybrid_camera(hybrid_state)
    cv2.namedWindow("All Faces", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Top Face", cv2.WINDOW_NORMAL)
    fps = 0.0
    prev = time.time()
    human_persistence = 0
    tracked_body_boxes: List[Dict[str, object]] = []

    try:
        while True:
            frame = get_latest_go2_hybrid_frame(hybrid_state)
            if frame is None:
                mode = go2_hybrid_status(hybrid_state)
                wait_main = make_waiting_view(f"Waiting for Go2 camera... ({mode})", size=(1280, 720))
                wait_top = make_waiting_view("No Go2 frame yet", size=(480, 480))
                cv2.imshow("All Faces", wait_main)
                cv2.imshow("Top Face", wait_top)
                if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                    break
                time.sleep(0.01 if not hybrid_state.get("using_ssh") else 0.08)
                continue
            cv2.setWindowTitle("All Faces", f"All Faces ({go2_hybrid_status(hybrid_state)})")
            if detector is None:
                detector = create_detector((frame.shape[1], frame.shape[0]))
            faces = detect_faces(frame, detector, min_face_size=min_face_size)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
            fps = 0.9 * fps + 0.1 * (1.0 / dt if dt > 0 else 0.0)

            display = frame.copy()
            top_face = None
            top_score = -1.0
            top_name = "?"
            top_nature = None

            for (px, py, pw, ph) in body_boxes:
                cv2.rectangle(display, (px, py), (px + pw, py + ph), (0, 150, 255), 2)
                put_label(display, "Body", (px, py + ph + 18), scale=0.5, bg=(0, 120, 200))

            face_known_present = False
            for face in faces:
                x, y, w, h = face_box(face)
                feature = encode_face(frame, face, recognizer)
                if feature is None:
                    display_name, conf = "Unknown", 0.0
                else:
                    display_name, conf = classify_feature(feature, embeddings, labels, id2identity, threshold)
                if display_name != "Unknown":
                    face_known_present = True
                person_face_box = (x, y, w, h)
                human_overlap = any(_box_iou(person_face_box, person) >= 0.05 for person in body_boxes)
                nature = "Human" if human_overlap or scene_is_human or display_name != "Unknown" else "Non-human"
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                put_label(display, f"{display_name} ({conf:.1f}%) {nature}", (x, max(22, y)))
                if conf > top_score:
                    top_score = conf
                    top_name = display_name
                    top_face = frame[y:y + h, x:x + w].copy()
                    top_nature = nature

            scene_status = "Human" if scene_is_human or face_known_present else "Non-human"
            put_label(display, f"Scene: {scene_status}", (10, 60), scale=0.6, bg=(40, 40, 40))
            if top_face is not None:
                top_view = fit_to_square(top_face, 480)
                top_label = f"Top: {top_name} ({top_score:.1f}%)"
                if top_nature:
                    top_label = f"{top_label} [{top_nature}]"
                put_label(top_view, top_label, (10, 30), scale=0.8, bg=(40, 40, 40))
            else:
                top_view = np.zeros((480, 480, 3), dtype=np.uint8)
                put_label(top_view, "No face", (10, 30), scale=0.8, bg=(40, 40, 40))

            put_label(display, f"FPS: {fps:.1f}", (10, 30), scale=0.7, bg=(40, 40, 40))
            cv2.imshow("All Faces", display)
            cv2.imshow("Top Face", top_view)

            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord("q")):
                break
    finally:
        stop_go2_hybrid_camera(hybrid_state)
        cv2.destroyAllWindows()


def download_models():
    ensure_models(download=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Embedding face recognition with OpenCV YuNet + SFace.")
    sub = ap.add_subparsers(dest="cmd")

    sub.add_parser("download-models", help="Download YuNet and SFace ONNX model files")

    s1 = sub.add_parser("collect", help="Collect images for one person using a local webcam")
    s1.add_argument("--role", required=True, choices=sorted(VALID_ROLES), help="Person role")
    s1.add_argument("--label", required=True, help="Person name")
    s1.add_argument("--shots", type=int, default=60)
    s1.add_argument("--cam", type=int, default=0)
    s1.add_argument("--min-face-size", type=int, default=MIN_FACE_SIZE)

    sub.add_parser("train", help="Train the embedding database on dataset folders")

    s3 = sub.add_parser("run", help="Run live recognition using a local webcam")
    s3.add_argument("--cam", type=int, default=0)
    s3.add_argument("--threshold", type=float, default=MATCH_THRESHOLD)
    s3.add_argument("--min-face-size", type=int, default=MIN_FACE_SIZE)

    s4 = sub.add_parser("collect-webrtc", help="Collect images using the Go2 WebRTC camera")
    s4.add_argument("--role", required=True, choices=sorted(VALID_ROLES), help="Person role")
    s4.add_argument("--label", required=True, help="Person name")
    s4.add_argument("--shots", type=int, default=60)
    s4.add_argument("--min-face-size", type=int, default=MIN_FACE_SIZE)

    s5 = sub.add_parser("run-webrtc", help="Run live recognition using the Go2 WebRTC camera")
    s5.add_argument("--threshold", type=float, default=MATCH_THRESHOLD)
    s5.add_argument("--min-face-size", type=int, default=MIN_FACE_SIZE)

    s6 = sub.add_parser("collect-ssh", help="Collect images by pulling Go2 frames over SSH/SCP")
    s6.add_argument("--role", required=True, choices=sorted(VALID_ROLES), help="Person role")
    s6.add_argument("--label", required=True, help="Person name")
    s6.add_argument("--shots", type=int, default=60)
    s6.add_argument("--min-face-size", type=int, default=MIN_FACE_SIZE)

    s7 = sub.add_parser("run-ssh", help="Run live recognition by pulling Go2 frames over SSH/SCP")
    s7.add_argument("--threshold", type=float, default=MATCH_THRESHOLD)
    s7.add_argument("--min-face-size", type=int, default=MIN_FACE_SIZE)

    s8 = sub.add_parser("collect-go2", help="Collect images using Go2 with WebRTC first and SSH fallback")
    s8.add_argument("--role", required=True, choices=sorted(VALID_ROLES), help="Person role")
    s8.add_argument("--label", required=True, help="Person name")
    s8.add_argument("--shots", type=int, default=60)
    s8.add_argument("--min-face-size", type=int, default=MIN_FACE_SIZE)

    s9 = sub.add_parser("run-go2", help="Run live recognition using Go2 with WebRTC first and SSH fallback")
    s9.add_argument("--threshold", type=float, default=MATCH_THRESHOLD)
    s9.add_argument("--min-face-size", type=int, default=MIN_FACE_SIZE)

    args = ap.parse_args()
    if args.cmd == "download-models":
        download_models()
    elif args.cmd == "collect":
        collect(role=args.role, label=args.label, cam_index=args.cam, shots=args.shots, min_face_size=args.min_face_size)
    elif args.cmd == "train":
        train()
    elif args.cmd == "run":
        run(cam_index=args.cam, threshold=args.threshold, min_face_size=args.min_face_size)
    elif args.cmd == "collect-webrtc":
        collect_webrtc(role=args.role, label=args.label, shots=args.shots, min_face_size=args.min_face_size)
    elif args.cmd == "run-webrtc":
        run_webrtc(threshold=args.threshold, min_face_size=args.min_face_size)
    elif args.cmd == "collect-ssh":
        collect_ssh(role=args.role, label=args.label, shots=args.shots, min_face_size=args.min_face_size)
    elif args.cmd == "run-ssh":
        run_ssh(threshold=args.threshold, min_face_size=args.min_face_size)
    elif args.cmd == "collect-go2":
        collect_go2(role=args.role, label=args.label, shots=args.shots, min_face_size=args.min_face_size)
    elif args.cmd == "run-go2":
        run_go2(threshold=args.threshold, min_face_size=args.min_face_size)
    else:
        ap.print_help()

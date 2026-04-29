import asyncio
import json
import time
from pathlib import Path
from queue import Queue

import cv2

from embedding_face_recognition_dual_display import (
    GO2_IP,
    get_latest_frame,
    start_go2_webrtc_camera,
    stop_go2_webrtc_camera,
)


MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
FRAME_PATH = MODEL_DIR / "go2_first_frame.jpg"
STATUS_PATH = MODEL_DIR / "go2_camera_probe_status.json"


def write_status(payload: dict):
    STATUS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    frame_queue: Queue = Queue(maxsize=1)
    start_ts = time.time()
    write_status(
        {
            "status": "starting",
            "ip": GO2_IP,
            "started_at": start_ts,
        }
    )

    loop, thread, session, state = start_go2_webrtc_camera(frame_queue)
    deadline = time.time() + 25.0

    try:
        while time.time() < deadline:
            frame = get_latest_frame(frame_queue)
            if frame is not None:
                cv2.imwrite(str(FRAME_PATH), frame)
                payload = {
                    "status": "frame_saved",
                    "ip": GO2_IP,
                    "mode": state.get("mode"),
                    "attempts": state.get("attempts", []),
                    "shape": list(frame.shape),
                    "saved_to": str(FRAME_PATH),
                    "elapsed_s": round(time.time() - start_ts, 2),
                }
                write_status(payload)
                print(json.dumps(payload, indent=2), flush=True)
                return

            if state.get("error") is not None:
                payload = {
                    "status": "error",
                    "ip": GO2_IP,
                    "mode": state.get("mode"),
                    "attempts": state.get("attempts", []),
                    "error": str(state["error"]),
                    "elapsed_s": round(time.time() - start_ts, 2),
                }
                write_status(payload)
                raise RuntimeError(payload["error"])

            time.sleep(0.1)

        payload = {
            "status": "timeout",
            "ip": GO2_IP,
            "mode": state.get("mode"),
            "attempts": state.get("attempts", []),
            "elapsed_s": round(time.time() - start_ts, 2),
            "saved_to": str(FRAME_PATH),
        }
        write_status(payload)
        raise TimeoutError("No Go2 camera frame received within 25 seconds.")
    finally:
        stop_go2_webrtc_camera(loop, session)
        try:
            thread.join(timeout=3.0)
        except Exception:
            pass


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[Go2CameraProbe] {exc}", flush=True)
        raise

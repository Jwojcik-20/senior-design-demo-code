import asyncio
import json
import os
from pathlib import Path

import cv2
from aiortc import MediaStreamTrack
from unitree_webrtc_connect.constants import RTC_TOPIC

from go2_connection import GO2_IP, connect_best_go2, patch_unitree_local_signaling, start_go2_video_stream


patch_unitree_local_signaling()

BRIDGE_DIR = Path(os.environ.get("GO2_BRIDGE_DIR", str(Path("models") / "go2_webrtc_bridge")))
BRIDGE_DIR.mkdir(parents=True, exist_ok=True)
LATEST_FILE = BRIDGE_DIR / "latest.jpg"
READY_FILE = BRIDGE_DIR / "ready.txt"
ERROR_FILE = BRIDGE_DIR / "error.txt"


async def ensure_normal_motion_mode(conn):
    try:
        response = await conn.datachannel.pub_sub.publish_request_new(
            RTC_TOPIC["MOTION_SWITCHER"],
            {"api_id": 1001},
        )
        data = json.loads(response["data"]["data"])
        current_mode = data.get("name")
        if current_mode != "normal":
            await conn.datachannel.pub_sub.publish_request_new(
                RTC_TOPIC["MOTION_SWITCHER"],
                {"api_id": 1002, "parameter": {"name": "normal"}},
            )
            await asyncio.sleep(2.0)
    except Exception:
        pass


async def main():
    READY_FILE.write_text("starting", encoding="utf-8")
    if ERROR_FILE.exists():
        ERROR_FILE.unlink()

    conn, label = await connect_best_go2(ip=GO2_IP)
    READY_FILE.write_text(f"connected:{label}", encoding="utf-8")
    await ensure_normal_motion_mode(conn)
    first_frame = asyncio.Event()

    async def recv_camera_stream(track: MediaStreamTrack):
        while True:
            frame = await track.recv()
            img = frame.to_ndarray(format="bgr24")
            tmp = LATEST_FILE.with_suffix(".tmp.jpg")
            cv2.imwrite(str(tmp), img)
            tmp.replace(LATEST_FILE)
            if not first_frame.is_set():
                READY_FILE.write_text(f"streaming:{label}", encoding="utf-8")
                first_frame.set()

    await start_go2_video_stream(conn, recv_camera_stream, first_frame_timeout=15)
    await asyncio.wait_for(first_frame.wait(), timeout=15)

    try:
        while True:
            await asyncio.sleep(1.0)
    finally:
        await conn.disconnect()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        ERROR_FILE.write_text(str(exc), encoding="utf-8")
        raise

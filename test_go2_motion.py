import asyncio
import json
import logging
import os
import sys

from unitree_webrtc_connect.constants import RTC_TOPIC, SPORT_CMD
from unitree_webrtc_connect.webrtc_driver import UnitreeWebRTCConnection
from go2_connection import GO2_IP, connect_best_go2, patch_unitree_local_signaling

logging.basicConfig(level=logging.FATAL)
HOLD_SECONDS = float(os.getenv("GO2_MOTION_HOLD_SECONDS", "3.0"))
patch_unitree_local_signaling()
LINEAR_SPEED = float(os.getenv("GO2_LINEAR_SPEED", "1.00"))
LATERAL_SPEED = float(os.getenv("GO2_LATERAL_SPEED", "0.72"))
YAW_SPEED = float(os.getenv("GO2_YAW_SPEED", "1.80"))
SKIP_AUTOSTOP = os.getenv("GO2_SKIP_AUTOSTOP", "0") == "1"

DIRECTIONAL_MOVES = {
    "Forward": {"x": LINEAR_SPEED, "y": 0.0, "z": 0.0},
    "Backward": {"x": -LINEAR_SPEED, "y": 0.0, "z": 0.0},
    "Left": {"x": 0.0, "y": 0.0, "z": YAW_SPEED},
    "Right": {"x": 0.0, "y": 0.0, "z": -YAW_SPEED},
    "StrafeLeft": {"x": 0.0, "y": LATERAL_SPEED, "z": 0.0},
    "StrafeRight": {"x": 0.0, "y": -LATERAL_SPEED, "z": 0.0},
}


async def ensure_normal_motion_mode(conn: UnitreeWebRTCConnection):
    response = await conn.datachannel.pub_sub.publish_request_new(
        RTC_TOPIC["MOTION_SWITCHER"],
        {"api_id": 1001},
    )
    data = json.loads(response["data"]["data"])
    current_mode = data.get("name")
    print(f"[Motion] Current mode: {current_mode}")
    if current_mode != "normal":
        response = await conn.datachannel.pub_sub.publish_request_new(
            RTC_TOPIC["MOTION_SWITCHER"],
            {
                "api_id": 1002,
                "parameter": {"name": "normal"},
            },
        )
        print(f"[Motion] Switched to normal mode: {response}")
        await asyncio.sleep(5.0)


async def main():
    if len(sys.argv) < 2:
        available = ", ".join(sorted(set(SPORT_CMD.keys()) | set(DIRECTIONAL_MOVES.keys())))
        print(f"Usage: python test_go2_motion.py <MotionName>\nAvailable: {available}")
        sys.exit(1)

    motion_name = sys.argv[1]
    if motion_name not in SPORT_CMD and motion_name not in DIRECTIONAL_MOVES:
        available = ", ".join(sorted(set(SPORT_CMD.keys()) | set(DIRECTIONAL_MOVES.keys())))
        print(f"Unknown motion '{motion_name}'.\nAvailable: {available}")
        sys.exit(1)

    conn, _ = await connect_best_go2(ip=GO2_IP)
    try:
        print(f"[MotionTest] Connected to {GO2_IP}")

        await ensure_normal_motion_mode(conn)

        print(f"[MotionTest] Sending {motion_name}...")
        if motion_name in DIRECTIONAL_MOVES:
            response = await conn.datachannel.pub_sub.publish_request_new(
                RTC_TOPIC["SPORT_MOD"],
                {"api_id": SPORT_CMD["Move"], "parameter": DIRECTIONAL_MOVES[motion_name]},
            )
        else:
            response = await conn.datachannel.pub_sub.publish_request_new(
                RTC_TOPIC["SPORT_MOD"],
                {"api_id": SPORT_CMD[motion_name]},
            )
        print(f"[MotionTest] Response: {response}")

        if motion_name in DIRECTIONAL_MOVES and SKIP_AUTOSTOP:
            print("[MotionTest] Directional move left running until a separate StopMove command is sent.")
            return

        await asyncio.sleep(HOLD_SECONDS)
        if motion_name in DIRECTIONAL_MOVES:
            stop_response = await conn.datachannel.pub_sub.publish_request_new(
                RTC_TOPIC["SPORT_MOD"],
                {"api_id": SPORT_CMD["StopMove"]},
            )
            print(f"[MotionTest] StopMove response: {stop_response}")
    finally:
        await conn.disconnect()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
        sys.exit(0)

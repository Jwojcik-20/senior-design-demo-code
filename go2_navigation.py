from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from unitree_webrtc_connect.constants import RTC_TOPIC, SPORT_CMD
from unitree_webrtc_connect.webrtc_driver import UnitreeWebRTCConnection

from go2_connection import GO2_IP, connect_best_go2, patch_unitree_local_signaling


BASE_DIR = Path(__file__).resolve().parent
WAYPOINTS_DIR = BASE_DIR / "models"
WAYPOINTS_PATH = WAYPOINTS_DIR / "go2_waypoints.json"
NAV_TARGET_PATH = WAYPOINTS_DIR / "go2_navigation_target.json"
NAV_LINEAR_SPEED = max(0.05, float(os.getenv("GO2_NAV_LINEAR_SPEED", "0.70")))
NAV_YAW_SPEED = max(0.10, float(os.getenv("GO2_NAV_YAW_SPEED", "1.60")))
NAV_DISTANCE_LIMIT = max(0.10, float(os.getenv("GO2_NAV_DISTANCE_LIMIT", "4.0")))
NAV_TURN_TOLERANCE = max(0.01, float(os.getenv("GO2_NAV_TURN_TOLERANCE", "0.08")))
NAV_DISTANCE_TOLERANCE = max(0.01, float(os.getenv("GO2_NAV_DISTANCE_TOLERANCE", "0.08")))
NAV_PHASE_SETTLE = max(0.0, float(os.getenv("GO2_NAV_PHASE_SETTLE", "0.2")))
NAV_MAX_TURN_STEP = max(0.05, float(os.getenv("GO2_NAV_MAX_TURN_STEP", "0.70")))
NAV_MAX_DRIVE_STEP = max(0.05, float(os.getenv("GO2_NAV_MAX_DRIVE_STEP", "0.50")))
NAV_POSE_REFRESH_TIMEOUT = max(0.5, float(os.getenv("GO2_NAV_POSE_REFRESH_TIMEOUT", "2.5")))
NAV_MAX_ITERATIONS = max(3, int(os.getenv("GO2_NAV_MAX_ITERATIONS", "30")))
NAV_PREFERRED_POSE_TOPIC = os.getenv("GO2_NAV_POSE_TOPIC", RTC_TOPIC["ROBOTODOM"]).strip()

POSE_TOPICS = [
    RTC_TOPIC["ROBOTODOM"],
    RTC_TOPIC["LIDAR_MAPPING_ODOM"],
    RTC_TOPIC["LIDAR_LOCALIZATION_ODOM"],
    RTC_TOPIC["SLAM_ODOMETRY"],
]


@dataclass
class PoseSnapshot:
    x: float
    y: float
    z: float
    yaw: float
    topic: str
    captured_at: float
    source: str
    raw: dict[str, Any]

    def to_waypoint(self, name: str) -> dict[str, Any]:
        return {
            "name": name,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "yaw": self.yaw,
            "topic": self.topic,
            "captured_at": self.captured_at,
            "source": self.source,
            "raw": self.raw,
        }


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def load_waypoints() -> list[dict[str, Any]]:
    data = _load_json(WAYPOINTS_PATH, {"waypoints": []})
    waypoints = data.get("waypoints", [])
    return [wp for wp in waypoints if isinstance(wp, dict) and wp.get("name")]


def save_waypoints(waypoints: list[dict[str, Any]]) -> None:
    WAYPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"waypoints": sorted(waypoints, key=lambda wp: str(wp.get("name", "")).lower())}
    WAYPOINTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def upsert_waypoint(name: str, snapshot: PoseSnapshot) -> dict[str, Any]:
    normalized = name.strip()
    if not normalized:
        raise ValueError("Waypoint name cannot be empty.")

    waypoints = load_waypoints()
    replacement = snapshot.to_waypoint(normalized)
    updated = False
    for idx, waypoint in enumerate(waypoints):
        if str(waypoint.get("name", "")).lower() == normalized.lower():
            waypoints[idx] = replacement
            updated = True
            break
    if not updated:
        waypoints.append(replacement)
    save_waypoints(waypoints)
    return replacement


def delete_waypoint(name: str) -> bool:
    normalized = name.strip().lower()
    waypoints = load_waypoints()
    remaining = [wp for wp in waypoints if str(wp.get("name", "")).lower() != normalized]
    if len(remaining) == len(waypoints):
        return False
    save_waypoints(remaining)
    return True


def rename_waypoint(old_name: str, new_name: str) -> dict[str, Any]:
    old_normalized = old_name.strip()
    new_normalized = new_name.strip()
    if not old_normalized or not new_normalized:
        raise ValueError("Waypoint names cannot be empty.")

    waypoints = load_waypoints()
    source_index = None
    target_index = None
    for idx, waypoint in enumerate(waypoints):
        lowered = str(waypoint.get("name", "")).lower()
        if lowered == old_normalized.lower():
            source_index = idx
        if lowered == new_normalized.lower():
            target_index = idx

    if source_index is None:
        raise ValueError(f"Waypoint '{old_name}' was not found.")
    if target_index is not None and target_index != source_index:
        raise ValueError(f"Waypoint '{new_name}' already exists.")

    updated = dict(waypoints[source_index])
    updated["name"] = new_normalized
    waypoints[source_index] = updated
    save_waypoints(waypoints)
    return updated


def get_waypoint(name: str) -> dict[str, Any] | None:
    normalized = name.strip().lower()
    for waypoint in load_waypoints():
        if str(waypoint.get("name", "")).lower() == normalized:
            return waypoint
    return None


def save_navigation_target(payload: dict[str, Any]) -> None:
    WAYPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    NAV_TARGET_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_navigation_target() -> dict[str, Any] | None:
    data = _load_json(NAV_TARGET_PATH, None)
    return data if isinstance(data, dict) else None


def _first_number(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _recursive_find(obj: Any, keys: tuple[str, ...]) -> dict[str, float]:
    found: dict[str, float] = {}

    def walk(node: Any) -> None:
        if len(found) == len(keys):
            return
        if isinstance(node, dict):
            for key, value in node.items():
                lowered = str(key).lower()
                for candidate in keys:
                    if lowered == candidate and candidate not in found:
                        number = _first_number(value)
                        if number is not None:
                            found[candidate] = number
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(obj)
    return found


def _extract_position(payload: dict[str, Any]) -> tuple[float, float, float] | None:
    candidate_dicts: list[dict[str, Any]] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            candidate_dicts.append(node)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(payload)

    for node in candidate_dicts:
        keys = {str(key).lower() for key in node.keys()}
        if {"x", "y"} <= keys:
            x = _first_number(node.get("x"))
            y = _first_number(node.get("y"))
            z = _first_number(node.get("z")) or 0.0
            if x is not None and y is not None:
                return x, y, z

    flat = _recursive_find(payload, ("x", "y", "z"))
    if "x" in flat and "y" in flat:
        return flat["x"], flat["y"], flat.get("z", 0.0)
    return None


def _extract_yaw(payload: dict[str, Any]) -> float | None:
    direct = _recursive_find(payload, ("yaw", "heading", "theta"))
    for key in ("yaw", "heading", "theta"):
        if key in direct:
            return direct[key]

    quaternion_keys = ("qx", "qy", "qz", "qw")
    quaternion = _recursive_find(payload, quaternion_keys)
    if len(quaternion) == 4:
        qx = quaternion["qx"]
        qy = quaternion["qy"]
        qz = quaternion["qz"]
        qw = quaternion["qw"]
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        return math.atan2(siny_cosp, cosy_cosp)

    candidate_dicts: list[dict[str, Any]] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            candidate_dicts.append(node)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(payload)
    for node in candidate_dicts:
        keys = {str(key).lower() for key in node.keys()}
        if {"x", "y", "z", "w"} <= keys:
            qx = _first_number(node.get("x"))
            qy = _first_number(node.get("y"))
            qz = _first_number(node.get("z"))
            qw = _first_number(node.get("w"))
            if None not in (qx, qy, qz, qw):
                siny_cosp = 2.0 * (qw * qz + qx * qy)
                cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
                return math.atan2(siny_cosp, cosy_cosp)
    return None


def extract_pose_snapshot(message: dict[str, Any], topic: str) -> PoseSnapshot | None:
    payload = message
    if isinstance(message, dict) and isinstance(message.get("data"), dict):
        payload = message["data"]

    position = _extract_position(payload)
    if position is None:
        return None
    yaw = _extract_yaw(payload)
    if yaw is None:
        yaw = 0.0

    x, y, z = position
    source = "topic_payload"
    return PoseSnapshot(
        x=x,
        y=y,
        z=z,
        yaw=yaw,
        topic=topic,
        captured_at=time.time(),
        source=source,
        raw=payload if isinstance(payload, dict) else {"payload": payload},
    )


class PoseListener:
    def __init__(self):
        self.latest_pose: PoseSnapshot | None = None
        self.latest_by_topic: dict[str, PoseSnapshot] = {}
        self.update_counts: dict[str, int] = {}
        self.active_topic: str | None = None
        self.preferred_topic = NAV_PREFERRED_POSE_TOPIC if NAV_PREFERRED_POSE_TOPIC in POSE_TOPICS else None
        self._event = asyncio.Event()

    def on_message(self, topic: str, message: dict[str, Any]) -> None:
        snapshot = extract_pose_snapshot(message, topic)
        if snapshot is None:
            return
        self.latest_pose = snapshot
        self.latest_by_topic[topic] = snapshot
        self.update_counts[topic] = self.update_counts.get(topic, 0) + 1
        if self.active_topic is None:
            self.active_topic = self._best_topic()
        self._event.set()

    def _best_topic(self) -> str | None:
        if self.preferred_topic and self.preferred_topic in self.latest_by_topic:
            return self.preferred_topic
        for topic in POSE_TOPICS:
            if topic in self.latest_by_topic:
                return topic
        return next(iter(self.latest_by_topic.keys()), None)

    def get_update_count(self, topic: str | None = None) -> int:
        active = topic or self.active_topic
        if not active:
            return 0
        return self.update_counts.get(active, 0)

    async def wait_for_pose(
        self,
        timeout: float = 8.0,
        *,
        preferred_topic: str | None = None,
        fresh_after: int | None = None,
    ) -> PoseSnapshot:
        deadline = time.monotonic() + timeout
        while True:
            topic = preferred_topic or self.active_topic or self._best_topic()
            if topic and topic in self.latest_by_topic:
                count = self.update_counts.get(topic, 0)
                if fresh_after is None or count > fresh_after:
                    self.active_topic = topic
                    return self.latest_by_topic[topic]
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Timed out waiting for a fresh pose update.")
            self._event.clear()
            await asyncio.wait_for(self._event.wait(), timeout=remaining)




async def ensure_normal_motion_mode(conn: UnitreeWebRTCConnection):
    response = await conn.datachannel.pub_sub.publish_request_new(
        RTC_TOPIC["MOTION_SWITCHER"],
        {"api_id": 1001},
    )
    data = json.loads(response["data"]["data"])
    current_mode = data.get("name")
    if current_mode != "normal":
        await conn.datachannel.pub_sub.publish_request_new(
            RTC_TOPIC["MOTION_SWITCHER"],
            {
                "api_id": 1002,
                "parameter": {"name": "normal"},
            },
        )
        await asyncio.sleep(5.0)


def _normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


async def _send_move(conn: UnitreeWebRTCConnection, *, x: float = 0.0, y: float = 0.0, z: float = 0.0):
    return await conn.datachannel.pub_sub.publish_request_new(
        RTC_TOPIC["SPORT_MOD"],
        {"api_id": SPORT_CMD["Move"], "parameter": {"x": x, "y": y, "z": z}},
    )


async def _stop_move(conn: UnitreeWebRTCConnection):
    return await conn.datachannel.pub_sub.publish_request_new(
        RTC_TOPIC["SPORT_MOD"],
        {"api_id": SPORT_CMD["StopMove"]},
    )


async def _timed_turn(conn: UnitreeWebRTCConnection, angle: float):
    if abs(angle) <= NAV_TURN_TOLERANCE:
        return
    yaw_rate = NAV_YAW_SPEED if angle >= 0 else -NAV_YAW_SPEED
    duration = abs(angle) / NAV_YAW_SPEED
    await _send_move(conn, z=yaw_rate)
    await asyncio.sleep(duration)
    await _stop_move(conn)
    if NAV_PHASE_SETTLE:
        await asyncio.sleep(NAV_PHASE_SETTLE)


async def _timed_forward(conn: UnitreeWebRTCConnection, distance: float):
    if distance <= NAV_DISTANCE_TOLERANCE:
        return
    duration = distance / NAV_LINEAR_SPEED
    await _send_move(conn, x=NAV_LINEAR_SPEED)
    await asyncio.sleep(duration)
    await _stop_move(conn)
    if NAV_PHASE_SETTLE:
        await asyncio.sleep(NAV_PHASE_SETTLE)


async def connect_pose_session(timeout: float = 8.0) -> tuple[UnitreeWebRTCConnection, str, PoseListener, PoseSnapshot]:
    patch_unitree_local_signaling()
    conn, label = await connect_best_go2(ip=GO2_IP)
    listener = PoseListener()
    for topic in POSE_TOPICS:
        conn.datachannel.pub_sub.subscribe(topic, callback=lambda message, topic=topic: listener.on_message(topic, message))
    snapshot = await listener.wait_for_pose(timeout=timeout)
    snapshot.source = label
    return conn, label, listener, snapshot


async def close_pose_session(conn: UnitreeWebRTCConnection):
    try:
        for topic in POSE_TOPICS:
            conn.datachannel.pub_sub.unsubscribe(topic)
    finally:
        await conn.disconnect()


async def capture_current_pose(timeout: float = 8.0) -> PoseSnapshot:
    conn, label, listener, snapshot = await connect_pose_session(timeout=timeout)
    try:
        snapshot.source = label
        return snapshot
    finally:
        await close_pose_session(conn)


async def record_waypoint(name: str, timeout: float = 8.0) -> dict[str, Any]:
    snapshot = await capture_current_pose(timeout=timeout)
    return upsert_waypoint(name, snapshot)


def _build_navigation_target_from_pose(target: dict[str, Any], current: PoseSnapshot) -> dict[str, Any]:
    dx = float(target.get("x", 0.0)) - current.x
    dy = float(target.get("y", 0.0)) - current.y
    dz = float(target.get("z", 0.0)) - current.z
    distance = math.hypot(dx, dy)
    heading_to_target = math.atan2(dy, dx)
    yaw_error = _normalize_angle(float(target.get("yaw", 0.0)) - current.yaw)
    heading_error = _normalize_angle(heading_to_target - current.yaw)
    return {
        "created_at": time.time(),
        "target": target,
        "current": current.to_waypoint("current"),
        "delta": {
            "dx": dx,
            "dy": dy,
            "dz": dz,
            "distance_xy": distance,
            "heading_to_target": heading_to_target,
            "heading_error": heading_error,
            "yaw_error": yaw_error,
        },
        "status": "preview_only",
        "pose_topic": current.topic,
    }


async def build_navigation_target(name: str, timeout: float = 8.0) -> dict[str, Any]:
    target = get_waypoint(name)
    if target is None:
        raise ValueError(f"Waypoint '{name}' was not found.")

    current = await capture_current_pose(timeout=timeout)
    payload = _build_navigation_target_from_pose(target, current)
    save_navigation_target(payload)
    return payload


async def go_to_waypoint(name: str, timeout: float = 8.0) -> dict[str, Any]:
    target = get_waypoint(name)
    if target is None:
        raise ValueError(f"Waypoint '{name}' was not found.")

    conn, label, listener, current = await connect_pose_session(timeout=timeout)
    try:
        await ensure_normal_motion_mode(conn)
        payload = _build_navigation_target_from_pose(target, current)
        steps: list[dict[str, Any]] = []
        payload["connection"] = label
        payload["execution"] = {
            "mode": "closed_loop_incremental",
            "linear_speed": NAV_LINEAR_SPEED,
            "yaw_speed": NAV_YAW_SPEED,
            "distance_limit": NAV_DISTANCE_LIMIT,
            "max_turn_step": NAV_MAX_TURN_STEP,
            "max_drive_step": NAV_MAX_DRIVE_STEP,
            "steps": steps,
        }
        if float(payload["delta"]["distance_xy"]) > NAV_DISTANCE_LIMIT:
            raise ValueError(
                f"Waypoint '{name}' is {float(payload['delta']['distance_xy']):.2f}m away, which exceeds the current safety limit of {NAV_DISTANCE_LIMIT:.2f}m."
            )

        active_topic = listener.active_topic or current.topic
        reached = False
        for _ in range(NAV_MAX_ITERATIONS):
            payload = _build_navigation_target_from_pose(target, current)
            payload["connection"] = label
            payload["execution"] = {
                "mode": "closed_loop_incremental",
                "linear_speed": NAV_LINEAR_SPEED,
                "yaw_speed": NAV_YAW_SPEED,
                "distance_limit": NAV_DISTANCE_LIMIT,
                "max_turn_step": NAV_MAX_TURN_STEP,
                "max_drive_step": NAV_MAX_DRIVE_STEP,
                "steps": steps,
            }
            delta = payload["delta"]
            distance = float(delta["distance_xy"])
            heading_error = float(delta["heading_error"])
            yaw_error = float(delta["yaw_error"])
            if distance <= NAV_DISTANCE_TOLERANCE and abs(yaw_error) <= NAV_TURN_TOLERANCE:
                reached = True
                break

            pose_count = listener.get_update_count(active_topic)
            if distance > NAV_DISTANCE_TOLERANCE:
                if abs(heading_error) > NAV_TURN_TOLERANCE:
                    turn_angle = max(-NAV_MAX_TURN_STEP, min(NAV_MAX_TURN_STEP, heading_error))
                    await _timed_turn(conn, turn_angle)
                    steps.append({"action": "turn_to_target", "angle": turn_angle})
                else:
                    drive_distance = min(NAV_MAX_DRIVE_STEP, distance)
                    await _timed_forward(conn, drive_distance)
                    steps.append({"action": "forward_step", "distance": drive_distance})
            else:
                turn_angle = max(-NAV_MAX_TURN_STEP, min(NAV_MAX_TURN_STEP, yaw_error))
                await _timed_turn(conn, turn_angle)
                steps.append({"action": "final_yaw", "angle": turn_angle})

            current = await listener.wait_for_pose(
                timeout=NAV_POSE_REFRESH_TIMEOUT,
                preferred_topic=active_topic,
                fresh_after=pose_count,
            )
            active_topic = listener.active_topic or current.topic

        if not reached:
            final_payload = _build_navigation_target_from_pose(target, current)
            final_payload["status"] = "incomplete"
            final_payload["connection"] = label
            final_payload["execution"] = {
                "mode": "closed_loop_incremental",
                "linear_speed": NAV_LINEAR_SPEED,
                "yaw_speed": NAV_YAW_SPEED,
                "distance_limit": NAV_DISTANCE_LIMIT,
                "max_turn_step": NAV_MAX_TURN_STEP,
                "max_drive_step": NAV_MAX_DRIVE_STEP,
                "steps": steps,
                "final_topic": active_topic,
            }
            final_payload["executed_at"] = time.time()
            save_navigation_target(final_payload)
            raise RuntimeError(
                f"Waypoint '{name}' did not converge. Remaining distance={float(final_payload['delta']['distance_xy']):.3f}m, "
                f"yaw_error={float(final_payload['delta']['yaw_error']):.3f}rad."
            )

        payload = _build_navigation_target_from_pose(target, current)
        payload["status"] = "executed_closed_loop"
        payload["executed_at"] = time.time()
        payload["connection"] = label
        payload["execution"] = {
            "mode": "closed_loop_incremental",
            "linear_speed": NAV_LINEAR_SPEED,
            "yaw_speed": NAV_YAW_SPEED,
            "distance_limit": NAV_DISTANCE_LIMIT,
            "max_turn_step": NAV_MAX_TURN_STEP,
            "max_drive_step": NAV_MAX_DRIVE_STEP,
            "final_topic": active_topic,
            "steps": steps,
        }
        save_navigation_target(payload)
        return payload
    finally:
        try:
            await _stop_move(conn)
        finally:
            await close_pose_session(conn)


def format_waypoint(waypoint: dict[str, Any]) -> str:
    return (
        f"{waypoint['name']}: "
        f"x={waypoint.get('x', 0.0):.3f}, "
        f"y={waypoint.get('y', 0.0):.3f}, "
        f"z={waypoint.get('z', 0.0):.3f}, "
        f"yaw={waypoint.get('yaw', 0.0):.3f}, "
        f"topic={waypoint.get('topic', '?')}"
    )


def format_pose_snapshot(snapshot: PoseSnapshot) -> str:
    return (
        f"x={snapshot.x:.3f}, y={snapshot.y:.3f}, z={snapshot.z:.3f}, "
        f"yaw={snapshot.yaw:.3f}, topic={snapshot.topic}, source={snapshot.source}"
    )


async def _main_async(args: argparse.Namespace) -> int:
    if args.command == "record":
        waypoint = await record_waypoint(args.name, timeout=args.timeout)
        print(f"Saved waypoint: {format_waypoint(waypoint)}")
        return 0
    if args.command == "current":
        snapshot = await capture_current_pose(timeout=args.timeout)
        print(format_waypoint(snapshot.to_waypoint("current")))
        return 0
    if args.command == "list":
        for waypoint in load_waypoints():
            print(format_waypoint(waypoint))
        return 0
    if args.command == "delete":
        deleted = delete_waypoint(args.name)
        print("Deleted." if deleted else "Waypoint not found.")
        return 0 if deleted else 1
    if args.command == "rename":
        waypoint = rename_waypoint(args.old_name, args.new_name)
        print(f"Renamed waypoint: {format_waypoint(waypoint)}")
        return 0
    if args.command == "goto-preview":
        payload = await build_navigation_target(args.name, timeout=args.timeout)
        delta = payload["delta"]
        print(
            f"Preview target '{args.name}': "
            f"dx={delta['dx']:.3f}, dy={delta['dy']:.3f}, "
            f"distance={delta['distance_xy']:.3f}, yaw_error={delta['yaw_error']:.3f}"
        )
        print(f"Saved target preview to {NAV_TARGET_PATH}")
        return 0
    if args.command == "goto":
        payload = await go_to_waypoint(args.name, timeout=args.timeout)
        delta = payload["delta"]
        print(
            f"Executed target '{args.name}': "
            f"dx={delta['dx']:.3f}, dy={delta['dy']:.3f}, "
            f"distance={delta['distance_xy']:.3f}, yaw_error={delta['yaw_error']:.3f}"
        )
        print(f"Saved execution record to {NAV_TARGET_PATH}")
        return 0
    raise ValueError(f"Unsupported command: {args.command}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Go2 waypoint recording and storage.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    record = subparsers.add_parser("record", help="Capture current pose and save as a waypoint.")
    record.add_argument("name")
    record.add_argument("--timeout", type=float, default=8.0)

    current = subparsers.add_parser("current", help="Print the latest current pose.")
    current.add_argument("--timeout", type=float, default=8.0)

    subparsers.add_parser("list", help="List saved waypoints.")

    delete = subparsers.add_parser("delete", help="Delete a saved waypoint.")
    delete.add_argument("name")

    rename = subparsers.add_parser("rename", help="Rename a saved waypoint.")
    rename.add_argument("old_name")
    rename.add_argument("new_name")

    goto_preview = subparsers.add_parser("goto-preview", help="Build a live delta preview for a saved waypoint.")
    goto_preview.add_argument("name")
    goto_preview.add_argument("--timeout", type=float, default=8.0)

    goto_exec = subparsers.add_parser("goto", help="Execute a closed-loop incremental move to a saved waypoint.")
    goto_exec.add_argument("name")
    goto_exec.add_argument("--timeout", type=float, default=8.0)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import asyncio
import json
import math
import queue
import threading
import tkinter as tk
from tkinter import ttk

from unitree_webrtc_connect.constants import RTC_TOPIC

from go2_navigation import (
    NAV_PREFERRED_POSE_TOPIC,
    PoseSnapshot,
    close_pose_session,
    connect_pose_session,
    format_pose_snapshot,
    load_navigation_target,
    load_waypoints,
)


BG = "#08111d"
PANEL = "#0d1b2e"
FIELD = "#0a1627"
EDGE = "#214666"
TEXT = "#e7f2ff"
MUTED = "#8ea7c4"
ACCENT = "#47d7ff"
ACCENT_2 = "#7cffb2"
WARN = "#ff7a90"
GRID = "#16314d"
ROBOT = "#7cffb2"
WAYPOINT = "#47d7ff"
TARGET = "#ffd166"
TRAIL = "#3fd0a3"
LIDAR = "#5ca7ff"
OCCUPANCY = "#1d6fa7"

MAP_SIZE = 760
MAP_PADDING = 48
TRAIL_LIMIT = 120
LIDAR_POINT_LIMIT = 600
OCCUPANCY_CELL_SIZE = 0.08
OCCUPANCY_MIN_HITS = 2


class LiveMapUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.embedded = bool(getattr(self.root, "embedded_ui", False))
        if not self.embedded:
            self.root.title("Go2 Live Map")
            self.root.geometry("1280x820")
            self.root.minsize(1080, 720)
        self.root.configure(bg=BG)
        self.map_size = 660 if self.embedded else MAP_SIZE

        self.status_var = tk.StringVar(value="Idle")
        self.pose_var = tk.StringVar(value="Pose not connected")
        self.topic_var = tk.StringVar(value=f"Preferred topic: {NAV_PREFERRED_POSE_TOPIC}")
        self.scale_var = tk.StringVar(value="Scale: waiting for pose")

        self.queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.stop_event = threading.Event()
        self.worker: threading.Thread | None = None
        self.latest_pose: PoseSnapshot | None = None
        self.pose_trail: list[tuple[float, float]] = []
        self.lidar_points: list[tuple[float, float]] = []
        self.lidar_world_points: list[tuple[float, float]] = []
        self.occupancy_hits: dict[tuple[int, int], int] = {}

        self._configure_style()
        self._build()
        self._poll_queue()
        self.refresh_waypoints()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _configure_style(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(".", background=BG, foreground=TEXT, fieldbackground=FIELD)
        style.configure("App.TFrame", background=BG)
        style.configure("Panel.TFrame", background=PANEL)
        title_size = 18 if self.embedded else 22
        sub_size = 9 if self.embedded else 10
        style.configure("Title.TLabel", background=BG, foreground=TEXT, font=("Segoe UI Semibold", title_size))
        style.configure("Sub.TLabel", background=BG, foreground=MUTED, font=("Segoe UI", sub_size))
        style.configure("Field.TLabel", background=PANEL, foreground=TEXT, font=("Segoe UI", 10))
        style.configure("Info.TLabel", background=PANEL, foreground=MUTED, font=("Segoe UI", 10))
        style.configure("Section.TLabelframe", background=PANEL, foreground=ACCENT)
        style.configure("Section.TLabelframe.Label", background=PANEL, foreground=ACCENT, font=("Segoe UI Semibold", 11))
        style.configure("Action.TButton", background=PANEL, foreground=TEXT, borderwidth=0, padding=(12, 10), font=("Segoe UI Semibold", 10))
        style.configure("Accent.TButton", background=ACCENT, foreground="#04121d", borderwidth=0, padding=(12, 10), font=("Segoe UI Semibold", 10))
        style.configure("Warn.TButton", background="#3a1622", foreground="#ffd7de", borderwidth=0, padding=(12, 10), font=("Segoe UI Semibold", 10))

    def _build(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        header_pad = (14, 10, 14, 6) if self.embedded else (24, 18, 24, 8)
        body_pad = (14, 6, 14, 14) if self.embedded else (24, 10, 24, 24)
        header = ttk.Frame(self.root, style="App.TFrame", padding=header_pad)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="Go2 Live Map", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            header,
            text="Phase 2 viewer: robot pose and saved waypoints on a live 2D canvas.",
            style="Sub.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=((4, 0) if self.embedded else (6, 0)))

        body = ttk.Frame(self.root, style="App.TFrame", padding=body_pad)
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(0, weight=0)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        left = ttk.LabelFrame(body, text="Controls", style="Section.TLabelframe", padding=16)
        left.grid(row=0, column=0, sticky="ns", padx=(0, 16))
        left.columnconfigure(0, weight=1)

        ttk.Button(left, text="Start Live Map", style="Accent.TButton", command=self.start_stream).grid(row=0, column=0, sticky="ew")
        ttk.Button(left, text="Stop Live Map", style="Warn.TButton", command=self.stop_stream).grid(row=1, column=0, sticky="ew", pady=(10, 0))
        ttk.Button(left, text="Refresh Waypoints", style="Action.TButton", command=self.refresh_waypoints).grid(row=2, column=0, sticky="ew", pady=(10, 0))

        ttk.Label(left, text="Status", style="Field.TLabel").grid(row=3, column=0, sticky="w", pady=(18, 4))
        ttk.Label(left, textvariable=self.status_var, style="Info.TLabel", wraplength=260, justify="left").grid(row=4, column=0, sticky="w")

        ttk.Label(left, text="Live Pose", style="Field.TLabel").grid(row=5, column=0, sticky="w", pady=(18, 4))
        ttk.Label(left, textvariable=self.pose_var, style="Info.TLabel", wraplength=260, justify="left").grid(row=6, column=0, sticky="w")

        ttk.Label(left, text="Topic", style="Field.TLabel").grid(row=7, column=0, sticky="w", pady=(18, 4))
        ttk.Label(left, textvariable=self.topic_var, style="Info.TLabel", wraplength=260, justify="left").grid(row=8, column=0, sticky="w")

        ttk.Label(left, text="Map Scale", style="Field.TLabel").grid(row=9, column=0, sticky="w", pady=(18, 4))
        ttk.Label(left, textvariable=self.scale_var, style="Info.TLabel", wraplength=260, justify="left").grid(row=10, column=0, sticky="w")

        ttk.Label(
            left,
            text="Legend\nGreen: robot\nMint: trail\nCyan: saved waypoints\nYellow: target/selected\nBlue: lidar points\nDeep blue: persistent occupancy",
            style="Info.TLabel",
            justify="left",
        ).grid(row=11, column=0, sticky="w", pady=(18, 0))

        right = ttk.LabelFrame(body, text="Map Canvas", style="Section.TLabelframe", padding=16)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        self.selected_waypoint_var = tk.StringVar(value="Selected: none")
        ttk.Label(right, textvariable=self.selected_waypoint_var, style="Info.TLabel").grid(row=0, column=0, sticky="w")

        self.canvas = tk.Canvas(
            right,
            bg=FIELD,
            highlightthickness=1,
            highlightbackground=EDGE,
            width=self.map_size,
            height=self.map_size,
            relief="flat",
        )
        self.canvas.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        self.canvas.bind("<Button-1>", self._on_canvas_click)

        self._selected_waypoint_name: str | None = None
        self._waypoint_screen_regions: list[tuple[str, tuple[float, float, float, float]]] = []
        self._waypoints_cache = load_waypoints()
        self._draw_map()

    def _poll_queue(self):
        while True:
            try:
                kind, payload = self.queue.get_nowait()
            except queue.Empty:
                break
            if kind == "status":
                self.status_var.set(str(payload))
            elif kind == "pose":
                pose = payload
                assert isinstance(pose, PoseSnapshot)
                self.latest_pose = pose
                self.pose_trail.append((pose.x, pose.y))
                if len(self.pose_trail) > TRAIL_LIMIT:
                    self.pose_trail = self.pose_trail[-TRAIL_LIMIT:]
                self.pose_var.set(format_pose_snapshot(pose))
                self.topic_var.set(f"Preferred topic: {NAV_PREFERRED_POSE_TOPIC}\nActive topic: {pose.topic}")
                self._draw_map()
            elif kind == "lidar":
                self.lidar_points = payload if isinstance(payload, list) else []
                self._accumulate_occupancy(self._lidar_world_points())
                self._draw_map()
            elif kind == "lidar_world":
                self.lidar_world_points = payload if isinstance(payload, list) else []
                self._accumulate_occupancy(self.lidar_world_points)
                self._draw_map()
            elif kind == "error":
                self.status_var.set(f"Error: {payload}")
            elif kind == "stopped":
                self.status_var.set("Live map stopped")
        self.root.after(100, self._poll_queue)

    def refresh_waypoints(self):
        self._waypoints_cache = load_waypoints()
        if self._selected_waypoint_name and not any(wp.get("name") == self._selected_waypoint_name for wp in self._waypoints_cache):
            self._selected_waypoint_name = None
            self.selected_waypoint_var.set("Selected: none")
        self._draw_map()

    def start_stream(self):
        if self.worker and self.worker.is_alive():
            return
        self.stop_event.clear()

        def worker():
            asyncio.run(self._stream_loop())

        self.worker = threading.Thread(target=worker, daemon=True)
        self.worker.start()
        self.status_var.set("Connecting live pose stream...")

    def stop_stream(self):
        self.stop_event.set()

    async def _stream_loop(self):
        try:
            conn, label, listener, snapshot = await connect_pose_session(timeout=8.0)
        except Exception as exc:
            self.queue.put(("error", exc))
            return

        lidar_topics = [RTC_TOPIC["ULIDAR"], RTC_TOPIC["ULIDAR_ARRAY"]]
        for topic in lidar_topics:
            conn.datachannel.pub_sub.subscribe(topic, callback=self._on_lidar_message)
        self.queue.put(("status", f"Connected via {label}"))
        self.queue.put(("pose", snapshot))
        active_topic = listener.active_topic or snapshot.topic
        try:
            while not self.stop_event.is_set():
                pose_count = listener.get_update_count(active_topic)
                try:
                    snapshot = await listener.wait_for_pose(
                        timeout=2.5,
                        preferred_topic=active_topic,
                        fresh_after=pose_count,
                    )
                except Exception:
                    continue
                active_topic = listener.active_topic or snapshot.topic
                self.queue.put(("pose", snapshot))
        finally:
            for topic in lidar_topics:
                try:
                    conn.datachannel.pub_sub.unsubscribe(topic)
                except Exception:
                    pass
            await close_pose_session(conn)
            self.queue.put(("stopped", None))

    def _extract_lidar_points(self, payload):
        points: list[tuple[float, float]] = []

        def try_add(x, y):
            try:
                points.append((float(x), float(y)))
            except Exception:
                return

        def walk(node):
            if len(points) >= LIDAR_POINT_LIMIT:
                return
            if isinstance(node, dict):
                lowered = {str(k).lower(): v for k, v in node.items()}
                if "x" in lowered and "y" in lowered:
                    try_add(lowered["x"], lowered["y"])
                elif "point" in lowered:
                    walk(lowered["point"])
                else:
                    for value in node.values():
                        walk(value)
            elif isinstance(node, list):
                if len(node) >= 2 and all(isinstance(v, (int, float)) for v in node[:2]):
                    try_add(node[0], node[1])
                else:
                    for item in node:
                        walk(item)

        walk(payload)
        return points[:LIDAR_POINT_LIMIT]

    def _extract_voxel_world_points(self, payload):
        if not isinstance(payload, dict):
            return []
        data = payload.get("data")
        if not isinstance(data, dict):
            return []
        positions = data.get("positions")
        origin = payload.get("origin")
        resolution = payload.get("resolution")
        if positions is None or not isinstance(origin, list) or len(origin) < 2 or resolution is None:
            return []
        try:
            resolution = float(resolution)
            ox = float(origin[0])
            oy = float(origin[1])
        except Exception:
            return []

        values = positions.tolist() if hasattr(positions, "tolist") else positions
        if not isinstance(values, list):
            return []

        world_points: list[tuple[float, float]] = []
        triples = min(len(values) // 3, LIDAR_POINT_LIMIT)
        for i in range(triples):
            base = i * 3
            try:
                vx = float(values[base])
                vy = float(values[base + 1])
            except Exception:
                continue
            world_points.append((ox + vx * resolution, oy + vy * resolution))
        return world_points[:LIDAR_POINT_LIMIT]

    def _on_lidar_message(self, message):
        payload = message
        if isinstance(message, dict) and isinstance(message.get("data"), dict):
            payload = message["data"]
        world_points = self._extract_voxel_world_points(payload)
        if world_points:
            self.queue.put(("lidar_world", world_points))
            return
        if isinstance(payload, dict):
            payload = payload.get("data", payload)
        points = self._extract_lidar_points(payload)
        if points:
            self.queue.put(("lidar", points))

    def _map_bounds(self):
        xs: list[float] = []
        ys: list[float] = []
        if self.latest_pose is not None:
            xs.append(self.latest_pose.x)
            ys.append(self.latest_pose.y)
        for x, y in self.pose_trail:
            xs.append(x)
            ys.append(y)
        for waypoint in self._waypoints_cache:
            xs.append(float(waypoint.get("x", 0.0)))
            ys.append(float(waypoint.get("y", 0.0)))
        target = load_navigation_target()
        if isinstance(target, dict):
            target_waypoint = target.get("target")
            if isinstance(target_waypoint, dict):
                xs.append(float(target_waypoint.get("x", 0.0)))
                ys.append(float(target_waypoint.get("y", 0.0)))
        if self.latest_pose is not None:
            for lx, ly in self._lidar_world_points():
                xs.append(lx)
                ys.append(ly)
        for lx, ly in self.lidar_world_points:
            xs.append(lx)
            ys.append(ly)

        if not xs:
            xs = [0.0]
            ys = [0.0]
        min_x = min(xs)
        max_x = max(xs)
        min_y = min(ys)
        max_y = max(ys)
        span_x = max(1.0, max_x - min_x)
        span_y = max(1.0, max_y - min_y)
        margin_x = span_x * 0.2
        margin_y = span_y * 0.2
        return (
            min_x - margin_x,
            max_x + margin_x,
            min_y - margin_y,
            max_y + margin_y,
        )

    def _world_to_canvas(self, x: float, y: float, bounds: tuple[float, float, float, float]) -> tuple[float, float]:
        min_x, max_x, min_y, max_y = bounds
        width = self.map_size - 2 * MAP_PADDING
        height = self.map_size - 2 * MAP_PADDING
        x_ratio = 0.5 if max_x == min_x else (x - min_x) / (max_x - min_x)
        y_ratio = 0.5 if max_y == min_y else (y - min_y) / (max_y - min_y)
        canvas_x = MAP_PADDING + x_ratio * width
        canvas_y = self.map_size - MAP_PADDING - y_ratio * height
        return canvas_x, canvas_y

    def _draw_grid(self):
        for step in range(0, self.map_size + 1, 80):
            self.canvas.create_line(step, 0, step, self.map_size, fill=GRID, width=1)
            self.canvas.create_line(0, step, self.map_size, step, fill=GRID, width=1)

    def _lidar_world_points(self):
        if self.latest_pose is None:
            return []
        cos_yaw = math.cos(self.latest_pose.yaw)
        sin_yaw = math.sin(self.latest_pose.yaw)
        world_points: list[tuple[float, float]] = []
        for lx, ly in self.lidar_points[:LIDAR_POINT_LIMIT]:
            wx = self.latest_pose.x + (lx * cos_yaw - ly * sin_yaw)
            wy = self.latest_pose.y + (lx * sin_yaw + ly * cos_yaw)
            world_points.append((wx, wy))
        return world_points

    def _accumulate_occupancy(self, points):
        for wx, wy in points:
            cell = (
                int(round(wx / OCCUPANCY_CELL_SIZE)),
                int(round(wy / OCCUPANCY_CELL_SIZE)),
            )
            self.occupancy_hits[cell] = self.occupancy_hits.get(cell, 0) + 1

    def _draw_occupancy(self, bounds):
        for (cx_idx, cy_idx), hits in self.occupancy_hits.items():
            if hits < OCCUPANCY_MIN_HITS:
                continue
            wx = cx_idx * OCCUPANCY_CELL_SIZE
            wy = cy_idx * OCCUPANCY_CELL_SIZE
            cx, cy = self._world_to_canvas(wx, wy, bounds)
            radius = 3
            self.canvas.create_rectangle(
                cx - radius,
                cy - radius,
                cx + radius,
                cy + radius,
                fill=OCCUPANCY,
                outline="",
            )

    def _draw_map(self):
        self.canvas.delete("all")
        self._draw_grid()
        bounds = self._map_bounds()
        self._waypoint_screen_regions.clear()

        min_x, max_x, min_y, max_y = bounds
        self.scale_var.set(
            f"x:[{min_x:.2f}, {max_x:.2f}]  y:[{min_y:.2f}, {max_y:.2f}]"
        )

        self._draw_occupancy(bounds)

        if len(self.pose_trail) >= 2:
            trail_coords: list[float] = []
            for x, y in self.pose_trail:
                cx, cy = self._world_to_canvas(x, y, bounds)
                trail_coords.extend((cx, cy))
            self.canvas.create_line(*trail_coords, fill=TRAIL, width=2, smooth=True)

        for wx, wy in self._lidar_world_points():
            cx, cy = self._world_to_canvas(wx, wy, bounds)
            self.canvas.create_oval(cx - 1.5, cy - 1.5, cx + 1.5, cy + 1.5, fill=LIDAR, outline="")
        for wx, wy in self.lidar_world_points:
            cx, cy = self._world_to_canvas(wx, wy, bounds)
            self.canvas.create_oval(cx - 1.5, cy - 1.5, cx + 1.5, cy + 1.5, fill=LIDAR, outline="")

        for waypoint in self._waypoints_cache:
            name = str(waypoint.get("name", ""))
            x = float(waypoint.get("x", 0.0))
            y = float(waypoint.get("y", 0.0))
            cx, cy = self._world_to_canvas(x, y, bounds)
            color = TARGET if name == self._selected_waypoint_name else WAYPOINT
            radius = 8 if name == self._selected_waypoint_name else 6
            self.canvas.create_oval(cx - radius, cy - radius, cx + radius, cy + radius, fill=color, outline="")
            self.canvas.create_text(cx + 12, cy - 10, text=name, fill=TEXT, anchor="w", font=("Segoe UI", 10))
            self._waypoint_screen_regions.append((name, (cx - 14, cy - 14, cx + 14, cy + 14)))

        target = load_navigation_target()
        if isinstance(target, dict):
            target_waypoint = target.get("target")
            if isinstance(target_waypoint, dict):
                tx = float(target_waypoint.get("x", 0.0))
                ty = float(target_waypoint.get("y", 0.0))
                cx, cy = self._world_to_canvas(tx, ty, bounds)
                self.canvas.create_oval(cx - 10, cy - 10, cx + 10, cy + 10, outline=TARGET, width=2)
                self.canvas.create_line(cx - 14, cy, cx + 14, cy, fill=TARGET, width=2)
                self.canvas.create_line(cx, cy - 14, cx, cy + 14, fill=TARGET, width=2)
                self.canvas.create_text(cx + 14, cy + 14, text="Target", fill=TARGET, anchor="w", font=("Segoe UI Semibold", 10))

        if self.latest_pose is not None:
            rx, ry = self._world_to_canvas(self.latest_pose.x, self.latest_pose.y, bounds)
            heading_len = 22
            end_x = rx + heading_len * math.cos(self.latest_pose.yaw)
            end_y = ry - heading_len * math.sin(self.latest_pose.yaw)
            self.canvas.create_oval(rx - 9, ry - 9, rx + 9, ry + 9, fill=ROBOT, outline="")
            self.canvas.create_line(rx, ry, end_x, end_y, fill=ROBOT, width=3)
            self.canvas.create_text(rx + 12, ry + 14, text="Robot", fill=ROBOT, anchor="w", font=("Segoe UI Semibold", 10))

    def _on_canvas_click(self, event):
        for name, (x1, y1, x2, y2) in self._waypoint_screen_regions:
            if x1 <= event.x <= x2 and y1 <= event.y <= y2:
                self._selected_waypoint_name = name
                self.selected_waypoint_var.set(f"Selected: {name}")
                self._draw_map()
                return

    def _on_close(self):
        self.stop_stream()
        self.root.after(150, self.root.destroy)


def main():
    root = tk.Tk()
    LiveMapUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

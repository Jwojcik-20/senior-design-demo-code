import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from go2_navigation import (
    NAV_TARGET_PATH,
    NAV_PREFERRED_POSE_TOPIC,
    WAYPOINTS_PATH,
    build_navigation_target,
    capture_current_pose,
    delete_waypoint,
    format_pose_snapshot,
    format_waypoint,
    go_to_waypoint,
    load_waypoints,
    rename_waypoint,
    record_waypoint,
)


BG = "#ffffff"
PANEL = "#f7f7f7"
FIELD = "#ffffff"
EDGE = "#d7d7d7"
TEXT = "#111111"
MUTED = "#4b4b4b"
ACCENT = "#c1121f"
ACCENT_2 = "#1f7a1f"
WARN = "#b00020"


class WaypointUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.embedded = bool(getattr(self.root, "embedded_ui", False))
        if not self.embedded:
            self.root.title("Go2 Waypoint Manager")
            self.root.geometry("980x640")
            self.root.minsize(860, 560)
        self.root.configure(bg=BG)

        self.name_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Idle")
        self.detail_var = tk.StringVar(value=f"Storage: {WAYPOINTS_PATH}")
        self.live_pose_var = tk.StringVar(value=f"Preferred pose topic: {NAV_PREFERRED_POSE_TOPIC}")
        self.queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.current_worker: threading.Thread | None = None

        self._configure_style()
        self._build()
        self.refresh_waypoints()
        self._poll_queue()

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
        style.configure("Title.TLabel", background=BG, foreground=ACCENT, font=("Segoe UI Semibold", title_size))
        style.configure("Sub.TLabel", background=BG, foreground=TEXT, font=("Segoe UI", sub_size))
        style.configure("Field.TLabel", background=PANEL, foreground=TEXT, font=("Segoe UI", 10))
        style.configure("Info.TLabel", background=PANEL, foreground=MUTED, font=("Segoe UI", 10))
        style.configure("Section.TLabelframe", background=PANEL, foreground=ACCENT)
        style.configure("Section.TLabelframe.Label", background=PANEL, foreground=ACCENT, font=("Segoe UI Semibold", 11))
        style.configure("Input.TEntry", fieldbackground=FIELD, foreground=TEXT, insertcolor=ACCENT, bordercolor=EDGE, lightcolor=EDGE, darkcolor=EDGE, padding=6)
        style.configure("Action.TButton", background="#efefef", foreground=TEXT, borderwidth=0, padding=(10, 8), font=("Segoe UI Semibold", 10))
        style.configure("Accent.TButton", background=ACCENT, foreground="#ffffff", borderwidth=0, padding=(10, 8), font=("Segoe UI Semibold", 10))
        style.configure("Warn.TButton", background="#ffe5e5", foreground=WARN, borderwidth=0, padding=(10, 8), font=("Segoe UI Semibold", 10))
        style.configure(
            "Waypoints.Treeview",
            background=FIELD,
            fieldbackground=FIELD,
            foreground=TEXT,
            bordercolor=EDGE,
            lightcolor=EDGE,
            darkcolor=EDGE,
            rowheight=34,
            relief="flat",
            font=("Segoe UI", 10),
        )
        style.configure(
            "Waypoints.Treeview.Heading",
            background="#fff0f0",
            foreground=ACCENT,
            bordercolor=EDGE,
            lightcolor=EDGE,
            darkcolor=EDGE,
            font=("Segoe UI Semibold", 10),
            relief="flat",
        )
        style.map(
            "Waypoints.Treeview",
            background=[("selected", "#ffe5e5")],
            foreground=[("selected", TEXT)],
        )
        style.map(
            "Waypoints.Treeview.Heading",
            background=[("active", "#ffe0e0")],
            foreground=[("active", ACCENT)],
        )

    def _build(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        header_pad = (14, 10, 14, 6) if self.embedded else (24, 18, 24, 8)
        body_pad = (14, 6, 14, 14) if self.embedded else (24, 10, 24, 24)
        header = ttk.Frame(self.root, style="App.TFrame", padding=header_pad)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="Go2 Waypoint Manager", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(header, text="Phase 1: capture and store named robot poses from odometry topics.", style="Sub.TLabel").grid(row=1, column=0, sticky="w", pady=((4, 0) if self.embedded else (6, 0)))

        body = ttk.Frame(self.root, style="App.TFrame", padding=body_pad)
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(0, weight=2)
        body.columnconfigure(1, weight=3)
        body.rowconfigure(0, weight=1)

        controls = ttk.LabelFrame(body, text="Capture", style="Section.TLabelframe", padding=16)
        controls.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
        controls.columnconfigure(0, weight=1)
        controls.columnconfigure(1, weight=1)

        ttk.Label(controls, text="Waypoint Name", style="Field.TLabel").grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Entry(controls, textvariable=self.name_var, style="Input.TEntry").grid(row=1, column=0, columnspan=2, sticky="ew", pady=(5, 12))
        ttk.Button(controls, text="Capture Current Pose", style="Accent.TButton", command=self.capture_waypoint).grid(row=2, column=0, columnspan=2, sticky="ew")
        ttk.Button(controls, text="Read Live Pose", style="Action.TButton", command=self.read_live_pose).grid(row=3, column=0, columnspan=2, sticky="ew", pady=(10, 0))

        ttk.Label(controls, text="Live Pose", style="Field.TLabel").grid(row=4, column=0, columnspan=2, sticky="w", pady=(14, 4))
        ttk.Label(controls, textvariable=self.live_pose_var, style="Info.TLabel", wraplength=280, justify="left").grid(row=5, column=0, columnspan=2, sticky="w")
        ttk.Label(controls, textvariable=self.status_var, style="Info.TLabel").grid(row=6, column=0, columnspan=2, sticky="w", pady=(14, 4))
        ttk.Label(controls, textvariable=self.detail_var, style="Info.TLabel", wraplength=280, justify="left").grid(row=7, column=0, columnspan=2, sticky="w")

        ttk.Button(controls, text="Refresh List", style="Action.TButton", command=self.refresh_waypoints).grid(row=8, column=0, sticky="ew", pady=(18, 0), padx=(0, 6))
        ttk.Button(controls, text="Delete Selected", style="Warn.TButton", command=self.delete_selected).grid(row=8, column=1, sticky="ew", pady=(18, 0), padx=(6, 0))
        ttk.Button(controls, text="Rename Selected", style="Action.TButton", command=self.rename_selected).grid(row=9, column=0, sticky="ew", pady=(10, 0), padx=(0, 6))
        ttk.Button(controls, text="Go To Selected", style="Accent.TButton", command=self.go_to_selected).grid(row=9, column=1, sticky="ew", pady=(10, 0), padx=(6, 0))

        roster = ttk.LabelFrame(body, text="Saved Waypoints", style="Section.TLabelframe", padding=16)
        roster.grid(row=0, column=1, sticky="nsew")
        roster.columnconfigure(0, weight=1)
        roster.rowconfigure(0, weight=1)

        self.tree = ttk.Treeview(roster, columns=("pose",), show="tree headings", height=18, style="Waypoints.Treeview")
        self.tree.heading("#0", text="Name")
        self.tree.heading("pose", text="Pose")
        self.tree.column("#0", width=180, anchor="w")
        self.tree.column("pose", width=460, anchor="w")
        self.tree.grid(row=0, column=0, sticky="nsew")
        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        self.tree.tag_configure("odd", background=FIELD, foreground=TEXT)
        self.tree.tag_configure("even", background="#f4f4f4", foreground=TEXT)

        scrollbar = ttk.Scrollbar(roster, orient="vertical", command=self.tree.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=scrollbar.set)

    def _poll_queue(self):
        while True:
            try:
                kind, payload = self.queue.get_nowait()
            except queue.Empty:
                break
            if kind == "status":
                self.status_var.set(str(payload))
            elif kind == "detail":
                self.detail_var.set(str(payload))
            elif kind == "refresh":
                self.refresh_waypoints()
            elif kind == "error":
                messagebox.showerror("Waypoint Capture", str(payload))
                self.status_var.set("Capture failed")
            elif kind == "done":
                self.status_var.set("Capture complete")
                self.detail_var.set(str(payload))
            elif kind == "pose":
                self.live_pose_var.set(str(payload))
        self.root.after(100, self._poll_queue)

    def refresh_waypoints(self):
        self.tree.delete(*self.tree.get_children())
        for index, waypoint in enumerate(load_waypoints()):
            summary = f"x={waypoint.get('x', 0.0):.3f}, y={waypoint.get('y', 0.0):.3f}, yaw={waypoint.get('yaw', 0.0):.3f}"
            tag = "even" if index % 2 == 0 else "odd"
            self.tree.insert("", "end", iid=waypoint["name"], text=waypoint["name"], values=(summary,), tags=(tag,))
        self.detail_var.set(f"Storage: {WAYPOINTS_PATH}")

    def capture_waypoint(self):
        name = self.name_var.get().strip()
        if not name:
            messagebox.showwarning("Waypoint Name", "Enter a waypoint name first.")
            return
        existing = next((wp for wp in load_waypoints() if str(wp.get("name", "")).lower() == name.lower()), None)
        if existing and not messagebox.askyesno("Overwrite Waypoint", f"Waypoint '{name}' already exists. Overwrite it?"):
            self.status_var.set("Idle")
            self.detail_var.set(f"Skipped overwrite for '{name}'.")
            return
        if self.current_worker and self.current_worker.is_alive():
            messagebox.showinfo("Waypoint Capture", "A capture is already running.")
            return

        def worker():
            self.queue.put(("status", f"Capturing pose for '{name}'..."))
            self.queue.put(("detail", "Waiting for odometry from the Go2 connection."))
            try:
                waypoint = asyncio.run(record_waypoint(name))
            except Exception as exc:
                self.queue.put(("error", exc))
                return
            self.queue.put(("refresh", None))
            self.queue.put(("done", format_waypoint(waypoint)))

        import asyncio

        self.current_worker = threading.Thread(target=worker, daemon=True)
        self.current_worker.start()

    def read_live_pose(self):
        if self.current_worker and self.current_worker.is_alive():
            messagebox.showinfo("Live Pose", "Another waypoint task is already running.")
            return

        def worker():
            self.queue.put(("status", "Reading live pose..."))
            self.queue.put(("detail", "Waiting for a fresh pose update from the preferred odometry topic."))
            try:
                snapshot = asyncio.run(capture_current_pose())
            except Exception as exc:
                self.queue.put(("error", exc))
                return
            self.queue.put(("pose", format_pose_snapshot(snapshot)))
            self.queue.put(("status", "Live pose updated"))
            self.queue.put(("detail", f"Pose topic locked to {snapshot.topic}"))

        import asyncio

        self.current_worker = threading.Thread(target=worker, daemon=True)
        self.current_worker.start()

    def delete_selected(self):
        selection = self.tree.selection()
        if not selection:
            return
        name = selection[0]
        if not messagebox.askyesno("Delete Waypoint", f"Delete waypoint '{name}'?"):
            return
        if delete_waypoint(name):
            self.refresh_waypoints()
            self.status_var.set("Deleted")
            self.detail_var.set(f"Removed waypoint '{name}'.")

    def rename_selected(self):
        selection = self.tree.selection()
        if not selection:
            return
        old_name = selection[0]
        new_name = self.name_var.get().strip()
        if not new_name:
            messagebox.showwarning("Rename Waypoint", "Enter the new name in the Waypoint Name field.")
            return
        if new_name.lower() != old_name.lower():
            existing = next((wp for wp in load_waypoints() if str(wp.get("name", "")).lower() == new_name.lower()), None)
            if existing:
                messagebox.showwarning("Rename Waypoint", f"A waypoint named '{new_name}' already exists.")
                return
        try:
            waypoint = rename_waypoint(old_name, new_name)
        except Exception as exc:
            messagebox.showerror("Rename Waypoint", str(exc))
            return
        self.refresh_waypoints()
        self.tree.selection_set(waypoint["name"])
        self.tree.focus(waypoint["name"])
        self.status_var.set("Renamed")
        self.detail_var.set(format_waypoint(waypoint))

    def go_to_selected(self):
        selection = self.tree.selection()
        if not selection:
            return
        name = selection[0]
        if self.current_worker and self.current_worker.is_alive():
            messagebox.showinfo("Go To Waypoint", "Another waypoint task is already running.")
            return

        def worker():
            self.queue.put(("status", f"Executing target '{name}'..."))
            self.queue.put(("detail", "Capturing current pose and running closed-loop turn/drive corrections until the robot converges."))
            try:
                payload = asyncio.run(go_to_waypoint(name))
            except Exception as exc:
                self.queue.put(("error", exc))
                return
            delta = payload["delta"]
            self.queue.put(("status", "Target execution complete"))
            self.queue.put(
                (
                    "detail",
                    f"Target '{name}': dx={delta['dx']:.3f}, dy={delta['dy']:.3f}, "
                    f"distance={delta['distance_xy']:.3f}, yaw_error={delta['yaw_error']:.3f}. "
                    f"Saved to {NAV_TARGET_PATH}",
                )
            )

        import asyncio

        self.current_worker = threading.Thread(target=worker, daemon=True)
        self.current_worker.start()

    def _on_select(self, _event=None):
        selection = self.tree.selection()
        if not selection:
            return
        name = selection[0]
        for waypoint in load_waypoints():
            if waypoint.get("name") == name:
                self.name_var.set(name)
                self.detail_var.set(format_waypoint(waypoint))
                break


def main():
    root = tk.Tk()
    WaypointUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

import os
import queue
import socket
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

try:
    from unitree_webrtc_connect.constants import SPORT_CMD
except Exception:
    SPORT_CMD = {}


BASE_DIR = Path(__file__).resolve().parent
SCRIPT_PATH = BASE_DIR / "test_go2_motion.py"

BG = "#ffffff"
PANEL = "#f7f7f7"
FIELD = "#ffffff"
EDGE = "#d7d7d7"
TEXT = "#111111"
MUTED = "#4b4b4b"
ACCENT = "#c1121f"
ACCENT_2 = "#1f7a1f"
WARN = "#b00020"
LOG_BG = "#ffffff"

MOTION_BUTTONS = [
    "Sit",
    "StandUp",
    "Stretch",
    "Hello",
    "WiggleHips",
    "Dance1",
    "Dance2",
    "Pose",
]

DIRECTION_BUTTONS = [
    ("", "Forward", ""),
    ("Left", "StopMove", "Right"),
    ("StrafeLeft", "Backward", "StrafeRight"),
]
DIRECTIONAL_NAMES = [motion for row in DIRECTION_BUTTONS for motion in row if motion]
ALL_MOTIONS = sorted(set(MOTION_BUTTONS) | set(DIRECTIONAL_NAMES) | set(SPORT_CMD.keys()))
HEALTH_PORTS = (9991, 8081, 22)


class MotionUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.embedded = bool(getattr(self.root, "embedded_ui", False))
        if not self.embedded:
            self.root.title("Go2 Motion Control")
            self.root.geometry("980x700")
            self.root.minsize(860, 600)
        self.root.configure(bg=BG)

        self.log_queue: queue.Queue[str] = queue.Queue()
        self.process_lock = threading.Lock()
        self.active_process: subprocess.Popen | None = None

        self.go2_ip_var = tk.StringVar(value=os.getenv("GO2_IP", "192.168.123.161"))
        self.hold_seconds_var = tk.StringVar(value=os.getenv("GO2_MOTION_HOLD_SECONDS", "1.0"))
        self.connect_timeout_var = tk.StringVar(value=os.getenv("GO2_CONNECT_TIMEOUT", "20"))
        self.datachannel_timeout_var = tk.StringVar(value=os.getenv("GO2_DATACHANNEL_TIMEOUT", "20"))
        self.connect_retries_var = tk.StringVar(value=os.getenv("GO2_CONNECT_RETRIES", "3"))
        self.linear_speed_var = tk.StringVar(value=os.getenv("GO2_LINEAR_SPEED", "1.00"))
        self.lateral_speed_var = tk.StringVar(value=os.getenv("GO2_LATERAL_SPEED", "0.72"))
        self.yaw_speed_var = tk.StringVar(value=os.getenv("GO2_YAW_SPEED", "1.80"))
        self.status_var = tk.StringVar(value="Idle")
        self.command_var = tk.StringVar(value="No motion sent yet.")
        self.health_var = tk.StringVar(value="Link: checking...")
        self.motion_picker_var = tk.StringVar(value="StandUp")
        self._health_job = None
        self._held_motion: str | None = None
        self._hold_active = False

        self._configure_style()
        self._build()
        self._poll_log_queue()
        self._schedule_health_check(initial=True)

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
        style.configure("Health.TLabel", background="#f3f3f3", foreground=TEXT, font=("Segoe UI Semibold", 10), padding=(10, 6))
        style.configure("Field.TLabel", background=PANEL, foreground=TEXT, font=("Segoe UI", 10))
        style.configure("Info.TLabel", background=PANEL, foreground=MUTED, font=("Segoe UI", 10))
        style.configure("Section.TLabelframe", background=PANEL, foreground=ACCENT)
        style.configure("Section.TLabelframe.Label", background=PANEL, foreground=ACCENT, font=("Segoe UI Semibold", 11))
        style.configure(
            "Action.TButton",
            background="#efefef",
            foreground=TEXT,
            borderwidth=0,
            padding=(10, 8),
            font=("Segoe UI Semibold", 10),
        )
        style.map("Action.TButton", background=[("active", "#e2e2e2"), ("pressed", "#d8d8d8")])
        style.configure(
            "Accent.TButton",
            background=ACCENT,
            foreground="#ffffff",
            borderwidth=0,
            padding=(10, 8),
            font=("Segoe UI Semibold", 10),
        )
        style.map("Accent.TButton", background=[("active", "#d11f2c"), ("pressed", "#a20f1a")])
        style.configure(
            "Warn.TButton",
            background="#ffe5e5",
            foreground=WARN,
            borderwidth=0,
            padding=(10, 8),
            font=("Segoe UI Semibold", 10),
        )
        style.map("Warn.TButton", background=[("active", "#ffd6d6"), ("pressed", "#ffcaca")])
        style.configure(
            "Input.TEntry",
            fieldbackground=FIELD,
            foreground=TEXT,
            insertcolor=ACCENT,
            bordercolor=EDGE,
            lightcolor=EDGE,
            darkcolor=EDGE,
            padding=6,
        )
        style.configure(
            "Input.TCombobox",
            fieldbackground=FIELD,
            foreground=TEXT,
            background=FIELD,
            bordercolor=EDGE,
            lightcolor=EDGE,
            darkcolor=EDGE,
            arrowcolor=ACCENT,
            arrowsize=16,
            padding=4,
        )
        style.map(
            "Input.TCombobox",
            foreground=[("readonly", TEXT), ("focus", TEXT)],
            fieldbackground=[("readonly", FIELD), ("focus", FIELD)],
            background=[("readonly", FIELD), ("focus", FIELD)],
            selectforeground=[("readonly", TEXT), ("focus", TEXT)],
            selectbackground=[("readonly", FIELD), ("focus", FIELD)],
        )

    def _build(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        header_pad = (14, 10, 14, 6) if self.embedded else (24, 18, 24, 8)
        body_pad = (14, 6, 14, 14) if self.embedded else (24, 10, 24, 24)
        header = ttk.Frame(self.root, style="App.TFrame", padding=header_pad)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        header.columnconfigure(1, weight=0)

        ttk.Label(header, text="Go2 Motion Control", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            header,
            text="Send basic movement commands through the existing Go2 WebRTC control path.",
            style="Sub.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=((4, 0) if self.embedded else (6, 0)))
        self.health_label = ttk.Label(header, textvariable=self.health_var, style="Health.TLabel")
        self.health_label.grid(row=0, column=1, rowspan=2, sticky="e", padx=(16, 10))
        action_bar = ttk.Frame(header, style="App.TFrame")
        action_bar.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        action_bar.columnconfigure(0, weight=1)
        action_bar.columnconfigure(1, weight=1)
        action_bar.columnconfigure(2, weight=1)
        ttk.Button(
            action_bar,
            text="START ROBOT",
            style="Accent.TButton",
            command=lambda: self.run_motion("StandUp"),
        ).grid(row=0, column=0, sticky="ew", padx=(0, 8))
        ttk.Button(
            action_bar,
            text="STOP ROBOT",
            style="Warn.TButton",
            command=lambda: self.run_motion("StopMove"),
        ).grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(
            action_bar,
            text="STOP PROCESS",
            style="Warn.TButton",
            command=self.stop_active_process,
        ).grid(row=0, column=2, sticky="ew", padx=(8, 0))

        body = ttk.Frame(self.root, style="App.TFrame", padding=body_pad)
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)
        body.rowconfigure(1, weight=1)

        connection = ttk.LabelFrame(body, text="Connection", style="Section.TLabelframe", padding=16)
        connection.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
        connection.columnconfigure(0, weight=1)
        connection.columnconfigure(1, weight=1)

        ttk.Label(connection, text="Go2 IP", style="Field.TLabel").grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Entry(connection, textvariable=self.go2_ip_var, width=24, style="Input.TEntry").grid(row=1, column=0, columnspan=2, sticky="ew", pady=(5, 12))

        ttk.Label(connection, text="Hold Seconds", style="Field.TLabel").grid(row=2, column=0, sticky="w")
        ttk.Combobox(
            connection,
            textvariable=self.hold_seconds_var,
            values=("0.5", "1.0", "1.5", "2.0", "3.0"),
            state="readonly",
            style="Input.TCombobox",
            width=10,
        ).grid(row=3, column=0, sticky="ew", pady=(5, 12), padx=(0, 6))

        ttk.Label(connection, text="Connect Timeout", style="Field.TLabel").grid(row=2, column=1, sticky="w")
        ttk.Combobox(
            connection,
            textvariable=self.connect_timeout_var,
            values=("10", "12", "15", "20", "25", "30"),
            state="readonly",
            style="Input.TCombobox",
            width=10,
        ).grid(row=3, column=1, sticky="ew", pady=(5, 12), padx=(6, 0))

        ttk.Label(connection, text="Datachannel Timeout", style="Field.TLabel").grid(row=4, column=0, sticky="w")
        ttk.Combobox(
            connection,
            textvariable=self.datachannel_timeout_var,
            values=("10", "12", "15", "20", "25", "30"),
            state="readonly",
            style="Input.TCombobox",
            width=10,
        ).grid(row=5, column=0, sticky="ew", pady=(5, 12), padx=(0, 6))

        ttk.Label(connection, text="Connect Retries", style="Field.TLabel").grid(row=4, column=1, sticky="w")
        ttk.Combobox(
            connection,
            textvariable=self.connect_retries_var,
            values=("1", "2", "3", "4", "5"),
            state="readonly",
            style="Input.TCombobox",
            width=10,
        ).grid(row=5, column=1, sticky="ew", pady=(5, 12), padx=(6, 0))

        ttk.Label(connection, text="Linear Speed", style="Field.TLabel").grid(row=6, column=0, sticky="w")
        ttk.Combobox(
            connection,
            textvariable=self.linear_speed_var,
            values=("0.40", "0.60", "0.80", "1.00", "1.20"),
            state="readonly",
            style="Input.TCombobox",
            width=10,
        ).grid(row=7, column=0, sticky="ew", pady=(5, 12), padx=(0, 6))

        ttk.Label(connection, text="Yaw Speed", style="Field.TLabel").grid(row=6, column=1, sticky="w")
        ttk.Combobox(
            connection,
            textvariable=self.yaw_speed_var,
            values=("0.80", "1.20", "1.50", "1.80", "2.00"),
            state="readonly",
            style="Input.TCombobox",
            width=10,
        ).grid(row=7, column=1, sticky="ew", pady=(5, 12), padx=(6, 0))

        ttk.Label(connection, text="Lateral Speed", style="Field.TLabel").grid(row=8, column=0, sticky="w")
        ttk.Combobox(
            connection,
            textvariable=self.lateral_speed_var,
            values=("0.30", "0.45", "0.60", "0.72", "0.90"),
            state="readonly",
            style="Input.TCombobox",
            width=10,
        ).grid(row=9, column=0, sticky="ew", pady=(5, 12), padx=(0, 6))

        status = tk.Label(
            connection,
            textvariable=self.status_var,
            bg="#fff5f5",
            fg=ACCENT,
            padx=12,
            pady=7,
            font=("Segoe UI Semibold", 10),
            relief="flat",
        )
        status.grid(row=10, column=0, columnspan=2, sticky="ew", pady=(2, 0))

        drive = ttk.LabelFrame(body, text="Directional Pad", style="Section.TLabelframe", padding=16)
        drive.grid(row=1, column=0, sticky="nsew", padx=(0, 16), pady=(16, 0))
        drive.columnconfigure(0, weight=1)
        ttk.Label(drive, text="Press and hold to move. Release to stop.", style="Info.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 8))
        pad = ttk.Frame(drive, style="Panel.TFrame")
        pad.grid(row=1, column=0, sticky="ew")
        for c in range(3):
            pad.columnconfigure(c, weight=1)
        for r, row in enumerate(DIRECTION_BUTTONS):
            for c, motion in enumerate(row):
                if not motion:
                    ttk.Frame(pad, style="Panel.TFrame").grid(row=r, column=c, sticky="ew", padx=6, pady=6)
                    continue
                style = "Warn.TButton" if motion == "StopMove" else "Action.TButton"
                if motion != "StopMove":
                    button = ttk.Button(
                        pad,
                        text=motion,
                        style=style,
                    )
                    button.grid(row=r, column=c, sticky="ew", padx=6, pady=6, ipady=5)
                    button.bind("<ButtonPress-1>", lambda _event, name=motion: self.start_hold_motion(name))
                    button.bind("<ButtonRelease-1>", lambda _event: self.stop_hold_motion())
                    button.bind("<Leave>", lambda _event: self.stop_hold_motion())
                    continue
                ttk.Button(
                    pad,
                    text=motion,
                    style=style,
                    command=lambda name=motion: self.run_motion(name),
                ).grid(row=r, column=c, sticky="ew", padx=6, pady=6, ipady=5)

        quick = ttk.LabelFrame(body, text="Quick Motions", style="Section.TLabelframe", padding=16)
        quick.grid(row=1, column=1, sticky="nsew", pady=(16, 0))
        quick.columnconfigure(0, weight=1)
        quick.columnconfigure(1, weight=1)
        ttk.Label(quick, text="Quick Motions", style="Info.TLabel").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))
        ttk.Label(quick, text="Custom Command", style="Field.TLabel").grid(row=1, column=0, columnspan=2, sticky="w")
        ttk.Combobox(
            quick,
            textvariable=self.motion_picker_var,
            values=ALL_MOTIONS,
            state="readonly",
            style="Input.TCombobox",
            width=24,
        ).grid(row=2, column=0, sticky="ew", pady=(5, 12), padx=(0, 6))
        ttk.Button(
            quick,
            text="Run Selected",
            style="Accent.TButton",
            command=self.run_selected_motion,
        ).grid(row=2, column=1, sticky="ew", pady=(5, 12), padx=(6, 0))

        for idx, motion in enumerate(MOTION_BUTTONS):
            row = 3 + idx // 2
            col = idx % 2
            style = "Accent.TButton" if motion in {"Sit", "StandUp"} else "Action.TButton"
            ttk.Button(
                quick,
                text=motion,
                style=style,
                command=lambda name=motion: self.run_motion(name),
            ).grid(row=row, column=col, sticky="ew", pady=(0, 8), padx=(0, 6) if col == 0 else (6, 0))

        output = ttk.LabelFrame(body, text="Run Log", style="Section.TLabelframe", padding=12)
        output.grid(row=0, column=1, sticky="nsew")
        output.columnconfigure(0, weight=1)
        output.rowconfigure(1, weight=1)

        meta = ttk.Frame(output, style="Panel.TFrame")
        meta.grid(row=0, column=0, sticky="ew")
        meta.columnconfigure(1, weight=1)
        ttk.Label(meta, text="Command", style="Info.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(meta, textvariable=self.command_var, style="Info.TLabel").grid(row=0, column=1, sticky="e")

        self.log_text = ScrolledText(
            output,
            wrap=tk.WORD,
            font=("Cascadia Code", 9),
            bg=LOG_BG,
            fg=TEXT,
            insertbackground=ACCENT,
            relief="flat",
            borderwidth=0,
            height=18,
            padx=10,
            pady=10,
        )
        self.log_text.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        self.log_text.configure(state="disabled")

        ttk.Button(output, text="Clear Log", style="Action.TButton", command=self.clear_log).grid(row=2, column=0, sticky="w", pady=(10, 0))

    def _append_log(self, text: str):
        self.log_text.configure(state="normal")
        self.log_text.insert(tk.END, text)
        self.log_text.see(tk.END)
        self.log_text.configure(state="disabled")

    def _poll_log_queue(self):
        while True:
            try:
                line = self.log_queue.get_nowait()
            except queue.Empty:
                break
            if line.startswith("__STATUS__"):
                self.status_var.set(line.replace("__STATUS__", "", 1))
                continue
            if line.startswith("__HEALTH__"):
                self._set_health_label(line.replace("__HEALTH__", "", 1))
                continue
            if line.startswith("__HEALTH_TICK__"):
                self._schedule_health_check()
                continue
            if line.startswith("__COMMAND__"):
                self.run_motion(line.replace("__COMMAND__", "", 1))
                continue
            self._append_log(line + "\n")
        self.root.after(100, self._poll_log_queue)

    def clear_log(self):
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", tk.END)
        self.log_text.configure(state="disabled")

    def _spawn(self, motion_name: str, *, skip_autostop: bool = False) -> bool:
        with self.process_lock:
            if self.active_process is not None and self.active_process.poll() is None:
                self._append_log("[Busy] Stop the active command first.\n")
                return False

            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            env["GO2_IP"] = self.go2_ip_var.get().strip()
            env["GO2_MOTION_HOLD_SECONDS"] = self.hold_seconds_var.get().strip()
            env["GO2_CONNECT_TIMEOUT"] = self.connect_timeout_var.get().strip()
            env["GO2_DATACHANNEL_TIMEOUT"] = self.datachannel_timeout_var.get().strip()
            env["GO2_CONNECT_RETRIES"] = self.connect_retries_var.get().strip()
            env["GO2_LINEAR_SPEED"] = self.linear_speed_var.get().strip()
            env["GO2_LATERAL_SPEED"] = self.lateral_speed_var.get().strip()
            env["GO2_YAW_SPEED"] = self.yaw_speed_var.get().strip()
            if skip_autostop:
                env["GO2_SKIP_AUTOSTOP"] = "1"

            command = [sys.executable, str(SCRIPT_PATH), motion_name]
            self.status_var.set("Running")
            self.command_var.set(" ".join(command))
            self._append_log(f"\n>>> {' '.join(command)}\n")
            process = subprocess.Popen(
                command,
                cwd=str(BASE_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
            self.active_process = process

        def reader():
            try:
                assert process.stdout is not None
                for line in process.stdout:
                    self.log_queue.put(line.rstrip("\n"))
            finally:
                return_code = process.wait()
                self.log_queue.put(f"[Process exited with code {return_code}]")
                self.log_queue.put("__STATUS__Idle")
                with self.process_lock:
                    if self.active_process is process:
                        self.active_process = None

        threading.Thread(target=reader, daemon=True).start()
        return True

    def run_motion(self, motion_name: str):
        self._spawn(motion_name)

    def run_selected_motion(self):
        motion_name = self.motion_picker_var.get().strip()
        if motion_name:
            self.run_motion(motion_name)

    def start_hold_motion(self, motion_name: str):
        if self._hold_active:
            return
        if self._spawn(motion_name, skip_autostop=True):
            self._hold_active = True
            self._held_motion = motion_name
            self.status_var.set(f"Holding {motion_name}")

    def stop_hold_motion(self):
        if not self._hold_active:
            return
        held_motion = self._held_motion
        self._hold_active = False
        self._held_motion = None
        if held_motion:
            self._append_log(f"[Hold] Released {held_motion}\n")
        self._queue_stop_when_ready()

    def stop_active_process(self):
        with self.process_lock:
            process = self.active_process
        if process is None or process.poll() is not None:
            return
        process.terminate()
        self.status_var.set("Stopping")
        self._append_log("[Requested process termination]\n")

    def _schedule_health_check(self, initial: bool = False):
        if self._health_job is not None:
            self.root.after_cancel(self._health_job)
        delay_ms = 200 if initial else 4000
        self._health_job = self.root.after(delay_ms, self._start_health_check)

    def _start_health_check(self):
        ip = self.go2_ip_var.get().strip()
        self.health_var.set(f"Link: checking {ip}")

        def worker():
            if not ip:
                self.log_queue.put("__HEALTH__IP missing")
                self.log_queue.put("__HEALTH_TICK__")
                return
            for port in HEALTH_PORTS:
                try:
                    with socket.create_connection((ip, port), timeout=1.0):
                        self.log_queue.put(f"__HEALTH__Reachable on {port}")
                        self.log_queue.put("__HEALTH_TICK__")
                        return
                except OSError:
                    continue
            self.log_queue.put("__HEALTH__Offline")
            self.log_queue.put("__HEALTH_TICK__")

        threading.Thread(target=worker, daemon=True).start()

    def _set_health_label(self, text: str):
        self.health_var.set(f"Link: {text}")
        if "Reachable" in text:
            self.health_label.configure(foreground=ACCENT_2)
        elif "checking" in text.lower():
            self.health_label.configure(foreground=ACCENT)
        else:
            self.health_label.configure(foreground=WARN)

    def _queue_stop_when_ready(self):
        def worker():
            for _ in range(40):
                with self.process_lock:
                    process = self.active_process
                if process is None or process.poll() is not None:
                    self.log_queue.put("__COMMAND__StopMove")
                    return
                threading.Event().wait(0.05)
            self.log_queue.put("__COMMAND__StopMove")

        threading.Thread(target=worker, daemon=True).start()


def main():
    root = tk.Tk()
    MotionUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

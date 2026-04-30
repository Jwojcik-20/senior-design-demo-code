import os
import queue
import shutil
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText


BASE_DIR = Path(__file__).resolve().parent
SCRIPT_PATH = BASE_DIR / "embedding_face_recognition_dual_display.py"
DISPLAY_SCRIPT_PATH = BASE_DIR / "display_video_channel_with_faces_login.py"
DATASET_DIR = BASE_DIR / "dataset"
VALID_ROLES = ("student", "staff", "guest")
CONNECTION_MODES = ("hybrid", "auto", "localsta", "remote", "ssh")
BG = "#ffffff"
PANEL = "#f7f7f7"
PANEL_ALT = "#f2f2f2"
TEXT = "#111111"
MUTED = "#4b4b4b"
ACCENT = "#c1121f"
ACCENT_2 = "#1f7a1f"
WARN = "#b00020"
FIELD = "#ffffff"
FIELD_EDGE = "#d7d7d7"
LOG_BG = "#ffffff"


class ProtocolUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.embedded = bool(getattr(self.root, "embedded_ui", False))
        if not self.embedded:
            self.root.title("Embedding Protocol Manager")
            self.root.geometry("1200x720")
            self.root.minsize(980, 620)
        self.root.configure(bg=BG)

        self.log_queue: queue.Queue[str] = queue.Queue()
        self.process_lock = threading.Lock()
        self.active_process: subprocess.Popen | None = None

        self.role_var = tk.StringVar(value="student")
        self.label_var = tk.StringVar(value="")
        self.shots_var = tk.StringVar(value="80")
        self.cam_var = tk.StringVar(value="0")
        self.go2_ip_var = tk.StringVar(value=os.getenv("GO2_IP", "192.168.123.161"))
        self.go2_mode_var = tk.StringVar(value=os.getenv("GO2_CONNECTION_MODE", "hybrid"))
        self.go2_serial_var = tk.StringVar(value=os.getenv("GO2_SERIAL", ""))
        self.unitree_email_var = tk.StringVar(value=os.getenv("UNITREE_EMAIL", ""))
        self.unitree_pass_var = tk.StringVar(value=os.getenv("UNITREE_PASS", ""))
        self.min_face_size_var = tk.StringVar(value="50")
        self.status_var = tk.StringVar(value="Idle")
        self.workflow_hint_var = tk.StringVar(value="")
        self.roster_index_map: dict[int, Path | None] = {}

        self._configure_style()
        self._build()
        self.refresh_roster()
        self._poll_log_queue()

    def _configure_style(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(".", background=BG, foreground=TEXT, fieldbackground=FIELD)
        style.configure("App.TFrame", background=BG)
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("PanelAlt.TFrame", background=PANEL_ALT)
        title_size = 18 if self.embedded else 22
        sub_size = 9 if self.embedded else 10
        style.configure("Title.TLabel", background=BG, foreground=ACCENT, font=("Segoe UI Semibold", title_size))
        style.configure("Sub.TLabel", background=BG, foreground=TEXT, font=("Segoe UI", sub_size))
        style.configure("Section.TLabelframe", background=PANEL, foreground=ACCENT)
        style.configure("Section.TLabelframe.Label", background=PANEL, foreground=ACCENT, font=("Segoe UI Semibold", 11))
        style.configure("Info.TLabel", background=PANEL, foreground=MUTED, font=("Segoe UI", 10))
        style.configure("Field.TLabel", background=PANEL, foreground=TEXT, font=("Segoe UI", 10))
        style.configure(
            "Action.TButton",
            background=PANEL_ALT,
            foreground=TEXT,
            borderwidth=0,
            focusthickness=0,
            padding=(10, 8),
            font=("Segoe UI Semibold", 10),
        )
        style.map("Action.TButton", background=[("active", "#e2e2e2"), ("pressed", "#d8d8d8")])
        style.configure(
            "Accent.TButton",
            background=ACCENT,
            foreground="#ffffff",
            borderwidth=0,
            focusthickness=0,
            padding=(10, 8),
            font=("Segoe UI Semibold", 10),
        )
        style.map("Accent.TButton", background=[("active", "#d11f2c"), ("pressed", "#a20f1a")])
        style.configure(
            "Warn.TButton",
            background="#ffe5e5",
            foreground=WARN,
            borderwidth=0,
            focusthickness=0,
            padding=(10, 8),
            font=("Segoe UI Semibold", 10),
        )
        style.map("Warn.TButton", background=[("active", "#ffd6d6"), ("pressed", "#ffcaca")])
        style.configure("TEntry", fieldbackground=FIELD, foreground=TEXT, insertcolor=ACCENT, bordercolor=FIELD_EDGE, lightcolor=FIELD_EDGE, darkcolor=FIELD_EDGE)
        style.configure("TCombobox", fieldbackground=FIELD, foreground=TEXT, arrowcolor=ACCENT, bordercolor=FIELD_EDGE, lightcolor=FIELD_EDGE, darkcolor=FIELD_EDGE)
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", FIELD)],
            foreground=[("readonly", TEXT)],
            selectbackground=[("readonly", FIELD)],
            selectforeground=[("readonly", TEXT)],
            background=[("readonly", FIELD)],
            arrowcolor=[("readonly", ACCENT)],
        )

    def _build(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        header_pad = (14, 10, 14, 6) if self.embedded else (20, 14, 20, 6)
        body_pad = (14, 6, 14, 14) if self.embedded else (20, 8, 20, 20)
        info_wrap = 220 if self.embedded else 250
        header = ttk.Frame(self.root, style="App.TFrame", padding=header_pad)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        header.columnconfigure(1, weight=0)

        ttk.Label(header, text="Embedding Face Protocol Manager", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            header,
            text="Collect human training images, download model assets, and train the embedding recognizer without typing commands.",
            style="Sub.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=((4, 0) if self.embedded else (6, 0)))

        status_chip = tk.Label(
            header,
            textvariable=self.status_var,
            bg="#fff5f5",
            fg=ACCENT,
            padx=16,
            pady=7,
            font=("Segoe UI Semibold", 10),
            relief="flat",
        )
        status_chip.grid(row=0, column=1, rowspan=2, sticky="e")

        body = ttk.Frame(self.root, style="App.TFrame", padding=body_pad)
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(0, weight=2)
        body.columnconfigure(1, weight=3)
        body.columnconfigure(2, weight=2)
        body.rowconfigure(0, weight=1)

        controls = ttk.LabelFrame(body, text="Guided Workflow", style="Section.TLabelframe", padding=18)
        controls.grid(row=0, column=0, sticky="nsw", padx=(0, 12))
        controls.configure(labelanchor="n")
        controls.columnconfigure(0, weight=1)

        ttk.Label(
            controls,
            text="Follow the steps in order. The setup fields below are shared across the collection and Go2 actions.",
            style="Info.TLabel",
            wraplength=info_wrap,
            justify="left",
        ).grid(row=0, column=0, sticky="w", pady=(0, 10))

        fields = ttk.Frame(controls, style="Panel.TFrame")
        fields.grid(row=1, column=0, sticky="ew")
        fields.columnconfigure(0, weight=1)
        fields.columnconfigure(1, weight=1)

        ttk.Label(fields, text="Role", style="Field.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(fields, text="Person Label", style="Field.TLabel").grid(row=0, column=1, sticky="w", padx=(10, 0))
        ttk.Combobox(fields, textvariable=self.role_var, values=VALID_ROLES, width=11, state="readonly").grid(row=1, column=0, sticky="ew", pady=(4, 8))
        ttk.Entry(fields, textvariable=self.label_var, width=16).grid(row=1, column=1, sticky="ew", pady=(4, 8), padx=(8, 0))

        ttk.Label(fields, text="Shots", style="Field.TLabel").grid(row=2, column=0, sticky="w")
        ttk.Label(fields, text="Camera Index", style="Field.TLabel").grid(row=2, column=1, sticky="w", padx=(10, 0))
        ttk.Entry(fields, textvariable=self.shots_var, width=11).grid(row=3, column=0, sticky="ew", pady=(4, 8))
        ttk.Entry(fields, textvariable=self.cam_var, width=11).grid(row=3, column=1, sticky="ew", pady=(4, 8), padx=(8, 0))

        ttk.Label(fields, text="Go2 IP", style="Field.TLabel").grid(row=4, column=0, sticky="w")
        ttk.Label(fields, text="Go2 Mode", style="Field.TLabel").grid(row=4, column=1, sticky="w", padx=(10, 0))
        ttk.Entry(fields, textvariable=self.go2_ip_var, width=16).grid(row=5, column=0, sticky="ew", pady=(4, 8))
        ttk.Combobox(fields, textvariable=self.go2_mode_var, values=CONNECTION_MODES, width=11, state="readonly").grid(row=5, column=1, sticky="ew", pady=(4, 8), padx=(8, 0))

        ttk.Label(fields, text="Go2 Serial", style="Field.TLabel").grid(row=6, column=0, sticky="w")
        ttk.Label(fields, text="Min Face Size", style="Field.TLabel").grid(row=6, column=1, sticky="w", padx=(10, 0))
        ttk.Entry(fields, textvariable=self.go2_serial_var, width=16).grid(row=7, column=0, sticky="ew", pady=(4, 8))
        ttk.Entry(fields, textvariable=self.min_face_size_var, width=11).grid(row=7, column=1, sticky="ew", pady=(4, 8), padx=(8, 0))

        ttk.Label(fields, text="Unitree Email", style="Field.TLabel").grid(row=8, column=0, sticky="w")
        ttk.Label(fields, text="Unitree Password", style="Field.TLabel").grid(row=8, column=1, sticky="w", padx=(10, 0))
        ttk.Entry(fields, textvariable=self.unitree_email_var, width=16).grid(row=9, column=0, sticky="ew", pady=(4, 10))
        ttk.Entry(fields, textvariable=self.unitree_pass_var, width=16, show="*").grid(row=9, column=1, sticky="ew", pady=(4, 10), padx=(8, 0))

        actions = ttk.Frame(controls, style="Panel.TFrame")
        actions.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        actions.columnconfigure(0, weight=1)
        actions.columnconfigure(1, weight=1)

        ttk.Label(actions, text="Workflow Actions", style="Field.TLabel").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))

        action_specs = [
            ("1. Download Models", "Accent.TButton", self.download_models),
            ("2A. Collect Laptop", "Accent.TButton", self.collect_local),
            ("2B. Collect Go2", "Action.TButton", self.collect_go2),
            ("3. Train Model", "Accent.TButton", self.train_model),
            ("4A. Run Laptop", "Action.TButton", self.run_local),
            ("4B. Run Go2", "Accent.TButton", self.run_go2),
        ]

        for index, (label, style_name, command) in enumerate(action_specs):
            row = 1 + index // 2
            column = index % 2
            ttk.Button(
                actions,
                text=label,
                style=style_name,
                command=command,
            ).grid(
                row=row,
                column=column,
                sticky="ew",
                pady=(0, 8),
                padx=(0, 6) if column == 0 else (6, 0),
            )

        self.workflow_hint = tk.Label(
            controls,
            textvariable=self.workflow_hint_var,
            bg="#fff5f5",
            fg=ACCENT,
            anchor="w",
            justify="left",
            padx=12,
            pady=10,
            font=("Segoe UI", 10, "bold"),
            wraplength=290,
        )
        self.workflow_hint.grid(row=3, column=0, sticky="ew", pady=(6, 0))

        output = ttk.LabelFrame(body, text="Run Log", style="Section.TLabelframe", padding=14)
        output.grid(row=0, column=1, sticky="nsew")
        output.columnconfigure(0, weight=1)
        output.rowconfigure(1, weight=1)

        meta = ttk.Frame(output, style="Panel.TFrame")
        meta.grid(row=0, column=0, sticky="ew")
        meta.columnconfigure(1, weight=1)
        self.command_var = tk.StringVar(value="No command started yet.")
        ttk.Label(meta, text="Command", style="Info.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(meta, textvariable=self.command_var, style="Info.TLabel").grid(row=0, column=1, sticky="e")

        self.log_text = ScrolledText(
            output,
            wrap=tk.WORD,
            font=("Cascadia Code", 10),
            bg=LOG_BG,
            fg=TEXT,
            insertbackground=ACCENT,
            relief="flat",
            borderwidth=0,
            padx=14,
            pady=14,
        )
        self.log_text.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        self.log_text.configure(state="disabled")

        footer = ttk.Frame(output, style="Panel.TFrame")
        footer.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        footer.columnconfigure(0, weight=1)
        ttk.Button(footer, text="Clear Log", style="Action.TButton", command=self.clear_log).grid(row=0, column=0, sticky="w")
        ttk.Button(footer, text="Stop Active Command", style="Warn.TButton", command=self.stop_active_process).grid(row=0, column=1, sticky="e")

        roster = ttk.LabelFrame(body, text="System Roster", style="Section.TLabelframe", padding=14)
        roster.grid(row=0, column=2, sticky="nsew")
        roster.columnconfigure(0, weight=1)
        roster.rowconfigure(1, weight=1)

        roster_meta = ttk.Frame(roster, style="Panel.TFrame")
        roster_meta.grid(row=0, column=0, sticky="ew")
        roster_meta.columnconfigure(0, weight=1)
        self.roster_count_var = tk.StringVar(value="0 identities")
        ttk.Label(roster_meta, textvariable=self.roster_count_var, style="Info.TLabel").grid(row=0, column=0, sticky="w")
        roster_actions = ttk.Frame(roster_meta, style="Panel.TFrame")
        roster_actions.grid(row=0, column=1, sticky="e")
        ttk.Button(roster_actions, text="Refresh", style="Action.TButton", command=self.refresh_roster).grid(row=0, column=0, sticky="e", padx=(0, 8))
        ttk.Button(roster_actions, text="Remove Selected", style="Warn.TButton", command=self.remove_selected_person).grid(row=0, column=1, sticky="e")

        self.roster_list = tk.Listbox(
            roster,
            bg=LOG_BG,
            fg=TEXT,
            selectbackground="#ffe5e5",
            selectforeground=TEXT,
            highlightthickness=1,
            highlightbackground=FIELD_EDGE,
            relief="flat",
            font=("Segoe UI", 10),
            activestyle="none",
            width=28,
        )
        self.roster_list.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        self.roster_list.bind("<<ListboxSelect>>", self._on_roster_select)

    def _validate_common(self, require_label: bool) -> dict | None:
        label = self.label_var.get().strip()
        shots = self.shots_var.get().strip()
        cam = self.cam_var.get().strip()
        go2_ip = self.go2_ip_var.get().strip()
        go2_mode = self.go2_mode_var.get().strip().lower()
        go2_serial = self.go2_serial_var.get().strip()
        unitree_email = self.unitree_email_var.get().strip()
        unitree_pass = self.unitree_pass_var.get().strip()
        min_face_size = self.min_face_size_var.get().strip()

        if require_label and not label:
            messagebox.showerror("Missing label", "Enter a person label before starting collection.")
            return None
        try:
            shots_int = int(shots)
            cam_int = int(cam)
            min_face_size_int = int(min_face_size)
        except ValueError:
            messagebox.showerror("Invalid input", "Shots, camera index, and min face size must be integers.")
            return None
        if go2_mode not in CONNECTION_MODES:
            messagebox.showerror("Invalid input", "Go2 mode must be hybrid, auto, localsta, remote, or ssh.")
            return None
        if go2_mode == "remote":
            if not go2_serial:
                messagebox.showerror("Missing serial", "Enter the Go2 serial number for remote mode.")
                return None
            if not unitree_email or not unitree_pass:
                messagebox.showerror("Missing credentials", "Enter the Unitree email and password for remote mode.")
                return None
        return {
            "label": label,
            "shots": shots_int,
            "cam": cam_int,
            "go2_ip": go2_ip,
            "go2_mode": go2_mode,
            "go2_serial": go2_serial,
            "unitree_email": unitree_email,
            "unitree_pass": unitree_pass,
            "min_face_size": min_face_size_int,
            "role": self.role_var.get(),
        }

    def _go2_env(self, data: dict) -> dict:
        env = {
            "GO2_IP": data["go2_ip"],
            "GO2_CONNECTION_MODE": data["go2_mode"],
            "PYTHONIOENCODING": "utf-8",
            "LIVEINTERFACE_SOURCE": "go2",
            "FACE_BACKEND": "embedding",
        }
        if data["go2_serial"]:
            env["GO2_SERIAL"] = data["go2_serial"]
        if data["unitree_email"]:
            env["UNITREE_EMAIL"] = data["unitree_email"]
        if data["unitree_pass"]:
            env["UNITREE_PASS"] = data["unitree_pass"]
        return env

    def _python_cmd(self, *parts: str) -> list[str]:
        return [sys.executable, str(SCRIPT_PATH), *parts]

    def _display_cmd(self) -> list[str]:
        return [sys.executable, str(DISPLAY_SCRIPT_PATH)]

    def _spawn(self, command: list[str], extra_env: dict | None = None):
        with self.process_lock:
            if self.active_process is not None and self.active_process.poll() is None:
                messagebox.showwarning("Busy", "A protocol is already running. Stop it or wait for it to finish.")
                return

            env = os.environ.copy()
            if extra_env:
                env.update(extra_env)

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
                self.log_queue.put("__REFRESH_ROSTER__")
                with self.process_lock:
                    if self.active_process is process:
                        self.active_process = None

        threading.Thread(target=reader, daemon=True).start()

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
            if line == "__REFRESH_ROSTER__":
                self.refresh_roster()
                continue
            self._append_log(line + "\n")
        self.root.after(100, self._poll_log_queue)

    def clear_log(self):
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", tk.END)
        self.log_text.configure(state="disabled")

    def stop_active_process(self):
        with self.process_lock:
            process = self.active_process
        if process is None or process.poll() is not None:
            return
        process.terminate()
        self.status_var.set("Stopping")
        self._append_log("[Requested process termination]\n")

    def _set_hint(self, text: str):
        self.workflow_hint_var.set(text)

    def _on_roster_select(self, event=None):
        selection = self.roster_list.curselection()
        if not selection:
            return
        person_dir = self.roster_index_map.get(selection[0])
        if person_dir is None:
            return
        self.role_var.set(person_dir.parent.name)
        self.label_var.set(person_dir.name)
        image_count = len(list(person_dir.glob("*.jpg")))
        self._set_hint(
            f"Selected {person_dir.name} in {person_dir.parent.name}. "
            f"This person currently has {image_count} images. "
            f"Collect more images, retrain, or remove this person."
        )

    def refresh_roster(self):
        self.roster_list.delete(0, tk.END)
        self.roster_index_map = {}
        total = 0
        total_images = 0
        row_index = 0
        for role in VALID_ROLES:
            role_dir = DATASET_DIR / role
            people = sorted([p for p in role_dir.iterdir() if p.is_dir()], key=lambda p: p.name.lower()) if role_dir.is_dir() else []
            role_count = len(people)
            role_images = 0
            for person_dir in people:
                role_images += len(list(person_dir.glob("*.jpg")))
            self.roster_list.insert(tk.END, f"{role.upper()}  [{role_count} people, {role_images} images]")
            self.roster_index_map[row_index] = None
            row_index += 1
            if people:
                for person_dir in people:
                    image_count = len(list(person_dir.glob("*.jpg")))
                    self.roster_list.insert(tk.END, f"  {person_dir.name}  ({image_count} images)")
                    self.roster_index_map[row_index] = person_dir
                    row_index += 1
                    total += 1
                    total_images += image_count
            else:
                self.roster_list.insert(tk.END, "  [none]")
                self.roster_index_map[row_index] = None
                row_index += 1
        self.roster_count_var.set(f"{total} identities | {total_images} images")
        if total == 0:
            self._set_hint("Start at Step 1, then Step 2 to collect your first person.")
        else:
            self._set_hint("First-time flow: Step 1 download models, Step 2 collect a person, Step 3 train, Step 4 run recognition.")

    def remove_selected_person(self):
        selection = self.roster_list.curselection()
        if not selection:
            messagebox.showinfo("No selection", "Select a person entry from the roster first.")
            return
        index = selection[0]
        person_dir = self.roster_index_map.get(index)
        if person_dir is None:
            messagebox.showinfo("Invalid selection", "Select an actual person entry, not a role header.")
            return
        confirm = messagebox.askyesno(
            "Remove person",
            f"Remove '{person_dir.name}' from role '{person_dir.parent.name}'?\n\nThis deletes the dataset folder and its images.",
        )
        if not confirm:
            return
        try:
            shutil.rmtree(person_dir)
            self._append_log(f"[Roster] Removed {person_dir.parent.name}/{person_dir.name}\n")
            self.refresh_roster()
            self._set_hint("Person removed. Run Step 3 to retrain the model before recognition.")
        except Exception as exc:
            messagebox.showerror("Remove failed", f"Could not remove {person_dir.name}: {exc}")

    def download_models(self):
        self._set_hint("Step 1 running. Wait for the model download to finish, then move to Step 2.")
        self._spawn(self._python_cmd("download-models"))

    def collect_local(self):
        data = self._validate_common(require_label=True)
        if data is None:
            return
        self._set_hint(
            f"Step 2 running on laptop camera for {data['label']} ({data['role']}). "
            "Stand in view of the camera and let it save the requested shots."
        )
        self._spawn(
            self._python_cmd(
                "collect",
                "--role", data["role"],
                "--label", data["label"],
                "--shots", str(data["shots"]),
                "--cam", str(data["cam"]),
                "--min-face-size", str(data["min_face_size"]),
            )
        )

    def collect_go2(self):
        data = self._validate_common(require_label=True)
        if data is None:
            return
        self._set_hint(
            f"Step 2 running on Go2 camera for {data['label']} ({data['role']}). "
            f"Mode: {data['go2_mode']}. Make sure the robot is reachable before starting."
        )
        command = "collect-go2" if data["go2_mode"] == "hybrid" else ("collect-ssh" if data["go2_mode"] == "ssh" else "collect-webrtc")
        self._spawn(
            self._python_cmd(
                command,
                "--role", data["role"],
                "--label", data["label"],
                "--shots", str(data["shots"]),
                "--min-face-size", str(data["min_face_size"]),
            ),
            self._go2_env(data),
        )

    def train_model(self):
        self._set_hint("Step 3 running. This rebuilds the face database from everyone currently listed in the roster.")
        self._spawn(self._python_cmd("train"))

    def run_local(self):
        data = self._validate_common(require_label=False)
        if data is None:
            return
        self._set_hint("Step 4 running on the laptop camera. Use this to verify recognition before moving to the Go2.")
        self._spawn(
            self._python_cmd(
                "run",
                "--cam", str(data["cam"]),
                "--min-face-size", str(data["min_face_size"]),
            )
        )

    def run_go2(self):
        data = self._validate_common(require_label=False)
        if data is None:
            return
        self._set_hint(
            f"Step 4 running on the Go2 camera in {data['go2_mode']} mode. "
            "Use remote mode only when the serial and Unitree credentials are set."
        )
        self._spawn(self._display_cmd(), self._go2_env(data))


def main():
    root = tk.Tk()
    ProtocolUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

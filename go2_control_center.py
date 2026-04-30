from pathlib import Path
import tkinter as tk
from tkinter import ttk

from embedding_protocol_ui import ProtocolUI
from go2_live_map_ui import BG as MAP_BG
from go2_live_map_ui import LiveMapUI
from go2_motion_ui import MotionUI
from go2_waypoint_ui import WaypointUI


APP_BG = MAP_BG
PANEL = "#0d1b2e"
TEXT = "#e7f2ff"
MUTED = "#8ea7c4"
ACCENT = "#47d7ff"
EDGE = "#214666"
ASSET_DIRS = ("IMAGES", "images")


def resolve_asset_path(filename: str) -> Path:
    base_dir = Path(__file__).resolve().parent
    for folder in ASSET_DIRS:
        candidate = base_dir / folder / filename
        if candidate.exists():
            return candidate
    return base_dir / ASSET_DIRS[0] / filename


class TabRoot(tk.Frame):
    def __init__(self, master):
        super().__init__(master, bg=APP_BG)
        self.embedded_ui = True

    def title(self, _text: str):
        return None

    def geometry(self, _value: str):
        return None

    def minsize(self, _width: int, _height: int):
        return None

    def protocol(self, _name: str, _callback):
        return None


class Go2ControlCenter:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Go2 Control Center")
        self.root.geometry("1440x860")
        self.root.minsize(1180, 760)
        self.root.configure(bg=APP_BG)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        self.header_logo = self._load_image("Logo.png", max_width=320, max_height=80)
        self.app_icon = self._load_image("logononame.png", max_width=48, max_height=48)

        self._configure_style()
        self._build()
        if self.app_icon is not None:
            self.root.iconphoto(True, self.app_icon)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _load_image(self, filename: str, *, max_width: int, max_height: int) -> tk.PhotoImage | None:
        asset_path = resolve_asset_path(filename)
        if not asset_path.exists():
            return None
        try:
            image = tk.PhotoImage(file=str(asset_path))
        except tk.TclError:
            return None

        width = max(1, image.width())
        height = max(1, image.height())
        factor_w = max(1, (width + max_width - 1) // max_width)
        factor_h = max(1, (height + max_height - 1) // max_height)
        factor = max(factor_w, factor_h)
        if factor > 1:
            image = image.subsample(factor, factor)
        return image

    def _configure_style(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(".", background=APP_BG, foreground=TEXT)
        style.configure("App.TFrame", background=APP_BG)
        style.configure("Title.TLabel", background=APP_BG, foreground=TEXT, font=("Segoe UI Semibold", 22))
        style.configure("Sub.TLabel", background=APP_BG, foreground=MUTED, font=("Segoe UI", 9))
        style.configure("Control.TNotebook", background=APP_BG, borderwidth=0)
        style.configure(
            "Control.TNotebook.Tab",
            background=PANEL,
            foreground=MUTED,
            padding=(14, 8),
            font=("Segoe UI Semibold", 10),
            borderwidth=0,
        )
        style.map(
            "Control.TNotebook.Tab",
            background=[("selected", "#13304c"), ("active", "#17395a")],
            foreground=[("selected", ACCENT), ("active", TEXT)],
        )

    def _build(self):
        header = ttk.Frame(self.root, style="App.TFrame", padding=(18, 12, 18, 6))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        header.columnconfigure(1, weight=0)
        ttk.Label(header, text="Go2 Control Center", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            header,
            text="Motion, waypoint management, live mapping, and embedding protocols in one workspace.",
            style="Sub.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))
        if self.header_logo is not None:
            brand_card = tk.Frame(
                header,
                bg=PANEL,
                highlightbackground=EDGE,
                highlightthickness=1,
                padx=14,
                pady=10,
            )
            brand_card.grid(row=0, column=1, rowspan=2, sticky="e", padx=(18, 0))
            tk.Label(brand_card, image=self.header_logo, bg=PANEL).grid(row=0, column=0, sticky="e")
            tk.Label(
                brand_card,
                text="Senior Design Demo",
                fg=TEXT,
                bg=PANEL,
                font=("Segoe UI Semibold", 10),
            ).grid(row=1, column=0, sticky="w", pady=(8, 0))
            tk.Label(
                brand_card,
                text="Go2 autonomy, mapping, and face-recognition controls",
                fg=MUTED,
                bg=PANEL,
                font=("Segoe UI", 9),
            ).grid(row=2, column=0, sticky="w")

        shell = ttk.Frame(self.root, style="App.TFrame", padding=(12, 0, 12, 12))
        shell.grid(row=1, column=0, sticky="nsew")
        shell.columnconfigure(0, weight=1)
        shell.rowconfigure(0, weight=1)

        self.notebook = ttk.Notebook(shell, style="Control.TNotebook")
        self.notebook.grid(row=0, column=0, sticky="nsew")

        self.motion_host = TabRoot(self.notebook)
        self.waypoint_host = TabRoot(self.notebook)
        self.map_host = TabRoot(self.notebook)
        self.embedding_host = TabRoot(self.notebook)

        self.notebook.add(self.motion_host, text="Motion")
        self.notebook.add(self.waypoint_host, text="Waypoints")
        self.notebook.add(self.map_host, text="Live Map")
        self.notebook.add(self.embedding_host, text="Embedding")

        self.motion_ui = MotionUI(self.motion_host)
        self.waypoint_ui = WaypointUI(self.waypoint_host)
        self.map_ui = LiveMapUI(self.map_host)
        self.embedding_ui = ProtocolUI(self.embedding_host)

    def _on_close(self):
        try:
            self.map_ui.stop_stream()
        except Exception:
            pass
        self.root.after(150, self.root.destroy)


def main():
    root = tk.Tk()
    Go2ControlCenter(root)
    root.mainloop()


if __name__ == "__main__":
    main()

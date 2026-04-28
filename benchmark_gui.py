#!/usr/bin/env python3
"""
Benchmark Suite GUI
===================
A dark-themed tkinter launcher for both the Ray Tracer and Image Processor
benchmarks.  Displays live system specs and benchmark scores/timings.

Requirements: Python 3.8+  (tkinter is included in the standard library)
Optional    : psutil  →  pip install psutil   (for RAM display)
"""

import io
import multiprocessing
import json
import os
import platform
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk

# ── Make sibling benchmark modules importable ─────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import pyrender_benchmark       as _rt   # noqa: E402
import parallel_image_processor as _ip   # noqa: E402

# ── Single source image used by both benchmarks ───────────────────────────────
def _find_source_image():
    """Return the first image found in test_input/, or None if the folder is empty."""
    _exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    folder = os.path.join(_HERE, "test_input")
    if os.path.isdir(folder):
        for name in sorted(os.listdir(folder)):
            if os.path.splitext(name)[1].lower() in _exts:
                return os.path.join(folder, name)
    return None

_SOURCE_IMAGE = _find_source_image()

# ── Colour palette ────────────────────────────────────────────────────────────
_BG    = "#0d0d1a"   # window background
_CARD  = "#1a1a2e"   # card background
_CARD2 = "#16213e"   # inner panels
_RT    = "#ff4757"   # red    – ray tracer accent
_IP    = "#7c4dff"   # purple – image processor accent
_GRN   = "#2ed573"   # green  – system info labels
_YEL   = "#ffa502"   # yellow – status bar
_TXT   = "#dfe6e9"   # main text
_DIM   = "#636e72"   # dimmed text / separators
_WHT   = "#ffffff"   # bright white

# ── Quartile reference thresholds ─────────────────────────────────────────────
# Scores below bound[0] → Q1, below bound[1] → Q2, below bound[2] → Q3, else Q4
_QUARTILE_BOUNDS = {
    "rt_single": (4_000,   12_000,  30_000),
    "rt_multi":  (50_000, 250_000, 700_000),
    "ip_single": (4_000,   12_000,  30_000),
    "ip_multi":  (50_000, 250_000, 700_000),
}
_QUARTILE_COLORS = {"Q1": "#e17055", "Q2": _YEL, "Q3": _GRN, "Q4": "#00cec9"}

# ─────────────────────────────────────────────────────────────────────────────
# Thread-local stdout proxy
# ─────────────────────────────────────────────────────────────────────────────
_tlocal      = threading.local()
_real_stdout = sys.stdout


class _StdoutProxy(io.TextIOBase):
    """Routes writes to a per-thread Queue writer when one is installed,
    otherwise falls through to the original stdout."""

    def write(self, text: str) -> int:
        writer = getattr(_tlocal, "writer", None)
        if writer is not None:
            return writer.write(text)
        return _real_stdout.write(text)

    def flush(self):
        writer = getattr(_tlocal, "writer", None)
        if writer is not None:
            writer.flush()
        else:
            _real_stdout.flush()


class _QueueWriter(io.TextIOBase):
    """Writes text into a Queue for the GUI's polling loop."""

    def __init__(self, q: queue.Queue):
        self._q = q

    def write(self, text: str) -> int:
        if text:
            self._q.put(("log", text))
        return len(text)

    def flush(self):
        pass


# Install the proxy once so all print() calls in worker threads are captured.
sys.stdout = _StdoutProxy()


# ─────────────────────────────────────────────────────────────────────────────
# System information
# ─────────────────────────────────────────────────────────────────────────────
import subprocess as _sp


def _wmic(args: list[str]) -> list[str]:
    """Run a wmic query and return non-header, non-empty lines."""
    try:
        out = _sp.check_output(
            ["wmic"] + args, text=True,
            stderr=_sp.DEVNULL, timeout=6,
        )
        return [l.strip() for l in out.splitlines()
                if l.strip() and l.strip().lower() not in ("name", "caption",
                "totalphysicalmemory", "numberoflogicalprocessors")]
    except Exception:
        return []


def _sys_info() -> dict:
    # CPU name
    cpu_lines = _wmic(["cpu", "get", "name"])
    cpu = cpu_lines[0] if cpu_lines else "Unknown"

    # Logical core count
    core_lines = _wmic(["cpu", "get", "NumberOfLogicalProcessors"])
    try:
        cores = int(core_lines[0])
    except (IndexError, ValueError):
        cores = multiprocessing.cpu_count()

    # RAM  (TotalPhysicalMemory is in bytes)
    ram_lines = _wmic(["computersystem", "get", "TotalPhysicalMemory"])
    try:
        ram = f"{int(ram_lines[0]) / 1_073_741_824:.1f} GB"
    except (IndexError, ValueError):
        ram = "N/A"

    # OS caption
    os_lines = _wmic(["os", "get", "caption"])
    os_str = os_lines[0] if os_lines else f"{platform.system()} {platform.release()}"

    # Python version
    py_ver = platform.python_version()

    # GPUs
    gpu_lines = _wmic(["path", "win32_VideoController", "get", "name"])
    gpus = gpu_lines if gpu_lines else ["N/A"]

    return {
        "cpu":   cpu,
        "cores": cores,
        "ram":   ram,
        "os":    os_str,
        "python": py_ver,
        "gpus":  gpus,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main application
# ─────────────────────────────────────────────────────────────────────────────
class App(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Benchmark Suite")
        self.configure(bg=_BG)
        self.minsize(880, 760)
        self._center(960, 820)

        self._q     = queue.Queue()
        self._busy  = False

        # Preview image references (must be kept alive to avoid GC)
        self._preview_imgs:   dict[str, object] = {}
        self._preview_labels: dict[str, tk.Label] = {}

        # Panel canvas for the Image Processor card
        self._ip_canvas: tk.Canvas | None = None
        self._ip_panel_photos: list = []   # keep PhotoImage refs from GC
        self._ip_panel_count: int  = 0

        # Shared canvas for the Ray Tracer card
        self._rt_canvas: tk.Canvas | None = None
        self._rt_canvas_photos: list = []
        self._rt_render_w: int = _rt.RENDER_WIDTH
        self._rt_render_h: int = _rt.RENDER_HEIGHT

        # Score display variables
        self._scores = {
            "rt_single": tk.StringVar(value="—"),
            "rt_multi":  tk.StringVar(value="—"),
            "ip_single": tk.StringVar(value="—"),
            "ip_multi":  tk.StringVar(value="—"),
        }
        self._status = tk.StringVar(value="Ready")
        self._pb: dict[str, ttk.Progressbar] = {}
        self._buttons: list[tk.Button] = []

        # Live elapsed timer per card
        self._timer_vars    = {"rt": tk.StringVar(value="—"), "ip": tk.StringVar(value="—")}
        self._timer_start:  dict[str, float] = {}
        self._timer_running = {"rt": False, "ip": False}

        # Per-test elapsed time and quartile rank
        self._time_vars = {
            "rt_single": tk.StringVar(value=""),
            "rt_multi":  tk.StringVar(value=""),
            "ip_single": tk.StringVar(value=""),
            "ip_multi":  tk.StringVar(value=""),
        }
        self._rank_vars = {
            "rt_single": tk.StringVar(value=""),
            "rt_multi":  tk.StringVar(value=""),
            "ip_single": tk.StringVar(value=""),
            "ip_multi":  tk.StringVar(value=""),
        }
        self._rank_labels: dict[str, tk.Label] = {}

        # Score history (persisted to JSON)
        self._score_history_file = os.path.join(_HERE, "benchmark_scores.json")
        self._current_session: dict = {}

        self._apply_style()
        self._build_ui()
        self._poll_queue()

    # ── Window helpers ────────────────────────────────────────────────────────
    def _center(self, w: int, h: int):
        self.update_idletasks()
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        self.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}")

    def _apply_style(self):
        s = ttk.Style()
        s.theme_use("clam")
        s.configure("TScrollbar",
                    background=_CARD2, troughcolor=_BG,
                    bordercolor=_BG, arrowcolor=_DIM)
        s.configure("RT.Horizontal.TProgressbar",
                    troughcolor=_CARD2, background=_RT, bordercolor=_CARD2)
        s.configure("IP.Horizontal.TProgressbar",
                    troughcolor=_CARD2, background=_IP, bordercolor=_CARD2)

    # ── UI construction ───────────────────────────────────────────────────────
    def _build_ui(self):
        self._header()
        self._sysinfo_panel()
        self._cards_row()
        self._history_panel()
        self._statusbar()
        self._log_panel()

    def _header(self):
        f = tk.Frame(self, bg=_CARD, pady=14)
        f.pack(fill="x")
        tk.Label(f, text="⚡  BENCHMARK SUITE",
                 font=("Segoe UI", 18, "bold"), bg=_CARD, fg=_WHT).pack()
        tk.Label(f, text="Ray Tracer  ·  Image Processor  ·  System Scores",
                 font=("Segoe UI", 9), bg=_CARD, fg=_DIM).pack()
        tk.Button(
            f, text="↺  Reset", font=("Segoe UI", 8, "bold"),
            bg="#2a2a3e", fg=_TXT,
            activebackground="#3a3a4e", activeforeground=_WHT,
            relief="flat", padx=8, pady=4, cursor="hand2",
            command=self._reset,
        ).place(relx=1.0, rely=0.5, anchor="e", x=-12)

    def _sysinfo_panel(self):
        outer = tk.Frame(self, bg=_BG, padx=10, pady=8)
        outer.pack(fill="x")
        f = tk.Frame(outer, bg=_CARD2, padx=14, pady=10)
        f.pack(fill="x")

        tk.Label(f, text="SYSTEM INFO",
                 font=("Segoe UI", 7, "bold"), bg=_CARD2, fg=_DIM
                 ).grid(row=0, column=0, columnspan=20, sticky="w", pady=(0, 6))

        nfo = _sys_info()
        row1 = [
            ("CPU",    nfo["cpu"]),
            ("Cores",  str(nfo["cores"])),
            ("RAM",    nfo["ram"]),
            ("OS",     nfo["os"]),
            ("Python", nfo["python"]),
        ]
        for col, (label, value) in enumerate(row1):
            pad = (24, 2) if col > 0 else (0, 2)
            tk.Label(f, text=label,
                     font=("Segoe UI", 8, "bold"), bg=_CARD2, fg=_GRN
                     ).grid(row=1, column=col * 2, padx=pad, sticky="w")
            tk.Label(f, text=value,
                     font=("Segoe UI", 8), bg=_CARD2, fg=_TXT, wraplength=160
                     ).grid(row=1, column=col * 2 + 1, sticky="w")

        # GPU row — one label per detected GPU
        gpu_str = "  /  ".join(nfo["gpus"])
        tk.Label(f, text="GPU",
                 font=("Segoe UI", 8, "bold"), bg=_CARD2, fg=_GRN
                 ).grid(row=2, column=0, pady=(6, 0), sticky="w")
        tk.Label(f, text=gpu_str,
                 font=("Segoe UI", 8), bg=_CARD2, fg=_TXT, wraplength=800
                 ).grid(row=2, column=1, columnspan=19, pady=(6, 0), sticky="w")

    def _cards_row(self):
        row = tk.Frame(self, bg=_BG, padx=10, pady=4)
        row.pack(fill="x")
        row.columnconfigure(0, weight=1)
        row.columnconfigure(1, weight=1)
        self._build_rt_card(row)
        self._build_ip_card(row)

    # ── Thumbnail loader ──────────────────────────────────────────────────────
    def _load_thumbnail(self, path: str, max_w: int = 380, max_h: int = 180):
        """Return a tk.PhotoImage thumbnail, or None on failure."""
        try:
            from PIL import Image as _PILImg
            _PILImg.MAX_IMAGE_PIXELS = None
            img = _PILImg.open(path).convert("RGB")
            img.thumbnail((max_w, max_h), _PILImg.LANCZOS)
            # Convert to PhotoImage via PPM bytes (no temp file)
            buf = io.BytesIO()
            img.save(buf, format="PPM")
            buf.seek(0)
            return tk.PhotoImage(data=buf.read())
        except Exception:
            # Fallback: tkinter native PNG only
            try:
                return tk.PhotoImage(file=path)
            except Exception:
                return None

    def _show_preview(self, bench: str, path: str):
        """Load *path* as a thumbnail and display it in the card placeholder."""
        if bench not in self._preview_labels:
            return
        photo = self._load_thumbnail(path)
        lbl   = self._preview_labels[bench]
        if photo:
            self._preview_imgs[bench] = photo   # keep reference
            lbl.configure(image=photo, text="", bg="#08081a")
        else:
            lbl.configure(text="(preview unavailable)",
                          image="", bg=_CARD2)

    def _load_rt_preview(self):
        """Render a low-res spheres preview in a background thread and show it."""
        try:
            rgb = _rt.render_preview(320, 180, max_depth=3)
        except Exception:
            return
        if not self._rt_preview_active:
            return
        try:
            from PIL import Image as _PILImg, ImageTk
            img = _PILImg.frombytes("RGB", (320, 180), rgb)
            canvas = self._rt_canvas
            cw = canvas.winfo_width() or 380
            img = img.resize((cw, 190), _PILImg.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self._rt_canvas_photos.append(photo)

            def _show():
                if not self._rt_preview_active:
                    return
                canvas.delete("all")
                canvas.create_image(0, 0, anchor="nw", image=photo)

            canvas.after(0, _show)
        except Exception:
            pass

    def _load_source_image_to_canvas(self, canvas, photos_list):
        """Load the source image to *canvas* in a background thread."""
        src = _SOURCE_IMAGE
        if not os.path.isfile(src):
            return

        def _bg():
            try:
                from PIL import Image as _PILImg
                _PILImg.MAX_IMAGE_PIXELS = None
                img = _PILImg.open(src).convert("RGB")
                img.thumbnail((760, 190), _PILImg.LANCZOS)
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                png_bytes = buf.getvalue()
                self.after(0, lambda: self._draw_png_on_canvas(
                    canvas, png_bytes, photos_list))
            except Exception:
                pass

        threading.Thread(target=_bg, daemon=True).start()

    def _draw_png_on_canvas(self, canvas, png_bytes, photos_list):
        """Render PNG bytes onto *canvas*. Must be called from the main thread."""
        if canvas is None:
            return
        cw = canvas.winfo_width() or 380
        ch = 190
        try:
            from PIL import Image as _PILImg, ImageTk
            _PILImg.MAX_IMAGE_PIXELS = None
            buf = io.BytesIO(png_bytes)
            img = _PILImg.open(buf).convert("RGB")
            img = img.resize((cw, ch), _PILImg.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            photos_list.clear()
            photos_list.append(photo)
            canvas.delete("all")
            canvas.create_image(0, 0, anchor="nw", image=photo)
        except Exception:
            canvas.delete("all")
            canvas.create_text(cw // 2, ch // 2,
                               text="(preview unavailable)",
                               fill=_DIM, font=("Consolas", 8))

    # ── Score box ─────────────────────────────────────────────────────────────
    def _score_box(self, parent, label: str, var: tk.StringVar,
                   col: int, color: str, key: str = ""):
        f = tk.Frame(parent, bg=_CARD2, padx=14, pady=10)
        f.grid(row=0, column=col, sticky="ew",
               padx=(0 if col == 0 else 8, 0))
        parent.columnconfigure(col, weight=1)
        tk.Label(f, text=label,
                 font=("Segoe UI", 7, "bold"), bg=_CARD2, fg=_DIM).pack()
        tk.Label(f, textvariable=var,
                 font=("Segoe UI", 24, "bold"), bg=_CARD2, fg=color).pack()
        tk.Label(f, text="pts",
                 font=("Segoe UI", 7), bg=_CARD2, fg=_DIM).pack()
        if key:
            tk.Label(f, textvariable=self._time_vars[key],
                     font=("Segoe UI", 8), bg=_CARD2, fg=_DIM).pack()
            rl = tk.Label(f, textvariable=self._rank_vars[key],
                          font=("Segoe UI", 10, "bold"), bg=_CARD2, fg=_DIM)
            rl.pack()
            self._rank_labels[key] = rl

    # ── Ray Tracer card ────────────────────────────────────────────────────────
    def _build_rt_card(self, parent):
        card = tk.Frame(parent, bg=_CARD, padx=14, pady=12,
                        highlightthickness=2, highlightbackground=_RT)
        card.grid(row=0, column=0, sticky="nsew", padx=(0, 6), pady=4)

        tk.Label(card, text="🎨  RAY TRACER",
                 font=("Segoe UI", 12, "bold"), bg=_CARD, fg=_RT).pack(anchor="w")
        tk.Label(card,
                 text="pyrender_benchmark.py  ·  3-D ray-traced scene "
                      "with sunset HDRI environment",
                 font=("Segoe UI", 7), bg=_CARD, fg=_DIM).pack(anchor="w")
        tk.Frame(card, height=1, bg=_RT).pack(fill="x", pady=(8, 10))

        sf = tk.Frame(card, bg=_CARD)
        sf.pack(fill="x", pady=(0, 10))
        self._score_box(sf, "SINGLE-CORE", self._scores["rt_single"], 0, _RT, key="rt_single")
        self._score_box(sf, "MULTI-CORE",  self._scores["rt_multi"],  1, _RT, key="rt_multi")

        pb = ttk.Progressbar(card, style="RT.Horizontal.TProgressbar",
                             mode="indeterminate")
        pb.pack(fill="x", pady=(0, 2))
        self._pb["rt"] = pb
        tk.Label(card, textvariable=self._timer_vars["rt"],
                 font=("Consolas", 8), bg=_CARD, fg=_RT, anchor="e"
                 ).pack(fill="x", pady=(0, 4))

        bf = tk.Frame(card, bg=_CARD)
        bf.pack(fill="x", pady=(0, 6))
        for text, m, side in [
            ("Single-Core", "single", "left"),
            ("Multi-Core",  "multi",  "left"),
            ("Run Both",    "both",   "right"),
        ]:
            b = tk.Button(
                bf, text=text, font=("Segoe UI", 9, "bold"),
                bg=_RT if m != "both" else "#2a2a3e",
                fg=_WHT,
                activebackground="#c0392b" if m != "both" else "#444",
                relief="flat", padx=10, pady=6, cursor="hand2",
                command=lambda mode=m: self._run("rt", mode),
            )
            if side == "left":
                b.pack(side="left", padx=(0, 4))
            else:
                b.pack(side="right")
            self._buttons.append(b)

        # Image preview canvas (same source image as Image Processor card)
        tk.Label(card, text="IMAGE PREVIEW",
                 font=("Segoe UI", 7, "bold"), bg=_CARD, fg=_DIM
                 ).pack(anchor="w", pady=(4, 2))
        rt_canvas = tk.Canvas(card, bg="#08081a", height=190,
                              highlightthickness=0)
        rt_canvas.pack(fill="x")
        self._rt_canvas = rt_canvas
        self._rt_canvas_photos = []
        self._rt_preview_active = True  # cleared when a benchmark run starts
        # Render a low-res preview of the spheres scene in the background
        rt_canvas.after(50, lambda: rt_canvas.create_text(
            rt_canvas.winfo_width() // 2 or 190, 95,
            text="Rendering preview…",
            fill=_DIM, font=("Segoe UI", 8), tags="placeholder"))
        threading.Thread(target=self._load_rt_preview, daemon=True).start()

    # ── Image Processor card ───────────────────────────────────────────────────
    def _build_ip_card(self, parent):
        card = tk.Frame(parent, bg=_CARD, padx=14, pady=12,
                        highlightthickness=2, highlightbackground=_IP)
        card.grid(row=0, column=1, sticky="nsew", padx=(6, 0), pady=4)

        tk.Label(card, text="🖼  IMAGE PROCESSOR",
                 font=("Segoe UI", 12, "bold"), bg=_CARD, fg=_IP).pack(anchor="w")
        tk.Label(card,
                 text="parallel_image_processor.py  ·  parallel filter "
                      "pipeline on real images",
                 font=("Segoe UI", 7), bg=_CARD, fg=_DIM).pack(anchor="w")
        tk.Frame(card, height=1, bg=_IP).pack(fill="x", pady=(8, 10))

        sf = tk.Frame(card, bg=_CARD)
        sf.pack(fill="x", pady=(0, 10))
        self._score_box(sf, "SINGLE-THREAD", self._scores["ip_single"], 0, _IP, key="ip_single")
        self._score_box(sf, "MULTI-CORE",    self._scores["ip_multi"],  1, _IP, key="ip_multi")

        pb = ttk.Progressbar(card, style="IP.Horizontal.TProgressbar",
                             mode="indeterminate")
        pb.pack(fill="x", pady=(0, 2))
        self._pb["ip"] = pb
        tk.Label(card, textvariable=self._timer_vars["ip"],
                 font=("Consolas", 8), bg=_CARD, fg=_IP, anchor="e"
                 ).pack(fill="x", pady=(0, 4))

        bf = tk.Frame(card, bg=_CARD)
        bf.pack(fill="x", pady=(0, 6))
        for text, m, side in [
            ("Single-Thread", "single", "left"),
            ("Multi-Core",    "multi",  "left"),
            ("Run Both",      "both",   "right"),
        ]:
            b = tk.Button(
                bf, text=text, font=("Segoe UI", 9, "bold"),
                bg=_IP if m != "both" else "#2a2a3e",
                fg=_WHT,
                activebackground="#5e35b1" if m != "both" else "#444",
                relief="flat", padx=10, pady=6, cursor="hand2",
                command=lambda mode=m: self._run("ip", mode),
            )
            if side == "left":
                b.pack(side="left", padx=(0, 4))
            else:
                b.pack(side="right")
            self._buttons.append(b)

        # Panel-by-panel image preview canvas
        tk.Label(card, text="IMAGE PREVIEW  ·  panel processing",
                 font=("Segoe UI", 7, "bold"), bg=_CARD, fg=_DIM
                 ).pack(anchor="w", pady=(4, 2))
        canvas = tk.Canvas(card, bg="#08081a", height=190, highlightthickness=0)
        canvas.pack(fill="x")
        self._ip_canvas = canvas
        self._ip_panel_photos = []
        self.after(250, lambda: self._load_source_image_to_canvas(
            self._ip_canvas, self._ip_panel_photos))

    # ── Status bar ─────────────────────────────────────────────────────────────
    def _statusbar(self):
        f = tk.Frame(self, bg=_CARD2, pady=4)
        f.pack(fill="x")
        tk.Label(f, textvariable=self._status,
                 font=("Consolas", 8), bg=_CARD2, fg=_YEL,
                 anchor="w", padx=10).pack(fill="x")

    # ── Output log ─────────────────────────────────────────────────────────────
    def _log_panel(self):
        f = tk.Frame(self, bg=_BG, padx=10)
        f.pack(fill="both", expand=True, pady=(2, 10))
        tk.Label(f, text="OUTPUT LOG",
                 font=("Segoe UI", 7, "bold"), bg=_BG, fg=_DIM).pack(anchor="w")
        inner = tk.Frame(f, bg=_BG)
        inner.pack(fill="both", expand=True)

        self._log = tk.Text(
            inner, bg="#08081a", fg=_TXT,
            font=("Consolas", 8), relief="flat",
            wrap="word", state="disabled",
            insertbackground=_WHT, bd=0,
            highlightthickness=1, highlightbackground=_CARD2,
        )
        sb = ttk.Scrollbar(inner, command=self._log.yview)
        self._log.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self._log.pack(fill="both", expand=True)

    # ── History panel ─────────────────────────────────────────────────────────
    def _history_panel(self):
        outer = tk.Frame(self, bg=_BG, padx=10, pady=2)
        outer.pack(fill="x")
        f = tk.Frame(outer, bg=_CARD2, padx=14, pady=8)
        f.pack(fill="x")
        tk.Label(f, text="SCORE HISTORY  ·  last 5 runs  (colour = quartile  Q1 worst → Q4 best)",
                 font=("Segoe UI", 7, "bold"), bg=_CARD2, fg=_DIM
                 ).pack(anchor="w", pady=(0, 4))
        self._history_text = tk.Text(
            f, bg=_CARD2, fg=_TXT, font=("Consolas", 7),
            relief="flat", state="disabled", height=4,
            highlightthickness=0, wrap="none",
        )
        self._history_text.pack(fill="x")
        for q, c in _QUARTILE_COLORS.items():
            self._history_text.tag_configure(q, foreground=c)
        self._history_text.tag_configure("dim", foreground=_DIM)
        self._history_text.tag_configure("hdr",
            foreground=_DIM, font=("Consolas", 7, "bold"))
        self._refresh_history()

    def _refresh_history(self):
        history = self._load_history()
        t = self._history_text
        t.configure(state="normal")
        t.delete("1.0", "end")
        t.insert("end",
            f"  {'Date':<18}  {'RT-1C':>12}  {'RT-nC':>14}  "
            f"{'IP-1T':>12}  {'IP-nC':>14}\n", "hdr")
        t.insert("end", "  " + "\u2500" * 72 + "\n", "dim")
        for entry in list(reversed(history[-5:])):
            date = entry.get("date", "?")
            t.insert("end", f"  {date:<18}  ")
            for k, width in [("rt_single", 12), ("rt_multi", 14),
                              ("ip_single", 12), ("ip_multi", 14)]:
                v = entry.get(k)
                if v is None:
                    t.insert("end", f"{'—':>{width}}  ", "dim")
                else:
                    rank = self._compute_rank(k, v)
                    t.insert("end", f"{f'{v:,} {rank}':>{width}}  ", rank)
            t.insert("end", "\n")
        if not history:
            t.insert("end", "  No runs recorded yet.\n", "dim")
        t.configure(state="disabled")

    # ── Score persistence & ranking ───────────────────────────────────────────
    def _load_history(self) -> list:
        if not os.path.isfile(self._score_history_file):
            return []
        try:
            with open(self._score_history_file, encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return []

    def _save_session(self):
        if not self._current_session or len(self._current_session) <= 1:
            return   # only the date key, no scores yet
        history = self._load_history()
        history.append(self._current_session)
        try:
            with open(self._score_history_file, "w", encoding="utf-8") as fh:
                json.dump(history, fh, indent=2)
        except Exception:
            pass
        self._refresh_history()

    def _compute_rank(self, key: str, score: int) -> str:
        q1, q2, q3 = _QUARTILE_BOUNDS.get(key, (4_000, 12_000, 30_000))
        if score < q1:  return "Q1"
        if score < q2:  return "Q2"
        if score < q3:  return "Q3"
        return "Q4"

    # ── Timer helpers ─────────────────────────────────────────────────────────
    def _start_timer(self, bench: str):
        self._timer_start[bench] = time.perf_counter()
        self._timer_running[bench] = True
        self._timer_vars[bench].set("0.0 s")
        self._tick_timer(bench)

    def _stop_timer(self, bench: str):
        if self._timer_running.get(bench):
            elapsed = time.perf_counter() - self._timer_start.get(bench, 0.0)
            self._timer_vars[bench].set(f"{elapsed:.2f} s  \u2714")
        self._timer_running[bench] = False

    def _tick_timer(self, bench: str):
        if self._timer_running.get(bench):
            elapsed = time.perf_counter() - self._timer_start.get(bench, 0.0)
            self._timer_vars[bench].set(f"{elapsed:.1f} s")
            self.after(100, lambda: self._tick_timer(bench))

    # ── Reset ─────────────────────────────────────────────────────────────────
    def _reset(self):
        if self._busy:
            return
        for v in self._scores.values():
            v.set("\u2014")
        for v in self._timer_vars.values():
            v.set("\u2014")
        for v in self._time_vars.values():
            v.set("")
        for v in self._rank_vars.values():
            v.set("")
        for lbl in self._rank_labels.values():
            lbl.configure(fg=_DIM)
        self._log.configure(state="normal")
        self._log.delete("1.0", "end")
        self._log.configure(state="disabled")
        self._status.set("Ready")
        self._current_session = {}
        self._rt_canvas_photos = []
        self._ip_panel_photos  = []
        self._ip_src_img       = None
        if self._rt_canvas:
            self._rt_canvas.delete("all")
            self._rt_canvas.configure(bg="#08081a")
            self._rt_canvas.create_text(
                self._rt_canvas.winfo_width() // 2 or 190, 95,
                text="Rendering preview…",
                fill=_DIM, font=("Segoe UI", 8), tags="placeholder")
            self._rt_preview_active = True
            threading.Thread(target=self._load_rt_preview, daemon=True).start()
        if self._ip_canvas:
            self.after(100, lambda: self._load_source_image_to_canvas(
                self._ip_canvas, self._ip_panel_photos))

    # ── Benchmark launcher ────────────────────────────────────────────────────
    def _run(self, bench: str, mode: str):
        if self._busy:
            self._log_append("⚠  A benchmark is already running — please wait.\n")
            return
        self._busy = True
        self._current_session = {"date": time.strftime("%Y-%m-%d %H:%M")}
        for b in self._buttons:
            b.configure(state="disabled")
        threading.Thread(
            target=self._worker, args=(bench, mode), daemon=True
        ).start()

    def _worker(self, bench: str, mode: str):
        """Runs entirely in the background thread."""
        q = self._q
        _tlocal.writer = _QueueWriter(q)   # captures print() in this thread
        try:
            modes = ["single", "multi"] if mode == "both" else [mode]
            label = "Ray Tracer" if bench == "rt" else "Image Processor"
            q.put(("status", f"Running {label}  [{mode.upper()}] …"))
            q.put(("pb_start", bench))
            q.put(("log", f"\n{'─' * 56}\n"))
            q.put(("log", f"  {label}  —  {mode.upper()} mode\n"))
            q.put(("log", f"{'─' * 56}\n"))

            for m in modes:
                if bench == "rt":
                    q.put(("rt_chunk_init", _rt.RENDER_HEIGHT))
                    score = elapsed = workers = 0
                    for item in _rt.run_benchmark_stream(m):
                        if item[0] == "chunk":
                            _, row_start, row_end, _w, _h, _rgb = item
                            q.put(("rt_chunk", row_start, row_end, _rt.RENDER_WIDTH, _rt.RENDER_HEIGHT, _rgb))
                        else:  # "result"
                            _, score, elapsed, workers, _chunks = item
                    q.put(("rt_score", m, score, elapsed, workers))
                else:
                    img_path = _find_source_image()
                    if img_path is None:
                        q.put(("log", "\n[ERROR] No image found in test_input/.\n"
                                      "        Please add a .jpg or .png file to the test_input folder.\n"))
                        q.put(("ip_score", m, 0, 0.0, 1, _ip.PANEL_COUNT))
                        continue
                    panel_count = _ip.PANEL_COUNT
                    t_start = time.perf_counter()
                    for item in _ip.process_image_panels(img_path, m):
                        if item[0] == "init":
                            _, iw, ih, full_png = item
                            q.put(("ip_panel_init", iw, ih, panel_count,
                                   full_png))
                        else:  # "panel"
                            _, idx, y0, y1, iw, ih, png_bytes, _ms = item
                            q.put(("ip_panel", idx, panel_count,
                                   y0, y1, iw, ih, png_bytes))
                    elapsed = time.perf_counter() - t_start
                    ref = (_ip.REF_TIME_SINGLE if m == "single"
                           else _ip.REF_TIME_MULTI)
                    scale = 1 if m == "single" else multiprocessing.cpu_count()
                    score = int(1000 * scale * ref / max(elapsed, 0.001))
                    workers_n = (1 if m == "single"
                                 else multiprocessing.cpu_count())
                    q.put(("ip_score", m, score, elapsed, workers_n,
                           panel_count))

        except Exception as exc:
            import traceback
            q.put(("log", f"\n[ERROR] {exc}\n{traceback.format_exc()}\n"))
        finally:
            _tlocal.writer = None
            q.put(("pb_stop", bench))
            q.put(("done", None))

    # ── Queue polling (runs on the GUI / main thread via after()) ─────────────
    def _poll_queue(self):
        try:
            while True:
                self._dispatch(self._q.get_nowait())
        except queue.Empty:
            pass
        self.after(40, self._poll_queue)

    def _dispatch(self, msg):
        kind = msg[0]

        if kind == "log":
            # Normalise carriage-return progress lines to newlines
            text = msg[1].replace("\r", "\n")
            if text.strip():
                self._log_append(text)

        elif kind == "status":
            self._status.set(msg[1])

        elif kind == "pb_start":
            self._pb[msg[1]].start(10)
            self._start_timer(msg[1])

        elif kind == "pb_stop":
            self._pb[msg[1]].stop()
            self._pb[msg[1]].configure(value=0)
            self._stop_timer(msg[1])

        elif kind == "show_image":
            _, bench, path = msg
            self._show_preview(bench, path)

        elif kind == "ip_panel_init":
            _, img_w, img_h, panel_count, full_png_bytes = msg
            self._ip_panel_count  = panel_count
            self._ip_panel_photos = []
            self._ip_src_img      = None   # cached at canvas size, loaded on first panel
            canvas = self._ip_canvas
            if canvas:
                canvas.delete("all")
                canvas.configure(bg="#000000")

        elif kind == "ip_panel":
            _, idx, panel_count, y0_src, y1_src, img_w, img_h, png_bytes = msg
            canvas = self._ip_canvas
            if canvas is None:
                return
            cw = canvas.winfo_width() or 380
            ch = 190
            ph = ch / panel_count
            display_y = int(idx * ph)
            display_h = max(1, round(ph))
            try:
                from PIL import Image as _PILImg, ImageTk
                _PILImg.MAX_IMAGE_PIXELS = None
                # Load + cache the source image at canvas size once
                if not getattr(self, "_ip_src_img", None):
                    _src = _find_source_image()
                    if _src:
                        src = _PILImg.open(_src).convert("RGB")
                        self._ip_src_img = src.resize((cw, ch), _PILImg.LANCZOS)
                    else:
                        self._ip_src_img = None
                band = self._ip_src_img.crop((0, display_y, cw, display_y + display_h))
                photo = ImageTk.PhotoImage(band)
                self._ip_panel_photos.append(photo)   # prevent GC
                canvas.delete(f"panel_{idx}")
                canvas.create_image(0, display_y, anchor="nw",
                                    image=photo, tags=f"panel_{idx}")
            except Exception:
                canvas.create_rectangle(0, display_y, cw, display_y + display_h,
                                        fill=_IP, outline="",
                                        tags=f"panel_{idx}")

        elif kind == "rt_chunk_init":
            _, render_h = msg
            self._rt_preview_active = False   # prevent preview thread from drawing
            self._rt_canvas_photos = []
            canvas = self._rt_canvas
            if canvas:
                canvas.delete("all")
                canvas.configure(bg="#000000")

        elif kind == "rt_chunk":
            _, row_start, row_end, render_w, render_h, rgb_bytes = msg
            canvas = self._rt_canvas
            if canvas is None:
                return
            cw = canvas.winfo_width() or 380
            ch = 190
            y0 = int(row_start / render_h * ch)
            y1 = int(row_end   / render_h * ch)
            band_h = max(1, y1 - y0)
            try:
                from PIL import Image as _PILImg, ImageTk
                src_band_h = row_end - row_start
                band = _PILImg.frombytes("RGB", (render_w, src_band_h), rgb_bytes)
                band = band.resize((cw, band_h), _PILImg.LANCZOS)
                photo = ImageTk.PhotoImage(band)
                self._rt_canvas_photos.append(photo)
                canvas.create_image(0, y0, anchor="nw", image=photo)
            except Exception:
                canvas.create_rectangle(0, y0, cw, y0 + band_h,
                                        fill=_RT, outline="")

        elif kind == "rt_score":
            _, m, score, elapsed, workers = msg
            key = f"rt_{m}"
            self._scores[key].set(f"{score:,}")
            self._time_vars[key].set(f"{elapsed:.2f} s")
            rank = self._compute_rank(key, score)
            self._rank_vars[key].set(rank)
            if key in self._rank_labels:
                self._rank_labels[key].configure(fg=_QUARTILE_COLORS.get(rank, _DIM))
            self._current_session[key] = score
            self._log_append(
                f"\n  ✔  Ray Tracer [{m.upper()}]  —  {rank}\n"
                f"     Score   : {score:>10,} pts\n"
                f"     Time    : {elapsed:>10.2f} s\n"
                f"     Workers : {workers:>10}\n"
            )
            # Show source image on RT canvas after benchmark finishes
            self._rt_canvas_photos = []
            self._rt_src_img = None
            if self._rt_canvas:
                self.after(50, lambda: self._load_source_image_to_canvas(
                    self._rt_canvas, self._rt_canvas_photos))

        elif kind == "ip_score":
            _, m, score, elapsed, workers, count = msg
            key = f"ip_{m}"
            self._scores[key].set(f"{score:,}")
            self._time_vars[key].set(f"{elapsed:.2f} s")
            rank = self._compute_rank(key, score)
            self._rank_vars[key].set(rank)
            if key in self._rank_labels:
                self._rank_labels[key].configure(fg=_QUARTILE_COLORS.get(rank, _DIM))
            self._current_session[key] = score
            self._log_append(
                f"\n  ✔  Image Processor [{m.upper()}]  —  {rank}\n"
                f"     Score   : {score:>10,} pts\n"
                f"     Time    : {elapsed:>10.2f} s\n"
                f"     Workers : {workers:>10}\n"
                f"     Images  : {count:>10}\n"
            )
            # Restore source image on IP canvas after benchmark finishes
            self._ip_panel_photos = []
            self._ip_src_img = None
            if self._ip_canvas:
                self._ip_canvas.configure(bg="#08081a")
                self.after(50, lambda: self._load_source_image_to_canvas(
                    self._ip_canvas, self._ip_panel_photos))

        elif kind == "done":
            self._busy = False
            self._status.set("Ready")
            for b in self._buttons:
                b.configure(state="normal")
            self._save_session()

    # ── Log helper ─────────────────────────────────────────────────────────────
    def _log_append(self, text: str):
        self._log.configure(state="normal")
        self._log.insert("end", text)
        self._log.see("end")
        self._log.configure(state="disabled")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Required on Windows so spawned worker processes don't re-run the GUI
    multiprocessing.freeze_support()
    App().mainloop()

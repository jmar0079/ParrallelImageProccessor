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
import os
import platform
import queue
import sys
import threading
import tkinter as tk
from tkinter import ttk

# ── Make sibling benchmark modules importable ─────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import pyrender_benchmark       as _rt   # noqa: E402
import parallel_image_processor as _ip   # noqa: E402

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
def _detect_gpus() -> list[str]:
    """Return a list of GPU name strings using the best available method."""
    # 1. Try GPUtil (pip install gputil)
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        if gpus:
            return [g.name for g in gpus]
    except Exception:
        pass

    # 2. Try pynvml / nvidia-ml-py (NVIDIA only)
    try:
        import pynvml
        pynvml.nvmlInit()
        names = [
            pynvml.nvmlDeviceGetName(pynvml.nvmlDeviceGetHandleByIndex(i))
            for i in range(pynvml.nvmlDeviceGetCount())
        ]
        if names:
            return names
    except Exception:
        pass

    # 3. wmi (Windows only, no extra install needed usually)
    try:
        import wmi
        c = wmi.WMI()
        names = [v.Name for v in c.Win32_VideoController() if v.Name]
        if names:
            return names
    except Exception:
        pass

    # 4. Fallback: wmic subprocess (Windows; no third-party deps)
    try:
        import subprocess
        out = subprocess.check_output(
            ["wmic", "path", "win32_VideoController", "get", "name"],
            text=True, stderr=subprocess.DEVNULL, timeout=5,
        )
        names = [l.strip() for l in out.splitlines()
                 if l.strip() and l.strip().lower() != "name"]
        if names:
            return names
    except Exception:
        pass

    return ["N/A"]


def _sys_info() -> dict:
    info = {
        "cpu":    platform.processor() or platform.machine() or "Unknown",
        "cores":  multiprocessing.cpu_count(),
        "os":     f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "ram":    "N/A",
        "gpus":   _detect_gpus(),
    }
    try:
        import psutil  # optional
        info["ram"] = f"{psutil.virtual_memory().total / 1_073_741_824:.1f} GB"
    except ImportError:
        pass
    return info


# ─────────────────────────────────────────────────────────────────────────────
# Main application
# ─────────────────────────────────────────────────────────────────────────────
class App(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Benchmark Suite")
        self.configure(bg=_BG)
        self.minsize(880, 660)
        self._center(960, 720)

        self._q     = queue.Queue()
        self._busy  = False

        # Preview image references (must be kept alive to avoid GC)
        self._preview_imgs:   dict[str, object] = {}
        self._preview_labels: dict[str, tk.Label] = {}

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
        self._statusbar()
        self._log_panel()

    def _header(self):
        f = tk.Frame(self, bg=_CARD, pady=14)
        f.pack(fill="x")
        tk.Label(f, text="⚡  BENCHMARK SUITE",
                 font=("Segoe UI", 18, "bold"), bg=_CARD, fg=_WHT).pack()
        tk.Label(f, text="Ray Tracer  ·  Image Processor  ·  System Scores",
                 font=("Segoe UI", 9), bg=_CARD, fg=_DIM).pack()

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

    # ── Score box ─────────────────────────────────────────────────────────────
    def _score_box(self, parent, label: str, var: tk.StringVar,
                   col: int, color: str):
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
        self._score_box(sf, "SINGLE-CORE", self._scores["rt_single"], 0, _RT)
        self._score_box(sf, "MULTI-CORE",  self._scores["rt_multi"],  1, _RT)

        pb = ttk.Progressbar(card, style="RT.Horizontal.TProgressbar",
                             mode="indeterminate")
        pb.pack(fill="x", pady=(0, 6))
        self._pb["rt"] = pb

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

        # Image preview placeholder (Ray Tracer output)
        tk.Label(card, text="RENDERED OUTPUT",
                 font=("Segoe UI", 7, "bold"), bg=_CARD, fg=_DIM
                 ).pack(anchor="w", pady=(4, 2))
        prev = tk.Label(card, text="— run a benchmark to see the render —",
                        font=("Segoe UI", 7), bg=_CARD2, fg=_DIM,
                        width=50, height=10, anchor="center")
        prev.pack(fill="x")
        self._preview_labels["rt"] = prev

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
        self._score_box(sf, "SINGLE-THREAD", self._scores["ip_single"], 0, _IP)
        self._score_box(sf, "MULTI-CORE",    self._scores["ip_multi"],  1, _IP)

        pb = ttk.Progressbar(card, style="IP.Horizontal.TProgressbar",
                             mode="indeterminate")
        pb.pack(fill="x", pady=(0, 6))
        self._pb["ip"] = pb

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

        # Image preview placeholder (input / output)
        tk.Label(card, text="IMAGE PREVIEW",
                 font=("Segoe UI", 7, "bold"), bg=_CARD, fg=_DIM
                 ).pack(anchor="w", pady=(4, 2))
        prev = tk.Label(card, text="— run a benchmark to see the image —",
                        font=("Segoe UI", 7), bg=_CARD2, fg=_DIM,
                        width=50, height=10, anchor="center")
        prev.pack(fill="x")
        self._preview_labels["ip"] = prev
        # Show the input image right away if it exists
        _sunset = os.path.join(_HERE, "test_input", "sunset.jpg")
        if os.path.isfile(_sunset):
            self.after(200, lambda: self._show_preview("ip", _sunset))

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

    # ── Benchmark launcher ────────────────────────────────────────────────────
    def _run(self, bench: str, mode: str):
        if self._busy:
            self._log_append("⚠  A benchmark is already running — please wait.\n")
            return
        self._busy = True
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
                    score, elapsed, workers, chunks = _rt.run_benchmark(m)
                    img_path = _rt._save_image(
                        _rt.RENDER_WIDTH, _rt.RENDER_HEIGHT, chunks,
                        base=os.path.join(_HERE, "benchmark_render"),
                    )
                    q.put(("rt_score", m, score, elapsed, workers, img_path))
                    q.put(("show_image", "rt", img_path))
                else:
                    in_dir = os.path.join(_HERE, "test_input")
                    score, elapsed, workers, count = _ip.run_benchmark(
                        m, input_dir=in_dir
                    )
                    q.put(("ip_score", m, score, elapsed, workers, count))
                    # Show first processed output if saved, else keep input
                    out_dir = os.path.join(_HERE, "benchmark_output")
                    if os.path.isdir(out_dir):
                        outs = sorted(
                            f for f in os.listdir(out_dir)
                            if f.lower().endswith((".png", ".jpg", ".ppm"))
                        )
                        if outs:
                            q.put(("show_image", "ip",
                                   os.path.join(out_dir, outs[0])))

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

        elif kind == "pb_stop":
            self._pb[msg[1]].stop()
            self._pb[msg[1]].configure(value=0)

        elif kind == "show_image":
            _, bench, path = msg
            self._show_preview(bench, path)

        elif kind == "rt_score":
            _, m, score, elapsed, workers, img_path = msg
            self._scores[f"rt_{m}"].set(f"{score:,}")
            self._log_append(
                f"\n  ✔  Ray Tracer [{m.upper()}]\n"
                f"     Score   : {score:>10,} pts\n"
                f"     Time    : {elapsed:>10.2f} s\n"
                f"     Workers : {workers:>10}\n"
                f"     Image   : {img_path}\n"
            )

        elif kind == "ip_score":
            _, m, score, elapsed, workers, count = msg
            self._scores[f"ip_{m}"].set(f"{score:,}")
            self._log_append(
                f"\n  ✔  Image Processor [{m.upper()}]\n"
                f"     Score   : {score:>10,} pts\n"
                f"     Time    : {elapsed:>10.2f} s\n"
                f"     Workers : {workers:>10}\n"
                f"     Images  : {count:>10}\n"
            )

        elif kind == "done":
            self._busy = False
            self._status.set("Ready")
            for b in self._buttons:
                b.configure(state="normal")

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

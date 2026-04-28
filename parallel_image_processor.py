#!/usr/bin/env python3
"""
Parallel Image Processor  v1.0
================================
Console-based parallel image processing tool + benchmark.

Modes:
  benchmark  — compare Single-Thread vs Multi-Core on synthetic images
  process    — apply chosen filters to every image in an input folder

Operations: blur, sharpen, edge, emboss, sepia, grayscale, contrast, autocontrast

Score formula (same convention as PyRender Benchmark):
  single score = 1 000 × (REF_TIME / elapsed)
  multi  score = 1 000 × cpu_count × (REF_TIME / elapsed)

Requirements : Python 3.8+
Optional     : Pillow  →  pip install Pillow   (strongly recommended)
               NumPy   →  pip install numpy    (faster pixel math)
"""

import io
import math
import multiprocessing
import os
import shutil
import struct
import sys
import tempfile
import time
import zlib
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Version & configuration
# ──────────────────────────────────────────────────────────────────────────────
VERSION = "1.0"

# Benchmark workload (synthetic images)
BENCH_WIDTH  = 1920
BENCH_HEIGHT = 1080
BENCH_COUNT  = 8          # number of synthetic test images
BENCH_PASSES = 4          # filter passes per image  (increases CPU load)

# Operations applied during benchmark  (all available operations)
BENCH_OPERATIONS = ["blur", "sharpen", "contrast", "sepia", "edge"]

# Number of horizontal strips the image is split into for panel processing
PANEL_COUNT = 8

# Reference time → 1 000 pts on the calibration machine (seconds)
REF_TIME_SINGLE = 60.0
REF_TIME_MULTI  = 60.0

# Supported input extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".ppm"}

# ──────────────────────────────────────────────────────────────────────────────
# Optional-dependency detection
# ──────────────────────────────────────────────────────────────────────────────
try:
    from PIL import Image as _PILImage, ImageFilter, ImageOps, ImageEnhance
    _PILImage.MAX_IMAGE_PIXELS = None   # allow very large images
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

try:
    import numpy as _np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


# ══════════════════════════════════════════════════════════════════════════════
# Minimal PNG encoder  (used when Pillow is not installed)
# ══════════════════════════════════════════════════════════════════════════════

def _png_chunk(tag: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(tag + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)


def _encode_png(pixels: bytes, width: int, height: int) -> bytes:
    """Encode raw RGB bytes to a valid PNG binary (no external libs)."""
    raw = b""
    stride = width * 3
    for y in range(height):
        raw += b"\x00" + pixels[y * stride: (y + 1) * stride]   # filter-none rows
    sig  = b"\x89PNG\r\n\x1a\n"
    ihdr = _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    idat = _png_chunk(b"IDAT", zlib.compress(raw, 6))
    iend = _png_chunk(b"IEND", b"")
    return sig + ihdr + idat + iend


def _save_raw_as_png(path: str, pixels: bytes, width: int, height: int) -> None:
    with open(path, "wb") as fh:
        fh.write(_encode_png(pixels, width, height))


def _save_raw_as_ppm(path: str, pixels: bytes, width: int, height: int) -> None:
    with open(path, "wb") as fh:
        fh.write(f"P6\n{width} {height}\n255\n".encode())
        fh.write(pixels)


# ══════════════════════════════════════════════════════════════════════════════
# Synthetic image generation  (no dependencies)
# ══════════════════════════════════════════════════════════════════════════════

def _generate_synthetic_raw(width: int, height: int, variant: int) -> bytes:
    """
    Generate a graduated colour-pattern image as raw RGB bytes.
    Different variants produce different gradients/patterns to exercise
    all pixel paths during the benchmark.
    """
    buf = bytearray(width * height * 3)
    idx = 0
    pi2 = math.pi * 2.0
    for y in range(height):
        fy = y / max(height - 1, 1)
        for x in range(width):
            fx = x / max(width - 1, 1)
            v  = variant % 8
            if v == 0:
                r, g, b = fx, fy, 1.0 - fx
            elif v == 1:
                r, g, b = fy, 1.0 - fy, fx
            elif v == 2:
                r, g, b = fx * fy, 1.0 - fx, fx
            elif v == 3:
                s = math.sin(fx * pi2 * 3) * 0.5 + 0.5
                r, g, b = s, fy, 1.0 - s
            elif v == 4:
                r = math.sin(fx * pi2 * 4 + variant) * 0.5 + 0.5
                g = math.cos(fy * pi2 * 4 + variant) * 0.5 + 0.5
                b = math.sin((fx + fy) * pi2 * 2)    * 0.5 + 0.5
            elif v == 5:
                r = fx;  g = 1.0 - fy;  b = (fx + fy) * 0.5
            elif v == 6:
                freq = pi2 * 6
                r = math.sin(fx * freq) ** 2
                g = math.cos(fy * freq) ** 2
                b = math.sin((fx - fy) * freq) ** 2
            else:
                d = math.sqrt((fx - 0.5) ** 2 + (fy - 0.5) ** 2) * 2.0
                r, g, b = 1.0 - d, d * 0.5, d
            buf[idx]   = int(max(0.0, min(1.0, r)) * 255)
            buf[idx+1] = int(max(0.0, min(1.0, g)) * 255)
            buf[idx+2] = int(max(0.0, min(1.0, b)) * 255)
            idx += 3
    return bytes(buf)


def _generate_synthetic_pil(width: int, height: int, variant: int):
    """Generate a synthetic PIL Image."""
    raw = _generate_synthetic_raw(width, height, variant)
    return _PILImage.frombytes("RGB", (width, height), raw)


# ══════════════════════════════════════════════════════════════════════════════
# Pure-Python image filters  (slow fallback, no dependencies required)
# ══════════════════════════════════════════════════════════════════════════════

def _raw_grayscale(px: bytes, w: int, h: int) -> bytes:
    buf = bytearray(len(px))
    for i in range(0, len(px), 3):
        lum = int(px[i] * 0.299 + px[i+1] * 0.587 + px[i+2] * 0.114)
        buf[i] = buf[i+1] = buf[i+2] = lum
    return bytes(buf)


def _raw_sepia(px: bytes, w: int, h: int) -> bytes:
    buf = bytearray(len(px))
    for i in range(0, len(px), 3):
        r, g, b = px[i], px[i+1], px[i+2]
        buf[i]   = min(255, int(r * 0.393 + g * 0.769 + b * 0.189))
        buf[i+1] = min(255, int(r * 0.349 + g * 0.686 + b * 0.168))
        buf[i+2] = min(255, int(r * 0.272 + g * 0.534 + b * 0.131))
    return bytes(buf)


def _raw_contrast(px: bytes, w: int, h: int, factor: float = 1.5) -> bytes:
    buf = bytearray(len(px))
    mid = 128.0
    for i in range(0, len(px), 3):
        buf[i]   = max(0, min(255, int((px[i]   - mid) * factor + mid)))
        buf[i+1] = max(0, min(255, int((px[i+1] - mid) * factor + mid)))
        buf[i+2] = max(0, min(255, int((px[i+2] - mid) * factor + mid)))
    return bytes(buf)


def _raw_box_blur(px: bytes, w: int, h: int, radius: int = 3) -> bytes:
    """Separable box blur (horizontal then vertical pass)."""
    # --- horizontal pass ---
    tmp = bytearray(len(px))
    for y in range(h):
        for x in range(w):
            sr = sg = sb = cnt = 0
            for kx in range(max(0, x - radius), min(w, x + radius + 1)):
                base = (y * w + kx) * 3
                sr += px[base]; sg += px[base+1]; sb += px[base+2]
                cnt += 1
            base = (y * w + x) * 3
            tmp[base]   = sr // cnt
            tmp[base+1] = sg // cnt
            tmp[base+2] = sb // cnt
    # --- vertical pass ---
    out = bytearray(len(px))
    for y in range(h):
        for x in range(w):
            sr = sg = sb = cnt = 0
            for ky in range(max(0, y - radius), min(h, y + radius + 1)):
                base = (ky * w + x) * 3
                sr += tmp[base]; sg += tmp[base+1]; sb += tmp[base+2]
                cnt += 1
            base = (y * w + x) * 3
            out[base]   = sr // cnt
            out[base+1] = sg // cnt
            out[base+2] = sb // cnt
    return bytes(out)


def _raw_sharpen(px: bytes, w: int, h: int) -> bytes:
    """3×3 sharpen kernel applied to each colour channel."""
    K = (0, -1, 0, -1, 5, -1, 0, -1, 0)
    buf = bytearray(len(px))
    # Copy border pixels unchanged
    for y in range(h):
        for x in range(w):
            base = (y * w + x) * 3
            buf[base:base+3] = px[base:base+3]
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            sr = sg = sb = 0
            ki = 0
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    b2 = ((y + dy) * w + (x + dx)) * 3
                    k  = K[ki]
                    sr += px[b2]   * k
                    sg += px[b2+1] * k
                    sb += px[b2+2] * k
                    ki += 1
            base = (y * w + x) * 3
            buf[base]   = max(0, min(255, sr))
            buf[base+1] = max(0, min(255, sg))
            buf[base+2] = max(0, min(255, sb))
    return bytes(buf)


def _raw_sobel_edge(px: bytes, w: int, h: int) -> bytes:
    """Sobel edge-detection → greyscale magnitude output."""
    gray = [int(px[i] * 0.299 + px[i+1] * 0.587 + px[i+2] * 0.114)
            for i in range(0, len(px), 3)]
    buf = bytearray(len(px))
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            def g(dx, dy): return gray[(y + dy) * w + (x + dx)]
            gx = (-g(-1,-1) - 2*g(-1, 0) - g(-1, 1) +
                   g( 1,-1) + 2*g( 1, 0) + g( 1, 1))
            gy = (-g(-1,-1) - 2*g( 0,-1) - g( 1,-1) +
                   g(-1, 1) + 2*g( 0, 1) + g( 1, 1))
            mag  = min(255, int(math.sqrt(gx * gx + gy * gy)))
            base = (y * w + x) * 3
            buf[base] = buf[base+1] = buf[base+2] = mag
    return bytes(buf)


def _apply_raw_operations(px: bytes, w: int, h: int,
                          ops: list, passes: int) -> bytes:
    """Run each operation *passes* times using pure Python."""
    for _ in range(passes):
        for op in ops:
            if   op == "grayscale":    px = _raw_grayscale(px, w, h)
            elif op == "sepia":        px = _raw_sepia(px, w, h)
            elif op == "contrast":     px = _raw_contrast(px, w, h)
            elif op == "blur":         px = _raw_box_blur(px, w, h)
            elif op == "sharpen":      px = _raw_sharpen(px, w, h)
            elif op == "edge":         px = _raw_sobel_edge(px, w, h)
    return px


# ══════════════════════════════════════════════════════════════════════════════
# Pillow-based filters  (fast, preferred when Pillow is installed)
# ══════════════════════════════════════════════════════════════════════════════

def _apply_pil_operations(img, ops: list, passes: int):
    """Apply each filter *passes* times to a PIL Image and return the result."""
    for _ in range(passes):
        for op in ops:
            if op == "blur":
                img = img.filter(ImageFilter.GaussianBlur(radius=3))
            elif op == "sharpen":
                img = img.filter(
                    ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
            elif op == "edge":
                img = img.filter(ImageFilter.FIND_EDGES)
            elif op == "emboss":
                img = img.filter(ImageFilter.EMBOSS)
            elif op == "grayscale":
                img = ImageOps.grayscale(img).convert("RGB")
            elif op == "sepia":
                gray = ImageOps.grayscale(img)
                r = gray.point(lambda v: min(255, int(v * 1.08)))
                g = gray.point(lambda v: int(v * 0.85))
                b = gray.point(lambda v: int(v * 0.66))
                img = _PILImage.merge("RGB", (r, g, b))
            elif op == "contrast":
                img = ImageEnhance.Contrast(img).enhance(1.5)
            elif op == "autocontrast":
                img = ImageOps.autocontrast(img)
    return img


# ══════════════════════════════════════════════════════════════════════════════
# Multiprocessing worker
#   MUST be a top-level function to be picklable on Windows.
# ══════════════════════════════════════════════════════════════════════════════

def _worker(args):
    """
    Process one image and optionally save the result.

    args = (index, source, width, height, ops, passes, use_pil, out_dir)
      source  : str (file path) | bytes (raw RGB for pure-Python path)
                PIL Image is not passed directly (not picklable)
    Returns (index, width, height, elapsed_ms)
    """
    idx, source, width, height, ops, passes, use_pil, out_dir = args
    t0 = time.perf_counter()

    if use_pil:
        from PIL import Image as _Img
        if isinstance(source, str):
            img = _Img.open(source).convert("RGB")
            width, height = img.size
            stem = Path(source).stem
        else:
            # source is raw bytes of a pre-generated synthetic image
            img = _Img.frombytes("RGB", (width, height), source)
            stem = f"synthetic_{idx:04d}"
        result = _apply_pil_operations(img, ops, passes)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            op_tag = "_".join(ops) if len(ops) <= 3 else f"{len(ops)}filters"
            out_path = os.path.join(out_dir, f"{stem}_{op_tag}_{idx:04d}.png")
            result.save(out_path)
    else:
        if isinstance(source, str):
            # Minimal P6 PPM reader (pure Python)
            with open(source, "rb") as fh:
                magic = fh.readline().strip()
                if magic != b"P6":
                    raise ValueError(f"Unsupported format for pure-Python mode: {source}")
                while True:
                    line = fh.readline().strip()
                    if not line.startswith(b"#"):
                        break
                w2, h2 = map(int, line.split())
                fh.readline()   # maxval
                raw = fh.read()
            width, height, source = w2, h2, raw
        result = _apply_raw_operations(source, width, height, ops, passes)
        if out_dir:
            out_path = os.path.join(out_dir, f"processed_{idx:04d}.png")
            _save_raw_as_png(out_path, result, width, height)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return idx, width, height, elapsed_ms


# ══════════════════════════════════════════════════════════════════════════════
# Panel worker  (top-level so it is picklable on Windows)
# ══════════════════════════════════════════════════════════════════════════════

def _panel_worker(args):
    """
    Process one horizontal strip (panel) of an image with PIL.
    args = (idx, image_path, x0, y0, x1, y1, ops, passes[, src_y0, src_y1])
    src_y0/src_y1: Y coordinates in the *original* full image (for GUI placement).
    Returns (idx, src_y0, src_y1, png_bytes, elapsed_ms)
    """
    idx, image_path, x0, y0, x1, y1, ops, passes = args[:8]
    src_y0 = args[8] if len(args) > 8 else y0
    src_y1 = args[9] if len(args) > 9 else y1
    t0 = time.perf_counter()
    from PIL import Image as _Img
    _Img.MAX_IMAGE_PIXELS = None
    img = _Img.open(image_path).convert("RGB")
    panel = img.crop((x0, y0, x1, y1))
    img.close()
    result = _apply_pil_operations(panel, ops, passes)
    buf = io.BytesIO()
    result.save(buf, format="PNG")
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return idx, src_y0, src_y1, buf.getvalue(), elapsed_ms


# ══════════════════════════════════════════════════════════════════════════════
# Console display helpers
# ══════════════════════════════════════════════════════════════════════════════

_BANNER = (
    "\n"
    "  ╔═══════════════════════════════════════════════════════════╗\n"
    "  ║    Parallel Image Processor  ·  Console Edition          ║\n"
    "  ║                              v{ver:<6}                      ║\n"
    "  ╚═══════════════════════════════════════════════════════════╝"
)


def _print_banner():
    print(_BANNER.format(ver=VERSION))
    cpu = multiprocessing.cpu_count()
    pil = "yes" if _HAS_PIL   else "no  (pip install Pillow)"
    np_ = "yes" if _HAS_NUMPY else "no  (pip install numpy)"
    print(f"  ├─ CPU    : {cpu} logical core(s)")
    print(f"  ├─ OS     : {sys.platform}")
    print(f"  ├─ Python : {sys.version.split()[0]}")
    print(f"  ├─ Pillow : {pil}")
    print(f"  └─ NumPy  : {np_}\n")


def _print_progress(done: int, total: int, elapsed: float, label: str = "") -> None:
    pct  = done / total
    fill = int(pct * 40)
    bar  = "█" * fill + "░" * (40 - fill)
    eta  = (f"{elapsed / pct * (1.0 - pct):5.1f}s"
            if elapsed > 0 and pct > 0 else " ----")
    tag  = f"  {label}" if label else ""
    print(f"\r  [{bar}] {pct * 100:5.1f}%  {elapsed:6.1f}s elapsed  ETA {eta}{tag}",
          end="", flush=True)


def _print_section(label: str, width_px: int, height_px: int,
                   count: int, passes: int, workers: int) -> None:
    sep = "─" * (53 - len(label))
    print(f"\n  ┌─ {label} {sep}┐")
    print(f"  │  {width_px} × {height_px} px  │  {count} image(s)"
          f"  │  {passes} pass(es)  │  {workers} worker(s)  │")
    print(f"  └{'─' * 63}┘\n")


def _print_result(label: str, score: int, elapsed: float, workers: int,
                  img_count: int) -> None:
    bar = "█" * min(50, max(1, score // 80))
    sep = "─" * 58
    print(f"\n  {sep}")
    print(f"  {label}")
    print(f"  {sep}")
    print(f"  Score    : {score:>10,} pts")
    print(f"  Time     : {elapsed:>10.2f} s")
    print(f"  Images   : {img_count:>10}")
    print(f"  Workers  : {workers:>10}")
    print(f"  Img/s    : {img_count / elapsed:>10.2f}")
    print(f"  {bar}")
    print(f"  {sep}")


def _print_summary(results: dict, saved_to: str) -> None:
    sep = "═" * 58
    print(f"\n  {sep}")
    print("  BENCHMARK SUMMARY")
    print(f"  {sep}")
    if "single" in results:
        sc, st = results["single"]
        print(f"  Single-Thread : {sc:>10,} pts   ({st:.2f} s)")
    if "multi" in results:
        mc, mt = results["multi"]
        print(f"  Multi-Core    : {mc:>10,} pts   ({mt:.2f} s)")
        if "single" in results:
            ratio = mc / max(results["single"][0], 1)
            print(f"  MP Ratio      :    {ratio:8.2f}×  "
                  f"({multiprocessing.cpu_count()} cores available)")
    if saved_to:
        print(f"\n  Output folder : {saved_to}")
    print(f"  {sep}\n")


# ══════════════════════════════════════════════════════════════════════════════
# Benchmark runners
# ══════════════════════════════════════════════════════════════════════════════

def _build_task_list(sources, width: int, height: int,
                     ops: list, passes: int,
                     use_pil: bool, out_dir: str) -> list:
    """Build the list of args tuples for the worker function."""
    return [
        (i, src, width, height, ops, passes, use_pil, out_dir)
        for i, src in enumerate(sources)
    ]


def process_image_panels(image_path: str, mode: str,
                         ops: list = None, passes: int = None,
                         panel_count: int = None):
    """
    Generator: opens *image_path* at full native resolution, splits into
    *panel_count* horizontal strips, applies filter ops to each, and yields
    panel results as they complete.

    Single mode  → panels processed sequentially top-to-bottom.
    Multi  mode  → panels processed in parallel; yielded as each finishes
                   (may arrive out of order, showing parallelism visually).

    Yields: first ("init", img_w, img_h, preview_png_bytes)
            then  ("panel", idx, src_y0, src_y1, img_w, img_h, png_bytes, ms)
    """
    if not _HAS_PIL:
        raise RuntimeError("Pillow is required for panel processing.  "
                           "Run: pip install Pillow")
    if ops is None:
        ops = BENCH_OPERATIONS
    if passes is None:
        passes = BENCH_PASSES
    if panel_count is None:
        panel_count = PANEL_COUNT

    from PIL import Image as _Img
    _Img.MAX_IMAGE_PIXELS = None

    # ── Load the full-resolution image once ──────────────────────────────
    print(f"  Opening image …", flush=True)
    img = _Img.open(image_path).convert("RGB")
    img_w, img_h = img.size
    mpx = img_w * img_h / 1_000_000
    print(f"  Full-resolution: {img_w}×{img_h} px  ({mpx:.0f} Mpx)")

    # Small preview thumbnail for the GUI canvas (created before cropping)
    preview = img.copy()
    preview.thumbnail((760, 190), _Img.LANCZOS)
    preview_buf = io.BytesIO()
    preview.save(preview_buf, format="PNG")
    preview_png_bytes = preview_buf.getvalue()
    preview.close()

    # ── Pre-crop each strip and save to a temp BMP (uncompressed = instant I/O)
    # This avoids having every worker independently decode the full 1.5 GB image.
    tmp_dir = tempfile.mkdtemp(prefix="pip_panels_")
    panel_meta = []   # (idx, bmp_path, src_y0, src_y1, strip_h)
    print(f"  Slicing {panel_count} panels …", flush=True)
    for i in range(panel_count):
        y0 = (i * img_h) // panel_count
        y1 = ((i + 1) * img_h) // panel_count
        strip = img.crop((0, y0, img_w, y1))
        bmp_path = os.path.join(tmp_dir, f"panel_{i:02d}.bmp")
        strip.save(bmp_path, format="BMP")
        strip.close()
        panel_meta.append((i, bmp_path, y0, y1, y1 - y0))
        print(f"    sliced {i + 1}/{panel_count}", end="\r", flush=True)
    img.close()
    print()

    try:
        yield "init", img_w, img_h, preview_png_bytes
        del preview_png_bytes

        cpu_count = multiprocessing.cpu_count()
        workers   = 1 if mode == "single" else cpu_count

        # Each task passes src_y0/src_y1 so workers return proper canvas coords
        tasks = [
            (i, bmp_path, 0, 0, img_w, strip_h, ops, passes, src_y0, src_y1)
            for i, bmp_path, src_y0, src_y1, strip_h in panel_meta
        ]

        label = "SINGLE-THREAD" if mode == "single" else "MULTI-CORE"
        print(f"  [{label}]  {panel_count} panels  ·  {workers} worker(s)  ·  "
              f"{len(ops)} filter(s) × {passes} pass(es)")

        if workers == 1:
            for task in tasks:
                idx, src_y0, src_y1, png_bytes, elapsed_ms = _panel_worker(task)
                print(f"    panel {idx + 1:>2}/{panel_count}  {elapsed_ms:7.0f} ms")
                yield "panel", idx, src_y0, src_y1, img_w, img_h, png_bytes, elapsed_ms
        else:
            with multiprocessing.Pool(workers) as pool:
                for result in pool.imap_unordered(_panel_worker, tasks):
                    idx, src_y0, src_y1, png_bytes, elapsed_ms = result
                    print(f"    panel {idx + 1:>2}/{panel_count}  {elapsed_ms:7.0f} ms")
                    yield "panel", idx, src_y0, src_y1, img_w, img_h, png_bytes, elapsed_ms
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def run_benchmark(mode: str, ops: list = None, out_dir: str = "",
                  input_dir: str = ""):
    """
    Run the image-processing benchmark in 'single' or 'multi' mode.

    If *input_dir* contains supported images those are used as the workload
    (replicated to reach BENCH_COUNT tasks when fewer images are found).
    Otherwise synthetic gradient images are generated automatically.

    Returns (score, elapsed_seconds, worker_count, image_count)
    """
    if ops is None:
        ops = BENCH_OPERATIONS

    cpu_count = multiprocessing.cpu_count()
    workers   = 1 if mode == "single" else cpu_count
    use_pil   = _HAS_PIL

    # ── Prefer real images from input_dir (or the default test_input folder) ──
    script_dir    = os.path.dirname(os.path.abspath(__file__))
    default_input = os.path.join(script_dir, "test_input")
    search_dir    = input_dir if input_dir else default_input

    real_files = []
    if os.path.isfile(search_dir) and Path(search_dir).suffix.lower() in IMAGE_EXTENSIONS:
        real_files = [search_dir]
    elif os.path.isdir(search_dir):
        real_files = sorted(
            str(p) for p in Path(search_dir).iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )

    if real_files:
        # Replicate the file list to reach BENCH_COUNT tasks
        sources = [real_files[i % len(real_files)] for i in range(BENCH_COUNT)]
        bench_w = bench_h = 0   # worker reads dimensions from the file
        print(f"  Using {len(real_files)} real image(s) from: {search_dir}")
        print(f"  ({BENCH_COUNT} tasks = {len(real_files)} image(s) × "
              f"{BENCH_COUNT // max(len(real_files), 1) or 1} repetition(s))")
    else:
        # Fall back to synthetic gradient images
        print("  Generating synthetic test images ...", end="", flush=True)
        sources = [
            _generate_synthetic_raw(BENCH_WIDTH, BENCH_HEIGHT, i)
            for i in range(BENCH_COUNT)
        ]
        bench_w, bench_h = BENCH_WIDTH, BENCH_HEIGHT
        print(f" {BENCH_COUNT} images ready.")

    label = "SINGLE-THREAD" if mode == "single" else "MULTI-CORE"
    _print_section(label, bench_w, bench_h,
                   BENCH_COUNT, BENCH_PASSES, workers)

    tasks  = _build_task_list(sources, bench_w, bench_h,
                              ops, BENCH_PASSES, use_pil, out_dir)
    total  = len(tasks)
    done   = 0
    t0     = time.perf_counter()

    if workers == 1:
        for task in tasks:
            _worker(task)
            done += 1
            _print_progress(done, total, time.perf_counter() - t0)
    else:
        with multiprocessing.Pool(workers) as pool:
            for _ in pool.imap_unordered(_worker, tasks):
                done += 1
                _print_progress(done, total, time.perf_counter() - t0)

    elapsed = time.perf_counter() - t0
    print()

    ref_time    = REF_TIME_SINGLE if mode == "single" else REF_TIME_MULTI
    scale       = 1 if mode == "single" else cpu_count
    score       = int(1000 * scale * ref_time / elapsed)

    return score, elapsed, workers, BENCH_COUNT


# ══════════════════════════════════════════════════════════════════════════════
# Folder processing mode
# ══════════════════════════════════════════════════════════════════════════════

def process_folder(in_dir: str, out_dir: str, ops: list,
                   passes: int = 1, workers: int = None) -> dict:
    """
    Apply *ops* to every supported image in *in_dir* and save to *out_dir*.

    Returns {'elapsed': float, 'count': int, 'workers': int}
    """
    if workers is None:
        workers = multiprocessing.cpu_count()

    in_path  = Path(in_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        p for p in in_path.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_files:
        print(f"  No supported images found in: {in_dir}")
        return {}

    use_pil = _HAS_PIL
    if not use_pil:
        ppm_only = [p for p in image_files if p.suffix.lower() == ".ppm"]
        if len(ppm_only) != len(image_files):
            print("  WARNING: Pillow not installed. Only .ppm files can be read "
                  "in pure-Python mode.")
        image_files = ppm_only

    tasks = [
        (i, str(f), 0, 0, ops, passes, use_pil, str(out_path))
        for i, f in enumerate(image_files)
    ]

    label = "MULTI-CORE" if workers > 1 else "SINGLE-THREAD"
    _print_section(label, 0, 0, len(tasks), passes, workers)

    total  = len(tasks)
    done   = 0
    t0     = time.perf_counter()

    if workers == 1:
        for task in tasks:
            idx, *_, elapsed_ms = _worker(task)
            done += 1
            _print_progress(done, total, time.perf_counter() - t0,
                            label=image_files[idx].name[:30])
    else:
        with multiprocessing.Pool(workers) as pool:
            for idx, *_, elapsed_ms in pool.imap_unordered(_worker, tasks):
                done += 1
                _print_progress(done, total, time.perf_counter() - t0,
                                label=image_files[idx].name[:30])

    elapsed = time.perf_counter() - t0
    print()
    return {"elapsed": elapsed, "count": total, "workers": workers}


# ══════════════════════════════════════════════════════════════════════════════
# Interactive console menu
# ══════════════════════════════════════════════════════════════════════════════

_ALL_OPS = ["blur", "sharpen", "edge", "emboss", "sepia",
            "grayscale", "contrast", "autocontrast"]


def _choose_operations() -> list:
    print("\n  Available operations:")
    for i, op in enumerate(_ALL_OPS, 1):
        print(f"    {i}. {op}")
    print("    A. All operations  (benchmark default)")
    raw = input("\n  Enter numbers separated by commas, or A for all: ").strip()
    if raw.upper() == "A":
        return list(_ALL_OPS)
    chosen = []
    for part in raw.split(","):
        part = part.strip()
        if part.isdigit():
            n = int(part) - 1
            if 0 <= n < len(_ALL_OPS):
                chosen.append(_ALL_OPS[n])
    return chosen if chosen else list(BENCH_OPERATIONS)


def _menu_benchmark() -> None:
    """Run the full single + multi-core benchmark."""
    results = {}
    out_dir = ""

    script_dir  = os.path.dirname(os.path.abspath(__file__))
    default_in  = os.path.join(script_dir, "test_input")
    has_real    = os.path.isdir(default_in) and any(
        Path(default_in).iterdir()
    )

    if has_real:
        print(f"\n  Found images in test_input/ — those will be used as the workload.")
    else:
        print("\n  No images in test_input/ — synthetic images will be generated.")

    save = input("\n  Save processed benchmark images? (y/N): ").strip().lower()
    if save == "y":
        out_dir = os.path.join(script_dir, "benchmark_output")
        os.makedirs(out_dir, exist_ok=True)

    # ── Single-Thread ─────────────────────────────────────────────────────────
    input("\n  Press ENTER to start Single-Thread test ...")
    score, elapsed, workers, count = run_benchmark(
        "single", out_dir=out_dir, input_dir=default_in)
    _print_result("SINGLE-THREAD RESULT", score, elapsed, workers, count)
    results["single"] = (score, elapsed)

    # ── Multi-Core ────────────────────────────────────────────────────────────
    if multiprocessing.cpu_count() > 1:
        input("\n  Press ENTER to start Multi-Core test ...")
        score, elapsed, workers, count = run_benchmark(
            "multi", out_dir=out_dir, input_dir=default_in)
        _print_result("MULTI-CORE RESULT", score, elapsed, workers, count)
        results["multi"] = (score, elapsed)

    _print_summary(results, out_dir)


def _menu_process() -> None:
    """Process images from a user-supplied folder."""
    in_dir = input("\n  Input folder path  : ").strip().strip('"')
    if not os.path.isdir(in_dir):
        print(f"  ERROR: '{in_dir}' is not a valid directory.")
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_out = os.path.join(script_dir, "processed_output")
    raw_out = input(f"  Output folder path [{default_out}]: ").strip().strip('"')
    out_dir = raw_out if raw_out else default_out

    ops = _choose_operations()

    raw_passes = input("\n  Filter passes per image [1]: ").strip()
    passes = int(raw_passes) if raw_passes.isdigit() and int(raw_passes) > 0 else 1

    raw_workers = input(
        f"  Worker processes (1 = single-thread, "
        f"{multiprocessing.cpu_count()} = all cores) [{multiprocessing.cpu_count()}]: "
    ).strip()
    workers = (int(raw_workers)
               if raw_workers.isdigit() and int(raw_workers) > 0
               else multiprocessing.cpu_count())
    workers = min(workers, multiprocessing.cpu_count())

    print(f"\n  Operations : {', '.join(ops)}")
    print(f"  Passes     : {passes}")
    print(f"  Workers    : {workers}")
    confirm = input("  Start processing? (Y/n): ").strip().lower()
    if confirm == "n":
        return

    info = process_folder(in_dir, out_dir, ops, passes, workers)
    if info:
        sep = "─" * 58
        print(f"\n  {sep}")
        print("  PROCESSING COMPLETE")
        print(f"  {sep}")
        print(f"  Images processed : {info['count']}")
        print(f"  Total time       : {info['elapsed']:.2f} s")
        print(f"  Images per sec   : {info['count'] / info['elapsed']:.2f}")
        print(f"  Workers used     : {info['workers']}")
        print(f"  Output folder    : {out_dir}")
        print(f"  {sep}\n")


def _main_menu() -> None:
    """Interactive top-level menu."""
    while True:
        print("\n  ┌─ MAIN MENU ──────────────────────────────────────────────┐")
        print("  │  1. Run benchmark (single-thread vs multi-core)          │")
        print("  │  2. Process a folder of images                           │")
        print("  │  3. Exit                                                 │")
        print("  └──────────────────────────────────────────────────────────┘")
        choice = input("\n  Select an option [1-3]: ").strip()
        if choice == "1":
            _menu_benchmark()
        elif choice == "2":
            _menu_process()
        elif choice == "3":
            print("\n  Goodbye!\n")
            break
        else:
            print("  Invalid choice. Please enter 1, 2, or 3.")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    _print_banner()
    _main_menu()


if __name__ == "__main__":
    # Required on Windows so spawned worker processes do not re-run main().
    multiprocessing.freeze_support()
    main()

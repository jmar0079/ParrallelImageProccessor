#!/usr/bin/env python3
"""
PyRender Benchmark v1.0
=======================
A Cinebench-inspired CPU benchmark that ray-traces a 3-D scene and converts
render time into a comparable point score.

  Single-Core test — renders using exactly one thread (no parallelism).
  Multi-Core  test — renders using every logical CPU core via multiprocessing.

Score formula (mirrors Cinebench R23 convention):
  single score = 1 000 × (REF_TIME / actual_seconds)
  multi  score = 1 000 × cpu_count × (REF_TIME / actual_seconds)

Requirements: Python 3.8+
Optional     : Pillow  → pip install Pillow   (for PNG output; PPM fallback)
"""

import math
import multiprocessing
import os
import sys
import time

# ──────────────────────────────────────────────────────────────────────────────
# Configuration — tweak to make the benchmark shorter / longer
# ──────────────────────────────────────────────────────────────────────────────
VERSION       = "1.0"
RENDER_WIDTH  = 640           # output image width  (pixels)
RENDER_HEIGHT = 360           # output image height (pixels)
MAX_DEPTH     = 5             # maximum reflection bounces per ray
SAMPLES       = 1             # samples per pixel  (1 = no AA; 2 = 2×2 AA)

# Calibration: the time (seconds) on the reference machine that earns 1 000 pts.
# Lower  → harder to score high  |  Higher → easier to score high.
REF_TIME_SINGLE = 90.0
REF_TIME_MULTI  = 90.0

EPSILON = 1e-4
INF     = float("inf")

# ──────────────────────────────────────────────────────────────────────────────
# Environment map  (loaded once per worker process from test_input/sunset.jpg)
# ──────────────────────────────────────────────────────────────────────────────
_env_pixels = None   # raw RGB bytes, or b"" if unavailable
_env_w      = 0
_env_h      = 0


def _load_env_image():
    """Load test_input/sunset.jpg as an equirectangular environment map.

    Called once at the start of each worker process; results are cached in
    module-level globals so subsequent chunks reuse the same data.
    """
    global _env_pixels, _env_w, _env_h
    if _env_pixels is not None:
        return                           # already loaded (or already failed)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path       = os.path.join(script_dir, "test_input", "sunset.jpg")
    if not os.path.isfile(path):
        _env_pixels = b""
        return
    try:
        from PIL import Image
        img        = Image.open(path).convert("RGB")
        _env_w, _env_h = img.size
        _env_pixels = img.tobytes()
    except Exception:
        _env_pixels = b""


# ──────────────────────────────────────────────────────────────────────────────
# Vec3 — minimalist 3-D vector (pure Python for maximum CPU load)
# ──────────────────────────────────────────────────────────────────────────────
class Vec3:
    __slots__ = ("x", "y", "z")

    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    # --- arithmetic -----------------------------------------------------------
    def __add__(self, o):    return Vec3(self.x + o.x, self.y + o.y, self.z + o.z)
    def __sub__(self, o):    return Vec3(self.x - o.x, self.y - o.y, self.z - o.z)
    def __neg__(self):       return Vec3(-self.x, -self.y, -self.z)
    def __mul__(self, t):
        if isinstance(t, Vec3):
            return Vec3(self.x * t.x, self.y * t.y, self.z * t.z)
        return Vec3(self.x * t, self.y * t, self.z * t)
    def __rmul__(self, t):   return self.__mul__(t)
    def __truediv__(self, t): return Vec3(self.x / t, self.y / t, self.z / t)

    # --- geometry -------------------------------------------------------------
    def dot(self, o):    return self.x * o.x + self.y * o.y + self.z * o.z
    def length(self):    return math.sqrt(self.dot(self))
    def normalize(self):
        l = self.length()
        if l < 1e-10:
            return Vec3()
        return Vec3(self.x / l, self.y / l, self.z / l)

    def reflect(self, n):
        """Reflect this vector around the surface normal n."""
        return self - n * (2.0 * self.dot(n))

    def clamp01(self):
        return Vec3(
            max(0.0, min(1.0, self.x)),
            max(0.0, min(1.0, self.y)),
            max(0.0, min(1.0, self.z)),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Scene data structures
# ──────────────────────────────────────────────────────────────────────────────
class Material:
    __slots__ = ("color", "ambient", "diffuse", "specular", "shininess", "reflect")

    def __init__(self, color, ambient=0.08, diffuse=0.8,
                 specular=0.5, shininess=32.0, reflect=0.0):
        self.color     = color
        self.ambient   = ambient
        self.diffuse   = diffuse
        self.specular  = specular
        self.shininess = shininess
        self.reflect   = reflect


class Sphere:
    __slots__ = ("cx", "cy", "cz", "radius", "mat")

    def __init__(self, cx, cy, cz, radius, mat):
        self.cx, self.cy, self.cz = float(cx), float(cy), float(cz)
        self.radius = float(radius)
        self.mat    = mat

    def normal_at(self, px, py, pz):
        """Return the outward unit normal at surface point (px, py, pz)."""
        dx = px - self.cx
        dy = py - self.cy
        dz = pz - self.cz
        l  = math.sqrt(dx * dx + dy * dy + dz * dz)
        return dx / l, dy / l, dz / l


class Light:
    __slots__ = ("px", "py", "pz", "cr", "cg", "cb", "intensity")

    def __init__(self, pos: Vec3, color: Vec3, intensity: float = 1.0):
        self.px, self.py, self.pz = pos.x, pos.y, pos.z
        self.cr, self.cg, self.cb = color.x, color.y, color.z
        self.intensity = intensity


# ──────────────────────────────────────────────────────────────────────────────
# Scene definition  — a Cornell-box-inspired still life with 13 spheres and
#                     3 point lights, mimicking Cinebench's depth of complexity.
# ──────────────────────────────────────────────────────────────────────────────
def build_scene():
    """Return (spheres, lights) for the standardised benchmark scene."""

    def mat(r, g, b, ambient=0.08, diffuse=0.8,
            specular=0.5, shininess=32.0, reflect=0.0):
        return Material(Vec3(r, g, b), ambient, diffuse, specular, shininess, reflect)

    spheres = [
        # Ground — enormous sphere acting as a flat plane
        Sphere(  0, -10003,    -5, 10000, mat(0.75, 0.75, 0.75, diffuse=0.9, specular=0.15, reflect=0.25)),
        # Far back wall
        Sphere(  0,      0, -10015, 10000, mat(0.65, 0.65, 0.80, diffuse=0.8, specular=0.05)),

        # Central mirror sphere (high reflectivity)
        Sphere(  0,    0.0,    -5,   1.2, mat(0.90, 0.90, 1.00, specular=1.0, shininess=512, reflect=0.92)),
        # Red   (left)
        Sphere(-2.6,   0.0,    -6,   1.0, mat(0.90, 0.12, 0.12, specular=0.7, shininess=64, reflect=0.08)),
        # Blue  (right)
        Sphere( 2.6,   0.0,    -6,   1.0, mat(0.12, 0.25, 0.95, specular=0.9, shininess=128, reflect=0.42)),
        # Gold  (front-left, semi-mirror)
        Sphere(-1.4,  -0.35,  -3.6,  0.7, mat(1.00, 0.78, 0.15, specular=1.0, shininess=256, reflect=0.58)),
        # Green (front-right, matte)
        Sphere( 1.4,  -0.35,  -3.6,  0.7, mat(0.12, 0.80, 0.18, specular=0.5, shininess=32)),
        # Purple (floating above centre)
        Sphere(  0,    1.55,  -4.5,  0.6, mat(0.62, 0.08, 0.85, specular=0.7, shininess=64)),
        # Teal  (far-left)
        Sphere(-3.8,   0.5,    -8,   0.85, mat(0.05, 0.72, 0.68, specular=0.6, shininess=48, reflect=0.18)),
        # Orange (far-right)
        Sphere( 3.8,   0.5,    -8,   0.85, mat(1.00, 0.48, 0.05, specular=0.8, shininess=80)),
        # White  (far-centre, slightly reflective)
        Sphere(  0,   -0.55,   -8,   0.5,  mat(1.00, 1.00, 1.00, specular=1.0, shininess=256, reflect=0.12)),
        # Small red accent
        Sphere(-2.0,   1.30,  -5.8,  0.4,  mat(1.00, 0.12, 0.20, specular=0.8, shininess=96)),
        # Small blue accent
        Sphere( 2.0,   1.30,  -5.8,  0.4,  mat(0.10, 0.50, 1.00, specular=0.9, shininess=96, reflect=0.30)),
    ]

    lights = [
        Light(Vec3(-5,  8, -1), Vec3(1.00, 0.95, 0.88), 1.3),
        Light(Vec3( 5,  5, -2), Vec3(0.75, 0.80, 1.00), 0.9),
        Light(Vec3( 0, 10, -6), Vec3(1.00, 1.00, 1.00), 0.55),
    ]

    return spheres, lights


# ──────────────────────────────────────────────────────────────────────────────
# Core ray-tracing  (deliberately kept in plain Python floats to maximise
# the CPU load — matching the intent of Cinebench's workload design)
# ──────────────────────────────────────────────────────────────────────────────
def _sphere_hit(ox, oy, oz, dx, dy, dz, cx, cy, cz, r):
    """
    Ray–sphere intersection using the half-coefficient form of the quadratic.
    Assumes the direction (dx, dy, dz) is a unit vector (|D|² = 1).

    Returns the nearest positive t, or INF if there is no valid hit.
    """
    bx = ox - cx;  by = oy - cy;  bz = oz - cz
    b  = bx*dx + by*dy + bz*dz          # B·D  (half b coefficient)
    c  = bx*bx + by*by + bz*bz - r*r   # |B|² - r²
    disc = b*b - c                       # discriminant  (= b² - ac, a=1)
    if disc < 0.0:
        return INF
    sq = math.sqrt(disc)
    t1 = -b - sq
    if t1 > EPSILON:
        return t1
    t2 = -b + sq
    return t2 if t2 > EPSILON else INF


def _trace(ox, oy, oz, dx, dy, dz, spheres, lights, depth):
    """
    Cast a single ray and return the Vec3 radiance at the hit point.
    Implements Blinn-Phong shading + hard shadows + recursive reflections.
    """
    # ── Find the closest sphere intersection ──────────────────────────────────
    t_min = INF
    hit   = None
    for s in spheres:
        t = _sphere_hit(ox, oy, oz, dx, dy, dz, s.cx, s.cy, s.cz, s.radius)
        if t < t_min:
            t_min = t
            hit   = s

    # ── Miss → sample the environment map (sunset.jpg) or fall back to sky ────
    if hit is None:
        if _env_pixels:
            # Equirectangular projection: map unit ray direction → UV
            u  = (math.atan2(dx, -dz) / (2.0 * math.pi) + 0.5) % 1.0
            v  = max(0.0, min(1.0, 0.5 - math.asin(max(-1.0, min(1.0, dy))) / math.pi))
            px = min(int(u * _env_w), _env_w - 1)
            py = min(int(v * _env_h), _env_h - 1)
            off = (py * _env_w + px) * 3
            return Vec3(_env_pixels[off]     / 255.0,
                        _env_pixels[off + 1] / 255.0,
                        _env_pixels[off + 2] / 255.0)
        # Fallback: procedural sky gradient when the image is unavailable
        sky = 0.5 * (dy + 1.0)
        return Vec3(0.08 + 0.12 * sky,
                    0.08 + 0.12 * sky,
                    0.15 + 0.30 * sky)

    # ── Hit point and outward surface normal ──────────────────────────────────
    hx = ox + dx * t_min
    hy = oy + dy * t_min
    hz = oz + dz * t_min
    nx, ny, nz = hit.normal_at(hx, hy, hz)

    mat = hit.mat

    # View direction = negated ray direction (already unit)
    vx, vy, vz = -dx, -dy, -dz

    # ── Ambient contribution ──────────────────────────────────────────────────
    cr = mat.color.x * mat.ambient
    cg = mat.color.y * mat.ambient
    cb = mat.color.z * mat.ambient

    # ── Per-light diffuse + specular with hard shadows ────────────────────────
    for L in lights:
        # Direction from hit point to light, and its distance
        lx = L.px - hx;  ly = L.py - hy;  lz = L.pz - hz
        ld = math.sqrt(lx*lx + ly*ly + lz*lz)
        lx /= ld;  ly /= ld;  lz /= ld

        # Shadow ray — origin nudged along normal to avoid self-intersection
        sox = hx + nx * EPSILON
        soy = hy + ny * EPSILON
        soz = hz + nz * EPSILON

        in_shadow = False
        for s2 in spheres:
            ts = _sphere_hit(sox, soy, soz, lx, ly, lz,
                             s2.cx, s2.cy, s2.cz, s2.radius)
            if ts < ld - EPSILON:
                in_shadow = True
                break
        if in_shadow:
            continue

        # Lambertian diffuse
        diff = max(0.0, nx*lx + ny*ly + nz*lz) * mat.diffuse

        # Blinn-Phong specular via half-vector H = normalise(L + V)
        hbx = lx + vx;  hby = ly + vy;  hbz = lz + vz
        hbl = math.sqrt(hbx*hbx + hby*hby + hbz*hbz)
        if hbl > 1e-10:
            hbx /= hbl;  hby /= hbl;  hbz /= hbl
        spec = max(0.0, nx*hbx + ny*hby + nz*hbz) ** mat.shininess * mat.specular

        li  = L.intensity
        lcr = L.cr * li;  lcg = L.cg * li;  lcb = L.cb * li

        cr += mat.color.x * diff * lcr + spec * lcr
        cg += mat.color.y * diff * lcg + spec * lcg
        cb += mat.color.z * diff * lcb + spec * lcb

    # ── Recursive reflection ──────────────────────────────────────────────────
    if depth > 0 and mat.reflect > 0.0:
        # R = D - 2(D·N)N
        dn  = dx*nx + dy*ny + dz*nz
        rdx = dx - 2.0*dn*nx
        rdy = dy - 2.0*dn*ny
        rdz = dz - 2.0*dn*nz
        # Nudge origin along normal to avoid self-intersection
        rc = _trace(hx + nx*EPSILON, hy + ny*EPSILON, hz + nz*EPSILON,
                    rdx, rdy, rdz, spheres, lights, depth - 1)
        rf = mat.reflect
        cr = cr * (1.0 - rf) + rc.x * rf
        cg = cg * (1.0 - rf) + rc.y * rf
        cb = cb * (1.0 - rf) + rc.z * rf

    return Vec3(max(0.0, min(1.0, cr)),
                max(0.0, min(1.0, cg)),
                max(0.0, min(1.0, cb)))


# ──────────────────────────────────────────────────────────────────────────────
# Worker — called inside each child process by multiprocessing.Pool
#          (must be a module-level function so it can be pickled on Windows)
# ──────────────────────────────────────────────────────────────────────────────
def _render_chunk(args):
    """
    Render a horizontal band of rows.

    args = (row_start, row_end, width, height, max_depth, samples)

    Returns (row_start, bytes) where bytes is raw R G B triplets for every
    pixel in rows [row_start, row_end).
    """
    row_start, row_end, width, height, max_depth, samples = args

    _load_env_image()          # populate env map globals once per process
    spheres, lights = build_scene()

    fov_tan = math.tan(math.pi / 6.0)   # half-angle for a 60° vertical FOV
    aspect  = width / height
    inv_s   = 1.0 / samples
    inv_s2  = 1.0 / (samples * samples)

    buf = bytearray(3 * width * (row_end - row_start))
    idx = 0

    for y in range(row_start, row_end):
        for x in range(width):
            acc_r = acc_g = acc_b = 0.0

            for sy in range(samples):
                for sx in range(samples):
                    # Sub-pixel UV in [0, 1]
                    u  = (x + (sx + 0.5) * inv_s) / width
                    v  = (y + (sy + 0.5) * inv_s) / height

                    # Camera ray direction (right-handed, -Z forward)
                    rdx =  (2.0 * u - 1.0) * aspect * fov_tan
                    rdy = -(2.0 * v - 1.0) * fov_tan
                    rdz = -1.0
                    # Normalise
                    rl   = math.sqrt(rdx*rdx + rdy*rdy + 1.0)
                    rdx /= rl;  rdy /= rl;  rdz /= rl

                    c = _trace(0.0, 0.0, 0.0,
                               rdx, rdy, rdz,
                               spheres, lights, max_depth)
                    acc_r += c.x;  acc_g += c.y;  acc_b += c.z

            # Average samples then apply γ = 2.2 correction and quantise
            buf[idx]   = int(max(0.0, min(1.0, acc_r * inv_s2)) ** (1.0 / 2.2) * 255.999)
            buf[idx+1] = int(max(0.0, min(1.0, acc_g * inv_s2)) ** (1.0 / 2.2) * 255.999)
            buf[idx+2] = int(max(0.0, min(1.0, acc_b * inv_s2)) ** (1.0 / 2.2) * 255.999)
            idx += 3

    return row_start, bytes(buf)


# ──────────────────────────────────────────────────────────────────────────────
# Image output
# ──────────────────────────────────────────────────────────────────────────────
def _save_image(width, height, chunks, base="benchmark_render"):
    """
    Assemble row-chunks, write to disk, and return the saved file path.
    Prefers PNG (requires Pillow); falls back to PPM which needs no libraries.
    """
    chunks.sort(key=lambda c: c[0])
    pixel_data = b"".join(d for _, d in chunks)

    ppm_path = base + ".ppm"
    with open(ppm_path, "wb") as fh:
        fh.write(f"P6\n{width} {height}\n255\n".encode())
        fh.write(pixel_data)

    # Try to write PNG via Pillow and remove the intermediate PPM
    try:
        from PIL import Image
        img = Image.frombytes("RGB", (width, height), pixel_data)
        png_path = base + ".png"
        img.save(png_path, optimize=True)
        try:
            os.remove(ppm_path)
        except OSError:
            pass
        return png_path
    except ImportError:
        return ppm_path


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark runner
# ──────────────────────────────────────────────────────────────────────────────
def _print_progress(done, total, elapsed):
    pct  = done / total
    fill = int(pct * 40)
    bar  = "█" * fill + "░" * (40 - fill)
    if elapsed > 0 and pct > 0:
        eta = f"{elapsed / pct * (1.0 - pct):5.1f}s"
    else:
        eta = " ----"
    print(f"\r  [{bar}] {pct * 100:5.1f}%  {elapsed:6.1f}s elapsed  ETA {eta}",
          end="", flush=True)


def run_benchmark(mode: str = "multi"):
    """
    Run the render benchmark in  'single'  or  'multi'  mode.

    Returns (score, elapsed_seconds, worker_count, chunks).
    """
    cpu_count = multiprocessing.cpu_count()
    workers   = 1 if mode == "single" else cpu_count

    # Split the image into small row-bands for even load distribution
    chunk_h   = max(1, RENDER_HEIGHT // max(workers * 8, 16))
    task_list = []
    y = 0
    while y < RENDER_HEIGHT:
        y_end = min(y + chunk_h, RENDER_HEIGHT)
        task_list.append((y, y_end, RENDER_WIDTH, RENDER_HEIGHT, MAX_DEPTH, SAMPLES))
        y = y_end

    n_chunks = len(task_list)
    label    = "SINGLE-CORE" if mode == "single" else "MULTI-CORE"

    print(f"\n  ┌─ {label} {'─' * (53 - len(label))}┐")
    print(f"  │  {RENDER_WIDTH} × {RENDER_HEIGHT} px  │  depth {MAX_DEPTH}  │  "
          f"{SAMPLES}×{SAMPLES} spp  │  {workers} worker(s)  │  {n_chunks} chunks  │")
    print(f"  └{'─' * 63}┘\n")

    t0     = time.perf_counter()
    chunks = []
    done   = 0

    if workers == 1:
        # Single-core path: run directly in the main process
        for args in task_list:
            chunks.append(_render_chunk(args))
            done += 1
            _print_progress(done, n_chunks, time.perf_counter() - t0)
    else:
        # Multi-core path: distribute across all CPU cores
        with multiprocessing.Pool(workers) as pool:
            for result in pool.imap_unordered(_render_chunk, task_list):
                chunks.append(result)
                done += 1
                _print_progress(done, n_chunks, time.perf_counter() - t0)

    elapsed = time.perf_counter() - t0
    print()  # end the progress line

    # Score — single score × cpu_count for multi (Cinebench convention)
    ref_time    = REF_TIME_SINGLE if mode == "single" else REF_TIME_MULTI
    score_scale = 1 if mode == "single" else cpu_count
    score       = int(1000 * score_scale * ref_time / elapsed)

    return score, elapsed, workers, chunks


# ──────────────────────────────────────────────────────────────────────────────
# Display helpers
# ──────────────────────────────────────────────────────────────────────────────
_BANNER = (
    "\n"
    "  ╔═══════════════════════════════════════════════════════════╗\n"
    "  ║      PyRender Benchmark  ·  Cinebench-Style Scoring       ║\n"
    "  ║                         v{ver:<6}                           ║\n"
    "  ╚═══════════════════════════════════════════════════════════╝"
)


def _print_banner():
    print(_BANNER.format(ver=VERSION))
    print(f"  ├─ CPU : {multiprocessing.cpu_count()} logical core(s)")
    print(f"  ├─ OS  : {sys.platform}")
    print(f"  └─ Py  : {sys.version.split()[0]}\n")


def _print_result(label, score, elapsed, workers):
    bar = "█" * min(50, max(1, score // 80))
    sep = "─" * 58
    print(f"\n  {sep}")
    print(f"  {label}")
    print(f"  {sep}")
    print(f"  Score   : {score:>10,} pts")
    print(f"  Time    : {elapsed:>10.2f} s")
    print(f"  Workers : {workers:>10}")
    print(f"  {bar}")
    print(f"  {sep}")


def _print_summary(results, img_path):
    sep = "═" * 58
    print(f"\n  {sep}")
    print("  BENCHMARK SUMMARY")
    print(f"  {sep}")
    if "single" in results:
        sc, st = results["single"]
        print(f"  Single-Core : {sc:>10,} pts   ({st:.2f} s)")
    if "multi" in results:
        mc, mt = results["multi"]
        print(f"  Multi-Core  : {mc:>10,} pts   ({mt:.2f} s)")
        if "single" in results:
            ratio = mc / results["single"][0]
            print(f"  Scaling     :    {ratio:6.2f}×  "
                  f"({multiprocessing.cpu_count()} cores)")
    print(f"\n  Render saved  : {img_path}")
    print(f"  {sep}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
def main():
    _print_banner()

    results     = {}
    last_chunks = []

    # ── Single-core test ──────────────────────────────────────────────────────
    input("  Press ENTER to start Single-Core test ...")
    score, elapsed, workers, chunks = run_benchmark("single")
    _print_result("SINGLE-CORE RESULT", score, elapsed, workers)
    results["single"] = (score, elapsed)
    last_chunks = chunks

    # ── Multi-core test (skip if only one logical core is available) ──────────
    if multiprocessing.cpu_count() > 1:
        input("\n  Press ENTER to start Multi-Core test ...")
        score, elapsed, workers, chunks = run_benchmark("multi")
        _print_result("MULTI-CORE RESULT", score, elapsed, workers)
        results["multi"] = (score, elapsed)
        last_chunks = chunks

    # ── Save the last rendered frame ──────────────────────────────────────────
    print("\n  Saving rendered image ...", end="", flush=True)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = _save_image(RENDER_WIDTH, RENDER_HEIGHT, last_chunks,
                           base=os.path.join(script_dir, "benchmark_render"))
    print(f" {img_path}")

    # ── Final summary ─────────────────────────────────────────────────────────
    _print_summary(results, img_path)


if __name__ == "__main__":
    # Required on Windows so spawned worker processes do not re-run main()
    multiprocessing.freeze_support()
    main()

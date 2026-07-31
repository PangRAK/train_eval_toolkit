"""Visualization helpers for the streaming VQA evaluation.

Only two things are used by ``01_StreamingEval_InternVL3.py``:

* :func:`draw_alarm`  — overlay a red alert border (+ event label / description) on a
  frame when the model flags an event. Adapted for the binary VQA output
  (``category`` = event/normal, plus a short ``description``).
* :class:`VideoFromPIL` — stream annotated PIL frames to an H.264 ``.mp4`` via ffmpeg.

This file was copied from another project and carried a lot of unrelated code
(text-feature loading, sliding-window helpers, score graphs, …). It has been
trimmed to just what this evaluation needs.
"""
from __future__ import annotations

import os
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Font
# ---------------------------------------------------------------------------
# The original code hard-coded ``./utils/arial.ttf`` (which is not present here).
# Resolve a real TrueType font from a few common locations, cache it by size, and
# fall back to PIL's built-in bitmap font so visualization never crashes.
_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "DejaVuSans.ttf",
)
_FONT_CACHE = {}


def _load_font(size: int):
    size = max(8, int(size))
    if size not in _FONT_CACHE:
        font = None
        for path in _FONT_CANDIDATES:
            try:
                font = ImageFont.truetype(path, size=size)
                break
            except OSError:
                continue
        _FONT_CACHE[size] = font or ImageFont.load_default()
    return _FONT_CACHE[size]


# ---------------------------------------------------------------------------
# Alarm overlay
# ---------------------------------------------------------------------------
def draw_alarm(image, alarm: bool, label: str | None = None, description: str | None = None) -> Image.Image:
    """Return ``image`` with a red alert border + label drawn when ``alarm`` is True.

    Adapted from the original multi-purpose ``draw()`` (which also rendered prompt
    top-k scores, a sliding-queue panel, etc.) down to the binary VQA case:

    * ``alarm``       — event-vs-normal decision for this clip (True => draw alert).
    * ``label``       — event name to show, e.g. ``"climbing"``.
    * ``description`` — optional short scene text from the VQA output.

    ``image`` may be a PIL image or an ``HxWx3`` RGB ndarray. A new image is always
    returned; the input is never mutated. When ``alarm`` is False the frame is
    returned unchanged (no overlay).
    """
    img = Image.fromarray(image) if isinstance(image, np.ndarray) else image.copy()
    if not alarm:
        return img

    drawer = ImageDraw.Draw(img)
    w, h = img.size

    # Red frame border (thickness scales with frame height).
    border = max(2, int(h / 120))
    for i in range(border):
        drawer.rectangle([i, i, w - 1 - i, h - 1 - i], outline=(255, 0, 0))

    # "[ALARM] <EVENT>" banner, top-left, just inside the border.
    text = f"[ALARM] {label.upper()}" if label else "[ALARM]"
    x, y = border + 6, border + 4
    drawer.text((x, y), text, font=_load_font(max(14, int(h / 22))),
                fill=(255, 0, 0), stroke_width=2, stroke_fill=(0, 0, 0))

    # Optional one-line description beneath the banner.
    if description:
        drawer.text((x, y + int(h / 18)), description[:90], font=_load_font(max(11, int(h / 40))),
                    fill=(255, 255, 255), stroke_width=1, stroke_fill=(0, 0, 0))

    return img


# ---------------------------------------------------------------------------
# Video writer
# ---------------------------------------------------------------------------
class VideoFromPIL:
    """Stream PIL frames to an H.264/AVC ``.mp4`` via ffmpeg.

    Frame size is fixed by the first frame (later frames are resized to match).
    The ffmpeg process starts lazily on the first :meth:`add_frame`, so a writer
    that never receives a frame writes nothing. Usage::

        writer = VideoFromPIL("out.mp4", fps=10)
        for frame in frames:        # PIL.Image
            writer.add_frame(frame)
        writer.save()
    """

    def __init__(self, output_path: str, fps: float = 2.0):
        if fps <= 0:
            raise ValueError("fps must be > 0")
        self.output_path = output_path
        self.fps = float(fps)
        self.size = None
        self._proc = None

    def _start(self):
        width, height = self.size
        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}", "-r", f"{self.fps:.6f}", "-i", "-",
            "-an", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v", "libx264", "-profile:v", "high", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", self.output_path,
        ]
        try:
            self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        except FileNotFoundError as exc:
            raise RuntimeError("ffmpeg executable was not found.") from exc

    def _err(self) -> str:
        if self._proc is None or self._proc.stderr is None:
            return "unknown ffmpeg error"
        return self._proc.stderr.read().decode("utf-8", errors="replace").strip() or "unknown ffmpeg error"

    def add_frame(self, pil_image: Image.Image):
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        if self.size is None:
            self.size = pil_image.size
            self._start()
        elif pil_image.size != self.size:
            pil_image = pil_image.resize(self.size)
        try:
            self._proc.stdin.write(np.asarray(pil_image, dtype=np.uint8).tobytes())
        except BrokenPipeError as exc:
            raise RuntimeError(self._err()) from exc

    def save(self):
        """Flush and finalize the file. Raises if no frame was ever added."""
        if self._proc is None:
            raise RuntimeError("No frames were added.")
        if self._proc.stdin is not None:
            self._proc.stdin.close()
        return_code = self._proc.wait()
        if return_code != 0:
            raise RuntimeError(self._err())
        self._proc = None

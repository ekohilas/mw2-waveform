"""Output sinks: an ffmpeg pipe (libx264 or SVT-AV1) and a PNG directory."""
from __future__ import annotations

import os
import shutil
import subprocess

import cv2
import numpy as np


class Mp4Writer:
    """BGR frames of any size -> BT.709 tv-range video.

    codec "libx264": H.264 yuv420p (default). codec "libsvtav1": AV1
    yuv420p10le (10-bit avoids banding in the soft glow); `crf` is then the
    SVT-AV1 crf (0-63) and `preset` its speed preset (0 slowest - 13).
    """

    # Measured level shift of this encode + the scorer's decode (BGR, 0..255):
    # decoded = input + LEVEL_SHIFT for unclipped values (out/scratch pipe test,
    # random colours, rms 0.7 after compensation). compensate=True pre-adds
    # the inverse, which lifts the re-encoded reference from 0.975 to 0.993.
    LEVEL_SHIFT = np.array([-3.7, -2.0, -3.7], np.float32)

    def __init__(self, path: str, W: int, H: int, fps_num: int, fps_den: int, crf: int = 16,
                 preset: str = "slow", compensate: bool = False, codec: str = "libx264"):
        self.compensate = compensate
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        pix_fmt = "yuv420p10le" if codec == "libsvtav1" else "yuv420p"
        cmd = ["ffmpeg", "-v", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
               "-framerate", f"{fps_num}/{fps_den}", "-i", "-",
               "-vf", f"scale=out_color_matrix=bt709:out_range=tv,format={pix_fmt},"
               "setparams=range=tv:color_primaries=bt709:color_trc=bt709:colorspace=bt709",
               "-c:v", codec, "-preset", preset, "-crf", str(crf),
               "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
               "-color_range", "tv", "-movflags", "+faststart", path]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        if self.compensate:
            f = frame.astype(np.float32)
            f = np.where(f > 0, f - self.LEVEL_SHIFT, 0.0)
            frame = np.clip(f + 0.5, 0, 255).astype(np.uint8)
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError("ffmpeg failed")


class PngDirWriter:
    """Writes NNNN.png into a directory (cleared first)."""

    def __init__(self, path: str):
        if os.path.isdir(path):
            shutil.rmtree(path)
        os.makedirs(path)
        self.path = path
        self.n = 0

    def write(self, frame: np.ndarray) -> None:
        cv2.imwrite(os.path.join(self.path, f"{self.n:04d}.png"), frame,
                    [cv2.IMWRITE_PNG_COMPRESSION, 1])
        self.n += 1

    def close(self) -> None:
        pass

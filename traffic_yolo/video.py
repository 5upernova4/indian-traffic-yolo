"""Browser-friendly MP4 writing (H.264 when available, ffmpeg re-encode fallback).

Adapted from the v1 Flask app.
"""

from __future__ import annotations

import os
import shutil
import subprocess

import cv2


class VideoSink:
    """Writes frames to ``path`` as H.264 MP4 if OpenCV supports it, else mp4v + ffmpeg re-encode."""

    def __init__(self, path: str, fps: float, size: tuple[int, int]) -> None:
        self.path = path
        self.size = size
        self._raw = None
        self.writer = None
        for codec in ("avc1", "H264", "X264"):
            try:
                w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec), fps, size)
                if w.isOpened():
                    self.writer = w
                    break
                w.release()
            except Exception:
                pass
        if self.writer is None:
            root, _ = os.path.splitext(path)
            self._raw = f"{root}_raw.mp4"
            self.writer = cv2.VideoWriter(self._raw, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    def write(self, frame) -> None:
        self.writer.write(frame)

    def close(self) -> str:
        self.writer.release()
        if self._raw is None:
            return self.path
        if reencode_h264(self._raw, self.path):
            os.remove(self._raw)
        else:  # no ffmpeg: keep the mp4v file (downloads fine, may not play inline)
            os.replace(self._raw, self.path)
        return self.path


def reencode_h264(src: str, dst: str) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", src, "-vcodec", "libx264", "-crf", "26", "-preset", "fast",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", dst]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False

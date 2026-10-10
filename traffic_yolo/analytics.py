"""Traffic analytics accumulated while processing a video (or a single image)."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .classes import CLASS_NAMES


class DensityHeatmap:
    """Accumulates vehicle anchor points into a low-resolution grid and renders an overlay."""

    def __init__(self, width: int, height: int, cell: int = 8, sigma_px: float = 18.0) -> None:
        self.width, self.height, self.cell = width, height, cell
        self.grid = np.zeros((max(1, height // cell), max(1, width // cell)), np.float32)
        self.sigma = max(sigma_px / cell, 0.5)

    def add(self, points) -> None:
        gh, gw = self.grid.shape
        for x, y in points:
            gx, gy = int(x / self.cell), int(y / self.cell)
            if 0 <= gx < gw and 0 <= gy < gh:
                self.grid[gy, gx] += 1.0

    def normalised(self) -> np.ndarray:
        k = int(self.sigma * 6) | 1
        g = cv2.GaussianBlur(self.grid, (k, k), self.sigma)
        m = float(g.max())
        return g / m if m > 0 else g

    def render(self, background: np.ndarray | None = None, alpha: float = 0.55) -> np.ndarray:
        heat = cv2.resize(self.normalised(), (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        colour = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        if background is None:
            return colour
        bg = cv2.resize(background, (self.width, self.height))
        mask = (heat[..., None] ** 0.7) * alpha
        return (bg * (1 - mask) + colour * mask).astype(np.uint8)


@dataclass
class TrafficAnalytics:
    """Collects per-frame detections, class counts over time, unique tracks and line crossings."""

    fps: float = 25.0
    bin_seconds: float = 1.0
    names: list[str] = field(default_factory=lambda: list(CLASS_NAMES))
    rows: list[dict] = field(default_factory=list)
    _per_bin: dict[int, Counter] = field(default_factory=lambda: defaultdict(Counter))
    _bin_frames: Counter = field(default_factory=Counter)
    _track_class: dict[int, int] = field(default_factory=dict)
    crossings: list[dict] = field(default_factory=list)
    frames_processed: int = 0

    def add_frame(self, frame_idx: int, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray,
                  track_ids: np.ndarray | None = None, voted_cls: dict[int, int] | None = None) -> None:
        t = frame_idx / self.fps if self.fps else 0.0
        b = int(t // self.bin_seconds)
        self._bin_frames[b] += 1
        self.frames_processed += 1
        for i in range(len(xyxy)):
            c = int(cls[i])
            tid = int(track_ids[i]) if track_ids is not None and track_ids[i] is not None and track_ids[i] >= 0 else None
            self._per_bin[b][c] += 1
            if tid is not None:
                self._track_class[tid] = voted_cls.get(tid, c) if voted_cls else c
            x1, y1, x2, y2 = (float(v) for v in xyxy[i][:4])
            self.rows.append({
                "frame": frame_idx, "time_s": round(t, 3), "track_id": "" if tid is None else tid,
                "class_id": c, "class_name": self.names[c] if 0 <= c < len(self.names) else str(c),
                "confidence": round(float(conf[i]), 4),
                "x1": round(x1, 1), "y1": round(y1, 1), "x2": round(x2, 1), "y2": round(y2, 1),
            })

    def add_crossings(self, crossings, fps: float | None = None) -> None:
        fps = fps or self.fps
        for c in crossings:
            self.crossings.append({
                "line": c.line, "track_id": c.track_id, "class_id": c.cls,
                "class_name": self.names[c.cls] if 0 <= c.cls < len(self.names) else str(c.cls),
                "direction": c.direction, "frame": c.frame, "time_s": round(c.frame / fps, 3) if fps else 0.0,
            })

    # ------------------------------------------------------------------ summaries
    def timeline(self) -> list[dict]:
        """Average number of visible vehicles per class in each time bin."""
        out = []
        for b in sorted(self._bin_frames):
            n = self._bin_frames[b]
            out.append({"t_start_s": b * self.bin_seconds,
                        **{name: round(self._per_bin[b].get(i, 0) / n, 3) for i, name in enumerate(self.names)}})
        return out

    def unique_tracks(self) -> dict[str, int]:
        cnt = Counter(self._track_class.values())
        return {name: cnt.get(i, 0) for i, name in enumerate(self.names)}

    def detections_per_class(self) -> dict[str, int]:
        cnt = Counter(r["class_id"] for r in self.rows)
        return {name: cnt.get(i, 0) for i, name in enumerate(self.names)}

    def line_counts(self) -> dict:
        out: dict = {}
        for c in self.crossings:
            line = out.setdefault(c["line"], {"total": 0, "in": {n: 0 for n in self.names}, "out": {n: 0 for n in self.names}})
            line["total"] += 1
            line[c["direction"]][c["class_name"]] = line[c["direction"]].get(c["class_name"], 0) + 1
        return out

    def summary(self, **extra) -> dict:
        return {
            "frames_processed": self.frames_processed,
            "fps": self.fps,
            "duration_s": round(self.frames_processed / self.fps, 2) if self.fps else None,
            "detections_per_class": self.detections_per_class(),
            "unique_vehicles_per_class": self.unique_tracks(),
            "line_counts": self.line_counts(),
            "timeline": self.timeline(),
            **extra,
        }

    # ------------------------------------------------------------------ export
    def write_csv(self, path: Path | str) -> None:
        cols = ["frame", "time_s", "track_id", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2"]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(self.rows)

    def write_crossings_csv(self, path: Path | str) -> None:
        cols = ["line", "track_id", "class_id", "class_name", "direction", "frame", "time_s"]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(self.crossings)

    def write_json(self, path: Path | str, **extra) -> dict:
        s = self.summary(**extra)
        Path(path).write_text(json.dumps(s, indent=2))
        return s

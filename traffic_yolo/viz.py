"""Drawing helpers (OpenCV) for detections, tracks, counting lines and the counts HUD."""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from .classes import CLASS_NAMES, DISPLAY_NAMES, color_for

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _scale(img: np.ndarray) -> tuple[int, float]:
    h, w = img.shape[:2]
    m = min(h, w)
    return max(1, round(m / 400)), max(0.4, m / 1400)


def draw_label(img, text, x, y, color, font_scale, thickness):
    (tw, th), base = cv2.getTextSize(text, FONT, font_scale, thickness)
    y0 = max(y - th - base - 4, 0)
    cv2.rectangle(img, (x, y0), (x + tw + 6, y0 + th + base + 4), color, -1)
    # dark text on light fills, white text on dark fills
    lum = 0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2]
    cv2.putText(img, text, (x + 3, y0 + th + 2), FONT, font_scale, (20, 20, 20) if lum > 150 else (255, 255, 255),
                thickness, cv2.LINE_AA)


def draw_detections(img: np.ndarray, xyxy, conf, cls, track_ids=None, names=CLASS_NAMES,
                    show_conf: bool = True) -> np.ndarray:
    """Draw boxes in place (and return the image)."""
    t, fs = _scale(img)
    for i in range(len(xyxy)):
        x1, y1, x2, y2 = (int(round(v)) for v in xyxy[i][:4])
        c = int(cls[i])
        color = color_for(c)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, t, cv2.LINE_AA)
        label = names[c] if 0 <= c < len(names) else str(c)
        if track_ids is not None and track_ids[i] is not None and track_ids[i] >= 0:
            label = f"#{int(track_ids[i])} {label}"
        if show_conf:
            label += f" {float(conf[i]):.2f}"
        draw_label(img, label, x1, y1, color, fs * 0.8, max(1, t - 1))
    return img


class TrailBuffer:
    """Keeps the last N anchor points of each track for drawing motion trails."""

    def __init__(self, maxlen: int = 24) -> None:
        self.maxlen = maxlen
        self.trails: dict[int, deque] = {}

    def update(self, track_ids, anchors, classes) -> None:
        for tid, pt, c in zip(track_ids, anchors, classes):
            d = self.trails.setdefault(int(tid), deque(maxlen=self.maxlen))
            d.append((int(pt[0]), int(pt[1]), int(c)))

    def draw(self, img, active_ids=None) -> None:
        t, _ = _scale(img)
        for tid, pts in self.trails.items():
            if active_ids is not None and tid not in active_ids:
                continue
            for j in range(1, len(pts)):
                cv2.line(img, pts[j - 1][:2], pts[j][:2], color_for(pts[j][2]), max(1, t - 1), cv2.LINE_AA)


def draw_line_counter(img, p1, p2, name: str, n_in: int, n_out: int, flash: bool = False) -> None:
    t, fs = _scale(img)
    p1i, p2i = (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1]))
    color = (80, 255, 255) if flash else (255, 255, 255)
    cv2.line(img, p1i, p2i, (0, 0, 0), t + 4, cv2.LINE_AA)
    cv2.line(img, p1i, p2i, color, t + 1, cv2.LINE_AA)
    for p in (p1i, p2i):
        cv2.circle(img, p, t + 4, color, -1, cv2.LINE_AA)
    label = f"{name}: in {n_in} | out {n_out}"
    draw_label(img, label, min(p1i[0], p2i[0]) + 8, min(p1i[1], p2i[1]) - 6, (30, 30, 30), fs * 0.9, max(1, t - 1))


def draw_hud(img, counts: dict[str, int], title: str = "Vehicles counted") -> None:
    """Semi-transparent panel listing per-class totals (top-left corner)."""
    t, fs = _scale(img)
    fs *= 0.85
    rows = [(DISPLAY_NAMES.get(k, k), v, color_for(k)) for k, v in counts.items()]
    (_, th), _ = cv2.getTextSize("Ag", FONT, fs, 1)
    line_h = int(th * 1.9)
    w = int(max(cv2.getTextSize(f"{n}: 0000", FONT, fs, 1)[0][0] for n, _, _ in rows + [(title, 0, 0)]) + th * 3)
    h = line_h * (len(rows) + 1) + th
    x0, y0 = 10, 10
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + w, y0 + h), (15, 15, 20), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    cv2.putText(img, title, (x0 + th, y0 + line_h), FONT, fs, (255, 255, 255), max(1, t - 1), cv2.LINE_AA)
    for i, (name, v, col) in enumerate(rows, start=2):
        y = y0 + line_h * i
        cv2.circle(img, (x0 + th, y - th // 2), max(3, th // 2), col, -1, cv2.LINE_AA)
        cv2.putText(img, f"{name}: {v}", (x0 + th * 2, y), FONT, fs, (235, 235, 235), max(1, t - 1), cv2.LINE_AA)

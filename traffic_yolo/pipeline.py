"""Detection / tracking / counting pipeline built on Ultralytics YOLO + ByteTrack."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .analytics import DensityHeatmap, TrafficAnalytics
from .classes import CLASS_NAMES, V1_TO_V2, build_id_map
from .counting import LineCounter, TrackClassVoter, box_anchor, parse_line
from .video import VideoSink
from .viz import TrailBuffer, draw_detections, draw_hud, draw_line_counter

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = ROOT / "weights" / "yolo11s_uvh26.pt"
# Inference size used by default for the shipped v2 weights (picked on the held-out test split,
# see docs/metrics/comparison.json). Other weights default to their own training size.
SHIPPED_IMGSZ = {"yolo11s_uvh26": 640}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def default_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


@dataclass
class Detections:
    xyxy: np.ndarray  # (N, 4) pixel coords in the *original* image
    conf: np.ndarray  # (N,)
    cls: np.ndarray  # (N,) v2 class ids
    ids: np.ndarray | None = None  # (N,) track ids (tracking only)

    def __len__(self) -> int:
        return len(self.xyxy)

    def select(self, keep: np.ndarray) -> "Detections":
        return Detections(self.xyxy[keep], self.conf[keep], self.cls[keep], None if self.ids is None else self.ids[keep])


class Detector:
    """Thin wrapper around an Ultralytics model.

    Ultralytics letterboxes the input to ``imgsz`` (keeping aspect ratio) and maps the
    boxes back to original-image pixels, so no manual resizing is needed here.

    Models with a different label space (the 5-class v1 model) are remapped into the
    six v2 classes so the rest of the pipeline is model-agnostic.
    """

    def __init__(self, weights: str | Path | None = None, device: str | None = None, imgsz: int | None = None,
                 conf: float = 0.25, iou: float = 0.5, tracker: str = "bytetrack.yaml") -> None:
        from ultralytics import YOLO

        self.weights = Path(weights or DEFAULT_WEIGHTS)
        if not self.weights.exists():
            raise FileNotFoundError(f"weights not found: {self.weights}")
        self.is_onnx = self.weights.suffix == ".onnx"
        self.model = YOLO(str(self.weights), task="detect")
        self.device = device or ("cpu" if self.is_onnx else default_device())
        self.conf, self.iou, self.tracker = conf, iou, tracker
        self.imgsz = imgsz or self._native_imgsz()
        names = self.model.names
        self.source_names = [names[i] for i in sorted(names)]
        if self.source_names == CLASS_NAMES:
            self.id_map = None
        else:
            self.id_map = build_id_map(names, V1_TO_V2)  # raises KeyError for unknown label spaces
        self.names = list(CLASS_NAMES)

    def warmup(self) -> None:
        """Run dummy inferences so the first real request is not slowed by lazy initialisation."""
        for shape in ((720, 1280, 3), (self.imgsz, self.imgsz, 3)):  # 16:9 frames and square images
            self.predict(np.zeros(shape, np.uint8))

    def _native_imgsz(self) -> int:
        if self.weights.stem in SHIPPED_IMGSZ:
            return SHIPPED_IMGSZ[self.weights.stem]
        if self.is_onnx:
            import onnxruntime as ort

            sess = ort.InferenceSession(str(self.weights), providers=["CPUExecutionProvider"])
            shape = sess.get_inputs()[0].shape
            if isinstance(shape[-1], int):  # fixed-size export
                return int(shape[-1])
            meta = sess.get_modelmeta().custom_metadata_map.get("imgsz", "[640, 640]")  # dynamic export
            return int(max(int(v) for v in meta.strip("[]").split(",")))
        sz = self.model.overrides.get("imgsz", 640)
        return int(sz[0] if isinstance(sz, (list, tuple)) else sz)

    def _to_dets(self, result) -> Detections:
        b = result.boxes
        xyxy = b.xyxy.cpu().numpy().astype(np.float32)
        conf = b.conf.cpu().numpy().astype(np.float32)
        cls = b.cls.cpu().numpy().astype(int)
        ids = b.id.cpu().numpy().astype(int) if getattr(b, "id", None) is not None else None
        det = Detections(xyxy, conf, cls, ids)
        if self.id_map is not None:
            mapped = np.array([-1 if self.id_map.get(int(c)) is None else self.id_map[int(c)] for c in cls], dtype=int)
            det.cls = mapped
            det = det.select(mapped >= 0)
        return det

    def predict(self, image: np.ndarray | str) -> Detections:
        r = self.model.predict(image, imgsz=self.imgsz, conf=self.conf, iou=self.iou, device=self.device, verbose=False)[0]
        return self._to_dets(r)

    def track(self, frame: np.ndarray) -> Detections:
        # ByteTrack also associates low-score boxes, so feed it everything above 0.1.
        r = self.model.track(frame, persist=True, tracker=self.tracker, imgsz=self.imgsz, conf=min(self.conf, 0.1),
                             iou=self.iou, device=self.device, verbose=False)[0]
        det = self._to_dets(r)
        if det.ids is None:
            det = Detections(det.xyxy[:0], det.conf[:0], det.cls[:0], np.zeros(0, int))
        return det

    def reset_tracker(self) -> None:
        p = getattr(self.model, "predictor", None)
        for t in getattr(p, "trackers", []) or []:
            t.reset()


# --------------------------------------------------------------------------- images
def process_image(src: str | Path | np.ndarray, out_dir: str | Path, detector: Detector) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(src)) if not isinstance(src, np.ndarray) else src
    if img is None:
        raise ValueError(f"could not read image {src}")
    t0 = time.perf_counter()
    det = detector.predict(img)
    ms = (time.perf_counter() - t0) * 1000
    an = TrafficAnalytics(fps=0.0)
    an.add_frame(0, det.xyxy, det.conf, det.cls)
    vis = draw_detections(img.copy(), det.xyxy, det.conf, det.cls)
    draw_hud(vis, an.detections_per_class(), title=f"Vehicles: {len(det)}")
    cv2.imwrite(str(out_dir / "annotated.jpg"), vis)
    an.write_csv(out_dir / "detections.csv")
    summary = {
        "type": "image",
        "source": str(src) if not isinstance(src, np.ndarray) else "array",
        "model": detector.weights.name,
        "imgsz": detector.imgsz,
        "image_size": [img.shape[1], img.shape[0]],
        "inference_ms": round(ms, 1),
        "total_vehicles": len(det),
        "detections_per_class": an.detections_per_class(),
    }
    (out_dir / "summary.json").write_text(__import__("json").dumps(summary, indent=2))
    return summary


# --------------------------------------------------------------------------- videos
def default_lines(width: int, height: int, position: float = 0.6, orientation: str = "horizontal"):
    if orientation == "vertical":
        x = width * position
        return [("line1", (x, 0.0), (x, float(height)))]
    y = height * position
    return [("line1", (0.0, y), (float(width), y))]


def process_video(src: str | Path, out_dir: str | Path, detector: Detector, lines: list | None = None,
                  stride: int = 1, max_frames: int | None = None, write_video: bool = True,
                  progress: Callable[[int, int], None] | None = None, bin_seconds: float = 1.0) -> dict:
    """Track vehicles through a video, count line crossings and export analytics.

    ``lines`` is a list of ``(name, p1, p2)`` tuples in pixels or ``"x1,y1,x2,y2"`` strings
    (fractions of the frame if all values <= 1). Defaults to one horizontal line at 60 % height.

    Writes ``annotated.mp4``, ``detections.csv``, ``crossings.csv``, ``summary.json`` and
    ``heatmap.jpg`` into ``out_dir`` and returns the summary dict.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise ValueError(f"could not open video {src}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps == fps and fps > 0 else 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if max_frames:
        total = min(total, max_frames) if total else max_frames

    parsed = []
    for i, ln in enumerate(lines or default_lines(w, h)):
        if isinstance(ln, str):
            p1, p2 = parse_line(ln, w, h)
            parsed.append((f"line{i + 1}", p1, p2))
        else:
            parsed.append(ln)
    counters = [LineCounter(p1, p2, name) for name, p1, p2 in parsed]
    voter, trails = TrackClassVoter(), TrailBuffer()
    heat = DensityHeatmap(w, h)
    an = TrafficAnalytics(fps=fps, bin_seconds=bin_seconds)
    sink = VideoSink(str(out_dir / "annotated.mp4"), fps, (w, h)) if write_video else None
    detector.reset_tracker()

    first_frame = None
    det = Detections(np.zeros((0, 4)), np.zeros(0), np.zeros(0, int), np.zeros(0, int))
    voted = np.zeros(0, int)
    flash: dict[str, int] = {}
    infer_ms = []
    idx = 0
    t_start = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames and idx >= max_frames):
            break
        if first_frame is None:
            first_frame = frame.copy()
        if idx % stride == 0:
            t0 = time.perf_counter()
            det = detector.track(frame)
            infer_ms.append((time.perf_counter() - t0) * 1000)
            voted = np.array([voter.update(t, c, s) for t, c, s in zip(det.ids, det.cls, det.conf)], dtype=int)
            anchors = [box_anchor(b) for b in det.xyxy]
            for lc in counters:
                new = lc.update(det.ids, anchors, voted, frame=idx)
                if new:
                    flash[lc.name] = idx
                an.add_crossings(new, fps)
            heat.add(anchors)
            trails.update(det.ids, anchors, voted)
            an.add_frame(idx, det.xyxy, det.conf, voted, det.ids)
        if sink is not None:
            vis = frame.copy()
            trails.draw(vis, active_ids=set(int(i) for i in det.ids))
            draw_detections(vis, det.xyxy, det.conf, voted, det.ids, show_conf=False)
            for lc in counters:
                c = lc.counts()
                draw_line_counter(vis, lc.p1, lc.p2, lc.name, sum(c["in"].values()), sum(c["out"].values()),
                                  flash=idx - flash.get(lc.name, -999) < max(3, int(fps // 4)))
            totals = {n: 0 for n in CLASS_NAMES}
            for cr in an.crossings:
                totals[cr["class_name"]] += 1
            draw_hud(vis, totals, title=f"Counted: {sum(totals.values())}")
            sink.write(vis)
        idx += 1
        if progress:
            progress(idx, max(total, idx))
    cap.release()
    video_path = sink.close() if sink is not None else None
    wall = time.perf_counter() - t_start

    an.write_csv(out_dir / "detections.csv")
    an.write_crossings_csv(out_dir / "crossings.csv")
    if first_frame is not None:
        cv2.imwrite(str(out_dir / "heatmap.jpg"), heat.render(first_frame))
    summary = an.write_json(
        out_dir / "summary.json",
        type="video",
        source=str(src),
        model=detector.weights.name,
        imgsz=detector.imgsz,
        frame_size=[w, h],
        stride=stride,
        lines=[{"name": n, "p1": list(map(float, p1)), "p2": list(map(float, p2))} for n, p1, p2 in parsed],
        total_counted=sum(lc.total for lc in counters),
        mean_inference_ms=round(float(np.mean(infer_ms)), 1) if infer_ms else None,
        processing_fps=round(idx / wall, 2) if wall > 0 else None,
        annotated_video=Path(video_path).name if video_path else None,
    )
    return summary


def is_video(path: str | Path) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTS

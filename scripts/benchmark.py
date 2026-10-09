"""Inference speed on real 1280x720 frames.

    python scripts/benchmark.py --images ../_data/uvh26/yolo/images/test --n 200

Reports Ultralytics' per-stage timings (pre-process / inference / post-process, ms per
frame, batch 1) after warm-up, plus end-to-end wall time per frame.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def bench(weights: str, device: str, images: list[Path], imgsz: int | None, warmup: int = 10) -> dict:
    from traffic_yolo.pipeline import Detector

    det = Detector(weights, device=device, imgsz=imgsz)
    frames = [cv2.imread(str(p)) for p in images]
    for f in frames[:warmup]:
        det.predict(f)
    pre, inf, post, wall = [], [], [], []
    for f in frames:
        t0 = time.perf_counter()
        r = det.model.predict(f, imgsz=det.imgsz, conf=det.conf, iou=det.iou, device=det.device, verbose=False)[0]
        wall.append((time.perf_counter() - t0) * 1000)
        pre.append(r.speed["preprocess"])
        inf.append(r.speed["inference"])
        post.append(r.speed["postprocess"])
    return {
        "weights": Path(weights).name, "device": device, "imgsz": det.imgsz, "frames": len(frames),
        "frame_size": [frames[0].shape[1], frames[0].shape[0]],
        "preprocess_ms": round(float(np.mean(pre)), 2), "inference_ms": round(float(np.mean(inf)), 2),
        "postprocess_ms": round(float(np.mean(post)), 2), "end_to_end_ms": round(float(np.mean(wall)), 2),
        "fps_end_to_end": round(1000 / float(np.mean(wall)), 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default="docs/metrics/speed.json")
    args = ap.parse_args()
    imgs = sorted(Path(args.images).glob("*.jpg"))[: args.n]
    runs = [
        ("weights/yolo11s_uvh26.pt", "mps", None),
        ("weights/yolo11s_uvh26.onnx", "cpu", None),
        ("weights/yolo11s_uvh26.onnx", "cpu", 640),
        ("weights/v1_legacy.onnx", "cpu", None),
    ]
    results = []
    for w, d, sz in runs:
        if not Path(w).exists():
            continue
        r = bench(w, d, imgs, sz)
        print(json.dumps(r))
        results.append(r)
    import onnxruntime
    import torch
    import ultralytics

    out = {"machine": f"{platform.machine()} {platform.platform()}", "chip": "Apple M5 (10-core CPU), 16 GB",
           "torch": torch.__version__, "onnxruntime": onnxruntime.__version__, "ultralytics": ultralytics.__version__,
           "results": results}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

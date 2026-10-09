"""Command-line interface.

    python -m traffic_yolo predict      <image|folder|video> [--out runs/predict]
    python -m traffic_yolo track-count  <video> [--line 0,0.6,1,0.6] [--stride 1]
    python -m traffic_yolo evaluate     <images_dir> [--labels labels_dir] [--weights weights/v1_legacy.onnx]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _detector(args):
    from .pipeline import Detector

    return Detector(args.weights, device=args.device, imgsz=args.imgsz, conf=args.conf, iou=args.iou)


def cmd_predict(args) -> None:
    from .pipeline import IMAGE_EXTS, is_video, process_image, process_video

    det = _detector(args)
    src = Path(args.source)
    out = Path(args.out)
    if src.is_dir():
        files = sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        totals: dict[str, int] = {}
        for f in files:
            s = process_image(f, out / f.stem, det)
            for k, v in s["detections_per_class"].items():
                totals[k] = totals.get(k, 0) + v
            print(f"{f.name}: {s['total_vehicles']} vehicles  ({s['inference_ms']} ms)")
        print(json.dumps({"images": len(files), "detections_per_class": totals}, indent=2))
    elif is_video(src):
        s = process_video(src, out, det, stride=args.stride, max_frames=args.max_frames)
        print(json.dumps({k: s[k] for k in ("frames_processed", "unique_vehicles_per_class", "processing_fps")}, indent=2))
    else:
        s = process_image(src, out, det)
        print(json.dumps(s, indent=2))
    print(f"results written to {out}/")


def cmd_track_count(args) -> None:
    from .pipeline import process_video

    det = _detector(args)

    def progress(i: int, n: int) -> None:
        if i % 50 == 0 or i == n:
            print(f"\r  frame {i}/{n}", end="", file=sys.stderr, flush=True)

    s = process_video(args.source, args.out, det, lines=args.line or None, stride=args.stride,
                      max_frames=args.max_frames, write_video=not args.no_video, progress=progress)
    print(file=sys.stderr)
    print(json.dumps({k: s[k] for k in ("frames_processed", "total_counted", "line_counts",
                                        "unique_vehicles_per_class", "processing_fps")}, indent=2))
    print(f"results written to {args.out}/")


def cmd_evaluate(args) -> None:
    from .evaluate import evaluate, save

    det = _detector(args)
    r = evaluate(det, args.images, args.labels, limit=args.limit)
    print(f"{'class':12s} {'inst':>6s} {'P':>6s} {'R':>6s} {'AP50':>6s} {'AP50-95':>8s}")
    for name, v in r["per_class"].items():
        print(f"{name:12s} {v['instances']:6d} {v['precision']:6.3f} {v['recall']:6.3f} {v['AP50']:6.3f} {v['AP50_95']:8.3f}")
    print(f"{'all':12s} {r['instances']:6d} {r['precision']:6.3f} {r['recall']:6.3f} {r['mAP50']:6.3f} {r['mAP50_95']:8.3f}")
    if args.save:
        save(r, args.save)
        print(f"saved {args.save}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m traffic_yolo", description="Indian traffic vehicle detection, tracking and counting")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--weights", default=None, help="model weights (.pt or .onnx); default weights/yolo11s_uvh26.pt")
    common.add_argument("--device", default=None, help="mps | cpu | cuda:0 (auto by default)")
    common.add_argument("--imgsz", type=int, default=None, help="inference size (default: the model's training size)")
    common.add_argument("--conf", type=float, default=0.25)
    common.add_argument("--iou", type=float, default=0.5)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("predict", parents=[common], help="detect vehicles in an image, a folder of images or a video")
    sp.add_argument("source")
    sp.add_argument("--out", default="runs/predict")
    sp.add_argument("--stride", type=int, default=1)
    sp.add_argument("--max-frames", type=int, default=None)
    sp.set_defaults(func=cmd_predict)

    sp = sub.add_parser("track-count", parents=[common], help="track vehicles in a video and count line crossings")
    sp.add_argument("source")
    sp.add_argument("--out", default="runs/track")
    sp.add_argument("--line", action="append", help="x1,y1,x2,y2 in pixels or 0-1 fractions; repeatable "
                                                     "(default: horizontal line at 60%% height)")
    sp.add_argument("--stride", type=int, default=1, help="run the model on every Nth frame")
    sp.add_argument("--max-frames", type=int, default=None)
    sp.add_argument("--no-video", action="store_true", help="skip writing the annotated video")
    sp.set_defaults(func=cmd_track_count)

    sp = sub.add_parser("evaluate", parents=[common], help="mAP / per-class AP on a YOLO-format image folder")
    sp.add_argument("images", help="folder of images (labels are looked up in ../../labels/<split>)")
    sp.add_argument("--labels", default=None)
    sp.add_argument("--limit", type=int, default=None)
    sp.add_argument("--save", default=None, help="write the metrics JSON here")
    sp.set_defaults(func=cmd_evaluate)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

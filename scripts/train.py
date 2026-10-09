"""Fine-tune YOLO11 on the converted UVH-26 subset.

    python scripts/train.py --data ../_data/uvh26/yolo/dataset.yaml --epochs 40

Hyper-parameters used for the released weights are the defaults below
(see docs/metrics/train_args.yaml for the exact resolved config).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from traffic_yolo.pipeline import default_device  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--device", default=None)
    ap.add_argument("--project", default="runs/train")
    ap.add_argument("--name", default="yolo11s_uvh26")
    ap.add_argument("--fraction", type=float, default=1.0, help="use a fraction of the train set (for quick timing runs)")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    from ultralytics import YOLO

    if args.resume:
        YOLO(str(Path(args.project) / args.name / "weights" / "last.pt")).train(resume=True)
        return

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device or default_device(),
        patience=args.patience,
        project=str(Path(args.project).resolve()),
        name=args.name,
        exist_ok=True,
        fraction=args.fraction,
        seed=26,
        deterministic=False,
        cos_lr=True,
        close_mosaic=5,
        # augmentation: Ultralytics defaults (mosaic, HSV jitter, scale 0.5, translate 0.1, h-flip 0.5)
        # plus a little mixup; no rotation/vertical flips because CCTV cameras are upright.
        mixup=0.05,
        plots=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()

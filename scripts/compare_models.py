"""Score v1 (legacy) and v2 on the held-out UVH-26 test subset with the same evaluator.

    python scripts/compare_models.py --data ../_data/uvh26/yolo

Writes docs/metrics/test_v1.json, docs/metrics/test_v2.json and docs/metrics/comparison.json.
Also runs Ultralytics' own validator on v2 as a cross-check (docs/metrics/test_v2_ultralytics.json).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from traffic_yolo.evaluate import evaluate, save, subset_map  # noqa: E402
from traffic_yolo.pipeline import Detector  # noqa: E402

V1_CLASSES = ["car", "two_wheeler", "bus", "truck"]  # the v2 classes the v1 model can express


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True, help="converted dataset root (contains images/test)")
    ap.add_argument("--v1", default="weights/v1_legacy.onnx")
    ap.add_argument("--v2", default="weights/yolo11s_uvh26.pt")
    ap.add_argument("--out", type=Path, default=Path("docs/metrics"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-ultralytics", action="store_true")
    args = ap.parse_args()
    test_imgs = args.data / "images" / "test"

    results = {}
    # Each model runs at its native resolution (the v1 ONNX graph has a fixed 416x416 input).
    for tag, w in [("v1", args.v1), ("v2", args.v2)]:
        det = Detector(w)
        print(f"== {tag}: {w} (imgsz {det.imgsz}, {det.device})")
        r = evaluate(det, test_imgs, limit=args.limit)
        r["subset_v1_classes"] = subset_map(r, V1_CLASSES)
        save(r, args.out / f"test_{tag}.json")
        results[tag] = r
        print(json.dumps({k: r[k] for k in ("mAP50", "mAP50_95", "precision", "recall")}, indent=1))

    comp = {
        "test_set": {"images": results["v2"]["images"], "instances": results["v2"]["instances"]},
        "models": {
            tag: {"weights": r["model"], "imgsz": r["imgsz"], "mAP50": r["mAP50"], "mAP50_95": r["mAP50_95"],
                  "precision": r["precision"], "recall": r["recall"],
                  "mAP50_v1_classes": r["subset_v1_classes"]["mAP50"],
                  "mAP50_95_v1_classes": r["subset_v1_classes"]["mAP50_95"],
                  "per_class_AP50": {k: v["AP50"] for k, v in r["per_class"].items()},
                  "per_class_AP50_95": {k: v["AP50_95"] for k, v in r["per_class"].items()}}
            for tag, r in results.items()
        },
    }
    save(comp, args.out / "comparison.json")

    if not args.skip_ultralytics:
        from ultralytics import YOLO

        yaml = args.data / "dataset.yaml"
        m = YOLO(args.v2)
        v = m.val(data=str(yaml), split="test", imgsz=results["v2"]["imgsz"], batch=8, device="mps", plots=True,
                  project=str(Path("runs/val").resolve()), name="v2_test", exist_ok=True, verbose=True)
        names = v.names
        per = {names[int(c)]: {"AP50": float(v.box.ap50[i]), "AP50_95": float(v.box.ap[i]),
                               "precision": float(v.box.p[i]), "recall": float(v.box.r[i])}
               for i, c in enumerate(v.box.ap_class_index)}
        save({"mAP50": float(v.box.map50), "mAP50_95": float(v.box.map), "precision": float(v.box.mp),
              "recall": float(v.box.mr), "per_class": per, "speed_ms": v.speed}, args.out / "test_v2_ultralytics.json")
        for f in ("confusion_matrix_normalized.png", "BoxPR_curve.png"):
            src = Path(v.save_dir) / f
            if src.exists():
                shutil.copy(src, args.out.parent / "figures" / f"ultralytics_{f}")


if __name__ == "__main__":
    main()

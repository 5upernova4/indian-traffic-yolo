"""Model-agnostic detection evaluation on a YOLO-format split.

Both the v1 model (5 classes, remapped) and the v2 model are scored with exactly the
same matching code, so the comparison is like-for-like. Matching and AP follow
Ultralytics' validator (greedy IoU matching at IoU 0.50:0.95, 101-point interpolated AP),
so v2 numbers agree with ``yolo val`` to within rounding.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .classes import CLASS_NAMES
from .data import parse_label_lines

IOUV = np.linspace(0.5, 0.95, 10)


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between (M,4) and (N,4) xyxy boxes -> (M,N)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    area_a = (a[:, 2:] - a[:, :2]).prod(1)
    area_b = (b[:, 2:] - b[:, :2]).prod(1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def match_predictions(pred_cls: np.ndarray, gt_cls: np.ndarray, iou: np.ndarray) -> np.ndarray:
    """(N_pred, 10) bool matrix of true positives at each IoU threshold (Ultralytics' greedy matching)."""
    correct = np.zeros((len(pred_cls), len(IOUV)), bool)
    if len(pred_cls) == 0 or len(gt_cls) == 0:
        return correct
    iou = iou * (gt_cls[:, None] == pred_cls[None, :])
    for i, thr in enumerate(IOUV):
        m = np.array(np.nonzero(iou >= thr)).T
        if len(m):
            if len(m) > 1:
                m = m[iou[m[:, 0], m[:, 1]].argsort()[::-1]]
                m = m[np.unique(m[:, 1], return_index=True)[1]]
                m = m[np.unique(m[:, 0], return_index=True)[1]]
            correct[m[:, 1].astype(int), i] = True
    return correct


def confusion_update(cm: np.ndarray, pred_xyxy, pred_conf, pred_cls, gt_xyxy, gt_cls, conf: float = 0.25,
                     iou_thr: float = 0.45) -> None:
    """Accumulate an (nc+1)x(nc+1) confusion matrix; rows = predicted, cols = true, last = background."""
    nc = cm.shape[0] - 1
    keep = pred_conf >= conf
    pred_xyxy, pred_cls = pred_xyxy[keep], pred_cls[keep]
    if len(gt_cls) == 0:
        for c in pred_cls:
            cm[c, nc] += 1
        return
    if len(pred_cls) == 0:
        for g in gt_cls:
            cm[nc, g] += 1
        return
    iou = box_iou(gt_xyxy, pred_xyxy)
    m = np.array(np.nonzero(iou > iou_thr)).T
    if len(m):
        m = m[iou[m[:, 0], m[:, 1]].argsort()[::-1]]
        m = m[np.unique(m[:, 1], return_index=True)[1]]
        m = m[iou[m[:, 0], m[:, 1]].argsort()[::-1]]
        m = m[np.unique(m[:, 0], return_index=True)[1]]
    matched_gt, matched_pred = set(), set()
    for g, p in m:
        cm[pred_cls[p], gt_cls[g]] += 1
        matched_gt.add(int(g))
        matched_pred.add(int(p))
    for g in range(len(gt_cls)):
        if g not in matched_gt:
            cm[nc, gt_cls[g]] += 1
    for p in range(len(pred_cls)):
        if p not in matched_pred:
            cm[pred_cls[p], nc] += 1


def load_gt(label_path: Path, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    boxes = parse_label_lines(label_path.read_text()) if label_path.exists() else []
    if not boxes:
        return np.zeros((0, 4), np.float32), np.zeros(0, int)
    a = np.array([b[1:] for b in boxes], np.float32)
    xyxy = np.stack([(a[:, 0] - a[:, 2] / 2) * w, (a[:, 1] - a[:, 3] / 2) * h,
                     (a[:, 0] + a[:, 2] / 2) * w, (a[:, 1] + a[:, 3] / 2) * h], 1)
    return xyxy, np.array([b[0] for b in boxes], int)


def evaluate(detector, images_dir: str | Path, labels_dir: str | Path | None = None, limit: int | None = None,
             names: list[str] = CLASS_NAMES, progress: bool = True) -> dict:
    """Score ``detector`` (a :class:`traffic_yolo.pipeline.Detector`) on a YOLO-format image folder."""
    import cv2
    from ultralytics.utils.metrics import ap_per_class

    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir) if labels_dir else images_dir.parent.parent / "labels" / images_dir.name
    files = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if limit:
        files = files[:limit]
    # mAP needs the full ranking, so evaluate at a very low confidence (as `yolo val` does).
    old = detector.conf, detector.iou
    detector.conf, detector.iou = 0.001, 0.7
    nc = len(names)
    stats = {"tp": [], "conf": [], "pred_cls": [], "target_cls": []}
    cm = np.zeros((nc + 1, nc + 1), np.int64)
    times = []
    try:
        for k, f in enumerate(files):
            img = cv2.imread(str(f))
            h, w = img.shape[:2]
            gt_xyxy, gt_cls = load_gt(labels_dir / f"{f.stem}.txt", w, h)
            t0 = time.perf_counter()
            det = detector.predict(img)
            times.append(time.perf_counter() - t0)
            iou = box_iou(gt_xyxy, det.xyxy)
            stats["tp"].append(match_predictions(det.cls, gt_cls, iou))
            stats["conf"].append(det.conf)
            stats["pred_cls"].append(det.cls)
            stats["target_cls"].append(gt_cls)
            confusion_update(cm, det.xyxy, det.conf, det.cls, gt_xyxy, gt_cls)
            if progress and (k + 1) % 100 == 0:
                print(f"  evaluated {k + 1}/{len(files)}", flush=True)
    finally:
        detector.conf, detector.iou = old
    tp = np.concatenate(stats["tp"], 0)
    conf = np.concatenate(stats["conf"], 0)
    pred_cls = np.concatenate(stats["pred_cls"], 0)
    target_cls = np.concatenate(stats["target_cls"], 0)
    _, _, p, r, _, ap, ucls, *_ = ap_per_class(tp, conf, pred_cls, target_cls, names=dict(enumerate(names)))
    per_class = {}
    counts = np.bincount(target_cls, minlength=nc)
    for i, name in enumerate(names):
        if i in ucls:
            j = int(np.where(ucls == i)[0][0])
            per_class[name] = {"instances": int(counts[i]), "precision": float(p[j]), "recall": float(r[j]),
                               "AP50": float(ap[j, 0]), "AP50_95": float(ap[j].mean())}
        else:
            per_class[name] = {"instances": int(counts[i]), "precision": 0.0, "recall": 0.0, "AP50": 0.0,
                               "AP50_95": 0.0}
    present = [v for v in per_class.values() if v["instances"] > 0]
    return {
        "model": detector.weights.name,
        "imgsz": detector.imgsz,
        "images": len(files),
        "instances": int(len(target_cls)),
        "precision": float(np.mean([v["precision"] for v in present])),
        "recall": float(np.mean([v["recall"] for v in present])),
        "mAP50": float(np.mean([v["AP50"] for v in present])),
        "mAP50_95": float(np.mean([v["AP50_95"] for v in present])),
        "per_class": per_class,
        "confusion_matrix": {"labels": list(names) + ["background"], "rows": "predicted", "cols": "true",
                             "conf": 0.25, "iou": 0.45, "matrix": cm.tolist()},
        "mean_end_to_end_ms": float(np.mean(times) * 1000) if times else None,
    }


def subset_map(result: dict, classes: list[str]) -> dict:
    """mAP restricted to a subset of classes (e.g. the four classes the v1 model knows)."""
    pc = result["per_class"]
    return {"classes": classes,
            "mAP50": float(np.mean([pc[c]["AP50"] for c in classes])),
            "mAP50_95": float(np.mean([pc[c]["AP50_95"] for c in classes]))}


def save(result: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(result, indent=2))

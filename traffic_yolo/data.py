"""COCO -> YOLO conversion helpers for UVH-26 (pure functions, no I/O heavy lifting).

Kept free of torch / ultralytics imports so the tests run anywhere.
"""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path
from typing import Iterable

from .classes import CLASS_NAMES, UVH26_TO_V2, build_id_map

YoloBox = tuple[int, float, float, float, float]  # cls, cx, cy, w, h (normalised)


def coco_bbox_to_yolo(bbox: Iterable[float], img_w: int, img_h: int, min_size: float = 1.0) -> tuple[float, float, float, float] | None:
    """Convert a COCO ``[x, y, w, h]`` pixel box to normalised YOLO ``cx, cy, w, h``.

    The box is clipped to the image first. Boxes thinner than ``min_size`` pixels
    after clipping are discarded (returns ``None``).
    """
    x, y, w, h = (float(v) for v in bbox)
    x1, y1 = max(0.0, x), max(0.0, y)
    x2, y2 = min(float(img_w), x + w), min(float(img_h), y + h)
    bw, bh = x2 - x1, y2 - y1
    if bw < min_size or bh < min_size:
        return None
    return ((x1 + bw / 2) / img_w, (y1 + bh / 2) / img_h, bw / img_w, bh / img_h)


def coco_to_yolo_labels(coco: dict, mapping: dict[str, str | None] = UVH26_TO_V2) -> dict[str, list[YoloBox]]:
    """Map every image in a COCO dict to its list of YOLO boxes in the v2 class space.

    Images with no (remaining) boxes are still returned with an empty list, so they
    are kept as background images.
    """
    cat_names = {c["id"]: c["name"] for c in coco["categories"]}
    id_map = build_id_map(cat_names, mapping)
    images = {im["id"]: im for im in coco["images"]}
    labels: dict[str, list[YoloBox]] = {im["file_name"]: [] for im in coco["images"]}
    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        target = id_map[ann["category_id"]]
        if target is None:
            continue
        im = images[ann["image_id"]]
        box = coco_bbox_to_yolo(ann["bbox"], im["width"], im["height"])
        if box is None:
            continue
        labels[im["file_name"]].append((target, *box))
    return labels


def format_label_lines(boxes: Iterable[YoloBox]) -> str:
    """Render boxes in the YOLO txt format (one ``cls cx cy w h`` per line)."""
    return "".join(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n" for c, cx, cy, w, h in boxes)


def parse_label_lines(text: str) -> list[YoloBox]:
    out: list[YoloBox] = []
    for line in text.strip().splitlines():
        parts = line.split()
        if len(parts) == 5:
            out.append((int(parts[0]), *(float(p) for p in parts[1:])))
    return out


def deterministic_sample(items: Iterable[str], n: int, seed: int) -> list[str]:
    """Sample ``n`` items reproducibly (independent of input order)."""
    pool = sorted(set(items))
    if n >= len(pool):
        return pool
    return sorted(random.Random(seed).sample(pool, n))


def split_items(items: Iterable[str], val_fraction: float, seed: int) -> tuple[list[str], list[str]]:
    """Reproducible train/val split of a list of image names."""
    pool = sorted(set(items))
    rng = random.Random(seed)
    rng.shuffle(pool)
    n_val = int(round(len(pool) * val_fraction))
    return sorted(pool[n_val:]), sorted(pool[:n_val])


def class_histogram(labels: dict[str, list[YoloBox]], names: Iterable[str] | None = None) -> dict[str, int]:
    """Count instances per class across a label dict, keyed by class name."""
    if names is not None:
        names = set(names)
    counts: Counter[int] = Counter()
    for name, boxes in labels.items():
        if names is not None and name not in names:
            continue
        counts.update(b[0] for b in boxes)
    return {CLASS_NAMES[i]: counts.get(i, 0) for i in range(len(CLASS_NAMES))}


def dataset_yaml(root: Path | str, train: str = "images/train", val: str = "images/val", test: str | None = "images/test") -> str:
    """Return the Ultralytics dataset YAML for a converted dataset root."""
    lines = [f"path: {Path(root).resolve()}", f"train: {train}", f"val: {val}"]
    if test:
        lines.append(f"test: {test}")
    lines.append("names:")
    lines += [f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES)]
    return "\n".join(lines) + "\n"

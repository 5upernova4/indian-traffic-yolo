"""Build the README figures from measured metrics.

    python scripts/make_figures.py --run runs/train/yolo11s_uvh26 --data ../_data/uvh26/yolo

Inputs: the training run's results.csv, docs/metrics/test_v1.json + test_v2.json, and a few
held-out UVH-26 test images (for prediction examples). Outputs go to docs/figures/.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from traffic_yolo.classes import CLASS_NAMES, DISPLAY_NAMES  # noqa: E402

# palette: validated categorical slots (light surface) + neutral for the legacy baseline
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#9a9893"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
ATTRIB = "Images: UVH-26 (IISc AIM), CC-BY-4.0"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold", "font.size": 9.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False,
})


def training_curves(run: Path, out: Path) -> None:
    import pandas as pd

    df = pd.read_csv(run / "results.csv")
    df.columns = [c.strip() for c in df.columns]
    ep = df["epoch"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, key, title in [(axes[0], "box_loss", "Box loss"), (axes[1], "cls_loss", "Classification loss")]:
        ax.plot(ep, df[f"train/{key}"], color=BLUE, lw=2, label="train")
        ax.plot(ep, df[f"val/{key}"], color=ORANGE, lw=2, label="val")
        ax.set_title(title, loc="left")
        ax.set_xlabel("epoch")
        ax.legend()
    ax = axes[2]
    ax.plot(ep, df["metrics/mAP50(B)"], color=BLUE, lw=2, label="mAP@50")
    ax.plot(ep, df["metrics/mAP50-95(B)"], color=AQUA, lw=2, label="mAP@50-95")
    best = int(df["metrics/mAP50-95(B)"].idxmax())
    ax.scatter([ep[best]], [df["metrics/mAP50-95(B)"][best]], s=40, color=AQUA, edgecolor=SURFACE, lw=2, zorder=3)
    ax.annotate(f"best {df['metrics/mAP50-95(B)'][best]:.3f}", (ep[best], df["metrics/mAP50-95(B)"][best]),
                textcoords="offset points", xytext=(-10, -16), ha="right", color=INK2, fontsize=8.5)
    ax.set_title("Validation mAP", loc="left")
    ax.set_xlabel("epoch")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "training_curves.png", dpi=160)
    plt.close(fig)


def per_class_ap(v2: dict, out: Path) -> None:
    pc = v2["per_class"]
    names = [DISPLAY_NAMES[n] for n in CLASS_NAMES]
    ap50 = [pc[n]["AP50"] for n in CLASS_NAMES]
    ap = [pc[n]["AP50_95"] for n in CLASS_NAMES]
    inst = [pc[n]["instances"] for n in CLASS_NAMES]
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    h = 0.36
    ax.barh(y - h / 2 - 0.01, ap50, h, color=BLUE, label="AP@50")
    ax.barh(y + h / 2 + 0.01, ap, h, color=ORANGE, label="AP@50-95")
    for i in range(len(names)):
        ax.text(ap50[i] + 0.01, y[i] - h / 2, f"{ap50[i]:.2f}", va="center", color=INK, fontsize=8.5)
        ax.text(ap[i] + 0.01, y[i] + h / 2, f"{ap[i]:.2f}", va="center", color=INK, fontsize=8.5)
    ax.set_yticks(y, [f"{n}\n({k:,} boxes)" for n, k in zip(names, inst)])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.grid(axis="y", visible=False)
    ax.set_title(f"v2 per-class AP on the held-out test set ({v2['images']} images)", loc="left")
    ax.legend(loc="lower right", ncol=2)
    fig.tight_layout()
    fig.savefig(out / "per_class_ap.png", dpi=160)
    plt.close(fig)


def v1_vs_v2(v1: dict, v2: dict, out: Path) -> None:
    groups = ["mAP@50\n(6 classes)", "mAP@50-95\n(6 classes)", "mAP@50\n(4 classes v1 knows)", "mAP@50-95\n(4 classes v1 knows)"]
    a = [v1["mAP50"], v1["mAP50_95"], v1["subset_v1_classes"]["mAP50"], v1["subset_v1_classes"]["mAP50_95"]]
    b = [v2["mAP50"], v2["mAP50_95"], v2["subset_v1_classes"]["mAP50"], v2["subset_v1_classes"]["mAP50_95"]]
    x = np.arange(len(groups))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.bar(x - w / 2 - 0.01, a, w, color=GRAY, label=f"v1 (YOLOv8n, {v1['imgsz']}px)")
    ax.bar(x + w / 2 + 0.01, b, w, color=BLUE, label=f"v2 (YOLO11s fine-tuned on UVH-26, {v2['imgsz']}px)")
    for i in range(len(groups)):
        ax.text(x[i] - w / 2, a[i] + 0.015, f"{a[i]:.2f}", ha="center", color=INK, fontsize=8.5)
        ax.text(x[i] + w / 2, b[i] + 0.015, f"{b[i]:.2f}", ha="center", color=INK, fontsize=8.5)
    ax.set_xticks(x, groups)
    ax.set_ylim(0, 1)
    ax.grid(axis="x", visible=False)
    ax.set_title("v1 vs v2 on the same held-out UVH-26 test images", loc="left")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "v1_vs_v2.png", dpi=160)
    plt.close(fig)


def imgsz_comparison(metrics: Path, out: Path) -> None:
    sizes = sorted(int(p.stem.split("_")[-1]) for p in metrics.glob("test_v2_[0-9]*.json"))
    if len(sizes) < 2:
        return
    res = {sz: json.loads((metrics / f"test_v2_{sz}.json").read_text()) for sz in sizes}
    names = [DISPLAY_NAMES[n] for n in CLASS_NAMES]
    x = np.arange(len(names) + 1)
    w = 0.8 / len(sizes)
    colors = [BLUE, ORANGE, AQUA]
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    for k, sz in enumerate(sizes):
        vals = [res[sz]["per_class"][n]["AP50_95"] for n in CLASS_NAMES] + [res[sz]["mAP50_95"]]
        xs = x - 0.4 + w * (k + 0.5)
        ax.bar(xs, vals, w * 0.94, color=colors[k % 3], label=f"inference at {sz}px")
        for xi, v in zip(xs, vals):
            ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", color=INK, fontsize=7.5)
    ax.set_xticks(x, names + ["All (mAP)"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("AP@50-95")
    ax.grid(axis="x", visible=False)
    ax.set_title("v2 (trained at 640px): held-out AP@50-95 by inference size", loc="left")
    ax.legend(loc="upper right", ncol=len(sizes))
    fig.tight_layout()
    fig.savefig(out / "imgsz_comparison.png", dpi=160)
    plt.close(fig)


def confusion(v2: dict, out: Path) -> None:
    cm = np.array(v2["confusion_matrix"]["matrix"], float)
    labels = [DISPLAY_NAMES.get(n, n.capitalize()) for n in v2["confusion_matrix"]["labels"]]
    norm = cm / np.clip(cm.sum(0, keepdims=True), 1, None)  # each true-class column sums to 1
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.grid(False)
    for i in range(norm.shape[0]):
        for j in range(norm.shape[1]):
            if cm[i, j] == 0:
                continue
            ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="#ffffff" if norm[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("True class")
    ax.set_ylabel("Predicted class")
    ax.set_title("v2 confusion matrix (conf 0.25, IoU 0.45), column-normalised", loc="left")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=160)
    plt.close(fig)


def _footer(img: np.ndarray, text: str) -> np.ndarray:
    bar = np.full((28, img.shape[1], 3), 250, np.uint8)
    cv2.putText(bar, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1, cv2.LINE_AA)
    return np.vstack([img, bar])


def prediction_grid(data: Path, out: Path, samples_dir: Path, names: list[str]) -> None:
    from traffic_yolo.pipeline import Detector, process_image

    det = Detector()
    tiles = []
    tmp = out / "_tmp"
    for n in names:
        src = data / "images" / "test" / n
        samples_dir.mkdir(parents=True, exist_ok=True)
        small = cv2.imread(str(src))
        cv2.imwrite(str(samples_dir / n), small, [cv2.IMWRITE_JPEG_QUALITY, 85])
        process_image(src, tmp / n, det)
        tiles.append(cv2.resize(cv2.imread(str(tmp / n / "annotated.jpg")), (960, 540), interpolation=cv2.INTER_AREA))
    rows = [np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]
    grid = _footer(np.vstack(rows), f"v2 predictions on held-out test images. {ATTRIB}.")
    cv2.imwrite(str(out / "predictions_grid.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
    shutil.rmtree(tmp, ignore_errors=True)


def side_by_side(data: Path, out: Path, name: str) -> None:
    from traffic_yolo.pipeline import Detector, process_image

    tmp = out / "_tmp"
    tiles = []
    for tag, w in [("v1 (legacy)", "weights/v1_legacy.onnx"), ("v2 (this release)", None)]:
        det = Detector(w)
        tag = f"{tag}, {det.imgsz}px"
        process_image(data / "images" / "test" / name, tmp / tag, det)
        t = cv2.resize(cv2.imread(str(tmp / tag / "annotated.jpg")), (960, 540), interpolation=cv2.INTER_AREA)
        cv2.rectangle(t, (0, 500), (960, 540), (20, 20, 20), -1)
        cv2.putText(t, tag, (12, 528), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(t)
    img = _footer(np.hstack(tiles), f"Same held-out test image, confidence >= 0.25. {ATTRIB}.")
    cv2.imwrite(str(out / "v1_vs_v2_example.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--metrics", type=Path, default=Path("docs/metrics"))
    ap.add_argument("--out", type=Path, default=Path("docs/figures"))
    ap.add_argument("--samples", nargs="*", default=[])
    ap.add_argument("--compare-sample", default=None)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    v1 = json.loads((args.metrics / "test_v1.json").read_text())
    v2 = json.loads((args.metrics / "test_v2.json").read_text())
    training_curves(args.run, args.out)
    per_class_ap(v2, args.out)
    v1_vs_v2(v1, v2, args.out)
    confusion(v2, args.out)
    imgsz_comparison(args.metrics, args.out)
    if args.data and args.samples:
        prediction_grid(args.data, args.out, Path("docs/samples"), args.samples)
    if args.data and args.compare_sample:
        side_by_side(args.data, args.out, args.compare_sample)


if __name__ == "__main__":
    main()

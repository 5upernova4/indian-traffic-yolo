"""Download a reproducible subset of UVH-26 and convert it to YOLO format.

    python scripts/prepare_uvh26.py --out ../_data/uvh26 --n-train 4200 --n-test 800

* Uses the majority-vote (MV) annotations.
* Samples images deterministically (``--seed``) from the official train and val splits.
* The sampled *train* images are split again into train / val (for model selection);
  the sampled *official val* images become the held-out **test** set.
* Images are fetched over plain HTTPS with a few parallel workers; PNGs are downscaled to JPEG (long side ``--long-side``) and the PNG is deleted
  straight away, so peak disk use stays small. The script is resumable.

UVH-26 is released by IISc AIM under CC-BY-4.0: https://huggingface.co/datasets/iisc-aim/UVH-26
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import requests
from huggingface_hub import HfApi, hf_hub_download, hf_hub_url

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from traffic_yolo.data import (  # noqa: E402
    class_histogram,
    coco_to_yolo_labels,
    dataset_yaml,
    deterministic_sample,
    format_label_lines,
    split_items,
)

REPO = "iisc-aim/UVH-26"
ANN = {"train": "UVH-26-Train/UVH-26-MV-Train.json", "val": "UVH-26-Val/UVH-26-MV-Val.json"}


def downscale_to_jpeg(src: Path, dst: Path, long_side: int, quality: int = 92) -> None:
    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"could not read {src}")
    h, w = img.shape[:2]
    scale = long_side / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, quality])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("../_data/uvh26"))
    ap.add_argument("--n-train", type=int, default=4200, help="images sampled from the official train split")
    ap.add_argument("--n-test", type=int, default=800, help="images sampled from the official val split (held-out test)")
    ap.add_argument("--val-fraction", type=float, default=0.15, help="share of sampled train images used for model selection")
    ap.add_argument("--seed", type=int, default=26)
    ap.add_argument("--long-side", type=int, default=1280)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--manifest", type=Path, default=None, help="optional path to also write the sample manifest")
    args = ap.parse_args()

    raw = args.out / "raw"
    yolo = args.out / "yolo"
    raw.mkdir(parents=True, exist_ok=True)

    # 1. annotations + file listing -------------------------------------------------
    coco = {}
    for split, rel in ANN.items():
        path = Path(hf_hub_download(REPO, rel, repo_type="dataset", local_dir=raw))
        coco[split] = json.loads(path.read_text())
    listing = [s.rfilename for s in HfApi().dataset_info(REPO).siblings if s.rfilename.endswith(".png")]
    repo_path = {}
    for f in listing:
        top = "train" if f.startswith("UVH-26-Train/") else "val"
        repo_path[(top, Path(f).name)] = f

    # 2. deterministic sample + split ----------------------------------------------
    labels = {s: coco_to_yolo_labels(c) for s, c in coco.items()}
    avail = {s: [n for n in labels[s] if (s, n) in repo_path] for s in labels}
    train_pool = deterministic_sample(avail["train"], args.n_train, args.seed)
    test_set = deterministic_sample(avail["val"], args.n_test, args.seed)
    train_set, val_set = split_items(train_pool, args.val_fraction, args.seed)
    plan = {"train": ("train", train_set), "val": ("train", val_set), "test": ("val", test_set)}
    manifest = {
        "source": f"https://huggingface.co/datasets/{REPO}",
        "annotations": "MV (majority vote)",
        "seed": args.seed,
        "long_side": args.long_side,
        "splits": {k: {"from_official": src, "images": names} for k, (src, names) in plan.items()},
        "instances": {k: class_histogram(labels[src], names) for k, (src, names) in plan.items()},
    }
    yolo.mkdir(parents=True, exist_ok=True)
    (yolo / "manifest.json").write_text(json.dumps(manifest, indent=1))
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, indent=1))
    for k, (_, names) in plan.items():
        print(f"{k:5s}: {len(names):5d} images  {manifest['instances'][k]}")

    # 3. labels ---------------------------------------------------------------------
    for split, (src, names) in plan.items():
        d = yolo / "labels" / split
        d.mkdir(parents=True, exist_ok=True)
        for n in names:
            (d / f"{Path(n).stem}.txt").write_text(format_label_lines(labels[src][n]))
    (yolo / "dataset.yaml").write_text(dataset_yaml(yolo))

    # 4. images (parallel HTTP download -> jpeg -> delete png) ------------------------
    todo = []
    for split, (src, names) in plan.items():
        for n in names:
            dst = yolo / "images" / split / f"{Path(n).stem}.jpg"
            if not dst.exists():
                todo.append((repo_path[(src, n)], dst))
    print(f"{len(todo)} images to download", flush=True)
    tmp = raw / "png"
    tmp.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    def fetch(item: tuple[str, Path]) -> bool:
        rel, dst = item
        png = tmp / Path(rel).name
        url = hf_hub_url(REPO, rel, repo_type="dataset")
        for attempt in range(6):
            try:
                with session.get(url, stream=True, timeout=120) as r:
                    r.raise_for_status()
                    with open(png, "wb") as f:
                        for chunk in r.iter_content(1 << 20):
                            f.write(chunk)
                if png.stat().st_size < 10_000 or png.read_bytes()[:4] != b"\x89PNG":
                    raise RuntimeError("bad payload")
                downscale_to_jpeg(png, dst, args.long_side)
                png.unlink()
                return True
            except Exception as e:  # network hiccups / throttling: back off and retry
                png.unlink(missing_ok=True)
                time.sleep(5 * (attempt + 1))
                last = e
        print(f"  failed {rel}: {last!r}", flush=True)
        return False

    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(args.workers) as pool:
        for ok in pool.map(fetch, todo):
            done += 1
            if done % 100 == 0 or done == len(todo):
                rate = done / (time.time() - t0)
                print(f"  {done}/{len(todo)} images ({rate:.2f} img/s, eta {(len(todo) - done) / rate / 60:.0f} min)",
                      flush=True)

    missing = [d for _, d in todo if not d.exists()]
    print(f"finished; {len(missing)} images missing")


if __name__ == "__main__":
    main()

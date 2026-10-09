"""Make a short SYNTHETIC video from a single still image (slow pan + zoom).

There is no real traffic video in this repo. This clip only exists to sanity-check the
tracking / counting / video-export code paths: every vehicle moves because the virtual
camera moves, not because traffic moves.

    python scripts/make_synthetic_clip.py docs/samples/<image>.jpg runs/synthetic.mp4
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("out")
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--size", default="960x540")
    args = ap.parse_args()
    W, H = (int(v) for v in args.size.split("x"))
    img = cv2.imread(args.image)
    ih, iw = img.shape[:2]
    n = int(args.seconds * args.fps)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))
    for i in range(n):
        t = i / max(n - 1, 1)
        t = t * t * (3 - 2 * t)  # ease in/out
        zoom = 0.62 + 0.10 * t  # window size as a fraction of the image
        cw, ch = iw * zoom, ih * zoom
        # pan from the bottom-left towards the top-right of the image
        cx = cw / 2 + (iw - cw) * (0.15 + 0.7 * t)
        cy = ih - ch / 2 - (ih - ch) * t
        m = np.float32([[W / cw, 0, -(cx - cw / 2) * W / cw], [0, H / ch, -(cy - ch / 2) * H / ch]])
        frame = cv2.warpAffine(img, m, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        cv2.putText(frame, "SYNTHETIC: pan over a still image", (W - 330, H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(frame)
    writer.release()
    print(f"wrote {n} frames to {args.out}")


if __name__ == "__main__":
    main()

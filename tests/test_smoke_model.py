"""End-to-end smoke tests with the shipped weights (CPU). Skipped if ultralytics is not installed."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

pytest.importorskip("ultralytics")
pytest.importorskip("onnxruntime")

from traffic_yolo.classes import CLASS_NAMES  # noqa: E402
from traffic_yolo.pipeline import Detector, process_image, process_video  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = sorted((ROOT / "docs" / "samples").glob("*.jpg"))
ONNX = ROOT / "weights" / "yolo11s_uvh26.onnx"


@pytest.fixture(scope="module")
def detector():
    if not ONNX.exists() or not SAMPLES:
        pytest.skip("weights or sample images missing")
    return Detector(ONNX, device="cpu")


def test_detects_vehicles_in_sample(detector, tmp_path):
    s = process_image(SAMPLES[0], tmp_path, detector)
    assert s["total_vehicles"] > 0
    assert set(s["detections_per_class"]) == set(CLASS_NAMES)
    assert (tmp_path / "annotated.jpg").exists() and (tmp_path / "detections.csv").exists()


def test_boxes_are_in_original_pixel_space(detector):
    img = cv2.imread(str(SAMPLES[0]))
    det = detector.predict(img)
    h, w = img.shape[:2]
    assert (det.xyxy[:, 0] >= -1).all() and (det.xyxy[:, 2] <= w + 1).all()
    assert (det.xyxy[:, 1] >= -1).all() and (det.xyxy[:, 3] <= h + 1).all()
    # boxes should not all be squashed into a 960x960 (or 416x416) corner
    assert det.xyxy[:, 2].max() > 0.5 * w or det.xyxy[:, 3].max() > 0.5 * h


def test_track_count_on_tiny_synthetic_clip(detector, tmp_path):
    img = cv2.imread(str(SAMPLES[0]))
    h, w = img.shape[:2]
    clip = tmp_path / "clip.mp4"
    vw = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 10, (w // 2, h // 2))
    for i in range(12):  # slow vertical pan
        m = np.float32([[0.5, 0, 0], [0, 0.5, -i * 4]])
        vw.write(cv2.warpAffine(img, m, (w // 2, h // 2), borderMode=cv2.BORDER_REFLECT))
    vw.release()
    s = process_video(clip, tmp_path / "out", detector, write_video=False)
    assert s["frames_processed"] == 12
    assert sum(s["unique_vehicles_per_class"].values()) > 0
    saved = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert saved["lines"][0]["name"] == "line1"

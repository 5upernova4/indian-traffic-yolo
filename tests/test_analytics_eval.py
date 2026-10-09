import json

import numpy as np
import pytest

from traffic_yolo.analytics import DensityHeatmap, TrafficAnalytics
from traffic_yolo.counting import Crossing
from traffic_yolo.evaluate import box_iou, confusion_update, match_predictions


def test_box_iou():
    a = np.array([[0, 0, 10, 10]], float)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], float)
    assert box_iou(a, b)[0] == pytest.approx([1.0, 1 / 3, 0.0], abs=1e-6)
    assert box_iou(a[:0], b).shape == (0, 3)


def test_match_predictions_respects_class_and_iou():
    gt_cls = np.array([0, 1])
    gt = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], float)
    pred = np.array([[0, 0, 10, 10], [20, 20, 30, 30], [0, 0, 10, 10], [21, 21, 30, 30]], float)
    pred_cls = np.array([0, 0, 0, 1])
    tp = match_predictions(pred_cls, gt_cls, box_iou(gt, pred))
    assert tp[0].all()               # perfect match
    assert not tp[1].any()           # wrong class
    assert not tp[2].any()           # duplicate of an already matched gt
    assert tp[3, 0] and not tp[3, -1]  # IoU 0.81: TP at 0.5, FP at 0.95


def test_confusion_matrix():
    cm = np.zeros((3, 3), int)  # 2 classes + background
    gt = np.array([[0, 0, 10, 10], [20, 20, 30, 30], [50, 50, 60, 60]], float)
    gt_cls = np.array([0, 1, 1])
    pred = np.array([[0, 0, 10, 10], [20, 20, 30, 30], [80, 80, 90, 90]], float)
    confusion_update(cm, pred, np.array([0.9, 0.9, 0.9]), np.array([0, 0, 1]), gt, gt_cls)
    assert cm[0, 0] == 1   # correct
    assert cm[0, 1] == 1   # class-1 predicted as class 0
    assert cm[2, 1] == 1   # missed class-1 object
    assert cm[1, 2] == 1   # false positive


def test_heatmap_peaks_where_points_are():
    h = DensityHeatmap(320, 240, cell=8)
    h.add([(160, 120)] * 20 + [(10, 10)])
    n = h.normalised()
    gy, gx = np.unravel_index(n.argmax(), n.shape)
    assert (gx, gy) == (20, 15)
    img = h.render(np.zeros((240, 320, 3), np.uint8))
    assert img.shape == (240, 320, 3)


def test_analytics_summary_and_export(tmp_path):
    a = TrafficAnalytics(fps=10, bin_seconds=1)
    for f in range(20):
        a.add_frame(f, np.array([[0, 0, 10, 10], [5, 5, 20, 20]]), np.array([0.9, 0.8]), np.array([0, 2]),
                    np.array([1, 2]))
    a.add_crossings([Crossing("line1", 1, 0, "in", 12), Crossing("line1", 2, 2, "out", 15)])
    s = a.write_json(tmp_path / "s.json", source="x")
    assert s["frames_processed"] == 20 and s["duration_s"] == 2.0
    assert s["unique_vehicles_per_class"]["car"] == 1 and s["unique_vehicles_per_class"]["two_wheeler"] == 1
    assert s["line_counts"]["line1"]["total"] == 2
    assert s["line_counts"]["line1"]["in"]["car"] == 1 and s["line_counts"]["line1"]["out"]["two_wheeler"] == 1
    assert len(s["timeline"]) == 2 and s["timeline"][0]["car"] == 1.0
    assert json.loads((tmp_path / "s.json").read_text())["source"] == "x"
    a.write_csv(tmp_path / "d.csv")
    a.write_crossings_csv(tmp_path / "c.csv")
    assert len((tmp_path / "d.csv").read_text().strip().splitlines()) == 41
    assert "line1,1,0,car,in,12,1.2" in (tmp_path / "c.csv").read_text()

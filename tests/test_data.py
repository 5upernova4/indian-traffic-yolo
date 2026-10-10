import pytest

from traffic_yolo.classes import CLASS_IDS, CLASS_NAMES, UVH26_TO_V2, V1_NAMES, V1_TO_V2, build_id_map
from traffic_yolo.data import (
    class_histogram,
    coco_bbox_to_yolo,
    coco_to_yolo_labels,
    dataset_yaml,
    deterministic_sample,
    format_label_lines,
    parse_label_lines,
    split_items,
)

UVH26_CATEGORIES = ["Hatchback", "Sedan", "SUV", "MUV", "Bus", "Truck", "Three-wheeler", "Two-wheeler", "LCV",
                    "Mini-bus", "Tempo-traveller", "Bicycle", "Van", "Others"]


def make_coco(anns, w=1920, h=1080):
    return {
        "images": [{"id": 0, "file_name": "a.png", "width": w, "height": h},
                   {"id": 1, "file_name": "empty.png", "width": w, "height": h}],
        "categories": [{"id": i + 1, "name": n} for i, n in enumerate(UVH26_CATEGORIES)],
        "annotations": [{"id": k, "image_id": 0, "category_id": c, "bbox": b, "iscrowd": 0} for k, (c, b) in enumerate(anns)],
    }


def test_every_uvh26_category_is_mapped():
    for name in UVH26_CATEGORIES:
        assert name in UVH26_TO_V2


@pytest.mark.parametrize("src,dst", [
    ("Hatchback", "car"), ("Sedan", "car"), ("SUV", "car"), ("MUV", "car"), ("Van", "car"),
    ("Three-wheeler", "auto"), ("Two-wheeler", "two_wheeler"), ("Bicycle", "two_wheeler"),
    ("Bus", "bus"), ("Mini-bus", "bus"), ("Tempo-traveller", "bus"), ("Truck", "truck"), ("LCV", "lcv"),
])
def test_uvh26_mapping(src, dst):
    assert UVH26_TO_V2[src] == dst


def test_other_is_dropped():
    assert UVH26_TO_V2["Other"] is None and UVH26_TO_V2["Others"] is None


def test_v1_mapping_ids():
    m = build_id_map(V1_NAMES, V1_TO_V2)
    assert m == {0: None, 1: CLASS_IDS["bus"], 2: CLASS_IDS["car"], 3: CLASS_IDS["two_wheeler"], 4: CLASS_IDS["truck"]}


def test_unknown_class_raises():
    with pytest.raises(KeyError):
        build_id_map(["Spaceship"], UVH26_TO_V2)


def test_bbox_conversion_centre_and_size():
    cx, cy, w, h = coco_bbox_to_yolo([960 - 50, 540 - 25, 100, 50], 1920, 1080)
    assert cx == pytest.approx(0.5) and cy == pytest.approx(0.5)
    assert w == pytest.approx(100 / 1920) and h == pytest.approx(50 / 1080)


def test_bbox_is_clipped_to_image():
    cx, cy, w, h = coco_bbox_to_yolo([-10, -10, 30, 30], 100, 100)
    assert (cx, cy, w, h) == pytest.approx((0.1, 0.1, 0.2, 0.2))
    cx, cy, w, h = coco_bbox_to_yolo([90, 90, 30, 30], 100, 100)
    assert (cx, cy, w, h) == pytest.approx((0.95, 0.95, 0.1, 0.1))


def test_degenerate_bbox_dropped():
    assert coco_bbox_to_yolo([10, 10, 0.5, 20], 100, 100) is None
    assert coco_bbox_to_yolo([200, 200, 10, 10], 100, 100) is None  # fully outside


def test_coco_to_yolo_maps_and_drops():
    coco = make_coco([
        (1, [0, 0, 192, 108]),      # Hatchback -> car
        (7, [0, 0, 192, 108]),      # Three-wheeler -> auto
        (12, [0, 0, 192, 108]),     # Bicycle -> two_wheeler
        (11, [0, 0, 192, 108]),     # Tempo-traveller -> bus
        (14, [0, 0, 192, 108]),     # Others -> dropped
    ])
    labels = coco_to_yolo_labels(coco)
    assert set(labels) == {"a.png", "empty.png"}
    assert labels["empty.png"] == []  # background image kept
    assert [b[0] for b in labels["a.png"]] == [CLASS_IDS[c] for c in ("car", "auto", "two_wheeler", "bus")]
    assert labels["a.png"][0][1:] == pytest.approx((0.05, 0.05, 0.1, 0.1))


def test_crowd_annotations_skipped():
    coco = make_coco([(1, [0, 0, 10, 10])])
    coco["annotations"][0]["iscrowd"] = 1
    assert coco_to_yolo_labels(coco)["a.png"] == []


def test_label_roundtrip():
    boxes = [(0, 0.5, 0.25, 0.1, 0.2), (5, 0.123456, 0.654321, 0.01, 0.02)]
    text = format_label_lines(boxes)
    assert text.count("\n") == 2
    assert parse_label_lines(text) == [pytest.approx(b) for b in boxes]


def test_sampling_is_deterministic_and_order_independent():
    items = [f"{i}.png" for i in range(1000)]
    a = deterministic_sample(items, 50, seed=26)
    b = deterministic_sample(list(reversed(items)), 50, seed=26)
    assert a == b and len(a) == 50 and len(set(a)) == 50
    assert deterministic_sample(items, 50, seed=27) != a
    assert deterministic_sample(items[:10], 50, seed=1) == sorted(items[:10])


def test_split_is_disjoint_and_complete():
    items = [f"{i}.png" for i in range(200)]
    tr, va = split_items(items, 0.15, seed=26)
    assert len(va) == 30 and len(tr) == 170
    assert not set(tr) & set(va) and set(tr) | set(va) == set(items)
    assert split_items(items, 0.15, seed=26) == (tr, va)


def test_class_histogram():
    labels = {"a": [(0, .5, .5, .1, .1), (0, .5, .5, .1, .1), (3, .5, .5, .1, .1)], "b": [(5, .5, .5, .1, .1)]}
    assert class_histogram(labels) == {"car": 2, "auto": 0, "two_wheeler": 0, "bus": 1, "truck": 0, "lcv": 1}
    assert class_histogram(labels, names=["b"])["lcv"] == 1 and class_histogram(labels, names=["b"])["car"] == 0


def test_dataset_yaml_lists_all_classes(tmp_path):
    y = dataset_yaml(tmp_path)
    for i, n in enumerate(CLASS_NAMES):
        assert f"  {i}: {n}" in y
    assert "test: images/test" in y

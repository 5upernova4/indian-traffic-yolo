"""Class taxonomy used by the v2 model, plus mappings from other label spaces.

The v2 detector predicts six vehicle classes that match how traffic on Indian
roads is usually described: cars, autos, two-wheelers, buses, trucks and light
commercial vehicles (LCVs).
"""

from __future__ import annotations

CLASS_NAMES: list[str] = ["car", "auto", "two_wheeler", "bus", "truck", "lcv"]
CLASS_IDS: dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}

# Human-friendly labels for UI / plots.
DISPLAY_NAMES: dict[str, str] = {
    "car": "Car",
    "auto": "Auto-rickshaw",
    "two_wheeler": "Two-wheeler",
    "bus": "Bus",
    "truck": "Truck",
    "lcv": "LCV",
}

# BGR colours (OpenCV) per class, chosen to stay distinguishable on video.
CLASS_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "car": (94, 197, 34),          # green
    "auto": (11, 158, 245),        # amber
    "two_wheeler": (212, 182, 6),  # cyan
    "bus": (246, 92, 139),         # violet
    "truck": (68, 68, 239),        # red
    "lcv": (153, 72, 236),         # pink
}

# UVH-26 (14 fine-grained categories) -> 6 v2 classes. ``None`` means the
# category is dropped (its boxes are removed from the labels).
UVH26_TO_V2: dict[str, str | None] = {
    "Hatchback": "car",
    "Sedan": "car",
    "SUV": "car",
    "MUV": "car",
    "Van": "car",
    "Three-wheeler": "auto",
    "Two-wheeler": "two_wheeler",
    "Bicycle": "two_wheeler",
    "Bus": "bus",
    "Mini-bus": "bus",
    "Tempo-traveller": "bus",
    "Truck": "truck",
    "LCV": "lcv",
    "Other": None,
    "Others": None,  # the released JSON spells it "Others"
}

# The original v1 model (weights/v1_legacy.onnx) is a YOLOv8n with five classes
# (read from the ONNX metadata). Map them into the v2 space for a like-for-like
# comparison. Ambulance has no counterpart in UVH-26, so it is ignored.
V1_NAMES: list[str] = ["Ambulance", "Bus", "Car", "Motorcycle", "Truck"]
V1_TO_V2: dict[str, str | None] = {
    "Ambulance": None,
    "Bus": "bus",
    "Car": "car",
    "Motorcycle": "two_wheeler",
    "Truck": "truck",
}


def color_for(cls: int | str) -> tuple[int, int, int]:
    """Return the BGR colour for a class id or name."""
    name = CLASS_NAMES[cls] if isinstance(cls, int) else cls
    return CLASS_COLORS_BGR.get(name, (200, 200, 200))


def build_id_map(src_names: dict[int, str] | list[str], mapping: dict[str, str | None]) -> dict[int, int | None]:
    """Turn a name->name mapping into a src_id -> v2_id mapping.

    Unknown source names raise ``KeyError`` so a silent mislabel cannot happen.
    """
    items = src_names.items() if isinstance(src_names, dict) else enumerate(src_names)
    out: dict[int, int | None] = {}
    for src_id, name in items:
        if name not in mapping:
            raise KeyError(f"No mapping defined for source class {name!r}")
        target = mapping[name]
        out[int(src_id)] = None if target is None else CLASS_IDS[target]
    return out

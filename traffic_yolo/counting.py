"""Virtual counting lines for tracked vehicles.

A :class:`LineCounter` watches the *anchor point* of each track (bottom-centre of
its box, i.e. roughly where the vehicle touches the road). A vehicle is counted
when its anchor moves from one side of the line segment to the other, and the
crossing point actually falls on the segment (not on its infinite extension).
Each track is counted at most once per line.

Direction convention, for a line drawn from ``p1`` to ``p2``:

* ``"in"``  - the anchor moved from the left-hand side of ``p1 -> p2`` to the right-hand side.
* ``"out"`` - the opposite.

In image coordinates (y grows downwards) a horizontal line drawn left-to-right
therefore counts vehicles moving *down* the frame as ``in`` and vehicles moving
*up* the frame as ``out``.

Pure Python / NumPy so it is fast to unit-test.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

Point = tuple[float, float]


def side_of_line(p1: Point, p2: Point, pt: Point) -> float:
    """Signed area test: >0 if ``pt`` is on one side of p1->p2, <0 on the other, 0 on the line."""
    return (p2[0] - p1[0]) * (pt[1] - p1[1]) - (p2[1] - p1[1]) * (pt[0] - p1[0])


def segments_intersect(a1: Point, a2: Point, b1: Point, b2: Point) -> bool:
    """True if segment a1-a2 intersects segment b1-b2 (touching counts)."""
    d1 = side_of_line(b1, b2, a1)
    d2 = side_of_line(b1, b2, a2)
    d3 = side_of_line(a1, a2, b1)
    d4 = side_of_line(a1, a2, b2)
    if ((d1 > 0 > d2) or (d1 < 0 < d2)) and ((d3 > 0 > d4) or (d3 < 0 < d4)):
        return True

    def on_seg(p: Point, q: Point, r: Point) -> bool:
        return min(p[0], q[0]) <= r[0] <= max(p[0], q[0]) and min(p[1], q[1]) <= r[1] <= max(p[1], q[1])

    return (d1 == 0 and on_seg(b1, b2, a1)) or (d2 == 0 and on_seg(b1, b2, a2)) or \
           (d3 == 0 and on_seg(a1, a2, b1)) or (d4 == 0 and on_seg(a1, a2, b2))


def box_anchor(xyxy: np.ndarray | list[float]) -> Point:
    """Bottom-centre of an ``x1, y1, x2, y2`` box."""
    x1, _, x2, y2 = (float(v) for v in xyxy[:4])
    return ((x1 + x2) / 2.0, y2)


@dataclass
class Crossing:
    line: str
    track_id: int
    cls: int
    direction: str  # "in" | "out"
    frame: int


@dataclass
class LineCounter:
    """Counts tracks that cross the segment ``p1 -> p2``."""

    p1: Point
    p2: Point
    name: str = "line"
    _last_pos: dict[int, Point] = field(default_factory=dict, repr=False)
    _last_side: dict[int, int] = field(default_factory=dict, repr=False)
    _counted: set[int] = field(default_factory=set, repr=False)
    crossings: list[Crossing] = field(default_factory=list)

    def update(self, track_ids, anchors, classes, frame: int = 0) -> list[Crossing]:
        """Feed one frame of tracks. Returns the crossings that happened on this frame."""
        new: list[Crossing] = []
        for tid, pt, cls in zip(track_ids, anchors, classes):
            tid = int(tid)
            pt = (float(pt[0]), float(pt[1]))
            s = side_of_line(self.p1, self.p2, pt)
            side = 0 if s == 0 else (1 if s > 0 else -1)
            prev_side = self._last_side.get(tid)
            prev_pos = self._last_pos.get(tid)
            if (
                side != 0
                and prev_side is not None
                and side != prev_side
                and tid not in self._counted
                and prev_pos is not None
                and segments_intersect(prev_pos, pt, self.p1, self.p2)
            ):
                # left (-) -> right (+) of p1->p2 in image coords is "in"
                direction = "in" if side > 0 else "out"
                c = Crossing(self.name, tid, int(cls), direction, frame)
                self.crossings.append(c)
                self._counted.add(tid)
                new.append(c)
            if side != 0:
                self._last_side[tid] = side
                self._last_pos[tid] = pt
            elif prev_pos is None:
                self._last_pos[tid] = pt
        return new

    @property
    def total(self) -> int:
        return len(self.crossings)

    def counts(self) -> dict[str, dict[int, int]]:
        """``{"in": {cls: n}, "out": {cls: n}}``"""
        out: dict[str, Counter] = {"in": Counter(), "out": Counter()}
        for c in self.crossings:
            out[c.direction][c.cls] += 1
        return {k: dict(v) for k, v in out.items()}


class TrackClassVoter:
    """Stabilises a track's class by majority vote over its history (confidence weighted).

    Detectors can flip between visually close classes (e.g. lcv / truck) from frame to
    frame; counting with the voted class keeps the per-class tallies consistent.
    """

    def __init__(self) -> None:
        self._votes: dict[int, Counter] = defaultdict(Counter)

    def update(self, track_id: int, cls: int, conf: float = 1.0) -> int:
        self._votes[int(track_id)][int(cls)] += float(conf)
        return self.get(track_id)

    def get(self, track_id: int) -> int:
        votes = self._votes.get(int(track_id))
        if not votes:
            return -1
        return max(votes.items(), key=lambda kv: (kv[1], -kv[0]))[0]

    def classes(self) -> dict[int, int]:
        return {tid: self.get(tid) for tid in self._votes}


def parse_line(spec: str, width: int, height: int) -> tuple[Point, Point]:
    """Parse ``"x1,y1,x2,y2"``. Values <= 1 are treated as fractions of the frame size."""
    vals = [float(v) for v in spec.replace(" ", "").split(",")]
    if len(vals) != 4:
        raise ValueError(f"line must be x1,y1,x2,y2 (got {spec!r})")
    if all(0.0 <= v <= 1.0 for v in vals):
        vals = [vals[0] * width, vals[1] * height, vals[2] * width, vals[3] * height]
    return (vals[0], vals[1]), (vals[2], vals[3])

import numpy as np
import pytest

from traffic_yolo.counting import LineCounter, TrackClassVoter, box_anchor, parse_line, segments_intersect, side_of_line


def run(counter, tracks):
    """tracks: list of frames; each frame is a list of (track_id, (x, y), cls)."""
    for f, frame in enumerate(tracks):
        if frame:
            ids, pts, cls = zip(*frame)
        else:
            ids, pts, cls = (), (), ()
        counter.update(ids, pts, cls, frame=f)
    return counter


def horizontal(y=100.0, x1=0.0, x2=200.0):
    return LineCounter((x1, y), (x2, y), "h")


def test_side_of_line_sign():
    assert side_of_line((0, 0), (10, 0), (5, 5)) > 0   # below in image coords
    assert side_of_line((0, 0), (10, 0), (5, -5)) < 0
    assert side_of_line((0, 0), (10, 0), (5, 0)) == 0


def test_segments_intersect():
    assert segments_intersect((0, -1), (0, 1), (-1, 0), (1, 0))
    assert not segments_intersect((5, -1), (5, 1), (-1, 0), (1, 0))  # misses the segment
    assert segments_intersect((1, -1), (1, 1), (-1, 0), (1, 0))     # touches the end point


def test_downward_crossing_counts_as_in():
    c = run(horizontal(), [[(1, (50, 80), 0)], [(1, (50, 95), 0)], [(1, (50, 110), 0)]])
    assert c.total == 1
    assert c.crossings[0].direction == "in" and c.crossings[0].frame == 2
    assert c.counts() == {"in": {0: 1}, "out": {}}


def test_upward_crossing_counts_as_out():
    c = run(horizontal(), [[(7, (50, 130), 3)], [(7, (50, 90), 3)]])
    assert c.counts() == {"in": {}, "out": {3: 1}}


def test_track_counted_only_once_even_if_it_oscillates():
    frames = [[(1, (50, y), 0)] for y in (90, 110, 90, 110, 90)]
    assert run(horizontal(), frames).total == 1


def test_no_count_without_crossing():
    frames = [[(1, (50, y), 0)] for y in (10, 20, 30, 40, 99)]
    assert run(horizontal(), frames).total == 0


def test_crossing_outside_segment_is_ignored():
    # line spans x in [0, 200]; vehicle crosses y=100 at x=300
    frames = [[(1, (300, 90), 0)], [(1, (300, 110), 0)]]
    assert run(horizontal(), frames).total == 0


def test_touching_the_line_then_crossing():
    frames = [[(1, (50, 90), 0)], [(1, (50, 100), 0)], [(1, (50, 100), 0)], [(1, (50, 105), 0)]]
    c = run(horizontal(), frames)
    assert c.total == 1 and c.crossings[0].direction == "in" and c.crossings[0].frame == 3


def test_new_track_on_far_side_is_not_counted():
    # a track that first appears already past the line has no "before" position
    assert run(horizontal(), [[(1, (50, 150), 0)], [(1, (50, 160), 0)]]).total == 0


def test_multiple_tracks_and_classes():
    frames = [
        [(1, (20, 90), 0), (2, (60, 120), 1), (3, (100, 50), 2)],
        [(1, (20, 105), 0), (2, (60, 95), 1), (3, (100, 60), 2)],
        [(1, (20, 120), 0), (2, (60, 80), 1), (3, (100, 101), 2)],
    ]
    c = run(horizontal(), frames)
    assert c.total == 3
    assert c.counts() == {"in": {0: 1, 2: 1}, "out": {1: 1}}


def test_track_gap_still_counts():
    # track disappears for a few frames (occlusion) and reappears across the line
    frames = [[(1, (50, 90), 0)], [], [], [(1, (50, 130), 0)]]
    assert run(horizontal(), frames).total == 1


def test_diagonal_and_vertical_lines():
    v = LineCounter((100, 0), (100, 200), "v")
    run(v, [[(1, (80, 50), 0)], [(1, (120, 50), 0)]])
    # p1->p2 points down the frame, so its right-hand side (positive) is to the left on screen:
    # moving left-to-right goes from the positive to the negative side -> "out"
    assert v.total == 1 and v.crossings[0].direction == "out"
    d = LineCounter((0, 0), (200, 200), "d")
    run(d, [[(1, (150, 50), 0)], [(1, (50, 150), 0)]])
    assert d.total == 1


def test_box_anchor_is_bottom_centre():
    assert box_anchor([10, 20, 30, 60]) == (20.0, 60.0)
    assert box_anchor(np.array([0, 0, 1, 1, 0.9])) == (0.5, 1.0)


def test_parse_line_fractions_and_pixels():
    assert parse_line("0,0.5,1,0.5", 1920, 1080) == ((0.0, 540.0), (1920.0, 540.0))
    assert parse_line("10, 20, 300, 400", 1920, 1080) == ((10.0, 20.0), (300.0, 400.0))
    with pytest.raises(ValueError):
        parse_line("1,2,3", 10, 10)


def test_class_voter_majority():
    v = TrackClassVoter()
    for c, s in [(4, 0.9), (5, 0.4), (4, 0.8), (5, 0.3), (5, 0.3)]:
        v.update(1, c, s)
    assert v.get(1) == 4  # 1.7 vs 1.0 confidence-weighted
    assert v.get(99) == -1
    v.update(2, 3)
    assert v.classes() == {1: 4, 2: 3}

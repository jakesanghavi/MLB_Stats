"""Tests for sample_ball: bidirectional interpolation across multi-frame gaps."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from read_play import sample_ball, BALL_INTERP_MAX_GAP


def _near(a, b, tol=1e-6):
    return all(abs(x - y) <= tol for x, y in zip(a, b))


def test_exact_sample():
    track = [(0.0, 1.0, 2.0, 3.0), (0.1, 2.0, 3.0, 4.0)]
    assert sample_ball(track, 0.0) == (1.0, 2.0, 3.0)
    assert sample_ball(track, 0.1) == (2.0, 3.0, 4.0)


def test_linear_midpoint_two_points():
    track = [(0.0, 0.0, 0.0, 0.0), (1.0, 10.0, 20.0, 30.0)]
    got = sample_ball(track, 0.5)
    assert abs(got[0] - 5.0) < 1e-6
    assert abs(got[2] - 15.0) < 1e-6
    # Y is ballistic, not the linear midpoint (10)
    assert got[1] > 10.0
    g, dt, y1, y2 = 32.174, 1.0, 0.0, 20.0
    vy = (y2 - y1) / dt + 0.5 * g * dt
    y_mid = y1 + vy * 0.5 - 0.5 * g * 0.25
    assert abs(got[1] - y_mid) < 1e-6


def test_packed_tuple_format():
    track = [(0.0, (0.0, 0.0, 0.0)), (1.0, (4.0, 6.0, 8.0))]
    got = sample_ball(track, 0.5)
    assert abs(got[0] - 2.0) < 1e-6
    assert abs(got[2] - 4.0) < 1e-6


def test_multi_frame_gap_uses_both_sides():
    # 5 missing interior samples; query in the hole must use past AND future.
    track = [(0.00, 0.0, 0.0, 0.0),
             (0.03, 3.0, 0.0, 0.0),
             (0.21, 21.0, 0.0, 0.0),
             (0.24, 24.0, 0.0, 0.0)]
    got = sample_ball(track, 0.12)
    assert got is not None
    # x should sit between the bracketing samples, not snap to either end
    assert 3.0 < got[0] < 21.0


def test_no_extrapolation():
    track = [(1.0, 1.0, 1.0, 1.0), (2.0, 2.0, 2.0, 2.0)]
    assert sample_ball(track, 0.5) is None
    assert sample_ball(track, 2.5) is None
    assert sample_ball([], 1.0) is None


def test_max_gap_not_bridged():
    track = [(0.0, 0.0, 0.0, 0.0), (BALL_INTERP_MAX_GAP + 0.5, 10.0, 10.0, 10.0)]
    assert sample_ball(track, 1.0) is None
    # same points with a looser cap still interpolate
    got = sample_ball(track, 1.0, max_gap=5.0)
    assert got is not None


def test_missing_apex_beats_linear():
    """Parabola y = 4t - t^2 is not gravity-scale; still, ballistic Y must
    not collapse to the chord between the last samples below the peak.
    """
    ts = [i * 0.25 for i in range(0, 17)]  # 0 .. 4
    full = [(t, t, 4 * t - t * t, 0.0) for t in ts]
    track = [p for p in full if p[2] <= 3.0 + 1e-9]
    t_peak = 2.0
    got = sample_ball(track, t_peak)
    assert got is not None
    lo = max(p for p in track if p[0] < t_peak)
    hi = min(p for p in track if p[0] > t_peak)
    linear_y = lo[2] + (hi[2] - lo[2]) * (t_peak - lo[0]) / (hi[0] - lo[0])
    assert got[1] > linear_y + 0.05, (got[1], linear_y)


def test_ballistic_fly_recovers_apex():
    """A ~100 ft fly with samples stripped above 70 ft — the hole is ~2.8s.
    Gravity + past/future endpoints should restore the apex.
    """
    g = 32.174
    y0, vy0 = 3.0, 80.0
    ts = [i * 0.05 for i in range(0, 120)]
    full = []
    for t in ts:
        y = y0 + vy0 * t - 0.5 * g * t * t
        if y < 0:
            break
        full.append((t, 40.0 * t, y, -10.0 * t))
    track = [p for p in full if p[2] <= 70.0]
    t_peak = vy0 / g
    true_y = y0 + vy0 * vy0 / (2.0 * g)
    got = sample_ball(track, t_peak)
    assert got is not None, "2.8s high-fly hole must be filled"
    assert abs(got[1] - true_y) < 2.0, (got[1], true_y)


def test_held_ball_does_not_pop_up():
    """A caught ball: fast in, slow across the hole, fast out. Height should
    not invent a 15 ft pop-up between the glove and the throw.
    """
    track = [
        (0.00, 50.0, 3.2, -56.0),
        (0.03, 52.0, 3.0, -58.0),
        (1.63, 59.0, 6.8, -66.5),
        (1.66, 57.0, 7.1, -68.6),
    ]
    got = sample_ball(track, 0.83)
    assert got is not None
    linear_y = 3.0 + (6.8 - 3.0) * (0.83 - 0.03) / (1.63 - 0.03)
    assert abs(got[1] - linear_y) < 1.0, got


def test_every_query_in_a_hole():
    track = [(0.0, 0.0, 10.0, 0.0), (0.5, 5.0, 10.0, 0.0)]
    for t in (0.1, 0.2, 0.3, 0.4):
        got = sample_ball(track, t)
        assert got is not None
        assert abs(got[0] - 10.0 * t) < 1e-6


def test_real_play_fills_sub_two_second_dropout():
    play = Path("/tmp/gd/play_823004_a6115b75-7846-3fb7-bbe4-995d0295512b")
    if not play.exists():
        return
    from read_play import PlayReader
    r = PlayReader(play)
    track = r.ball_track()
    t0 = r.frames[0]["time"]
    # 1.57s hole after the batted ball (rel t 21.20 .. 22.83)
    t = t0 + 22.0
    old = sample_ball(track, t, max_gap=0.2)
    new = sample_ball(track, t)
    assert old is None, "0.2s cap should still miss this hole"
    assert new is not None and all(math.isfinite(c) for c in new)


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print("ok", fn.__name__)
    print(f"{len(tests)} tests passed")

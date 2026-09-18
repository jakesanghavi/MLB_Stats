"""Tests for sample_ball: fly-ball gaps only, no time cap."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from read_play import sample_ball, BALL_INTERP_HEIGHT, BALL_DENSE_GAP


def test_exact_sample():
    track = [(0.0, 1.0, 2.0, 3.0), (0.1, 2.0, 3.0, 4.0)]
    assert sample_ball(track, 0.0) == (1.0, 2.0, 3.0)
    assert sample_ball(track, 0.1) == (2.0, 3.0, 4.0)


def test_dense_gap_always_filled():
    """Sampling-cadence holes interpolate even when the ball is low."""
    track = [(0.0, 0.0, 3.0, 0.0), (0.1, 4.0, 3.0, 8.0)]
    got = sample_ball(track, 0.05)
    assert got is not None
    assert abs(got[0] - 2.0) < 1e-6
    assert abs(got[2] - 4.0) < 1e-6


def test_packed_tuple_dense():
    track = [(0.0, (0.0, 6.0, 0.0)), (0.1, (4.0, 6.0, 8.0))]
    got = sample_ball(track, 0.05)
    assert got is not None
    assert abs(got[0] - 2.0) < 1e-6
    assert abs(got[2] - 4.0) < 1e-6


def test_low_abnormal_gap_skipped():
    """Grounder / catch / throw holes must not be filled."""
    track = [(0.0, 0.0, 3.0, 0.0), (1.6, 10.0, 6.0, -8.0)]
    assert sample_ball(track, 0.8) is None
    track24 = [(0.0, 0.0, BALL_INTERP_HEIGHT - 0.1, 0.0), (2.0, 10.0, 40.0, 0.0)]
    assert sample_ball(track24, 1.0) is None


def test_high_abnormal_gap_filled():
    """Last seen above 25 ft → fill, including long holes."""
    track = [(0.0, 0.0, 40.0, 0.0), (8.0, 80.0, 30.0, -40.0)]
    got = sample_ball(track, 4.0)
    assert got is not None
    assert 0.0 < got[0] < 80.0


def test_no_extrapolation():
    track = [(1.0, 1.0, 40.0, 1.0), (2.0, 2.0, 40.0, 2.0)]
    assert sample_ball(track, 0.5) is None
    assert sample_ball(track, 2.5) is None
    assert sample_ball([], 1.0) is None


def test_threshold_is_last_seen_height():
    """Gate is the sample before the hole, not the one after."""
    # last seen high, reappears low — still interpolate
    track = [(0.0, 0.0, 40.0, 0.0), (3.0, 30.0, 5.0, -20.0)]
    assert sample_ball(track, 1.5) is not None
    # last seen low, reappears high — do not interpolate
    track = [(0.0, 0.0, 5.0, 0.0), (3.0, 30.0, 40.0, -20.0)]
    assert sample_ball(track, 1.5) is None


def test_ballistic_fly_recovers_apex():
    """~100 ft fly with samples stripped above 70 ft (~2.8s hole)."""
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
    assert track[-1][2] >= BALL_INTERP_HEIGHT or max(p[2] for p in track) >= BALL_INTERP_HEIGHT
    t_peak = vy0 / g
    true_y = y0 + vy0 * vy0 / (2.0 * g)
    got = sample_ball(track, t_peak)
    assert got is not None, "high-fly hole must be filled"
    assert abs(got[1] - true_y) < 2.0, (got[1], true_y)


def test_long_high_fly_no_time_cap():
    """An 8s hole still fills when the last seen sample is a high fly."""
    g = 32.174
    y0, vy0 = 4.0, 90.0
    # two samples: going up through 40 ft, coming down through 30 ft, 8s apart
    t1, t2 = 0.5, 8.5
    y1 = y0 + vy0 * t1 - 0.5 * g * t1 * t1
    y2 = y0 + vy0 * t2 - 0.5 * g * t2 * t2
    assert y1 >= BALL_INTERP_HEIGHT
    track = [(t1, 20.0, y1, -5.0), (t2, 200.0, max(y2, 5.0), -80.0)]
    t_peak = vy0 / g
    got = sample_ball(track, t_peak)
    assert got is not None
    assert got[1] > y1, "apex should be above the last seen height"


def test_held_low_ball_not_interpolated():
    track = [
        (0.00, 50.0, 3.2, -56.0),
        (0.03, 52.0, 3.0, -58.0),
        (1.63, 59.0, 6.8, -66.5),
        (1.66, 57.0, 7.1, -68.6),
    ]
    assert sample_ball(track, 0.83) is None


def test_real_play_skips_low_dropout():
    play = Path("/tmp/gd/play_823004_a6115b75-7846-3fb7-bbe4-995d0295512b")
    if not play.exists():
        return
    from read_play import PlayReader
    r = PlayReader(play)
    track = r.ball_track()
    t0 = r.frames[0]["time"]
    # 1.57s hole after the batted ball — last seen ~3 ft, must stay empty
    t = t0 + 22.0
    assert sample_ball(track, t) is None
    # 6.3s between throws also low
    assert sample_ball(track, t0 + 26.0) is None
    # dense 30 fps holes during the pitch still fill
    assert sample_ball(track, t0 + 20.05) is not None


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print("ok", fn.__name__)
    print(f"{len(tests)} tests passed")

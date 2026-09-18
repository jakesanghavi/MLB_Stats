"""Tests for sample_ball: fly-ball gaps only, both ends >= 25 ft, no time cap."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from read_play import sample_ball, BALL_INTERP_HEIGHT


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
    track = [(0.0, 0.0, 3.0, 0.0), (1.6, 10.0, 6.0, -8.0)]
    assert sample_ball(track, 0.8) is None
    track24 = [(0.0, 0.0, BALL_INTERP_HEIGHT - 0.1, 0.0), (2.0, 10.0, 40.0, 0.0)]
    assert sample_ball(track24, 1.0) is None


def test_both_ends_must_be_high():
    """A high fly must not be joined to a replacement ball near the field."""
    # last seen high, next is a second ball at 3 ft — do not interpolate
    assert sample_ball([(0.0, -16.0, 81.0, -26.0), (4.4, -3.0, 3.4, -6.0)], 2.0) is None
    # last seen low, next high — do not interpolate
    assert sample_ball([(0.0, 0.0, 5.0, 0.0), (3.0, 30.0, 40.0, -20.0)], 1.5) is None
    # both ends high — fill, even if the hole is long
    got = sample_ball([(0.0, 0.0, 40.0, -20.0), (8.0, 80.0, 30.0, -80.0)], 4.0)
    assert got is not None
    assert abs(got[0] - 40.0) < 1e-6
    assert abs(got[2] - (-50.0)) < 1e-6  # field-Z is linear
    assert got[1] > 40.0  # ballistic apex above the chord


def test_no_extrapolation():
    track = [(1.0, 1.0, 40.0, 1.0), (2.0, 2.0, 40.0, 2.0)]
    assert sample_ball(track, 0.5) is None
    assert sample_ball(track, 2.5) is None
    assert sample_ball([], 1.0) is None


def test_ballistic_fly_recovers_apex():
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
    assert got is not None, "high-fly hole must be filled"
    assert abs(got[1] - true_y) < 2.0, (got[1], true_y)


def test_xz_linear_not_hermite():
    """Field X/Z must be the chord, not a cubic overshoot."""
    track = [
        (0.00, 0.0, 40.0, 0.0),
        (0.03, 1.0, 41.0, -1.0),
        (4.03, 81.0, 35.0, -81.0),
        (4.06, 82.0, 34.0, -82.0),
    ]
    got = sample_ball(track, 2.03)
    assert got is not None
    assert abs(got[0] - 41.0) < 0.2
    assert abs(got[2] - (-41.0)) < 0.2


def test_held_low_ball_not_interpolated():
    track = [
        (0.00, 50.0, 3.2, -56.0),
        (0.03, 52.0, 3.0, -58.0),
        (1.63, 59.0, 6.8, -66.5),
        (1.66, 57.0, 7.1, -68.6),
    ]
    assert sample_ball(track, 0.83) is None


def test_play_822849_fly_not_second_ball():
    play = Path("/tmp/gd/play_822849_8313f274-c733-325e-8df0-beaee0ddb6e1")
    if not play.exists():
        return
    from read_play import PlayReader
    r = PlayReader(play)
    track = r.ball_track()
    t0 = r.frames[0]["time"]
    fly = sample_ball(track, t0 + 23.40)
    assert fly is not None, "4.4s fly hole (both ends ~80-90 ft) must fill"
    # ballistic Y recovers the Statcast apex (~164.6 ft); linear field X/Z
    # stay on the chord between the last/next high samples
    assert 160.0 < fly[1] < 170.0, fly
    assert -110.0 < fly[0] < -15.0, fly
    assert -80.0 < fly[2] < -20.0, fly
    # after the catch the next ball is a replacement at ~3 ft — do not fill
    for rel in (26.80, 27.20, 27.50, 28.00, 28.40):
        assert sample_ball(track, t0 + rel) is None, rel


def test_real_play_skips_low_dropout():
    play = Path("/tmp/gd/play_823004_a6115b75-7846-3fb7-bbe4-995d0295512b")
    if not play.exists():
        return
    from read_play import PlayReader
    r = PlayReader(play)
    track = r.ball_track()
    t0 = r.frames[0]["time"]
    assert sample_ball(track, t0 + 22.0) is None
    assert sample_ball(track, t0 + 26.0) is None
    assert sample_ball(track, t0 + 20.05) is not None


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print("ok", fn.__name__)
    print(f"{len(tests)} tests passed")

"""play_window / pitch_release_time must follow the requested play GUID.

Mannequin clips bleed neighboring pitches. Taking the first action-0 in the
file is why a 16s 0-1 PA (gamePk 822845 / 9b398372…) windowed the called
strike instead of the GIDP, while later pitches with a ~20s+ gap still worked.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from read_play import PlayReader
from reconstruct3d import _contact_time, _pitch_release_time


def _frame(t, events=None, ball=None, polys=None):
    return {
        "time": t,
        "num": 0,
        "ball": ball,
        "gameEvents": events or [],
        "actorPoses": [],
        "ballPolynomials": polys or [],
    }


def _reader(frames, play_id):
    r = PlayReader.__new__(PlayReader)
    r.dir = Path(".")
    r.info = {"gamePk": 822845, "playId": play_id}
    r.metadata = {}
    r.manifest = {}
    r.labels = {}
    r.uniforms = {}
    r.bone_id_map = None
    r.frames = frames
    r.chunks_read = 0
    return r


def _two_pitch_clip(p1="fa811c95-c5a3-3b8d-8116-463820e132a9",
                    p2="9b398372-272c-3787-bb1a-aebacf8eedab"):
    """Synthetic 822845-style clip: 16s between releases, one liveAction span."""
    frames = [
        _frame(0.0),
        _frame(2.0, [{"time": 2.0, "dataType": 4, "data": {"mode": True}}]),
        _frame(4.0, [{"time": 4.0, "dataType": 7,
                      "data": {"action": 0, "index": 0, "playId": p1}}],
               ball={"x": 0.0, "y": 6.0, "z": -50.0},
               polys=[{"dataType": 5, "id": p1}]),
        _frame(4.4, ball={"x": 0.1, "y": 3.0, "z": -4.0}),
        _frame(7.0, ball={"x": 2.0, "y": 5.0, "z": -40.0}),
        _frame(20.0, [{"time": 20.0, "dataType": 7,
                       "data": {"action": 0, "index": 1, "playId": p2}}],
               ball={"x": 0.0, "y": 6.0, "z": -50.0},
               polys=[{"dataType": 5, "id": p2}]),
        _frame(20.4, [{"time": 20.4, "dataType": 12, "data": {}}],
               ball={"x": 0.4, "y": 2.4, "z": -2.4},
               polys=[{"dataType": 2, "id": p2}]),
        _frame(21.5, ball={"x": -20.0, "y": 0.6, "z": -124.0}),
        _frame(25.0, [{"time": 25.0, "dataType": 4, "data": {"mode": False}}]),
        _frame(30.0),
    ]
    return frames, p1, p2


def test_matches_requested_guid_not_first_release():
    frames, p1, p2 = _two_pitch_clip()
    r = _reader(frames, p2)
    assert r.pitch_release_time() == 20.0
    assert _pitch_release_time(r) == 20.0
    w0, w1 = r.play_window()
    assert w0 > 4.4, w0  # must not start on the called-strike pitch
    assert w0 <= 18.0, w0  # 2s lead before the GIDP
    assert w1 >= 21.5, w1
    assert w1 <= 27.1, w1


def test_first_pitch_of_same_clip_still_windows_itself():
    frames, p1, p2 = _two_pitch_clip()
    r = _reader(frames, p1)
    assert r.pitch_release_time() == 4.0
    w0, w1 = r.play_window()
    assert w0 <= 4.0
    assert w1 < 20.0, w1  # ball gap + live trim stay on pitch 1


def test_unmatched_guid_falls_back_to_first_release():
    frames, p1, p2 = _two_pitch_clip()
    r = _reader(frames, "not-in-clip")
    assert r.pitch_release_time() == 4.0


def test_single_release_unchanged():
    pid = "8313f274-c733-325e-8df0-beaee0ddb6e1"
    frames = [
        _frame(0.0, [{"time": 0.7, "dataType": 7,
                      "data": {"action": 1, "playId": "prev-pitch"}}]),
        _frame(2.0, [{"time": 2.0, "dataType": 4, "data": {"mode": True}}]),
        _frame(18.0, [{"time": 18.0, "dataType": 4, "data": {"mode": False}}]),
        _frame(20.0, [{"time": 20.0, "dataType": 7,
                       "data": {"action": 0, "playId": pid}}],
               ball={"x": 0.0, "y": 6.0, "z": -50.0}),
        _frame(21.5, [{"time": 21.5, "dataType": 4, "data": {"mode": True}}],
               ball={"x": 1.0, "y": 3.0, "z": -10.0}),
        _frame(24.0, ball={"x": -8.0, "y": 4.0, "z": -40.0}),
        _frame(30.0, [{"time": 30.0, "dataType": 4, "data": {"mode": False}}]),
    ]
    r = _reader(frames, pid)
    assert r.pitch_release_time() == 20.0
    w0, w1 = r.play_window()
    assert 17.5 <= w0 <= 20.0, w0
    assert w1 >= 24.0


def test_contact_follows_requested_pitch():
    frames, p1, p2 = _two_pitch_clip()
    # prior-pitch foul impact must not win
    frames.insert(4, _frame(4.6, [{"time": 4.6, "dataType": 12, "data": {}}],
                            ball={"x": 0.2, "y": 2.5, "z": -3.0}))
    r = _reader(frames, p2)
    assert _contact_time(r) == 20.4


def test_real_822845_gidp_if_present():
    d = Path("/tmp/gd/play_822845_9b398372-272c-3787-bb1a-aebacf8eedab")
    if not d.exists():
        print("skip real 822845 (no play dir)")
        return
    r = PlayReader(d)
    t0 = r.frames[0]["time"]
    t_pitch = r.pitch_release_time()
    assert r.info.get("playId") == "9b398372-272c-3787-bb1a-aebacf8eedab"
    assert abs((t_pitch - t0) - 20.018) < 0.05, t_pitch - t0
    w0, w1 = r.play_window()
    assert (w0 - t0) > 10.0, w0 - t0
    assert (w0 - t0) < 20.1, w0 - t0
    assert (w1 - t0) > 21.5, w1 - t0
    print("ok real 822845 window", round(w0 - t0, 3), "..", round(w1 - t0, 3))


if __name__ == "__main__":
    test_matches_requested_guid_not_first_release()
    test_first_pitch_of_same_clip_still_windows_itself()
    test_unmatched_guid_falls_back_to_first_release()
    test_single_release_unchanged()
    test_contact_follows_requested_pitch()
    test_real_822845_gidp_if_present()
    print("ok")

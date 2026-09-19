"""Head/gaze coverage and POV look-target smoke tests."""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from reconstruct3d import _slerp_quat, _sample_pose, _sample_bat
from views import (
    EASY_BLEND_S, HEAD_POSE, HEAD_POSES, easy_event_weight, look_from_eye,
    look_target, look_pose, smart_forward, smooth_head_series,
)


def test_slerp_90():
    # 90 deg about Y: (0, sin(a/2), 0, cos(a/2))
    q1 = [0.0, 0.0, 0.0, 1.0]
    q2 = [0.0, math.sin(math.pi / 4), 0.0, math.cos(math.pi / 4)]
    mid = _slerp_quat(q1, q2, 0.5)
    expect = [0.0, math.sin(math.pi / 8), 0.0, math.cos(math.pi / 8)]
    assert abs(abs(np.dot(mid, expect)) - 1.0) < 1e-6, (mid, expect)
    print("ok slerp")


def test_sample_pose_slerps():
    q1 = [0.0, 0.0, 0.0, 1.0]
    q2 = [0.0, math.sin(math.pi / 4), 0.0, math.cos(math.pi / 4)]
    track = [
        (0.0, {"rootPos": {"x": 0, "y": 0, "z": 0},
               "jointRotations": {"joint_Neck": q1}, "scale": 1.0}),
        (1.0, {"rootPos": {"x": 10, "y": 0, "z": 0},
               "jointRotations": {"joint_Neck": q2}, "scale": 1.0}),
    ]
    p = _sample_pose(track, 0.5)
    assert abs(p["rootPos"]["x"] - 5.0) < 1e-9
    mid = p["jointRotations"]["joint_Neck"]
    expect = [0.0, math.sin(math.pi / 8), 0.0, math.cos(math.pi / 8)]
    assert abs(abs(np.dot(mid, expect)) - 1.0) < 1e-6, mid
    print("ok sample_pose slerp")


def test_look_target():
    assert look_target("P", None) == (0.0, 2.5, 0.0)
    assert look_target("Batter", None) == (0.0, 5.0, -60.5)
    assert look_target("P", (3.0, 8.0, -20.0)) == (3.0, 8.0, -20.0)
    eye = (0.0, 6.0, -60.0)
    hp = look_from_eye(eye, (0.0, 2.5, 0.0))
    pos, fwd, up = hp
    assert fwd[2] > 0.9, fwd  # toward the plate (+Z)
    assert abs(np.dot(fwd, up)) < 1e-6
    print("ok look_target")


def test_smooth_kills_spike():
    heads = []
    for i in range(20):
        fwd = [0.0, 0.0, 1.0]
        if i == 10:
            fwd = [1.0, 0.0, 0.0]
        heads.append([0.0, 6.0, 0.0, *fwd, 0.0, 1.0, 0.0])
    sm = smooth_head_series(heads, fps=20.0)
    spike = sm[10][3:6]
    # should not snap all the way to +X
    assert spike[0] < 0.55, spike
    assert spike[2] > 0.7, spike
    print("ok smooth", [round(v, 3) for v in spike])


def test_head_pose_modes():
    eye = np.array([0.0, 6.0, -60.0])
    neck = np.array([0.0, 0.0, 1.0])  # facing plate
    ball = (0.0, 8.0, -20.0)
    neck_look = look_pose("FOLLOW_NECK", eye, neck, (0, 1, 0), "P", ball)
    assert neck_look[1][2] > 0.95, neck_look[1]
    always = look_pose("ALWAYS_BALL", eye, np.array([1.0, 0.0, 0.0]), (0, 1, 0), "P", ball)
    to_ball = np.array(ball) - eye
    to_ball = to_ball / np.linalg.norm(to_ball)
    assert abs(np.dot(always[1], to_ball)) > 0.99, always[1]
    # ball in front of neck -> SMART uses ball
    smart = look_pose("SMART_VISION", eye, neck, (0, 1, 0), "P", ball)
    assert abs(np.dot(smart[1], to_ball)) > 0.99, smart[1]
    # ball behind, plate in front -> SMART keeps plate, ALWAYS still tracks ball
    behind = (0.0, 6.0, -200.0)
    smart_b = smart_forward(eye, neck, "P", behind)
    always_b = look_pose("ALWAYS_BALL", eye, neck, (0, 1, 0), "P", behind)
    to_behind = np.array(behind) - eye
    to_behind = to_behind / np.linalg.norm(to_behind)
    assert smart_b[2] > 0.7, smart_b  # still facing +Z / plate-ish
    assert np.dot(always_b[1], to_behind) > 0.95, always_b[1]
    # clamp: only a back target, turn at most 80 deg
    clamped = smart_forward(eye, neck, "P", behind, cone_deg=80.0)
    # neck +Z vs behind is ~180; clamp 80 leaves a +Z component
    assert clamped[2] > 0.0, clamped
    print("ok head_pose modes")


def test_easy_vision_flips_at_contact():
    assert HEAD_POSE == "EASY_VISION"
    assert "EASY_VISION" in HEAD_POSES
    eye = np.array([0.0, 6.0, -60.0])
    neck = np.array([0.0, 0.0, 1.0])
    ball = (0.0, 8.0, -20.0)
    before = look_pose("EASY_VISION", eye, neck, (0, 1, 0), "P", ball, contacted=False)
    to_ball = np.array(ball) - eye
    to_ball = to_ball / np.linalg.norm(to_ball)
    assert abs(np.dot(before[1], to_ball)) > 0.99, before[1]
    after = look_pose("EASY_VISION", eye, neck, (0, 1, 0), "P", ball, contacted=True)
    assert after[1][2] > 0.95, after[1]
    print("ok easy_vision contact flip")


def test_easy_vision_blends_only_at_events():
    assert easy_event_weight(1.0, 2.0) == 0.0
    assert easy_event_weight(2.0 + EASY_BLEND_S, 2.0) == 1.0
    mid = easy_event_weight(2.0 + 0.5 * EASY_BLEND_S, 2.0)
    assert 0.4 < mid < 0.6, mid
    eye = np.array([0.0, 6.0, -60.0])
    neck = np.array([0.0, 0.0, 1.0])
    fly = (40.0, 8.0, -20.0)  # off toward 1B so ball ≠ neck
    to_fly = np.array(fly) - eye
    to_fly = to_fly / np.linalg.norm(to_fly)
    far_before = look_pose(
        "EASY_VISION", eye, neck, (0, 1, 0), "P", fly,
        t=1.70, t_release=2.0, t_contact=2.4,
    )
    assert abs(np.dot(far_before[1], to_fly)) > 0.99, far_before[1]
    far_after = look_pose(
        "EASY_VISION", eye, neck, (0, 1, 0), "P", fly,
        t=2.4 + EASY_BLEND_S + 0.05, t_release=2.0, t_contact=2.4,
    )
    assert far_after[1][2] > 0.95, far_after[1]
    at_contact = look_pose(
        "EASY_VISION", eye, neck, (0, 1, 0), "P", fly,
        t=2.4 + 0.5 * EASY_BLEND_S, t_release=2.0, t_contact=2.4,
    )
    assert 0.15 < float(np.dot(at_contact[1], to_fly)) < 0.95, at_contact[1]
    assert 0.15 < at_contact[1][2] < 0.95, at_contact[1]
    held = (0.0, 5.0, -59.0)
    pitch = (0.0, 6.0, 0.0)
    at_release = look_pose(
        "EASY_VISION", eye, neck, (0, 1, 0), "P", pitch,
        t=2.0 + 0.5 * EASY_BLEND_S, t_release=2.0, t_contact=2.4,
        ball_xyz_prerelease=held,
    )
    to_held = np.array(held) - eye
    to_held = to_held / np.linalg.norm(to_held)
    to_pitch = np.array(pitch) - eye
    to_pitch = to_pitch / np.linalg.norm(to_pitch)
    assert 0.15 < float(np.dot(at_release[1], to_pitch)) < 0.99, at_release[1]
    assert float(np.dot(at_release[1], to_held)) > 0.15, at_release[1]
    print("ok easy_vision event blends")


def test_sample_bat_lerps():
    track = [
        (0.0, (0.0, 1.0, 0.0), (0.0, 3.0, 0.0)),
        (0.2, (2.0, 1.0, 0.0), (2.0, 3.0, 0.0)),
    ]
    handle, head = _sample_bat(track, 0.1)
    assert abs(handle[0] - 1.0) < 1e-9, handle
    assert abs(head[0] - 1.0) < 1e-9, head
    assert _sample_bat(track, 2.0) is None
    # nearest-neighbor would have snapped to an endpoint
    assert abs(handle[0] - 0.0) > 0.4
    print("ok sample_bat lerp")


def test_wire_has_no_head(play_dir=None):
    d = Path(play_dir or "/tmp/gd/play_822849_8313f274-c733-325e-8df0-beaee0ddb6e1")
    if not d.exists():
        print("skip wire-head coverage (no play dir)")
        return
    from read_play import PlayReader
    reader = PlayReader(d)
    n = 0
    heads = 0
    for f in reader.frames:
        for a in f.get("actorPoses") or []:
            jr = a.get("jointRotations") or {}
            if not jr:
                continue
            n += 1
            if "joint_Head" in jr or "joint_EyeLT" in jr:
                heads += 1
    assert n > 0
    assert heads == 0, heads
    print(f"ok no Head/eye quats in {n} actor poses")


if __name__ == "__main__":
    test_slerp_90()
    test_sample_pose_slerps()
    test_look_target()
    test_smooth_kills_spike()
    test_head_pose_modes()
    test_easy_vision_flips_at_contact()
    test_easy_vision_blends_only_at_events()
    test_sample_bat_lerps()
    test_wire_has_no_head()
    print("ok")

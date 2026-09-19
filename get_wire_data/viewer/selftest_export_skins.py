"""Packed poses and home/away sides for the player-mesh wrap."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_play import _fill_missing_sides, _pack_pose


def test_pack_pose():
    bones = ["joint_Pelvis", "joint_Neck"]
    pose = {
        "rootPos": {"x": 1.25, "y": 3.4, "z": -2.5},
        "scale": 1.0,
        "jointRotations": {
            "joint_Pelvis": [0.0, 0.0, 0.0, 1.0],
            "joint_Neck": [0.1, 0.2, 0.3, 0.4],
        },
    }
    packed = _pack_pose(pose, bones)
    assert packed[:4] == [1.25, 3.4, -2.5, 1.0]
    assert packed[4:8] == [0.0, 0.0, 0.0, 1.0]
    assert packed[8:12] == [0.1, 0.2, 0.3, 0.4]
    missing = _pack_pose({"rootPos": {"x": 0, "y": 0, "z": 0}}, bones)
    assert missing[4:8] == [0.0, 0.0, 0.0, 0.0]
    print("ok pack pose")


def test_fill_missing_sides():
    actors = [
        {"type": "pitcher", "side": "home"},
        {"type": "batter", "side": None},
        {"type": "fielder", "side": None},
        {"type": "umpire", "side": None},
        {"type": "coach", "side": None},
    ]
    _fill_missing_sides(actors)
    assert actors[1]["side"] == "away"
    assert actors[2]["side"] == "home"
    assert actors[3]["side"] is None
    assert actors[4]["side"] == "away"
    print("ok fill missing sides")


if __name__ == "__main__":
    test_pack_pose()
    test_fill_missing_sides()
    print("ok")

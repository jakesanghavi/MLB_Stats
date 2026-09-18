"""Static contracts for the viewer JS/HTML (no browser)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
JS = (ROOT / "viewer.js").read_text()
HTML = (ROOT / "index.html").read_text()


def test_code_only_head_pose():
    assert 'let headPose = "EASY_VISION"' in JS
    assert "EASY_VISION" in JS
    assert 'data-head=' not in HTML
    assert "Follow neck" not in HTML
    assert "head-pose" not in HTML
    print("ok code-only EASY_VISION")


def test_no_head_spheres():
    assert "headMarkers" not in JS
    assert "SphereGeometry(0.28" not in JS
    print("ok no head spheres")


def test_thicken_knobs():
    for name in ("LIMB_THICKEN", "TRAIL_THICKEN", "BALL_THICKEN"):
        assert f"const {name}" in JS, name
    print("ok thicken knobs")


def test_smooth_playhead():
    assert "lerpXyz(play.ball, t, true)" in JS
    assert "catmull3" in JS
    assert "poseBat(lerpBat(t))" in JS
    assert "Math.round(u * (play.times.length - 1))" not in JS
    print("ok smooth ball/bat")


def test_bat_mesh():
    assert "data/bat.glb" in JS
    assert "BAT_MODEL_LEN_FT" in JS
    print("ok bat.glb")


if __name__ == "__main__":
    test_code_only_head_pose()
    test_no_head_spheres()
    test_thicken_knobs()
    test_smooth_playhead()
    test_bat_mesh()
    print("ok")

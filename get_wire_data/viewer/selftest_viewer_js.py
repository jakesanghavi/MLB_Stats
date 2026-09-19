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


def test_trail_flag_and_save_video():
    assert "const SHOW_BALL_TRAIL = true" in JS
    assert "SHOW_BALL_TRAIL && trail.length >= 6" in JS
    assert "const EASY_BLEND_S" in JS
    assert "function saveVideo" in JS
    assert "SAVE_VIDEO_FPS = 30" in JS
    assert "SAVE_VIDEO_BITRATE = 16_000_000" in JS
    assert 'id="btn-save"' in HTML
    assert "Save video" in HTML
    print("ok trail flag + save video")


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


def test_player_mesh_flag():
    assert "const SHOW_PLAYER_MESH = true" in JS
    assert "SkeletonUtils" in JS
    assert "loadPlayerMesh" in JS
    assert "poseSkin" in JS
    assert "player mesh failed, using stick figures" in JS
    assert "false = stick figures, no jersey/head/hat/glove assets" in JS
    print("ok player mesh flag")


if __name__ == "__main__":
    test_code_only_head_pose()
    test_no_head_spheres()
    test_thicken_knobs()
    test_trail_flag_and_save_video()
    test_smooth_playhead()
    test_bat_mesh()
    test_player_mesh_flag()
    print("ok")

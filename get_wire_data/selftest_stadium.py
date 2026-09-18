"""Tests for stadium face keeping (largest-area, not quadric)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stadium import (
    STADIUM_FACE_TARGET, STADIUM_CACHE_METHOD, FIELD_CACHE_METHOD,
    _keep_largest_faces, load_park_meshes, BALLPARKS,
)


def test_keep_largest_faces_picks_big_tris():
    verts = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0],
        [0, 0, 0], [10, 0, 0], [0, 10, 0],
        [0, 0, 0], [0.1, 0, 0], [0, 0.1, 0],
    ], float)
    faces = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], int)
    v, f = _keep_largest_faces(verts, faces, 1)
    a, b, c = v[f[0, 0]], v[f[0, 1]], v[f[0, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a))
    assert abs(area - 50.0) < 1e-6, area


def test_keep_largest_under_budget_unchanged():
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float)
    faces = np.array([[0, 1, 2]], int)
    v, f = _keep_largest_faces(verts, faces, 10)
    assert np.allclose(v, verts) and np.array_equal(f, faces)


def test_tex_stadium_cache_is_largest_area():
    glb = BALLPARKS / "5325_TEX.glb"
    if not glb.exists():
        return
    park = load_park_meshes(glb, want_field=False, want_stadium=True)
    verts, faces = park["stadium"]
    assert len(faces) == STADIUM_FACE_TARGET, len(faces)
    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum()
    # quadric-to-16k kept ~0.13e6 ft²; largest 40k keeps well over 1e6
    assert area > 1.0e6, area
    cache = np.load(glb.with_suffix(".stadium.npz"))
    assert str(cache["method"]) == STADIUM_CACHE_METHOD
    assert int(cache["target"]) == STADIUM_FACE_TARGET


def test_field_cache_method_tag():
    glb = BALLPARKS / "5325_TEX.glb"
    if not glb.exists():
        return
    park = load_park_meshes(glb, want_field=True, want_stadium=False)
    assert "field" in park and len(park["field"][1]) > 1000
    cache = np.load(glb.with_suffix(".field.npz"))
    if "method" in cache.files:
        assert str(cache["method"]) == FIELD_CACHE_METHOD


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print("ok", fn.__name__)
    print(f"{len(tests)} tests passed")

"""Resolve, decode, and downsample a Gameday 3D ballpark .glb.

The stadium files are Draco-compressed and far too dense for matplotlib, so we
decode with DracoPy, apply the node TRS (scale 0.01 + 90° X on current parks),
then scale meters→feet (Gameday ``FI = 3.28084``).

The field is a simple surface, so quadric-decimate it. The stadium is tens of
thousands of disconnected scraps (seats, rails); quadric simplification
destroys it. We keep the largest-area faces instead. Cached as
``*.field.npz`` / ``*.stadium.npz`` next to the .glb.
"""
import os
import time
from pathlib import Path

import numpy as np

from glb import load_glb_nodes

ASSETS = Path(__file__).resolve().parent / "assets"
BALLPARKS = ASSETS / "ballparks"
ASSET_BASE = os.environ.get(
    "FV_ASSET_BASE",
    "https://fv-assets.mlb.com/v/58242e42c28513752c1cb5776bdf0da7f0679d5e",
)

# venueId -> Gameday park abbreviation (VA table in gd.@bvg_poser.min.js).
# Home-team abbreviation from boxscore is used first; this is the fallback.
VENUE_ABBR = {
    3: "BOS", 15: "ARI", 22: "LAD", 680: "SEA", 2395: "SF", 2529: "ATH",
    2680: "SD", 2681: "PHI", 2889: "STL", 3289: "NYM", 3309: "WSH",
    3312: "MIN", 3313: "NYY", 4705: "ATL", 5325: "TEX",
}

FIELD_FACE_TARGET = 6000
STADIUM_FACE_TARGET = 40000
FIELD_CACHE_METHOD = "decimate"
STADIUM_CACHE_METHOD = "largest-area"
# Gameday viewer: after GLTFLoader applies node TRS the mesh is in meters;
# they then do scene.scale.set(FI, FI, FI). Tracking data is already feet.
M_TO_FT = 3.28084


def _home_abbr(reader):
    try:
        return reader.metadata["boxscore"]["teams"]["home"]["team"]["abbreviation"]
    except (KeyError, TypeError):
        return None


def _venue_id(reader):
    return reader.metadata.get("venueId")


def _candidates(venue_id, abbr):
    names = []
    if venue_id is not None and abbr:
        names.append(f"{venue_id}_{abbr}.glb")
    if venue_id is not None:
        names.append(f"{venue_id}_*.glb")
    return names


def find_ballpark_glb(reader, explicit=None):
    """Locate ``{venueId}_{ABBR}.glb``, downloading from the CDN if needed."""
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
        raise FileNotFoundError(f"ballpark glb not found: {p}")

    venue_id = _venue_id(reader)
    abbr = _home_abbr(reader) or VENUE_ABBR.get(venue_id)
    BALLPARKS.mkdir(parents=True, exist_ok=True)

    if venue_id is not None and abbr:
        local = BALLPARKS / f"{venue_id}_{abbr}.glb"
        if local.exists():
            return local

    if venue_id is not None:
        matches = sorted(BALLPARKS.glob(f"{venue_id}_*.glb"))
        if matches:
            return matches[0]

    if venue_id is None or not abbr:
        raise FileNotFoundError(
            "cannot resolve ballpark file (need metadata venueId + home abbreviation)"
        )

    url = f"{ASSET_BASE}/models/ballparks/{venue_id}_{abbr}.glb"
    dest = BALLPARKS / f"{venue_id}_{abbr}.glb"
    print(f"downloading ballpark {url}")
    try:
        import requests
        r = requests.get(url, timeout=60)
        r.raise_for_status()
    except Exception as e:
        raise FileNotFoundError(
            f"failed to download {url} ({e}). Place the file at {dest}."
        ) from e
    dest.write_bytes(r.content)
    print(f"saved {dest} ({len(r.content)} bytes)")
    return dest


def _decimate(verts, faces, target):
    if len(faces) <= target:
        return verts, faces
    import trimesh
    mesh = trimesh.Trimesh(verts, faces, process=False)
    simple = mesh.simplify_quadric_decimation(face_count=int(target))
    return np.asarray(simple.vertices, float), np.asarray(simple.faces, int)


def _keep_largest_faces(verts, faces, target):
    """Keep the ``target`` largest-area triangles; drop tiny seat/rail scraps."""
    n = len(faces)
    if n <= target:
        return verts, faces
    a = verts[faces[:, 0]]
    b = verts[faces[:, 1]]
    c = verts[faces[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    keep = np.argpartition(area, -int(target))[-int(target):]
    faces = np.ascontiguousarray(faces[keep])
    used, inv = np.unique(faces, return_inverse=True)
    return verts[used], inv.reshape(-1, 3)


def _cache_path(glb_path, kind):
    return Path(glb_path).with_suffix(f".{kind}.npz")


def _load_cached(path, target, method=None):
    if not path.exists():
        return None
    data = np.load(path)
    try:
        if int(data["target"]) != int(target):
            return None
        if abs(float(data["m_to_ft"]) - M_TO_FT) > 1e-4:
            return None
        if method is not None:
            if "method" in data.files:
                if str(data["method"]) != str(method):
                    return None
            elif method != FIELD_CACHE_METHOD:
                return None  # old stadium caches have no method tag
    except (KeyError, ValueError, TypeError):
        return None  # old meter-scale cache
    return data["verts"], data["faces"]


def _save_cached(path, verts, faces, target, method):
    np.savez_compressed(path, verts=verts, faces=faces,
                        target=np.array(target), m_to_ft=np.array(M_TO_FT),
                        method=np.array(method))


def _pick_nodes(nodes, kind):
    suffix = "_Field" if kind == "field" else "_Stadium"
    exact = [k for k in nodes if k.endswith(suffix)]
    if exact:
        return exact
    # some parks use un-prefixed names
    alt = "Field" if kind == "field" else "Stadium"
    return [k for k in nodes if k == alt]


def _merge(nodes, names):
    vs, fs, n = [], [], 0
    for name in names:
        v, f = nodes[name]
        vs.append(v)
        fs.append(f + n)
        n += len(v)
    return np.vstack(vs), np.vstack(fs)


def load_park_meshes(glb_path, want_field=True, want_stadium=True,
                     field_faces=FIELD_FACE_TARGET, stadium_faces=STADIUM_FACE_TARGET):
    """Return dict with optional 'field' / 'stadium' -> (verts, faces) in world feet."""
    glb_path = Path(glb_path)
    out = {}
    needed = []
    if want_field:
        cached = _load_cached(_cache_path(glb_path, "field"), field_faces,
                              method=FIELD_CACHE_METHOD)
        if cached is not None:
            out["field"] = cached
        else:
            needed.append("field")
    if want_stadium:
        cached = _load_cached(_cache_path(glb_path, "stadium"), stadium_faces,
                              method=STADIUM_CACHE_METHOD)
        if cached is not None:
            out["stadium"] = cached
        else:
            needed.append("stadium")
    if not needed:
        return out

    t_dec = time.perf_counter()
    print(f"decoding ballpark {glb_path.name} ...")
    nodes = load_glb_nodes(glb_path)
    print("  nodes:", ", ".join(f"{k} ({len(v[1])} tris)" for k, v in nodes.items()))

    if "field" in needed:
        names = _pick_nodes(nodes, "field")
        if not names:
            raise KeyError(f"no *_Field node in {glb_path} (have {list(nodes)})")
        t = time.perf_counter()
        verts, faces = _merge(nodes, names)
        n0 = len(faces)
        verts, faces = _decimate(verts, faces, field_faces)
        verts = verts * M_TO_FT
        _save_cached(_cache_path(glb_path, "field"), verts, faces, field_faces,
                     FIELD_CACHE_METHOD)
        out["field"] = (verts, faces)
        print(f"  field -> {len(faces)} tris (from {n0})  "
              f"z[{verts[:, 2].min():.1f},{verts[:, 2].max():.1f}] ft  "
              f"({time.perf_counter() - t:.2f}s)")
    if "stadium" in needed:
        names = _pick_nodes(nodes, "stadium")
        if not names:
            raise KeyError(f"no *_Stadium node in {glb_path} (have {list(nodes)})")
        t = time.perf_counter()
        verts, faces = _merge(nodes, names)
        n0 = len(faces)
        verts, faces = _keep_largest_faces(verts, faces, stadium_faces)
        verts = verts * M_TO_FT
        _save_cached(_cache_path(glb_path, "stadium"), verts, faces, stadium_faces,
                     STADIUM_CACHE_METHOD)
        out["stadium"] = (verts, faces)
        print(f"  stadium -> {len(faces)} tris (largest of {n0})  "
              f"z[{verts[:, 2].min():.1f},{verts[:, 2].max():.1f}] ft  "
              f"({time.perf_counter() - t:.2f}s)")
    print(f"  ballpark decode done ({time.perf_counter() - t_dec:.2f}s)")
    return out

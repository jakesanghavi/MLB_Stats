"""Load meshes from a binary glTF (.glb).

``load_glb_mesh`` reads a single uncompressed primitive (used for bat.glb).
``load_glb_nodes`` walks every node, applies that node's TRS, and decodes
``KHR_draco_mesh_compression`` primitives via DracoPy when present — that is
what the ballpark files need. See STADIUM_NOTES.md.
"""
import json
import struct
from pathlib import Path

import numpy as np

_COMP = {5120: ("b", 1), 5121: ("B", 1), 5122: ("h", 2), 5123: ("H", 2),
         5125: ("I", 4), 5126: ("f", 4)}
_NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def _read_accessor(gltf, bin_data, idx):
    acc = gltf["accessors"][idx]
    bv = gltf["bufferViews"][acc["bufferView"]]
    comp, size = _COMP[acc["componentType"]]
    n = _NCOMP[acc["type"]]
    base = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
    stride = bv.get("byteStride") or (size * n)
    out = np.empty((acc["count"], n), dtype=np.float64)
    for i in range(acc["count"]):
        off = base + i * stride
        out[i] = struct.unpack_from("<" + comp * n, bin_data, off)
    return out[:, 0] if n == 1 else out


def _parse_glb(path):
    d = Path(path).read_bytes()
    assert d[:4] == b"glTF", "not a GLB"
    clen, = struct.unpack_from("<I", d, 12)
    gltf = json.loads(d[20:20 + clen])
    bin_start = 20 + clen
    blen, _btype = struct.unpack_from("<II", d, bin_start)
    bin_data = d[bin_start + 8: bin_start + 8 + blen]
    return gltf, bin_data


def _quat_mat(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], float)


def _apply_trs(verts, node):
    s = np.array(node.get("scale", [1.0, 1.0, 1.0]), float)
    R = _quat_mat(node.get("rotation", [0.0, 0.0, 0.0, 1.0]))
    t = np.array(node.get("translation", [0.0, 0.0, 0.0]), float)
    return (R @ (verts * s).T).T + t


def _decode_primitive(gltf, bin_data, prim):
    """Return (verts Nx3, faces Mx3) for one primitive, Draco or raw."""
    ext = (prim.get("extensions") or {}).get("KHR_draco_mesh_compression")
    if ext is not None:
        try:
            import DracoPy
        except ImportError as e:
            raise SystemExit(
                "Draco-compressed mesh (ballpark) needs DracoPy: pip install DracoPy"
            ) from e
        bv = gltf["bufferViews"][ext["bufferView"]]
        off = bv.get("byteOffset", 0)
        blob = bin_data[off: off + bv["byteLength"]]
        dec = DracoPy.decode(blob)
        verts = np.asarray(dec.points, float).reshape(-1, 3)
        faces = np.asarray(dec.faces, int)
        if faces.ndim == 1:
            faces = faces.reshape(-1, 3)
        return verts, faces
    verts = _read_accessor(gltf, bin_data, prim["attributes"]["POSITION"])
    idx = _read_accessor(gltf, bin_data, prim["indices"]).astype(int)
    return verts, idx.reshape(-1, 3)


def load_glb_mesh(path, mesh_index=0, prim_index=0):
    """Return (vertices Nx3 float, faces Mx3 int) of a plain-glTF .glb mesh."""
    gltf, bin_data = _parse_glb(path)
    prim = gltf["meshes"][mesh_index]["primitives"][prim_index]
    return _decode_primitive(gltf, bin_data, prim)


def load_glb_nodes(path):
    """Return {node_name: (verts Nx3, faces Mx3)} with each node's TRS applied.

    Multiple primitives on one node are merged. Used for ballpark files.
    """
    gltf, bin_data = _parse_glb(path)
    out = {}
    for node in gltf.get("nodes", []):
        mi = node.get("mesh")
        if mi is None:
            continue
        name = node.get("name") or f"mesh_{mi}"
        vs, fs, n = [], [], 0
        for prim in gltf["meshes"][mi]["primitives"]:
            v, f = _decode_primitive(gltf, bin_data, prim)
            vs.append(v)
            fs.append(f + n)
            n += len(v)
        verts = _apply_trs(np.vstack(vs), node)
        faces = np.vstack(fs)
        out[name] = (verts, faces)
    return out

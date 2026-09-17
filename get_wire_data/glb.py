"""Minimal loader for the mesh of a plain (non-Draco) binary glTF (.glb).

Only what the 3D reconstruction needs: the first mesh primitive's vertex
positions and triangle indices, in the model's rest pose. Used for the bat mesh
(bat.glb); Draco-compressed assets (e.g. stadiums) are out of scope here — see
STADIUM_NOTES.md.
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


def load_glb_mesh(path, mesh_index=0, prim_index=0):
    """Return (vertices Nx3 float, faces Mx3 int) of a plain-glTF .glb mesh."""
    d = Path(path).read_bytes()
    assert d[:4] == b"glTF", "not a GLB"
    clen, = struct.unpack_from("<I", d, 12)
    gltf = json.loads(d[20:20 + clen])
    # BIN chunk follows the JSON chunk
    bin_start = 20 + clen
    blen, btype = struct.unpack_from("<II", d, bin_start)
    bin_data = d[bin_start + 8: bin_start + 8 + blen]

    prim = gltf["meshes"][mesh_index]["primitives"][prim_index]
    verts = _read_accessor(gltf, bin_data, prim["attributes"]["POSITION"])
    idx = _read_accessor(gltf, bin_data, prim["indices"]).astype(int)
    faces = idx.reshape(-1, 3)
    return verts, faces

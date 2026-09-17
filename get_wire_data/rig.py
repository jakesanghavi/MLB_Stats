"""
Skeleton rig + forward kinematics for MLB Gameday 3D actors.

The web viewer loads a glTF character rig (``{assetBase}/models/generic/
generic-lod.gltf``, assetBase = https://fv-assets.mlb.com/v/<hash>) and poses it
by: setting the Pelvis (boneId 0) world position to the actor's ``rootPos`` and
setting each tracked bone's local rotation to the decoded quaternion, keeping the
rig's bind-pose local translations. The skeleton lives under an identity
``Armature`` node and is authored in feet, so FK yields world coordinates
directly (same frame as the tracked ball: x/z ground plane, y up).

RigSkeleton parses only the node hierarchy + bind TRS from the glTF JSON (the
external .bin mesh buffer is not needed). ``fk(root_pos, quats_by_name, scale)``
returns world joint positions; ``bone_edges`` lists parent→child joint segments
for drawing a skeleton.
"""
import json
from pathlib import Path

import numpy as np

DEFAULT_RIG = Path(__file__).resolve().parent / "assets" / "generic-lod.gltf"
PELVIS = "joint_Pelvis"


def _quat_matrix(q):
    x, y, z, w = q
    n = (x * x + y * y + z * z + w * w) ** 0.5
    if n == 0:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def _trs(translation, rotation, scale):
    m = np.eye(4)
    r = _quat_matrix(rotation) if rotation is not None else np.eye(3)
    s = np.array(scale if scale is not None else [1.0, 1.0, 1.0])
    m[:3, :3] = r * s  # column-scale == R @ diag(s)
    if translation is not None:
        m[:3, 3] = translation
    return m


class RigSkeleton:
    def __init__(self, gltf_path=DEFAULT_RIG):
        g = json.loads(Path(gltf_path).read_text())
        self.nodes = g["nodes"]
        self.name_to_idx = {n.get("name"): i for i, n in enumerate(self.nodes)}
        self.parent = {}
        for i, n in enumerate(self.nodes):
            for c in n.get("children", []):
                self.parent[c] = i
        self.roots = [i for i in range(len(self.nodes)) if i not in self.parent]

        # bind-pose local matrices
        self.bind_local = [
            _trs(n.get("translation"), n.get("rotation"), n.get("scale"))
            for n in self.nodes
        ]
        # joint node indices (the rig's tracked/skeletal bones)
        self.joint_idx = [i for i, n in enumerate(self.nodes)
                          if (n.get("name") or "").startswith("joint_")]
        # topologically ordered node list (parents before children)
        self.order = self._topo_order()
        # parent→child edges between joint nodes (skip non-joint ancestors)
        self.bone_edges = []
        for j in self.joint_idx:
            p = self.parent.get(j)
            while p is not None and p not in set(self.joint_idx):
                p = self.parent.get(p)
            if p is not None:
                self.bone_edges.append((p, j))

    def _topo_order(self):
        order = []
        seen = set()

        def visit(i):
            if i in seen:
                return
            seen.add(i)
            order.append(i)
            for c in self.nodes[i].get("children", []):
                visit(c)
        for r in self.roots:
            visit(r)
        return order

    def fk(self, root_pos, quats_by_name, scale=1.0):
        """Return {node_index: world_xyz(3,)} for the posed skeleton.

        root_pos: (x, y, z) world position of the Pelvis (feet).
        quats_by_name: {bone_name: [x, y, z, w]} decoded local rotations.
        scale: actor scale (applied at the Pelvis).
        """
        pelvis = self.name_to_idx[PELVIS]
        quats_by_idx = {}
        for name, q in quats_by_name.items():
            idx = self.name_to_idx.get(name)
            if idx is not None:
                quats_by_idx[idx] = q

        world = [None] * len(self.nodes)
        for i in self.order:
            if i == pelvis:
                rot = quats_by_idx.get(pelvis)
                local = _trs(list(root_pos),
                             rot if rot is not None else self.nodes[i].get("rotation"),
                             [scale, scale, scale])
            elif i in quats_by_idx:
                n = self.nodes[i]
                local = _trs(n.get("translation"), quats_by_idx[i], n.get("scale"))
            else:
                local = self.bind_local[i]
            p = self.parent.get(i)
            world[i] = local if p is None else world[p] @ local

        return {i: world[i][:3, 3] for i in range(len(self.nodes)) if world[i] is not None}

    def segments(self, world_pos):
        """List of (p0, p1) world-point pairs for each joint bone edge."""
        segs = []
        for p, c in self.bone_edges:
            if p in world_pos and c in world_pos:
                segs.append((world_pos[p], world_pos[c]))
        return segs


if __name__ == "__main__":
    rig = RigSkeleton()
    print("nodes:", len(rig.nodes), "joints:", len(rig.joint_idx),
          "bone edges:", len(rig.bone_edges), "roots:", [rig.nodes[r].get("name") for r in rig.roots])
    # sanity: bind pose (no data quats) with pelvis at (0, 3.3, 0)
    wp = rig.fk((0.0, 3.3, 0.0), {}, 1.0)
    ys = [p[1] for p in wp.values()]
    print("bind-pose height span (ft): %.2f .. %.2f" % (min(ys), max(ys)))

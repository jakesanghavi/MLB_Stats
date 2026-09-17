"""
Full 3D reconstruction of a downloaded Gameday 3D play.

Poses every actor's skeleton via forward kinematics (rig.RigSkeleton) from the
decoded root positions + joint quaternions, and draws:
  - each actor's skeleton (colored bone segments by role),
  - the bat mesh (bat.glb) placed from the tracked handle/head positions,
  - the ball (small sphere) with a yellow halo and a persistent path trail,
over the action window.

Camera / zoom options (--view):
  action  : fit the actors present in the window (default)
  full    : fit the whole clip
  infield : fixed home-plate/infield framing
  follow  : auto-zoom that tracks the ball (holds last position through gaps),
            with --zoom controlling the half-width.

The field-Z axis is inverted (outfield up), matching the 2D animator.

    python reconstruct3d.py <play_dir> [out.mp4] [--view follow] [--zoom 45]
                            [--fps 20] [--azim -72] [--elev 16] [--full]
"""
import argparse
import bisect
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from read_play import PlayReader
from rig import RigSkeleton
from glb import load_glb_mesh

TYPE_COLORS = {
    "pitcher": "#d12d49", "batter": "#005A9C", "catcher": "#EB6E1F",
    "fielder": "#1fbe3a", "umpire": "#888888", "plate-umpire": "#555555",
    "coach": "#000000", "runner": "#775eef", "unknown": "#bbbbbb",
}
TRACK_ALL_TRAILS = True
_BALL_GAP_BREAK = 0.3
_BAT_MODEL_LEN = 2.843  # bat.glb knob->barrel extent (ft)
ASSETS = Path(__file__).resolve().parent / "assets"


# ---- geometry helpers -------------------------------------------------------
def _to_plot(p):
    """world (x, y=height, z) -> plot (x, z, y) so height is the vertical axis."""
    return (p[0], p[2], p[1])


def _rot_align(a, b):
    """Rotation matrix taking unit vector a to unit vector b."""
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = float(np.linalg.norm(v))
    if s < 1e-8:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))


def _bat_world_verts(verts, handle, head):
    """Map bat model (knob at y=0, barrel at y~2.843) onto world handle->head."""
    handle = np.array(handle, float)
    b = np.array(head, float) - handle
    L = float(np.linalg.norm(b))
    if L < 1e-4:
        return None
    R = _rot_align(np.array([0.0, 1.0, 0.0]), b / L)
    s = L / _BAT_MODEL_LEN
    return (R @ (verts.T * s)).T + handle


def _unit_sphere(nu=10, nv=7):
    u = np.linspace(0, 2 * np.pi, nu + 1)
    v = np.linspace(0, np.pi, nv + 1)
    pts = np.array([[np.cos(uu) * np.sin(vv), np.cos(vv), np.sin(uu) * np.sin(vv)]
                    for vv in v for uu in u])
    faces = []
    w = nu + 1
    for j in range(nv):
        for i in range(nu):
            a = j * w + i; bb = a + 1; c = a + w; dd = c + 1
            faces.append([a, bb, dd]); faces.append([a, dd, c])
    return pts, np.array(faces)


# ---- pose sampling (per-actor interpolation, no blink) ----------------------
def _actor_pose_tracks(reader):
    tracks = {}
    for f in reader.frames:
        for a in f.get("actorPoses", []):
            if a.get("rootPos") and a.get("jointRotations"):
                tracks.setdefault(a["uid"], []).append((f["time"], a))
    for uid in tracks:
        tracks[uid].sort(key=lambda x: x[0])
    return tracks


def _sample_pose(track, t, max_gap=1.0):
    times = [x[0] for x in track]
    if not times or t < times[0] or t > times[-1]:
        return None
    j = bisect.bisect_left(times, t)
    if j < len(times) and times[j] == t:
        return track[j][1]
    lo, hi = max(0, j - 1), min(len(times) - 1, j)
    t1, p1 = track[lo]; t2, p2 = track[hi]
    if t2 - t1 > max_gap:
        return None
    f = (t - t1) / (t2 - t1) if t2 > t1 else 0.0
    r1, r2 = p1["rootPos"], p2["rootPos"]
    root = {"x": r1["x"] + (r2["x"] - r1["x"]) * f,
            "y": r1["y"] + (r2["y"] - r1["y"]) * f,
            "z": r1["z"] + (r2["z"] - r1["z"]) * f}
    nearest = p1 if f < 0.5 else p2
    return {"rootPos": root, "jointRotations": nearest["jointRotations"],
            "scale": nearest.get("scale", 1.0)}


def _bat_track(reader):
    out = []
    for f in reader.frames:
        ib = f.get("inferredBat")
        if not (ib and ib.get("headPosition") and ib.get("handlePosition")):
            continue
        hp, hh = ib["headPosition"], ib["handlePosition"]
        head = (hp["x"], hp["y"], hp["z"]); handle = (hh["x"], hh["y"], hh["z"])
        # skip the model-space default (knob at origin, barrel straight up)
        if abs(handle[0]) < 1e-6 and abs(handle[1]) < 1e-6 and abs(handle[2]) < 1e-6:
            continue
        out.append((f["time"], handle, head))
    out.sort(key=lambda x: x[0])
    return out


def _sample_gap(track, t, max_gap):
    times = [x[0] for x in track]
    if not times or t < times[0] or t > times[-1]:
        return None
    j = bisect.bisect_left(times, t)
    if j < len(times) and times[j] == t:
        return track[j][1:]
    lo, hi = max(0, j - 1), min(len(times) - 1, j)
    if track[hi][0] - track[lo][0] > max_gap:
        return None
    return track[hi if (t - track[lo][0]) >= (track[hi][0] - t) else lo][1:]


def _lerp_ball(track, t, max_gap=0.2):
    times = [x[0] for x in track]
    if not times or t < times[0] or t > times[-1]:
        return None
    j = bisect.bisect_left(times, t)
    if j < len(times) and times[j] == t:
        return track[j][1]
    lo, hi = max(0, j - 1), min(len(times) - 1, j)
    t1, v1 = track[lo]; t2, v2 = track[hi]
    if t2 - t1 > max_gap:
        return None
    f = (t - t1) / (t2 - t1) if t2 > t1 else 0.0
    return tuple(a + (b - a) * f for a, b in zip(v1, v2))


def reconstruct3d(play_dir, out_path="reconstruction3d.mp4", view="action",
                  zoom=45.0, fps=20, azim=-72.0, elev=16.0, full=False):
    reader = PlayReader(play_dir)
    if not reader.frames:
        raise SystemExit("no frames")
    rig = RigSkeleton()
    bat_v, bat_f = load_glb_mesh(ASSETS / "bat.glb")
    # rbi-ball.glb is ~1100 verts — too heavy for matplotlib; a generated
    # sphere plus the yellow halo is the readable stand-in at field scale.
    ball_v, ball_f = _unit_sphere()
    ball_r = 0.35  # ft (exaggerated a bit so it reads at field scale)

    t0 = reader.frames[0]["time"]
    if full:
        view = "full"
    w0, w1 = (reader.frames[0]["time"], reader.frames[-1]["time"]) if view == "full" \
        else reader.play_window()
    print(f"window {w0 - t0:.2f}..{w1 - t0:.2f}s ({w1 - w0:.2f}s) view={view}")

    pose_tracks = _actor_pose_tracks(reader)
    ball_track = [(t, (x, y, z)) for t, x, y, z in reader.ball_track()]
    bat_track = _bat_track(reader)
    grid = np.arange(w0, w1, 1.0 / fps)

    print("running FK: %d frames x %d actors ..." % (len(grid), len(pose_tracks)))
    F_segs, F_cols, F_ball, F_bat, F_center = [], [], [], [], []
    for t in grid:
        segs, cols, roots = [], [], []
        for uid, tr in pose_tracks.items():
            pose = _sample_pose(tr, t)
            if pose is None:
                continue
            rp = (pose["rootPos"]["x"], pose["rootPos"]["y"], pose["rootPos"]["z"])
            roots.append((rp[0], rp[2]))
            wp = rig.fk(rp, pose["jointRotations"], pose.get("scale", 1.0))
            col = TYPE_COLORS.get(reader.actor_type(uid), TYPE_COLORS["unknown"])
            for p, c in rig.segments(wp):
                segs.append([_to_plot(p), _to_plot(c)]); cols.append(col)
        F_segs.append(segs); F_cols.append(cols)
        ball = _lerp_ball(ball_track, t)
        F_ball.append(ball)
        bh = _sample_gap(bat_track, t, 0.3)
        F_bat.append(_bat_world_verts(bat_v, bh[0], bh[1]) if bh else None)
        # follow target: the ball only (held through gaps below), so the camera
        # locks on the action instead of being pulled around by deep fielders.
        F_center.append((ball[0], ball[2]) if ball is not None else None)

    # static bounds
    allpts = np.array([pt for segs in F_segs for seg in segs for pt in seg])
    if view == "infield":
        xlo, xhi, zlo, zhi = -80, 80, -175, 30
    else:  # action / full
        xlo, xhi = np.percentile(allpts[:, 0], [1, 99])
        zlo, zhi = np.percentile(allpts[:, 1], [1, 99])
        px = (xhi - xlo) * 0.08 + 5; pz = (zhi - zlo) * 0.08 + 5
        xlo, xhi, zlo, zhi = xlo - px, xhi + px, zlo - pz, zhi + pz

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_zlim(0, 10)
    ax.set_xlabel("field X (ft)"); ax.set_ylabel("field Z (ft, inverted)")
    ax.set_zlabel("height (ft)")
    ax.view_init(elev=elev, azim=azim)

    def apply_bounds(cx=None, cz=None):
        if view == "follow" and cx is not None:
            ax.set_xlim(cx - zoom, cx + zoom)
            ax.set_ylim(cz + zoom, cz - zoom)  # inverted (outfield up)
            ax.set_box_aspect((2 * zoom, 2 * zoom, 10))
        else:
            ax.set_xlim(xlo, xhi)
            ax.set_ylim(zhi, zlo)  # inverted
            ax.set_box_aspect(((xhi - xlo), (zhi - zlo), 12))

    apply_bounds()

    coll = Line3DCollection([[(0, 0, 0), (0, 0, 0)]], linewidths=1.6)
    ax.add_collection3d(coll)
    bat_coll = Poly3DCollection([], facecolor="#8a5a2b", edgecolor="#5c3a17", linewidths=0.3)
    ax.add_collection3d(bat_coll)
    ball_coll = Poly3DCollection([], facecolor="#f7f7f7", edgecolor="#cccccc", linewidths=0.2)
    ax.add_collection3d(ball_coll)
    halo = ax.scatter([], [], [], s=140, c="#ffd21e", alpha=0.35, edgecolors="none", depthshade=False)
    trail_line, = ax.plot([], [], [], "-", color="#ff9e00", lw=1.7, alpha=0.85)
    title = ax.set_title("")

    legend_types = ["pitcher", "batter", "catcher", "fielder", "umpire", "coach"]
    handles = [Line2D([0], [0], color=TYPE_COLORS[t], lw=3, label=t.capitalize()) for t in legend_types]
    handles += [Line2D([0], [0], color="#8a5a2b", lw=4, label="Bat"),
                Line2D([0], [0], marker="o", color="w", label="Ball",
                       markerfacecolor="#ffd21e", markeredgecolor="black", markersize=9)]
    ax.legend(handles=handles, loc="upper right", fontsize=8)

    # initial follow center: first tracked ball, else infield
    first_c = next((c for c in F_center if c is not None), (0.0, -60.0))
    trail_pts = []
    state = {"prev_t": None, "center": first_c}
    if view == "follow":
        apply_bounds(*first_c)

    def ball_polys(center):
        cx, cy, cz = center  # world x,y,z
        wv = ball_v * ball_r + np.array([cx, cy, cz])
        pv = np.column_stack([wv[:, 0], wv[:, 2], wv[:, 1]])  # to plot coords
        return [pv[f] for f in ball_f]

    def update(i):
        if i == 0:
            trail_pts.clear(); state["prev_t"] = None
        coll.set_segments(F_segs[i]); coll.set_color(F_cols[i])

        bv = F_bat[i]
        if bv is not None:
            pv = np.column_stack([bv[:, 0], bv[:, 2], bv[:, 1]])
            bat_coll.set_verts([pv[f] for f in bat_f])
        else:
            bat_coll.set_verts([])

        b = F_ball[i]
        if b is not None:
            x, y, z = b
            ball_coll.set_verts(ball_polys((x, y, z)))
            halo._offsets3d = ([x], [z], [y])
            gap = state["prev_t"] is not None and (grid[i] - state["prev_t"]) > _BALL_GAP_BREAK
            if gap:
                trail_pts.append((np.nan, np.nan, np.nan))
            trail_pts.append((x, z, y))
            state["prev_t"] = grid[i]
        else:
            ball_coll.set_verts([]); halo._offsets3d = ([], [], [])
        if TRACK_ALL_TRAILS and trail_pts:
            arr = np.array(trail_pts, float)
            trail_line.set_data(arr[:, 0], arr[:, 1]); trail_line.set_3d_properties(arr[:, 2])

        if view == "follow":
            if F_center[i] is not None:
                state["center"] = F_center[i]  # lock on ball; hold last through gaps
            apply_bounds(state["center"][0], state["center"][1])
        n = len({tuple(c) for c in F_cols[i]}) if F_cols[i] else 0
        title.set_text(f"Gameday 3D reconstruction  |  t={grid[i] - w0:5.2f}s  |  actors={n}")
        return coll, bat_coll, ball_coll, halo, trail_line, title

    anim = FuncAnimation(fig, update, frames=len(grid), blit=False, interval=1000 / fps)
    anim.save(out_path, writer=FFMpegWriter(fps=fps, bitrate=3200))
    plt.close(fig)
    print(f"wrote {out_path} ({len(grid)} frames @ {fps}fps)")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("play_dir")
    ap.add_argument("out", nargs="?", default="reconstruction3d.mp4")
    ap.add_argument("--view", choices=["action", "full", "infield", "follow"], default="action")
    ap.add_argument("--zoom", type=float, default=45.0, help="follow half-width (ft)")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--azim", type=float, default=-72.0)
    ap.add_argument("--elev", type=float, default=16.0)
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    reconstruct3d(args.play_dir, args.out, view=args.view, zoom=args.zoom,
                  fps=args.fps, azim=args.azim, elev=args.elev, full=args.full)

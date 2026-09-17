"""
Full 3D reconstruction of a downloaded Gameday 3D play.

Uses the decoded per-actor root positions + joint quaternions and the character
rig (rig.RigSkeleton) to run forward kinematics and draw every actor's skeleton
in 3D, plus the tracked ball, over the action window. This verifies we captured
enough tracking data to rebuild the play in 3D.

    python reconstruct3d.py <play_dir> [out.mp4] [--full] [--fps 20]

Coordinates: world x/z is the field plane, y is height (feet). Plotted with the
height (y) as the vertical axis.
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
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from read_play import PlayReader
from rig import RigSkeleton

TYPE_COLORS = {
    "pitcher": "#d12d49", "batter": "#005A9C", "catcher": "#EB6E1F",
    "fielder": "#1fbe3a", "umpire": "#888888", "plate-umpire": "#555555",
    "coach": "#000000", "runner": "#775eef", "unknown": "#bbbbbb",
}
TRACK_ALL_TRAILS = True
_BALL_GAP_BREAK = 0.3


def _actor_pose_tracks(reader):
    """uid -> sorted [(time, pose_dict)] with rootPos + jointRotations + scale."""
    tracks = {}
    for f in reader.frames:
        for a in f.get("actorPoses", []):
            if a.get("rootPos") and a.get("jointRotations"):
                tracks.setdefault(a["uid"], []).append((f["time"], a))
    for uid in tracks:
        tracks[uid].sort(key=lambda x: x[0])
    return tracks


def _sample_pose(track, t, max_gap=1.0):
    """Interpolate rootPos linearly, take nearest-sample joint quats, near time t."""
    times = [x[0] for x in track]
    if not times or t < times[0] or t > times[-1]:
        return None
    j = bisect.bisect_left(times, t)
    if j < len(times) and times[j] == t:
        return track[j][1]
    lo = max(0, j - 1)
    hi = min(len(times) - 1, j)
    t1, p1 = track[lo]
    t2, p2 = track[hi]
    if t2 - t1 > max_gap:
        return None
    f = (t - t1) / (t2 - t1) if t2 > t1 else 0.0
    r1, r2 = p1["rootPos"], p2["rootPos"]
    root = (r1["x"] + (r2["x"] - r1["x"]) * f,
            r1["y"] + (r2["y"] - r1["y"]) * f,
            r1["z"] + (r2["z"] - r1["z"]) * f)
    nearest = p1 if f < 0.5 else p2
    return {"rootPos": {"x": root[0], "y": root[1], "z": root[2]},
            "jointRotations": nearest["jointRotations"],
            "scale": nearest.get("scale", 1.0)}


def _sample_ball(track, t, max_gap=0.2):
    times = [x[0] for x in track]
    if not times or t < times[0] or t > times[-1]:
        return None
    j = bisect.bisect_left(times, t)
    if j < len(times) and times[j] == t:
        return track[j][1]
    lo, hi = max(0, j - 1), min(len(times) - 1, j)
    t1, v1 = track[lo]
    t2, v2 = track[hi]
    if t2 - t1 > max_gap:
        return None
    f = (t - t1) / (t2 - t1) if t2 > t1 else 0.0
    return tuple(a + (b - a) * f for a, b in zip(v1, v2))


def reconstruct3d(play_dir, out_path="reconstruction3d.mp4", full=False, fps=20):
    reader = PlayReader(play_dir)
    if not reader.frames:
        raise SystemExit("no frames")
    rig = RigSkeleton()
    t0 = reader.frames[0]["time"]
    w0, w1 = (reader.frames[0]["time"], reader.frames[-1]["time"]) if full else reader.play_window()
    print(f"window {w0 - t0:.2f}..{w1 - t0:.2f}s ({w1 - w0:.2f}s)")

    pose_tracks = _actor_pose_tracks(reader)
    ball_track = [(t, (x, y, z)) for t, x, y, z in reader.ball_track()]
    grid = np.arange(w0, w1, 1.0 / fps)

    # precompute per-frame skeleton segments + colors, and ball points
    print("running FK over %d frames x %d actors ..." % (len(grid), len(pose_tracks)))
    frames_segs, frames_cols, frames_ball = [], [], []
    for t in grid:
        segs, cols = [], []
        for uid, tr in pose_tracks.items():
            pose = _sample_pose(tr, t)
            if pose is None:
                continue
            rp = (pose["rootPos"]["x"], pose["rootPos"]["y"], pose["rootPos"]["z"])
            wp = rig.fk(rp, pose["jointRotations"], pose.get("scale", 1.0))
            col = TYPE_COLORS.get(reader.actor_type(uid), TYPE_COLORS["unknown"])
            for p, c in rig.segments(wp):
                # plot axes: X=x, Y=z, Z=y(height)
                segs.append([(p[0], p[2], p[1]), (c[0], c[2], c[1])])
                cols.append(col)
        frames_segs.append(segs)
        frames_cols.append(cols)
        frames_ball.append(_sample_ball(ball_track, t))

    # bounds from all segment endpoints
    allpts = np.array([pt for segs in frames_segs for seg in segs for pt in seg])
    xlo, xhi = np.percentile(allpts[:, 0], [1, 99])
    ylo, yhi = np.percentile(allpts[:, 1], [1, 99])
    padx = (xhi - xlo) * 0.08 + 5
    pady = (yhi - ylo) * 0.08 + 5

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(xlo - padx, xhi + padx)
    ax.set_ylim(ylo - pady, yhi + pady)
    ax.set_zlim(0, 8)
    ax.set_box_aspect(((xhi - xlo) + 2 * padx, (yhi - ylo) + 2 * pady, 24))
    ax.set_xlabel("field X (ft)")
    ax.set_ylabel("field Z (ft)")
    ax.set_zlabel("height (ft)")
    ax.view_init(elev=16, azim=-72)

    coll = Line3DCollection([[(0, 0, 0), (0, 0, 0)]], linewidths=1.6)
    ax.add_collection3d(coll)
    ball_scat = ax.scatter([], [], [], c="#ffd21e", edgecolors="black", s=45, depthshade=False)
    trail_line, = ax.plot([], [], [], "-", color="#ff9e00", lw=1.6, alpha=0.8)
    title = ax.set_title("")

    legend_types = ["pitcher", "batter", "catcher", "fielder", "umpire", "coach"]
    handles = [Line2D([0], [0], color=TYPE_COLORS[t], lw=3, label=t.capitalize())
               for t in legend_types]
    handles.append(Line2D([0], [0], marker="o", color="w", label="Ball",
                          markerfacecolor="#ffd21e", markeredgecolor="black", markersize=8))
    ax.legend(handles=handles, loc="upper right", fontsize=8)

    trail_pts = []
    state = {"prev_t": None}

    def update(i):
        if i == 0:
            trail_pts.clear(); state["prev_t"] = None
        coll.set_segments(frames_segs[i])
        coll.set_color(frames_cols[i])
        b = frames_ball[i]
        if b is not None:
            x, y, z = b
            ball_scat._offsets3d = ([x], [z], [y])
            gap = state["prev_t"] is not None and (grid[i] - state["prev_t"]) > _BALL_GAP_BREAK
            if TRACK_ALL_TRAILS:
                if gap:
                    trail_pts.append((np.nan, np.nan, np.nan))
                trail_pts.append((x, z, y))
            state["prev_t"] = grid[i]
        else:
            ball_scat._offsets3d = ([], [], [])
        if TRACK_ALL_TRAILS and trail_pts:
            arr = np.array(trail_pts, dtype=float)
            trail_line.set_data(arr[:, 0], arr[:, 1])
            trail_line.set_3d_properties(arr[:, 2])
        n_actors = len({tuple(c) for c in frames_cols[i]}) if frames_cols[i] else 0
        title.set_text(f"Gameday 3D reconstruction  |  t={grid[i] - w0:5.2f}s  |  "
                       f"bones={len(frames_segs[i])}")
        return coll, ball_scat, trail_line, title

    anim = FuncAnimation(fig, update, frames=len(grid), blit=False, interval=1000 / fps)
    anim.save(out_path, writer=FFMpegWriter(fps=fps, bitrate=3000))
    plt.close(fig)
    print(f"wrote {out_path} ({len(grid)} frames @ {fps}fps)")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("play_dir")
    ap.add_argument("out", nargs="?", default="reconstruction3d.mp4")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--fps", type=int, default=20)
    args = ap.parse_args()
    reconstruct3d(args.play_dir, args.out, full=args.full, fps=args.fps)

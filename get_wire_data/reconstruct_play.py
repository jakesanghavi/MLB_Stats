"""
Reconstruct a top-down animation of a downloaded Gameday 3D play from the raw
files, as a verification that the tracking data was captured completely.

Renders every actor's world root position (colored by role) and the tracked ball
over the course of the play, top-down (field-plane x/z; y is height). This is a
data-completeness check: if the whole play was captured, you see all fielders,
the batter/runners, umpires and the ball move coherently through the play.

    python reconstruct_play.py <play_dir> [out.mp4]

Full rigged-mannequin rendering (bending the skeleton with the decoded joint
quaternions) additionally needs the character rig asset (bone hierarchy + bind
pose) that the web viewer loads separately; that is a follow-up. Root positions
+ ball fully exercise the decoded tracking stream.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from read_play import PlayReader

TYPE_COLORS = {
    "pitcher": "#d12d49", "batter": "#005A9C", "catcher": "#EB6E1F",
    "fielder": "#1fbe3a", "umpire": "#888888", "plate-umpire": "#555555",
    "coach": "#000000", "runner": "#775eef", "unknown": "#bbbbbb",
}


def _percentile_limits(vals, lo=1, hi=99, pad=0.08):
    a, b = np.percentile(vals, lo), np.percentile(vals, hi)
    span = max(b - a, 1.0)
    return a - span * pad, b + span * pad


def reconstruct(play_dir, out_path=None, max_frames=700, fps=30):
    reader = PlayReader(play_dir)
    frames = reader.frames
    if not frames:
        raise SystemExit("no frames decoded")

    step = max(1, len(frames) // max_frames)
    sel = frames[::step]
    t0 = frames[0]["time"]

    # bounds from all actor roots + ball
    xs, zs = [], []
    for f in frames:
        for a in f["actorPoses"]:
            if a["rootPos"]:
                xs.append(a["rootPos"]["x"]); zs.append(a["rootPos"]["z"])
    for _, x, _, z in reader.ball_track():
        xs.append(x); zs.append(z)
    xlim = _percentile_limits(np.array(xs))
    zlim = _percentile_limits(np.array(zs))

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_aspect("equal")
    ax.set_xlim(*xlim); ax.set_ylim(*zlim)
    ax.set_xlabel("field X (ft)"); ax.set_ylabel("field Z (ft)")
    ax.set_facecolor("#eef6ec")

    actor_scatter = ax.scatter([], [], s=90, edgecolors="black", linewidths=0.6, zorder=3)
    ball_dot = ax.scatter([], [], s=60, c="#ffd21e", edgecolors="black", linewidths=1.2, zorder=5)
    ball_trail, = ax.plot([], [], "-", color="#ff9e00", lw=1.5, alpha=0.7, zorder=4)
    title = ax.set_title("")

    legend_types = ["pitcher", "batter", "catcher", "fielder", "umpire", "coach"]
    handles = [Line2D([0], [0], marker="o", color="w", label=t.capitalize(),
                      markerfacecolor=TYPE_COLORS[t], markeredgecolor="black", markersize=9)
               for t in legend_types]
    handles.append(Line2D([0], [0], marker="o", color="w", label="Ball",
                          markerfacecolor="#ffd21e", markeredgecolor="black", markersize=8))
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)

    ball_hist = []

    def update(i):
        f = sel[i]
        pts, cols = [], []
        for a in f["actorPoses"]:
            rp = a["rootPos"]
            if not rp:
                continue
            pts.append((rp["x"], rp["z"]))
            cols.append(TYPE_COLORS.get(reader.actor_type(a["uid"]), TYPE_COLORS["unknown"]))
        if pts:
            actor_scatter.set_offsets(np.array(pts))
            actor_scatter.set_facecolors(cols)
        else:
            actor_scatter.set_offsets(np.empty((0, 2)))

        b = f.get("ball")
        if b:
            ball_dot.set_offsets(np.array([[b["x"], b["z"]]]))
            ball_dot.set_sizes([40 + max(0.0, b["y"]) * 8])  # bigger when higher
            ball_hist.append((b["x"], b["z"]))
        else:
            ball_dot.set_offsets(np.empty((0, 2)))
        if len(ball_hist) > 40:
            del ball_hist[0]
        if ball_hist:
            bh = np.array(ball_hist)
            ball_trail.set_data(bh[:, 0], bh[:, 1])

        title.set_text(f"Gameday 3D reconstruction  |  t={f['time'] - t0:5.1f}s  "
                       f"|  actors={len(pts)}  |  frame {f['num']}")
        return actor_scatter, ball_dot, ball_trail, title

    anim = FuncAnimation(fig, update, frames=len(sel), blit=False, interval=1000 / fps)

    out = out_path or "reconstruction.mp4"
    anim.save(out, writer=FFMpegWriter(fps=fps, bitrate=2400))
    plt.close(fig)
    print(f"wrote {out}  ({len(sel)} frames @ {fps}fps, step={step})")
    return out


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    out = sys.argv[2] if len(sys.argv) > 2 else "reconstruction.mp4"
    reconstruct(d, out)

"""
Reconstruct a top-down animation of a downloaded Gameday 3D play from the raw
files, as a verification that the tracking data was captured completely.

Renders every actor's world root position (colored by role) and the tracked ball
top-down (field-plane x/z; y is height). Z is drawn inverted so the outfield is
up. By default it clips to the actual action window (see PlayReader.play_window)
and resamples on a uniform timeline, interpolating each actor between its own
30 fps samples so actors do not blink in/out as the tracked set changes.

    python reconstruct_play.py <play_dir> [out.mp4] [--full] [--fps 30]

Full rigged-mannequin rendering (bending the skeleton with the decoded joint
quaternions) additionally needs the character rig asset (bone hierarchy + bind
pose) the web viewer loads separately; that is a follow-up.
"""
import argparse
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

# When True, keep the entire ball path on screen for the whole play (past steps
# persist as the animation evolves). When False, only a short recent trail is
# shown and it resets across ball gaps.
TRACK_ALL_TRAILS = True

# A jump larger than this (seconds) between consecutive ball samples starts a new
# trail segment (so disjoint ball phases aren't joined by a fake straight line).
_BALL_GAP_BREAK = 0.3


def _sample(track, t, max_gap):
    """Interpolate a sorted [(t, (x, z[, y])), ...] track at time t.

    Returns the interpolated tuple when t lies within the track and the
    surrounding samples are <= max_gap apart, else None (actor absent / big gap).
    """
    n = len(track)
    if n == 0 or t < track[0][0] or t > track[-1][0]:
        return None
    lo, hi = 0, n - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if track[mid][0] < t:
            lo = mid + 1
        else:
            hi = mid
    if track[lo][0] == t:
        return track[lo][1]
    t1, v1 = track[lo - 1]
    t2, v2 = track[lo]
    if t2 - t1 > max_gap:
        return None
    f = (t - t1) / (t2 - t1) if t2 > t1 else 0.0
    return tuple(a + (b - a) * f for a, b in zip(v1, v2))


def _percentile_limits(vals, lo=1, hi=99, pad=0.1):
    a, b = np.percentile(vals, lo), np.percentile(vals, hi)
    span = max(b - a, 1.0)
    return a - span * pad, b + span * pad


def reconstruct(play_dir, out_path="reconstruction.mp4", full=False, fps=30):
    reader = PlayReader(play_dir)
    if not reader.frames:
        raise SystemExit("no frames decoded")
    t0 = reader.frames[0]["time"]

    if full:
        w0, w1 = reader.frames[0]["time"], reader.frames[-1]["time"]
    else:
        w0, w1 = reader.play_window()
    print(f"window: {w0 - t0:.2f}s .. {w1 - t0:.2f}s  ({w1 - w0:.2f}s, {'full' if full else 'action'})")

    # per-actor root tracks (uid -> sorted [(t,(x,z))]) and ball track [(t,(x,z,y))]
    raw_tracks = reader.actor_tracks()
    actor_tracks = {uid: [(t, (rp["x"], rp["z"])) for t, rp in tr]
                    for uid, tr in raw_tracks.items()}
    ball_track = [(t, (x, z, y)) for t, x, y, z in reader.ball_track()]

    # bounds from actors present in the window
    xs, zs = [], []
    for uid, tr in actor_tracks.items():
        for t, (x, z) in tr:
            if w0 <= t <= w1:
                xs.append(x); zs.append(z)
    for t, (x, z, _) in ball_track:
        if w0 <= t <= w1:
            xs.append(x); zs.append(z)
    if not xs:
        raise SystemExit("no positions in window")
    xlim = _percentile_limits(np.array(xs))
    zlim = _percentile_limits(np.array(zs))

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_aspect("equal")
    ax.set_xlim(*xlim)
    ax.set_ylim(zlim[1], zlim[0])  # inverted Z (outfield up)
    ax.set_xlabel("field X (ft)")
    ax.set_ylabel("field Z (ft, inverted)")
    ax.set_facecolor("#eef6ec")

    actor_scatter = ax.scatter([], [], s=90, edgecolors="black", linewidths=0.6, zorder=3)
    ball_dot = ax.scatter([], [], s=60, c="#ffd21e", edgecolors="black", linewidths=1.2, zorder=5)
    ball_trail, = ax.plot([], [], "-", color="#ff9e00", lw=1.6, alpha=0.75, zorder=4)
    title = ax.set_title("")

    legend_types = ["pitcher", "batter", "catcher", "fielder", "umpire", "coach"]
    handles = [Line2D([0], [0], marker="o", color="w", label=t.capitalize(),
                      markerfacecolor=TYPE_COLORS[t], markeredgecolor="black", markersize=9)
               for t in legend_types]
    handles.append(Line2D([0], [0], marker="o", color="w", label="Ball",
                          markerfacecolor="#ffd21e", markeredgecolor="black", markersize=8))
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)

    grid = np.arange(w0, w1, 1.0 / fps)
    trail_xz = []          # accumulated trail points (NaN pairs break segments)
    recent = []            # short recent trail (TRACK_ALL_TRAILS == False)
    state = {"prev_ball_t": None}

    def update(i):
        t = grid[i]
        if i == 0:  # reset persistent state when (re)rendering from the start
            trail_xz.clear(); recent.clear(); state["prev_ball_t"] = None

        pts, cols = [], []
        for uid, tr in actor_tracks.items():
            v = _sample(tr, t, max_gap=1.0)  # hold/interp within an actor's own span
            if v is not None:
                pts.append(v)
                cols.append(TYPE_COLORS.get(reader.actor_type(uid), TYPE_COLORS["unknown"]))
        if pts:
            actor_scatter.set_offsets(np.array(pts))
            actor_scatter.set_facecolors(cols)
        else:
            actor_scatter.set_offsets(np.empty((0, 2)))

        bv = _sample(ball_track, t, max_gap=0.2)  # ball only near real samples
        if bv is not None:
            x, z, y = bv
            ball_dot.set_offsets(np.array([[x, z]]))
            ball_dot.set_sizes([40 + max(0.0, y) * 8])
            gap = state["prev_ball_t"] is not None and (t - state["prev_ball_t"]) > _BALL_GAP_BREAK
            if TRACK_ALL_TRAILS:
                if gap:
                    trail_xz.append((np.nan, np.nan))  # break, don't connect across gaps
                trail_xz.append((x, z))
            else:
                if gap:
                    recent.clear()
                recent.append((x, z))
                if len(recent) > 40:
                    del recent[0]
            state["prev_ball_t"] = t
        else:
            ball_dot.set_offsets(np.empty((0, 2)))
            if not TRACK_ALL_TRAILS:
                recent.clear()

        pathpts = trail_xz if TRACK_ALL_TRAILS else recent
        if pathpts:
            arr = np.array(pathpts, dtype=float)
            ball_trail.set_data(arr[:, 0], arr[:, 1])
        else:
            ball_trail.set_data([], [])

        title.set_text(f"Gameday 3D reconstruction  |  t={t - w0:5.2f}s  |  actors={len(pts)}")
        return actor_scatter, ball_dot, ball_trail, title

    anim = FuncAnimation(fig, update, frames=len(grid), blit=False, interval=1000 / fps)
    anim.save(out_path, writer=FFMpegWriter(fps=fps, bitrate=2400))
    plt.close(fig)
    print(f"wrote {out_path}  ({len(grid)} frames @ {fps}fps)")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("play_dir")
    ap.add_argument("out", nargs="?", default="reconstruction.mp4")
    ap.add_argument("--full", action="store_true", help="render the whole clip instead of the action window")
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()
    reconstruct(args.play_dir, args.out, full=args.full, fps=args.fps)

"""
Full 3D reconstruction of a downloaded Gameday 3D play.

Poses every actor's skeleton via forward kinematics (rig.RigSkeleton) from the
decoded root positions + joint quaternions, and draws:
  - each actor's skeleton (colored bone segments by role),
  - the bat mesh (bat.glb) placed from the tracked handle/head positions,
  - the ball (small sphere) with a yellow halo and a persistent path trail,
  - optionally the ballpark field / stadium (INCLUDE_FIELD / INCLUDE_STADIUM),
over the action window.

Camera / zoom options (--view):
  action  : fit the actors present in the window (default)
  full    : fit the whole clip
  infield : fixed home-plate/infield framing
  follow  : start behind the pitcher looking at home; on pitch release, track
            from behind the ball (relative to its motion). After contact the
            camera slews smoothly to stay behind the new heading
            (--zoom = follow half-width).

The field-Z axis is inverted (outfield up), matching the 2D animator.

    python reconstruct3d.py <play_dir> [out.mp4] [--view follow] [--zoom 70]
                            [--fps 20] [--azim -72] [--elev 16] [--full]
                            [--field] [--stadium]
"""
import argparse
import bisect
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgba

sys.path.insert(0, str(Path(__file__).resolve().parent))
from read_play import PlayReader, sample_ball
from rig import RigSkeleton
from glb import load_glb_mesh

TYPE_COLORS = {
    "pitcher": "#d12d49", "batter": "#005A9C", "catcher": "#EB6E1F",
    "fielder": "#000000", "umpire": "#888888", "plate-umpire": "#555555",
    "coach": "#bbbbbb", "runner": "#775eef", "unknown": "#bbbbbb",
}
TRACK_ALL_TRAILS = True
INCLUDE_FIELD = True      # dirt/grass plane from the ballpark glb
INCLUDE_STADIUM = True    # bowl / stands from the ballpark glb
_BALL_GAP_BREAK = 0.3
_BAT_MODEL_LEN = 2.843  # bat.glb knob->barrel extent (ft)
ASSETS = Path(__file__).resolve().parent / "assets"
_FIELD_COLOR = "#5b9e4a"
_STADIUM_COLOR = "#c4beb3"
# Behind the pitcher, looking toward home. matplotlib's eye is opposite
# sin(azim) on an inverted field-Z axis, so +90 is the outfield/mound side.
_PITCHER_AZIM = 90.0
_FOLLOW_AZIM_RATE = 140.0   # deg/s; a 180° reverse takes ~1.3s
_FOLLOW_CENTER_TAU = 0.12   # seconds to ease the look-at point onto the ball
_FOLLOW_MIN_SPEED = 12.0    # ft/s; ignore heading when the ball is basically still


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


def _mesh_plot_tris(verts, faces):
    """world (x, y=height, z) triangles -> plot (x, z, y) triangles."""
    pv = np.column_stack([verts[:, 0], verts[:, 2], verts[:, 1]])
    return pv[faces]


def _pitch_release_time(reader):
    """playEvent action 0 (pitch released), absolute seconds."""
    for t, dt, data in reader.events():
        if dt == 7 and (data or {}).get("action") == 0:
            return t
    return None


def _pitcher_xz(reader, t_ref):
    """Pitcher root (x, field-z) near t_ref, else the rubber."""
    if t_ref is None:
        t_ref = reader.frames[0]["time"]
    for f in reader.frames:
        if abs(f["time"] - t_ref) > 0.3:
            continue
        for a in f.get("actorPoses") or []:
            rp = a.get("rootPos")
            if rp and reader.actor_type(a["uid"]) == "pitcher":
                return (float(rp["x"]), float(rp["z"]))
        break
    return (0.0, -60.5)


def _angle_diff(cur, target):
    """Shortest signed delta from cur to target, in (-180, 180]."""
    return (target - cur + 180.0) % 360.0 - 180.0


def _slew_azim(cur, target, dt, rate=_FOLLOW_AZIM_RATE, vx=0.0):
    """Move azim toward target, capped at `rate` deg/s. A 180° reverse
    prefers the 1B side if vx >= 0, 3B otherwise."""
    d = _angle_diff(cur, target)
    if abs(abs(d) - 180.0) < 0.5:
        d = 180.0 if vx >= 0.0 else -180.0
    step = rate * dt
    if abs(d) <= step:
        return target
    return cur + math.copysign(step, d)


def _behind_azim(vx, vz, fallback):
    """Azimuth sitting opposite the ball's ground velocity, looking at it."""
    if math.hypot(vx, vz) < _FOLLOW_MIN_SPEED:
        return fallback
    # R ∥ velocity so the eye (center - dist*R) sits behind the ball.
    return math.degrees(math.atan2(vz, vx))


def _canvas_rgba(fig):
    """Copy the Agg canvas as an (H, W, 4) uint8 array."""
    fig.canvas.draw()
    return np.array(fig.canvas.buffer_rgba())


def _composite(bg, overlay):
    """Alpha-blend overlay over bg (both HxWx4 uint8)."""
    a = overlay[..., 3:4].astype(np.float32) * (1.0 / 255.0)
    rgb = overlay[..., :3].astype(np.float32) * a + bg[..., :3].astype(np.float32) * (1.0 - a)
    out = np.empty_like(bg)
    out[..., :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    out[..., 3] = 255
    return out


def _write_mp4_rgb(frames, out_path, fps):
    """Pipe RGB frames (H,W,3) uint8 to ffmpeg. Returns (n_frames, elapsed_s)."""
    first = next(frames)
    h, w = first.shape[:2]
    w -= w % 2
    h -= h % 2
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "-",
        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-b:v", "3200k", "-movflags", "+faststart",
        str(out_path),
    ]
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n = 0
    try:
        proc.stdin.write(np.ascontiguousarray(first[:h, :w, :3]).tobytes())
        n = 1
        for frame in frames:
            proc.stdin.write(np.ascontiguousarray(frame[:h, :w, :3]).tobytes())
            n += 1
    finally:
        if proc.stdin is not None:
            proc.stdin.close()
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg exited {rc} writing {out_path}")
    return n, time.perf_counter() - t0


def reconstruct3d(play_dir, out_path="reconstruction3d.mp4", view="action",
                  zoom=70.0, fps=20, azim=-72.0, elev=16.0, full=False,
                  include_field=None, include_stadium=None, ballpark_glb=None,
                  preview_t=None):
    if include_field is None:
        include_field = INCLUDE_FIELD
    if include_stadium is None:
        include_stadium = INCLUDE_STADIUM
    t_run = time.perf_counter()
    def _log(msg):
        print(f"{msg}  [{time.perf_counter() - t_run:.2f}s]")

    t = time.perf_counter()
    reader = PlayReader(play_dir)
    if not reader.frames:
        raise SystemExit("no frames")
    _log(f"play decoded  frames={len(reader.frames)} ({time.perf_counter() - t:.2f}s)")
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
    _log(f"window {w0 - t0:.2f}..{w1 - t0:.2f}s ({w1 - w0:.2f}s) view={view} "
         f"field={include_field} stadium={include_stadium}")

    park = {}
    if include_field or include_stadium:
        from stadium import find_ballpark_glb, load_park_meshes
        t = time.perf_counter()
        glb_path = find_ballpark_glb(reader, explicit=ballpark_glb)
        park = load_park_meshes(glb_path, want_field=include_field,
                                want_stadium=include_stadium)
        counts = {k: len(v[1]) for k, v in park.items()}
        _log(f"  park meshes {counts} ({time.perf_counter() - t:.2f}s)")

    pose_tracks = _actor_pose_tracks(reader)
    ball_track = list(reader.ball_track())
    bat_track = _bat_track(reader)
    t_release = _pitch_release_time(reader)
    pitch_xz = _pitcher_xz(reader, t_release if t_release is not None else w0)
    # Look at the pitcher so the pre-release shot is from behind him toward home.
    pitcher_look = pitch_xz
    grid = np.arange(w0, w1, 1.0 / fps)
    if view == "follow":
        print(f"  follow: pitcher-cam until release "
              f"t_release={None if t_release is None else t_release - t0:.2f}s "
              f"pitcher=({pitch_xz[0]:.1f},{pitch_xz[1]:.1f})")

    print("running FK: %d frames x %d actors ..." % (len(grid), len(pose_tracks)))
    t_fk = time.perf_counter()
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
        ball = sample_ball(ball_track, t)
        F_ball.append(ball)
        bh = _sample_gap(bat_track, t, 0.3)
        F_bat.append(_bat_world_verts(bat_v, bh[0], bh[1]) if bh else None)
        F_center.append((ball[0], ball[2]) if ball is not None else None)
    _log(f"  FK done ({time.perf_counter() - t_fk:.2f}s)")

    # Ground velocity (world x, world z) at each sample, for behind-the-ball azim.
    F_vel = []
    nball = len(F_ball)
    for i, b in enumerate(F_ball):
        if b is None:
            F_vel.append(None)
            continue
        lo = i - 1
        while lo >= 0 and F_ball[lo] is None:
            lo -= 1
        hi = i + 1
        while hi < nball and F_ball[hi] is None:
            hi += 1
        a = F_ball[lo] if lo >= 0 else None
        c = F_ball[hi] if hi < nball else None
        if a is not None and c is not None and grid[hi] - grid[lo] > 1e-4:
            dt = grid[hi] - grid[lo]
            F_vel.append(((c[0] - a[0]) / dt, (c[2] - a[2]) / dt))
        elif a is not None and grid[i] - grid[lo] > 1e-4:
            dt = grid[i] - grid[lo]
            F_vel.append(((b[0] - a[0]) / dt, (b[2] - a[2]) / dt))
        elif c is not None and grid[hi] - grid[i] > 1e-4:
            dt = grid[hi] - grid[i]
            F_vel.append(((c[0] - b[0]) / dt, (c[2] - b[2]) / dt))
        else:
            F_vel.append((0.0, 0.0))

    # static bounds — frame the play, not the whole park (park is a backdrop)
    allpts = np.array([pt for segs in F_segs for seg in segs for pt in seg])
    if view == "infield":
        xlo, xhi, zlo, zhi = -100, 100, -160, 22
    else:  # action / full: tight actor fit, small pad, do not grow to stadium
        xlo, xhi = float(allpts[:, 0].min()), float(allpts[:, 0].max())
        zlo, zhi = float(allpts[:, 1].min()), float(allpts[:, 1].max())
        pad = 8.0
        xlo, xhi, zlo, zhi = xlo - pad, xhi + pad, zlo - pad, zhi + pad

    fig = plt.figure(figsize=(12, 8), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_position([0.0, 0.0, 1.0, 1.0])
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ball_h = [b[1] for b in F_ball if b is not None]
    ymax = max(ball_h) if ball_h else 8.0
    z_hi = max(ymax + 25.0, 16.0)
    if include_stadium:
        z_hi = max(z_hi, 70.0)
        z_lo = -2.0
    elif include_field:
        z_lo = -1.0
    else:
        z_lo = 0.0
    ax.set_zlim(z_lo, z_hi)
    zspan = z_hi - z_lo
    ax.set_axis_off()
    ax.grid(False)
    # Draw in artist zorder, not matplotlib's 3D painter sort (that puts
    # large field tris on top of skeletons).
    ax.computed_zorder = False
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor((1, 1, 1, 0))
    ax.yaxis.pane.set_edgecolor((1, 1, 1, 0))
    ax.zaxis.pane.set_edgecolor((1, 1, 1, 0))
    ax.view_init(elev=elev, azim=_PITCHER_AZIM if view == "follow" else azim)

    def apply_bounds(cx=None, cz=None, cam_azim=None):
        if view == "follow" and cx is not None:
            ax.set_xlim(cx - zoom, cx + zoom)
            ax.set_ylim(cz + zoom, cz - zoom)  # inverted (outfield up)
            ax.set_box_aspect((2 * zoom, 2 * zoom, zspan), zoom=1.6)
            ax.view_init(elev=elev, azim=_PITCHER_AZIM if cam_azim is None else cam_azim)
        else:
            ax.set_xlim(xlo, xhi)
            ax.set_ylim(zhi, zlo)  # inverted
            ax.set_box_aspect(((xhi - xlo), (zhi - zlo), zspan), zoom=1.6)

    apply_bounds()

    def _add_park_collections():
        """Field + stadium as one collection so faces depth-sort together.

        Stadium stays slightly translucent: matplotlib has no z-buffer, so
        opaque roof slabs would paint over the diamond. Actors stay a higher
        zorder on top.
        """
        tris, cols = [], []
        if "field" in park:
            fv, ff = park["field"]
            tris.append(_mesh_plot_tris(fv, ff))
            cols.append(np.repeat([to_rgba(_FIELD_COLOR, 0.95)], len(ff), axis=0))
        if "stadium" in park:
            sv, sf = park["stadium"]
            tris.append(_mesh_plot_tris(sv, sf))
            cols.append(np.repeat([to_rgba(_STADIUM_COLOR, 0.40)], len(sf), axis=0))
        if not tris:
            return None
        pc = Poly3DCollection(np.concatenate(tris), facecolors=np.concatenate(cols),
                              linewidths=0, antialiaseds=False)
        pc.set_zorder(1)
        ax.add_collection3d(pc)
        return pc

    park_coll = _add_park_collections()

    coll = Line3DCollection([[(0, 0, 0), (0, 0, 0)]], linewidths=1.6)
    coll.set_zorder(3)
    ax.add_collection3d(coll)
    bat_coll = Poly3DCollection([], facecolor="#8a5a2b", edgecolor="#5c3a17", linewidths=0.3)
    bat_coll.set_zorder(4)
    ax.add_collection3d(bat_coll)
    ball_coll = Poly3DCollection([], facecolor="#f7f7f7", edgecolor="#cccccc", linewidths=0.2)
    ball_coll.set_zorder(6)
    ax.add_collection3d(ball_coll)
    halo = ax.scatter([], [], [], s=140, c="#ffd21e", alpha=0.35, edgecolors="none", depthshade=False)
    halo.set_zorder(7)
    trail_line, = ax.plot([], [], [], "-", color="#ff9e00", lw=1.7, alpha=0.85)
    trail_line.set_zorder(5)
    title = ax.set_title("")

    legend_types = ["pitcher", "batter", "catcher", "fielder", "umpire", "coach"]
    handles = [Line2D([0], [0], color=TYPE_COLORS[t], lw=3, label=t.capitalize()) for t in legend_types]
    handles += [Line2D([0], [0], color="#8a5a2b", lw=4, label="Bat"),
                Line2D([0], [0], marker="o", color="w", label="Ball",
                       markerfacecolor="#ffd21e", markeredgecolor="black", markersize=9)]
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.85,
              borderpad=0.4)

    trail_pts = []
    dt_frame = 1.0 / fps
    state = {
        "prev_t": None,
        "center": pitcher_look,
        "azim": _PITCHER_AZIM,
        "tracking": False,
    }
    if view == "follow":
        apply_bounds(state["center"][0], state["center"][1], state["azim"])

    def ball_polys(center):
        cx, cy, cz = center  # world x,y,z
        wv = ball_v * ball_r + np.array([cx, cy, cz])
        pv = np.column_stack([wv[:, 0], wv[:, 2], wv[:, 1]])  # to plot coords
        return [pv[f] for f in ball_f]

    def update(i):
        if i == 0:
            trail_pts.clear()
            state["prev_t"] = None
            state["center"] = pitcher_look
            state["azim"] = _PITCHER_AZIM
            state["tracking"] = False
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
            released = t_release is None or grid[i] >= t_release
            if released:
                state["tracking"] = True
            if state["tracking"] and F_center[i] is not None:
                target = F_center[i]
            else:
                target = pitcher_look
            k = 1.0 - math.exp(-dt_frame / _FOLLOW_CENTER_TAU)
            cx, cz = state["center"]
            state["center"] = (cx + (target[0] - cx) * k, cz + (target[1] - cz) * k)

            vx = vz = 0.0
            if state["tracking"] and F_vel[i] is not None:
                vx, vz = F_vel[i]
                target_az = _behind_azim(vx, vz, state["azim"])
            else:
                target_az = _PITCHER_AZIM
            state["azim"] = _slew_azim(state["azim"], target_az, dt_frame, vx=vx)
            apply_bounds(state["center"][0], state["center"][1], state["azim"])
        n = len({tuple(c) for c in F_cols[i]}) if F_cols[i] else 0
        title.set_text(f"Gameday 3D reconstruction  |  t={grid[i] - w0:5.2f}s  |  actors={n}")
        extras = [c for c in (park_coll,) if c is not None]
        return (coll, bat_coll, ball_coll, halo, trail_line, title, *extras)

    if str(out_path).lower().endswith(".png"):
        if preview_t is not None:
            fi = int(round((t0 + preview_t - w0) * fps))
        elif view == "follow" and t_release is not None:
            fi = int(round((t_release - w0) * fps))
        else:
            fi = 40
        fi = max(0, min(fi, len(grid) - 1))
        t = time.perf_counter()
        # run from 0 so follow azim/center have eased to this frame
        for k in range(fi + 1):
            update(k)
        fig.savefig(out_path, dpi=130, facecolor=fig.get_facecolor())
        plt.close(fig)
        _log(f"wrote {out_path} (preview frame, {time.perf_counter() - t:.2f}s)")
        return out_path

    bake_park = view != "follow" and bool(park)
    park_bg = None
    if bake_park:
        t = time.perf_counter()
        coll.set_segments([])
        bat_coll.set_verts([])
        ball_coll.set_verts([])
        halo._offsets3d = ([], [], [])
        trail_line.set_data([], [])
        trail_line.set_3d_properties([])
        title.set_text("")
        leg = ax.get_legend()
        if leg is not None:
            leg.set_visible(False)
        park_bg = _canvas_rgba(fig)
        if park_coll is not None:
            park_coll.remove()
            park_coll = None
        if leg is not None:
            leg.set_visible(True)
        fig.patch.set_facecolor((1, 1, 1, 0))
        ax.patch.set_facecolor((1, 1, 1, 0))
        _log(f"  baked park background {park_bg.shape[1]}x{park_bg.shape[0]} "
             f"({time.perf_counter() - t:.2f}s)")

    def _frames():
        for i in range(len(grid)):
            update(i)
            overlay = _canvas_rgba(fig)
            if park_bg is not None:
                yield _composite(park_bg, overlay)[..., :3]
            else:
                yield overlay[..., :3]

    n, dt = _write_mp4_rgb(_frames(), out_path, fps)
    plt.close(fig)
    _log(f"wrote {out_path} ({n} frames @ {fps}fps, {dt:.2f}s, {n / dt:.1f} fps)")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("play_dir")
    ap.add_argument("out", nargs="?", default="reconstruction3d.mp4")
    ap.add_argument("--view", choices=["action", "full", "infield", "follow"], default="action")
    ap.add_argument("--zoom", type=float, default=70.0, help="follow half-width (ft)")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--azim", type=float, default=-72.0)
    ap.add_argument("--elev", type=float, default=16.0)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--field", action="store_true", default=None,
                    help="draw the ballpark field mesh (overrides INCLUDE_FIELD)")
    ap.add_argument("--stadium", action="store_true", default=None,
                    help="draw the ballpark stadium mesh (overrides INCLUDE_STADIUM)")
    ap.add_argument("--ballpark", default=None, help="path to {venueId}_{ABBR}.glb")
    ap.add_argument("--t", type=float, default=None,
                    help="if out is .png, preview this seconds-from-clip-start")
    args = ap.parse_args()
    include_field = INCLUDE_FIELD if args.field is None else True
    include_stadium = INCLUDE_STADIUM if args.stadium is None else True
    reconstruct3d(args.play_dir, args.out, view=args.view, zoom=args.zoom,
                  fps=args.fps, azim=args.azim, elev=args.elev, full=args.full,
                  include_field=include_field, include_stadium=include_stadium,
                  ballpark_glb=args.ballpark, preview_t=args.t)

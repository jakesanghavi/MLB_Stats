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
  follow  : umpire-behind-the-catcher view through the pitch; after contact
            (or a sudden batted-ball turn) switches to ball tracking
            (--zoom = follow half-width).

The field-Z axis is inverted (outfield up), matching the 2D animator.

    python reconstruct3d.py <play_dir> [out.mp4] [--view follow] [--zoom 70]
                            [--fps 20] [--azim -72] [--elev 16] [--full]
                            [--field] [--stadium]
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
INCLUDE_FIELD = False      # dirt/grass plane from the ballpark glb
INCLUDE_STADIUM = False    # bowl / stands from the ballpark glb
_BALL_GAP_BREAK = 0.3
_BAT_MODEL_LEN = 2.843  # bat.glb knob->barrel extent (ft)
ASSETS = Path(__file__).resolve().parent / "assets"
_FIELD_COLOR = "#5b9e4a"
_DIRT_COLOR = "#c2a36b"
_STADIUM_COLOR = "#c4beb3"
# Umpire-over-the-catcher view: from +Z (behind home) looking at the mound.
# azim 84 is a hair toward 1B so we look over the umpire's shoulder, not
# through his torso. elev is steep enough to clear the mask.
_PLATE_AZIM = 84.0
_PLATE_ELEV = 34.0
_PLATE_HALF_X = 15.0   # ft, batter's boxes + a slice of the infield
_PLATE_Z_FAR = -75.0   # past the mound
_PLATE_H = 14.0
_PLATE_BEHIND = 14.0   # ft behind the rearmost plate actor (umpire/catcher)


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


def _mesh_plot_tris(verts, faces):
    """world (x, y=height, z) triangles -> plot (x, z, y) triangles."""
    pv = np.column_stack([verts[:, 0], verts[:, 2], verts[:, 1]])
    return pv[faces]


def _face_metrics(verts, faces):
    """Per-face longest/shortest edge, area, aspect, centroid."""
    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    e1 = np.linalg.norm(b - a, axis=1)
    e2 = np.linalg.norm(c - b, axis=1)
    e3 = np.linalg.norm(a - c, axis=1)
    emax = np.maximum(np.maximum(e1, e2), e3)
    emin = np.minimum(np.minimum(e1, e2), e3)
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    aspect = emax / np.maximum(emin, 1e-9)
    cent = (a + b + c) / 3.0
    return emax, emin, area, aspect, cent


def _clean_field_faces(verts, faces, max_aspect=10.0, max_sliver_area=50.0):
    """Drop decimation slivers. Those thin tris rasterize as white streaks;
    the large remaining faces are the actual grass cover."""
    _emax, _emin, area, aspect, cent = _face_metrics(verts, faces)
    sliver = (aspect > max_aspect) & (area < max_sliver_area)
    # Field node also carries a chunk of backstop net behind the plate.
    backstop = (cent[:, 1] > 5.0) & (cent[:, 2] > 0.0)
    return verts, faces[~(sliver | backstop)]


def _clean_stadium_faces(verts, faces, min_y=8.0, max_aspect=25.0,
                         max_z_vertex=8.0, max_z_span=120.0, max_edge=150.0):
    """Drop ground overlays, the backstop, wrapping bowl tris, and slivers."""
    emax, _emin, _area, aspect, cent = _face_metrics(verts, faces)
    zmin = verts[faces][:, :, 2].min(axis=1)
    zmax = verts[faces][:, :, 2].max(axis=1)
    keep = ((cent[:, 1] >= min_y)
            & (aspect < max_aspect)
            & (zmax < max_z_vertex)
            & ((zmax - zmin) < max_z_span)
            & (emax < max_edge))
    return verts, faces[keep]


def _quad_tiles(x0, x1, z0, z1, y, step=30.0):
    """Small ground quads. One huge polygon loses matplotlib's painter sort
    and shows up as white holes; local tiles composite correctly."""
    xs = np.arange(min(x0, x1), max(x0, x1), step)
    zs = np.arange(min(z0, z1), max(z0, z1), step)
    tris = []
    for x in xs:
        for z in zs:
            p = np.array([[x, z, y],
                          [x + step, z, y],
                          [x + step, z + step, y],
                          [x, z + step, y]], float)
            tris.append(p[[0, 1, 2]])
            tris.append(p[[0, 2, 3]])
    return tris


def _infield_dirt():
    """Tan infield skin (diamond a bit larger than the 90-ft square)."""
    y = 0.04  # sit just above the grass tiles
    pts = np.array([
        [0.0, 8.0, y],       # slightly behind the plate
        [78.0, -63.0, y],    # past 1B
        [0.0, -142.0, y],    # past 2B
        [-78.0, -63.0, y],   # past 3B
    ], float)
    return [pts[[0, 1, 2]], pts[[0, 2, 3]]]


def _park_collection(tris, color, alpha):
    from matplotlib.colors import to_rgba
    if not tris:
        return None
    fc = np.repeat([to_rgba(color, alpha)], len(tris), axis=0)
    return Poly3DCollection(tris, facecolors=fc, linewidths=0,
                            edgecolors="none", antialiaseds=False, shade=False)


def _pitch_contact_times(reader, ball_track):
    """Pitch time and first contact / odd-turn time (absolute seconds)."""
    t_pitch = t_hit = t_play1 = None
    for t, dt, data in reader.events():
        data = data or {}
        if dt == 7:
            if data.get("action") == 0 and t_pitch is None:
                t_pitch = t
            elif data.get("action") == 1 and t_play1 is None:
                t_play1 = t
    for f in reader.frames:
        if t_hit is not None:
            break
        for p in f.get("ballPolynomials") or []:
            if p.get("dataType") == 2:  # BallHitData
                t_hit = f["time"]
                break
    t_flip = None
    if t_pitch is not None:
        prev, saw_in = None, False
        for t, x, y, z in ball_track:
            if t < t_pitch - 0.05:
                prev = (t, x, y, z)
                continue
            if prev is None:
                prev = (t, x, y, z)
                continue
            dt = t - prev[0]
            if dt > 1e-4:
                vz = (z - prev[3]) / dt
                vx = (x - prev[1]) / dt
                if vz > 40:
                    saw_in = True
                if saw_in and (vz < -25 or abs(vx) > 80):
                    t_flip = t
                    break
            prev = (t, x, y, z)
    t_contact = t_hit or t_flip or t_play1
    return t_pitch, t_contact


def _plate_near_z(reader, t_pitch):
    """Field-Z of a camera plane behind the umpire/catcher (not inside them)."""
    z_back = 8.0
    if t_pitch is None:
        return z_back + _PLATE_BEHIND
    for f in reader.frames:
        if abs(f["time"] - t_pitch) > 0.2:
            continue
        for a in f.get("actorPoses") or []:
            rp = a.get("rootPos")
            if not rp:
                continue
            typ = reader.actor_type(a["uid"])
            if typ in ("plate-umpire", "catcher") or (typ == "umpire" and rp["z"] > 0):
                z_back = max(z_back, rp["z"])
        break
    return z_back + _PLATE_BEHIND


def reconstruct3d(play_dir, out_path="reconstruction3d.mp4", view="action",
                  zoom=70.0, fps=20, azim=-72.0, elev=16.0, full=False,
                  include_field=None, include_stadium=None, ballpark_glb=None):
    if include_field is None:
        include_field = INCLUDE_FIELD
    if include_stadium is None:
        include_stadium = INCLUDE_STADIUM
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
    print(f"window {w0 - t0:.2f}..{w1 - t0:.2f}s ({w1 - w0:.2f}s) view={view} "
          f"field={include_field} stadium={include_stadium}")

    park = {}
    if include_field or include_stadium:
        from stadium import find_ballpark_glb, load_park_meshes
        glb_path = find_ballpark_glb(reader, explicit=ballpark_glb)
        park = load_park_meshes(glb_path, want_field=include_field,
                                want_stadium=include_stadium)

    pose_tracks = _actor_pose_tracks(reader)
    raw_ball = reader.ball_track()
    ball_track = [(t, (x, y, z)) for t, x, y, z in raw_ball]
    bat_track = _bat_track(reader)
    t_pitch, t_contact = _pitch_contact_times(reader, raw_ball)
    z_near = _plate_near_z(reader, t_pitch)
    grid = np.arange(w0, w1, 1.0 / fps)
    if view == "follow":
        print(f"  follow: plate-cam until contact "
              f"t_pitch={None if t_pitch is None else t_pitch - t0:.2f}s "
              f"t_contact={None if t_contact is None else t_contact - t0:.2f}s "
              f"z_near={z_near:.1f}ft")

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
    if include_stadium:
        ax.set_zlim(-2, 70)
        zspan = 72
    elif include_field:
        ax.set_zlim(-1, 16)
        zspan = 17
    else:
        ax.set_zlim(0, 10)
        zspan = 12
    ax.set_axis_off()
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor((1, 1, 1, 0))
    ax.yaxis.pane.set_edgecolor((1, 1, 1, 0))
    ax.zaxis.pane.set_edgecolor((1, 1, 1, 0))
    ax.view_init(elev=_PLATE_ELEV if view == "follow" else elev,
                 azim=_PLATE_AZIM if view == "follow" else azim)

    def _set_stadium_visible(on):
        if stadium_coll is not None:
            stadium_coll.set_visible(on)

    def apply_plate():
        # Hide the bowl: matplotlib will otherwise paint backstop tris
        # across the near plane and fill the umpire camera with beige.
        _set_stadium_visible(False)
        ax.set_xlim(-_PLATE_HALF_X, _PLATE_HALF_X)
        ax.set_ylim(_PLATE_Z_FAR, z_near)  # look from behind home toward mound
        ax.set_zlim(0, _PLATE_H)
        ax.set_box_aspect((2 * _PLATE_HALF_X, z_near - _PLATE_Z_FAR, _PLATE_H), zoom=1.45)
        ax.view_init(elev=_PLATE_ELEV, azim=_PLATE_AZIM)

    def apply_bounds(cx=None, cz=None):
        _set_stadium_visible(True)
        if view == "follow" and cx is not None:
            ax.set_xlim(cx - zoom, cx + zoom)
            ax.set_ylim(cz + zoom, cz - zoom)  # inverted (outfield up)
            ax.set_zlim(-2, 70) if include_stadium else ax.set_zlim(-1, 16) if include_field else ax.set_zlim(0, 10)
            ax.set_box_aspect((2 * zoom, 2 * zoom, zspan), zoom=1.6)
            ax.view_init(elev=elev, azim=azim)
        else:
            ax.set_xlim(xlo, xhi)
            ax.set_ylim(zhi, zlo)  # inverted
            ax.set_box_aspect(((xhi - xlo), (zhi - zlo), zspan), zoom=1.6)

    def _add_park_mesh(kind, color, alpha):
        if kind not in park:
            return None
        verts, faces = park[kind]
        if kind == "field":
            verts, faces = _clean_field_faces(verts, faces)
        elif kind == "stadium":
            verts, faces = _clean_stadium_faces(verts, faces)
        if len(faces) == 0:
            return None
        pc = _park_collection(list(_mesh_plot_tris(verts, faces)), color, alpha)
        if pc is not None:
            ax.add_collection3d(pc)
        return pc

    # grass tiles + infield dirt first so mesh holes aren't white paper
    if include_field:
        grass = _park_collection(_quad_tiles(-150, 150, -400, 20, y=-0.2, step=30.0),
                                 _FIELD_COLOR, 1.0)
        if grass is not None:
            ax.add_collection3d(grass)
        dirt = _park_collection(_infield_dirt(), _DIRT_COLOR, 1.0)
        if dirt is not None:
            ax.add_collection3d(dirt)
    field_coll = _add_park_mesh("field", _FIELD_COLOR, 0.95)
    stadium_coll = _add_park_mesh("stadium", _STADIUM_COLOR, 0.38)

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
    if include_field:
        handles.append(Line2D([0], [0], color=_FIELD_COLOR, lw=6, label="Field"))
    if include_stadium:
        handles.append(Line2D([0], [0], color=_STADIUM_COLOR, lw=6, label="Stadium"))
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.85,
              borderpad=0.4)

    # initial follow center: first tracked ball, else infield
    first_c = next((c for c in F_center if c is not None), (0.0, -60.0))
    trail_pts = []
    state = {"prev_t": None, "center": first_c, "mode": "plate"}
    if view == "follow":
        apply_plate()
    else:
        apply_bounds()

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
            tracking = t_contact is not None and grid[i] >= t_contact
            if tracking:
                if F_center[i] is not None:
                    state["center"] = F_center[i]
                apply_bounds(state["center"][0], state["center"][1])
                state["mode"] = "ball"
            else:
                apply_plate()
                state["mode"] = "plate"
        n = len({tuple(c) for c in F_cols[i]}) if F_cols[i] else 0
        title.set_text(f"Gameday 3D reconstruction  |  t={grid[i] - w0:5.2f}s  |  actors={n}")
        extras = [c for c in (field_coll, stadium_coll) if c is not None]
        return (coll, bat_coll, ball_coll, halo, trail_line, title, *extras)

    if str(out_path).lower().endswith(".png"):
        if view == "follow" and t_pitch is not None:
            fi = int(round((t_pitch - w0) * fps))
        else:
            fi = 40
        fi = max(0, min(fi, len(grid) - 1))
        update(fi)
        fig.savefig(out_path, dpi=130, facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"wrote {out_path} (preview frame)")
        return out_path
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
    args = ap.parse_args()
    include_field = INCLUDE_FIELD if args.field is None else True
    include_stadium = INCLUDE_STADIUM if args.stadium is None else True
    reconstruct3d(args.play_dir, args.out, view=args.view, zoom=args.zoom,
                  fps=args.fps, azim=args.azim, elev=args.elev, full=args.full,
                  include_field=include_field, include_stadium=include_stadium,
                  ballpark_glb=args.ballpark)

"""Name lookup and start-of-play role/position guesses for POV views."""
import csv
import math
from pathlib import Path

import numpy as np

# world xz: +X toward 1B, +Z toward the catcher, outfield −Z
_BAG_1B = (63.64, -63.64)
_BAG_2B = (0.0, -127.28)
_BAG_3B = (-63.64, -63.64)

DEFENSE_ANCHORS = {
    "P": ((0.0, -60.5), 28.0),
    "C": ((0.0, 4.0), 18.0),
    "1B": (_BAG_1B, 48.0),
    "2B": ((18.0, -122.0), 42.0),
    "SS": ((-18.0, -122.0), 42.0),
    "3B": (_BAG_3B, 48.0),
    "LF": ((-110.0, -250.0), 95.0),
    "CF": ((0.0, -300.0), 95.0),
    "RF": ((110.0, -250.0), 95.0),
}
RUNNER_BAGS = {
    "1B runner": (_BAG_1B, 22.0),
    "2B runner": (_BAG_2B, 22.0),
    "3B runner": (_BAG_3B, 22.0),
}
UMPIRE_ANCHORS = {
    "Home plate umpire": ((0.0, 8.0), 25.0),
    "First base umpire": ((85.0, -85.0), 50.0),
    "Second base umpire": ((0.0, -165.0), 55.0),
    "Third base umpire": ((-85.0, -85.0), 50.0),
}
DEFENSE_ORDER = ["P", "C", "1B", "2B", "3B", "SS", "LF", "CF", "RF"]
OFFENSE_ORDER = ["Batter", "1B runner", "2B runner", "3B runner"]
OFFICIAL_ORDER = [
    "Home plate umpire", "First base umpire", "Second base umpire",
    "Third base umpire", "First base coach", "Third base coach",
]


def _repo_root():
    return Path(__file__).resolve().parents[2]


def _datapack_dirs():
    root = _repo_root()
    out = []
    for name in ("DataPack", "Datapack", "datapack"):
        d = root / name / "Misc_Data"
        if d.is_dir():
            out.append(d)
    return out


def load_bios(extra_dirs=None):
    """MLB playerId -> fullName from batter/pitcher bios CSVs."""
    names = {}
    dirs = list(_datapack_dirs())
    if extra_dirs:
        dirs.extend(Path(d) for d in extra_dirs)
    files = []
    for d in dirs:
        files.extend(sorted(d.glob("*batter_bios*.csv")))
        files.extend(sorted(d.glob("*pitcher_bios*.csv")))
        files.extend(sorted(d.glob("batter_bios_*.csv")))
        files.extend(sorted(d.glob("pitcher_bios_*.csv")))
    seen = set()
    for path in files:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        try:
            with path.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    pid = row.get("id") or row.get("playerId") or row.get("mlbam")
                    name = row.get("fullName") or row.get("name")
                    if not pid or not name:
                        continue
                    try:
                        names[int(pid)] = name.strip()
                    except ValueError:
                        continue
        except OSError:
            continue
    return names


def names_from_boxscore(metadata):
    names = {}
    teams = ((metadata or {}).get("boxscore") or {}).get("teams") or {}
    for side in teams.values():
        players = (side or {}).get("players") or {}
        if not isinstance(players, dict):
            continue
        for rec in players.values():
            person = (rec or {}).get("person") or {}
            pid = person.get("id")
            name = person.get("nameFirstLast") or person.get("fullName") or person.get("boxscoreName")
            if pid and name:
                names[int(pid)] = name
    return names


def resolve_name(player_id, bios, boxscore_names):
    if player_id is None:
        return None
    try:
        pid = int(player_id)
    except (TypeError, ValueError):
        return None
    if pid <= 0:
        return None
    return bios.get(pid) or boxscore_names.get(pid)


def _dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _xz(actor):
    p = actor.get("start")
    if not p:
        return None
    return (float(p[0]), float(p[2]))


def _label(slot, name):
    if slot and name:
        return f"{slot} - {name}"
    return slot or name


PLATE_LOOK = (0.0, 2.5, 0.0)
MOUND_LOOK = (0.0, 5.0, -60.5)
BAG_LOOK = {
    "1B runner": (0.0, 3.0, -127.28),
    "2B runner": (-63.64, 3.0, -63.64),
    "3B runner": (0.0, 2.5, 0.0),
}
OFFENSE_LOOK = {"Batter", "C", "1B runner", "2B runner", "3B runner"}

# Code-only look mode (not in the GUI). Change HEAD_POSE here or pass --head-pose.
# FOLLOW_NECK   bind-pose face on the tracked neck
# ALWAYS_BALL   eyes look at the ball, else mound/plate
# SMART_VISION  ball/role target if it sits in the neck cone, else clamp
# EASY_VISION   ALWAYS_BALL until contact, then FOLLOW_NECK
HEAD_POSES = ("FOLLOW_NECK", "ALWAYS_BALL", "SMART_VISION", "EASY_VISION")
HEAD_POSE = "EASY_VISION"
SMART_CONE_DEG = 80.0
EYE_PUSH = 0.35
# Ease EASY_VISION only across pitch release and bat contact. 0 elsewhere.
EASY_BLEND_S = 0.18


def _unit(v):
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return None
    return v / n


def _ang_deg(a, b):
    a, b = _unit(a), _unit(b)
    if a is None or b is None:
        return 180.0
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))


def _slerp_dir(a, b, f):
    a, b = _unit(a), _unit(b)
    if a is None:
        return b
    if b is None:
        return a
    f = float(np.clip(f, 0.0, 1.0))
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if dot > 0.9995:
        return _unit(a + f * (b - a))
    theta = math.acos(dot)
    s = math.sin(theta)
    return (math.sin((1.0 - f) * theta) * a + math.sin(f * theta) * b) / s


def _turn_toward(neck, want, max_deg):
    ang = _ang_deg(neck, want)
    if ang <= max_deg:
        return _unit(want)
    if ang < 1e-3:
        return _unit(neck)
    return _slerp_dir(neck, want, max_deg / ang)


def _basis_from_fwd(eye, fwd, up=None, push=EYE_PUSH):
    fwd = _unit(fwd)
    if fwd is None:
        return None
    if up is None:
        up = np.array([0.0, 1.0, 0.0])
    up = np.asarray(up, dtype=float)
    upu = _unit(up)
    if upu is not None and abs(float(np.dot(fwd, upu))) > 0.95:
        up = np.array([0.0, 0.0, 1.0])
    up = up - fwd * float(np.dot(up, fwd))
    up = _unit(up)
    if up is None:
        return None
    eye = np.asarray(eye, dtype=float)
    return eye + fwd * push, fwd, up


def role_fallback(slot):
    if slot in BAG_LOOK:
        return BAG_LOOK[slot]
    if slot in OFFENSE_LOOK:
        return MOUND_LOOK
    return PLATE_LOOK


def attention_targets(slot, ball_xyz=None):
    """(priority, xyz) — lower priority wins. Ball, then next bag, then mound/plate."""
    out = []
    if ball_xyz is not None:
        out.append((0, (float(ball_xyz[0]), float(ball_xyz[1]), float(ball_xyz[2]))))
    if slot in BAG_LOOK:
        out.append((1, BAG_LOOK[slot]))
    out.append((2, role_fallback(slot)))
    return out


def smart_forward(eye, neck_fwd, slot, ball_xyz=None, cone_deg=SMART_CONE_DEG):
    """Best attention dir inside the neck cone, else clamp toward it."""
    neck = _unit(neck_fwd)
    if neck is None:
        return None
    eye = np.asarray(eye, dtype=float)
    ranked = []
    for pri, tgt in attention_targets(slot, ball_xyz):
        d = _unit(np.asarray(tgt, dtype=float) - eye)
        if d is None:
            continue
        ranked.append((pri, _ang_deg(neck, d), d))
    if not ranked:
        return neck
    in_cone = [r for r in ranked if r[1] <= cone_deg]
    if in_cone:
        in_cone.sort(key=lambda x: (x[0], x[1]))
        return in_cone[0][2]
    ranked.sort(key=lambda x: x[0])
    return _turn_toward(neck, ranked[0][2], cone_deg)


def easy_event_weight(t, event_t, tau=EASY_BLEND_S):
    """0 before ``event_t``, 1 after ``event_t + tau``, smoothstep in between."""
    if event_t is None or t is None:
        return None
    if t <= event_t:
        return 0.0
    if t >= event_t + tau:
        return 1.0
    u = (t - event_t) / tau
    return u * u * (3.0 - 2.0 * u)


def _blend_basis(a, b, w):
    if a is None:
        return b
    if b is None:
        return a
    w = float(np.clip(w, 0.0, 1.0))
    pos = (1.0 - w) * np.asarray(a[0], float) + w * np.asarray(b[0], float)
    fwd = _slerp_dir(a[1], b[1], w)
    up = (1.0 - w) * np.asarray(a[2], float) + w * np.asarray(b[2], float)
    up = up - fwd * float(np.dot(up, fwd))
    up = _unit(up)
    if up is None:
        up = np.asarray(b[2], float)
    return pos, fwd, up


def look_pose(mode, eye, neck_fwd, neck_up=None, slot=None, ball_xyz=None,
              contacted=False, t=None, t_release=None, t_contact=None,
              ball_xyz_prerelease=None):
    """(pos, fwd, up) for a HEAD_POSE mode. ``eye`` is the unpushed eye midpoint.

    EASY_VISION eases only at ``t_release`` and ``t_contact``. Far from those
    instants the look matches a hard switch (ball before contact, neck after).
    """
    mode = mode if mode in HEAD_POSES else HEAD_POSE
    if mode == "EASY_VISION":
        w_c = easy_event_weight(t, t_contact)
        if w_c is None:
            w_c = 1.0 if contacted else 0.0
        ball_look = look_from_eye(eye, look_target(slot, ball_xyz))
        neck_look = _basis_from_fwd(eye, neck_fwd, neck_up)
        if w_c >= 1.0:
            return neck_look
        if w_c > 0.0:
            return _blend_basis(ball_look, neck_look, w_c)
        w_r = easy_event_weight(t, t_release)
        if w_r is not None and 0.0 < w_r < 1.0:
            pre = look_from_eye(eye, look_target(slot, ball_xyz_prerelease))
            return _blend_basis(pre, ball_look, w_r)
        return ball_look
    if mode == "FOLLOW_NECK":
        return _basis_from_fwd(eye, neck_fwd, neck_up)
    if mode == "ALWAYS_BALL":
        return look_from_eye(eye, look_target(slot, ball_xyz))
    # SMART_VISION
    fwd = smart_forward(eye, neck_fwd, slot, ball_xyz)
    return _basis_from_fwd(eye, fwd)


def look_target(slot, ball_xyz=None):
    """Attention point: the ball when we have it, else mound/plate by role.

    The wire has no gaze. Bind-pose head −X is just the neck pointing a static
    skull, which jitters and often faces the stands or the dirt.
    """
    if ball_xyz is not None:
        return (float(ball_xyz[0]), float(ball_xyz[1]), float(ball_xyz[2]))
    if slot in OFFENSE_LOOK:
        return MOUND_LOOK
    return PLATE_LOOK


def look_from_eye(eye, target, push=0.35):
    """Camera at the eyes, looking at ``target``, world-up."""
    pos = np.asarray(eye, dtype=float)
    tgt = np.asarray(target, dtype=float)
    fwd = tgt - pos
    n = float(np.linalg.norm(fwd))
    if n < 1e-6:
        return None
    fwd = fwd / n
    up = np.array([0.0, 1.0, 0.0])
    if abs(float(np.dot(fwd, up))) > 0.95:
        up = np.array([0.0, 0.0, 1.0])
    up = up - fwd * float(np.dot(up, fwd))
    un = float(np.linalg.norm(up))
    if un < 1e-6:
        return None
    up = up / un
    return pos + fwd * push, fwd, up


def smooth_head_series(heads, fps, tau_pos=0.08, tau_fwd=0.15):
    """EMA on exported [pos,fwd,up] samples. ``heads`` items are lists or None."""
    if fps <= 0:
        return heads
    a_pos = 1.0 - math.exp(-1.0 / (fps * tau_pos))
    a_fwd = 1.0 - math.exp(-1.0 / (fps * tau_fwd))
    out = []
    prev = None
    for h in heads:
        if h is None:
            out.append(None)
            prev = None
            continue
        pos = np.array(h[:3], dtype=float)
        fwd = np.array(h[3:6], dtype=float)
        up = np.array(h[6:9], dtype=float)
        if prev is None:
            prev = (pos, fwd, up)
        else:
            pos = prev[0] * (1.0 - a_pos) + pos * a_pos
            fwd = prev[1] * (1.0 - a_fwd) + fwd * a_fwd
            fn = float(np.linalg.norm(fwd))
            fwd = fwd / fn if fn > 1e-8 else prev[1]
            up = prev[2] * (1.0 - a_fwd) + up * a_fwd
            up = up - fwd * float(np.dot(up, fwd))
            un = float(np.linalg.norm(up))
            up = up / un if un > 1e-8 else prev[2]
            prev = (pos, fwd, up)
        out.append([float(v) for v in (*prev[0], *prev[1], *prev[2])])
    return out


def classify_views(actors):
    """Guess defense / batter / runners / officials from start-of-play xz.

    ``actors`` items: {uid, type, playerId, name, start: [x,y,z] or None}
    Returns {players: [...], officials: [...]} view dicts.
    """
    present = [a for a in actors if _xz(a)]
    used = set()
    slots = {}  # slot -> actor

    def take(slot, actor):
        slots[slot] = actor
        used.add(id(actor))

    # Pitcher / catcher from Gameday type when they are on the dirt.
    for a in present:
        xz = _xz(a)
        if a.get("type") == "pitcher" and _dist(xz, DEFENSE_ANCHORS["P"][0]) <= 35:
            take("P", a)
            break
    for a in present:
        if id(a) in used:
            continue
        xz = _xz(a)
        if a.get("type") == "catcher" and _dist(xz, DEFENSE_ANCHORS["C"][0]) <= 22:
            take("C", a)
            break

    defense_pool = [
        a for a in present
        if id(a) not in used and a.get("type") in ("fielder", "pitcher", "catcher")
    ]
    pairs = []
    for a in defense_pool:
        xz = _xz(a)
        for slot, (pt, lim) in DEFENSE_ANCHORS.items():
            if slot in slots:
                continue
            d = _dist(xz, pt)
            if d <= lim:
                pairs.append((d, slot, a))
    pairs.sort(key=lambda x: x[0])
    for d, slot, a in pairs:
        if slot in slots or id(a) in used:
            continue
        take(slot, a)

    # Batter: labeled batter/runner nearest home, or anyone in the box.
    offense_pool = [
        a for a in present
        if id(a) not in used and a.get("type") in ("batter", "runner")
    ]
    home_cands = []
    for a in offense_pool:
        xz = _xz(a)
        d = _dist(xz, (0.0, 0.0))
        if d <= 16:
            home_cands.append((d, a))
    if not home_cands:
        for a in present:
            if id(a) in used:
                continue
            if a.get("type") in ("catcher", "plate-umpire", "umpire"):
                continue
            xz = _xz(a)
            d = _dist(xz, (0.0, 0.0))
            if d <= 10:
                home_cands.append((d, a))
    if home_cands:
        home_cands.sort(key=lambda x: x[0])
        take("Batter", home_cands[0][1])

    runner_pool = [a for a in present if id(a) not in used]
    rpairs = []
    for a in runner_pool:
        if a.get("type") not in ("batter", "runner", "fielder"):
            continue
        xz = _xz(a)
        for slot, (pt, lim) in RUNNER_BAGS.items():
            d = _dist(xz, pt)
            if d <= lim:
                rpairs.append((d, slot, a))
    rpairs.sort(key=lambda x: x[0])
    for d, slot, a in rpairs:
        if slot in slots or id(a) in used:
            continue
        # Prefer Gameday batter/runner over a leftover fielder.
        if a.get("type") == "fielder" and any(
            b.get("type") in ("batter", "runner") and id(b) not in used
            and _dist(_xz(b), RUNNER_BAGS[slot][0]) <= RUNNER_BAGS[slot][1]
            for b in runner_pool
        ):
            continue
        take(slot, a)

    # Officials.
    umps = [a for a in present if id(a) not in used
            and a.get("type") in ("umpire", "plate-umpire")]
    for a in umps:
        if a.get("type") == "plate-umpire":
            take("Home plate umpire", a)
            break
    ump_pairs = []
    for a in umps:
        if id(a) in used:
            continue
        xz = _xz(a)
        for slot, (pt, lim) in UMPIRE_ANCHORS.items():
            if slot in slots:
                continue
            d = _dist(xz, pt)
            if d <= lim:
                ump_pairs.append((d, slot, a))
    ump_pairs.sort(key=lambda x: x[0])
    for d, slot, a in ump_pairs:
        if slot in slots or id(a) in used:
            continue
        take(slot, a)

    coaches = [a for a in present if id(a) not in used and a.get("type") == "coach"]
    for a in coaches:
        xz = _xz(a)
        slot = "First base coach" if xz[0] >= 0 else "Third base coach"
        if slot not in slots:
            take(slot, a)

    leftover_named = [
        a for a in present
        if id(a) not in used and a.get("name")
        and a.get("type") not in ("umpire", "plate-umpire", "coach")
    ]

    players = []
    for slot in DEFENSE_ORDER + OFFENSE_ORDER:
        a = slots.get(slot)
        if not a:
            continue
        players.append({
            "uid": a["uid"],
            "slot": slot,
            "name": a.get("name"),
            "label": _label(slot, a.get("name")),
            "group": "offense" if slot in OFFENSE_ORDER else "defense",
        })
    for a in leftover_named:
        players.append({
            "uid": a["uid"],
            "slot": None,
            "name": a.get("name"),
            "label": a["name"],
            "group": "defense",
        })

    officials = []
    for slot in OFFICIAL_ORDER:
        a = slots.get(slot)
        if not a:
            continue
        officials.append({
            "uid": a["uid"],
            "slot": slot,
            "name": None,
            "label": slot,
            "group": "official",
        })
    return {"players": players, "officials": officials}

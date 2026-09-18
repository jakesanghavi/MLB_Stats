"""Name lookup and start-of-play role/position guesses for POV views."""
import csv
from pathlib import Path

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

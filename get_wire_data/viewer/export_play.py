"""Export a downloaded play to compact JSON for the three.js viewer."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from read_play import PlayReader, sample_ball
from rig import RigSkeleton
from reconstruct3d import (
    TYPE_COLORS, _actor_pose_tracks, _sample_pose, _bat_track, _sample_bat,
    _pitch_release_time, _contact_time, _pitcher_xz,
)
from stadium import find_ballpark_glb, _home_abbr, _venue_id
from player_assets import actor_side, ensure_player_assets, outfit_role, player_sides
from views import (
    HEAD_POSE, HEAD_POSES, load_bios, names_from_boxscore, resolve_name,
    classify_views, smooth_head_series,
)


def _r(v, n=3):
    return None if v is None else round(float(v), n)


def _xyz(p, n=3):
    return [_r(p[0], n), _r(p[1], n), _r(p[2], n)]


def _bone_order(pose_tracks):
    names = set()
    for track in pose_tracks.values():
        for _t, pose in track:
            names.update(pose.get("jointRotations") or {})
    names.discard("joint_Pelvis")
    return ["joint_Pelvis"] + sorted(names)


def _pack_pose(pose, bones):
    rp = pose["rootPos"]
    out = [_r(rp["x"], 4), _r(rp["y"], 4), _r(rp["z"], 4),
           _r(pose.get("scale", 1.0), 4)]
    joints = pose.get("jointRotations") or {}
    for name in bones:
        q = joints.get(name)
        if not q or len(q) < 4:
            out.extend([0.0, 0.0, 0.0, 0.0])
        else:
            out.extend([_r(q[0], 5), _r(q[1], 5), _r(q[2], 5), _r(q[3], 5)])
    return out


def _fill_missing_sides(actors_out):
    """Batters/coaches without a roster id inherit the batting-side majority."""
    known = [a["side"] for a in actors_out if a.get("side")]
    batting = None
    batters = [a["side"] for a in actors_out
               if a.get("type") == "batter" and a.get("side")]
    if batters:
        batting = max(set(batters), key=batters.count)
    elif known:
        # fielders are the other side
        fielding = max(set(known), key=known.count)
        batting = "away" if fielding == "home" else "home"
    fielding = "away" if batting == "home" else "home" if batting else None
    for a in actors_out:
        if a.get("side"):
            continue
        if a.get("type") in ("umpire", "plate-umpire"):
            continue
        if a.get("type") in ("batter", "coach"):
            a["side"] = batting
        else:
            a["side"] = fielding


def export_play(play_dir, out_json, fps=20.0, full=False, head_pose=None):
    t0_wall = time.perf_counter()
    play_dir = Path(play_dir)
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    reader = PlayReader(play_dir)
    if not reader.frames:
        raise SystemExit("no frames")
    rig = RigSkeleton()
    clip0 = reader.frames[0]["time"]
    if full:
        w0, w1 = reader.frames[0]["time"], reader.frames[-1]["time"]
    else:
        w0, w1 = reader.play_window()
    grid = np.arange(w0, w1, 1.0 / fps)
    pose_tracks = _actor_pose_tracks(reader)
    ball_track = list(reader.ball_track())
    bat_track = _bat_track(reader)
    t_release = _pitch_release_time(reader)
    t_contact = _contact_time(reader)
    pitch_xz = _pitcher_xz(reader, t_release if t_release is not None else w0)

    actors_out = []
    uids = sorted(pose_tracks)
    bios = load_bios()
    box_names = names_from_boxscore(reader.metadata)
    sides = player_sides(reader.metadata)
    bones = _bone_order(pose_tracks)
    classify_in = []
    for uid in uids:
        pid = reader.actor_label(uid).get("actor")
        name = resolve_name(pid, bios, box_names)
        start_pose = _sample_pose(pose_tracks[uid], w0)
        if start_pose is None:
            for tt, p in pose_tracks[uid]:
                if w0 <= tt <= w1:
                    start_pose = p
                    break
        start = None
        if start_pose and start_pose.get("rootPos"):
            rp = start_pose["rootPos"]
            start = [_r(rp["x"]), _r(rp["y"]), _r(rp["z"])]
        atype = reader.actor_type(uid)
        classify_in.append({
            "uid": int(uid) if str(uid).isdigit() else uid,
            "type": atype,
            "playerId": pid,
            "name": name,
            "start": start,
        })
        actors_out.append({
            "uid": int(uid) if str(uid).isdigit() else uid,
            "playerId": pid if isinstance(pid, int) and pid > 0 else None,
            "name": name,
            "type": atype,
            "outfit": outfit_role(atype),
            "side": actor_side(atype, pid, sides),
            "color": TYPE_COLORS.get(atype, TYPE_COLORS["unknown"]),
            "frames": [],
            "pose": [],
            "head": [],
        })
    _fill_missing_sides(actors_out)
    views = classify_views(classify_in)
    uid_slot = {v["uid"]: v.get("slot") for v in views["players"] + views["officials"]}
    mode = head_pose if head_pose in HEAD_POSES else HEAD_POSE
    for a in actors_out:
        a["slot"] = uid_slot.get(a["uid"])

    ball_frames = []
    bat_frames = []
    amin = np.array([1e9, 1e9, 1e9])
    amax = np.array([-1e9, -1e9, -1e9])

    for t in grid:
        b = sample_ball(ball_track, t)
        ball_frames.append(_xyz(b, 5) if b is not None else None)
        bh = _sample_bat(bat_track, t)
        if bh:
            bat_frames.append({"handle": _xyz(bh[0], 5), "head": _xyz(bh[1], 5)})
        else:
            bat_frames.append(None)
        for ai, uid in enumerate(uids):
            pose = _sample_pose(pose_tracks[uid], t)
            if pose is None:
                actors_out[ai]["frames"].append(None)
                actors_out[ai]["pose"].append(None)
                actors_out[ai]["head"].append(None)
                continue
            actors_out[ai]["pose"].append(_pack_pose(pose, bones))
            rp = (pose["rootPos"]["x"], pose["rootPos"]["y"], pose["rootPos"]["z"])
            mats = rig.fk_matrices(rp, pose["jointRotations"], pose.get("scale", 1.0))
            wp = {j: mats[j][:3, 3] for j in range(len(mats)) if mats[j] is not None}
            segs = []
            for p, c in rig.segments(wp):
                segs.extend(_xyz(p))
                segs.extend(_xyz(c))
                amin = np.minimum(amin, p)
                amax = np.maximum(amax, p)
            actors_out[ai]["frames"].append(segs)
            eye = rig.eye_position(mats)
            hp = rig.head_pose(mats)
            if eye is None or hp is None:
                actors_out[ai]["head"].append(None)
                continue
            _pos, neck_fwd, neck_up = hp
            actors_out[ai]["head"].append(
                [_r(v) for v in (*eye, *neck_fwd, *neck_up)]
            )

    for a in actors_out:
        a["head"] = [
            None if h is None else [_r(v) for v in h]
            for h in smooth_head_series(a["head"], fps)
        ]

    glb_name = None
    venue_id = _venue_id(reader)
    abbr = _home_abbr(reader)
    try:
        glb_path = find_ballpark_glb(reader)
        glb_name = glb_path.name
    except FileNotFoundError:
        glb_path = None

    payload = {
        "gamePk": reader.info.get("gamePk"),
        "playId": reader.info.get("playId"),
        "fps": fps,
        "clipStart": clip0,
        "window": [_r(w0 - clip0, 4), _r(w1 - clip0, 4)],
        "duration": _r(w1 - w0, 4),
        "tRelease": None if t_release is None else _r(t_release - w0, 4),
        "tContact": None if t_contact is None else _r(t_contact - w0, 4),
        "pitcherLook": [_r(pitch_xz[0]), 5.0, _r(pitch_xz[1])],
        "bounds": {
            "actors": {"min": [_r(x) for x in amin], "max": [_r(x) for x in amax]},
        },
        "ballpark": {
            "file": glb_name,
            "venueId": venue_id,
            "abbr": abbr,
            "mToFt": 3.28084,
        },
        "times": [_r(t - w0, 4) for t in grid],
        "ball": ball_frames,
        "bat": bat_frames,
        "bones": bones,
        "skins": ensure_player_assets(out_json.parent, play_dir),
        "actors": actors_out,
        "views": views,
        "headPose": mode,
    }
    out_json.write_text(json.dumps(payload, separators=(",", ":")))
    dt = time.perf_counter() - t0_wall
    n_views = len(views["players"]) + len(views["officials"])
    print(f"exported {out_json}  {len(grid)} frames  {out_json.stat().st_size / 1e6:.2f} MB  "
          f"{n_views} views  head={mode}  ({dt:.2f}s)")
    for v in views["players"] + views["officials"]:
        print(f"  view  {v['label']}")
    return payload, glb_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("play_dir")
    ap.add_argument("out", nargs="?", default=None)
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--head-pose", choices=list(HEAD_POSES), default=HEAD_POSE)
    args = ap.parse_args()
    out = args.out or str(Path(__file__).resolve().parent / "data" / "play.json")
    export_play(args.play_dir, out, fps=args.fps, full=args.full, head_pose=args.head_pose)


if __name__ == "__main__":
    main()

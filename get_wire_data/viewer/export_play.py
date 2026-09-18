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
    TYPE_COLORS, _actor_pose_tracks, _sample_pose, _bat_track, _sample_gap,
    _pitch_release_time, _pitcher_xz,
)
from stadium import find_ballpark_glb, _home_abbr, _venue_id
from views import (
    load_bios, names_from_boxscore, resolve_name, classify_views,
    look_target, look_from_eye, smooth_head_series,
)


def _r(v, n=3):
    return None if v is None else round(float(v), n)


def _xyz(p):
    return [_r(p[0]), _r(p[1]), _r(p[2])]


def export_play(play_dir, out_json, fps=20.0, full=False):
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
    pitch_xz = _pitcher_xz(reader, t_release if t_release is not None else w0)

    actors_out = []
    uids = sorted(pose_tracks)
    bios = load_bios()
    box_names = names_from_boxscore(reader.metadata)
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
        classify_in.append({
            "uid": int(uid) if str(uid).isdigit() else uid,
            "type": reader.actor_type(uid),
            "playerId": pid,
            "name": name,
            "start": start,
        })
        actors_out.append({
            "uid": int(uid) if str(uid).isdigit() else uid,
            "playerId": pid if isinstance(pid, int) and pid > 0 else None,
            "name": name,
            "type": reader.actor_type(uid),
            "color": TYPE_COLORS.get(reader.actor_type(uid), TYPE_COLORS["unknown"]),
            "frames": [],
            "head": [],
        })
    views = classify_views(classify_in)
    uid_slot = {v["uid"]: v.get("slot") for v in views["players"] + views["officials"]}

    ball_frames = []
    bat_frames = []
    amin = np.array([1e9, 1e9, 1e9])
    amax = np.array([-1e9, -1e9, -1e9])

    for t in grid:
        b = sample_ball(ball_track, t)
        ball_frames.append(_xyz(b) if b is not None else None)
        bh = _sample_gap(bat_track, t, 0.3)
        if bh:
            bat_frames.append({"handle": _xyz(bh[0]), "head": _xyz(bh[1])})
        else:
            bat_frames.append(None)
        for ai, uid in enumerate(uids):
            pose = _sample_pose(pose_tracks[uid], t)
            if pose is None:
                actors_out[ai]["frames"].append(None)
                actors_out[ai]["head"].append(None)
                continue
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
            if eye is None:
                actors_out[ai]["head"].append(None)
                continue
            slot = uid_slot.get(actors_out[ai]["uid"])
            tgt = look_target(slot, b)
            hp = look_from_eye(eye, tgt)
            if hp is None:
                actors_out[ai]["head"].append(None)
            else:
                pos, fwd, up = hp
                actors_out[ai]["head"].append([_r(v) for v in (*pos, *fwd, *up)])

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
        "actors": actors_out,
        "views": views,
    }
    out_json.write_text(json.dumps(payload, separators=(",", ":")))
    dt = time.perf_counter() - t0_wall
    n_views = len(views["players"]) + len(views["officials"])
    print(f"exported {out_json}  {len(grid)} frames  {out_json.stat().st_size / 1e6:.2f} MB  "
          f"{n_views} views  ({dt:.2f}s)")
    for v in views["players"] + views["officials"]:
        print(f"  view  {v['label']}")
    return payload, glb_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("play_dir")
    ap.add_argument("out", nargs="?", default=None)
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    out = args.out or str(Path(__file__).resolve().parent / "data" / "play.json")
    export_play(args.play_dir, out, fps=args.fps, full=args.full)


if __name__ == "__main__":
    main()

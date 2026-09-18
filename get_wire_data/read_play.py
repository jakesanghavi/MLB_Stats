"""
Read a downloaded Gameday 3D play (a directory of raw files written by
``MannequinClient.download_raw``) back into workable Python structures.

Directory layout (per play):
    play.json        {gamePk, playId, version}
    versions.json
    manifest.json    {records:[{index,startTime,duration,isGap}], ...}
    metadata.json    {boneIdMap, batBoneIdMap, ruleSettings, boxscore, ...}
    labels.json      {uid: {actor: playerId, type: "pitcher"|"batter"|...}}
    uniforms.json
    {index}.bin      TrackingDataWire chunks (one per non-gap manifest record)

Usage:
    from read_play import PlayReader
    play = PlayReader("/path/to/play_dir")
    print(play.summary())
    for t, x, y, z in play.ball_track():
        ...
    for uid, track in play.actor_tracks().items():
        ...
"""
import bisect
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from get_wires import decode_tracking_data

# Interpolate through multi-frame tracking dropouts (common on high flies),
# but do not bridge separate ball phases (e.g. the several-second hole
# between a throw landing and the next throw).
BALL_INTERP_MAX_GAP = 4.0
_G_FT = 32.174  # ft/s^2; Y is ballistic between bracketing samples


def _load_json(path):
    return json.loads(Path(path).read_text()) if Path(path).exists() else None


def _ball_xyz(sample):
    """Unpack (t, x, y, z) or (t, (x, y, z)) -> (t, x, y, z)."""
    if len(sample) == 2:
        t, v = sample
        return float(t), float(v[0]), float(v[1]), float(v[2])
    return float(sample[0]), float(sample[1]), float(sample[2]), float(sample[3])


def sample_ball(track, t, max_gap=BALL_INTERP_MAX_GAP):
    """Best-estimate ball (x, y, z) at time ``t`` from a sorted sample track.

    Uses the samples before *and* after ``t``:
      * X/Z — cubic Hermite with Catmull-Rom tangents from extra neighbors
        when they exist, otherwise linear.
      * Y (height) — ballistic under gravity, so a high-fly hole missing the
        apex is reconstructed instead of flattened to a chord.

    Returns None outside the track, when a side is missing (no extrapolation),
    or when the bracketing gap is wider than ``max_gap`` seconds.
    """
    n = len(track)
    if n == 0:
        return None
    times = [_ball_xyz(p)[0] for p in track]
    j = bisect.bisect_left(times, t)
    if j < n and abs(times[j] - t) < 1e-9:
        _, x, y, z = _ball_xyz(track[j])
        return (x, y, z)
    if j <= 0 or j >= n:
        return None
    i1, i2 = j - 1, j
    t1 = times[i1]
    t2 = times[i2]
    dt = t2 - t1
    if dt <= 1e-9 or dt > max_gap:
        return None
    _, x1, y1, z1 = _ball_xyz(track[i1])
    _, x2, y2, z2 = _ball_xyz(track[i2])
    p1 = (x1, y1, z1)
    p2 = (x2, y2, z2)
    if i1 - 1 >= 0:
        t0, x0, y0, z0 = _ball_xyz(track[i1 - 1])
        span = t2 - t0
        v1 = tuple((b - a) / span for a, b in zip((x0, y0, z0), p2)) if span > 1e-9 \
            else tuple((b - a) / dt for a, b in zip(p1, p2))
    else:
        v1 = tuple((b - a) / dt for a, b in zip(p1, p2))
    if i2 + 1 < n:
        t3, x3, y3, z3 = _ball_xyz(track[i2 + 1])
        span = t3 - t1
        v2 = tuple((b - a) / span for a, b in zip(p1, (x3, y3, z3))) if span > 1e-9 \
            else tuple((b - a) / dt for a, b in zip(p1, p2))
    else:
        v2 = tuple((b - a) / dt for a, b in zip(p1, p2))
    u = (t - t1) / dt
    u2 = u * u
    u3 = u2 * u
    h00 = 2 * u3 - 3 * u2 + 1
    h10 = u3 - 2 * u2 + u
    h01 = -2 * u3 + 3 * u2
    h11 = u3 - u2
    x, _, z = (h00 * a + h10 * dt * va + h01 * b + h11 * dt * vb
               for a, va, b, vb in zip(p1, v1, p2, v2))
    implied = math.dist(p1, p2) / dt
    spd1 = math.dist((x0, y0, z0), p1) / (t1 - t0) if i1 - 1 >= 0 else implied
    spd2 = math.dist(p2, (x3, y3, z3)) / (t3 - t2) if i2 + 1 < n else implied
    # Nearly stopped vs the flight on either side → held/caught, not airborne.
    held = (spd1 > 15.0 and spd2 > 15.0 and implied < 0.3 * min(spd1, spd2))
    if held:
        y = y1 + (y2 - y1) * u
    else:
        dt_local = t - t1
        vy = (y2 - y1) / dt + 0.5 * _G_FT * dt
        y = y1 + vy * dt_local - 0.5 * _G_FT * dt_local * dt_local
    return (x, max(y, 0.0), z)


class PlayReader:
    def __init__(self, play_dir):
        self.dir = Path(play_dir)
        self.info = _load_json(self.dir / "play.json") or {}
        self.metadata = _load_json(self.dir / "metadata.json") or {}
        self.manifest = _load_json(self.dir / "manifest.json") or {}
        self.labels = _load_json(self.dir / "labels.json") or {}
        self.uniforms = _load_json(self.dir / "uniforms.json") or {}
        self.bone_id_map = self.metadata.get("boneIdMap")

        records = self.manifest.get("records", []) if isinstance(self.manifest, dict) else []
        records = sorted(records, key=lambda r: r.get("startTime", r.get("index", 0)))

        self.frames = []
        self.chunks_read = 0
        for rec in records:
            idx = rec.get("index")
            if idx is None or rec.get("isGap"):
                continue
            chunk = self.dir / f"{idx}.bin"
            if not chunk.exists():
                continue
            decoded = decode_tracking_data(chunk.read_bytes(), self.bone_id_map)
            self.frames.extend(decoded["frames"])
            self.chunks_read += 1

        self.frames.sort(key=lambda f: (f.get("time") or 0, f.get("num", 0)))

    # -- convenience views ------------------------------------------------
    def actor_label(self, uid):
        return self.labels.get(str(uid), {})

    def actor_type(self, uid):
        return self.actor_label(uid).get("type", "unknown")

    def ball_track(self):
        """[(time, x, y, z)] for every frame with a tracked ball position."""
        out = []
        for f in self.frames:
            b = f.get("ball")
            if b:
                out.append((f["time"], b["x"], b["y"], b["z"]))
        return out

    def ball_at(self, t, max_gap=BALL_INTERP_MAX_GAP):
        """Interpolated ball (x, y, z) at time ``t``, or None."""
        return sample_ball(self.ball_track(), t, max_gap=max_gap)

    def actor_tracks(self):
        """uid -> [(time, rootPos_dict)] time series of each actor's root."""
        tracks = {}
        for f in self.frames:
            for a in f.get("actorPoses", []):
                if a.get("rootPos"):
                    tracks.setdefault(a["uid"], []).append((f["time"], a["rootPos"]))
        return tracks

    def events(self):
        """[(time, dataType, data)] flattened game events across the play."""
        out = []
        for f in self.frames:
            for e in f.get("gameEvents", []):
                out.append((e.get("time", f["time"]), e.get("dataType"), e.get("data")))
        return out

    def live_action_intervals(self, min_seconds=0.75):
        """[(start, end)] intervals where liveAction mode==true (the ball is live).

        Short sub-second blips (warmup/setup jitter) are dropped. An unterminated
        final 'true' is closed at the end of the clip.
        """
        if not self.frames:
            return []
        toggles = []
        for f in self.frames:
            for e in f.get("gameEvents", []):
                if e.get("dataType") == 4:  # liveAction
                    toggles.append((e.get("time", f["time"]), bool((e.get("data") or {}).get("mode"))))
        toggles.sort(key=lambda x: x[0])
        intervals = []
        start = None
        for t, mode in toggles:
            if mode and start is None:
                start = t
            elif not mode and start is not None:
                intervals.append((start, t))
                start = None
        if start is not None:
            intervals.append((start, self.frames[-1]["time"]))
        return [iv for iv in intervals if iv[1] - iv[0] >= min_seconds]

    def play_window(self, max_seconds=25.0, lead=2.0, trail=2.0, gap_merge=8.0):
        """Estimate the *actual* action window (absolute start, end seconds).

        Gameday 3D clips are over-inclusive (long lead-in/out, sometimes bleed
        from adjacent plays). We anchor on the in-stream ``playEvent{action:0}``
        (the pitch), take the contiguous ball-active window after it, and trim the
        end to the sustained ``liveAction`` (ball-live) interval that contains the
        pitch so trailing dead-ball tracking is excluded. Capped at ``max_seconds``.
        Falls back to the first tracked-ball time, then to the whole clip.
        """
        if not self.frames:
            return None
        t0 = self.frames[0]["time"]
        tN = self.frames[-1]["time"]

        t_pitch = None
        for f in self.frames:
            for e in f.get("gameEvents", []):
                if e.get("dataType") == 7 and (e.get("data") or {}).get("action") == 0:
                    t_pitch = e.get("time", f["time"])
                    break
            if t_pitch is not None:
                break

        ball_t = [t for t, _, _, _ in self.ball_track()]
        if t_pitch is None:
            if not ball_t:
                return (t0, tN)
            t_pitch = ball_t[0]

        # walk the contiguous ball-active segment covering/after the pitch
        seg_end = t_pitch
        started = False
        prev = None
        for bt in ball_t:
            if bt < t_pitch - 1.0:
                prev = bt
                continue
            if not started:
                started, seg_end, prev = True, bt, bt
                continue
            if bt - prev <= gap_merge:
                seg_end, prev = bt, bt
            else:
                break

        # sustained live-action interval overlapping [pitch, ball segment end]
        start_anchor = t_pitch
        t_end = seg_end
        candidates = [iv for iv in self.live_action_intervals()
                      if iv[1] >= t_pitch - 1.0 and iv[0] <= seg_end + 1.0]
        if candidates:
            containing = [iv for iv in candidates if iv[0] <= t_pitch <= iv[1]]
            live = max(containing or candidates, key=lambda iv: iv[1] - iv[0])
            start_anchor = min(t_pitch, live[0])
            t_end = min(seg_end, live[1])  # trim trailing dead-ball tracking

        t_end = min(t_end, t_pitch + max_seconds)
        return (max(t0, start_anchor - lead), min(tN, t_end + trail))

    def summary(self):
        tracks = self.actor_tracks()
        types = {}
        for uid in tracks:
            t = self.actor_type(uid)
            types[t] = types.get(t, 0) + 1
        times = [f["time"] for f in self.frames if f.get("time") is not None]
        return {
            "gamePk": self.info.get("gamePk"),
            "playId": self.info.get("playId"),
            "version": self.info.get("version"),
            "chunks_read": self.chunks_read,
            "frames": len(self.frames),
            "frames_with_ball": len(self.ball_track()),
            "unique_actors": len(tracks),
            "actor_types": types,
            "duration_s": round(max(times) - min(times), 2) if times else 0,
            "events": len(self.events()),
        }


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    r = PlayReader(d)
    print(json.dumps(r.summary(), indent=2))

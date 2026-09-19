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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from get_wires import decode_tracking_data

# Short holes are just 30 fps sampling. Abnormal (longer) holes are only
# filled when BOTH ends of the hole are at least this high — the same fly
# ball, not a catch connecting to a replacement ball. No time cap; a high
# fly may be missing for several seconds around the apex.
BALL_INTERP_HEIGHT = 25.0  # ft above the field
BALL_DENSE_GAP = 0.2       # seconds
_G_FT = 32.174             # ft/s^2; Y is ballistic between bracketing samples


def _load_json(path):
    return json.loads(Path(path).read_text()) if Path(path).exists() else None


def _ball_xyz(sample):
    """Unpack (t, x, y, z) or (t, (x, y, z)) -> (t, x, y, z)."""
    if len(sample) == 2:
        t, v = sample
        return float(t), float(v[0]), float(v[1]), float(v[2])
    return float(sample[0]), float(sample[1]), float(sample[2]), float(sample[3])


def sample_ball(track, t, min_height=BALL_INTERP_HEIGHT, dense_gap=BALL_DENSE_GAP):
    """Best-estimate ball (x, y, z) at time ``t`` from a sorted sample track.

    Always interpolates sampling-cadence holes (``dense_gap``, default 0.2s).

    Abnormal holes are filled only when *both* the last sample before the gap
    and the next sample after it are at least ``min_height`` ft high (default
    25). That is the whole gate: a fly missing its apex is filled with no
    duration cap; a catch connecting to a second/replacement ball is not,
    because that new ball is near the field. X/Z are linear in time; Y is
    ballistic under gravity. No extrapolation.
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
    if dt <= 1e-9:
        return None
    _, x1, y1, z1 = _ball_xyz(track[i1])
    _, x2, y2, z2 = _ball_xyz(track[i2])
    if dt > dense_gap and (y1 < min_height or y2 < min_height):
        return None
    u = (t - t1) / dt
    x = x1 + (x2 - x1) * u
    z = z1 + (z2 - z1) * u
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

    def ball_at(self, t, min_height=BALL_INTERP_HEIGHT):
        """Interpolated ball (x, y, z) at time ``t``, or None."""
        return sample_ball(self.ball_track(), t, min_height=min_height)

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

    def pitch_releases(self):
        """[(time, playId)] for every in-clip ``playEvent{action:0}`` (pitch released)."""
        out = []
        for t, dt, data in self.events():
            if dt == 7 and (data or {}).get("action") == 0:
                out.append((t, (data or {}).get("playId")))
        return out

    def pitch_release_time(self, play_id=None):
        """Absolute seconds of the pitch-released event for this play.

        Mannequin clips often bleed a neighboring pitch. Prefer the
        ``playEvent{action:0}`` whose ``playId`` matches the requested GUID
        (``play.json`` / ``self.info``). Fall back to the first release in
        the clip when the GUID is missing or unmatched.
        """
        if play_id is None:
            play_id = self.info.get("playId")
        releases = self.pitch_releases()
        if play_id:
            for t, pid in releases:
                if pid == play_id:
                    return t
        return releases[0][0] if releases else None

    def play_window(self, max_seconds=25.0, lead=2.0, trail=2.0, gap_merge=8.0):
        """Estimate the *actual* action window (absolute start, end seconds).

        Gameday 3D clips are over-inclusive (long lead-in/out, sometimes bleed
        from adjacent plays). We anchor on the in-stream ``playEvent{action:0}``
        whose ``playId`` matches the requested GUID (not merely the first
        release in the file), take the contiguous ball-active window after it,
        and trim the end to the sustained ``liveAction`` (ball-live) interval
        that contains the pitch so trailing dead-ball tracking is excluded.
        The start is not rewound through an earlier pitch when ``liveAction``
        stays on across pitches (typical with a runner on). Capped at
        ``max_seconds``. Falls back to the first tracked-ball time, then to
        the whole clip.
        """
        if not self.frames:
            return None
        t0 = self.frames[0]["time"]
        tN = self.frames[-1]["time"]

        releases = self.pitch_releases()
        t_pitch = self.pitch_release_time()

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
            proposed = min(t_pitch, live[0])
            earlier = [t for t, _pid in releases if t < t_pitch - 0.05]
            # liveAction often stays true across a whole PA when a runner is
            # on; do not rewind the window into the previous pitch.
            if not earlier or proposed > max(earlier):
                start_anchor = proposed
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

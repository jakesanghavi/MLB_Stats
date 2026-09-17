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
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from get_wires import decode_tracking_data


def _load_json(path):
    return json.loads(Path(path).read_text()) if Path(path).exists() else None


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

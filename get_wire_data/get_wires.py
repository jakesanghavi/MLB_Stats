"""
Fetch and decode MLB Gameday 3D (FieldVision / "mannequin") tracking data.

Reverse-engineered from the public Gameday web bundle (gd.min.js +
gd.@bvg_poser.min.js). The 3D replay is driven by size/offset FlatBuffers
served from the mannequin tracking API:

    {domain}/mannequin/{gamePk}/plays/{playId}/versions.json
    {domain}/mannequin/{gamePk}/plays/{playId}/{version}/manifest.json
    {domain}/mannequin/{gamePk}/plays/{playId}/{version}/metadata.json
    {domain}/mannequin/{gamePk}/plays/{playId}/{version}/labels.json
    {domain}/mannequin/{gamePk}/plays/{playId}/{version}/uniforms.json
    {domain}/mannequin/{gamePk}/plays/{playId}/{version}/{index}.bin   <- tracking chunks

    domain = https://fieldvision-hls.mlbinfra.com          (prod)
             https://fieldvision-hls-beta.mlbinfra.com     (beta)

Each ``{index}.bin`` is a single (non size-prefixed) ``TrackingDataWire``
FlatBuffer. ``manifest.json`` lists which chunk indices exist:

    { "version": "...", "records": [ { "index": 11, "startTime": ..., "duration": ... }, ... ] }

TrackingDataWire
  version : string
  frames  : [TrackingFrameWire]
  limits  : LimitsWire (start/end seconds)
  eventKeyFrame : [GameEventWire]

TrackingFrameWire (one per rendered frame)
  actorPoses      : [ActorPoseWire]        rigged skeleton (root pos + packed quats)
  ballPosition    : Vec3Wire               ball xyz for this frame
  gameEvents      : [GameEventWire]         union of *EventDataWire
  trackedEvents   : [TrackedEventWire]
  inferredBat     : TrackingBatPositionWire
  ballPolynomials : [BallPolynomialWire]    union of Ball*DataWire (pitch/hit/throw/...)
  rawJoints       : [SkeletalPlayerWire]    absolute joint positions per player
  num, time, timestamp, isGap, gapDuration

AUTH: the mannequin API requires the logged-in user's MLB bearer token
(the ``Authorization`` header) plus ``x-mannequin-client: gameday``. Grab a
fresh token from DevTools (Network tab -> any *.bin request -> Authorization
header) while logged in and viewing a Gameday 3D play, then pass it via
--token or the MLB_BEARER_TOKEN env var.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import requests

# Make the generated ``MLB`` FlatBuffers package importable regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from MLB.TrackingDataWire import TrackingDataWire
from MLB.TrackingFrameWire import TrackingFrameWire
from MLB.ActorPoseWire import ActorPoseWire
from MLB.SkeletalPlayerWire import SkeletalPlayerWire
from MLB.BallPolynomialWire import BallPolynomialWire
from MLB.GameEventWire import GameEventWire
from MLB.TrackedEventWire import TrackedEventWire
from MLB.TrackingBatPositionWire import TrackingBatPositionWire

# Union member order (index == discriminator value) taken verbatim from the bundle.
from MLB.BallBounceDataWire import BallBounceDataWire
from MLB.BallHitDataWire import BallHitDataWire
from MLB.BallHitLaunchWire import BallHitLaunchWire
from MLB.BallHitRefinedWire import BallHitRefinedWire
from MLB.BallPitchDataWire import BallPitchDataWire
from MLB.BallPitchRefinedWire import BallPitchRefinedWire
from MLB.BallThrowData import BallThrowData
from MLB.BallPickOffDataWire import BallPickOffDataWire
from MLB.BallPitchReleasePointWire import BallPitchReleasePointWire
from MLB.BallPitchSpinWire import BallPitchSpinWire

from MLB.CountEventDataWire import CountEventDataWire
from MLB.TeamScoreEventDataWire import TeamScoreEventDataWire
from MLB.BattingOrderEventDataWire import BattingOrderEventDataWire
from MLB.LiveActionEventDataWire import LiveActionEventDataWire
from MLB.InningEventDataWire import InningEventDataWire
from MLB.AtBatEventDataWire import AtBatEventDataWire
from MLB.PlayEventDataWire import PlayEventDataWire
from MLB.HandedEventDataWire import HandedEventDataWire
from MLB.PositionAssignmentEventDataWire import PositionAssignmentEventDataWire
from MLB.GumboTimecodeEventDataWire import GumboTimecodeEventDataWire
from MLB.StatusEventDataWire import StatusEventDataWire
from MLB.BatImpactEventDataWire import BatImpactEventDataWire
from MLB.HighFrequencyBatMarkerEventDataWire import HighFrequencyBatMarkerEventDataWire
from MLB.ABSEventDataWire import ABSEventDataWire

# discriminator value -> concrete union member class
BALL_POLY_UNION = {
    1: BallBounceDataWire, 2: BallHitDataWire, 3: BallHitLaunchWire,
    4: BallHitRefinedWire, 5: BallPitchDataWire, 6: BallPitchRefinedWire,
    7: BallThrowData, 8: BallPickOffDataWire, 9: BallPitchReleasePointWire,
    10: BallPitchSpinWire,
}
GAME_EVENT_UNION = {
    1: CountEventDataWire, 2: TeamScoreEventDataWire, 3: BattingOrderEventDataWire,
    4: LiveActionEventDataWire, 5: InningEventDataWire, 6: AtBatEventDataWire,
    7: PlayEventDataWire, 8: HandedEventDataWire, 9: PositionAssignmentEventDataWire,
    10: GumboTimecodeEventDataWire, 11: StatusEventDataWire, 12: BatImpactEventDataWire,
    13: HighFrequencyBatMarkerEventDataWire, 14: ABSEventDataWire,
}


# ---------------------------------------------------------------------------
# Generic FlatBuffers table -> dict
# ---------------------------------------------------------------------------
def _decode_str(v):
    return v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else v


def table_to_dict(obj):
    """Reflectively flatten a generated FlatBuffers *table* accessor to a dict.

    Handles scalars, strings, nested tables/structs and vectors. Union fields
    (Data/DataType pairs) are resolved by the callers that know the union map.
    """
    if obj is None:
        return None
    out = {}
    names = [n for n in dir(obj) if n[:1].isupper() and not n.startswith("GetRootAs")]
    lengths = {n[:-6] for n in names if n.endswith("Length")}

    # vectors first
    for base in lengths:
        if not hasattr(obj, base):
            continue
        n = getattr(obj, base + "Length")()
        acc = getattr(obj, base)
        arr = []
        for i in range(n):
            try:
                v = acc(i)
            except TypeError:
                v = None
            arr.append(_convert(v))
        out[_lc(base)] = arr

    skip = set()
    for base in lengths:
        skip.update({base, base + "Length", base + "IsNone", base + "AsNumpy"})

    for name in names:
        if name in skip or name in ("Init",):
            continue
        if name.endswith(("Length", "IsNone", "AsNumpy")):
            continue
        meth = getattr(obj, name)
        if not callable(meth):
            continue
        try:
            val = meth()  # zero-arg accessors only
        except TypeError:
            continue
        out[_lc(name)] = _convert(val)
    return out


def _convert(v):
    if v is None:
        return None
    if isinstance(v, (bytes, bytearray)):
        return _decode_str(v)
    if hasattr(v, "Init") and hasattr(v, "_tab"):  # nested table/struct accessor
        return table_to_dict(v)
    return v


def _lc(name):
    return name[0].lower() + name[1:]


def _vec3(v):
    return None if v is None else {"x": v.X(), "y": v.Y(), "z": v.Z()}


def _resolve_union(owner, union_map):
    """Return (type_id, decoded_dict) for a Data()/DataType() union field."""
    tid = owner.DataType()
    tab = owner.Data()
    if not tid or tab is None:
        return tid, None
    cls = union_map.get(tid)
    if cls is None:
        return tid, None
    member = cls()
    member.Init(tab.Bytes, tab.Pos)
    return tid, table_to_dict(member)


# ---------------------------------------------------------------------------
# Structured per-frame decode
# ---------------------------------------------------------------------------
def decode_skeletal_player(sp, bone_id_map=None):
    n = sp.JointPositionsLength()
    ids_n = sp.JointIdsLength()
    joints = {}
    for i in range(ids_n):
        jid = sp.JointIds(i)
        key = (bone_id_map or {}).get(str(jid), str(jid))
        base = 3 * i
        if base + 2 < n:
            joints[key] = [sp.JointPositions(base), sp.JointPositions(base + 1), sp.JointPositions(base + 2)]
    return {
        "trackId": sp.TrackId(),
        "playerId": sp.PlayerId(),
        "jerseyNumber": sp.JerseyNumber(),
        "positionId": sp.PositionId(),
        "roleId": sp.RoleId(),
        "joints": joints,
    }


def decode_actor_pose(ap):
    return {
        "uid": ap.Uid(),
        "rootPos": _vec3(ap.RootPos()),
        "batRootPos": _vec3(ap.BatRootPos()),
        "ground": ap.Ground(),
        "apex": ap.Apex(),
        "scale": ap.Scale(),
        "nodeIds": [ap.NodeIds(i) for i in range(ap.NodeIdsLength())],
        "packedQuats": [ap.PackedQuats(i) for i in range(ap.PackedQuatsLength())],
    }


def decode_frame(fr, bone_id_map=None):
    ball = _vec3(fr.BallPosition())

    players = [decode_skeletal_player(fr.RawJoints(i), bone_id_map)
               for i in range(fr.RawJointsLength())]

    poses = [decode_actor_pose(fr.ActorPoses(i)) for i in range(fr.ActorPosesLength())]

    ball_polys = []
    for i in range(fr.BallPolynomialsLength()):
        bp = fr.BallPolynomials(i)
        tid, data = _resolve_union(bp, BALL_POLY_UNION)
        ball_polys.append({
            "id": _decode_str(bp.Id()),
            "timestamp": _decode_str(bp.Timestamp()),
            "dataType": tid,
            "data": data,
        })

    game_events = []
    for i in range(fr.GameEventsLength()):
        ge = fr.GameEvents(i)
        tid, data = _resolve_union(ge, GAME_EVENT_UNION)
        game_events.append({
            "time": ge.Time(),
            "isKeyFramed": ge.IsKeyFramed(),
            "dataType": tid,
            "data": data,
        })

    tracked_events = [table_to_dict(fr.TrackedEvents(i)) for i in range(fr.TrackedEventsLength())]

    inferred_bat = None
    ib = fr.InferredBat()
    if ib is not None:
        inferred_bat = {"headPosition": _vec3(ib.HeadPosition()),
                        "handlePosition": _vec3(ib.HandlePosition())}

    return {
        "num": fr.Num(),
        "time": fr.Time(),
        "timestamp": _decode_str(fr.Timestamp()),
        "isGap": fr.IsGap(),
        "gapDuration": fr.GapDuration(),
        "ball": ball,
        "players": players,
        "actorPoses": poses,
        "ballPolynomials": ball_polys,
        "gameEvents": game_events,
        "trackedEvents": tracked_events,
        "inferredBat": inferred_bat,
    }


def decode_tracking_data(buf, bone_id_map=None):
    """Decode a single ``{index}.bin`` chunk (a TrackingDataWire root)."""
    td = TrackingDataWire.GetRootAs(bytearray(buf), 0)
    limits = td.Limits()
    return {
        "version": _decode_str(td.Version()),
        "limits": None if limits is None else {"start": limits.Start(), "end": limits.End()},
        "frames": [decode_frame(td.Frames(i), bone_id_map) for i in range(td.FramesLength())],
    }


# ---------------------------------------------------------------------------
# Mannequin tracking API client
# ---------------------------------------------------------------------------
class MannequinClient:
    PROD = "https://fieldvision-hls.mlbinfra.com"
    BETA = "https://fieldvision-hls-beta.mlbinfra.com"

    def __init__(self, game_pk, play_id=None, token=None, client_id="gameday",
                 env="prod", base_path="mannequin"):
        self.game_pk = game_pk
        self.play_id = play_id
        self.token = token
        self.client_id = client_id
        self.domain = self.BETA if env == "beta" else self.PROD
        self.base_path = base_path
        self.root = f"{game_pk}/plays/{play_id}" if play_id else f"{game_pk}"
        self._version = None

    @property
    def headers(self):
        h = {}
        if self.token:
            h["Authorization"] = self.token
        if self.client_id:
            h["x-mannequin-client"] = self.client_id
        return h

    def _url(self, path):
        return f"{self.domain}/{self.base_path}/{path}"

    def _get_json(self, path):
        r = requests.get(self._url(path), headers=self.headers, timeout=30)
        r.raise_for_status()
        return r.json()

    def _get_bytes(self, path):
        r = requests.get(self._url(path), headers=self.headers, timeout=30)
        r.raise_for_status()
        return r.content

    def versions(self):
        return self._get_json(f"{self.root}/versions.json")

    def latest_version(self):
        if self._version:
            return self._version
        v = self.versions()
        # versions.json shapes vary; accept list or {versions:[...]} / {latest:..}
        if isinstance(v, dict):
            cand = v.get("versions") or v.get("available") or list(v.values())
        else:
            cand = v
        vers = [str(x) for x in cand] if isinstance(cand, list) else [str(cand)]
        vers = [x for x in vers if x and x[0].isdigit()]
        vers.sort(key=lambda s: [int(p) for p in s.split(".") if p.isdigit()])
        self._version = vers[-1] if vers else None
        return self._version

    def manifest(self, version=None):
        version = version or self.latest_version()
        return self._get_json(f"{self.root}/{version}/manifest.json")

    def metadata(self, version=None):
        version = version or self.latest_version()
        return self._get_json(f"{self.root}/{version}/metadata.json")

    def labels(self, version=None):
        version = version or self.latest_version()
        return self._get_json(f"{self.root}/{version}/labels.json")

    def uniforms(self, version=None):
        version = version or self.latest_version()
        return self._get_json(f"{self.root}/{version}/uniforms.json")

    def chunk_bytes(self, index, version=None):
        version = version or self.latest_version()
        return self._get_bytes(f"{self.root}/{version}/{index}.bin")

    def download_play(self, version=None):
        """Fetch every chunk in the manifest and return merged per-play tracking."""
        version = version or self.latest_version()
        meta = self.metadata(version)
        bone_id_map = (meta or {}).get("boneIdMap")
        manifest = self.manifest(version)
        records = manifest.get("records", []) if isinstance(manifest, dict) else []
        records = sorted(records, key=lambda r: r.get("startTime", r.get("index", 0)))

        frames = []
        for rec in records:
            idx = rec.get("index")
            if idx is None:
                continue
            buf = self.chunk_bytes(idx, version)
            decoded = decode_tracking_data(buf, bone_id_map)
            frames.extend(decoded["frames"])

        frames.sort(key=lambda f: (f.get("time") if f.get("time") is not None else 0, f.get("num", 0)))
        return {
            "gamePk": self.game_pk,
            "playId": self.play_id,
            "version": version,
            "metadata": meta,
            "manifest": manifest,
            "frameCount": len(frames),
            "frames": frames,
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Fetch & decode MLB Gameday 3D tracking data for a play.")
    ap.add_argument("--game-pk", type=int, required=True)
    ap.add_argument("--play-id", required=True, help="Play GUID (from the mannequin URL / gumbo playId)")
    ap.add_argument("--token", default=os.environ.get("MLB_BEARER_TOKEN"),
                    help="MLB bearer token (or set MLB_BEARER_TOKEN). Grab from DevTools while logged in.")
    ap.add_argument("--env", choices=["prod", "beta"], default="prod")
    ap.add_argument("--client-id", default="gameday")
    ap.add_argument("--version", default=None, help="Force a specific data version (default: latest)")
    ap.add_argument("--out", default=None, help="Write merged tracking JSON to this path")
    args = ap.parse_args()

    if not args.token:
        ap.error("No token. Pass --token or set MLB_BEARER_TOKEN (see module docstring for how to get one).")

    client = MannequinClient(args.game_pk, args.play_id, token=args.token,
                             client_id=args.client_id, env=args.env)
    play = client.download_play(args.version)
    print(f"version={play['version']}  frames={play['frameCount']}")
    if play["frames"]:
        f0 = play["frames"][0]
        print(f"frame0: t={f0['time']} ball={f0['ball']} players={len(f0['players'])} "
              f"poses={len(f0['actorPoses'])} events={len(f0['gameEvents'])}")

    out = args.out or f"tracking_{args.game_pk}_{args.play_id}.json"
    Path(out).write_text(json.dumps(play, indent=2, default=str))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

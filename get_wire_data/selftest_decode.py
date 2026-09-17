"""Round-trip self-test for the Gameday 3D wire decoder.

Builds a synthetic ``TrackingDataWire`` (identical byte layout to a real
``{index}.bin``) using the flatc-generated builders, then decodes it with
``get_wires.decode_tracking_data`` and asserts every value survives the trip.

This validates the reconstructed schema + decoder without needing a live
mannequin API token (which requires an MLB login).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import flatbuffers
import MLB.SkeletalPlayerWire as SP
import MLB.ActorPoseWire as AP
import MLB.BallHitLaunchWire as BHL
import MLB.BallPolynomialWire as BP
import MLB.CountEventDataWire as CNT
import MLB.GameEventWire as GE
import MLB.TrackingFrameWire as TF
import MLB.TrackingDataWire as TD
from MLB.Vec3Wire import CreateVec3Wire
from MLB.LimitsWire import CreateLimitsWire

from get_wires import decode_tracking_data


def build_buffer():
    b = flatbuffers.Builder(1024)

    # ---- SkeletalPlayerWire (raw joints): 2 joints ----
    jp = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    SP.StartJointPositionsVector(b, len(jp))
    for v in reversed(jp):
        b.PrependFloat32(v)
    jp_off = b.EndVector()
    jids = [0, 42]  # bone ids
    SP.StartJointIdsVector(b, len(jids))
    for v in reversed(jids):
        b.PrependUint32(v)
    jids_off = b.EndVector()
    SP.Start(b)
    SP.AddPositionId(b, 6)
    SP.AddTrackId(b, 777)
    SP.AddJointPositions(b, jp_off)
    SP.AddJointIds(b, jids_off)
    SP.AddPlayerId(b, 123456)
    SP.AddJerseyNumber(b, 27)
    SP.AddRoleId(b, 2)
    sp_off = SP.End(b)

    TF.StartRawJointsVector(b, 1)
    b.PrependUOffsetTRelative(sp_off)
    rawjoints_off = b.EndVector()

    # ---- ActorPoseWire ----
    pq = [111, 222, 333]
    AP.StartPackedQuatsVector(b, len(pq))
    for v in reversed(pq):
        b.PrependUint32(v)
    pq_off = b.EndVector()
    nids = [0, 1, 2]
    AP.StartNodeIdsVector(b, len(nids))
    for v in reversed(nids):
        b.PrependUint16(v)
    nids_off = b.EndVector()
    AP.Start(b)
    AP.AddUid(b, 999)
    AP.AddRootPos(b, CreateVec3Wire(b, 10.0, 11.0, 12.0))
    AP.AddPackedQuats(b, pq_off)
    AP.AddNodeIds(b, nids_off)
    AP.AddGround(b, 0.5)
    AP.AddApex(b, 7.5)
    AP.AddScale(b, 1.0)
    ap_off = AP.End(b)
    TF.StartActorPosesVector(b, 1)
    b.PrependUOffsetTRelative(ap_off)
    poses_off = b.EndVector()

    # ---- BallPolynomialWire (union member 3 = BallHitLaunchWire) ----
    BHL.Start(b)
    BHL.AddSpeed(b, 95.0)
    BHL.AddAngle(b, 22.0)
    BHL.AddDirection(b, -5.0)
    bhl_off = BHL.End(b)
    bp_id = b.CreateString("poly-1")
    bp_ts = b.CreateString("2026-09-16T00:00:00Z")
    BP.Start(b)
    BP.AddId(b, bp_id)
    BP.AddTimestamp(b, bp_ts)
    BP.AddDataType(b, 3)
    BP.AddData(b, bhl_off)
    bp_off = BP.End(b)
    TF.StartBallPolynomialsVector(b, 1)
    b.PrependUOffsetTRelative(bp_off)
    bpoly_off = b.EndVector()

    # ---- GameEventWire (union member 1 = CountEventDataWire) ----
    CNT.Start(b)
    CNT.AddBalls(b, 2)
    CNT.AddStrikes(b, 1)
    CNT.AddOuts(b, 0)
    cnt_off = CNT.End(b)
    GE.Start(b)
    GE.AddDataType(b, 1)
    GE.AddData(b, cnt_off)
    GE.AddTime(b, 0.25)
    GE.AddIsKeyFramed(b, 1)
    ge_off = GE.End(b)
    TF.StartGameEventsVector(b, 1)
    b.PrependUOffsetTRelative(ge_off)
    gev_off = b.EndVector()

    # ---- TrackingFrameWire ----
    ts_off = b.CreateString("2026-09-16T00:00:00.100Z")
    TF.Start(b)
    TF.AddActorPoses(b, poses_off)
    TF.AddBallPosition(b, CreateVec3Wire(b, 0.1, 55.0, 3.2))
    TF.AddGameEvents(b, gev_off)
    TF.AddRawJoints(b, rawjoints_off)
    TF.AddBallPolynomials(b, bpoly_off)
    TF.AddNum(b, 1)
    TF.AddTime(b, 0.1)
    TF.AddTimestamp(b, ts_off)
    TF.AddIsGap(b, False)
    TF.AddGapDuration(b, 0.0)
    frame_off = TF.End(b)
    TD.StartFramesVector(b, 1)
    b.PrependUOffsetTRelative(frame_off)
    frames_off = b.EndVector()

    # ---- TrackingDataWire (root) ----
    ver_off = b.CreateString("1.6.1")
    TD.Start(b)
    TD.AddVersion(b, ver_off)
    TD.AddFrames(b, frames_off)
    TD.AddLimits(b, CreateLimitsWire(b, 0.0, 5.0))
    root = TD.End(b)
    b.Finish(root)
    return bytes(b.Output())


def main():
    buf = build_buffer()
    bone_id_map = {"0": "joint_Pelvis", "42": "joint_HeadEnd"}
    out = decode_tracking_data(buf, bone_id_map)

    assert out["version"] == "1.6.1", out["version"]
    assert out["limits"] == {"start": 0.0, "end": 5.0}, out["limits"]
    assert len(out["frames"]) == 1
    f = out["frames"][0]

    assert f["num"] == 1 and abs(f["time"] - 0.1) < 1e-6
    assert f["timestamp"] == "2026-09-16T00:00:00.100Z"
    assert f["ball"] == {"x": round(0.1, 6) if False else f["ball"]["x"], "y": 55.0, "z": f["ball"]["z"]}
    assert abs(f["ball"]["x"] - 0.1) < 1e-6 and abs(f["ball"]["z"] - 3.2) < 1e-6

    assert len(f["players"]) == 1
    p = f["players"][0]
    assert p["trackId"] == 777 and p["playerId"] == 123456 and p["jerseyNumber"] == 27
    assert p["positionId"] == 6 and p["roleId"] == 2
    assert p["joints"]["joint_Pelvis"] == [1.0, 2.0, 3.0]
    assert p["joints"]["joint_HeadEnd"] == [4.0, 5.0, 6.0]

    assert len(f["actorPoses"]) == 1
    ap = f["actorPoses"][0]
    assert ap["uid"] == 999
    assert ap["rootPos"] == {"x": 10.0, "y": 11.0, "z": 12.0}
    # 3 packed quats -> 3 joint rotations keyed by bone name (node 0 maps via bone_id_map)
    assert set(ap["jointRotations"].keys()) == {"joint_Pelvis", "1", "2"}
    for q in ap["jointRotations"].values():
        assert len(q) == 4 and abs(sum(c * c for c in q) - 1.0) < 1e-3  # unit quaternion

    assert len(f["ballPolynomials"]) == 1
    bp = f["ballPolynomials"][0]
    assert bp["id"] == "poly-1" and bp["dataType"] == 3
    assert abs(bp["data"]["speed"] - 95.0) < 1e-4
    assert abs(bp["data"]["angle"] - 22.0) < 1e-4
    assert abs(bp["data"]["direction"] - (-5.0)) < 1e-4

    assert len(f["gameEvents"]) == 1
    ge = f["gameEvents"][0]
    assert ge["dataType"] == 1 and ge["isKeyFramed"] == 1
    assert abs(ge["time"] - 0.25) < 1e-6
    assert ge["data"] == {"balls": 2, "strikes": 1, "outs": 0}, ge["data"]

    print("OK: %d byte buffer decoded and all assertions passed" % len(buf))
    import json
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()

"""Classification smoke tests for POV view labels."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from reconstruct3d import _actor_pose_tracks, _sample_pose
from read_play import PlayReader
from views import (
    classify_views, load_bios, names_from_boxscore, resolve_name,
)


def _actors(play_dir):
    reader = PlayReader(play_dir)
    w0, w1 = reader.play_window()
    tracks = _actor_pose_tracks(reader)
    bios = load_bios()
    box = names_from_boxscore(reader.metadata)
    out = []
    for uid in tracks:
        pid = reader.actor_label(uid).get("actor")
        pose = _sample_pose(tracks[uid], w0)
        if pose is None:
            for tt, p in tracks[uid]:
                if w0 <= tt <= w1:
                    pose = p
                    break
        start = None
        if pose and pose.get("rootPos"):
            rp = pose["rootPos"]
            start = [rp["x"], rp["y"], rp["z"]]
        out.append({
            "uid": uid,
            "type": reader.actor_type(uid),
            "playerId": pid,
            "name": resolve_name(pid, bios, box),
            "start": start,
        })
    return classify_views(out)


def test_empty_bases_tex():
    d = Path("/tmp/gd/play_822849_8313f274-c733-325e-8df0-beaee0ddb6e1")
    if not d.exists():
        print("skip empty-bases (no play dir)")
        return
    views = _actors(d)
    slots = {v["slot"]: v for v in views["players"]}
    assert slots["P"]["name"] == "Jacob deGrom", slots.get("P")
    assert slots["C"]["name"] == "Danny Jansen"
    assert slots["Batter"]["name"] == "Adley Rutschman"
    assert "1B runner" not in slots
    assert "2B runner" not in slots
    assert "3B runner" not in slots
    off = {v["slot"] for v in views["officials"]}
    assert "Home plate umpire" in off
    assert "First base coach" in off
    assert "Third base coach" in off
    print("ok empty-bases", [v["label"] for v in views["players"]])
    print("   officials", [v["label"] for v in views["officials"]])


def test_runners_stl():
    d = Path("/tmp/gd/play_823004_a6115b75-7846-3fb7-bbe4-995d0295512b")
    if not d.exists():
        print("skip runners (no play dir)")
        return
    views = _actors(d)
    slots = {v["slot"]: v for v in views["players"]}
    assert "Batter" in slots, slots
    assert "1B runner" in slots, slots
    assert "3B runner" in slots, slots
    assert "2B runner" not in slots
    # runners were labeled batter in the wire
    assert slots["1B runner"]["name"]
    assert slots["3B runner"]["name"]
    assert slots["Batter"]["uid"] != slots["1B runner"]["uid"]
    print("ok runners", [v["label"] for v in views["players"]])


if __name__ == "__main__":
    test_empty_bases_tex()
    test_runners_stl()
    print("ok")

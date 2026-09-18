"""In-memory play serving and no-reuse port."""
import json
import sys
import threading
from pathlib import Path
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent))
import serve


def test_api_play_is_memory_not_disk():
    payload = {
        "gamePk": 1,
        "playId": "abc-new-play",
        "views": {"players": [{"slot": "P", "label": "P - Jake Bennett", "name": "Jake Bennett"}]},
        "actors": [],
    }
    serve.CURRENT["json"] = json.dumps(payload).encode()
    serve.CURRENT["whoami"] = json.dumps({"pitcher": "P - Jake Bennett", "playId": "abc-new-play"}).encode()
    serve.CURRENT["meta"] = {"playId": "abc-new-play", "pitcher": "P - Jake Bennett"}
    serve.CURRENT["index"] = serve.render_index(serve.CURRENT["meta"])
    httpd = serve.ViewerServer(("127.0.0.1", 0), serve.Handler)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        who = json.load(urlopen(f"http://127.0.0.1:{port}/api/whoami"))
        assert who["pitcher"] == "P - Jake Bennett", who
        play = json.load(urlopen(f"http://127.0.0.1:{port}/api/play"))
        assert play["playId"] == "abc-new-play"
        assert play["views"]["players"][0]["name"] == "Jake Bennett"
        disk = json.load(urlopen(f"http://127.0.0.1:{port}/data/play.json"))
        assert disk["playId"] == "abc-new-play"
        html = urlopen(f"http://127.0.0.1:{port}/").read().decode()
        assert "Jake Bennett" in html
        assert "abc-new-play" in html
        print("ok memory play", port)
    finally:
        httpd.shutdown()


def test_no_reuse_address():
    assert serve.ViewerServer.allow_reuse_address is False
    print("ok allow_reuse_address false")


if __name__ == "__main__":
    test_no_reuse_address()
    test_api_play_is_memory_not_disk()
    print("ok")

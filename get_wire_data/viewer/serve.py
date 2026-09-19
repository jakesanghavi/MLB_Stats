"""Export a play and serve the three.js viewer.

    python3 serve.py /path/to/play_dir [--port 8765] [--fps 20] [--head-pose EASY_VISION]

The process keeps the exported play in memory and will not bind a port that
is already in use (HTTPServer's default SO_REUSEADDR lets a second serve.py
start while the first one still answers the browser).
"""
import argparse
import json
import shutil
import sys
import urllib.request
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import unquote, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from export_play import export_play
from stadium import ASSET_BASE
from views import HEAD_POSE, HEAD_POSES

_ASSET_UA = "Mozilla/5.0 (compatible; gameday3d-player-assets/1.0)"


def _install_glb(name, dest, cdn_rel):
    """Copy assets/{name} into dest, or download it from the FieldVision CDN."""
    src = Path(__file__).resolve().parent.parent / "assets" / name
    dest = Path(dest)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    if not src.exists() or src.stat().st_size <= 0:
        src.parent.mkdir(parents=True, exist_ok=True)
        url = f"{ASSET_BASE.rstrip('/')}/{cdn_rel}"
        print(f"downloading {name}  {url}", flush=True)
        req = urllib.request.Request(url, headers={"User-Agent": _ASSET_UA})
        with urllib.request.urlopen(req, timeout=60) as r:
            src.write_bytes(r.read())
    if not src.exists() or src.stat().st_size <= 0:
        return False
    try:
        dest.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dest)
    print(f"{name} -> {dest}", flush=True)
    return True

VIEWER_ROOT = Path(__file__).resolve().parent
DATA = VIEWER_ROOT / "data"

# Filled by prepare(); Handler serves these bytes, not whatever is on disk.
CURRENT = {
    "json": b"{}",
    "whoami": b"{}",
    "index": b"",
    "meta": {},
}


def _pitcher_label(payload):
    for v in (payload.get("views") or {}).get("players") or []:
        if v.get("slot") == "P":
            return v.get("label") or v.get("name") or "P"
    return None


def _meta(payload, play_dir):
    return {
        "gamePk": payload.get("gamePk"),
        "playId": payload.get("playId"),
        "headPose": payload.get("headPose"),
        "pitcher": _pitcher_label(payload),
        "playDir": str(play_dir),
        "nViews": len((payload.get("views") or {}).get("players") or []),
    }


def render_index(meta):
    html = (VIEWER_ROOT / "index.html").read_text(encoding="utf-8")
    rev = str(meta.get("playId") or "play")
    html = html.replace("viewer.js?v=play-cache", f"viewer.js?v={rev}")
    banner = (
        f"{meta.get('gamePk') or ''}  {meta.get('playId') or ''}  "
        f"{meta.get('pitcher') or ''}"
    ).strip()
    inject = (
        f"<script>window.PLAY_META={json.dumps(meta, separators=(',', ':'))};</script>\n"
        f"<div id=\"play-banner\">{banner}</div>\n"
    )
    html = html.replace("<body>\n", "<body>\n" + inject, 1)
    return html.encode("utf-8")


def prepare(play_dir, fps, full, head_pose=None):
    play_dir = Path(play_dir).expanduser().resolve()
    if not play_dir.is_dir():
        raise SystemExit(f"play directory not found: {play_dir}")
    print(f"play_dir  {play_dir}", flush=True)
    DATA.mkdir(parents=True, exist_ok=True)
    payload, glb_path = export_play(
        play_dir, DATA / "play.json", fps=fps, full=full, head_pose=head_pose
    )
    dest = DATA / "ballpark.glb"
    if glb_path and Path(glb_path).exists():
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        try:
            dest.symlink_to(Path(glb_path).resolve())
        except OSError:
            shutil.copy2(glb_path, dest)
        print(f"ballpark -> {dest} ({Path(glb_path).name})", flush=True)
    else:
        print("no ballpark glb (field/stadium will be a flat plane)", flush=True)
    if not _install_glb("bat.glb", DATA / "bat.glb", "models/bat.glb"):
        print("no bat.glb (viewer will use a cylinder)", flush=True)
    if not _install_glb("rbi-ball.glb", DATA / "rbi-ball.glb", "models/rbi-ball.glb"):
        print("no rbi-ball.glb (viewer will use a yellow sphere)", flush=True)
    meta = _meta(payload, play_dir)
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    CURRENT["json"] = body
    CURRENT["whoami"] = json.dumps(meta, indent=2).encode("utf-8")
    CURRENT["meta"] = meta
    CURRENT["index"] = render_index(meta)
    (DATA / "whoami.json").write_bytes(CURRENT["whoami"])
    print(
        f"serving  gamePk={meta.get('gamePk')}  playId={meta.get('playId')}  "
        f"{meta.get('pitcher') or ''}  ({len(body) / 1e6:.2f} MB in memory)",
        flush=True,
    )
    return payload


class ViewerServer(ThreadingHTTPServer):
    # Default HTTPServer.allow_reuse_address is True, so a second serve.py can
    # bind the same port while the first process still handles the browser.
    allow_reuse_address = False


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(VIEWER_ROOT), **kwargs)

    def log_message(self, fmt, *args):
        meta = CURRENT.get("meta") or {}
        extra = f"  play={meta.get('playId')} {meta.get('pitcher') or ''}"
        sys.stderr.write("%s - %s%s\n" % (self.address_string(), fmt % args, extra))

    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        play_id = (CURRENT.get("meta") or {}).get("playId")
        if play_id:
            self.send_header("X-Play-Id", str(play_id))
        super().end_headers()

    def _send_bytes(self, body, content_type):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        if path in ("/", "/index.html"):
            self._send_bytes(CURRENT["index"] or render_index(CURRENT.get("meta") or {}), "text/html; charset=utf-8")
            return
        if path in ("/api/play", "/data/play.json"):
            self._send_bytes(CURRENT["json"], "application/json")
            return
        if path in ("/api/whoami", "/data/whoami.json"):
            self._send_bytes(CURRENT["whoami"], "application/json")
            return
        super().do_GET()


def _die_port_in_use(host, port):
    print(
        f"ERROR: {host}:{port} is already taken.\n"
        "Another viewer is still running and is what the browser will show\n"
        "(often the previous play). In that terminal hit Ctrl+C, or:\n"
        f"  lsof -iTCP:{port} -sTCP:LISTEN\n"
        "then start serve.py again. Or pass --port 8766.",
        file=sys.stderr,
        flush=True,
    )
    raise SystemExit(2)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("play_dir")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--head-pose", choices=list(HEAD_POSES), default=HEAD_POSE)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    prepare(args.play_dir, args.fps, args.full, args.head_pose)
    try:
        httpd = ViewerServer((args.host, args.port), Handler)
    except OSError as exc:
        print(exc, file=sys.stderr)
        _die_port_in_use(args.host, args.port)
    meta = CURRENT["meta"]
    print(f"viewer  http://127.0.0.1:{args.port}/", flush=True)
    print(
        f"whoami  http://127.0.0.1:{args.port}/api/whoami   "
        f"({meta.get('pitcher') or meta.get('playId')})",
        flush=True,
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()

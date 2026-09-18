"""Export a play and serve the three.js viewer.

    python3 serve.py /path/to/play_dir [--port 8765] [--fps 20] [--head-pose SMART_VISION]
"""
import argparse
import shutil
import sys
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_play import export_play
from views import HEAD_POSE, HEAD_POSES

VIEWER_ROOT = Path(__file__).resolve().parent
DATA = VIEWER_ROOT / "data"


def prepare(play_dir, fps, full, head_pose=None):
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
    return payload


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(VIEWER_ROOT), **kwargs)

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("play_dir")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--head-pose", choices=list(HEAD_POSES), default=HEAD_POSE)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()
    prepare(args.play_dir, args.fps, args.full, args.head_pose)
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"viewer  http://127.0.0.1:{args.port}/", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()

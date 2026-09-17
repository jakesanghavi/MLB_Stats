import requests
import struct
import flatbuffers
import json
import pkgutil
import importlib
import MLB  # Your generated FlatBuffers Python package

# -----------------------------
# CONFIG
# -----------------------------
HEADERS = {
    'sec-ch-ua-platform': '"macOS"',
    'authorization': 'eyJraWQiOiIya0M2cWZGcDBEWWlsZkpreHo2ODk5QlpTa3cxeklKODBYZGFreVM0bS1ZIiwiYWxnIjoiUlMyNTYifQ.eyJ2ZXIiOjEsImp0aSI6IkFULmNJaDVwQW54ajlPNW5teGkyRnZlMnBya2syWkI4YjRpM2QyZ0w5MFVGNjQiLCJpc3MiOiJodHRwczovL2lkcy5tbGIuY29tL29hdXRoMi9hdXMxbTA4OHlLMDdub0JmaDM1NiIsImF1ZCI6ImFwaTovL21sYl9kZWZhdWx0IiwiaWF0IjoxNzc1NDk2MzAwLCJleHAiOjE3NzU1ODI3MDAsImNpZCI6IjBvYXA3d2E4NTdqY3ZQbFo1MzU1IiwidWlkIjoiMDB1ODV0emd4Y3Z6dWxrTHgzNTYiLCJzY3AiOlsib3BlbmlkIiwiZW1haWwiXSwiYXV0aF90aW1lIjoxNzc1NDg2NTI3LCJzdWIiOiJqYWtlLnNhbmdoYXZpQGdtYWlsLmNvbSIsImlwaWQiOiI0NTc1MzE2OCIsIm1sYl9lbnRpdGxlbWVudCI6dHJ1ZSwiZ3VpZCI6ImE1N2M1MTYzZGVjOWRkNjkyMTY0NDY1YmYyMjVkZTg0In0.F4iV0Uv9crSLwV4QbiAZu93f0OQ4MAmgzQHCI-kkabki7jZDP4TKFBU2_iM2KhW7wPFJ1AUClv2X4Wm9Z02NeITcpQwI5HT2gbHmP6Hbsb1gP4TrptGgyxXrrSQzmJFH91yGMJ_FjQa_uFLTUFQghOv_8dPBlMMMENTBeD-KbRef8C17A2Rbq4_G1IPtLDeiiB9JPqLNgDCv0OqacTwd54DKrqgcG_tLhc94CkoaQLh4lCqZ45YfrymBhsUgyyvNuqeOOmoCPTZI9MY0PJqO8kTl7EOKN768pj4j7cNWWKUlRmLb4ooBpjqUw1luL7Q9tIucfKyoVtImQRjA9LdnlQ',
    'x-mannequin-client': 'gameday',
    'Referer': 'https://www.mlb.com/gameday/padres-vs-red-sox/2026/04/05/824781/final/box',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36',
    'sec-ch-ua': '"Chromium";v="146", "Not-A.Brand";v="24", "Google Chrome";v="146"',
    'sec-ch-ua-mobile': '?0',
}

BASE_URL = "https://fieldvision-hls-beta.mlbinfra.com/mannequin/824781/plays/b61b507c-eeca-39bb-bdc9-56b55198ba70/1.6.1/"
BIN_FILES = ["11.bin", "12.bin", "13.bin"]  # example bin files

# -----------------------------
# FETCH BIN FILES
# -----------------------------
def fetch_bins():
    buffers = {}
    for file in BIN_FILES:
        url = BASE_URL + file
        resp = requests.get(url, headers=HEADERS)
        print(f"Fetching {file}: {resp.status_code}")
        if resp.status_code == 200:
            buffers[file] = resp.content
        else:
            print(f"❌ Failed: {file}")
    return buffers

# -----------------------------
# SIZE-PREFIXED FLATBUFFER ROOT
# -----------------------------
def get_root_buffer(buf):
    """
    Mimic JS getSizePrefixedRoot:
    - Skip 4-byte size prefix
    - Read 4-byte root table offset
    - Return sliced buffer at root
    """
    if len(buf) < 8:
        return buf  # too small to have prefix
    root_offset = struct.unpack_from("<I", buf, 4)[0]
    return buf[4 + root_offset:]

# -----------------------------
# UNIVERSAL UNPACKER
# -----------------------------
def unpack_fb(obj):
    if obj is None:
        return None

    result = {}
    for attr in dir(obj):
        if attr.startswith("_") or attr.endswith("Length") or attr.endswith("Array"):
            continue

        method = getattr(obj, attr)
        if callable(method):
            try:
                value = method()
            except TypeError:
                continue
            except Exception:
                continue

            # Convert bytes -> string
            if isinstance(value, bytes):
                try:
                    value = value.decode("utf-8")
                except UnicodeDecodeError:
                    value = str(value)

            # Nested FlatBuffers
            if hasattr(value, "unpack") and callable(getattr(value, "unpack")):
                result[attr] = unpack_fb(value)
            else:
                result[attr] = value

    # Handle vectors
    for attr in dir(obj):
        if attr.endswith("Length"):
            base = attr[:-6]
            length_method = getattr(obj, attr)
            accessor_method = getattr(obj, base, None)
            if accessor_method and callable(accessor_method):
                length = length_method()
                result[base] = [accessor_method(i) for i in range(length)]

        elif attr.endswith("Array"):
            base = attr[:-5]
            accessor_method = getattr(obj, attr)
            if callable(accessor_method):
                array_val = accessor_method()
                if array_val is not None:
                    result[base] = [
                        v if not isinstance(v, bytes) else v.decode("utf-8")
                        for v in array_val
                    ]

    return result

# -----------------------------
# TRY EVERY MLB CLASS
# -----------------------------
def try_all_classes(buf):
    unpacked = {}
    root_buf = get_root_buffer(buf)
    for _, name, _ in pkgutil.iter_modules(MLB.__path__):
        try:
            module = importlib.import_module(f"MLB.{name}")
            for cls_name in dir(module):
                cls = getattr(module, cls_name)
                if isinstance(cls, type) and hasattr(cls, "GetRootAs"):
                    try:
                        obj = cls.GetRootAs(root_buf, 0)
                        data = unpack_fb(obj)
                        if any(v != 0 and v is not None for v in flatten_values(data)):
                            unpacked[cls_name] = data
                    except Exception:
                        continue
        except Exception:
            continue
    return unpacked

def flatten_values(d):
    """Flatten nested dict to list of all leaf values"""
    if isinstance(d, dict):
        for v in d.values():
            yield from flatten_values(v)
    elif isinstance(d, list):
        for v in d:
            yield from flatten_values(v)
    else:
        yield d

# -----------------------------
# MAIN
# -----------------------------
def main():
    buffers = fetch_bins()
    final_result = {}

    for fname, buf in buffers.items():
        print(f"\n🔹 Processing {fname}")
        data = try_all_classes(buf)
        if not data:
            data = "UNKNOWN FORMAT OR EMPTY"
        final_result[fname] = data

    print("\n🔥 FINAL RESULT:\n")
    print(json.dumps(final_result, indent=2))

if __name__ == "__main__":
    main()

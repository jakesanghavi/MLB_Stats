"""Resolve Gameday 3D player-mesh URLs on the FieldVision CDN.

The tracking wire is pose only. The human-shaped body (head, jersey, cap,
gloves, shoes) is ``generic-lod.gltf`` + ``generic-lod.bin`` plus JPEG
atlases. See PLAYER_MESH_NOTES.md.
"""
import json
import shutil
import urllib.request
from pathlib import Path
from urllib.parse import quote

from stadium import ASSET_BASE

_UA = "Mozilla/5.0 (compatible; gameday3d-player-assets/1.0)"
_ASSET_GLTF = Path(__file__).resolve().parent / "assets" / "generic-lod.gltf"

OUTFIT_ROLES = (
    "pitcher", "batter", "catcher", "fielder",
    "coach", "umpire", "plate-umpire",
)

# playEvent kind -> variants.json / material slot
KIND_TO_SLOT = {
    "Jersey": "Jersey_Top",
    "Pants": "Jersey_Bottom",
    "Cap": "Cap",
}

LEGACY_TEXTURES = (
    "Skin_Diffuse.jpg",
    "Jersey_Top_Diffuse.jpg",
    "Jersey_Bottom_Diffuse.jpg",
    "Shoes_Diffuse.jpg",
    "Gear_Diffuse.jpg",
    "CatcherGear_Diffuse.jpg",
    "UniformAccessories_Diffuse.jpg",
)


def _join(*parts):
    base = ASSET_BASE.rstrip("/")
    tail = "/".join(p.strip("/") for p in parts)
    return f"{base}/{tail}"


def generic_gltf_url():
    return _join("models", "generic", "generic-lod.gltf")


def generic_bin_url():
    return _join("models", "generic", "generic-lod.bin")


def variants_url():
    return _join("skins", "materials", "variants.json")


def outfit_url(role):
    if role not in OUTFIT_ROLES:
        raise ValueError(f"unknown outfit role {role!r}")
    return _join("skins", "outfits", f"{role}.json")


def legacy_texture_url(name):
    return _join("skins", "textures", "legacy", name)


def team_texture_url(rel_path):
    """``2026/TEX/tex_uni_top_buttonup_home_diffuse.jpg`` -> absolute URL."""
    return _join("skins", "textures", quote(rel_path, safe="/"))


def load_uniforms(play_dir):
    p = Path(play_dir) / "uniforms.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def codes_from_uniforms(uniforms):
    """side -> {Jersey_Top|Jersey_Bottom|Cap: code}."""
    out = {}
    for side in ("home", "away"):
        block = uniforms.get(side) or {}
        slots = {}
        for item in block.get("items") or []:
            slot = KIND_TO_SLOT.get(item.get("kind"))
            if slot and item.get("code"):
                slots[slot] = item["code"]
        if slots:
            out[side] = {"teamId": block.get("teamId"), "slots": slots}
    return out


def resolve_play_textures(uniforms, variants):
    """Map home/away material slots to CDN URLs via variants.json."""
    catalog = (variants or {}).get("textures") or {}
    resolved = {}
    for side, info in codes_from_uniforms(uniforms).items():
        slots = {}
        for slot, code in info["slots"].items():
            rel = catalog.get(code)
            slots[slot] = {
                "code": code,
                "relative": rel,
                "url": team_texture_url(rel) if rel else None,
            }
        resolved[side] = {"teamId": info["teamId"], "slots": slots}
    return resolved


OUTFIT_FOR_TYPE = {
    "pitcher": "pitcher",
    "catcher": "catcher",
    "batter": "batter",
    "fielder": "fielder",
    "runner": "fielder",
    "coach": "coach",
    "umpire": "umpire",
    "plate-umpire": "plate-umpire",
}


def outfit_role(actor_type):
    return OUTFIT_FOR_TYPE.get(actor_type or "", "fielder")


def player_sides(metadata):
    """playerId -> 'home' | 'away' from the play boxscore."""
    teams = ((metadata or {}).get("boxscore") or {}).get("teams") or {}
    out = {}
    for side in ("home", "away"):
        players = (teams.get(side) or {}).get("players") or {}
        if not isinstance(players, dict):
            continue
        for rec in players.values():
            pid = ((rec or {}).get("person") or {}).get("id")
            if pid:
                out[int(pid)] = side
    return out


def actor_side(actor_type, player_id, sides):
    if actor_type in ("umpire", "plate-umpire"):
        return None
    if player_id is not None:
        try:
            pid = int(player_id)
        except (TypeError, ValueError):
            pid = None
        if pid and pid in sides:
            return sides[pid]
    if actor_type in ("batter", "coach"):
        return None  # filled later from the batter majority
    return None


def _download(url, dest):
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        dest.write_bytes(r.read())
    return dest


def ensure_player_assets(data_root, play_dir=None):
    """Download the generic character + this play's team textures into data_root.

    Layout matches the glTF relative URIs so GLTFLoader can resolve them::

        data/models/generic/generic-lod.gltf
        data/models/generic/generic-lod.bin
        data/skins/textures/legacy/*.jpg
        data/skins/textures/{year}/{ABBR}/*.jpg
        data/skins/outfits/{role}.json
        data/skins/materials/variants.json
    """
    data_root = Path(data_root)
    generic_dir = data_root / "models" / "generic"
    generic_dir.mkdir(parents=True, exist_ok=True)
    dest_gltf = generic_dir / "generic-lod.gltf"
    if _ASSET_GLTF.exists():
        if not dest_gltf.exists() or dest_gltf.stat().st_size != _ASSET_GLTF.stat().st_size:
            shutil.copy2(_ASSET_GLTF, dest_gltf)
    else:
        _download(generic_gltf_url(), dest_gltf)
    _download(generic_bin_url(), generic_dir / "generic-lod.bin")

    variants_path = data_root / "skins" / "materials" / "variants.json"
    _download(variants_url(), variants_path)
    variants = json.loads(variants_path.read_text())

    outfits = {}
    for role in OUTFIT_ROLES:
        dest = data_root / "skins" / "outfits" / f"{role}.json"
        _download(outfit_url(role), dest)
        outfits[role] = json.loads(dest.read_text()).get("items") or []

    for name in LEGACY_TEXTURES:
        _download(legacy_texture_url(name),
                  data_root / "skins" / "textures" / "legacy" / name)

    uniforms = load_uniforms(play_dir) if play_dir else {}
    resolved = resolve_play_textures(uniforms, variants)
    sides = {}
    for side, info in resolved.items():
        slots = {}
        for slot, rec in info["slots"].items():
            rel = rec.get("relative")
            if not rel:
                continue
            dest = data_root / "skins" / "textures" / rel
            _download(rec["url"], dest)
            slots[slot] = f"data/skins/textures/{rel}"
        sides[side] = slots

    return {
        "model": "data/models/generic/generic-lod.gltf",
        "lod": 0,
        "outfits": outfits,
        "sides": sides,
    }


def catalog(play_dir=None, variants=None):
    """URLs Gameday fetches for the generic body + this play’s uniforms."""
    uniforms = load_uniforms(play_dir) if play_dir else {}
    return {
        "assetBase": ASSET_BASE,
        "model": generic_gltf_url(),
        "meshBin": generic_bin_url(),
        "variants": variants_url(),
        "outfits": {role: outfit_url(role) for role in OUTFIT_ROLES},
        "legacyTextures": {name: legacy_texture_url(name) for name in LEGACY_TEXTURES},
        "uniforms": resolve_play_textures(uniforms, variants),
    }


if __name__ == "__main__":
    import sys
    play = sys.argv[1] if len(sys.argv) > 1 else None
    variants = None
    if play:
        vpath = Path(play) / "variants.json"
        # optional local cache; otherwise only codes are printed
        if vpath.exists():
            variants = json.loads(vpath.read_text())
    print(json.dumps(catalog(play, variants), indent=2))

"""Resolve Gameday 3D player-mesh URLs on the FieldVision CDN.

The tracking wire is pose only. The human-shaped body (head, jersey, cap,
gloves, shoes) is ``generic-lod.gltf`` + ``generic-lod.bin`` plus JPEG
atlases. See PLAYER_MESH_NOTES.md.
"""
import json
from pathlib import Path
from urllib.parse import quote

from stadium import ASSET_BASE

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

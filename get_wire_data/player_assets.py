"""Resolve Gameday 3D player-mesh URLs on the FieldVision CDN.

The tracking wire is pose only. The human-shaped body (head, jersey, cap,
gloves, shoes) is ``generic-lod.gltf`` + ``generic-lod.bin`` plus JPEG
atlases. Helmets / gloves / catcher gear are tinted from per-team material
JSON + mask PNGs. See PLAYER_MESH_NOTES.md.
"""
import json
import shutil
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import quote

import numpy as np
from PIL import Image

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

MATERIAL_SLOTS = (
    "Jersey_Top", "Jersey_Bottom", "Cap", "Skin",
    "Gear", "CatcherGear", "Shoes", "UniformAccessories",
)

LEGACY_TEXTURES = (
    "Skin_Diffuse.jpg",
    "Jersey_Top_Diffuse.jpg",
    "Jersey_Bottom_Diffuse.jpg",
    "Shoes_Diffuse.jpg",
    "Gear_Diffuse.jpg",
    "CatcherGear_Diffuse.jpg",
    "UniformAccessories_Diffuse.jpg",
    "Gear_Mask.png",
    "CatcherGear_Mask.png",
    "Shoes_Mask.png",
    "UniformAccessories_Mask.png",
    "UMP_Home_Jersey_Albedo.jpg",
    "UMP_Home_Pants_Albedo.jpg",
)

# Gameday VA table (teamId -> latest abbreviation).
TEAM_ID_ABBR = {
    108: "LAA", 109: "ARI", 110: "BAL", 111: "BOS", 112: "CHC",
    113: "CIN", 114: "CLE", 115: "COL", 116: "DET", 117: "HOU",
    118: "KC", 119: "LAD", 120: "WSH", 121: "NYM", 133: "ATH",
    134: "PIT", 135: "SD", 136: "SEA", 137: "SF", 138: "STL",
    139: "TB", 140: "TEX", 141: "TOR", 142: "MIN", 143: "PHI",
    144: "ATL", 145: "CWS", 146: "MIA", 147: "NYY", 158: "MIL",
}

# Fallback only — used when skins/materials/{id}_{ABBR}_{HOME|AWAY}.json 404s.
TEAM_COLORS = {
    "ARI": ("#A71930", "#E3D4AD", "#000000"),
    "ATL": ("#CE1141", "#13274F", "#EAAA00"),
    "BAL": ("#DF4601", "#000000", "#FFFFFF"),
    "BOS": ("#BD3039", "#0C2340", "#FFFFFF"),
    "CHC": ("#0E3386", "#CC3433", "#FFFFFF"),
    "CWS": ("#27251F", "#C4CED4", "#FFFFFF"),
    "CIN": ("#C6011F", "#000000", "#FFFFFF"),
    "CLE": ("#00385D", "#E50022", "#FFFFFF"),
    "COL": ("#333366", "#C4CED4", "#131413"),
    "DET": ("#0C2340", "#FA4616", "#FFFFFF"),
    "HOU": ("#002D62", "#EB6E1F", "#F4911E"),
    "KC": ("#004687", "#BD9B60", "#FFFFFF"),
    "LAA": ("#003263", "#BA0021", "#862633"),
    "LAD": ("#005A9C", "#EF3E42", "#A5ACAF"),
    "MIA": ("#00A3E0", "#EF3340", "#41748D"),
    "MIL": ("#12284B", "#FFC52F", "#FFFFFF"),
    "MIN": ("#002B5C", "#D31145", "#B9975B"),
    "NYM": ("#002D72", "#FF5910", "#FFFFFF"),
    "NYY": ("#003087", "#E4002C", "#0C2340"),
    "ATH": ("#003831", "#EFB21E", "#A2AAAD"),
    "OAK": ("#003831", "#EFB21E", "#A2AAAD"),
    "PHI": ("#E81828", "#002D72", "#FFFFFF"),
    "PIT": ("#27251F", "#FDB827", "#FFFFFF"),
    "SD": ("#2F241D", "#FFC425", "#FFFFFF"),
    "SF": ("#FD5A1E", "#27251F", "#EFD19F"),
    "SEA": ("#0C2C56", "#005C5C", "#C4CED4"),
    "STL": ("#C41E3A", "#0C2340", "#FEDB00"),
    "TB": ("#092C5C", "#8FBCE6", "#F5D130"),
    "TEX": ("#003278", "#C0111F", "#FFFFFF"),
    "TOR": ("#134A8E", "#1D2D5C", "#E8291C"),
    "WSH": ("#AB0003", "#14225A", "#FFFFFF"),
}

GLOVE_BROWN = "#6B3F2A"
UMPIRE_BLACK = "#1A1A1A"

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


def material_template_url(stem):
    return _join("skins", "materials", f"{stem}.json")


def material_stems(team_id, abbr, side):
    """CDN stems Gameday fetches for ``skins/materials/{stem}.json``."""
    if side == "umpire" or (abbr or "").lower() == "umpire":
        return ["0_umpire"]
    try:
        tid = int(team_id)
    except (TypeError, ValueError):
        tid = 0
    abbr = (abbr or TEAM_ID_ABBR.get(tid) or "MLB").upper()
    if abbr in ("MLB", "UMPIRE"):
        tid = 0
    suffix = "HOME" if side == "home" else "AWAY"
    stems = [f"{tid}_{abbr}_{suffix}"]
    if abbr == "ATH":
        stems.append(f"{tid}_OAK_{suffix}")
    elif abbr == "OAK":
        stems.append(f"{tid}_ATH_{suffix}")
    return stems


def load_uniforms(play_dir):
    p = Path(play_dir) / "uniforms.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def load_metadata(play_dir):
    p = Path(play_dir) / "metadata.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def codes_from_uniforms(uniforms):
    """side -> {Jersey_Top|Jersey_Bottom|Cap: code}."""
    out = {}
    for side in ("home", "away"):
        block = uniforms.get(side) or {}
        slots = {}
        variant = None
        for item in block.get("items") or []:
            slot = KIND_TO_SLOT.get(item.get("kind"))
            if slot and item.get("code"):
                slots[slot] = item["code"]
            if item.get("kind") == "Jersey" and item.get("variant") is not None:
                variant = item.get("variant")
        if slots:
            out[side] = {
                "teamId": block.get("teamId"),
                "slots": slots,
                "variant": variant,
            }
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
        resolved[side] = {
            "teamId": info["teamId"],
            "variant": info.get("variant"),
            "slots": slots,
        }
    return resolved


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
        return "umpire"
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


def play_teams(play_dir):
    """home/away -> {teamId, abbr, variant} from uniforms + boxscore."""
    uniforms = load_uniforms(play_dir) if play_dir else {}
    metadata = load_metadata(play_dir) if play_dir else {}
    teams = ((metadata or {}).get("boxscore") or {}).get("teams") or {}
    out = {}
    for side in ("home", "away"):
        block = uniforms.get(side) or {}
        team = (teams.get(side) or {}).get("team") or {}
        tid = block.get("teamId") or team.get("id")
        abbr = team.get("abbreviation") or TEAM_ID_ABBR.get(int(tid) if tid else -1)
        variant = None
        for item in block.get("items") or []:
            if item.get("kind") == "Jersey":
                variant = item.get("variant")
                break
        out[side] = {"teamId": tid, "abbr": abbr, "variant": variant}
    return out


def expand_tints(tints):
    """Match Gameday primary/secondary/tertiary/quaternary defaults."""
    t = [c for c in (tints or []) if c]
    if not t:
        return None
    primary = t[0]
    secondary = t[1] if len(t) > 1 else t[0]
    tertiary = t[2] if len(t) > 2 else primary
    if len(t) > 3:
        quaternary = t[3]
    elif len(t) > 2:
        quaternary = secondary
    else:
        quaternary = primary
    return [primary, secondary, tertiary, quaternary]


def _hex_rgb(color):
    h = str(color).lstrip("#")
    if len(h) != 6:
        return (0.0, 0.0, 0.0)
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def bake_tinted_map(base_path, mask_path, tints, dest):
    """base * Gameday mask mix (R/G/B/(1-A) -> primary/secondary/tertiary/quaternary)."""
    colors = expand_tints(tints)
    if not colors:
        shutil.copy2(base_path, dest)
        return dest
    base_img = Image.open(base_path).convert("RGB")
    mask_img = Image.open(mask_path).convert("RGBA").resize(base_img.size, Image.BILINEAR)
    base = np.asarray(base_img, dtype=np.float32) / 255.0
    mask = np.asarray(mask_img, dtype=np.float32) / 255.0
    cols = [np.array(_hex_rgb(c), dtype=np.float32) for c in colors]
    color = np.ones_like(base)
    for i in range(3):
        w = mask[..., i][..., None]
        color = color * (1.0 - w) + cols[i] * w
    w = (1.0 - mask[..., 3])[..., None]
    color = color * (1.0 - w) + cols[3] * w
    out = np.clip(base * color, 0.0, 1.0)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((out * 255.0).astype(np.uint8), "RGB").save(dest, quality=92)
    return dest


def fallback_template(abbr, side):
    """Team-color tints when the CDN material JSON is missing."""
    if side == "umpire":
        black = [UMPIRE_BLACK, "#2A2A2A", UMPIRE_BLACK, UMPIRE_BLACK]
        return {
            "Jersey_Top": {"base": "legacy/UMP_Home_Jersey_Albedo.jpg"},
            "Jersey_Bottom": {"base": "legacy/UMP_Home_Pants_Albedo.jpg"},
            "Cap": {"base": "legacy/UMP_Home_Jersey_Albedo.jpg"},
            "Skin": {"base": "legacy/Skin_Diffuse.jpg"},
            "Gear": {
                "base": "legacy/Gear_Diffuse.jpg",
                "mask": "legacy/Gear_Mask.png",
                "tints": black,
            },
            "CatcherGear": {
                "base": "legacy/CatcherGear_Diffuse.jpg",
                "mask": "legacy/CatcherGear_Mask.png",
                "tints": black,
            },
            "Shoes": {
                "base": "legacy/Shoes_Diffuse.jpg",
                "mask": "legacy/Shoes_Mask.png",
                "tints": [UMPIRE_BLACK, "#FCFAFA"],
            },
            "UniformAccessories": {
                "base": "legacy/UniformAccessories_Diffuse.jpg",
                "mask": "legacy/UniformAccessories_Mask.png",
                "tints": black,
            },
        }
    c1, c2, c3 = TEAM_COLORS.get((abbr or "").upper(), ("#333333", "#888888", "#FFFFFF"))
    return {
        "Skin": {"base": "legacy/Skin_Diffuse.jpg"},
        "Gear": {
            "base": "legacy/Gear_Diffuse.jpg",
            "mask": "legacy/Gear_Mask.png",
            "tints": [c1, c2, c3, c1],
        },
        "CatcherGear": {
            "base": "legacy/CatcherGear_Diffuse.jpg",
            "mask": "legacy/CatcherGear_Mask.png",
            "tints": [c2, c1, c3, c2],
        },
        "Shoes": {
            "base": "legacy/Shoes_Diffuse.jpg",
            "mask": "legacy/Shoes_Mask.png",
            "tints": [GLOVE_BROWN, "#4A2C1A"],
        },
        "UniformAccessories": {
            "base": "legacy/UniformAccessories_Diffuse.jpg",
            "mask": "legacy/UniformAccessories_Mask.png",
            "tints": [c2, c1, c3],
        },
    }


def slot_with_variant(slot, variant):
    """Apply material-JSON alt-uniform tint overrides when present."""
    if not slot:
        return {}
    out = dict(slot)
    variants = slot.get("variants") or {}
    if variant is not None and str(variant) in variants:
        over = variants[str(variant)]
        if over.get("tints"):
            out["tints"] = over["tints"]
        if over.get("base"):
            out["base"] = over["base"]
    return out


def _download(url, dest):
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        dest.write_bytes(r.read())
    return dest


def _try_download(url, dest):
    try:
        return _download(url, dest)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError):
        return None


def fetch_material_template(data_root, team_id, abbr, side):
    mats = data_root / "skins" / "materials"
    for stem in material_stems(team_id, abbr, side):
        dest = mats / f"{stem}.json"
        if dest.exists() and dest.stat().st_size > 0:
            return json.loads(dest.read_text()), stem
        got = _try_download(material_template_url(stem), dest)
        if got:
            return json.loads(got.read_text()), stem
    return fallback_template(abbr, side), None


def ensure_player_assets(data_root, play_dir=None):
    """Download the generic character + this play's team textures into data_root.

    Layout matches the glTF relative URIs so GLTFLoader can resolve them::

        data/models/generic/generic-lod.gltf
        data/models/generic/generic-lod.bin
        data/skins/textures/legacy/*.jpg
        data/skins/textures/{year}/{ABBR}/*.jpg
        data/skins/baked/{side}_{slot}.jpg
        data/skins/outfits/{role}.json
        data/skins/materials/variants.json
        data/skins/materials/{id}_{ABBR}_{HOME|AWAY}.json
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

    tex_root = data_root / "skins" / "textures"
    for name in LEGACY_TEXTURES:
        _try_download(legacy_texture_url(name), tex_root / "legacy" / name)

    uniforms = load_uniforms(play_dir) if play_dir else {}
    resolved = resolve_play_textures(uniforms, variants)
    teams = play_teams(play_dir) if play_dir else {}

    sides = {}
    for side in ("home", "away", "umpire"):
        info = teams.get(side) or {}
        tid = info.get("teamId")
        abbr = info.get("abbr")
        variant = info.get("variant")
        if side != "umpire" and not tid and not abbr:
            continue
        template, _stem = fetch_material_template(data_root, tid, abbr, side)
        maps = {}
        for slot in MATERIAL_SLOTS:
            spec = slot_with_variant(template.get(slot) or {}, variant)
            # Live-game jersey / pants / cap codes beat the material-JSON year.
            live = ((resolved.get(side) or {}).get("slots") or {}).get(slot) or {}
            rel = live.get("relative") or spec.get("base")
            if not rel:
                continue
            src = tex_root / rel
            if live.get("url"):
                _try_download(live["url"], src)
            else:
                _try_download(team_texture_url(rel), src)
            if not src.exists() or src.stat().st_size <= 0:
                continue
            mask_rel = spec.get("mask")
            tints = spec.get("tints")
            if mask_rel and tints:
                mask_path = tex_root / mask_rel
                _try_download(team_texture_url(mask_rel), mask_path)
                if mask_path.exists():
                    baked = data_root / "skins" / "baked" / f"{side}_{slot}.jpg"
                    bake_tinted_map(src, mask_path, tints, baked)
                    maps[slot] = f"data/skins/baked/{side}_{slot}.jpg"
                    continue
            maps[slot] = f"data/skins/textures/{rel}"
        if maps:
            sides[side] = maps

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
        "legacyTextures": {name: legacy_texture_url(name) for name in LEGACY_TEXTURES if name.endswith(".jpg")},
        "uniforms": resolve_play_textures(uniforms, variants),
        "materials": {"umpire": material_template_url("0_umpire")},
    }


if __name__ == "__main__":
    import sys
    play = sys.argv[1] if len(sys.argv) > 1 else None
    variants = None
    if play:
        vpath = Path(play) / "variants.json"
        if vpath.exists():
            variants = json.loads(vpath.read_text())
    print(json.dumps(catalog(play, variants), indent=2))

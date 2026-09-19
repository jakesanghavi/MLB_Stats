"""URL helpers for the generic skinned character + team atlases."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from player_assets import (
    KIND_TO_SLOT, OUTFIT_ROLES, TEAM_COLORS, actor_side, catalog,
    codes_from_uniforms, expand_tints, fallback_template, generic_bin_url,
    generic_gltf_url, material_stems, outfit_role, outfit_url, player_sides,
    resolve_play_textures, team_texture_url, variants_url,
)


def test_cdn_paths():
    assert generic_gltf_url().endswith("/models/generic/generic-lod.gltf")
    assert generic_bin_url().endswith("/models/generic/generic-lod.bin")
    assert variants_url().endswith("/skins/materials/variants.json")
    assert outfit_url("catcher").endswith("/skins/outfits/catcher.json")
    assert "fv-assets.mlb.com/v/" in generic_gltf_url()
    for role in OUTFIT_ROLES:
        outfit_url(role)
    try:
        outfit_url("mascot")
        assert False, "expected ValueError"
    except ValueError:
        pass
    print("ok cdn paths")


def test_uniforms_to_urls():
    uniforms = {
        "home": {"teamId": 140, "items": [
            {"kind": "Jersey", "code": "140_jersey_1_2026"},
            {"kind": "Pants", "code": "140_pants_1_2026"},
            {"kind": "Cap", "code": "140_hat_1_2026"},
        ]},
        "away": {"teamId": 111, "items": [
            {"kind": "Jersey", "code": "111_jersey_2_2026"},
        ]},
    }
    codes = codes_from_uniforms(uniforms)
    assert codes["home"]["slots"][KIND_TO_SLOT["Jersey"]] == "140_jersey_1_2026"
    assert codes["home"]["slots"]["Cap"] == "140_hat_1_2026"
    variants = {"textures": {
        "140_jersey_1_2026": "2026/TEX/tex_uni_top_buttonup_home_diffuse.jpg",
        "111_jersey_2_2026": "2026/BOS/bos_uni_top_buttonup_road_diffuse.jpg",
    }}
    got = resolve_play_textures(uniforms, variants)
    home_j = got["home"]["slots"]["Jersey_Top"]
    assert home_j["relative"] == "2026/TEX/tex_uni_top_buttonup_home_diffuse.jpg"
    assert home_j["url"].endswith("/skins/textures/2026/TEX/tex_uni_top_buttonup_home_diffuse.jpg")
    assert got["away"]["slots"]["Jersey_Top"]["url"].endswith(
        "/skins/textures/2026/BOS/bos_uni_top_buttonup_road_diffuse.jpg"
    )
    assert got["home"]["slots"]["Jersey_Bottom"]["url"] is None  # pants code not in catalog
    print("ok uniforms to urls")


def test_catalog_lists_legacy():
    c = catalog()
    assert "Skin_Diffuse.jpg" in c["legacyTextures"]
    assert set(c["outfits"]) == set(OUTFIT_ROLES)
    print("ok catalog")


def test_outfit_role_and_side():
    assert outfit_role("pitcher") == "pitcher"
    assert outfit_role("plate-umpire") == "plate-umpire"
    assert outfit_role("unknown") == "fielder"
    assert outfit_role("runner") == "fielder"
    sides = {676477: "away", 670032: "home"}
    assert actor_side("pitcher", 676477, sides) == "away"
    assert actor_side("batter", 670032, sides) == "home"
    assert actor_side("umpire", 1, sides) == "umpire"
    assert actor_side("plate-umpire", 1, sides) == "umpire"
    meta = {"boxscore": {"teams": {
        "home": {"players": {"ID1": {"person": {"id": 10}}}},
        "away": {"players": {"ID2": {"person": {"id": 20}}}},
    }}}
    assert player_sides(meta) == {10: "home", 20: "away"}
    print("ok outfit role and side")


def test_real_822845_uniforms_if_present():
    d = Path("/tmp/gd/play_822845_9b398372-272c-3787-bb1a-aebacf8eedab")
    if not d.exists():
        print("skip real 822845 uniforms")
        return
    uniforms = json.loads((d / "uniforms.json").read_text())
    codes = codes_from_uniforms(uniforms)
    assert codes["home"]["teamId"] == 140
    assert codes["home"]["slots"]["Jersey_Top"] == "140_jersey_1_2026"
    assert codes["away"]["slots"]["Jersey_Top"] == "111_jersey_2_2026"
    print("ok real 822845 uniforms")


def test_material_stems_and_tints():
    assert material_stems(140, "TEX", "home") == ["140_TEX_HOME"]
    assert material_stems(111, "BOS", "away") == ["111_BOS_AWAY"]
    assert material_stems(0, "umpire", "umpire") == ["0_umpire"]
    assert "133_OAK_HOME" in material_stems(133, "ATH", "home")
    assert expand_tints(["#111111", "#222222"]) == [
        "#111111", "#222222", "#111111", "#111111",
    ]
    assert expand_tints(["#A", "#B", "#C"]) == ["#A", "#B", "#C", "#B"]
    ump = fallback_template("TEX", "umpire")
    assert ump["Jersey_Top"]["base"].endswith("UMP_Home_Jersey_Albedo.jpg")
    assert ump["Gear"]["tints"][0].startswith("#1")
    tex = fallback_template("TEX", "home")
    assert tex["Gear"]["tints"][0] == TEAM_COLORS["TEX"][0]
    assert tex["CatcherGear"]["tints"][0] == TEAM_COLORS["TEX"][1]
    assert tex["Shoes"]["tints"][0].startswith("#6")
    print("ok material stems and tints")


if __name__ == "__main__":
    test_cdn_paths()
    test_uniforms_to_urls()
    test_catalog_lists_legacy()
    test_outfit_role_and_side()
    test_real_822845_uniforms_if_present()
    test_material_stems_and_tints()
    print("ok")

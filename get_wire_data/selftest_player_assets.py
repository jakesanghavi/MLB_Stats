"""URL helpers for the generic skinned character + team atlases."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from player_assets import (
    KIND_TO_SLOT, OUTFIT_ROLES, catalog, codes_from_uniforms,
    generic_bin_url, generic_gltf_url, outfit_url, resolve_play_textures,
    team_texture_url, variants_url,
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


if __name__ == "__main__":
    test_cdn_paths()
    test_uniforms_to_urls()
    test_catalog_lists_legacy()
    test_real_822845_uniforms_if_present()
    print("ok")

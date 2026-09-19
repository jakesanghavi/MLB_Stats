# Player meshes (jerseys, heads, hats, gloves, shoes)

Gameday 3D’s people are **not** extra tracking data. They are one generic
skinned character from the public FieldVision CDN, posed with the joint
quaternions we already decode. We only ever used the glTF **skeleton**
(joint names + bind TRS) and drew cylinders between joints. The mesh
buffer, outfit pieces, and team textures were never loaded.

## What we already have vs what Gameday draws

| Piece | We have it? | What we do with it |
|---|---|---|
| `actorPoses` root + packed quats | yes, in the `.bin` tracking chunks | FK → bone segments |
| `generic-lod.gltf` joint hierarchy | yes, `assets/generic-lod.gltf` | `RigSkeleton` only |
| `generic-lod.bin` (~6.4 MB skinned mesh) | yes, `ensure_player_assets` | cloned onto each actor when `SHOW_PLAYER_MESH` |
| Per-role outfit JSON (which pieces are on) | yes | show/hide jersey, cap, gloves, pads |
| `uniforms.json` (jersey / pants / cap codes) | yes, downloaded with the play | resolved through `variants.json` |
| Team + skin JPEG atlases | yes | `Jersey_Top` / `Jersey_Bottom` / `Cap` maps |

`rig.py` says this explicitly: *“the external .bin mesh buffer is not needed.”*
That is why the viewer looks like a stick figure with thick limbs: the
human-shaped geometry lives in `generic-lod.bin`, not in the tracking wire.

## Where Gameday loads the body

Same `assetBase` as the ballpark (`kI` in `gd.@bvg_poser.min.js`):

```
https://fv-assets.mlb.com/v/58242e42c28513752c1cb5776bdf0da7f0679d5e
```

The poser default is `modelName: "generic/generic-lod.gltf"`. It loads:

```
{assetBase}/models/generic/generic-lod.gltf
{assetBase}/models/generic/generic-lod.bin          # 6,399,384 bytes, 148 skinned primitives
```

The glTF is a single `Generic_LOD` character (scale 0.01, +90° X — same
cm→m trick as the park). It already contains the built-up human:

- `Head_30`, `Eyes`, `Mouth`
- `Jersey_Standard_ButtonUp`, `Pants_Standard_Full`, `Undershirt_Standard`, `Belt_Standard`
- `Headgear_Cap`, `Headgear_Catcher`, `Headgear_*_Helmet`
- `Gloves_Batter`, `Gloves_*_Fielder`, `Gloves_*_Catcher`
- `Shoes_Standard`
- `CatcherGear_Armor_*`, elbow/shin guards, sleeves, wristbands, …

Each piece has LOD0–LOD3. Materials on the mesh are:

`Skin`, `Jersey_Top`, `Jersey_Bottom`, `Cap`, `Shoes`, `Gear`,
`CatcherGear`, `UniformAccessories`.

Gameday clones this template per actor, then:

1. Sets `joint_Pelvis` to `rootPos` and writes the tracked local quats
   (exactly what `RigSkeleton.fk` does).
2. Shows/hides pieces from a per-role outfit list.
3. Swaps the jersey/pants/cap albedo for the team’s season texture.

## Outfit lists (which hat / glove / pads)

```
{assetBase}/skins/outfits/{role}.json
```

Roles that 200: `pitcher`, `batter`, `catcher`, `fielder`, `coach`,
`umpire`, `plate-umpire`.

Pitcher (cap + fielder glove):

```json
["Arms_Cutoff", "Arms_Hands", "Belt_Standard", "Gloves_*_Fielder",
 "Head_30", "Eyes", "Headgear_Cap", "Jersey_Standard_ButtonUp",
 "Pants_Standard_Full", "Shoes_Standard", "Undershirt_Standard"]
```

Batter (helmet + batting gloves). Catcher (mask + chest/shin + catcher mitt).
`*` is mirrored for L/R handedness.

## Team textures (the Rangers “T”, Sox road gray, …)

Play-level `uniforms.json` is only **codes**, not meshes:

```json
{"home": {"teamId": 140, "items": [
  {"kind": "Jersey", "code": "140_jersey_1_2026"},
  {"kind": "Pants",  "code": "140_pants_1_2026"},
  {"kind": "Cap",    "code": "140_hat_1_2026"}
]}}
```

The poser maps those codes through a public catalog:

```
{assetBase}/skins/materials/variants.json
```

~1000 entries. Example for this game (TEX home / BOS road):

```
140_jersey_1_2026 -> 2026/TEX/tex_uni_top_buttonup_home_diffuse.jpg
111_jersey_2_2026 -> 2026/BOS/bos_uni_top_buttonup_road_diffuse.jpg
```

Resolved URL:

```
{assetBase}/skins/textures/{path from variants.json}
```

Fallback / default albedos shipped next to the generic rig:

```
{assetBase}/skins/textures/legacy/Skin_Diffuse.jpg
{assetBase}/skins/textures/legacy/Jersey_Top_Diffuse.jpg
{assetBase}/skins/textures/legacy/Jersey_Bottom_Diffuse.jpg
{assetBase}/skins/textures/legacy/Shoes_Diffuse.jpg
{assetBase}/skins/textures/legacy/Gear_Diffuse.jpg
{assetBase}/skins/textures/legacy/CatcherGear_Diffuse.jpg
{assetBase}/skins/textures/legacy/UniformAccessories_Diffuse.jpg
```

Those JPEGs are 512×512 atlases (head + hands on `Skin`, shoe/glove islands
on `Shoes`/`Gear`, logo + cap bill on the team jersey). They are not
separate head/hat/glove models.

## Helmets, gloves, umpire gear (mask tints)

Jersey logos come from `variants.json`. Helmets, batting gloves, catcher
pads, and shoes are **not** extra JPEGs — Gameday fetches a per-team
material template and tints a shared mask:

```
{assetBase}/skins/materials/{teamId}_{ABBR}_{HOME|AWAY}.json
{assetBase}/skins/materials/0_umpire.json
```

Example `140_TEX_HOME.json`: `Gear` (helmet + batting gloves) has
`legacy/Gear_Mask.png` and tints `[navy, white, red, navy]`. `CatcherGear`
and `Shoes` work the same way. Umpires use `UMP_Home_Jersey_Albedo.jpg`
(black). `ensure_player_assets` bakes those mixes to
`data/skins/baked/{side}_{slot}.jpg`.

If the material JSON 404s: umpire gear is black, fielder gloves stay brown,
helmets use team C1, other gear uses C2 with C1/C3 as accents.

## Viewer flag

`SHOW_PLAYER_MESH` in `viewer/viewer.js` (code-only, default `true`).

- `true`: wrap each actor in a clone of `generic-lod` (outfit + team maps),
  posed with the exported joint quats.
- `false`: the previous stick-figure cylinders. No glTF load.

If the mesh fails to load, the viewer falls back to sticks.

`export_play` always writes `bones` / `pose` / `skins` so flipping the flag
only needs a hard-reload of `viewer.js`. `ensure_player_assets` caches the
CDN files under `viewer/data/` (gitignored).

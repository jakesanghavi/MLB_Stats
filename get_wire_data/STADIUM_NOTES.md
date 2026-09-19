# Stadium (ballpark) 3D assets

The Gameday 3D viewer renders the actual ballpark mesh. `reconstruct3d.py` can
draw a downsampled, untextured version in matplotlib when `INCLUDE_FIELD` /
`INCLUDE_STADIUM` are True (or `--field` / `--stadium`).

Needs: `pip install DracoPy trimesh fast-simplification`

## Where the assets live

Same public CDN as the player rig:

```
assetBase = https://fv-assets.mlb.com/v/<hash>
            (current: 58242e42c28513752c1cb5776bdf0da7f0679d5e)

Player rig   : {assetBase}/models/generic/generic-lod.gltf   (+ generic-lod.bin)
               (skinned human — head/jersey/hat/gloves/shoes. See PLAYER_MESH_NOTES.md)
Outfits      : {assetBase}/skins/outfits/{role}.json
Team textures: {assetBase}/skins/materials/variants.json
               + {assetBase}/skins/textures/{year}/{ABBR}/...
Bat          : {assetBase}/models/bat.glb                     (plain glTF)
Ball         : {assetBase}/models/rbi-ball.glb                (plain glTF)
Stadium      : {assetBase}/models/ballparks/{venueId}_{ABBR}.glb
Screens      : {assetBase}/models/ballparks/screens/{name}.glb
Crowd        : {assetBase}/crowd/{name}.glb
Draco decoder: {assetBase}/libs/draco/                        (wasm, used by viewer)
```

`{hash}` comes from the poser bundle (`kI` constant in `gd.@bvg_poser.min.js`) and
changes when MLB redeploys; re-read it from the bundle if downloads 404.

`stadium.find_ballpark_glb` looks in `assets/ballparks/` then downloads that URL.
`.glb` / `.npz` caches there are gitignored.

## Venue → file name

The stadium file is `{venueId}_{abbreviation}.glb`. `venueId` is in each play's
`metadata.json`. Abbreviation is the home team (`boxscore.teams.home.team.abbreviation`)
with a small `VENUE_ABBR` fallback in `stadium.py`. Example: game 823004 has
`venueId = 2889` → `STL` → `models/ballparks/2889_STL.glb`.

## What the file contains

`2889_STL.glb` (~5.8 MB) declares:

```
extensionsUsed: ["KHR_texture_transform", "KHR_draco_mesh_compression"]
nodes: MLB_Ballpark_StartingCube, STL_Field, STL_Stadium, CameraCollider
```

- Geometry is Draco-compressed. `glb.load_glb_nodes` decodes it with DracoPy.
- Node TRS on the field/stadium is scale `0.01` + 90° about X; that yields **meters**
  (cm→m). Gameday then multiplies the whole scene by `FI = 3.28084` (m→ft). Tracking
  is already feet, home at the origin, outfield −Z. After `× 3.28084` the mesh
  snaps to the diamond: home, rubber at z = −60.5, bases, Busch CF wall at 400 ft.
- `STL_Field` ~18k tris (grass/dirt). `STL_Stadium` ~400k tris (bowl). We
  skip the collider and starting cube, then quadric-decimate to ~6k / ~16k tris
  for matplotlib.

## Renderer limits

matplotlib has no textures and poor z-order, so the park is a green field +
translucent gray bowl — not the Gameday look. A real engine (pyrender, three.js)
is the path if we want the JPEG atlas.

## Related decoded data we already have

- `metadata.json`: `venueId`, `ruleSettings`, `boneIdMap`, `batBoneIdMap`, `boxscore`.
- Per frame: ball position, per-actor root + joint quaternions, bat head/handle,
  events.

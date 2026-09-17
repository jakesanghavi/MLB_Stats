# Stadium (ballpark) 3D assets — notes for later

The Gameday 3D viewer renders the actual ballpark mesh. We can download it, but
integrating it into the Python reconstruction is deferred (Draco + heavy mesh).
These notes capture everything needed to pick it up later.

## Where the assets live

Same public CDN as the player rig:

```
assetBase = https://fv-assets.mlb.com/v/<hash>
            (current: 58242e42c28513752c1cb5776bdf0da7f0679d5e)

Player rig   : {assetBase}/models/generic/generic-lod.gltf   (+ generic-lod.bin)
Bat          : {assetBase}/models/bat.glb                     (plain glTF)
Ball         : {assetBase}/models/rbi-ball.glb                (plain glTF)
Stadium      : {assetBase}/models/ballparks/{venueId}_{ABBR}.glb
Screens      : {assetBase}/models/ballparks/screens/{name}.glb
Crowd        : {assetBase}/crowd/{name}.glb
Draco decoder: {assetBase}/libs/draco/                        (wasm, used by viewer)
```

`{hash}` comes from the poser bundle (`kI` constant in `gd.@bvg_poser.min.js`) and
changes when MLB redeploys; re-read it from the bundle if downloads 404.

## Venue → file name

The stadium file is `{venueId}_{abbreviation}.glb`. `venueId` is in each play's
`metadata.json` (`venueId`). The venue→abbreviation map is the `VA` array in the
poser bundle. Example: our sample play (game 823004) has `venueId = 2889` → `STL`
→ `models/ballparks/2889_STL.glb`.

Selected entries (venueId: ABBR): 2889 STL, 3 BOS, 22 LAD, 3313 NYY, 2680 SD,
3289 NYM, 680 SEA, 2395 SF, 3312 MIN, 2681 PHI, 4705 ATL, 15 ARI, 3309 WSH,
2529 ATH, 5325 TEX, ... (full list = `VA` in the bundle).

## Why it's deferred

`2889_STL.glb` (~5.8 MB) declares:

```
extensionsUsed: ["KHR_texture_transform", "KHR_draco_mesh_compression"]
meshes: MLB_Ballpark_StartingCube, STL_Field, STL_Stadium, CameraCollider
```

- The **geometry is Draco-compressed** (`KHR_draco_mesh_compression`). The node
  hierarchy/transforms are readable from the glTF JSON, but vertex data needs a
  Draco decoder. None is installed here (`DracoPy` / `trimesh` / `pygltflib` are
  all missing), so `glb.load_glb_mesh` (plain-glTF only) cannot read it.
- Even decoded, the stadium is a large textured mesh. **matplotlib is a poor mesh
  renderer** (slow, no textures, would dwarf the ~6 ft skeletons). Rendering the
  ballpark properly wants a real 3D engine.

## How to do it later

1. Add a Draco-capable loader: `pip install DracoPy` (decode
   `KHR_draco_mesh_compression` buffer views) or `trimesh[easy]` (which can load
   Draco glTF), or run the CDN wasm decoder.
2. Decide the renderer:
   - **Recommended**: move 3D playback to a real engine — `pyrender`/`trimesh`
     scene, Open3D, or export the scene (skeletons + bat + ball + stadium) to a
     glТF/USD and view in a WebGL/three.js viewer. This gets textures, correct
     scale, and interactivity for free.
   - **matplotlib stopgap**: don't draw the full mesh; extract just the field
     plane / a simplified outline (e.g., decimate, or use the `STL_Field` mesh
     bounds) as a ground reference under the players.
3. Placement: the stadium mesh is authored in the same world frame as the
   tracking (field feet, home plate near origin). It should drop in without extra
   alignment; verify against known landmarks (mound at z ≈ -60 ft, home at 0).

## Related decoded data we already have

- `metadata.json`: `venueId`, `ruleSettings` (strike-zone factors), `boneIdMap`,
  `batBoneIdMap`, `boxscore`.
- Per frame: ball position, per-actor root + joint quaternions (→ skeletons via
  `rig.py`), bat head/handle (`inferredBat`), events.

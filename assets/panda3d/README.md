Panda3D Asset Slots

This directory is the drop-in root for optional real 3D assets used by the
Panda3D runtime. The main pygame trainer does not require these files.

Supported model formats:
- `.glb`
- `.gltf`
- `.obj`
- `.bam`
- `.egg`

Recommended structure:

```text
assets/panda3d/
  manifest.json
  aircraft/
    plane_red.glb
    plane_blue.glb
    plane_green.glb
    plane_yellow.glb
  helicopters/
    helicopter_green.glb
  vehicles/
    truck_olive.glb
  scenery/
    building_hangar.glb
    building_tower.glb
    trees_pine_cluster.glb
  personnel/
    soldiers_patrol.glb
```

The runtime reads `manifest.json` first and resolves the first existing path for
each asset entry. If an asset is missing, the runtime falls back to an internal
3D primitive so the scene still runs.

This keeps the bridge usable while higher-quality art is being sourced.

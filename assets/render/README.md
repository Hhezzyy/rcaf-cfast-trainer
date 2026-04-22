Render Asset Slots

This directory is the drop-in root for optional real 3D assets used by the
ModernGL runtime. The trainer can still run with internal primitives if a model
file is missing.

Supported model formats:
- `.glb`
- `.gltf`
- `.obj`

Recommended structure:

```text
assets/render/
  manifest.json
  rapid_tracking/
    plane_fixed_wing.obj
    helicopter_green.obj
    truck_olive.obj
    vehicle_tracked.obj
    building_hangar.obj
    building_tower.obj
    road_paved_segment.obj
    road_dirt_segment.obj
    terrain_hill_mound.obj
    terrain_lake_patch.obj
    terrain_rock_cluster.obj
    trees_pine_cluster.obj
    soldiers_patrol.obj
  auditory/
    auditory_ball.obj
    auditory_ball.mtl
    auditory_tunnel_segment.obj
    auditory_tunnel_segment.mtl
    auditory_tunnel_rib.obj
    auditory_tunnel_rib.mtl
  scenery/
    building_hangar.glb
    building_tower.glb
    trees_pine_cluster.glb
    shrubs_low_cluster.obj
    shrubs_low_cluster.mtl
    trees_field_cluster.obj
    trees_field_cluster.mtl
    forest_canopy_patch.obj
    forest_canopy_patch.mtl
```

The runtime reads `manifest.json` first and resolves the first existing path for
each asset entry. If an asset is missing, the runtime falls back to an internal
3D primitive so the scene still runs.

Optional manifest tuning fields:
- `hpr_offset_deg`: heading, pitch, roll offsets applied after the scene pose
- `pos_offset`: x, y, z offsets applied after the scene placement
- `scale`: multiplies the scene-authored scale for loaded models

This keeps the 3D scenes usable while higher-quality art is being sourced.

# Spatial Integration

## Purpose

- Spatial memory and mental-rotation task backed by the optional Godot companion renderer.
- The active Godot path tests visible scene evidence only: object presence/absence, category counts, relative spacing, viewpoint matching, top-down map recognition, aircraft routes, aircraft color presence/count, and visible aircraft order.
- Hidden asset IDs such as `HUM2` and grid-cell coordinates such as `M14` are intentionally not valid question content.

## Important Files

- `cfast_trainer/spatial_integration.py`: Python-side legacy engine, payloads, and `build_spatial_integration_test(...)`.
- `cfast_trainer/godot_owned.py`: Godot-owned Spatial Integration config, allowed question families, and drill-code aliases.
- `godot/cfast_3d/scripts/spatial_integration_runtime.gd`: Godot scene generation, study camera, question generation, answer UI, scoring events, and visual map cards.
- `tests/test_spatial_integration_core.py`
- `tests/test_spatial_integration_headless_sim.py`
- `tests/test_spatial_integration_drills.py`
- `tests/test_godot_bridge.py`
- `tests/test_godot_chunk_generation_source.py`
- `tests/godot_spatial_integration_question_probe.gd`

## Lifecycle / Data Flow

1. Python chooses a Spatial Integration run through `spatial_integration_godot_config(...)`.
2. `cfast_trainer/godot_bridge.py` starts the Godot-owned runtime for Spatial Integration.
3. Godot builds each study scene from chunk terrain, visible landmarks, routes, and aircraft tracks.
4. Study views are oblique/rotated; questions hide the scene and show only answer controls.
5. Answers are sent back to Python as progress/completion events for the normal result path.

## Current Question Rules

- All generated Spatial Integration questions use option cards, not typed grid answers.
- `scene_reconstruction` shows four clickable top-down map previews after the question; one is exact and three are difficulty-scaled decoys.
- Static scene questions must use visible categories: building, vehicle, human, sheep, tower, forest, tent, or coarse regions.
- Aircraft questions may refer to visible aircraft colors and visible motion order, but not hidden IDs.
- The north marker is an in-scene marker during study only; it is off-map, larger than a HUD cue, and hidden during questions.

## Persistence / Test Hooks

- Persistence is indirect through the shared app shell and result conversion path.
- `tests/godot_spatial_integration_question_probe.gd` is the strongest fairness check for generated Godot questions.
- `tests/test_godot_chunk_generation_source.py` pins source-level invariants for fair question families, terrain generation, and map preview support.
- Run Godot compile/import checks after GDScript edits:

  ```bash
  /Applications/Godot.app/Contents/MacOS/Godot --headless --path godot/cfast_3d --quit
  ```

## Common Safe Edit Points

- Add or remove active question families in `cfast_trainer/godot_owned.py` and the `STATIC_KINDS` / `AIRCRAFT_KINDS` lists in the Godot runtime together.
- Tune top-down reconstruction distractor difficulty inside `_static_reconstruction_decoys(...)`.
- Tune relation distance with `RELATION_CLOSE_DISTANCE_CELLS`.
- Add Godot probe assertions whenever a question family is changed.

## Common Risk Areas

- GDScript warnings are treated as errors by the project import, so avoid Variant-inferred `:=` locals from untyped helpers.
- Do not reintroduce hidden labels, answer tokens, or grid-cell names into user-facing prompts/card labels.
- Question screens must not display the studied scene or north marker.
- Large answer cards need enough vertical space in `SpatialPanel`; preview cards are taller than text-only choices.

# Rapid Tracking

Rapid Tracking now lives behind the stable `cfast_trainer.rapid_tracking` import path as a package with clearer seams:

- `simulation.py` owns deterministic runtime reset/reseed hooks on top of the legacy RT core.
- `scene.py` adds the exercise lifecycle used by the app shell: `enter()`, `exit()`, `reset()`, `handle_event()`, `render()`, `resize()`, and `snapshot()`.
- `renderer.py` owns the RT-specific HUD, fallback 2D view, GL scene queueing, and dev panel.
- `procgen.py` exposes the current compound-style generator behind `TrainingWorldBuilder` so richer terrain/road settlement generation can plug in later without disturbing the rest of the exercise.

## Lifecycle

- `build_rapid_tracking_test()` returns a `RapidTrackingEngine`, which is also a `RapidTrackingExercise`.
- The app binds shell resources with `bind_screen_context(...)`, then calls `enter()` on construction and `exit()` on close.
- Resets are in-process. `reset()` keeps the same seed unless one is passed, `reseed()` advances deterministically, `restart_practice()` and `restart_scored()` jump straight into those modes, and `return_to_instructions()` rebuilds the same seeded run at the intro phase.

## Dev Controls

When app dev tools are enabled:

- `F2`: toggle the debug overlay.
- `F3`: toggle camera/target diagnostics.
- `F5`: reset the current run with the same seed and current mode.
- `F6`: reseed and restart the current mode.
- `Shift+F5`: return to instructions with the same seed.
- On-screen dev buttons mirror reset, reseed, instructions, practice, scored, debug, and camera actions.

## Simulation vs Rendering

- The simulation remains deterministic and renderer-agnostic.
- The Modern GL path is still the canonical 3D scene renderer through the shared `RapidTrackingGlScene`.
- The package renderer owns only presentation: queueing the GL scene, drawing the fallback schematic view, and drawing RT HUD/debug/dev overlays.
- Rapid Tracking scenery now prefers file-backed OBJ assets through `assets/panda3d/manifest.json` while keeping the existing asset IDs and builtin fallbacks intact.
- Static RT roads, buildings, terrain, and distant backdrop features are cached twice: first as deterministic scene instances by `scene_seed`, then as pretransformed world-space triangle groups for coarse culling and cheaper far-scene rendering.
- Dynamic RT renderables stay separate from that cache: active targets, filtered ambient movers, the capture box, reticle, and debug overlays are still evaluated per frame.

## Future Procgen Hooks

`TrainingWorldBuilder` is the extension seam for richer training worlds. The current `RapidTrackingV1TrainingWorldBuilder` still uses the existing compound layout, but future builders can add terrain, movement-cost fields, POI scoring, road routing, local settlement layout, and validation/repair without changing the exercise lifecycle or shell integration.

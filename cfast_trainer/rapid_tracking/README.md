# Rapid Tracking

Rapid Tracking now lives behind the stable `cfast_trainer.rapid_tracking` import path as a package with clearer seams:

- `simulation.py` owns deterministic runtime reset/reseed hooks on top of the legacy RT core.
- `scene.py` adds the exercise lifecycle used by the app shell: `enter()`, `exit()`, `reset()`, `handle_event()`, `render()`, `resize()`, and `snapshot()`.
- `renderer.py` owns the flat pygame tracking field, reticle, capture box, HUD, and dev panel.
- `cfast_trainer/godot_bridge.py` can stream Rapid Tracking snapshots to the optional Godot companion window.

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
- The pygame presentation remains the authoritative control surface and fallback renderer.
- When Godot 4 is installed, the app launches the companion project for an FPS-style low-poly terrain/target view. It receives target, reticle, capture-box, camera, and scene payload state over localhost UDP JSON.
- Dynamic renderables are evaluated per frame from the engine payload: active target position, capture box, reticle, and debug data.
- The package has no Python-side depth renderer or ModernGL/OpenGL dependency.

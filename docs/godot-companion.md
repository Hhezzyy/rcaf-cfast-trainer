# Godot Companion Renderer

## Scope

The Godot project in `godot/cfast_3d/` is a companion renderer for five guide-backed 3D tasks:

- Auditory Capacity
- Rapid Tracking
- Spatial Integration
- Trace Test 1
- Trace Test 2

Instrument Comprehension aircraft-image flows are rendered in the main pygame window from a local 3D mesh asset; they do not stream to Godot.

The bridge supports two runtime models:

- Snapshot companion mode: Python owns timing, scoring, input, persistence, menus, and pause flow while Godot receives whitelisted visual snapshots over localhost UDP JSON from `cfast_trainer/godot_bridge.py`.
- Godot-owned mode: the active 3D runtime owns phase progress, live simulation, input, and authoritative result metrics. Python still owns menus, pause flow, persistence, and result materialization.

Auditory Capacity, Rapid Tracking, Spatial Integration, Trace Test 1, and Trace Test 2 use the Godot-owned runtime family when launched from the Tests menu. Their drills, workouts, benchmark probes, and adaptive blocks must build from that same runtime family rather than a stale fallback renderer.

If Godot is missing, fails to launch, or exits during a test, the existing pygame renderer remains active.
The companion is also suppressed automatically during pytest runs and when SDL uses the
`dummy` video driver, so automated tests cannot open Godot windows. Set
`CFAST_ENABLE_GODOT_IN_TESTS=1` only for an explicit local integration run.

## macOS Install And Run

Install or verify Godot 4.6.2:

```bash
./scripts/ensure_godot_macos.sh
```

Import the Godot project:

```bash
/Applications/Godot.app/Contents/MacOS/Godot --headless --path godot/cfast_3d --import
```

Run the Godot project manually:

```bash
/Applications/Godot.app/Contents/MacOS/Godot --path godot/cfast_3d --windowed --resolution 960x540 --max-fps 60
/Applications/Godot.app/Contents/MacOS/Godot --path godot/cfast_3d --fullscreen --max-fps 60
```

Normal app launch:

```bash
./run.command
```

`run.command` calls `scripts/ensure_godot_macos.sh`, then exports:

```bash
CFAST_GODOT_BIN=/Applications/Godot.app/Contents/MacOS/Godot
```

## Low-Spec Defaults

- Resolution scale: `0.67`
- Procedural atlas target: `512x512`
- Mesh style: low-poly primitives
- Active/ambient target count: 3-8
- Draw distance: short to medium
- Fog: on
- Lighting: ambient plus one directional light
- Shadows: off
- Reflections: off
- Anti-aliasing: off

## Spatial Integration Notes

- Spatial Integration is Godot-owned for the active 3D path.
- Study views use oblique/rotated scene cameras; question screens hide the studied scene.
- The north marker is scene geometry during study only, positioned off the north edge of the map, and hidden during questions.
- Fair-question rules are enforced in the Godot runtime: no hidden object IDs, no grid-cell answer prompts, and no typed grid-cell input for active Spatial Integration questions.
- The reconstruction question presents four clickable top-down map previews below the prompt. One is exact; the other three are generated with difficulty-scaled similarity.

## Input Conventions

- Auditory Capacity ball control uses aircraft-style pitch in Godot-owned mode: keyboard Up and stick-forward move the ball down, while keyboard Down and stick-back move it up. The source probe in `tests/test_godot_auditory_runtime_source.py` protects that sign convention.
- Rapid Tracking scans all connected joypads for motion and hold-zoom input, with a deadzone before the axis contribution is applied.
- Python wrapper drills that still use pygame input keep their own regression coverage so legacy drill shells do not silently drop joystick-derived control.

## Bridge Contract

Python launches Godot with:

```bash
/Applications/Godot.app/Contents/MacOS/Godot --path godot/cfast_3d --windowed --resolution 960x540 --max-fps 60 -- --listen-port <port> --session-id <id>
```

When the pygame app is in fullscreen or borderless mode, Python launches the companion with `--fullscreen` instead of the windowed resolution. Set `CFAST_GODOT_WINDOW_MODE=windowed`, `fullscreen`, or `maximized` to force a companion mode independent of the main app.

Python sends UDP JSON packets to `127.0.0.1:<port>` with:

- `schema`
- `kind`
- `title`
- `phase`
- `renderer_backend`
- `window_mode`
- `performance`
- `payload`

The current renderer backend recorded by Python is `godot_4` only when a Godot packet has been sent successfully for the active test. Otherwise results keep `pygame_2d`.

Godot-owned runtimes send control packets such as `godot_ready`, `godot_phase_advance`, `godot_progress`, `godot_complete`, and `godot_error` back to Python. The shell routes those packets by `run_key` first, then by `test_code` or runtime kind, so nested workout/adaptive engines advance instead of leaving the user on an instruction or practice screen.

## References

- Godot macOS downloads: https://godotengine.org/download/macos/
- Godot command line usage: https://docs.godotengine.org/en/4.6/tutorials/editor/command_line_tutorial.html

# Godot Companion Renderer

## Scope

The Godot project in `godot/cfast_3d/` is a companion renderer for five guide-backed 3D tasks:

- Auditory Capacity
- Rapid Tracking
- Spatial Integration
- Trace Test 1
- Trace Test 2

Instrument Comprehension aircraft-image flows are rendered in the main pygame window from a local 3D mesh asset; they do not stream to Godot.

Python remains the source of truth for timing, scoring, input, persistence, menus, and pause flow. Godot runs as a separate window and receives whitelisted visual snapshots over localhost UDP JSON from `cfast_trainer/godot_bridge.py`.

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

## References

- Godot macOS downloads: https://godotengine.org/download/macos/
- Godot command line usage: https://docs.godotengine.org/en/4.6/tutorials/editor/command_line_tutorial.html

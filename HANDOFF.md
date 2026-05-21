# Handoff (RCAF CFAST Trainer)

## Current status
- Tests: run `.venv/bin/python -m pytest -q` after large cleanup passes.
- Bare `python` is not on PATH on this laptop; use `.venv/bin/python` or `python3`.
- Current renderer direction: pygame app/control window plus optional Godot 4 companion window for Auditory Capacity, Rapid Tracking, Spatial Integration, Trace Test 1, and Trace Test 2.
- Python remains source of truth for timing, scoring, input, persistence, and menus. Godot receives UDP JSON visual snapshots only.
- Python-side OpenGL/ModernGL renderers and assets remain removed.
- Spatial Integration Godot questions are fairness-constrained: no hidden object IDs, no grid-cell prompts, option cards only, scene hidden during questions, and reconstruction answers show four top-down map previews.

## How to run
- Normal macOS launch:
  `./run.command`
- The shortcut is the expected day-to-day path. It runs the Godot ensure/import helper,
  exports `CFAST_GODOT_BIN=/Applications/Godot.app/Contents/MacOS/Godot` when present,
  then runs `.venv/bin/python -m cfast_trainer` from the repo root.
- If you want a Finder/Desktop shortcut, create an alias to `run.command`; do not copy
  it away from the repo because it expects `.venv` beside it.
- Godot install/check:
  `./scripts/ensure_godot_macos.sh`
- Godot import:
  `/Applications/Godot.app/Contents/MacOS/Godot --headless --path godot/cfast_3d --import`
- Manual Godot project run:
  `/Applications/Godot.app/Contents/MacOS/Godot --path godot/cfast_3d --resolution 960x540 --max-fps 60`
- Test:
  `.venv/bin/python -m pytest -q`

## Known UX issue
- Fullscreen can trap/freeze. If it happens:
  - macOS Force Quit: Option+Command+Esc
  - For safety run windowed by default (set `settings.fullscreen=False` or set_mode flags to 0).

## Next work areas
- Fine-tune each Godot test presenter now that the simple bridge is in place.
- Spatial Integration: add screenshot/video regression coverage for top-down map answer previews and aircraft motion-order questions.
- Airborne Numerical: expand question types (fuel endurance, parcel effects) and align overlays/text with candidate guide.
- UI robustness: add a reliable exit hotkey + windowed-safe default.
- Add persistence for settings and session summaries.

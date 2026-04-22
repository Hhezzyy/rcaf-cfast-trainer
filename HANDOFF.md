# Handoff (RCAF CFAST Trainer)

## Current status
- Tests: `.venv/bin/python -m pytest -q` is green on this Mac as of 2026-04-22.
- Bare `python` is not on PATH on this laptop; use `.venv/bin/python` or `python3`.
- Recent verified work: removed stale Panda3D references after the renderer cleanup, documented the current ModernGL/render-asset layout, and kept pygame fallback render paths working for non-OpenGL tests.
- The large pre-existing diff also moves former `assets/panda3d/` models to `assets/render/` and deletes old Panda3D runtime/tests. I only labelled the parts whose purpose was clear from the code and tests.

## How to run
- Activate venv:
  `source .venv/bin/activate`
- Run:
  `.venv/bin/python -m cfast_trainer`
- Test:
  `.venv/bin/python -m pytest -q`

## Known UX issue
- Fullscreen can trap/freeze. If it happens:
  - macOS Force Quit: Option+Command+Esc
  - For safety run windowed by default (set `settings.fullscreen=False` or set_mode flags to 0).

## Next work areas
- Airborne Numerical: expand question types (fuel endurance, parcel effects) and align overlays/text with candidate guide.
- UI robustness: add a reliable exit hotkey + windowed-safe default.
- Add persistence for settings and session summaries.

# Handoff (RCAF CFAST Trainer)

## Current status
- Tests: `python -m pytest -q` should be green.
- Recent work: fixed `app.py` airborne UI helpers (indentation + method placement) and made `TimedTextInputTest.submit_answer` accept non-string inputs by coercing to str.

## How to run
- Activate venv:
  `source .venv/bin/activate`
- Run:
  `python -m cfast_trainer`

## Known UX issue
- Fullscreen can trap/freeze. If it happens:
  - macOS Force Quit: Option+Command+Esc
  - For safety run windowed by default (set `settings.fullscreen=False` or set_mode flags to 0).

## Next work areas
- Airborne Numerical: expand question types (fuel endurance, parcel effects) and align overlays/text with candidate guide.
- UI robustness: add a reliable exit hotkey + windowed-safe default.
- Add persistence for settings and session summaries.

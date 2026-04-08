# Situational Awareness

## Purpose

- Multi-channel monitoring task that mixes pictorial, coded, numerical, and aural updates with timed queries.
- Tracks visible contacts, cue cards, top-strip summaries, announcements, and query-answer windows.

## Important Files

- `cfast_trainer/situational_awareness.py`: config, scenario generation, runtime state machine, payloads, and `build_situational_awareness_test(...)`.
- `cfast_trainer/sa_drills.py`: targeted training segments and pressure/mixed drill builders.
- `cfast_trainer/sa_workouts.py`: workout sequencing and menu entries.
- `cfast_trainer/app.py`: shared rendering, audio cue handling, and submission plumbing.
- `tests/test_situational_awareness_core.py`
- `tests/test_situational_awareness_ui.py`
- `tests/test_situational_awareness_headless_sim.py`
- `tests/test_sa_drills.py`
- `tests/test_sa_workouts.py`

## Lifecycle / Data Flow

1. `build_situational_awareness_test(...)` returns a `SituationalAwarenessTest`.
2. Training segments define active channels, query families, and pressure modifiers.
3. `update()` advances scenario clocks, contact sweeps, cue cards, announcements, and active queries.
4. `snapshot()` emits `SituationalAwarenessPayload`, which the shell renders and uses for answer input.
5. On results, the shared shell handles persistence through the generic results path.

## Input / Rendering Dependencies

- Rendering is shell-owned and driven by `SituationalAwarenessPayload`.
- Input can be grid-cell text, mouse clicks, or numbered choice answers depending on `answer_mode`.
- Aural content depends on the shell path remaining aligned with payload announcements and active channels.

## Persistence / Test Hooks

- Persistence enters through shared app-shell result recording rather than subsystem-local DB code.
- Core determinism and query behavior are tested in `tests/test_situational_awareness_core.py`.
- UI-specific answer-mode rendering lives in `tests/test_situational_awareness_ui.py`.
- Timed scripted flows are covered in `tests/test_situational_awareness_headless_sim.py`.

## Common Safe Edit Points

- Training segments and profile tuning in `cfast_trainer/sa_drills.py`.
- Workout structure in `cfast_trainer/sa_workouts.py`.
- Scenario-family or query helpers in `cfast_trainer/situational_awareness.py` when the payload contract stays stable.

## Common Risk Areas

- Query timing, scenario timing, and channel timing are tightly coupled.
- Payload-field changes can break both rendering and audio cue synchronization.
- This subsystem is screenshot-heavy; visual regressions are easiest to validate with full-window captures.

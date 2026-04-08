# Visual Search

## Purpose

- Deterministic board-search task that asks the user to find the numbered cell matching a target tile.
- Scales from simpler 3x4 boards to dense late-level symbol overload boards.

## Important Files

- `cfast_trainer/visual_search.py`: config, profiles, payload/scoring, generator, and `build_visual_search_test(...)`.
- `cfast_trainer/vs_drills.py`: preview, clean-scan, mixed-tempo, pressure, and higher-order search drill builders.
- `cfast_trainer/vs_workouts.py`: workout plan structure and menu entries.
- `cfast_trainer/app.py`: shell rendering for live boards and composite tokens.
- `tests/test_visual_search_core.py`
- `tests/test_visual_search_ui.py`
- `tests/test_visual_search_headless_sim.py`
- `tests/test_vs_drills.py`
- `tests/test_vs_workouts.py`

## Lifecycle / Data Flow

1. `build_visual_search_test(...)` returns a generic `TimedTextInputTest`.
2. `VisualSearchGenerator.next_problem(...)` chooses a task kind, builds a board, and assigns visible numeric cell codes.
3. The shell renders the payload and collects the two-digit answer.
4. Results flow through the generic scoring/persistence path.

## Input / Rendering Dependencies

- Input is keyboard-first through the shared typed-answer shell flow.
- Rendering lives in `cfast_trainer/app.py`; dense symbol boards rely on the shell correctly parsing composite token strings.
- No subsystem-local renderer module exists.

## Persistence / Test Hooks

- Persistence is handled by the shared results path.
- Core board-shape and level behavior are covered in `tests/test_visual_search_core.py`.
- Rendering of composite tokens and larger boards is covered in `tests/test_visual_search_ui.py`.
- Practice/scored summary flow is covered in `tests/test_visual_search_headless_sim.py`.

## Common Safe Edit Points

- `VisualSearchProfile` and token-bank helpers in `cfast_trainer/visual_search.py`.
- Drill composition in `cfast_trainer/vs_drills.py`.
- Workout structure in `cfast_trainer/vs_workouts.py`.

## Common Risk Areas

- Level 8-10 behavior is intentionally specific and already pinned by tests.
- `VisualSearchPayload` changes can silently break shell rendering or answer interpretation.
- Small generator changes can affect determinism and late-level overload assumptions.

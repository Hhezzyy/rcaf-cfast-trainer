# Shared App Shell

## Purpose

- The shared pygame shell owns app startup, menu routing, loading screens, pause flows, settings screens, renderer bootstrap/failure handling, activity-session wiring, diagnostic persistence, and result persistence.
- This is the highest-leverage navigation document for UI, input, and cross-subsystem bugs.

## Important Files

- `cfast_trainer/app.py`: `App`, `LoadingScreen`, `MenuScreen`, `CognitiveTestScreen`, workout screens, benchmark/adaptive screens, input/settings stores, and `run(...)`.
- `cfast_trainer/__main__.py`: CLI entrypoint and `--headless-sim` wiring.
- `cfast_trainer/runtime_defaults.py`: stored window mode, OpenGL preference, and auditory runtime defaults.
- `cfast_trainer/runtime_ui_policy.py`: small runtime UI feature flags.
- `cfast_trainer/persistence.py`: `ResultsStore` and history/session materialization.
- `cfast_trainer/results.py`: engine-to-result conversion.
- `tests/test_app_shell_hardening.py`
- `tests/test_cognitive_test_screen_pause_menu.py`
- `tests/test_loading_screen.py`
- `tests/test_menu_screen_mouse.py`
- `tests/test_smoke.py`
- `tests/test_smoke_ui_tests_menu.py`
- `tests/test_smoke_ui_individual_drills_menu.py`
- `tests/test_smoke_ui_workouts_menu.py`
- `tests/test_intro_briefings.py`

## Lifecycle / Data Flow

1. `cfast_trainer/__main__.py` calls `run(...)` or `run_headless_sim(...)`.
2. `run(...)` bootstraps pygame, window mode, renderer selection, settings stores, and the root menu tree.
3. `MenuScreen` routes into `LoadingScreen`, then into `CognitiveTestScreen`, workout screens, benchmark screens, or adaptive screens.
4. Those screens call engine methods like `start_practice()`, `start_scored()`, `update()`, `snapshot()`, and `submit_answer(...)`.
5. On completion, the shell converts runtime state into `AttemptResult` records and persists them.

## Input / Rendering Dependencies

- Keyboard, mouse, joystick calibration, input profiles, and button bindings are all configured here.
- Renderer bootstrap, explicit failure handling, diagnostic persistence, and Panda3D handoff start here.
- Many subsystem payloads are rendered directly by `CognitiveTestScreen`, so shell changes can affect many tests at once.

## Persistence / Test Hooks

- `ResultsStore` is the shared persistence entry point.
- Headless shell scenarios are available through `python -m cfast_trainer --headless-sim <scenario>`.
- Shell safety nets live in the pause, smoke, intro, and hardening tests listed above.

## Common Safe Edit Points

- Menu labels, screen wiring, and non-functional copy in `cfast_trainer/app.py`.
- Runtime defaults in `cfast_trainer/runtime_defaults.py`.
- Documentation-only updates in `docs/` and `.github/`.

## Common Risk Areas

- `cfast_trainer/app.py` is large and cross-cutting; avoid unrelated cleanup while fixing one bug.
- Pause flow, loading flow, and persistence flow each touch many subsystem types.
- Renderer selection, failure paths, and input setup behave differently in normal and headless modes, and renderer failures are expected to be loud plus diagnostic-rich.

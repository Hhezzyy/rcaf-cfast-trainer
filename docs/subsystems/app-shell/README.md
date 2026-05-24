# Shared App Shell

## Purpose

- The shared pygame shell owns app startup, menu routing, loading screens, pause flows, settings screens, display bootstrap, activity-session wiring, and result persistence.
- This is the highest-leverage navigation document for UI, input, and cross-subsystem bugs.

## Important Files

- `run.command`: standard macOS shortcut for day-to-day app launches; it checks/imports the optional Godot companion and then execs `.venv/bin/python -m cfast_trainer` from the repo root.
- `cfast_trainer/app.py`: `App`, `LoadingScreen`, `MenuScreen`, `CognitiveTestScreen`, workout screens, benchmark/adaptive screens, input/settings stores, and `run(...)`.
- `cfast_trainer/godot_bridge.py`: optional Godot 4 companion launcher and UDP JSON state bridge for the 3D-presented tests.
- `godot/cfast_3d/`: lightweight Godot companion project.
- `scripts/ensure_godot_macos.sh`: installs/checks `/Applications/Godot.app` and imports the companion project on macOS.
- `cfast_trainer/__main__.py`: CLI entrypoint and `--headless-sim` wiring.
- `cfast_trainer/runtime_defaults.py`: stored window mode and auditory runtime defaults.
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

1. On macOS, `run.command` is the normal user launch path. It checks Godot first, exports `CFAST_GODOT_BIN` when available, and invokes `python -m cfast_trainer` through the local `.venv`.
2. `cfast_trainer/__main__.py` calls `run(...)` or `run_headless_sim(...)`.
3. `run(...)` bootstraps pygame, window mode, settings stores, and the root menu tree.
4. `MenuScreen` routes into `LoadingScreen`, then into `CognitiveTestScreen`, workout screens, benchmark screens, or adaptive screens.
5. Those screens call engine methods like `start_practice()`, `start_scored()`, `update()`, `snapshot()`, and `submit_answer(...)`.
6. On completion, the shell converts runtime state into `AttemptResult` records and persists them.

## Input / Rendering Dependencies

- Keyboard, mouse, joystick calibration, input profiles, and button bindings are all configured here.
- Pygame remains the control window and fallback renderer. Auditory Capacity, Rapid Tracking, Spatial Integration, Trace Test 1, and Trace Test 2 can run as Godot-owned 3D activities in the optional Godot 4 companion window. Instrument Comprehension aircraft-image flows render local 3D mesh assets inside pygame.
- Godot-owned activities report ready/progress/phase/complete/error messages back to Python. `App._route_godot_control_command(...)` forwards those packets by `run_key`, `test_code`, and runtime kind so nested workout, benchmark, and adaptive block engines advance correctly.
- Python remains responsible for menus, pause flow, persistence, and final result storage, even when Godot owns the live 3D simulation.
- Many subsystem payloads are rendered directly by `CognitiveTestScreen`, so shell changes can affect many tests at once.

## Persistence / Test Hooks

- `ResultsStore` is the shared persistence entry point.
- Headless shell scenarios are available through `python -m cfast_trainer --headless-sim <scenario>`.
- `tests/test_godot_bridge.py` covers companion payload serialization, fake-process bridge lifecycle, nested Godot-owned message routing, UI render sync, and a skipped import smoke test when Godot is unavailable.
- `tests/test_drill_runtime_audit.py` checks menu drills, canonical drill builders, and workout blocks against the expected runtime families.
- Shell safety nets live in the pause, smoke, intro, and hardening tests listed above.

## Common Safe Edit Points

- Menu labels, screen wiring, and non-functional copy in `cfast_trainer/app.py`.
- Runtime defaults in `cfast_trainer/runtime_defaults.py`.
- Documentation-only updates in `docs/` and `.github/`.

## Common Risk Areas

- `cfast_trainer/app.py` is large and cross-cutting; avoid unrelated cleanup while fixing one bug.
- Pause flow, loading flow, and persistence flow each touch many subsystem types.
- Display bootstrap and input setup behave differently in normal and headless modes.

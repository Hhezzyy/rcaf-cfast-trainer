# RCAF CFAST Trainer

Offline-first training app for CFASC CFAST-style aircrew selection aptitude domains.

## Repo Map

### App Entrypoints

- `python -m cfast_trainer` starts the app through `cfast_trainer/__main__.py`.
- `cfast_trainer/app.py` is the main pygame shell, screen router, loader flow, pause flow, settings UI, and persistence glue.
- `./run.command` is the macOS convenience launcher for a local `.venv`.

### Core Architecture

1. `cfast_trainer/__main__.py` parses CLI args and forwards to `run()` or `run_headless_sim()` in `cfast_trainer/app.py`.
2. `cfast_trainer/app.py` builds the `App`, bootstraps rendering/input stores, and routes between `MenuScreen`, `LoadingScreen`, `CognitiveTestScreen`, workout screens, benchmark screens, and adaptive-session screens.
3. Each whole-test subsystem usually lives in `cfast_trainer/<subsystem>.py` and exposes `build_*_test(...)`.
4. Drill builders usually live in `cfast_trainer/<prefix>_drills.py`; workout plans usually live in `cfast_trainer/<prefix>_workouts.py`.
5. Results are converted to `AttemptResult` data in `cfast_trainer/results.py` and persisted through `cfast_trainer/persistence.py`.
6. Difficulty/adaptive orchestration lives in `cfast_trainer/adaptive_difficulty.py`, `cfast_trainer/adaptive_scheduler.py`, `cfast_trainer/training_modes.py`, and `cfast_trainer/canonical_drill_registry.py`.

### Where Tests Live

- All tests live in `tests/`.
- Common patterns:
  - `tests/test_<subsystem>_core.py`: deterministic engine or generator rules.
  - `tests/test_<subsystem>_ui.py`: pygame shell rendering/input expectations.
  - `tests/test_<subsystem>_headless_sim.py`: scripted no-window flows.
  - `tests/test_<prefix>_drills.py` and `tests/test_<prefix>_workouts.py`: drill/workout plan coverage.
- Shell-wide tests are concentrated in files like `tests/test_app_shell_hardening.py`, `tests/test_cognitive_test_screen_pause_menu.py`, `tests/test_loading_screen.py`, and the smoke menu tests.

### Rendering, Input, Persistence, Settings

- Renderer bootstrap, failure handling, and diagnostics:
  - `cfast_trainer/app.py`
  - `cfast_trainer/modern_gl_renderer.py`
  - `cfast_trainer/gl_scenes.py`
  - `cfast_trainer/render_assets.py`
- Panda3D bridge and optional runtime:
  - `cfast_trainer/panda3d_launcher.py`
  - `cfast_trainer/panda3d_runtime.py`
  - `cfast_trainer/panda3d_assets.py`
  - `assets/panda3d/README.md`
- HOTAS/input calibration, profiles, and joystick bindings:
  - `cfast_trainer/app.py` (`InputProfilesStore`, `AxisCalibrationScreen`, `JoystickBindingsScreen`)
- Persistence and history:
  - `cfast_trainer/persistence.py`
  - `cfast_trainer/results.py`
- Runtime defaults and UI policy:
  - `cfast_trainer/runtime_defaults.py`
  - `cfast_trainer/runtime_ui_policy.py`

### Safe First Files By Bug Category

| Bug category | Safest files to inspect first | Why |
| --- | --- | --- |
| Menu, pause, intro, loading, screen routing | `cfast_trainer/app.py`, `tests/test_app_shell_hardening.py`, `tests/test_cognitive_test_screen_pause_menu.py`, `tests/test_loading_screen.py` | The shared shell and pause logic are centralized. |
| Wrong instructions or test briefings | `cfast_trainer/app.py`, `tests/test_intro_briefings.py` | Intro overlays and guide briefing text are shell-owned. |
| Whole-test scoring or timing | `cfast_trainer/<subsystem>.py`, `tests/test_<subsystem>_core.py`, `tests/test_<subsystem>_headless_sim.py` | Most scoring/timer logic lives in the subsystem engine, not the shell. |
| Drill selection or workout sequencing | `cfast_trainer/<prefix>_drills.py`, `cfast_trainer/<prefix>_workouts.py`, matching `tests/test_<prefix>_drills.py`, `tests/test_<prefix>_workouts.py` | Drill/workout composition is isolated from the shell. |
| Adaptive plan selection | `cfast_trainer/adaptive_scheduler.py`, `cfast_trainer/canonical_drill_registry.py`, `tests/test_adaptive_scheduler.py` | Ranking and drill selection are concentrated here. |
| Benchmark battery order or summaries | `cfast_trainer/benchmark.py`, `tests/test_benchmark.py` | Probe order and session summary logic live together. |
| Persistence/history regressions | `cfast_trainer/persistence.py`, `tests/test_persistence.py`, `tests/test_cognitive_test_screen_persistence.py` | DB schema, writes, and shell persistence hooks are covered there. |
| OpenGL/Panda3D startup, bootstrap, or diagnostics issues | `cfast_trainer/app.py`, `cfast_trainer/modern_gl_renderer.py`, `cfast_trainer/panda3d_runtime.py`, `tests/test_3d_renderer_selection.py`, `tests/test_gl_bootstrap.py` | Bootstrap, failure handling, diagnostics, and renderer selection are shared infra. |
| HOTAS/input binding problems | `cfast_trainer/app.py`, `tests/test_app_window_mode.py`, `tests/test_cognitive_test_screen_pause_menu.py`, `tests/test_sensory_motor_apparatus_ui.py` | Input stores and shell mapping screens live in `app.py`. |

### Major Test And Drill Subsystems

| Subsystem | Whole-test code | Drill/workout code | Primary tests |
| --- | --- | --- | --- |
| Numerical Operations | `cfast_trainer/numerical_operations.py` | `cfast_trainer/no_drills.py`, `cfast_trainer/no_workouts.py` | `tests/test_numerical_operations_core.py`, `tests/test_numerical_operations_headless_sim.py`, `tests/test_no_drills.py`, `tests/test_no_workouts.py` |
| Math Reasoning | `cfast_trainer/math_reasoning.py` | `cfast_trainer/mr_drills.py`, `cfast_trainer/mr_workouts.py` | `tests/test_math_reasoning_core.py`, `tests/test_math_reasoning_headless_sim.py`, `tests/test_mr_drills.py`, `tests/test_mr_workouts.py` |
| Airborne Numerical | `cfast_trainer/airborne_numerical.py` | `cfast_trainer/ant_drills.py`, `cfast_trainer/ant_workouts.py` | `tests/test_airborne_numerical.py`, `tests/test_ant_drills.py`, `tests/test_ant_workouts.py` |
| Angles / Bearings / Degrees | `cfast_trainer/angles_bearings_degrees.py` | `cfast_trainer/abd_drills.py`, `cfast_trainer/abd_workouts.py` | `tests/test_angles_bearings_degrees_core.py`, `tests/test_angles_bearings_degrees_headless_sim.py`, `tests/test_abd_drills.py`, `tests/test_abd_workouts.py` |
| Auditory Capacity | `cfast_trainer/auditory_capacity.py`, `cfast_trainer/auditory_capacity_view.py` | `cfast_trainer/ac_drills.py`, `cfast_trainer/ac_workouts.py` | `tests/test_auditory_capacity.py`, `tests/test_auditory_capacity_core.py`, `tests/test_auditory_capacity_headless_sim.py`, `tests/test_auditory_capacity_panda3d.py`, `tests/test_ac_drills.py`, `tests/test_ac_workouts.py` |
| Colours Letters Numbers | `cfast_trainer/colours_letters_numbers.py` | `cfast_trainer/cln_drills.py`, `cfast_trainer/cln_workouts.py` | `tests/test_colours_letters_numbers_core.py`, `tests/test_colours_letters_numbers_headless_sim.py`, `tests/test_cln_drills.py`, `tests/test_cln_workouts.py` |
| Cognitive Updating | `cfast_trainer/cognitive_updating.py` | `cfast_trainer/cu_drills.py`, `cfast_trainer/cu_workouts.py` | `tests/test_cognitive_updating_core.py`, `tests/test_cognitive_updating_headless_sim.py`, `tests/test_cognitive_updating_ui.py`, `tests/test_cu_drills.py`, `tests/test_cu_workouts.py` |
| Digit Recognition | `cfast_trainer/digit_recognition.py` | `cfast_trainer/dr_drills.py`, `cfast_trainer/dr_workouts.py` | `tests/test_digit_recognition_core.py`, `tests/test_digit_recognition_headless_sim.py`, `tests/test_dr_drills.py`, `tests/test_dr_workouts.py` |
| Instrument Comprehension | `cfast_trainer/instrument_comprehension.py`, `cfast_trainer/instrument_aircraft_cards.py`, `cfast_trainer/instrument_orientation_solver.py` | `cfast_trainer/ic_drills.py`, `cfast_trainer/ic_workouts.py` | `tests/test_instrument_comprehension_core.py`, `tests/test_instrument_comprehension_headless_sim.py`, `tests/test_instrument_comprehension_ui.py`, `tests/test_instrument_aircraft_cards.py`, `tests/test_instrument_orientation_solver.py`, `tests/test_ic_drills.py`, `tests/test_ic_workouts.py` |
| Rapid Tracking | `cfast_trainer/rapid_tracking/`, `cfast_trainer/rapid_tracking_view.py`, `cfast_trainer/rapid_tracking_gl.py`, `cfast_trainer/rapid_tracking_panda3d.py` | `cfast_trainer/rt_drills.py`, `cfast_trainer/rt_workouts.py` | `tests/test_rapid_tracking_core.py`, `tests/test_rapid_tracking_ui.py`, `tests/test_rapid_tracking_scene.py`, `tests/test_rapid_tracking_headless_sim.py`, `tests/test_rapid_tracking_gl.py`, `tests/test_rapid_tracking_panda3d.py`, `tests/test_rt_drills.py`, `tests/test_rt_workouts.py` |
| Sensory Motor Apparatus | `cfast_trainer/sensory_motor_apparatus.py` | `cfast_trainer/sma_drills.py`, `cfast_trainer/sma_workouts.py` | `tests/test_sensory_motor_apparatus_core.py`, `tests/test_sensory_motor_apparatus_headless_sim.py`, `tests/test_sensory_motor_apparatus_ui.py`, `tests/test_sma_drills.py`, `tests/test_sma_workouts.py` |
| Situational Awareness | `cfast_trainer/situational_awareness.py` | `cfast_trainer/sa_drills.py`, `cfast_trainer/sa_workouts.py` | `tests/test_situational_awareness_core.py`, `tests/test_situational_awareness_ui.py`, `tests/test_situational_awareness_headless_sim.py`, `tests/test_sa_drills.py`, `tests/test_sa_workouts.py` |
| Spatial Integration | `cfast_trainer/spatial_integration.py`, `cfast_trainer/spatial_integration_gl.py`, `cfast_trainer/spatial_integration_panda3d.py` | `cfast_trainer/si_drills.py`, `cfast_trainer/si_workouts.py` | `tests/test_spatial_integration_core.py`, `tests/test_spatial_integration_ui.py`, `tests/test_spatial_integration_headless_sim.py`, `tests/test_spatial_integration_gl.py`, `tests/test_spatial_integration_panda3d.py`, `tests/test_spatial_integration_drills.py`, `tests/test_spatial_integration_workouts.py` |
| System Logic | `cfast_trainer/system_logic.py` | `cfast_trainer/sl_drills.py`, `cfast_trainer/sl_workouts.py` | `tests/test_system_logic_core.py`, `tests/test_system_logic_headless_sim.py`, `tests/test_system_logic_ui.py`, `tests/test_sl_drills.py`, `tests/test_sl_workouts.py` |
| Table Reading | `cfast_trainer/table_reading.py`, `cfast_trainer/table_reading_cards/` | `cfast_trainer/tbl_drills.py`, `cfast_trainer/tbl_workouts.py` | `tests/test_table_reading_core.py`, `tests/test_table_reading_ui.py`, `tests/test_table_reading_headless_sim.py`, `tests/test_tbl_drills.py`, `tests/test_tbl_workouts.py` |
| Target Recognition | `cfast_trainer/target_recognition.py` | `cfast_trainer/tr_drills.py`, `cfast_trainer/tr_workouts.py` | `tests/test_target_recognition_core.py`, `tests/test_target_recognition_ui.py`, `tests/test_target_recognition_headless_sim.py`, `tests/test_tr_drills.py`, `tests/test_tr_workouts.py` |
| Trace Test 1 | `cfast_trainer/trace_test_1.py`, `cfast_trainer/trace_test_1_gl.py`, `cfast_trainer/trace_test_1_panda3d.py` | `cfast_trainer/trace_drills.py`, `cfast_trainer/trace_workouts.py` | `tests/test_trace_test_1_core.py`, `tests/test_trace_test_1_headless_sim.py`, `tests/test_trace_test_1_gl.py`, `tests/test_trace_test_1_panda3d.py`, `tests/test_trace_drills.py`, `tests/test_trace_workouts.py` |
| Trace Test 2 | `cfast_trainer/trace_test_2.py`, `cfast_trainer/trace_test_2_gl.py`, `cfast_trainer/trace_test_2_panda3d.py` | `cfast_trainer/trace_drills.py`, `cfast_trainer/trace_workouts.py` | `tests/test_trace_test_2_core.py`, `tests/test_trace_test_2_headless_sim.py`, `tests/test_trace_test_2_gl.py`, `tests/test_trace_test_2_panda3d.py`, `tests/test_trace_drills.py`, `tests/test_trace_workouts.py` |
| Vigilance | `cfast_trainer/vigilance.py` | `cfast_trainer/vig_drills.py`, `cfast_trainer/vig_workouts.py` | `tests/test_vigilance_core.py`, `tests/test_vigilance_headless_sim.py`, `tests/test_vigilance_ui.py`, `tests/test_vigilance_smoke_ui.py`, `tests/test_vig_drills.py`, `tests/test_vig_workouts.py` |
| Visual Search | `cfast_trainer/visual_search.py` | `cfast_trainer/vs_drills.py`, `cfast_trainer/vs_workouts.py` | `tests/test_visual_search_core.py`, `tests/test_visual_search_ui.py`, `tests/test_visual_search_headless_sim.py`, `tests/test_vs_drills.py`, `tests/test_vs_workouts.py` |

### Local Docs Worth Reading First

- Shared app shell map: `docs/subsystems/app-shell/README.md`
- Target Recognition map: `docs/subsystems/target-recognition/README.md`
- Digit Recognition map: `docs/subsystems/digit-recognition/README.md`
- Situational Awareness map: `docs/subsystems/situational-awareness/README.md`
- Visual Search map: `docs/subsystems/visual-search/README.md`
- Benchmark map: `docs/subsystems/benchmark/README.md`
- Adaptive scheduler map: `docs/subsystems/adaptive-scheduler/README.md`
- Rapid Tracking local map: `cfast_trainer/rapid_tracking/README.md`
- Panda3D asset map: `assets/panda3d/README.md`
- Contributor workflow and branch selection: `docs/codex-playbook.md`
- Test coverage overview: `docs/test-matrix.md`
- Screenshot convention: `docs/screenshots/README.md`

GitHub may only visibly show `main` from the remote view, but branch selection in this repo should follow the most recently updated local branch. `docs/codex-playbook.md` is the source of truth for that workflow.

## Requirements

- CPython 3.11+ (3.13 tested)
- VS Code with the Python and Pylance extensions
- Git (recommended for moving between machines)

## Setup (Windows 11)

1. Install CPython 3.13.12 (x64) from python.org.
2. Create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install dependencies:

   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   python -m pip install -r requirements-dev.txt
   ```

4. In VS Code, select the `.venv` interpreter.

## Setup (macOS Monterey, Intel)

1. Install CPython 3.13.12 from python.org.
2. Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   python -m pip install -r requirements-dev.txt
   ```

4. In VS Code, select the `.venv` interpreter.

Optional: install `requirements-3d.txt` when working on the Panda3D or richer 3D paths.

## VS Code

- Recommended extensions: `.vscode/extensions.json`
- Python test settings: `.vscode/settings.json`

## Verify

- `python --version` should report 3.11+.
- `python -m pytest` runs the test suite.
- `python -m cfast_trainer` opens the app (main menu).
- `python -m cfast_trainer --headless-sim boot` exercises the shell without opening a real window.

## Share / Move Between PC And Mac

- Do not copy `.venv/` between machines.
- Preferred: push to a git remote and clone on the other machine.
- Alternative zip (tracked files only):

  ```bash
  git archive --format zip -o rcaf-cfast-trainer.zip HEAD
  ```

- For ChatGPT projects/chats: upload the zip produced by `git archive`.

After moving, recreate the virtual environment and reinstall dependencies on the target machine.

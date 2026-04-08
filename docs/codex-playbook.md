# Codex Playbook

## Branch Selection

- Start by identifying the most recently updated local branch, not by assuming `main`.
- Command:

  ```bash
  git for-each-ref --sort=-committerdate --format='%(refname:short)|%(committerdate:iso8601)|%(objectname:short)' refs/heads
  ```

- Confirm the working tree before changing anything:

  ```bash
  git status --short --branch
  ```

- Stay on the most recently updated local branch unless the task explicitly requires a different one.

## Run Commands

- Main app:

  ```bash
  python -m cfast_trainer
  ```

- macOS convenience launcher:

  ```bash
  ./run.command
  ```

- CLI entrypoint without module mode:

  ```bash
  python cfast_trainer/__main__.py
  ```

- Headless shell scenarios:

  ```bash
  python -m cfast_trainer --headless-sim boot
  python -m cfast_trainer --headless-sim tests_menu
  python -m cfast_trainer --headless-sim benchmark_intro
  ```

## Test Commands

- Full suite:

  ```bash
  python -m pytest -q
  ```

- Target a subsystem:

  ```bash
  python -m pytest -q tests/test_target_recognition_core.py
  python -m pytest -q tests/test_visual_search_ui.py
  python -m pytest -q tests/test_adaptive_scheduler.py
  ```

- Shell/pause/loading sanity checks:

  ```bash
  python -m pytest -q tests/test_app_shell_hardening.py tests/test_cognitive_test_screen_pause_menu.py tests/test_loading_screen.py
  ```

## Lint / Type Check

- Ruff is configured in `pyproject.toml`:

  ```bash
  python -m ruff check .
  ```

- No dedicated repo type-check command is currently configured.

## Headless Environment

- Most pygame UI tests run with dummy SDL drivers.
- Useful env vars:

  ```bash
  export PYGAME_HIDE_SUPPORT_PROMPT=1
  export SDL_VIDEODRIVER=dummy
  export SDL_AUDIODRIVER=dummy
  export CFAST_DISABLE_TTS=1
  ```

- `run(headless=True)` and `--headless-sim` set these automatically.

## Where Screenshots Help Most

- Main menu, tests menu, individual drills menu, and workout menus.
- Shared pause menu, settings overlays, and loading screens.
- Target Recognition multi-panel layouts.
- Visual Search dense late-level boards.
- Situational Awareness grid plus cue card plus active query.
- Benchmark/adaptive intro, block, and results transitions.
- Renderer fallback/failure screens and any Panda3D/OpenGL mismatch.
- HOTAS calibration, input profile, and joystick binding screens.

## Scoped Change Rules

- Work from the most recently updated local branch.
- Read only the files needed for the current phase.
- Do not refactor unrelated files.
- Prefer extending existing tests over creating new test structure.
- Preserve macOS behavior first, but keep Windows compatibility in code and tests where practical.
- Do not make gameplay or scoring changes unless the task explicitly calls for them.
- Treat screenshots as authoritative for UI symptoms and visual regressions.

## Suggested Workflow

### UI Bugs

1. Start with the screenshot and identify the exact screen.
2. Read `cfast_trainer/app.py` plus the matching `tests/test_*_ui.py` file first.
3. Only then read the subsystem payload producer if the bug looks data-driven.
4. Extend the closest existing UI or shell test.

### Rendering Bugs

1. Decide whether the issue is fallback 2D, Modern GL, or Panda3D.
2. Inspect `cfast_trainer/app.py` bootstrap code first, then the renderer-specific module.
3. Compare against `tests/test_3d_renderer_selection.py`, `tests/test_gl_bootstrap.py`, and any subsystem renderer tests.
4. Use full-window screenshots and note platform, renderer path, and window mode.

### Scoring Bugs

1. Start in the whole-test module, not the shell.
2. Read the matching `tests/test_<subsystem>_core.py` and `tests/test_<subsystem>_headless_sim.py` first.
3. Check `cfast_trainer/results.py` and `cfast_trainer/persistence.py` only after confirming the engine summary is wrong or right.
4. Avoid shell refactors unless the bug is clearly in result capture or UI submission flow.

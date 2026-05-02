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
- GitHub may only visibly expose `main` from the remote view; local refs are the source of truth for choosing the starting branch.

## Run Commands

- Default macOS/operator launch:

  ```bash
  ./run.command
  ```

- The user normally intends to run the app through `run.command`. Treat that shortcut
  as the default launch path unless the task explicitly involves Windows, CI, or a
  manual/debug terminal run.
- On macOS, `run.command` also checks/imports the optional Godot companion project before
  launching Python. If Godot is unavailable, the app continues with pygame fallback.

- Godot install/check:

  ```bash
  ./scripts/ensure_godot_macos.sh
  ```

- Godot import:

  ```bash
  /Applications/Godot.app/Contents/MacOS/Godot --headless --path godot/cfast_3d --import
  ```

- Manual Godot companion run:

  ```bash
  /Applications/Godot.app/Contents/MacOS/Godot --path godot/cfast_3d --windowed --resolution 960x540 --max-fps 60
  ```

- Manual equivalent when the virtual environment is already active:

  ```bash
  python -m cfast_trainer
  ```

- Manual equivalent without activating the virtual environment:

  ```bash
  .venv/bin/python -m cfast_trainer
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
- HOTAS calibration, input profile, and joystick binding screens.

## Scoped Change Rules

- Work from the most recently updated local branch.
- Read only the files needed for the current phase.
- Do not refactor unrelated files.
- Prefer extending existing tests over creating new test structure.
- Preserve macOS behavior first, but keep Windows compatibility in code and tests where practical.
- Do not make gameplay or scoring changes unless the task explicitly calls for them.
- Treat screenshots as authoritative for UI symptoms and visual regressions.
- If you materially move file ownership, subsystem ownership, or primary test coverage, update `README.md`, the relevant subsystem README, and `docs/test-matrix.md` in the same PR.

## Suggested Workflow

### UI Bugs

1. Start with the screenshot and identify the exact screen.
2. Read `cfast_trainer/app.py` plus the matching `tests/test_*_ui.py` file first.
3. Only then read the subsystem payload producer if the bug looks data-driven.
4. Extend the closest existing UI or shell test.

### Rendering Bugs

1. Start with the screen-specific pygame render path in `cfast_trainer/app.py` or the matching subsystem renderer.
2. Compare against the closest `tests/test_*_ui.py` file and the app-shell/window-mode tests.
3. Use full-window screenshots and note platform, window mode, and input hardware.

### Scoring Bugs

1. Start in the whole-test module, not the shell.
2. Read the matching `tests/test_<subsystem>_core.py` and `tests/test_<subsystem>_headless_sim.py` first.
3. Check `cfast_trainer/results.py` and `cfast_trainer/persistence.py` only after confirming the engine summary is wrong or right.
4. Avoid shell refactors unless the bug is clearly in result capture or UI submission flow.

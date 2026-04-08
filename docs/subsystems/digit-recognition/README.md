# Digit Recognition

## Purpose

- Short-term visual memory task with staged show, mask, and question phases.
- Supports recall, count-target, and different-digit question kinds.

## Important Files

- `cfast_trainer/digit_recognition.py`: question generation, runtime state machine, payloads, events, and `build_digit_recognition_test(...)`.
- `cfast_trainer/dr_drills.py`: drill families and timed drill wrappers.
- `cfast_trainer/dr_workouts.py`: workout sequencing and menu entries.
- `cfast_trainer/app.py`: shared `CognitiveTestScreen` rendering/input path for digit payloads.
- `tests/test_digit_recognition_core.py`
- `tests/test_digit_recognition_headless_sim.py`
- `tests/test_dr_drills.py`
- `tests/test_dr_workouts.py`

## Lifecycle / Data Flow

1. `build_digit_recognition_test(...)` constructs a `DigitRecognitionTest`.
2. The engine advances through `_Stage.SHOW`, `_Stage.MASK`, and `_Stage.QUESTION`.
3. `snapshot()` exposes `DigitRecognitionPayload` for the shell to render.
4. `submit_answer(...)` records phase events and updates scored summaries used by shared persistence.

## Input / Rendering Dependencies

- Rendering stays inside the shared pygame shell in `cfast_trainer/app.py`.
- Input is keyboard-driven through the generic answer flow in `CognitiveTestScreen`.
- The shell expects payload timing and display-line behavior to remain consistent across stages.

## Persistence / Test Hooks

- Final persistence is shared-shell behavior through `cfast_trainer/persistence.py`.
- Determinism, question-bank coverage, and timer boundaries are tested in `tests/test_digit_recognition_core.py`.
- Practice/scored scripted behavior and special payload layouts are covered in `tests/test_digit_recognition_headless_sim.py`.

## Common Safe Edit Points

- `DigitRecognitionProfile` defaults and helper methods in `cfast_trainer/digit_recognition.py`.
- New drill families or timing wrappers in `cfast_trainer/dr_drills.py`.
- Workout ordering in `cfast_trainer/dr_workouts.py`.

## Common Risk Areas

- Stage timing changes can easily break both UI assumptions and headless tests.
- `DigitRecognitionPayload.display_lines` shape is consumed directly by the shell.
- Question-kind rotation and determinism are relied on by the core tests.

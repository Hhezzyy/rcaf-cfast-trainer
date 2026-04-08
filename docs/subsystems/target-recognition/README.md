# Target Recognition

## Purpose

- Mixed-panel visual search task that asks the user to register targets across scene, light, scan, and system panels.
- Whole-test engine and deterministic content generation live in `cfast_trainer/target_recognition.py`.

## Important Files

- `cfast_trainer/target_recognition.py`: config, payload shape, generator, scorer, and `build_target_recognition_test(...)`.
- `cfast_trainer/tr_drills.py`: training variants and timed drill builders.
- `cfast_trainer/tr_workouts.py`: workout block ordering and menu entries.
- `cfast_trainer/app.py`: Target Recognition rendering/input state inside `CognitiveTestScreen`.
- `tests/test_target_recognition_core.py`
- `tests/test_target_recognition_ui.py`
- `tests/test_target_recognition_headless_sim.py`
- `tests/test_tr_drills.py`
- `tests/test_tr_workouts.py`

## Lifecycle / Data Flow

1. `build_target_recognition_test(...)` returns a generic `TimedTextInputTest`.
2. `TargetRecognitionGenerator.next_problem(...)` builds the multi-panel payload and expected count.
3. `CognitiveTestScreen` renders panel-specific UI from `TargetRecognitionPayload` and forwards mouse/keyboard submissions.
4. On completion, the shell converts the engine to an `AttemptResult` and persistence happens through `cfast_trainer/results.py` and `cfast_trainer/persistence.py`.

## Input / Rendering Dependencies

- Rendering is shell-owned in `cfast_trainer/app.py`.
- Input is mostly mouse-driven for the live panels, with generic shell hotkeys layered on top.
- No subsystem-local renderer module exists; the shell keeps panel animation state for scene/light/scan/system presentation.

## Persistence / Test Hooks

- Persistence is indirect through the shared app shell and `ResultsStore`.
- Core determinism, scoring, and payload rules are covered in `tests/test_target_recognition_core.py`.
- Shell rendering and input expectations live in `tests/test_target_recognition_ui.py`.
- Practice-to-scored timing coverage lives in `tests/test_target_recognition_headless_sim.py`.

## Common Safe Edit Points

- `TargetRecognitionConfig` and payload-only additions in `cfast_trainer/target_recognition.py`.
- Generator tuning inside `TargetRecognitionGenerator` when the change is content/layout specific.
- Drill composition in `cfast_trainer/tr_drills.py`.
- Workout order or menu labels in `cfast_trainer/tr_workouts.py`.

## Common Risk Areas

- `TargetRecognitionPayload` shape changes can break shell rendering and existing UI tests.
- Multi-panel scoring assumptions depend on the generator and scorer staying aligned.
- Shell-owned panel state in `cfast_trainer/app.py` is easy to desync if you change panel semantics without updating the UI path.

# Adaptive Scheduler

## Purpose

- Selects the next drill or block based on saved evidence, recent history, and primitive-level ranking.
- Powers the live adaptive session rather than a single standalone test.

## Important Files

- `cfast_trainer/adaptive_scheduler.py`: evidence collection, ranking, block selection, live session runtime, and `build_adaptive_session_plan(...)`.
- `cfast_trainer/canonical_drill_registry.py`: canonical drill metadata, builder lookup, and difficulty-family links.
- `cfast_trainer/training_modes.py`: mode support and fatigue-probe helpers.
- `cfast_trainer/persistence.py`: attempt history loading for adaptive evidence.
- `cfast_trainer/primitive_ranking.py`: lower-level ranking inputs.
- `cfast_trainer/app.py`: `AdaptiveSessionScreen` and shell integration.
- `tests/test_adaptive_scheduler.py`
- `tests/test_canonical_drill_registry.py`
- `tests/test_training_modes.py`

## Lifecycle / Data Flow

1. Recent attempt history is loaded from persistence.
2. `collect_adaptive_evidence(...)` and `rank_adaptive_primitives(...)` turn history into ranked target areas.
3. `build_adaptive_session_plan(...)` chooses the next live block.
4. `AdaptiveSession` runs that block, records the result, and can re-plan after each completion.

## Input / Rendering Dependencies

- The scheduler itself is rendering-agnostic.
- UI behavior for intros, block overlays, pause flow, and result transitions lives in `cfast_trainer/app.py`.
- Runtime block input/rendering is delegated to whichever drill/test engine the scheduler selects.

## Persistence / Test Hooks

- This subsystem depends heavily on `AttemptHistoryEntry` data from `cfast_trainer/persistence.py`.
- `tests/test_adaptive_scheduler.py` is the primary safety net for evidence ranking, candidate selection, and session behavior.
- `tests/test_canonical_drill_registry.py` and `tests/test_training_modes.py` protect adjacent mapping layers.

## Common Safe Edit Points

- Human-readable notes or debug metadata in `cfast_trainer/adaptive_scheduler.py`.
- Canonical drill metadata additions in `cfast_trainer/canonical_drill_registry.py` when you also update tests.
- Mode-support policy in `cfast_trainer/training_modes.py`.

## Common Risk Areas

- Selection logic affects what users practice next, even when gameplay/scoring is unchanged.
- History parsing, weighting, and ranking are tightly coupled.
- Registry changes can break adaptive selection if builder names, codes, or supported modes drift out of sync.

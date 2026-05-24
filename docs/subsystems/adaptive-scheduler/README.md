# Adaptive Scheduler

## Purpose

- Selects the next drill or block based on saved evidence, recent history, and primitive-level ranking.
- Powers the live adaptive session rather than a single standalone test.
- Keeps Python/Pygame authoritative for timing, scoring, persistence, adaptive difficulty, and scheduling. Godot-owned drills may render/simulate 3D activity, but their metrics are normalized in Python before ranking.

## Important Files

- `cfast_trainer/adaptive_scheduler.py`: evidence collection, ranking, block selection, live session runtime, and `build_adaptive_session_plan(...)`.
- `cfast_trainer/skill_evidence.py`: normalized family-aware evidence scores used by ranking and adaptive difficulty.
- `cfast_trainer/adaptive_difficulty.py`: family difficulty ladders, learning-zone target bands, and live difficulty controller policy.
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
2. `AttemptResult.metrics` are standardized by `results.py`, including common aliases such as `score_ratio`, `timeout_rate`, `first_half_throughput`, `canonical_drill_code`, and Godot-compatible tracking/spatial/multitask keys.
3. `skill_evidence_from_metrics(...)` converts those metrics into a normalized `SkillEvidenceScore` for the activity's difficulty family. Raw scores are not compared directly across unrelated tests.
4. `collect_adaptive_evidence(...)` and `rank_adaptive_primitives(...)` turn normalized evidence into primitive weakness, fatigue, post-error, stability, retention, confidence, and catalog coverage signals.
5. `build_adaptive_session_plan(...)` chooses the next live block using weakness targeting, learning-zone fit, coverage-first exploration, and variety.
6. `AdaptiveSession` runs that block, records the result, and can re-plan after each completion.

## Coverage-First Exploration

- Eligible catalog primitives and drills are measured before the scheduler over-focuses on already scored weak areas. Fatigue-gated drills do not count against coverage until they are eligible.
- Cold-start scheduling gives never-attempted primitives and drills strong novelty and coverage priority so the session gathers baseline evidence.
- Mixed scheduling keeps filling unattempted and under-sampled drill gaps while weak measured primitives regain weight.
- Exploit scheduling starts once primitive and drill coverage are broad enough. Weakness, retention, fatigue, and instability dominate again, but new catalog drills still get a bounded exploration path.
- Primitive baseline evidence may come from mapped drill, integrated-test, benchmark, or adaptive-block history. Drill coverage requires canonical adaptive drill history, so a benchmark can establish a weak primitive without falsely marking every training drill attempted.

## Learning-Zone Targets

- Cognitive drills default to roughly `0.75-0.85` normalized mastery.
- Search/scanning targets `0.70-0.85` while preserving throughput improvement and false-alarm control.
- Memory-span and multitask recall targets `0.60-0.80` exact success.
- Psychomotor tracking targets `0.70-0.85` using time-on-target/control quality, not raw point score alone.
- Benchmark/full-test simulation remains fixed and comparable; adaptive targeting is not attached to benchmark launches.

## Session Intents

`build_adaptive_session_plan(..., session_intent=...)` accepts:

- `balanced_adaptive` for the default weighting.
- `weakness_focus` to emphasize weakest primitives and reduce exploration.
- `flow_training` to favor blocks near the learning-zone band.
- `benchmark_recovery` to reduce pressure after a fixed benchmark-style run.
- `fatigue_training` to allow longer/fatigue-probe blocks when eligible.
- `exam_sim_support` to keep timing and mode choices closer to realistic practice while still using adaptive evidence.

## Input / Rendering Dependencies

- The scheduler itself is rendering-agnostic.
- UI behavior for intros, block overlays, pause flow, and result transitions lives in `cfast_trainer/app.py`.
- Runtime block input/rendering is delegated to whichever drill/test engine the scheduler selects.
- For Godot-owned activities, Godot can own live visuals and simulation, but Python receives/derives normalized scoring metrics and persists them through `AttemptResult`.

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
- Benchmark scoring and difficulty must stay fixed; do not route benchmark attempts through live adaptive difficulty.

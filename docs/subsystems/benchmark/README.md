# Benchmark

## Purpose

- Fixed probe battery that runs the official test family in a defined order and summarizes per-probe plus overall results.
- Lives alongside the app shell as an orchestration subsystem rather than a single cognitive test.

## Important Files

- `cfast_trainer/benchmark.py`: probe plan models, session runtime, summary models, and `build_benchmark_plan(...)`.
- `cfast_trainer/app.py`: `BenchmarkScreen`, loading transitions, pause behavior, and result persistence glue.
- `cfast_trainer/guide_skill_catalog.py`: official test catalog that the benchmark plan asserts against.
- `tests/test_benchmark.py`
- `tests/test_intro_briefings.py`

## Lifecycle / Data Flow

1. `build_benchmark_plan(...)` creates the ordered list of probe builders.
2. `BenchmarkSession` stages intro, probe, probe-results, and overall results.
3. Each probe wraps an underlying test engine and normalizes its result shape.
4. Benchmark summaries and telemetry are then persisted through the shared results store path.

## Input / Rendering Dependencies

- Rendering and pause/menu behavior live in `cfast_trainer/app.py`.
- The session delegates probe-specific rendering/input to the wrapped engine through the shell.
- Loading transitions and intro/result screens are shared-shell concerns.

## Persistence / Test Hooks

- Benchmark writes through the shared `ResultsStore` path after probe/session completion.
- `tests/test_benchmark.py` covers plan/session behavior, persistence hooks, and shell integration.

## Common Safe Edit Points

- Plan notes, non-scoring copy, or probe metadata in `cfast_trainer/benchmark.py`.
- Benchmark UI-only presentation in `cfast_trainer/app.py` if the session contract stays stable.

## Common Risk Areas

- Probe order must stay aligned with `OFFICIAL_GUIDE_TESTS`.
- Summary aggregation affects adaptive history and downstream reporting.
- Mid-session restart or skip behavior touches both benchmark runtime and shared shell pause flow.

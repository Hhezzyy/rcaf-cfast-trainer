from __future__ import annotations

import os
from dataclasses import dataclass
from typing import cast

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from cfast_trainer.ant_drills import AntDrillMode
from cfast_trainer.app import App, AntWorkoutScreen, CognitiveTestScreen, MenuItem, MenuScreen
from cfast_trainer.ant_workouts import AntWorkoutBlockPlan, AntWorkoutPlan, AntWorkoutSession
from cfast_trainer.cognitive_core import Phase
from cfast_trainer.cognitive_core import TestSnapshot as SnapshotModel
from cfast_trainer.table_reading import TableReadingPayload, TableReadingPart, TableReadingGenerator
from cfast_trainer.tbl_drills import build_tbl_mixed_tempo_drill


@dataclass
class _FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class _FakeTableReadingEngine:
    def __init__(self, payload: TableReadingPayload, *, title: str) -> None:
        self._payload = payload
        self._title = title
        self.submissions: list[str] = []

    def snapshot(self) -> SnapshotModel:
        return SnapshotModel(
            title=self._title,
            phase=Phase.PRACTICE,
            prompt=self._payload.stem,
            input_hint="",
            time_remaining_s=None,
            attempted_scored=0,
            correct_scored=0,
            payload=self._payload,
        )

    def can_exit(self) -> bool:
        return True

    def start_practice(self) -> None:
        pass

    def start_scored(self) -> None:
        pass

    def submit_answer(self, raw: str) -> bool:
        self.submissions.append(str(raw))
        return True

    def update(self) -> None:
        pass


def _sample_payload(*, part: TableReadingPart) -> TableReadingPayload:
    generator = TableReadingGenerator(seed=202)
    problem = generator.next_problem_for_selection(difficulty=0.5, part=part)
    return cast(TableReadingPayload, problem.payload)


def _build_screen(engine: object) -> tuple[App, CognitiveTestScreen]:
    pygame.init()
    surface = pygame.display.set_mode((960, 540))
    font = pygame.font.Font(None, 36)
    app = App(surface=surface, font=font)
    root = MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True)
    app.push(root)
    screen = CognitiveTestScreen(app, engine_factory=lambda: engine)
    app.push(screen)
    return app, screen


def test_table_reading_drill_title_still_routes_to_real_renderer(monkeypatch) -> None:
    payload = _sample_payload(part=TableReadingPart.PART_ONE)
    _app, screen = _build_screen(
        _FakeTableReadingEngine(payload, title="Table Reading: Part 1 Anchor")
    )
    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        called = {"value": False}
        original = screen._render_table_reading_screen

        def wrapped(surface, snap, payload):
            called["value"] = True
            return original(surface, snap, payload)

        monkeypatch.setattr(screen, "_render_table_reading_screen", wrapped)
        screen.render(surface)

        assert called["value"] is True
    finally:
        pygame.quit()


def test_table_reading_choice_keys_submit_immediately() -> None:
    payload = _sample_payload(part=TableReadingPart.PART_TWO)
    engine = _FakeTableReadingEngine(payload, title="Table Reading: Part 2 Prime")
    _app, screen = _build_screen(engine)
    try:
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_d, "mod": 0, "unicode": "d"})
        )

        assert engine.submissions == ["3"]
    finally:
        pygame.quit()


def test_table_reading_workout_block_uses_real_runtime_screen() -> None:
    pygame.init()
    try:
        surface = pygame.display.set_mode((960, 540))
        font = pygame.font.Font(None, 36)
        app = App(surface=surface, font=font)
        root = MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True)
        app.push(root)

        clock = _FakeClock()
        plan = AntWorkoutPlan(
            code="table_reading_workout",
            title="Table Reading Workout UI",
            description="UI regression workout.",
            notes=("Untimed reflections.",),
            blocks=(
                AntWorkoutBlockPlan(
                    block_id="part1",
                    label="Part 1 Anchor",
                    description="Warm-up.",
                    focus_skills=("Single-table lookup",),
                    drill_code="tbl_part1_anchor",
                    mode=AntDrillMode.BUILD,
                    duration_min=0.25,
                ),
            ),
        )
        session = AntWorkoutSession(
            clock=clock,
            seed=606,
            plan=plan,
            starting_level=5,
        )
        session.activate()
        session.append_text("focus")
        session.activate()
        session.append_text("scan")
        session.activate()
        session.activate()

        screen = AntWorkoutScreen(app, session=session, test_code="table_reading_workout")
        app.push(screen)
        screen.render(surface)

        runtime = screen._runtime_screen
        assert runtime is not None
        assert runtime._engine is not None
        snap = runtime._engine.snapshot()
        assert str(snap.title).startswith("Table Reading")
        assert isinstance(snap.payload, TableReadingPayload)
    finally:
        pygame.quit()


def test_table_reading_real_drill_engine_uses_table_payload_on_live_screen() -> None:
    clock = _FakeClock()
    engine = build_tbl_mixed_tempo_drill(
        clock=clock,
        seed=505,
        difficulty=0.5,
        mode=AntDrillMode.BUILD,
    )
    engine.start_scored()
    _app, screen = _build_screen(engine)
    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        screen.render(surface)

        payload = engine._current.payload
        assert isinstance(payload, TableReadingPayload)
        assert payload.part in (TableReadingPart.PART_ONE, TableReadingPart.PART_TWO)
    finally:
        pygame.quit()

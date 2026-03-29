from __future__ import annotations

import os
from dataclasses import dataclass

import pygame

from cfast_trainer.app import App, CognitiveTestScreen, DifficultySettingsStore, MenuItem, MenuScreen
from cfast_trainer.cognitive_core import Phase
from cfast_trainer.trace_drills import build_tt1_lateral_anchor_drill, build_tt2_steady_anchor_drill
from cfast_trainer.trace_test_1 import TraceTest1Config, build_trace_test_1_test
from cfast_trainer.trace_test_2 import TraceTest2Config, build_trace_test_2_test


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += float(dt)


def _build_screen(
    *,
    engine_factory,
    test_code: str,
    review_mode: bool = False,
    settings_path=None,
) -> CognitiveTestScreen:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    pygame.init()
    surface = pygame.Surface((960, 540))
    font = pygame.font.Font(None, 36)
    difficulty_settings_store = None
    if settings_path is not None:
        difficulty_settings_store = DifficultySettingsStore(settings_path)
        difficulty_settings_store.set_review_mode_enabled(review_mode)
    app = App(
        surface=surface,
        font=font,
        opengl_enabled=False,
        difficulty_settings_store=difficulty_settings_store,
    )
    app.push(MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True))
    screen = CognitiveTestScreen(app, engine_factory=engine_factory, test_code=test_code)
    app.push(screen)
    return screen


def test_trace_test_1_arrow_input_only_registers_after_answer_open() -> None:
    clock = FakeClock()
    screen = _build_screen(
        engine_factory=lambda: build_trace_test_1_test(
            clock=clock,
            seed=17,
            difficulty=0.5,
            config=TraceTest1Config(
                practice_questions=1,
                practice_observe_s=1.0,
                scored_observe_s=1.0,
            ),
        ),
        test_code="trace_test_1",
    )
    screen._engine.start_practice()

    screen.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_LEFT, "unicode": ""}))
    assert screen._engine.phase is Phase.PRACTICE

    clock.advance(0.50)
    screen._engine.update()
    screen.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_LEFT, "unicode": ""}))
    assert screen._engine.phase is Phase.PRACTICE_DONE


def test_trace_test_1_arrow_input_opens_answer_checker_in_review_mode(tmp_path) -> None:
    clock = FakeClock()
    screen = _build_screen(
        engine_factory=lambda: build_trace_test_1_test(
            clock=clock,
            seed=17,
            difficulty=0.5,
            config=TraceTest1Config(
                practice_questions=1,
                practice_observe_s=1.0,
                scored_observe_s=1.0,
            ),
        ),
        test_code="trace_test_1",
        review_mode=True,
        settings_path=tmp_path / "difficulty-settings.json",
    )
    screen._engine.start_practice()

    clock.advance(0.50)
    screen._engine.update()
    payload = screen._engine.snapshot().payload
    assert payload is not None

    expected = {
        1: "LEFT",
        2: "RIGHT",
        3: "UP",
        4: "DOWN",
    }[int(payload.correct_code)]

    screen.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_LEFT, "unicode": ""}))

    assert screen._review_state is not None
    assert screen._review_state.submitted_raw == "LEFT"
    assert screen._review_state.correct_choice_code is None
    assert screen._review_state.submitted_choice_code is None
    assert screen._review_state.correct_answer_text == expected


def test_trace_test_2_letter_choice_submits_only_after_observe_stage() -> None:
    clock = FakeClock()
    screen = _build_screen(
        engine_factory=lambda: build_trace_test_2_test(
            clock=clock,
            seed=19,
            difficulty=0.5,
            config=TraceTest2Config(
                practice_questions=1,
                practice_observe_s=0.5,
                scored_observe_s=0.5,
            ),
        ),
        test_code="trace_test_2",
    )
    screen._engine.start_practice()

    screen.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_a, "unicode": "a"}))
    assert screen._engine.phase is Phase.PRACTICE

    clock.advance(0.55)
    screen._engine.update()
    screen.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_a, "unicode": "a"}))
    assert screen._engine.phase is Phase.PRACTICE_DONE


def test_trace_test_2_up_down_then_enter_submits_selection() -> None:
    clock = FakeClock()
    screen = _build_screen(
        engine_factory=lambda: build_trace_test_2_test(
            clock=clock,
            seed=23,
            difficulty=0.5,
            config=TraceTest2Config(
                practice_questions=1,
                practice_observe_s=0.5,
                scored_observe_s=0.5,
            ),
        ),
        test_code="trace_test_2",
    )
    screen._engine.start_practice()

    clock.advance(0.55)
    screen._engine.update()
    screen.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""}))
    screen.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": "\r"}))
    assert screen._engine.phase is Phase.PRACTICE_DONE


def test_trace_test_2_digit_then_enter_submits_selection() -> None:
    clock = FakeClock()
    screen = _build_screen(
        engine_factory=lambda: build_trace_test_2_test(
            clock=clock,
            seed=29,
            difficulty=0.5,
            config=TraceTest2Config(
                practice_questions=1,
                practice_observe_s=0.5,
                scored_observe_s=0.5,
            ),
        ),
        test_code="trace_test_2",
    )
    screen._engine.start_practice()

    clock.advance(0.55)
    screen._engine.update()
    screen.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_3, "unicode": "3"}))
    screen.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": "\r"}))
    assert screen._engine.phase is Phase.PRACTICE_DONE


def test_trace_test_1_custom_drill_title_routes_to_trace_renderer() -> None:
    clock = FakeClock()
    screen = _build_screen(
        engine_factory=lambda: build_tt1_lateral_anchor_drill(
            clock=clock,
            seed=31,
            difficulty=0.5,
        ),
        test_code="tt1_lateral_anchor",
    )
    screen._engine.start_practice()
    called = {"tt1": False}

    def _fake_render(surface, snap, payload) -> None:
        called["tt1"] = True

    screen._render_trace_test_1_screen = _fake_render  # type: ignore[method-assign]
    screen.render(pygame.Surface((960, 540)))

    assert called["tt1"] is True


def test_trace_test_2_custom_drill_title_routes_to_trace_renderer() -> None:
    clock = FakeClock()
    screen = _build_screen(
        engine_factory=lambda: build_tt2_steady_anchor_drill(
            clock=clock,
            seed=37,
            difficulty=0.5,
        ),
        test_code="tt2_steady_anchor",
    )
    screen._engine.start_practice()
    called = {"tt2": False}

    def _fake_render(surface, snap, payload) -> None:
        called["tt2"] = True

    screen._render_trace_test_2_screen = _fake_render  # type: ignore[method-assign]
    screen.render(pygame.Surface((960, 540)))

    assert called["tt2"] is True

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


def test_trace_test_1_arrow_input_faults_wrong_choice_during_observe_stage() -> None:
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

    payload = screen._engine.snapshot().payload
    assert payload is not None
    wrong_key, wrong_raw = {
        1: (pygame.K_RIGHT, "RIGHT"),
        2: (pygame.K_LEFT, "LEFT"),
        3: (pygame.K_DOWN, "DOWN"),
        4: (pygame.K_UP, "UP"),
    }[int(payload.correct_code)]

    screen.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": wrong_key, "unicode": ""}))
    assert screen._engine.phase is Phase.PRACTICE
    assert screen._engine.events() == []

    clock.advance(1.05)
    screen._engine.update()
    assert screen._engine.events()[-1].raw == wrong_raw
    assert screen._engine.events()[-1].is_correct is False
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

    clock.advance(0.20)
    screen._engine.update()
    payload = screen._engine.snapshot().payload
    assert payload is not None
    assert float(payload.observe_progress) < float(payload.answer_open_progress)

    expected = {
        1: "LEFT",
        2: "RIGHT",
        3: "UP",
        4: "DOWN",
    }[int(payload.correct_code)]
    wrong_key, wrong_raw = {
        1: (pygame.K_RIGHT, "RIGHT"),
        2: (pygame.K_LEFT, "LEFT"),
        3: (pygame.K_DOWN, "DOWN"),
        4: (pygame.K_UP, "UP"),
    }[int(payload.correct_code)]

    screen.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": wrong_key, "unicode": ""}))

    assert screen._review_state is not None
    assert screen._review_state.submitted_raw == wrong_raw
    assert screen._review_state.correct_choice_code is None
    assert screen._review_state.submitted_choice_code is None
    assert screen._review_state.correct_answer_text == expected
    assert screen._review_state.blocks_runtime is False
    assert screen._review_clock is not None
    assert screen._review_clock.is_paused() is False


def test_trace_test_1_review_overlay_auto_clears_without_second_submit(tmp_path) -> None:
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

    clock.advance(0.20)
    screen._engine.update()
    screen.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_LEFT, "unicode": ""}))

    assert screen._review_state is not None

    now = screen._review_clock.now() if screen._review_clock is not None else 0.0
    clock.advance(1.0)
    screen.render(pygame.Surface((960, 540)))

    assert screen._review_state is None
    if screen._review_clock is not None:
        assert screen._review_clock.now() > now


def test_trace_test_1_review_overlay_detects_no_input_timeout(tmp_path) -> None:
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

    payload = screen._engine.snapshot().payload
    assert payload is not None
    expected = {
        1: "LEFT",
        2: "RIGHT",
        3: "UP",
        4: "DOWN",
    }[int(payload.correct_code)]

    clock.advance(1.05)
    screen.render(pygame.Surface((960, 540)))

    assert screen._review_state is not None
    assert screen._review_state.submitted_raw == "NO INPUT"
    assert screen._review_state.correct_answer_text == expected
    assert screen._review_state.blocks_runtime is False


def test_trace_test_1_answer_input_does_not_swap_prompt_immediately(tmp_path) -> None:
    clock = FakeClock()
    screen = _build_screen(
        engine_factory=lambda: build_trace_test_1_test(
            clock=clock,
            seed=17,
            difficulty=0.5,
            config=TraceTest1Config(
                practice_questions=2,
                practice_observe_s=1.0,
                scored_observe_s=1.0,
            ),
        ),
        test_code="trace_test_1",
        review_mode=True,
        settings_path=tmp_path / "difficulty-settings.json",
    )
    screen._engine.start_practice()

    clock.advance(0.20)
    screen._engine.update()
    before = screen._engine.snapshot().payload
    assert before is not None

    screen.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_LEFT, "unicode": ""}))
    after = screen._engine.snapshot().payload
    assert after is not None
    assert after.prompt_index == before.prompt_index

    clock.advance(0.20)
    screen._engine.update()
    mid = screen._engine.snapshot().payload
    assert mid is not None
    assert mid.prompt_index == before.prompt_index

    clock.advance(0.40)
    screen._engine.update()
    next_payload = screen._engine.snapshot().payload
    assert next_payload is not None
    assert next_payload.prompt_index == before.prompt_index

    clock.advance(0.20)
    screen._engine.update()
    final_payload = screen._engine.snapshot().payload
    assert final_payload is not None
    assert final_payload.prompt_index == before.prompt_index + 1


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

from __future__ import annotations

import os
from dataclasses import dataclass

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from cfast_trainer.app import App, CognitiveTestScreen, DifficultySettingsStore, MenuItem, MenuScreen
from cfast_trainer.cognitive_core import Phase, Problem
from cfast_trainer.cognitive_core import TestSnapshot as SnapshotModel


class FakeClock:
    def __init__(self) -> None:
        self._now = 0.0

    def now(self) -> float:
        return self._now

    def advance(self, delta_s: float) -> None:
        self._now += float(delta_s)


@dataclass(frozen=True, slots=True)
class _FakeOption:
    code: int
    text: str


@dataclass(frozen=True, slots=True)
class _FakePayload:
    options: tuple[_FakeOption, ...]


class _ReviewEngine:
    def __init__(self, *, clock: FakeClock, payload: object | None) -> None:
        self._clock = clock
        self._phase = Phase.PRACTICE
        self._difficulty = 0.5
        self._title = "Fake Review Test"
        self._payload = payload
        self._current = Problem(prompt="Question 1", answer=2, payload=payload)
        self._presented_at_s = self._clock.now()
        self.submit_count = 0

    def snapshot(self) -> SnapshotModel:
        return SnapshotModel(
            title=self._title,
            phase=self._phase,
            prompt=self._current.prompt,
            input_hint="Type answer then Enter",
            time_remaining_s=30.0,
            attempted_scored=0,
            correct_scored=0,
            payload=self._payload,
        )

    def can_exit(self) -> bool:
        return True

    def start_practice(self) -> None:
        return

    def start_scored(self) -> None:
        self._phase = Phase.SCORED

    def submit_answer(self, raw: str) -> bool:
        _ = raw
        self.submit_count += 1
        self._current = Problem(prompt="Question 2", answer=3, payload=self._payload)
        self._presented_at_s = self._clock.now()
        return True

    def update(self) -> None:
        return


def _build_screen(*, tmp_path, review_mode: bool, payload: object | None) -> tuple[CognitiveTestScreen, FakeClock]:
    pygame.init()
    surface = pygame.display.set_mode((960, 540))
    font = pygame.font.Font(None, 36)
    store = DifficultySettingsStore(tmp_path / "difficulty-settings.json")
    store.set_review_mode_enabled(review_mode)
    app = App(surface=surface, font=font, difficulty_settings_store=store)
    app.push(MenuScreen(app, "Main", [MenuItem("Quit", app.quit)], is_root=True))
    clock = FakeClock()
    engine = _ReviewEngine(clock=clock, payload=payload)
    screen = CognitiveTestScreen(app, engine_factory=lambda: engine, test_code="numerical_operations")
    app.push(screen)
    return screen, clock


def test_review_mode_pauses_after_typed_submit_until_second_enter(tmp_path) -> None:
    screen, base_clock = _build_screen(tmp_path=tmp_path, review_mode=True, payload=None)
    try:
        screen._input = "17"
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "mod": 0, "unicode": ""})
        )

        assert screen._review_state is not None
        assert screen._review_state.submitted_raw == "17"
        assert screen._review_state.correct_answer_text == "2"
        assert screen._review_clock is not None
        assert screen._review_clock.is_paused() is True

        paused_now = screen._review_clock.now()
        base_clock.advance(5.0)
        assert screen._review_clock.now() == paused_now

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "mod": 0, "unicode": ""})
        )

        assert screen._review_state is None
        assert screen._review_clock.is_paused() is False
    finally:
        pygame.quit()


def test_review_mode_tracks_correct_and_selected_multiple_choice_codes(tmp_path) -> None:
    payload = _FakePayload(
        options=(
            _FakeOption(code=1, text="One"),
            _FakeOption(code=2, text="Two"),
            _FakeOption(code=3, text="Three"),
        )
    )
    screen, _clock = _build_screen(tmp_path=tmp_path, review_mode=True, payload=payload)
    try:
        screen._input = "1"
        screen._math_choice = 1
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "mod": 0, "unicode": ""})
        )

        assert screen._review_state is not None
        assert screen._review_state.correct_choice_code == 2
        assert screen._review_state.submitted_choice_code == 1
        assert screen._review_state.correct_answer_text == "S"
    finally:
        pygame.quit()


def test_review_mode_off_keeps_fast_submit_behavior(tmp_path) -> None:
    screen, _clock = _build_screen(tmp_path=tmp_path, review_mode=False, payload=None)
    try:
        screen._input = "17"
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "mod": 0, "unicode": ""})
        )

        assert screen._review_state is None
        assert screen._input == ""
    finally:
        pygame.quit()

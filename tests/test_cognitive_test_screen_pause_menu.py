from __future__ import annotations

import os
from typing import cast

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame
import pytest

from cfast_trainer.app import (
    App,
    CognitiveTestScreen,
    DifficultySettingsStore,
    MenuItem,
    MenuScreen,
)
from cfast_trainer.cognitive_core import Phase
from cfast_trainer.cognitive_core import TestSnapshot as SnapshotModel


class _FakeEngine:
    def __init__(self, *, phase: Phase = Phase.PRACTICE, title: str = "Fake Test") -> None:
        self._phase = phase
        self._title = title
        self.update_count = 0
        self._difficulty = 0.5
        self._noise_level_override: float | None = None
        self._distortion_level_override: float | None = None
        self._noise_source_override: str | None = None

    def snapshot(self) -> SnapshotModel:
        return SnapshotModel(
            title=self._title,
            phase=self._phase,
            prompt="",
            input_hint="",
            time_remaining_s=60.0 if self._phase is Phase.SCORED else None,
            attempted_scored=0,
            correct_scored=0,
            payload=None,
        )

    def can_exit(self) -> bool:
        return self._phase in (
            Phase.INSTRUCTIONS,
            Phase.PRACTICE,
            Phase.PRACTICE_DONE,
            Phase.RESULTS,
        )

    def start_practice(self) -> None:
        if self._phase is Phase.INSTRUCTIONS:
            self._phase = Phase.PRACTICE

    def start_scored(self) -> None:
        if self._phase is Phase.PRACTICE_DONE:
            self._phase = Phase.SCORED

    def submit_answer(self, raw: str) -> bool:
        token = str(raw).strip().lower()
        if token in {"__skip_practice__", "skip_practice"} and self._phase is Phase.PRACTICE:
            self._phase = Phase.PRACTICE_DONE
            return True
        if token in {"__skip_section__", "skip_section", "__skip_all__", "skip_all"} and (
            self._phase is Phase.SCORED
        ):
            self._phase = Phase.RESULTS
            return True
        return False

    def update(self) -> None:
        self.update_count += 1

    def set_audio_overrides(
        self,
        *,
        noise_level: float | None = None,
        distortion_level: float | None = None,
        noise_source: str | None = None,
    ) -> None:
        self._noise_level_override = noise_level
        self._distortion_level_override = distortion_level
        self._noise_source_override = noise_source


def _build_app_and_screen(
    *,
    phase: Phase = Phase.PRACTICE,
    title: str = "Fake Test",
    test_code: str | None = None,
    difficulty_settings_store: DifficultySettingsStore | None = None,
) -> tuple[App, CognitiveTestScreen, list[_FakeEngine]]:
    pygame.init()
    surface = pygame.display.set_mode((960, 540))
    font = pygame.font.Font(None, 36)
    app = App(
        surface=surface,
        font=font,
        difficulty_settings_store=difficulty_settings_store,
    )
    root = MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True)
    app.push(root)
    created: list[_FakeEngine] = []

    def factory() -> _FakeEngine:
        engine_phase = phase if not created else Phase.INSTRUCTIONS
        engine = _FakeEngine(phase=engine_phase, title=title)
        if test_code is not None:
            engine._difficulty = (app.effective_difficulty_level(test_code) - 1) / 9.0
        created.append(engine)
        return engine

    screen = CognitiveTestScreen(app, engine_factory=factory, test_code=test_code)
    app.push(screen)
    return app, screen, created


def test_pause_menu_escape_then_resume_resumes_test() -> None:
    app, screen, _engines = _build_app_and_screen()
    try:
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE, "mod": 0, "unicode": ""})
        )
        assert screen._pause_menu_active is True

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "mod": 0, "unicode": ""})
        )
        assert screen._pause_menu_active is False
        assert len(app._screens) == 2
    finally:
        pygame.quit()


def test_pause_menu_escape_opens_from_instructions() -> None:
    app, screen, _engines = _build_app_and_screen(phase=Phase.INSTRUCTIONS)
    try:
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE, "mod": 0, "unicode": ""})
        )
        assert screen._pause_menu_active is True
        assert len(app._screens) == 2
    finally:
        pygame.quit()


def test_pause_menu_main_menu_returns_to_root() -> None:
    app, screen, _engines = _build_app_and_screen()
    try:
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE, "mod": 0, "unicode": ""})
        )
        for _ in range(2):
            screen.handle_event(
                pygame.event.Event(
                    pygame.KEYDOWN,
                    {"key": pygame.K_DOWN, "mod": 0, "unicode": ""},
                )
            )
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "mod": 0, "unicode": ""})
        )

        assert len(app._screens) == 1
    finally:
        pygame.quit()


def test_pause_menu_freezes_engine_updates_while_open() -> None:
    _app, screen, engines = _build_app_and_screen()
    try:
        surface = pygame.display.get_surface()
        assert surface is not None

        engine = engines[-1]
        screen.render(surface)
        before_pause = engine.update_count
        assert before_pause > 0

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE, "mod": 0, "unicode": ""})
        )
        screen.render(surface)
        assert engine.update_count == before_pause
    finally:
        pygame.quit()


def test_pause_menu_mouse_click_activates_main_menu_row() -> None:
    app, screen, _engines = _build_app_and_screen(phase=Phase.INSTRUCTIONS)
    try:
        surface = pygame.display.get_surface()
        assert surface is not None

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE, "mod": 0, "unicode": ""})
        )
        screen.render(surface)
        hitbox = screen._pause_menu_hitboxes[2]
        screen.handle_event(
            pygame.event.Event(
                pygame.MOUSEBUTTONDOWN,
                {"button": 1, "pos": hitbox.center},
            )
        )

        assert len(app._screens) == 1
    finally:
        pygame.quit()


def test_pause_settings_apply_and_restart_rebuilds_screen_with_new_difficulty(tmp_path) -> None:
    store = DifficultySettingsStore(tmp_path / "difficulty-settings.json")
    app, screen, engines = _build_app_and_screen(
        phase=Phase.PRACTICE,
        test_code="rapid_tracking",
        difficulty_settings_store=store,
    )
    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        first_engine = engines[-1]

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE, "mod": 0, "unicode": ""})
        )
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "mod": 0, "unicode": ""})
        )
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "mod": 0, "unicode": ""})
        )
        screen.render(surface)
        assert screen._pause_menu_mode == "settings"

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RIGHT, "mod": 0, "unicode": ""})
        )
        assert store.test_level("rapid_tracking") == 5
        assert first_engine._difficulty == pytest.approx((5 - 1) / 9.0)

        for _ in range(4):
            screen.handle_event(
                pygame.event.Event(
                    pygame.KEYDOWN,
                    {"key": pygame.K_DOWN, "mod": 0, "unicode": ""},
                )
            )
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "mod": 0, "unicode": ""})
        )

        new_screen = cast(CognitiveTestScreen, app._screens[-1])
        assert new_screen is not screen
        assert store.test_level("rapid_tracking") == 6
        assert app.effective_difficulty_level("rapid_tracking") == 6
        assert len(engines) == 2
        assert engines[-1]._difficulty == pytest.approx((6 - 1) / 9.0)
    finally:
        pygame.quit()


def test_intro_difficulty_is_staged_until_enter_starts_practice(tmp_path) -> None:
    store = DifficultySettingsStore(tmp_path / "difficulty-settings.json")
    app, screen, engines = _build_app_and_screen(
        phase=Phase.INSTRUCTIONS,
        test_code="rapid_tracking",
        difficulty_settings_store=store,
    )
    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        first_engine = engines[-1]

        screen.render(surface)
        assert screen._get_intro_difficulty_level() == 5
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RIGHT, "mod": 0, "unicode": ""})
        )

        assert store.test_level("rapid_tracking") == 5
        assert first_engine._difficulty == pytest.approx((5 - 1) / 9.0)

        for _ in range(8):
            screen.render(surface)

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "mod": 0, "unicode": ""})
        )

        new_screen = cast(CognitiveTestScreen, app._screens[-1])
        assert new_screen is not screen
        assert store.test_level("rapid_tracking") == 6
        assert app.effective_difficulty_level("rapid_tracking") == 6
        assert len(engines) == 2
        assert engines[-1]._difficulty == pytest.approx((6 - 1) / 9.0)
        assert engines[-1].snapshot().phase is Phase.PRACTICE
    finally:
        pygame.quit()


def test_intro_difficulty_change_from_practice_done_restarts_to_beginning(tmp_path) -> None:
    store = DifficultySettingsStore(tmp_path / "difficulty-settings.json")
    store.set_test_level(test_code="rapid_tracking", level=7)
    app, screen, engines = _build_app_and_screen(
        phase=Phase.PRACTICE_DONE,
        test_code="rapid_tracking",
        difficulty_settings_store=store,
    )
    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        first_engine = engines[-1]

        screen.render(surface)
        for _ in range(8):
            screen.render(surface)

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_LEFT, "mod": 0, "unicode": ""})
        )
        assert store.test_level("rapid_tracking") == 7
        assert engines[-1]._difficulty == pytest.approx((7 - 1) / 9.0)

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "mod": 0, "unicode": ""})
        )

        new_screen = cast(CognitiveTestScreen, app._screens[-1])
        assert new_screen is not screen
        assert store.test_level("rapid_tracking") == 6
        assert first_engine.snapshot().phase is Phase.PRACTICE_DONE
        assert first_engine._difficulty == pytest.approx((7 - 1) / 9.0)
        assert engines[-1] is not first_engine
        assert engines[-1].snapshot().phase is Phase.INSTRUCTIONS
        assert engines[-1]._difficulty == pytest.approx((6 - 1) / 9.0)
    finally:
        pygame.quit()


def test_intro_loading_blocks_enter_until_practice_stage_is_ready() -> None:
    _app, screen, engines = _build_app_and_screen(phase=Phase.INSTRUCTIONS)
    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        engine = engines[-1]

        screen.render(surface)
        assert screen._intro_loading_complete(Phase.INSTRUCTIONS) is False

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "mod": 0, "unicode": ""})
        )
        assert engine.snapshot().phase is Phase.INSTRUCTIONS

        for _ in range(8):
            screen.render(surface)

        assert screen._intro_loading_complete(Phase.INSTRUCTIONS) is True
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "mod": 0, "unicode": ""})
        )
        assert engine.snapshot().phase is Phase.PRACTICE
    finally:
        pygame.quit()


def test_practice_done_loading_blocks_enter_until_scored_stage_is_ready() -> None:
    _app, screen, engines = _build_app_and_screen(phase=Phase.PRACTICE_DONE)
    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        engine = engines[-1]

        screen.render(surface)
        assert screen._intro_loading_complete(Phase.PRACTICE_DONE) is False

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "mod": 0, "unicode": ""})
        )
        assert engine.snapshot().phase is Phase.PRACTICE_DONE

        for _ in range(8):
            screen.render(surface)

        assert screen._intro_loading_complete(Phase.PRACTICE_DONE) is True
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "mod": 0, "unicode": ""})
        )
        assert engine.snapshot().phase is Phase.SCORED
    finally:
        pygame.quit()


def test_pause_settings_changes_auditory_mix_controls() -> None:
    _app, screen, engines = _build_app_and_screen(title="Auditory Capacity")
    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        engine = engines[-1]

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE, "mod": 0, "unicode": ""})
        )
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "mod": 0, "unicode": ""})
        )
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "mod": 0, "unicode": ""})
        )
        screen.render(surface)

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "mod": 0, "unicode": ""})
        )
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RIGHT, "mod": 0, "unicode": ""})
        )
        assert engine._noise_level_override == 0.1

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "mod": 0, "unicode": ""})
        )
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RIGHT, "mod": 0, "unicode": ""})
        )
        assert engine._distortion_level_override == 0.1

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "mod": 0, "unicode": ""})
        )
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RIGHT, "mod": 0, "unicode": ""})
        )
        assert engine._noise_source_override is not None
    finally:
        pygame.quit()

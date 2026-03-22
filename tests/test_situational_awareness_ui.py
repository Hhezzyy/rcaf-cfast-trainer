from __future__ import annotations

import os
from dataclasses import dataclass

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from cfast_trainer.app import App, CognitiveTestScreen, MenuItem, MenuScreen
from cfast_trainer.cognitive_core import Phase
from cfast_trainer.cognitive_core import TestSnapshot as SnapshotModel
from cfast_trainer.situational_awareness import (
    SituationalAwarenessActiveQuery,
    SituationalAwarenessAnswerChoice,
    SituationalAwarenessAnswerMode,
    SituationalAwarenessCueCard,
    SituationalAwarenessPayload,
    SituationalAwarenessQueryKind,
    SituationalAwarenessScenarioFamily,
    SituationalAwarenessVisibleContact,
)


@dataclass
class _FakeSituationAwarenessEngine:
    payload: SituationalAwarenessPayload
    phase: Phase = Phase.PRACTICE
    title: str = "Situational Awareness"

    def __post_init__(self) -> None:
        self.submissions: list[str] = []

    def snapshot(self) -> SnapshotModel:
        prompt = (
            self.payload.active_query.prompt
            if self.payload.active_query is not None
            else "Stand by for the next query."
        )
        return SnapshotModel(
            title=self.title,
            phase=self.phase,
            prompt=prompt,
            input_hint="",
            time_remaining_s=None,
            attempted_scored=0,
            correct_scored=0,
            payload=self.payload,
        )

    def can_exit(self) -> bool:
        return True

    def start_practice(self) -> None:
        return

    def start_scored(self) -> None:
        return

    def submit_answer(self, raw: str) -> bool:
        self.submissions.append(str(raw))
        return True

    def update(self) -> None:
        return


def _sample_payload(*, answer_mode: SituationalAwarenessAnswerMode) -> SituationalAwarenessPayload:
    action_choices = (
        SituationalAwarenessAnswerChoice(1, "Safe now."),
        SituationalAwarenessAnswerChoice(2, "Wait one sweep, then go."),
        SituationalAwarenessAnswerChoice(3, "Unsafe. Hold clear of traffic."),
        SituationalAwarenessAnswerChoice(4, "Request a fresh picture first."),
    )
    active_query = SituationalAwarenessActiveQuery(
        query_id=101,
        kind=(
            SituationalAwarenessQueryKind.SAFE_TO_MOVE
            if answer_mode is SituationalAwarenessAnswerMode.CHOICE
            else SituationalAwarenessQueryKind.FUTURE_LOCATION
        ),
        answer_mode=answer_mode,
        prompt=(
            "Where will LEEDS be in 12s?"
            if answer_mode is SituationalAwarenessAnswerMode.GRID_CELL
            else "Is it safe for LEEDS to proceed to F5 now?"
        ),
        correct_answer_token="E5" if answer_mode is SituationalAwarenessAnswerMode.GRID_CELL else "2",
        expires_in_s=9.0,
        subject_callsign="LEEDS",
        future_offset_s=12 if answer_mode is SituationalAwarenessAnswerMode.GRID_CELL else None,
        answer_choices=action_choices if answer_mode is SituationalAwarenessAnswerMode.CHOICE else (),
    )
    return SituationalAwarenessPayload(
        scenario_family=SituationalAwarenessScenarioFamily.MERGE_CONFLICT,
        scenario_label="Conflict / Safety 1/1",
        scenario_index=1,
        scenario_total=1,
        active_channels=("pictorial", "coded", "numerical", "aural"),
        active_query_kinds=("current_location", "future_location"),
        focus_label="Current + future location",
        segment_label="Picture Anchor",
        segment_index=1,
        segment_total=1,
        segment_time_remaining_s=42.0,
        scenario_elapsed_s=18.0,
        scenario_time_remaining_s=42.0,
        next_query_in_s=None,
        visible_contacts=(
            SituationalAwarenessVisibleContact("LEEDS", "friendly", 2.0, 4.0, "E2", "E", 0.95),
            SituationalAwarenessVisibleContact("RAVEN", "hostile", 6.0, 4.0, "E6", "W", 0.65),
        ),
        cue_card=SituationalAwarenessCueCard("LEEDS", "friendly", "F5", "11:00:35", "FL220", "CH 3", 0.82),
        waypoints=(),
        top_strip_text="LEEDS channel 3, direct F5.",
        top_strip_fade=0.72,
        display_clock_text="11:00:16",
        active_query=active_query,
        answer_mode=answer_mode,
        correct_answer_token=active_query.correct_answer_token,
        announcement_token=("sa", 1),
        announcement_lines=("Conflict update.", active_query.prompt),
    )


def _build_screen(engine: _FakeSituationAwarenessEngine) -> CognitiveTestScreen:
    pygame.init()
    surface = pygame.display.set_mode((960, 540))
    font = pygame.font.Font(None, 36)
    app = App(surface=surface, font=font)
    root = MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True)
    app.push(root)
    screen = CognitiveTestScreen(app, engine_factory=lambda: engine)
    app.push(screen)
    return screen


def test_situational_awareness_renderer_uses_sparse_grid_and_query_layout() -> None:
    engine = _FakeSituationAwarenessEngine(payload=_sample_payload(answer_mode=SituationalAwarenessAnswerMode.GRID_CELL))
    screen = _build_screen(engine)
    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        screen.render(surface)

        assert len(screen._sa_grid_hitboxes) == 100
        assert screen._sa_option_hitboxes == {}
    finally:
        pygame.quit()


def test_situational_awareness_grid_click_submits_grid_cell() -> None:
    engine = _FakeSituationAwarenessEngine(payload=_sample_payload(answer_mode=SituationalAwarenessAnswerMode.GRID_CELL))
    screen = _build_screen(engine)
    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        screen.render(surface)

        rect = screen._sa_grid_hitboxes["E5"]
        screen.handle_event(
            pygame.event.Event(
                pygame.MOUSEBUTTONDOWN,
                {"button": 1, "pos": rect.center},
            )
        )

        assert engine.submissions == ["E5"]
    finally:
        pygame.quit()


def test_situational_awareness_choice_click_submits_choice_code() -> None:
    engine = _FakeSituationAwarenessEngine(payload=_sample_payload(answer_mode=SituationalAwarenessAnswerMode.CHOICE))
    screen = _build_screen(engine)
    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        screen.render(surface)

        rect = screen._sa_option_hitboxes[2]
        screen.handle_event(
            pygame.event.Event(
                pygame.MOUSEBUTTONDOWN,
                {"button": 1, "pos": rect.center},
            )
        )

        assert engine.submissions == ["2"]
    finally:
        pygame.quit()


def test_situational_awareness_keyboard_choice_submission_is_immediate() -> None:
    engine = _FakeSituationAwarenessEngine(payload=_sample_payload(answer_mode=SituationalAwarenessAnswerMode.CHOICE))
    screen = _build_screen(engine)
    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        screen.render(surface)

        screen.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_2, "unicode": "2"}))

        assert engine.submissions == ["2"]
    finally:
        pygame.quit()

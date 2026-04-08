from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from importlib.machinery import ModuleSpec
from types import ModuleType

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

if "moderngl" not in sys.modules:
    moderngl_stub = ModuleType("moderngl")
    moderngl_stub.__spec__ = ModuleSpec("moderngl", loader=None)
    sys.modules["moderngl"] = moderngl_stub

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
    SituationalAwarenessWaypoint,
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
        SituationalAwarenessAnswerChoice(1, "Escort and hold"),
        SituationalAwarenessAnswerChoice(2, "Shadow only"),
        SituationalAwarenessAnswerChoice(3, "Break south"),
        SituationalAwarenessAnswerChoice(4, "Intercept now"),
        SituationalAwarenessAnswerChoice(5, "Hold position"),
    )
    active_query = SituationalAwarenessActiveQuery(
        query_id=101,
        kind=(
            SituationalAwarenessQueryKind.RULE_ACTION
            if answer_mode is SituationalAwarenessAnswerMode.CHOICE
            else SituationalAwarenessQueryKind.ACTUAL_DESTINATION
        ),
        answer_mode=answer_mode,
        prompt=(
            "Where is LEEDS actually heading?"
            if answer_mode is SituationalAwarenessAnswerMode.GRID_CELL
            else "Based on the chatter and rule call, what should LEEDS do?"
        ),
        correct_answer_token="E5" if answer_mode is SituationalAwarenessAnswerMode.GRID_CELL else "2",
        expires_in_s=9.0,
        subject_callsign="LEEDS",
        future_offset_s=None,
        answer_choices=action_choices if answer_mode is SituationalAwarenessAnswerMode.CHOICE else (),
    )
    return SituationalAwarenessPayload(
        scenario_family=SituationalAwarenessScenarioFamily.MERGE_CONFLICT,
        scenario_label="Tactical Merge 1/1",
        scenario_index=1,
        scenario_total=1,
        active_channels=("pictorial", "coded", "numerical", "aural"),
        active_query_kinds=("current_location", "actual_destination", "rule_action"),
        focus_label="Full mixed picture",
        segment_label="Picture Anchor",
        segment_index=1,
        segment_total=1,
        segment_time_remaining_s=42.0,
        scenario_elapsed_s=18.0,
        scenario_time_remaining_s=42.0,
        next_query_in_s=None,
        visible_contacts=(
            SituationalAwarenessVisibleContact("LEEDS", "Leeds", "friendly", "helicopter", 2.0, 4.0, "E2", "E", "E5", 0.95),
            SituationalAwarenessVisibleContact("RAVEN TWO", "Raven Two", "enemy", "tank", 6.0, 4.0, "E6", "W", "D4", 0.65),
        ),
        cue_card=SituationalAwarenessCueCard(
            "LEEDS",
            "Leeds",
            "friendly",
            "helicopter",
            "C7",
            "E5",
            "RAVEN TWO at D4 | Variation 3 - shadow only",
            "Shadow only",
            "CH 3",
            0.82,
        ),
        waypoints=(
            SituationalAwarenessWaypoint("C7", 7, 2),
            SituationalAwarenessWaypoint("E5", 5, 4),
        ),
        top_strip_text="LEEDS instructed C7, actually heading E5.",
        top_strip_fade=0.72,
        display_clock_text="11:00:16",
        active_query=active_query,
        answer_mode=answer_mode,
        correct_answer_token=active_query.correct_answer_token,
        announcement_token=("sa", "radio", 1),
        announcement_lines=("RAVEN TWO is enemy tank, trending D4.",),
        radio_log=(
            "LEEDS is friendly helicopter, check in channel 3.",
            "RAVEN TWO is enemy tank, trending D4.",
        ),
        speech_prefetch_lines=(
            "LEEDS is friendly helicopter, check in channel 3.",
            "RAVEN TWO is enemy tank, trending D4.",
        ),
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

        assert len(screen._sa_option_hitboxes) == 5
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

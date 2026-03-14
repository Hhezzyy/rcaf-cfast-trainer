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
    SituationalAwarenessPayload,
    SituationalAwarenessQueryKind,
    SituationalAwarenessScenarioFamily,
    SituationalAwarenessStatusEntry,
    SituationalAwarenessTrack,
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


def _sample_payload(
    *,
    answer_mode: SituationalAwarenessAnswerMode,
) -> SituationalAwarenessPayload:
    action_choices = (
        SituationalAwarenessAnswerChoice(1, "Keep lead direct and vector #2 away."),
        SituationalAwarenessAnswerChoice(2, "Hold lead and keep #2 direct."),
        SituationalAwarenessAnswerChoice(3, "Delay both tracks one sweep."),
        SituationalAwarenessAnswerChoice(4, "Swap both to standby channel."),
    )
    active_query = SituationalAwarenessActiveQuery(
        query_id=101,
        kind=(
            SituationalAwarenessQueryKind.ACTION_SELECTION
            if answer_mode is SituationalAwarenessAnswerMode.ACTION
            else (
                SituationalAwarenessQueryKind.FUTURE_POSITION
                if answer_mode is SituationalAwarenessAnswerMode.GRID_CELL
                else SituationalAwarenessQueryKind.CODE_OR_STATUS_RECALL
            )
        ),
        answer_mode=answer_mode,
        prompt=(
            "Where will track 2 be in 20s?"
            if answer_mode is SituationalAwarenessAnswerMode.GRID_CELL
            else (
                "Which indexed track is squawking 4123 at FL240 on CH 2?"
                if answer_mode is SituationalAwarenessAnswerMode.TRACK_INDEX
                else "Best immediate action near E5?"
            )
        ),
        correct_answer_token=(
            "E5"
            if answer_mode is SituationalAwarenessAnswerMode.GRID_CELL
            else ("2" if answer_mode is SituationalAwarenessAnswerMode.TRACK_INDEX else "1")
        ),
        expires_in_s=9.0,
        subject_callsign="R2",
        future_offset_s=20 if answer_mode is SituationalAwarenessAnswerMode.GRID_CELL else None,
        answer_choices=action_choices if answer_mode is SituationalAwarenessAnswerMode.ACTION else (),
    )
    return SituationalAwarenessPayload(
        scenario_family=SituationalAwarenessScenarioFamily.MERGE_CONFLICT,
        scenario_label="Merge Conflict 1/1",
        scenario_index=1,
        scenario_total=1,
        active_channels=("pictorial", "coded", "numerical", "aural"),
        active_query_kinds=("future_position", "contact_identification"),
        focus_label="Picture tracking",
        segment_label="Picture Anchor",
        segment_index=1,
        segment_total=1,
        segment_time_remaining_s=42.0,
        scenario_elapsed_s=18.0,
        scenario_time_remaining_s=42.0,
        next_query_in_s=None,
        tracks=(
            SituationalAwarenessTrack(1, "R1", 2.0, 4.0, "E2", "E", 5, 4011, 1, 220, "NORMAL", "ECHO"),
            SituationalAwarenessTrack(2, "R2", 6.0, 4.0, "E6", "W", 5, 4123, 2, 240, "LOW", "DELTA"),
            SituationalAwarenessTrack(3, "R3", 4.0, 1.0, "B4", "S", 3, 4334, 3, 300, "NORMAL", "CHARLIE"),
            SituationalAwarenessTrack(4, "T4", 8.0, 8.0, "I8", "NW", 2, 4555, 4, 180, "NORMAL", "ALFA"),
        ),
        status_entries=(
            SituationalAwarenessStatusEntry(1, "R1", "E2", "E", 5, 4011, 1, 220, "NORMAL", "ECHO"),
            SituationalAwarenessStatusEntry(2, "R2", "E6", "W", 5, 4123, 2, 240, "LOW", "DELTA"),
            SituationalAwarenessStatusEntry(3, "R3", "B4", "S", 3, 4334, 3, 300, "NORMAL", "CHARLIE"),
            SituationalAwarenessStatusEntry(4, "T4", "I8", "NW", 2, 4555, 4, 180, "NORMAL", "ALFA"),
        ),
        waypoints=(
            SituationalAwarenessWaypoint("ALFA", 1, 1),
            SituationalAwarenessWaypoint("BRAVO", 8, 1),
            SituationalAwarenessWaypoint("CHARLIE", 8, 8),
            SituationalAwarenessWaypoint("DELTA", 1, 8),
            SituationalAwarenessWaypoint("ECHO", 5, 2),
        ),
        recent_feed_lines=(
            "T+08 R2 switch channel 2 and continue DELTA.",
            "T+16 R1 maintain heading east.",
        ),
        active_query=active_query,
        answer_mode=answer_mode,
        correct_answer_token=active_query.correct_answer_token,
        announcement_token=("sa", 1),
        announcement_lines=("Traffic merge update.", active_query.prompt),
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


def test_situational_awareness_renderer_uses_grid_status_and_query_layout() -> None:
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


def test_situational_awareness_track_row_click_submits_index() -> None:
    engine = _FakeSituationAwarenessEngine(payload=_sample_payload(answer_mode=SituationalAwarenessAnswerMode.TRACK_INDEX))
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


def test_situational_awareness_action_cards_submit_numeric_choice() -> None:
    engine = _FakeSituationAwarenessEngine(payload=_sample_payload(answer_mode=SituationalAwarenessAnswerMode.ACTION))
    screen = _build_screen(engine)
    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        screen.render(surface)

        assert len(screen._sa_option_hitboxes) == 4
        rect = screen._sa_option_hitboxes[3]
        screen.handle_event(
            pygame.event.Event(
                pygame.MOUSEBUTTONDOWN,
                {"button": 1, "pos": rect.center},
            )
        )

        assert engine.submissions == ["3"]
    finally:
        pygame.quit()


def test_situational_awareness_title_prefix_routes_to_real_renderer() -> None:
    engine = _FakeSituationAwarenessEngine(
        payload=_sample_payload(answer_mode=SituationalAwarenessAnswerMode.TRACK_INDEX),
        title="Situational Awareness: Tempo",
    )
    screen = _build_screen(engine)
    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        screen.render(surface)
        assert len(screen._sa_option_hitboxes) == 4
    finally:
        pygame.quit()

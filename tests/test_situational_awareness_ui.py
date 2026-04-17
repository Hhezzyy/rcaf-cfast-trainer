from __future__ import annotations

import os
import sys
from dataclasses import dataclass, replace
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


class _RecordingFont:
    def __init__(self, base: pygame.font.Font, sink: list[str]) -> None:
        self._base = base
        self._sink = sink

    def render(self, text: str, antialias: bool, color: object) -> pygame.Surface:
        self._sink.append(str(text))
        return self._base.render(text, antialias, color)

    def __getattr__(self, name: str) -> object:
        return getattr(self._base, name)


def _sample_payload(*, answer_mode: SituationalAwarenessAnswerMode) -> SituationalAwarenessPayload:
    action_choices = (
        SituationalAwarenessAnswerChoice(1, "Escort and hold"),
        SituationalAwarenessAnswerChoice(2, "Shadow only"),
        SituationalAwarenessAnswerChoice(3, "Break south"),
        SituationalAwarenessAnswerChoice(4, "Intercept now"),
        SituationalAwarenessAnswerChoice(5, "Hold position"),
    )
    if answer_mode is SituationalAwarenessAnswerMode.GRID_CELL:
        kind = SituationalAwarenessQueryKind.ACTUAL_DESTINATION
        prompt = "Where is LEEDS actually heading?"
        correct_answer = "E5"
        answer_choices = ()
        accepted_tokens = ()
        entry_label = ""
        entry_placeholder = ""
        entry_max_chars = 0
    elif answer_mode is SituationalAwarenessAnswerMode.CHOICE:
        kind = SituationalAwarenessQueryKind.RULE_ACTION
        prompt = "Based on the chatter and rule call, what should LEEDS do?"
        correct_answer = "2"
        answer_choices = action_choices
        accepted_tokens = ()
        entry_label = ""
        entry_placeholder = ""
        entry_max_chars = 0
    elif answer_mode is SituationalAwarenessAnswerMode.NUMERIC:
        kind = SituationalAwarenessQueryKind.ALTITUDE
        prompt = "What altitude is LEEDS at?"
        correct_answer = "180"
        answer_choices = ()
        accepted_tokens = ()
        entry_label = "Altitude"
        entry_placeholder = "180"
        entry_max_chars = 3
    else:
        kind = SituationalAwarenessQueryKind.CURRENT_ALLEGIANCE
        prompt = "Is LEEDS friendly, hostile, or unknown?"
        correct_answer = "FRIENDLY"
        answer_choices = ()
        accepted_tokens = ("FRIENDLY", "HOSTILE", "UNKNOWN")
        entry_label = "Affiliation"
        entry_placeholder = "yellow / red / white"
        entry_max_chars = 10
    active_query = SituationalAwarenessActiveQuery(
        query_id=101,
        kind=kind,
        answer_mode=answer_mode,
        prompt=prompt,
        correct_answer_token=correct_answer,
        expires_in_s=9.0,
        subject_callsign="LEEDS",
        future_offset_s=None,
        answer_choices=answer_choices,
        accepted_tokens=accepted_tokens,
        entry_label=entry_label,
        entry_placeholder=entry_placeholder,
        entry_max_chars=entry_max_chars,
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
            next_waypoint="E5",
            next_waypoint_at_text="11:00:46",
            altitude_text="FL180",
            communications_text="RAVEN TWO at D4 | Variation 3 - shadow only",
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
        round_index=2,
        round_total=3,
        north_heading_deg=90,
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


def _install_recording_fonts(*fonts: object) -> list[str]:
    captured: list[str] = []
    for obj in fonts:
        for attr in ("_small_font", "_tiny_font", "_mid_font", "_big_font"):
            font = getattr(obj, attr, None)
            if isinstance(font, _RecordingFont) or font is None:
                continue
            setattr(obj, attr, _RecordingFont(font, captured))
    return captured


def test_situational_awareness_renderer_uses_sparse_grid_and_query_layout() -> None:
    engine = _FakeSituationAwarenessEngine(payload=_sample_payload(answer_mode=SituationalAwarenessAnswerMode.GRID_CELL))
    screen = _build_screen(engine)
    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        captured = _install_recording_fonts(screen)
        screen.render(surface)

        assert len(screen._sa_grid_hitboxes) == 100
        assert screen._sa_option_hitboxes == {}
        assert "INCOMING INFORMATION" in captured
        assert "ASSET INFORMATION" in captured
        assert "yellow = friendly   red = hostile   white = unknown   cell = 2 km" in captured
        assert "NEXT WAYPOINT" in captured
        assert "NEXT WAYPOINT AT" in captured
        assert "ALTITUDE" in captured
        assert "COMMUNICATIONS" in captured
        assert "Round 2/3" in captured
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


def test_situational_awareness_keyboard_numeric_submission_is_immediate() -> None:
    engine = _FakeSituationAwarenessEngine(payload=_sample_payload(answer_mode=SituationalAwarenessAnswerMode.NUMERIC))
    screen = _build_screen(engine)
    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        screen.render(surface)

        for key, char in (
            (pygame.K_1, "1"),
            (pygame.K_8, "8"),
            (pygame.K_0, "0"),
        ):
            screen.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": key, "unicode": char}))
        screen.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""}))

        assert engine.submissions == ["180"]
    finally:
        pygame.quit()


def test_situational_awareness_keyboard_token_submission_is_immediate() -> None:
    engine = _FakeSituationAwarenessEngine(payload=_sample_payload(answer_mode=SituationalAwarenessAnswerMode.TOKEN))
    screen = _build_screen(engine)
    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        screen.render(surface)

        for key, char in (
            (pygame.K_y, "y"),
            (pygame.K_e, "e"),
            (pygame.K_l, "l"),
            (pygame.K_l, "l"),
            (pygame.K_o, "o"),
            (pygame.K_w, "w"),
        ):
            screen.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": key, "unicode": char}))
        screen.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""}))

        assert engine.submissions == ["YELLOW"]
    finally:
        pygame.quit()


def test_situational_awareness_renderer_updates_focused_panel_callsign() -> None:
    engine = _FakeSituationAwarenessEngine(payload=_sample_payload(answer_mode=SituationalAwarenessAnswerMode.GRID_CELL))
    screen = _build_screen(engine)
    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        captured = _install_recording_fonts(screen)
        screen.render(surface)

        assert "LEEDS" in captured
        assert "RAVEN TWO" not in captured

        assert engine.payload.cue_card is not None
        engine.payload = replace(
            engine.payload,
            cue_card=replace(
                engine.payload.cue_card,
                callsign="RAVEN TWO",
                spoken_callsign="Raven Two",
                allegiance="enemy",
                asset_type="tank",
                next_waypoint="D4",
                next_waypoint_at_text="11:01:02",
                altitude_text="FL220",
                communications_text="CH 4 | Break south",
            ),
        )
        captured.clear()
        screen.render(surface)

        assert "RAVEN TWO" in captured
    finally:
        pygame.quit()

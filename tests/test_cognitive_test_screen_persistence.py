from __future__ import annotations

import os
import sqlite3

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from cfast_trainer.app import App, CognitiveTestScreen, MenuItem, MenuScreen
from cfast_trainer.cognitive_core import AttemptSummary, Phase
from cfast_trainer.cognitive_core import TestSnapshot as SnapshotModel
from cfast_trainer.persistence import ResultsStore


class _ResultsFakeEngine:
    seed = 123
    difficulty = 0.5
    practice_questions = 1
    scored_duration_s = 60.0

    def snapshot(self) -> SnapshotModel:
        return SnapshotModel(
            title="Fake Results",
            phase=Phase.RESULTS,
            prompt="Results\nAttempted: 3\nCorrect: 2",
            input_hint="",
            time_remaining_s=None,
            attempted_scored=3,
            correct_scored=2,
            payload=None,
        )

    def scored_summary(self) -> AttemptSummary:
        return AttemptSummary(
            attempted=3,
            correct=2,
            accuracy=2.0 / 3.0,
            duration_s=60.0,
            throughput_per_min=3.0,
            mean_response_time_s=0.5,
            total_score=2.0,
            max_score=3.0,
            score_ratio=2.0 / 3.0,
        )

    def events(self) -> list[object]:
        return []

    def can_exit(self) -> bool:
        return True

    def start_practice(self) -> None:
        return

    def start_scored(self) -> None:
        return

    def submit_answer(self, raw: str) -> bool:
        _ = raw
        return False

    def update(self) -> None:
        return


def test_results_screen_persists_once_and_surfaces_save_summary(tmp_path) -> None:
    pygame.init()
    try:
        surface = pygame.display.set_mode((960, 540))
        font = pygame.font.Font(None, 36)
        store = ResultsStore(tmp_path / "results.sqlite3")
        app = App(
            surface=surface,
            font=font,
            results_store=store,
            app_version="test",
        )
        app.push(MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True))

        engine = _ResultsFakeEngine()
        screen = CognitiveTestScreen(
            app,
            engine_factory=lambda: engine,
            test_code="fake_results",
        )
        app.push(screen)

        screen.render(surface)
        screen.render(surface)

        with sqlite3.connect(store.path) as conn:
            attempt_count = conn.execute("SELECT COUNT(*) FROM attempt").fetchone()
            activity_count = conn.execute("SELECT COUNT(*) FROM activity_session").fetchone()
            telemetry_kinds = [
                row[0] for row in conn.execute("SELECT kind FROM telemetry_event ORDER BY seq").fetchall()
            ]

        assert attempt_count == (1,)
        assert activity_count == (1,)
        assert telemetry_kinds == ["activity_started", "activity_completed"]
        assert any("Saved locally." in line for line in screen._results_persistence_lines)
        session = store.session_summary()
        assert session is not None
        assert session.activity_count == 1
        assert session.completed_activity_count == 1
        assert session.attempt_count == 1
    finally:
        pygame.quit()

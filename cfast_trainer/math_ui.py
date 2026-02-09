"""Pygame UI adapter for the Numerical Operations (Mathematics Reasoning) test.

This module provides a screen class that presents the mathematics reasoning test
to the user.  It handles the sequence of instruction reading, a brief
practice session, the timed test itself and a final results display.  All
deterministic logic (problem generation, timing, scoring) lives in
``math_core``; this UI layer is responsible only for rendering and capturing
input.  During the timed phase the screen displays the current problem, the
user's typed response and the remaining time.  When the allotted time
expires the user is shown a summary of their performance.

The screen makes no assumptions about persistence or networking and runs
entirely locally.  It must be integrated into the main menu by creating
an instance and pushing it onto the application's screen stack.
"""

from __future__ import annotations

import pygame
from dataclasses import dataclass
from typing import Optional

from .app import App, Screen
from .math_core import MathProblem, MathProblemGenerator, MathTestEngine


@dataclass
class _PracticeState:
    """Internal state for the practice phase.

    A small number of practice problems are presented prior to the timed
    assessment.  Each attempt records the user's response and provides
    immediate feedback.  Practice results are not logged or persisted.
    """

    generator: MathProblemGenerator
    remaining: int
    current_problem: Optional[MathProblem] = None
    user_input: str = ""
    feedback: Optional[str] = None


class MathTestScreen:
    """Screen implementing the mathematics reasoning test.

    The screen maintains a simple state machine with four modes:

    * ``instructions``: show test instructions and wait for the user to
      proceed.
    * ``practice``: present a handful of practice problems without timing.
    * ``test``: run the timed test using ``MathTestEngine``.
    * ``results``: display aggregated performance metrics.

    Transitions occur based on user input (Enter/Space) or when the timer
    expires.  Pressing Esc or Backspace returns to the previous menu from
    any screen except during the active test (to discourage accidental
    cancellation).  Once results are displayed, any key press will dismiss
    the screen.
    """

    def __init__(self, app: App) -> None:
        self._app = app
        self._state: str = "instructions"
        # Practice: three sample problems using a fresh generator
        self._practice = _PracticeState(
            generator=MathProblemGenerator(seed=None, difficulty=1),
            remaining=3,
        )
        # Timed test: two‑minute duration; seed defaults to None for variability
        self._engine = MathTestEngine(duration_seconds=120.0, seed=None, difficulty=1)
        self._user_input: str = ""
        self._last_key_down: Optional[int] = None

    # -- Event handling -----------------------------------------------------
    def handle_event(self, event: pygame.event.Event) -> None:
        # Global escape/back handling: in instructions/practice/results allow exit
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_ESCAPE, pygame.K_BACKSPACE):
                # In test state we ignore escapes to prevent cancelling
                if self._state in ("instructions", "practice", "results"):
                    self._app.pop()
                    return
            # Space and Enter/Return keys are considered the same for our purposes
            if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
                self._last_key_down = event.key
        elif event.type == pygame.KEYUP:
            # Only trigger actions on key release to avoid repeats
            if self._last_key_down is not None and event.key == self._last_key_down:
                self._on_action_key()
                self._last_key_down = None
                return

        # Handle typed character input when appropriate
        if self._state in ("practice", "test") and event.type == pygame.KEYDOWN:
            # Digits and minus sign
            if event.key == pygame.K_MINUS:
                if not self._user_input:
                    self._user_input = "-"
            elif event.unicode and event.unicode.isdigit():
                self._user_input += event.unicode
            elif event.key in (pygame.K_BACKSPACE, pygame.K_DELETE):
                # Remove last character
                self._user_input = self._user_input[:-1]

    def _on_action_key(self) -> None:
        """Respond to Enter/Space key releases based on the current state."""
        if self._state == "instructions":
            # Begin practice
            self._state = "practice"
            # Prime the first practice problem
            self._advance_practice()
        elif self._state == "practice":
            # Submit the current practice answer
            self._submit_practice_answer()
        elif self._state == "test":
            # Submit answer for current test problem
            self._submit_test_answer()
        elif self._state == "results":
            # Any key dismisses the results screen
            self._app.pop()

    # -- Practice phase helpers --------------------------------------------
    def _advance_practice(self) -> None:
        """Load the next practice problem if any remain."""
        if self._practice.remaining <= 0:
            # No more practice problems; transition to test
            self._start_test()
            return
        problem = self._practice.generator.next_problem()
        # Represent the problem index by id(id) value of dataclass but we need number
        # We'll store the answer in state for checking
        self._practice.current_problem = problem
        self._practice.feedback = None
        self._user_input = ""
        self._practice.remaining -= 1

    def _submit_practice_answer(self) -> None:
        """Check the user's answer for the current practice problem."""
        problem = getattr(self._practice, "current_problem", None)
        if problem is None:
            return
        # Attempt to parse the user's input; treat empty or invalid as incorrect
        try:
            answer = int(self._user_input)
        except Exception:
            answer = None  # type: ignore[assignment]
        correct = answer is not None and answer == problem.answer
        self._practice.feedback = "Correct!" if correct else f"Incorrect, answer is {problem.answer}"
        # After feedback, automatically advance to next practice problem on next action key
        # We'll treat any Enter/Space as simply clearing the feedback and loading next
        self._advance_practice()

    # -- Test phase helpers -------------------------------------------------
    def _start_test(self) -> None:
        """Begin the timed test and generate the first problem."""
        self._state = "test"
        self._engine.start()
        # Reset user input for first problem
        self._user_input = ""

    def _submit_test_answer(self) -> None:
        """Record the user's answer for the current test problem."""
        # Convert the user input to an integer if possible
        try:
            answer = int(self._user_input) if self._user_input else None
        except ValueError:
            answer = None
        self._engine.submit_answer(answer)
        # Clear input for the next problem
        self._user_input = ""
        if self._engine.finished:
            self._show_results()

    def _show_results(self) -> None:
        """Transition to results state."""
        self._state = "results"
        # Capture summary once to avoid recomputing each frame
        self._results_summary = self._engine.summary()

    # -- Rendering ----------------------------------------------------------
    def render(self, surface: pygame.Surface) -> None:
        # Basic background
        surface.fill((10, 10, 14))
        if self._state == "instructions":
            self._render_instructions(surface)
        elif self._state == "practice":
            self._render_practice(surface)
        elif self._state == "test":
            self._render_test(surface)
        elif self._state == "results":
            self._render_results(surface)

    def _render_instructions(self, surface: pygame.Surface) -> None:
        font = self._app.font
        lines = [
            "Numerical Operations Test",
            "",
            "You will be presented with simple arithmetic problems",
            "involving addition, subtraction, multiplication and division.",
            "Your goal is to answer as many problems correctly as possible",
            "within a 2‑minute time limit.",
            "",
            "Press Enter or Space to begin a short practice session.",
            "Press Esc to return to the previous menu.",
        ]
        y = 60
        for line in lines:
            text = font.render(line, True, (235, 235, 245))
            surface.blit(text, (40, y))
            y += 36

    def _render_practice(self, surface: pygame.Surface) -> None:
        font = self._app.font
        # Show current practice problem if available
        problem = getattr(self._practice, "current_problem", None)
        y = 60
        text = font.render("Practice Session", True, (235, 235, 245))
        surface.blit(text, (40, y))
        y += 48
        if problem is not None:
            display = font.render(problem.display + self._user_input, True, (235, 235, 245))
            surface.blit(display, (60, y))
            y += 42
            if self._practice.feedback:
                feedback = font.render(self._practice.feedback, True, (180, 220, 180) if "Correct" in self._practice.feedback else (220, 180, 180))
                surface.blit(feedback, (60, y))
        else:
            # Should not happen; fallback
            hint = font.render("Loading practice problem...", True, (180, 180, 190))
            surface.blit(hint, (60, y))

        # Prompt
        prompt = font.render("Press Enter to submit and continue", True, (140, 140, 150))
        surface.blit(prompt, (40, surface.get_height() - 60))

    def _render_test(self, surface: pygame.Surface) -> None:
        font = self._app.font
        # Show remaining time
        time_left = max(0.0, self._engine.time_remaining)
        minutes = int(time_left) // 60
        seconds = int(time_left) % 60
        time_str = f"Time: {minutes:02d}:{seconds:02d}"
        time_surf = font.render(time_str, True, (235, 235, 245))
        surface.blit(time_surf, (40, 40))

        # Show current problem and typed input
        problem = self._engine.current_problem
        if problem is not None:
            text = font.render("Problem", True, (235, 235, 245))
            surface.blit(text, (40, 120))
            problem_surf = font.render(problem.display + self._user_input, True, (235, 235, 245))
            surface.blit(problem_surf, (60, 160))
        else:
            # Should not happen during active test; fallback
            info = font.render("Test complete", True, (180, 180, 190))
            surface.blit(info, (40, 120))

        # Instruction for test
        hint = font.render("Type answer and press Enter", True, (140, 140, 150))
        surface.blit(hint, (40, surface.get_height() - 60))

        # Check if time expired outside of submit handler
        if self._engine.finished and self._state == "test":
            # Immediately show results
            self._show_results()

    def _render_results(self, surface: pygame.Surface) -> None:
        font = self._app.font
        lines = ["Test Complete", ""]
        # Build results lines
        summary = getattr(self, "_results_summary", {})
        lines.append(f"Problems attempted: {summary.get('attempted', 0)}")
        lines.append(f"Correct answers: {summary.get('correct', 0)}")
        # Format accuracy as percentage
        accuracy = summary.get('accuracy', 0.0)
        lines.append(f"Accuracy: {accuracy * 100:.0f}%")
        # Throughput per minute
        throughput = summary.get('throughput', 0.0)
        lines.append(f"Throughput: {throughput:.2f} problems/min")
        avg_time = summary.get('avg_response_time', 0.0)
        lines.append(f"Avg response time: {avg_time:.2f} s")
        lines.append("")
        lines.append("Press any key to return to the menu")
        y = 60
        for line in lines:
            text = font.render(line, True, (235, 235, 245))
            surface.blit(text, (40, y))
            y += 36

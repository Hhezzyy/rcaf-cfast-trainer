from __future__ import annotations

from dataclasses import dataclass

from .auditory_capacity import score_sequence_answer
from .cognitive_core import AnswerScorer, Problem


@dataclass(frozen=True, slots=True)
class LookupRetainPromptSpec:
    target_label: str
    target_digits: str
    steps: tuple[str, ...]

    @property
    def answer_digits(self) -> int:
        return max(1, len(self.target_digits))

    def render_prompt(self) -> str:
        lines = [f"Find the {self.target_label.lower()} and enter it."]
        lines.extend(step for step in self.steps if str(step).strip())
        lines.append("Enter the original value using exact digits.")
        return "\n".join(lines)


def expected_digits_for_problem(problem: Problem) -> str:
    payload = problem.payload
    answer_digits = getattr(payload, "answer_digits", None)
    if answer_digits is None:
        answer_digits = max(1, len(str(abs(int(problem.answer)))))
    answer_digits = max(1, int(answer_digits))
    return f"{int(problem.answer):0{answer_digits}d}"


class LookupRetainScorer(AnswerScorer):
    def score(self, *, problem: Problem, user_answer: int, raw: str) -> float:
        _ = user_answer
        return float(score_sequence_answer(expected_digits_for_problem(problem), raw))

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .clock import Clock
from .cognitive_core import (
    AnswerScorer,
    Problem,
    SeededRng,
    TimedTextInputTest,
    clamp01,
)


@dataclass(frozen=True, slots=True)
class VisualSearchConfig:
    # Guide describes this as a short test; keep timed block compact by default.
    scored_duration_s: float = 4.0 * 60.0
    practice_questions: int = 4


class VisualSearchTaskKind(StrEnum):
    ALPHANUMERIC = "alphanumeric"
    SYMBOL_CODE = "symbol_code"
    WARNING_SIGN = "warning_sign"
    COLOR_PATTERN = "color_pattern"


@dataclass(frozen=True, slots=True)
class VisualSearchPayload:
    kind: VisualSearchTaskKind
    rows: int
    cols: int
    target: str
    cells: tuple[str, ...]  # row-major order
    cell_codes: tuple[int, ...]  # row-major numeric labels shown in each block
    full_credit_error: int
    zero_credit_error: int


class VisualSearchScorer(AnswerScorer):
    """Exact count is full credit; near misses receive partial estimation credit."""

    def score(self, *, problem: Problem, user_answer: int, raw: str) -> float:
        _ = raw
        payload = problem.payload
        if not isinstance(payload, VisualSearchPayload):
            return 1.0 if int(user_answer) == int(problem.answer) else 0.0

        full = max(0, int(payload.full_credit_error))
        zero = max(full + 1, int(payload.zero_credit_error))
        err = abs(int(user_answer) - int(problem.answer))

        if err <= full:
            return 1.0
        if err >= zero:
            return 0.0
        return clamp01((zero - err) / float(zero - full))


class VisualSearchGenerator:
    """Deterministic generator for mixed visual scan/search counting trials."""

    _ALPHANUMERIC_TOKENS = ("A", "B", "E", "F", "G", "H", "K", "L", "M", "P", "R", "S")
    _SYMBOL_TOKENS = (
        "X_MARK",
        "DOUBLE_CROSS",
        "S_BEND",
        "RING_SPOKE",
        "L_HOOK",
        "BOX",
        "STAR",
        "TRIANGLE",
        "LOLLIPOP",
        "PIN",
        "BOLT",
        "FORK",
    )
    _WARNING_TOKENS = ("FUEL", "OXY", "HYD", "ELEC", "ICE", "ENG", "FIRE", "WARN", "RAD", "NAV")
    _COLOR_PATTERN_TOKENS = ("RG", "GR", "RB", "BR", "GY", "YG", "BY", "YB", "RW", "WR", "BW", "WB")

    def __init__(self, *, seed: int) -> None:
        self._rng = SeededRng(seed)
        # Visual Search guide scope: letters and line figures only.
        self._kinds = (
            VisualSearchTaskKind.ALPHANUMERIC,
            VisualSearchTaskKind.SYMBOL_CODE,
        )
        self._last_kind: VisualSearchTaskKind | None = None

    def next_problem(self, *, difficulty: float) -> Problem:
        _ = clamp01(difficulty)
        kind = self._pick_kind()

        rows = 3
        cols = 4
        cell_count = rows * cols

        bank = self._token_bank(kind)
        target = str(self._rng.choice(bank))
        distractors = tuple(tok for tok in bank if tok != target)

        cells = [str(self._rng.choice(distractors)) for _ in range(cell_count)]
        target_idx = int(self._rng.randint(0, cell_count - 1))
        cells[target_idx] = target

        # Visible per-cell numeric labels that the user types as the answer.
        code_pool = tuple(range(10, 10 + cell_count))
        cell_codes = tuple(int(v) for v in self._rng.sample(code_pool, k=cell_count))
        correct_code = int(cell_codes[target_idx])

        full_credit_error = 0
        zero_credit_error = 1

        payload = VisualSearchPayload(
            kind=kind,
            rows=rows,
            cols=cols,
            target=target,
            cells=tuple(cells),
            cell_codes=cell_codes,
            full_credit_error=full_credit_error,
            zero_credit_error=zero_credit_error,
        )

        return Problem(
            prompt=self._prompt_for(kind=kind, target=target),
            answer=correct_code,
            payload=payload,
        )

    def _pick_kind(self) -> VisualSearchTaskKind:
        kind = self._rng.choice(self._kinds)
        if self._last_kind is not None and kind == self._last_kind and self._rng.random() < 0.65:
            alternatives = tuple(k for k in self._kinds if k != kind)
            kind = self._rng.choice(alternatives)
        self._last_kind = kind
        return kind

    def _token_bank(self, kind: VisualSearchTaskKind) -> tuple[str, ...]:
        if kind is VisualSearchTaskKind.ALPHANUMERIC:
            return self._ALPHANUMERIC_TOKENS
        if kind is VisualSearchTaskKind.SYMBOL_CODE:
            return self._SYMBOL_TOKENS
        if kind is VisualSearchTaskKind.WARNING_SIGN:
            return self._WARNING_TOKENS
        return self._COLOR_PATTERN_TOKENS

    def _prompt_for(self, *, kind: VisualSearchTaskKind, target: str) -> str:
        if kind is VisualSearchTaskKind.ALPHANUMERIC:
            prefix = "Letters"
        else:
            prefix = "Line figures"
        return f"{prefix}: find the matching tile and enter its block number."


def build_visual_search_test(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    config: VisualSearchConfig | None = None,
) -> TimedTextInputTest:
    cfg = config or VisualSearchConfig()

    instructions = [
        "Visual Search Test",
        "",
        "Scan the numbered tiles and match the tile shown underneath the grid.",
        "Each board uses 12 numbered blocks labelled 10 to 21.",
        "Enter the number from the matching block.",
        "Targets alternate between letters and line-figure symbols.",
        "",
        "Controls:",
        "- Type the two-digit block number",
        "- Press Enter to submit",
        "",
        "You will get a short practice, then a timed scored block.",
    ]

    return TimedTextInputTest(
        title="Visual Search",
        instructions=instructions,
        generator=VisualSearchGenerator(seed=seed),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        practice_questions=cfg.practice_questions,
        scored_duration_s=cfg.scored_duration_s,
        scorer=VisualSearchScorer(),
    )

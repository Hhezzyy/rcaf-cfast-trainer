from __future__ import annotations

from dataclasses import dataclass, field
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
class VisualSearchProfile:
    allowed_kinds: tuple[VisualSearchTaskKind, ...] = field(
        default_factory=lambda: (
            VisualSearchTaskKind.ALPHANUMERIC,
            VisualSearchTaskKind.SYMBOL_CODE,
        )
    )
    similarity_floor: float = 0.05
    similarity_ceiling: float = 0.85
    family_switch_floor: float = 0.15
    family_switch_ceiling: float = 0.75
    preview_emphasis: bool = False


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
    class_count: int = 1
    active_classes: tuple[str, ...] = ()
    salience_level: float = 0.0
    switch_mode: str = ""
    priority_label: str = ""


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
    """Deterministic generator for mixed visual search trials."""

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
    _LETTER_CONFUSABLE_CLUSTERS = (
        ("E", "F", "H", "K", "L"),
        ("A", "M", "R"),
        ("B", "G", "P", "R", "S"),
    )
    _SYMBOL_CONFUSABLE_CLUSTERS = (
        ("X_MARK", "DOUBLE_CROSS", "STAR", "BOLT"),
        ("L_HOOK", "PIN", "FORK"),
        ("BOX", "TRIANGLE", "RING_SPOKE"),
        ("S_BEND", "LOLLIPOP"),
    )

    def __init__(self, *, seed: int, profile: VisualSearchProfile | None = None) -> None:
        self._rng = SeededRng(seed)
        self._profile = profile or VisualSearchProfile()
        self._kinds = self._normalize_allowed_kinds(self._profile.allowed_kinds)
        self._last_kind: VisualSearchTaskKind | None = None

    def next_problem(self, *, difficulty: float) -> Problem:
        normalized_difficulty = clamp01(difficulty)
        kind = self._pick_kind(difficulty=normalized_difficulty)
        rows, cols = self._grid_shape(difficulty=normalized_difficulty)
        cell_count = rows * cols

        bank = self._token_bank(kind)
        target = str(self._rng.choice(bank))
        distractors = self._build_distractor_pool(
            kind=kind,
            target=target,
            difficulty=normalized_difficulty,
            count=cell_count,
        )

        cells = [str(distractors[idx]) for idx in range(cell_count)]
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

    def _grid_shape(self, *, difficulty: float) -> tuple[int, int]:
        if difficulty < 0.25:
            return (2, 3)
        if difficulty < 0.60:
            return (3, 4)
        if difficulty < 0.85:
            return (4, 4)
        return (4, 5)

    def _pick_kind(self, *, difficulty: float) -> VisualSearchTaskKind:
        if len(self._kinds) == 1:
            kind = self._kinds[0]
            self._last_kind = kind
            return kind
        kind = self._rng.choice(self._kinds)
        switch_bias = self._lerp(
            self._profile.family_switch_floor,
            self._profile.family_switch_ceiling,
            clamp01(difficulty),
        )
        if self._last_kind is not None and kind == self._last_kind and self._rng.random() < switch_bias:
            alternatives = tuple(k for k in self._kinds if k != kind)
            kind = self._rng.choice(alternatives)
        self._last_kind = kind
        return kind

    def _build_distractor_pool(
        self,
        *,
        kind: VisualSearchTaskKind,
        target: str,
        difficulty: float,
        count: int,
    ) -> tuple[str, ...]:
        bank = self._token_bank(kind)
        confusable = self._confusable_members(kind=kind, target=target)
        broad = tuple(token for token in bank if token != target and token not in confusable)
        similarity = self._lerp(
            self._profile.similarity_floor,
            self._profile.similarity_ceiling,
            clamp01(difficulty),
        )
        similar_pool = confusable if confusable else tuple(token for token in bank if token != target)
        broad_pool = broad if broad else tuple(token for token in bank if token != target)
        picks: list[str] = []
        for _ in range(max(1, int(count))):
            use_similar = bool(similar_pool) and self._rng.random() < similarity
            pool = similar_pool if use_similar else broad_pool
            picks.append(str(self._rng.choice(pool)))
        return tuple(picks)

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
        if self._profile.preview_emphasis:
            return f"{prefix}: preview the target first, then scan the board and enter its block number."
        return f"{prefix}: find the matching tile and enter its block number."

    def _confusable_members(self, *, kind: VisualSearchTaskKind, target: str) -> tuple[str, ...]:
        if kind is VisualSearchTaskKind.ALPHANUMERIC:
            clusters = self._LETTER_CONFUSABLE_CLUSTERS
        elif kind is VisualSearchTaskKind.SYMBOL_CODE:
            clusters = self._SYMBOL_CONFUSABLE_CLUSTERS
        else:
            return ()
        seen: set[str] = set()
        ordered: list[str] = []
        for cluster in clusters:
            if target not in cluster:
                continue
            for token in cluster:
                if token == target or token in seen:
                    continue
                seen.add(token)
                ordered.append(token)
        return tuple(ordered)

    @staticmethod
    def _normalize_allowed_kinds(
        allowed_kinds: tuple[VisualSearchTaskKind, ...] | list[VisualSearchTaskKind],
    ) -> tuple[VisualSearchTaskKind, ...]:
        supported = {
            VisualSearchTaskKind.ALPHANUMERIC,
            VisualSearchTaskKind.SYMBOL_CODE,
        }
        cleaned: list[VisualSearchTaskKind] = []
        seen: set[VisualSearchTaskKind] = set()
        for kind in tuple(allowed_kinds):
            if kind not in supported or kind in seen:
                continue
            seen.add(kind)
            cleaned.append(kind)
        if cleaned:
            return tuple(cleaned)
        return (
            VisualSearchTaskKind.ALPHANUMERIC,
            VisualSearchTaskKind.SYMBOL_CODE,
        )

    @staticmethod
    def _lerp(start: float, end: float, amount: float) -> float:
        t = clamp01(amount)
        return float(start) + (float(end) - float(start)) * t


def build_visual_search_test(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    config: VisualSearchConfig | None = None,
    profile: VisualSearchProfile | None = None,
) -> TimedTextInputTest:
    cfg = config or VisualSearchConfig()

    instructions = [
        "Visual Search Test",
        "",
        "Scan the numbered tiles and match the tile shown underneath the grid.",
        "Each board uses 12 numbered blocks labelled 10 to 21.",
        "Enter the number from the matching block.",
        "Targets use letters and line-figure symbols from the same live test family.",
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
        generator=VisualSearchGenerator(seed=seed, profile=profile),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        practice_questions=cfg.practice_questions,
        scored_duration_s=cfg.scored_duration_s,
        scorer=VisualSearchScorer(),
    )

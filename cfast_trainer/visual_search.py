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


_OFFICIAL_GRID_BY_LEVEL: tuple[tuple[int, int], ...] = (
    (3, 4),
    (3, 4),
    (4, 4),
    (4, 4),
    (4, 5),
    (4, 5),
    (5, 5),
    (5, 6),
    (6, 6),
    (7, 6),
)
_VISUAL_SEARCH_VARIANT_MARKS: tuple[str, ...] = (
    "T",
    "TR",
    "R",
    "BR",
    "B",
    "BL",
    "L",
    "TL",
    "C",
    "DOT",
)


def _difficulty_to_level(difficulty: float) -> int:
    return max(1, min(10, int(round(clamp01(difficulty) * 9.0)) + 1))


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
    grid_by_level: tuple[tuple[int, int], ...] = _OFFICIAL_GRID_BY_LEVEL
    high_band_unique_level: int = 8
    same_base_overload_level: int = 9
    high_band_symbol_only: bool = False
    variant_marks: tuple[str, ...] = _VISUAL_SEARCH_VARIANT_MARKS


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
    input_digits: int = 2
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
    _ALPHANUMERIC_STRING_BY_LEVEL = {
        9: 3,
        10: 4,
    }

    def __init__(self, *, seed: int, profile: VisualSearchProfile | None = None) -> None:
        self._rng = SeededRng(seed)
        self._profile = profile or VisualSearchProfile()
        self._kinds = self._normalize_allowed_kinds(self._profile.allowed_kinds)
        self._last_kind: VisualSearchTaskKind | None = None

    def next_problem(self, *, difficulty: float) -> Problem:
        normalized_difficulty = clamp01(difficulty)
        level = _difficulty_to_level(normalized_difficulty)
        kind = self._pick_kind(difficulty=normalized_difficulty, level=level)
        rows, cols = self._grid_shape(level=level)
        cell_count = rows * cols

        bank = self._token_bank(kind)
        string_length = self._string_length_for(kind=kind, level=level)
        target_base = str(self._rng.choice(bank))
        same_base_level = max(1, int(self._profile.same_base_overload_level))
        if string_length > 1:
            target, cells = self._build_high_difficulty_string_board(
                count=cell_count,
                length=string_length,
                level=level,
            )
            target_idx = cells.index(target)
        elif level >= same_base_level:
            target, cells = self._build_same_base_overload_board(
                kind=kind,
                target_base=target_base,
                count=cell_count,
                level=level,
            )
            target_idx = cells.index(target)
        elif level >= max(1, int(self._profile.high_band_unique_level)):
            target, cells = self._build_high_band_unique_board(
                kind=kind,
                target_base=target_base,
                count=cell_count,
            )
            target_idx = cells.index(target)
        else:
            target = target_base
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
        cell_codes = self._build_cell_codes(count=cell_count)
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

    def _grid_shape(self, *, level: int) -> tuple[int, int]:
        shapes = self._profile.grid_by_level or _OFFICIAL_GRID_BY_LEVEL
        index = max(0, min(len(shapes) - 1, int(level) - 1))
        rows, cols = shapes[index]
        return (max(1, int(rows)), max(1, int(cols)))

    def _string_length_for(self, *, kind: VisualSearchTaskKind, level: int) -> int:
        if kind is not VisualSearchTaskKind.ALPHANUMERIC:
            return 0
        return int(self._ALPHANUMERIC_STRING_BY_LEVEL.get(int(level), 0))

    def _build_cell_codes(self, *, count: int, rng: SeededRng | None = None) -> tuple[int, ...]:
        active_rng = self._rng if rng is None else rng
        needed = max(1, int(count))
        code_pool = tuple(range(10, 100))
        return tuple(int(v) for v in active_rng.sample(code_pool, k=needed))

    def _pick_kind(self, *, difficulty: float, level: int) -> VisualSearchTaskKind:
        if (
            level >= max(1, int(self._profile.same_base_overload_level))
            and self._profile.high_band_symbol_only
            and VisualSearchTaskKind.SYMBOL_CODE in self._kinds
        ):
            self._last_kind = VisualSearchTaskKind.SYMBOL_CODE
            return VisualSearchTaskKind.SYMBOL_CODE
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

    def _build_high_band_unique_board(
        self,
        *,
        kind: VisualSearchTaskKind,
        target_base: str,
        count: int,
    ) -> tuple[str, list[str]]:
        marks = self._variant_marks()
        bases = list(self._high_band_base_pool(kind=kind, target_base=target_base, count=count))
        while len(bases) * len(marks) < max(1, int(count)):
            for token in self._token_bank(kind):
                token = str(token)
                if token in bases:
                    continue
                bases.append(token)
                if len(bases) * len(marks) >= max(1, int(count)):
                    break

        target_mark = str(self._rng.choice(marks))
        target = self._compose_variant_token(target_base, target_mark)
        pool = [
            self._compose_variant_token(base, mark)
            for base in bases
            for mark in marks
            if self._compose_variant_token(base, mark) != target
        ]
        selected = self._rng.sample(tuple(pool), k=max(0, int(count) - 1))
        cells = [str(token) for token in selected]
        cells.append(target)
        cells = [str(token) for token in self._rng.sample(tuple(cells), k=len(cells))]
        return target, cells

    def _build_same_base_overload_board(
        self,
        *,
        kind: VisualSearchTaskKind,
        target_base: str,
        count: int,
        level: int,
    ) -> tuple[str, list[str]]:
        _ = kind
        signatures = self._variant_signatures()
        level_signatures = [
            signature for signature in signatures if len(signature) == 1
        ] or list(signatures)
        if level >= 10:
            level_signatures = [signature for signature in signatures if len(signature) == 2] or list(signatures)

        target_signature = tuple(
            level_signatures[int(self._rng.randint(0, len(level_signatures) - 1))]
        )
        target = self._compose_variant_token(target_base, target_signature)
        distractor_signatures = self._sorted_same_base_signatures(
            target_signature=target_signature,
            prefer_single_first=level <= 9,
        )

        needed = max(0, int(count) - 1)
        if len(distractor_signatures) < needed:
            repeats = list(distractor_signatures)
            while len(repeats) < needed:
                repeats.extend(distractor_signatures)
            distractor_signatures = tuple(repeats[:needed])
        else:
            distractor_signatures = distractor_signatures[:needed]

        cells = [
            self._compose_variant_token(target_base, signature)
            for signature in distractor_signatures
        ]
        cells.append(target)
        cells = [str(token) for token in self._rng.sample(tuple(cells), k=len(cells))]
        return target, cells

    def _build_high_difficulty_string_board(
        self,
        *,
        count: int,
        length: int,
        level: int,
    ) -> tuple[str, list[str]]:
        char_pool = self._ALPHANUMERIC_TOKENS
        target = self._build_string_token(length=length)
        distractors: list[str] = []
        seen = {target}
        positions = tuple(range(max(1, int(length))))
        while len(distractors) < max(0, int(count) - 1):
            max_changes = 1 if level <= 9 and len(distractors) < max(1, int(count) // 2) else 2
            change_count = min(
                len(positions),
                1 if max_changes <= 1 or self._rng.random() < 0.72 else 2,
            )
            chosen_positions = tuple(self._rng.sample(positions, k=change_count))
            chars = list(target)
            for pos in chosen_positions:
                current = target[pos]
                replacement_pool = self._string_confusable_chars(current)
                if not replacement_pool:
                    replacement_pool = tuple(ch for ch in char_pool if ch != current)
                chars[pos] = str(self._rng.choice(replacement_pool))
            candidate = "".join(chars)
            if candidate in seen:
                fallback = self._build_string_token(length=length)
                if fallback in seen:
                    continue
                candidate = fallback
            seen.add(candidate)
            distractors.append(candidate)
        cells = list(distractors)
        cells.append(target)
        cells = [str(token) for token in self._rng.sample(tuple(cells), k=len(cells))]
        return target, cells

    def _build_string_token(self, *, length: int) -> str:
        length = max(1, int(length))
        chars = [str(self._rng.choice(self._ALPHANUMERIC_TOKENS)) for _ in range(length)]
        if length > 1 and len(set(chars)) == 1:
            replacement_pool = tuple(ch for ch in self._ALPHANUMERIC_TOKENS if ch != chars[-1])
            chars[-1] = str(self._rng.choice(replacement_pool))
        return "".join(chars)

    def _string_confusable_chars(self, token: str) -> tuple[str, ...]:
        base = str(token)
        for cluster in self._LETTER_CONFUSABLE_CLUSTERS:
            if base not in cluster:
                continue
            return tuple(ch for ch in cluster if ch != base)
        return tuple(ch for ch in self._ALPHANUMERIC_TOKENS if ch != base)

    def _high_band_base_pool(
        self,
        *,
        kind: VisualSearchTaskKind,
        target_base: str,
        count: int,
    ) -> tuple[str, ...]:
        marks = tuple(
            str(mark).upper()
            for mark in (self._profile.variant_marks or _VISUAL_SEARCH_VARIANT_MARKS)
            if str(mark).strip()
        ) or _VISUAL_SEARCH_VARIANT_MARKS
        bank = self._token_bank(kind)
        bases: list[str] = [str(target_base)]
        for token in self._confusable_members(kind=kind, target=target_base):
            token = str(token)
            if token not in bases:
                bases.append(token)
        if len(bases) * len(marks) >= max(1, int(count)):
            return tuple(bases)
        for token in bank:
            token = str(token)
            if token in bases:
                continue
            bases.append(token)
            if len(bases) * len(marks) >= max(1, int(count)):
                break
        return tuple(bases)

    @staticmethod
    def _compose_variant_token(base: str, mark: str | tuple[str, ...] | list[str]) -> str:
        if isinstance(mark, str):
            marks = tuple(part.strip().upper() for part in mark.split("+") if part.strip())
        else:
            marks = tuple(str(part).strip().upper() for part in tuple(mark) if str(part).strip())
        if not marks:
            return str(base)
        return f"{str(base)}@{'+'.join(marks)}"

    @staticmethod
    def token_base(token: str) -> str:
        raw = str(token)
        return raw.split("@", 1)[0]

    @staticmethod
    def token_mark(token: str) -> str:
        return "+".join(VisualSearchGenerator.token_marks(token))

    @staticmethod
    def token_marks(token: str) -> tuple[str, ...]:
        raw = str(token)
        if "@" not in raw:
            return ()
        _base, suffix = raw.split("@", 1)
        return tuple(part.strip().upper() for part in suffix.split("+") if part.strip())

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
        base_target = self.token_base(target)
        if kind is VisualSearchTaskKind.ALPHANUMERIC:
            clusters = self._LETTER_CONFUSABLE_CLUSTERS
        elif kind is VisualSearchTaskKind.SYMBOL_CODE:
            clusters = self._SYMBOL_CONFUSABLE_CLUSTERS
        else:
            return ()
        seen: set[str] = set()
        ordered: list[str] = []
        for cluster in clusters:
            if base_target not in cluster:
                continue
            for token in cluster:
                if token == base_target or token in seen:
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

    def _variant_marks(self) -> tuple[str, ...]:
        marks: list[str] = []
        seen: set[str] = set()
        for raw in self._profile.variant_marks or _VISUAL_SEARCH_VARIANT_MARKS:
            mark = str(raw).strip().upper()
            if not mark or mark in seen:
                continue
            seen.add(mark)
            marks.append(mark)
        return tuple(marks) or _VISUAL_SEARCH_VARIANT_MARKS

    def _variant_signatures(self) -> tuple[tuple[str, ...], ...]:
        marks = self._variant_marks()
        signatures: list[tuple[str, ...]] = [(mark,) for mark in marks]
        for first_index, first in enumerate(marks):
            for second in marks[first_index + 1 :]:
                signatures.append((first, second))
        return tuple(signatures)

    def _sorted_same_base_signatures(
        self,
        *,
        target_signature: tuple[str, ...],
        prefer_single_first: bool,
    ) -> tuple[tuple[str, ...], ...]:
        target_set = set(target_signature)
        mark_index = {mark: index for index, mark in enumerate(self._variant_marks())}

        def signature_key(signature: tuple[str, ...]) -> tuple[int, int, int, tuple[int, ...]]:
            signature_set = set(signature)
            distance = len(signature_set.symmetric_difference(target_set))
            single_penalty = 0
            if prefer_single_first:
                single_penalty = 0 if len(signature) == 1 else 1
            return (
                single_penalty,
                distance,
                len(signature),
                tuple(mark_index[mark] for mark in signature),
            )

        ordered = [signature for signature in self._variant_signatures() if signature != target_signature]
        ordered.sort(key=signature_key)
        return tuple(ordered)


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
        "Boards scale from the standard 3x4 layout up to denser late-level matrices.",
        "Enter the two-digit number from the matching block, then press Enter to submit.",
        "Targets use letters and line-figure symbols from the same live test family.",
        "",
        "Controls:",
        "- Type the two-digit block number, then press Enter",
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

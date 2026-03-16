from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .clock import Clock
from .content_variants import stable_variant_id
from .cognitive_core import AnswerScorer, Problem, SeededRng, TimedTextInputTest, clamp01, lerp_int


@dataclass(frozen=True, slots=True)
class TableReadingConfig:
    # Candidate guide indicates ~11 minutes including instructions.
    scored_duration_s: float = 9.0 * 60.0
    practice_questions: int = 2


class TableReadingPart(StrEnum):
    PART_ONE = "part_one_cross_reference"
    PART_TWO = "part_two_multi_table"


@dataclass(frozen=True, slots=True)
class TableReadingOption:
    code: int
    label: str
    value: int


@dataclass(frozen=True, slots=True)
class TableReadingTable:
    title: str
    row_header: str
    column_header: str
    row_labels: tuple[str, ...]
    column_labels: tuple[str, ...]
    values: tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class TableReadingPayload:
    part: TableReadingPart
    stem: str
    primary_table: TableReadingTable
    secondary_table: TableReadingTable | None
    primary_row_label: str
    primary_column_label: str
    secondary_row_label: str | None
    secondary_column_label: str | None
    options: tuple[TableReadingOption, ...]
    correct_code: int
    correct_value: int
    estimate_tolerance: int
    content_family: str = ""
    variant_id: str = ""
    content_pack: str = ""


@dataclass(frozen=True, slots=True)
class _PartOneCardPack:
    family: str
    table: TableReadingTable


@dataclass(frozen=True, slots=True)
class _PartTwoCardSet:
    family: str
    index_table: TableReadingTable
    correction_table: TableReadingTable


def _build_grid(
    row_count: int,
    col_count: int,
    *,
    base: int,
    row_gain: int,
    col_gain: int,
    row_curve: int,
    col_curve: int,
    cross: int,
) -> tuple[tuple[int, ...], ...]:
    rows: list[tuple[int, ...]] = []
    for row_idx in range(row_count):
        current: list[int] = []
        for col_idx in range(col_count):
            value = base
            value += row_idx * row_gain
            value += col_idx * col_gain
            value += (row_idx % 3) * row_curve
            value += (col_idx % 4) * col_curve
            value += ((row_idx + col_idx) % 3) * cross
            current.append(int(value))
        rows.append(tuple(current))
    return tuple(rows)


def _build_part_one_pack(
    *,
    family: str,
    title: str,
    row_header: str,
    column_header: str,
    row_prefix: str,
    row_count: int,
    column_prefix: str,
    column_count: int,
    base: int,
    row_gain: int,
    col_gain: int,
    row_curve: int,
    col_curve: int,
    cross: int,
) -> _PartOneCardPack:
    table = TableReadingTable(
        title=title,
        row_header=row_header,
        column_header=column_header,
        row_labels=tuple(f"{row_prefix}{idx + 1}" for idx in range(row_count)),
        column_labels=tuple(f"{column_prefix}{idx + 1}" for idx in range(column_count)),
        values=_build_grid(
            row_count,
            column_count,
            base=base,
            row_gain=row_gain,
            col_gain=col_gain,
            row_curve=row_curve,
            col_curve=col_curve,
            cross=cross,
        ),
    )
    return _PartOneCardPack(family=family, table=table)


def _build_part_two_index_table(
    *,
    title: str,
    row_header: str,
    column_header: str,
    row_labels: tuple[str, ...],
    column_labels: tuple[str, ...],
    base: int,
    row_divisor: int,
    col_divisor: int,
    bias: int,
) -> TableReadingTable:
    values: list[tuple[int, ...]] = []
    for row_idx, _row_label in enumerate(row_labels):
        row_values: list[int] = []
        for col_idx, _col_label in enumerate(column_labels):
            value = base
            value += row_idx // max(1, row_divisor)
            value += col_idx // max(1, col_divisor)
            value += ((row_idx + col_idx) % 2) * bias
            row_values.append(max(1, min(15, int(value))))
        values.append(tuple(row_values))
    return TableReadingTable(
        title=title,
        row_header=row_header,
        column_header=column_header,
        row_labels=row_labels,
        column_labels=column_labels,
        values=tuple(values),
    )


def _build_part_two_correction_table(
    *,
    title: str,
    row_header: str,
    column_header: str,
    row_labels: tuple[str, ...],
    column_labels: tuple[str, ...],
    base: int,
    row_gain: int,
    col_gain: int,
    cross: int,
) -> TableReadingTable:
    values: list[tuple[int, ...]] = []
    for row_idx, _row_label in enumerate(row_labels):
        row_values: list[int] = []
        for col_idx, _col_label in enumerate(column_labels):
            value = base + (row_idx * row_gain) + (col_idx * col_gain)
            value += ((row_idx + col_idx) % 3) * cross
            row_values.append(int(value))
        values.append(tuple(row_values))
    return TableReadingTable(
        title=title,
        row_header=row_header,
        column_header=column_header,
        row_labels=row_labels,
        column_labels=column_labels,
        values=tuple(values),
    )


def _build_part_two_set(
    *,
    family: str,
    index_title: str,
    correction_title: str,
    index_row_header: str,
    index_column_header: str,
    index_row_labels: tuple[str, ...],
    index_column_labels: tuple[str, ...],
    correction_column_header: str,
    correction_column_labels: tuple[str, ...],
    base: int,
    row_divisor: int,
    col_divisor: int,
    bias: int,
    correction_base: int,
    correction_row_gain: int,
    correction_col_gain: int,
    correction_cross: int,
) -> _PartTwoCardSet:
    correction_row_labels = tuple(str(idx) for idx in range(1, 16))
    return _PartTwoCardSet(
        family=family,
        index_table=_build_part_two_index_table(
            title=index_title,
            row_header=index_row_header,
            column_header=index_column_header,
            row_labels=index_row_labels,
            column_labels=index_column_labels,
            base=base,
            row_divisor=row_divisor,
            col_divisor=col_divisor,
            bias=bias,
        ),
        correction_table=_build_part_two_correction_table(
            title=correction_title,
            row_header="Drift Index",
            column_header=correction_column_header,
            row_labels=correction_row_labels,
            column_labels=correction_column_labels,
            base=correction_base,
            row_gain=correction_row_gain,
            col_gain=correction_col_gain,
            cross=correction_cross,
        ),
    )


_PART_ONE_PACKS: tuple[_PartOneCardPack, ...] = (
    _build_part_one_pack(
        family="lookup",
        title="Card A - Lookup Table",
        row_header="Row",
        column_header="Column",
        row_prefix="R",
        row_count=10,
        column_prefix="C",
        column_count=10,
        base=171,
        row_gain=18,
        col_gain=13,
        row_curve=3,
        col_curve=2,
        cross=4,
    ),
    _build_part_one_pack(
        family="station",
        title="Card D - Station Table",
        row_header="Station",
        column_header="Band",
        row_prefix="S",
        row_count=8,
        column_prefix="B",
        column_count=12,
        base=142,
        row_gain=24,
        col_gain=11,
        row_curve=4,
        col_curve=3,
        cross=5,
    ),
    _build_part_one_pack(
        family="sector",
        title="Card E - Sector Table",
        row_header="Sector",
        column_header="Gate",
        row_prefix="SX",
        row_count=12,
        column_prefix="G",
        column_count=8,
        base=196,
        row_gain=15,
        col_gain=17,
        row_curve=5,
        col_curve=2,
        cross=3,
    ),
    _build_part_one_pack(
        family="dispatch",
        title="Card J - Dispatch Table",
        row_header="Dispatch",
        column_header="Window",
        row_prefix="D",
        row_count=9,
        column_prefix="W",
        column_count=11,
        base=156,
        row_gain=19,
        col_gain=15,
        row_curve=4,
        col_curve=3,
        cross=6,
    ),
    _build_part_one_pack(
        family="range",
        title="Card K - Range Table",
        row_header="Range",
        column_header="Band",
        row_prefix="RG",
        row_count=11,
        column_prefix="B",
        column_count=9,
        base=188,
        row_gain=17,
        col_gain=14,
        row_curve=5,
        col_curve=4,
        cross=4,
    ),
)

_PART_TWO_CARD_SETS: tuple[_PartTwoCardSet, ...] = (
    _build_part_two_set(
        family="wind",
        index_title="Card B - Wind Index Table",
        correction_title="Card C - Angle Correction Table",
        index_row_header="Air Speed",
        index_column_header="Wind Velocity",
        index_row_labels=("80", "100", "120", "140", "160", "180", "200", "220"),
        index_column_labels=("4", "8", "12", "16", "20", "24", "28", "32"),
        correction_column_header="Wind Angle",
        correction_column_labels=("15", "25", "35", "45", "55", "65", "75", "85", "95"),
        base=1,
        row_divisor=1,
        col_divisor=2,
        bias=1,
        correction_base=1,
        correction_row_gain=1,
        correction_col_gain=1,
        correction_cross=0,
    ),
    _build_part_two_set(
        family="crosswind",
        index_title="Card F - Crosswind Drift Table",
        correction_title="Card G - Course Correction Table",
        index_row_header="Ground Speed",
        index_column_header="Crosswind",
        index_row_labels=("90", "110", "130", "150", "170", "190", "210", "230"),
        index_column_labels=("5", "10", "15", "20", "25", "30", "35"),
        correction_column_header="Track Angle",
        correction_column_labels=("10", "20", "30", "40", "50", "60", "70", "80"),
        base=2,
        row_divisor=2,
        col_divisor=1,
        bias=1,
        correction_base=1,
        correction_row_gain=1,
        correction_col_gain=1,
        correction_cross=1,
    ),
    _build_part_two_set(
        family="offset",
        index_title="Card H - Offset Index Table",
        correction_title="Card I - Lateral Correction Table",
        index_row_header="Track Speed",
        index_column_header="Offset Drift",
        index_row_labels=("100", "120", "140", "160", "180", "200", "220", "240"),
        index_column_labels=("6", "12", "18", "24", "30", "36"),
        correction_column_header="Correction Angle",
        correction_column_labels=("20", "30", "40", "50", "60", "70", "80", "90"),
        base=2,
        row_divisor=1,
        col_divisor=1,
        bias=0,
        correction_base=2,
        correction_row_gain=1,
        correction_col_gain=1,
        correction_cross=1,
    ),
    _build_part_two_set(
        family="descent",
        index_title="Card L - Descent Index Table",
        correction_title="Card M - Recovery Angle Table",
        index_row_header="Track Speed",
        index_column_header="Sink Rate",
        index_row_labels=("90", "110", "130", "150", "170", "190", "210", "230"),
        index_column_labels=("4", "8", "12", "16", "20", "24", "28"),
        correction_column_header="Recovery Arc",
        correction_column_labels=("15", "25", "35", "45", "55", "65", "75", "85"),
        base=1,
        row_divisor=2,
        col_divisor=1,
        bias=1,
        correction_base=2,
        correction_row_gain=1,
        correction_col_gain=1,
        correction_cross=1,
    ),
    _build_part_two_set(
        family="timing",
        index_title="Card N - Timing Index Table",
        correction_title="Card O - Timing Adjustment Table",
        index_row_header="Ground Speed",
        index_column_header="Timing Error",
        index_row_labels=("80", "100", "120", "140", "160", "180", "200", "220"),
        index_column_labels=("5", "10", "15", "20", "25", "30", "35", "40"),
        correction_column_header="Adjustment Angle",
        correction_column_labels=("10", "20", "30", "40", "50", "60", "70", "80", "90"),
        base=1,
        row_divisor=1,
        col_divisor=2,
        bias=1,
        correction_base=1,
        correction_row_gain=1,
        correction_col_gain=1,
        correction_cross=0,
    ),
)

_PART_ONE_PACK_BY_FAMILY = {pack.family: pack for pack in _PART_ONE_PACKS}
_PART_TWO_SET_BY_FAMILY = {card_set.family: card_set for card_set in _PART_TWO_CARD_SETS}
_PART_ONE_TITLE_TO_FAMILY = {pack.table.title: pack.family for pack in _PART_ONE_PACKS}
_PART_TWO_TITLES_TO_FAMILY = {
    (card_set.index_table.title, card_set.correction_table.title): card_set.family
    for card_set in _PART_TWO_CARD_SETS
}


def table_reading_family_for_payload(payload: TableReadingPayload) -> str:
    if payload.part is TableReadingPart.PART_ONE:
        return _PART_ONE_TITLE_TO_FAMILY.get(payload.primary_table.title, "unknown")
    secondary_title = payload.secondary_table.title if payload.secondary_table is not None else ""
    return _PART_TWO_TITLES_TO_FAMILY.get((payload.primary_table.title, secondary_title), "unknown")


class TableReadingScorer(AnswerScorer):
    """Exact table value gets full credit; near values receive partial credit."""

    def score(self, *, problem: Problem, user_answer: int, raw: str) -> float:
        payload = problem.payload
        if not isinstance(payload, TableReadingPayload):
            return 1.0 if int(user_answer) == int(problem.answer) else 0.0

        selected_value = int(user_answer)
        code_map = {option.code: option.value for option in payload.options}
        if selected_value in code_map:
            selected_value = int(code_map[selected_value])

        delta = abs(selected_value - int(payload.correct_value))
        if delta == 0:
            return 1.0

        tolerance = max(1, int(payload.estimate_tolerance))
        max_delta = tolerance * 2
        if delta >= max_delta:
            return 0.0
        return float(max_delta - delta) / float(max_delta)


class TableReadingGenerator:
    """Deterministic generator for single-table and chained-table reading tasks."""

    def __init__(self, *, seed: int) -> None:
        self._rng = SeededRng(seed)
        self._recent_part_one_families: list[str] = []
        self._recent_part_two_families: list[str] = []

    @staticmethod
    def supported_part_one_families() -> tuple[str, ...]:
        return tuple(pack.family for pack in _PART_ONE_PACKS)

    @staticmethod
    def supported_part_two_families() -> tuple[str, ...]:
        return tuple(card_set.family for card_set in _PART_TWO_CARD_SETS)

    def next_problem(self, *, difficulty: float) -> Problem:
        return self.next_problem_for_selection(difficulty=difficulty)

    def next_problem_for_selection(
        self,
        *,
        difficulty: float,
        part: TableReadingPart | None = None,
        family: str | None = None,
        profile: str = "default",
    ) -> Problem:
        d = clamp01(difficulty)
        active_part = part
        if active_part is None:
            part_two_probability = 0.55 + (0.35 * d)
            active_part = (
                TableReadingPart.PART_TWO
                if self._rng.random() < part_two_probability
                else TableReadingPart.PART_ONE
            )
        if active_part is TableReadingPart.PART_ONE:
            return self._part_one_problem(d, family=family, profile=profile)
        return self._part_two_problem(d, family=family, profile=profile)

    def _part_one_problem(self, difficulty: float, *, family: str | None, profile: str) -> Problem:
        pack = self._select_part_one_pack(family)
        row_idx, col_idx = self._select_part_one_indices(pack.table, profile=profile)

        row_label = pack.table.row_labels[row_idx]
        col_label = pack.table.column_labels[col_idx]
        correct_value = int(pack.table.values[row_idx][col_idx])
        option_step = self._part_one_option_step(difficulty=difficulty, profile=profile)
        options, correct_code, tolerance = self._build_options(
            correct_value=correct_value,
            option_step=option_step,
        )

        stem = (
            f"Part 1: Using {pack.table.title}, find the value at "
            f"{pack.table.row_header} {row_label} and {pack.table.column_header} {col_label}."
        )
        prompt = self._prompt_from(stem=stem, options=options)
        payload = TableReadingPayload(
            part=TableReadingPart.PART_ONE,
            stem=stem,
            primary_table=pack.table,
            secondary_table=None,
            primary_row_label=row_label,
            primary_column_label=col_label,
            secondary_row_label=None,
            secondary_column_label=None,
            options=options,
            correct_code=correct_code,
            correct_value=correct_value,
            estimate_tolerance=tolerance,
            content_family=pack.family,
            variant_id=stable_variant_id(
                "part_one",
                pack.family,
                row_label,
                col_label,
            ),
            content_pack="part_one",
        )
        return Problem(prompt=prompt, answer=correct_value, payload=payload)

    def _part_two_problem(self, difficulty: float, *, family: str | None, profile: str) -> Problem:
        card_set = self._select_part_two_set(family)
        speed_idx, wind_idx, angle_idx = self._select_part_two_indices(card_set, profile=profile)

        speed_label = card_set.index_table.row_labels[speed_idx]
        wind_label = card_set.index_table.column_labels[wind_idx]
        angle_label = card_set.correction_table.column_labels[angle_idx]

        drift_index = int(card_set.index_table.values[speed_idx][wind_idx])
        correction_row_idx = card_set.correction_table.row_labels.index(str(drift_index))
        correct_value = int(card_set.correction_table.values[correction_row_idx][angle_idx])
        option_step = self._part_two_option_step(difficulty=difficulty, profile=profile)
        options, correct_code, tolerance = self._build_options(
            correct_value=correct_value,
            option_step=option_step,
        )

        stem = (
            f"Part 2: Use {card_set.index_table.title} then {card_set.correction_table.title}. "
            f"{card_set.index_table.row_header}={speed_label}, "
            f"{card_set.index_table.column_header}={wind_label}, "
            f"{card_set.correction_table.column_header}={angle_label}. "
            "What is the final correction?"
        )
        prompt = self._prompt_from(stem=stem, options=options)
        payload = TableReadingPayload(
            part=TableReadingPart.PART_TWO,
            stem=stem,
            primary_table=card_set.index_table,
            secondary_table=card_set.correction_table,
            primary_row_label=speed_label,
            primary_column_label=wind_label,
            secondary_row_label=str(drift_index),
            secondary_column_label=angle_label,
            options=options,
            correct_code=correct_code,
            correct_value=correct_value,
            estimate_tolerance=tolerance,
            content_family=card_set.family,
            variant_id=stable_variant_id(
                "part_two",
                card_set.family,
                speed_label,
                wind_label,
                angle_label,
            ),
            content_pack="part_two",
        )
        return Problem(prompt=prompt, answer=correct_value, payload=payload)

    def _select_part_one_pack(self, family: str | None) -> _PartOneCardPack:
        if family is None:
            pool = [
                pack
                for pack in _PART_ONE_PACKS
                if pack.family not in self._recent_part_one_families[-2:]
            ] or list(_PART_ONE_PACKS)
            selected = pool[self._rng.randint(0, len(pool) - 1)]
            self._recent_part_one_families.append(selected.family)
            if len(self._recent_part_one_families) > 3:
                del self._recent_part_one_families[:-3]
            return selected
        token = str(family).strip().lower()
        if token not in _PART_ONE_PACK_BY_FAMILY:
            raise ValueError(f"Unknown Table Reading Part 1 family: {family}")
        return _PART_ONE_PACK_BY_FAMILY[token]

    def _select_part_two_set(self, family: str | None) -> _PartTwoCardSet:
        if family is None:
            pool = [
                card_set
                for card_set in _PART_TWO_CARD_SETS
                if card_set.family not in self._recent_part_two_families[-2:]
            ] or list(_PART_TWO_CARD_SETS)
            selected = pool[self._rng.randint(0, len(pool) - 1)]
            self._recent_part_two_families.append(selected.family)
            if len(self._recent_part_two_families) > 3:
                del self._recent_part_two_families[:-3]
            return selected
        token = str(family).strip().lower()
        if token not in _PART_TWO_SET_BY_FAMILY:
            raise ValueError(f"Unknown Table Reading Part 2 family: {family}")
        return _PART_TWO_SET_BY_FAMILY[token]

    def _select_part_one_indices(
        self,
        table: TableReadingTable,
        *,
        profile: str,
    ) -> tuple[int, int]:
        row_idx = self._rng.randint(0, len(table.row_labels) - 1)
        col_idx = self._rng.randint(0, len(table.column_labels) - 1)
        normalized = str(profile).strip().lower()
        if normalized in {"scan", "pressure"} and len(table.row_labels) >= 6 and len(table.column_labels) >= 6:
            row_half = len(table.row_labels) // 2
            col_half = len(table.column_labels) // 2
            if self._rng.randint(0, 1) == 0:
                row_idx = self._rng.randint(0, max(0, row_half - 1))
                col_idx = self._rng.randint(col_half, len(table.column_labels) - 1)
            else:
                row_idx = self._rng.randint(row_half, len(table.row_labels) - 1)
                col_idx = self._rng.randint(0, max(0, col_half - 1))
        return row_idx, col_idx

    def _select_part_two_indices(
        self,
        card_set: _PartTwoCardSet,
        *,
        profile: str,
    ) -> tuple[int, int, int]:
        normalized = str(profile).strip().lower()
        if normalized == "prime":
            speed_limit = max(1, (len(card_set.index_table.row_labels) * 3) // 5)
            wind_limit = max(1, (len(card_set.index_table.column_labels) * 3) // 5)
            angle_limit = max(1, len(card_set.correction_table.column_labels) // 2)
            return (
                self._rng.randint(0, speed_limit - 1),
                self._rng.randint(0, wind_limit - 1),
                self._rng.randint(0, angle_limit - 1),
            )
        return (
            self._rng.randint(0, len(card_set.index_table.row_labels) - 1),
            self._rng.randint(0, len(card_set.index_table.column_labels) - 1),
            self._rng.randint(0, len(card_set.correction_table.column_labels) - 1),
        )

    def _part_one_option_step(self, *, difficulty: float, profile: str) -> int:
        normalized = str(profile).strip().lower()
        if normalized == "anchor":
            return lerp_int(12, 5, difficulty)
        if normalized == "scan":
            return lerp_int(7, 2, difficulty)
        if normalized == "pressure":
            return lerp_int(5, 1, difficulty)
        return lerp_int(8, 3, difficulty)

    def _part_two_option_step(self, *, difficulty: float, profile: str) -> int:
        normalized = str(profile).strip().lower()
        if normalized == "prime":
            return lerp_int(4, 2, difficulty)
        if normalized == "pressure":
            return lerp_int(2, 1, difficulty)
        return lerp_int(3, 1, difficulty)

    def _build_options(
        self,
        *,
        correct_value: int,
        option_step: int,
    ) -> tuple[tuple[TableReadingOption, ...], int, int]:
        values: list[int] = [int(correct_value)]
        spread = max(1, int(option_step))

        while len(values) < 5:
            direction = -1 if self._rng.randint(0, 1) == 0 else 1
            scale = self._rng.randint(1, 3)
            jitter = self._rng.randint(0, max(1, spread // 2))
            candidate = int(correct_value) + (direction * scale * spread)
            candidate += -jitter if direction < 0 else jitter
            if candidate < 0 or candidate in values:
                continue
            values.append(candidate)

        order = self._rng.sample([0, 1, 2, 3, 4], k=5)
        shuffled = [values[idx] for idx in order]
        labels = ("A", "B", "C", "D", "E")

        options = tuple(
            TableReadingOption(code=index + 1, label=labels[index], value=value)
            for index, value in enumerate(shuffled)
        )
        correct_code = next(opt.code for opt in options if opt.value == int(correct_value))

        nearest_delta = min(
            abs(option.value - int(correct_value))
            for option in options
            if option.value != int(correct_value)
        )
        tolerance = max(1, nearest_delta)
        return options, int(correct_code), int(tolerance)

    def _prompt_from(self, *, stem: str, options: tuple[TableReadingOption, ...]) -> str:
        lines = [stem, ""]
        for option in options:
            lines.append(f"{option.code}) {option.label}: {option.value}")
        return "\n".join(lines)


def build_table_reading_test(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    config: TableReadingConfig | None = None,
) -> TimedTextInputTest:
    cfg = config or TableReadingConfig()

    instructions = [
        "Table Reading Test",
        "",
        "Work quickly and accurately by scanning and cross-referencing lookup cards.",
        "Part 1: use one table card to find a row/column value.",
        "Part 2: use an index card and a correction card in sequence to find the final answer.",
        "",
        "Controls:",
        "- Press A, S, D, F, or G to choose an option",
        "- Use Up/Down to move between options",
        "- Press Enter to submit",
        "",
        "Printable cards are in cfast_trainer/table_reading_cards/.",
        "Once the timed block starts, continue until completion.",
    ]

    return TimedTextInputTest(
        title="Table Reading",
        instructions=instructions,
        generator=TableReadingGenerator(seed=seed),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        practice_questions=cfg.practice_questions,
        scored_duration_s=cfg.scored_duration_s,
        scorer=TableReadingScorer(),
    )

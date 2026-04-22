from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .clock import Clock
from .content_variants import content_metadata_from_payload, stable_variant_id
from .cognitive_core import (
    AnswerScorer,
    Phase,
    Problem,
    QuestionEvent,
    SeededRng,
    TestSnapshot,
    TimedTextInputTest,
    clamp01,
    lerp_int,
)


TABLE_READING_MIN_SIZE = 5
TABLE_READING_MAX_SIZE = 50


@dataclass(frozen=True, slots=True)
class TableReadingConfig:
    # Candidate guide indicates ~11 minutes including instructions.
    scored_duration_s: float = 9.0 * 60.0
    practice_questions: int = 2


class TableReadingPart(StrEnum):
    PART_ONE = "part_one_cross_reference"
    PART_TWO = "part_two_multi_table"


class TableReadingItemKind(StrEnum):
    SINGLE_TABLE_LOOKUP = "single_table_lookup"
    TWO_TABLE_LOOKUP = "two_table_lookup"
    THREE_TABLE_LOOKUP = "three_table_lookup"
    REVERSE_LOOKUP = "reverse_lookup"
    LETTER_SEARCH = "letter_search"


class TableReadingAnswerMode(StrEnum):
    MULTIPLE_CHOICE = "multiple_choice"
    NUMERIC = "numeric"
    SINGLE_LETTER = "single_letter"
    LETTER_STRING = "letter_string"


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
    values: tuple[tuple[int | str, ...], ...]


@dataclass(frozen=True, slots=True)
class TableReadingDataTab:
    title: str
    tables: tuple[TableReadingTable, ...]


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
    item_kind: TableReadingItemKind = TableReadingItemKind.SINGLE_TABLE_LOOKUP
    answer_mode: TableReadingAnswerMode = TableReadingAnswerMode.MULTIPLE_CHOICE
    data_tabs: tuple[TableReadingDataTab, ...] = ()
    correct_answer_text: str = ""
    input_max_length: int = 8
    display_answer_text: str = ""


@dataclass(frozen=True, slots=True)
class _PartOneSpec:
    family: str
    title: str
    row_header: str
    column_header: str
    base: int
    row_gain: int
    col_gain: int
    cross: int


@dataclass(frozen=True, slots=True)
class _PartTwoSpec:
    family: str
    index_title: str
    correction_title: str
    index_row_header: str
    index_column_header: str
    correction_column_header: str
    base: int
    row_gain: int
    col_gain: int


_PART_ONE_SPECS: tuple[_PartOneSpec, ...] = (
    _PartOneSpec("lookup", "Card A - Lookup Table", "Row", "Column", 171, 17, 11, 4),
    _PartOneSpec("station", "Card D - Station Table", "Station", "Band", 142, 19, 13, 5),
    _PartOneSpec("sector", "Card E - Sector Table", "Sector", "Gate", 196, 15, 17, 3),
    _PartOneSpec("dispatch", "Card J - Dispatch Table", "Dispatch", "Window", 156, 18, 14, 6),
    _PartOneSpec("range", "Card K - Range Table", "Range", "Band", 188, 16, 12, 4),
)

_PART_TWO_SPECS: tuple[_PartTwoSpec, ...] = (
    _PartTwoSpec(
        "wind",
        "Card B - Wind Index Table",
        "Card C - Angle Correction Table",
        "Air Speed",
        "Wind Velocity",
        "Wind Angle",
        31,
        3,
        5,
    ),
    _PartTwoSpec(
        "crosswind",
        "Card F - Crosswind Drift Table",
        "Card G - Course Correction Table",
        "Ground Speed",
        "Crosswind",
        "Track Angle",
        43,
        4,
        6,
    ),
    _PartTwoSpec(
        "offset",
        "Card H - Offset Index Table",
        "Card I - Lateral Correction Table",
        "Track Speed",
        "Offset Drift",
        "Correction Angle",
        37,
        5,
        4,
    ),
    _PartTwoSpec(
        "descent",
        "Card L - Descent Index Table",
        "Card M - Recovery Angle Table",
        "Track Speed",
        "Sink Rate",
        "Recovery Arc",
        47,
        4,
        7,
    ),
    _PartTwoSpec(
        "timing",
        "Card N - Timing Index Table",
        "Card O - Timing Adjustment Table",
        "Ground Speed",
        "Timing Error",
        "Adjustment Angle",
        29,
        6,
        5,
    ),
)

_PART_ONE_BY_FAMILY = {spec.family: spec for spec in _PART_ONE_SPECS}
_PART_TWO_BY_FAMILY = {spec.family: spec for spec in _PART_TWO_SPECS}
_DEFAULT_KIND_SEQUENCE: tuple[TableReadingItemKind, ...] = (
    TableReadingItemKind.SINGLE_TABLE_LOOKUP,
    TableReadingItemKind.TWO_TABLE_LOOKUP,
    TableReadingItemKind.THREE_TABLE_LOOKUP,
    TableReadingItemKind.REVERSE_LOOKUP,
    TableReadingItemKind.LETTER_SEARCH,
)


def table_reading_table_size_for_difficulty(difficulty: float) -> int:
    return lerp_int(TABLE_READING_MIN_SIZE, TABLE_READING_MAX_SIZE, clamp01(difficulty))


def table_reading_family_for_payload(payload: TableReadingPayload) -> str:
    return str(payload.content_family or "unknown")


def table_reading_event_user_answer(problem: Problem, raw: str) -> int:
    payload = problem.payload
    token = str(raw).strip()
    if isinstance(payload, TableReadingPayload):
        if payload.answer_mode in (
            TableReadingAnswerMode.SINGLE_LETTER,
            TableReadingAnswerMode.LETTER_STRING,
        ):
            return 0
        if payload.answer_mode is TableReadingAnswerMode.MULTIPLE_CHOICE and token.isdigit():
            return int(token)
    try:
        return int(token)
    except ValueError:
        return 0


def _normalize_text_answer(raw: object) -> str:
    return "".join(ch for ch in str(raw).upper() if ch.isalnum())


def _alpha_label(index: int) -> str:
    n = max(0, int(index))
    chars: list[str] = []
    while True:
        n, rem = divmod(n, 26)
        chars.append(chr(ord("A") + rem))
        if n == 0:
            break
        n -= 1
    return "".join(reversed(chars))


def _alpha_labels(size: int) -> tuple[str, ...]:
    return tuple(_alpha_label(idx) for idx in range(max(1, int(size))))


def _numeric_values(
    *,
    size: int,
    base: int,
    row_gain: int,
    col_gain: int,
    cross: int,
) -> tuple[tuple[int, ...], ...]:
    row_stride = max(1, size) * (max(1, col_gain) + max(1, cross) + 3)
    rows: list[tuple[int, ...]] = []
    for row_idx in range(size):
        current: list[int] = []
        for col_idx in range(size):
            value = base + row_idx * row_stride + col_idx * col_gain
            value += (row_idx % 4) * row_gain
            value += ((row_idx + col_idx) % 5) * cross
            current.append(int(value))
        rows.append(tuple(current))
    return tuple(rows)


def _index_values(size: int, *, offset: int = 0) -> tuple[tuple[int, ...], ...]:
    rows: list[tuple[int, ...]] = []
    for row_idx in range(size):
        rows.append(tuple(((row_idx + col_idx + offset) % size) + 1 for col_idx in range(size)))
    return tuple(rows)


def _letter_values(size: int, *, offset: int) -> tuple[tuple[str, ...], ...]:
    rows: list[tuple[str, ...]] = []
    for row_idx in range(size):
        current: list[str] = []
        for col_idx in range(size):
            letter_idx = (row_idx * 7 + col_idx * 11 + offset + (row_idx * col_idx)) % 26
            current.append(chr(ord("A") + letter_idx))
        rows.append(tuple(current))
    return tuple(rows)


def _data_tabs_for(tables: tuple[TableReadingTable, ...]) -> tuple[TableReadingDataTab, ...]:
    if len(tables) <= 1:
        return (TableReadingDataTab("Table", tables[:1]),)
    if len(tables) == 2:
        return (
            TableReadingDataTab("Table 1", (tables[0],)),
            TableReadingDataTab("Table 2", (tables[1],)),
        )
    return (
        TableReadingDataTab("Table 1", (tables[0],)),
        TableReadingDataTab("Tables 2-3", (tables[1], tables[2])),
    )


class TableReadingScorer(AnswerScorer):
    """Score table answers while allowing numeric, choice, and letter-entry modes."""

    def score(self, *, problem: Problem, user_answer: int, raw: str) -> float:
        payload = problem.payload
        if not isinstance(payload, TableReadingPayload):
            return 1.0 if int(user_answer) == int(problem.answer) else 0.0

        if payload.answer_mode in (
            TableReadingAnswerMode.SINGLE_LETTER,
            TableReadingAnswerMode.LETTER_STRING,
        ):
            submitted = _normalize_text_answer(raw)
            expected = _normalize_text_answer(payload.correct_answer_text)
            return 1.0 if submitted == expected and expected != "" else 0.0

        selected_value: int
        token = str(raw).strip()
        if payload.answer_mode is TableReadingAnswerMode.MULTIPLE_CHOICE:
            code_map = {int(option.code): int(option.value) for option in payload.options}
            if token.isdigit() and int(token) in code_map:
                selected_value = code_map[int(token)]
            elif int(user_answer) in code_map:
                selected_value = code_map[int(user_answer)]
            else:
                selected_value = int(user_answer)
        else:
            try:
                selected_value = int(token)
            except ValueError:
                selected_value = int(user_answer)

        delta = abs(selected_value - int(payload.correct_value))
        if delta == 0:
            return 1.0

        tolerance = max(0, int(payload.estimate_tolerance))
        if tolerance <= 0:
            return 0.0
        max_delta = tolerance * 2
        if delta >= max_delta:
            return 0.0
        return float(max_delta - delta) / float(max_delta)


class TableReadingGenerator:
    """Deterministic generator for CFASC-style tabbed table reading tasks."""

    def __init__(self, *, seed: int) -> None:
        self._rng = SeededRng(seed)
        self._recent_part_one_families: list[str] = []
        self._recent_part_two_families: list[str] = []
        self._default_kind_index = 0

    @staticmethod
    def supported_part_one_families() -> tuple[str, ...]:
        return tuple(spec.family for spec in _PART_ONE_SPECS)

    @staticmethod
    def supported_part_two_families() -> tuple[str, ...]:
        return tuple(spec.family for spec in _PART_TWO_SPECS)

    def next_problem(self, *, difficulty: float) -> Problem:
        return self.next_problem_for_selection(difficulty=difficulty)

    def next_problem_for_selection(
        self,
        *,
        difficulty: float,
        part: TableReadingPart | None = None,
        family: str | None = None,
        profile: str = "default",
        item_kind: TableReadingItemKind | str | None = None,
        answer_mode: TableReadingAnswerMode | str | None = None,
    ) -> Problem:
        d = clamp01(difficulty)
        active_kind = self._resolve_item_kind(part=part, profile=profile, item_kind=item_kind)
        active_answer_mode = (
            None if answer_mode is None else TableReadingAnswerMode(str(answer_mode))
        )

        if active_kind is TableReadingItemKind.SINGLE_TABLE_LOOKUP:
            return self._single_table_problem(
                d,
                family=family,
                profile=profile,
                answer_mode=active_answer_mode,
            )
        if active_kind is TableReadingItemKind.TWO_TABLE_LOOKUP:
            return self._two_table_problem(
                d,
                family=family,
                profile=profile,
                answer_mode=active_answer_mode,
            )
        if active_kind is TableReadingItemKind.THREE_TABLE_LOOKUP:
            return self._three_table_problem(d, family=family, profile=profile)
        if active_kind is TableReadingItemKind.REVERSE_LOOKUP:
            return self._reverse_lookup_problem(d, family=family, profile=profile)
        return self._letter_search_problem(d, profile=profile, answer_mode=active_answer_mode)

    def _resolve_item_kind(
        self,
        *,
        part: TableReadingPart | None,
        profile: str,
        item_kind: TableReadingItemKind | str | None,
    ) -> TableReadingItemKind:
        if item_kind is not None:
            return TableReadingItemKind(str(item_kind))
        profile_key = str(profile).strip().lower()
        if profile_key in {"three", "compute", "three_table"}:
            return TableReadingItemKind.THREE_TABLE_LOOKUP
        if profile_key in {"reverse", "distractor"}:
            return TableReadingItemKind.REVERSE_LOOKUP
        if profile_key in {"letter", "letters", "letter_search"}:
            return TableReadingItemKind.LETTER_SEARCH
        if part is TableReadingPart.PART_ONE:
            return TableReadingItemKind.SINGLE_TABLE_LOOKUP
        if part is TableReadingPart.PART_TWO:
            return TableReadingItemKind.TWO_TABLE_LOOKUP
        kind = _DEFAULT_KIND_SEQUENCE[self._default_kind_index % len(_DEFAULT_KIND_SEQUENCE)]
        self._default_kind_index += 1
        return kind

    def _single_table_problem(
        self,
        difficulty: float,
        *,
        family: str | None,
        profile: str,
        answer_mode: TableReadingAnswerMode | None,
    ) -> Problem:
        spec = self._select_part_one_spec(family)
        table = self._part_one_table(spec=spec, size=table_reading_table_size_for_difficulty(difficulty))
        row_idx, col_idx = self._select_indices(table, profile=profile)
        row_label = table.row_labels[row_idx]
        col_label = table.column_labels[col_idx]
        correct_value = int(table.values[row_idx][col_idx])
        mode = answer_mode or TableReadingAnswerMode.MULTIPLE_CHOICE
        options: tuple[TableReadingOption, ...] = ()
        correct_code = 0
        tolerance = self._part_one_option_step(difficulty=difficulty, profile=profile)
        if mode is TableReadingAnswerMode.MULTIPLE_CHOICE:
            options, correct_code, tolerance = self._build_options(
                correct_value=correct_value,
                option_step=tolerance,
            )
        stem = (
            f"Using {table.title}, find the value at "
            f"{table.row_header} {row_label} and {table.column_header} {col_label}."
        )
        payload = self._payload(
            part=TableReadingPart.PART_ONE,
            stem=stem,
            tables=(table,),
            primary_row_label=row_label,
            primary_column_label=col_label,
            secondary_row_label=None,
            secondary_column_label=None,
            options=options,
            correct_code=correct_code,
            correct_value=correct_value,
            estimate_tolerance=tolerance,
            content_family=spec.family,
            content_pack="part_one",
            item_kind=TableReadingItemKind.SINGLE_TABLE_LOOKUP,
            answer_mode=mode,
            correct_answer_text=str(correct_value),
            input_max_length=max(3, len(str(correct_value))),
            variant_parts=("single", spec.family, row_label, col_label, mode.value),
        )
        return Problem(prompt=self._prompt_from(stem=stem, options=options), answer=correct_value, payload=payload)

    def _two_table_problem(
        self,
        difficulty: float,
        *,
        family: str | None,
        profile: str,
        answer_mode: TableReadingAnswerMode | None,
    ) -> Problem:
        spec = self._select_part_two_spec(family)
        size = table_reading_table_size_for_difficulty(difficulty)
        index_table, correction_table = self._part_two_tables(spec=spec, size=size)
        row_idx, col_idx = self._select_indices(index_table, profile=profile)
        angle_idx = self._rng.randint(0, len(correction_table.column_labels) - 1)
        row_label = index_table.row_labels[row_idx]
        col_label = index_table.column_labels[col_idx]
        angle_label = correction_table.column_labels[angle_idx]
        index_value = int(index_table.values[row_idx][col_idx])
        correction_row_idx = correction_table.row_labels.index(str(index_value))
        correct_value = int(correction_table.values[correction_row_idx][angle_idx])
        mode = answer_mode or TableReadingAnswerMode.MULTIPLE_CHOICE
        options: tuple[TableReadingOption, ...] = ()
        correct_code = 0
        tolerance = self._part_two_option_step(difficulty=difficulty, profile=profile)
        if mode is TableReadingAnswerMode.MULTIPLE_CHOICE:
            options, correct_code, tolerance = self._build_options(
                correct_value=correct_value,
                option_step=tolerance,
            )
        stem = (
            f"Use {index_table.title} then {correction_table.title}. "
            f"{index_table.row_header}={row_label}, "
            f"{index_table.column_header}={col_label}, "
            f"{correction_table.column_header}={angle_label}. "
            "What is the final correction?"
        )
        payload = self._payload(
            part=TableReadingPart.PART_TWO,
            stem=stem,
            tables=(index_table, correction_table),
            primary_row_label=row_label,
            primary_column_label=col_label,
            secondary_row_label=str(index_value),
            secondary_column_label=angle_label,
            options=options,
            correct_code=correct_code,
            correct_value=correct_value,
            estimate_tolerance=tolerance,
            content_family=spec.family,
            content_pack="part_two",
            item_kind=TableReadingItemKind.TWO_TABLE_LOOKUP,
            answer_mode=mode,
            correct_answer_text=str(correct_value),
            input_max_length=max(3, len(str(correct_value))),
            variant_parts=("two", spec.family, row_label, col_label, angle_label, mode.value),
        )
        return Problem(prompt=self._prompt_from(stem=stem, options=options), answer=correct_value, payload=payload)

    def _three_table_problem(
        self,
        difficulty: float,
        *,
        family: str | None,
        profile: str,
    ) -> Problem:
        spec = self._select_part_two_spec(family)
        size = table_reading_table_size_for_difficulty(difficulty)
        table_a, table_b, table_c = self._three_tables(spec=spec, size=size)
        row_idx, first_col_idx = self._select_indices(table_a, profile=profile)
        second_col_idx = self._rng.randint(0, size - 1)
        third_col_idx = self._rng.randint(0, size - 1)
        row_label = table_a.row_labels[row_idx]
        first_col_label = table_a.column_labels[first_col_idx]
        second_col_label = table_b.column_labels[second_col_idx]
        third_col_label = table_c.column_labels[third_col_idx]
        first_index = int(table_a.values[row_idx][first_col_idx])
        second_index = int(table_b.values[first_index - 1][second_col_idx])
        correct_value = int(table_c.values[second_index - 1][third_col_idx])
        tolerance = max(1, self._part_two_option_step(difficulty=difficulty, profile=profile))
        stem = (
            f"Use {table_a.title}, {table_b.title}, then {table_c.title}. "
            f"Start at {table_a.row_header} {row_label} and {table_a.column_header} {first_col_label}; "
            f"use {table_b.column_header} {second_col_label}; then use "
            f"{table_c.column_header} {third_col_label}. Enter the final value."
        )
        payload = self._payload(
            part=TableReadingPart.PART_TWO,
            stem=stem,
            tables=(table_a, table_b, table_c),
            primary_row_label=row_label,
            primary_column_label=first_col_label,
            secondary_row_label=str(first_index),
            secondary_column_label=second_col_label,
            options=(),
            correct_code=0,
            correct_value=correct_value,
            estimate_tolerance=tolerance,
            content_family=spec.family,
            content_pack="part_three",
            item_kind=TableReadingItemKind.THREE_TABLE_LOOKUP,
            answer_mode=TableReadingAnswerMode.NUMERIC,
            correct_answer_text=str(correct_value),
            input_max_length=max(3, len(str(correct_value))),
            variant_parts=("three", spec.family, row_label, first_col_label, second_col_label, third_col_label),
        )
        return Problem(prompt=stem, answer=correct_value, payload=payload)

    def _reverse_lookup_problem(
        self,
        difficulty: float,
        *,
        family: str | None,
        profile: str,
    ) -> Problem:
        spec = self._select_part_one_spec(family)
        table = self._part_one_table(spec=spec, size=table_reading_table_size_for_difficulty(difficulty))
        row_idx, col_idx = self._select_indices(table, profile=profile)
        row_label = table.row_labels[row_idx]
        col_label = table.column_labels[col_idx]
        correct_value = int(table.values[row_idx][col_idx])
        answer_text = f"{row_label}{col_label}"
        stem = (
            f"In {table.title}, find the cell containing {correct_value}. "
            f"Enter the {table.row_header} label followed by the {table.column_header} label with no spaces."
        )
        payload = self._payload(
            part=TableReadingPart.PART_ONE,
            stem=stem,
            tables=(table,),
            primary_row_label=row_label,
            primary_column_label=col_label,
            secondary_row_label=None,
            secondary_column_label=None,
            options=(),
            correct_code=0,
            correct_value=correct_value,
            estimate_tolerance=0,
            content_family=spec.family,
            content_pack="reverse_lookup",
            item_kind=TableReadingItemKind.REVERSE_LOOKUP,
            answer_mode=TableReadingAnswerMode.LETTER_STRING,
            correct_answer_text=answer_text,
            input_max_length=max(2, len(answer_text)),
            variant_parts=("reverse", spec.family, correct_value, row_label, col_label),
        )
        return Problem(prompt=stem, answer=0, payload=payload)

    def _letter_search_problem(
        self,
        difficulty: float,
        *,
        profile: str,
        answer_mode: TableReadingAnswerMode | None,
    ) -> Problem:
        size = table_reading_table_size_for_difficulty(difficulty)
        labels = _alpha_labels(size)
        table = TableReadingTable(
            title="Card P - Letter Search Table",
            row_header="Row",
            column_header="Column",
            row_labels=labels,
            column_labels=labels,
            values=_letter_values(size, offset=self._rng.randint(0, 25)),
        )
        requested_mode = answer_mode
        if requested_mode is TableReadingAnswerMode.SINGLE_LETTER:
            count = 1
        elif requested_mode is TableReadingAnswerMode.LETTER_STRING:
            count = max(2, min(4, 2 + self._rng.randint(0, 2)))
        else:
            count = 1 if difficulty < 0.35 or self._rng.random() < 0.45 else 2 + self._rng.randint(0, 2)
        positions = self._rng.sample(range(size * size), k=count)
        cells: list[tuple[int, int]] = [(pos // size, pos % size) for pos in positions]
        answer_text = "".join(str(table.values[row][col]) for row, col in cells)
        mode = (
            TableReadingAnswerMode.SINGLE_LETTER
            if len(answer_text) == 1
            else TableReadingAnswerMode.LETTER_STRING
        )
        coords = "; ".join(
            f"{table.row_header} {table.row_labels[row]} / {table.column_header} {table.column_labels[col]}"
            for row, col in cells
        )
        stem = f"Read the letter{'s' if count > 1 else ''} at: {coords}. Enter them in order."
        first_row, first_col = cells[0]
        payload = self._payload(
            part=TableReadingPart.PART_ONE,
            stem=stem,
            tables=(table,),
            primary_row_label=table.row_labels[first_row],
            primary_column_label=table.column_labels[first_col],
            secondary_row_label=None,
            secondary_column_label=None,
            options=(),
            correct_code=0,
            correct_value=0,
            estimate_tolerance=0,
            content_family="letters",
            content_pack="letter_search",
            item_kind=TableReadingItemKind.LETTER_SEARCH,
            answer_mode=mode,
            correct_answer_text=answer_text,
            input_max_length=max(1, len(answer_text)),
            variant_parts=("letters", coords, answer_text),
        )
        return Problem(prompt=stem, answer=0, payload=payload)

    def _payload(
        self,
        *,
        part: TableReadingPart,
        stem: str,
        tables: tuple[TableReadingTable, ...],
        primary_row_label: str,
        primary_column_label: str,
        secondary_row_label: str | None,
        secondary_column_label: str | None,
        options: tuple[TableReadingOption, ...],
        correct_code: int,
        correct_value: int,
        estimate_tolerance: int,
        content_family: str,
        content_pack: str,
        item_kind: TableReadingItemKind,
        answer_mode: TableReadingAnswerMode,
        correct_answer_text: str,
        input_max_length: int,
        variant_parts: tuple[object, ...],
    ) -> TableReadingPayload:
        return TableReadingPayload(
            part=part,
            stem=stem,
            primary_table=tables[0],
            secondary_table=tables[1] if len(tables) >= 2 else None,
            primary_row_label=primary_row_label,
            primary_column_label=primary_column_label,
            secondary_row_label=secondary_row_label,
            secondary_column_label=secondary_column_label,
            options=options,
            correct_code=int(correct_code),
            correct_value=int(correct_value),
            estimate_tolerance=int(estimate_tolerance),
            content_family=content_family,
            variant_id=stable_variant_id(*variant_parts),
            content_pack=content_pack,
            item_kind=item_kind,
            answer_mode=answer_mode,
            data_tabs=_data_tabs_for(tables),
            correct_answer_text=str(correct_answer_text),
            input_max_length=max(1, int(input_max_length)),
            display_answer_text=str(correct_answer_text),
        )

    def _part_one_table(self, *, spec: _PartOneSpec, size: int) -> TableReadingTable:
        labels = _alpha_labels(size)
        jitter = self._rng.randint(0, 19)
        return TableReadingTable(
            title=spec.title,
            row_header=spec.row_header,
            column_header=spec.column_header,
            row_labels=labels,
            column_labels=labels,
            values=_numeric_values(
                size=size,
                base=spec.base + jitter,
                row_gain=spec.row_gain,
                col_gain=spec.col_gain,
                cross=spec.cross,
            ),
        )

    def _part_two_tables(
        self,
        *,
        spec: _PartTwoSpec,
        size: int,
    ) -> tuple[TableReadingTable, TableReadingTable]:
        labels = _alpha_labels(size)
        index_table = TableReadingTable(
            title=spec.index_title,
            row_header=spec.index_row_header,
            column_header=spec.index_column_header,
            row_labels=labels,
            column_labels=labels,
            values=_index_values(size, offset=self._rng.randint(0, max(1, size - 1))),
        )
        correction_table = TableReadingTable(
            title=spec.correction_title,
            row_header="Index",
            column_header=spec.correction_column_header,
            row_labels=tuple(str(idx + 1) for idx in range(size)),
            column_labels=labels,
            values=_numeric_values(
                size=size,
                base=spec.base * 4 + self._rng.randint(0, 11),
                row_gain=spec.row_gain,
                col_gain=spec.col_gain,
                cross=2,
            ),
        )
        return index_table, correction_table

    def _three_tables(
        self,
        *,
        spec: _PartTwoSpec,
        size: int,
    ) -> tuple[TableReadingTable, TableReadingTable, TableReadingTable]:
        labels = _alpha_labels(size)
        table_a = TableReadingTable(
            title=f"{spec.index_title} - Start",
            row_header=spec.index_row_header,
            column_header=spec.index_column_header,
            row_labels=labels,
            column_labels=labels,
            values=_index_values(size, offset=self._rng.randint(0, max(1, size - 1))),
        )
        table_b = TableReadingTable(
            title=f"{spec.correction_title} - Transfer",
            row_header="Index A",
            column_header="Transfer Column",
            row_labels=tuple(str(idx + 1) for idx in range(size)),
            column_labels=labels,
            values=_index_values(size, offset=self._rng.randint(0, max(1, size - 1))),
        )
        table_c = TableReadingTable(
            title=f"{spec.correction_title} - Final",
            row_header="Index B",
            column_header=spec.correction_column_header,
            row_labels=tuple(str(idx + 1) for idx in range(size)),
            column_labels=labels,
            values=_numeric_values(
                size=size,
                base=spec.base * 6 + self._rng.randint(0, 13),
                row_gain=spec.row_gain + 1,
                col_gain=spec.col_gain + 1,
                cross=3,
            ),
        )
        return table_a, table_b, table_c

    def _select_part_one_spec(self, family: str | None) -> _PartOneSpec:
        if family is not None:
            token = str(family).strip().lower()
            if token in _PART_ONE_BY_FAMILY:
                return _PART_ONE_BY_FAMILY[token]
            raise ValueError(f"Unknown Table Reading Part 1 family: {family}")
        pool = [
            spec
            for spec in _PART_ONE_SPECS
            if spec.family not in self._recent_part_one_families[-2:]
        ] or list(_PART_ONE_SPECS)
        selected = pool[self._rng.randint(0, len(pool) - 1)]
        self._recent_part_one_families.append(selected.family)
        if len(self._recent_part_one_families) > 3:
            del self._recent_part_one_families[:-3]
        return selected

    def _select_part_two_spec(self, family: str | None) -> _PartTwoSpec:
        if family is not None:
            token = str(family).strip().lower()
            if token in _PART_TWO_BY_FAMILY:
                return _PART_TWO_BY_FAMILY[token]
            raise ValueError(f"Unknown Table Reading Part 2 family: {family}")
        pool = [
            spec
            for spec in _PART_TWO_SPECS
            if spec.family not in self._recent_part_two_families[-2:]
        ] or list(_PART_TWO_SPECS)
        selected = pool[self._rng.randint(0, len(pool) - 1)]
        self._recent_part_two_families.append(selected.family)
        if len(self._recent_part_two_families) > 3:
            del self._recent_part_two_families[:-3]
        return selected

    def _select_indices(self, table: TableReadingTable, *, profile: str) -> tuple[int, int]:
        row_idx = self._rng.randint(0, len(table.row_labels) - 1)
        col_idx = self._rng.randint(0, len(table.column_labels) - 1)
        normalized = str(profile).strip().lower()
        if normalized in {"scan", "pressure", "distractor"} and len(table.row_labels) >= 6 and len(table.column_labels) >= 6:
            row_half = len(table.row_labels) // 2
            col_half = len(table.column_labels) // 2
            if self._rng.randint(0, 1) == 0:
                row_idx = self._rng.randint(0, max(0, row_half - 1))
                col_idx = self._rng.randint(col_half, len(table.column_labels) - 1)
            else:
                row_idx = self._rng.randint(row_half, len(table.row_labels) - 1)
                col_idx = self._rng.randint(0, max(0, col_half - 1))
        return row_idx, col_idx

    def _part_one_option_step(self, *, difficulty: float, profile: str) -> int:
        normalized = str(profile).strip().lower()
        if normalized == "anchor":
            return lerp_int(12, 5, difficulty)
        if normalized == "scan":
            return lerp_int(7, 2, difficulty)
        if normalized in {"pressure", "distractor"}:
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
        return options, int(correct_code), max(1, int(nearest_delta))

    def _prompt_from(self, *, stem: str, options: tuple[TableReadingOption, ...]) -> str:
        if not options:
            return stem
        lines = [stem, ""]
        for option in options:
            lines.append(f"{option.code}) {option.label}: {option.value}")
        return "\n".join(lines)


class TableReadingTest(TimedTextInputTest):
    def submit_answer(self, raw: object) -> bool:
        if self._phase not in (Phase.PRACTICE, Phase.SCORED):
            return False
        expired = self._phase is Phase.SCORED and self.time_remaining_s() == 0

        raw_in = raw if isinstance(raw, str) else str(raw)
        raw_token = raw_in.strip()
        if expired and raw_token == "":
            self._phase = Phase.RESULTS
            return False
        if raw_token == "":
            return False

        assert self._current is not None
        assert self._presented_at_s is not None

        user_answer = table_reading_event_user_answer(self._current, raw_in)
        answered_at_s = self._clock.now()
        response_time_s = max(0.0, answered_at_s - self._presented_at_s)
        if self._scorer is None:
            score = 1.0 if user_answer == self._current.answer else 0.0
        else:
            score = float(self._scorer.score(problem=self._current, user_answer=user_answer, raw=raw_in))
            score = 0.0 if score < 0.0 else 1.0 if score > 1.0 else score
        is_full_correct = score >= 1.0 - 1e-9

        self._events.append(
            QuestionEvent(
                index=len(self._events),
                phase=self._phase,
                prompt=self._current.prompt,
                correct_answer=self._current.answer,
                user_answer=user_answer,
                is_correct=is_full_correct,
                presented_at_s=self._presented_at_s,
                answered_at_s=answered_at_s,
                response_time_s=response_time_s,
                raw=raw_in,
                score=score,
                max_score=1.0,
                content_metadata=content_metadata_from_payload(self._current.payload),
            )
        )

        if self._phase is Phase.SCORED:
            self._scored_attempted += 1
            self._scored_max_score += 1.0
            self._scored_total_score += score
            if is_full_correct:
                self._scored_correct += 1
        else:
            self._practice_answered += 1

        if expired and self._phase is Phase.SCORED:
            self._phase = Phase.RESULTS
            self._current = None
            self._presented_at_s = None
            return True

        if self._phase is Phase.PRACTICE and self._practice_answered >= self._practice_questions:
            self._phase = Phase.PRACTICE_DONE
            self._current = None
            self._presented_at_s = None
            return True

        self._deal_new_problem()
        return True

    def snapshot(self) -> TestSnapshot:
        snap = super().snapshot()
        payload = snap.payload
        if isinstance(payload, TableReadingPayload) and snap.phase in (Phase.PRACTICE, Phase.SCORED):
            return TestSnapshot(
                title=snap.title,
                phase=snap.phase,
                prompt=snap.prompt,
                input_hint=_input_hint_for_payload(payload),
                time_remaining_s=snap.time_remaining_s,
                attempted_scored=snap.attempted_scored,
                correct_scored=snap.correct_scored,
                payload=snap.payload,
                practice_feedback=snap.practice_feedback,
            )
        return snap


def _input_hint_for_payload(payload: TableReadingPayload) -> str:
    if payload.answer_mode is TableReadingAnswerMode.MULTIPLE_CHOICE:
        return "A/S/D/F/G or 1-5: answer | Tab: tables"
    if payload.answer_mode is TableReadingAnswerMode.NUMERIC:
        return "Digits then Enter | Tab: tables"
    if payload.answer_mode is TableReadingAnswerMode.SINGLE_LETTER:
        return "One letter then Enter | Tab: tables"
    return "Letters then Enter | Tab: tables"


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
        "Use the Question tab for the prompt and answer field.",
        "Use the table tabs to inspect one, two, or three lookup cards.",
        "Some items use choices, some require a typed number, and some require letters.",
        "",
        "Controls:",
        "- Press Tab or Shift+Tab to switch information tabs",
        "- Press A, S, D, F, or G for multiple-choice items",
        "- Type digits or letters for typed-answer items, then press Enter",
        "",
        "Printable cards are in cfast_trainer/table_reading_cards/.",
        "Once the timed block starts, continue until completion.",
    ]

    return TableReadingTest(
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

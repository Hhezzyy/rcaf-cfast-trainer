from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .clock import Clock
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


_PART_ONE_TABLE = TableReadingTable(
    title="Card A - Lookup Table",
    row_header="Row",
    column_header="Column",
    row_labels=("R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10"),
    column_labels=("C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10"),
    values=(
        (171, 186, 201, 197, 212, 234, 249, 245, 260, 275),
        (192, 193, 213, 214, 241, 242, 262, 263, 283, 284),
        (213, 219, 225, 238, 263, 269, 275, 281, 287, 312),
        (215, 226, 244, 255, 266, 277, 288, 299, 310, 328),
        (236, 259, 275, 272, 288, 304, 301, 317, 340, 337),
        (264, 266, 287, 289, 310, 312, 314, 342, 344, 365),
        (285, 292, 299, 306, 313, 320, 353, 360, 367, 374),
        (287, 299, 311, 323, 335, 354, 366, 378, 390, 383),
        (308, 325, 323, 340, 364, 362, 379, 396, 394, 411),
        (329, 332, 354, 364, 367, 389, 392, 395, 417, 420),
    ),
)

_PART_TWO_INDEX_TABLE = TableReadingTable(
    title="Card B - Wind Index Table",
    row_header="Air Speed",
    column_header="Wind Velocity",
    row_labels=("80", "100", "120", "140", "160", "180", "200", "220"),
    column_labels=("4", "8", "12", "16", "20", "24", "28", "32"),
    values=(
        (4, 6, 8, 10, 12, 13, 14, 15),
        (3, 5, 7, 9, 11, 12, 13, 14),
        (3, 4, 6, 8, 10, 11, 12, 13),
        (2, 4, 5, 7, 9, 10, 11, 12),
        (2, 3, 5, 6, 8, 9, 10, 11),
        (1, 3, 4, 6, 7, 8, 9, 10),
        (1, 2, 4, 5, 6, 7, 8, 9),
        (1, 2, 3, 4, 5, 6, 7, 8),
    ),
)

_PART_TWO_CORRECTION_TABLE = TableReadingTable(
    title="Card C - Angle Correction Table",
    row_header="Drift Index",
    column_header="Wind Angle",
    row_labels=("1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"),
    column_labels=("15", "25", "35", "45", "55", "65", "75", "85", "95"),
    values=(
        (1, 1, 1, 1, 1, 1, 1, 1, 1),
        (1, 1, 1, 1, 1, 2, 2, 2, 2),
        (1, 1, 2, 2, 2, 3, 3, 3, 4),
        (1, 1, 2, 2, 3, 3, 4, 4, 4),
        (1, 1, 2, 3, 3, 4, 5, 5, 6),
        (1, 2, 3, 3, 4, 5, 5, 6, 7),
        (1, 2, 3, 4, 5, 5, 7, 8, 8),
        (2, 2, 4, 5, 5, 6, 8, 8, 9),
        (2, 3, 4, 5, 6, 7, 8, 9, 11),
        (2, 3, 4, 6, 7, 8, 9, 11, 12),
        (2, 3, 5, 6, 7, 9, 10, 11, 13),
        (2, 4, 5, 7, 8, 10, 11, 13, 14),
        (3, 4, 6, 7, 9, 10, 12, 14, 15),
        (3, 4, 6, 8, 9, 11, 13, 15, 16),
        (3, 5, 7, 8, 10, 12, 14, 16, 18),
    ),
)


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
    """Deterministic generator for table scanning and cross-reference tasks."""

    def __init__(self, *, seed: int) -> None:
        self._rng = SeededRng(seed)

    def next_problem(self, *, difficulty: float) -> Problem:
        d = clamp01(difficulty)
        part_two_probability = 0.55 + (0.35 * d)
        if self._rng.random() < part_two_probability:
            return self._part_two_problem(d)
        return self._part_one_problem(d)

    def _part_one_problem(self, difficulty: float) -> Problem:
        row_idx = self._rng.randint(0, len(_PART_ONE_TABLE.row_labels) - 1)
        col_idx = self._rng.randint(0, len(_PART_ONE_TABLE.column_labels) - 1)

        row_label = _PART_ONE_TABLE.row_labels[row_idx]
        col_label = _PART_ONE_TABLE.column_labels[col_idx]
        correct_value = int(_PART_ONE_TABLE.values[row_idx][col_idx])

        option_step = lerp_int(6, 2, difficulty)
        options, correct_code, tolerance = self._build_options(
            correct_value=correct_value,
            option_step=option_step,
        )

        stem = f"Part 1: Using Card A, find the value at {row_label} and {col_label}."
        prompt = self._prompt_from(stem=stem, options=options)

        payload = TableReadingPayload(
            part=TableReadingPart.PART_ONE,
            stem=stem,
            primary_table=_PART_ONE_TABLE,
            secondary_table=None,
            primary_row_label=row_label,
            primary_column_label=col_label,
            secondary_row_label=None,
            secondary_column_label=None,
            options=options,
            correct_code=correct_code,
            correct_value=correct_value,
            estimate_tolerance=tolerance,
        )
        return Problem(prompt=prompt, answer=correct_value, payload=payload)

    def _part_two_problem(self, difficulty: float) -> Problem:
        speed_idx = self._rng.randint(0, len(_PART_TWO_INDEX_TABLE.row_labels) - 1)
        wind_idx = self._rng.randint(0, len(_PART_TWO_INDEX_TABLE.column_labels) - 1)
        angle_idx = self._rng.randint(0, len(_PART_TWO_CORRECTION_TABLE.column_labels) - 1)

        speed_label = _PART_TWO_INDEX_TABLE.row_labels[speed_idx]
        wind_label = _PART_TWO_INDEX_TABLE.column_labels[wind_idx]
        angle_label = _PART_TWO_CORRECTION_TABLE.column_labels[angle_idx]

        drift_index = int(_PART_TWO_INDEX_TABLE.values[speed_idx][wind_idx])
        correction_row_idx = _PART_TWO_CORRECTION_TABLE.row_labels.index(str(drift_index))
        correct_value = int(_PART_TWO_CORRECTION_TABLE.values[correction_row_idx][angle_idx])

        option_step = lerp_int(2, 1, difficulty)
        options, correct_code, tolerance = self._build_options(
            correct_value=correct_value,
            option_step=option_step,
        )

        stem = (
            "Part 2: Use Card B then Card C. "
            f"Air Speed={speed_label}, Wind Velocity={wind_label}, Wind Angle={angle_label}. "
            "What is the drift correction?"
        )
        prompt = self._prompt_from(stem=stem, options=options)

        payload = TableReadingPayload(
            part=TableReadingPart.PART_TWO,
            stem=stem,
            primary_table=_PART_TWO_INDEX_TABLE,
            secondary_table=_PART_TWO_CORRECTION_TABLE,
            primary_row_label=speed_label,
            primary_column_label=wind_label,
            secondary_row_label=str(drift_index),
            secondary_column_label=angle_label,
            options=options,
            correct_code=correct_code,
            correct_value=correct_value,
            estimate_tolerance=tolerance,
        )
        return Problem(prompt=prompt, answer=correct_value, payload=payload)

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
        "Work quickly and accurately by scanning and cross-referencing lookup tables.",
        "Part 1: use one table (Card A) to find row/column values.",
        "Part 2: use Card B and Card C in sequence to compute drift correction.",
        "",
        "Controls:",
        "- Press 1, 2, 3, 4, or 5 to choose an option",
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

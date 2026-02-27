from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from itertools import combinations

from .clock import Clock
from .cognitive_core import (
    AnswerScorer,
    Problem,
    SeededRng,
    TimedTextInputTest,
    clamp01,
    lerp_int,
)

_GRID_SIZE = 10
_ROW_LABELS = tuple("ABCDEFGHIJ")

_CALLSIGN_POOL = (
    "R1",
    "R2",
    "R3",
    "T4",
    "K5",
    "N6",
    "F7",
    "M8",
    "H9",
    "P1",
    "L2",
    "S3",
)

_HEADING_VECTORS: dict[str, tuple[int, int]] = {
    "N": (0, -1),
    "NE": (1, -1),
    "E": (1, 0),
    "SE": (1, 1),
    "S": (0, 1),
    "SW": (-1, 1),
    "W": (-1, 0),
    "NW": (-1, -1),
}

_CARDINAL_HEADINGS = ("N", "E", "S", "W")


@dataclass(frozen=True, slots=True)
class SituationalAwarenessConfig:
    # Training-friendly default (shorter than guide estimate).
    scored_duration_s: float = 12.0 * 60.0
    practice_questions: int = 3


class SituationalAwarenessQuestionKind(StrEnum):
    POSITION_PROJECTION = "position_projection"
    CONFLICT_PREDICTION = "conflict_prediction"
    ACTION_SELECTION = "action_selection"


@dataclass(frozen=True, slots=True)
class SituationalAwarenessContact:
    callsign: str
    x: int
    y: int
    heading: str
    speed_cells_per_min: int
    squawk: int
    fuel_state: str = "NORMAL"


@dataclass(frozen=True, slots=True)
class SituationalAwarenessOption:
    code: int
    text: str
    cell_label: str | None = None


@dataclass(frozen=True, slots=True)
class SituationalAwarenessPayload:
    kind: SituationalAwarenessQuestionKind
    stem: str
    horizon_min: int
    contacts: tuple[SituationalAwarenessContact, ...]
    options: tuple[SituationalAwarenessOption, ...]
    correct_code: int
    query_callsign: str | None
    conflict_pair: tuple[str, str] | None
    correct_cell: str | None


def _clamp_grid(value: int) -> int:
    if value < 0:
        return 0
    if value >= _GRID_SIZE:
        return _GRID_SIZE - 1
    return int(value)


def cell_label_from_xy(x: int, y: int) -> str:
    ix = _clamp_grid(x)
    iy = _clamp_grid(y)
    return f"{_ROW_LABELS[iy]}{ix}"


def cell_xy_from_label(label: str) -> tuple[int, int] | None:
    token = str(label).strip().upper()
    if len(token) < 2:
        return None
    row_token = token[0]
    col_token = token[1:]
    if row_token not in _ROW_LABELS:
        return None
    if not col_token.isdigit():
        return None
    x = int(col_token)
    y = _ROW_LABELS.index(row_token)
    if x < 0 or x >= _GRID_SIZE:
        return None
    return x, y


def project_contact_xy(contact: SituationalAwarenessContact, minutes: int) -> tuple[int, int]:
    heading = str(contact.heading).upper()
    dx, dy = _HEADING_VECTORS.get(heading, (0, 0))
    speed = max(0, int(contact.speed_cells_per_min))
    dt = max(0, int(minutes))
    x = contact.x + (dx * speed * dt)
    y = contact.y + (dy * speed * dt)
    return _clamp_grid(x), _clamp_grid(y)


def project_contact_cell(contact: SituationalAwarenessContact, minutes: int) -> str:
    x, y = project_contact_xy(contact, minutes)
    return cell_label_from_xy(x, y)


class SituationalAwarenessScorer(AnswerScorer):
    """Exact option is full credit; near projection misses receive partial credit."""

    def score(self, *, problem: Problem, user_answer: int, raw: str) -> float:
        _ = raw
        payload = problem.payload
        if not isinstance(payload, SituationalAwarenessPayload):
            return 1.0 if int(user_answer) == int(problem.answer) else 0.0

        selected_code = int(user_answer)
        if selected_code == int(payload.correct_code):
            return 1.0

        if payload.kind is not SituationalAwarenessQuestionKind.POSITION_PROJECTION:
            return 0.0
        if payload.correct_cell is None:
            return 0.0

        selected_option = next((opt for opt in payload.options if opt.code == selected_code), None)
        if selected_option is None or selected_option.cell_label is None:
            return 0.0

        selected_xy = cell_xy_from_label(selected_option.cell_label)
        correct_xy = cell_xy_from_label(payload.correct_cell)
        if selected_xy is None or correct_xy is None:
            return 0.0

        dist = abs(selected_xy[0] - correct_xy[0]) + abs(selected_xy[1] - correct_xy[1])
        if dist <= 1:
            return 0.5
        if dist == 2:
            return 0.25
        return 0.0


class SituationalAwarenessGenerator:
    """Deterministic mixed-information situational awareness trial generator."""

    def __init__(self, *, seed: int) -> None:
        self._rng = SeededRng(seed)
        self._last_kind: SituationalAwarenessQuestionKind | None = None

    def next_problem(self, *, difficulty: float) -> Problem:
        d = clamp01(difficulty)
        kind = self._pick_kind(d)
        if kind is SituationalAwarenessQuestionKind.POSITION_PROJECTION:
            return self._position_projection_problem(d)
        if kind is SituationalAwarenessQuestionKind.CONFLICT_PREDICTION:
            return self._conflict_prediction_problem(d)
        return self._action_selection_problem(d)

    def _pick_kind(self, difficulty: float) -> SituationalAwarenessQuestionKind:
        if difficulty < 0.35:
            kinds = (
                SituationalAwarenessQuestionKind.POSITION_PROJECTION,
                SituationalAwarenessQuestionKind.CONFLICT_PREDICTION,
            )
        elif difficulty < 0.70:
            kinds = (
                SituationalAwarenessQuestionKind.POSITION_PROJECTION,
                SituationalAwarenessQuestionKind.CONFLICT_PREDICTION,
                SituationalAwarenessQuestionKind.ACTION_SELECTION,
            )
        else:
            # Heavier emphasis on conflict/action at higher difficulty.
            kinds = (
                SituationalAwarenessQuestionKind.POSITION_PROJECTION,
                SituationalAwarenessQuestionKind.CONFLICT_PREDICTION,
                SituationalAwarenessQuestionKind.ACTION_SELECTION,
                SituationalAwarenessQuestionKind.ACTION_SELECTION,
            )

        pick = self._rng.choice(kinds)
        if self._last_kind is not None and pick == self._last_kind and self._rng.random() < 0.65:
            alternatives = tuple(kind for kind in kinds if kind != pick)
            pick = self._rng.choice(alternatives)
        self._last_kind = pick
        return pick

    def _position_projection_problem(self, difficulty: float) -> Problem:
        horizon = self._rng.randint(1, lerp_int(2, 4, difficulty))
        contact_count = lerp_int(3, 5, difficulty)
        callsigns = self._rng.sample(_CALLSIGN_POOL, k=contact_count)

        query_contact = self._random_motion_contact(
            callsign=callsigns[0],
            horizon_min=horizon,
            difficulty=difficulty,
            prefer_cardinal=False,
        )
        others = [
            self._random_motion_contact(
                callsign=callsign,
                horizon_min=horizon,
                difficulty=difficulty,
                prefer_cardinal=True,
            )
            for callsign in callsigns[1:]
        ]
        contacts = (query_contact, *others)

        correct_cell = project_contact_cell(query_contact, horizon)
        options, correct_code = self._cell_options(correct_cell)

        stem = f"After {horizon} min, where will {query_contact.callsign} be?"
        prompt = self._compose_prompt(
            aural=(
                f"{query_contact.callsign} maintain heading {query_contact.heading} at "
                f"{query_contact.speed_cells_per_min} blk/min."
            ),
            visual=self._contact_line(contacts),
            code=self._code_line(contacts),
            numeric=f"Projection horizon {horizon} min.",
            query=stem,
            options=options,
        )

        payload = SituationalAwarenessPayload(
            kind=SituationalAwarenessQuestionKind.POSITION_PROJECTION,
            stem=stem,
            horizon_min=horizon,
            contacts=contacts,
            options=options,
            correct_code=correct_code,
            query_callsign=query_contact.callsign,
            conflict_pair=None,
            correct_cell=correct_cell,
        )
        return Problem(prompt=prompt, answer=correct_code, payload=payload)

    def _conflict_prediction_problem(self, difficulty: float) -> Problem:
        horizon = self._rng.randint(2, lerp_int(3, 4, difficulty))
        callsigns = self._rng.sample(_CALLSIGN_POOL, k=4)

        heading_pairs = [("E", "W"), ("N", "S")]
        if difficulty >= 0.55:
            heading_pairs.extend([("NE", "SW"), ("NW", "SE")])

        meet_x = 5
        meet_y = 5
        heading_a = "E"
        heading_b = "W"
        for _ in range(96):
            pair = self._rng.choice(tuple(heading_pairs))
            cand_a = pair[0]
            cand_b = pair[1]
            mx = self._rng.randint(1, _GRID_SIZE - 2)
            my = self._rng.randint(1, _GRID_SIZE - 2)

            ax, ay = self._start_for_target(mx, my, cand_a, speed=1, horizon_min=horizon)
            bx, by = self._start_for_target(mx, my, cand_b, speed=1, horizon_min=horizon)
            if self._in_bounds(ax, ay) and self._in_bounds(bx, by):
                meet_x = mx
                meet_y = my
                heading_a = cand_a
                heading_b = cand_b
                break

        contact_a = self._make_contact(
            callsign=callsigns[0],
            x=self._start_for_target(meet_x, meet_y, heading_a, speed=1, horizon_min=horizon)[0],
            y=self._start_for_target(meet_x, meet_y, heading_a, speed=1, horizon_min=horizon)[1],
            heading=heading_a,
            speed=1,
        )
        contact_b = self._make_contact(
            callsign=callsigns[1],
            x=self._start_for_target(meet_x, meet_y, heading_b, speed=1, horizon_min=horizon)[0],
            y=self._start_for_target(meet_x, meet_y, heading_b, speed=1, horizon_min=horizon)[1],
            heading=heading_b,
            speed=1,
        )
        contact_c = self._make_contact(callsigns[2], x=0, y=0, heading="N", speed=0)
        contact_d = self._make_contact(
            callsigns[3], x=_GRID_SIZE - 1, y=_GRID_SIZE - 1, heading="S", speed=0
        )
        contacts = (contact_a, contact_b, contact_c, contact_d)

        correct_pair = tuple(sorted((contact_a.callsign, contact_b.callsign)))
        all_pairs = [tuple(sorted((a.callsign, b.callsign))) for a, b in combinations(contacts, 2)]
        distractors = [pair for pair in all_pairs if pair != correct_pair]
        sampled = self._rng.sample(distractors, k=3)
        pair_values = [correct_pair, *sampled]
        order = self._rng.sample([0, 1, 2, 3], k=4)
        shuffled = [pair_values[idx] for idx in order]

        options = tuple(
            SituationalAwarenessOption(code=index + 1, text=f"{pair[0]} & {pair[1]}")
            for index, pair in enumerate(shuffled)
        )
        correct_code = next(
            opt.code for opt, pair in zip(options, shuffled, strict=True) if pair == correct_pair
        )

        stem = f"Which pair occupies the same block at +{horizon} min?"
        prompt = self._compose_prompt(
            aural=f"{contact_a.callsign} and {contact_b.callsign} report converging traffic.",
            visual=self._contact_line(contacts),
            code=self._code_line(contacts),
            numeric=f"Predict positions after {horizon} min.",
            query=stem,
            options=options,
        )

        payload = SituationalAwarenessPayload(
            kind=SituationalAwarenessQuestionKind.CONFLICT_PREDICTION,
            stem=stem,
            horizon_min=horizon,
            contacts=contacts,
            options=options,
            correct_code=correct_code,
            query_callsign=None,
            conflict_pair=correct_pair,
            correct_cell=cell_label_from_xy(meet_x, meet_y),
        )
        return Problem(prompt=prompt, answer=correct_code, payload=payload)

    def _action_selection_problem(self, difficulty: float) -> Problem:
        horizon = self._rng.randint(2, lerp_int(3, 4, difficulty))
        callsigns = self._rng.sample(_CALLSIGN_POOL, k=3)

        meet_x = self._rng.randint(2, _GRID_SIZE - 3)
        meet_y = self._rng.randint(2, _GRID_SIZE - 3)
        priority = self._make_contact(
            callsign=callsigns[0],
            x=self._start_for_target(meet_x, meet_y, "E", speed=1, horizon_min=horizon)[0],
            y=self._start_for_target(meet_x, meet_y, "E", speed=1, horizon_min=horizon)[1],
            heading="E",
            speed=1,
            fuel_state="MIN",
        )
        traffic = self._make_contact(
            callsign=callsigns[1],
            x=self._start_for_target(meet_x, meet_y, "W", speed=1, horizon_min=horizon)[0],
            y=self._start_for_target(meet_x, meet_y, "W", speed=1, horizon_min=horizon)[1],
            heading="W",
            speed=1,
            fuel_state="NORMAL",
        )
        support = self._random_motion_contact(
            callsign=callsigns[2],
            horizon_min=horizon,
            difficulty=difficulty,
            prefer_cardinal=True,
        )
        contacts = (priority, traffic, support)

        correct_text = f"Vector {traffic.callsign} away; keep {priority.callsign} direct."
        option_texts = (
            correct_text,
            f"Hold {priority.callsign}; keep {traffic.callsign} direct.",
            "Maintain both routes and monitor one more sweep.",
            "Change both to backup channel before vectoring.",
        )
        order = self._rng.sample([0, 1, 2, 3], k=4)
        shuffled = [option_texts[idx] for idx in order]
        options = tuple(
            SituationalAwarenessOption(code=index + 1, text=text)
            for index, text in enumerate(shuffled)
        )
        correct_code = next(opt.code for opt in options if opt.text == correct_text)

        merge_cell = cell_label_from_xy(meet_x, meet_y)
        stem = "Best immediate action?"
        prompt = self._compose_prompt(
            aural=f"{priority.callsign} reports MIN fuel; {traffic.callsign} normal.",
            visual=self._contact_line(contacts),
            code=self._code_line(contacts),
            numeric=f"Projected merge {merge_cell} in {horizon} min.",
            query=stem,
            options=options,
        )

        payload = SituationalAwarenessPayload(
            kind=SituationalAwarenessQuestionKind.ACTION_SELECTION,
            stem=stem,
            horizon_min=horizon,
            contacts=contacts,
            options=options,
            correct_code=correct_code,
            query_callsign=priority.callsign,
            conflict_pair=tuple(sorted((priority.callsign, traffic.callsign))),
            correct_cell=merge_cell,
        )
        return Problem(prompt=prompt, answer=correct_code, payload=payload)

    def _make_contact(
        self,
        callsign: str,
        *,
        x: int,
        y: int,
        heading: str,
        speed: int,
        fuel_state: str = "NORMAL",
    ) -> SituationalAwarenessContact:
        return SituationalAwarenessContact(
            callsign=str(callsign),
            x=_clamp_grid(x),
            y=_clamp_grid(y),
            heading=str(heading).upper(),
            speed_cells_per_min=max(0, int(speed)),
            squawk=1000 + self._rng.randint(0, 6777),
            fuel_state=str(fuel_state).upper(),
        )

    def _random_motion_contact(
        self,
        *,
        callsign: str,
        horizon_min: int,
        difficulty: float,
        prefer_cardinal: bool,
    ) -> SituationalAwarenessContact:
        headings: tuple[str, ...]
        if prefer_cardinal or difficulty < 0.55:
            headings = _CARDINAL_HEADINGS
        else:
            headings = tuple(_HEADING_VECTORS.keys())

        max_speed = 1 if difficulty < 0.65 else 2
        for _ in range(96):
            heading = self._rng.choice(headings)
            speed = self._rng.randint(1, max_speed)
            x_range, y_range = self._valid_start_ranges(
                heading=heading, speed=speed, horizon_min=horizon_min
            )
            if x_range is None or y_range is None:
                continue
            x = self._rng.randint(x_range[0], x_range[1])
            y = self._rng.randint(y_range[0], y_range[1])
            return self._make_contact(callsign, x=x, y=y, heading=heading, speed=speed)

        # Safe fallback if no sampled combination fit (should be rare).
        return self._make_contact(callsign, x=4, y=4, heading="E", speed=1)

    def _valid_start_ranges(
        self,
        *,
        heading: str,
        speed: int,
        horizon_min: int,
    ) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
        dx, dy = _HEADING_VECTORS.get(str(heading).upper(), (0, 0))
        total_dx = int(dx) * int(speed) * int(horizon_min)
        total_dy = int(dy) * int(speed) * int(horizon_min)

        x_lo = max(0, -total_dx)
        x_hi = min(_GRID_SIZE - 1, (_GRID_SIZE - 1) - total_dx)
        y_lo = max(0, -total_dy)
        y_hi = min(_GRID_SIZE - 1, (_GRID_SIZE - 1) - total_dy)

        x_range = (x_lo, x_hi) if x_lo <= x_hi else None
        y_range = (y_lo, y_hi) if y_lo <= y_hi else None
        return x_range, y_range

    def _start_for_target(
        self,
        target_x: int,
        target_y: int,
        heading: str,
        *,
        speed: int,
        horizon_min: int,
    ) -> tuple[int, int]:
        dx, dy = _HEADING_VECTORS.get(str(heading).upper(), (0, 0))
        x = int(target_x) - (dx * int(speed) * int(horizon_min))
        y = int(target_y) - (dy * int(speed) * int(horizon_min))
        return x, y

    @staticmethod
    def _in_bounds(x: int, y: int) -> bool:
        return 0 <= int(x) < _GRID_SIZE and 0 <= int(y) < _GRID_SIZE

    def _cell_options(
        self, correct_cell: str
    ) -> tuple[tuple[SituationalAwarenessOption, ...], int]:
        correct_xy = cell_xy_from_label(correct_cell)
        assert correct_xy is not None
        cx, cy = correct_xy

        offsets = (
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (1, -1),
            (-1, 1),
            (1, 1),
            (-2, 0),
            (2, 0),
            (0, -2),
            (0, 2),
        )
        shuffled_offsets = self._rng.sample(offsets, k=len(offsets))

        labels: list[str] = [correct_cell]
        for dx, dy in shuffled_offsets:
            x = cx + int(dx)
            y = cy + int(dy)
            if not self._in_bounds(x, y):
                continue
            candidate = cell_label_from_xy(x, y)
            if candidate in labels:
                continue
            labels.append(candidate)
            if len(labels) == 4:
                break

        while len(labels) < 4:
            candidate = cell_label_from_xy(
                self._rng.randint(0, _GRID_SIZE - 1), self._rng.randint(0, _GRID_SIZE - 1)
            )
            if candidate in labels:
                continue
            labels.append(candidate)

        order = self._rng.sample([0, 1, 2, 3], k=4)
        shuffled = [labels[idx] for idx in order]
        options = tuple(
            SituationalAwarenessOption(code=index + 1, text=label, cell_label=label)
            for index, label in enumerate(shuffled)
        )
        correct_code = next(opt.code for opt in options if opt.cell_label == correct_cell)
        return options, correct_code

    @staticmethod
    def _contact_line(contacts: tuple[SituationalAwarenessContact, ...]) -> str:
        chunks = [
            (
                f"{c.callsign}@{cell_label_from_xy(c.x, c.y)}/"
                f"{c.heading}{c.speed_cells_per_min}/{c.fuel_state}"
            )
            for c in contacts
        ]
        return " | ".join(chunks)

    @staticmethod
    def _code_line(contacts: tuple[SituationalAwarenessContact, ...]) -> str:
        return " ".join(f"{c.callsign}:{int(c.squawk):04d}" for c in contacts)

    @staticmethod
    def _compose_prompt(
        *,
        aural: str,
        visual: str,
        code: str,
        numeric: str,
        query: str,
        options: tuple[SituationalAwarenessOption, ...],
    ) -> str:
        lines = [
            "Situational Awareness Feed",
            f"AURAL: {aural}",
            f"VIS: {visual}",
            f"CODE: {code}",
            f"NUM: {numeric}",
            f"QUERY: {query}",
        ]
        for option in options:
            lines.append(f"{option.code}) {option.text}")
        return "\n".join(lines[:10])


def build_situational_awareness_test(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    config: SituationalAwarenessConfig | None = None,
) -> TimedTextInputTest:
    cfg = config or SituationalAwarenessConfig()

    instructions = [
        "Situational Awareness Test",
        "",
        "Monitor mixed aural, visual, numerical, and coded updates.",
        "Build and update a mental picture of moving aircraft and constraints.",
        "Answer future-state and action-priority queries quickly and accurately.",
        "",
        "Grid reference: rows A-J and columns 0-9.",
        "Contact format: CALLSIGN@CELL/HEADING+SPEED/FUEL.",
        "",
        "Controls:",
        "- Click highlighted map cells (projection trials) or option rows",
        "- Keyboard fallback: 1, 2, 3, or 4 then Enter",
        "- Up/Down cycles options when using keyboard",
        "",
        "Timed block default is 12 minutes for repeatable training sessions.",
    ]

    return TimedTextInputTest(
        title="Situational Awareness",
        instructions=instructions,
        generator=SituationalAwarenessGenerator(seed=seed),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        practice_questions=cfg.practice_questions,
        scored_duration_s=cfg.scored_duration_s,
        scorer=SituationalAwarenessScorer(),
    )

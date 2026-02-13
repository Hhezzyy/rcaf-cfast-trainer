from __future__ import annotations

from dataclasses import dataclass

from .clock import Clock
from .cognitive_core import (
    AnswerScorer,
    Problem,
    SeededRng,
    TimedTextInputTest,
    lerp_int,
    round_half_up,
)

# -----------------------------------------------------------------------------
# Airborne Numerical (training recreation)
# - Answers are HHMM (24h) and must be exactly 4 digits (UI enforces)
# - No backspace (UI enforces)
# - Deterministic generation (seeded RNG)
# -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RouteLeg:
    frm: int
    to: int
    distance: int


@dataclass(frozen=True, slots=True)
class AirborneScenario:
    template_name: str
    node_names: tuple[str, ...]
    edge_distances: tuple[int, ...]  # aligned to template edges
    route: tuple[int, ...]  # node indices
    legs: tuple[RouteLeg, ...]

    start_time_hhmm: str
    speed_value: int
    speed_unit: str
    distance_unit: str

    fuel_burn_per_hr: int
    parcel_weight: int
    start_fuel_liters: int

    # overlay “menus” (for tables/graphs)
    weight_speed_table: tuple[tuple[int, int], ...]  # (kg, speed)
    speed_fuel_table: tuple[tuple[int, int], ...]  # (speed, L/hr)

    # per-problem label (helps UI/tests)
    question_kind: str  # "arrival_time" | "empty_time"
    target_label: str   # destination code or "EMPTY"

    # convenience aliases (fixes Pylance red + keeps old UI code working)
    @property
    def parcel_weight_kg(self) -> int:
        return self.parcel_weight

    @property
    def fuel_burn_lph(self) -> int:
        return self.fuel_burn_per_hr


@dataclass(frozen=True, slots=True)
class MapTemplate:
    name: str
    nodes: tuple[tuple[float, float], ...]  # normalized (0..1) positions
    edges: tuple[tuple[int, int], ...]  # (node_a, node_b)


TEMPLATES: tuple[MapTemplate, ...] = (
    MapTemplate(
        name="T1",
        nodes=((0.10, 0.20), (0.40, 0.20), (0.70, 0.25), (0.25, 0.55), (0.55, 0.55), (0.85, 0.60)),
        edges=((0, 1), (1, 2), (1, 3), (3, 4), (4, 2), (4, 5)),
    ),
    MapTemplate(
        name="T2",
        nodes=((0.15, 0.25), (0.40, 0.10), (0.70, 0.20), (0.25, 0.55), (0.55, 0.50), (0.80, 0.60)),
        edges=((0, 1), (1, 2), (0, 3), (3, 4), (4, 2), (4, 5)),
    ),
    MapTemplate(
        name="T3",
        nodes=((0.10, 0.20), (0.35, 0.10), (0.65, 0.12), (0.30, 0.55), (0.60, 0.50), (0.85, 0.55)),
        edges=((0, 1), (1, 2), (0, 3), (3, 4), (4, 2), (4, 5)),
    ),
    MapTemplate(
        name="T4",
        nodes=((0.10, 0.25), (0.35, 0.10), (0.65, 0.15), (0.30, 0.55), (0.60, 0.55), (0.85, 0.60)),
        edges=((0, 1), (1, 2), (0, 3), (3, 4), (4, 2), (4, 5)),
    ),
    MapTemplate(
        name="T5",
        nodes=((0.10, 0.20), (0.35, 0.15), (0.60, 0.12), (0.85, 0.20), (0.25, 0.55), (0.55, 0.55), (0.85, 0.60)),
        edges=((0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 2), (5, 6), (6, 3)),
    ),
    MapTemplate(
        name="T6",
        nodes=((0.12, 0.18), (0.40, 0.12), (0.70, 0.15), (0.20, 0.52), (0.50, 0.55), (0.82, 0.55)),
        edges=((0, 1), (1, 2), (0, 3), (3, 4), (4, 2), (4, 5)),
    ),
)

TEMPLATES_BY_NAME = {t.name: t for t in TEMPLATES}

_NODE_CODES = (
    "ALP", "BRV", "CHL", "DLT", "ECO", "FOX", "GLF", "HIL", "IND", "JUL",
    "KIL", "MKE", "NVR", "OPS", "PRM", "QNT", "RDL", "SNR", "TNG", "ULS",
    "VCT", "WHS", "XRY", "YLD", "ZUL",
)


def _hhmm_str_to_minutes(hhmm: str) -> int:
    hh = int(hhmm[:2])
    mm = int(hhmm[2:])
    return hh * 60 + mm


def _minutes_to_hhmm_str(minutes: int) -> str:
    minutes %= 24 * 60
    hh = minutes // 60
    mm = minutes % 60
    return f"{hh:02d}{mm:02d}"


def _minutes_to_hhmm_int(minutes: int) -> int:
    return int(_minutes_to_hhmm_str(minutes))


def _route_minutes(legs: tuple[RouteLeg, ...], speed_value: int) -> int:
    # Round once at final answer (your preference).
    total_dist = sum(l.distance for l in legs)
    return int(round_half_up((total_dist / max(speed_value, 1)) * 60.0))


class AirborneTimeScorer(AnswerScorer):
    """Score HHMM answers with a +/-30 minute linear tolerance.
    Exact match = 1.0, >=30 min error = 0.0.
    """

    def score(self, problem: Problem, user_answer: int) -> float:
        correct_hhmm = int(problem.answer)
        user_hhmm = int(user_answer)

        correct_min = _hhmm_str_to_minutes(f"{correct_hhmm:04d}")
        user_min = _hhmm_str_to_minutes(f"{user_hhmm:04d}")

        diff = abs(user_min - correct_min)
        diff = min(diff, 24 * 60 - diff)

        if diff <= 0:
            return 1.0
        if diff >= 30:
            return 0.0
        return float(lerp_int(1000, 0, diff / 30.0)) / 1000.0


# Back-compat name
AirborneArrivalTimeScorer = AirborneTimeScorer


def _pick_node_names(rng: SeededRng, n: int) -> tuple[str, ...]:
    picks = rng.sample(list(_NODE_CODES), k=n)
    return tuple(picks)


def _make_weight_speed_table(rng: SeededRng, speed_unit: str) -> tuple[tuple[tuple[int, int], ...], int, int]:
    weights = sorted(rng.sample(list(range(150, 851, 50)), k=6))
    parcel_weight = int(rng.choice(weights))

    if speed_unit == "km/h":
        max_speed = int(rng.randint(360, 520))
        min_speed = int(max(220, max_speed - rng.randint(80, 170)))
    else:
        max_speed = int(rng.randint(190, 310))
        min_speed = int(max(120, max_speed - rng.randint(40, 90)))

    w0, w1 = weights[0], weights[-1]
    rows: list[tuple[int, int]] = []
    for w in weights:
        t = 0.0 if w1 == w0 else (w - w0) / (w1 - w0)
        spd = int(round_half_up(max_speed + (min_speed - max_speed) * t))
        rows.append((w, spd))

    chosen_speed = next(spd for w, spd in rows if w == parcel_weight)
    return (tuple(rows), parcel_weight, chosen_speed)


def _make_speed_fuel_table(rng: SeededRng, speed_unit: str, chosen_speed: int) -> tuple[tuple[tuple[int, int], ...], int]:
    if speed_unit == "km/h":
        step = 40
        lo, hi = 220, 600
        base = rng.randint(800, 1600)
        factor = rng.random() * 2.5 + 3.8  # ~3.8..6.3
    else:
        step = 20
        lo, hi = 120, 360
        base = rng.randint(600, 1200)
        factor = rng.random() * 2.2 + 3.0  # ~3.0..5.2

    speeds = {chosen_speed}
    while len(speeds) < 6:
        delta = step * rng.randint(-3, 3)
        cand = int(max(lo, min(hi, chosen_speed + delta)))
        speeds.add(cand)

    speeds_sorted = sorted(speeds)
    rows: list[tuple[int, int]] = []
    burn_at = 0
    for spd in speeds_sorted:
        jitter = rng.randint(-120, 120)
        burn = int(max(400, round_half_up(base + spd * factor + jitter)))
        rows.append((spd, burn))
        if spd == chosen_speed:
            burn_at = burn

    return (tuple(rows), burn_at)


class AirborneNumericalGenerator:
    def __init__(self, rng: SeededRng):
        self._rng = rng

    def generate(self) -> Problem:
        scenario, prompt, answer = self._build_problem()
        return Problem(prompt=prompt, answer=int(answer), payload=scenario)

    def check(self, problem: Problem, user_answer: int) -> bool:
        return int(user_answer) == int(problem.answer)

    def _build_problem(self) -> tuple[AirborneScenario, str, int]:
        rng = self._rng
        template = rng.choice(list(TEMPLATES))
        node_names = _pick_node_names(rng, len(template.nodes))

        # Units (consistent within a problem)
        speed_unit = rng.choice(["km/h", "knots"])
        distance_unit = "km" if speed_unit == "km/h" else "NM"

        # Edge distances aligned with template.edges
        if distance_unit == "km":
            edge_distances = tuple(int(rng.randint(40, 180)) for _ in template.edges)
        else:
            edge_distances = tuple(int(rng.randint(25, 120)) for _ in template.edges)

        # Route: 2-4 legs
        start_node = int(rng.randint(0, len(template.nodes) - 1))
        route = [start_node]
        max_legs = int(rng.randint(2, 4))  # supports up to 3 vias
        for _ in range(max_legs):
            cur = route[-1]
            options = [b if a == cur else a for a, b in template.edges if a == cur or b == cur]
            options2 = [o for o in options if o not in route]
            nxt = int(rng.choice(options2 if options2 else options))
            route.append(nxt)
        route_t = tuple(route)

        legs: list[RouteLeg] = []
        for a, b in zip(route_t, route_t[1:], strict=False):
            edge_key = tuple(sorted((a, b)))
            idx = next(i for i, e in enumerate(template.edges) if tuple(sorted(e)) == edge_key)
            legs.append(RouteLeg(frm=a, to=b, distance=int(edge_distances[idx])))

        # Menus (tables/graphs)
        weight_speed_table, parcel_weight, speed_value = _make_weight_speed_table(rng, speed_unit)
        speed_fuel_table, fuel_burn = _make_speed_fuel_table(rng, speed_unit, speed_value)
        start_fuel = int(rng.randint(2000, 9000) // 50 * 50)

        # Start time
        start_minutes = int(rng.randint(0, 23) * 60 + rng.randint(0, 59))
        start_time_hhmm = _minutes_to_hhmm_str(start_minutes)

        # Question kind mix: include fuel-based time questions now
        question_kind = "arrival_time" if rng.random() < 0.6 else "empty_time"

        if question_kind == "arrival_time":
            dest = node_names[route_t[-1]]
            minutes = _route_minutes(tuple(legs), speed_value)
            answer = _minutes_to_hhmm_int(start_minutes + minutes)
            target_label = dest
            prompt = f"ARRIVAL TIME at {dest} (HHMM). Enter 4 digits:"
        else:
            endurance_min = int(round_half_up((start_fuel / max(fuel_burn, 1)) * 60.0))
            answer = _minutes_to_hhmm_int(start_minutes + endurance_min)
            target_label = "EMPTY"
            prompt = "EMPTY TIME (HHMM). Enter 4 digits:"

        scenario = AirborneScenario(
            template_name=template.name,
            node_names=node_names,
            edge_distances=edge_distances,
            route=route_t,
            legs=tuple(legs),
            start_time_hhmm=start_time_hhmm,
            speed_value=int(speed_value),
            speed_unit=speed_unit,
            distance_unit=distance_unit,
            fuel_burn_per_hr=int(fuel_burn),
            parcel_weight=int(parcel_weight),
            start_fuel_liters=start_fuel,
            weight_speed_table=weight_speed_table,
            speed_fuel_table=speed_fuel_table,
            question_kind=question_kind,
            target_label=target_label,
        )
        return scenario, prompt, int(answer)


def build_airborne_numerical_test(
    clock: Clock,
    seed: int,
    *,
    practice: bool = True,
    difficulty: float = 0.5,  # accepted for UI compatibility (can be used later)
    scored_duration_s: float = 35.0 * 60.0,
) -> TimedTextInputTest:
    rng = SeededRng(seed)
    gen = AirborneNumericalGenerator(rng)  # keep as-is for now
    scorer = AirborneArrivalTimeScorer()

    intro = [
        "Airborne Numerical",
        "",
        "Find the required time (HHMM) for each question.",
        "Enter 4 digits (24-hour clock). No backspace.",
        "",
        "Controls:",
        "  Hold A: Show distances",
        "  Hold S: Intro",
        "  Hold D: Speed & Fuel Consumption",
        "  Hold F: Speed & Parcel Weight",
        "",
        "Scoring:",
        "  Practice can show feedback. Scored section does not.",
    ]

    return TimedTextInputTest(
        title="Airborne Numerical Test",
        clock=clock,
        generator=gen,
        scorer=scorer,
        instructions=intro,
        practice_questions=6 if practice else 0,
        scored_duration_s=scored_duration_s,
        seconds_per_question=None,
    )
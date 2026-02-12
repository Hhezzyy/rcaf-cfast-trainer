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


# -----------------------------
# Map / scenario payload models
# -----------------------------

@dataclass(frozen=True, slots=True)
class MapNode:
    # Node circle center on a 960x540 canvas.
    pos: tuple[int, int]
    # Label box anchor (top-left) chosen so text never sits on the edge lines.
    label_anchor: tuple[int, int]


@dataclass(frozen=True, slots=True)
class MapEdge:
    a: int
    b: int


@dataclass(frozen=True, slots=True)
class MapTemplate:
    name: str
    nodes: tuple[MapNode, ...]
    edges: tuple[MapEdge, ...]


@dataclass(frozen=True, slots=True)
class RouteLeg:
    frm: int
    to: int
    distance: int  # NM or km, consistent per question


@dataclass(frozen=True, slots=True)
class AirborneScenario:
    template_name: str
    node_names: tuple[str, ...]
    # Edge distances aligned to template.edges order (same length).
    edge_distances: tuple[int, ...]
    route: tuple[int, ...]  # node indices along the route, inclusive endpoints
    legs: tuple[RouteLeg, ...]
    start_time_hhmm: str  # 4 digits
    speed_value: int
    speed_unit: str       # "kt" or "km/h"
    distance_unit: str    # "NM" or "km"
    fuel_burn_per_hr: int
    parcel_weight: int

    @property
    def parcel_weight_kg(self) -> int:
        return self.parcel_weight


# -----------------------------
# Time helpers (HHMM)
# -----------------------------

def _hhmm_str_to_minutes(hhmm: str) -> int:
    # Assumes caller validated exactly 4 digits.
    hh = int(hhmm[0:2])
    mm = int(hhmm[2:4])
    hh = hh % 24
    mm = mm % 60
    return hh * 60 + mm


def _minutes_to_hhmm_str(total_minutes: int) -> str:
    total_minutes %= 1440
    hh = total_minutes // 60
    mm = total_minutes % 60
    return f"{hh:02d}{mm:02d}"


def _wrap_minute_error(a_min: int, b_min: int) -> int:
    # Smallest absolute error on a circular 24h clock.
    diff = abs(a_min - b_min) % 1440
    return min(diff, 1440 - diff)


# -----------------------------
# Scoring model
# -----------------------------

class AirborneArrivalTimeScorer:
    """Partial credit for arrival time (HHMM).

    Rules from your spec:
    - Raw input must be exactly 4 digits, else score = 0.
    - Score is 1.0 if exact; linearly decays to 0.0 at 30 minutes error.
    - No negative scoring.
    """

    def __init__(self, *, zero_at_minutes: int = 30) -> None:
        self._zero_at = max(1, int(zero_at_minutes))

    def score(self, *, problem: Problem, user_answer: int, raw: str) -> float:
        raw = raw.strip()

        if len(raw) != 4 or not raw.isdigit():
            return 0.0

        expected = problem.payload
        if not isinstance(expected, AirborneScenario):
            # If payload is missing/malformed, fall back to strict equality.
            return 1.0 if user_answer == problem.answer else 0.0

        user_min = _hhmm_str_to_minutes(raw)
        correct_hhmm = arrival_time_hhmm(expected)
        correct_min = _hhmm_str_to_minutes(correct_hhmm)

        err = _wrap_minute_error(user_min, correct_min)
        if err >= self._zero_at:
            return 0.0
        return 1.0 - (err / self._zero_at)


# -----------------------------
# Generator
# -----------------------------

@dataclass(frozen=True, slots=True)
class AirborneNumericalConfig:
    scored_duration_s: float = 35.0 * 60.0
    practice_questions: int = 5
    # Partial credit hits 0 at this many minutes error.
    zero_score_at_minutes: int = 30


_TEMPLATES: tuple[MapTemplate, ...] = (
    # Template 1: simple 6-node mesh
    MapTemplate(
        name="T1",
        nodes=(
            MapNode(pos=(160, 140), label_anchor=(110, 80)),
            MapNode(pos=(360, 110), label_anchor=(330, 50)),
            MapNode(pos=(560, 140), label_anchor=(590, 90)),
            MapNode(pos=(240, 320), label_anchor=(170, 350)),
            MapNode(pos=(480, 320), label_anchor=(500, 350)),
            MapNode(pos=(720, 260), label_anchor=(750, 220)),
        ),
        edges=(
            MapEdge(0, 1),
            MapEdge(1, 2),
            MapEdge(0, 3),
            MapEdge(1, 3),
            MapEdge(1, 4),
            MapEdge(2, 4),
            MapEdge(4, 5),
            MapEdge(2, 5),
        ),
    ),
    # Template 2: 7-node ring + chords
    MapTemplate(
        name="T2",
        nodes=(
            MapNode(pos=(170, 120), label_anchor=(110, 60)),
            MapNode(pos=(360, 90), label_anchor=(330, 30)),
            MapNode(pos=(560, 120), label_anchor=(600, 70)),
            MapNode(pos=(680, 250), label_anchor=(710, 210)),
            MapNode(pos=(560, 380), label_anchor=(600, 390)),
            MapNode(pos=(360, 410), label_anchor=(300, 450)),
            MapNode(pos=(170, 360), label_anchor=(100, 390)),
        ),
        edges=(
            MapEdge(0, 1),
            MapEdge(1, 2),
            MapEdge(2, 3),
            MapEdge(3, 4),
            MapEdge(4, 5),
            MapEdge(5, 6),
            MapEdge(6, 0),
            MapEdge(1, 5),
            MapEdge(2, 4),
        ),
    ),
    # Template 3: 6-node “S” path with cross-links
    MapTemplate(
        name="T3",
        nodes=(
            MapNode(pos=(150, 120), label_anchor=(90, 60)),
            MapNode(pos=(310, 170), label_anchor=(260, 110)),
            MapNode(pos=(470, 120), label_anchor=(500, 70)),
            MapNode(pos=(630, 170), label_anchor=(660, 130)),
            MapNode(pos=(470, 320), label_anchor=(500, 350)),
            MapNode(pos=(310, 370), label_anchor=(240, 400)),
        ),
        edges=(
            MapEdge(0, 1),
            MapEdge(1, 2),
            MapEdge(2, 3),
            MapEdge(3, 4),
            MapEdge(4, 5),
            MapEdge(5, 0),
            MapEdge(1, 5),
            MapEdge(2, 4),
        ),
    ),
    # Template 4: 8-node grid-ish
    MapTemplate(
        name="T4",
        nodes=(
            MapNode(pos=(170, 140), label_anchor=(110, 80)),
            MapNode(pos=(350, 140), label_anchor=(300, 80)),
            MapNode(pos=(530, 140), label_anchor=(560, 90)),
            MapNode(pos=(710, 140), label_anchor=(740, 100)),
            MapNode(pos=(170, 320), label_anchor=(110, 350)),
            MapNode(pos=(350, 320), label_anchor=(300, 350)),
            MapNode(pos=(530, 320), label_anchor=(560, 350)),
            MapNode(pos=(710, 320), label_anchor=(740, 350)),
        ),
        edges=(
            MapEdge(0, 1),
            MapEdge(1, 2),
            MapEdge(2, 3),
            MapEdge(4, 5),
            MapEdge(5, 6),
            MapEdge(6, 7),
            MapEdge(0, 4),
            MapEdge(1, 5),
            MapEdge(2, 6),
            MapEdge(3, 7),
            MapEdge(1, 6),
        ),
    ),
    # Template 5: 7-node hub/spokes
    MapTemplate(
        name="T5",
        nodes=(
            MapNode(pos=(480, 220), label_anchor=(500, 170)),  # hub
            MapNode(pos=(240, 140), label_anchor=(170, 80)),
            MapNode(pos=(720, 140), label_anchor=(750, 100)),
            MapNode(pos=(200, 320), label_anchor=(120, 350)),
            MapNode(pos=(760, 320), label_anchor=(790, 350)),
            MapNode(pos=(360, 400), label_anchor=(300, 450)),
            MapNode(pos=(600, 400), label_anchor=(620, 450)),
        ),
        edges=(
            MapEdge(0, 1),
            MapEdge(0, 2),
            MapEdge(0, 3),
            MapEdge(0, 4),
            MapEdge(0, 5),
            MapEdge(0, 6),
            MapEdge(1, 3),
            MapEdge(2, 4),
            MapEdge(5, 6),
        ),
    ),
)

# Exported for UI rendering (fixed geometry; deterministic per scenario).
TEMPLATES_BY_NAME: dict[str, MapTemplate] = {t.name: t for t in _TEMPLATES}


_CODES: tuple[str, ...] = (
    # Short, uniform-width-ish codes (avoid clipping; UI will still draw boxed labels).
    "ALP", "BRV", "CHL", "DLT", "ECH", "FOX", "GLF", "HIL", "IND", "JUL",
    "KIL", "LIM", "MKE", "NVR", "OSC", "PAP", "QBC", "ROM", "SIE", "TNG",
    "UNF", "VIC", "WHS", "XRY", "YKR", "ZUL",
)


def _route_minutes(s: AirborneScenario) -> int:
    # Speed is constant for now; per-leg time uses distance/speed.
    if s.speed_value <= 0:
        return 0
    minutes = 0
    for leg in s.legs:
        minutes += round_half_up((leg.distance / s.speed_value) * 60.0)
    return minutes


def arrival_time_hhmm(s: AirborneScenario) -> str:
    """Compute arrival time as HHMM, using round-half-up for travel time."""

    return _minutes_to_hhmm_str(_hhmm_str_to_minutes(s.start_time_hhmm) + _route_minutes(s))


class AirborneNumericalGenerator:
    """Generates arrival-time questions with 1–3 vias (2–4 legs), seeded and deterministic."""

    def __init__(self, *, seed: int) -> None:
        self._rng = SeededRng(seed)

    def next_problem(self, *, difficulty: float) -> Problem:
        difficulty = max(0.0, min(1.0, difficulty))

        template = _TEMPLATES[self._rng.randint(0, len(_TEMPLATES) - 1)]
        node_names = self._assign_node_names(len(template.nodes))

        speed_unit, dist_unit, speed = self._pick_speed_units_and_value(difficulty)

        # Vias: 1..3 (legs 2..4) with difficulty increasing max.
        max_vias = 1 if difficulty < 0.4 else 2 if difficulty < 0.7 else 3
        vias = self._rng.randint(1, max_vias)
        legs_count = vias + 1

        route = self._pick_route(template, legs_count)
        legs, edge_distances = self._assign_edge_distances(template, route, speed, dist_unit, difficulty)

        start_time = self._pick_start_time()

        scenario = AirborneScenario(
            template_name=template.name,
            node_names=node_names,
            edge_distances=edge_distances,
            route=route,
            legs=legs,
            start_time_hhmm=start_time,
            speed_value=speed,
            speed_unit=speed_unit,
            distance_unit=dist_unit,
            fuel_burn_per_hr=self._pick_fuel_burn(difficulty),
            parcel_weight=self._pick_parcel_weight(difficulty),
        )

        # Correct arrival time as HHMM, stored as int for core consistency.
        arrival_hhmm = arrival_time_hhmm(scenario)
        answer_int = int(arrival_hhmm)  # leading zeros irrelevant; scorer uses raw formatting

        dest_name = node_names[route[-1]]
        prompt = f"ARRIVAL TIME at {dest_name} (HHMM). Enter 4 digits:"

        return Problem(prompt=prompt, answer=answer_int, tolerance=0, payload=scenario)

    def _assign_node_names(self, n: int) -> tuple[str, ...]:
        # Deterministic selection without replacement.
        pool = list(_CODES)
        out: list[str] = []
        for _ in range(n):
            idx = self._rng.randint(0, len(pool) - 1)
            out.append(pool.pop(idx))
        return tuple(out)

    def _pick_speed_units_and_value(self, difficulty: float) -> tuple[str, str, int]:
        # Keep consistent units within a question.
        use_nm = self._rng.uniform(0.0, 1.0) < 0.7  # mostly nautical-style
        if use_nm:
            # Choose speeds with integer NM/min (multiples of 60).
            choices = [240, 300, 360, 420, 480]
            speed = choices[self._rng.randint(0, len(choices) - 1)]
            return "kt", "NM", speed
        # Metric variant
        choices = [180, 240, 300, 360, 420]
        speed = choices[self._rng.randint(0, len(choices) - 1)]
        return "km/h", "km", speed

    def _pick_route(self, template: MapTemplate, legs_count: int) -> tuple[int, ...]:
        # Random walk without revisiting nodes.
        adj = _adjacency(template)
        for _ in range(200):
            start = self._rng.randint(0, len(template.nodes) - 1)
            path = [start]
            while len(path) < (legs_count + 1):
                cur = path[-1]
                nbrs = [x for x in adj[cur] if x not in path]
                if not nbrs:
                    break
                nxt = nbrs[self._rng.randint(0, len(nbrs) - 1)]
                path.append(nxt)
            if len(path) == legs_count + 1:
                return tuple(path)
        # Fallback: just pick first available chain.
        nodes = list(range(len(template.nodes)))
        return tuple(nodes[: legs_count + 1])

    def _assign_edge_distances(
        self,
        template: MapTemplate,
        route: tuple[int, ...],
        speed: int,
        dist_unit: str,
        difficulty: float,
    ) -> tuple[tuple[RouteLeg, ...], tuple[int, ...]]:
        # Build map from undirected edge key to index in template.edges.
        edge_index: dict[tuple[int, int], int] = {}
        for i, e in enumerate(template.edges):
            key = (e.a, e.b) if e.a < e.b else (e.b, e.a)
            edge_index[key] = i

        # Start with baseline distances for all edges.
        # Use multiples of speed/60 to keep per-leg minutes simple when we want.
        per_min = max(1, speed // 60)
        min_min = 6 if difficulty < 0.5 else 8
        max_min = lerp_int(18, 28, difficulty)

        dists = [per_min * self._rng.randint(min_min, max_min) for _ in template.edges]

        legs: list[RouteLeg] = []
        # Force route edges to match chosen per-leg times (integer minutes).
        for i in range(len(route) - 1):
            a = route[i]
            b = route[i + 1]
            key = (a, b) if a < b else (b, a)
            idx = edge_index.get(key)
            if idx is None:
                # Should not happen; if it does, keep baseline and proceed.
                dist = dists[0]
            else:
                minutes = self._rng.randint(min_min, max_min)
                dist = per_min * minutes
                dists[idx] = dist
            legs.append(RouteLeg(frm=a, to=b, distance=dist))

        return tuple(legs), tuple(dists)

    def _pick_start_time(self) -> str:
        # 24h clock; keep minutes on 5-min steps.
        hh = self._rng.randint(6, 21)
        mm = self._rng.randint(0, 11) * 5
        return f"{hh:02d}{mm:02d}"

    def _pick_fuel_burn(self, difficulty: float) -> int:
        lo = 900
        hi = lerp_int(1800, 5200, difficulty)
        burn = self._rng.randint(lo, hi)
        if difficulty < 0.4:
            burn = int(round(burn / 100) * 100)
        return burn

    def _pick_parcel_weight(self, difficulty: float) -> int:
        lo = 50
        hi = lerp_int(200, 1200, difficulty)
        w = self._rng.randint(lo, hi)
        if difficulty < 0.4:
            w = int(round(w / 10) * 10)
        return w


def _adjacency(t: MapTemplate) -> dict[int, list[int]]:
    adj: dict[int, list[int]] = {i: [] for i in range(len(t.nodes))}
    for e in t.edges:
        adj[e.a].append(e.b)
        adj[e.b].append(e.a)
    return adj


def build_airborne_numerical_test(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    config: AirborneNumericalConfig | None = None,
) -> TimedTextInputTest:
    cfg = config or AirborneNumericalConfig()

    instructions = [
        "Airborne Numerical Test",
        "",
        "Compute ARRIVAL TIME using the map and info panels.",
        "Answers must be 4 digits (HHMM, 24-hour clock).",
        "Incorrect answers are not penalized; they just don't score.",
        "",
        "Controls (hold keys; they do NOT type into the answer):",
        "- Hold A: Introduction overlay",
        "- Hold S: Speed & Fuel overlay",
        "- Hold D: Speed & Parcel overlay",
        "- Hold F: Show distances on the map",
        "",
        "Type digits for your answer, then press Enter to submit.",
        "No backspace (training matches test constraints).",
    ]

    generator = AirborneNumericalGenerator(seed=seed)
    scorer: AnswerScorer = AirborneArrivalTimeScorer(zero_at_minutes=cfg.zero_score_at_minutes)

    return TimedTextInputTest(
        title="Airborne Numerical Test",
        instructions=instructions,
        generator=generator,
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        practice_questions=cfg.practice_questions,
        scored_duration_s=cfg.scored_duration_s,
        scorer=scorer,
    )

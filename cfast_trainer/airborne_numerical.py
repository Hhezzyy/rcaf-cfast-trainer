from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

from .adaptive_difficulty import difficulty_level_for_ratio, difficulty_profile_for_code
from .clock import Clock
from .content_variants import stable_variant_id
from .cognitive_core import (
    AnswerScorer,
    Problem,
    SeededRng,
    TimedTextInputTest,
    round_half_up,
)

# -----------------------------------------------------------------------------
# Airborne Numerical (training recreation)
# - Deterministic generation (seeded RNG)
# - 4-digit input UI is preserved; all generated answers fit within it
# - Some reference pages can be charts instead of tables; chart-driven questions
#   carry a tolerance so estimated reads are still marked fairly
# -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RouteLeg:
    frm: int
    to: int
    distance: int


@dataclass(frozen=True, slots=True)
class UnitProfile:
    speed_unit: str
    distance_unit: str
    speed_minutes: int
    fuel_unit: str
    fuel_minutes: int
    distance_range: tuple[int, int]
    speed_max_range: tuple[int, int]
    speed_floor: int
    speed_drop_range: tuple[int, int]
    speed_neighbor_step: int
    fuel_base_range: tuple[int, int]
    fuel_factor_range: tuple[float, float]
    fuel_minimum: int
    start_fuel_range: tuple[int, int]
    start_fuel_step: int
    speed_chart_steps: tuple[int, ...]
    fuel_chart_steps: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class AirborneScenario:
    template_name: str
    node_names: tuple[str, ...]
    edge_distances: tuple[int, ...]  # aligned to template edges
    route: tuple[int, ...]  # node indices
    legs: tuple[RouteLeg, ...]

    start_time_hhmm: str
    given_time_label: str
    given_time_hhmm: str
    speed_value: int
    speed_unit: str
    speed_minutes: int
    distance_unit: str

    fuel_burn_per_hr: int
    fuel_unit: str
    fuel_minutes: int
    parcel_weight: int
    start_fuel_liters: int

    route_distance_total: int
    route_travel_minutes: int
    arrival_time_hhmm: str
    empty_time_hhmm: str
    fuel_used_on_route: int

    # overlay “menus” (table or chart)
    weight_speed_table: tuple[tuple[int, int], ...]  # (kg, speed)
    speed_fuel_table: tuple[tuple[int, int], ...]  # (speed, fuel burn)
    parcel_reference_format: str  # "table" | "chart"
    fuel_reference_format: str  # "table" | "chart"
    parcel_chart_step: int
    fuel_chart_step: int

    # per-problem label (helps UI/tests)
    question_kind: str
    target_label: str
    answer_format: str  # "hhmm" | "number"
    answer_label: str
    answer_unit_label: str
    answer_digits: int
    content_family: str = ""
    variant_id: str = ""
    content_pack: str = ""
    input_digits: int = 4

    # convenience aliases (keeps older UI code working)
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


@dataclass(frozen=True, slots=True)
class AirborneDifficultyProfile:
    family: str
    level: int
    question_kinds: tuple[tuple[str, float], ...]
    min_legs: int = 1
    max_legs: int = 4
    speed_minutes: tuple[int, ...] = (60, 1)
    fuel_minutes: tuple[int, ...] = (60, 1)
    parcel_reference_formats: tuple[str, ...] = ("table", "chart")
    fuel_reference_formats: tuple[str, ...] = ("table", "chart")


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
        nodes=(
            (0.10, 0.20),
            (0.35, 0.15),
            (0.60, 0.12),
            (0.85, 0.20),
            (0.25, 0.55),
            (0.55, 0.55),
            (0.85, 0.60),
        ),
        edges=((0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 2), (5, 6), (6, 3)),
    ),
    MapTemplate(
        name="T6",
        nodes=((0.12, 0.18), (0.40, 0.12), (0.70, 0.15), (0.20, 0.52), (0.50, 0.55), (0.82, 0.55)),
        edges=((0, 1), (1, 2), (0, 3), (3, 4), (4, 2), (4, 5)),
    ),
    MapTemplate(
        name="T7",
        nodes=((0.08, 0.22), (0.28, 0.10), (0.52, 0.18), (0.80, 0.12), (0.18, 0.55), (0.44, 0.62), (0.76, 0.58)),
        edges=((0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 2), (5, 6), (6, 3)),
    ),
    MapTemplate(
        name="T8",
        nodes=((0.12, 0.18), (0.34, 0.26), (0.62, 0.12), (0.86, 0.22), (0.24, 0.58), (0.54, 0.50), (0.80, 0.60)),
        edges=((0, 1), (1, 2), (2, 3), (1, 4), (4, 5), (5, 2), (5, 6), (6, 3)),
    ),
    MapTemplate(
        name="T9",
        nodes=((0.10, 0.18), (0.32, 0.10), (0.58, 0.14), (0.84, 0.22), (0.18, 0.50), (0.44, 0.60), (0.74, 0.56)),
        edges=((0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 2), (5, 6), (6, 3), (2, 6)),
    ),
    MapTemplate(
        name="T10",
        nodes=((0.08, 0.16), (0.28, 0.24), (0.52, 0.10), (0.76, 0.18), (0.16, 0.52), (0.44, 0.46), (0.68, 0.62), (0.90, 0.52)),
        edges=((0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 2), (5, 6), (6, 3), (6, 7), (3, 7)),
    ),
)

TEMPLATES_BY_NAME = {t.name: t for t in TEMPLATES}

UNIT_PROFILES: tuple[UnitProfile, ...] = (
    UnitProfile(
        speed_unit="km/h",
        distance_unit="km",
        speed_minutes=60,
        fuel_unit="L/hr",
        fuel_minutes=60,
        distance_range=(40, 180),
        speed_max_range=(360, 520),
        speed_floor=220,
        speed_drop_range=(80, 170),
        speed_neighbor_step=20,
        fuel_base_range=(800, 1600),
        fuel_factor_range=(3.8, 6.3),
        fuel_minimum=400,
        start_fuel_range=(2200, 9000),
        start_fuel_step=50,
        speed_chart_steps=(10, 20, 25, 50),
        fuel_chart_steps=(100, 200, 250, 500),
    ),
    UnitProfile(
        speed_unit="knots",
        distance_unit="NM",
        speed_minutes=60,
        fuel_unit="L/hr",
        fuel_minutes=60,
        distance_range=(25, 120),
        speed_max_range=(190, 310),
        speed_floor=120,
        speed_drop_range=(40, 90),
        speed_neighbor_step=10,
        fuel_base_range=(600, 1200),
        fuel_factor_range=(3.0, 5.2),
        fuel_minimum=350,
        start_fuel_range=(1800, 7200),
        start_fuel_step=50,
        speed_chart_steps=(5, 10, 20, 25),
        fuel_chart_steps=(50, 100, 200, 250),
    ),
    UnitProfile(
        speed_unit="km/30min",
        distance_unit="km",
        speed_minutes=30,
        fuel_unit="L/30min",
        fuel_minutes=30,
        distance_range=(20, 110),
        speed_max_range=(110, 240),
        speed_floor=70,
        speed_drop_range=(25, 70),
        speed_neighbor_step=10,
        fuel_base_range=(180, 520),
        fuel_factor_range=(1.8, 3.2),
        fuel_minimum=100,
        start_fuel_range=(700, 3600),
        start_fuel_step=25,
        speed_chart_steps=(5, 10, 20),
        fuel_chart_steps=(25, 50, 100),
    ),
    UnitProfile(
        speed_unit="NM/30min",
        distance_unit="NM",
        speed_minutes=30,
        fuel_unit="L/30min",
        fuel_minutes=30,
        distance_range=(15, 95),
        speed_max_range=(75, 180),
        speed_floor=50,
        speed_drop_range=(18, 55),
        speed_neighbor_step=5,
        fuel_base_range=(150, 420),
        fuel_factor_range=(1.5, 2.8),
        fuel_minimum=80,
        start_fuel_range=(600, 3200),
        start_fuel_step=25,
        speed_chart_steps=(5, 10, 20),
        fuel_chart_steps=(20, 50, 100),
    ),
    UnitProfile(
        speed_unit="mi/min",
        distance_unit="miles",
        speed_minutes=1,
        fuel_unit="L/min",
        fuel_minutes=1,
        distance_range=(12, 60),
        speed_max_range=(6, 11),
        speed_floor=3,
        speed_drop_range=(2, 5),
        speed_neighbor_step=1,
        fuel_base_range=(3, 8),
        fuel_factor_range=(0.5, 1.1),
        fuel_minimum=2,
        start_fuel_range=(120, 650),
        start_fuel_step=5,
        speed_chart_steps=(1, 2),
        fuel_chart_steps=(1, 2, 5),
    ),
    UnitProfile(
        speed_unit="km/min",
        distance_unit="km",
        speed_minutes=1,
        fuel_unit="L/min",
        fuel_minutes=1,
        distance_range=(18, 90),
        speed_max_range=(7, 13),
        speed_floor=4,
        speed_drop_range=(2, 6),
        speed_neighbor_step=1,
        fuel_base_range=(4, 10),
        fuel_factor_range=(0.6, 1.3),
        fuel_minimum=2,
        start_fuel_range=(140, 720),
        start_fuel_step=5,
        speed_chart_steps=(1, 2),
        fuel_chart_steps=(1, 2, 5),
    ),
)

_NODE_CODES = (
    "ALP",
    "BRV",
    "CHL",
    "DLT",
    "ECO",
    "FOX",
    "GLF",
    "HIL",
    "IND",
    "JUL",
    "KIL",
    "MKE",
    "NVR",
    "OPS",
    "PRM",
    "QNT",
    "RDL",
    "SNR",
    "TNG",
    "ULS",
    "VCT",
    "WHS",
    "XRY",
    "YLD",
    "ZUL",
)

QUESTION_KINDS: tuple[tuple[str, float], ...] = (
    ("arrival_time", 0.15),
    ("takeoff_time", 0.12),
    ("empty_time", 0.12),
    ("fuel_endurance", 0.11),
    ("fuel_burned", 0.14),
    ("distance_travelled", 0.12),
    ("parcel_weight", 0.12),
    ("parcel_effect", 0.12),
)

PRACTICE_KIND_ORDER: tuple[str, ...] = (
    "arrival_time",
    "takeoff_time",
    "empty_time",
    "fuel_endurance",
    "fuel_burned",
    "distance_travelled",
    "parcel_weight",
    "parcel_effect",
)

PRACTICE_REFERENCE_PAIRS: tuple[tuple[str, str], ...] = (
    ("table", "table"),
    ("chart", "table"),
    ("table", "chart"),
    ("chart", "chart"),
    ("table", "chart"),
    ("chart", "table"),
)

_PROMPT_VARIANTS: dict[str, tuple[str, ...]] = {
    "arrival_time": (
        "ARRIVAL TIME at {dest} (HHMM). Enter 4 digits:",
        "ROUTE ARRIVAL for {dest} (HHMM). Enter 4 digits:",
        "LANDING TIME at {dest} (HHMM). Enter 4 digits:",
    ),
    "takeoff_time": (
        "TAKE OFF TIME for {dest} (HHMM). Enter 4 digits:",
        "DEPARTURE TIME for {dest} (HHMM). Enter 4 digits:",
        "LAUNCH TIME needed for {dest} (HHMM). Enter 4 digits:",
    ),
    "empty_time": (
        "EMPTY TIME (HHMM). Enter 4 digits:",
        "TANKS DRY at what time (HHMM)? Enter 4 digits:",
        "FUEL EMPTY TIME (HHMM). Enter 4 digits:",
    ),
    "fuel_endurance": (
        "FUEL ENDURANCE (minutes). Enter 4 digits:",
        "USABLE ENDURANCE before dry tanks (minutes). Enter 4 digits:",
        "MINUTES OF FUEL REMAINING. Enter 4 digits:",
    ),
    "fuel_burned": (
        "FUEL BURNED to {dest} ({fuel_value_unit}). Enter 4 digits:",
        "FUEL USED on route to {dest} ({fuel_value_unit}). Enter 4 digits:",
        "ROUTE FUEL CONSUMPTION to {dest} ({fuel_value_unit}). Enter 4 digits:",
    ),
    "distance_travelled": (
        "DISTANCE TRAVELLED to {dest} ({distance_unit}). Enter 4 digits:",
        "ROUTE DISTANCE to {dest} ({distance_unit}). Enter 4 digits:",
        "TOTAL DISTANCE FLOWN to {dest} ({distance_unit}). Enter 4 digits:",
    ),
    "parcel_weight": (
        "PARCEL WEIGHT for {dest} (kg). Enter 4 digits:",
        "PAYLOAD WEIGHT for {dest} (kg). Enter 4 digits:",
        "LOAD MASS for {dest} (kg). Enter 4 digits:",
    ),
    "parcel_effect": (
        "PARCEL EFFECT on speed ({speed_unit}). Enter 4 digits:",
        "PAYLOAD SPEED LOSS ({speed_unit}). Enter 4 digits:",
        "SPEED PENALTY from parcel load ({speed_unit}). Enter 4 digits:",
    ),
}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _difficulty_to_level(difficulty: float) -> int:
    return difficulty_level_for_ratio("airborne_numerical", _clamp01(difficulty))


def _normalize_question_weights(
    options: tuple[tuple[str, float], ...],
) -> tuple[tuple[str, float], ...]:
    if not options:
        raise ValueError("question weight profile must not be empty")
    total = sum(max(0.0, float(weight)) for _label, weight in options)
    if total <= 0.0:
        uniform = 1.0 / float(len(options))
        return tuple((label, uniform) for label, _weight in options)
    return tuple((label, max(0.0, float(weight)) / total) for label, weight in options)


def build_ant_airborne_difficulty_profile(
    level: int,
    *,
    family: str = "full",
) -> AirborneDifficultyProfile:
    clamped = max(1, min(10, int(level)))
    token = str(family).strip().lower() or "full"
    shared = difficulty_profile_for_code("airborne_numerical", clamped, mode="build")
    allow_minute_resolution = shared.axes.time_pressure >= 0.42
    allow_chart_references = shared.axes.source_integration_depth >= 0.28
    prefer_chart_references = shared.axes.source_integration_depth >= 0.58
    high_leg_floor = 3 if shared.axes.content_complexity >= 0.72 else 1
    advanced_speed_minutes = (60, 1) if allow_minute_resolution else (60,)
    advanced_fuel_minutes = (60, 1) if allow_minute_resolution else (60,)
    advanced_parcel_formats = (
        ("chart",)
        if prefer_chart_references
        else ("table", "chart") if allow_chart_references else ("table",)
    )
    advanced_fuel_formats = ("table", "chart") if allow_chart_references else ("table",)

    if token == "route_time":
        if clamped <= 2:
            kinds = (("arrival_time", 1.0),)
            return AirborneDifficultyProfile(
                family=token,
                level=clamped,
                question_kinds=kinds,
                max_legs=2,
                speed_minutes=(60,),
                parcel_reference_formats=("table",),
            )
        if clamped <= 4:
            kinds = _normalize_question_weights((("arrival_time", 0.7), ("takeoff_time", 0.3)))
            return AirborneDifficultyProfile(
                family=token,
                level=clamped,
                question_kinds=kinds,
                max_legs=3,
                speed_minutes=(60,),
                parcel_reference_formats=("table",),
            )
        if clamped <= 6:
            kinds = _normalize_question_weights((("arrival_time", 0.55), ("takeoff_time", 0.45)))
            return AirborneDifficultyProfile(
                family=token,
                level=clamped,
                question_kinds=kinds,
                max_legs=4,
                speed_minutes=(60,),
                parcel_reference_formats=("table",),
            )
        return AirborneDifficultyProfile(
            family=token,
            level=clamped,
            question_kinds=_normalize_question_weights(
                (("arrival_time", 0.5), ("takeoff_time", 0.5))
            ),
            min_legs=high_leg_floor,
            max_legs=4,
            speed_minutes=advanced_speed_minutes,
            parcel_reference_formats=advanced_parcel_formats,
        )

    if token == "endurance":
        if clamped <= 2:
            kinds = (("fuel_endurance", 1.0),)
            return AirborneDifficultyProfile(
                family=token,
                level=clamped,
                question_kinds=kinds,
                max_legs=2,
                fuel_minutes=(60,),
                fuel_reference_formats=("table",),
            )
        if clamped <= 4:
            kinds = _normalize_question_weights((("fuel_endurance", 0.65), ("empty_time", 0.35)))
            return AirborneDifficultyProfile(
                family=token,
                level=clamped,
                question_kinds=kinds,
                max_legs=3,
                fuel_minutes=(60,),
                fuel_reference_formats=("table",),
            )
        if clamped <= 6:
            kinds = _normalize_question_weights((("fuel_endurance", 0.5), ("empty_time", 0.5)))
            return AirborneDifficultyProfile(
                family=token,
                level=clamped,
                question_kinds=kinds,
                max_legs=3,
                fuel_minutes=(60,),
                fuel_reference_formats=("table", "chart"),
            )
        return AirborneDifficultyProfile(
            family=token,
            level=clamped,
            question_kinds=_normalize_question_weights((("fuel_endurance", 0.45), ("empty_time", 0.55))),
            max_legs=4,
            fuel_minutes=advanced_fuel_minutes,
            fuel_reference_formats=advanced_fuel_formats,
        )

    if token == "fuel_burn":
        if clamped <= 2:
            return AirborneDifficultyProfile(
                family=token,
                level=clamped,
                question_kinds=(("fuel_burned", 1.0),),
                max_legs=2,
                speed_minutes=(60,),
                fuel_minutes=(60,),
                parcel_reference_formats=("table",),
                fuel_reference_formats=("table",),
            )
        if clamped <= 4:
            return AirborneDifficultyProfile(
                family=token,
                level=clamped,
                question_kinds=(("fuel_burned", 1.0),),
                max_legs=3,
                speed_minutes=(60,),
                fuel_minutes=(60,),
                parcel_reference_formats=("table",),
                fuel_reference_formats=("table",),
            )
        if clamped <= 6:
            return AirborneDifficultyProfile(
                family=token,
                level=clamped,
                question_kinds=(("fuel_burned", 1.0),),
                max_legs=3,
                speed_minutes=(60,),
                fuel_minutes=(60,),
                parcel_reference_formats=("table",),
                fuel_reference_formats=("table",),
            )
        return AirborneDifficultyProfile(
            family=token,
            level=clamped,
            question_kinds=(("fuel_burned", 1.0),),
            min_legs=high_leg_floor,
            max_legs=4,
            speed_minutes=advanced_speed_minutes,
            fuel_minutes=advanced_fuel_minutes,
            parcel_reference_formats=advanced_parcel_formats,
            fuel_reference_formats=advanced_fuel_formats,
        )

    if token == "distance":
        if clamped <= 2:
            return AirborneDifficultyProfile(
                family=token,
                level=clamped,
                question_kinds=(("distance_travelled", 1.0),),
                max_legs=2,
                speed_minutes=(60,),
                fuel_minutes=(60,),
            )
        if clamped <= 4:
            return AirborneDifficultyProfile(
                family=token,
                level=clamped,
                question_kinds=(("distance_travelled", 1.0),),
                max_legs=3,
                speed_minutes=(60,),
                fuel_minutes=(60,),
            )
        if clamped <= 6:
            return AirborneDifficultyProfile(
                family=token,
                level=clamped,
                question_kinds=(("distance_travelled", 1.0),),
                max_legs=4,
                speed_minutes=(60,),
                fuel_minutes=(60,),
            )
        return AirborneDifficultyProfile(
            family=token,
            level=clamped,
            question_kinds=(("distance_travelled", 1.0),),
            min_legs=max(1, high_leg_floor + 1),
            max_legs=4,
            speed_minutes=advanced_speed_minutes,
            fuel_minutes=advanced_fuel_minutes,
        )

    if token == "payload":
        if clamped <= 2:
            return AirborneDifficultyProfile(
                family=token,
                level=clamped,
                question_kinds=(("parcel_weight", 1.0),),
                max_legs=2,
                speed_minutes=(60,),
                parcel_reference_formats=("table",),
            )
        if clamped <= 4:
            kinds = _normalize_question_weights((("parcel_weight", 0.65), ("parcel_effect", 0.35)))
            return AirborneDifficultyProfile(
                family=token,
                level=clamped,
                question_kinds=kinds,
                max_legs=3,
                speed_minutes=(60,),
                parcel_reference_formats=("table",),
            )
        if clamped <= 6:
            kinds = _normalize_question_weights((("parcel_weight", 0.55), ("parcel_effect", 0.45)))
            return AirborneDifficultyProfile(
                family=token,
                level=clamped,
                question_kinds=kinds,
                max_legs=3,
                speed_minutes=(60,),
                parcel_reference_formats=("table", "chart"),
            )
        return AirborneDifficultyProfile(
            family=token,
            level=clamped,
            question_kinds=_normalize_question_weights((("parcel_weight", 0.45), ("parcel_effect", 0.55))),
            max_legs=4,
            speed_minutes=advanced_speed_minutes,
            parcel_reference_formats=advanced_parcel_formats,
        )

    if token != "full":
        raise ValueError(f"Unknown airborne difficulty family: {family}")

    if clamped <= 2:
        kinds = _normalize_question_weights(
            (
                ("arrival_time", 0.18),
                ("takeoff_time", 0.08),
                ("empty_time", 0.08),
                ("fuel_endurance", 0.18),
                ("fuel_burned", 0.10),
                ("distance_travelled", 0.16),
                ("parcel_weight", 0.14),
                ("parcel_effect", 0.08),
            )
        )
        return AirborneDifficultyProfile(
            family=token,
            level=clamped,
            question_kinds=kinds,
            max_legs=2,
            speed_minutes=(60,),
            fuel_minutes=(60,),
            parcel_reference_formats=("table",),
            fuel_reference_formats=("table",),
        )
    if clamped <= 4:
        kinds = _normalize_question_weights(
            (
                ("arrival_time", 0.16),
                ("takeoff_time", 0.10),
                ("empty_time", 0.10),
                ("fuel_endurance", 0.14),
                ("fuel_burned", 0.12),
                ("distance_travelled", 0.14),
                ("parcel_weight", 0.12),
                ("parcel_effect", 0.12),
            )
        )
        return AirborneDifficultyProfile(
            family=token,
            level=clamped,
            question_kinds=kinds,
            max_legs=3,
            speed_minutes=(60,),
            fuel_minutes=(60,),
            parcel_reference_formats=("table",),
            fuel_reference_formats=("table",),
        )
    if clamped <= 6:
        return AirborneDifficultyProfile(
            family=token,
            level=clamped,
            question_kinds=_normalize_question_weights(QUESTION_KINDS),
            max_legs=4,
            speed_minutes=(60,),
            fuel_minutes=(60,),
            parcel_reference_formats=("table", "chart"),
            fuel_reference_formats=("table",),
        )
    if clamped <= 8:
        kinds = _normalize_question_weights(
            (
                ("arrival_time", 0.13),
                ("takeoff_time", 0.13),
                ("empty_time", 0.13),
                ("fuel_endurance", 0.11),
                ("fuel_burned", 0.13),
                ("distance_travelled", 0.12),
                ("parcel_weight", 0.11),
                ("parcel_effect", 0.14),
            )
        )
        return AirborneDifficultyProfile(
            family=token,
            level=clamped,
            question_kinds=kinds,
            max_legs=4,
            speed_minutes=advanced_speed_minutes,
            fuel_minutes=advanced_fuel_minutes,
            parcel_reference_formats=advanced_parcel_formats,
            fuel_reference_formats=advanced_fuel_formats,
        )
    return AirborneDifficultyProfile(
        family=token,
        level=clamped,
        question_kinds=_normalize_question_weights(QUESTION_KINDS),
        max_legs=4,
        speed_minutes=advanced_speed_minutes,
        fuel_minutes=advanced_fuel_minutes,
        parcel_reference_formats=advanced_parcel_formats,
        fuel_reference_formats=advanced_fuel_formats,
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


def _minute_diff_hhmm(a_hhmm: int, b_hhmm: int) -> int:
    a = _hhmm_str_to_minutes(f"{int(a_hhmm):04d}")
    b = _hhmm_str_to_minutes(f"{int(b_hhmm):04d}")
    diff = abs(a - b)
    return min(diff, 24 * 60 - diff)


def _pick_node_names(rng: SeededRng, n: int) -> tuple[str, ...]:
    picks = rng.sample(list(_NODE_CODES), k=n)
    return tuple(picks)


def _weighted_choice(rng: SeededRng, options: tuple[tuple[str, float], ...]) -> str:
    ticket = rng.random()
    total = 0.0
    for label, weight in options:
        total += weight
        if ticket <= total:
            return label
    return options[-1][0]


def _airborne_content_family(question_kind: str) -> str:
    if question_kind in {"arrival_time", "takeoff_time"}:
        return "route_time"
    if question_kind in {"empty_time", "fuel_endurance"}:
        return "endurance"
    if question_kind == "fuel_burned":
        return "fuel_burn"
    if question_kind == "distance_travelled":
        return "distance"
    return "payload"


def _airborne_prompt(
    rng: SeededRng,
    *,
    question_kind: str,
    dest: str,
    unit_profile: UnitProfile,
) -> str:
    variants = _PROMPT_VARIANTS[question_kind]
    template = str(rng.choice(variants))
    return template.format(
        dest=dest,
        distance_unit=unit_profile.distance_unit,
        speed_unit=unit_profile.speed_unit,
        fuel_value_unit=unit_profile.fuel_unit.split("/")[0],
    )


def _scenario_matches_profile(
    scenario: AirborneScenario,
    profile: AirborneDifficultyProfile,
) -> bool:
    allowed_kinds = {kind for kind, _weight in profile.question_kinds}
    if scenario.question_kind not in allowed_kinds:
        return False

    legs = len(scenario.legs)
    if legs < profile.min_legs or legs > profile.max_legs:
        return False
    if scenario.speed_minutes not in profile.speed_minutes:
        return False
    if scenario.fuel_minutes not in profile.fuel_minutes:
        return False
    if scenario.parcel_reference_format not in profile.parcel_reference_formats:
        return False
    if scenario.fuel_reference_format not in profile.fuel_reference_formats:
        return False

    if profile.family == "route_time" and profile.level >= 9:
        return legs >= 3 and (
            scenario.parcel_reference_format == "chart" or scenario.speed_minutes != 60
        )

    if profile.family == "endurance" and profile.level >= 9:
        return scenario.fuel_reference_format == "chart" or scenario.fuel_minutes != 60

    if profile.family == "fuel_burn" and profile.level >= 9:
        has_chart = (
            scenario.fuel_reference_format == "chart"
            or scenario.parcel_reference_format == "chart"
        )
        return legs >= 3 and (
            has_chart or scenario.speed_minutes != 60 or scenario.fuel_minutes != 60
        )

    if profile.family == "distance" and profile.level >= 9:
        return legs >= 4 and scenario.speed_minutes != 60

    if profile.family == "payload" and profile.level >= 9:
        return scenario.parcel_reference_format == "chart"

    if profile.family == "full" and profile.level >= 9:
        return (
            scenario.parcel_reference_format == "chart"
            or scenario.fuel_reference_format == "chart"
            or scenario.speed_minutes != 60
            or scenario.fuel_minutes != 60
        )

    return True


def _route_minutes(route_distance: int, speed_value: int, speed_minutes: int) -> int:
    return int(round_half_up((route_distance / max(speed_value, 1)) * float(speed_minutes)))


def _fuel_used_for_minutes(route_minutes: int, fuel_burn: int, fuel_minutes: int) -> int:
    return int(round_half_up((route_minutes / float(max(fuel_minutes, 1))) * fuel_burn))


def _fuel_endurance_minutes(start_fuel: int, fuel_burn: int, fuel_minutes: int) -> int:
    return int(round_half_up((start_fuel / float(max(fuel_burn, 1))) * float(fuel_minutes)))


def _pick_chart_step(max_value: int, candidates: tuple[int, ...]) -> int:
    best = candidates[0]
    best_score = abs(math.ceil(max_value / max(best, 1)) - 5)
    for step in candidates[1:]:
        score = abs(math.ceil(max_value / max(step, 1)) - 5)
        if score < best_score:
            best = step
            best_score = score
    return best


def _speed_tolerance_units(scenario: AirborneScenario) -> int:
    if scenario.parcel_reference_format != "chart":
        return 0
    return max(1, int(math.ceil(scenario.parcel_chart_step / 2.0)))


def _fuel_tolerance_units(scenario: AirborneScenario) -> int:
    if scenario.fuel_reference_format != "chart":
        return 0
    return max(1, int(math.ceil(scenario.fuel_chart_step / 2.0)))


def _time_tolerance_from_speed(scenario: AirborneScenario) -> int:
    speed_tol = _speed_tolerance_units(scenario)
    if speed_tol <= 0:
        return 0
    base = scenario.route_travel_minutes
    low_speed = max(1, scenario.speed_value - speed_tol)
    high_speed = scenario.speed_value + speed_tol
    variants = (
        _route_minutes(scenario.route_distance_total, low_speed, scenario.speed_minutes),
        _route_minutes(scenario.route_distance_total, high_speed, scenario.speed_minutes),
    )
    return max(abs(base - variant) for variant in variants)


def _time_tolerance_from_fuel(scenario: AirborneScenario) -> int:
    fuel_tol = _fuel_tolerance_units(scenario)
    if fuel_tol <= 0:
        return 0
    base_endurance = _hhmm_str_to_minutes(scenario.empty_time_hhmm) - _hhmm_str_to_minutes(
        scenario.start_time_hhmm
    )
    variants: list[int] = []
    for burn in {
        max(1, scenario.fuel_burn_per_hr - fuel_tol),
        scenario.fuel_burn_per_hr + fuel_tol,
    }:
        endurance = int(
            round_half_up(
                (scenario.start_fuel_liters / float(max(burn, 1))) * float(scenario.fuel_minutes)
            )
        )
        variants.append(endurance)
    return max(abs(base_endurance - variant) for variant in variants)


def _numeric_tolerance_from_graphs(scenario: AirborneScenario) -> int:
    speed_tol = _speed_tolerance_units(scenario)
    fuel_tol = _fuel_tolerance_units(scenario)
    if scenario.question_kind == "fuel_endurance":
        return _time_tolerance_from_fuel(scenario)
    if scenario.question_kind == "fuel_burned":
        baseline = scenario.fuel_used_on_route
        values: list[int] = []
        speeds = {scenario.speed_value}
        burns = {scenario.fuel_burn_per_hr}
        if speed_tol > 0:
            speeds.update(
                {
                    max(1, scenario.speed_value - speed_tol),
                    scenario.speed_value + speed_tol,
                }
            )
        if fuel_tol > 0:
            burns.update(
                {max(1, scenario.fuel_burn_per_hr - fuel_tol), scenario.fuel_burn_per_hr + fuel_tol}
            )
        for speed in speeds:
            route_minutes = _route_minutes(
                scenario.route_distance_total,
                speed,
                scenario.speed_minutes,
            )
            for burn in burns:
                values.append(_fuel_used_for_minutes(route_minutes, burn, scenario.fuel_minutes))
        return max(abs(baseline - value) for value in values)
    if scenario.question_kind == "parcel_effect" and scenario.parcel_reference_format == "chart":
        return max(1, speed_tol * 2)
    if scenario.question_kind == "parcel_weight" and scenario.parcel_reference_format == "chart":
        weights = [weight for weight, _ in scenario.weight_speed_table]
        step = min(
            (right - left for left, right in zip(weights, weights[1:], strict=False)),
            default=50,
        )
        return max(1, int(step))
    return 0


def _build_problem_tolerance(scenario: AirborneScenario) -> int:
    if scenario.answer_format == "hhmm":
        if scenario.question_kind in {"arrival_time", "takeoff_time"}:
            return _time_tolerance_from_speed(scenario)
        if scenario.question_kind == "empty_time":
            return _time_tolerance_from_fuel(scenario)
        return 0
    return _numeric_tolerance_from_graphs(scenario)


class AirborneScorer(AnswerScorer):
    def score(self, *, problem: Problem, user_answer: int, raw: str) -> float:
        _ = raw
        scenario = cast(AirborneScenario | None, problem.payload)
        tolerance = max(0, int(problem.tolerance))
        if scenario is not None and scenario.answer_format == "hhmm":
            diff = _minute_diff_hhmm(int(problem.answer), int(user_answer))
            return 1.0 if diff <= tolerance else 0.0
        diff = abs(int(user_answer) - int(problem.answer))
        return 1.0 if diff <= tolerance else 0.0


class AirborneNumericalGenerator:
    def __init__(self, rng: SeededRng, *, scripted_diverse_problems: int = 0):
        self._rng = rng
        self._scripted_specs = self._build_scripted_specs(scripted_diverse_problems)
        self._recent_variants: list[str] = []

    def generate(self, *, profile: AirborneDifficultyProfile | None = None) -> Problem:
        for _ in range(128):
            spec = self._scripted_specs[0] if self._scripted_specs else None
            if spec is None:
                if profile is None:
                    scenario, prompt, answer = self._build_problem()
                else:
                    scenario, prompt, answer = self._build_problem_from_profile(profile)
            else:
                question_kind, unit_profile, parcel_format, fuel_format = spec
                scenario, prompt, answer = self._build_problem(
                    forced_question_kind=question_kind,
                    forced_unit_profile=unit_profile,
                    forced_parcel_reference_format=parcel_format,
                    forced_fuel_reference_format=fuel_format,
                )
            if 0 <= int(answer) <= 9999:
                if spec is not None:
                    self._scripted_specs.pop(0)
                tolerance = _build_problem_tolerance(scenario)
                return Problem(
                    prompt=prompt,
                    answer=int(answer),
                    tolerance=tolerance,
                    payload=scenario,
                )
        raise RuntimeError("unable to generate a 4-digit airborne numerical answer")

    def check(self, problem: Problem, user_answer: int) -> bool:
        scenario = cast(AirborneScenario | None, problem.payload)
        if scenario is not None and scenario.answer_format == "hhmm":
            return _minute_diff_hhmm(problem.answer, user_answer) <= max(0, problem.tolerance)
        return abs(int(user_answer) - int(problem.answer)) <= max(0, int(problem.tolerance))

    def next_problem(self, *, difficulty: float) -> Problem:
        if self._scripted_specs:
            return self.generate()
        level = _difficulty_to_level(difficulty)
        profile = build_ant_airborne_difficulty_profile(level, family="full")
        return self.generate(profile=profile)

    def _build_scripted_specs(
        self,
        scripted_diverse_problems: int,
    ) -> list[tuple[str, UnitProfile, str, str]]:
        count = max(0, int(scripted_diverse_problems))
        if count <= 0:
            return []

        specs: list[tuple[str, UnitProfile, str, str]] = []
        unit_offset = int(self._rng.randint(0, len(UNIT_PROFILES) - 1))
        ref_offset = int(self._rng.randint(0, len(PRACTICE_REFERENCE_PAIRS) - 1))
        for idx in range(count):
            kind = PRACTICE_KIND_ORDER[idx % len(PRACTICE_KIND_ORDER)]
            unit_profile = UNIT_PROFILES[(unit_offset + idx) % len(UNIT_PROFILES)]
            parcel_format, fuel_format = PRACTICE_REFERENCE_PAIRS[
                (ref_offset + idx) % len(PRACTICE_REFERENCE_PAIRS)
            ]
            specs.append((kind, unit_profile, parcel_format, fuel_format))
        return specs

    def _build_problem_from_profile(
        self,
        profile: AirborneDifficultyProfile,
    ) -> tuple[AirborneScenario, str, int]:
        for _ in range(1024):
            question_kind = _weighted_choice(self._rng, profile.question_kinds)
            scenario, prompt, answer = self._build_problem(forced_question_kind=question_kind)
            if _scenario_matches_profile(scenario, profile):
                return scenario, prompt, answer
        raise RuntimeError(
            f"unable to generate airborne numerical problem for profile {profile.family} L{profile.level}"
        )

    def _build_problem(
        self,
        *,
        forced_question_kind: str | None = None,
        forced_unit_profile: UnitProfile | None = None,
        forced_parcel_reference_format: str | None = None,
        forced_fuel_reference_format: str | None = None,
    ) -> tuple[AirborneScenario, str, int]:
        rng = self._rng
        template = rng.choice(list(TEMPLATES))
        unit_profile = forced_unit_profile or rng.choice(list(UNIT_PROFILES))
        node_names = _pick_node_names(rng, len(template.nodes))

        edge_lo, edge_hi = unit_profile.distance_range
        edge_distances = tuple(int(rng.randint(edge_lo, edge_hi)) for _ in template.edges)

        start_node = int(rng.randint(0, len(template.nodes) - 1))
        route = [start_node]
        max_legs = int(rng.randint(2, 4))
        for _ in range(max_legs):
            cur = route[-1]
            options = [b if a == cur else a for a, b in template.edges if a == cur or b == cur]
            unseen = [node for node in options if node not in route]
            nxt = int(rng.choice(unseen if unseen else options))
            route.append(nxt)
        route_t = tuple(route)

        legs: list[RouteLeg] = []
        for a, b in zip(route_t, route_t[1:], strict=False):
            edge_key = tuple(sorted((a, b)))
            idx = next(
                i
                for i, edge in enumerate(template.edges)
                if tuple(sorted(edge)) == edge_key
            )
            legs.append(RouteLeg(frm=a, to=b, distance=int(edge_distances[idx])))
        legs_t = tuple(legs)

        weight_speed_table, parcel_weight, speed_value = _make_weight_speed_table(rng, unit_profile)
        speed_fuel_table, fuel_burn = _make_speed_fuel_table(rng, unit_profile, speed_value)
        start_fuel = int(
            round_half_up(
                rng.randint(*unit_profile.start_fuel_range)
                / float(max(unit_profile.start_fuel_step, 1))
            )
            * unit_profile.start_fuel_step
        )

        start_minutes = int(rng.randint(0, 23) * 60 + rng.randint(0, 59))
        start_time_hhmm = _minutes_to_hhmm_str(start_minutes)

        route_distance_total = sum(leg.distance for leg in legs_t)
        route_travel_minutes = _route_minutes(
            route_distance_total,
            speed_value,
            unit_profile.speed_minutes,
        )
        arrival_time_hhmm = _minutes_to_hhmm_str(start_minutes + route_travel_minutes)
        empty_minutes = _fuel_endurance_minutes(start_fuel, fuel_burn, unit_profile.fuel_minutes)
        empty_time_hhmm = _minutes_to_hhmm_str(start_minutes + empty_minutes)
        fuel_used_on_route = _fuel_used_for_minutes(
            route_travel_minutes,
            fuel_burn,
            unit_profile.fuel_minutes,
        )
        clean_speed = int(weight_speed_table[0][1])
        parcel_effect = int(max(0, clean_speed - speed_value))

        parcel_reference_format = forced_parcel_reference_format or (
            "chart" if rng.random() < 0.45 else "table"
        )
        fuel_reference_format = forced_fuel_reference_format or (
            "chart" if rng.random() < 0.45 else "table"
        )
        parcel_chart_step = _pick_chart_step(
            max(speed for _, speed in weight_speed_table),
            unit_profile.speed_chart_steps,
        )
        fuel_chart_step = _pick_chart_step(
            max(burn for _, burn in speed_fuel_table),
            unit_profile.fuel_chart_steps,
        )

        question_kind = forced_question_kind or _weighted_choice(rng, QUESTION_KINDS)
        dest = node_names[route_t[-1]]

        if question_kind == "arrival_time":
            answer = _minutes_to_hhmm_int(start_minutes + route_travel_minutes)
            prompt = _airborne_prompt(
                rng,
                question_kind=question_kind,
                dest=dest,
                unit_profile=unit_profile,
            )
            given_time_label = "Time Now"
            given_time_hhmm = start_time_hhmm
            target_label = dest
            answer_format = "hhmm"
            answer_label = "Arrival Time"
            answer_unit_label = "HHMM"
        elif question_kind == "takeoff_time":
            answer = _minutes_to_hhmm_int(start_minutes)
            prompt = _airborne_prompt(
                rng,
                question_kind=question_kind,
                dest=dest,
                unit_profile=unit_profile,
            )
            given_time_label = "Arrival Time"
            given_time_hhmm = arrival_time_hhmm
            target_label = dest
            answer_format = "hhmm"
            answer_label = "Take Off"
            answer_unit_label = "HHMM"
        elif question_kind == "empty_time":
            answer = _minutes_to_hhmm_int(start_minutes + empty_minutes)
            prompt = _airborne_prompt(
                rng,
                question_kind=question_kind,
                dest=dest,
                unit_profile=unit_profile,
            )
            given_time_label = "Time Now"
            given_time_hhmm = start_time_hhmm
            target_label = "EMPTY"
            answer_format = "hhmm"
            answer_label = "Empty Time"
            answer_unit_label = "HHMM"
        elif question_kind == "fuel_endurance":
            answer = int(empty_minutes)
            prompt = _airborne_prompt(
                rng,
                question_kind=question_kind,
                dest=dest,
                unit_profile=unit_profile,
            )
            given_time_label = "Time Now"
            given_time_hhmm = start_time_hhmm
            target_label = "ENDURANCE"
            answer_format = "number"
            answer_label = "Fuel Endurance"
            answer_unit_label = "min"
        elif question_kind == "fuel_burned":
            answer = int(fuel_used_on_route)
            prompt = _airborne_prompt(
                rng,
                question_kind=question_kind,
                dest=dest,
                unit_profile=unit_profile,
            )
            given_time_label = "Time Now"
            given_time_hhmm = start_time_hhmm
            target_label = dest
            answer_format = "number"
            answer_label = "Fuel Used"
            answer_unit_label = unit_profile.fuel_unit.split("/")[0]
        elif question_kind == "distance_travelled":
            answer = int(route_distance_total)
            prompt = _airborne_prompt(
                rng,
                question_kind=question_kind,
                dest=dest,
                unit_profile=unit_profile,
            )
            given_time_label = "Time Now"
            given_time_hhmm = start_time_hhmm
            target_label = dest
            answer_format = "number"
            answer_label = "Distance"
            answer_unit_label = unit_profile.distance_unit
        elif question_kind == "parcel_weight":
            answer = int(parcel_weight)
            prompt = _airborne_prompt(
                rng,
                question_kind=question_kind,
                dest=dest,
                unit_profile=unit_profile,
            )
            given_time_label = "Arrival Time"
            given_time_hhmm = arrival_time_hhmm
            target_label = dest
            answer_format = "number"
            answer_label = "Parcel Weight"
            answer_unit_label = "kg"
        else:
            answer = parcel_effect
            prompt = _airborne_prompt(
                rng,
                question_kind=question_kind,
                dest=dest,
                unit_profile=unit_profile,
            )
            given_time_label = "Time Now"
            given_time_hhmm = start_time_hhmm
            target_label = dest
            answer_format = "number"
            answer_label = "Parcel Effect"
            answer_unit_label = unit_profile.speed_unit

        content_family = _airborne_content_family(question_kind)
        content_pack = f"{parcel_reference_format}_{fuel_reference_format}"
        variant_id = stable_variant_id(
            template.name,
            question_kind,
            content_pack,
            len(legs_t),
            unit_profile.speed_unit,
            unit_profile.speed_minutes,
            unit_profile.fuel_minutes,
        )
        if variant_id in self._recent_variants:
            if forced_question_kind is None:
                return self._build_problem(
                    forced_unit_profile=forced_unit_profile,
                    forced_parcel_reference_format=forced_parcel_reference_format,
                    forced_fuel_reference_format=forced_fuel_reference_format,
                )
        self._recent_variants.append(variant_id)
        if len(self._recent_variants) > 4:
            del self._recent_variants[:-4]

        scenario = AirborneScenario(
            template_name=template.name,
            node_names=node_names,
            edge_distances=edge_distances,
            route=route_t,
            legs=legs_t,
            start_time_hhmm=start_time_hhmm,
            given_time_label=given_time_label,
            given_time_hhmm=given_time_hhmm,
            speed_value=int(speed_value),
            speed_unit=unit_profile.speed_unit,
            speed_minutes=unit_profile.speed_minutes,
            distance_unit=unit_profile.distance_unit,
            fuel_burn_per_hr=int(fuel_burn),
            fuel_unit=unit_profile.fuel_unit,
            fuel_minutes=unit_profile.fuel_minutes,
            parcel_weight=int(parcel_weight),
            start_fuel_liters=start_fuel,
            route_distance_total=int(route_distance_total),
            route_travel_minutes=int(route_travel_minutes),
            arrival_time_hhmm=arrival_time_hhmm,
            empty_time_hhmm=empty_time_hhmm,
            fuel_used_on_route=int(fuel_used_on_route),
            weight_speed_table=weight_speed_table,
            speed_fuel_table=speed_fuel_table,
            parcel_reference_format=parcel_reference_format,
            fuel_reference_format=fuel_reference_format,
            parcel_chart_step=int(parcel_chart_step),
            fuel_chart_step=int(fuel_chart_step),
            question_kind=question_kind,
            target_label=target_label,
            answer_format=answer_format,
            answer_label=answer_label,
            answer_unit_label=answer_unit_label,
            answer_digits=4,
            content_family=content_family,
            variant_id=variant_id,
            content_pack=content_pack,
        )
        return scenario, prompt, int(answer)


def _make_weight_speed_table(
    rng: SeededRng, profile: UnitProfile
) -> tuple[tuple[tuple[int, int], ...], int, int]:
    weights = [0] + sorted(rng.sample(list(range(150, 851, 50)), k=5))
    parcel_weight = int(rng.choice(weights[1:]))

    min_required_drop = len(weights) - 1
    max_speed_lo = max(profile.speed_max_range[0], profile.speed_floor + min_required_drop)
    max_speed = int(rng.randint(max_speed_lo, profile.speed_max_range[1]))
    max_total_drop = min(profile.speed_drop_range[1], max_speed - profile.speed_floor)
    min_total_drop = min(max_total_drop, max(5, profile.speed_drop_range[0]))
    total_drop = int(rng.randint(min_total_drop, max_total_drop))
    remaining = total_drop - (len(weights) - 1)
    gaps = [1] * (len(weights) - 1)
    for idx in range(len(gaps)):
        max_extra = remaining
        if max_extra <= 0:
            break
        extra = int(rng.randint(0, max_extra))
        gaps[idx] += extra
        remaining -= extra
    gaps[-1] += remaining

    speeds = [max_speed]
    current_speed = max_speed
    for gap in gaps:
        current_speed -= gap
        speeds.append(current_speed)

    rows = [(weight, speed) for weight, speed in zip(weights, speeds, strict=True)]

    chosen_speed = next(speed for weight, speed in rows if weight == parcel_weight)
    return tuple(rows), parcel_weight, chosen_speed


def _make_speed_fuel_table(
    rng: SeededRng,
    profile: UnitProfile,
    chosen_speed: int,
) -> tuple[tuple[tuple[int, int], ...], int]:
    speeds = {chosen_speed}

    offsets = [1, -1, 2, -2, 3, -3, 4, -4, 5, -5]
    for offset in offsets:
        if len(speeds) >= 6:
            break
        candidate = chosen_speed + (offset * profile.speed_neighbor_step)
        candidate = max(profile.speed_floor, min(profile.speed_max_range[1], candidate))
        speeds.add(int(candidate))

    if len(speeds) < 6:
        pool = [
            speed
            for speed in range(
                profile.speed_floor,
                profile.speed_max_range[1] + 1,
                max(1, profile.speed_neighbor_step),
            )
            if speed not in speeds
        ]
        picks = rng.sample(pool, k=min(len(pool), 6 - len(speeds)))
        speeds.update(int(speed) for speed in picks)

    speeds_sorted = sorted(speeds)
    base = rng.randint(*profile.fuel_base_range)
    factor_lo, factor_hi = profile.fuel_factor_range
    factor = factor_lo + (rng.random() * (factor_hi - factor_lo))

    rows: list[tuple[int, int]] = []
    chosen_burn = profile.fuel_minimum
    for speed in speeds_sorted:
        jitter = rng.randint(-2, 2) if profile.fuel_minutes == 1 else rng.randint(-120, 120)
        burn = int(max(profile.fuel_minimum, round_half_up(base + speed * factor + jitter)))
        rows.append((speed, burn))
        if speed == chosen_speed:
            chosen_burn = burn

    return tuple(rows), int(chosen_burn)


def build_airborne_numerical_test(
    clock: Clock,
    seed: int,
    *,
    practice: bool = True,
    difficulty: float = 0.5,
    scored_duration_s: float = 35.0 * 60.0,
) -> TimedTextInputTest:
    rng = SeededRng(seed)
    practice_questions = len(PRACTICE_KIND_ORDER) if practice else 0
    gen = AirborneNumericalGenerator(rng, scripted_diverse_problems=practice_questions)
    scorer = AirborneScorer()

    intro = [
        "Airborne Numerical",
        "",
        (
            "Questions can ask for arrival time, take off time, empty time, "
            "fuel endurance, fuel used, distance travelled, parcel weight, "
            "or parcel effect on speed."
        ),
        (
            "Some reference pages are tables. Others are bar charts, so you may "
            "need to estimate between grid lines."
        ),
        "All answers use 4 digits. Use a leading zero for non-time answers when needed.",
        "",
        "Controls:",
        "  Hold A: Show distances",
        "  Hold S: Introduction",
        "  Hold D: Speed & Fuel reference",
        "  Hold F: Speed & Parcel reference",
        "",
        "Scoring:",
        "  Table-based reads expect exact answers.",
        "  Chart-based reads allow a small tolerance for fair estimation.",
    ]

    return TimedTextInputTest(
        title="Airborne Numerical Test",
        clock=clock,
        generator=gen,
        scorer=scorer,
        instructions=intro,
        seed=seed,
        difficulty=difficulty,
        practice_questions=practice_questions,
        scored_duration_s=scored_duration_s,
    )

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import cast

from .clock import Clock
from .cognitive_core import (
    AnswerScorer,
    AttemptSummary,
    Phase,
    Problem,
    QuestionEvent,
    SeededRng,
    TestSnapshot,
    clamp01,
)


@dataclass(frozen=True, slots=True)
class SpatialIntegrationConfig:
    practice_scenes_per_part: int = 2
    static_scored_duration_s: float = 10.0 * 60.0
    aircraft_scored_duration_s: float = 13.0 * 60.0
    static_study_s: float = 12.0
    aircraft_study_s: float = 15.0
    question_time_limit_s: float = 8.0
    skip_practice_for_testing: bool = False
    start_part: str = "STATIC"  # "STATIC" | "AIRCRAFT"


class SpatialIntegrationPart(StrEnum):
    STATIC = "static"
    AIRCRAFT = "aircraft"


class SpatialIntegrationSceneView(StrEnum):
    TOPDOWN = "topdown"
    OBLIQUE = "oblique"
    PROFILE = "profile"


class SpatialIntegrationTrialStage(StrEnum):
    STUDY = "study"
    MEMORIZE = "study"
    QUESTION = "question"


class SpatialIntegrationQuestionKind(StrEnum):
    LANDMARK_GRID = "landmark_grid"
    SCENE_RECONSTRUCTION = "scene_reconstruction"
    AIRCRAFT_ROUTE_SELECTION = "aircraft_route_selection"
    AIRCRAFT_LOCATION_GRID = "aircraft_location_grid"


class SpatialIntegrationAnswerMode(StrEnum):
    GRID_CLICK = "grid_click"
    OPTION_PICK = "option_pick"


@dataclass(frozen=True, slots=True)
class SpatialIntegrationPoint:
    x: int
    y: int
    z: int


@dataclass(frozen=True, slots=True)
class SpatialIntegrationVector:
    dx: int
    dy: int
    dz: int


@dataclass(frozen=True, slots=True)
class SpatialIntegrationLandmark:
    label: str
    x: int
    y: int
    kind: str = "landmark"


@dataclass(frozen=True, slots=True)
class SpatialIntegrationHill:
    label: str
    x: int
    y: int
    radius: int
    height: int


@dataclass(frozen=True, slots=True)
class SpatialIntegrationReferenceView:
    label: str
    scene_view: SpatialIntegrationSceneView
    heading_deg: int


@dataclass(frozen=True, slots=True)
class SpatialIntegrationOption:
    code: int
    label: str
    answer_token: str
    candidate_landmarks: tuple[SpatialIntegrationLandmark, ...] = ()
    candidate_route: tuple[SpatialIntegrationPoint, ...] = ()
    candidate_aircraft: SpatialIntegrationPoint | None = None
    heading_deg: int = 0


@dataclass(frozen=True, slots=True)
class SpatialIntegrationPayload:
    part: SpatialIntegrationPart
    trial_stage: SpatialIntegrationTrialStage
    block_kind: str  # "practice" | "scored"
    scene_id: int
    scene_index_in_block: int
    scenes_in_block: int | None
    study_view_index: int
    study_views_in_scene: int
    question_index_in_scene: int
    questions_in_scene: int
    stage_time_remaining_s: float | None
    part_time_remaining_s: float | None
    kind: SpatialIntegrationQuestionKind
    answer_mode: SpatialIntegrationAnswerMode | None
    stem: str
    query_label: str
    north_arrow_deg: int
    scene_view: SpatialIntegrationSceneView
    grid_cols: int
    grid_rows: int
    alt_levels: int
    reference_views: tuple[SpatialIntegrationReferenceView, ...]
    active_reference_view: SpatialIntegrationReferenceView | None
    hills: tuple[SpatialIntegrationHill, ...]
    landmarks: tuple[SpatialIntegrationLandmark, ...]
    answer_map_landmarks: tuple[SpatialIntegrationLandmark, ...]
    route_points: tuple[SpatialIntegrationPoint, ...]
    route_current_index: int
    aircraft_prev: SpatialIntegrationPoint
    aircraft_now: SpatialIntegrationPoint
    velocity: SpatialIntegrationVector
    show_aircraft_motion: bool
    options: tuple[SpatialIntegrationOption, ...]
    correct_code: int
    correct_point: SpatialIntegrationPoint
    correct_answer_token: str
    answer_map_route_points: tuple[SpatialIntegrationPoint, ...] = ()


def _clamp(v: int, lo: int, hi: int) -> int:
    return int(lo if v < lo else hi if v > hi else v)


def _grid_cell_token(x: int, y: int) -> str:
    return f"{chr(ord('A') + int(y))}{int(x) + 1}"


def _normalize_grid_cell_token(raw: str) -> str | None:
    token = "".join(ch for ch in str(raw).upper().strip() if ch.isalnum())
    if len(token) < 2:
        return None
    letters = "".join(ch for ch in token if ch.isalpha())
    digits = "".join(ch for ch in token if ch.isdigit())
    if len(letters) != 1 or digits == "":
        return None
    row = letters[0]
    if not ("A" <= row <= "Z"):
        return None
    return f"{row}{int(digits)}"


def _cell_from_token(token: str, *, grid_cols: int, grid_rows: int) -> SpatialIntegrationPoint | None:
    norm = _normalize_grid_cell_token(token)
    if norm is None:
        return None
    row = ord(norm[0]) - ord("A")
    col = int(norm[1:]) - 1
    if not (0 <= row < int(grid_rows) and 0 <= col < int(grid_cols)):
        return None
    return SpatialIntegrationPoint(x=col, y=row, z=0)


def _encode_point(point: SpatialIntegrationPoint, *, grid_cols: int, grid_rows: int) -> int:
    cols = max(1, int(grid_cols))
    rows = max(1, int(grid_rows))
    if not (0 <= point.x < cols and 0 <= point.y < rows):
        return 0
    return int((point.y * cols) + point.x + 1)


def _part_name(part: SpatialIntegrationPart) -> str:
    return "Part 1" if part is SpatialIntegrationPart.STATIC else "Part 2"


def _scene_title(part: SpatialIntegrationPart) -> str:
    return "Landscape Integration" if part is SpatialIntegrationPart.STATIC else "Aircraft / Route Integration"


class SpatialIntegrationScorer(AnswerScorer):
    """Spatial Integration uses exact scoring for both grid and fixed-choice questions."""

    def score(self, *, problem: Problem, user_answer: int, raw: str) -> float:
        payload = problem.payload
        if not isinstance(payload, SpatialIntegrationPayload):
            return 1.0 if int(user_answer) == int(problem.answer) else 0.0
        if payload.answer_mode is SpatialIntegrationAnswerMode.GRID_CLICK:
            token = _normalize_grid_cell_token(raw)
            return 1.0 if token == payload.correct_answer_token else 0.0
        return 1.0 if int(user_answer) == int(payload.correct_code) else 0.0


@dataclass(frozen=True, slots=True)
class _SpatialIntegrationQuestion:
    kind: SpatialIntegrationQuestionKind
    answer_mode: SpatialIntegrationAnswerMode
    stem: str
    query_label: str
    correct_code: int
    correct_point: SpatialIntegrationPoint
    correct_answer_token: str
    options: tuple[SpatialIntegrationOption, ...]
    answer_map_landmarks: tuple[SpatialIntegrationLandmark, ...]
    answer_map_route_points: tuple[SpatialIntegrationPoint, ...] = ()


@dataclass(frozen=True, slots=True)
class _SpatialIntegrationSceneCluster:
    scene_id: int
    part: SpatialIntegrationPart
    scene_view: SpatialIntegrationSceneView
    north_arrow_deg: int
    grid_cols: int
    grid_rows: int
    alt_levels: int
    reference_views: tuple[SpatialIntegrationReferenceView, ...]
    hills: tuple[SpatialIntegrationHill, ...]
    landmarks: tuple[SpatialIntegrationLandmark, ...]
    route_points: tuple[SpatialIntegrationPoint, ...]
    route_current_index: int
    aircraft_prev: SpatialIntegrationPoint
    aircraft_now: SpatialIntegrationPoint
    velocity: SpatialIntegrationVector
    show_aircraft_motion: bool
    questions: tuple[_SpatialIntegrationQuestion, ...]


class SpatialIntegrationGenerator:
    _STATIC_OBJECT_SPECS = (
        ("BLD1", "building"),
        ("BLD2", "building"),
        ("SOL1", "foot_soldiers"),
        ("SOL2", "foot_soldiers"),
        ("SHP1", "sheep"),
        ("SHP2", "sheep"),
        ("WOOD", "forest"),
        ("TRK1", "truck"),
        ("TWR1", "tower"),
        ("TENT", "tent"),
    )
    _AIRCRAFT_OBJECT_SPECS = (
        ("BLD1", "building"),
        ("SOL1", "foot_soldiers"),
        ("SHP1", "sheep"),
        ("WOOD", "forest"),
        ("TRK1", "truck"),
        ("TWR1", "tower"),
        ("TENT", "tent"),
    )
    _ROUTE_TEMPLATES = (
        (
            SpatialIntegrationPoint(1, 5, 1),
            SpatialIntegrationPoint(2, 4, 2),
            SpatialIntegrationPoint(3, 3, 3),
            SpatialIntegrationPoint(5, 3, 3),
            SpatialIntegrationPoint(6, 4, 2),
            SpatialIntegrationPoint(5, 5, 1),
        ),
        (
            SpatialIntegrationPoint(1, 2, 1),
            SpatialIntegrationPoint(2, 2, 2),
            SpatialIntegrationPoint(4, 3, 3),
            SpatialIntegrationPoint(5, 5, 3),
            SpatialIntegrationPoint(4, 6, 2),
            SpatialIntegrationPoint(2, 6, 1),
        ),
        (
            SpatialIntegrationPoint(1, 4, 2),
            SpatialIntegrationPoint(2, 3, 3),
            SpatialIntegrationPoint(4, 2, 4),
            SpatialIntegrationPoint(5, 3, 3),
            SpatialIntegrationPoint(6, 5, 2),
            SpatialIntegrationPoint(5, 6, 1),
        ),
    )

    def __init__(self, *, seed: int) -> None:
        self._rng = SeededRng(seed)
        self._scene_id = 0

    def next_scene_cluster(
        self,
        *,
        part: SpatialIntegrationPart,
        difficulty: float,
    ) -> _SpatialIntegrationSceneCluster:
        d = clamp01(difficulty)
        self._scene_id += 1
        if part is SpatialIntegrationPart.STATIC:
            return self._build_static_scene(scene_id=self._scene_id, difficulty=d)
        return self._build_aircraft_scene(scene_id=self._scene_id, difficulty=d)

    def _build_static_scene(
        self,
        *,
        scene_id: int,
        difficulty: float,
    ) -> _SpatialIntegrationSceneCluster:
        grid_cols = 8
        grid_rows = 8
        hills = self._sample_hills()
        landmarks = self._sample_landmarks(
            grid_cols=grid_cols,
            grid_rows=grid_rows,
            specs=self._STATIC_OBJECT_SPECS,
            count=7,
        )
        primary_targets = cast(tuple[SpatialIntegrationLandmark, ...], self._rng.sample(landmarks, k=2))

        options = self._build_static_reconstruction_options(
            correct_landmarks=landmarks,
            grid_cols=grid_cols,
            grid_rows=grid_rows,
        )
        correct_option = next(opt for opt in options if opt.answer_token == "correct")

        questions = (
            _SpatialIntegrationQuestion(
                kind=SpatialIntegrationQuestionKind.LANDMARK_GRID,
                answer_mode=SpatialIntegrationAnswerMode.GRID_CLICK,
                stem=f"Click the grid cell where the {primary_targets[0].label} was located.",
                query_label=primary_targets[0].label,
                correct_code=0,
                correct_point=SpatialIntegrationPoint(primary_targets[0].x, primary_targets[0].y, 0),
                correct_answer_token=_grid_cell_token(primary_targets[0].x, primary_targets[0].y),
                options=(),
                answer_map_landmarks=self._grid_context_landmarks(
                    landmarks=landmarks,
                    anchor_point=SpatialIntegrationPoint(primary_targets[0].x, primary_targets[0].y, 0),
                    difficulty=difficulty,
                    omit_label=primary_targets[0].label,
                ),
            ),
            _SpatialIntegrationQuestion(
                kind=SpatialIntegrationQuestionKind.LANDMARK_GRID,
                answer_mode=SpatialIntegrationAnswerMode.GRID_CLICK,
                stem=f"Click the grid cell where the {primary_targets[1].label} was located.",
                query_label=primary_targets[1].label,
                correct_code=0,
                correct_point=SpatialIntegrationPoint(primary_targets[1].x, primary_targets[1].y, 0),
                correct_answer_token=_grid_cell_token(primary_targets[1].x, primary_targets[1].y),
                options=(),
                answer_map_landmarks=self._grid_context_landmarks(
                    landmarks=landmarks,
                    anchor_point=SpatialIntegrationPoint(primary_targets[1].x, primary_targets[1].y, 0),
                    difficulty=difficulty,
                    omit_label=primary_targets[1].label,
                ),
            ),
            _SpatialIntegrationQuestion(
                kind=SpatialIntegrationQuestionKind.SCENE_RECONSTRUCTION,
                answer_mode=SpatialIntegrationAnswerMode.OPTION_PICK,
                stem="Which top-down map matches the studied landscape?",
                query_label="FULL SCENE",
                correct_code=int(correct_option.code),
                correct_point=SpatialIntegrationPoint(0, 0, 0),
                correct_answer_token=str(correct_option.code),
                options=options,
                answer_map_landmarks=(),
            ),
        )

        return _SpatialIntegrationSceneCluster(
            scene_id=scene_id,
            part=SpatialIntegrationPart.STATIC,
            scene_view=SpatialIntegrationSceneView.OBLIQUE,
            north_arrow_deg=0,
            grid_cols=grid_cols,
            grid_rows=grid_rows,
            alt_levels=4,
            reference_views=(
                SpatialIntegrationReferenceView("View 1", SpatialIntegrationSceneView.OBLIQUE, 24),
                SpatialIntegrationReferenceView("View 2", SpatialIntegrationSceneView.OBLIQUE, -42),
                SpatialIntegrationReferenceView("View 3", SpatialIntegrationSceneView.OBLIQUE, 132),
            ),
            hills=hills,
            landmarks=landmarks,
            route_points=(),
            route_current_index=0,
            aircraft_prev=SpatialIntegrationPoint(0, 0, 0),
            aircraft_now=SpatialIntegrationPoint(0, 0, 0),
            velocity=SpatialIntegrationVector(0, 0, 0),
            show_aircraft_motion=False,
            questions=questions,
        )

    def _build_aircraft_scene(
        self,
        *,
        scene_id: int,
        difficulty: float,
    ) -> _SpatialIntegrationSceneCluster:
        grid_cols = 8
        grid_rows = 8
        hills = self._sample_hills()
        landmarks = self._sample_landmarks(
            grid_cols=grid_cols,
            grid_rows=grid_rows,
            specs=self._AIRCRAFT_OBJECT_SPECS,
            count=5,
        )
        route = tuple(cast(tuple[SpatialIntegrationPoint, ...], self._rng.choice(self._ROUTE_TEMPLATES)))
        route_current_index = int(self._rng.randint(2, len(route) - 2))
        aircraft_prev = route[route_current_index - 1]
        aircraft_now = route[route_current_index]
        aircraft_next = route[route_current_index + 1]
        velocity = SpatialIntegrationVector(
            dx=int(aircraft_next.x - aircraft_now.x),
            dy=int(aircraft_next.y - aircraft_now.y),
            dz=int(aircraft_next.z - aircraft_now.z),
        )

        route_options = self._build_aircraft_route_options(
            correct_route=route,
            aircraft_now=aircraft_now,
            grid_cols=grid_cols,
            grid_rows=grid_rows,
            variant="route",
        )
        route_correct = next(opt for opt in route_options if opt.answer_token == "correct")

        continuation_options = self._build_aircraft_route_options(
            correct_route=route,
            aircraft_now=aircraft_next,
            grid_cols=grid_cols,
            grid_rows=grid_rows,
            variant="continuation",
        )
        continuation_correct = next(
            opt for opt in continuation_options if opt.answer_token == "correct"
        )

        questions = (
            _SpatialIntegrationQuestion(
                kind=SpatialIntegrationQuestionKind.AIRCRAFT_ROUTE_SELECTION,
                answer_mode=SpatialIntegrationAnswerMode.OPTION_PICK,
                stem="Which route map matches the aircraft track shown in the studied views?",
                query_label="ROUTE",
                correct_code=int(route_correct.code),
                correct_point=aircraft_now,
                correct_answer_token=str(route_correct.code),
                options=route_options,
                answer_map_landmarks=landmarks,
            ),
            _SpatialIntegrationQuestion(
                kind=SpatialIntegrationQuestionKind.AIRCRAFT_ROUTE_SELECTION,
                answer_mode=SpatialIntegrationAnswerMode.OPTION_PICK,
                stem="Which path continuation shows the aircraft in the correct next position?",
                query_label="CONTINUATION",
                correct_code=int(continuation_correct.code),
                correct_point=aircraft_next,
                correct_answer_token=str(continuation_correct.code),
                options=continuation_options,
                answer_map_landmarks=landmarks,
            ),
            _SpatialIntegrationQuestion(
                kind=SpatialIntegrationQuestionKind.AIRCRAFT_LOCATION_GRID,
                answer_mode=SpatialIntegrationAnswerMode.GRID_CLICK,
                stem="Click the grid cell where the aircraft was located in the studied scene.",
                query_label="AIRCRAFT",
                correct_code=0,
                correct_point=aircraft_now,
                correct_answer_token=_grid_cell_token(aircraft_now.x, aircraft_now.y),
                options=(),
                answer_map_landmarks=self._grid_context_landmarks(
                    landmarks=landmarks,
                    anchor_point=aircraft_now,
                    difficulty=difficulty,
                ),
                answer_map_route_points=self._grid_context_route_points(
                    route_points=route,
                    focus_index=route_current_index,
                    difficulty=difficulty,
                ),
            ),
        )

        return _SpatialIntegrationSceneCluster(
            scene_id=scene_id,
            part=SpatialIntegrationPart.AIRCRAFT,
            scene_view=SpatialIntegrationSceneView.OBLIQUE,
            north_arrow_deg=0,
            grid_cols=grid_cols,
            grid_rows=grid_rows,
            alt_levels=5,
            reference_views=(
                SpatialIntegrationReferenceView("View 1", SpatialIntegrationSceneView.OBLIQUE, 28),
                SpatialIntegrationReferenceView("View 2", SpatialIntegrationSceneView.OBLIQUE, -54),
                SpatialIntegrationReferenceView("View 3", SpatialIntegrationSceneView.OBLIQUE, 146),
            ),
            hills=hills,
            landmarks=landmarks,
            route_points=route,
            route_current_index=route_current_index,
            aircraft_prev=aircraft_prev,
            aircraft_now=aircraft_now,
            velocity=velocity,
            show_aircraft_motion=True,
            questions=questions,
        )

    def _sample_hills(self) -> tuple[SpatialIntegrationHill, ...]:
        presets = (
            (
                SpatialIntegrationHill("H1", 2, 2, 2, 2),
                SpatialIntegrationHill("H2", 5, 3, 2, 3),
                SpatialIntegrationHill("H3", 4, 6, 2, 2),
            ),
            (
                SpatialIntegrationHill("H1", 1, 3, 2, 2),
                SpatialIntegrationHill("H2", 5, 2, 3, 3),
                SpatialIntegrationHill("H3", 6, 6, 2, 2),
            ),
            (
                SpatialIntegrationHill("H1", 2, 5, 2, 2),
                SpatialIntegrationHill("H2", 4, 2, 2, 2),
                SpatialIntegrationHill("H3", 6, 4, 3, 3),
            ),
        )
        return tuple(cast(tuple[SpatialIntegrationHill, ...], self._rng.choice(presets)))

    def _sample_landmarks(
        self,
        *,
        grid_cols: int,
        grid_rows: int,
        specs: tuple[tuple[str, str], ...],
        count: int,
    ) -> tuple[SpatialIntegrationLandmark, ...]:
        chosen_specs = cast(tuple[tuple[str, str], ...], self._rng.sample(specs, k=min(count, len(specs))))
        points: list[tuple[int, int]] = []
        occupancy: dict[tuple[int, int], int] = {}
        while len(points) < len(chosen_specs):
            duplicate_ok = bool(points) and self._rng.random() < 0.28
            if duplicate_ok:
                repeated = cast(tuple[int, int], self._rng.choice(points))
                if occupancy.get(repeated, 0) < 2:
                    candidate = repeated
                else:
                    candidate = (
                        int(self._rng.randint(1, grid_cols - 2)),
                        int(self._rng.randint(1, grid_rows - 2)),
                    )
            else:
                candidate = (
                    int(self._rng.randint(1, grid_cols - 2)),
                    int(self._rng.randint(1, grid_rows - 2)),
                )
            if occupancy.get(candidate, 0) >= 2:
                continue
            occupancy[candidate] = occupancy.get(candidate, 0) + 1
            points.append(candidate)
        return tuple(
            SpatialIntegrationLandmark(label=str(label), x=int(pt[0]), y=int(pt[1]), kind=str(kind))
            for (label, kind), pt in zip(chosen_specs, points, strict=True)
        )

    def _grid_context_landmarks(
        self,
        *,
        landmarks: tuple[SpatialIntegrationLandmark, ...],
        anchor_point: SpatialIntegrationPoint,
        difficulty: float,
        omit_label: str | None = None,
    ) -> tuple[SpatialIntegrationLandmark, ...]:
        filtered = tuple(
            landmark
            for landmark in landmarks
            if omit_label is None or str(landmark.label).upper() != str(omit_label).upper()
        )
        if not filtered:
            return ()
        keep_fraction = max(0.0, 1.0 - clamp01(difficulty))
        keep_count = max(0, min(len(filtered), int(round(len(filtered) * keep_fraction))))
        if keep_count >= len(filtered):
            return filtered
        if keep_count <= 0:
            return ()

        ranked = sorted(
            filtered,
            key=lambda landmark: (
                abs(int(landmark.x) - int(anchor_point.x)) + abs(int(landmark.y) - int(anchor_point.y)),
                str(landmark.label),
            ),
        )
        selected = set(ranked[:keep_count])
        return tuple(landmark for landmark in filtered if landmark in selected)

    def _grid_context_route_points(
        self,
        *,
        route_points: tuple[SpatialIntegrationPoint, ...],
        focus_index: int,
        difficulty: float,
    ) -> tuple[SpatialIntegrationPoint, ...]:
        if not route_points:
            return ()
        keep_fraction = max(0.0, 1.0 - clamp01(difficulty))
        keep_count = max(0, min(len(route_points), int(round(len(route_points) * keep_fraction))))
        if keep_count >= len(route_points):
            return route_points
        if keep_count < 2:
            return ()

        center = max(0, min(len(route_points) - 1, int(focus_index)))
        start = max(0, center - ((keep_count - 1) // 2))
        end = start + keep_count
        if end > len(route_points):
            end = len(route_points)
            start = end - keep_count
        return route_points[start:end]

    def _build_static_reconstruction_options(
        self,
        *,
        correct_landmarks: tuple[SpatialIntegrationLandmark, ...],
        grid_cols: int,
        grid_rows: int,
    ) -> tuple[SpatialIntegrationOption, ...]:
        candidates: list[tuple[str, tuple[SpatialIntegrationLandmark, ...]]] = [
            ("correct", correct_landmarks),
            (
                "rotated",
                tuple(
                    SpatialIntegrationLandmark(
                        label=lm.label,
                        x=_clamp((grid_cols - 1) - lm.y, 0, grid_cols - 1),
                        y=_clamp(lm.x, 0, grid_rows - 1),
                        kind=lm.kind,
                    )
                    for lm in correct_landmarks
                ),
            ),
            (
                "mirrored",
                tuple(
                    SpatialIntegrationLandmark(
                        label=lm.label,
                        x=_clamp((grid_cols - 1) - lm.x, 0, grid_cols - 1),
                        y=lm.y,
                        kind=lm.kind,
                    )
                    for lm in correct_landmarks
                ),
            ),
            (
                "swapped",
                self._swap_landmarks(correct_landmarks),
            ),
        ]
        order = self._rng.sample((0, 1, 2, 3), k=4)
        options: list[SpatialIntegrationOption] = []
        for code, idx in enumerate(order, start=1):
            token, landmarks = candidates[idx]
            options.append(
                SpatialIntegrationOption(
                    code=code,
                    label=f"{code}",
                    answer_token=token,
                    candidate_landmarks=landmarks,
                )
            )
        return tuple(options)

    def _build_aircraft_route_options(
        self,
        *,
        correct_route: tuple[SpatialIntegrationPoint, ...],
        aircraft_now: SpatialIntegrationPoint,
        grid_cols: int,
        grid_rows: int,
        variant: str,
    ) -> tuple[SpatialIntegrationOption, ...]:
        mirrored = tuple(
            SpatialIntegrationPoint(
                x=_clamp((grid_cols - 1) - point.x, 0, grid_cols - 1),
                y=point.y,
                z=point.z,
            )
            for point in correct_route
        )
        rotated = tuple(
            SpatialIntegrationPoint(
                x=_clamp((grid_cols - 1) - point.y, 0, grid_cols - 1),
                y=_clamp(point.x, 0, grid_rows - 1),
                z=point.z,
            )
            for point in correct_route
        )
        shifted = tuple(
            SpatialIntegrationPoint(
                x=_clamp(point.x + 1, 0, grid_cols - 1),
                y=_clamp(point.y - 1, 0, grid_rows - 1),
                z=point.z,
            )
            for point in correct_route
        )
        candidates = [
            ("correct", correct_route, aircraft_now),
            ("mirrored", mirrored, self._mirror_aircraft_point(aircraft_now, grid_cols=grid_cols)),
            ("rotated", rotated, self._rotate_aircraft_point(aircraft_now, grid_cols=grid_cols, grid_rows=grid_rows)),
            ("shifted", shifted, SpatialIntegrationPoint(_clamp(aircraft_now.x + 1, 0, grid_cols - 1), _clamp(aircraft_now.y - 1, 0, grid_rows - 1), aircraft_now.z)),
        ]
        if variant == "continuation":
            candidates[3] = (
                "loop_timing",
                tuple(reversed(correct_route)),
                cast(SpatialIntegrationPoint, correct_route[max(0, len(correct_route) - 2)]),
            )

        order = self._rng.sample((0, 1, 2, 3), k=4)
        return tuple(
            SpatialIntegrationOption(
                code=code,
                label=f"{code}",
                answer_token=candidates[idx][0],
                candidate_route=candidates[idx][1],
                candidate_aircraft=candidates[idx][2],
            )
            for code, idx in enumerate(order, start=1)
        )

    @staticmethod
    def _swap_landmarks(
        landmarks: tuple[SpatialIntegrationLandmark, ...],
    ) -> tuple[SpatialIntegrationLandmark, ...]:
        if len(landmarks) < 2:
            return landmarks
        first = landmarks[0]
        second = landmarks[1]
        swapped = [
            SpatialIntegrationLandmark(
                label=first.label,
                x=second.x,
                y=second.y,
                kind=first.kind,
            ),
            SpatialIntegrationLandmark(
                label=second.label,
                x=first.x,
                y=first.y,
                kind=second.kind,
            ),
        ]
        swapped.extend(landmarks[2:])
        return tuple(swapped)

    @staticmethod
    def _mirror_aircraft_point(point: SpatialIntegrationPoint, *, grid_cols: int) -> SpatialIntegrationPoint:
        return SpatialIntegrationPoint(
            x=_clamp((grid_cols - 1) - point.x, 0, grid_cols - 1),
            y=point.y,
            z=point.z,
        )

    @staticmethod
    def _rotate_aircraft_point(
        point: SpatialIntegrationPoint,
        *,
        grid_cols: int,
        grid_rows: int,
    ) -> SpatialIntegrationPoint:
        return SpatialIntegrationPoint(
            x=_clamp((grid_cols - 1) - point.y, 0, grid_cols - 1),
            y=_clamp(point.x, 0, grid_rows - 1),
            z=point.z,
        )


class SpatialIntegrationEngine:
    _PART_ORDER = (
        SpatialIntegrationPart.STATIC,
        SpatialIntegrationPart.AIRCRAFT,
    )

    def __init__(
        self,
        *,
        clock: Clock,
        seed: int,
        difficulty: float = 0.5,
        config: SpatialIntegrationConfig | None = None,
    ) -> None:
        cfg = config or SpatialIntegrationConfig()
        if cfg.practice_scenes_per_part < 0:
            raise ValueError("practice_scenes_per_part must be >= 0")
        if cfg.static_scored_duration_s <= 0.0:
            raise ValueError("static_scored_duration_s must be > 0")
        if cfg.aircraft_scored_duration_s <= 0.0:
            raise ValueError("aircraft_scored_duration_s must be > 0")
        if cfg.static_study_s < 0.1 or cfg.aircraft_study_s < 0.1:
            raise ValueError("study duration must be >= 0.1")
        if cfg.question_time_limit_s < 0.1:
            raise ValueError("question_time_limit_s must be >= 0.1")

        start = str(cfg.start_part).strip().upper()
        self._start_part = (
            SpatialIntegrationPart.AIRCRAFT
            if start in {"B", "AIRCRAFT", "PART2"}
            else SpatialIntegrationPart.STATIC
        )
        self._part_idx = self._PART_ORDER.index(self._start_part)

        self._clock = clock
        self._seed = int(seed)
        self._difficulty = clamp01(difficulty)
        self._cfg = cfg
        self._generator = SpatialIntegrationGenerator(seed=self._seed)
        self._scorer = SpatialIntegrationScorer()

        self._phase = Phase.INSTRUCTIONS
        self._last_update_at_s = self._clock.now()

        self._block_is_practice = True
        self._practice_done_prompt = "Practice complete."
        self._pending_done_action: str | None = None
        self._practice_feedback: str | None = None

        self._current_problem: Problem | None = None
        self._current_scene: _SpatialIntegrationSceneCluster | None = None
        self._current_question_idx = 0
        self._current_study_view_idx = 0
        self._block_scene_index = 0
        self._block_scene_target = 0
        self._block_started_at_s: float | None = None
        self._block_duration_s: float | None = None
        self._stage_ends_at_s: float | None = None
        self._question_presented_at_s: float | None = None
        self._pending_part_end_after_question = False

        self._events: list[QuestionEvent] = []
        self._scored_attempted = 0
        self._scored_correct = 0
        self._scored_total_score = 0.0
        self._scored_max_score = 0.0
        self._scored_elapsed_s = 0.0

    @property
    def phase(self) -> Phase:
        return self._phase

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def difficulty(self) -> float:
        return float(self._difficulty)

    def can_exit(self) -> bool:
        return self._phase in (
            Phase.INSTRUCTIONS,
            Phase.PRACTICE,
            Phase.PRACTICE_DONE,
            Phase.RESULTS,
        )

    def events(self) -> list[QuestionEvent]:
        return list(self._events)

    def start_practice(self) -> None:
        if self._phase is not Phase.INSTRUCTIONS:
            return
        self._part_idx = self._PART_ORDER.index(self._start_part)
        if self._cfg.skip_practice_for_testing:
            part = self._active_part()
            self._phase = Phase.PRACTICE_DONE
            self._pending_done_action = "start_scored"
            self._practice_done_prompt = (
                f"{_part_name(part)} practice skipped (testing mode). Press Enter to start scored."
            )
            self._current_problem = None
            self._current_scene = None
            return
        self._begin_block(is_practice=True)

    def start_scored(self) -> None:
        if self._phase is Phase.PRACTICE_DONE:
            if self._pending_done_action == "start_scored":
                self._begin_block(is_practice=False)
                return
            if self._pending_done_action == "start_next_practice":
                self._part_idx += 1
                if self._part_idx >= len(self._PART_ORDER):
                    self._to_results()
                    return
                if self._cfg.skip_practice_for_testing:
                    part = self._active_part()
                    self._phase = Phase.PRACTICE_DONE
                    self._pending_done_action = "start_scored"
                    self._practice_done_prompt = (
                        f"{_part_name(part)} practice skipped (testing mode). Press Enter to start scored."
                    )
                    self._current_problem = None
                    self._current_scene = None
                    return
                self._begin_block(is_practice=True)
                return
            return

        if self._phase is Phase.INSTRUCTIONS:
            self._part_idx = self._PART_ORDER.index(self._start_part)
            self._begin_block(is_practice=False)

    def submit_answer(self, raw: str) -> bool:
        raw_in = str(raw).strip()
        if self._handle_control_command(raw_in):
            return True

        if self._phase not in (Phase.PRACTICE, Phase.SCORED):
            return False
        if self._current_problem is None:
            return False
        payload = self._payload_or_none(self._current_problem)
        if payload is None or payload.trial_stage is not SpatialIntegrationTrialStage.QUESTION:
            return False
        if raw_in == "":
            return False

        score = 0.0
        user_answer = 0
        if payload.answer_mode is SpatialIntegrationAnswerMode.GRID_CLICK:
            point = _cell_from_token(
                raw_in,
                grid_cols=int(payload.grid_cols),
                grid_rows=int(payload.grid_rows),
            )
            if point is None:
                return False
            user_answer = _encode_point(point, grid_cols=payload.grid_cols, grid_rows=payload.grid_rows)
            score = self._scorer.score(problem=self._current_problem, user_answer=user_answer, raw=raw_in)
        else:
            try:
                user_answer = int(raw_in)
            except ValueError:
                return False
            score = self._scorer.score(problem=self._current_problem, user_answer=user_answer, raw=raw_in)

        now = self._clock.now()
        presented = now if self._question_presented_at_s is None else self._question_presented_at_s
        response_time_s = max(0.0, now - presented)
        self._record_event(
            user_answer=user_answer,
            raw=raw_in,
            score=float(score),
            response_time_s=float(response_time_s),
            answered_at_s=float(now),
        )
        self._after_question_completion(score=float(score))
        return True

    def update(self) -> None:
        now = self._clock.now()
        dt = max(0.0, now - self._last_update_at_s)
        self._last_update_at_s = now
        if self._phase is Phase.SCORED:
            self._scored_elapsed_s += dt

        if self._phase not in (Phase.PRACTICE, Phase.SCORED):
            return
        if self._current_problem is None:
            return
        payload = self._payload_or_none(self._current_problem)
        if payload is None:
            return

        if self._phase is Phase.SCORED and self._block_time_remaining_s() == 0.0:
            self._pending_part_end_after_question = True

        if self._stage_ends_at_s is None or now < self._stage_ends_at_s:
            return

        if payload.trial_stage is SpatialIntegrationTrialStage.STUDY:
            if self._current_scene is not None and self._current_study_view_idx + 1 < len(self._current_scene.reference_views):
                self._current_study_view_idx += 1
                self._set_trial_stage(SpatialIntegrationTrialStage.STUDY)
                self._stage_ends_at_s = now + self._study_step_duration(scene=self._current_scene)
            else:
                self._set_trial_stage(SpatialIntegrationTrialStage.QUESTION)
                self._question_presented_at_s = now
                self._stage_ends_at_s = now + float(self._cfg.question_time_limit_s)
            return

        self._timeout_current_question(now=now)

    def time_remaining_s(self) -> float | None:
        if self._phase is not Phase.SCORED:
            return None
        return self._block_time_remaining_s()

    def current_prompt(self) -> str:
        if self._phase is Phase.INSTRUCTIONS:
            return (
                "Spatial Integration\n"
                "Part 1: study 3 landscape views shown one at a time, then answer 3 follow-up questions.\n"
                "Part 2: study 3 aircraft / route views shown one at a time, then answer 3 follow-up questions.\n\n"
                "Grid questions: click a cell or type a token like B4 then Enter.\n"
                "Option questions: click a card or press 1-4 then Enter.\n"
                "Testing shortcuts: F10 skip practice, F11 skip part, F8 skip all."
            )
        if self._phase is Phase.PRACTICE_DONE:
            return self._practice_done_prompt
        if self._phase is Phase.RESULTS:
            s = self.scored_summary()
            acc = int(round(s.accuracy * 100.0))
            rt = "n/a" if s.mean_response_time_s is None else f"{s.mean_response_time_s:.2f}s"
            return (
                "Results\n"
                f"Attempted: {s.attempted}\n"
                f"Correct: {s.correct}\n"
                f"Accuracy: {acc}%\n"
                f"Mean RT: {rt}\n"
                f"Throughput: {s.throughput_per_min:.1f}/min"
            )
        if self._current_problem is None:
            return ""
        payload = self._payload_or_none(self._current_problem)
        if payload is None:
            return self._current_problem.prompt
        if payload.trial_stage is SpatialIntegrationTrialStage.STUDY:
            active_view = "View"
            if payload.active_reference_view is not None:
                active_view = str(payload.active_reference_view.label)
            return (
                f"{_part_name(payload.part)} {payload.block_kind.title()} "
                f"Scene {payload.scene_index_in_block}\n"
                f"Study {active_view} ({payload.study_view_index}/{payload.study_views_in_scene})."
            )
        return self._current_problem.prompt

    def scored_summary(self) -> AttemptSummary:
        attempted = int(self._scored_attempted)
        correct = int(self._scored_correct)
        duration_s = max(0.001, float(self._scored_elapsed_s))
        accuracy = 0.0 if attempted == 0 else correct / attempted
        throughput = (attempted / duration_s) * 60.0
        rts = [e.response_time_s for e in self._events if e.phase is Phase.SCORED]
        mean_rt = None if not rts else sum(rts) / len(rts)
        total_score = float(self._scored_total_score)
        max_score = float(self._scored_max_score)
        score_ratio = 0.0 if max_score <= 0.0 else total_score / max_score
        return AttemptSummary(
            attempted=attempted,
            correct=correct,
            accuracy=float(accuracy),
            duration_s=float(duration_s),
            throughput_per_min=float(throughput),
            mean_response_time_s=mean_rt,
            total_score=total_score,
            max_score=max_score,
            score_ratio=score_ratio,
        )

    def snapshot(self) -> TestSnapshot:
        payload = self._snapshot_payload()
        hint = "Click a cell / type B4, or press 1-4 then Enter"
        if isinstance(payload, SpatialIntegrationPayload):
            if payload.trial_stage is SpatialIntegrationTrialStage.STUDY:
                hint = "Study the current view"
            elif payload.answer_mode is SpatialIntegrationAnswerMode.GRID_CLICK:
                hint = "Click a grid cell or type B4 then Enter"
            else:
                hint = "Click an option or press 1-4 then Enter"
        return TestSnapshot(
            title="Spatial Integration",
            phase=self._phase,
            prompt=self.current_prompt(),
            input_hint=hint,
            time_remaining_s=self.time_remaining_s(),
            attempted_scored=int(self._scored_attempted),
            correct_scored=int(self._scored_correct),
            payload=payload,
            practice_feedback=self._practice_feedback,
        )

    def _begin_block(self, *, is_practice: bool) -> None:
        self._block_is_practice = bool(is_practice)
        self._phase = Phase.PRACTICE if is_practice else Phase.SCORED
        self._pending_done_action = None
        self._practice_done_prompt = ""
        self._practice_feedback = None
        self._pending_part_end_after_question = False
        self._block_scene_index = 0
        self._block_scene_target = int(self._cfg.practice_scenes_per_part) if is_practice else 0
        self._block_started_at_s = self._clock.now()
        self._block_duration_s = None if is_practice else self._scored_duration_for_part(self._active_part())
        if is_practice and self._block_scene_target <= 0:
            self._complete_active_block()
            return
        self._deal_next_scene()

    def _deal_next_scene(self) -> None:
        part = self._active_part()
        self._block_scene_index += 1
        self._current_question_idx = 0
        self._current_study_view_idx = 0
        self._current_scene = self._generator.next_scene_cluster(part=part, difficulty=self._difficulty)
        question = self._current_scene.questions[0]
        self._current_problem = self._make_problem(
            scene=self._current_scene,
            question=question,
            trial_stage=SpatialIntegrationTrialStage.STUDY,
        )
        self._question_presented_at_s = None
        self._stage_ends_at_s = self._clock.now() + self._study_step_duration(scene=self._current_scene)

    def _make_problem(
        self,
        *,
        scene: _SpatialIntegrationSceneCluster,
        question: _SpatialIntegrationQuestion,
        trial_stage: SpatialIntegrationTrialStage,
    ) -> Problem:
        part_remaining = self._block_time_remaining_s()
        if question.answer_mode is SpatialIntegrationAnswerMode.GRID_CLICK:
            answer = _encode_point(question.correct_point, grid_cols=scene.grid_cols, grid_rows=scene.grid_rows)
        else:
            answer = int(question.correct_code)
        payload = SpatialIntegrationPayload(
            part=scene.part,
            trial_stage=trial_stage,
            block_kind="practice" if self._block_is_practice else "scored",
            scene_id=int(scene.scene_id),
            scene_index_in_block=int(self._block_scene_index),
            scenes_in_block=int(self._block_scene_target) if self._block_is_practice else None,
            study_view_index=(
                min(len(scene.reference_views), max(1, self._current_study_view_idx + 1))
                if trial_stage is SpatialIntegrationTrialStage.STUDY
                else len(scene.reference_views)
            ),
            study_views_in_scene=len(scene.reference_views),
            question_index_in_scene=0 if trial_stage is SpatialIntegrationTrialStage.STUDY else int(self._current_question_idx + 1),
            questions_in_scene=len(scene.questions),
            stage_time_remaining_s=(
                self._study_step_duration(scene=scene)
                if trial_stage is SpatialIntegrationTrialStage.STUDY
                else float(self._cfg.question_time_limit_s)
            ),
            part_time_remaining_s=part_remaining,
            kind=question.kind,
            answer_mode=None if trial_stage is SpatialIntegrationTrialStage.STUDY else question.answer_mode,
            stem=question.stem,
            query_label=question.query_label,
            north_arrow_deg=int(scene.north_arrow_deg),
            scene_view=scene.scene_view,
            grid_cols=int(scene.grid_cols),
            grid_rows=int(scene.grid_rows),
            alt_levels=int(scene.alt_levels),
            reference_views=scene.reference_views,
            active_reference_view=(
                scene.reference_views[
                    max(0, min(len(scene.reference_views) - 1, int(self._current_study_view_idx)))
                ]
                if trial_stage is SpatialIntegrationTrialStage.STUDY and scene.reference_views
                else None
            ),
            hills=scene.hills,
            landmarks=scene.landmarks,
            answer_map_landmarks=question.answer_map_landmarks,
            route_points=scene.route_points,
            route_current_index=int(scene.route_current_index),
            aircraft_prev=scene.aircraft_prev,
            aircraft_now=scene.aircraft_now,
            velocity=scene.velocity,
            show_aircraft_motion=bool(scene.show_aircraft_motion),
            options=question.options,
            correct_code=int(question.correct_code),
            correct_point=question.correct_point,
            correct_answer_token=str(question.correct_answer_token),
            answer_map_route_points=question.answer_map_route_points,
        )
        return Problem(
            prompt=question.stem,
            answer=int(answer),
            payload=payload,
        )

    def _set_trial_stage(self, stage: SpatialIntegrationTrialStage) -> None:
        if self._current_problem is None or self._current_scene is None:
            return
        question = self._current_scene.questions[self._current_question_idx]
        self._current_problem = self._make_problem(
            scene=self._current_scene,
            question=question,
            trial_stage=stage,
        )

    def _after_question_completion(self, *, score: float) -> None:
        if self._phase is Phase.PRACTICE:
            self._practice_feedback = "Practice: correct" if score >= 1.0 else "Practice: incorrect"

        self._stage_ends_at_s = None
        self._question_presented_at_s = None

        should_end_part = bool(self._pending_part_end_after_question)
        self._pending_part_end_after_question = False

        if self._current_scene is None:
            self._complete_active_block()
            return

        self._current_question_idx += 1
        if self._current_question_idx < len(self._current_scene.questions) and not should_end_part:
            question = self._current_scene.questions[self._current_question_idx]
            self._current_problem = self._make_problem(
                scene=self._current_scene,
                question=question,
                trial_stage=SpatialIntegrationTrialStage.QUESTION,
            )
            self._question_presented_at_s = self._clock.now()
            self._stage_ends_at_s = self._clock.now() + float(self._cfg.question_time_limit_s)
            return

        if self._block_is_practice:
            if self._block_scene_index >= self._block_scene_target:
                self._complete_active_block()
            else:
                self._deal_next_scene()
            return

        if should_end_part or self._block_time_remaining_s() == 0.0:
            self._complete_active_block()
            return

        self._deal_next_scene()

    def _record_event(
        self,
        *,
        user_answer: int,
        raw: str,
        score: float,
        response_time_s: float,
        answered_at_s: float,
    ) -> None:
        assert self._current_problem is not None
        presented = answered_at_s if self._question_presented_at_s is None else self._question_presented_at_s
        event = QuestionEvent(
            index=len(self._events),
            phase=self._phase,
            prompt=self._current_problem.prompt,
            correct_answer=int(self._current_problem.answer),
            user_answer=int(user_answer),
            is_correct=bool(score >= 1.0 - 1e-9),
            presented_at_s=float(presented),
            answered_at_s=float(answered_at_s),
            response_time_s=float(response_time_s),
            raw=str(raw),
            score=float(score),
            max_score=1.0,
        )
        self._events.append(event)
        if self._phase is Phase.SCORED:
            self._scored_attempted += 1
            self._scored_max_score += 1.0
            self._scored_total_score += float(score)
            if score >= 1.0 - 1e-9:
                self._scored_correct += 1

    def _timeout_current_question(self, *, now: float) -> None:
        self._record_event(
            user_answer=0,
            raw="TIMEOUT",
            score=0.0,
            response_time_s=max(0.0, now - (self._question_presented_at_s or now)),
            answered_at_s=now,
        )
        if self._phase is Phase.PRACTICE:
            self._practice_feedback = "Practice: timeout"
        self._after_question_completion(score=0.0)

    def _complete_active_block(self) -> None:
        part = self._active_part()
        self._current_problem = None
        self._current_scene = None
        self._stage_ends_at_s = None
        self._question_presented_at_s = None

        if self._block_is_practice:
            self._phase = Phase.PRACTICE_DONE
            self._pending_done_action = "start_scored"
            self._practice_done_prompt = (
                f"{_part_name(part)} practice complete. Press Enter to start scored."
            )
            return

        if self._part_idx + 1 < len(self._PART_ORDER):
            next_part = self._PART_ORDER[self._part_idx + 1]
            self._phase = Phase.PRACTICE_DONE
            self._pending_done_action = "start_next_practice"
            self._practice_done_prompt = (
                f"{_part_name(part)} scored complete. "
                f"Press Enter to begin {_part_name(next_part)} practice."
            )
            return

        self._to_results()

    def _to_results(self) -> None:
        self._phase = Phase.RESULTS
        self._pending_done_action = None
        self._current_problem = None
        self._current_scene = None
        self._stage_ends_at_s = None
        self._question_presented_at_s = None

    def _handle_control_command(self, raw: str) -> bool:
        token = str(raw).strip().lower()
        if token in {"__skip_all__", "skip_all"}:
            self._to_results()
            return True

        if token in {"__skip_practice__", "skip_practice"}:
            if self._phase is not Phase.PRACTICE:
                return False
            part = self._active_part()
            self._phase = Phase.PRACTICE_DONE
            self._pending_done_action = "start_scored"
            self._practice_done_prompt = (
                f"{_part_name(part)} practice skipped. Press Enter to start scored."
            )
            self._current_problem = None
            self._current_scene = None
            self._stage_ends_at_s = None
            self._question_presented_at_s = None
            return True

        if token in {"__skip_section__", "__skip_part__", "skip_section", "skip_part"}:
            part = self._active_part()
            if self._part_idx + 1 < len(self._PART_ORDER):
                next_part = self._PART_ORDER[self._part_idx + 1]
                self._phase = Phase.PRACTICE_DONE
                self._pending_done_action = "start_next_practice"
                self._practice_done_prompt = (
                    f"{_part_name(part)} skipped. "
                    f"Press Enter to begin {_part_name(next_part)} practice."
                )
                self._current_problem = None
                self._current_scene = None
                self._stage_ends_at_s = None
                self._question_presented_at_s = None
                return True
            self._to_results()
            return True

        return False

    def _snapshot_payload(self) -> object | None:
        if self._current_problem is None:
            return None
        payload = self._payload_or_none(self._current_problem)
        if payload is None:
            return self._current_problem.payload
        stage_remaining = None
        if self._stage_ends_at_s is not None:
            stage_remaining = max(0.0, self._stage_ends_at_s - self._clock.now())
        return replace(
            payload,
            stage_time_remaining_s=stage_remaining,
            part_time_remaining_s=self._block_time_remaining_s(),
        )

    def _block_time_remaining_s(self) -> float | None:
        if self._block_is_practice or self._block_started_at_s is None or self._block_duration_s is None:
            return None
        remaining = float(self._block_duration_s) - (self._clock.now() - self._block_started_at_s)
        return max(0.0, remaining)

    def _study_duration_for_part(self, part: SpatialIntegrationPart) -> float:
        return float(
            self._cfg.static_study_s
            if part is SpatialIntegrationPart.STATIC
            else self._cfg.aircraft_study_s
        )

    def _study_step_duration(self, *, scene: _SpatialIntegrationSceneCluster) -> float:
        return float(self._study_duration_for_part(scene.part) / max(1, len(scene.reference_views)))

    def _scored_duration_for_part(self, part: SpatialIntegrationPart) -> float:
        return float(
            self._cfg.static_scored_duration_s
            if part is SpatialIntegrationPart.STATIC
            else self._cfg.aircraft_scored_duration_s
        )

    def _active_part(self) -> SpatialIntegrationPart:
        idx = max(0, min(len(self._PART_ORDER) - 1, int(self._part_idx)))
        return self._PART_ORDER[idx]

    @staticmethod
    def _payload_or_none(problem: Problem) -> SpatialIntegrationPayload | None:
        payload = problem.payload
        return payload if isinstance(payload, SpatialIntegrationPayload) else None


def build_spatial_integration_test(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    config: SpatialIntegrationConfig | None = None,
) -> SpatialIntegrationEngine:
    cfg = config or SpatialIntegrationConfig()

    skip_env = os.environ.get("CFAST_SPATIAL_SKIP_PRACTICE", "").strip().lower()
    if skip_env in {"1", "true", "yes", "on"}:
        cfg = replace(cfg, skip_practice_for_testing=True)

    start_env = os.environ.get("CFAST_SPATIAL_START_SECTION", "").strip().upper()
    if start_env in {"A", "STATIC", "PART1"}:
        cfg = replace(cfg, start_part="STATIC")
    elif start_env in {"B", "AIRCRAFT", "PART2", "C"}:
        cfg = replace(cfg, start_part="AIRCRAFT")

    return SpatialIntegrationEngine(
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        config=cfg,
    )

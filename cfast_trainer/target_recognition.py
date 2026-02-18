from __future__ import annotations

from dataclasses import dataclass

from .clock import Clock
from .cognitive_core import (
    AnswerScorer,
    Problem,
    SeededRng,
    TimedTextInputTest,
    clamp01,
    lerp_int,
)


@dataclass(frozen=True, slots=True)
class TargetRecognitionConfig:
    # Guide indicates ~25 minutes including instructions.
    scored_duration_s: float = 23.0 * 60.0
    practice_questions: int = 4


@dataclass(frozen=True, slots=True)
class TargetRecognitionSceneEntity:
    shape: str  # "truck" | "tank" | "building"
    affiliation: str  # "hostile" | "friendly" | "neutral"
    damaged: bool
    high_priority: bool


@dataclass(frozen=True, slots=True)
class TargetRecognitionSceneCriteria:
    shape: str
    affiliation: str
    require_damaged: bool | None
    require_high_priority: bool | None


@dataclass(frozen=True, slots=True)
class TargetRecognitionPayload:
    scene_rows: int
    scene_cols: int
    scene_cells: tuple[str, ...]
    scene_entities: tuple[TargetRecognitionSceneEntity, ...]
    scene_target: str
    scene_has_target: bool
    scene_target_options: tuple[str, ...]
    light_pattern: tuple[str, str, str]
    light_target_pattern: tuple[str, str, str]
    light_has_target: bool
    scan_tokens: tuple[str, ...]
    scan_target: str
    scan_has_target: bool
    system_rows: tuple[str, ...]  # compatibility mirror (first cycle, first column)
    system_target: str
    system_has_target: bool
    system_cycles: tuple["TargetRecognitionSystemCycle", ...]
    system_step_interval_s: float
    full_credit_error: int
    zero_credit_error: int


@dataclass(frozen=True, slots=True)
class TargetRecognitionSystemCycle:
    target: str
    columns: tuple[tuple[str, ...], ...]


class TargetRecognitionScorer(AnswerScorer):
    """Exact panel-count matches get full credit; near misses get estimation credit."""

    def score(self, *, problem: Problem, user_answer: int, raw: str) -> float:
        _ = raw
        payload = problem.payload
        if not isinstance(payload, TargetRecognitionPayload):
            return 1.0 if int(user_answer) == int(problem.answer) else 0.0

        full = max(0, int(payload.full_credit_error))
        zero = max(full + 1, int(payload.zero_credit_error))
        err = abs(int(user_answer) - int(problem.answer))

        if err <= full:
            return 1.0
        if err >= zero:
            return 0.0
        return clamp01((zero - err) / float(zero - full))


class TargetRecognitionGenerator:
    """Deterministic mixed-panel target-recognition trial stream."""

    _SCENE_SHAPES = ("truck", "tank", "building")
    _SCENE_AFFILIATIONS = ("hostile", "friendly", "neutral")
    _SCAN_TOKENS = ("<>", "<|", "|>", "[]", "{}", "()", "/\\", "\\/", "==", "=~")
    _LIGHT_COLORS = ("G", "B", "Y", "R")
    _ALNUM = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"

    def __init__(self, *, seed: int) -> None:
        self._rng = SeededRng(seed)

    def next_problem(self, *, difficulty: float) -> Problem:
        d = clamp01(difficulty)
        presence_prob = self._presence_probability(d)

        scene_rows = lerp_int(5, 7, d)
        scene_cols = lerp_int(8, 11, d)
        scene_has_target = self._rng.random() < presence_prob
        (
            scene_entities,
            scene_target_criteria,
            scene_target_options,
            scene_cells,
        ) = self._build_scene_entities(
            rows=scene_rows,
            cols=scene_cols,
            has_target=scene_has_target,
            difficulty=d,
        )
        scene_target = self._scene_criteria_label(scene_target_criteria)

        light_target_pattern = self._build_light_pattern()
        light_has_target = self._rng.random() < presence_prob
        light_pattern = (
            light_target_pattern
            if light_has_target
            else self._build_different_light_pattern(light_target_pattern)
        )

        scan_target = str(self._rng.choice(self._SCAN_TOKENS))
        scan_has_target = self._rng.random() < presence_prob
        scan_tokens = self._build_scan_tokens(
            count=lerp_int(10, 16, d),
            target=scan_target,
            has_target=scan_has_target,
        )

        system_row_count = lerp_int(16, 22, d)
        system_target = self._random_alnum_code(length=4)
        # User-requested behavior: this panel continuously cycles and keeps a target in-stream.
        system_has_target = True
        system_cycles = self._build_system_cycles(
            row_count=system_row_count,
            first_target=system_target,
            cycle_count=lerp_int(3, 5, d),
        )
        system_rows = system_cycles[0].columns[0] if system_cycles else ()
        system_step_interval_s = self._system_step_interval_s(d)

        expected_matches = int(scene_has_target) + int(light_has_target) + int(scan_has_target) + int(
            system_has_target
        )

        payload = TargetRecognitionPayload(
            scene_rows=scene_rows,
            scene_cols=scene_cols,
            scene_cells=scene_cells,
            scene_entities=scene_entities,
            scene_target=scene_target,
            scene_has_target=scene_has_target,
            scene_target_options=scene_target_options,
            light_pattern=light_pattern,
            light_target_pattern=light_target_pattern,
            light_has_target=light_has_target,
            scan_tokens=scan_tokens,
            scan_target=scan_target,
            scan_has_target=scan_has_target,
            system_rows=system_rows,
            system_target=system_target,
            system_has_target=system_has_target,
            system_cycles=system_cycles,
            system_step_interval_s=system_step_interval_s,
            full_credit_error=0,
            zero_credit_error=3,
        )

        return Problem(
            prompt="Use mouse only to register targets in each panel.",
            answer=expected_matches,
            payload=payload,
        )

    def _presence_probability(self, difficulty: float) -> float:
        # Harder settings reduce match frequency to force sustained scan switching.
        return max(0.45, min(0.7, 0.7 - (difficulty * 0.22)))

    def _build_scene_entities(
        self,
        *,
        rows: int,
        cols: int,
        has_target: bool,
        difficulty: float,
    ) -> tuple[
        tuple[TargetRecognitionSceneEntity, ...],
        TargetRecognitionSceneCriteria,
        tuple[str, ...],
        tuple[str, ...],
    ]:
        count = max(1, int(rows * cols))
        damaged_prob = 0.20 + (difficulty * 0.28)
        priority_prob = 0.14 + (difficulty * 0.24)

        entities: list[TargetRecognitionSceneEntity] = []
        for _ in range(count):
            entities.append(
                TargetRecognitionSceneEntity(
                    shape=str(self._rng.choice(self._SCENE_SHAPES)),
                    affiliation=str(self._rng.choice(self._SCENE_AFFILIATIONS)),
                    damaged=self._rng.random() < damaged_prob,
                    high_priority=self._rng.random() < priority_prob,
                )
            )
        entities_tuple = tuple(entities)
        if has_target:
            criteria = self._pick_present_scene_criteria(entities=entities_tuple, difficulty=difficulty)
        else:
            criteria = self._pick_absent_scene_criteria(entities=entities_tuple)
        target_options = self._build_scene_target_options(
            criteria=criteria,
            entities=entities_tuple,
            has_target=has_target,
            difficulty=difficulty,
        )
        cells = tuple(self._scene_entity_code(e) for e in entities)
        return entities_tuple, criteria, target_options, cells

    def _pick_present_scene_criteria(
        self,
        *,
        entities: tuple[TargetRecognitionSceneEntity, ...],
        difficulty: float,
    ) -> TargetRecognitionSceneCriteria:
        if not entities:
            return TargetRecognitionSceneCriteria(
                shape="truck",
                affiliation="friendly",
                require_damaged=None,
                require_high_priority=None,
            )

        priority_pool = [e for e in entities if e.damaged or e.high_priority]
        if priority_pool and self._rng.random() < (0.45 + (difficulty * 0.45)):
            anchor = priority_pool[int(self._rng.randint(0, len(priority_pool) - 1))]
        else:
            anchor = entities[int(self._rng.randint(0, len(entities) - 1))]

        require_damaged = anchor.damaged if (anchor.damaged and self._rng.random() < 0.8) else None
        require_hp = anchor.high_priority if (anchor.high_priority and self._rng.random() < 0.8) else None

        if require_damaged is None and require_hp is None:
            if anchor.high_priority:
                require_hp = True
            elif anchor.damaged:
                require_damaged = True

        return TargetRecognitionSceneCriteria(
            shape=anchor.shape,
            affiliation=anchor.affiliation,
            require_damaged=require_damaged,
            require_high_priority=require_hp,
        )

    def _pick_absent_scene_criteria(
        self,
        *,
        entities: tuple[TargetRecognitionSceneEntity, ...],
    ) -> TargetRecognitionSceneCriteria:
        absent = self._scene_absent_candidates(entities=entities)
        if absent:
            return absent[int(self._rng.randint(0, len(absent) - 1))]
        # Fallback: still deterministic and specific even if dense scene accidentally covers all candidates.
        return TargetRecognitionSceneCriteria(
            shape="truck",
            affiliation="hostile",
            require_damaged=True,
            require_high_priority=True,
        )

    def _build_scene_target_options(
        self,
        *,
        criteria: TargetRecognitionSceneCriteria,
        entities: tuple[TargetRecognitionSceneEntity, ...],
        has_target: bool,
        difficulty: float,
    ) -> tuple[str, ...]:
        labels: list[str] = [self._scene_criteria_label(criteria)]
        seen = set(labels)
        target_count = lerp_int(4, 6, difficulty)
        target_count = max(1, target_count)

        if has_target:
            pool = self._scene_present_candidates(entities=entities)
        else:
            pool = self._scene_absent_candidates(entities=entities)

        for candidate in pool:
            label = self._scene_criteria_label(candidate)
            if label in seen:
                continue
            labels.append(label)
            seen.add(label)
            if len(labels) >= target_count:
                break
        return tuple(labels)

    def _scene_present_candidates(
        self,
        *,
        entities: tuple[TargetRecognitionSceneEntity, ...],
    ) -> tuple[TargetRecognitionSceneCriteria, ...]:
        dedup: dict[tuple[str, str, bool | None, bool | None], TargetRecognitionSceneCriteria] = {}
        for entity in entities:
            variants = [
                TargetRecognitionSceneCriteria(
                    shape=entity.shape,
                    affiliation=entity.affiliation,
                    require_damaged=None,
                    require_high_priority=None,
                )
            ]
            if entity.damaged:
                variants.append(
                    TargetRecognitionSceneCriteria(
                        shape=entity.shape,
                        affiliation=entity.affiliation,
                        require_damaged=True,
                        require_high_priority=None,
                    )
                )
            if entity.high_priority:
                variants.append(
                    TargetRecognitionSceneCriteria(
                        shape=entity.shape,
                        affiliation=entity.affiliation,
                        require_damaged=None,
                        require_high_priority=True,
                    )
                )
            if entity.damaged and entity.high_priority:
                variants.append(
                    TargetRecognitionSceneCriteria(
                        shape=entity.shape,
                        affiliation=entity.affiliation,
                        require_damaged=True,
                        require_high_priority=True,
                    )
                )
            for candidate in variants:
                key = (
                    candidate.shape,
                    candidate.affiliation,
                    candidate.require_damaged,
                    candidate.require_high_priority,
                )
                dedup[key] = candidate

        candidates = list(dedup.values())
        # Deterministic shuffle of the candidate pool.
        for i in range(len(candidates) - 1, 0, -1):
            j = int(self._rng.randint(0, i))
            candidates[i], candidates[j] = candidates[j], candidates[i]
        return tuple(candidates)

    def _scene_absent_candidates(
        self,
        *,
        entities: tuple[TargetRecognitionSceneEntity, ...],
    ) -> tuple[TargetRecognitionSceneCriteria, ...]:
        candidates: list[TargetRecognitionSceneCriteria] = []
        for shape in self._SCENE_SHAPES:
            for affiliation in self._SCENE_AFFILIATIONS:
                for req_d in (None, True):
                    for req_h in (None, True):
                        candidates.append(
                            TargetRecognitionSceneCriteria(
                                shape=shape,
                                affiliation=affiliation,
                                require_damaged=req_d,
                                require_high_priority=req_h,
                            )
                        )
        absent = [c for c in candidates if not any(self._scene_matches(e, c) for e in entities)]
        # Deterministic shuffle of the candidate pool.
        for i in range(len(absent) - 1, 0, -1):
            j = int(self._rng.randint(0, i))
            absent[i], absent[j] = absent[j], absent[i]
        return tuple(absent)

    @staticmethod
    def _scene_matches(entity: TargetRecognitionSceneEntity, criteria: TargetRecognitionSceneCriteria) -> bool:
        if entity.shape != criteria.shape:
            return False
        if entity.affiliation != criteria.affiliation:
            return False
        if criteria.require_damaged is True and not entity.damaged:
            return False
        if criteria.require_high_priority is True and not entity.high_priority:
            return False
        return True

    @staticmethod
    def _scene_entity_code(entity: TargetRecognitionSceneEntity) -> str:
        shape_code = {
            "truck": "TRK",
            "tank": "TNK",
            "building": "BLD",
        }.get(entity.shape, "UNK")
        side_code = {
            "hostile": "H",
            "friendly": "F",
            "neutral": "N",
        }.get(entity.affiliation, "N")
        flags = ("D" if entity.damaged else "") + ("P" if entity.high_priority else "")
        return f"{shape_code}:{side_code}{flags}"

    @staticmethod
    def _scene_criteria_label(criteria: TargetRecognitionSceneCriteria) -> str:
        words: list[str] = []
        if criteria.require_damaged is True:
            words.append("Damaged")
        words.append(criteria.affiliation.title())
        words.append(criteria.shape.title())
        label = " ".join(words)
        if criteria.require_high_priority is True:
            label = f"{label} (HP)"
        return label

    def _build_light_pattern(self) -> tuple[str, str, str]:
        return tuple(str(self._rng.choice(self._LIGHT_COLORS)) for _ in range(3))

    def _build_different_light_pattern(
        self, target_pattern: tuple[str, str, str]
    ) -> tuple[str, str, str]:
        for _ in range(32):
            pattern = self._build_light_pattern()
            if pattern != target_pattern:
                return pattern
        # Defensive fallback to guarantee a mismatch.
        first = self._LIGHT_COLORS[0]
        alt = self._LIGHT_COLORS[1]
        return (alt if target_pattern[0] == first else first, target_pattern[1], target_pattern[2])

    def _build_scan_tokens(self, *, count: int, target: str, has_target: bool) -> tuple[str, ...]:
        count = max(1, int(count))
        distractors = tuple(tok for tok in self._SCAN_TOKENS if tok != target)
        tokens = [str(self._rng.choice(distractors)) for _ in range(count)]
        if has_target:
            idx = int(self._rng.randint(0, count - 1))
            tokens[idx] = target
        return tuple(tokens)

    def _system_step_interval_s(self, difficulty: float) -> float:
        # Slow but increasing cadence as difficulty rises.
        return max(0.20, min(0.80, 0.70 - (difficulty * 0.35)))

    def _build_system_cycles(
        self,
        *,
        row_count: int,
        first_target: str,
        cycle_count: int,
    ) -> tuple[TargetRecognitionSystemCycle, ...]:
        rows = max(6, int(row_count))
        n_cycles = max(2, int(cycle_count))
        n_cols = 3
        target_col = 1

        targets = [first_target]
        while len(targets) < n_cycles:
            candidate = self._random_alnum_code(length=4)
            if candidate in targets:
                continue
            targets.append(candidate)

        cycles: list[TargetRecognitionSystemCycle] = []
        for target in targets:
            cols: list[tuple[str, ...]] = []
            for col_idx in range(n_cols):
                values = [self._random_alnum_code(length=4, exclude=target) for _ in range(rows)]
                if col_idx == target_col:
                    values[0] = target
                cols.append(tuple(values))
            cycles.append(TargetRecognitionSystemCycle(target=target, columns=tuple(cols)))
        return tuple(cycles)

    def _random_alnum_code(self, *, length: int, exclude: str | None = None) -> str:
        length = max(1, int(length))
        while True:
            code = "".join(str(self._rng.choice(self._ALNUM)) for _ in range(length))
            if exclude is not None and code == exclude:
                continue
            return code


def build_target_recognition_test(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    config: TargetRecognitionConfig | None = None,
) -> TimedTextInputTest:
    cfg = config or TargetRecognitionConfig()

    instructions = [
        "Target Recognition Test",
        "",
        "This task combines multiple simultaneous visual scan/search panels:",
        "- Scene panel (imagery + warning signs)",
        "- Light panel (color pattern)",
        "- Scan panel (symbol/code stream)",
        "- System panel (alphanumeric strings)",
        "",
        "For each screen, compare every panel against its target strip.",
        "Scene target list entries are active instructions (shape + affiliation + status flags).",
        "Use mouse only: click targets directly in each panel when they match.",
        "",
        "You will complete a short practice, then a timed scored block.",
    ]

    return TimedTextInputTest(
        title="Target Recognition",
        instructions=instructions,
        generator=TargetRecognitionGenerator(seed=seed),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        practice_questions=cfg.practice_questions,
        scored_duration_s=cfg.scored_duration_s,
        scorer=TargetRecognitionScorer(),
    )

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .clock import Clock
from .content_variants import content_metadata_from_payload, stable_variant_id
from .cognitive_core import (
    AnswerScorer,
    AttemptSummary,
    Phase,
    Problem,
    QuestionEvent,
    SeededRng,
    TestSnapshot,
    clamp01,
    lerp_int,
)


@dataclass(frozen=True, slots=True)
class InstrumentComprehensionConfig:
    # Candidate guide indicates ~26 minutes total including instructions.
    scored_duration_s: float = 22.0 * 60.0
    practice_questions: int = 1


class InstrumentComprehensionTrialKind(StrEnum):
    INSTRUMENTS_TO_DESCRIPTION = "instruments_to_description"
    INSTRUMENTS_TO_AIRCRAFT = "instruments_to_aircraft"
    AIRCRAFT_TO_INSTRUMENTS = "aircraft_to_instruments"


class InstrumentAircraftViewPreset(StrEnum):
    FRONT_LEFT = "front_left"
    FRONT_RIGHT = "front_right"
    PROFILE_LEFT = "profile_left"
    PROFILE_RIGHT = "profile_right"
    TOP_OBLIQUE = "top_oblique"


class InstrumentOptionRenderMode(StrEnum):
    AIRCRAFT = "aircraft"
    INSTRUMENT_PANEL = "instrument_panel"
    DESCRIPTION = "description"


class InstrumentHeadingDisplayMode(StrEnum):
    ROTATING_ROSE = "rotating_rose"
    MOVING_ARROW = "moving_arrow"


@dataclass(frozen=True, slots=True)
class InstrumentState:
    speed_kts: int
    altitude_ft: int
    vertical_rate_fpm: int
    bank_deg: int
    pitch_deg: int
    heading_deg: int
    slip: int  # -1=slip left, 0=balanced, 1=slip right


@dataclass(frozen=True, slots=True)
class InstrumentOption:
    code: int
    state: InstrumentState
    description: str
    view_preset: InstrumentAircraftViewPreset | None = None
    distractor_tag: str = ""
    distractor_profile_tag: str = ""


@dataclass(frozen=True, slots=True)
class InstrumentComprehensionPayload:
    kind: InstrumentComprehensionTrialKind
    prompt_state: InstrumentState
    prompt_description: str
    options: tuple[InstrumentOption, ...]
    option_errors: tuple[int, ...]
    full_credit_error: int
    zero_credit_error: int
    prompt_view_preset: InstrumentAircraftViewPreset | None = None
    option_render_mode: InstrumentOptionRenderMode = InstrumentOptionRenderMode.DESCRIPTION
    heading_display_mode: InstrumentHeadingDisplayMode = InstrumentHeadingDisplayMode.ROTATING_ROSE
    content_family: str = ""
    variant_id: str = ""
    content_pack: str = "instrument_comprehension"


from .instrument_orientation_solver import (
    INSTRUMENT_COMMON_MISREAD_TAGS,
    INSTRUMENT_DISTRACTOR_FALLBACKS,
    apply_distractor_profile,
    display_match_error,
    display_observation_from_state,
    describe_instrument_state,
    lower_band_profile_pool,
    nearest_profile_candidates,
    solve_instrument_interpretation,
    interpretation_error,
)


@dataclass(frozen=True, slots=True)
class _InstrumentOptionSeed:
    state: InstrumentState
    distractor_tag: str
    distractor_profile_tag: str


_PART_ORDER = (
    InstrumentComprehensionTrialKind.INSTRUMENTS_TO_AIRCRAFT,
    InstrumentComprehensionTrialKind.AIRCRAFT_TO_INSTRUMENTS,
    InstrumentComprehensionTrialKind.INSTRUMENTS_TO_DESCRIPTION,
)

_PART1_VIEW_PRESETS: tuple[InstrumentAircraftViewPreset, ...] = (
    InstrumentAircraftViewPreset.FRONT_LEFT,
    InstrumentAircraftViewPreset.FRONT_RIGHT,
    InstrumentAircraftViewPreset.PROFILE_LEFT,
    InstrumentAircraftViewPreset.PROFILE_RIGHT,
    InstrumentAircraftViewPreset.TOP_OBLIQUE,
)
_PART2_PROMPT_VIEW_PRESET = InstrumentAircraftViewPreset.FRONT_LEFT


def instrument_aircraft_view_preset_for_code(code: int) -> InstrumentAircraftViewPreset:
    idx = max(0, min(len(_PART1_VIEW_PRESETS) - 1, int(code) - 1))
    return _PART1_VIEW_PRESETS[idx]


def instrument_aircraft_reverse_prompt_view_preset() -> InstrumentAircraftViewPreset:
    return _PART2_PROMPT_VIEW_PRESET

def altimeter_hand_turns(altitude_ft: int) -> tuple[float, float]:
    """Return clockwise turns from the 12 o'clock position.

    The short hand makes one full revolution per 10,000 ft and indicates
    thousands. The long hand makes one full revolution per 1,000 ft and
    indicates hundreds.
    """

    altitude = max(0, int(altitude_ft))
    thousands_turn = (altitude % 10000) / 10000.0
    hundreds_turn = (altitude % 1000) / 1000.0
    return thousands_turn, hundreds_turn


def airspeed_turn(speed_kts: int) -> float:
    """Return the clockwise turns for a 0-360 knot circular airspeed dial."""

    speed = max(0, min(360, int(speed_kts)))
    return speed / 360.0


class InstrumentComprehensionScorer(AnswerScorer):
    """Exact option gets full credit; near interpretation gets partial credit."""

    def score(self, *, problem: Problem, user_answer: int, raw: str) -> float:
        _ = raw

        payload = problem.payload
        if not isinstance(payload, InstrumentComprehensionPayload):
            return 1.0 if int(user_answer) == int(problem.answer) else 0.0

        by_code = {opt.code: idx for idx, opt in enumerate(payload.options)}
        idx = by_code.get(int(user_answer))
        if idx is None or idx >= len(payload.option_errors):
            return 0.0

        err = max(0, int(payload.option_errors[idx]))
        full = max(0, int(payload.full_credit_error))
        zero = max(full + 1, int(payload.zero_credit_error))

        if err <= full:
            return 1.0
        if err >= zero:
            return 0.0
        return clamp01((zero - err) / float(zero - full))


class InstrumentComprehensionGenerator:
    """Deterministic generator for the guide-style two-part instrument test."""

    def __init__(self, *, seed: int) -> None:
        self._rng = SeededRng(seed)
        self._kind_offset = int(self._rng.randint(0, len(_PART_ORDER) - 1))
        self._trial_index = 0

    def next_problem(self, *, difficulty: float) -> Problem:
        kind = _PART_ORDER[(self._kind_offset + self._trial_index) % len(_PART_ORDER)]
        self._trial_index += 1
        return self.next_problem_for_kind(kind=kind, difficulty=difficulty)

    def next_problem_for_kind(
        self,
        *,
        kind: InstrumentComprehensionTrialKind,
        difficulty: float,
    ) -> Problem:
        d = clamp01(difficulty)
        option_render_mode = {
            InstrumentComprehensionTrialKind.INSTRUMENTS_TO_AIRCRAFT: InstrumentOptionRenderMode.AIRCRAFT,
            InstrumentComprehensionTrialKind.AIRCRAFT_TO_INSTRUMENTS: InstrumentOptionRenderMode.INSTRUMENT_PANEL,
            InstrumentComprehensionTrialKind.INSTRUMENTS_TO_DESCRIPTION: InstrumentOptionRenderMode.DESCRIPTION,
        }[kind]
        heading_display_mode = self._sample_heading_display_mode()
        sampled_state = self._sample_state(difficulty=d)
        solved_state = solve_instrument_interpretation(
            prompt_state=sampled_state,
            kind=kind,
            heading_display_mode=heading_display_mode,
        )
        prompt_observation = display_observation_from_state(
            solved_state,
            heading_display_mode,
        )
        prompt_view_preset = (
            instrument_aircraft_reverse_prompt_view_preset()
            if kind is InstrumentComprehensionTrialKind.AIRCRAFT_TO_INSTRUMENTS
            else None
        )
        option_seeds = self._build_option_seeds(
            state=solved_state,
            difficulty=d,
            kind=kind,
            heading_display_mode=heading_display_mode,
        )
        order = self._rng.sample((0, 1, 2, 3, 4), k=5)
        options: list[InstrumentOption] = []
        for code, idx in enumerate(order, start=1):
            seed = option_seeds[idx]
            option = InstrumentOption(
                code=code,
                state=seed.state,
                description=describe_instrument_state(seed.state),
                view_preset=(
                    instrument_aircraft_view_preset_for_code(code)
                    if kind is InstrumentComprehensionTrialKind.INSTRUMENTS_TO_AIRCRAFT
                    else None
                ),
                distractor_tag=seed.distractor_tag,
                distractor_profile_tag=seed.distractor_profile_tag,
            )
            options.append(option)

        options = self._repair_option_render_collisions(
            options=options,
            correct_state=solved_state,
            difficulty=d,
            kind=kind,
            heading_display_mode=heading_display_mode,
            prompt_view_preset=prompt_view_preset,
        )
        self._validate_aircraft_option_semantics(
            options=options,
            prompt_state=solved_state,
            kind=kind,
            prompt_view_preset=prompt_view_preset,
        )
        option_errors = tuple(
            display_match_error(
                prompt_observation,
                option.state,
                kind=kind,
                heading_display_mode=heading_display_mode,
            )
            for option in options
        )
        correct_indices = [
            idx for idx, option in enumerate(options) if option.distractor_tag == "correct"
        ]
        if len(correct_indices) != 1:
            raise RuntimeError("Instrument Comprehension expected exactly one correct option")
        correct_idx = correct_indices[0]
        min_error = min(option_errors)
        if option_errors[correct_idx] != min_error or option_errors.count(min_error) != 1:
            raise RuntimeError("Instrument Comprehension correct option lost display-model uniqueness")
        correct_code = int(options[correct_idx].code)
        payload = InstrumentComprehensionPayload(
            kind=kind,
            prompt_state=solved_state,
            prompt_description=describe_instrument_state(solved_state),
            options=tuple(options),
            option_errors=option_errors,
            full_credit_error=0,
            zero_credit_error=lerp_int(130, 70, d),
            prompt_view_preset=prompt_view_preset,
            option_render_mode=option_render_mode,
            heading_display_mode=heading_display_mode,
            content_family=str(kind.value),
            variant_id=stable_variant_id(
                kind.value,
                option_render_mode.value,
                heading_display_mode.value,
            ),
        )
        return Problem(
            prompt=self._prompt_for(kind=kind),
            answer=int(correct_code),
            payload=payload,
        )

    def _prompt_for(self, *, kind: InstrumentComprehensionTrialKind) -> str:
        if kind is InstrumentComprehensionTrialKind.INSTRUMENTS_TO_AIRCRAFT:
            return "Read the instruments and choose the matching aircraft image (A/S/D/F/G)."
        if kind is InstrumentComprehensionTrialKind.AIRCRAFT_TO_INSTRUMENTS:
            return "Read the aircraft image and choose the matching instrument panel (A/S/D/F/G)."
        return (
            "Read the instrument panel and choose the best matching "
            "flight description (A/S/D/F/G)."
        )

    def _sample_state(self, *, difficulty: float) -> InstrumentState:
        speed_step = max(5, lerp_int(20, 5, difficulty))
        altitude_step = max(100, lerp_int(500, 100, difficulty))
        heading_step = max(5, lerp_int(45, 5, difficulty))

        speed = self._sample_quantized(120, 360, speed_step)
        altitude = self._sample_quantized(1000, 9500, altitude_step)
        heading = self._sample_quantized(0, 359, heading_step)

        bank_mag = self._rng.randint(lerp_int(8, 4, difficulty), lerp_int(35, 22, difficulty))
        bank_sign = -1 if self._rng.random() < 0.5 else 1
        bank = int(bank_sign * bank_mag)

        pitch_mag = self._rng.randint(lerp_int(2, 1, difficulty), lerp_int(12, 8, difficulty))
        pitch_sign = -1 if self._rng.random() < 0.55 else 1
        pitch = int(pitch_sign * pitch_mag)

        vertical_sign = (
            -1 if pitch < 0 else 1 if pitch > 0 else (-1 if self._rng.random() < 0.5 else 1)
        )
        vertical_mag = self._rng.randint(0, lerp_int(1200, 2200, difficulty))
        vertical_rate = int(vertical_sign * (vertical_mag // 100) * 100)

        slip_choices = (0, 0, -1, 1) if difficulty >= 0.45 else (0, 0, 0, -1, 1)
        slip = int(self._rng.choice(slip_choices))

        return InstrumentState(
            speed_kts=int(speed),
            altitude_ft=int(altitude),
            vertical_rate_fpm=int(vertical_rate),
            bank_deg=int(bank),
            pitch_deg=int(pitch),
            heading_deg=int(heading),
            slip=int(slip),
        )

    def _sample_heading_display_mode(self) -> InstrumentHeadingDisplayMode:
        return InstrumentHeadingDisplayMode(
            self._rng.choice(
                (
                    InstrumentHeadingDisplayMode.ROTATING_ROSE,
                    InstrumentHeadingDisplayMode.MOVING_ARROW,
                )
            )
        )

    def _build_option_seeds(
        self,
        *,
        state: InstrumentState,
        difficulty: float,
        kind: InstrumentComprehensionTrialKind,
        heading_display_mode: InstrumentHeadingDisplayMode,
    ) -> tuple[_InstrumentOptionSeed, ...]:
        used_states = {state}
        used_tags = {"correct"}
        seeds: list[_InstrumentOptionSeed] = [
            _InstrumentOptionSeed(
                state=state,
                distractor_tag="correct",
                distractor_profile_tag="correct",
            )
        ]

        for tag in INSTRUMENT_COMMON_MISREAD_TAGS:
            seeds.append(
                self._pick_tagged_distractor(
                    requested_tag=tag,
                    used_states=used_states,
                    used_tags=used_tags,
                    state=state,
                    difficulty=difficulty,
                    kind=kind,
                    heading_display_mode=heading_display_mode,
                )
            )

        seeds.append(
            self._pick_easier_band_random_distractor(
                used_states=used_states,
                used_tags=used_tags,
                state=state,
                difficulty=difficulty,
                kind=kind,
                heading_display_mode=heading_display_mode,
            )
        )
        seeds.append(
            self._pick_nearest_distractor(
                used_states=used_states,
                used_tags=used_tags,
                state=state,
                difficulty=difficulty,
                kind=kind,
                heading_display_mode=heading_display_mode,
            )
        )
        return tuple(seeds)

    def _pick_tagged_distractor(
        self,
        *,
        requested_tag: str,
        used_states: set[InstrumentState],
        used_tags: set[str],
        state: InstrumentState,
        difficulty: float,
        kind: InstrumentComprehensionTrialKind,
        heading_display_mode: InstrumentHeadingDisplayMode,
        candidate_tags: tuple[str, ...] | None = None,
    ) -> _InstrumentOptionSeed:
        ordered_tags = (
            tuple(candidate_tags)
            if candidate_tags is not None
            else (requested_tag, *self._fallback_tags(kind))
        )
        for tag in ordered_tags:
            if tag in used_tags and tag != requested_tag:
                continue
            candidate_state = apply_distractor_profile(
                state,
                profile_tag=tag,
                kind=kind,
                heading_display_mode=heading_display_mode,
                difficulty=difficulty,
            )
            if candidate_state in used_states:
                continue
            if (
                interpretation_error(
                    state,
                    candidate_state,
                    kind=kind,
                    heading_display_mode=heading_display_mode,
                )
                <= 0
            ):
                continue
            used_states.add(candidate_state)
            used_tags.add(tag)
            return _InstrumentOptionSeed(
                state=candidate_state,
                distractor_tag=requested_tag,
                distractor_profile_tag=tag,
            )
        raise RuntimeError(f"Could not build Instrument Comprehension distractor for tag {requested_tag}")

    def _pick_easier_band_random_distractor(
        self,
        *,
        used_states: set[InstrumentState],
        used_tags: set[str],
        state: InstrumentState,
        difficulty: float,
        kind: InstrumentComprehensionTrialKind,
        heading_display_mode: InstrumentHeadingDisplayMode,
    ) -> _InstrumentOptionSeed:
        base_pool = tuple(
            tag
            for tag in self._lower_band_profile_pool(kind=kind, difficulty=difficulty)
            if tag not in used_tags
        )
        if base_pool:
            start = int(self._rng.randint(0, len(base_pool) - 1))
            ordered = base_pool[start:] + base_pool[:start]
        else:
            ordered = ()
        return self._pick_tagged_distractor(
            requested_tag="easier_band_random",
            used_states=used_states,
            used_tags=used_tags,
            state=state,
            difficulty=difficulty,
            kind=kind,
            heading_display_mode=heading_display_mode,
            candidate_tags=tuple(ordered) + self._fallback_tags(kind),
        )

    def _pick_nearest_distractor(
        self,
        *,
        used_states: set[InstrumentState],
        used_tags: set[str],
        state: InstrumentState,
        difficulty: float,
        kind: InstrumentComprehensionTrialKind,
        heading_display_mode: InstrumentHeadingDisplayMode,
    ) -> _InstrumentOptionSeed:
        ranked: list[tuple[int, int, str, InstrumentState]] = []
        for idx, tag in enumerate(self._nearest_profile_candidates(kind=kind, difficulty=difficulty)):
            if tag in used_tags:
                continue
            candidate_state = apply_distractor_profile(
                state,
                profile_tag=tag,
                kind=kind,
                heading_display_mode=heading_display_mode,
                difficulty=difficulty,
            )
            if candidate_state in used_states:
                continue
            err = interpretation_error(
                state,
                candidate_state,
                kind=kind,
                heading_display_mode=heading_display_mode,
            )
            if err <= 0:
                continue
            ranked.append((int(err), int(idx), tag, candidate_state))
        if not ranked:
            return self._pick_tagged_distractor(
                requested_tag="nearest_solver",
                used_states=used_states,
                used_tags=used_tags,
                state=state,
                difficulty=difficulty,
                kind=kind,
                heading_display_mode=heading_display_mode,
                candidate_tags=self._fallback_tags(kind),
            )
        _err, _idx, tag, candidate_state = min(ranked, key=lambda item: (item[0], item[1], item[2]))
        used_states.add(candidate_state)
        used_tags.add(tag)
        return _InstrumentOptionSeed(
            state=candidate_state,
            distractor_tag="nearest_solver",
            distractor_profile_tag=tag,
        )

    def _repair_option_render_collisions(
        self,
        *,
        options: list[InstrumentOption],
        correct_state: InstrumentState,
        difficulty: float,
        kind: InstrumentComprehensionTrialKind,
        heading_display_mode: InstrumentHeadingDisplayMode,
        prompt_view_preset: InstrumentAircraftViewPreset | None,
    ) -> list[InstrumentOption]:
        if kind not in (
            InstrumentComprehensionTrialKind.INSTRUMENTS_TO_AIRCRAFT,
            InstrumentComprehensionTrialKind.AIRCRAFT_TO_INSTRUMENTS,
        ):
            return options

        from .instrument_aircraft_cards import (
            aircraft_card_pose_distance,
            aircraft_card_pose_signature,
        )

        repaired = list(options)
        threshold = 26.0
        for idx, option in enumerate(repaired):
            if option.distractor_tag == "correct":
                continue
            attempts = 0
            while True:
                current_signature = aircraft_card_pose_signature(
                    option.state,
                    view_preset=(
                        option.view_preset
                        if kind is InstrumentComprehensionTrialKind.INSTRUMENTS_TO_AIRCRAFT
                        else (prompt_view_preset or InstrumentAircraftViewPreset.FRONT_LEFT)
                    ),
                )
                collision = False
                for earlier in repaired[:idx]:
                    earlier_signature = aircraft_card_pose_signature(
                        earlier.state,
                        view_preset=(
                            earlier.view_preset
                            if kind is InstrumentComprehensionTrialKind.INSTRUMENTS_TO_AIRCRAFT
                            else (prompt_view_preset or InstrumentAircraftViewPreset.FRONT_LEFT)
                        ),
                    )
                    if aircraft_card_pose_distance(current_signature, earlier_signature) < threshold:
                        collision = True
                        break
                if not collision:
                    break
                attempts += 1
                if attempts > 8:
                    break
                used_states = {correct_state, option.state, *(opt.state for opt in repaired[:idx])}
                used_tags = {
                    "correct",
                    option.distractor_profile_tag,
                    *(opt.distractor_profile_tag for opt in repaired[:idx]),
                }
                if option.distractor_tag == "easier_band_random":
                    replacement = self._pick_easier_band_random_distractor(
                        used_states=used_states,
                        used_tags=used_tags,
                        state=correct_state,
                        difficulty=difficulty,
                        kind=kind,
                        heading_display_mode=heading_display_mode,
                    )
                elif option.distractor_tag == "nearest_solver":
                    replacement = self._pick_nearest_distractor(
                        used_states=used_states,
                        used_tags=used_tags,
                        state=correct_state,
                        difficulty=difficulty,
                        kind=kind,
                        heading_display_mode=heading_display_mode,
                    )
                else:
                    replacement = self._pick_tagged_distractor(
                        requested_tag=option.distractor_tag,
                        used_states=used_states,
                        used_tags=used_tags,
                        state=correct_state,
                        difficulty=difficulty,
                        kind=kind,
                        heading_display_mode=heading_display_mode,
                    )
                repaired[idx] = InstrumentOption(
                    code=option.code,
                    state=replacement.state,
                    description=describe_instrument_state(replacement.state),
                    view_preset=option.view_preset,
                    distractor_tag=replacement.distractor_tag,
                    distractor_profile_tag=replacement.distractor_profile_tag,
                )
                option = repaired[idx]
        return repaired

    def _validate_aircraft_option_semantics(
        self,
        *,
        options: list[InstrumentOption],
        prompt_state: InstrumentState,
        kind: InstrumentComprehensionTrialKind,
        prompt_view_preset: InstrumentAircraftViewPreset | None,
    ) -> None:
        if kind not in (
            InstrumentComprehensionTrialKind.INSTRUMENTS_TO_AIRCRAFT,
            InstrumentComprehensionTrialKind.AIRCRAFT_TO_INSTRUMENTS,
        ):
            return

        from .instrument_aircraft_cards import aircraft_card_semantic_drift_tags

        if kind is InstrumentComprehensionTrialKind.AIRCRAFT_TO_INSTRUMENTS:
            prompt_tags = aircraft_card_semantic_drift_tags(
                prompt_state,
                view_preset=prompt_view_preset or InstrumentAircraftViewPreset.FRONT_LEFT,
            )
            if prompt_tags:
                raise RuntimeError(
                    f"Instrument aircraft prompt semantic drift: {', '.join(prompt_tags)}"
                )

        for option in options:
            preset = (
                option.view_preset
                if kind is InstrumentComprehensionTrialKind.INSTRUMENTS_TO_AIRCRAFT
                else (prompt_view_preset or InstrumentAircraftViewPreset.FRONT_LEFT)
            )
            drift_tags = aircraft_card_semantic_drift_tags(
                option.state,
                view_preset=preset or InstrumentAircraftViewPreset.FRONT_LEFT,
            )
            if drift_tags:
                raise RuntimeError(
                    f"Instrument aircraft option semantic drift for code {option.code}: "
                    + ", ".join(drift_tags)
                )

    def _lower_band_profile_pool(
        self,
        *,
        kind: InstrumentComprehensionTrialKind,
        difficulty: float,
    ) -> tuple[str, ...]:
        return lower_band_profile_pool(kind=kind, difficulty=difficulty)

    def _nearest_profile_candidates(
        self,
        *,
        kind: InstrumentComprehensionTrialKind,
        difficulty: float,
    ) -> tuple[str, ...]:
        return nearest_profile_candidates(kind=kind, difficulty=difficulty)

    def _fallback_tags(self, kind: InstrumentComprehensionTrialKind) -> tuple[str, ...]:
        _ = kind
        return INSTRUMENT_DISTRACTOR_FALLBACKS

    def _sample_quantized(self, lo: int, hi: int, step: int) -> int:
        if step <= 1:
            return int(self._rng.randint(lo, hi))
        min_q = lo // step
        max_q = hi // step
        q = int(self._rng.randint(min_q, max_q))
        v = q * step
        if v < lo:
            v += step
        if v > hi:
            v -= step
        return int(max(lo, min(hi, v)))

    @staticmethod
    def _clamp(v: int, lo: int, hi: int) -> int:
        return int(lo if v < lo else hi if v > hi else v)


class InstrumentComprehensionEngine:
    """Three-part guide-style instrument comprehension engine.

    Part 1: instruments -> aircraft orientation image
    Part 2: aircraft image -> instrument panel
    Part 3: instruments -> flight description
    """

    def __init__(
        self,
        *,
        clock: Clock,
        seed: int,
        difficulty: float = 0.5,
        config: InstrumentComprehensionConfig | None = None,
    ) -> None:
        cfg = config or InstrumentComprehensionConfig()
        self._clock = clock
        self._seed = int(seed)
        self._difficulty = clamp01(difficulty)
        self._cfg = cfg
        self._generator = InstrumentComprehensionGenerator(seed=self._seed)
        self._scorer = InstrumentComprehensionScorer()
        self._instructions = [
            "Instrument Comprehension",
            "",
            "Part 1: read the attitude and heading instruments, "
            "then choose the matching aircraft image.",
            "Part 2: read the aircraft image, "
            "then choose the matching full instrument panel.",
            "Part 3: read the full instrument panel, "
            "then choose the matching flight description.",
            "",
            "Some heading questions use the rotating compass card. Others keep the compass fixed and move the red heading arrow.",
            "",
            "Controls:",
            "- Press A, S, D, F, or G to choose an option",
            "- Press Enter to submit",
        ]

        self._phase = Phase.INSTRUCTIONS
        self._part_idx = 0
        self._current: Problem | None = None
        self._presented_at_s: float | None = None
        self._practice_answered = 0
        self._scored_started_at_s: float | None = None
        self._pending_done_action: str | None = None  # "start_scored" | "next_part_practice"

        self._events: list[QuestionEvent] = []
        self._scored_attempted = 0
        self._scored_correct = 0
        self._scored_total_score = 0.0
        self._scored_max_score = 0.0

    @property
    def phase(self) -> Phase:
        return self._phase

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def difficulty(self) -> float:
        return float(self._difficulty)

    def events(self) -> list[QuestionEvent]:
        return list(self._events)

    def instructions(self) -> list[str]:
        return list(self._instructions)

    def can_exit(self) -> bool:
        return self._phase in (
            Phase.INSTRUCTIONS,
            Phase.PRACTICE,
            Phase.PRACTICE_DONE,
            Phase.RESULTS,
        )

    def start(self) -> None:
        self.start_practice()

    def start_practice(self) -> None:
        if self._phase is not Phase.INSTRUCTIONS:
            return
        if int(self._cfg.practice_questions) <= 0:
            self._phase = Phase.PRACTICE_DONE
            self._pending_done_action = "start_scored"
            self._current = None
            self._presented_at_s = None
            return
        self._part_idx = 0
        self._practice_answered = 0
        self._pending_done_action = None
        self._phase = Phase.PRACTICE
        self._deal_new_problem()

    def start_scored(self) -> None:
        if self._phase is Phase.RESULTS:
            return
        if self._phase is Phase.INSTRUCTIONS:
            self._part_idx = 0
            self._phase = Phase.SCORED
            self._pending_done_action = None
            self._scored_started_at_s = self._clock.now()
            self._deal_new_problem()
            return
        if self._phase is not Phase.PRACTICE_DONE:
            return

        if self._pending_done_action == "next_part_practice":
            if self._part_idx + 1 >= len(_PART_ORDER):
                self._phase = Phase.RESULTS
                self._current = None
                self._presented_at_s = None
                return
            self._part_idx += 1
            self._practice_answered = 0
            self._pending_done_action = None
            if int(self._cfg.practice_questions) <= 0:
                self._phase = Phase.SCORED
                self._scored_started_at_s = self._clock.now()
                self._deal_new_problem()
                return
            self._phase = Phase.PRACTICE
            self._deal_new_problem()
            return

        self._phase = Phase.SCORED
        self._pending_done_action = None
        self._scored_started_at_s = self._clock.now()
        self._deal_new_problem()

    def update(self) -> None:
        if self._phase is not Phase.SCORED:
            return
        remaining = self.time_remaining_s()
        if remaining is not None and remaining <= 0.0:
            self._finish_scored_part()

    def time_remaining_s(self) -> float | None:
        if self._phase is not Phase.SCORED or self._scored_started_at_s is None:
            return None
        per_part_duration = self._scored_duration_per_part_s()
        elapsed = self._clock.now() - self._scored_started_at_s
        return max(0.0, per_part_duration - elapsed)

    def current_prompt(self) -> str:
        part_num = self._part_idx + 1
        if self._phase is Phase.INSTRUCTIONS:
            return "Press Enter to begin Part 1 practice."
        if self._phase is Phase.PRACTICE_DONE:
            if self._pending_done_action == "next_part_practice":
                return (
                    f"Part {part_num} complete. Press Enter to continue "
                    f"to Part {part_num + 1} practice."
                )
            return f"Part {part_num} practice complete. Press Enter to start the scored block."
        if self._phase is Phase.RESULTS:
            s = self.scored_summary()
            acc_pct = int(round(s.accuracy * 100))
            rt = "n/a" if s.mean_response_time_s is None else f"{s.mean_response_time_s:.2f}s"
            return (
                f"Results\nAttempted: {s.attempted}\nCorrect: {s.correct}\n"
                f"Accuracy: {acc_pct}%\nMean RT: {rt}\nThroughput: {s.throughput_per_min:.1f}/min"
            )
        if self._current is None:
            return ""
        return self._current.prompt

    def submit_answer(self, raw: object) -> bool:
        raw_in = raw if isinstance(raw, str) else str(raw)
        raw = raw_in.strip()
        if self._handle_control_command(raw):
            return True

        if self._phase not in (Phase.PRACTICE, Phase.SCORED):
            return False
        expired = self._phase is Phase.SCORED and self.time_remaining_s() == 0
        if expired and raw == "":
            self._finish_scored_part()
            return False
        if raw == "":
            return False

        try:
            user_answer = int(raw)
        except ValueError:
            if expired:
                self._finish_scored_part()
            return False

        assert self._current is not None
        assert self._presented_at_s is not None

        answered_at_s = self._clock.now()
        response_time_s = max(0.0, answered_at_s - self._presented_at_s)
        score = float(
            self._scorer.score(
                problem=self._current,
                user_answer=user_answer,
                raw=raw_in,
            )
        )
        if score < 0.0:
            score = 0.0
        elif score > 1.0:
            score = 1.0
        is_full_correct = score >= 1.0 - 1e-9

        event = QuestionEvent(
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
        self._events.append(event)

        if self._phase is Phase.SCORED:
            self._scored_attempted += 1
            self._scored_total_score += score
            self._scored_max_score += 1.0
            if is_full_correct:
                self._scored_correct += 1
        else:
            self._practice_answered += 1

        if expired and self._phase is Phase.SCORED:
            self._finish_scored_part()
            return True

        if (
            self._phase is Phase.PRACTICE
            and self._practice_answered >= int(self._cfg.practice_questions)
        ):
            self._phase = Phase.PRACTICE_DONE
            self._pending_done_action = "start_scored"
            self._current = None
            self._presented_at_s = None
            return True

        self._deal_new_problem()
        return True

    def scored_summary(self) -> AttemptSummary:
        duration_s = float(self._cfg.scored_duration_s)
        attempted = self._scored_attempted
        correct = self._scored_correct
        accuracy = 0.0 if attempted == 0 else correct / attempted
        throughput = (attempted / duration_s) * 60.0 if duration_s > 0 else 0.0
        rts = [e.response_time_s for e in self._events if e.phase is Phase.SCORED]
        mean_rt = None if not rts else sum(rts) / len(rts)
        total_score = float(self._scored_total_score)
        max_score = float(self._scored_max_score)
        score_ratio = 0.0 if max_score == 0.0 else total_score / max_score
        return AttemptSummary(
            attempted=attempted,
            correct=correct,
            accuracy=accuracy,
            duration_s=duration_s,
            throughput_per_min=throughput,
            mean_response_time_s=mean_rt,
            total_score=total_score,
            max_score=max_score,
            score_ratio=score_ratio,
        )

    def snapshot(self) -> TestSnapshot:
        payload = None if self._current is None else self._current.payload
        return TestSnapshot(
            title="Instrument Comprehension",
            phase=self._phase,
            prompt=self.current_prompt(),
            input_hint="Use A/S/D/F/G or 1-5 then Enter",
            time_remaining_s=self.time_remaining_s(),
            attempted_scored=self._scored_attempted,
            correct_scored=self._scored_correct,
            payload=payload,
        )

    def _deal_new_problem(self) -> None:
        kind = _PART_ORDER[self._part_idx]
        self._current = self._generator.next_problem_for_kind(
            kind=kind,
            difficulty=self._difficulty,
        )
        self._presented_at_s = self._clock.now()

    def _finish_scored_part(self) -> None:
        self._current = None
        self._presented_at_s = None
        self._scored_started_at_s = None
        if self._part_idx + 1 < len(_PART_ORDER):
            self._phase = Phase.PRACTICE_DONE
            self._pending_done_action = "next_part_practice"
            return
        self._to_results()

    def _scored_duration_per_part_s(self) -> float:
        return float(self._cfg.scored_duration_s) / float(max(1, len(_PART_ORDER)))

    def _to_results(self) -> None:
        self._phase = Phase.RESULTS
        self._pending_done_action = None
        self._current = None
        self._presented_at_s = None
        self._scored_started_at_s = None

    def _handle_control_command(self, raw: str) -> bool:
        token = str(raw).strip().lower()
        if token in {"__skip_all__", "skip_all"}:
            self._to_results()
            return True

        if token in {"__skip_practice__", "skip_practice"}:
            if self._phase is not Phase.PRACTICE:
                return False
            self._phase = Phase.PRACTICE_DONE
            self._pending_done_action = "start_scored"
            self._current = None
            self._presented_at_s = None
            return True

        if token in {"__skip_section__", "skip_section"}:
            if self._phase is Phase.SCORED:
                self._finish_scored_part()
                return True
            if self._phase is Phase.PRACTICE_DONE:
                if self._part_idx + 1 < len(_PART_ORDER):
                    self._part_idx += 1
                    self._practice_answered = 0
                    self._phase = Phase.PRACTICE
                    self._pending_done_action = None
                    self._deal_new_problem()
                    return True
                self._to_results()
                return True
            return False

        return False


def build_instrument_comprehension_test(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    config: InstrumentComprehensionConfig | None = None,
) -> InstrumentComprehensionEngine:
    return InstrumentComprehensionEngine(
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        config=config,
    )

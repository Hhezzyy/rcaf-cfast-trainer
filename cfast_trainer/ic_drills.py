from __future__ import annotations

from dataclasses import dataclass, field, replace

from .ant_drills import (
    ANT_DRILL_MODE_PROFILES,
    AntAdaptiveDifficultyConfig,
    AntDrillMode,
    TimedCapDrill,
)
from .clock import Clock
from .cognitive_core import Phase, Problem, clamp01, lerp_int
from .instrument_comprehension import (
    InstrumentComprehensionGenerator,
    InstrumentComprehensionScorer,
    InstrumentComprehensionTrialKind,
    InstrumentState,
)


def _norm_heading(deg: int) -> int:
    return int(deg) % 360


def _normalize_mode(mode: AntDrillMode | str) -> AntDrillMode:
    if isinstance(mode, AntDrillMode):
        return mode
    return AntDrillMode(str(mode).strip().lower())


@dataclass(frozen=True, slots=True)
class IcDrillConfig:
    practice_questions: int | None = None
    scored_duration_s: float | None = None
    adaptive: AntAdaptiveDifficultyConfig = field(default_factory=AntAdaptiveDifficultyConfig)


class InstrumentComprehensionTimedDrill(TimedCapDrill):
    def _input_hint(self) -> str:
        if self.phase not in (Phase.PRACTICE, Phase.SCORED):
            return "Press Enter to continue"
        return (
            f"L{self._current_level()} | Cap {self._item_remaining_s():0.1f}s | "
            "A/S/D/F/G or 1-5 then Enter"
        )


class _IcSinglePartGenerator(InstrumentComprehensionGenerator):
    _kind = InstrumentComprehensionTrialKind.INSTRUMENTS_TO_AIRCRAFT

    def next_problem(self, *, difficulty: float) -> Problem:
        return self.next_problem_for_kind(kind=self._kind, difficulty=difficulty)


class IcHeadingAnchorGenerator(_IcSinglePartGenerator):
    _kind = InstrumentComprehensionTrialKind.INSTRUMENTS_TO_AIRCRAFT

    def _sample_state(self, *, difficulty: float) -> InstrumentState:
        d = clamp01(difficulty)
        speed = self._sample_quantized(160, 300, max(10, lerp_int(25, 10, d)))
        altitude = self._sample_quantized(2000, 8000, max(500, lerp_int(1000, 500, d)))
        heading = self._sample_quantized(0, 359, max(30, lerp_int(90, 45, d)))
        bank = int(self._rng.randint(-4, 4))
        pitch = int(self._rng.randint(-2, 2))
        vertical_rate = 0 if pitch == 0 else (100 if pitch > 0 else -100)
        return InstrumentState(
            speed_kts=int(speed),
            altitude_ft=int(altitude),
            vertical_rate_fpm=int(vertical_rate),
            bank_deg=bank,
            pitch_deg=pitch,
            heading_deg=int(heading),
            slip=0,
        )

    def _build_distractors(
        self,
        *,
        state: InstrumentState,
        difficulty: float,
        kind: InstrumentComprehensionTrialKind,
    ) -> tuple[InstrumentState, InstrumentState, InstrumentState, InstrumentState]:
        _ = kind
        d = clamp01(difficulty)
        near = max(20, lerp_int(70, 30, d))
        mid = max(35, lerp_int(110, 55, d))
        far = max(55, lerp_int(150, 85, d))
        return (
            replace(state, heading_deg=_norm_heading(state.heading_deg + near)),
            replace(state, heading_deg=_norm_heading(state.heading_deg - near)),
            replace(state, heading_deg=_norm_heading(state.heading_deg + mid)),
            replace(state, heading_deg=_norm_heading(state.heading_deg + far)),
        )


class IcAttitudeFrameGenerator(_IcSinglePartGenerator):
    _kind = InstrumentComprehensionTrialKind.INSTRUMENTS_TO_AIRCRAFT

    def _sample_state(self, *, difficulty: float) -> InstrumentState:
        d = clamp01(difficulty)
        speed = self._sample_quantized(170, 320, max(10, lerp_int(25, 10, d)))
        altitude = self._sample_quantized(2000, 9000, max(500, lerp_int(1000, 500, d)))
        heading = self._sample_quantized(0, 359, 90 if d < 0.7 else 45)
        bank_mag = self._rng.randint(lerp_int(10, 6, d), lerp_int(28, 18, d))
        bank = int(bank_mag if self._rng.random() < 0.5 else -bank_mag)
        pitch_mag = self._rng.randint(lerp_int(3, 2, d), lerp_int(10, 7, d))
        pitch = int(pitch_mag if self._rng.random() < 0.5 else -pitch_mag)
        vertical_rate = 0 if abs(pitch) <= 1 else (500 if pitch > 0 else -500)
        return InstrumentState(
            speed_kts=int(speed),
            altitude_ft=int(altitude),
            vertical_rate_fpm=int(vertical_rate),
            bank_deg=bank,
            pitch_deg=pitch,
            heading_deg=int(heading),
            slip=0,
        )

    def _build_distractors(
        self,
        *,
        state: InstrumentState,
        difficulty: float,
        kind: InstrumentComprehensionTrialKind,
    ) -> tuple[InstrumentState, InstrumentState, InstrumentState, InstrumentState]:
        _ = kind
        d = clamp01(difficulty)
        pitch_delta = max(2, lerp_int(8, 4, d))
        bank_delta = max(4, lerp_int(14, 7, d))
        coarse_heading = 0 if d < 0.85 else 45
        return (
            replace(state, bank_deg=-state.bank_deg),
            replace(state, pitch_deg=-state.pitch_deg, vertical_rate_fpm=-state.vertical_rate_fpm),
            replace(
                state,
                bank_deg=-state.bank_deg,
                pitch_deg=-state.pitch_deg,
                vertical_rate_fpm=-state.vertical_rate_fpm,
            ),
            replace(
                state,
                heading_deg=_norm_heading(state.heading_deg + coarse_heading),
                bank_deg=state.bank_deg + (bank_delta if state.bank_deg <= 0 else -bank_delta),
                pitch_deg=state.pitch_deg + (pitch_delta if state.pitch_deg <= 0 else -pitch_delta),
            ),
        )


class IcPart1OrientationRunGenerator(_IcSinglePartGenerator):
    _kind = InstrumentComprehensionTrialKind.INSTRUMENTS_TO_AIRCRAFT


class IcReversePanelPrimeGenerator(_IcSinglePartGenerator):
    _kind = InstrumentComprehensionTrialKind.AIRCRAFT_TO_INSTRUMENTS

    def _sample_state(self, *, difficulty: float) -> InstrumentState:
        d = clamp01(difficulty)
        speed = self._sample_quantized(160, 320, max(10, lerp_int(30, 10, d)))
        altitude = self._sample_quantized(2000, 9000, max(500, lerp_int(1000, 500, d)))
        heading = self._sample_quantized(0, 359, 90 if d < 0.7 else 45)
        bank_mag = self._rng.randint(lerp_int(8, 5, d), lerp_int(24, 16, d))
        bank = int(bank_mag if self._rng.random() < 0.5 else -bank_mag)
        pitch_mag = self._rng.randint(lerp_int(2, 1, d), lerp_int(8, 5, d))
        pitch = int(pitch_mag if self._rng.random() < 0.5 else -pitch_mag)
        vertical_rate = 0 if abs(pitch) <= 1 else (700 if pitch > 0 else -700)
        return InstrumentState(
            speed_kts=int(speed),
            altitude_ft=int(altitude),
            vertical_rate_fpm=int(vertical_rate),
            bank_deg=bank,
            pitch_deg=pitch,
            heading_deg=int(heading),
            slip=0,
        )

    def _build_distractors(
        self,
        *,
        state: InstrumentState,
        difficulty: float,
        kind: InstrumentComprehensionTrialKind,
    ) -> tuple[InstrumentState, InstrumentState, InstrumentState, InstrumentState]:
        _ = kind
        d = clamp01(difficulty)
        speed_delta = max(20, lerp_int(50, 20, d))
        altitude_delta = max(500, lerp_int(1600, 600, d))
        heading_delta = max(45, lerp_int(90, 45, d))
        pitch_delta = max(3, lerp_int(7, 4, d))
        return (
            replace(state, speed_kts=max(120, min(360, state.speed_kts + speed_delta))),
            replace(state, altitude_ft=max(1000, min(9500, state.altitude_ft + altitude_delta))),
            replace(
                state,
                heading_deg=_norm_heading(state.heading_deg + heading_delta),
                bank_deg=-state.bank_deg,
            ),
            replace(
                state,
                pitch_deg=-state.pitch_deg if state.pitch_deg != 0 else pitch_delta,
                vertical_rate_fpm=(
                    -state.vertical_rate_fpm
                    if state.vertical_rate_fpm != 0
                    else (700 if pitch_delta > 0 else -700)
                ),
            ),
        )


class IcReversePanelRunGenerator(_IcSinglePartGenerator):
    _kind = InstrumentComprehensionTrialKind.AIRCRAFT_TO_INSTRUMENTS


class IcDescriptionPrimeGenerator(_IcSinglePartGenerator):
    _kind = InstrumentComprehensionTrialKind.INSTRUMENTS_TO_DESCRIPTION

    def _sample_state(self, *, difficulty: float) -> InstrumentState:
        d = clamp01(difficulty)
        speed = self._sample_quantized(140, 320, max(10, lerp_int(30, 10, d)))
        altitude = self._sample_quantized(2000, 9000, max(500, lerp_int(1000, 500, d)))
        heading = self._sample_quantized(0, 359, 90 if d < 0.65 else 45)
        bank_mag = self._rng.randint(lerp_int(8, 5, d), lerp_int(24, 16, d))
        bank = int(bank_mag if self._rng.random() < 0.5 else -bank_mag)
        pitch_mag = self._rng.randint(lerp_int(2, 1, d), lerp_int(8, 5, d))
        pitch = int(pitch_mag if self._rng.random() < 0.5 else -pitch_mag)
        vertical_rate = 0 if abs(pitch) <= 1 else (700 if pitch > 0 else -700)
        return InstrumentState(
            speed_kts=int(speed),
            altitude_ft=int(altitude),
            vertical_rate_fpm=int(vertical_rate),
            bank_deg=bank,
            pitch_deg=pitch,
            heading_deg=int(heading),
            slip=0,
        )

    def _build_distractors(
        self,
        *,
        state: InstrumentState,
        difficulty: float,
        kind: InstrumentComprehensionTrialKind,
    ) -> tuple[InstrumentState, InstrumentState, InstrumentState, InstrumentState]:
        _ = kind
        d = clamp01(difficulty)
        speed_delta = max(20, lerp_int(50, 20, d))
        altitude_delta = max(500, lerp_int(1800, 600, d))
        heading_delta = max(45, lerp_int(90, 45, d))
        pitch_delta = max(3, lerp_int(7, 4, d))
        return (
            replace(state, speed_kts=max(120, min(360, state.speed_kts + speed_delta))),
            replace(state, altitude_ft=max(1000, min(9500, state.altitude_ft + altitude_delta))),
            replace(
                state,
                heading_deg=_norm_heading(state.heading_deg + heading_delta),
                bank_deg=-state.bank_deg,
            ),
            replace(
                state,
                pitch_deg=-state.pitch_deg if state.pitch_deg != 0 else pitch_delta,
                vertical_rate_fpm=(
                    -state.vertical_rate_fpm if state.vertical_rate_fpm != 0 else (700 if pitch_delta > 0 else -700)
                ),
            ),
        )


class IcDescriptionRunGenerator(_IcSinglePartGenerator):
    _kind = InstrumentComprehensionTrialKind.INSTRUMENTS_TO_DESCRIPTION


class _IcSequenceGenerator(InstrumentComprehensionGenerator):
    def __init__(
        self,
        *,
        seed: int,
        kind_sequence: tuple[InstrumentComprehensionTrialKind, ...],
    ) -> None:
        super().__init__(seed=seed)
        self._kind_sequence = kind_sequence
        self._sequence_index = 0

    def next_problem(self, *, difficulty: float) -> Problem:
        kind = self._kind_sequence[self._sequence_index % len(self._kind_sequence)]
        self._sequence_index += 1
        return self.next_problem_for_kind(kind=kind, difficulty=difficulty)


class IcMixedPartRunGenerator(_IcSequenceGenerator):
    def __init__(self, *, seed: int) -> None:
        super().__init__(
            seed=seed,
            kind_sequence=(
                InstrumentComprehensionTrialKind.INSTRUMENTS_TO_AIRCRAFT,
                InstrumentComprehensionTrialKind.INSTRUMENTS_TO_AIRCRAFT,
                InstrumentComprehensionTrialKind.AIRCRAFT_TO_INSTRUMENTS,
                InstrumentComprehensionTrialKind.AIRCRAFT_TO_INSTRUMENTS,
                InstrumentComprehensionTrialKind.INSTRUMENTS_TO_DESCRIPTION,
                InstrumentComprehensionTrialKind.INSTRUMENTS_TO_DESCRIPTION,
            ),
        )


class IcPressureRunGenerator(_IcSequenceGenerator):
    def __init__(self, *, seed: int) -> None:
        super().__init__(
            seed=seed,
            kind_sequence=(
                InstrumentComprehensionTrialKind.INSTRUMENTS_TO_AIRCRAFT,
                InstrumentComprehensionTrialKind.AIRCRAFT_TO_INSTRUMENTS,
                InstrumentComprehensionTrialKind.INSTRUMENTS_TO_DESCRIPTION,
            ),
        )


def _build_ic_drill(
    *,
    title_base: str,
    instructions: tuple[str, ...],
    generator: InstrumentComprehensionGenerator,
    clock: Clock,
    seed: int,
    difficulty: float,
    mode: AntDrillMode | str,
    config: IcDrillConfig,
    base_caps_by_level: tuple[float, ...],
) -> InstrumentComprehensionTimedDrill:
    normalized_mode = _normalize_mode(mode)
    profile = ANT_DRILL_MODE_PROFILES[normalized_mode]
    practice_questions = (
        profile.practice_questions if config.practice_questions is None else int(config.practice_questions)
    )
    scored_duration_s = (
        profile.scored_duration_s if config.scored_duration_s is None else float(config.scored_duration_s)
    )
    return InstrumentComprehensionTimedDrill(
        title=f"{title_base} ({profile.label})",
        instructions=instructions,
        generator=generator,
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        practice_questions=practice_questions,
        scored_duration_s=scored_duration_s,
        mode=normalized_mode,
        base_caps_by_level=base_caps_by_level,
        adaptive_config=config.adaptive,
        scorer=InstrumentComprehensionScorer(),
        immediate_feedback_override=True,
    )


def build_ic_heading_anchor_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: IcDrillConfig | None = None,
) -> InstrumentComprehensionTimedDrill:
    cfg = config or IcDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    return _build_ic_drill(
        title_base="Instrument Comprehension: Heading Anchor",
        instructions=(
            "Instrument Comprehension: Heading Anchor",
            f"Mode: {profile.label}",
            "Part 1 only. Keep bank and pitch nearly level so the heading picture becomes automatic first.",
            "Use the real aircraft-image answer layout and commit fast when the compass rose changes.",
            "Press Enter to begin practice.",
        ),
        generator=IcHeadingAnchorGenerator(seed=seed),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
        base_caps_by_level=(15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0),
    )


def build_ic_attitude_frame_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: IcDrillConfig | None = None,
) -> InstrumentComprehensionTimedDrill:
    cfg = config or IcDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    return _build_ic_drill(
        title_base="Instrument Comprehension: Attitude Frame",
        instructions=(
            "Instrument Comprehension: Attitude Frame",
            f"Mode: {profile.label}",
            "Part 1 only. Hold heading coarse and clean up bank and pitch image discrimination first.",
            "Focus on the aircraft attitude frame before you worry about finer heading offsets.",
            "Press Enter to begin practice.",
        ),
        generator=IcAttitudeFrameGenerator(seed=seed),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
        base_caps_by_level=(15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0),
    )


def build_ic_part1_orientation_run_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.TEMPO,
    config: IcDrillConfig | None = None,
) -> InstrumentComprehensionTimedDrill:
    cfg = config or IcDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    return _build_ic_drill(
        title_base="Instrument Comprehension: Part 1 Orientation Run",
        instructions=(
            "Instrument Comprehension: Part 1 Orientation Run",
            f"Mode: {profile.label}",
            "Part 1 only. Use the real aircraft-card answer set with heading, bank, and pitch changing together.",
            "Train quick orientation matching without leaving the A/S/D/F/G answer flow.",
            "Press Enter to begin practice.",
        ),
        generator=IcPart1OrientationRunGenerator(seed=seed),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
        base_caps_by_level=(13.0, 12.0, 11.0, 10.0, 9.0, 8.5, 8.0, 7.0, 6.5, 6.0),
    )


def build_ic_reverse_panel_prime_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: IcDrillConfig | None = None,
) -> InstrumentComprehensionTimedDrill:
    cfg = config or IcDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    return _build_ic_drill(
        title_base="Instrument Comprehension: Reverse Panel Prime",
        instructions=(
            "Instrument Comprehension: Reverse Panel Prime",
            f"Mode: {profile.label}",
            "Part 2 only. Read one aircraft image, then choose the matching full instrument panel.",
            "Start with easier one-dimension distractors before the full reverse-match run.",
            "Press Enter to begin practice.",
        ),
        generator=IcReversePanelPrimeGenerator(seed=seed),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
        base_caps_by_level=(16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0),
    )


def build_ic_reverse_panel_run_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.TEMPO,
    config: IcDrillConfig | None = None,
) -> InstrumentComprehensionTimedDrill:
    cfg = config or IcDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    return _build_ic_drill(
        title_base="Instrument Comprehension: Reverse Panel Run",
        instructions=(
            "Instrument Comprehension: Reverse Panel Run",
            f"Mode: {profile.label}",
            "Part 2 only. Use full aircraft-image-to-instrument-panel matching under tempo pressure.",
            "Keep the same A/S/D/F/G flow and reset immediately after misses.",
            "Press Enter to begin practice.",
        ),
        generator=IcReversePanelRunGenerator(seed=seed),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
        base_caps_by_level=(14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.5, 7.5, 7.0, 6.5),
    )


def build_ic_description_prime_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: IcDrillConfig | None = None,
) -> InstrumentComprehensionTimedDrill:
    cfg = config or IcDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    return _build_ic_drill(
        title_base="Instrument Comprehension: Description Prime",
        instructions=(
            "Instrument Comprehension: Description Prime",
            f"Mode: {profile.label}",
            "Part 3 only. Start with cleaner one-dimension distractors before the full interpretation run.",
            "Read the full instrument panel, then match the best description without overthinking the wording.",
            "Press Enter to begin practice.",
        ),
        generator=IcDescriptionPrimeGenerator(seed=seed),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
        base_caps_by_level=(16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0),
    )


def build_ic_description_run_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.TEMPO,
    config: IcDrillConfig | None = None,
) -> InstrumentComprehensionTimedDrill:
    cfg = config or IcDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    return _build_ic_drill(
        title_base="Instrument Comprehension: Description Run",
        instructions=(
            "Instrument Comprehension: Description Run",
            f"Mode: {profile.label}",
            "Part 3 only. Use the real full-panel-to-description interpretation flow.",
            "Commit to the best description quickly and use the next item to reset after misses.",
            "Press Enter to begin practice.",
        ),
        generator=IcDescriptionRunGenerator(seed=seed),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
        base_caps_by_level=(14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.5, 7.5, 7.0, 6.5),
    )


def build_ic_mixed_part_run_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.TEMPO,
    config: IcDrillConfig | None = None,
) -> InstrumentComprehensionTimedDrill:
    cfg = config or IcDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    return _build_ic_drill(
        title_base="Instrument Comprehension: Mixed Part Run",
        instructions=(
            "Instrument Comprehension: Mixed Part Run",
            f"Mode: {profile.label}",
            "Balanced rhythm: two Part 1 items, two Part 2 items, then two Part 3 items, repeating.",
            "Use the part changes to reset cleanly without changing the live instrument UI.",
            "Press Enter to begin practice.",
        ),
        generator=IcMixedPartRunGenerator(seed=seed),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
        base_caps_by_level=(12.0, 11.0, 10.0, 9.5, 9.0, 8.0, 7.5, 7.0, 6.5, 6.0),
    )


def build_ic_pressure_run_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.STRESS,
    config: IcDrillConfig | None = None,
) -> InstrumentComprehensionTimedDrill:
    cfg = config or IcDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    return _build_ic_drill(
        title_base="Instrument Comprehension: Pressure Run",
        instructions=(
            "Instrument Comprehension: Pressure Run",
            f"Mode: {profile.label}",
            "Balanced mixed pressure: Part 1, Part 2, and Part 3 alternate item-by-item under the hardest cap profile.",
            "Recover immediately after misses and keep the same real IC answer flow throughout.",
            "Press Enter to begin practice.",
        ),
        generator=IcPressureRunGenerator(seed=seed),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
        base_caps_by_level=(10.0, 9.5, 9.0, 8.5, 8.0, 7.0, 6.5, 6.0, 5.5, 5.0),
    )

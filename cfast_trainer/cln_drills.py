from __future__ import annotations

from dataclasses import dataclass, field

from .ant_drills import ANT_DRILL_MODE_PROFILES, AntDrillAttemptSummary, AntDrillMode
from .clock import Clock
from .cognitive_core import Phase, Problem, QuestionEvent, SeededRng, TestSnapshot, clamp01
from .colours_letters_numbers import (
    ColoursLettersNumbersDiamond,
    ColoursLettersNumbersGenerationProfile,
    ColoursLettersNumbersGenerator,
    ColoursLettersNumbersMemoryChallenge,
    ColoursLettersNumbersMemoryMode,
    ColoursLettersNumbersOption,
    ColoursLettersNumbersRuntimePayload,
    ColoursLettersNumbersTrainingPayload,
)
from .numerical_operations import NumericalOperationsProblemProfile


def _normalize_mode(mode: AntDrillMode | str) -> AntDrillMode:
    return mode if isinstance(mode, AntDrillMode) else AntDrillMode(str(mode).strip().lower())


def _level_from_difficulty(difficulty: float) -> int:
    return max(1, min(10, int(round(clamp01(float(difficulty)) * 9.0)) + 1))


def _lerp_float(lo: float, hi: float, difficulty: float) -> float:
    d = clamp01(float(difficulty))
    return float(lo) + ((float(hi) - float(lo)) * d)


def _lerp_int(lo: int, hi: int, difficulty: float) -> int:
    return int(round(_lerp_float(float(lo), float(hi), difficulty)))


@dataclass(frozen=True, slots=True)
class ClnDrillConfig:
    practice_rounds: int | None = None
    scored_duration_s: float | None = None


@dataclass(frozen=True, slots=True)
class ClnDrillProfile:
    active_channels: tuple[str, ...]
    memory_mode: ColoursLettersNumbersMemoryMode = ColoursLettersNumbersMemoryMode.DELAYED_CHOICE
    generation_profile: ColoursLettersNumbersGenerationProfile = field(
        default_factory=ColoursLettersNumbersGenerationProfile
    )
    sequence_show_easy: float = 2.0
    sequence_show_hard: float = 1.0
    recall_delay_easy: float = 1.0
    recall_delay_hard: float = 3.0
    recall_delay_max_easy: float = 1.5
    recall_delay_max_hard: float = 4.0
    recall_window_easy: float = 8.0
    recall_window_hard: float = 5.0
    diamond_spawn_easy: float = 1.8
    diamond_spawn_hard: float = 0.8
    diamond_spawn_max_easy: float = 2.2
    diamond_spawn_max_hard: float = 1.0
    diamond_speed_easy: float = 0.12
    diamond_speed_hard: float = 0.34
    diamond_speed_max_easy: float = 0.18
    diamond_speed_max_hard: float = 0.44
    max_live_diamonds_easy: int = 1
    max_live_diamonds_hard: int = 4
    typed_memory_max_length: int = 8
    panel_prompt: str = ""
    control_hint: str = ""
    input_label: str = "Math Answer"
    show_text_entry: bool = True
    static_text: str = "--"
    top_hint_override: str | None = None
    required_math_answers_before_memory: int = 0

    def has_memory(self) -> bool:
        return "memory" in self.active_channels

    def has_math(self) -> bool:
        return "math" in self.active_channels

    def has_colour(self) -> bool:
        return "colour" in self.active_channels


@dataclass(slots=True)
class _LiveDiamond:
    id: int
    color: str
    row: int
    x_norm: float
    speed_norm_per_s: float


class ClnDrillEngine:
    _LANE_COLORS = ("RED", "YELLOW", "GREEN", "BLUE")
    _HIT_ZONE_START = 0.54
    _HIT_ZONE_END = 0.98
    _MAX_UPDATE_DT_S = 0.25

    def __init__(
        self,
        *,
        title: str,
        instructions: tuple[str, ...],
        profile: ClnDrillProfile,
        clock: Clock,
        seed: int,
        difficulty: float,
        practice_rounds: int,
        scored_duration_s: float,
        mode: AntDrillMode,
    ) -> None:
        self._title = str(title)
        self._instructions = tuple(instructions)
        self._profile = profile
        self._clock = clock
        self._seed = int(seed)
        self._difficulty = clamp01(float(difficulty))
        self._practice_rounds = max(0, int(practice_rounds))
        self._scored_duration_s = max(1.0, float(scored_duration_s))
        self._mode = mode

        self._challenge_generator = ColoursLettersNumbersGenerator(
            SeededRng(self._seed),
            profile=self._profile.generation_profile,
        )
        self._rng = SeededRng(self._seed ^ 0x51D0_6F19)

        self._phase = Phase.INSTRUCTIONS
        self._memory_current: ColoursLettersNumbersMemoryChallenge | None = None
        self._math_current: Problem | None = None
        self._memory_cycle_started_at_s: float | None = None
        self._memory_prompted_at_s: float | None = None
        self._memory_recall_delay_s_current = 0.0
        self._memory_answered = False
        self._last_update_s: float | None = None
        self._spawn_cooldown_s = 0.0
        self._diamonds: list[_LiveDiamond] = []
        self._next_diamond_id = 1
        self._practice_progress = 0
        self._scored_started_at_s: float | None = None
        self._events: list[QuestionEvent] = []
        self._last_feedback = ""
        self._display_points = 0.0
        self._display_cleared = 0
        self._display_missed = 0
        self._display_attempted = 0
        self._display_correct = 0
        self._scored_attempted = 0
        self._scored_correct = 0
        self._scored_points = 0.0
        self._scored_cleared = 0
        self._scored_missed = 0
        self._memory_timeouts = 0
        self._timeout_streak = 0
        self._max_timeout_streak = 0
        self._math_answers_since_memory_cycle = 0

    @property
    def phase(self) -> Phase:
        return self._phase

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def difficulty(self) -> float:
        return float(self._difficulty)

    @property
    def practice_questions(self) -> int:
        return self._practice_rounds

    @property
    def scored_duration_s(self) -> float:
        return float(self._scored_duration_s)

    def can_exit(self) -> bool:
        return self._phase is not Phase.SCORED

    def events(self) -> list[QuestionEvent]:
        return list(self._events)

    def start_practice(self) -> None:
        if self._phase is not Phase.INSTRUCTIONS:
            return
        if self._practice_rounds <= 0:
            self._phase = Phase.PRACTICE_DONE
            return
        self._phase = Phase.PRACTICE
        self._reset_visible_counters()
        self._start_channels(self._clock.now())

    def start_scored(self) -> None:
        if self._phase is Phase.RESULTS:
            return
        self._phase = Phase.SCORED
        self._scored_started_at_s = self._clock.now()
        self._reset_visible_counters()
        self._last_feedback = ""
        self._start_channels(self._clock.now())

    def update(self) -> None:
        if self._phase not in (Phase.PRACTICE, Phase.SCORED):
            return
        if self._phase is Phase.SCORED and self.time_remaining_s() == 0.0:
            self._finish_results()
            return

        now = self._clock.now()
        if self._last_update_s is None:
            self._last_update_s = now
        dt = max(0.0, now - self._last_update_s)
        self._last_update_s = now
        dt = min(dt, self._MAX_UPDATE_DT_S)

        if self._profile.has_colour():
            self._update_diamonds(dt)
        if self._profile.has_memory():
            self._update_memory_cycle(now)

    def submit_answer(self, raw: str) -> bool:
        if self._phase not in (Phase.PRACTICE, Phase.SCORED):
            return False

        token = str(raw).strip()
        lowered = token.lower()
        if lowered in {"__skip_practice__", "skip_practice"} and self._phase is Phase.PRACTICE:
            self._phase = Phase.PRACTICE_DONE
            self._clear_channel_state()
            return True
        if lowered in {"__skip_section__", "skip_section", "__skip_all__", "skip_all"} and (
            self._phase is Phase.SCORED
        ):
            self._finish_results()
            return True
        if self._phase is Phase.SCORED and self.time_remaining_s() == 0.0:
            self._finish_results()
            return False

        upper = token.upper()
        if upper.startswith("MEMSEQ:"):
            return self._submit_memory_sequence(upper.split(":", 1)[1])
        if upper.startswith("MEM:"):
            return self._submit_memory_choice(upper.split(":", 1)[1])
        if upper.startswith("CLR:"):
            return self._submit_color(upper.split(":", 1)[1])
        return self._submit_math(token)

    def time_remaining_s(self) -> float | None:
        if self._phase is not Phase.SCORED:
            return None
        assert self._scored_started_at_s is not None
        elapsed = self._clock.now() - self._scored_started_at_s
        return max(0.0, self._scored_duration_s - elapsed)

    def scored_summary(self) -> AntDrillAttemptSummary:
        duration_s = float(self._scored_duration_s)
        attempted = int(self._scored_attempted)
        correct = int(self._scored_correct)
        accuracy = 0.0 if attempted == 0 else correct / attempted
        throughput = (attempted / duration_s) * 60.0 if duration_s > 0.0 else 0.0
        correct_per_min = (correct / duration_s) * 60.0 if duration_s > 0.0 else 0.0
        rts = [event.response_time_s for event in self._events if event.phase is Phase.SCORED]
        mean_rt = None if not rts else (sum(rts) / len(rts))
        max_score = (attempted * 2.0) + (self._scored_cleared * 0.5)
        score_ratio = 0.0 if max_score <= 0.0 else clamp01(max(0.0, self._scored_points) / max_score)
        fixation_rate = 0.0 if attempted == 0 else self._memory_timeouts / attempted
        return AntDrillAttemptSummary(
            attempted=attempted,
            correct=correct,
            accuracy=float(accuracy),
            duration_s=duration_s,
            throughput_per_min=float(throughput),
            mean_response_time_s=mean_rt,
            total_score=float(self._scored_points),
            max_score=float(max_score),
            score_ratio=float(score_ratio),
            correct_per_min=float(correct_per_min),
            timeouts=int(self._memory_timeouts),
            fixation_rate=float(fixation_rate),
            max_timeout_streak=int(self._max_timeout_streak),
            mode=self._mode.value,
            difficulty_level=_level_from_difficulty(self._difficulty),
            difficulty_level_start=_level_from_difficulty(self._difficulty),
            difficulty_level_end=_level_from_difficulty(self._difficulty),
            difficulty_change_count=0,
            adaptive_enabled=False,
            adaptive_window_size=0,
        )

    def snapshot(self) -> TestSnapshot:
        return TestSnapshot(
            title=self._title,
            phase=self._phase,
            prompt=self._prompt_text(),
            input_hint=self._input_hint(),
            time_remaining_s=self.time_remaining_s(),
            attempted_scored=self._display_attempted if self._phase is Phase.PRACTICE else self._scored_attempted,
            correct_scored=self._display_correct if self._phase is Phase.PRACTICE else self._scored_correct,
            payload=self._payload(),
            practice_feedback=self._last_feedback if self._last_feedback else None,
        )

    def _prompt_text(self) -> str:
        if self._phase is Phase.INSTRUCTIONS:
            return "\n".join(self._instructions)
        if self._phase is Phase.PRACTICE_DONE:
            return "Practice complete. Press Enter to begin the timed block."
        if self._phase is Phase.RESULTS:
            summary = self.scored_summary()
            return (
                "Results\n"
                f"Attempted: {summary.attempted}\n"
                f"Correct: {summary.correct}\n"
                f"Accuracy: {summary.accuracy * 100.0:.1f}%\n"
                f"Correct/min: {summary.correct_per_min:.1f}\n"
                f"Colour clear: {self._scored_cleared}\n"
                f"Colour miss: {self._scored_missed}\n"
                f"Score: {summary.total_score:.1f}\n"
                f"Timeouts: {summary.timeouts}"
            )
        if self._profile.has_math() and self._math_current is not None:
            return str(self._math_current.prompt)
        return self._profile.panel_prompt

    def _input_hint(self) -> str:
        if self._phase not in (Phase.PRACTICE, Phase.SCORED):
            return "Press Enter to continue"
        return self._profile.control_hint

    def _payload(self) -> ColoursLettersNumbersRuntimePayload | None:
        if self._phase not in (Phase.PRACTICE, Phase.SCORED):
            return None
        if not (self._profile.has_memory() or self._profile.has_math() or self._profile.has_colour()):
            return None

        target_sequence: str | None = None
        options: tuple[ColoursLettersNumbersOption, ...] = ()
        options_active = False
        memory_input_active = False
        top_hint_override = self._profile.top_hint_override
        memory_gate_open = self._memory_gate_open()
        if self._profile.has_memory() and self._memory_current is not None:
            if self._profile.memory_mode is ColoursLettersNumbersMemoryMode.VISIBLE_COPY:
                if not self._memory_answered:
                    target_sequence = self._memory_current.target_sequence
                    memory_input_active = True
                    top_hint_override = "Type the full sequence while it remains visible."
            else:
                elapsed = self._memory_elapsed()
                if elapsed < self._sequence_show_s():
                    target_sequence = self._memory_current.target_sequence
                elif (
                    not self._memory_answered
                    and elapsed >= (self._sequence_show_s() + self._memory_recall_delay_s_current)
                    and memory_gate_open
                ):
                    if self._profile.memory_mode is ColoursLettersNumbersMemoryMode.DELAYED_CHOICE:
                        options = self._memory_current.options
                        options_active = True
                    else:
                        memory_input_active = True
                elif not self._memory_answered and not memory_gate_open:
                    top_hint_override = "Solve one math question before the memory answer opens."
        if not self._profile.has_memory():
            options = ()
            options_active = False

        diamonds = tuple(
            ColoursLettersNumbersDiamond(
                id=diamond.id,
                color=diamond.color,
                row=diamond.row,
                x_norm=float(diamond.x_norm),
            )
            for diamond in self._diamonds
        )

        prompt_text = self._profile.panel_prompt
        if memory_input_active:
            prompt_text = "Type the stored sequence and press Enter."
        elif self._profile.has_memory() and options_active:
            prompt_text = "Pick the matching corner using A/S/D/F/G or the mouse."
        elif (
            self._profile.has_math()
            and self._math_current is not None
            and (
                not self._profile.has_memory()
                or not self._memory_answered
                or not memory_gate_open
            )
        ):
            prompt_text = str(self._math_current.prompt)
        elif self._profile.has_memory() and not options_active:
            prompt_text = "Hold the sequence in memory until the answer grid opens."

        show_text_entry = self._profile.show_text_entry or memory_input_active or self._profile.has_math()
        input_label = "Sequence Entry" if memory_input_active else self._profile.input_label
        static_text = self._profile.static_text
        if options_active and not show_text_entry:
            static_text = "A/S/D/F/G"

        return ColoursLettersNumbersTrainingPayload(
            target_sequence=target_sequence,
            options=options,
            options_active=options_active,
            memory_answered=(not self._profile.has_memory()) or self._memory_answered,
            math_answered=False,
            math_prompt=prompt_text,
            lane_colors=self._LANE_COLORS,
            lane_start_norm=self._HIT_ZONE_START,
            lane_end_norm=self._HIT_ZONE_END,
            diamonds=diamonds,
            missed_diamonds=self._display_missed,
            cleared_diamonds=self._display_cleared,
            points=float(self._display_points),
            memory_input_active=memory_input_active,
            memory_input_max_length=(
                self._profile.typed_memory_max_length if memory_input_active else 0
            ),
            input_label=input_label,
            show_text_entry=show_text_entry,
            static_text=static_text,
            control_hint=self._profile.control_hint,
            top_hint_override=top_hint_override,
            colour_active=self._profile.has_colour(),
            math_active=self._profile.has_math(),
            memory_active=self._profile.has_memory(),
        )

    def _start_channels(self, now: float) -> None:
        self._last_update_s = now
        self._diamonds.clear()
        self._spawn_cooldown_s = self._next_spawn_interval_s()
        if self._profile.has_memory():
            self._start_memory_cycle(now)
        else:
            self._memory_current = None
            self._memory_cycle_started_at_s = None
            self._memory_prompted_at_s = None
            self._memory_answered = True
        if self._profile.has_math():
            self._start_math_cycle(now)
        else:
            self._math_current = None

    def _start_memory_cycle(self, now: float) -> None:
        self._memory_current = self._challenge_generator.next_memory_challenge(difficulty=self._difficulty)
        self._memory_cycle_started_at_s = now
        self._math_answers_since_memory_cycle = 0
        self._memory_recall_delay_s_current = self._rng.uniform(
            self._recall_delay(),
            self._recall_delay_max(),
        )
        self._memory_prompted_at_s = (
            now
            if self._profile.memory_mode is ColoursLettersNumbersMemoryMode.VISIBLE_COPY
            else (now + self._sequence_show_s() + self._memory_recall_delay_s_current)
        )
        self._memory_answered = False

    def _start_math_cycle(self, now: float) -> None:
        prompt, answer = self._challenge_generator.next_math_challenge(difficulty=self._difficulty)
        self._math_current = Problem(prompt=prompt, answer=int(answer))
        self._presented_at_s = now

    def _memory_elapsed(self) -> float:
        if self._memory_cycle_started_at_s is None:
            return 0.0
        return max(0.0, self._clock.now() - self._memory_cycle_started_at_s)

    def _sequence_show_s(self) -> float:
        return _lerp_float(
            self._profile.sequence_show_easy,
            self._profile.sequence_show_hard,
            self._difficulty,
        )

    def _recall_delay(self) -> float:
        return _lerp_float(
            self._profile.recall_delay_easy,
            self._profile.recall_delay_hard,
            self._difficulty,
        )

    def _recall_delay_max(self) -> float:
        return max(
            self._recall_delay(),
            _lerp_float(
                self._profile.recall_delay_max_easy,
                self._profile.recall_delay_max_hard,
                self._difficulty,
            ),
        )

    def _recall_window(self) -> float:
        return _lerp_float(
            self._profile.recall_window_easy,
            self._profile.recall_window_hard,
            self._difficulty,
        )

    def _memory_gate_open(self) -> bool:
        required = max(0, int(self._profile.required_math_answers_before_memory))
        if required <= 0:
            return True
        return self._math_answers_since_memory_cycle >= required

    def _next_spawn_interval_s(self) -> float:
        return self._rng.uniform(
            _lerp_float(self._profile.diamond_spawn_easy, self._profile.diamond_spawn_hard, self._difficulty),
            _lerp_float(
                self._profile.diamond_spawn_max_easy,
                self._profile.diamond_spawn_max_hard,
                self._difficulty,
            ),
        )

    def _spawn_speed_norm_per_s(self) -> float:
        return self._rng.uniform(
            _lerp_float(self._profile.diamond_speed_easy, self._profile.diamond_speed_hard, self._difficulty),
            _lerp_float(
                self._profile.diamond_speed_max_easy,
                self._profile.diamond_speed_max_hard,
                self._difficulty,
            ),
        )

    def _max_live_diamonds(self) -> int:
        return max(
            1,
            _lerp_int(
                self._profile.max_live_diamonds_easy,
                self._profile.max_live_diamonds_hard,
                self._difficulty,
            ),
        )

    def _update_memory_cycle(self, now: float) -> None:
        if self._memory_current is None or self._memory_cycle_started_at_s is None:
            return

        elapsed = now - self._memory_cycle_started_at_s
        if self._profile.memory_mode is ColoursLettersNumbersMemoryMode.VISIBLE_COPY:
            if elapsed < self._recall_window():
                return
        else:
            total_cycle_s = self._sequence_show_s() + self._memory_recall_delay_s_current + self._recall_window()
            if elapsed < total_cycle_s:
                return

        if not self._memory_answered:
            self._record_memory_result(
                now=now,
                response="TIMEOUT",
                is_correct=False,
                is_timeout=True,
            )
        else:
            self._advance_practice_progress(kind="memory")

        if self._phase not in (Phase.PRACTICE, Phase.SCORED):
            return
        if self._profile.has_memory() and self._profile.has_math() is False and self._profile.has_colour() is False:
            self._start_memory_cycle(now)
            return
        self._start_memory_cycle(now)

    def _update_diamonds(self, dt: float) -> None:
        if dt <= 0.0:
            return

        self._spawn_cooldown_s -= dt
        while self._spawn_cooldown_s <= 0.0:
            overdue = -self._spawn_cooldown_s
            self._spawn_cooldown_s = self._next_spawn_interval_s() - overdue
            if len(self._diamonds) < self._max_live_diamonds():
                self._spawn_diamond()

        survivors: list[_LiveDiamond] = []
        for diamond in self._diamonds:
            diamond.x_norm += diamond.speed_norm_per_s * dt
            if diamond.x_norm >= 1.02:
                self._display_missed += 1
                self._display_points -= 1.0
                self._last_feedback = f"Missed {diamond.color} lane."
                if self._phase is Phase.SCORED:
                    self._scored_missed += 1
                    self._scored_points -= 1.0
                elif not self._profile.has_memory() and not self._profile.has_math():
                    self._advance_practice_progress(kind="colour")
                continue
            survivors.append(diamond)
        self._diamonds = survivors

    def _spawn_diamond(self) -> None:
        color = str(self._rng.choice(self._LANE_COLORS))
        row = self._sample_diamond_row()
        diamond = _LiveDiamond(
            id=self._next_diamond_id,
            color=color,
            row=row,
            x_norm=0.02,
            speed_norm_per_s=self._spawn_speed_norm_per_s(),
        )
        self._next_diamond_id += 1
        self._diamonds.append(diamond)

    def _sample_diamond_row(self) -> int:
        pick = self._rng.random()
        if pick < 0.20:
            return 0
        if pick < 0.55:
            return 1
        return 2

    def _submit_memory_sequence(self, raw_sequence: str) -> bool:
        if (
            not self._profile.has_memory()
            or self._profile.memory_mode is ColoursLettersNumbersMemoryMode.DELAYED_CHOICE
            or self._memory_current is None
            or self._memory_answered
        ):
            return False
        if self._profile.memory_mode is not ColoursLettersNumbersMemoryMode.VISIBLE_COPY:
            if self._memory_elapsed() < (self._sequence_show_s() + self._memory_recall_delay_s_current):
                return False
            if not self._memory_gate_open():
                return False
        cleaned = "".join(ch for ch in str(raw_sequence).upper() if "A" <= ch <= "Z")
        if cleaned == "":
            return False
        is_correct = cleaned == self._memory_current.target_sequence
        now = self._clock.now()
        self._record_memory_result(now=now, response=cleaned, is_correct=is_correct, is_timeout=False)
        if self._profile.has_math() and self._profile.required_math_answers_before_memory > 0:
            self._start_memory_cycle(now)
            self._start_math_cycle(now)
            return True
        self._start_memory_cycle(now)
        return True

    def _submit_memory_choice(self, raw_code: str) -> bool:
        if (
            not self._profile.has_memory()
            or self._profile.memory_mode is not ColoursLettersNumbersMemoryMode.DELAYED_CHOICE
            or self._memory_current is None
            or self._memory_answered
        ):
            return False
        if self._memory_elapsed() < (self._sequence_show_s() + self._memory_recall_delay_s_current):
            return False
        if not self._memory_gate_open():
            return False
        digits = "".join(ch for ch in str(raw_code) if ch.isdigit())
        if digits == "":
            return False
        code = int(digits)
        is_correct = code == int(self._memory_current.expected_option_code)
        now = self._clock.now()
        self._record_memory_result(now=now, response=str(code), is_correct=is_correct, is_timeout=False)
        if self._profile.has_math() and self._profile.required_math_answers_before_memory > 0:
            self._start_memory_cycle(now)
            self._start_math_cycle(now)
            return True
        if not self._profile.has_math() and not self._profile.has_colour():
            self._start_memory_cycle(now)
        return True

    def _record_memory_result(
        self,
        *,
        now: float,
        response: str,
        is_correct: bool,
        is_timeout: bool,
    ) -> None:
        if self._memory_current is None:
            return
        prompted_at = self._memory_prompted_at_s if self._memory_prompted_at_s is not None else now
        rt = max(0.0, now - prompted_at)
        self._events.append(
            QuestionEvent(
                index=len(self._events),
                phase=self._phase,
                prompt=self._memory_prompt_text(),
                correct_answer=int(self._memory_current.expected_option_code),
                user_answer=(0 if response == "TIMEOUT" else int(response) if response.isdigit() else 0),
                is_correct=is_correct,
                presented_at_s=prompted_at,
                answered_at_s=now,
                response_time_s=rt,
                raw="__timeout__" if is_timeout else f"MEM:{response}",
                score=(2.0 if is_correct else 0.0),
                max_score=2.0,
            )
        )
        self._memory_answered = True
        self._display_attempted += 1
        if is_correct:
            self._display_correct += 1
            self._display_points += 2.0
            self._last_feedback = "Memory correct. Commit and move."
            self._timeout_streak = 0
        else:
            if is_timeout:
                self._last_feedback = f"Memory timeout. Sequence was {self._memory_current.target_sequence}."
                self._memory_timeouts += 1
                self._timeout_streak += 1
                self._max_timeout_streak = max(self._max_timeout_streak, self._timeout_streak)
            else:
                self._last_feedback = f"Memory miss. Sequence was {self._memory_current.target_sequence}."
                self._timeout_streak = 0
        if self._phase is Phase.SCORED:
            self._scored_attempted += 1
            self._scored_points += 2.0 if is_correct else 0.0
            if is_correct:
                self._scored_correct += 1
        else:
            self._advance_practice_progress(kind="memory")

    def _memory_prompt_text(self) -> str:
        if self._profile.memory_mode is ColoursLettersNumbersMemoryMode.VISIBLE_COPY:
            return "Type the full sequence while it remains visible."
        if self._profile.memory_mode is ColoursLettersNumbersMemoryMode.DELAYED_EXACT:
            return "Type the stored sequence from memory."
        return "Pick the matching sequence using A/S/D/F/G or mouse."

    def _submit_math(self, raw_text: str) -> bool:
        if not self._profile.has_math() or self._math_current is None:
            return False
        text = str(raw_text).strip()
        if text == "":
            return False
        if text.startswith("-"):
            digits = "".join(ch for ch in text[1:] if ch.isdigit())
            if digits == "":
                return False
            value = -int(digits)
        else:
            digits = "".join(ch for ch in text if ch.isdigit())
            if digits == "":
                return False
            value = int(digits)
        is_correct = int(value) == int(self._math_current.answer)
        now = self._clock.now()
        base = self._presented_at_s if hasattr(self, "_presented_at_s") and self._presented_at_s is not None else now
        rt = max(0.0, now - base)
        self._events.append(
            QuestionEvent(
                index=len(self._events),
                phase=self._phase,
                prompt=str(self._math_current.prompt),
                correct_answer=int(self._math_current.answer),
                user_answer=int(value),
                is_correct=is_correct,
                presented_at_s=base,
                answered_at_s=now,
                response_time_s=rt,
                raw=str(value),
                score=(2.0 if is_correct else 0.0),
                max_score=2.0,
            )
        )
        self._math_answers_since_memory_cycle += 1
        self._display_attempted += 1
        if is_correct:
            self._display_correct += 1
            self._display_points += 2.0
            self._last_feedback = "Math correct. Commit and move."
        else:
            self._last_feedback = f"Math miss. Correct answer: {self._math_current.answer}"
        if self._phase is Phase.SCORED:
            self._scored_attempted += 1
            self._scored_points += 2.0 if is_correct else 0.0
            if is_correct:
                self._scored_correct += 1
        else:
            self._advance_practice_progress(kind="math")
        if not (
            self._profile.has_memory()
            and not self._memory_answered
            and not self._memory_gate_open()
        ):
            self._start_math_cycle(now)
        return True

    def _submit_color(self, raw_key: str) -> bool:
        if not self._profile.has_colour():
            return False
        key = str(raw_key).strip().upper()
        key_map = {
            "Q": "RED",
            "W": "YELLOW",
            "E": "GREEN",
            "R": "BLUE",
            "RED": "RED",
            "YELLOW": "YELLOW",
            "GREEN": "GREEN",
            "BLUE": "BLUE",
        }
        color = key_map.get(key)
        if color is None:
            return False

        lane_bounds = self._color_lane_bounds(color)
        if lane_bounds is None:
            return False
        lane_start, lane_end = lane_bounds
        lane_mid = (lane_start + lane_end) * 0.5
        candidates = [
            diamond
            for diamond in self._diamonds
            if diamond.color == color and lane_start <= diamond.x_norm <= lane_end
        ]
        if not candidates:
            return False
        best = min(candidates, key=lambda diamond: abs(diamond.x_norm - lane_mid))
        self._diamonds = [diamond for diamond in self._diamonds if diamond.id != best.id]
        self._display_cleared += 1
        self._display_points += 0.5
        self._last_feedback = f"Clear: {best.color} lane."
        if self._phase is Phase.SCORED:
            self._scored_cleared += 1
            self._scored_points += 0.5
        else:
            self._advance_practice_progress(kind="colour")
        return True

    def _color_lane_bounds(self, color: str) -> tuple[float, float] | None:
        try:
            idx = self._LANE_COLORS.index(color)
        except ValueError:
            return None
        lane_w = (self._HIT_ZONE_END - self._HIT_ZONE_START) / float(len(self._LANE_COLORS))
        lane_start = self._HIT_ZONE_START + (lane_w * float(idx))
        lane_end = lane_start + lane_w
        return lane_start, lane_end

    def _advance_practice_progress(self, *, kind: str) -> None:
        if self._phase is not Phase.PRACTICE:
            return
        if self._profile.has_memory() and kind != "memory":
            return
        if not self._profile.has_memory() and self._profile.has_math() and kind != "math":
            return
        if not self._profile.has_memory() and not self._profile.has_math() and kind != "colour":
            return
        self._practice_progress += 1
        if self._practice_progress >= self._practice_rounds:
            self._phase = Phase.PRACTICE_DONE
            self._clear_channel_state()

    def _reset_visible_counters(self) -> None:
        self._display_points = 0.0
        self._display_cleared = 0
        self._display_missed = 0
        self._display_attempted = 0
        self._display_correct = 0

    def _clear_channel_state(self) -> None:
        self._memory_current = None
        self._math_current = None
        self._memory_cycle_started_at_s = None
        self._memory_prompted_at_s = None
        self._memory_answered = False
        self._last_update_s = None
        self._diamonds.clear()

    def _finish_results(self) -> None:
        self._phase = Phase.RESULTS
        self._clear_channel_state()


def _build_cln_drill(
    *,
    title_base: str,
    instructions: tuple[str, ...],
    profile: ClnDrillProfile,
    clock: Clock,
    seed: int,
    difficulty: float,
    mode: AntDrillMode | str,
    config: ClnDrillConfig,
) -> ClnDrillEngine:
    normalized_mode = _normalize_mode(mode)
    mode_profile = ANT_DRILL_MODE_PROFILES[normalized_mode]
    practice_rounds = (
        mode_profile.practice_questions
        if config.practice_rounds is None
        else int(config.practice_rounds)
    )
    scored_duration_s = (
        mode_profile.scored_duration_s
        if config.scored_duration_s is None
        else float(config.scored_duration_s)
    )
    return ClnDrillEngine(
        title=f"{title_base} ({mode_profile.label})",
        instructions=instructions,
        profile=profile,
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        practice_rounds=practice_rounds,
        scored_duration_s=scored_duration_s,
        mode=normalized_mode,
    )


def build_cln_sequence_copy_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: ClnDrillConfig | None = None,
) -> ClnDrillEngine:
    cfg = config or ClnDrillConfig()
    mode_profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    profile = ClnDrillProfile(
        active_channels=("memory",),
        memory_mode=ColoursLettersNumbersMemoryMode.VISIBLE_COPY,
        generation_profile=ColoursLettersNumbersGenerationProfile(
            sequence_min_easy=5,
            sequence_min_hard=7,
            sequence_max_easy=6,
            sequence_max_hard=9,
            option_style="scaffold",
        ),
        sequence_show_easy=18.0,
        sequence_show_hard=10.0,
        recall_window_easy=18.0,
        recall_window_hard=10.0,
        panel_prompt="Type the full sequence and press Enter.",
        control_hint="Type letters then Enter",
        input_label="Sequence Entry",
        show_text_entry=True,
        static_text="--",
        top_hint_override="Type the full sequence while it remains visible.",
        typed_memory_max_length=10,
    )
    return _build_cln_drill(
        title_base="Colours, Letters and Numbers: Sequence Copy",
        instructions=(
            "Colours, Letters and Numbers: Sequence Copy",
            f"Mode: {mode_profile.label}",
            "Visible-supported sequence copy to prime encoding rhythm and chunking before delayed recall blocks.",
            "Type the letters directly while the sequence stays on screen.",
            "Press Enter to begin practice.",
        ),
        profile=profile,
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
    )


def build_cln_sequence_match_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: ClnDrillConfig | None = None,
) -> ClnDrillEngine:
    cfg = config or ClnDrillConfig()
    mode_profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    profile = ClnDrillProfile(
        active_channels=("memory",),
        generation_profile=ColoursLettersNumbersGenerationProfile(
            sequence_min_easy=5,
            sequence_min_hard=6,
            sequence_max_easy=6,
            sequence_max_hard=8,
            option_style="scaffold",
        ),
        sequence_show_easy=4.0,
        sequence_show_hard=2.2,
        recall_delay_easy=1.6,
        recall_delay_hard=3.0,
        recall_delay_max_easy=2.4,
        recall_delay_max_hard=4.0,
        recall_window_easy=8.0,
        recall_window_hard=4.5,
        panel_prompt="Wait for the answer grid, then pick the match.",
        control_hint="Memory: A/S/D/F/G or mouse",
        input_label="Status",
        show_text_entry=False,
        static_text="A/S/D/F/G",
    )
    return _build_cln_drill(
        title_base="Colours, Letters and Numbers: Sequence Match",
        instructions=(
            "Colours, Letters and Numbers: Sequence Match",
            f"Mode: {mode_profile.label}",
            "Show the sequence, hold it through a longer warm-up delay, then choose the matching corner with the real CLN memory controls.",
            "This is the bridge from scaffolded sequence work into the live memory response format.",
            "Press Enter to begin practice.",
        ),
        profile=profile,
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
    )


def build_cln_sequence_math_recall_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: ClnDrillConfig | None = None,
) -> ClnDrillEngine:
    cfg = config or ClnDrillConfig()
    mode_profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    level = _level_from_difficulty(difficulty)
    exact_recall = level <= 5
    profile = ClnDrillProfile(
        active_channels=("memory", "math"),
        memory_mode=(
            ColoursLettersNumbersMemoryMode.DELAYED_EXACT
            if exact_recall
            else ColoursLettersNumbersMemoryMode.DELAYED_CHOICE
        ),
        generation_profile=ColoursLettersNumbersGenerationProfile(
            sequence_min_easy=5,
            sequence_min_hard=6,
            sequence_max_easy=6,
            sequence_max_hard=8,
            option_style="scaffold" if exact_recall else "default",
            math_profile=NumericalOperationsProblemProfile(operand_profile="clean_compute"),
            math_difficulty_offset=-0.05,
        ),
        sequence_show_easy=4.2,
        sequence_show_hard=2.4,
        recall_delay_easy=1.6,
        recall_delay_hard=3.2,
        recall_delay_max_easy=2.2,
        recall_delay_max_hard=4.2,
        recall_window_easy=8.0,
        recall_window_hard=5.0,
        panel_prompt="Hold the sequence, solve one math question, then recall the sequence.",
        control_hint=(
            "Enter: math, then type letters and Enter"
            if exact_recall
            else "Enter: math, then A/S/D/F/G or mouse"
        ),
        input_label="Math Answer",
        show_text_entry=True,
        static_text="--",
        top_hint_override="Store the sequence, solve one math item, then recall it.",
        typed_memory_max_length=10,
        required_math_answers_before_memory=1,
    )
    return _build_cln_drill(
        title_base="Colours, Letters and Numbers: Sequence Hold + One Math",
        instructions=(
            "Colours, Letters and Numbers: Sequence Hold + One Math",
            f"Mode: {mode_profile.label}",
            "See the sequence, let it disappear, solve one math question, then recall the stored sequence.",
            "Lower levels use exact typed recall; higher levels switch to the real corner-choice memory format.",
            "Press Enter to begin practice.",
        ),
        profile=profile,
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
    )


def build_cln_math_prime_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: ClnDrillConfig | None = None,
) -> ClnDrillEngine:
    cfg = config or ClnDrillConfig()
    mode_profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    profile = ClnDrillProfile(
        active_channels=("math",),
        generation_profile=ColoursLettersNumbersGenerationProfile(
            math_profile=NumericalOperationsProblemProfile(operand_profile="fact_prime"),
            math_difficulty_offset=-0.15,
        ),
        panel_prompt="Solve the math prompt and press Enter.",
        control_hint="Enter: math",
        input_label="Math Answer",
        show_text_entry=True,
        static_text="--",
        top_hint_override="Math channel only. Keep the arithmetic clean and low-pressure.",
    )
    return _build_cln_drill(
        title_base="Colours, Letters and Numbers: Math Prime",
        instructions=(
            "Colours, Letters and Numbers: Math Prime",
            f"Mode: {mode_profile.label}",
            "Low-pressure CLN math warm-up using the same full-screen CLN math lane but reusing Numerical Operations arithmetic logic.",
            "Prime the arithmetic before memory interference and lane pressure start stacking on top of it.",
            "Press Enter to begin practice.",
        ),
        profile=profile,
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
    )


def build_cln_colour_lane_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: ClnDrillConfig | None = None,
) -> ClnDrillEngine:
    cfg = config or ClnDrillConfig()
    mode_profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    profile = ClnDrillProfile(
        active_channels=("colour",),
        diamond_spawn_easy=2.0,
        diamond_spawn_hard=0.9,
        diamond_spawn_max_easy=2.4,
        diamond_spawn_max_hard=1.2,
        diamond_speed_easy=0.11,
        diamond_speed_hard=0.28,
        diamond_speed_max_easy=0.16,
        diamond_speed_max_hard=0.36,
        max_live_diamonds_easy=1,
        max_live_diamonds_hard=3,
        panel_prompt="Clear matching diamonds inside the colour lanes.",
        control_hint="Colour lanes: Q/W/E/R",
        input_label="Status",
        show_text_entry=False,
        static_text="Q/W/E/R",
        top_hint_override="Colour channel only. Clear matching diamonds in the lane zone.",
    )
    return _build_cln_drill(
        title_base="Colours, Letters and Numbers: Colour Lane Warm-Up",
        instructions=(
            "Colours, Letters and Numbers: Colour Lane Warm-Up",
            f"Mode: {mode_profile.label}",
            "Lane-clearing drill using the real CLN colour-zone layout and Q/W/E/R mapping.",
            "Build scan rhythm and hit timing before memory and math start interfering.",
            "Press Enter to begin practice.",
        ),
        profile=profile,
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
    )


def build_cln_memory_math_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.TEMPO,
    config: ClnDrillConfig | None = None,
) -> ClnDrillEngine:
    cfg = config or ClnDrillConfig()
    mode_profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    profile = ClnDrillProfile(
        active_channels=("memory", "math"),
        generation_profile=ColoursLettersNumbersGenerationProfile(
            sequence_min_easy=5,
            sequence_min_hard=6,
            sequence_max_easy=6,
            sequence_max_hard=8,
            option_style="default",
            math_profile=NumericalOperationsProblemProfile(operand_profile="clean_compute"),
            math_difficulty_offset=0.0,
        ),
        sequence_show_easy=3.4,
        sequence_show_hard=2.1,
        recall_delay_easy=1.0,
        recall_delay_hard=2.8,
        recall_delay_max_easy=1.8,
        recall_delay_max_hard=4.0,
        recall_window_easy=7.0,
        recall_window_hard=5.0,
        panel_prompt="Solve the math while holding the sequence in memory.",
        control_hint="Memory: A/S/D/F/G or mouse  |  Enter: math",
        input_label="Math Answer",
        show_text_entry=True,
        static_text="--",
    )
    return _build_cln_drill(
        title_base="Colours, Letters and Numbers: Memory + Math",
        instructions=(
            "Colours, Letters and Numbers: Memory + Math",
            f"Mode: {mode_profile.label}",
            "Hold the sequence through the blank and keep solving typed arithmetic without colour-lane interference yet.",
            "Use this block to build cross-channel stability before the full three-channel CLN runs.",
            "Press Enter to begin practice.",
        ),
        profile=profile,
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
    )


def build_cln_memory_colour_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.TEMPO,
    config: ClnDrillConfig | None = None,
) -> ClnDrillEngine:
    cfg = config or ClnDrillConfig()
    mode_profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    profile = ClnDrillProfile(
        active_channels=("memory", "colour"),
        generation_profile=ColoursLettersNumbersGenerationProfile(
            sequence_min_easy=5,
            sequence_min_hard=6,
            sequence_max_easy=6,
            sequence_max_hard=8,
            option_style="default",
        ),
        sequence_show_easy=3.3,
        sequence_show_hard=2.0,
        recall_delay_easy=1.2,
        recall_delay_hard=3.0,
        recall_delay_max_easy=2.0,
        recall_delay_max_hard=4.5,
        recall_window_easy=7.0,
        recall_window_hard=5.0,
        diamond_spawn_easy=1.8,
        diamond_spawn_hard=0.8,
        diamond_spawn_max_easy=2.2,
        diamond_spawn_max_hard=1.0,
        diamond_speed_easy=0.12,
        diamond_speed_hard=0.32,
        diamond_speed_max_easy=0.18,
        diamond_speed_max_hard=0.40,
        max_live_diamonds_easy=1,
        max_live_diamonds_hard=4,
        panel_prompt="Clear the lanes while holding the sequence in memory.",
        control_hint="Memory: A/S/D/F/G or mouse  |  Colour lanes: Q/W/E/R",
        input_label="Status",
        show_text_entry=False,
        static_text="A/S/D/F/G + Q/W/E/R",
    )
    return _build_cln_drill(
        title_base="Colours, Letters and Numbers: Memory + Colour",
        instructions=(
            "Colours, Letters and Numbers: Memory + Colour",
            f"Mode: {mode_profile.label}",
            "Hold the sequence while clearing the colour lanes, without typed math competing for the same block yet.",
            "This is the first real multitask bridge into the full CLN structure.",
            "Press Enter to begin practice.",
        ),
        profile=profile,
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
    )


def build_cln_full_steady_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.TEMPO,
    config: ClnDrillConfig | None = None,
) -> ClnDrillEngine:
    cfg = config or ClnDrillConfig()
    mode_profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    profile = ClnDrillProfile(
        active_channels=("memory", "math", "colour"),
        generation_profile=ColoursLettersNumbersGenerationProfile(
            sequence_min_easy=5,
            sequence_min_hard=6,
            sequence_max_easy=6,
            sequence_max_hard=8,
            option_style="default",
            math_profile=NumericalOperationsProblemProfile(operand_profile="clean_compute"),
            math_difficulty_offset=0.05,
        ),
        sequence_show_easy=3.2,
        sequence_show_hard=2.0,
        recall_delay_easy=2.0,
        recall_delay_hard=4.5,
        recall_delay_max_easy=3.0,
        recall_delay_max_hard=6.0,
        recall_window_easy=7.0,
        recall_window_hard=4.8,
        diamond_spawn_easy=1.8,
        diamond_spawn_hard=0.9,
        diamond_spawn_max_easy=2.2,
        diamond_spawn_max_hard=1.1,
        diamond_speed_easy=0.12,
        diamond_speed_hard=0.34,
        diamond_speed_max_easy=0.18,
        diamond_speed_max_hard=0.42,
        max_live_diamonds_easy=1,
        max_live_diamonds_hard=4,
        panel_prompt="Run the full CLN structure at steady pressure.",
        control_hint="Memory: A/S/D/F/G or mouse  |  Colour lanes: Q/W/E/R  |  Enter: math",
        input_label="Math Answer",
        show_text_entry=True,
        static_text="--",
    )
    return _build_cln_drill(
        title_base="Colours, Letters and Numbers: Full Steady",
        instructions=(
            "Colours, Letters and Numbers: Full Steady",
            f"Mode: {mode_profile.label}",
            "All three CLN channels stay live together with moderate sequence timing, typed math, and lane clearing.",
            "This block should feel like the real task, just before the hardest pressure profile.",
            "Press Enter to begin practice.",
        ),
        profile=profile,
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
    )


def build_cln_full_pressure_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.STRESS,
    config: ClnDrillConfig | None = None,
) -> ClnDrillEngine:
    cfg = config or ClnDrillConfig()
    mode_profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    profile = ClnDrillProfile(
        active_channels=("memory", "math", "colour"),
        generation_profile=ColoursLettersNumbersGenerationProfile(
            sequence_min_easy=5,
            sequence_min_hard=7,
            sequence_max_easy=7,
            sequence_max_hard=9,
            option_style="tight",
            math_profile=NumericalOperationsProblemProfile(operand_profile="default"),
            math_difficulty_offset=0.15,
        ),
        sequence_show_easy=2.8,
        sequence_show_hard=1.7,
        recall_delay_easy=3.0,
        recall_delay_hard=6.0,
        recall_delay_max_easy=4.5,
        recall_delay_max_hard=9.0,
        recall_window_easy=6.0,
        recall_window_hard=4.2,
        diamond_spawn_easy=1.4,
        diamond_spawn_hard=0.65,
        diamond_spawn_max_easy=1.8,
        diamond_spawn_max_hard=0.85,
        diamond_speed_easy=0.18,
        diamond_speed_hard=0.42,
        diamond_speed_max_easy=0.24,
        diamond_speed_max_hard=0.52,
        max_live_diamonds_easy=2,
        max_live_diamonds_hard=5,
        panel_prompt="Full CLN pressure: hold memory, keep the math clean, and keep the lanes alive.",
        control_hint="Memory: A/S/D/F/G or mouse  |  Colour lanes: Q/W/E/R  |  Enter: math",
        input_label="Math Answer",
        show_text_entry=True,
        static_text="--",
    )
    return _build_cln_drill(
        title_base="Colours, Letters and Numbers: Full Pressure",
        instructions=(
            "Colours, Letters and Numbers: Full Pressure",
            f"Mode: {mode_profile.label}",
            "The full three-channel CLN structure under denser diamonds, harder arithmetic, and more disruptive memory timing.",
            "Use it as the late-workout pressure block, not as a perfection block.",
            "Press Enter to begin practice.",
        ),
        profile=profile,
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
    )

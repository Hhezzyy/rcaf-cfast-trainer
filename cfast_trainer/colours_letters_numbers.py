from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .clock import Clock
from .cognitive_core import AttemptSummary, Phase, SeededRng, TestSnapshot, clamp01, lerp_int


@dataclass(frozen=True, slots=True)
class ColoursLettersNumbersConfig:
    # Guide indicates ~20 minutes including instructions.
    scored_duration_s: float = 18.0 * 60.0
    practice_rounds: int = 3
    # Memory channel answer window after the recall pause opens.
    round_duration_s: float = 9.0
    sequence_show_s: float = 2.0
    memory_recall_delay_s: float = 5.0
    memory_recall_delay_max_s: float = 60.0
    diamond_spawn_interval_s: float = 1.30
    diamond_spawn_interval_max_s: float = 1.90
    diamond_speed_norm_per_s: float = 0.12
    diamond_speed_max_norm_per_s: float = 0.48
    max_live_diamonds: int = 4


class ColoursLettersNumbersQuestionKind(StrEnum):
    MEMORY_SEQUENCE = "memory_sequence"
    MATH = "math"


@dataclass(frozen=True, slots=True)
class ColoursLettersNumbersOption:
    code: int
    label: str


@dataclass(frozen=True, slots=True)
class ColoursLettersNumbersDiamond:
    id: int
    color: str
    row: int
    x_norm: float


@dataclass(frozen=True, slots=True)
class ColoursLettersNumbersTrial:
    target_sequence: str
    options: tuple[ColoursLettersNumbersOption, ...]
    expected_option_code: int
    math_prompt: str
    math_answer: int


@dataclass(frozen=True, slots=True)
class ColoursLettersNumbersPayload:
    target_sequence: str | None
    options: tuple[ColoursLettersNumbersOption, ...]
    options_active: bool
    memory_answered: bool
    math_answered: bool
    math_prompt: str
    lane_colors: tuple[str, ...]
    lane_start_norm: float
    lane_end_norm: float
    diamonds: tuple[ColoursLettersNumbersDiamond, ...]
    missed_diamonds: int
    cleared_diamonds: int
    points: float


@dataclass(frozen=True, slots=True)
class ColoursLettersNumbersEvent:
    phase: Phase
    kind: ColoursLettersNumbersQuestionKind
    expected: str
    response: str
    is_correct: bool
    response_time_s: float


@dataclass(slots=True)
class _LiveDiamond:
    id: int
    color: str
    row: int
    x_norm: float
    speed_norm_per_s: float


class ColoursLettersNumbersGenerator:
    def __init__(self, rng: SeededRng):
        self._rng = rng

    def next_trial(self, *, difficulty: float) -> ColoursLettersNumbersTrial:
        difficulty = 0.0 if difficulty <= 0.0 else 1.0 if difficulty >= 1.0 else float(difficulty)
        sequence = self._build_sequence(difficulty=difficulty)
        options = self._build_options(sequence)
        expected_option_code = next((o.code for o in options if o.label == sequence), 1)
        math_prompt, math_answer = self._build_math(difficulty=difficulty)
        return ColoursLettersNumbersTrial(
            target_sequence=sequence,
            options=options,
            expected_option_code=expected_option_code,
            math_prompt=math_prompt,
            math_answer=math_answer,
        )

    def _build_sequence(self, *, difficulty: float) -> str:
        alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ"
        n = lerp_int(5, 6, difficulty)
        chars = [alphabet[self._rng.randint(0, len(alphabet) - 1)] for _ in range(n)]
        return "".join(chars)

    def _build_options(self, sequence: str) -> tuple[ColoursLettersNumbersOption, ...]:
        variants = [sequence]
        seen = {sequence}
        while len(variants) < 5:
            idx = int(self._rng.randint(0, len(sequence) - 1))
            alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ"
            replacement = sequence[idx]
            while replacement == sequence[idx]:
                replacement = alphabet[self._rng.randint(0, len(alphabet) - 1)]
            candidate = f"{sequence[:idx]}{replacement}{sequence[idx + 1 :]}"
            if candidate in seen:
                continue
            variants.append(candidate)
            seen.add(candidate)

        for i in range(len(variants) - 1, 0, -1):
            j = int(self._rng.randint(0, i))
            variants[i], variants[j] = variants[j], variants[i]

        return tuple(
            ColoursLettersNumbersOption(code=i + 1, label=s) for i, s in enumerate(variants)
        )

    def _build_math(self, *, difficulty: float) -> tuple[str, int]:
        lo = lerp_int(2, 4, difficulty)
        hi = lerp_int(9, 16, difficulty)
        a = int(self._rng.randint(lo, hi))
        b = int(self._rng.randint(lo, hi))
        op = int(self._rng.randint(0, 2))
        if op == 0:
            return f"{a} + {b} =", a + b
        if op == 1:
            if a < b:
                a, b = b, a
            return f"{a} - {b} =", a - b
        return f"{a} x {b} =", a * b


class ColoursLettersNumbersTest:
    _LANE_COLORS = ("RED", "YELLOW", "GREEN", "BLUE")
    _HIT_ZONE_START = 0.54
    _HIT_ZONE_END = 0.98
    _MAX_UPDATE_DT_S = 0.25

    def __init__(
        self,
        *,
        clock: Clock,
        seed: int,
        difficulty: float = 0.5,
        config: ColoursLettersNumbersConfig | None = None,
    ) -> None:
        cfg = config or ColoursLettersNumbersConfig()
        if not (0.0 <= difficulty <= 1.0):
            raise ValueError("difficulty must be in [0.0, 1.0]")
        if cfg.practice_rounds < 0:
            raise ValueError("practice_rounds must be >= 0")
        if cfg.scored_duration_s <= 0.0:
            raise ValueError("scored_duration_s must be > 0")
        if cfg.round_duration_s <= 0.0:
            raise ValueError("round_duration_s must be > 0")
        if cfg.sequence_show_s <= 0.0:
            raise ValueError("sequence_show_s must be > 0")
        if cfg.memory_recall_delay_s < 0.0:
            raise ValueError("memory_recall_delay_s must be >= 0")
        if cfg.memory_recall_delay_max_s < cfg.memory_recall_delay_s:
            raise ValueError("memory_recall_delay_max_s must be >= memory_recall_delay_s")
        if cfg.diamond_spawn_interval_s <= 0.0:
            raise ValueError("diamond_spawn_interval_s must be > 0")
        if cfg.diamond_spawn_interval_max_s < cfg.diamond_spawn_interval_s:
            raise ValueError("diamond_spawn_interval_max_s must be >= diamond_spawn_interval_s")
        if cfg.diamond_speed_norm_per_s <= 0.0:
            raise ValueError("diamond_speed_norm_per_s must be > 0")
        if cfg.diamond_speed_max_norm_per_s < cfg.diamond_speed_norm_per_s:
            raise ValueError("diamond_speed_max_norm_per_s must be >= diamond_speed_norm_per_s")
        if cfg.max_live_diamonds <= 0:
            raise ValueError("max_live_diamonds must be > 0")

        self._title = "Colours, Letters and Numbers"
        self._clock = clock
        self._difficulty = float(difficulty)
        self._cfg = cfg

        self._gen = ColoursLettersNumbersGenerator(SeededRng(int(seed)))
        self._rng = SeededRng(int(seed) ^ 0xA53F_91B7)

        self._phase = Phase.INSTRUCTIONS

        self._memory_current: ColoursLettersNumbersTrial | None = None
        self._math_current: ColoursLettersNumbersTrial | None = None

        self._memory_cycle_started_at_s: float | None = None
        self._memory_recall_delay_s_current: float = cfg.memory_recall_delay_s
        self._last_update_s: float | None = None
        self._spawn_cooldown_s: float = self._next_spawn_interval_s()
        self._memory_answered = False

        self._memory_prompted_at_s: float | None = None
        self._math_prompted_at_s: float | None = None

        self._practice_memory_cycles_completed = 0
        self._scored_started_at_s: float | None = None
        self._scored_attempted = 0
        self._scored_correct = 0
        self._scored_points = 0.0
        self._scored_cleared = 0
        self._scored_missed = 0

        self._events: list[ColoursLettersNumbersEvent] = []
        self._diamonds: list[_LiveDiamond] = []
        self._next_diamond_id = 1

    def can_exit(self) -> bool:
        return self._phase is not Phase.SCORED

    @property
    def phase(self) -> Phase:
        return self._phase

    def start_practice(self) -> None:
        if self._phase is not Phase.INSTRUCTIONS:
            return
        if self._cfg.practice_rounds == 0:
            self._phase = Phase.PRACTICE_DONE
            return
        self._phase = Phase.PRACTICE
        self._start_channels()

    def start_scored(self) -> None:
        if self._phase is not Phase.PRACTICE_DONE:
            return
        self._phase = Phase.SCORED
        self._scored_started_at_s = self._clock.now()
        self._start_channels()

    def time_remaining_s(self) -> float | None:
        if self._phase is not Phase.SCORED:
            return None
        assert self._scored_started_at_s is not None
        elapsed = self._clock.now() - self._scored_started_at_s
        return max(0.0, self._cfg.scored_duration_s - elapsed)

    def update(self) -> None:
        if self._phase is Phase.SCORED and self.time_remaining_s() == 0.0:
            self._finish_to_results()
            return
        if self._phase not in (Phase.PRACTICE, Phase.SCORED):
            return
        if self._memory_cycle_started_at_s is None:
            return

        now = self._clock.now()
        if self._last_update_s is None:
            self._last_update_s = now
        dt = max(0.0, now - self._last_update_s)
        self._last_update_s = now
        dt = min(dt, self._MAX_UPDATE_DT_S)

        self._update_diamonds(dt)
        self._update_memory_cycle(now)

    def submit_answer(self, raw: str) -> bool:
        if self._phase not in (Phase.PRACTICE, Phase.SCORED):
            return False
        if self._memory_current is None or self._math_current is None:
            return False

        command = str(raw).strip()
        if command == "":
            return False

        if command.upper().startswith("MEM:"):
            return self._submit_memory(command[4:])
        if command.upper().startswith("CLR:"):
            return self._submit_color(command[4:])
        return self._submit_math(command)

    def scored_summary(self) -> AttemptSummary:
        duration_s = float(self._cfg.scored_duration_s)
        attempted = int(self._scored_attempted)
        correct = int(self._scored_correct)
        accuracy = 0.0 if attempted == 0 else correct / attempted
        throughput = (attempted / duration_s) * 60.0
        rts = [e.response_time_s for e in self._events if e.phase is Phase.SCORED]
        mean_rt = None if not rts else (sum(rts) / len(rts))

        total_score = float(self._scored_points)
        max_score = (attempted * 2.0) + (self._scored_cleared * 0.5)
        score_ratio = 0.0 if max_score <= 0.0 else clamp01(max(0.0, total_score) / max_score)
        return AttemptSummary(
            attempted=attempted,
            correct=correct,
            accuracy=float(accuracy),
            duration_s=duration_s,
            throughput_per_min=float(throughput),
            mean_response_time_s=mean_rt,
            total_score=total_score,
            max_score=max_score,
            score_ratio=score_ratio,
        )

    def snapshot(self) -> TestSnapshot:
        return TestSnapshot(
            title=self._title,
            phase=self._phase,
            prompt=self._prompt_text(),
            input_hint="Math answer: type number then Enter",
            time_remaining_s=self.time_remaining_s(),
            attempted_scored=self._scored_attempted,
            correct_scored=self._scored_correct,
            payload=self._payload(),
        )

    def _options_active(self) -> bool:
        if self._memory_cycle_started_at_s is None:
            return False
        elapsed = self._clock.now() - self._memory_cycle_started_at_s
        return elapsed >= (self._cfg.sequence_show_s + self._memory_recall_delay_s_current)

    def _sequence_visible(self) -> bool:
        if self._memory_cycle_started_at_s is None:
            return False
        elapsed = self._clock.now() - self._memory_cycle_started_at_s
        return elapsed < self._cfg.sequence_show_s

    def _payload(self) -> ColoursLettersNumbersPayload | None:
        if self._phase not in (Phase.PRACTICE, Phase.SCORED):
            return None
        if self._memory_current is None or self._math_current is None:
            return None

        options_active = self._options_active()
        target_sequence = self._memory_current.target_sequence if self._sequence_visible() else None

        diamonds = tuple(
            ColoursLettersNumbersDiamond(
                id=d.id,
                color=d.color,
                row=d.row,
                x_norm=float(d.x_norm),
            )
            for d in self._diamonds
        )
        return ColoursLettersNumbersPayload(
            target_sequence=target_sequence,
            options=self._memory_current.options,
            options_active=options_active,
            memory_answered=self._memory_answered,
            math_answered=False,
            math_prompt=self._math_current.math_prompt,
            lane_colors=self._LANE_COLORS,
            lane_start_norm=self._HIT_ZONE_START,
            lane_end_norm=self._HIT_ZONE_END,
            diamonds=diamonds,
            missed_diamonds=self._scored_missed,
            cleared_diamonds=self._scored_cleared,
            points=float(self._scored_points),
        )

    def _prompt_text(self) -> str:
        if self._phase is Phase.INSTRUCTIONS:
            return "\n".join(
                [
                    "Colours, Letters and Numbers",
                    "",
                    "1) Memorize the sequence shown at the top.",
                    "2) Hold it in memory during the blank gap, then pick the match.",
                    "3) Choose the matching corner with A/S/D/F or mouse click.",
                    "4) Solve math by typing a number then Enter (no math timer).",
                    "5) Clear moving diamonds with color keys while over matching zones:",
                    "   Q=Red, W=Yellow, E=Green, R=Blue.",
                    "   Blank gap varies between 5 and 60 seconds.",
                    "Memory, math, and colours run on separate loops.",
                    "Letting diamonds pass reduces score.",
                    "",
                    "Press Enter to start practice.",
                ]
            )
        if self._phase is Phase.PRACTICE_DONE:
            return "Practice complete. Press Enter to begin the timed test."
        if self._phase is Phase.RESULTS:
            s = self.scored_summary()
            mean_ms = (
                "—" if s.mean_response_time_s is None else f"{s.mean_response_time_s * 1000.0:.0f}"
            )
            mm = int(round(s.duration_s)) // 60
            ss = int(round(s.duration_s)) % 60
            return "\n".join(
                [
                    "Results",
                    "",
                    f"Attempted: {s.attempted}",
                    f"Correct:   {s.correct}",
                    f"Accuracy:  {s.accuracy * 100.0:.1f}%",
                    f"Time:      {mm:02d}:{ss:02d}",
                    f"Rate:      {s.throughput_per_min:.1f} / min",
                    f"Mean RT:   {mean_ms} ms",
                    f"Cleared:   {self._scored_cleared}",
                    f"Missed:    {self._scored_missed}",
                    f"Score:     {s.total_score:.1f}",
                    "",
                    "Press Enter to return.",
                ]
            )
        if self._math_current is None:
            return ""
        return self._math_current.math_prompt

    def _start_channels(self) -> None:
        now = self._clock.now()
        self._last_update_s = now
        self._spawn_cooldown_s = self._next_spawn_interval_s()
        self._diamonds.clear()
        self._start_memory_cycle(now)
        self._start_math_cycle(now)

    def _start_memory_cycle(self, now: float) -> None:
        trial = self._gen.next_trial(difficulty=self._difficulty)
        self._memory_current = trial
        self._memory_cycle_started_at_s = now
        self._memory_recall_delay_s_current = self._rng.uniform(
            self._cfg.memory_recall_delay_s,
            self._cfg.memory_recall_delay_max_s,
        )
        self._memory_prompted_at_s = (
            now + self._cfg.sequence_show_s + self._memory_recall_delay_s_current
        )
        self._memory_answered = False

    def _start_math_cycle(self, now: float) -> None:
        trial = self._gen.next_trial(difficulty=self._difficulty)
        self._math_current = trial
        self._math_prompted_at_s = now

    def _update_memory_cycle(self, now: float) -> None:
        if self._memory_current is None or self._memory_cycle_started_at_s is None:
            return

        elapsed = now - self._memory_cycle_started_at_s
        total_cycle_s = (
            self._cfg.sequence_show_s
            + self._memory_recall_delay_s_current
            + self._cfg.round_duration_s
        )
        if elapsed < total_cycle_s:
            return

        if not self._memory_answered:
            self._record_memory_result(
                now=now,
                response="TIMEOUT",
                is_correct=False,
            )

        if self._phase is Phase.PRACTICE:
            self._practice_memory_cycles_completed += 1
            if self._practice_memory_cycles_completed >= self._cfg.practice_rounds:
                self._phase = Phase.PRACTICE_DONE
                self._clear_task_state()
                return

        self._start_memory_cycle(now)

    def _clear_task_state(self) -> None:
        self._memory_current = None
        self._math_current = None
        self._memory_cycle_started_at_s = None
        self._last_update_s = None
        self._memory_prompted_at_s = None
        self._math_prompted_at_s = None
        self._memory_answered = False
        self._diamonds.clear()

    def _finish_to_results(self) -> None:
        self._phase = Phase.RESULTS
        self._clear_task_state()

    def _update_diamonds(self, dt: float) -> None:
        if dt <= 0.0:
            return

        self._spawn_cooldown_s -= dt
        while self._spawn_cooldown_s <= 0.0:
            overdue = -self._spawn_cooldown_s
            self._spawn_cooldown_s = self._next_spawn_interval_s() - overdue

            spawn_slots = max(0, self._cfg.max_live_diamonds - len(self._diamonds))
            if spawn_slots > 0:
                self._spawn_diamond()

        survivors: list[_LiveDiamond] = []
        for d in self._diamonds:
            d.x_norm += d.speed_norm_per_s * dt
            if d.x_norm >= 1.02:
                if self._phase is Phase.SCORED:
                    self._scored_missed += 1
                    self._scored_points -= 1.0
                continue
            survivors.append(d)
        self._diamonds = survivors

    def _spawn_diamond(self) -> None:
        color = str(self._rng.choice(self._LANE_COLORS))
        row = self._sample_diamond_row()
        d = _LiveDiamond(
            id=self._next_diamond_id,
            color=color,
            row=row,
            x_norm=0.02,
            speed_norm_per_s=self._rng.uniform(
                self._cfg.diamond_speed_norm_per_s,
                self._cfg.diamond_speed_max_norm_per_s,
            ),
        )
        self._next_diamond_id += 1
        self._diamonds.append(d)

    def _next_spawn_interval_s(self) -> float:
        return self._rng.uniform(
            self._cfg.diamond_spawn_interval_s,
            self._cfg.diamond_spawn_interval_max_s,
        )

    def _sample_diamond_row(self) -> int:
        # Bias the stream slightly lower on screen while preserving all three rows.
        pick = self._rng.random()
        if pick < 0.20:
            return 0
        if pick < 0.55:
            return 1
        return 2

    def _record_memory_result(self, *, now: float, response: str, is_correct: bool) -> None:
        if self._memory_current is None:
            return
        base = self._memory_prompted_at_s if self._memory_prompted_at_s is not None else now
        rt = max(0.0, now - base)
        self._events.append(
            ColoursLettersNumbersEvent(
                phase=self._phase,
                kind=ColoursLettersNumbersQuestionKind.MEMORY_SEQUENCE,
                expected=str(self._memory_current.expected_option_code),
                response=str(response),
                is_correct=is_correct,
                response_time_s=rt,
            )
        )
        self._memory_answered = True

        if self._phase is Phase.SCORED:
            self._scored_attempted += 1
            if is_correct:
                self._scored_correct += 1
                self._scored_points += 2.0

    def _submit_memory(self, raw_code: str) -> bool:
        if self._memory_current is None:
            return False
        if not self._options_active():
            return False
        if self._memory_answered:
            return False

        code_text = "".join(ch for ch in str(raw_code) if ch.isdigit())
        if code_text == "":
            return False
        code = int(code_text)
        is_correct = code == int(self._memory_current.expected_option_code)

        now = self._clock.now()
        self._record_memory_result(now=now, response=str(code), is_correct=is_correct)
        return True

    def _submit_math(self, raw_text: str) -> bool:
        if self._math_current is None:
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

        is_correct = int(value) == int(self._math_current.math_answer)

        now = self._clock.now()
        base = self._math_prompted_at_s if self._math_prompted_at_s is not None else now
        rt = max(0.0, now - base)

        self._events.append(
            ColoursLettersNumbersEvent(
                phase=self._phase,
                kind=ColoursLettersNumbersQuestionKind.MATH,
                expected=str(self._math_current.math_answer),
                response=str(value),
                is_correct=is_correct,
                response_time_s=rt,
            )
        )

        if self._phase is Phase.SCORED:
            self._scored_attempted += 1
            if is_correct:
                self._scored_correct += 1
                self._scored_points += 2.0

        self._start_math_cycle(now)
        return True

    def _submit_color(self, raw_key: str) -> bool:
        key = str(raw_key).strip().upper()
        key_map = {
            "Q": "RED",
            "W": "YELLOW",
            "E": "GREEN",
            "R": "BLUE",
            # Backward-compatible aliases.
            "RED": "RED",
            "YELLOW": "YELLOW",
            "GREEN": "GREEN",
            "BLUE": "BLUE",
            "Y": "YELLOW",
            "G": "GREEN",
            "B": "BLUE",
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
            d for d in self._diamonds if d.color == color and lane_start <= d.x_norm <= lane_end
        ]
        if not candidates:
            return False

        best = min(candidates, key=lambda d: abs(d.x_norm - lane_mid))
        self._diamonds = [d for d in self._diamonds if d.id != best.id]

        if self._phase is Phase.SCORED:
            self._scored_cleared += 1
            self._scored_points += 0.5
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


def build_colours_letters_numbers_test(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    practice: bool = True,
    scored_duration_s: float | None = None,
    config: ColoursLettersNumbersConfig | None = None,
) -> ColoursLettersNumbersTest:
    cfg = config or ColoursLettersNumbersConfig()
    if not practice:
        cfg = ColoursLettersNumbersConfig(
            scored_duration_s=cfg.scored_duration_s,
            practice_rounds=0,
            round_duration_s=cfg.round_duration_s,
            sequence_show_s=cfg.sequence_show_s,
            memory_recall_delay_s=cfg.memory_recall_delay_s,
            memory_recall_delay_max_s=cfg.memory_recall_delay_max_s,
            diamond_spawn_interval_s=cfg.diamond_spawn_interval_s,
            diamond_spawn_interval_max_s=cfg.diamond_spawn_interval_max_s,
            diamond_speed_norm_per_s=cfg.diamond_speed_norm_per_s,
            diamond_speed_max_norm_per_s=cfg.diamond_speed_max_norm_per_s,
            max_live_diamonds=cfg.max_live_diamonds,
        )
    if scored_duration_s is not None:
        cfg = ColoursLettersNumbersConfig(
            scored_duration_s=float(scored_duration_s),
            practice_rounds=cfg.practice_rounds,
            round_duration_s=cfg.round_duration_s,
            sequence_show_s=cfg.sequence_show_s,
            memory_recall_delay_s=cfg.memory_recall_delay_s,
            memory_recall_delay_max_s=cfg.memory_recall_delay_max_s,
            diamond_spawn_interval_s=cfg.diamond_spawn_interval_s,
            diamond_spawn_interval_max_s=cfg.diamond_spawn_interval_max_s,
            diamond_speed_norm_per_s=cfg.diamond_speed_norm_per_s,
            diamond_speed_max_norm_per_s=cfg.diamond_speed_max_norm_per_s,
            max_live_diamonds=cfg.max_live_diamonds,
        )
    return ColoursLettersNumbersTest(
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        config=cfg,
    )

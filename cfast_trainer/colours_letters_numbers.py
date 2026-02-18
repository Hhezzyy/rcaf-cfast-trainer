from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .clock import Clock
from .cognitive_core import AttemptSummary, Phase, SeededRng, TestSnapshot, lerp_int


class ColoursLettersNumbersQuestionKind(str, Enum):
    LETTER_MATCH = "letter_match"
    EQUATION = "equation"
    COLOR_COUNT = "color_count"


@dataclass(frozen=True, slots=True)
class ColoursLettersNumbersOption:
    code: int
    label: str


@dataclass(frozen=True, slots=True)
class ColoursLettersNumbersTrial:
    kind: ColoursLettersNumbersQuestionKind
    target: str
    options: tuple[ColoursLettersNumbersOption, ...]
    equation_text: str
    color_bars: tuple[str, ...]
    prompt: str
    expected: str


@dataclass(frozen=True, slots=True)
class ColoursLettersNumbersPayload:
    kind: ColoursLettersNumbersQuestionKind
    target: str
    options: tuple[ColoursLettersNumbersOption, ...]
    equation_text: str
    color_bars: tuple[str, ...]
    prompt: str


@dataclass(frozen=True, slots=True)
class ColoursLettersNumbersEvent:
    phase: Phase
    kind: ColoursLettersNumbersQuestionKind
    expected: str
    response: str
    is_correct: bool
    response_time_s: float


class ColoursLettersNumbersGenerator:
    def __init__(self, rng: SeededRng):
        self._rng = rng
        self._index = 0

    def next_trial(self, *, difficulty: float) -> ColoursLettersNumbersTrial:
        difficulty = 0.0 if difficulty <= 0.0 else 1.0 if difficulty >= 1.0 else float(difficulty)
        kind = (
            ColoursLettersNumbersQuestionKind.LETTER_MATCH
            if self._index % 3 == 0
            else (
                ColoursLettersNumbersQuestionKind.EQUATION
                if self._index % 3 == 1
                else ColoursLettersNumbersQuestionKind.COLOR_COUNT
            )
        )

        target = self._build_word(length=5)
        options = self._build_options(target)

        equation_text, equation_answer = self._build_equation(difficulty)
        color_bars = self._build_color_bars(difficulty)

        if kind is ColoursLettersNumbersQuestionKind.LETTER_MATCH:
            expected_code = next((str(o.code) for o in options if o.label == target), "1")
            prompt = "WHICH OPTION MATCHES THE TARGET STRING? (1-4)"
            expected = expected_code
        elif kind is ColoursLettersNumbersQuestionKind.EQUATION:
            prompt = f"SOLVE: {equation_text}"
            expected = str(equation_answer)
        else:
            target_color = color_bars[self._rng.randint(0, len(color_bars) - 1)]
            prompt = f"HOW MANY {target_color} BARS ARE SHOWN?"
            expected = str(sum(1 for c in color_bars if c == target_color))

        self._index += 1
        return ColoursLettersNumbersTrial(
            kind=kind,
            target=target,
            options=options,
            equation_text=equation_text,
            color_bars=color_bars,
            prompt=prompt,
            expected=expected,
        )

    def _build_word(self, *, length: int) -> str:
        alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ"
        chars = [alphabet[self._rng.randint(0, len(alphabet) - 1)] for _ in range(length)]
        return "".join(chars)

    def _build_mutation(self, word: str) -> str:
        alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ"
        idx = int(self._rng.randint(0, len(word) - 1))
        old = word[idx]
        replacement = old
        while replacement == old:
            replacement = alphabet[self._rng.randint(0, len(alphabet) - 1)]
        return f"{word[:idx]}{replacement}{word[idx + 1:]}"

    def _build_options(self, target: str) -> tuple[ColoursLettersNumbersOption, ...]:
        values = [target]
        seen = {target}
        while len(values) < 4:
            candidate = self._build_mutation(target)
            if candidate in seen:
                continue
            values.append(candidate)
            seen.add(candidate)

        # Deterministic in-place Fisher-Yates shuffle.
        for i in range(len(values) - 1, 0, -1):
            j = int(self._rng.randint(0, i))
            values[i], values[j] = values[j], values[i]

        return tuple(ColoursLettersNumbersOption(code=i + 1, label=v) for i, v in enumerate(values))

    def _build_equation(self, difficulty: float) -> tuple[str, int]:
        lo = lerp_int(2, 4, difficulty)
        hi = lerp_int(9, 14, difficulty)
        a = int(self._rng.randint(lo, hi))
        b = int(self._rng.randint(lo, hi))

        op_pick = int(self._rng.randint(0, 2))
        if op_pick == 0:
            return f"{a} + {b} =", a + b
        if op_pick == 1:
            if a < b:
                a, b = b, a
            return f"{a} - {b} =", a - b
        return f"{a} x {b} =", a * b

    def _build_color_bars(self, difficulty: float) -> tuple[str, ...]:
        palette = ("RED", "YELLOW", "GREEN", "BLUE")
        n = lerp_int(4, 7, difficulty)
        return tuple(palette[self._rng.randint(0, len(palette) - 1)] for _ in range(n))


class ColoursLettersNumbersTest:
    def __init__(
        self,
        *,
        clock: Clock,
        seed: int,
        difficulty: float = 0.5,
        practice_questions: int = 4,
        scored_duration_s: float = 240.0,
    ) -> None:
        if not (0.0 <= difficulty <= 1.0):
            raise ValueError("difficulty must be in [0.0, 1.0]")
        if practice_questions < 0:
            raise ValueError("practice_questions must be >= 0")
        if scored_duration_s <= 0.0:
            raise ValueError("scored_duration_s must be > 0")

        self._title = "Colours, Letters and Numbers"
        self._clock = clock
        self._difficulty = float(difficulty)
        self._practice_questions = int(practice_questions)
        self._scored_duration_s = float(scored_duration_s)

        self._gen = ColoursLettersNumbersGenerator(SeededRng(int(seed)))

        self._phase = Phase.INSTRUCTIONS
        self._current: ColoursLettersNumbersTrial | None = None
        self._question_presented_at_s: float | None = None

        self._practice_answered = 0
        self._scored_started_at_s: float | None = None
        self._scored_attempted = 0
        self._scored_correct = 0
        self._events: list[ColoursLettersNumbersEvent] = []

    def can_exit(self) -> bool:
        return self._phase is not Phase.SCORED

    @property
    def phase(self) -> Phase:
        return self._phase

    def start_practice(self) -> None:
        if self._phase is not Phase.INSTRUCTIONS:
            return
        if self._practice_questions == 0:
            self._phase = Phase.PRACTICE_DONE
            return
        self._phase = Phase.PRACTICE
        self._deal_new_trial()

    def start_scored(self) -> None:
        if self._phase is not Phase.PRACTICE_DONE:
            return
        self._phase = Phase.SCORED
        self._scored_started_at_s = self._clock.now()
        self._deal_new_trial()

    def time_remaining_s(self) -> float | None:
        if self._phase is not Phase.SCORED:
            return None
        assert self._scored_started_at_s is not None
        elapsed = self._clock.now() - self._scored_started_at_s
        return max(0.0, self._scored_duration_s - elapsed)

    def update(self) -> None:
        if self._phase is Phase.SCORED and self.time_remaining_s() == 0.0:
            self._finish_to_results()

    def submit_answer(self, raw: str) -> bool:
        if self._phase not in (Phase.PRACTICE, Phase.SCORED):
            return False

        expired = self._phase is Phase.SCORED and self.time_remaining_s() == 0.0

        assert self._current is not None
        assert self._question_presented_at_s is not None

        raw_text = str(raw).strip()
        if raw_text.startswith("-"):
            rest = "".join(ch for ch in raw_text[1:] if ch.isdigit())
            user = f"-{rest}" if rest else ""
        else:
            user = "".join(ch for ch in raw_text if ch.isdigit())
        if user == "":
            if expired:
                self._finish_to_results()
            return False

        answered_at_s = self._clock.now()
        rt = max(0.0, answered_at_s - self._question_presented_at_s)
        expected = self._current.expected

        try:
            is_correct = int(user) == int(expected)
        except ValueError:
            is_correct = user == expected

        self._events.append(
            ColoursLettersNumbersEvent(
                phase=self._phase,
                kind=self._current.kind,
                expected=expected,
                response=user,
                is_correct=is_correct,
                response_time_s=rt,
            )
        )

        if self._phase is Phase.SCORED:
            self._scored_attempted += 1
            if is_correct:
                self._scored_correct += 1
        else:
            self._practice_answered += 1

        if expired and self._phase is Phase.SCORED:
            self._finish_to_results()
            return True

        if self._phase is Phase.PRACTICE and self._practice_answered >= self._practice_questions:
            self._phase = Phase.PRACTICE_DONE
            self._current = None
            self._question_presented_at_s = None
            return True

        self._deal_new_trial()
        return True

    def scored_summary(self) -> AttemptSummary:
        duration_s = float(self._scored_duration_s)
        attempted = int(self._scored_attempted)
        correct = int(self._scored_correct)
        accuracy = 0.0 if attempted == 0 else correct / attempted
        throughput = (attempted / duration_s) * 60.0
        rts = [e.response_time_s for e in self._events if e.phase is Phase.SCORED]
        mean_rt = None if not rts else (sum(rts) / len(rts))
        return AttemptSummary(
            attempted=attempted,
            correct=correct,
            accuracy=float(accuracy),
            duration_s=duration_s,
            throughput_per_min=float(throughput),
            mean_response_time_s=mean_rt,
        )

    def snapshot(self) -> TestSnapshot:
        return TestSnapshot(
            title=self._title,
            phase=self._phase,
            prompt=self._prompt_text(),
            input_hint="Type numeric answer then Enter",
            time_remaining_s=self.time_remaining_s(),
            attempted_scored=self._scored_attempted,
            correct_scored=self._scored_correct,
            payload=self._payload(),
        )

    def _payload(self) -> ColoursLettersNumbersPayload | None:
        if self._phase not in (Phase.PRACTICE, Phase.SCORED):
            return None
        assert self._current is not None
        return ColoursLettersNumbersPayload(
            kind=self._current.kind,
            target=self._current.target,
            options=self._current.options,
            equation_text=self._current.equation_text,
            color_bars=self._current.color_bars,
            prompt=self._current.prompt,
        )

    def _prompt_text(self) -> str:
        if self._phase is Phase.INSTRUCTIONS:
            return "\n".join(
                [
                    "Colours, Letters and Numbers",
                    "",
                    "Shift attention between multiple task channels.",
                    "- Match letter strings",
                    "- Solve simple arithmetic",
                    "- Monitor colour bars",
                    "",
                    "Type numeric answers and press Enter.",
                    "Press Enter to start practice.",
                ]
            )
        if self._phase is Phase.PRACTICE_DONE:
            return "Practice complete. Press Enter to begin the timed test."
        if self._phase is Phase.RESULTS:
            s = self.scored_summary()
            mean_ms = "—" if s.mean_response_time_s is None else f"{s.mean_response_time_s * 1000.0:.0f}"
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
                    "",
                    "Press Enter to return.",
                ]
            )
        assert self._current is not None
        return self._current.prompt

    def _deal_new_trial(self) -> None:
        self._current = self._gen.next_trial(difficulty=self._difficulty)
        self._question_presented_at_s = self._clock.now()

    def _finish_to_results(self) -> None:
        self._phase = Phase.RESULTS
        self._current = None
        self._question_presented_at_s = None


def build_colours_letters_numbers_test(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    practice: bool = True,
    scored_duration_s: float = 240.0,
) -> ColoursLettersNumbersTest:
    return ColoursLettersNumbersTest(
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        practice_questions=(4 if practice else 0),
        scored_duration_s=float(scored_duration_s),
    )

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .clock import Clock
from .cognitive_core import AttemptSummary, Phase, SeededRng, TestSnapshot, lerp_int


class DigitRecognitionQuestionKind(str, Enum):
    RECALL = "recall"
    COUNT_TARGET = "count_target"


@dataclass(frozen=True, slots=True)
class DigitRecognitionTrial:
    digits: str
    kind: DigitRecognitionQuestionKind
    prompt: str
    expected: str


@dataclass(frozen=True, slots=True)
class DigitRecognitionPayload:
    display_digits: str | None
    accepting_input: bool


class _Stage(str, Enum):
    SHOW = "show"
    MASK = "mask"
    QUESTION = "question"


@dataclass(frozen=True, slots=True)
class DigitRecognitionEvent:
    phase: Phase
    kind: DigitRecognitionQuestionKind
    expected: str
    response: str
    is_correct: bool
    response_time_s: float


class DigitRecognitionGenerator:
    def __init__(self, rng: SeededRng):
        self._rng = rng
        self._index = 0

    def next_trial(self, *, difficulty: float) -> DigitRecognitionTrial:
        difficulty = 0.0 if difficulty <= 0.0 else 1.0 if difficulty >= 1.0 else float(difficulty)

        lo = lerp_int(5, 7, difficulty)
        hi = lerp_int(8, 11, difficulty)
        n = int(self._rng.randint(lo, hi))

        digits = "".join(str(self._rng.randint(0, 9)) for _ in range(n))

        kind = (
            DigitRecognitionQuestionKind.RECALL
            if (self._index % 2 == 0)
            else DigitRecognitionQuestionKind.COUNT_TARGET
        )

        if kind is DigitRecognitionQuestionKind.RECALL:
            prompt = "ENTER THE DIGIT STRING:"
            expected = digits
        else:
            target = str(self._rng.randint(0, 9))
            prompt = f"HOW MANY {target}s WERE THERE?"
            expected = str(sum(1 for ch in digits if ch == target))

        self._index += 1
        return DigitRecognitionTrial(digits=digits, kind=kind, prompt=prompt, expected=expected)


class DigitRecognitionTest:
    def __init__(
        self,
        *,
        clock: Clock,
        seed: int,
        difficulty: float = 0.5,
        practice_questions: int = 3,
        scored_duration_s: float = 360.0,
        display_s: float = 1.25,
        mask_s: float = 0.25,
    ) -> None:
        if not (0.0 <= difficulty <= 1.0):
            raise ValueError("difficulty must be in [0.0, 1.0]")
        if practice_questions < 0:
            raise ValueError("practice_questions must be >= 0")
        if scored_duration_s <= 0.0:
            raise ValueError("scored_duration_s must be > 0")
        if display_s <= 0.0:
            raise ValueError("display_s must be > 0")
        if mask_s < 0.0:
            raise ValueError("mask_s must be >= 0")

        self._title = "Digit Recognition"
        self._clock = clock
        self._difficulty = float(difficulty)

        self._practice_questions = int(practice_questions)
        self._scored_duration_s = float(scored_duration_s)
        self._display_s = float(display_s)
        self._mask_s = float(mask_s)

        self._gen = DigitRecognitionGenerator(SeededRng(int(seed)))

        self._phase = Phase.INSTRUCTIONS
        self._stage: _Stage | None = None
        self._stage_started_at_s: float | None = None
        self._question_presented_at_s: float | None = None
        self._current: DigitRecognitionTrial | None = None

        self._practice_answered = 0

        self._scored_started_at_s: float | None = None
        self._scored_attempted = 0
        self._scored_correct = 0

        self._events: list[DigitRecognitionEvent] = []

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
            return

        if self._phase not in (Phase.PRACTICE, Phase.SCORED):
            return
        if self._stage is None or self._stage_started_at_s is None:
            return

        now = self._clock.now()
        elapsed = now - self._stage_started_at_s

        if self._stage is _Stage.SHOW and elapsed >= self._display_s:
            self._stage = _Stage.MASK
            self._stage_started_at_s = now
            return

        if self._stage is _Stage.MASK and elapsed >= self._mask_s:
            self._stage = _Stage.QUESTION
            self._stage_started_at_s = now
            self._question_presented_at_s = now
            return

    def submit_answer(self, raw: str) -> bool:
        if self._phase not in (Phase.PRACTICE, Phase.SCORED):
            return False

        expired = self._phase is Phase.SCORED and self.time_remaining_s() == 0.0

        if self._stage is not _Stage.QUESTION:
            if expired:
                self._finish_to_results()
            return False

        assert self._current is not None
        assert self._question_presented_at_s is not None

        user = "".join(ch for ch in str(raw) if ch.isdigit())
        if user == "":
            if expired:
                self._finish_to_results()
            return False

        answered_at_s = self._clock.now()
        rt = max(0.0, answered_at_s - self._question_presented_at_s)

        expected = self._current.expected
        if self._current.kind is DigitRecognitionQuestionKind.RECALL:
            is_correct = user == expected
        else:
            try:
                is_correct = int(user) == int(expected)
            except ValueError:
                is_correct = False

        self._events.append(
            DigitRecognitionEvent(
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
            self._stage = None
            self._stage_started_at_s = None
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
            input_hint="Type digits then Enter",
            time_remaining_s=self.time_remaining_s(),
            attempted_scored=self._scored_attempted,
            correct_scored=self._scored_correct,
            payload=self._payload(),
        )

    def _payload(self) -> DigitRecognitionPayload | None:
        if self._phase not in (Phase.PRACTICE, Phase.SCORED):
            return None
        if self._stage is _Stage.SHOW:
            assert self._current is not None
            return DigitRecognitionPayload(display_digits=" ".join(self._current.digits), accepting_input=False)
        if self._stage is _Stage.MASK:
            return DigitRecognitionPayload(display_digits=None, accepting_input=False)
        if self._stage is _Stage.QUESTION:
            return DigitRecognitionPayload(display_digits=None, accepting_input=True)
        return None

    def _prompt_text(self) -> str:
        if self._phase is Phase.INSTRUCTIONS:
            return "\n".join(
                [
                    "Digit Recognition",
                    "",
                    "Remember strings of digits of varying lengths.",
                    "Answer questions about the digit string shown.",
                    "",
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

        if self._stage is _Stage.SHOW:
            return "MEMORIZE"
        if self._stage is _Stage.MASK:
            return ""
        assert self._current is not None
        return self._current.prompt

    def _deal_new_trial(self) -> None:
        self._current = self._gen.next_trial(difficulty=self._difficulty)
        self._stage = _Stage.SHOW
        now = self._clock.now()
        self._stage_started_at_s = now
        self._question_presented_at_s = None

    def _finish_to_results(self) -> None:
        self._phase = Phase.RESULTS
        self._current = None
        self._stage = None
        self._stage_started_at_s = None
        self._question_presented_at_s = None


def build_digit_recognition_test(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    practice: bool = True,
    scored_duration_s: float = 360.0,
) -> DigitRecognitionTest:
    return DigitRecognitionTest(
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        practice_questions=(3 if practice else 0),
        scored_duration_s=float(scored_duration_s),
        display_s=1.25,
        mask_s=0.25,
    )

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from .clock import Clock
from .cognitive_core import AttemptSummary, Phase, SeededRng, TestSnapshot, lerp_int


class DigitRecognitionQuestionKind(StrEnum):
    RECALL = "recall"
    COUNT_TARGET = "count_target"
    DIFFERENT_DIGIT = "different_digit"


@dataclass(frozen=True, slots=True)
class DigitRecognitionProfile:
    allowed_kinds: tuple[DigitRecognitionQuestionKind, ...] = (
        DigitRecognitionQuestionKind.RECALL,
        DigitRecognitionQuestionKind.COUNT_TARGET,
        DigitRecognitionQuestionKind.DIFFERENT_DIGIT,
    )
    min_length_easy: int = 5
    min_length_hard: int = 7
    max_length_easy: int = 8
    max_length_hard: int = 11
    display_s_easy: float = 1.25
    display_s_hard: float = 1.25
    mask_s_easy: float = 0.25
    mask_s_hard: float = 0.25
    visible_supported: bool = False
    string_profile: Literal["default", "friendly", "noisy"] = "default"

    def normalized_kinds(self) -> tuple[DigitRecognitionQuestionKind, ...]:
        unique = tuple(dict.fromkeys(self.allowed_kinds))
        if not unique:
            return (
                DigitRecognitionQuestionKind.RECALL,
                DigitRecognitionQuestionKind.COUNT_TARGET,
                DigitRecognitionQuestionKind.DIFFERENT_DIGIT,
            )
        return unique

    def display_s_for(self, difficulty: float) -> float:
        d = 0.0 if difficulty <= 0.0 else 1.0 if difficulty >= 1.0 else float(difficulty)
        return float(self.display_s_easy) + ((float(self.display_s_hard) - float(self.display_s_easy)) * d)

    def mask_s_for(self, difficulty: float) -> float:
        d = 0.0 if difficulty <= 0.0 else 1.0 if difficulty >= 1.0 else float(difficulty)
        return float(self.mask_s_easy) + ((float(self.mask_s_hard) - float(self.mask_s_easy)) * d)


@dataclass(frozen=True, slots=True)
class DigitRecognitionTrial:
    digits: str
    comparison_digits: str | None
    kind: DigitRecognitionQuestionKind
    prompt: str
    expected: str


@dataclass(frozen=True, slots=True)
class DigitRecognitionPayload:
    display_digits: str | None
    display_lines: tuple[str, ...] | None
    accepting_input: bool
    prompt_text: str = ""
    input_digits: int = 16
    family: str = ""


@dataclass(frozen=True, slots=True)
class DigitRecognitionTrainingSpec:
    kind: DigitRecognitionQuestionKind
    display_lines: tuple[str, ...]
    question_prompt: str
    expected_digits: str
    display_answer_text: str
    input_digits: int
    answer_digits: int
    initial_display_s: float
    mask_s: float
    keep_display_visible_during_question: bool
    question_cap_s: float | None = None
    mask_display_lines: tuple[str, ...] = ()
    mask_prompt: str = ""
    family_tag: str = ""
    span_length: int = 0
    delay_s: float = 0.0
    query_complexity: int = 0
    interference_rate: int = 0


class _Stage(StrEnum):
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
    def __init__(self, rng: SeededRng, *, profile: DigitRecognitionProfile | None = None):
        self._rng = rng
        self._index = 0
        self._profile = profile or DigitRecognitionProfile()
        self._allowed_kinds = self._profile.normalized_kinds()

    def next_digit_string(self, *, difficulty: float) -> str:
        difficulty = 0.0 if difficulty <= 0.0 else 1.0 if difficulty >= 1.0 else float(difficulty)
        lo = lerp_int(self._profile.min_length_easy, self._profile.min_length_hard, difficulty)
        hi = lerp_int(self._profile.max_length_easy, self._profile.max_length_hard, difficulty)
        n = int(self._rng.randint(lo, hi))
        return self._build_digit_string(length=n, difficulty=difficulty)

    def next_trial(self, *, difficulty: float) -> DigitRecognitionTrial:
        difficulty = 0.0 if difficulty <= 0.0 else 1.0 if difficulty >= 1.0 else float(difficulty)
        digits = self.next_digit_string(difficulty=difficulty)
        n = len(digits)
        kind = self._pick_kind()

        if kind is DigitRecognitionQuestionKind.RECALL:
            prompt = "ENTER THE DIGIT STRING:"
            expected = digits
            comparison_digits = None
        else:
            if kind is DigitRecognitionQuestionKind.COUNT_TARGET:
                target = str(self._rng.randint(0, 9))
                prompt = f"HOW MANY {target}s WERE THERE?"
                expected = str(sum(1 for ch in digits if ch == target))
                comparison_digits = None
            else:
                pos = int(self._rng.randint(0, n - 1))
                original = digits[pos]
                replacement_pool = self._replacement_pool(
                    digits=digits,
                    original_digit=original,
                    difficulty=difficulty,
                )
                changed = str(self._rng.choice(replacement_pool))
                comparison_digits = digits[:pos] + changed + digits[pos + 1 :]
                prompt = "SECOND STRING: WHAT DIGIT WAS DIFFERENT?"
                expected = changed

        self._index += 1
        return DigitRecognitionTrial(
            digits=digits,
            comparison_digits=comparison_digits,
            kind=kind,
            prompt=prompt,
            expected=expected,
        )

    def _pick_kind(self) -> DigitRecognitionQuestionKind:
        if len(self._allowed_kinds) == 1:
            return self._allowed_kinds[0]
        if self._allowed_kinds == (
            DigitRecognitionQuestionKind.RECALL,
            DigitRecognitionQuestionKind.COUNT_TARGET,
            DigitRecognitionQuestionKind.DIFFERENT_DIGIT,
        ):
            if self._index % 2 == 0:
                return DigitRecognitionQuestionKind.RECALL
            if (self._index // 2) % 2 == 0:
                return DigitRecognitionQuestionKind.COUNT_TARGET
            return DigitRecognitionQuestionKind.DIFFERENT_DIGIT
        return DigitRecognitionQuestionKind(str(self._rng.choice(self._allowed_kinds)))

    def _build_digit_string(self, *, length: int, difficulty: float) -> str:
        length = max(1, int(length))
        style = self._profile.string_profile
        if style == "friendly":
            return self._build_friendly_digit_string(length)
        if style == "noisy":
            return self._build_noisy_digit_string(length=length, difficulty=difficulty)
        return "".join(str(self._rng.randint(0, 9)) for _ in range(length))

    def _build_friendly_digit_string(self, length: int) -> str:
        pieces: list[str] = []
        while sum(len(piece) for piece in pieces) < length:
            family = str(self._rng.choice(("repeat", "pair", "step", "mirror")))
            remaining = length - sum(len(piece) for piece in pieces)
            if family == "repeat":
                digit = str(self._rng.randint(0, 9))
                piece = digit * min(remaining, int(self._rng.choice((2, 3))))
            elif family == "pair":
                a = str(self._rng.randint(0, 9))
                b = str(self._rng.randint(0, 9))
                piece = (a + b) * max(1, min(2, remaining // 2))
            elif family == "step":
                start = int(self._rng.randint(0, 7))
                width = min(remaining, int(self._rng.choice((2, 3))))
                piece = "".join(str((start + idx) % 10) for idx in range(width))
            else:
                left = str(self._rng.randint(0, 9))
                right = str(self._rng.randint(0, 9))
                piece = left + right + left
            pieces.append(piece[:remaining])
        return "".join(pieces)[:length]

    def _build_noisy_digit_string(self, *, length: int, difficulty: float) -> str:
        pool_size = 4 if difficulty < 0.5 else 3 if difficulty < 0.8 else 2
        pool = [str(self._rng.randint(0, 9)) for _ in range(pool_size)]
        digits: list[str] = []
        while len(digits) < length:
            if digits and self._rng.random() < (0.20 + (0.35 * difficulty)):
                digits.append(digits[-1])
                continue
            digits.append(str(self._rng.choice(pool)))
        return "".join(digits[:length])

    def _replacement_pool(
        self,
        *,
        digits: str,
        original_digit: str,
        difficulty: float,
    ) -> list[str]:
        if difficulty >= 0.7:
            preferred = [digit for digit in dict.fromkeys(digits) if digit != original_digit]
            if preferred:
                return preferred
        return [str(d) for d in range(10) if str(d) != original_digit]


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
        profile: DigitRecognitionProfile | None = None,
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
        self._profile = profile or DigitRecognitionProfile()

        self._gen = DigitRecognitionGenerator(SeededRng(int(seed)), profile=self._profile)

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
            lines = [" ".join(self._current.digits)]
            if self._current.comparison_digits is not None:
                lines.append(" ".join(self._current.comparison_digits))
            return DigitRecognitionPayload(
                display_digits=lines[0],
                display_lines=tuple(lines),
                accepting_input=False,
            )
        if self._stage is _Stage.MASK:
            return DigitRecognitionPayload(
                display_digits=None,
                display_lines=None,
                accepting_input=False,
            )
        if self._stage is _Stage.QUESTION:
            return DigitRecognitionPayload(
                display_digits=None,
                display_lines=None,
                accepting_input=True,
            )
        return None

    def _prompt_text(self) -> str:
        if self._phase is Phase.INSTRUCTIONS:
            return "\n".join(
                [
                    "Digit Recognition",
                    "",
                    "Remember strings of digits of varying lengths.",
                    "Some questions ask you to retype the full string.",
                    "Others ask for information about it, such as counts or",
                    "which digit changed between two similar strings.",
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
    profile = DigitRecognitionProfile()
    return DigitRecognitionTest(
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        practice_questions=(3 if practice else 0),
        scored_duration_s=float(scored_duration_s),
        display_s=profile.display_s_for(difficulty),
        mask_s=profile.mask_s_for(difficulty),
        profile=profile,
    )

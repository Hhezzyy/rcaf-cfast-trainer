from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .ant_drills import ANT_DRILL_MODE_PROFILES, AntDrillAttemptSummary, AntDrillMode
from .clock import Clock
from .cognitive_core import AnswerScorer, Phase, Problem, QuestionEvent, SeededRng, TestSnapshot
from .digit_recognition import (
    DigitRecognitionGenerator,
    DigitRecognitionPayload,
    DigitRecognitionProfile,
    DigitRecognitionQuestionKind,
    DigitRecognitionTrainingSpec,
)
from .lookup_retain import expected_digits_for_problem
from .runtime_ui_policy import runtime_visible_timers_enabled


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _difficulty_to_level(difficulty: float) -> int:
    return max(1, min(10, int(round(_clamp01(difficulty) * 9.0)) + 1))


@dataclass(frozen=True, slots=True)
class DigitRecognitionDrillConfig:
    practice_questions: int | None = None
    scored_duration_s: float | None = None


def _normalize_mode(mode: AntDrillMode | str) -> AntDrillMode:
    return mode if isinstance(mode, AntDrillMode) else AntDrillMode(str(mode).strip().lower())


def _format_digits(value: str, *, chunk: int = 1) -> str:
    if chunk <= 1:
        return " ".join(value)
    groups = [value[index : index + chunk] for index in range(0, len(value), chunk)]
    return "   ".join(groups)


def _format_interference(value: str, *, chunk: int = 2) -> str:
    groups = [value[index : index + chunk] for index in range(0, len(value), chunk)]
    return " ".join(groups)


def _question_cap(level: int, caps: tuple[float, ...]) -> float:
    clamped = max(1, min(len(caps), int(level)))
    return float(caps[clamped - 1])


class DigitRecognitionExactScorer(AnswerScorer):
    def score(self, *, problem: Problem, user_answer: int, raw: str) -> float:
        _ = user_answer
        digits = "".join(ch for ch in str(raw) if ch.isdigit())
        return 1.0 if digits == expected_digits_for_problem(problem) else 0.0


class _BaseDigitRecognitionProblemGenerator:
    def cap_for_problem(self, *, problem: Problem, level: int) -> float | None:
        _ = problem, level
        return None


class DrVisibleCopyGenerator(_BaseDigitRecognitionProblemGenerator):
    def __init__(self, *, seed: int) -> None:
        self._digits = DigitRecognitionGenerator(
            SeededRng(seed),
            profile=DigitRecognitionProfile(
                allowed_kinds=(DigitRecognitionQuestionKind.RECALL,),
                min_length_easy=5,
                min_length_hard=7,
                max_length_easy=7,
                max_length_hard=10,
                visible_supported=True,
                string_profile="friendly",
            ),
        )

    def next_problem(self, *, difficulty: float) -> Problem:
        digits = self._digits.next_digit_string(difficulty=difficulty)
        return Problem(
            prompt="Type the full string while it remains visible.",
            answer=int(digits),
            payload=DigitRecognitionTrainingSpec(
                kind=DigitRecognitionQuestionKind.RECALL,
                display_lines=(_format_digits(digits, chunk=2),),
                question_prompt="Type the full string while it remains visible.",
                expected_digits=digits,
                display_answer_text=digits,
                input_digits=max(4, len(digits)),
                answer_digits=len(digits),
                initial_display_s=0.0,
                mask_s=0.0,
                keep_display_visible_during_question=True,
            ),
        )


class DrPositionProbeGenerator(_BaseDigitRecognitionProblemGenerator):
    def __init__(self, *, seed: int) -> None:
        self._rng = SeededRng(seed)
        self._digits = DigitRecognitionGenerator(
            SeededRng(seed + 101),
            profile=DigitRecognitionProfile(
                allowed_kinds=(DigitRecognitionQuestionKind.RECALL,),
                min_length_easy=6,
                min_length_hard=8,
                max_length_easy=8,
                max_length_hard=11,
                visible_supported=True,
                string_profile="friendly",
            ),
        )

    def next_problem(self, *, difficulty: float) -> Problem:
        level = _difficulty_to_level(difficulty)
        digits = self._digits.next_digit_string(difficulty=difficulty)
        if level <= 5 or len(digits) <= 6:
            index = int(self._rng.randint(0, len(digits) - 1))
            expected = digits[index]
            prompt = f"Digits stay visible. Enter digit {index + 1} from the original string."
        else:
            width = 2 if level <= 8 else 3
            width = min(width, len(digits))
            start = int(self._rng.randint(0, len(digits) - width))
            expected = digits[start : start + width]
            prompt = (
                f"Digits stay visible. Enter digits {start + 1}-{start + width} "
                "from the original string."
            )
        return Problem(
            prompt=prompt,
            answer=int(expected),
            payload=DigitRecognitionTrainingSpec(
                kind=DigitRecognitionQuestionKind.RECALL,
                display_lines=(_format_digits(digits, chunk=2),),
                question_prompt=prompt,
                expected_digits=expected,
                display_answer_text=expected,
                input_digits=max(2, len(expected)),
                answer_digits=len(expected),
                initial_display_s=(1.8, 1.6, 1.4, 1.2, 1.0, 0.9, 0.8, 0.75, 0.7, 0.6)[level - 1],
                mask_s=0.0,
                keep_display_visible_during_question=True,
            ),
        )


class DrVisualDigitQueryGenerator(_BaseDigitRecognitionProblemGenerator):
    def __init__(self, *, seed: int) -> None:
        self._rng = SeededRng(seed)
        self._digits = DigitRecognitionGenerator(
            SeededRng(seed + 101),
            profile=DigitRecognitionProfile(
                allowed_kinds=(DigitRecognitionQuestionKind.RECALL,),
                min_length_easy=5,
                min_length_hard=9,
                max_length_easy=7,
                max_length_hard=12,
                display_s_easy=1.8,
                display_s_hard=0.7,
                mask_s_easy=0.35,
                mask_s_hard=0.90,
                string_profile="friendly",
            ),
        )

    def next_problem(self, *, difficulty: float) -> Problem:
        level = _difficulty_to_level(difficulty)
        digits = self._digits.next_digit_string(difficulty=difficulty)
        query_complexity = 1 if level <= 3 else 2 if level <= 6 else 3
        if query_complexity == 1:
            index = int(self._rng.randint(0, len(digits) - 1))
            expected = digits[index]
            prompt = f"Enter digit {index + 1} from the original string."
        elif query_complexity == 2:
            width = min(2, len(digits))
            start = int(self._rng.randint(0, len(digits) - width))
            expected = digits[start : start + width]
            prompt = f"Enter digits {start + 1}-{start + width} from the original string."
        else:
            count_target = str(self._rng.randint(0, 9))
            expected = str(sum(1 for ch in digits if ch == count_target))
            prompt = f"How many {count_target}s were in the original string?"
        interference_rate = 1 if level <= 3 else 2 if level <= 6 else 3
        interference_lines = tuple(
            _format_interference(self._digits.next_digit_string(difficulty=min(1.0, difficulty + 0.2)))
            for _ in range(interference_rate)
        )
        mask_s = (0.30, 0.36, 0.42, 0.52, 0.60, 0.68, 0.76, 0.84, 0.92, 1.00)[level - 1]
        display_s = (1.80, 1.65, 1.50, 1.35, 1.20, 1.05, 0.95, 0.85, 0.78, 0.70)[level - 1]
        return Problem(
            prompt=prompt,
            answer=int(expected),
            payload=DigitRecognitionTrainingSpec(
                kind=DigitRecognitionQuestionKind.RECALL,
                display_lines=(_format_digits(digits, chunk=2),),
                question_prompt=prompt,
                expected_digits=expected,
                display_answer_text=expected,
                input_digits=max(2, len(expected)),
                answer_digits=len(expected),
                initial_display_s=display_s,
                mask_s=mask_s,
                keep_display_visible_during_question=False,
                question_cap_s=(13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.5, 6.0, 5.5)[
                    level - 1
                ],
                mask_display_lines=interference_lines,
                mask_prompt="Ignore the interference stream.",
                family_tag="visual_digit_query",
                span_length=len(digits),
                delay_s=mask_s,
                query_complexity=query_complexity,
                interference_rate=interference_rate,
            ),
        )


class DrRecallAfterInterferenceGenerator(_BaseDigitRecognitionProblemGenerator):
    def __init__(self, *, seed: int) -> None:
        self._digits = DigitRecognitionGenerator(
            SeededRng(seed),
            profile=DigitRecognitionProfile(
                allowed_kinds=(DigitRecognitionQuestionKind.RECALL,),
                min_length_easy=5,
                min_length_hard=8,
                max_length_easy=7,
                max_length_hard=11,
                display_s_easy=1.6,
                display_s_hard=0.65,
                mask_s_easy=0.45,
                mask_s_hard=1.10,
                string_profile="default",
            ),
        )

    def next_problem(self, *, difficulty: float) -> Problem:
        level = _difficulty_to_level(difficulty)
        digits = self._digits.next_digit_string(difficulty=difficulty)
        interference_rate = 1 if level <= 3 else 2 if level <= 6 else 4
        interference_lines = tuple(
            _format_interference(self._digits.next_digit_string(difficulty=min(1.0, difficulty + 0.25)))
            for _ in range(interference_rate)
        )
        mask_s = (0.45, 0.52, 0.60, 0.70, 0.82, 0.92, 1.00, 1.10, 1.18, 1.25)[level - 1]
        display_s = (1.60, 1.50, 1.40, 1.28, 1.15, 1.00, 0.90, 0.80, 0.72, 0.65)[level - 1]
        return Problem(
            prompt="After the interference, enter the original digit string.",
            answer=int(digits),
            payload=DigitRecognitionTrainingSpec(
                kind=DigitRecognitionQuestionKind.RECALL,
                display_lines=(_format_digits(digits, chunk=2),),
                question_prompt="After the interference, enter the original digit string.",
                expected_digits=digits,
                display_answer_text=digits,
                input_digits=max(4, len(digits)),
                answer_digits=len(digits),
                initial_display_s=display_s,
                mask_s=mask_s,
                keep_display_visible_during_question=False,
                question_cap_s=(14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.5, 6.0)[
                    level - 1
                ],
                mask_display_lines=interference_lines,
                mask_prompt="Ignore the interference digits and hold the original string.",
                family_tag="recall_after_interference",
                span_length=len(digits),
                delay_s=mask_s,
                query_complexity=1,
                interference_rate=interference_rate,
            ),
        )


class DrVisibleFamilyPrimerGenerator(_BaseDigitRecognitionProblemGenerator):
    def __init__(self, *, seed: int) -> None:
        self._base = DigitRecognitionGenerator(
            SeededRng(seed),
            profile=DigitRecognitionProfile(
                allowed_kinds=(
                    DigitRecognitionQuestionKind.COUNT_TARGET,
                    DigitRecognitionQuestionKind.DIFFERENT_DIGIT,
                    DigitRecognitionQuestionKind.DIFFERENCE_COUNT,
                ),
                min_length_easy=6,
                min_length_hard=8,
                max_length_easy=8,
                max_length_hard=11,
                visible_supported=True,
                string_profile="friendly",
            ),
        )

    def next_problem(self, *, difficulty: float) -> Problem:
        trial = self._base.next_trial(difficulty=difficulty)
        lines = [_format_digits(trial.digits, chunk=2)]
        if trial.comparison_digits is not None:
            lines.append(_format_digits(trial.comparison_digits, chunk=2))
        prompt = f"Digits stay visible. {trial.prompt}"
        return Problem(
            prompt=prompt,
            answer=int(trial.expected),
            payload=DigitRecognitionTrainingSpec(
                kind=trial.kind,
                display_lines=tuple(lines),
                question_prompt=prompt,
                expected_digits=trial.expected,
                display_answer_text=trial.expected,
                input_digits=max(2, len(trial.expected)),
                answer_digits=len(trial.expected),
                initial_display_s=(1.6, 1.5, 1.4, 1.3, 1.1, 1.0, 0.9, 0.8, 0.75, 0.7)[
                    _difficulty_to_level(difficulty) - 1
                ],
                mask_s=0.0,
                keep_display_visible_during_question=True,
            ),
        )


class _HiddenMemoryFamilyGenerator(_BaseDigitRecognitionProblemGenerator):
    def __init__(
        self,
        *,
        seed: int,
        allowed_kinds: tuple[DigitRecognitionQuestionKind, ...],
        string_profile: str,
        display_easy: float,
        display_hard: float,
        mask_easy: float,
        mask_hard: float,
    ) -> None:
        self._profile = DigitRecognitionProfile(
            allowed_kinds=allowed_kinds,
            min_length_easy=5,
            min_length_hard=7,
            max_length_easy=8,
            max_length_hard=11,
            display_s_easy=display_easy,
            display_s_hard=display_hard,
            mask_s_easy=mask_easy,
            mask_s_hard=mask_hard,
            string_profile=string_profile,
        )
        self._base = DigitRecognitionGenerator(SeededRng(seed), profile=self._profile)

    def _problem_from_trial(self, *, trial, difficulty: float) -> Problem:
        lines = [_format_digits(trial.digits)]
        if trial.comparison_digits is not None:
            lines.append(_format_digits(trial.comparison_digits))
        return Problem(
            prompt=trial.prompt,
            answer=int(trial.expected),
            payload=DigitRecognitionTrainingSpec(
                kind=trial.kind,
                display_lines=tuple(lines),
                question_prompt=trial.prompt,
                expected_digits=trial.expected,
                display_answer_text=trial.expected,
                input_digits=max(2, len(trial.expected)),
                answer_digits=len(trial.expected),
                initial_display_s=self._profile.display_s_for(difficulty),
                mask_s=self._profile.mask_s_for(difficulty),
                keep_display_visible_during_question=False,
            ),
        )

    def next_problem(self, *, difficulty: float) -> Problem:
        return self._problem_from_trial(
            trial=self._base.next_trial(difficulty=difficulty),
            difficulty=difficulty,
        )


class DrRecallRunGenerator(_HiddenMemoryFamilyGenerator):
    def __init__(self, *, seed: int) -> None:
        super().__init__(
            seed=seed,
            allowed_kinds=(DigitRecognitionQuestionKind.RECALL,),
            string_profile="default",
            display_easy=1.6,
            display_hard=0.6,
            mask_easy=0.35,
            mask_hard=0.12,
        )


class DrCountTargetGenerator(_HiddenMemoryFamilyGenerator):
    def __init__(self, *, seed: int) -> None:
        super().__init__(
            seed=seed,
            allowed_kinds=(DigitRecognitionQuestionKind.COUNT_TARGET,),
            string_profile="noisy",
            display_easy=1.5,
            display_hard=0.55,
            mask_easy=0.30,
            mask_hard=0.10,
        )


class DrDifferentDigitGenerator(_HiddenMemoryFamilyGenerator):
    def __init__(self, *, seed: int) -> None:
        super().__init__(
            seed=seed,
            allowed_kinds=(DigitRecognitionQuestionKind.DIFFERENT_DIGIT,),
            string_profile="noisy",
            display_easy=1.6,
            display_hard=0.6,
            mask_easy=0.35,
            mask_hard=0.12,
        )


class DrDifferenceCountGenerator(_HiddenMemoryFamilyGenerator):
    def __init__(self, *, seed: int) -> None:
        super().__init__(
            seed=seed,
            allowed_kinds=(DigitRecognitionQuestionKind.DIFFERENCE_COUNT,),
            string_profile="noisy",
            display_easy=1.55,
            display_hard=0.58,
            mask_easy=0.32,
            mask_hard=0.12,
        )


class DrGroupedFamilyRunGenerator(_BaseDigitRecognitionProblemGenerator):
    _order = (
        DigitRecognitionQuestionKind.RECALL,
        DigitRecognitionQuestionKind.COUNT_TARGET,
        DigitRecognitionQuestionKind.DIFFERENT_DIGIT,
        DigitRecognitionQuestionKind.DIFFERENCE_COUNT,
    )

    def __init__(self, *, seed: int, group_size: int = 2) -> None:
        self._group_size = max(1, int(group_size))
        self._family_index = 0
        self._items_in_family = 0
        self._generators = {
            DigitRecognitionQuestionKind.RECALL: DrRecallRunGenerator(seed=seed + 101),
            DigitRecognitionQuestionKind.COUNT_TARGET: DrCountTargetGenerator(seed=seed + 202),
            DigitRecognitionQuestionKind.DIFFERENT_DIGIT: DrDifferentDigitGenerator(seed=seed + 303),
            DigitRecognitionQuestionKind.DIFFERENCE_COUNT: DrDifferenceCountGenerator(seed=seed + 404),
        }

    def next_problem(self, *, difficulty: float) -> Problem:
        family = self._order[self._family_index]
        problem = self._generators[family].next_problem(difficulty=difficulty)
        self._items_in_family += 1
        if self._items_in_family >= self._group_size:
            self._items_in_family = 0
            self._family_index = (self._family_index + 1) % len(self._order)
        return problem


class DrMixedPressureGenerator(_BaseDigitRecognitionProblemGenerator):
    def __init__(self, *, seed: int) -> None:
        self._rng = SeededRng(seed)
        self._last_kind: DigitRecognitionQuestionKind | None = None
        self._generators = {
            DigitRecognitionQuestionKind.RECALL: DrRecallRunGenerator(seed=seed + 101),
            DigitRecognitionQuestionKind.COUNT_TARGET: DrCountTargetGenerator(seed=seed + 202),
            DigitRecognitionQuestionKind.DIFFERENT_DIGIT: DrDifferentDigitGenerator(seed=seed + 303),
            DigitRecognitionQuestionKind.DIFFERENCE_COUNT: DrDifferenceCountGenerator(seed=seed + 404),
        }

    def next_problem(self, *, difficulty: float) -> Problem:
        families = list(self._generators)
        chosen = DigitRecognitionQuestionKind(str(self._rng.choice(families)))
        if self._last_kind is chosen and self._rng.random() < 0.5:
            alternates = [family for family in families if family is not chosen]
            chosen = DigitRecognitionQuestionKind(str(self._rng.choice(alternates)))
        self._last_kind = chosen
        local = min(1.0, float(difficulty) + 0.15)
        return self._generators[chosen].next_problem(difficulty=local)


class _DrStage(StrEnum):
    SHOW = "show"
    MASK = "mask"
    QUESTION = "question"


class DigitRecognitionTimedDrill:
    def __init__(
        self,
        *,
        title: str,
        instructions: tuple[str, ...],
        generator: _BaseDigitRecognitionProblemGenerator,
        clock: Clock,
        seed: int,
        difficulty: float,
        practice_questions: int,
        scored_duration_s: float,
        mode: AntDrillMode,
        base_question_caps_by_level: tuple[float, ...],
    ) -> None:
        self._title = str(title)
        self._instructions = tuple(instructions)
        self._generator = generator
        self._clock = clock
        self._seed = int(seed)
        self._difficulty = _clamp01(difficulty)
        self._practice_questions = max(0, int(practice_questions))
        self._scored_duration_s = max(1.0, float(scored_duration_s))
        self._mode = mode
        self._mode_profile = ANT_DRILL_MODE_PROFILES[mode]
        self._base_question_caps_by_level = tuple(float(cap) for cap in base_question_caps_by_level)
        self._scorer = DigitRecognitionExactScorer()

        self._phase = Phase.INSTRUCTIONS
        self._current: Problem | None = None
        self._stage: _DrStage | None = None
        self._stage_started_at_s: float | None = None
        self._question_presented_at_s: float | None = None
        self._question_cap_s: float | None = None
        self._practice_answered = 0
        self._scored_started_at_s: float | None = None
        self._scored_attempted = 0
        self._scored_correct = 0
        self._scored_timeouts = 0
        self._scored_total_score = 0.0
        self._scored_max_score = 0.0
        self._max_timeout_streak = 0
        self._timeout_streak = 0
        self._events: list[QuestionEvent] = []
        self._last_feedback = ""

    @property
    def phase(self) -> Phase:
        return self._phase

    @property
    def difficulty(self) -> float:
        return float(self._difficulty)

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def practice_questions(self) -> int:
        return self._practice_questions

    @property
    def scored_duration_s(self) -> float:
        return float(self._scored_duration_s)

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
        if self._practice_questions <= 0:
            self._phase = Phase.PRACTICE_DONE
            return
        self._phase = Phase.PRACTICE
        self._last_feedback = ""
        self._deal_new_problem()

    def start_scored(self) -> None:
        if self._phase in (Phase.SCORED, Phase.RESULTS):
            return
        self._phase = Phase.SCORED
        self._last_feedback = ""
        self._scored_started_at_s = self._clock.now()
        self._deal_new_problem()

    def update(self) -> None:
        if self._phase not in (Phase.PRACTICE, Phase.SCORED):
            return
        if self._phase is Phase.SCORED and self.time_remaining_s() == 0.0:
            self._finish_results()
            return
        if self._current is None or self._stage is None or self._stage_started_at_s is None:
            return
        spec = self._current.payload
        if not isinstance(spec, DigitRecognitionTrainingSpec):
            return
        now = self._clock.now()
        elapsed = now - self._stage_started_at_s
        if self._stage is _DrStage.SHOW and elapsed >= spec.initial_display_s:
            if spec.mask_s > 0.0:
                self._stage = _DrStage.MASK
                self._stage_started_at_s = now
                return
            self._stage = _DrStage.QUESTION
            self._stage_started_at_s = now
            self._question_presented_at_s = now
            return
        if self._stage is _DrStage.MASK and elapsed >= spec.mask_s:
            self._stage = _DrStage.QUESTION
            self._stage_started_at_s = now
            self._question_presented_at_s = now
            return
        if (
            self._stage is _DrStage.QUESTION
            and self._question_presented_at_s is not None
            and self._item_remaining_s() <= 0.0
        ):
            self._record_timeout()

    def submit_answer(self, raw: str) -> bool:
        if self._phase not in (Phase.PRACTICE, Phase.SCORED):
            return False
        token = str(raw).strip().lower()
        if token in {"__skip_practice__", "skip_practice"} and self._phase is Phase.PRACTICE:
            self._phase = Phase.PRACTICE_DONE
            self._reset_current()
            return True
        if token in {"__skip_section__", "skip_section", "__skip_all__", "skip_all"} and (
            self._phase is Phase.SCORED
        ):
            self._finish_results()
            return True
        if self._phase is Phase.SCORED and self.time_remaining_s() == 0.0:
            self._finish_results()
            return False
        if self._current is None or self._stage is not _DrStage.QUESTION:
            return False
        digits = "".join(ch for ch in str(raw) if ch.isdigit())
        if digits == "":
            return False
        answered_at_s = self._clock.now()
        presented_at_s = self._question_presented_at_s or answered_at_s
        response_time_s = max(0.0, answered_at_s - presented_at_s)
        score_value = self._scorer.score(problem=self._current, user_answer=int(digits), raw=digits)
        is_correct = score_value >= 0.999999
        self._events.append(
            QuestionEvent(
                index=len(self._events),
                phase=self._phase,
                prompt=str(self._current.prompt),
                correct_answer=int(self._current.answer),
                user_answer=int(digits),
                is_correct=is_correct,
                presented_at_s=presented_at_s,
                answered_at_s=answered_at_s,
                response_time_s=response_time_s,
                raw=digits,
                score=score_value,
                max_score=1.0,
            )
        )
        if self._phase is Phase.SCORED:
            self._scored_attempted += 1
            self._scored_total_score += float(score_value)
            self._scored_max_score += 1.0
            if is_correct:
                self._scored_correct += 1
                self._timeout_streak = 0
            else:
                self._timeout_streak = 0
        else:
            self._practice_answered += 1
        self._last_feedback = (
            "Correct. Commit and move."
            if is_correct
            else f"Incorrect. Correct answer: {self._display_answer(self._current)}"
        )
        if self._phase is Phase.SCORED and self.time_remaining_s() == 0.0:
            self._finish_results()
            return True
        if self._phase is Phase.PRACTICE and self._practice_answered >= self._practice_questions:
            self._phase = Phase.PRACTICE_DONE
            self._reset_current()
            return True
        self._deal_new_problem()
        return True

    def snapshot(self) -> TestSnapshot:
        return TestSnapshot(
            title=self._title,
            phase=self._phase,
            prompt=self.current_prompt(),
            input_hint=self._input_hint(),
            time_remaining_s=self.time_remaining_s(),
            attempted_scored=self._scored_attempted,
            correct_scored=self._scored_correct,
            payload=self._runtime_payload(),
            practice_feedback=self._last_feedback if self._last_feedback else None,
        )

    def current_prompt(self) -> str:
        if self._phase is Phase.INSTRUCTIONS:
            return "\n".join(self._instructions)
        if self._phase is Phase.PRACTICE_DONE:
            return "Practice complete. Press Enter to start the timed block."
        if self._phase is Phase.RESULTS:
            summary = self.scored_summary()
            return (
                "Results\n"
                f"Attempted: {summary.attempted}\n"
                f"Correct: {summary.correct}\n"
                f"Exact accuracy: {summary.accuracy * 100.0:.1f}%\n"
                f"Correct/min: {summary.correct_per_min:.1f}\n"
                f"Timeouts: {summary.timeouts} ({summary.fixation_rate * 100.0:.1f}%)\n"
                f"Max timeout streak: {summary.max_timeout_streak}\n"
                f"Score ratio: {summary.score_ratio * 100.0:.1f}%"
            )
        if self._current is None or not isinstance(self._current.payload, DigitRecognitionTrainingSpec):
            return ""
        if self._stage is _DrStage.SHOW:
            return "MEMORIZE"
        if self._stage is _DrStage.MASK:
            return self._current.payload.mask_prompt
        return self._current.payload.question_prompt

    def time_remaining_s(self) -> float | None:
        if self._phase is not Phase.SCORED:
            return None
        assert self._scored_started_at_s is not None
        return max(0.0, self._scored_duration_s - (self._clock.now() - self._scored_started_at_s))

    def scored_summary(self) -> AntDrillAttemptSummary:
        duration_s = float(self._scored_duration_s)
        attempted = int(self._scored_attempted)
        correct = int(self._scored_correct)
        accuracy = 0.0 if attempted == 0 else correct / attempted
        throughput = (attempted / duration_s) * 60.0 if duration_s > 0.0 else 0.0
        correct_per_min = (correct / duration_s) * 60.0 if duration_s > 0.0 else 0.0
        fixation_rate = 0.0 if attempted == 0 else self._scored_timeouts / attempted
        rts = [event.response_time_s for event in self._events if event.phase is Phase.SCORED]
        mean_rt = None if not rts else sum(rts) / len(rts)
        return AntDrillAttemptSummary(
            attempted=attempted,
            correct=correct,
            accuracy=float(accuracy),
            duration_s=duration_s,
            throughput_per_min=float(throughput),
            mean_response_time_s=mean_rt,
            total_score=float(self._scored_total_score),
            max_score=float(self._scored_max_score),
            score_ratio=(
                0.0
                if self._scored_max_score <= 0.0
                else float(self._scored_total_score) / float(self._scored_max_score)
            ),
            correct_per_min=float(correct_per_min),
            timeouts=int(self._scored_timeouts),
            fixation_rate=float(fixation_rate),
            max_timeout_streak=int(self._max_timeout_streak),
            mode=self._mode.value,
            difficulty_level=_difficulty_to_level(self._difficulty),
            difficulty_level_start=_difficulty_to_level(self._difficulty),
            difficulty_level_end=_difficulty_to_level(self._difficulty),
            difficulty_change_count=0,
            adaptive_enabled=False,
            adaptive_window_size=0,
        )

    def _input_hint(self) -> str:
        if self._phase not in (Phase.PRACTICE, Phase.SCORED):
            return "Press Enter to continue"
        if self._stage is not _DrStage.QUESTION or self._question_presented_at_s is None:
            return f"L{_difficulty_to_level(self._difficulty)} | Observe, then type digits"
        level = _difficulty_to_level(self._difficulty)
        if runtime_visible_timers_enabled():
            return (
                f"L{level} | "
                f"Cap {self._item_remaining_s():0.1f}s | Type digits then Enter"
            )
        return (
            f"L{level} | Type digits then Enter"
        )

    def _deal_new_problem(self) -> None:
        self._current = self._generator.next_problem(difficulty=self._difficulty)
        now = self._clock.now()
        self._stage_started_at_s = now
        spec = self._current.payload
        if not isinstance(spec, DigitRecognitionTrainingSpec):
            self._stage = _DrStage.QUESTION
            self._question_presented_at_s = now
            self._question_cap_s = self._resolve_question_cap(self._current)
            return
        self._question_cap_s = self._resolve_question_cap(self._current)
        if spec.initial_display_s <= 0.0 and spec.mask_s <= 0.0:
            self._stage = _DrStage.QUESTION
            self._question_presented_at_s = now
            return
        self._stage = _DrStage.SHOW
        self._question_presented_at_s = None

    def _resolve_question_cap(self, problem: Problem) -> float:
        level = _difficulty_to_level(self._difficulty)
        resolver = getattr(self._generator, "cap_for_problem", None)
        if callable(resolver):
            resolved = resolver(problem=problem, level=level)
            if resolved is not None:
                return max(2.0, min(60.0, float(resolved) * self._mode_profile.cap_scale))
        payload = problem.payload
        if isinstance(payload, DigitRecognitionTrainingSpec) and payload.question_cap_s is not None:
            return max(2.0, min(60.0, float(payload.question_cap_s) * self._mode_profile.cap_scale))
        return max(
            2.0,
            min(
                60.0,
                _question_cap(level, self._base_question_caps_by_level) * self._mode_profile.cap_scale,
            ),
        )

    def _item_remaining_s(self) -> float:
        if self._question_presented_at_s is None or self._question_cap_s is None:
            return 0.0
        return max(0.0, self._question_cap_s - (self._clock.now() - self._question_presented_at_s))

    def _record_timeout(self) -> None:
        if self._current is None or self._question_presented_at_s is None:
            return
        answered_at_s = self._clock.now()
        response_time_s = max(0.0, answered_at_s - self._question_presented_at_s)
        self._events.append(
            QuestionEvent(
                index=len(self._events),
                phase=self._phase,
                prompt=str(self._current.prompt),
                correct_answer=int(self._current.answer),
                user_answer=0,
                is_correct=False,
                presented_at_s=self._question_presented_at_s,
                answered_at_s=answered_at_s,
                response_time_s=response_time_s,
                raw="__timeout__",
                score=0.0,
                max_score=1.0,
                is_timeout=True,
            )
        )
        if self._phase is Phase.SCORED:
            self._scored_attempted += 1
            self._scored_timeouts += 1
            self._scored_max_score += 1.0
            self._timeout_streak += 1
            self._max_timeout_streak = max(self._max_timeout_streak, self._timeout_streak)
        else:
            self._practice_answered += 1
        self._last_feedback = f"Timeout. Correct answer: {self._display_answer(self._current)}"
        if self._phase is Phase.SCORED and self.time_remaining_s() == 0.0:
            self._finish_results()
            return
        if self._phase is Phase.PRACTICE and self._practice_answered >= self._practice_questions:
            self._phase = Phase.PRACTICE_DONE
            self._reset_current()
            return
        self._deal_new_problem()

    def _runtime_payload(self) -> DigitRecognitionPayload | None:
        if self._phase not in (Phase.PRACTICE, Phase.SCORED) or self._current is None:
            return None
        spec = self._current.payload
        if not isinstance(spec, DigitRecognitionTrainingSpec):
            return None
        if self._stage is _DrStage.SHOW:
            return DigitRecognitionPayload(
                display_digits=spec.display_lines[0] if spec.display_lines else None,
                display_lines=spec.display_lines,
                accepting_input=False,
                prompt_text="",
                input_digits=spec.input_digits,
                family=spec.family_tag or spec.kind.value,
            )
        if self._stage is _DrStage.MASK:
            return DigitRecognitionPayload(
                display_digits=spec.mask_display_lines[0] if spec.mask_display_lines else None,
                display_lines=spec.mask_display_lines or None,
                accepting_input=False,
                prompt_text=spec.mask_prompt,
                input_digits=spec.input_digits,
                family=spec.family_tag or spec.kind.value,
            )
        display_lines = spec.display_lines if spec.keep_display_visible_during_question else None
        return DigitRecognitionPayload(
            display_digits=None if not display_lines else display_lines[0],
            display_lines=display_lines,
            accepting_input=True,
            prompt_text=spec.question_prompt,
            input_digits=spec.input_digits,
            family=spec.family_tag or spec.kind.value,
        )

    def _display_answer(self, problem: Problem) -> str:
        payload = problem.payload
        if isinstance(payload, DigitRecognitionTrainingSpec):
            return payload.display_answer_text
        return expected_digits_for_problem(problem)

    def _finish_results(self) -> None:
        self._phase = Phase.RESULTS
        self._reset_current()

    def _reset_current(self) -> None:
        self._current = None
        self._stage = None
        self._stage_started_at_s = None
        self._question_presented_at_s = None
        self._question_cap_s = None


def _build_drill(
    *,
    title_base: str,
    instructions: tuple[str, ...],
    generator: _BaseDigitRecognitionProblemGenerator,
    clock: Clock,
    seed: int,
    difficulty: float,
    mode: AntDrillMode | str,
    config: DigitRecognitionDrillConfig,
    base_question_caps_by_level: tuple[float, ...],
) -> DigitRecognitionTimedDrill:
    normalized_mode = _normalize_mode(mode)
    profile = ANT_DRILL_MODE_PROFILES[normalized_mode]
    practice_questions = (
        profile.practice_questions if config.practice_questions is None else int(config.practice_questions)
    )
    scored_duration_s = (
        profile.scored_duration_s if config.scored_duration_s is None else float(config.scored_duration_s)
    )
    return DigitRecognitionTimedDrill(
        title=f"{title_base} ({profile.label})",
        instructions=instructions,
        generator=generator,
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        practice_questions=practice_questions,
        scored_duration_s=scored_duration_s,
        mode=normalized_mode,
        base_question_caps_by_level=base_question_caps_by_level,
    )


def build_dr_visible_copy_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: DigitRecognitionDrillConfig | None = None,
) -> DigitRecognitionTimedDrill:
    cfg = config or DigitRecognitionDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    return _build_drill(
        title_base="Digit Recognition: Visible Copy",
        instructions=(
            "Digit Recognition: Visible Copy",
            f"Mode: {profile.label}",
            "Type the full digit string while it remains visible so the encoding rhythm becomes automatic.",
            "Use this as the clean warm-up before hidden-memory pressure starts.",
            "Press Enter to begin practice.",
        ),
        generator=DrVisibleCopyGenerator(seed=seed),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
        base_question_caps_by_level=(14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.5, 6.0),
    )


def build_dr_position_probe_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: DigitRecognitionDrillConfig | None = None,
) -> DigitRecognitionTimedDrill:
    cfg = config or DigitRecognitionDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    return _build_drill(
        title_base="Digit Recognition: Position Probe",
        instructions=(
            "Digit Recognition: Position Probe",
            f"Mode: {profile.label}",
            "Scan the string left-to-right, then answer a position or short-slice prompt without losing the order.",
            "The digits stay visible when the prompt appears so this block can teach serial anchors before hidden recall.",
            "Press Enter to begin practice.",
        ),
        generator=DrPositionProbeGenerator(seed=seed),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
        base_question_caps_by_level=(15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.5),
    )


def build_dr_visible_family_primer_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: DigitRecognitionDrillConfig | None = None,
) -> DigitRecognitionTimedDrill:
    cfg = config or DigitRecognitionDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    return _build_drill(
        title_base="Digit Recognition: Visible Family Primer",
        instructions=(
            "Digit Recognition: Visible Family Primer",
            f"Mode: {profile.label}",
            "Touch the count-target, different-digit, and difference-count families while the strings are still visible.",
            "Use this to learn the family prompts before the hidden-memory blocks remove the support.",
            "Press Enter to begin practice.",
        ),
        generator=DrVisibleFamilyPrimerGenerator(seed=seed),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
        base_question_caps_by_level=(14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.5, 6.0),
    )


def build_dr_visual_digit_query_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: DigitRecognitionDrillConfig | None = None,
) -> DigitRecognitionTimedDrill:
    cfg = config or DigitRecognitionDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    return _build_drill(
        title_base="Digit Recognition: Visual Digit Query",
        instructions=(
            "Digit Recognition: Visual Digit Query",
            f"Mode: {profile.label}",
            "Memorize the string, ignore the interference stream, then answer a specific digit query from memory.",
            "Span length, delay, query complexity, and interference density all rise with level.",
            "Press Enter to begin practice.",
        ),
        generator=DrVisualDigitQueryGenerator(seed=seed),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
        base_question_caps_by_level=(13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.5, 6.0, 5.5),
    )


def build_dr_recall_run_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: DigitRecognitionDrillConfig | None = None,
) -> DigitRecognitionTimedDrill:
    cfg = config or DigitRecognitionDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    return _build_drill(
        title_base="Digit Recognition: Recall Run",
        instructions=(
            "Digit Recognition: Recall Run",
            f"Mode: {profile.label}",
            "Train exact hidden-memory full-string recall under shorter display and mask windows.",
            "The next-item banner shows the prior exact string without interrupting your rhythm.",
            "Press Enter to begin practice.",
        ),
        generator=DrRecallRunGenerator(seed=seed),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
        base_question_caps_by_level=(12.0, 11.0, 10.0, 9.5, 9.0, 8.0, 7.0, 6.0, 5.5, 5.0),
    )


def build_dr_recall_after_interference_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.TEMPO,
    config: DigitRecognitionDrillConfig | None = None,
) -> DigitRecognitionTimedDrill:
    cfg = config or DigitRecognitionDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    return _build_drill(
        title_base="Digit Recognition: Recall After Interference",
        instructions=(
            "Digit Recognition: Recall After Interference",
            f"Mode: {profile.label}",
            "Hold the original string through an explicit interference burst, then type the full recall exactly.",
            "Later levels lengthen the string, extend the delay, and raise the interference rate before the answer window opens.",
            "Press Enter to begin practice.",
        ),
        generator=DrRecallAfterInterferenceGenerator(seed=seed),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
        base_question_caps_by_level=(14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.5, 6.0),
    )


def build_dr_count_target_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.TEMPO,
    config: DigitRecognitionDrillConfig | None = None,
) -> DigitRecognitionTimedDrill:
    cfg = config or DigitRecognitionDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    return _build_drill(
        title_base="Digit Recognition: Count Target",
        instructions=(
            "Digit Recognition: Count Target",
            f"Mode: {profile.label}",
            "Count one target digit after the string disappears and move on fast when repeated digits get dense.",
            "This block trains count precision without leaving the typed Digit Recognition answer mode.",
            "Press Enter to begin practice.",
        ),
        generator=DrCountTargetGenerator(seed=seed),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
        base_question_caps_by_level=(11.0, 10.0, 9.5, 9.0, 8.5, 8.0, 7.0, 6.0, 5.5, 5.0),
    )


def build_dr_different_digit_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.TEMPO,
    config: DigitRecognitionDrillConfig | None = None,
) -> DigitRecognitionTimedDrill:
    cfg = config or DigitRecognitionDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    return _build_drill(
        title_base="Digit Recognition: Different Digit",
        instructions=(
            "Digit Recognition: Different Digit",
            f"Mode: {profile.label}",
            "Spot the one changed digit after the two strings disappear and keep your pace when the strings get more similar.",
            "Treat misses as resets, not as a reason to slow down.",
            "Press Enter to begin practice.",
        ),
        generator=DrDifferentDigitGenerator(seed=seed),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
        base_question_caps_by_level=(12.0, 11.0, 10.0, 9.5, 9.0, 8.0, 7.0, 6.5, 6.0, 5.5),
    )


def build_dr_difference_count_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.TEMPO,
    config: DigitRecognitionDrillConfig | None = None,
) -> DigitRecognitionTimedDrill:
    cfg = config or DigitRecognitionDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    return _build_drill(
        title_base="Digit Recognition: Difference Count",
        instructions=(
            "Digit Recognition: Difference Count",
            f"Mode: {profile.label}",
            "Compare the remembered strings and answer how many positions changed before the display fades from working memory.",
            "Use a quick left-to-right tally instead of re-solving the whole string from scratch.",
            "Press Enter to begin practice.",
        ),
        generator=DrDifferenceCountGenerator(seed=seed),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
        base_question_caps_by_level=(12.0, 11.0, 10.5, 10.0, 9.0, 8.5, 7.5, 6.5, 6.0, 5.5),
    )


def build_dr_grouped_family_run_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.TEMPO,
    config: DigitRecognitionDrillConfig | None = None,
) -> DigitRecognitionTimedDrill:
    cfg = config or DigitRecognitionDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    return _build_drill(
        title_base="Digit Recognition: Grouped Family Run",
        instructions=(
            "Digit Recognition: Grouped Family Run",
            f"Mode: {profile.label}",
            "Hidden-memory family chunks arrive in a fixed order: recall, count target, different digit, then difference count.",
            "Use the grouped order to stabilize each family before the late mixed block.",
            "Press Enter to begin practice.",
        ),
        generator=DrGroupedFamilyRunGenerator(seed=seed, group_size=2),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
        base_question_caps_by_level=(10.0, 9.5, 9.0, 8.5, 8.0, 7.0, 6.5, 6.0, 5.5, 5.0),
    )


def build_dr_mixed_pressure_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.STRESS,
    config: DigitRecognitionDrillConfig | None = None,
) -> DigitRecognitionTimedDrill:
    cfg = config or DigitRecognitionDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    return _build_drill(
        title_base="Digit Recognition: Mixed Pressure",
        instructions=(
            "Digit Recognition: Mixed Pressure",
            f"Mode: {profile.label}",
            "All four hidden-memory families rotate under the shortest display, mask, and answer timings in this library.",
            "Recover immediately after misses and let the next-item banner reset you.",
            "Press Enter to begin practice.",
        ),
        generator=DrMixedPressureGenerator(seed=seed),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
        base_question_caps_by_level=(9.0, 8.5, 8.0, 7.5, 7.0, 6.5, 6.0, 5.5, 5.0, 4.5),
    )

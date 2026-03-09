from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .clock import Clock
from .cognitive_core import Phase, SeededRng, TestSnapshot, clamp01, lerp_int


@dataclass(frozen=True, slots=True)
class VigilanceConfig:
    # Candidate guide reports ~8 minutes including instructions.
    practice_duration_s: float = 45.0
    scored_duration_s: float = 6.0 * 60.0
    spawn_interval_s: float = 0.85
    max_active_symbols: int = 7


class VigilanceSymbolKind(StrEnum):
    STAR = "star"
    DIAMOND = "diamond"
    TRIANGLE = "triangle"
    HEXAGON = "hexagon"


_SYMBOL_POINTS: dict[VigilanceSymbolKind, int] = {
    VigilanceSymbolKind.STAR: 1,
    VigilanceSymbolKind.DIAMOND: 2,
    VigilanceSymbolKind.TRIANGLE: 3,
    VigilanceSymbolKind.HEXAGON: 4,
}


@dataclass(frozen=True, slots=True)
class VigilanceSymbol:
    symbol_id: int
    row: int
    col: int
    kind: VigilanceSymbolKind
    points: int
    time_left_s: float


@dataclass(frozen=True, slots=True)
class VigilancePayload:
    rows: int
    cols: int
    symbols: tuple[VigilanceSymbol, ...]
    legend: tuple[tuple[VigilanceSymbolKind, int], ...]
    points_total: int
    captured_total: int
    missed_total: int


@dataclass(frozen=True, slots=True)
class VigilanceSummary:
    attempted: int
    correct: int
    accuracy: float
    duration_s: float
    throughput_per_min: float
    points: int
    points_per_min: float
    missed: int
    mean_capture_time_s: float | None


@dataclass(slots=True)
class _ActiveSymbol:
    symbol_id: int
    row: int
    col: int
    kind: VigilanceSymbolKind
    points: int
    spawned_at_s: float
    expires_at_s: float


class VigilanceEngine:
    """Continuous vigilance matrix: transient symbols with weighted point values."""

    _rows = 9
    _cols = 9
    _legend: tuple[tuple[VigilanceSymbolKind, int], ...] = tuple(
        (kind, points) for kind, points in _SYMBOL_POINTS.items()
    )

    def __init__(
        self,
        *,
        clock: Clock,
        seed: int,
        difficulty: float = 0.5,
        config: VigilanceConfig | None = None,
    ) -> None:
        cfg = config or VigilanceConfig()
        if cfg.practice_duration_s < 0.0:
            raise ValueError("practice_duration_s must be >= 0")
        if cfg.scored_duration_s <= 0.0:
            raise ValueError("scored_duration_s must be > 0")
        if cfg.spawn_interval_s <= 0.0:
            raise ValueError("spawn_interval_s must be > 0")
        if cfg.max_active_symbols <= 0:
            raise ValueError("max_active_symbols must be > 0")

        self._clock = clock
        self._seed = int(seed)
        self._difficulty = clamp01(difficulty)
        self._config = cfg
        self._rng = SeededRng(seed=self._seed)

        self._phase = Phase.INSTRUCTIONS
        self._phase_started_at_s = self._clock.now()

        self._active: list[_ActiveSymbol] = []
        self._next_symbol_id = 1
        self._next_spawn_at_s = 0.0

        self._practice_points = 0

        self._scored_attempted = 0
        self._scored_captured = 0
        self._scored_points = 0
        self._scored_missed = 0
        self._scored_capture_times: list[float] = []

    @property
    def phase(self) -> Phase:
        return self._phase

    def can_exit(self) -> bool:
        return self._phase in (
            Phase.INSTRUCTIONS,
            Phase.PRACTICE,
            Phase.PRACTICE_DONE,
            Phase.RESULTS,
        )

    def start_practice(self) -> None:
        if self._phase is not Phase.INSTRUCTIONS:
            return
        if self._config.practice_duration_s <= 0.0:
            self._phase = Phase.PRACTICE_DONE
            self._phase_started_at_s = self._clock.now()
            return

        self._phase = Phase.PRACTICE
        self._reset_phase_stream(blank_delay_s=0.90)

    def start_scored(self) -> None:
        if self._phase in (Phase.SCORED, Phase.RESULTS):
            return
        self._phase = Phase.SCORED
        self._phase_started_at_s = self._clock.now()
        self._active.clear()
        self._next_spawn_at_s = self._phase_started_at_s + 0.90

        self._scored_attempted = 0
        self._scored_captured = 0
        self._scored_points = 0
        self._scored_missed = 0
        self._scored_capture_times = []

    def submit_answer(self, raw: str) -> bool:
        if self._phase not in (Phase.PRACTICE, Phase.SCORED):
            return False

        row_col = self._parse_row_col(raw)
        if row_col is None:
            return False
        row, col = row_col

        now = self._clock.now()
        self._advance_stream(now)

        if self._phase is Phase.SCORED:
            self._scored_attempted += 1

        match_index: int | None = None
        for idx, symbol in enumerate(self._active):
            if symbol.row == row and symbol.col == col and symbol.expires_at_s > now:
                match_index = idx
                break

        if match_index is None:
            return True

        symbol = self._active.pop(match_index)
        if self._phase is Phase.PRACTICE:
            self._practice_points += int(symbol.points)
            return True

        self._scored_captured += 1
        self._scored_points += int(symbol.points)
        self._scored_capture_times.append(max(0.0, now - symbol.spawned_at_s))
        return True

    def update(self) -> None:
        now = self._clock.now()
        self._advance_stream(now)
        self._refresh_phase_boundaries(now)

    def time_remaining_s(self) -> float | None:
        now = self._clock.now()
        if self._phase is Phase.PRACTICE:
            rem = self._config.practice_duration_s - (now - self._phase_started_at_s)
            return max(0.0, rem)
        if self._phase is Phase.SCORED:
            rem = self._config.scored_duration_s - (now - self._phase_started_at_s)
            return max(0.0, rem)
        return None

    def snapshot(self) -> TestSnapshot:
        now = self._clock.now()
        payload = self._build_payload(now=now)
        return TestSnapshot(
            title="Vigilance",
            phase=self._phase,
            prompt=self.current_prompt(),
            input_hint="Select Row/Col field, then type digits 1-9",
            time_remaining_s=self.time_remaining_s(),
            attempted_scored=self._scored_attempted,
            correct_scored=self._scored_captured,
            payload=payload,
            practice_feedback=None,
        )

    def current_prompt(self) -> str:
        if self._phase is Phase.INSTRUCTIONS:
            return (
                "Vigilance Test\n"
                "The matrix starts blank. Symbols appear briefly, then disappear.\n"
                "Most symbols are stars (1 point). Rarer symbols are worth more points.\n"
                "Capture a symbol by entering its row and column while it is still visible.\n"
                "Select Row/Col input using mouse click or arrow keys.\n"
                "Press Enter to begin practice."
            )
        if self._phase is Phase.PRACTICE_DONE:
            return "Practice complete. Press Enter to start the timed test."
        if self._phase is Phase.RESULTS:
            summary = self.scored_summary()
            accuracy_pct = int(round(summary.accuracy * 100.0))
            mean_capture = (
                "n/a"
                if summary.mean_capture_time_s is None
                else f"{summary.mean_capture_time_s:.2f}s"
            )
            return (
                "Results\n"
                f"Points: {summary.points}\n"
                f"Captures: {summary.correct}/{summary.attempted} ({accuracy_pct}%)\n"
                f"Missed (expired): {summary.missed}\n"
                f"Mean capture time: {mean_capture}\n"
                f"Points per minute: {summary.points_per_min:.1f}"
            )
        return "Capture symbols by entering row and column before they disappear."

    def scored_summary(self) -> VigilanceSummary:
        duration_s = float(self._config.scored_duration_s)
        attempted = int(self._scored_attempted)
        correct = int(self._scored_captured)
        accuracy = 0.0 if attempted == 0 else correct / attempted
        throughput_per_min = (attempted / duration_s) * 60.0
        points = int(self._scored_points)
        points_per_min = (points / duration_s) * 60.0
        mean_capture_time_s = None
        if self._scored_capture_times:
            mean_capture_time_s = sum(self._scored_capture_times) / len(self._scored_capture_times)

        return VigilanceSummary(
            attempted=attempted,
            correct=correct,
            accuracy=accuracy,
            duration_s=duration_s,
            throughput_per_min=throughput_per_min,
            points=points,
            points_per_min=points_per_min,
            missed=int(self._scored_missed),
            mean_capture_time_s=mean_capture_time_s,
        )

    def _reset_phase_stream(self, *, blank_delay_s: float) -> None:
        self._phase_started_at_s = self._clock.now()
        self._active.clear()
        self._next_spawn_at_s = self._phase_started_at_s + max(0.0, float(blank_delay_s))
        self._practice_points = 0

    def _refresh_phase_boundaries(self, now: float) -> None:
        if self._phase is Phase.PRACTICE:
            if (now - self._phase_started_at_s) >= self._config.practice_duration_s:
                self._phase = Phase.PRACTICE_DONE
                self._phase_started_at_s = now
                self._active.clear()
        elif self._phase is Phase.SCORED:
            if (now - self._phase_started_at_s) >= self._config.scored_duration_s:
                self._phase = Phase.RESULTS
                self._phase_started_at_s = now
                self._active.clear()

    def _advance_stream(self, now: float) -> None:
        if self._phase not in (Phase.PRACTICE, Phase.SCORED):
            return

        max_active = self._max_active_symbols()
        spawn_loops = 0
        while self._next_spawn_at_s > 0.0 and self._next_spawn_at_s <= now and spawn_loops < 12:
            if len(self._active) < max_active:
                self._spawn_symbol(spawned_at_s=self._next_spawn_at_s)
            self._next_spawn_at_s += self._next_spawn_interval_s()
            spawn_loops += 1

        survivors: list[_ActiveSymbol] = []
        for symbol in self._active:
            if symbol.expires_at_s <= now:
                if self._phase is Phase.SCORED:
                    self._scored_missed += 1
                continue
            survivors.append(symbol)
        self._active = survivors

    def _max_active_symbols(self) -> int:
        base = lerp_int(4, int(self._config.max_active_symbols), self._difficulty)
        return max(1, int(base))

    def _next_spawn_interval_s(self) -> float:
        d = clamp01(self._difficulty)
        baseline = max(0.25, float(self._config.spawn_interval_s))
        target = max(0.25, baseline - (0.24 * d))
        jitter = self._rng.uniform(-0.12, 0.12)
        return max(0.20, target + jitter)

    def _spawn_symbol(self, *, spawned_at_s: float) -> None:
        occupied = {(sym.row, sym.col) for sym in self._active}
        available: list[tuple[int, int]] = []
        for row in range(1, self._rows + 1):
            for col in range(1, self._cols + 1):
                if (row, col) not in occupied:
                    available.append((row, col))
        if not available:
            return

        row, col = self._rng.choice(tuple(available))
        kind = self._pick_symbol_kind()
        points = int(_SYMBOL_POINTS[kind])
        lifetime = self._symbol_lifetime_s(kind=kind)

        self._active.append(
            _ActiveSymbol(
                symbol_id=self._next_symbol_id,
                row=int(row),
                col=int(col),
                kind=kind,
                points=points,
                spawned_at_s=spawned_at_s,
                expires_at_s=spawned_at_s + lifetime,
            )
        )
        self._next_symbol_id += 1

    def _pick_symbol_kind(self) -> VigilanceSymbolKind:
        roll = self._rng.random()
        if roll < 0.72:
            return VigilanceSymbolKind.STAR
        if roll < 0.88:
            return VigilanceSymbolKind.DIAMOND
        if roll < 0.96:
            return VigilanceSymbolKind.TRIANGLE
        return VigilanceSymbolKind.HEXAGON

    def _symbol_lifetime_s(self, *, kind: VigilanceSymbolKind) -> float:
        base = {
            VigilanceSymbolKind.STAR: 3.5,
            VigilanceSymbolKind.DIAMOND: 4.0,
            VigilanceSymbolKind.TRIANGLE: 4.4,
            VigilanceSymbolKind.HEXAGON: 4.9,
        }[kind]
        penalty = 1.1 * clamp01(self._difficulty)
        jitter = self._rng.uniform(-0.25, 0.25)
        return max(1.4, base - penalty + jitter)

    def _build_payload(self, *, now: float) -> VigilancePayload:
        symbols = tuple(
            VigilanceSymbol(
                symbol_id=symbol.symbol_id,
                row=symbol.row,
                col=symbol.col,
                kind=symbol.kind,
                points=symbol.points,
                time_left_s=max(0.0, symbol.expires_at_s - now),
            )
            for symbol in sorted(self._active, key=lambda sym: (sym.row, sym.col, sym.symbol_id))
            if symbol.expires_at_s > now
        )

        captured_total = self._scored_captured if self._phase is Phase.SCORED else 0
        missed_total = self._scored_missed if self._phase is Phase.SCORED else 0
        points_total = self._scored_points if self._phase is Phase.SCORED else self._practice_points

        return VigilancePayload(
            rows=self._rows,
            cols=self._cols,
            symbols=symbols,
            legend=self._legend,
            points_total=int(points_total),
            captured_total=int(captured_total),
            missed_total=int(missed_total),
        )

    def _parse_row_col(self, raw: str) -> tuple[int, int] | None:
        token = str(raw).strip()
        if token == "":
            return None

        row = -1
        col = -1

        for sep in (",", " ", "/", ":", ";"):
            if sep in token:
                parts = [part for part in token.replace(" ", "").split(sep) if part != ""]
                if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                    row = int(parts[0])
                    col = int(parts[1])
                    break

        if row < 0 or col < 0:
            digits = [ch for ch in token if ch.isdigit()]
            if len(digits) >= 2:
                row = int(digits[0])
                col = int(digits[1])

        if not (1 <= row <= self._rows and 1 <= col <= self._cols):
            return None
        return int(row), int(col)


def build_vigilance_test(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    config: VigilanceConfig | None = None,
) -> VigilanceEngine:
    return VigilanceEngine(
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        config=config,
    )

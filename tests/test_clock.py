from __future__ import annotations

from cfast_trainer.clock import PausableClock


class FakeClock:
    def __init__(self) -> None:
        self._now = 0.0

    def now(self) -> float:
        return self._now

    def advance(self, delta_s: float) -> None:
        self._now += float(delta_s)


def test_pausable_clock_freezes_elapsed_time_until_resumed() -> None:
    base = FakeClock()
    clock = PausableClock(base)

    assert clock.now() == 0.0

    base.advance(3.0)
    assert clock.now() == 3.0

    clock.pause()
    base.advance(7.0)
    assert clock.now() == 3.0
    assert clock.is_paused() is True

    clock.resume()
    assert clock.is_paused() is False
    assert clock.now() == 3.0

    base.advance(2.0)
    assert clock.now() == 5.0

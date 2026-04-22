from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from importlib.machinery import ModuleSpec
from types import ModuleType

if "moderngl" not in sys.modules:
    moderngl_stub = ModuleType("moderngl")
    moderngl_stub.__spec__ = ModuleSpec("moderngl", loader=None)
    sys.modules["moderngl"] = moderngl_stub

from cfast_trainer.cognitive_core import Phase
from cfast_trainer.dr_drills import build_dr_visible_copy_drill
from cfast_trainer.ic_drills import build_ic_heading_anchor_drill
from cfast_trainer.sl_drills import build_sl_quantitative_anchor_drill
from cfast_trainer.tbl_drills import build_tbl_part1_anchor_drill
from cfast_trainer.tr_drills import build_tr_scene_anchor_drill


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += float(dt)


def _assert_hint_hides_cap_countdown(engine: object) -> None:
    snap = engine.snapshot()
    assert snap.phase is Phase.SCORED
    assert snap.time_remaining_s is not None
    assert "Cap " not in snap.input_hint
    assert not re.search(r"\b\d+(?:\.\d)?s\b", snap.input_hint)


def test_runtime_drill_hints_hide_cap_countdowns_but_keep_internal_caps() -> None:
    cases = (
        (build_tbl_part1_anchor_drill, "_current_cap_s"),
        (build_sl_quantitative_anchor_drill, "_current_cap_s"),
        (build_ic_heading_anchor_drill, "_current_cap_s"),
        (build_tr_scene_anchor_drill, "_current_cap_s"),
        (build_dr_visible_copy_drill, "_question_cap_s"),
    )

    for index, (builder, cap_attr) in enumerate(cases):
        engine = builder(clock=FakeClock(), seed=400 + index, difficulty=0.5)
        engine.start_scored()

        _assert_hint_hides_cap_countdown(engine)
        assert float(getattr(engine, cap_attr)) > 0.0
        assert float(engine._item_remaining_s()) > 0.0

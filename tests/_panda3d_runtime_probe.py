from __future__ import annotations

import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame

from cfast_trainer.app import App, CognitiveTestScreen, MenuItem, MenuScreen
from cfast_trainer.auditory_capacity import build_auditory_capacity_test
from cfast_trainer.auditory_capacity_panda3d import panda3d_auditory_rendering_available
from cfast_trainer.rapid_tracking import build_rapid_tracking_test
from cfast_trainer.rapid_tracking_panda3d import panda3d_rapid_tracking_rendering_available
from cfast_trainer.spatial_integration import build_spatial_integration_test
from cfast_trainer.spatial_integration_panda3d import (
    panda3d_spatial_integration_rendering_available,
)
from cfast_trainer.trace_test_1 import build_trace_test_1_test
from cfast_trainer.trace_test_1_panda3d import panda3d_trace_test_1_rendering_available
from cfast_trainer.trace_test_2 import build_trace_test_2_test
from cfast_trainer.trace_test_2_panda3d import panda3d_trace_test_2_rendering_available


class _FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _panda_available(scene_name: str) -> bool:
    token = str(scene_name).strip().lower()
    return {
        "auditory": panda3d_auditory_rendering_available,
        "rapid_tracking": panda3d_rapid_tracking_rendering_available,
        "spatial_integration": panda3d_spatial_integration_rendering_available,
        "trace_test_1": panda3d_trace_test_1_rendering_available,
        "trace_test_2": panda3d_trace_test_2_rendering_available,
    }[token]()


def _force_panda_preferences() -> None:
    os.environ["CFAST_AUDITORY_RENDERER"] = "panda"
    os.environ["CFAST_RAPID_TRACKING_RENDERER"] = "panda"
    os.environ["CFAST_SPATIAL_INTEGRATION_RENDERER"] = "panda"
    os.environ["CFAST_TRACE_TEST_1_RENDERER"] = "panda"
    os.environ["CFAST_TRACE_TEST_2_RENDERER"] = "panda"


def _build_app(
    scene_name: str,
    *,
    surface: pygame.Surface,
    rapid_tracking_seed: int = 551,
) -> tuple[App, CognitiveTestScreen, str]:
    font = pygame.font.Font(None, 36)
    app = App(surface=surface, font=font, opengl_enabled=True)
    app.push(MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True))
    token = str(scene_name).strip().lower()

    if token == "auditory":
        clock = _FakeClock()
        screen = CognitiveTestScreen(
            app,
            engine_factory=lambda: build_auditory_capacity_test(
                clock=clock,
                seed=17,
                difficulty=0.58,
            ),
            test_code="auditory_capacity",
        )
        app.push(screen)
        screen._engine.start_practice()
        clock.advance(0.2)
        screen._engine.update()
        return app, screen, token

    if token == "rapid_tracking":
        clock = _FakeClock()
        screen = CognitiveTestScreen(
            app,
            engine_factory=lambda: build_rapid_tracking_test(
                clock=clock,
                seed=int(rapid_tracking_seed),
                difficulty=0.63,
            ),
            test_code="rapid_tracking",
        )
        app.push(screen)
        screen._engine.start_scored()
        clock.advance(0.6)
        screen._engine.update()
        return app, screen, token

    if token == "spatial_integration":
        clock = _FakeClock()
        screen = CognitiveTestScreen(
            app,
            engine_factory=lambda: build_spatial_integration_test(
                clock=clock,
                seed=91,
                difficulty=0.55,
            ),
            test_code="spatial_integration",
        )
        app.push(screen)
        screen._engine.start_practice()
        clock.advance(0.1)
        screen._engine.update()
        return app, screen, token

    if token == "trace_test_1":
        clock = _FakeClock()
        screen = CognitiveTestScreen(
            app,
            engine_factory=lambda: build_trace_test_1_test(
                clock=clock,
                seed=37,
                difficulty=0.6,
            ),
            test_code="trace_test_1",
        )
        app.push(screen)
        screen._engine.start_practice()
        clock.advance(0.1)
        screen._engine.update()
        return app, screen, token

    if token == "trace_test_2":
        clock = _FakeClock()
        screen = CognitiveTestScreen(
            app,
            engine_factory=lambda: build_trace_test_2_test(
                clock=clock,
                seed=71,
                difficulty=0.58,
            ),
            test_code="trace_test_2",
        )
        app.push(screen)
        screen._engine.start_practice()
        clock.advance(0.1)
        screen._engine.update()
        return app, screen, token

    raise ValueError(f"unknown scene: {scene_name}")


def _rapid_tracking_layout_signature(renderer) -> dict[str, object]:
    layout = getattr(renderer, "_layout", None)
    if layout is None:
        return {}

    def _anchors(items, limit: int = 4) -> list[list[object]]:
        return [
            [anchor.anchor_id, anchor.variant, round(float(anchor.x), 3), round(float(anchor.y), 3)]
            for anchor in tuple(items)[:limit]
        ]

    def _clusters(items, limit: int = 3) -> list[list[object]]:
        return [
            [
                cluster.cluster_id,
                cluster.asset_id,
                round(float(cluster.x), 3),
                round(float(cluster.y), 3),
                int(cluster.count),
                round(float(cluster.scale), 3),
            ]
            for cluster in tuple(items)[:limit]
        ]

    return {
        "seed": int(layout.seed),
        "orbit_direction": int(layout.orbit_direction),
        "orbit_phase_offset": round(float(layout.orbit_phase_offset), 4),
        "orbit_radius_scale": round(float(layout.orbit_radius_scale), 4),
        "altitude_bias": round(float(layout.altitude_bias), 4),
        "ridge_phase": round(float(layout.ridge_phase), 4),
        "ridge_amp_scale": round(float(layout.ridge_amp_scale), 4),
        "path_lateral_bias": round(float(layout.path_lateral_bias), 4),
        "compound_center_x": round(float(layout.compound_center_x), 4),
        "compound_center_y": round(float(layout.compound_center_y), 4),
        "building_anchors": _anchors(layout.building_anchors),
        "patrol_anchors": _anchors(layout.patrol_anchors),
        "road_anchors": _anchors(layout.road_anchors),
        "ridge_anchors": _anchors(layout.ridge_anchors),
        "shrub_clusters": _clusters(layout.shrub_clusters),
        "tree_clusters": _clusters(layout.tree_clusters),
        "forest_clusters": _clusters(layout.forest_clusters),
    }


def _scene_summary(
    *,
    scene_name: str,
    screen: CognitiveTestScreen,
    app: App,
) -> dict[str, object]:
    token = str(scene_name).strip().lower()
    avg = pygame.transform.average_color(app.surface)
    summary: dict[str, object] = {
        "kind": token,
        "avg_color": [int(avg[0]), int(avg[1]), int(avg[2])],
        "gl_scene_type": None,
    }

    if token == "auditory":
        renderer = screen._auditory_panda_renderer
        if renderer is None:
            raise AssertionError("auditory Panda renderer was not created")
        payload = screen._engine.snapshot().payload
        summary.update(
            renderer_type=type(renderer).__name__,
            renderer_size=list(renderer.size),
            has_payload=payload is not None,
            gate_count=len(getattr(renderer, "_gate_nodes", {})),
            loaded_asset_ids=list(getattr(renderer, "loaded_asset_ids", ())),
            fallback_asset_ids=list(getattr(renderer, "fallback_asset_ids", ())),
        )
        return summary

    if token == "rapid_tracking":
        renderer = screen._rapid_tracking_panda_renderer
        if renderer is None:
            raise AssertionError("rapid-tracking Panda renderer was not created")
        payload = screen._engine.snapshot().payload
        overlay = renderer.target_overlay_state()
        summary.update(
            renderer_type=type(renderer).__name__,
            renderer_size=list(renderer.size),
            target_kind=None if payload is None else payload.target_kind,
            session_seed=None if payload is None else int(payload.session_seed),
            target_cover_state=None if payload is None else str(payload.target_cover_state),
            overlay_on_screen=None if overlay is None else bool(overlay.on_screen),
            overlay_in_front=None if overlay is None else bool(overlay.in_front),
            camera_heading_deg=None
            if renderer.last_camera_rig_state() is None
            else float(renderer.last_camera_rig_state().heading_deg),
            camera_pitch_deg=None
            if renderer.last_camera_rig_state() is None
            else float(renderer.last_camera_rig_state().pitch_deg),
            road_segment_count=len(getattr(renderer, "_road_segment_nodes", ())),
            road_intersection_count=len(getattr(renderer, "_road_intersection_nodes", ())),
            layout_signature=_rapid_tracking_layout_signature(renderer),
        )
        return summary

    if token == "spatial_integration":
        renderer = screen._spatial_integration_panda_renderer
        if renderer is None:
            raise AssertionError("spatial-integration Panda renderer was not created")
        summary.update(
            renderer_type=type(renderer).__name__,
            renderer_size=list(renderer.size),
            landmark_count=len(getattr(renderer, "_landmark_nodes", ())),
        )
        return summary

    if token == "trace_test_1":
        renderer = screen._trace_test_1_panda_renderer
        if renderer is None:
            raise AssertionError("trace-test-1 Panda renderer was not created")
        red_hpr = [round(float(value), 3) for value in renderer._red_anchor.getHpr()]
        blue_hprs = [
            [round(float(value), 3) for value in anchor.getHpr()]
            for anchor in getattr(renderer, "_blue_anchors", ())
        ]
        summary.update(
            renderer_type=type(renderer).__name__,
            renderer_size=list(renderer.size),
            aircraft_count=1 + len(blue_hprs),
            blue_count=len(blue_hprs),
            red_hpr=red_hpr,
            blue_hpr_count=len(blue_hprs),
        )
        return summary

    if token == "trace_test_2":
        renderer = screen._trace_test_2_panda_renderer
        if renderer is None:
            raise AssertionError("trace-test-2 Panda renderer was not created")
        summary.update(
            renderer_type=type(renderer).__name__,
            renderer_size=list(renderer.size),
            aircraft_count=len(getattr(renderer, "_aircraft_nodes", {})),
            orientation_count=len(getattr(renderer, "_aircraft_orientation_by_code", {})),
        )
        return summary

    raise ValueError(f"unknown scene: {scene_name}")


def run_scene_probe(scene_name: str, *, rapid_tracking_seed: int = 551) -> dict[str, object]:
    _force_panda_preferences()
    if not _panda_available(scene_name):
        print(f"SKIP:Panda3D unavailable for {scene_name}", flush=True)
        raise SystemExit(77)

    pygame.init()
    screen: CognitiveTestScreen | None = None
    try:
        try:
            display_surface = pygame.display.set_mode((960, 540), pygame.HIDDEN)
        except Exception:
            display_surface = pygame.display.set_mode((960, 540))

        app, screen, token = _build_app(
            scene_name,
            surface=display_surface,
            rapid_tracking_seed=rapid_tracking_seed,
        )
        app.render()
        gl_scene = app.consume_gl_scene()
        if gl_scene is not None:
            raise AssertionError(f"expected Panda path, queued {type(gl_scene).__name__}")
        result = _scene_summary(scene_name=token, screen=screen, app=app)
        result["gl_scene_type"] = None
        return result
    finally:
        if screen is not None:
            close = getattr(screen, "close", None)
            if callable(close):
                close()
        pygame.quit()


def main(argv: list[str]) -> int:
    if len(argv) not in (2, 3):
        raise SystemExit("usage: _panda3d_runtime_probe.py <scene-name> [seed]")
    seed = 551 if len(argv) == 2 else int(argv[2])
    result = run_scene_probe(argv[1], rapid_tracking_seed=seed)
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

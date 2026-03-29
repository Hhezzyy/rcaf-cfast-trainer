from __future__ import annotations

import pygame
import pytest

from cfast_trainer.app import App, CognitiveTestScreen, MenuItem, MenuScreen
from cfast_trainer.aircraft_art import fixed_wing_heading_from_screen_heading
from cfast_trainer.trace_test_2 import (
    TraceTest2AircraftTrack,
    TraceTest2Generator,
    TraceTest2MotionKind,
    TraceTest2Payload,
    TraceTest2Point3,
    build_trace_test_2_test,
    trace_test_2_track_tangent,
)
from cfast_trainer.trace_test_2_gl import project_point, screen_heading_deg
from cfast_trainer.trace_test_2_panda3d import (
    TraceTest2Panda3DRenderer,
    panda3d_trace_test_2_rendering_available,
)


class _FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _angle_delta_deg(current: float, reference: float) -> float:
    return ((float(current) - float(reference) + 180.0) % 360.0) - 180.0


def _turn_track(*, motion_kind: TraceTest2MotionKind) -> TraceTest2AircraftTrack:
    end_x = -34.0 if motion_kind is TraceTest2MotionKind.LEFT else 34.0
    return TraceTest2AircraftTrack(
        code=1,
        color_name="RED",
        color_rgb=(220, 64, 72),
        waypoints=(
            TraceTest2Point3(0.0, 38.0, 9.0),
            TraceTest2Point3(0.0, 92.0, 9.0),
            TraceTest2Point3(end_x, 142.0, 9.0),
        ),
        motion_kind=motion_kind,
        direction_changed=True,
        ended_screen_x=end_x,
        ended_altitude_z=9.0,
    )


def _run_probe() -> dict[str, object]:
    pygame.init()
    screen: CognitiveTestScreen | None = None
    try:
        surface = pygame.display.set_mode((960, 540))
        font = pygame.font.Font(None, 36)
        app = App(surface=surface, font=font, opengl_enabled=True)
        app.push(MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True))
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
        app.render()
        scene = app.consume_gl_scene()
        assert scene is not None
        payload = scene.payload
        return {
            "scene_type": type(scene).__name__,
            "correct_code": int(payload.correct_code),
            "aircraft_count": len(payload.aircraft),
            "variant_id": str(payload.variant_id),
            "content_family": str(payload.content_family),
        }
    finally:
        if screen is not None:
            close = getattr(screen, "close", None)
            if callable(close):
                close()
        pygame.quit()


def _project_world_point(
    renderer: TraceTest2Panda3DRenderer,
    *,
    point: TraceTest2Point3,
) -> tuple[float, float]:
    from panda3d.core import Point2, Point3

    relative = renderer._base.cam.getRelativePoint(
        renderer._base.render,
        Point3(point.x, point.y, point.z),
    )
    projected = Point2()
    assert renderer._base.camLens.project(relative, projected)
    return (
        (float(projected.x) + 1.0) * 0.5 * renderer.size[0],
        (1.0 - float(projected.y)) * 0.5 * renderer.size[1],
    )


def test_panda3d_trace_test_2_rendering_disabled_for_dummy_video(monkeypatch) -> None:
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.delenv("CFAST_TRACE_TEST_2_RENDERER", raising=False)

    assert panda3d_trace_test_2_rendering_available() is False


def test_panda3d_trace_test_2_rendering_ignores_non_panda_preference(monkeypatch) -> None:
    monkeypatch.setenv("CFAST_TRACE_TEST_2_RENDERER", "pygame")
    monkeypatch.delenv("SDL_VIDEODRIVER", raising=False)
    monkeypatch.setattr(
        "cfast_trainer.trace_test_2_panda3d.importlib.util.find_spec",
        lambda _name: object(),
    )

    assert panda3d_trace_test_2_rendering_available() is True


def test_trace_test_2_aircraft_hpr_points_nose_along_motion_tangent() -> None:
    east = TraceTest2Panda3DRenderer._aircraft_hpr_from_tangent(tangent=(1.0, 0.0, 0.0))
    north = TraceTest2Panda3DRenderer._aircraft_hpr_from_tangent(tangent=(0.0, 1.0, 0.0))
    climb = TraceTest2Panda3DRenderer._aircraft_hpr_from_tangent(tangent=(0.0, 1.0, 1.0))

    assert east[0] == pytest.approx(90.0)
    assert north[0] == pytest.approx(0.0)
    assert climb[1] < 0.0


def test_trace_test_2_aircraft_hpr_rejects_zero_motion() -> None:
    with pytest.raises(ValueError, match="non-zero"):
        TraceTest2Panda3DRenderer._aircraft_hpr_from_tangent(tangent=(0.0, 0.0, 0.0))


def test_trace_test_2_panda_hpr_uses_world_tangent_not_screen_heading() -> None:
    track = _turn_track(motion_kind=TraceTest2MotionKind.RIGHT)
    progress = 0.5
    size = (960, 540)
    tangent = trace_test_2_track_tangent(track=track, progress=progress)
    world_hpr = TraceTest2Panda3DRenderer._aircraft_hpr_from_tangent(tangent=tangent)
    hpr = TraceTest2Panda3DRenderer._aircraft_hpr_for_track(
        track=track,
        progress=progress,
        size=size,
        tangent=tangent,
    )
    expected_heading = fixed_wing_heading_from_screen_heading(
        screen_heading_deg(track=track, progress=progress, size=size)
    )

    assert hpr == pytest.approx(world_hpr)
    assert abs(hpr[0] - expected_heading) >= 1.0
    assert hpr[2] == pytest.approx(world_hpr[2])


def test_trace_test_2_panda_left_turn_headings_change_monotonically_without_cardinal_snap() -> None:
    track = _turn_track(motion_kind=TraceTest2MotionKind.LEFT)
    headings = []
    for progress in (0.25, 0.40, 0.55, 0.70):
        tangent = trace_test_2_track_tangent(track=track, progress=progress)
        headings.append(
            TraceTest2Panda3DRenderer._aircraft_hpr_for_track(
                track=track,
                progress=progress,
                size=(960, 540),
                tangent=tangent,
            )[0]
        )
    deltas = [_angle_delta_deg(heading, 0.0) for heading in headings]

    assert all(later < earlier for earlier, later in zip(deltas, deltas[1:], strict=False))
    assert all(3.0 <= abs(later - earlier) <= 40.0 for earlier, later in zip(headings, headings[1:], strict=False))
    assert all(min(abs((heading % 90.0)), abs((heading % 90.0) - 90.0)) > 2.0 for heading in headings)


def test_trace_test_2_panda_right_turn_headings_change_monotonically_without_cardinal_snap() -> None:
    track = _turn_track(motion_kind=TraceTest2MotionKind.RIGHT)
    headings = []
    for progress in (0.25, 0.40, 0.55, 0.70):
        tangent = trace_test_2_track_tangent(track=track, progress=progress)
        headings.append(
            TraceTest2Panda3DRenderer._aircraft_hpr_for_track(
                track=track,
                progress=progress,
                size=(960, 540),
                tangent=tangent,
            )[0]
        )
    deltas = [_angle_delta_deg(heading, 0.0) for heading in headings]

    assert all(later > earlier for earlier, later in zip(deltas, deltas[1:], strict=False))
    assert all(3.0 <= abs(later - earlier) <= 40.0 for earlier, later in zip(headings, headings[1:], strict=False))
    assert all(min(abs((heading % 90.0)), abs((heading % 90.0) - 90.0)) > 2.0 for heading in headings)


def test_trace_test_2_panda_pitch_sign_changes_for_climb_vs_descent() -> None:
    climb_track = TraceTest2AircraftTrack(
        code=1,
        color_name="RED",
        color_rgb=(220, 64, 72),
        waypoints=(TraceTest2Point3(0.0, 38.0, 9.0), TraceTest2Point3(0.0, 138.0, 29.0)),
        motion_kind=TraceTest2MotionKind.CLIMB,
        direction_changed=True,
        ended_screen_x=0.0,
        ended_altitude_z=29.0,
    )
    descent_track = TraceTest2AircraftTrack(
        code=2,
        color_name="BLUE",
        color_rgb=(72, 128, 224),
        waypoints=(TraceTest2Point3(0.0, 38.0, 29.0), TraceTest2Point3(0.0, 138.0, 9.0)),
        motion_kind=TraceTest2MotionKind.STRAIGHT,
        direction_changed=True,
        ended_screen_x=0.0,
        ended_altitude_z=9.0,
    )

    climb_tangent = trace_test_2_track_tangent(track=climb_track, progress=0.5)
    descent_tangent = trace_test_2_track_tangent(track=descent_track, progress=0.5)
    climb_hpr = TraceTest2Panda3DRenderer._aircraft_hpr_for_track(
        track=climb_track,
        progress=0.5,
        size=(960, 540),
        tangent=climb_tangent,
    )
    descent_hpr = TraceTest2Panda3DRenderer._aircraft_hpr_for_track(
        track=descent_track,
        progress=0.5,
        size=(960, 540),
        tangent=descent_tangent,
    )

    assert climb_hpr[0] == pytest.approx(0.0)
    assert descent_hpr[0] == pytest.approx(0.0)
    assert climb_hpr[1] < 0.0
    assert descent_hpr[1] > 0.0


def test_trace_test_2_panda_multi_aircraft_orientation_matches_lattice_motion_kind() -> None:
    payload = TraceTest2Generator(seed=71).next_problem(difficulty=0.58).payload

    assert isinstance(payload, TraceTest2Payload)

    for track in payload.aircraft:
        tangent = trace_test_2_track_tangent(track=track, progress=0.5)
        hpr = TraceTest2Panda3DRenderer._aircraft_hpr_for_track(
            track=track,
            progress=0.5,
            size=(960, 540),
            tangent=tangent,
        )

        assert hpr == pytest.approx(
            TraceTest2Panda3DRenderer._aircraft_hpr_from_tangent(tangent=tangent)
        )
        assert hpr[2] == pytest.approx(0.0, abs=0.01)
        if track.motion_kind is TraceTest2MotionKind.LEFT:
            assert hpr[0] == pytest.approx(270.0)
        elif track.motion_kind is TraceTest2MotionKind.RIGHT:
            assert hpr[0] == pytest.approx(90.0)
        elif track.motion_kind is TraceTest2MotionKind.CLIMB:
            assert hpr[1] < 0.0
        elif track.motion_kind is TraceTest2MotionKind.DESCEND:
            assert hpr[1] > 0.0
        else:
            assert abs(hpr[1]) <= 0.01


def test_trace_test_2_panda_hpr_sampling_is_deterministic_for_same_track() -> None:
    track = _turn_track(motion_kind=TraceTest2MotionKind.RIGHT)
    progresses = (0.12, 0.37, 0.63, 0.88)

    samples_a = tuple(
        TraceTest2Panda3DRenderer._aircraft_hpr_for_track(
            track=track,
            progress=progress,
            size=(960, 540),
            tangent=trace_test_2_track_tangent(track=track, progress=progress),
        )
        for progress in progresses
    )
    samples_b = tuple(
        TraceTest2Panda3DRenderer._aircraft_hpr_for_track(
            track=track,
            progress=progress,
            size=(960, 540),
            tangent=trace_test_2_track_tangent(track=track, progress=progress),
        )
        for progress in progresses
    )

    assert samples_a == samples_b


def test_trace_test_2_panda_hpr_sampling_is_deterministic_for_same_seed() -> None:
    progresses = (0.18, 0.42, 0.66, 0.84)
    payload_a = TraceTest2Generator(seed=71).next_problem(difficulty=0.58).payload
    payload_b = TraceTest2Generator(seed=71).next_problem(difficulty=0.58).payload

    assert isinstance(payload_a, TraceTest2Payload)
    assert isinstance(payload_b, TraceTest2Payload)
    assert payload_a.aircraft == payload_b.aircraft

    samples_a = tuple(
        tuple(
            TraceTest2Panda3DRenderer._aircraft_hpr_for_track(
                track=track,
                progress=progress,
                size=(960, 540),
                tangent=trace_test_2_track_tangent(track=track, progress=progress),
            )
            for track in payload_a.aircraft
        )
        for progress in progresses
    )
    samples_b = tuple(
        tuple(
            TraceTest2Panda3DRenderer._aircraft_hpr_for_track(
                track=track,
                progress=progress,
                size=(960, 540),
                tangent=trace_test_2_track_tangent(track=track, progress=progress),
            )
            for track in payload_b.aircraft
        )
        for progress in progresses
    )

    assert samples_a == samples_b


def test_trace_test_2_camera_matches_gl_centering_and_ordering_contract() -> None:
    if not panda3d_trace_test_2_rendering_available():
        pytest.skip("Panda3D unavailable")

    pygame.init()
    renderer = TraceTest2Panda3DRenderer(size=(960, 540))
    try:
        center = TraceTest2Point3(0.0, 96.0, 9.0)
        left = TraceTest2Point3(-30.0, 96.0, 9.0)
        right = TraceTest2Point3(30.0, 96.0, 9.0)
        low = TraceTest2Point3(0.0, 96.0, 0.0)
        high = TraceTest2Point3(0.0, 96.0, 20.0)

        panda_center = _project_world_point(renderer, point=center)
        panda_left = _project_world_point(renderer, point=left)
        panda_right = _project_world_point(renderer, point=right)
        panda_low = _project_world_point(renderer, point=low)
        panda_high = _project_world_point(renderer, point=high)

        gl_center = project_point(center, size=(960, 540))
        gl_left = project_point(left, size=(960, 540))
        gl_right = project_point(right, size=(960, 540))
        gl_low = project_point(low, size=(960, 540))
        gl_high = project_point(high, size=(960, 540))

        assert abs(panda_center[0] - (renderer.size[0] * 0.5)) <= 24.0
        assert abs(gl_center[0] - 480.0) <= 0.01
        assert panda_left[0] < panda_center[0] < panda_right[0]
        assert gl_left[0] < gl_center[0] < gl_right[0]
        assert panda_high[1] < panda_low[1]
        assert gl_high[1] < gl_low[1]
    finally:
        renderer.close()
        pygame.quit()


def test_trace_test_2_screen_queues_modern_gl_runtime() -> None:
    probe = _run_probe()

    assert probe["scene_type"] == "TraceTest2GlScene"
    assert probe["correct_code"] == 4
    assert probe["aircraft_count"] >= 2
    assert probe["variant_id"]
    assert probe["content_family"] == "motion_memory"

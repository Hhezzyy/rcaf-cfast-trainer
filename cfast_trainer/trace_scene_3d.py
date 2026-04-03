from __future__ import annotations

import math
from dataclasses import dataclass

from .rapid_tracking_view import camera_space_to_viewport, world_to_camera_space
from .trace_test_1 import (
    TraceTest1Command,
    TraceTest1Payload,
    TraceTest1PromptPlan,
    TraceTest1SceneFrame,
    trace_test_1_aircraft_hpr,
    trace_test_1_scene_frames,
)
from .trace_test_2 import (
    TraceTest2AircraftTrack,
    TraceTest2Payload,
    TraceTest2Point3,
    trace_test_2_track_position,
    trace_test_2_track_tangent,
)

Point3 = tuple[float, float, float]

_TT1_CAMERA_FOV = (48.0, 38.0)
_TT2_CAMERA_FOV = (52.0, 40.0)
_TT2_TRAIL_LOOKBACK = 0.42
_TT2_TRAIL_PROGRESS_STEPS = (0.0, 0.5, 0.8)
_TT2_TRACK_SAMPLE_PROGRESS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


@dataclass(frozen=True, slots=True)
class TraceCameraPose:
    position: Point3
    target: Point3
    heading_deg: float
    pitch_deg: float
    h_fov_deg: float
    v_fov_deg: float
    near_clip: float = 0.12
    far_clip: float = 1200.0


@dataclass(frozen=True, slots=True)
class TraceTrailPoint:
    position: Point3
    hpr_deg: Point3
    alpha: float


@dataclass(frozen=True, slots=True)
class TraceAircraftPose:
    code: int
    asset_id: str
    position: Point3
    hpr_deg: Point3
    scale: Point3 = (1.0, 1.0, 1.0)
    color_rgba: tuple[float, float, float, float] | None = None


@dataclass(frozen=True, slots=True)
class TraceScene3dSnapshot:
    camera: TraceCameraPose
    aircraft: tuple[TraceAircraftPose, ...]
    ghosts: tuple[TraceAircraftPose, ...] = ()


def _vec_add(left: Point3, right: Point3) -> Point3:
    return (
        float(left[0] + right[0]),
        float(left[1] + right[1]),
        float(left[2] + right[2]),
    )


def _vec_sub(left: Point3, right: Point3) -> Point3:
    return (
        float(left[0] - right[0]),
        float(left[1] - right[1]),
        float(left[2] - right[2]),
    )


def _vec_scale(vector: Point3, scalar: float) -> Point3:
    return (
        float(vector[0] * scalar),
        float(vector[1] * scalar),
        float(vector[2] * scalar),
    )


def _vec_length(vector: Point3) -> float:
    return math.sqrt(
        (float(vector[0]) * float(vector[0]))
        + (float(vector[1]) * float(vector[1]))
        + (float(vector[2]) * float(vector[2]))
    )


def _vec_normalize(vector: Point3) -> Point3:
    length = _vec_length(vector)
    if length <= 1e-6:
        return (0.0, 0.0, 0.0)
    inv = 1.0 / length
    return (
        float(vector[0] * inv),
        float(vector[1] * inv),
        float(vector[2] * inv),
    )


def _wrap_heading_deg(dx: float, dy: float) -> float:
    return (math.degrees(math.atan2(float(dx), float(dy))) + 360.0) % 360.0


def _pitch_deg(dx: float, dy: float, dz: float) -> float:
    horizontal = max(1e-6, math.hypot(float(dx), float(dy)))
    return math.degrees(math.atan2(float(dz), horizontal))


def _camera_pose(
    *,
    position: Point3,
    target: Point3,
    h_fov_deg: float,
    v_fov_deg: float,
    near_clip: float = 0.12,
    far_clip: float = 1200.0,
) -> TraceCameraPose:
    delta = _vec_sub(target, position)
    return TraceCameraPose(
        position=position,
        target=target,
        heading_deg=_wrap_heading_deg(delta[0], delta[1]),
        pitch_deg=_pitch_deg(delta[0], delta[1], delta[2]),
        h_fov_deg=float(h_fov_deg),
        v_fov_deg=float(v_fov_deg),
        near_clip=float(near_clip),
        far_clip=float(far_clip),
    )


def _camera_space_vector(camera: TraceCameraPose, vector: Point3) -> Point3:
    origin = camera.position
    target = _vec_add(origin, vector)
    cam_x, cam_y, cam_z = world_to_camera_space(
        cam_world_x=float(origin[0]),
        cam_world_y=float(origin[1]),
        cam_world_z=float(origin[2]),
        heading_deg=float(camera.heading_deg),
        pitch_deg=float(camera.pitch_deg),
        target_world_x=float(target[0]),
        target_world_y=float(target[1]),
        target_world_z=float(target[2]),
    )
    return (float(cam_x), float(cam_y), float(cam_z))


def _project_world_point(
    *,
    camera: TraceCameraPose,
    point: Point3,
    size: tuple[int, int] = (960, 540),
) -> tuple[float, float, bool]:
    cam_x, cam_y, cam_z = world_to_camera_space(
        cam_world_x=float(camera.position[0]),
        cam_world_y=float(camera.position[1]),
        cam_world_z=float(camera.position[2]),
        heading_deg=float(camera.heading_deg),
        pitch_deg=float(camera.pitch_deg),
        target_world_x=float(point[0]),
        target_world_y=float(point[1]),
        target_world_z=float(point[2]),
    )
    screen_x, screen_y, _on_screen, in_front = camera_space_to_viewport(
        cam_x=float(cam_x),
        cam_y=float(cam_y),
        cam_z=float(cam_z),
        size=size,
        h_fov_deg=float(camera.h_fov_deg),
        v_fov_deg=float(camera.v_fov_deg),
    )
    return (float(screen_x), float(screen_y), bool(in_front))


def _frame_projected_forward_delta(
    *,
    frame: TraceTest1SceneFrame,
    camera: TraceCameraPose,
) -> tuple[float, float]:
    forward = _vec_normalize(
        (
            float(frame.world_forward[0]),
            float(frame.world_forward[1]),
            float(frame.world_forward[2]),
        )
    )
    if _vec_length(forward) <= 1e-6:
        return (0.0, 0.0)
    offset = _vec_scale(forward, 8.0)
    tail = _vec_sub(tuple(float(value) for value in frame.position), offset)
    nose = _vec_add(tuple(float(value) for value in frame.position), offset)
    tail_screen = _project_world_point(camera=camera, point=tail)
    nose_screen = _project_world_point(camera=camera, point=nose)
    if not tail_screen[2] or not nose_screen[2]:
        return (0.0, 0.0)
    return (
        float(nose_screen[0] - tail_screen[0]),
        float(nose_screen[1] - tail_screen[1]),
    )


def _trace_test_1_camera(
    *,
    points: tuple[Point3, ...],
    viewpoint_bearing_deg: int,
    reference_forward: Point3 | None = None,
    reference_up: Point3 | None = None,
) -> TraceCameraPose:
    min_x = min(point[0] for point in points)
    max_x = max(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_y = max(point[1] for point in points)
    min_z = min(point[2] for point in points)
    max_z = max(point[2] for point in points)
    center = (
        float((min_x + max_x) * 0.5),
        float((min_y + max_y) * 0.5),
        float((min_z + max_z) * 0.5),
    )
    span_x = max(12.0, float(max_x - min_x))
    span_y = max(18.0, float(max_y - min_y))
    span_z = max(8.0, float(max_z - min_z))
    distance = max(52.0, (span_x * 1.55) + (span_y * 0.72))
    height = max(13.5, (span_z * 1.8) + 6.0)
    forward = _vec_normalize(reference_forward or (0.0, 1.0, 0.0))
    if _vec_length(forward) <= 1e-6:
        forward = (0.0, 1.0, 0.0)
    up = _vec_normalize(reference_up or (0.0, 0.0, 1.0))
    if _vec_length(up) <= 1e-6:
        up = (0.0, 0.0, 1.0)
    right = _vec_normalize(
        (
            (forward[1] * up[2]) - (forward[2] * up[1]),
            (forward[2] * up[0]) - (forward[0] * up[2]),
            (forward[0] * up[1]) - (forward[1] * up[0]),
        )
    )
    if _vec_length(right) <= 1e-6:
        right = (1.0, 0.0, 0.0)
    bearing_rad = math.radians(int(viewpoint_bearing_deg) - 180)
    observer_dir = _vec_normalize(
        _vec_add(
            _vec_scale(_vec_scale(forward, -1.0), math.cos(bearing_rad)),
            _vec_scale(right, math.sin(bearing_rad)),
        )
    )
    offset = _vec_add(
        _vec_scale(observer_dir, distance),
        _vec_add(
            _vec_scale(up, height),
            _vec_scale(right, -(distance * 0.12)),
        ),
    )
    target = (
        float(center[0] + (forward[0] * max(6.0, span_y * 0.16))),
        float(center[1] + (forward[1] * max(6.0, span_y * 0.16))),
        float(center[2] + (forward[2] * max(4.0, span_z * 0.24))),
    )
    return _camera_pose(
        position=_vec_add(center, offset),
        target=target,
        h_fov_deg=_TT1_CAMERA_FOV[0],
        v_fov_deg=_TT1_CAMERA_FOV[1],
        far_clip=max(320.0, distance + 220.0),
    )


def _trace_test_2_camera(
    *,
    points: tuple[Point3, ...],
) -> TraceCameraPose:
    min_x = min(point[0] for point in points)
    max_x = max(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_y = max(point[1] for point in points)
    min_z = min(point[2] for point in points)
    max_z = max(point[2] for point in points)
    center = (
        float((min_x + max_x) * 0.5),
        float((min_y + max_y) * 0.5),
        float((min_z + max_z) * 0.5),
    )
    span_x = max(28.0, float(max_x - min_x))
    span_y = max(54.0, float(max_y - min_y))
    span_z = max(10.0, float(max_z - min_z))
    distance = max(112.0, (span_y * 1.05) + (span_x * 0.55))
    side = max(22.0, span_x * 0.42)
    height = max(22.0, (span_z * 2.6) + 18.0)
    position = (
        float(center[0] - side),
        float(min_y - distance),
        float(max_z + height),
    )
    target = (
        float(center[0]),
        float(center[1] + max(10.0, span_y * 0.10)),
        float(center[2] + (span_z * 0.12)),
    )
    return _camera_pose(
        position=position,
        target=target,
        h_fov_deg=_TT2_CAMERA_FOV[0],
        v_fov_deg=_TT2_CAMERA_FOV[1],
        far_clip=max(520.0, distance + span_y + 260.0),
    )


def _tt1_frame_pose(
    *,
    frame: TraceTest1SceneFrame,
    code: int,
    asset_id: str,
    scale: Point3,
) -> TraceAircraftPose:
    return TraceAircraftPose(
        code=int(code),
        asset_id=str(asset_id),
        position=tuple(float(value) for value in frame.position),
        hpr_deg=tuple(float(value) for value in trace_test_1_aircraft_hpr(frame)),
        scale=tuple(float(value) for value in scale),
    )


def _tt2_track_pose(
    *,
    track: TraceTest2AircraftTrack,
    progress: float,
    scale: Point3,
    alpha: float = 1.0,
) -> TraceAircraftPose:
    position = trace_test_2_track_position(track=track, progress=float(progress))
    tangent = trace_test_2_track_tangent(track=track, progress=float(progress))
    dx, dy, dz = (float(tangent[0]), float(tangent[1]), float(tangent[2]))
    horizontal = max(1e-6, math.hypot(dx, dy))
    asset_id = {
        "red": "plane_red",
        "blue": "plane_blue",
        "yellow": "plane_yellow",
        "silver": "plane_green",
    }.get(str(track.color_name).strip().lower(), "plane_blue")
    color_rgba = (
        float(track.color_rgb[0]) / 255.0,
        float(track.color_rgb[1]) / 255.0,
        float(track.color_rgb[2]) / 255.0,
        max(0.0, min(1.0, float(alpha))),
    )
    return TraceAircraftPose(
        code=int(track.code),
        asset_id=asset_id,
        position=(float(position.x), float(position.y), float(position.z)),
        hpr_deg=(
            math.degrees(math.atan2(dx, dy)) % 360.0,
            math.degrees(math.atan2(dz, horizontal)),
            0.0,
        ),
        scale=tuple(float(value) for value in scale),
        color_rgba=color_rgba,
    )


def _tt2_trail_points(
    *,
    track: TraceTest2AircraftTrack,
    progress: float,
) -> tuple[TraceTrailPoint, ...]:
    capped_progress = max(0.0, min(1.0, float(progress)))
    if capped_progress <= 0.04:
        return ()
    start = max(0.0, capped_progress - _TT2_TRAIL_LOOKBACK)
    points: list[TraceTrailPoint] = []
    for sample in _TT2_TRAIL_PROGRESS_STEPS:
        sample_progress = start + ((capped_progress - start) * float(sample))
        if sample_progress >= capped_progress - 1e-5:
            continue
        pose = _tt2_track_pose(
            track=track,
            progress=sample_progress,
            scale=(0.68, 0.68, 0.68),
            alpha=(0.12 + (0.16 * float(sample))),
        )
        points.append(
            TraceTrailPoint(
                position=pose.position,
                hpr_deg=pose.hpr_deg,
                alpha=0.18 + (0.18 * float(sample)),
            )
        )
    return tuple(points)


def build_trace_test_1_scene3d(
    *,
    payload: TraceTest1Payload,
) -> TraceScene3dSnapshot:
    points = (
        tuple(float(value) for value in payload.scene.red_frame.position),
        *(
            tuple(float(value) for value in frame.position)
            for frame in payload.scene.blue_frames
        ),
    )
    camera = _trace_test_1_camera(
        points=points,
        viewpoint_bearing_deg=int(payload.viewpoint_bearing_deg),
        reference_forward=tuple(float(value) for value in payload.scene.red_frame.world_forward),
        reference_up=tuple(float(value) for value in payload.scene.red_frame.world_up),
    )
    aircraft = [
        _tt1_frame_pose(
            frame=payload.scene.red_frame,
            code=1,
            asset_id="plane_red",
            scale=(1.55, 1.55, 1.55),
        )
    ]
    for idx, frame in enumerate(payload.scene.blue_frames):
        aircraft.append(
            _tt1_frame_pose(
                frame=frame,
                code=idx + 2,
                asset_id="plane_blue",
                scale=(1.18, 1.18, 1.18),
            )
        )
    return TraceScene3dSnapshot(
        camera=camera,
        aircraft=tuple(aircraft),
    )


def build_trace_test_2_scene3d(
    *,
    payload: TraceTest2Payload,
    practice_mode: bool = False,
) -> TraceScene3dSnapshot:
    current_progress = max(0.0, min(1.0, float(payload.observe_progress)))
    sample_points: list[Point3] = []
    for track in payload.aircraft:
        for sample in _TT2_TRACK_SAMPLE_PROGRESS:
            point = trace_test_2_track_position(track=track, progress=float(sample))
            sample_points.append((float(point.x), float(point.y), float(point.z)))
    camera = _trace_test_2_camera(points=tuple(sample_points))
    aircraft = tuple(
        _tt2_track_pose(
            track=track,
            progress=current_progress,
            scale=(1.22, 1.22, 1.22),
        )
        for track in payload.aircraft
    )
    ghosts: list[TraceAircraftPose] = []
    if practice_mode:
        for track in payload.aircraft:
            for trail_point in _tt2_trail_points(track=track, progress=current_progress):
                ghosts.append(
                    TraceAircraftPose(
                        code=int(track.code),
                        asset_id={
                            "red": "plane_red",
                            "blue": "plane_blue",
                            "yellow": "plane_yellow",
                            "silver": "plane_green",
                        }.get(str(track.color_name).strip().lower(), "plane_blue"),
                        position=trail_point.position,
                        hpr_deg=trail_point.hpr_deg,
                        scale=(0.66, 0.66, 0.66),
                        color_rgba=(
                            float(track.color_rgb[0]) / 255.0,
                            float(track.color_rgb[1]) / 255.0,
                            float(track.color_rgb[2]) / 255.0,
                            float(trail_point.alpha),
                        ),
                    )
                )
    return TraceScene3dSnapshot(
        camera=camera,
        aircraft=aircraft,
        ghosts=tuple(ghosts),
    )


def classify_trace_test_1_view_maneuver(
    *,
    prompt: TraceTest1PromptPlan,
    viewpoint_bearing_deg: int = 180,
) -> TraceTest1Command:
    early_progress = min(0.18, max(0.08, float(prompt.answer_open_progress) * 0.5))
    final_progress = min(0.82, max(0.68, float(prompt.answer_open_progress) + 0.24))
    early_frame = trace_test_1_scene_frames(prompt=prompt, progress=early_progress).red_frame
    final_frame = trace_test_1_scene_frames(prompt=prompt, progress=final_progress).red_frame
    camera = _trace_test_1_camera(
        points=(
            tuple(float(value) for value in early_frame.position),
            tuple(float(value) for value in final_frame.position),
        ),
        viewpoint_bearing_deg=int(viewpoint_bearing_deg),
        reference_forward=tuple(float(value) for value in early_frame.world_forward),
        reference_up=tuple(float(value) for value in early_frame.world_up),
    )
    early_forward = _frame_projected_forward_delta(frame=early_frame, camera=camera)
    final_forward = _frame_projected_forward_delta(frame=final_frame, camera=camera)
    delta = (
        float(final_forward[0] - early_forward[0]),
        float(final_forward[1] - early_forward[1]),
    )
    if abs(delta[0]) < 1e-4 and abs(delta[1]) < 1e-4:
        delta = final_forward
    if abs(delta[0]) >= abs(delta[1]):
        if delta[0] >= 0.0:
            return TraceTest1Command.RIGHT
        return TraceTest1Command.LEFT
    if delta[1] <= 0.0:
        return TraceTest1Command.PULL
    return TraceTest1Command.PUSH


def trace_test_1_camera_space_delta(
    *,
    prompt: TraceTest1PromptPlan,
    viewpoint_bearing_deg: int = 180,
) -> Point3:
    early_progress = min(0.18, max(0.08, float(prompt.answer_open_progress) * 0.5))
    final_progress = min(0.82, max(0.68, float(prompt.answer_open_progress) + 0.24))
    early_frame = trace_test_1_scene_frames(prompt=prompt, progress=early_progress).red_frame
    final_frame = trace_test_1_scene_frames(prompt=prompt, progress=final_progress).red_frame
    camera = _trace_test_1_camera(
        points=(
            tuple(float(value) for value in early_frame.position),
            tuple(float(value) for value in final_frame.position),
        ),
        viewpoint_bearing_deg=int(viewpoint_bearing_deg),
        reference_forward=tuple(float(value) for value in early_frame.world_forward),
        reference_up=tuple(float(value) for value in early_frame.world_up),
    )
    early_forward = _frame_projected_forward_delta(frame=early_frame, camera=camera)
    final_forward = _frame_projected_forward_delta(frame=final_frame, camera=camera)
    return (
        float(final_forward[0] - early_forward[0]),
        float(final_forward[1] - early_forward[1]),
        0.0,
    )


def trace_test_2_track_sample_points(
    *,
    track: TraceTest2AircraftTrack,
) -> tuple[Point3, ...]:
    return tuple(
        (
            float(point.x),
            float(point.y),
            float(point.z),
        )
        for point in (
            trace_test_2_track_position(track=track, progress=sample)
            for sample in _TT2_TRACK_SAMPLE_PROGRESS
        )
    )

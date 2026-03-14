from __future__ import annotations

from dataclasses import dataclass

from .spatial_integration import (
    SpatialIntegrationLandmark,
    SpatialIntegrationPayload,
    SpatialIntegrationPoint,
    SpatialIntegrationSceneView,
)


@dataclass(frozen=True, slots=True)
class SpatialIntegrationProjectedMarker:
    label: str
    screen_x: float
    screen_y: float


@dataclass(frozen=True, slots=True)
class SpatialIntegrationSceneLayout:
    scene_view: SpatialIntegrationSceneView
    horizon_y: float
    landmarks: tuple[SpatialIntegrationProjectedMarker, ...]
    aircraft_prev: tuple[float, float]
    aircraft_now: tuple[float, float]
    aircraft_future: tuple[float, float] | None


def _project_topdown(
    point: SpatialIntegrationPoint,
    *,
    grid_cols: int,
    grid_rows: int,
    size: tuple[int, int],
) -> tuple[float, float]:
    width = max(1, int(size[0]))
    height = max(1, int(size[1]))
    margin_x = width * 0.12
    margin_y = height * 0.24
    usable_w = max(1.0, width - (margin_x * 2.0))
    usable_h = max(1.0, height - (margin_y * 2.0))
    cell_w = usable_w / max(1, grid_cols)
    cell_h = usable_h / max(1, grid_rows)
    return (
        margin_x + ((float(point.x) + 0.5) * cell_w),
        margin_y + ((float(point.y) + 0.5) * cell_h),
    )


def _project_oblique(
    point: SpatialIntegrationPoint,
    *,
    grid_cols: int,
    grid_rows: int,
    alt_levels: int,
    size: tuple[int, int],
) -> tuple[float, float]:
    width = max(1, int(size[0]))
    height = max(1, int(size[1]))
    col_span = max(1, grid_cols - 1)
    row_span = max(1, grid_rows - 1)
    alt_span = max(1, alt_levels - 1)
    nx = float(point.x) / col_span
    ny = float(point.y) / row_span
    nz = float(point.z) / alt_span
    return (
        (width * 0.22) + (nx * width * 0.52) + (ny * width * 0.10),
        (height * 0.76) - (ny * height * 0.34) - (nz * height * 0.22),
    )


def _project(
    point: SpatialIntegrationPoint,
    *,
    payload: SpatialIntegrationPayload,
    size: tuple[int, int],
) -> tuple[float, float]:
    if payload.scene_view is SpatialIntegrationSceneView.TOPDOWN:
        return _project_topdown(
            point,
            grid_cols=int(payload.grid_cols),
            grid_rows=int(payload.grid_rows),
            size=size,
        )
    return _project_oblique(
        point,
        grid_cols=int(payload.grid_cols),
        grid_rows=int(payload.grid_rows),
        alt_levels=int(payload.alt_levels),
        size=size,
    )


def build_scene_layout(
    *,
    payload: SpatialIntegrationPayload | None,
    size: tuple[int, int],
) -> SpatialIntegrationSceneLayout:
    scene_view = (
        payload.scene_view if payload is not None else SpatialIntegrationSceneView.OBLIQUE
    )
    height = max(1, int(size[1]))
    horizon_ratio = 0.16 if scene_view is SpatialIntegrationSceneView.TOPDOWN else 0.42
    horizon_y = height * horizon_ratio

    if payload is None:
        empty_point = (size[0] * 0.5, size[1] * 0.58)
        return SpatialIntegrationSceneLayout(
            scene_view=scene_view,
            horizon_y=float(horizon_y),
            landmarks=(),
            aircraft_prev=empty_point,
            aircraft_now=empty_point,
            aircraft_future=None,
        )

    landmarks = tuple(
        SpatialIntegrationProjectedMarker(
            label=str(landmark.label),
            screen_x=float(
                _project(
                    SpatialIntegrationPoint(x=int(landmark.x), y=int(landmark.y), z=0),
                    payload=payload,
                    size=size,
                )[0]
            ),
            screen_y=float(
                _project(
                    SpatialIntegrationPoint(x=int(landmark.x), y=int(landmark.y), z=0),
                    payload=payload,
                    size=size,
                )[1]
            ),
        )
        for landmark in payload.landmarks
    )
    aircraft_prev = _project(payload.aircraft_prev, payload=payload, size=size)
    aircraft_now = _project(payload.aircraft_now, payload=payload, size=size)
    aircraft_future = None
    if payload.show_aircraft_motion:
        aircraft_future = _project(
            SpatialIntegrationPoint(
                x=int(payload.aircraft_now.x + payload.velocity.dx),
                y=int(payload.aircraft_now.y + payload.velocity.dy),
                z=int(payload.aircraft_now.z + payload.velocity.dz),
            ),
            payload=payload,
            size=size,
        )
    return SpatialIntegrationSceneLayout(
        scene_view=scene_view,
        horizon_y=float(horizon_y),
        landmarks=landmarks,
        aircraft_prev=(float(aircraft_prev[0]), float(aircraft_prev[1])),
        aircraft_now=(float(aircraft_now[0]), float(aircraft_now[1])),
        aircraft_future=(
            None
            if aircraft_future is None
            else (float(aircraft_future[0]), float(aircraft_future[1]))
        ),
    )

from __future__ import annotations

import os
import sys
import math
from dataclasses import dataclass, replace
from pathlib import Path

import pygame

from .aircraft_art import (
    apply_fixed_wing_view_rotation,
    instrument_card_pygame_palette,
    project_fixed_wing_faces,
    project_fixed_wing_point,
    rotate_fixed_wing_point,
)
from .instrument_comprehension import InstrumentAircraftViewPreset, InstrumentState
from .modern_gl_renderer import ModernInstrumentCardRenderer, _ColorVertex
from .render_assets import RenderAssetCatalog

_CANONICAL_CARD_SIZE = (448, 280)
_CARD_SPRITE_VERSION = "v20"


def _default_cache_dir() -> Path:
    env = os.environ.get("CFAST_INSTRUMENT_CARD_CACHE")
    if env:
        return Path(env).expanduser().resolve()
    home = Path.home()
    if sys.platform == "darwin":
        return home / "Library" / "Caches" / "cfast_trainer" / "instrument_cards"
    if os.name == "nt":
        local = os.environ.get("LOCALAPPDATA")
        base = Path(local) if local else (home / "AppData" / "Local")
        return base / "cfast_trainer" / "instrument_cards"
    return home / ".cache" / "cfast_trainer" / "instrument_cards"


@dataclass(frozen=True, slots=True)
class InstrumentAircraftCardKey:
    heading_deg: int
    pitch_deg: int
    bank_deg: int
    view_preset: InstrumentAircraftViewPreset

    @classmethod
    def from_state(
        cls,
        state: InstrumentState,
        *,
        view_preset: InstrumentAircraftViewPreset = InstrumentAircraftViewPreset.FRONT_LEFT,
    ) -> InstrumentAircraftCardKey:
        return cls(
            heading_deg=int(state.heading_deg) % 360,
            pitch_deg=max(-20, min(20, int(round(state.pitch_deg)))),
            bank_deg=max(-45, min(45, int(round(state.bank_deg)))),
            view_preset=view_preset,
        )

    def filename(self) -> str:
        return (
            f"{_CARD_SPRITE_VERSION}_"
            f"{self.view_preset}_"
            f"h{self.heading_deg:03d}_p{self.pitch_deg:+03d}_b{self.bank_deg:+03d}.png"
            .replace("+", "p")
            .replace("-", "m")
        )


@dataclass(frozen=True, slots=True)
class InstrumentAircraftCardViewProjection:
    view_yaw_deg: float
    view_pitch_deg: float
    view_roll_deg: float = 0.0
    scale: float = 16.0
    offset_x: float = 0.0
    offset_y: float = 0.0
    forward_x_mix: float = 0.0
    forward_y_mix: float = 0.0


@dataclass(frozen=True, slots=True)
class _InstrumentAircraftCardCameraSpec:
    camera_pos: tuple[float, float, float]
    look_at: tuple[float, float, float]
    fov_deg: float


@dataclass(frozen=True, slots=True)
class InstrumentAircraftCardPoseSignature:
    nose: tuple[int, int]
    tail: tuple[int, int]
    left_wing: tuple[int, int]
    right_wing: tuple[int, int]
    canopy: tuple[int, int]
    bounds: tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class InstrumentAircraftCardPresetSpec:
    projection: InstrumentAircraftCardViewProjection
    camera: _InstrumentAircraftCardCameraSpec
    view_root_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True, slots=True)
class InstrumentAircraftCardSemanticMetrics:
    projected_heading_deg: float
    wing_tilt_px: float
    pitch_cue_px: float


_REFERENCE_CAMERA = _InstrumentAircraftCardCameraSpec(
    camera_pos=(0.0, -22.0, 2.15),
    look_at=(0.0, 1.60, 0.32),
    fov_deg=16.0,
)


_PRESET_SPECS: dict[InstrumentAircraftViewPreset, InstrumentAircraftCardPresetSpec] = {
    InstrumentAircraftViewPreset.FRONT_LEFT: InstrumentAircraftCardPresetSpec(
        projection=InstrumentAircraftCardViewProjection(
            view_yaw_deg=0.0,
            view_pitch_deg=1.0,
            scale=12.8,
        ),
        camera=_REFERENCE_CAMERA,
    ),
    InstrumentAircraftViewPreset.FRONT_RIGHT: InstrumentAircraftCardPresetSpec(
        projection=InstrumentAircraftCardViewProjection(
            view_yaw_deg=0.0,
            view_pitch_deg=1.0,
            scale=12.8,
        ),
        camera=_REFERENCE_CAMERA,
    ),
    InstrumentAircraftViewPreset.PROFILE_LEFT: InstrumentAircraftCardPresetSpec(
        projection=InstrumentAircraftCardViewProjection(
            view_yaw_deg=0.0,
            view_pitch_deg=2.0,
            scale=12.4,
        ),
        camera=_REFERENCE_CAMERA,
    ),
    InstrumentAircraftViewPreset.PROFILE_RIGHT: InstrumentAircraftCardPresetSpec(
        projection=InstrumentAircraftCardViewProjection(
            view_yaw_deg=0.0,
            view_pitch_deg=2.0,
            scale=12.4,
        ),
        camera=_REFERENCE_CAMERA,
    ),
    InstrumentAircraftViewPreset.TOP_OBLIQUE: InstrumentAircraftCardPresetSpec(
        projection=InstrumentAircraftCardViewProjection(
            view_yaw_deg=0.0,
            view_pitch_deg=6.0,
            scale=12.0,
        ),
        camera=_REFERENCE_CAMERA,
    ),
}


def instrument_aircraft_card_preset_spec(
    preset: InstrumentAircraftViewPreset,
) -> InstrumentAircraftCardPresetSpec:
    return _PRESET_SPECS[preset]


def instrument_aircraft_card_view_projection(
    preset: InstrumentAircraftViewPreset,
) -> InstrumentAircraftCardViewProjection:
    return instrument_aircraft_card_preset_spec(preset).projection


def _camera_spec_for_preset(
    preset: InstrumentAircraftViewPreset,
) -> _InstrumentAircraftCardCameraSpec:
    return instrument_aircraft_card_preset_spec(preset).camera


def _view_root_offset_for_preset(
    preset: InstrumentAircraftViewPreset,
) -> tuple[float, float, float]:
    return instrument_aircraft_card_preset_spec(preset).view_root_offset


def aircraft_card_pose_signature(
    state: InstrumentState,
    *,
    view_preset: InstrumentAircraftViewPreset = InstrumentAircraftViewPreset.FRONT_LEFT,
) -> InstrumentAircraftCardPoseSignature:
    projection = instrument_aircraft_card_view_projection(view_preset)
    scale = max(40.0, float(projection.scale) * 6.0)

    def project(point: tuple[float, float, float]) -> tuple[int, int]:
        rotated = rotate_fixed_wing_point(
            point,
            heading_deg=float(state.heading_deg),
            pitch_deg=float(state.pitch_deg),
            bank_deg=float(state.bank_deg),
        )
        viewed = apply_fixed_wing_view_rotation(
            rotated,
            view_yaw_deg=projection.view_yaw_deg,
            view_pitch_deg=projection.view_pitch_deg,
            view_roll_deg=projection.view_roll_deg,
        )
        sx, sy, _depth = project_fixed_wing_point(
            viewed,
            cx=0,
            cy=0,
            scale=scale,
            forward_x_mix=projection.forward_x_mix,
            forward_y_mix=projection.forward_y_mix,
        )
        return int(sx), int(sy)

    faces = project_fixed_wing_faces(
        heading_deg=float(state.heading_deg),
        pitch_deg=float(state.pitch_deg),
        bank_deg=float(state.bank_deg),
        cx=0,
        cy=0,
        scale=scale,
        view_yaw_deg=projection.view_yaw_deg,
        view_pitch_deg=projection.view_pitch_deg,
        view_roll_deg=projection.view_roll_deg,
        forward_x_mix=projection.forward_x_mix,
        forward_y_mix=projection.forward_y_mix,
    )
    points = [point for face in faces for point in face.points]
    min_x = min(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_x = max(point[0] for point in points)
    max_y = max(point[1] for point in points)

    return InstrumentAircraftCardPoseSignature(
        nose=project((0.0, 3.42, 0.12)),
        tail=project((0.0, -2.48, 0.18)),
        left_wing=project((-3.86, 0.56, 0.16)),
        right_wing=project((3.86, 0.56, 0.16)),
        canopy=project((0.0, 1.42, 0.64)),
        bounds=(int(min_x), int(min_y), int(max_x), int(max_y)),
    )


def aircraft_card_pose_distance(
    left: InstrumentAircraftCardPoseSignature,
    right: InstrumentAircraftCardPoseSignature,
) -> float:
    landmarks = (
        (left.nose, right.nose),
        (left.tail, right.tail),
        (left.left_wing, right.left_wing),
        (left.right_wing, right.right_wing),
        (left.canopy, right.canopy),
    )
    distance = 0.0
    for a, b in landmarks:
        distance += abs(float(a[0] - b[0])) + abs(float(a[1] - b[1]))
    distance += 0.5 * sum(abs(float(a - b)) for a, b in zip(left.bounds, right.bounds, strict=False))
    return float(distance)


def aircraft_card_projected_heading_deg(signature: InstrumentAircraftCardPoseSignature) -> float:
    dx = float(signature.nose[0] - signature.tail[0])
    dy = float(signature.nose[1] - signature.tail[1])
    return _wrap_signed_deg(math.degrees(math.atan2(-dy, dx)))


def aircraft_card_wing_tilt_px(signature: InstrumentAircraftCardPoseSignature) -> float:
    return float(signature.left_wing[1] - signature.right_wing[1])


def aircraft_card_pitch_cue_px(signature: InstrumentAircraftCardPoseSignature) -> float:
    body_mid_y = (float(signature.nose[1]) + float(signature.tail[1])) * 0.5
    return body_mid_y - float(signature.canopy[1])


def aircraft_card_semantic_metrics(
    signature: InstrumentAircraftCardPoseSignature,
) -> InstrumentAircraftCardSemanticMetrics:
    return InstrumentAircraftCardSemanticMetrics(
        projected_heading_deg=aircraft_card_projected_heading_deg(signature),
        wing_tilt_px=aircraft_card_wing_tilt_px(signature),
        pitch_cue_px=aircraft_card_pitch_cue_px(signature),
    )


def aircraft_card_semantic_drift_tags(
    state: InstrumentState,
    *,
    view_preset: InstrumentAircraftViewPreset = InstrumentAircraftViewPreset.FRONT_LEFT,
    heading_tolerance_deg: float = 24.0,
    heading_margin_deg: float = 8.0,
    neutral_bank_sample_deg: int = 12,
    neutral_bank_symmetry_tolerance_px: float = 16.0,
    bank_delta_tolerance_px: float = 2.0,
    pitch_delta_tolerance_px: float = 0.5,
) -> tuple[str, ...]:
    signature = aircraft_card_pose_signature(state, view_preset=view_preset)
    metrics = aircraft_card_semantic_metrics(signature)
    tags: list[str] = []

    heading_deg = int(state.heading_deg) % 360
    if int(state.pitch_deg) == 0 and int(state.bank_deg) == 0:
        expected_heading = _wrap_signed_deg(90.0 - float(heading_deg))
        heading_err = _wrapped_angle_error(metrics.projected_heading_deg, expected_heading)
        nearby_heading_errors = [
            _wrapped_angle_error(
                metrics.projected_heading_deg,
                _wrap_signed_deg(90.0 - float((heading_deg + delta) % 360)),
            )
            for delta in (90, 180, 270)
        ]
        if heading_err > float(heading_tolerance_deg) or any(
            error + float(heading_margin_deg) < heading_err for error in nearby_heading_errors
        ):
            tags.append("heading_axis")

    neutral_bank_signature = aircraft_card_pose_signature(
        replace(state, bank_deg=0),
        view_preset=view_preset,
    )
    neutral_bank_metrics = aircraft_card_semantic_metrics(neutral_bank_signature)
    wing_delta = metrics.wing_tilt_px - neutral_bank_metrics.wing_tilt_px
    bank = int(state.bank_deg)
    if bank == 0 and abs(int(state.pitch_deg)) <= 2:
        left_bank_metrics = aircraft_card_semantic_metrics(
            aircraft_card_pose_signature(
                replace(state, bank_deg=-abs(int(neutral_bank_sample_deg))),
                view_preset=view_preset,
            )
        )
        right_bank_metrics = aircraft_card_semantic_metrics(
            aircraft_card_pose_signature(
                replace(state, bank_deg=abs(int(neutral_bank_sample_deg))),
                view_preset=view_preset,
            )
        )
        left_delta = left_bank_metrics.wing_tilt_px - metrics.wing_tilt_px
        right_delta = right_bank_metrics.wing_tilt_px - metrics.wing_tilt_px
        if (
            left_delta <= 0.0
            or right_delta >= 0.0
            or abs(abs(left_delta) - abs(right_delta)) > float(neutral_bank_symmetry_tolerance_px)
        ):
            tags.append("bank_neutral")
    elif abs(bank) >= 2 and (
        abs(wing_delta) < float(bank_delta_tolerance_px) or (wing_delta * float(bank)) > 0.0
    ):
        tags.append("bank_sign")

    neutral_pitch_signature = aircraft_card_pose_signature(
        replace(state, pitch_deg=0),
        view_preset=view_preset,
    )
    neutral_pitch_metrics = aircraft_card_semantic_metrics(neutral_pitch_signature)
    pitch_delta = metrics.pitch_cue_px - neutral_pitch_metrics.pitch_cue_px
    pitch = int(state.pitch_deg)
    if abs(pitch) >= 2:
        if abs(pitch_delta) < float(pitch_delta_tolerance_px) or (pitch_delta * float(pitch)) < 0.0:
            tags.append("pitch_sign")

    return tuple(tags)


def _wrap_signed_deg(angle_deg: float) -> float:
    wrapped = (float(angle_deg) + 180.0) % 360.0 - 180.0
    if wrapped == -180.0:
        return 180.0
    return wrapped


def _wrapped_angle_error(left: float, right: float) -> float:
    return abs(_wrap_signed_deg(float(left) - float(right)))


class InstrumentAircraftCardSpriteBank:
    def __init__(
        self,
        *,
        cache_dir: Path | None = None,
        asset_catalog: RenderAssetCatalog | None = None,
        allow_generation: bool = True,
    ) -> None:
        self._cache_dir = (cache_dir or _default_cache_dir()).resolve()
        self._asset_catalog = asset_catalog or RenderAssetCatalog()
        self._allow_generation = bool(allow_generation)
        self._surface_cache: dict[InstrumentAircraftCardKey, pygame.Surface] = {}
        self._scaled_cache: dict[tuple[InstrumentAircraftCardKey, int, int], pygame.Surface] = {}
        self._renderer: ModernInstrumentCardRenderer | None = None
        self._generation_failed = False

    def get_scaled_surface(
        self,
        *,
        state: InstrumentState,
        size: tuple[int, int],
        view_preset: InstrumentAircraftViewPreset = InstrumentAircraftViewPreset.FRONT_LEFT,
    ) -> pygame.Surface | None:
        key = InstrumentAircraftCardKey.from_state(state, view_preset=view_preset)
        cache_key = (key, int(size[0]), int(size[1]))
        cached = self._scaled_cache.get(cache_key)
        if cached is not None:
            return cached

        source = self.get_surface(state=state, view_preset=view_preset)
        if source is None:
            return None
        scaled = pygame.transform.smoothscale(source, size)
        self._scaled_cache[cache_key] = scaled
        return scaled

    def get_surface(
        self,
        *,
        state: InstrumentState,
        view_preset: InstrumentAircraftViewPreset = InstrumentAircraftViewPreset.FRONT_LEFT,
    ) -> pygame.Surface | None:
        key = InstrumentAircraftCardKey.from_state(state, view_preset=view_preset)
        cached = self._surface_cache.get(key)
        if cached is not None:
            return cached

        cache_path = self._cache_dir / key.filename()
        if cache_path.exists():
            loaded = self._normalize_surface(self._load_surface(cache_path))
            self._surface_cache[key] = loaded
            return loaded

        if not self._allow_generation or self._generation_failed:
            fallback = self._render_software_surface(key)
            self._surface_cache[key] = fallback
            return fallback

        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            self._asset_catalog.require("plane_red")
            renderer = self._get_renderer()
            renderer.render_card(
                key=key,
                destination=cache_path,
                draw_callback=self._render_modern_gl_card,
                size=_CANONICAL_CARD_SIZE,
            )
            loaded = self._normalize_surface(self._load_surface(cache_path))
            self._surface_cache[key] = loaded
            return loaded
        except Exception:
            self._generation_failed = True
            fallback = self._render_software_surface(key)
            self._surface_cache[key] = fallback
            return fallback

    def cache_path_for(
        self,
        *,
        state: InstrumentState,
        view_preset: InstrumentAircraftViewPreset = InstrumentAircraftViewPreset.FRONT_LEFT,
    ) -> Path:
        return self._cache_dir / InstrumentAircraftCardKey.from_state(
            state,
            view_preset=view_preset,
        ).filename()

    def _get_renderer(self) -> ModernInstrumentCardRenderer:
        if self._renderer is None:
            self._renderer = ModernInstrumentCardRenderer()
        return self._renderer

    def _render_software_surface(
        self,
        key: InstrumentAircraftCardKey,
    ) -> pygame.Surface:
        surface = pygame.Surface(_CANONICAL_CARD_SIZE, pygame.SRCALPHA)
        self._paint_card_background(surface)
        projection = instrument_aircraft_card_view_projection(key.view_preset)
        palette = instrument_card_pygame_palette()
        scale = max(10.5, float(min(surface.get_width(), surface.get_height())) * (projection.scale / 100.0))
        faces = project_fixed_wing_faces(
            heading_deg=float(key.heading_deg),
            pitch_deg=float(key.pitch_deg),
            bank_deg=float(key.bank_deg),
            cx=surface.get_rect().centerx + int(round(projection.offset_x)),
            cy=surface.get_rect().centery + int(round(projection.offset_y)),
            scale=scale,
            view_yaw_deg=projection.view_yaw_deg,
            view_pitch_deg=projection.view_pitch_deg,
            view_roll_deg=projection.view_roll_deg,
            forward_x_mix=projection.forward_x_mix,
            forward_y_mix=projection.forward_y_mix,
        )
        role_colors = {
            "body": palette.body,
            "accent": palette.accent,
            "canopy": palette.canopy,
            "engine": palette.engine,
        }
        for face in faces:
            base = role_colors.get(face.role, palette.body)
            fill = tuple(max(0, min(255, int(round(channel * face.shade)))) for channel in base)
            pygame.draw.polygon(surface, fill, face.points)
        pygame.draw.rect(surface, (170, 184, 212), surface.get_rect(), 1)
        return self._normalize_surface(surface)

    @staticmethod
    def _load_surface(path: Path) -> pygame.Surface:
        loaded = pygame.image.load(str(path))
        if pygame.display.get_init() and pygame.display.get_surface() is not None:
            return loaded.convert_alpha()
        return loaded.copy()

    @staticmethod
    def _aircraft_bounds(surface: pygame.Surface) -> tuple[int, int, int, int] | None:
        min_x = surface.get_width()
        min_y = surface.get_height()
        max_x = -1
        max_y = -1
        for y in range(surface.get_height()):
            for x in range(surface.get_width()):
                color = surface.get_at((x, y))
                if color.a <= 0:
                    continue
                if color.r <= 120 or color.r <= color.g + 20 or color.r <= color.b + 20:
                    continue
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)
        if max_x < min_x or max_y < min_y:
            return None
        return (min_x, min_y, max_x, max_y)

    def _normalize_surface(self, surface: pygame.Surface) -> pygame.Surface:
        bounds = self._aircraft_bounds(surface)
        if bounds is None:
            return surface

        min_x, min_y, max_x, max_y = bounds
        width = max_x - min_x + 1
        height = max_y - min_y + 1
        inset = 28
        target_w = max(1, surface.get_width() - inset * 2)
        target_h = max(1, surface.get_height() - inset * 2)
        scale = min(1.0, target_w / float(width), target_h / float(height))
        if scale >= 0.995 and min_x >= inset and min_y >= inset and max_x <= surface.get_width() - inset and max_y <= surface.get_height() - inset:
            return surface

        scaled_size = (
            max(1, int(round(surface.get_width() * scale))),
            max(1, int(round(surface.get_height() * scale))),
        )
        if surface.get_bitsize() in (24, 32):
            scaled = pygame.transform.smoothscale(surface, scaled_size)
        else:
            scaled = pygame.transform.scale(surface, scaled_size)
        scaled_bounds = self._aircraft_bounds(scaled)
        result = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        self._paint_card_background(result)
        if scaled_bounds is None:
            result.blit(scaled, scaled.get_rect(center=surface.get_rect().center))
            pygame.draw.rect(result, (170, 184, 212), result.get_rect(), 1)
            return result

        s_min_x, s_min_y, s_max_x, s_max_y = scaled_bounds
        bounds_w = s_max_x - s_min_x + 1
        bounds_h = s_max_y - s_min_y + 1
        dest_x = ((result.get_width() - bounds_w) // 2) - s_min_x
        dest_y = ((result.get_height() - bounds_h) // 2) - s_min_y
        result.blit(scaled, (dest_x, dest_y))
        pygame.draw.rect(result, (170, 184, 212), result.get_rect(), 1)
        return result

    @staticmethod
    def _paint_card_background(surface: pygame.Surface) -> None:
        rect = surface.get_rect()
        for y in range(rect.height):
            t = y / max(1, rect.height - 1)
            shade = int(round(232 - (t * 46)))
            pygame.draw.line(surface, (shade, shade, shade), (0, y), (rect.width, y))

        vignette = pygame.Surface(rect.size, pygame.SRCALPHA)
        for ring in range(4):
            alpha = 26 + ring * 12
            inset = ring * max(6, rect.width // 22)
            pygame.draw.rect(
                vignette,
                (24, 28, 36, alpha),
                pygame.Rect(
                    inset,
                    inset,
                    max(4, rect.width - inset * 2),
                    max(4, rect.height - inset * 2),
                ),
                0,
            )
        surface.blit(vignette, (0, 0), special_flags=pygame.BLEND_RGBA_SUB)

        floor_y = rect.bottom - max(10, rect.height // 9)
        pygame.draw.line(surface, (84, 88, 96), (rect.x, floor_y), (rect.right, floor_y), 2)
        for step in (0.2, 0.4, 0.6, 0.8):
            y = int(round(floor_y + ((rect.bottom - floor_y) * step)))
            shade = 124 - int(round(step * 24))
            pygame.draw.line(surface, (shade, shade, shade), (rect.x, y), (rect.right, y), 1)
        for lane in range(1, 5):
            x = rect.centerx + int(round((lane - 2.5) * (rect.w / 7.5)))
            pygame.draw.line(surface, (132, 136, 144), (x, floor_y), (rect.centerx, rect.bottom), 1)

    @staticmethod
    def _append_triangle(
        batch,
        a: tuple[float, float],
        b: tuple[float, float],
        c: tuple[float, float],
        color: tuple[int, int, int, int],
    ) -> None:
        rgba = tuple(float(channel) / 255.0 for channel in color)
        batch.triangles.extend(
            [
                _ColorVertex(float(a[0]), float(a[1]), *rgba),
                _ColorVertex(float(b[0]), float(b[1]), *rgba),
                _ColorVertex(float(c[0]), float(c[1]), *rgba),
            ]
        )

    @classmethod
    def _append_quad(
        cls,
        batch,
        a: tuple[float, float],
        b: tuple[float, float],
        c: tuple[float, float],
        d: tuple[float, float],
        color: tuple[int, int, int, int],
    ) -> None:
        cls._append_triangle(batch, a, b, c, color)
        cls._append_triangle(batch, a, c, d, color)

    @classmethod
    def _append_polygon(
        cls,
        batch,
        points: list[tuple[int, int]] | tuple[tuple[int, int], ...],
        color: tuple[int, int, int, int],
    ) -> None:
        if len(points) < 3:
            return
        origin = (float(points[0][0]), float(points[0][1]))
        for idx in range(1, len(points) - 1):
            cls._append_triangle(
                batch,
                origin,
                (float(points[idx][0]), float(points[idx][1])),
                (float(points[idx + 1][0]), float(points[idx + 1][1])),
                color,
            )

    @classmethod
    def _render_modern_gl_card(
        cls,
        batch,
        key: InstrumentAircraftCardKey,
        width: int,
        height: int,
    ) -> None:
        rect = pygame.Rect(0, 0, int(width), int(height))
        projection = instrument_aircraft_card_view_projection(key.view_preset)
        palette = instrument_card_pygame_palette()

        step_h = max(1, rect.height // 48)
        for y in range(0, rect.height, step_h):
            t = y / max(1, rect.height - 1)
            shade = int(round(232 - (t * 46)))
            y0 = float(y)
            y1 = float(min(rect.height, y + step_h))
            cls._append_quad(
                batch,
                (0.0, y0),
                (float(rect.width), y0),
                (float(rect.width), y1),
                (0.0, y1),
                (shade, shade, shade, 255),
            )

        floor_y = rect.bottom - max(10, rect.height // 9)
        cls._append_quad(
            batch,
            (0.0, float(floor_y - 1)),
            (float(rect.width), float(floor_y - 1)),
            (float(rect.width), float(floor_y + 1)),
            (0.0, float(floor_y + 1)),
            (84, 88, 96, 255),
        )
        for step in (0.2, 0.4, 0.6, 0.8):
            y = int(round(floor_y + ((rect.bottom - floor_y) * step)))
            shade = 124 - int(round(step * 24))
            cls._append_quad(
                batch,
                (0.0, float(y)),
                (float(rect.width), float(y)),
                (float(rect.width), float(y + 1)),
                (0.0, float(y + 1)),
                (shade, shade, shade, 255),
            )

        scale = max(10.5, float(min(rect.w, rect.h)) * (projection.scale / 100.0))
        faces = project_fixed_wing_faces(
            heading_deg=float(key.heading_deg),
            pitch_deg=float(key.pitch_deg),
            bank_deg=float(key.bank_deg),
            cx=rect.centerx + int(round(projection.offset_x)),
            cy=rect.centery + int(round(projection.offset_y)),
            scale=scale,
            view_yaw_deg=projection.view_yaw_deg,
            view_pitch_deg=projection.view_pitch_deg,
            view_roll_deg=projection.view_roll_deg,
            forward_x_mix=projection.forward_x_mix,
            forward_y_mix=projection.forward_y_mix,
        )
        role_colors = {
            "body": palette.body,
            "accent": palette.accent,
            "canopy": palette.canopy,
            "engine": palette.engine,
        }
        for face in faces:
            base = role_colors.get(face.role, palette.body)
            fill = tuple(max(0, min(255, int(round(channel * face.shade)))) for channel in base)
            cls._append_polygon(batch, face.points, (*fill, 255))

        border = (170, 184, 212, 255)
        cls._append_quad(
            batch,
            (0.0, 0.0),
            (float(rect.width), 0.0),
            (float(rect.width), 1.0),
            (0.0, 1.0),
            border,
        )
        cls._append_quad(
            batch,
            (0.0, float(rect.height - 1)),
            (float(rect.width), float(rect.height - 1)),
            (float(rect.width), float(rect.height)),
            (0.0, float(rect.height)),
            border,
        )
        cls._append_quad(
            batch,
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, float(rect.height)),
            (0.0, float(rect.height)),
            border,
        )
        cls._append_quad(
            batch,
            (float(rect.width - 1), 0.0),
            (float(rect.width), 0.0),
            (float(rect.width), float(rect.height)),
            (float(rect.width - 1), float(rect.height)),
            border,
        )

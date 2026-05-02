from __future__ import annotations

import math
from dataclasses import dataclass, replace

import pygame

from .aircraft_art import (
    apply_fixed_wing_view_rotation,
    draw_fixed_wing_pygame,
    instrument_card_pygame_palette,
    project_fixed_wing_faces,
    project_fixed_wing_point,
    rotate_fixed_wing_point,
)
from .instrument_comprehension import InstrumentAircraftViewPreset, InstrumentState

_CANONICAL_CARD_SIZE = (448, 280)
_CARD_SPRITE_VERSION = "mesh_v1"


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
    scale_ratio: float = 0.105
    offset_x_ratio: float = 0.0
    offset_y_ratio: float = 0.0
    forward_x_mix: float = 0.10
    forward_y_mix: float = 0.26


@dataclass(frozen=True, slots=True)
class InstrumentAircraftCardPoseSignature:
    nose: tuple[int, int]
    tail: tuple[int, int]
    left_wing: tuple[int, int]
    right_wing: tuple[int, int]
    canopy: tuple[int, int]
    bounds: tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class InstrumentAircraftCardSemanticMetrics:
    projected_heading_deg: float
    wing_tilt_px: float
    pitch_cue_px: float


_PRESET_PROJECTIONS: dict[InstrumentAircraftViewPreset, InstrumentAircraftCardViewProjection] = {
    InstrumentAircraftViewPreset.FRONT_LEFT: InstrumentAircraftCardViewProjection(
        view_yaw_deg=28.0,
        view_pitch_deg=4.0,
        scale_ratio=0.107,
        offset_y_ratio=-0.02,
    ),
    InstrumentAircraftViewPreset.FRONT_RIGHT: InstrumentAircraftCardViewProjection(
        view_yaw_deg=-28.0,
        view_pitch_deg=4.0,
        scale_ratio=0.107,
        offset_y_ratio=-0.02,
    ),
    InstrumentAircraftViewPreset.PROFILE_LEFT: InstrumentAircraftCardViewProjection(
        view_yaw_deg=84.0,
        view_pitch_deg=5.0,
        scale_ratio=0.104,
        forward_x_mix=0.04,
        forward_y_mix=0.20,
    ),
    InstrumentAircraftViewPreset.PROFILE_RIGHT: InstrumentAircraftCardViewProjection(
        view_yaw_deg=-84.0,
        view_pitch_deg=5.0,
        scale_ratio=0.104,
        forward_x_mix=0.04,
        forward_y_mix=0.20,
    ),
    InstrumentAircraftViewPreset.TOP_DOWN: InstrumentAircraftCardViewProjection(
        view_yaw_deg=0.0,
        view_pitch_deg=68.0,
        scale_ratio=0.118,
        forward_x_mix=0.02,
        forward_y_mix=0.04,
    ),
}


def instrument_aircraft_card_view_projection(
    preset: InstrumentAircraftViewPreset,
) -> InstrumentAircraftCardViewProjection:
    return _PRESET_PROJECTIONS[preset]


def aircraft_card_pose_signature(
    state: InstrumentState,
    *,
    view_preset: InstrumentAircraftViewPreset = InstrumentAircraftViewPreset.FRONT_LEFT,
) -> InstrumentAircraftCardPoseSignature:
    projection = instrument_aircraft_card_view_projection(view_preset)
    scale = 80.0

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
        nose=project((0.0, 3.10, 0.04)),
        tail=project((0.0, -2.78, 0.20)),
        left_wing=project((-4.15, 0.08, 0.14)),
        right_wing=project((4.15, 0.08, 0.14)),
        canopy=project((0.0, 1.40, 0.70)),
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
    neutral_bank_sample_deg: int = 12,
    neutral_bank_symmetry_tolerance_px: float = 18.0,
    bank_delta_tolerance_px: float = 1.2,
    pitch_delta_tolerance_px: float = 0.4,
) -> tuple[str, ...]:
    signature = aircraft_card_pose_signature(state, view_preset=view_preset)
    metrics = aircraft_card_semantic_metrics(signature)
    tags: list[str] = []

    neutral_bank_signature = aircraft_card_pose_signature(
        replace(state, bank_deg=0),
        view_preset=view_preset,
    )
    neutral_bank_metrics = aircraft_card_semantic_metrics(neutral_bank_signature)
    wing_delta = metrics.wing_tilt_px - neutral_bank_metrics.wing_tilt_px
    bank = int(state.bank_deg)
    if view_preset is InstrumentAircraftViewPreset.TOP_DOWN:
        pass
    elif bank == 0 and abs(int(state.pitch_deg)) <= 2:
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
    elif view_preset is not InstrumentAircraftViewPreset.TOP_DOWN and abs(bank) >= 2:
        if abs(wing_delta) < float(bank_delta_tolerance_px) or (wing_delta * float(bank)) > 0.0:
            tags.append("bank_sign")

    neutral_pitch_signature = aircraft_card_pose_signature(
        replace(state, pitch_deg=0),
        view_preset=view_preset,
    )
    neutral_pitch_metrics = aircraft_card_semantic_metrics(neutral_pitch_signature)
    pitch_delta = metrics.pitch_cue_px - neutral_pitch_metrics.pitch_cue_px
    pitch = int(state.pitch_deg)
    if view_preset is not InstrumentAircraftViewPreset.TOP_DOWN and abs(pitch) >= 2:
        if abs(pitch_delta) < float(pitch_delta_tolerance_px) or (pitch_delta * float(pitch)) < 0.0:
            tags.append("pitch_sign")

    return tuple(tags)


class InstrumentAircraftCardSpriteBank:
    def __init__(self) -> None:
        self._surface_cache: dict[InstrumentAircraftCardKey, pygame.Surface] = {}
        self._scaled_cache: dict[tuple[InstrumentAircraftCardKey, int, int], pygame.Surface] = {}

    def get_scaled_surface(
        self,
        *,
        state: InstrumentState,
        size: tuple[int, int],
        view_preset: InstrumentAircraftViewPreset = InstrumentAircraftViewPreset.FRONT_LEFT,
    ) -> pygame.Surface:
        key = InstrumentAircraftCardKey.from_state(state, view_preset=view_preset)
        cache_key = (key, int(size[0]), int(size[1]))
        cached = self._scaled_cache.get(cache_key)
        if cached is not None:
            return cached

        source = self.get_surface(state=state, view_preset=view_preset)
        if source.get_size() == size:
            scaled = source.copy()
        else:
            scaled = pygame.transform.smoothscale(source, size)
        self._scaled_cache[cache_key] = scaled
        return scaled

    def get_surface(
        self,
        *,
        state: InstrumentState,
        view_preset: InstrumentAircraftViewPreset = InstrumentAircraftViewPreset.FRONT_LEFT,
    ) -> pygame.Surface:
        key = InstrumentAircraftCardKey.from_state(state, view_preset=view_preset)
        cached = self._surface_cache.get(key)
        if cached is not None:
            return cached
        surface = self._render_software_surface(key)
        self._surface_cache[key] = surface
        return surface

    def _render_software_surface(
        self,
        key: InstrumentAircraftCardKey,
    ) -> pygame.Surface:
        surface = pygame.Surface(_CANONICAL_CARD_SIZE, pygame.SRCALPHA)
        self._paint_card_background(surface)
        projection = instrument_aircraft_card_view_projection(key.view_preset)
        scale = max(12.0, min(surface.get_size()) * projection.scale_ratio)
        draw_fixed_wing_pygame(
            surface,
            heading_deg=float(key.heading_deg),
            pitch_deg=float(key.pitch_deg),
            bank_deg=float(key.bank_deg),
            cx=surface.get_rect().centerx + int(round(surface.get_width() * projection.offset_x_ratio)),
            cy=surface.get_rect().centery + int(round(surface.get_height() * projection.offset_y_ratio)),
            scale=scale,
            palette=instrument_card_pygame_palette(),
            view_yaw_deg=projection.view_yaw_deg,
            view_pitch_deg=projection.view_pitch_deg,
            view_roll_deg=projection.view_roll_deg,
            forward_x_mix=projection.forward_x_mix,
            forward_y_mix=projection.forward_y_mix,
        )
        pygame.draw.rect(surface, (170, 184, 212), surface.get_rect(), 1)
        return self._normalize_surface(surface)

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
                if color.r <= 120 or color.r <= color.g + 18 or color.r <= color.b + 18:
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
        inset = 24
        target_w = max(1, surface.get_width() - inset * 2)
        target_h = max(1, surface.get_height() - inset * 2)
        scale = min(1.0, target_w / float(width), target_h / float(height))
        if (
            scale >= 0.995
            and min_x >= inset
            and min_y >= inset
            and max_x <= surface.get_width() - inset
            and max_y <= surface.get_height() - inset
        ):
            return surface

        scaled_size = (
            max(1, int(round(surface.get_width() * scale))),
            max(1, int(round(surface.get_height() * scale))),
        )
        scaled = pygame.transform.smoothscale(surface, scaled_size)
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


def _wrap_signed_deg(angle_deg: float) -> float:
    wrapped = (float(angle_deg) + 180.0) % 360.0 - 180.0
    if wrapped == -180.0:
        return 180.0
    return wrapped

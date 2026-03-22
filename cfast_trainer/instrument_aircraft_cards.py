from __future__ import annotations

import importlib.util
import os
import sys
import math
from dataclasses import dataclass, replace
from pathlib import Path

import pygame

from .aircraft_art import (
    apply_fixed_wing_view_rotation,
    build_panda_palette,
    build_panda3d_fixed_wing_model,
    draw_fixed_wing_pygame,
    instrument_card_pygame_palette,
    project_fixed_wing_faces,
    project_fixed_wing_point,
    rotate_fixed_wing_point,
)
from .instrument_comprehension import InstrumentAircraftViewPreset, InstrumentState
from .panda3d_assets import Panda3DAssetCatalog

_CANONICAL_CARD_SIZE = (448, 280)
_CARD_SPRITE_VERSION = "v20"


def panda3d_card_rendering_available() -> bool:
    return importlib.util.find_spec("direct.showbase.ShowBase") is not None


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
        asset_catalog: Panda3DAssetCatalog | None = None,
        allow_generation: bool = True,
    ) -> None:
        self._cache_dir = (cache_dir or _default_cache_dir()).resolve()
        self._asset_catalog = asset_catalog or Panda3DAssetCatalog()
        self._allow_generation = bool(allow_generation)
        self._surface_cache: dict[InstrumentAircraftCardKey, pygame.Surface] = {}
        self._scaled_cache: dict[tuple[InstrumentAircraftCardKey, int, int], pygame.Surface] = {}
        self._renderer: _Panda3DInstrumentCardRenderer | None = None
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
            return None

        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            loaded: pygame.Surface | None = None
            if panda3d_card_rendering_available():
                try:
                    renderer = self._get_renderer()
                    renderer.render_card(key=key, destination=cache_path)
                    loaded = self._normalize_surface(self._load_surface(cache_path))
                except Exception:
                    loaded = None
            if loaded is None:
                loaded = self._normalize_surface(self._render_pygame_fallback_card(key))
                pygame.image.save(loaded, str(cache_path))
            self._surface_cache[key] = loaded
            return loaded
        except Exception:
            self._generation_failed = True
            return None

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

    def _get_renderer(self) -> _Panda3DInstrumentCardRenderer:
        if self._renderer is None:
            self._renderer = _Panda3DInstrumentCardRenderer(asset_catalog=self._asset_catalog)
        return self._renderer

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

    @classmethod
    def _render_pygame_fallback_card(cls, key: InstrumentAircraftCardKey) -> pygame.Surface:
        surface = pygame.Surface(_CANONICAL_CARD_SIZE, pygame.SRCALPHA)
        rect = surface.get_rect()
        cls._paint_card_background(surface)

        projection = instrument_aircraft_card_view_projection(key.view_preset)
        draw_fixed_wing_pygame(
            surface,
            heading_deg=float(key.heading_deg),
            pitch_deg=float(key.pitch_deg),
            bank_deg=float(key.bank_deg),
            cx=rect.centerx + int(round(projection.offset_x)),
            cy=rect.centery + int(round(projection.offset_y)),
            scale=max(10.5, float(min(rect.w, rect.h)) * (projection.scale / 100.0)),
            palette=instrument_card_pygame_palette(),
            view_yaw_deg=projection.view_yaw_deg,
            view_pitch_deg=projection.view_pitch_deg,
            view_roll_deg=projection.view_roll_deg,
            forward_x_mix=projection.forward_x_mix,
            forward_y_mix=projection.forward_y_mix,
        )
        pygame.draw.rect(surface, (170, 184, 212), rect, 1)
        return surface


class _Panda3DInstrumentCardRenderer:
    def __init__(self, *, asset_catalog: Panda3DAssetCatalog) -> None:
        self._asset_catalog = asset_catalog

        from panda3d.core import AmbientLight, DirectionalLight, Vec4, loadPrcFileData

        loadPrcFileData("", "window-type offscreen")
        loadPrcFileData("", "audio-library-name null")
        loadPrcFileData("", f"win-size {_CANONICAL_CARD_SIZE[0]} {_CANONICAL_CARD_SIZE[1]}")
        loadPrcFileData("", "sync-video false")

        from direct.showbase.ShowBase import ShowBase

        self._base = ShowBase()
        self._base.disableMouse()
        self._base.setBackgroundColor(0.83, 0.85, 0.89, 1.0)
        default_camera = _camera_spec_for_preset(InstrumentAircraftViewPreset.FRONT_LEFT)
        self._base.camLens.setFov(default_camera.fov_deg)
        self._base.cam.setPos(*default_camera.camera_pos)
        self._base.cam.lookAt(*default_camera.look_at)

        ambient = AmbientLight("ambient")
        ambient.setColor(Vec4(0.34, 0.36, 0.39, 1.0))
        ambient_np = self._base.render.attachNewNode(ambient)
        self._base.render.setLight(ambient_np)

        key_light = DirectionalLight("key-light")
        key_light.setColor(Vec4(1.10, 1.02, 0.96, 1.0))
        key_light_np = self._base.render.attachNewNode(key_light)
        key_light_np.setHpr(22.0, -34.0, 0.0)
        self._base.render.setLight(key_light_np)

        fill_light = DirectionalLight("fill-light")
        fill_light.setColor(Vec4(0.38, 0.46, 0.58, 1.0))
        fill_light_np = self._base.render.attachNewNode(fill_light)
        fill_light_np.setHpr(-118.0, -16.0, 0.0)
        self._base.render.setLight(fill_light_np)

        rim_light = DirectionalLight("rim-light")
        rim_light.setColor(Vec4(0.56, 0.54, 0.60, 1.0))
        rim_light_np = self._base.render.attachNewNode(rim_light)
        rim_light_np.setHpr(164.0, -8.0, 0.0)
        self._base.render.setLight(rim_light_np)

        floor_z = self._build_room()
        self._view_root = self._base.render.attachNewNode("instrument-plane-view-root")
        self._plane_root = self._view_root.attachNewNode("instrument-plane-root")
        self._add_plane_shadow(floor_z=floor_z)
        plane = self._load_plane_asset()
        plane.reparentTo(self._plane_root)
        plane.setScale(0.52)
        plane.setPos(-0.78, 1.60, 0.34)

    def render_card(self, *, key: InstrumentAircraftCardKey, destination: Path) -> None:
        from panda3d.core import PNMImage

        projection = instrument_aircraft_card_view_projection(key.view_preset)
        camera = _camera_spec_for_preset(key.view_preset)
        self._base.camLens.setFov(camera.fov_deg)
        self._base.cam.setPos(*camera.camera_pos)
        self._base.cam.lookAt(*camera.look_at)
        self._view_root.setPos(*_view_root_offset_for_preset(key.view_preset))
        self._view_root.setHpr(
            float(projection.view_yaw_deg),
            float(projection.view_pitch_deg),
            float(projection.view_roll_deg),
        )
        self._plane_root.setHpr(-float(key.heading_deg), float(key.pitch_deg), -float(key.bank_deg))

        for _ in range(3):
            self._base.graphicsEngine.renderFrame()

        image = PNMImage()
        ok = self._base.win.getScreenshot(image)
        if not ok:
            raise RuntimeError("Failed to capture Instrument Comprehension card sprite.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        image.write(str(destination))

    def _build_room(self) -> float:
        room, floor_z = self._make_open_room()
        room.reparentTo(self._base.render)
        room.setPos(0.0, 1.75, 0.0)
        return float(floor_z)

    def _load_plane_asset(self):
        asset_path = self._asset_catalog.resolve_path("plane_red")
        if asset_path is not None:
            suffix = asset_path.suffix.lower()
            if suffix in {".glb", ".gltf"}:
                import gltf

                node = gltf.load_model(asset_path)
            else:
                node = self._base.loader.loadModel(str(asset_path))
            if not node.isEmpty():
                return node
        return self._build_fallback_plane()

    def _add_plane_shadow(self, *, floor_z: float) -> None:
        from panda3d.core import CardMaker, TransparencyAttrib

        shadow = CardMaker("instrument-plane-shadow")
        shadow.setFrame(-1.85, 1.85, -0.84, 0.84)
        shadow_np = self._plane_root.attachNewNode(shadow.generate())
        shadow_np.setPos(0.0, 1.60, float(floor_z) + 0.02)
        shadow_np.setHpr(0.0, -90.0, 0.0)
        shadow_np.setColor(0.05, 0.05, 0.06, 0.24)
        shadow_np.setTransparency(TransparencyAttrib.MAlpha)
        shadow_np.setScale(1.0, 1.0, 0.62)
        shadow_np.setTwoSided(True)

    def _make_open_room(self):
        from panda3d.core import CardMaker, NodePath

        root = NodePath("room")

        def add_panel(
            name: str,
            *,
            frame: tuple[float, float, float, float],
            pos: tuple[float, float, float],
            hpr: tuple[float, float, float],
            color: tuple[float, float, float, float],
        ) -> None:
            maker = CardMaker(name)
            maker.setFrame(*frame)
            node = root.attachNewNode(maker.generate())
            node.setPos(*pos)
            node.setHpr(*hpr)
            node.setColor(*color)
            node.setTwoSided(True)

        depth = 11.0
        width = 26.0
        height = 6.4
        add_panel(
            "back",
            frame=(-width / 2.0, width / 2.0, -height / 2.0, height / 2.0),
            pos=(0.0, depth, 0.0),
            hpr=(180.0, 0.0, 0.0),
            color=(0.84, 0.85, 0.87, 1.0),
        )
        floor_z = -height / 2.0
        add_panel(
            "floor",
            frame=(-width / 2.0, width / 2.0, -depth / 2.0, depth / 2.0),
            pos=(0.0, depth / 2.0, floor_z),
            hpr=(0.0, 90.0, 0.0),
            color=(0.66, 0.68, 0.71, 1.0),
        )

        self._add_floor_guides(root=root, width=width, depth=depth, floor_z=floor_z + 0.002)
        return root, floor_z

    def _add_floor_guides(self, *, root, width: float, depth: float, floor_z: float) -> None:
        from panda3d.core import LineSegs

        guides = LineSegs("floor-guides")
        guides.setThickness(1.0)
        guides.setColor(0.52, 0.52, 0.54, 1.0)
        for lane in range(-3, 4):
            x = lane * (width / 8.0)
            guides.moveTo(x, 0.0, floor_z)
            guides.drawTo(x * 0.38, depth, floor_z)
        for row in range(1, 5):
            y = row * (depth / 5.0)
            scale = 1.0 - (row * 0.12)
            guides.moveTo(-(width / 2.0) * scale, y, floor_z)
            guides.drawTo((width / 2.0) * scale, y, floor_z)
        root.attachNewNode(guides.create())

    def _build_fallback_plane(self):
        palette = build_panda_palette(
            body_color=(0.84, 0.18, 0.16, 1.0),
            accent_color=(0.50, 0.05, 0.06, 1.0),
            canopy_color=(0.74, 0.88, 0.95, 0.92),
            engine_color=(0.44, 0.04, 0.05, 1.0),
        )
        return build_panda3d_fixed_wing_model(palette=palette, name="instrument-plane")

    @staticmethod
    def _make_box(*, size: tuple[float, float, float], color: tuple[float, float, float, float]):
        from panda3d.core import CardMaker, NodePath

        sx, sy, sz = size
        root = NodePath("box")

        def add_face(
            name: str,
            frame: tuple[float, float, float, float],
            pos: tuple[float, float, float],
            hpr: tuple[float, float, float],
        ) -> None:
            maker = CardMaker(name)
            maker.setFrame(*frame)
            node = root.attachNewNode(maker.generate())
            node.setPos(*pos)
            node.setHpr(*hpr)
            node.setColor(*color)
            node.setTwoSided(True)

        add_face(
            "top",
            (-sx / 2.0, sx / 2.0, -sy / 2.0, sy / 2.0),
            (0.0, 0.0, sz / 2.0),
            (0.0, -90.0, 0.0),
        )
        add_face(
            "bottom",
            (-sx / 2.0, sx / 2.0, -sy / 2.0, sy / 2.0),
            (0.0, 0.0, -sz / 2.0),
            (0.0, 90.0, 0.0),
        )
        add_face(
            "front",
            (-sx / 2.0, sx / 2.0, -sz / 2.0, sz / 2.0),
            (0.0, sy / 2.0, 0.0),
            (0.0, 0.0, 0.0),
        )
        add_face(
            "back",
            (-sx / 2.0, sx / 2.0, -sz / 2.0, sz / 2.0),
            (0.0, -sy / 2.0, 0.0),
            (180.0, 0.0, 0.0),
        )
        add_face(
            "left",
            (-sy / 2.0, sy / 2.0, -sz / 2.0, sz / 2.0),
            (-sx / 2.0, 0.0, 0.0),
            (90.0, 0.0, 0.0),
        )
        add_face(
            "right",
            (-sy / 2.0, sy / 2.0, -sz / 2.0, sz / 2.0),
            (sx / 2.0, 0.0, 0.0),
            (-90.0, 0.0, 0.0),
        )
        return root

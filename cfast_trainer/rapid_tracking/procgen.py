from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .legacy import (
    RapidTrackingCompoundLayout,
    RapidTrackingLayoutPolicy,
    RapidTrackingScenarioBundle,
    RapidTrackingScenarioGenerator,
    _ellipse_contains_point,
    _readable_layout_evaluation,
    build_rapid_tracking_compound_layout,
    select_scene_seed,
)
from ..rapid_tracking_view import rapid_tracking_seed_unit, track_to_world_xy as rapid_tracking_track_to_world_xy


@dataclass(frozen=True, slots=True)
class RapidTrackingBackdropTerrain:
    feature_id: str
    profile: str
    world_x: float
    world_y: float
    heading_deg: float
    scale_x: float
    scale_y: float
    scale_z: float


def _lerp(a: float, b: float, t: float) -> float:
    return float(a) + ((float(b) - float(a)) * float(t))


def build_distant_terrain_ring(
    *,
    layout: RapidTrackingCompoundLayout,
) -> tuple[RapidTrackingBackdropTerrain, ...]:
    center_wx, center_wy = rapid_tracking_track_to_world_xy(
        track_x=float(layout.compound_center_x),
        track_y=float(layout.compound_center_y),
        path_lateral_bias=float(layout.path_lateral_bias),
    )
    seed = int(layout.seed)
    slot_count = 6
    angle_offset_deg = rapid_tracking_seed_unit(seed=seed, salt="backdrop:angle-offset") * 360.0
    features: list[RapidTrackingBackdropTerrain] = []
    for idx in range(slot_count):
        profile = "mountain" if idx % 3 == 0 else "hill"
        angle_jitter_deg = _lerp(
            -7.0,
            7.0,
            rapid_tracking_seed_unit(seed=seed, salt=f"backdrop:{idx}:angle-jitter"),
        )
        angle_deg = angle_offset_deg + ((360.0 / float(slot_count)) * float(idx)) + angle_jitter_deg
        angle_rad = math.radians(angle_deg)
        radius = (
            _lerp(
                520.0,
                660.0,
                rapid_tracking_seed_unit(seed=seed, salt=f"backdrop:{idx}:radius"),
            )
            if profile == "mountain"
            else _lerp(
                430.0,
                560.0,
                rapid_tracking_seed_unit(seed=seed, salt=f"backdrop:{idx}:radius"),
            )
        )
        scale_x = (
            _lerp(
                118.0,
                168.0,
                rapid_tracking_seed_unit(seed=seed, salt=f"backdrop:{idx}:scale-x"),
            )
            if profile == "mountain"
            else _lerp(
                78.0,
                122.0,
                rapid_tracking_seed_unit(seed=seed, salt=f"backdrop:{idx}:scale-x"),
            )
        )
        scale_y = (
            _lerp(
                92.0,
                138.0,
                rapid_tracking_seed_unit(seed=seed, salt=f"backdrop:{idx}:scale-y"),
            )
            if profile == "mountain"
            else _lerp(
                64.0,
                106.0,
                rapid_tracking_seed_unit(seed=seed, salt=f"backdrop:{idx}:scale-y"),
            )
        )
        scale_z = (
            _lerp(
                34.0,
                56.0,
                rapid_tracking_seed_unit(seed=seed, salt=f"backdrop:{idx}:scale-z"),
            )
            if profile == "mountain"
            else _lerp(
                18.0,
                30.0,
                rapid_tracking_seed_unit(seed=seed, salt=f"backdrop:{idx}:scale-z"),
            )
        )
        features.append(
            RapidTrackingBackdropTerrain(
                feature_id=f"backdrop-{idx}",
                profile=profile,
                world_x=float(center_wx + (math.cos(angle_rad) * radius)),
                world_y=float(center_wy + (math.sin(angle_rad) * radius)),
                heading_deg=float(
                    (
                        angle_deg
                        + _lerp(
                            -18.0,
                            18.0,
                            rapid_tracking_seed_unit(seed=seed, salt=f"backdrop:{idx}:heading"),
                        )
                    )
                    % 360.0
                ),
                scale_x=float(scale_x),
                scale_y=float(scale_y),
                scale_z=float(scale_z),
            )
        )
    return tuple(features)


class TrainingWorldBuilder(ABC):
    @abstractmethod
    def build_layout(self, *, seed: int) -> RapidTrackingCompoundLayout:
        raise NotImplementedError

    @abstractmethod
    def resolve_scene_seed(
        self,
        *,
        session_seed: int,
        layout_policy: RapidTrackingLayoutPolicy | str,
    ) -> int:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class RapidTrackingV1TrainingWorldBuilder(TrainingWorldBuilder):
    layout_policy: RapidTrackingLayoutPolicy | str = RapidTrackingLayoutPolicy.DEFAULT

    def build_layout(self, *, seed: int) -> RapidTrackingCompoundLayout:
        return build_rapid_tracking_compound_layout(seed=int(seed))

    def resolve_scene_seed(
        self,
        *,
        session_seed: int,
        layout_policy: RapidTrackingLayoutPolicy | str | None = None,
    ) -> int:
        policy = self.layout_policy if layout_policy is None else layout_policy
        return int(select_scene_seed(int(session_seed), policy))

    def build_bundle(
        self,
        *,
        session_seed: int,
        layout_policy: RapidTrackingLayoutPolicy | str | None = None,
    ) -> RapidTrackingScenarioBundle:
        scene_seed = self.resolve_scene_seed(
            session_seed=int(session_seed),
            layout_policy=self.layout_policy if layout_policy is None else layout_policy,
        )
        return RapidTrackingScenarioGenerator(seed=scene_seed).build()


__all__ = [
    "RapidTrackingBackdropTerrain",
    "build_distant_terrain_ring",
    "TrainingWorldBuilder",
    "RapidTrackingV1TrainingWorldBuilder",
    "_ellipse_contains_point",
    "_readable_layout_evaluation",
    "build_rapid_tracking_compound_layout",
    "select_scene_seed",
]

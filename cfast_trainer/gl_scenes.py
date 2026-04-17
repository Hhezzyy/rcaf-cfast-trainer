from __future__ import annotations

from dataclasses import dataclass

import pygame

from .auditory_capacity import AuditoryCapacityPayload
from .rapid_tracking import RapidTrackingPayload
from .spatial_integration import SpatialIntegrationPayload
from .trace_test_1 import TraceTest1Payload
from .trace_test_2 import TraceTest2Payload


@dataclass(frozen=True, slots=True)
class AuditoryGlScene:
    world: pygame.Rect
    payload: AuditoryCapacityPayload | None
    time_remaining_s: float | None
    time_fill_ratio: float | None
    frame_dt_s: float = 0.0
    advance_animation: bool = True


@dataclass(frozen=True, slots=True)
class RapidTrackingGlScene:
    world: pygame.Rect
    payload: RapidTrackingPayload | None
    active_phase: bool


@dataclass(frozen=True, slots=True)
class SpatialIntegrationGlScene:
    world: pygame.Rect
    payload: SpatialIntegrationPayload | None


@dataclass(frozen=True, slots=True)
class TraceTest1GlScene:
    world: pygame.Rect
    payload: TraceTest1Payload | None
    practice_mode: bool = False


@dataclass(frozen=True, slots=True)
class TraceTest2GlScene:
    world: pygame.Rect
    payload: TraceTest2Payload | None
    practice_mode: bool = False


GlScene = (
    AuditoryGlScene
    | RapidTrackingGlScene
    | SpatialIntegrationGlScene
    | TraceTest1GlScene
    | TraceTest2GlScene
)


def gl_scene_name(scene: GlScene) -> str:
    if isinstance(scene, AuditoryGlScene):
        return "auditory"
    if isinstance(scene, RapidTrackingGlScene):
        return "rapid_tracking"
    if isinstance(scene, SpatialIntegrationGlScene):
        return "spatial_integration"
    if isinstance(scene, TraceTest1GlScene):
        return "trace_test_1"
    return "trace_test_2"

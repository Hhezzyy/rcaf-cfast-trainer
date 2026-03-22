from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .adaptive_difficulty import difficulty_level_for_ratio, difficulty_profile_for_code
from .clock import Clock
from .content_variants import content_metadata_from_payload
from .cognitive_core import Phase, QuestionEvent, SeededRng, TestSnapshot, clamp01, lerp_int
from .rapid_tracking_view import (
    camera_pose_compat,
    camera_rig_state as rapid_tracking_camera_rig_state,
    estimated_target_world_z,
    target_projection as rapid_tracking_target_projection,
    track_to_world_xy,
)


@dataclass(frozen=True, slots=True)
class RapidTrackingConfig:
    # Candidate guide (page 12) describes ~16 minutes including instructions.
    practice_duration_s: float = 60.0
    scored_duration_s: float = 14.0 * 60.0
    tick_hz: float = 120.0

    # Direct free-look camera behavior.
    camera_limit: float = 5.2
    camera_control_speed: float = 1.52
    camera_control_response_hz: float = 8.6
    camera_yaw_rate_deg_s: float = 96.0
    camera_pitch_rate_deg_s: float = 78.0

    # Disturbance drift while tracking.
    view_limit: float = 1.0

    # HUD behavior: crosshair and target box appear after sustained on-target tracking.
    hud_acquire_s: float = 1.35
    hud_persist_s: float = 2.20

    # Guide-style scoring.
    on_target_radius: float = 0.10
    good_window_error: float = 0.19

    # Corner preview card when targets switch.
    target_preview_s: float = 2.25

    # Camera-box trigger capture.
    capture_box_half_width: float = 0.085
    capture_box_half_height: float = 0.075
    capture_zoom_s: float = 0.48
    capture_cooldown_s: float = 0.42
    capture_flash_s: float = 0.34
    capture_zoom_strength: float = 0.85
    capture_hold_bonus_interval_s: float = 0.25
    capture_hold_blend_response_hz: float = 8.0
    default_h_fov_deg: float = 130.0
    hold_zoom_h_fov_deg: float = 39.0


RAPID_TRACKING_TARGET_KIND_ORDER: tuple[str, ...] = (
    "soldier",
    "building",
    "truck",
    "helicopter",
    "jet",
)

RAPID_TRACKING_CHALLENGE_ORDER: tuple[str, ...] = (
    "lock_quality",
    "handoff_reacquisition",
    "occlusion_recovery",
    "capture_timing",
    "ground_tempo",
    "air_speed",
)


@dataclass(frozen=True, slots=True)
class RapidTrackingTrainingProfile:
    target_kinds: tuple[str, ...] = RAPID_TRACKING_TARGET_KIND_ORDER
    cover_modes: tuple[str, ...] = ("open", "building", "terrain")
    handoff_modes: tuple[str, ...] = ("smooth", "jump")
    turbulence_scale: float = 1.0
    camera_assist_override: float | None = None
    preview_duration_scale: float = 1.0
    capture_box_scale: float = 1.0
    capture_cooldown_scale: float = 1.0
    segment_duration_scale: float = 1.0


@dataclass(frozen=True, slots=True)
class RapidTrackingTrainingSegment:
    label: str
    duration_s: float
    focus_label: str = ""
    active_target_kinds: tuple[str, ...] = RAPID_TRACKING_TARGET_KIND_ORDER
    active_challenges: tuple[str, ...] = RAPID_TRACKING_CHALLENGE_ORDER
    profile: RapidTrackingTrainingProfile = RapidTrackingTrainingProfile()


@dataclass(frozen=True, slots=True)
class RapidTrackingDriftVector:
    vx: float
    vy: float
    duration_s: float


@dataclass(frozen=True, slots=True)
class RapidTrackingAnchor:
    anchor_id: str
    family: str
    variant: str
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class RapidTrackingFoliageCluster:
    cluster_id: str
    asset_id: str
    fallback: str
    x: float
    y: float
    radius: float
    count: int
    scale: float


@dataclass(frozen=True, slots=True)
class RapidTrackingCompoundLayout:
    seed: int
    orbit_direction: int
    orbit_phase_offset: float
    orbit_radius_scale: float
    altitude_bias: float
    ridge_phase: float
    ridge_amp_scale: float
    path_lateral_bias: float
    compound_center_x: float
    compound_center_y: float
    building_anchors: tuple[RapidTrackingAnchor, ...]
    patrol_anchors: tuple[RapidTrackingAnchor, ...]
    road_anchors: tuple[RapidTrackingAnchor, ...]
    helicopter_anchors: tuple[RapidTrackingAnchor, ...]
    jet_anchors: tuple[RapidTrackingAnchor, ...]
    ridge_anchors: tuple[RapidTrackingAnchor, ...]
    shrub_clusters: tuple[RapidTrackingFoliageCluster, ...]
    tree_clusters: tuple[RapidTrackingFoliageCluster, ...]
    forest_clusters: tuple[RapidTrackingFoliageCluster, ...]

    def all_anchors(self) -> tuple[RapidTrackingAnchor, ...]:
        return (
            *self.building_anchors,
            *self.patrol_anchors,
            *self.road_anchors,
            *self.helicopter_anchors,
            *self.jet_anchors,
            *self.ridge_anchors,
        )

    def anchor(self, anchor_id: str) -> RapidTrackingAnchor:
        token = str(anchor_id).strip().lower()
        for anchor in self.all_anchors():
            if anchor.anchor_id == token:
                return anchor
        raise KeyError(f"unknown rapid-tracking anchor: {anchor_id}")


@dataclass(frozen=True, slots=True)
class RapidTrackingSceneSegment:
    kind: str  # "soldier" | "building" | "truck" | "helicopter" | "jet"
    variant: str
    route_kind: str
    start_anchor_id: str
    end_anchor_id: str
    duration_s: float
    handoff: str = "smooth"  # "smooth" | "jump"
    cover_mode: str = "open"  # "open" | "building" | "terrain"
    focus_anchor_id: str = ""
    speed_profile: str = ""
    arc_x: float = 0.0
    arc_y: float = 0.0


@dataclass(frozen=True, slots=True)
class RapidTrackingPayload:
    target_rel_x: float
    target_rel_y: float
    target_world_x: float
    target_world_y: float
    focus_world_x: float
    focus_world_y: float
    reticle_x: float
    reticle_y: float
    camera_x: float
    camera_y: float
    camera_yaw_deg: float
    camera_pitch_deg: float
    target_vx: float
    target_vy: float
    target_visible: bool
    target_cover_state: str
    target_is_moving: bool
    target_kind: str
    target_variant: str
    target_handoff_mode: str
    difficulty: float
    difficulty_tier: str
    session_seed: int
    camera_assist_strength: float
    turbulence_strength: float
    target_time_to_switch_s: float
    target_switch_preview_s: float
    obscured_time_left_s: float
    phase_elapsed_s: float
    mean_error: float
    rms_error: float
    on_target_s: float
    on_target_ratio: float
    obscured_tracking_ratio: float
    hud_visible: bool
    lock_progress: float
    terrain_ridge_y: float
    scene_progress: float
    capture_box_half_width: float
    capture_box_half_height: float
    target_in_capture_box: bool
    capture_zoom: float
    capture_points: int
    capture_hits: int
    capture_attempts: int
    capture_accuracy: float
    capture_feedback: str
    capture_flash_s: float
    active_target_kinds: tuple[str, ...] = RAPID_TRACKING_TARGET_KIND_ORDER
    active_challenges: tuple[str, ...] = RAPID_TRACKING_CHALLENGE_ORDER
    focus_label: str = ""
    segment_label: str = ""
    segment_index: int = 1
    segment_total: int = 1
    segment_time_remaining_s: float = 0.0


@dataclass(frozen=True, slots=True)
class RapidTrackingSummary:
    attempted: int
    correct: int
    accuracy: float
    duration_s: float
    throughput_per_min: float
    mean_error: float
    rms_error: float
    on_target_s: float
    on_target_ratio: float
    obscured_time_s: float
    obscured_tracking_ratio: float
    moving_target_ratio: float
    total_score: float
    max_score: float
    score_ratio: float
    capture_points: int
    capture_hits: int
    capture_attempts: int
    capture_accuracy: float
    capture_max_points: int
    capture_score_ratio: float
    overshoot_count: int
    reversal_count: int


@dataclass(frozen=True, slots=True)
class RapidTrackingDifficultyProfile:
    tier: str
    scene_script: tuple[RapidTrackingSceneSegment, ...]
    practice_script: tuple[RapidTrackingSceneSegment, ...]
    loop_limit: int
    practice_loop_limit: int
    duration_scale: float
    turbulence_strength: float
    drift_duration_scale: float
    camera_assist_strength: float


@dataclass(frozen=True, slots=True)
class RapidTrackingScenarioBundle:
    layout: RapidTrackingCompoundLayout
    ground_script: tuple[RapidTrackingSceneSegment, ...]
    practice_script: tuple[RapidTrackingSceneSegment, ...]
    scored_script: tuple[RapidTrackingSceneSegment, ...]


def rapid_tracking_target_label(*, kind: str, variant: str = "") -> str:
    target_kind = str(kind).strip().lower()
    target_variant = str(variant).strip().lower()
    if target_kind == "building":
        if target_variant == "garage":
            return "GARAGE"
        if target_variant == "tower":
            return "TOWER"
        return "HANGAR"
    if target_kind == "soldier":
        return "SOLDIER"
    if target_kind == "truck":
        return "VEHICLE"
    if target_kind == "helicopter":
        return "HELICOPTER"
    if target_kind == "jet":
        return "JET"
    return target_kind.upper()


def rapid_tracking_target_description(*, kind: str, variant: str = "") -> str:
    target_kind = str(kind).strip().lower()
    target_variant = str(variant).strip().lower()
    if target_kind == "building":
        if target_variant == "garage":
            return "TRACK THE GARAGE UNTIL THE NEXT TARGET EMERGES"
        if target_variant == "tower":
            return "TRACK THE TOWER DURING THE BUILDING HANDOFF"
        return "TRACK THE HANGAR DURING THE BUILDING HANDOFF"
    if target_kind == "soldier":
        return "FOOT PATROL MOVING BETWEEN STRUCTURES AND COVER"
    if target_kind == "truck":
        return "ROAD VEHICLE MOVING THROUGH THE COMPOUND"
    if target_kind == "helicopter":
        return "ROTARY-WING AIRCRAFT CUTTING ACROSS THE COMPOUND"
    if target_kind == "jet":
        return "FAST FIXED-WING PASS THROUGH THE SCAN"
    return "TARGET"


def rapid_tracking_target_cue(*, kind: str, variant: str = "", handoff_mode: str = "") -> str:
    target_kind = str(kind).strip().lower()
    mode = str(handoff_mode).strip().lower()
    if target_kind == "soldier":
        cue = "FOOT PATROL / SLOW"
    elif target_kind == "building":
        cue = "BUILDING HANDOFF"
    elif target_kind == "truck":
        cue = "ROAD MOVEMENT / FAST"
    elif target_kind == "helicopter":
        cue = "AIR HANDOFF / FASTER"
    elif target_kind == "jet":
        cue = "FAST AIR PASS / FASTEST"
    else:
        cue = "TRACK TARGET"
    if mode == "jump":
        return f"{cue}  JUMP SWITCH"
    return f"{cue}  SMOOTH SWITCH"


def rapid_tracking_seed_unit(*, seed: int, salt: str) -> float:
    total = int(seed) & 0xFFFFFFFF
    for ch in str(salt):
        total ^= ord(ch) & 0xFFFFFFFF
        total = (total * 16777619) & 0xFFFFFFFF
        total ^= (total >> 13) & 0xFFFFFFFF
    return float(total & 0xFFFFFFFF) / float(0xFFFFFFFF)


def score_window(*, mean_error: float, good_window_error: float) -> float:
    threshold = max(0.001, float(good_window_error))
    value = max(0.0, float(mean_error))
    if value <= threshold:
        return 1.0
    fail_at = threshold * 2.0
    if value >= fail_at:
        return 0.0
    return (fail_at - value) / threshold


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _lerp(a: float, b: float, t: float) -> float:
    return a + ((b - a) * t)


def _rotate_point(x: float, y: float, angle_rad: float) -> tuple[float, float]:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return ((x * c) - (y * s), (x * s) + (y * c))


def _anchor_lookup(layout: RapidTrackingCompoundLayout) -> dict[str, RapidTrackingAnchor]:
    return {anchor.anchor_id: anchor for anchor in layout.all_anchors()}


class RapidTrackingDriftGenerator:
    """Deterministic drift profile for continuous moving-view tracking."""

    def __init__(self, *, seed: int) -> None:
        self._rng = SeededRng(seed)

    def next_vector(self, *, difficulty: float) -> RapidTrackingDriftVector:
        d = clamp01(difficulty)
        mag_lo = lerp_int(8, 14, d) / 100.0
        mag_hi = lerp_int(18, 30, d) / 100.0
        interval_lo = max(0.28, 0.86 - (0.32 * d))
        interval_hi = max(interval_lo + 0.06, 1.52 - (0.54 * d))

        magnitude = self._rng.uniform(mag_lo, mag_hi)
        angle = self._rng.uniform(0.0, math.tau)
        duration_s = self._rng.uniform(interval_lo, interval_hi)

        return RapidTrackingDriftVector(
            vx=math.cos(angle) * magnitude,
            vy=math.sin(angle) * magnitude,
            duration_s=duration_s,
        )


class RapidTrackingScenarioGenerator:
    _BASE_BUILDINGS: tuple[tuple[str, str, float, float], ...] = (
        ("hangar-a", "hangar", -1.38, 0.58),
        ("garage-a", "garage", -0.56, 0.36),
        ("tower-a", "tower", 0.46, 0.18),
        ("hangar-b", "hangar", 1.14, 0.28),
        ("tower-b", "tower", 1.38, -0.08),
        ("hangar-c", "hangar", 0.14, 0.54),
        ("garage-b", "garage", -1.18, 0.14),
        ("tower-c", "tower", -0.28, -0.02),
    )
    _BASE_PATROLS: tuple[tuple[str, float, float], ...] = (
        ("patrol-a", -2.18, 0.86),
        ("patrol-b", -1.54, 0.72),
        ("patrol-c", -0.78, 0.60),
        ("patrol-d", 0.06, 0.62),
        ("patrol-e", 0.92, 0.68),
        ("patrol-f", 1.72, 0.56),
        ("patrol-g", 1.26, 0.34),
        ("patrol-h", -1.04, 0.82),
    )
    _BASE_ROADS: tuple[tuple[str, float, float], ...] = (
        ("road-west-a", -1.82, 0.60),
        ("road-mid-a", -0.62, 0.60),
        ("road-east-a", 0.62, 0.60),
        ("road-east-b", 1.72, 0.60),
        ("road-west-v0", -1.18, 0.18),
        ("road-west-v1", -1.18, 0.82),
        ("road-east-v0", 1.28, 0.20),
        ("road-east-v1", 1.28, 0.80),
    )
    _BASE_HELOS: tuple[tuple[str, float, float], ...] = (
        ("helo-a", -1.78, 0.12),
        ("helo-b", -0.76, -0.06),
        ("helo-c", 0.18, -0.12),
        ("helo-d", 1.26, -0.04),
        ("helo-e", 1.82, 0.16),
        ("helo-f", 0.74, 0.20),
        ("helo-g", -0.30, 0.18),
    )
    _BASE_JETS: tuple[tuple[str, float, float], ...] = (
        ("jet-a", 1.82, -0.30),
        ("jet-b", 0.86, -0.40),
        ("jet-c", -0.18, -0.54),
        ("jet-d", -1.52, -0.72),
        ("jet-e", -2.22, -0.90),
        ("jet-f", 0.32, -0.22),
        ("jet-g", 1.44, -0.12),
    )
    _BASE_RIDGES: tuple[tuple[str, float, float], ...] = (
        ("ridge-a", -0.92, 0.26),
        ("ridge-b", 0.08, 0.30),
        ("ridge-c", 1.04, 0.18),
    )

    _GROUND_ROUTES: tuple[tuple[str, str], ...] = (
        ("road-west-a", "road-mid-a"),
        ("road-mid-a", "road-east-a"),
        ("road-east-a", "road-east-b"),
        ("road-east-b", "road-east-a"),
        ("road-east-a", "road-mid-a"),
        ("road-mid-a", "road-west-a"),
        ("road-west-v0", "road-west-v1"),
        ("road-west-v1", "road-west-v0"),
        ("road-east-v0", "road-east-v1"),
        ("road-east-v1", "road-east-v0"),
    )
    _PATROL_ROUTES: tuple[tuple[str, str], ...] = (
        ("patrol-a", "patrol-b"),
        ("patrol-b", "patrol-c"),
        ("patrol-c", "patrol-d"),
        ("patrol-d", "patrol-e"),
        ("patrol-e", "patrol-f"),
        ("patrol-f", "patrol-g"),
        ("patrol-h", "patrol-b"),
    )
    _HELO_ROUTES: tuple[tuple[str, str], ...] = (
        ("helo-a", "helo-d"),
        ("helo-b", "helo-e"),
        ("helo-c", "helo-f"),
        ("helo-g", "helo-e"),
    )
    _JET_ROUTES: tuple[tuple[str, str], ...] = (
        ("jet-a", "jet-d"),
        ("jet-g", "jet-e"),
        ("jet-f", "jet-e"),
    )
    _GROUND_TEMPLATES: tuple[tuple[str, str, str, str], ...] = (
        ("soldier", "open", "smooth", ""),
        ("building", "building", "smooth", ""),
        ("truck", "open", "smooth", ""),
        ("soldier", "terrain", "smooth", ""),
        ("building", "building", "jump", ""),
        ("truck", "open", "smooth", ""),
        ("soldier", "open", "smooth", ""),
    )
    _PRACTICE_TEMPLATES: tuple[tuple[str, str, str, str], ...] = (
        ("soldier", "open", "smooth", ""),
        ("building", "building", "smooth", ""),
        ("truck", "open", "smooth", ""),
        ("truck", "terrain", "smooth", ""),
        ("helicopter", "open", "smooth", ""),
        ("building", "building", "jump", ""),
        ("jet", "open", "smooth", ""),
        ("soldier", "open", "smooth", ""),
    )
    _SCORED_TEMPLATES: tuple[tuple[str, str, str, str], ...] = (
        ("soldier", "open", "smooth", ""),
        ("building", "building", "smooth", ""),
        ("truck", "open", "smooth", ""),
        ("truck", "terrain", "smooth", ""),
        ("helicopter", "open", "smooth", ""),
        ("building", "building", "jump", ""),
        ("soldier", "open", "smooth", ""),
        ("jet", "open", "smooth", ""),
        ("truck", "open", "smooth", ""),
        ("building", "building", "smooth", ""),
        ("helicopter", "terrain", "smooth", ""),
        ("jet", "open", "jump", ""),
        ("soldier", "terrain", "smooth", ""),
        ("building", "building", "smooth", ""),
        ("truck", "open", "smooth", ""),
        ("helicopter", "open", "smooth", ""),
    )

    def __init__(self, *, seed: int) -> None:
        self._seed = int(seed)

    def build(self) -> RapidTrackingScenarioBundle:
        layout = self._build_layout()
        ground_script = self._build_schedule(layout=layout, phase_key="ground", templates=self._GROUND_TEMPLATES)
        practice_script = self._build_schedule(
            layout=layout,
            phase_key="practice",
            templates=self._PRACTICE_TEMPLATES,
        )
        scored_script = self._build_schedule(layout=layout, phase_key="scored", templates=self._SCORED_TEMPLATES)
        return RapidTrackingScenarioBundle(
            layout=layout,
            ground_script=ground_script,
            practice_script=practice_script,
            scored_script=scored_script,
        )

    def build_training_script(
        self,
        *,
        layout: RapidTrackingCompoundLayout,
        phase_key: str,
        profile: RapidTrackingTrainingProfile,
        salt: str = "",
    ) -> tuple[RapidTrackingSceneSegment, ...]:
        token = str(phase_key).strip().lower()
        if token == "ground":
            templates = self._GROUND_TEMPLATES
        elif token == "practice":
            templates = self._PRACTICE_TEMPLATES
        else:
            templates = self._SCORED_TEMPLATES
        allowed_kinds = {str(kind).strip().lower() for kind in profile.target_kinds}
        allowed_cover = {str(mode).strip().lower() for mode in profile.cover_modes}
        allowed_handoffs = {str(mode).strip().lower() for mode in profile.handoff_modes}
        filtered = tuple(
            template
            for template in templates
            if str(template[0]).strip().lower() in allowed_kinds
            and str(template[1]).strip().lower() in allowed_cover
            and str(template[2]).strip().lower() in allowed_handoffs
        )
        active_templates = filtered or templates
        return self._build_schedule(
            layout=layout,
            phase_key=f"{token}:{salt}",
            templates=active_templates,
        )

    def _rng(self, *, salt: str) -> SeededRng:
        total = self._seed ^ 0x45D9F3B
        for ch in salt:
            total ^= ord(ch) & 0xFFFFFFFF
            total = (total * 1103515245 + 12345) & 0x7FFFFFFF
        return SeededRng(total or 1)

    def _transform_anchor(
        self,
        *,
        family: str,
        variant: str,
        anchor_id: str,
        x: float,
        y: float,
        rotation_rad: float,
        scale: float,
        jitter: float,
        rng: SeededRng,
    ) -> RapidTrackingAnchor:
        rx, ry = _rotate_point(x * scale, y * scale, rotation_rad)
        jitter_x = rng.uniform(-jitter, jitter)
        jitter_y = rng.uniform(-jitter, jitter)
        if family in {"road", "ridge"}:
            jitter_y *= 0.5
        if family == "jet":
            jitter_y *= 1.3
        return RapidTrackingAnchor(
            anchor_id=str(anchor_id).strip().lower(),
            family=str(family),
            variant=str(variant),
            x=_clamp(rx + jitter_x, -2.35, 2.05),
            y=_clamp(ry + jitter_y, -0.98, 0.92),
        )

    @staticmethod
    def _compound_center(
        *,
        buildings: tuple[RapidTrackingAnchor, ...],
        roads: tuple[RapidTrackingAnchor, ...],
        patrols: tuple[RapidTrackingAnchor, ...],
    ) -> tuple[float, float]:
        anchors = (
            *buildings,
            *roads[:4],
            *patrols[:4],
        )
        if not anchors:
            return (0.0, 0.28)
        return (
            sum(anchor.x for anchor in anchors) / len(anchors),
            sum(anchor.y for anchor in anchors) / len(anchors),
        )

    def _build_shrub_clusters(
        self,
        *,
        buildings: tuple[RapidTrackingAnchor, ...],
        roads: tuple[RapidTrackingAnchor, ...],
    ) -> tuple[RapidTrackingFoliageCluster, ...]:
        rng = self._rng(salt="foliage:shrubs")
        clusters: list[RapidTrackingFoliageCluster] = []
        bases = (*buildings[:4], *roads[:4])
        for idx, base in enumerate(bases):
            angle = rng.uniform(0.0, math.tau)
            dist = rng.uniform(0.08, 0.20)
            x = _clamp(base.x + (math.cos(angle) * dist), -2.35, 2.05)
            y = _clamp(base.y + (math.sin(angle) * dist), -0.98, 0.92)
            clusters.append(
                RapidTrackingFoliageCluster(
                    cluster_id=f"shrub-{idx}",
                    asset_id="shrubs_low_cluster",
                    fallback="shrub",
                    x=x,
                    y=y,
                    radius=rng.uniform(0.07, 0.15),
                    count=3 + int(rng.uniform(0.0, 3.0)),
                    scale=rng.uniform(0.86, 1.16),
                )
            )
        return tuple(clusters)

    def _build_tree_clusters(
        self,
        *,
        patrols: tuple[RapidTrackingAnchor, ...],
        ridges: tuple[RapidTrackingAnchor, ...],
        compound_center_x: float,
        compound_center_y: float,
    ) -> tuple[RapidTrackingFoliageCluster, ...]:
        rng = self._rng(salt="foliage:trees")
        clusters: list[RapidTrackingFoliageCluster] = []
        bases = (*ridges, *patrols[::2])
        for idx, base in enumerate(bases):
            outward_x = base.x - compound_center_x
            outward_y = base.y - compound_center_y
            outward_len = math.hypot(outward_x, outward_y)
            if outward_len <= 1e-6:
                outward_x, outward_y = (0.0, 1.0)
            else:
                outward_x /= outward_len
                outward_y /= outward_len
            offset = rng.uniform(0.10, 0.24)
            x = _clamp(base.x + (outward_x * offset), -2.35, 2.05)
            y = _clamp(base.y + (outward_y * offset), -0.98, 0.92)
            clusters.append(
                RapidTrackingFoliageCluster(
                    cluster_id=f"tree-{idx}",
                    asset_id="trees_field_cluster",
                    fallback="trees",
                    x=x,
                    y=y,
                    radius=rng.uniform(0.12, 0.22),
                    count=3 + int(rng.uniform(0.0, 3.0)),
                    scale=rng.uniform(1.04, 1.52),
                )
            )
        return tuple(clusters)

    def _build_forest_clusters(
        self,
        *,
        compound_center_x: float,
        compound_center_y: float,
    ) -> tuple[RapidTrackingFoliageCluster, ...]:
        rng = self._rng(salt="foliage:forest")
        clusters: list[RapidTrackingFoliageCluster] = []
        phase = rng.uniform(0.0, math.tau)
        for idx in range(5):
            ang = phase + (idx * (math.tau / 5.0))
            radius = rng.uniform(1.48, 2.10)
            x = _clamp(compound_center_x + (math.cos(ang) * radius), -2.35, 2.05)
            y = _clamp(compound_center_y + (math.sin(ang) * radius * 0.72), -0.98, 0.92)
            clusters.append(
                RapidTrackingFoliageCluster(
                    cluster_id=f"forest-{idx}",
                    asset_id="forest_canopy_patch",
                    fallback="forest",
                    x=x,
                    y=y,
                    radius=rng.uniform(0.24, 0.40),
                    count=5 + int(rng.uniform(0.0, 4.0)),
                    scale=rng.uniform(1.84, 2.62),
                )
            )
        return tuple(clusters)

    def _build_layout(self) -> RapidTrackingCompoundLayout:
        rng = self._rng(salt="layout")
        rotation_rad = rng.uniform(-0.10, 0.10)
        scale = rng.uniform(0.92, 1.08)
        orbit_direction = -1 if rng.random() < 0.5 else 1
        orbit_phase_offset = rng.uniform(0.0, math.tau)
        orbit_radius_scale = rng.uniform(0.92, 1.12)
        altitude_bias = rng.uniform(-1.0, 1.3)
        ridge_phase = rng.uniform(-math.pi, math.pi)
        ridge_amp_scale = rng.uniform(0.88, 1.18)
        path_lateral_bias = rng.uniform(-0.22, 0.22)

        buildings = tuple(
            self._transform_anchor(
                family="building",
                variant=variant,
                anchor_id=anchor_id,
                x=x,
                y=y,
                rotation_rad=rotation_rad,
                scale=scale,
                jitter=0.08,
                rng=rng,
            )
            for anchor_id, variant, x, y in self._BASE_BUILDINGS
        )
        patrols = tuple(
            self._transform_anchor(
                family="patrol",
                variant="patrol",
                anchor_id=anchor_id,
                x=x,
                y=y,
                rotation_rad=rotation_rad,
                scale=scale,
                jitter=0.06,
                rng=rng,
            )
            for anchor_id, x, y in self._BASE_PATROLS
        )
        roads = tuple(
            self._transform_anchor(
                family="road",
                variant="road",
                anchor_id=anchor_id,
                x=x,
                y=y,
                rotation_rad=rotation_rad,
                scale=scale,
                jitter=0.04,
                rng=rng,
            )
            for anchor_id, x, y in self._BASE_ROADS
        )
        helos = tuple(
            self._transform_anchor(
                family="helicopter",
                variant="air",
                anchor_id=anchor_id,
                x=x,
                y=y,
                rotation_rad=rotation_rad,
                scale=scale,
                jitter=0.08,
                rng=rng,
            )
            for anchor_id, x, y in self._BASE_HELOS
        )
        jets = tuple(
            self._transform_anchor(
                family="jet",
                variant="air",
                anchor_id=anchor_id,
                x=x,
                y=y,
                rotation_rad=rotation_rad,
                scale=scale,
                jitter=0.10,
                rng=rng,
            )
            for anchor_id, x, y in self._BASE_JETS
        )
        ridges = tuple(
            self._transform_anchor(
                family="ridge",
                variant="ridge",
                anchor_id=anchor_id,
                x=x,
                y=y,
                rotation_rad=rotation_rad,
                scale=scale,
                jitter=0.03,
                rng=rng,
            )
            for anchor_id, x, y in self._BASE_RIDGES
        )
        compound_center_x, compound_center_y = self._compound_center(
            buildings=buildings,
            roads=roads,
            patrols=patrols,
        )
        shrub_clusters = self._build_shrub_clusters(
            buildings=buildings,
            roads=roads,
        )
        tree_clusters = self._build_tree_clusters(
            patrols=patrols,
            ridges=ridges,
            compound_center_x=compound_center_x,
            compound_center_y=compound_center_y,
        )
        forest_clusters = self._build_forest_clusters(
            compound_center_x=compound_center_x,
            compound_center_y=compound_center_y,
        )
        return RapidTrackingCompoundLayout(
            seed=self._seed,
            orbit_direction=orbit_direction,
            orbit_phase_offset=orbit_phase_offset,
            orbit_radius_scale=orbit_radius_scale,
            altitude_bias=altitude_bias,
            ridge_phase=ridge_phase,
            ridge_amp_scale=ridge_amp_scale,
            path_lateral_bias=path_lateral_bias,
            compound_center_x=compound_center_x,
            compound_center_y=compound_center_y,
            building_anchors=buildings,
            patrol_anchors=patrols,
            road_anchors=roads,
            helicopter_anchors=helos,
            jet_anchors=jets,
            ridge_anchors=ridges,
            shrub_clusters=shrub_clusters,
            tree_clusters=tree_clusters,
            forest_clusters=forest_clusters,
        )

    def _build_schedule(
        self,
        *,
        layout: RapidTrackingCompoundLayout,
        phase_key: str,
        templates: tuple[tuple[str, str, str, str], ...],
    ) -> tuple[RapidTrackingSceneSegment, ...]:
        rng = self._rng(salt=f"schedule:{phase_key}")
        anchors = _anchor_lookup(layout)
        segments: list[RapidTrackingSceneSegment] = []
        previous_end = layout.patrol_anchors[0].anchor_id
        used_buildings: list[str] = []

        for idx, (kind, cover_mode, handoff, _reserved) in enumerate(templates):
            if kind == "building":
                building = self._nearest_building_anchor(
                    layout=layout,
                    anchors=anchors,
                    near_anchor_id=previous_end,
                    used_recent=tuple(used_buildings[-2:]),
                )
                used_buildings.append(building.anchor_id)
                segments.append(
                    RapidTrackingSceneSegment(
                        kind="building",
                        variant=building.variant,
                        route_kind="building_hold",
                        start_anchor_id=building.anchor_id,
                        end_anchor_id=building.anchor_id,
                        duration_s=1.35 + (0.16 * rapid_tracking_seed_unit(seed=self._seed, salt=f"building:{idx}")),
                        handoff=handoff,
                        cover_mode="building",
                        focus_anchor_id=building.anchor_id,
                        speed_profile="building",
                        arc_x=0.0,
                        arc_y=0.0,
                    )
                )
                previous_end = building.anchor_id
                continue

            if kind == "soldier":
                start_anchor_id, end_anchor_id = self._pick_route(
                    rng=rng,
                    route_pairs=self._PATROL_ROUTES,
                    previous_end=previous_end,
                )
                arc_x = rng.uniform(-0.20, 0.20)
                arc_y = rng.uniform(0.04, 0.12)
                duration_s = rng.uniform(9.0, 10.4)
                variant = "target"
                route_kind = "foot_patrol"
            elif kind == "truck":
                start_anchor_id, end_anchor_id = self._pick_route(
                    rng=rng,
                    route_pairs=self._GROUND_ROUTES,
                    previous_end=previous_end,
                )
                arc_x = 0.0
                arc_y = 0.0
                duration_s = rng.uniform(3.4, 4.2)
                variant = "olive"
                route_kind = "road_run"
            elif kind == "helicopter":
                start_anchor_id, end_anchor_id = self._pick_route(
                    rng=rng,
                    route_pairs=self._HELO_ROUTES,
                    previous_end=previous_end,
                )
                arc_x = rng.uniform(0.12, 0.28) * (-1.0 if rng.random() < 0.5 else 1.0)
                arc_y = rng.uniform(-0.20, 0.12)
                duration_s = rng.uniform(3.1, 3.9)
                variant = "green"
                route_kind = "helo_arc"
            else:
                start_anchor_id, end_anchor_id = self._pick_route(
                    rng=rng,
                    route_pairs=self._JET_ROUTES,
                    previous_end=previous_end,
                )
                arc_x = rng.uniform(-0.34, -0.14)
                arc_y = rng.uniform(-0.24, -0.10)
                duration_s = rng.uniform(2.2, 2.9)
                variant = "red" if rng.random() < 0.5 else "yellow"
                route_kind = "jet_pass"

            focus_anchor_id = ""
            if cover_mode == "terrain":
                focus_anchor_id = layout.ridge_anchors[idx % max(1, len(layout.ridge_anchors))].anchor_id

            segments.append(
                RapidTrackingSceneSegment(
                    kind=kind,
                    variant=variant,
                    route_kind=route_kind,
                    start_anchor_id=start_anchor_id,
                    end_anchor_id=end_anchor_id,
                    duration_s=duration_s,
                    handoff=handoff,
                    cover_mode=cover_mode,
                    focus_anchor_id=focus_anchor_id,
                    speed_profile=kind,
                    arc_x=arc_x,
                    arc_y=arc_y,
                )
            )
            previous_end = end_anchor_id

        return tuple(segments)

    @staticmethod
    def _pick_route(
        *,
        rng: SeededRng,
        route_pairs: tuple[tuple[str, str], ...],
        previous_end: str,
    ) -> tuple[str, str]:
        matching = [pair for pair in route_pairs if pair[0] == previous_end]
        if matching:
            return rng.choice(tuple(matching))
        return rng.choice(route_pairs)

    @staticmethod
    def _nearest_building_anchor(
        *,
        layout: RapidTrackingCompoundLayout,
        anchors: dict[str, RapidTrackingAnchor],
        near_anchor_id: str,
        used_recent: tuple[str, ...],
    ) -> RapidTrackingAnchor:
        ref = anchors.get(str(near_anchor_id).strip().lower())
        buildings = list(layout.building_anchors)
        if ref is None:
            return buildings[0]

        def score(anchor: RapidTrackingAnchor) -> tuple[float, float]:
            dx = anchor.x - ref.x
            dy = anchor.y - ref.y
            penalty = 0.0 if anchor.anchor_id not in used_recent else 0.35
            return ((dx * dx) + (dy * dy) + penalty, abs(dx))

        return min(buildings, key=score)


def build_rapid_tracking_compound_layout(*, seed: int) -> RapidTrackingCompoundLayout:
    return RapidTrackingScenarioGenerator(seed=seed).build().layout


class RapidTrackingEngine:
    """Continuous eye-hand tracking in a seeded helicopter-over-compound surveillance scene."""

    _MAX_UPDATE_DT_S = 0.50

    def __init__(
        self,
        *,
        clock: Clock,
        seed: int,
        difficulty: float = 0.5,
        config: RapidTrackingConfig | None = None,
        title: str = "Rapid Tracking",
        practice_segments: Sequence[RapidTrackingTrainingSegment] | None = None,
        scored_segments: Sequence[RapidTrackingTrainingSegment] | None = None,
    ) -> None:
        cfg = config or RapidTrackingConfig()
        if cfg.tick_hz <= 0.0:
            raise ValueError("tick_hz must be > 0")
        if cfg.practice_duration_s < 0.0:
            raise ValueError("practice_duration_s must be >= 0")
        if cfg.scored_duration_s <= 0.0:
            raise ValueError("scored_duration_s must be > 0")
        if cfg.camera_yaw_rate_deg_s <= 0.0:
            raise ValueError("camera_yaw_rate_deg_s must be > 0")
        if cfg.camera_pitch_rate_deg_s <= 0.0:
            raise ValueError("camera_pitch_rate_deg_s must be > 0")
        if cfg.hud_acquire_s <= 0.0:
            raise ValueError("hud_acquire_s must be > 0")
        if cfg.hud_persist_s <= 0.0:
            raise ValueError("hud_persist_s must be > 0")

        self._clock = clock
        self._seed = int(seed)
        self._difficulty = clamp01(difficulty)
        self._cfg = cfg
        self._tick_dt = 1.0 / float(cfg.tick_hz)
        self._rng = SeededRng(self._seed)
        self._drift_gen = RapidTrackingDriftGenerator(seed=self._seed + 41)
        self._scenario_generator = RapidTrackingScenarioGenerator(seed=self._seed)
        self._scenario = self._scenario_generator.build()
        self._compound_layout = self._scenario.layout
        self._anchor_lookup = _anchor_lookup(self._compound_layout)
        self._SCENE_SCRIPT = self._scenario.scored_script
        self._LOW_SCENE_SCRIPT = self._scenario.ground_script
        self._title = str(title)
        self._events: list[QuestionEvent] = []
        self._custom_segment_layout = practice_segments is not None or scored_segments is not None
        self._practice_segments = self._normalize_segments(practice_segments)
        self._scored_segments = self._normalize_segments(scored_segments)

        self._phase = Phase.INSTRUCTIONS
        self._phase_started_at_s = self._clock.now()
        self._last_update_at_s = self._phase_started_at_s
        self._accumulator_s = 0.0
        self._sim_elapsed_s = 0.0

        self._control_x = 0.0
        self._control_y = 0.0
        self._look_x = 0.0
        self._look_y = 0.0

        self._reticle_x = 0.0
        self._reticle_y = 0.0

        self._camera_x = 0.0
        self._camera_y = 0.0
        self._camera_yaw_deg = 0.0
        self._camera_pitch_deg = 0.0
        self._drift_x = 0.0
        self._drift_y = 0.0
        self._drift_until_s = 0.0

        self._target_x = 0.0
        self._target_y = 0.0
        self._target_vx = 0.0
        self._target_vy = 0.0
        self._target_is_moving = True
        self._target_kind = "soldier"
        self._target_variant = "target"
        self._target_handoff_mode = "smooth"
        self._target_cover_state = "open"
        self._difficulty_tier = self._difficulty_profile().tier
        self._loop_count = 0

        self._script_index = 0
        self._segment_started_s = 0.0
        self._segment_duration_s = 1.0
        self._segment_start_x = 0.0
        self._segment_start_y = 0.0
        self._segment_control_x = 0.0
        self._segment_control_y = 0.0
        self._segment_end_x = 0.0
        self._segment_end_y = 0.0
        self._segment_route_kind = "foot_patrol"
        self._segment_focus_anchor_id = ""
        self._target_switch_at_s = 0.0
        self._target_preview_until_s = 0.0

        self._terrain_ridge_y = 0.0
        self._target_terrain_occluded = False

        self._lock_hold_s = 0.0
        self._hud_visible_until_s = 0.0
        self._capture_points = 0
        self._capture_hits = 0
        self._capture_attempts = 0
        self._capture_max_points = 0
        self._capture_zoom_until_s = 0.0
        self._capture_hold_active = False
        self._capture_hold_blend = 0.0
        self._capture_hold_bonus_due_at_s = 0.0
        self._capture_last_at_s = -999.0
        self._capture_feedback_until_s = 0.0
        self._capture_feedback = ""

        self._practice_samples = 0
        self._practice_sum_error = 0.0

        self._scored_samples = 0
        self._scored_sum_error = 0.0
        self._scored_sum_error2 = 0.0
        self._scored_on_target_s = 0.0
        self._scored_obscured_s = 0.0
        self._scored_obscured_on_target_s = 0.0
        self._scored_moving_target_s = 0.0
        self._scored_overshoot_count = 0
        self._scored_reversal_count = 0
        self._prev_err_sign_x: int | None = None
        self._prev_err_sign_y: int | None = None

        self._window_elapsed_s = 0.0
        self._window_sum_error = 0.0
        self._window_samples = 0

        self._scored_attempted = 0
        self._scored_correct = 0
        self._scored_total_score = 0.0
        self._scored_max_score = 0.0

        self._active_training_segments: tuple[RapidTrackingTrainingSegment, ...] = ()
        self._active_training_segment_index = 0
        self._current_training_segment: RapidTrackingTrainingSegment | None = None
        self._training_segment_started_elapsed_s = 0.0
        self._active_training_script: tuple[RapidTrackingSceneSegment, ...] = ()

        self._reset_runtime_state(reset_scores=False)

    @property
    def phase(self) -> Phase:
        return self._phase

    @classmethod
    def _normalize_segments(
        cls,
        segments: Sequence[RapidTrackingTrainingSegment] | None,
    ) -> tuple[RapidTrackingTrainingSegment, ...]:
        if not segments:
            return ()
        normalized: list[RapidTrackingTrainingSegment] = []
        for raw in segments:
            label = str(raw.label).strip() or "Rapid Tracking Segment"
            focus_label = str(raw.focus_label).strip() or label
            duration_s = max(15.0, float(raw.duration_s))
            active_target_kinds = tuple(
                kind
                for kind in (
                    str(kind).strip().lower()
                    for kind in tuple(raw.active_target_kinds) or RAPID_TRACKING_TARGET_KIND_ORDER
                )
                if kind in RAPID_TRACKING_TARGET_KIND_ORDER
            ) or RAPID_TRACKING_TARGET_KIND_ORDER
            active_challenges = tuple(
                challenge
                for challenge in (
                    str(challenge).strip().lower()
                    for challenge in tuple(raw.active_challenges) or RAPID_TRACKING_CHALLENGE_ORDER
                )
                if challenge
            ) or RAPID_TRACKING_CHALLENGE_ORDER
            profile = raw.profile
            normalized.append(
                RapidTrackingTrainingSegment(
                    label=label,
                    duration_s=duration_s,
                    focus_label=focus_label,
                    active_target_kinds=active_target_kinds,
                    active_challenges=active_challenges,
                    profile=RapidTrackingTrainingProfile(
                        target_kinds=tuple(
                            kind
                            for kind in (
                                str(kind).strip().lower()
                                for kind in tuple(profile.target_kinds) or active_target_kinds
                            )
                            if kind in RAPID_TRACKING_TARGET_KIND_ORDER
                        )
                        or active_target_kinds,
                        cover_modes=tuple(
                            mode
                            for mode in (
                                str(mode).strip().lower()
                                for mode in tuple(profile.cover_modes) or ("open", "building", "terrain")
                            )
                            if mode in {"open", "building", "terrain"}
                        )
                        or ("open", "building", "terrain"),
                        handoff_modes=tuple(
                            mode
                            for mode in (
                                str(mode).strip().lower()
                                for mode in tuple(profile.handoff_modes) or ("smooth", "jump")
                            )
                            if mode in {"smooth", "jump"}
                        )
                        or ("smooth", "jump"),
                        turbulence_scale=max(0.0, float(profile.turbulence_scale)),
                        camera_assist_override=(
                            None
                            if profile.camera_assist_override is None
                            else clamp01(float(profile.camera_assist_override))
                        ),
                        preview_duration_scale=max(0.2, float(profile.preview_duration_scale)),
                        capture_box_scale=max(0.35, float(profile.capture_box_scale)),
                        capture_cooldown_scale=max(0.2, float(profile.capture_cooldown_scale)),
                        segment_duration_scale=max(0.3, float(profile.segment_duration_scale)),
                    ),
                )
            )
        return tuple(normalized)

    @property
    def seed(self) -> int:
        return self._seed

    def events(self) -> list[QuestionEvent]:
        return list(self._events)

    def can_exit(self) -> bool:
        return self._phase in (
            Phase.INSTRUCTIONS,
            Phase.PRACTICE,
            Phase.PRACTICE_DONE,
            Phase.RESULTS,
        )

    def _training_profile(self) -> RapidTrackingTrainingProfile | None:
        if self._current_training_segment is None:
            return None
        return self._current_training_segment.profile

    def _effective_camera_assist_strength(self, profile: RapidTrackingDifficultyProfile) -> float:
        training = self._training_profile()
        if training is not None and training.camera_assist_override is not None:
            return float(training.camera_assist_override)
        return float(profile.camera_assist_strength)

    def _effective_turbulence_strength(self, profile: RapidTrackingDifficultyProfile) -> float:
        training = self._training_profile()
        scale = 1.0 if training is None else float(training.turbulence_scale)
        return max(0.0, float(profile.turbulence_strength) * scale)

    def _effective_preview_duration_s(self) -> float:
        training = self._training_profile()
        scale = 1.0 if training is None else float(training.preview_duration_scale)
        return max(0.15, float(self._cfg.target_preview_s) * scale)

    def _effective_capture_box_half_width(self) -> float:
        training = self._training_profile()
        scale = 1.0 if training is None else float(training.capture_box_scale)
        return max(0.02, float(self._cfg.capture_box_half_width) * scale)

    def _effective_capture_box_half_height(self) -> float:
        training = self._training_profile()
        scale = 1.0 if training is None else float(training.capture_box_scale)
        return max(0.02, float(self._cfg.capture_box_half_height) * scale)

    def _effective_capture_cooldown_s(self) -> float:
        training = self._training_profile()
        scale = 1.0 if training is None else float(training.capture_cooldown_scale)
        return max(0.05, float(self._cfg.capture_cooldown_s) * scale)

    def _effective_segment_duration_scale(self, profile: RapidTrackingDifficultyProfile) -> float:
        training = self._training_profile()
        scale = 1.0 if training is None else float(training.segment_duration_scale)
        return float(profile.duration_scale) * scale

    def _active_target_kinds(self) -> tuple[str, ...]:
        if self._current_training_segment is None:
            return RAPID_TRACKING_TARGET_KIND_ORDER
        return self._current_training_segment.active_target_kinds

    def _active_challenges(self) -> tuple[str, ...]:
        if self._current_training_segment is None:
            return RAPID_TRACKING_CHALLENGE_ORDER
        return self._current_training_segment.active_challenges

    def _focus_label(self) -> str:
        if self._current_training_segment is None:
            return "Full mixed tracking"
        return self._current_training_segment.focus_label

    def _segment_label(self) -> str:
        if self._current_training_segment is None:
            return self._title
        return self._current_training_segment.label

    def _segment_total(self) -> int:
        return max(1, len(self._active_training_segments))

    def _segment_index(self) -> int:
        return min(self._segment_total(), self._active_training_segment_index + 1)

    def _segment_time_remaining_s(self) -> float:
        if self._current_training_segment is None:
            return max(0.0, float(self.time_remaining_s() or 0.0))
        return max(
            0.0,
            float(self._current_training_segment.duration_s)
            - max(0.0, float(self._sim_elapsed_s - self._training_segment_started_elapsed_s)),
        )

    def _set_active_segments_for_phase(self) -> None:
        if self._phase is Phase.PRACTICE:
            self._active_training_segments = self._practice_segments
        elif self._phase is Phase.SCORED:
            self._active_training_segments = self._scored_segments
        else:
            self._active_training_segments = ()
        self._active_training_segment_index = 0
        self._current_training_segment = None
        self._training_segment_started_elapsed_s = 0.0
        self._active_training_script = ()

    def _activate_training_segment(self, index: int) -> None:
        if not self._active_training_segments:
            self._current_training_segment = None
            self._active_training_script = ()
            return
        safe_index = max(0, min(int(index), len(self._active_training_segments) - 1))
        segment = self._active_training_segments[safe_index]
        phase_key = "practice" if self._phase is Phase.PRACTICE else "scored"
        self._active_training_segment_index = safe_index
        self._current_training_segment = segment
        self._training_segment_started_elapsed_s = float(self._sim_elapsed_s)
        self._active_training_script = self._scenario_generator.build_training_script(
            layout=self._compound_layout,
            phase_key=phase_key,
            profile=segment.profile,
            salt=f"{self._phase.value}:{safe_index}:{segment.label}",
        )

    def _refresh_active_training_segment(self) -> None:
        if not self._active_training_segments or self._current_training_segment is None:
            return
        while (
            self._active_training_segment_index + 1 < len(self._active_training_segments)
            and (self._sim_elapsed_s - self._training_segment_started_elapsed_s)
            >= float(self._current_training_segment.duration_s)
        ):
            if self._phase is Phase.SCORED and self._window_samples > 0:
                self._score_active_window()
                self._window_elapsed_s = 0.0
            self._activate_training_segment(self._active_training_segment_index + 1)
            self._reset_scene_state(reset_capture_totals=False)

    def _difficulty_profile(self) -> RapidTrackingDifficultyProfile:
        level = difficulty_level_for_ratio("rapid_tracking", clamp01(self._difficulty))
        shared = difficulty_profile_for_code("rapid_tracking", level, mode="build")
        pressure = shared.axes.time_pressure
        sensitivity = shared.axes.control_sensitivity
        concurrency = shared.axes.multitask_concurrency
        if level <= 3:
            return RapidTrackingDifficultyProfile(
                tier="low",
                scene_script=self._scenario.ground_script,
                practice_script=self._scenario.ground_script,
                loop_limit=1,
                practice_loop_limit=0,
                duration_scale=max(2.6, 4.4 - (pressure * 1.8)),
                turbulence_strength=max(0.0, sensitivity * 0.12),
                drift_duration_scale=max(1.8, 3.1 - (sensitivity * 0.8)),
                camera_assist_strength=max(0.55, 0.95 - (sensitivity * 0.35)),
            )
        if level >= 8:
            return RapidTrackingDifficultyProfile(
                tier="high",
                scene_script=self._scenario.scored_script,
                practice_script=self._scenario.practice_script,
                loop_limit=3,
                practice_loop_limit=0,
                duration_scale=max(0.62, 1.20 - (pressure * 0.62)),
                turbulence_strength=max(0.78, 0.82 + (sensitivity * 0.62)),
                drift_duration_scale=max(0.52, 1.02 - (pressure * 0.56) - (sensitivity * 0.10)),
                camera_assist_strength=0.0,
            )
        return RapidTrackingDifficultyProfile(
            tier="mid",
            scene_script=self._scenario.scored_script,
            practice_script=self._scenario.practice_script,
            loop_limit=1,
            practice_loop_limit=0,
            duration_scale=max(1.00, 2.0 - (pressure * 1.10)),
            turbulence_strength=max(0.24, 0.22 + (sensitivity * 0.46) + (concurrency * 0.08)),
            drift_duration_scale=max(0.90, 1.45 - (pressure * 0.42)),
            camera_assist_strength=0.0,
        )

    def _active_scene_script(self, profile: RapidTrackingDifficultyProfile) -> tuple[RapidTrackingSceneSegment, ...]:
        if self._current_training_segment is not None and self._active_training_script:
            return self._active_training_script
        if self._phase is Phase.PRACTICE:
            return profile.practice_script
        return profile.scene_script

    def _active_loop_limit(self, profile: RapidTrackingDifficultyProfile) -> int:
        if self._current_training_segment is not None:
            return 0
        if self._phase is Phase.PRACTICE:
            return int(profile.practice_loop_limit)
        return int(profile.loop_limit)

    def start_practice(self) -> None:
        if self._phase is not Phase.INSTRUCTIONS:
            return
        if self._cfg.practice_duration_s <= 0.0:
            self._phase = Phase.PRACTICE_DONE
            self._phase_started_at_s = self._clock.now()
            return
        self._phase = Phase.PRACTICE
        self._set_active_segments_for_phase()
        self._reset_runtime_state(reset_scores=False)

    def start_scored(self) -> None:
        if self._phase in (Phase.SCORED, Phase.RESULTS):
            return
        self._phase = Phase.SCORED
        self._set_active_segments_for_phase()
        self._reset_runtime_state(reset_scores=True)

    def submit_answer(self, raw: str) -> bool:
        token = str(raw).strip().upper()
        if token == "CAPTURE_HOLD_START":
            return self._capture_hold_start()
        if token == "CAPTURE_HOLD_END":
            return self._capture_hold_end()
        if token in {"CAPTURE", "TRIGGER", "SHOT"}:
            return self._capture_trigger()
        return False

    def set_control(self, *, horizontal: float, vertical: float) -> None:
        self._control_x = max(-1.0, min(1.0, float(horizontal)))
        self._control_y = max(-1.0, min(1.0, float(vertical)))

    def update(self) -> None:
        now = self._clock.now()
        dt = now - self._last_update_at_s
        self._last_update_at_s = now

        if dt <= 0.0:
            self._refresh_phase_boundaries(now)
            return

        dt = min(float(dt), self._MAX_UPDATE_DT_S)
        self._accumulator_s += dt

        while self._accumulator_s >= self._tick_dt:
            self._accumulator_s -= self._tick_dt
            if self._phase in (Phase.PRACTICE, Phase.SCORED):
                self._step(self._tick_dt)

        self._refresh_phase_boundaries(now)

    def time_remaining_s(self) -> float | None:
        now = self._clock.now()
        if self._phase is Phase.PRACTICE:
            rem = self._cfg.practice_duration_s - (now - self._phase_started_at_s)
            return max(0.0, rem)
        if self._phase is Phase.SCORED:
            rem = self._cfg.scored_duration_s - (now - self._phase_started_at_s)
            return max(0.0, rem)
        return None

    def snapshot(self) -> TestSnapshot:
        profile = self._difficulty_profile()
        rig, projection, target_world_x, target_world_y, focus_world_x, focus_world_y = (
            self._current_camera_solution()
        )
        compat_x, compat_y = camera_pose_compat(
            heading_deg=rig.heading_deg,
            pitch_deg=rig.pitch_deg,
            neutral_heading_deg=rig.neutral_heading_deg,
            neutral_pitch_deg=rig.neutral_pitch_deg,
        )

        preview_left = max(0.0, self._target_preview_until_s - self._sim_elapsed_s)
        hud_visible = self._sim_elapsed_s <= self._hud_visible_until_s
        lock_progress = clamp01(self._lock_hold_s / self._cfg.hud_acquire_s)
        target_in_capture_box = self._target_in_capture_box(require_visible=True)
        capture_flash_s = max(0.0, self._capture_feedback_until_s - self._sim_elapsed_s)
        capture_accuracy = (
            0.0 if self._capture_attempts <= 0 else self._capture_hits / self._capture_attempts
        )

        payload = RapidTrackingPayload(
            target_rel_x=float(projection.target_rel_x),
            target_rel_y=float(projection.target_rel_y),
            target_world_x=float(target_world_x),
            target_world_y=float(target_world_y),
            focus_world_x=float(focus_world_x),
            focus_world_y=float(focus_world_y),
            reticle_x=float(self._reticle_x),
            reticle_y=float(self._reticle_y),
            camera_x=float(compat_x),
            camera_y=float(compat_y),
            camera_yaw_deg=float(rig.heading_deg),
            camera_pitch_deg=float(rig.pitch_deg),
            target_vx=float(self._target_vx),
            target_vy=float(self._target_vy),
            target_visible=not self._target_terrain_occluded,
            target_cover_state=str(self._target_cover_state),
            target_is_moving=bool(self._target_is_moving),
            target_kind=str(self._target_kind),
            target_variant=str(self._target_variant),
            target_handoff_mode=str(self._target_handoff_mode),
            difficulty=float(self._difficulty),
            difficulty_tier=str(profile.tier),
            session_seed=int(self._seed),
            camera_assist_strength=float(self._effective_camera_assist_strength(profile)),
            turbulence_strength=float(self._effective_turbulence_strength(profile)),
            target_time_to_switch_s=max(0.0, self._target_switch_at_s - self._sim_elapsed_s),
            target_switch_preview_s=float(preview_left),
            obscured_time_left_s=0.0,
            phase_elapsed_s=float(self._phase_elapsed_s()),
            mean_error=self._current_mean_error(),
            rms_error=self._current_rms_error(),
            on_target_s=float(self._scored_on_target_s),
            on_target_ratio=self._current_on_target_ratio(),
            obscured_tracking_ratio=self._current_obscured_tracking_ratio(),
            hud_visible=bool(hud_visible),
            lock_progress=float(lock_progress),
            terrain_ridge_y=float(self._terrain_ridge_y),
            scene_progress=float(self._scene_progress()),
            capture_box_half_width=float(self._effective_capture_box_half_width()),
            capture_box_half_height=float(self._effective_capture_box_half_height()),
            target_in_capture_box=bool(target_in_capture_box),
            capture_zoom=float(self._current_capture_zoom()),
            capture_points=int(self._capture_points),
            capture_hits=int(self._capture_hits),
            capture_attempts=int(self._capture_attempts),
            capture_accuracy=float(capture_accuracy),
            capture_feedback=str(self._capture_feedback),
            capture_flash_s=float(capture_flash_s),
            active_target_kinds=self._active_target_kinds(),
            active_challenges=self._active_challenges(),
            focus_label=self._focus_label(),
            segment_label=self._segment_label(),
            segment_index=self._segment_index(),
            segment_total=self._segment_total(),
            segment_time_remaining_s=float(self._segment_time_remaining_s()),
        )

        return TestSnapshot(
            title=self._title,
            phase=self._phase,
            prompt=self.current_prompt(),
            input_hint=(
                "Pan freely with configured HOTAS movement, rudder or left-right input, and joystick axis 1 or up-down input. "
                "Keep the target centered, then hold the configured capture binding or Space when it is inside the "
                "center camera box."
            ),
            time_remaining_s=self.time_remaining_s(),
            attempted_scored=self._scored_attempted,
            correct_scored=self._scored_correct,
            payload=payload,
            practice_feedback=None,
        )

    def current_prompt(self) -> str:
        if self._phase is Phase.INSTRUCTIONS:
            return (
                f"{self._title}\n"
                "Scope a target-rich compound from a bobbing helicopter while the aircraft circles overhead.\n"
                "Foot patrols move slowly, road vehicles move faster, helicopters move faster still, "
                "and jets are the fastest passes.\n"
                "Targets may duck into buildings or disappear behind terrain; during building handoffs "
                "you track the structure until the next target emerges.\n"
                "Pan the camera freely, keep the target centered, and hold the configured capture binding or Space when the target "
                "is inside the center camera box to zoom and capture.\n"
                "Press Enter to begin practice."
            )
        if self._phase is Phase.PRACTICE_DONE:
            return "Practice complete. Press Enter to start the timed test."
        if self._phase is Phase.RESULTS:
            summary = self.scored_summary()
            acc = int(round(summary.accuracy * 100))
            on_target_pct = summary.on_target_ratio * 100.0
            obscured_pct = summary.obscured_tracking_ratio * 100.0
            return (
                "Results\n"
                f"Windows: {summary.attempted}\n"
                f"Accurate: {summary.correct} ({acc}%)\n"
                f"Mean Error: {summary.mean_error:.3f}\n"
                "On-Target Time: "
                f"{summary.on_target_s:.1f}s ({on_target_pct:.1f}%)\n"
                f"Obscured Tracking: {obscured_pct:.1f}%\n"
                "Camera Hits: "
                f"{summary.capture_hits}/{summary.capture_attempts}  "
                f"Points: {summary.capture_points}"
            )
        return "Track the active target through handoffs, keep lock, and capture it inside the center box."

    def scored_summary(self) -> RapidTrackingSummary:
        attempted = int(self._scored_attempted)
        correct = int(self._scored_correct)
        duration_s = float(self._cfg.scored_duration_s)
        accuracy = 0.0 if attempted == 0 else correct / attempted
        throughput_per_min = (attempted / duration_s) * 60.0
        mean_error = (
            0.0 if self._scored_samples == 0 else self._scored_sum_error / self._scored_samples
        )

        rms_error = 0.0
        if self._scored_samples > 0:
            rms_error = math.sqrt(self._scored_sum_error2 / self._scored_samples)

        on_target_ratio = (
            0.0 if duration_s <= 0.0 else min(1.0, self._scored_on_target_s / duration_s)
        )

        obscured_tracking_ratio = 0.0
        if self._scored_obscured_s > 0.0:
            obscured_tracking_ratio = self._scored_obscured_on_target_s / self._scored_obscured_s

        moving_target_ratio = (
            0.0 if duration_s <= 0.0 else min(1.0, self._scored_moving_target_s / duration_s)
        )

        max_score = float(self._scored_max_score)
        total_score = float(self._scored_total_score)
        score_ratio = 0.0 if max_score <= 0.0 else total_score / max_score
        capture_attempts = int(self._capture_attempts)
        capture_hits = int(self._capture_hits)
        capture_accuracy = 0.0 if capture_attempts == 0 else capture_hits / capture_attempts
        capture_max_points = int(self._capture_max_points)
        capture_score_ratio = (
            0.0 if capture_max_points <= 0 else float(self._capture_points) / float(capture_max_points)
        )

        return RapidTrackingSummary(
            attempted=attempted,
            correct=correct,
            accuracy=accuracy,
            duration_s=duration_s,
            throughput_per_min=throughput_per_min,
            mean_error=float(mean_error),
            rms_error=float(rms_error),
            on_target_s=float(self._scored_on_target_s),
            on_target_ratio=float(on_target_ratio),
            obscured_time_s=float(self._scored_obscured_s),
            obscured_tracking_ratio=float(obscured_tracking_ratio),
            moving_target_ratio=float(moving_target_ratio),
            total_score=total_score,
            max_score=max_score,
            score_ratio=score_ratio,
            capture_points=int(self._capture_points),
            capture_hits=capture_hits,
            capture_attempts=capture_attempts,
            capture_accuracy=float(capture_accuracy),
            capture_max_points=capture_max_points,
            capture_score_ratio=float(capture_score_ratio),
            overshoot_count=int(self._scored_overshoot_count),
            reversal_count=int(self._scored_reversal_count),
        )

    def _reset_runtime_state(self, *, reset_scores: bool) -> None:
        now = self._clock.now()
        self._phase_started_at_s = now
        self._last_update_at_s = now
        self._accumulator_s = 0.0
        self._sim_elapsed_s = 0.0
        self._set_active_segments_for_phase()
        if self._active_training_segments:
            self._activate_training_segment(0)
        else:
            self._current_training_segment = None
            self._active_training_script = ()

        self._look_x = 0.0
        self._look_y = 0.0
        self._reticle_x = 0.0
        self._reticle_y = 0.0
        self._reset_scene_state(reset_capture_totals=True)

        if not reset_scores:
            self._practice_samples = 0
            self._practice_sum_error = 0.0

        if reset_scores:
            self._events = []
            self._scored_samples = 0
            self._scored_sum_error = 0.0
            self._scored_sum_error2 = 0.0
            self._scored_on_target_s = 0.0
            self._scored_obscured_s = 0.0
            self._scored_obscured_on_target_s = 0.0
            self._scored_moving_target_s = 0.0
            self._scored_overshoot_count = 0
            self._scored_reversal_count = 0
            self._prev_err_sign_x = None
            self._prev_err_sign_y = None

            self._window_elapsed_s = 0.0
            self._window_sum_error = 0.0
            self._window_samples = 0

            self._scored_attempted = 0
            self._scored_correct = 0
            self._scored_total_score = 0.0
            self._scored_max_score = 0.0

    def _error_sign(self, value: float) -> int:
        deadband = max(float(self._cfg.on_target_radius) * 0.5, 1e-6)
        if value > deadband:
            return 1
        if value < -deadband:
            return -1
        return 0

    def _record_direction_change(self, *, axis: str, error_value: float) -> None:
        current = self._error_sign(error_value)
        previous = self._prev_err_sign_x if axis == "x" else self._prev_err_sign_y
        if current != 0:
            if previous is not None and previous != 0 and current != previous:
                self._scored_reversal_count += 1
                if abs(float(error_value)) > float(self._cfg.on_target_radius):
                    self._scored_overshoot_count += 1
            if axis == "x":
                self._prev_err_sign_x = current
            else:
                self._prev_err_sign_y = current

    def _reset_scene_state(self, *, reset_capture_totals: bool) -> None:
        self._camera_x = 0.0
        self._camera_y = 0.0
        self._camera_yaw_deg = 0.0
        self._camera_pitch_deg = 0.0
        self._drift_x = 0.0
        self._drift_y = 0.0
        self._drift_until_s = 0.0
        self._refresh_drift_vector()

        self._target_x = 0.0
        self._target_y = 0.0
        self._target_vx = 0.0
        self._target_vy = 0.0
        self._target_is_moving = True
        self._target_kind = "soldier"
        self._target_variant = "target"
        self._target_handoff_mode = "smooth"
        self._target_cover_state = "open"

        self._script_index = 0
        self._segment_started_s = float(self._sim_elapsed_s)
        self._segment_duration_s = 1.0
        self._segment_start_x = 0.0
        self._segment_start_y = 0.0
        self._segment_control_x = 0.0
        self._segment_control_y = 0.0
        self._segment_end_x = 0.0
        self._segment_end_y = 0.0
        self._segment_route_kind = "foot_patrol"
        self._segment_focus_anchor_id = ""
        self._loop_count = 0
        self._target_switch_at_s = float(self._sim_elapsed_s)
        self._target_preview_until_s = 0.0
        self._start_scene_segment(initial=True)
        self._advance_target()
        self._reset_camera_pose_to_target()

        self._terrain_ridge_y = 0.0
        self._target_terrain_occluded = False

        self._lock_hold_s = 0.0
        self._hud_visible_until_s = 0.0
        self._capture_zoom_until_s = 0.0
        self._capture_hold_active = False
        self._capture_hold_blend = 0.0
        self._capture_hold_bonus_due_at_s = 0.0
        self._capture_last_at_s = -999.0
        self._capture_feedback_until_s = 0.0
        self._capture_feedback = ""
        if reset_capture_totals:
            self._capture_points = 0
            self._capture_hits = 0
            self._capture_attempts = 0
            self._capture_max_points = 0

    def _step(self, dt: float) -> None:
        self._sim_elapsed_s += dt
        self._refresh_active_training_segment()

        if self._sim_elapsed_s >= self._drift_until_s:
            self._refresh_drift_vector()

        self._advance_camera(dt)
        self._advance_target()

        rig, projection, _target_world_x, _target_world_y, _focus_world_x, _focus_world_y = (
            self._current_camera_solution()
        )
        target_rel_x = projection.target_rel_x
        target_rel_y = projection.target_rel_y
        self._terrain_ridge_y = self._mountain_ridge_for(target_rel_x)
        covered_by_segment = self._target_cover_state == "terrain"
        auto_occluded = False
        if not covered_by_segment:
            auto_occluded = self._is_occluded_by_terrain(
                target_rel_x=target_rel_x,
                target_rel_y=target_rel_y,
                target_kind=self._target_kind,
            )
        self._target_terrain_occluded = covered_by_segment or auto_occluded

        err_x = target_rel_x - self._reticle_x
        err_y = target_rel_y - self._reticle_y
        tracking_error = math.sqrt((err_x * err_x) + (err_y * err_y))

        on_target = bool(projection.in_front) and tracking_error <= self._cfg.on_target_radius
        target_in_capture_box = self._target_in_capture_box(require_visible=True)
        self._update_hud_lock_state(dt=dt, on_target=on_target)
        self._advance_capture_hold_state(dt=dt, target_in_box=target_in_capture_box)

        if self._phase is Phase.PRACTICE:
            self._practice_samples += 1
            self._practice_sum_error += tracking_error
            return

        if self._phase is not Phase.SCORED:
            return

        self._scored_samples += 1
        self._scored_sum_error += tracking_error
        self._scored_sum_error2 += tracking_error * tracking_error
        self._record_direction_change(axis="x", error_value=err_x)
        self._record_direction_change(axis="y", error_value=err_y)

        if on_target:
            self._scored_on_target_s += dt

        if self._target_terrain_occluded:
            self._scored_obscured_s += dt
            if on_target:
                self._scored_obscured_on_target_s += dt

        if self._target_is_moving:
            self._scored_moving_target_s += dt

        self._window_elapsed_s += dt
        self._window_sum_error += tracking_error
        self._window_samples += 1

        if self._window_elapsed_s >= 1.0:
            self._score_active_window()
            self._window_elapsed_s = max(0.0, self._window_elapsed_s - 1.0)

    def _refresh_phase_boundaries(self, now: float) -> None:
        if self._phase is Phase.PRACTICE:
            if now - self._phase_started_at_s >= self._cfg.practice_duration_s:
                self._phase = Phase.PRACTICE_DONE
                self._phase_started_at_s = now
                self._last_update_at_s = now
                self._accumulator_s = 0.0
            return

        if self._phase is Phase.SCORED:
            if now - self._phase_started_at_s >= self._cfg.scored_duration_s:
                if self._window_samples > 0:
                    self._score_active_window()
                    self._window_elapsed_s = 0.0
                self._phase = Phase.RESULTS
                self._phase_started_at_s = now
                self._last_update_at_s = now
                self._accumulator_s = 0.0

    def _advance_camera(self, dt: float) -> None:
        self._camera_yaw_deg = (self._camera_yaw_deg + (self._control_x * self._cfg.camera_yaw_rate_deg_s * dt)) % 360.0
        self._camera_pitch_deg = max(
            -89.0,
            min(89.0, self._camera_pitch_deg + ((-self._control_y) * self._cfg.camera_pitch_rate_deg_s * dt)),
        )
        self._reticle_x = 0.0
        self._reticle_y = 0.0

    def _advance_target(self) -> None:
        while self._sim_elapsed_s >= self._target_switch_at_s:
            self._start_scene_segment(initial=False)

        seg_elapsed = self._sim_elapsed_s - self._segment_started_s
        if self._segment_duration_s <= 0.0:
            u = 1.0
        else:
            u = max(0.0, min(1.0, seg_elapsed / self._segment_duration_s))

        one_minus = 1.0 - u
        kind = self._target_kind
        dominant_ground_axis_x = abs(self._segment_end_x - self._segment_start_x) >= abs(
            self._segment_end_y - self._segment_start_y
        )

        if kind == "building":
            self._target_x = float(self._segment_end_x)
            self._target_y = float(self._segment_end_y)
            self._target_vx = 0.0
            self._target_vy = 0.0
            self._target_is_moving = False
            return

        if kind == "truck":
            if dominant_ground_axis_x:
                x = self._segment_start_x + ((self._segment_end_x - self._segment_start_x) * u)
                y = self._segment_start_y
                dx_du = self._segment_end_x - self._segment_start_x
                dy_du = 0.0
            else:
                x = self._segment_start_x
                y = self._segment_start_y + ((self._segment_end_y - self._segment_start_y) * u)
                dx_du = 0.0
                dy_du = self._segment_end_y - self._segment_start_y
        else:
            x = (
                (one_minus * one_minus * self._segment_start_x)
                + (2.0 * one_minus * u * self._segment_control_x)
                + (u * u * self._segment_end_x)
            )
            y = (
                (one_minus * one_minus * self._segment_start_y)
                + (2.0 * one_minus * u * self._segment_control_y)
                + (u * u * self._segment_end_y)
            )
            dx_du = (2.0 * one_minus * (self._segment_control_x - self._segment_start_x)) + (
                2.0 * u * (self._segment_end_x - self._segment_control_x)
            )
            dy_du = (2.0 * one_minus * (self._segment_control_y - self._segment_start_y)) + (
                2.0 * u * (self._segment_end_y - self._segment_control_y)
            )

        if self._segment_duration_s <= 0.0:
            vx = 0.0
            vy = 0.0
        else:
            inv = 1.0 / self._segment_duration_s
            vx = dx_du * inv
            vy = dy_du * inv

        extra_x = 0.0
        extra_y = 0.0
        extra_vx = 0.0
        extra_vy = 0.0
        phase = self._sim_elapsed_s + (self._script_index * 0.73) + (self._seed * 0.0009)
        if kind == "soldier":
            weave_amp = 0.018 + (0.010 * self._difficulty)
            stride_amp = 0.010 + (0.006 * self._difficulty)
            weave_freq = 2.4 + (0.7 * self._difficulty)
            stride_freq = 5.8 + (1.0 * self._difficulty)
            extra_x = math.sin(phase * weave_freq) * weave_amp
            extra_y = math.sin((phase * stride_freq) + 0.45) * stride_amp
            extra_vx = math.cos(phase * weave_freq) * weave_amp * weave_freq
            extra_vy = math.cos((phase * stride_freq) + 0.45) * stride_amp * stride_freq
        elif kind == "truck":
            surge_amp = 0.012 + (0.005 * self._difficulty)
            surge_freq = 1.9 + (0.5 * self._difficulty)
            surge = math.sin((phase * surge_freq) + 0.8) * surge_amp
            surge_v = math.cos((phase * surge_freq) + 0.8) * surge_amp * surge_freq
            if dominant_ground_axis_x:
                extra_x = surge
                extra_vx = surge_v
            else:
                extra_y = surge
                extra_vy = surge_v
        elif kind == "helicopter":
            hover_amp = 0.024 + (0.012 * self._difficulty)
            bob_amp = 0.016 + (0.010 * self._difficulty)
            hover_freq = 1.3 + (0.4 * self._difficulty)
            bob_freq = 2.3 + (0.6 * self._difficulty)
            extra_x = math.sin(phase * hover_freq) * hover_amp
            extra_y = math.sin((phase * bob_freq) + 0.65) * bob_amp
            extra_vx = math.cos(phase * hover_freq) * hover_amp * hover_freq
            extra_vy = math.cos((phase * bob_freq) + 0.65) * bob_amp * bob_freq
        elif kind == "jet":
            wx_amp = 0.018 + (0.028 * self._difficulty)
            wy_amp = 0.018 + (0.030 * self._difficulty)
            wx_freq = 2.0 + (1.0 * self._difficulty)
            wy_freq = 2.6 + (1.4 * self._difficulty)
            extra_x = math.sin(phase * wx_freq) * wx_amp
            extra_y = math.sin((phase * wy_freq) + 0.4) * wy_amp
            extra_vx = math.cos(phase * wx_freq) * wx_amp * wx_freq
            extra_vy = math.cos((phase * wy_freq) + 0.4) * wy_amp * wy_freq

        self._target_x = float(x + extra_x)
        self._target_y = float(y + extra_y)
        self._target_vx = float(vx + extra_vx)
        self._target_vy = float(vy + extra_vy)
        speed_threshold = 0.012 if kind == "soldier" else 0.020
        self._target_is_moving = math.hypot(self._target_vx, self._target_vy) > speed_threshold

    def _start_scene_segment(self, *, initial: bool) -> None:
        profile = self._difficulty_profile()
        script = self._active_scene_script(profile)
        if len(script) == 0:
            return

        tier_changed = self._difficulty_tier != profile.tier
        self._difficulty_tier = profile.tier

        if initial or tier_changed:
            self._script_index = 0
            self._loop_count = 0
        else:
            next_index = self._script_index + 1
            if next_index >= len(script):
                self._loop_count += 1
                loop_limit = self._active_loop_limit(profile)
                if loop_limit > 0 and self._loop_count >= loop_limit:
                    self._target_switch_at_s = math.inf
                    self._target_preview_until_s = 0.0
                    return
                next_index = 0
            self._script_index = next_index

        segment = script[self._script_index]

        prev_kind = self._target_kind
        prev_variant = self._target_variant
        prev_cover = self._target_cover_state
        next_kind = str(segment.kind)
        next_variant = str(segment.variant)
        next_handoff = "smooth" if initial else str(segment.handoff).strip().lower()

        start_anchor = self._anchor_lookup[segment.start_anchor_id]
        end_anchor = self._anchor_lookup[segment.end_anchor_id]

        if initial:
            start_x = float(start_anchor.x)
            start_y = float(start_anchor.y)
        else:
            if next_handoff == "jump":
                start_x = float(start_anchor.x)
                start_y = float(start_anchor.y)
            else:
                lane_blend = 0.18 if (next_kind == "building" or prev_kind == "building") else 0.24
                start_x = (self._target_x * (1.0 - lane_blend)) + (start_anchor.x * lane_blend)
                start_y = (self._target_y * (1.0 - lane_blend)) + (start_anchor.y * lane_blend)

        if next_kind == "truck":
            horizontal_route = abs(end_anchor.x - start_anchor.x) >= abs(end_anchor.y - start_anchor.y)
            if horizontal_route:
                start_y = float(start_anchor.y)
            else:
                start_x = float(start_anchor.x)

        scene_progress = self._scene_progress()
        pressure = clamp01((self._difficulty * 0.44) + (scene_progress * 0.60))
        if next_kind == "soldier":
            duration_scale = 1.04 - (0.16 * pressure)
            duration = max(4.6, float(segment.duration_s) * duration_scale)
            arc_scale = 0.90 + (0.12 * pressure)
        elif next_kind == "building":
            duration_scale = 0.98 - (0.10 * pressure)
            duration = max(1.0, min(2.2, float(segment.duration_s) * duration_scale))
            arc_scale = 0.0
        elif next_kind == "truck":
            duration_scale = 0.92 - (0.18 * pressure)
            duration = max(3.4, float(segment.duration_s) * duration_scale)
            arc_scale = 0.98 + (0.14 * pressure)
        elif next_kind == "helicopter":
            duration_scale = 0.84 - (0.20 * pressure)
            duration = max(2.7, float(segment.duration_s) * duration_scale)
            arc_scale = 1.06 + (0.20 * pressure)
        else:
            duration_scale = 0.72 - (0.24 * pressure)
            duration = max(1.8, float(segment.duration_s) * duration_scale)
            arc_scale = 1.18 + (0.24 * pressure)

        duration *= self._effective_segment_duration_scale(profile)

        self._segment_started_s = self._sim_elapsed_s
        self._segment_duration_s = duration
        self._segment_start_x = start_x
        self._segment_start_y = start_y
        self._segment_end_x = float(end_anchor.x)
        self._segment_end_y = float(end_anchor.y)
        self._segment_control_x = ((start_x + self._segment_end_x) * 0.5) + (
            float(segment.arc_x) * arc_scale
        )
        self._segment_control_y = ((start_y + self._segment_end_y) * 0.5) + (
            float(segment.arc_y) * arc_scale
        )
        self._segment_route_kind = str(segment.route_kind)
        self._segment_focus_anchor_id = str(segment.focus_anchor_id)
        self._target_switch_at_s = self._sim_elapsed_s + duration
        self._target_kind = next_kind
        self._target_variant = next_variant
        self._target_handoff_mode = next_handoff
        self._target_cover_state = str(segment.cover_mode)

        if (
            initial
            or next_kind != prev_kind
            or next_variant != prev_variant
            or next_handoff == "jump"
            or self._target_cover_state != prev_cover
        ):
            self._target_preview_until_s = self._sim_elapsed_s + self._effective_preview_duration_s()

    def _refresh_drift_vector(self) -> None:
        profile = self._difficulty_profile()
        turbulence_strength = self._effective_turbulence_strength(profile)
        if turbulence_strength <= 0.0:
            self._drift_x = 0.0
            self._drift_y = 0.0
            self._drift_until_s = self._sim_elapsed_s + 60.0
            return
        drift = self._drift_gen.next_vector(difficulty=self._difficulty)
        self._drift_x = float(drift.vx * turbulence_strength)
        self._drift_y = float(drift.vy * turbulence_strength)
        self._drift_until_s = self._sim_elapsed_s + float(
            drift.duration_s * profile.drift_duration_scale
        )

    def _mountain_ridge_for(self, rel_x: float) -> float:
        x = float(rel_x)
        phase = self._compound_layout.ridge_phase
        amp = self._compound_layout.ridge_amp_scale
        bias = self._compound_layout.path_lateral_bias
        major = 0.14 * amp * math.sin((x * (2.4 + (amp * 0.2))) + phase)
        mid = 0.09 * amp * math.sin((x * 4.7) + 1.15 - phase * 0.4)
        fine = 0.05 * math.sin((x * 8.4) - 0.55 + (bias * 4.0))
        return 0.12 + major + mid + fine

    def _is_occluded_by_terrain(
        self, *, target_rel_x: float, target_rel_y: float, target_kind: str
    ) -> bool:
        kind = str(target_kind).strip().lower()
        if kind == "building":
            return False
        if abs(target_rel_x) > (self._cfg.view_limit * 1.20):
            return False

        ridge_y = self._mountain_ridge_for(target_rel_x)
        if kind in {"soldier", "truck"}:
            return target_rel_y >= (ridge_y + 0.12)
        if kind == "helicopter":
            return target_rel_y >= (ridge_y + 0.22)
        return target_rel_y >= (ridge_y + 0.34)

    def _update_hud_lock_state(self, *, dt: float, on_target: bool) -> None:
        previous = self._lock_hold_s

        if on_target:
            self._lock_hold_s += dt
            if previous < self._cfg.hud_acquire_s <= self._lock_hold_s:
                self._hud_visible_until_s = self._sim_elapsed_s + self._cfg.hud_persist_s
            if self._lock_hold_s >= self._cfg.hud_acquire_s:
                self._hud_visible_until_s = max(
                    self._hud_visible_until_s, self._sim_elapsed_s + 0.25
                )
            return

        self._lock_hold_s = max(0.0, self._lock_hold_s - (dt * 1.9))

    @staticmethod
    def _capture_points_for_kind(kind: str) -> int:
        return 2 if str(kind).strip().lower() in {"helicopter", "jet"} else 1

    def _capture_hold_start(self) -> bool:
        if self._phase not in (Phase.PRACTICE, Phase.SCORED):
            return False
        self._capture_hold_active = True
        self._capture_hold_bonus_due_at_s = self._sim_elapsed_s + max(
            0.05,
            float(self._cfg.capture_hold_bonus_interval_s),
        )
        self._capture_trigger()
        return True

    def _capture_hold_end(self) -> bool:
        if self._phase not in (Phase.PRACTICE, Phase.SCORED):
            self._capture_hold_active = False
            self._capture_hold_bonus_due_at_s = 0.0
            return False
        self._capture_hold_active = False
        self._capture_hold_bonus_due_at_s = 0.0
        return True

    def _capture_trigger(self) -> bool:
        if self._phase not in (Phase.PRACTICE, Phase.SCORED):
            return False
        if (self._sim_elapsed_s - self._capture_last_at_s) < self._effective_capture_cooldown_s():
            return False

        self._capture_last_at_s = self._sim_elapsed_s
        self._capture_attempts += 1
        self._capture_max_points += self._capture_points_for_kind(self._target_kind)

        hit = self._target_in_capture_box(require_visible=True)
        if hit:
            points = self._capture_points_for_kind(self._target_kind)
            self._capture_hits += 1
            self._capture_points += points
            self._capture_zoom_until_s = self._sim_elapsed_s + self._cfg.capture_zoom_s
            self._capture_feedback = f"+{points}"
            self._capture_feedback_until_s = self._sim_elapsed_s + self._cfg.capture_flash_s
            return True

        self._capture_feedback = "MISS"
        self._capture_feedback_until_s = self._sim_elapsed_s + self._cfg.capture_flash_s
        return True

    def _advance_capture_hold_state(self, *, dt: float, target_in_box: bool) -> None:
        response_hz = max(0.1, float(self._cfg.capture_hold_blend_response_hz))
        alpha = min(1.0, dt * response_hz)
        target_blend = 1.0 if self._capture_hold_active else 0.0
        self._capture_hold_blend += (target_blend - self._capture_hold_blend) * alpha
        if not self._capture_hold_active:
            return

        interval = max(0.05, float(self._cfg.capture_hold_bonus_interval_s))
        if self._capture_hold_bonus_due_at_s <= 0.0:
            self._capture_hold_bonus_due_at_s = self._sim_elapsed_s + interval
        while self._sim_elapsed_s >= self._capture_hold_bonus_due_at_s:
            self._capture_max_points += 1
            if target_in_box:
                self._capture_points += 1
            self._capture_hold_bonus_due_at_s += interval

    def _target_in_capture_box(self, *, require_visible: bool) -> bool:
        if require_visible and self._target_terrain_occluded:
            return False
        _rig, projection, _target_world_x, _target_world_y, _focus_world_x, _focus_world_y = (
            self._current_camera_solution()
        )
        if not projection.in_front:
            return False
        return (
            abs(projection.target_rel_x - self._reticle_x) <= self._effective_capture_box_half_width()
            and abs(projection.target_rel_y - self._reticle_y) <= self._effective_capture_box_half_height()
        )

    def _current_capture_zoom(self) -> float:
        hold_zoom = max(0.0, min(1.0, float(self._capture_hold_blend)))
        remaining = self._capture_zoom_until_s - self._sim_elapsed_s
        if remaining <= 0.0 or self._cfg.capture_zoom_s <= 0.0:
            return hold_zoom
        t = 1.0 - max(0.0, min(1.0, remaining / self._cfg.capture_zoom_s))
        pulse = math.sin(t * math.pi)
        kick = 1.0 - t
        pulse_zoom = max(0.0, ((pulse * 0.72) + (kick * 0.28)) * self._cfg.capture_zoom_strength)
        return max(hold_zoom, pulse_zoom)

    def _scene_progress(self) -> float:
        if self._phase is Phase.PRACTICE:
            if self._cfg.practice_duration_s <= 0.0:
                return 1.0
            return clamp01((self._clock.now() - self._phase_started_at_s) / self._cfg.practice_duration_s)
        if self._phase in (Phase.SCORED, Phase.RESULTS):
            if self._cfg.scored_duration_s <= 0.0:
                return 1.0
            elapsed = self._clock.now() - self._phase_started_at_s
            if self._phase is Phase.RESULTS:
                elapsed = self._cfg.scored_duration_s
            return clamp01(elapsed / self._cfg.scored_duration_s)
        return 0.0

    def _score_active_window(self) -> None:
        mean_error = (
            0.0 if self._window_samples == 0 else self._window_sum_error / self._window_samples
        )
        window_score = score_window(
            mean_error=mean_error,
            good_window_error=self._cfg.good_window_error,
        )
        is_correct = window_score >= 1.0 - 1e-9
        window_end_s = max(0.0, float(self._clock.now() - self._phase_started_at_s))
        response_time_s = max(0.0, min(1.0, float(self._window_elapsed_s)))
        window_start_s = max(0.0, window_end_s - response_time_s)
        self._scored_attempted += 1
        self._scored_max_score += 1.0
        self._scored_total_score += window_score
        if is_correct:
            self._scored_correct += 1
        self._events.append(
            QuestionEvent(
                index=int(self._scored_attempted),
                phase=Phase.SCORED,
                prompt="Rapid Tracking Window",
                correct_answer=1,
                user_answer=1 if is_correct else 0,
                is_correct=is_correct,
                presented_at_s=window_start_s,
                answered_at_s=window_end_s,
                response_time_s=response_time_s,
                raw="",
                score=float(window_score),
                max_score=1.0,
                content_metadata=content_metadata_from_payload(
                    None,
                    extras={
                        "content_family": "rapid_tracking",
                        "variant_id": f"{self._target_kind}:{self._target_variant}:{self._segment_route_kind}",
                        "content_pack": "rapid_tracking",
                        "kind": self._target_kind,
                        "target_variant": self._target_variant,
                    },
                ),
            )
        )

        self._window_sum_error = 0.0
        self._window_samples = 0

    def _current_mean_error(self) -> float:
        if self._phase is Phase.PRACTICE:
            if self._practice_samples == 0:
                return 0.0
            return self._practice_sum_error / self._practice_samples
        if self._phase in (Phase.SCORED, Phase.RESULTS):
            if self._scored_samples == 0:
                return 0.0
            return self._scored_sum_error / self._scored_samples
        return 0.0

    def _current_rms_error(self) -> float:
        if self._phase in (Phase.SCORED, Phase.RESULTS):
            if self._scored_samples == 0:
                return 0.0
            return math.sqrt(self._scored_sum_error2 / self._scored_samples)
        return 0.0

    def _current_on_target_ratio(self) -> float:
        if self._phase in (Phase.SCORED, Phase.RESULTS):
            elapsed = self._clock.now() - self._phase_started_at_s
            if self._phase is Phase.RESULTS:
                elapsed = self._cfg.scored_duration_s
            if elapsed <= 0.0:
                return 0.0
            return max(0.0, min(1.0, self._scored_on_target_s / elapsed))
        return 0.0

    def _current_obscured_tracking_ratio(self) -> float:
        if self._scored_obscured_s <= 0.0:
            return 0.0
        return max(0.0, min(1.0, self._scored_obscured_on_target_s / self._scored_obscured_s))

    def _phase_elapsed_s(self) -> float:
        if self._phase in (Phase.PRACTICE, Phase.SCORED):
            return float(self._sim_elapsed_s)
        if self._phase is Phase.RESULTS:
            return float(self._cfg.scored_duration_s)
        return max(0.0, float(self._clock.now() - self._phase_started_at_s))

    def _focus_world_xy(self) -> tuple[float, float]:
        return track_to_world_xy(
            track_x=float(self._compound_layout.compound_center_x),
            track_y=float(self._compound_layout.compound_center_y),
            path_lateral_bias=float(self._compound_layout.path_lateral_bias),
        )

    def _target_world_xy(self) -> tuple[float, float]:
        return track_to_world_xy(
            track_x=float(self._target_x),
            track_y=float(self._target_y),
            path_lateral_bias=float(self._compound_layout.path_lateral_bias),
        )

    def _current_camera_solution(
        self,
    ) -> tuple[object, object, float, float, float, float]:
        focus_world_x, focus_world_y = self._focus_world_xy()
        target_world_x, target_world_y = self._target_world_xy()
        rig = rapid_tracking_camera_rig_state(
            elapsed_s=self._phase_elapsed_s(),
            seed=int(self._seed),
            progress=self._scene_progress(),
            camera_yaw_deg=float(self._camera_yaw_deg),
            camera_pitch_deg=float(self._camera_pitch_deg),
            zoom=self._current_capture_zoom(),
            target_kind=str(self._target_kind),
            target_world_x=float(target_world_x),
            target_world_y=float(target_world_y),
            focus_world_x=float(focus_world_x),
            focus_world_y=float(focus_world_y),
            turbulence_strength=float(self._effective_turbulence_strength(self._difficulty_profile())),
        )
        projection = rapid_tracking_target_projection(
            rig=rig,
            target_kind=str(self._target_kind),
            target_world_x=float(target_world_x),
            target_world_y=float(target_world_y),
            elapsed_s=self._phase_elapsed_s(),
            scene_progress=self._scene_progress(),
            seed=int(self._seed),
        )
        return (
            rig,
            projection,
            float(target_world_x),
            float(target_world_y),
            float(focus_world_x),
            float(focus_world_y),
        )

    def _reset_camera_pose_to_target(self) -> None:
        focus_world_x, focus_world_y = self._focus_world_xy()
        target_world_x, target_world_y = self._target_world_xy()
        rig = rapid_tracking_camera_rig_state(
            elapsed_s=self._phase_elapsed_s(),
            seed=int(self._seed),
            progress=self._scene_progress(),
            camera_yaw_deg=None,
            camera_pitch_deg=None,
            zoom=0.0,
            target_kind=str(self._target_kind),
            target_world_x=float(target_world_x),
            target_world_y=float(target_world_y),
            focus_world_x=float(focus_world_x),
            focus_world_y=float(focus_world_y),
            turbulence_strength=float(self._effective_turbulence_strength(self._difficulty_profile())),
        )
        target_world_z = estimated_target_world_z(
            kind=str(self._target_kind),
            target_world_x=float(target_world_x),
            target_world_y=float(target_world_y),
            elapsed_s=self._phase_elapsed_s(),
            scene_progress=self._scene_progress(),
            seed=int(self._seed),
        )
        self._camera_yaw_deg = math.degrees(
            math.atan2(
                float(target_world_x) - float(rig.cam_world_x),
                float(target_world_y) - float(rig.cam_world_y),
            )
        ) % 360.0
        horizontal_distance = math.hypot(
            float(target_world_x) - float(rig.cam_world_x),
            float(target_world_y) - float(rig.cam_world_y),
        )
        self._camera_pitch_deg = math.degrees(
            math.atan2(
                float(target_world_z) - float(rig.cam_world_z),
                max(1e-3, horizontal_distance),
            )
        )
        compat_x, compat_y = camera_pose_compat(
            heading_deg=float(self._camera_yaw_deg),
            pitch_deg=float(self._camera_pitch_deg),
            neutral_heading_deg=float(rig.neutral_heading_deg),
            neutral_pitch_deg=float(rig.neutral_pitch_deg),
        )
        self._camera_x = float(compat_x)
        self._camera_y = float(compat_y)


def build_rapid_tracking_test(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    config: RapidTrackingConfig | None = None,
    title: str = "Rapid Tracking",
    practice_segments: Sequence[RapidTrackingTrainingSegment] | None = None,
    scored_segments: Sequence[RapidTrackingTrainingSegment] | None = None,
) -> RapidTrackingEngine:
    return RapidTrackingEngine(
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        config=config,
        title=title,
        practice_segments=practice_segments,
        scored_segments=scored_segments,
    )

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, replace

from .clock import Clock
from .cognitive_core import Phase, TestSnapshot
from .telemetry import TelemetryEvent


KIND_AUDITORY_CAPACITY = "auditory_capacity"
KIND_RAPID_TRACKING = "rapid_tracking"
KIND_SPATIAL_INTEGRATION = "spatial_integration"
KIND_TRACE_TEST_1 = "trace_test_1"
KIND_TRACE_TEST_2 = "trace_test_2"

GODOT_OWNED_KINDS = {
    KIND_AUDITORY_CAPACITY,
    KIND_RAPID_TRACKING,
    KIND_SPATIAL_INTEGRATION,
    KIND_TRACE_TEST_1,
    KIND_TRACE_TEST_2,
}

_DIRECT_KIND_BY_CODE = {
    "auditory_capacity": KIND_AUDITORY_CAPACITY,
    "rapid_tracking": KIND_RAPID_TRACKING,
    "spatial_integration": KIND_SPATIAL_INTEGRATION,
    "trace_test_1": KIND_TRACE_TEST_1,
    "trace_test_2": KIND_TRACE_TEST_2,
    "auditory_capacity_workout": KIND_AUDITORY_CAPACITY,
    "rapid_tracking_workout": KIND_RAPID_TRACKING,
    "spatial_integration_workout": KIND_SPATIAL_INTEGRATION,
    "trace_test_1_workout": KIND_TRACE_TEST_1,
    "trace_test_2_workout": KIND_TRACE_TEST_2,
}

_SI_STATIC_KINDS = (
    "landmark_grid",
    "scene_reconstruction",
)
_SI_AIRCRAFT_KINDS = (
    "aircraft_route_selection",
    "aircraft_continuation_selection",
    "aircraft_location_grid",
)
_SI_ALL_KINDS = _SI_STATIC_KINDS + _SI_AIRCRAFT_KINDS


_AC_CHANNEL_ORDER = (
    "gates",
    "state_commands",
    "gate_directives",
    "digit_recall",
    "trigger",
    "distractors",
)
_AC_GODOT_TUBE_PLAY_SCALE = 1.18
_AC_GODOT_INNER_RX = 2.86
_AC_GODOT_INNER_RZ = 2.10


def rapid_tracking_godot_config(
    *,
    test_code: str,
    mode: str = "standard",
    difficulty: float = 0.5,
    duration_s: float | None = None,
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Declarative Godot run config for the expanded Rapid Tracking world."""
    token = str(test_code or "").strip().lower()
    normalized_mode = str(mode or "standard").strip().lower()
    ratio = max(0.0, min(1.0, _finite_float(difficulty, 0.5)))
    is_drill = token.startswith("rt_")
    is_workout = token == "rapid_tracking_workout"
    is_benchmark = normalized_mode == "benchmark"

    density_scale = 0.86 + ratio * 0.42
    if "lock_anchor" in token or "capture_timing" in token:
        density_scale *= 0.82
    if "pressure" in token:
        density_scale *= 1.14
    if is_benchmark:
        density_scale *= 0.92

    vehicle_bias = 1.0
    pedestrian_bias = 1.0
    air_target_enabled = False
    if "ground" in token or "terrain" in token:
        vehicle_bias = 1.12
        pedestrian_bias = 1.08
    if "building" in token:
        pedestrian_bias = 1.18
    if "air_speed" in token:
        air_target_enabled = True
        vehicle_bias = 0.78
        pedestrian_bias = 0.70
    if "pressure" in token:
        air_target_enabled = True

    world_size = 92.0 + ratio * 28.0
    terrain_resolution = int(round(18.0 + ratio * 10.0))
    road_density = 0.78 + ratio * 0.42
    town_count = int(round(4.0 + ratio * 4.0))
    vehicle_count = int(round((16.0 + ratio * 22.0) * density_scale * vehicle_bias))
    pedestrian_count = int(round((10.0 + ratio * 18.0) * density_scale * pedestrian_bias))

    rapid_world: dict[str, object] = {
        "world_size_m": world_size,
        "terrain_resolution": terrain_resolution,
        "hill_intensity": 1.05 + ratio * 1.25,
        "mountain_intensity": 1.0 + ratio * 1.8,
        "town_count": town_count,
        "road_density": road_density,
        "lake_count": int(round(1.0 + ratio * 2.0)),
        "river_count": int(round(1.0 + ratio * 1.0)),
        "tunnel_count": int(round(1.0 + ratio * 3.0)),
        "forest_patch_count": int(round(8.0 + ratio * 10.0)),
        "vehicle_count": max(6, vehicle_count),
        "pedestrian_count": max(4, pedestrian_count),
        "parked_asset_count": int(round((20.0 + ratio * 30.0) * density_scale)),
        "occlusion_density": 0.22 + ratio * 0.36,
        "air_distractor_count": int(round(3.0 + ratio * 4.0)),
        "ground_target_weight": 0.88 if not air_target_enabled else 0.64,
        "air_targets_enabled": air_target_enabled,
        "difficulty_scaled": True,
        "scene_style": "low_poly_large",
    }

    config: dict[str, object] = {
        "rapid_tracking": True,
        "difficulty_scaled": True,
        "mode": normalized_mode,
        "active_target_mix": "ground_heavy",
        "reticle_mode": "fixed_center",
        "unlimited_gimbal": True,
        "on_target_radius": max(0.055, 0.122 - ratio * 0.030),
        "capture_box_half_width": max(0.075, 0.165 - ratio * 0.035),
        "capture_box_half_height": max(0.065, 0.140 - ratio * 0.025),
        "capture_cooldown_s": max(0.14, 0.48 - ratio * 0.14),
        "camera_orbit_rate_scale": 0.88 + ratio * 0.42,
        "camera_turbulence_scale": 0.80 + ratio * 0.72,
        "target_speed_scale": 0.82 + ratio * 0.62,
        "handoff_interval_s": max(7.0, 15.5 - ratio * 5.2),
        "obscuration_scale": 0.72 + ratio * 0.75,
        "rapid_world": rapid_world,
    }
    if duration_s is not None:
        config["duration_s"] = max(1.0, _finite_float(duration_s, 180.0))
    if is_workout:
        config["workout"] = True
    if is_drill:
        config["drill"] = True
    if is_benchmark:
        config["benchmark"] = True
    if extra:
        config.update(_as_dict(extra))
        if "rapid_world" in extra:
            merged_world = dict(rapid_world)
            merged_world.update(_as_dict(extra["rapid_world"]))
            config["rapid_world"] = merged_world
    return config


def _ac_training_profile_payload(profile: object) -> dict[str, object]:
    return {item.name: getattr(profile, item.name) for item in fields(profile)}


def _ac_effective_segment_payload(
    *,
    base_config: object,
    difficulty_profile: object,
    difficulty: float,
    profile: object,
    use_default_difficulty_profile: bool,
    duration_s: float,
) -> dict[str, object]:
    gate_rate_scale = max(0.05, float(getattr(profile, "gate_rate_scale")))
    command_rate_scale = max(0.05, float(getattr(profile, "command_rate_scale")))
    directive_rate_scale = max(0.05, float(getattr(profile, "directive_rate_scale")))
    sequence_rate_scale = max(0.05, float(getattr(profile, "sequence_rate_scale")))
    beep_rate_scale = max(0.05, float(getattr(profile, "beep_rate_scale")))
    response_scale = max(0.35, float(getattr(profile, "response_window_scale")))
    tube_width_scale = max(0.40, float(getattr(profile, "tube_width_scale")))
    tube_height_scale = max(0.40, float(getattr(profile, "tube_height_scale")))
    disturbance_scale = max(0.0, float(getattr(profile, "disturbance_scale")))

    if use_default_difficulty_profile:
        gate_interval_s = float(getattr(difficulty_profile, "gate_interval_s"))
        callsign_count = int(getattr(difficulty_profile, "callsign_count"))
        digit_min = int(getattr(difficulty_profile, "digit_sequence_min_len"))
        digit_max = int(getattr(difficulty_profile, "digit_sequence_max_len"))
        noise_peak = float(getattr(difficulty_profile, "background_noise_peak"))
        distortion_peak = float(getattr(difficulty_profile, "background_distortion_peak"))
        instructor_noise = float(getattr(difficulty_profile, "instructor_noise_level"))
        instructor_distortion = float(getattr(difficulty_profile, "instructor_distortion_level"))
        instructor_rate_wpm = int(getattr(difficulty_profile, "instructor_rate_wpm"))
        ambient_layers = int(getattr(difficulty_profile, "ambient_layer_target"))
    else:
        gate_interval_s = 1.0 / max(0.05, float(getattr(base_config, "gate_spawn_rate")))
        callsign_count = max(1, int(getattr(base_config, "callsign_count")))
        digit_max = max(
            1,
            min(
                int(getattr(base_config, "digit_sequence_max_len")),
                int(getattr(profile, "digit_sequence_max_len")),
            ),
        )
        digit_min = max(
            1,
            min(digit_max, int(getattr(profile, "digit_sequence_min_len"))),
        )
        noise_peak = 0.30 + (0.50 * max(0.0, min(1.0, float(difficulty))))
        distortion_peak = 0.04 + (0.20 * max(0.0, min(1.0, float(difficulty))))
        instructor_noise = 0.0
        instructor_distortion = 0.0
        instructor_rate_wpm = 182
        ambient_layers = 1

    effective_gate_interval_s = max(0.25, gate_interval_s / gate_rate_scale)
    command_interval_s = (1.0 / max(0.05, float(getattr(base_config, "command_rate")))) / command_rate_scale
    directive_interval_s = max(5.2, effective_gate_interval_s * 1.45) / directive_rate_scale
    sequence_interval_s = max(24.0, float(getattr(base_config, "sequence_interval_s"))) / sequence_rate_scale
    beep_interval_s = max(10.0, float(getattr(base_config, "beep_interval_s"))) / beep_rate_scale
    sequence_first_s = max(7.0, min(max(28.0, float(duration_s) * 0.24), sequence_interval_s))

    return {
        "tube_half_width": max(
            0.20,
            float(getattr(base_config, "tube_half_width")) * tube_width_scale * _AC_GODOT_TUBE_PLAY_SCALE,
        ),
        "tube_half_height": max(
            0.16,
            float(getattr(base_config, "tube_half_height")) * tube_height_scale * _AC_GODOT_TUBE_PLAY_SCALE,
        ),
        "disturbance_gain": max(0.0, float(getattr(base_config, "disturbance_gain")) * disturbance_scale),
        "response_window_seconds": max(
            0.40, float(getattr(base_config, "response_window_seconds")) * response_scale
        ),
        "gate_interval_s": effective_gate_interval_s,
        "command_interval_s": max(0.75, command_interval_s),
        "directive_interval_s": max(4.0, directive_interval_s),
        "sequence_interval_s": max(7.0, sequence_interval_s),
        "sequence_first_s": sequence_first_s,
        "beep_interval_s": max(6.0, beep_interval_s),
        "callsign_count": callsign_count,
        "digit_sequence_min_len": digit_min,
        "digit_sequence_max_len": digit_max,
        "background_noise_level": max(
            0.0, min(1.0, noise_peak * max(0.0, float(getattr(profile, "noise_level_scale"))))
        ),
        "background_distortion_level": max(
            0.0,
            min(1.0, distortion_peak * max(0.0, float(getattr(profile, "distortion_level_scale")))),
        ),
        "instructor_noise_level": max(0.0, min(1.0, instructor_noise)),
        "instructor_distortion_level": max(0.0, min(1.0, instructor_distortion)),
        "instructor_rate_wpm": instructor_rate_wpm,
        "ambient_layer_target": ambient_layers,
    }


def _ac_segment_payload(
    *,
    segment: object,
    base_config: object,
    difficulty_profile: object,
    difficulty: float,
    use_default_difficulty_profile: bool,
) -> dict[str, object]:
    profile = getattr(segment, "profile")
    duration_s = max(0.0, float(getattr(segment, "duration_s")))
    return {
        "label": str(getattr(segment, "label")),
        "duration_s": duration_s,
        "active_channels": [str(channel) for channel in getattr(segment, "active_channels")],
        "profile": _ac_training_profile_payload(profile),
        "effective": _ac_effective_segment_payload(
            base_config=base_config,
            difficulty_profile=difficulty_profile,
            difficulty=difficulty,
            profile=profile,
            use_default_difficulty_profile=use_default_difficulty_profile,
            duration_s=duration_s,
        ),
    }


def _ac_standard_segment(*, duration_s: float):
    from .auditory_capacity import AuditoryCapacityTrainingProfile, AuditoryCapacityTrainingSegment

    return AuditoryCapacityTrainingSegment(
        label="Full Mixed",
        duration_s=float(duration_s),
        active_channels=_AC_CHANNEL_ORDER,
        profile=AuditoryCapacityTrainingProfile(),
    )


def _ac_drill_segments(
    *,
    test_code: str,
    mode: str,
    duration_s: float,
) -> tuple[object, ...]:
    from .ac_drills import _mode_scaled_profile, _repeat_segments, _segment
    from .ant_drills import AntDrillMode
    from .auditory_capacity import AuditoryCapacityTrainingProfile

    token = str(test_code or "").strip().lower()
    try:
        normalized_mode = AntDrillMode(str(mode).strip().lower())
    except Exception:
        normalized_mode = AntDrillMode.BUILD
    scored_duration_s = max(1.0, float(duration_s))

    if token == "ac_gate_anchor":
        profile = _mode_scaled_profile(
            profile=AuditoryCapacityTrainingProfile(
                enable_state_commands=False,
                enable_gate_directives=False,
                enable_digit_sequences=False,
                enable_trigger_cues=False,
                enable_distractors=False,
                gate_rate_scale=0.70,
                disturbance_scale=0.70,
                tube_width_scale=1.12,
                tube_height_scale=1.10,
                noise_level_scale=0.0,
            ),
            mode=normalized_mode,
        )
        return (_segment(label="Gate Flight", duration_s=scored_duration_s, active_channels=("gates",), profile=profile),)

    if token == "ac_state_command_prime":
        profile = _mode_scaled_profile(
            profile=AuditoryCapacityTrainingProfile(
                enable_gate_directives=False,
                enable_digit_sequences=False,
                enable_trigger_cues=False,
                enable_distractors=False,
                gate_rate_scale=0.60,
                command_rate_scale=0.85,
                disturbance_scale=0.78,
                tube_width_scale=1.08,
                tube_height_scale=1.06,
                noise_level_scale=0.0,
            ),
            mode=normalized_mode,
        )
        return (
            _segment(
                label="State Commands",
                duration_s=scored_duration_s,
                active_channels=("gates", "state_commands"),
                profile=profile,
            ),
        )

    if token == "ac_gate_directive_run":
        profile = _mode_scaled_profile(
            profile=AuditoryCapacityTrainingProfile(
                enable_state_commands=False,
                enable_digit_sequences=False,
                enable_trigger_cues=False,
                enable_distractors=True,
                gate_rate_scale=0.75,
                directive_rate_scale=0.90,
                disturbance_scale=0.86,
                tube_width_scale=1.04,
                tube_height_scale=1.04,
                noise_level_scale=0.30,
            ),
            mode=normalized_mode,
        )
        return (
            _segment(
                label="Gate Directives",
                duration_s=scored_duration_s,
                active_channels=("gates", "gate_directives", "distractors"),
                profile=profile,
            ),
        )

    if token == "ac_digit_sequence_prime":
        digit_max = 5 if normalized_mode.value in {"fresh", "build", "recovery"} else 6
        profile = _mode_scaled_profile(
            profile=AuditoryCapacityTrainingProfile(
                enable_state_commands=False,
                enable_gate_directives=False,
                enable_trigger_cues=False,
                enable_distractors=False,
                gate_rate_scale=0.58,
                sequence_rate_scale=0.92,
                disturbance_scale=0.76,
                tube_width_scale=1.08,
                tube_height_scale=1.08,
                noise_level_scale=0.0,
                digit_sequence_min_len=5,
                digit_sequence_max_len=digit_max,
            ),
            mode=normalized_mode,
        )
        return (
            _segment(
                label="Digit Recall",
                duration_s=scored_duration_s,
                active_channels=("gates", "digit_recall"),
                profile=profile,
            ),
        )

    if token == "ac_trigger_cue_anchor":
        profile = _mode_scaled_profile(
            profile=AuditoryCapacityTrainingProfile(
                enable_state_commands=False,
                enable_gate_directives=False,
                enable_digit_sequences=False,
                enable_distractors=False,
                gate_rate_scale=0.62,
                beep_rate_scale=0.92,
                disturbance_scale=0.74,
                tube_width_scale=1.08,
                tube_height_scale=1.08,
                noise_level_scale=0.0,
            ),
            mode=normalized_mode,
        )
        return (
            _segment(
                label="Trigger Cues",
                duration_s=scored_duration_s,
                active_channels=("gates", "trigger"),
                profile=profile,
            ),
        )

    if token == "ac_callsign_filter_run":
        profile = _mode_scaled_profile(
            profile=AuditoryCapacityTrainingProfile(
                enable_digit_sequences=False,
                enable_trigger_cues=False,
                enable_distractors=True,
                gate_rate_scale=0.90,
                command_rate_scale=0.92,
                directive_rate_scale=0.95,
                disturbance_scale=0.95,
                noise_level_scale=0.72,
            ),
            mode=normalized_mode,
        )
        return (
            _segment(
                label="Callsign Filter",
                duration_s=scored_duration_s,
                active_channels=("gates", "state_commands", "gate_directives", "distractors"),
                profile=profile,
            ),
        )

    if token == "ac_mixed_tempo":
        templates = (
            _segment(
                label="Gate Flight",
                duration_s=90.0,
                active_channels=("gates",),
                profile=_mode_scaled_profile(
                    profile=AuditoryCapacityTrainingProfile(
                        enable_state_commands=False,
                        enable_gate_directives=False,
                        enable_digit_sequences=False,
                        enable_trigger_cues=False,
                        enable_distractors=False,
                        gate_rate_scale=0.80,
                        disturbance_scale=0.82,
                        tube_width_scale=1.08,
                        tube_height_scale=1.06,
                        noise_level_scale=0.0,
                    ),
                    mode=normalized_mode,
                ),
            ),
            _segment(
                label="State Commands",
                duration_s=90.0,
                active_channels=("gates", "state_commands"),
                profile=_mode_scaled_profile(
                    profile=AuditoryCapacityTrainingProfile(
                        enable_gate_directives=False,
                        enable_digit_sequences=False,
                        enable_trigger_cues=False,
                        enable_distractors=False,
                        gate_rate_scale=0.70,
                        command_rate_scale=0.90,
                        disturbance_scale=0.84,
                        tube_width_scale=1.06,
                        tube_height_scale=1.04,
                        noise_level_scale=0.0,
                    ),
                    mode=normalized_mode,
                ),
            ),
            _segment(
                label="Gate Directives",
                duration_s=90.0,
                active_channels=("gates", "gate_directives", "distractors"),
                profile=_mode_scaled_profile(
                    profile=AuditoryCapacityTrainingProfile(
                        enable_state_commands=False,
                        enable_digit_sequences=False,
                        enable_trigger_cues=False,
                        enable_distractors=True,
                        gate_rate_scale=0.82,
                        directive_rate_scale=0.95,
                        disturbance_scale=0.92,
                        noise_level_scale=0.30,
                    ),
                    mode=normalized_mode,
                ),
            ),
            _segment(
                label="Digit Recall",
                duration_s=90.0,
                active_channels=("gates", "digit_recall"),
                profile=_mode_scaled_profile(
                    profile=AuditoryCapacityTrainingProfile(
                        enable_state_commands=False,
                        enable_gate_directives=False,
                        enable_trigger_cues=False,
                        enable_distractors=False,
                        gate_rate_scale=0.65,
                        sequence_rate_scale=0.95,
                        disturbance_scale=0.82,
                        tube_width_scale=1.04,
                        tube_height_scale=1.04,
                        noise_level_scale=0.0,
                        digit_sequence_min_len=5,
                        digit_sequence_max_len=6,
                    ),
                    mode=normalized_mode,
                ),
            ),
            _segment(
                label="Trigger + Callsign Filter",
                duration_s=90.0,
                active_channels=("gates", "state_commands", "gate_directives", "trigger", "distractors"),
                profile=_mode_scaled_profile(
                    profile=AuditoryCapacityTrainingProfile(
                        enable_digit_sequences=False,
                        enable_distractors=True,
                        gate_rate_scale=0.92,
                        command_rate_scale=0.94,
                        directive_rate_scale=0.96,
                        beep_rate_scale=0.96,
                        disturbance_scale=0.96,
                        noise_level_scale=0.72,
                    ),
                    mode=normalized_mode,
                ),
            ),
            _segment(
                label="Full Mixed",
                duration_s=90.0,
                active_channels=_AC_CHANNEL_ORDER,
                profile=_mode_scaled_profile(
                    profile=AuditoryCapacityTrainingProfile(),
                    mode=normalized_mode,
                ),
            ),
        )
        return _repeat_segments(total_duration_s=scored_duration_s, templates=templates)

    if token == "ac_pressure_run":
        profile = _mode_scaled_profile(
            profile=AuditoryCapacityTrainingProfile(
                enable_gates=True,
                enable_state_commands=True,
                enable_gate_directives=True,
                enable_digit_sequences=True,
                enable_trigger_cues=True,
                enable_distractors=True,
                gate_rate_scale=1.18,
                command_rate_scale=1.16,
                directive_rate_scale=1.14,
                sequence_rate_scale=1.05,
                beep_rate_scale=1.12,
                response_window_scale=0.86,
                disturbance_scale=1.18,
                tube_width_scale=0.96,
                tube_height_scale=0.96,
                noise_level_scale=1.18,
                distortion_level_scale=1.12,
                digit_sequence_min_len=5,
                digit_sequence_max_len=6,
            ),
            mode=normalized_mode,
        )
        return (
            _segment(
                label="Pressure Run",
                duration_s=scored_duration_s,
                active_channels=_AC_CHANNEL_ORDER,
                profile=profile,
            ),
        )

    return (_ac_standard_segment(duration_s=scored_duration_s),)


def _ac_workout_segments(*, difficulty: float, duration_s: float) -> tuple[object, ...]:
    from .ac_workouts import build_ac_workout_plan

    plan = build_ac_workout_plan(duration_scale=max(0.05, float(duration_s) / (90.0 * 60.0)))
    segments: list[object] = []
    for block in plan.blocks:
        block_duration_s = max(1.0, float(block.duration_min) * 60.0)
        segments.extend(
            _ac_drill_segments(
                test_code=str(block.drill_code),
                mode=str(block.mode.value),
                duration_s=block_duration_s,
            )
        )
    return tuple(segments) or (_ac_standard_segment(duration_s=duration_s),)


def auditory_capacity_godot_config(
    *,
    test_code: str,
    mode: str = "standard",
    difficulty: float = 0.5,
    duration_s: float | None = None,
    review_mode_enabled: bool = False,
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Declarative Godot run config mirroring the Python Auditory Capacity settings."""
    from .auditory_capacity import AuditoryCapacityConfig, AuditoryCapacityDifficultyProfile

    cfg = AuditoryCapacityConfig()
    difficulty_ratio = max(0.0, min(1.0, _finite_float(difficulty, 0.5)))
    difficulty_profile = AuditoryCapacityDifficultyProfile.from_ratio(difficulty_ratio)
    token = str(test_code or "").strip().lower()
    total_duration_s = max(
        1.0,
        _finite_float(
            duration_s,
            cfg.scored_duration_s if token != "auditory_capacity_workout" else 90.0 * 60.0,
        ),
    )
    use_default_difficulty_profile = token == "auditory_capacity"
    if token == "auditory_capacity_workout":
        segments = _ac_workout_segments(difficulty=difficulty_ratio, duration_s=total_duration_s)
    elif token.startswith("ac_"):
        segments = _ac_drill_segments(test_code=token, mode=mode, duration_s=total_duration_s)
    else:
        segments = (_ac_standard_segment(duration_s=total_duration_s),)
    segment_payloads = [
        _ac_segment_payload(
            segment=segment,
            base_config=cfg,
            difficulty_profile=difficulty_profile,
            difficulty=difficulty_ratio,
            use_default_difficulty_profile=use_default_difficulty_profile,
        )
        for segment in segments
    ]
    first_effective = dict(segment_payloads[0]["effective"] if segment_payloads else {})
    config: dict[str, object] = {
        "auditory_capacity": True,
        "difficulty_scaled": True,
        "use_default_difficulty_profile": use_default_difficulty_profile,
        "mode": str(mode or "standard"),
        "tick_hz": float(cfg.tick_hz),
        "control_gain": float(cfg.control_gain),
        "disturbance_gain": float(first_effective.get("disturbance_gain", cfg.disturbance_gain)),
        "tube_half_width": float(first_effective.get("tube_half_width", cfg.tube_half_width)),
        "tube_half_height": float(first_effective.get("tube_half_height", cfg.tube_half_height)),
        "base_tube_half_width": float(cfg.tube_half_width) * _AC_GODOT_TUBE_PLAY_SCALE,
        "base_tube_half_height": float(cfg.tube_half_height) * _AC_GODOT_TUBE_PLAY_SCALE,
        "tunnel_curvature_intensity": float(cfg.tunnel_curvature_intensity),
        "tunnel_twist_intensity": float(cfg.tunnel_twist_intensity),
        "gate_speed_norm_per_s": float(cfg.gate_speed_norm_per_s),
        "gate_spawn_rate": float(cfg.gate_spawn_rate),
        "gate_interval_s": float(first_effective.get("gate_interval_s", cfg.gate_interval_s)),
        "command_rate": float(cfg.command_rate),
        "command_interval_s": float(first_effective.get("command_interval_s", 1.0 / cfg.command_rate)),
        "directive_interval_s": float(first_effective.get("directive_interval_s", 5.6)),
        "distractor_rate": float(cfg.distractor_rate),
        "distractor_interval_s": float(1.0 / max(0.05, cfg.distractor_rate)),
        "callsign_count": int(first_effective.get("callsign_count", cfg.callsign_count)),
        "digit_sequence_min_len": int(first_effective.get("digit_sequence_min_len", 4)),
        "digit_sequence_max_len": int(first_effective.get("digit_sequence_max_len", cfg.digit_sequence_max_len)),
        "recall_prompt_rate": float(cfg.recall_prompt_rate),
        "response_window_seconds": float(first_effective.get("response_window_seconds", cfg.response_window_seconds)),
        "callsign_interval_s": float(cfg.callsign_interval_s),
        "base_beep_interval_s": float(cfg.beep_interval_s),
        "beep_interval_s": float(first_effective.get("beep_interval_s", cfg.beep_interval_s)),
        "beep_frequency_hz": 1120.0,
        "beep_duration_s": 0.12,
        "beep_volume_db": -6.0,
        "review_mode_enabled": bool(review_mode_enabled),
        "ambient_volume_db": -14.0,
        "ambient_layer_drop_db": -2.5,
        "primary_voice_volume": 62.0,
        "distractor_voice_volume": 76.0,
        "filler_voice_volume": 40.0,
        "secondary_voice_min_difficulty": 0.67,
        "filler_narrator_interval_s": 13.0,
        "color_command_interval_s": float(cfg.color_command_interval_s),
        "sequence_interval_s": float(first_effective.get("sequence_interval_s", cfg.sequence_interval_s)),
        "sequence_first_s": float(first_effective.get("sequence_first_s", 28.0)),
        "sequence_display_s": float(cfg.sequence_display_s),
        "sequence_response_s": float(cfg.sequence_response_s),
        "inner_rx": _AC_GODOT_INNER_RX,
        "inner_rz": _AC_GODOT_INNER_RZ,
        "ball_radius": 0.11,
        "physics_ball_radius": 0.28,
        "wall_bounce_factor": 0.42,
        "tube_chunk_count": 22,
        "tube_chunk_length": 2.0,
        "tube_radial_segments": 16,
        "active_channels": list(segment_payloads[0]["active_channels"] if segment_payloads else _AC_CHANNEL_ORDER),
        "segments": segment_payloads,
        "difficulty_profile": {
            item.name: getattr(difficulty_profile, item.name)
            for item in fields(difficulty_profile)
        },
        "audio": {
            "tts_required": True,
            "background_noise_level": float(first_effective.get("background_noise_level", 0.0)),
            "background_distortion_level": float(first_effective.get("background_distortion_level", 0.0)),
            "instructor_noise_level": float(first_effective.get("instructor_noise_level", 0.0)),
            "instructor_distortion_level": float(first_effective.get("instructor_distortion_level", 0.0)),
            "instructor_rate_wpm": int(first_effective.get("instructor_rate_wpm", 182)),
            "ambient_layer_target": int(first_effective.get("ambient_layer_target", 1)),
            "background_noise_source": None,
        },
    }
    if duration_s is not None:
        config["duration_s"] = total_duration_s
    if token == "auditory_capacity_workout":
        config["workout"] = True
    if token.startswith("ac_"):
        config["drill"] = True
    if extra:
        config.update(_as_dict(extra))
    return config


def spatial_integration_godot_config(
    *,
    test_code: str,
    mode: str = "standard",
    duration_s: float | None = None,
    practice_scenes_per_part: int = 0,
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Declarative Godot run config for Spatial Integration variants."""
    token = str(test_code or "").strip().lower()
    parts: tuple[str, ...]
    allowed: tuple[str, ...]
    if token in {"si_landmark_anchor"}:
        parts = ("static",)
        allowed = ("landmark_grid",)
    elif token in {"si_reconstruction_run"}:
        parts = ("static",)
        allowed = ("scene_reconstruction",)
    elif token in {"si_static_mixed_run", "si_static_multiview_integration"}:
        parts = ("static",)
        allowed = _SI_STATIC_KINDS
    elif token in {"si_route_anchor"}:
        parts = ("aircraft",)
        allowed = ("aircraft_route_selection",)
    elif token in {"si_continuation_prime"}:
        parts = ("aircraft",)
        allowed = ("aircraft_continuation_selection",)
    elif token in {"si_aircraft_grid_run"}:
        parts = ("aircraft",)
        allowed = ("aircraft_location_grid",)
    elif token in {"si_moving_aircraft_multiview_integration", "si_aircraft_multiview_integration"}:
        parts = ("aircraft",)
        allowed = _SI_AIRCRAFT_KINDS
    else:
        parts = ("static", "aircraft")
        allowed = _SI_ALL_KINDS

    config: dict[str, object] = {
        "spatial_integration": True,
        "parts": list(parts),
        "allowed_question_kinds": list(allowed),
        "static_study_s": 12.0,
        "aircraft_study_s": 15.0,
        "question_time_limit_s": 8.0,
        "practice_scenes_per_part": max(0, int(practice_scenes_per_part)),
        "grid_cols": 8,
        "grid_rows": 8,
        "alt_levels": 5,
        "mode": str(mode or "standard"),
    }
    if duration_s is not None:
        config["duration_s"] = max(1.0, float(duration_s))
    if extra:
        config.update(_as_dict(extra))
    return config


def trace_test_godot_config(
    *,
    test_code: str,
    mode: str = "standard",
    difficulty: float = 0.5,
    duration_s: float | None = None,
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Declarative Godot run config for Trace Test 1/2 movement and camera tuning."""
    token = str(test_code or "").strip().lower()
    ratio = max(0.0, min(1.0, _finite_float(difficulty, 0.5)))
    move_s = max(1.18, 1.62 - ratio * 0.28)
    pause_s = max(0.82, 1.08 - ratio * 0.20)
    if "pressure" in token:
        move_s *= 0.92
        pause_s *= 0.92
    trace2_clip_min_s = max(3.2, 4.10 - ratio * 0.60)
    trace2_clip_max_s = max(trace2_clip_min_s + 1.10, 7.00 - ratio * 1.00)
    config: dict[str, object] = {
        "trace": True,
        "difficulty_scaled": True,
        "mode": str(mode or "standard"),
        "trace_grid_half_x": 7,
        "trace_grid_half_z": 7,
        "trace_grid_max_y": 5,
        "trace_grid_scale": 0.64,
        "trace_grid_alt_scale": 0.60,
        "trace_move_duration_s": move_s,
        "trace_pause_duration_s": pause_s,
        "trace_camera_style": "fixed_square_orthographic",
        "trace_camera_margin": 1.18,
        "trace_panel_margin": 1.25,
        "trace_panel_visual_scale": 2.0,
        "trace1_response_window_s": max(1.0, 3.0 - ratio * 0.55),
        "trace2_observe_s": max(2.0, 5.0 - ratio * 0.55),
        "trace2_clip_min_s": trace2_clip_min_s,
        "trace2_clip_max_s": trace2_clip_max_s,
        "trace2_question_window_s": max(2.0, 7.0 - ratio * 1.2),
        "trace2_offscreen_margin_cells": 2.10 + ratio * 0.30,
        "trace2_close_camera_scale": max(0.52, 0.68 - ratio * 0.08),
        "trace2_aircraft_scale": 1.36 + ratio * 0.18,
        "trace2_question_overlay_enabled": True,
        "trace2_new_clip_after_answer": True,
    }
    if duration_s is not None:
        config["duration_s"] = max(1.0, float(duration_s))
    if token.startswith(("tt1_", "tt2_", "trace_")):
        config["drill"] = True
    if extra:
        config.update(_as_dict(extra))
    return config


def godot_kind_for_test_code(test_code: str | None) -> str | None:
    token = str(test_code or "").strip().lower()
    if token in _DIRECT_KIND_BY_CODE:
        return _DIRECT_KIND_BY_CODE[token]
    if token.startswith("ac_"):
        return KIND_AUDITORY_CAPACITY
    if token.startswith("rt_"):
        return KIND_RAPID_TRACKING
    if token.startswith("si_"):
        return KIND_SPATIAL_INTEGRATION
    if token.startswith("tt1_"):
        return KIND_TRACE_TEST_1
    if token.startswith("tt2_"):
        return KIND_TRACE_TEST_2
    if token.startswith("trace_"):
        return KIND_TRACE_TEST_1
    return None


def default_title_for_godot_kind(kind: str, test_code: str) -> str:
    normalized = str(kind)
    if normalized == KIND_AUDITORY_CAPACITY:
        return "Auditory Capacity"
    if normalized == KIND_RAPID_TRACKING:
        return "Rapid Tracking"
    if normalized == KIND_SPATIAL_INTEGRATION:
        return "Spatial Integration"
    if normalized == KIND_TRACE_TEST_1:
        return "Trace Test 1"
    if normalized == KIND_TRACE_TEST_2:
        return "Trace Test 2"
    return str(test_code).replace("_", " ").title()


def _default_phase_screens(kind: str, title: str) -> dict[str, object]:
    base_title = str(title)
    live_hint = "Use the Godot window controls. Esc opens pause."
    if kind == KIND_AUDITORY_CAPACITY:
        instruction = (
            "Fly the ball through the tunnel, respond to spoken commands, and use the cue keys when prompted."
        )
        controls = "Arrow keys or joystick move the ball. Q/W/E/R change ball colour. Space answers beep cues."
    elif kind == KIND_RAPID_TRACKING:
        instruction = "Keep the centered sight on the active target as the camera moves through the scene."
        controls = "Mouse, arrow keys, or joystick aim the camera. Space or Enter captures."
    elif kind == KIND_SPATIAL_INTEGRATION:
        instruction = "Study each generated scene from its shown views, then answer map and route questions."
        controls = "Mouse selects cells/cards. Number keys choose options. Enter submits."
    elif kind == KIND_TRACE_TEST_1:
        instruction = (
            "Watch the red aircraft and answer with the stick movement that would have produced "
            "the observed orientation change."
        )
        controls = (
            "Left/Right yaw left/right. Up/forward stick pushes the nose down. "
            "Down/back stick pulls the nose up. You are scored on matching the correct stick movement "
            "before the response window closes."
        )
    elif kind == KIND_TRACE_TEST_2:
        instruction = (
            "Watch a short clip of several aircraft moving one point at a time, then answer a "
            "full-screen multiple-choice question about what happened in the clip."
        )
        controls = (
            "Use 1-4 or A/S/D/F to choose the aircraft. Questions can ask which aircraft started "
            "on top, made a left or right turn, ended highest, left the screen, ended off screen, "
            "or did not change direction."
        )
    else:
        instruction = "Complete the Godot-owned 3D test."
        controls = live_hint
    return {
        "instructions": {
            "title": base_title,
            "heading": "Instructions",
            "body": instruction,
            "controls": controls,
            "footer": "Press Enter, Space, or numpad Del to start practice.",
        },
        "practice": {
            "title": base_title,
            "heading": "Practice",
            "body": "This is a short untimed-feeling warmup before the scored block.",
            "controls": live_hint,
        },
        "practice_done": {
            "title": base_title,
            "heading": "Practice Complete",
            "body": "The scored block is next.",
            "footer": "Press Enter, Space, or numpad Del to start the test.",
        },
        "results": {
            "title": base_title,
            "heading": "Results",
            "body": "Your result has been sent back to Python for saving.",
            "footer": "Press Enter, Space, or numpad Del to continue.",
        },
    }


@dataclass(frozen=True, slots=True)
class GodotOwnedTestSpec:
    kind: str
    test_code: str
    title: str
    seed: int
    difficulty: float
    launch_id: str
    run_key: str
    phase: str
    duration_s: float
    practice_enabled: bool = False
    practice_duration_s: float = 0.0
    mode: str = "standard"
    config: dict[str, object] = field(default_factory=dict)
    audio: dict[str, object] = field(default_factory=dict)
    assets: dict[str, object] = field(default_factory=dict)
    phase_screens: dict[str, object] = field(default_factory=dict)
    result_metadata: dict[str, object] = field(default_factory=dict)

    def with_phase(self, phase: Phase | str) -> "GodotOwnedTestSpec":
        phase_token = str(getattr(phase, "value", phase)).strip().lower() or "unknown"
        return replace(
            self,
            phase=phase_token,
            run_key=f"{self.test_code}:{self.seed}:{self.mode}:{self.launch_id}:{phase_token}",
        )


@dataclass(frozen=True, slots=True)
class GodotOwnedPayload:
    spec: GodotOwnedTestSpec
    progress: dict[str, object] = field(default_factory=dict)
    error: dict[str, object] | None = None


@dataclass(frozen=True, slots=True)
class GodotOwnedSummary:
    attempted: int
    correct: int
    accuracy: float
    duration_s: float
    throughput_per_min: float
    mean_response_time_s: float | None
    total_score: float = 0.0
    max_score: float = 0.0
    score_ratio: float = 0.0
    difficulty_level_start: int | None = None
    difficulty_level_end: int | None = None
    difficulty_change_count: int = 0


def _finite_float(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(number):
        return float(default)
    return float(number)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    number = _finite_float(value, default=float("nan"))
    if not math.isfinite(number):
        return None
    return float(number)


def _coerce_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _as_dict(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    return {}


def _as_list(value: object) -> list[object]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return []


def _difficulty_level_from_ratio(value: object) -> int:
    ratio = max(0.0, min(1.0, _finite_float(value, 0.5)))
    return max(1, min(10, int(round(ratio * 9.0)) + 1))


def _summary_from_mapping(
    data: Mapping[str, object],
    *,
    fallback_duration_s: float,
    fallback_difficulty: float,
) -> GodotOwnedSummary:
    attempted = max(0, _coerce_int(data.get("attempted", data.get("attempts", 0))))
    correct = max(0, _coerce_int(data.get("correct", data.get("hits", 0))))
    accuracy = _finite_float(data.get("accuracy", 0.0 if attempted <= 0 else correct / attempted))
    duration_s = max(0.0, _finite_float(data.get("duration_s", fallback_duration_s)))
    throughput = _finite_float(
        data.get(
            "throughput_per_min",
            0.0 if duration_s <= 0.0 else (float(attempted) / duration_s) * 60.0,
        )
    )
    total_score = _finite_float(data.get("total_score", data.get("score", float(correct))))
    max_score = _finite_float(data.get("max_score", max(float(attempted), total_score, 0.0)))
    score_ratio = _finite_float(
        data.get("score_ratio", 0.0 if max_score <= 0.0 else total_score / max_score)
    )
    level = _difficulty_level_from_ratio(fallback_difficulty)
    return GodotOwnedSummary(
        attempted=attempted,
        correct=correct,
        accuracy=max(0.0, min(1.0, accuracy)),
        duration_s=duration_s,
        throughput_per_min=throughput,
        mean_response_time_s=_optional_float(data.get("mean_response_time_s")),
        total_score=total_score,
        max_score=max_score,
        score_ratio=max(0.0, min(1.0, score_ratio)),
        difficulty_level_start=_coerce_int(data.get("difficulty_level_start", level), level),
        difficulty_level_end=_coerce_int(data.get("difficulty_level_end", level), level),
        difficulty_change_count=max(0, _coerce_int(data.get("difficulty_change_count", 0))),
    )


def _event_from_mapping(raw: Mapping[str, object], *, seq: int, fallback_kind: str) -> TelemetryEvent:
    extra = raw.get("extra")
    if not isinstance(extra, Mapping):
        extra = {
            str(key): value
            for key, value in raw.items()
            if key
            not in {
                "family",
                "kind",
                "phase",
                "seq",
                "item_index",
                "is_scored",
                "is_correct",
                "is_timeout",
                "response_time_ms",
                "score",
                "max_score",
                "difficulty_level",
                "occurred_at_ms",
                "prompt",
                "expected",
                "response",
                "extra",
            }
        } or None
    response_time_ms = raw.get("response_time_ms")
    if response_time_ms is None and raw.get("response_time_s") is not None:
        response_time_ms = int(round(_finite_float(raw.get("response_time_s")) * 1000.0))
    occurred_at_ms = raw.get("occurred_at_ms")
    if occurred_at_ms is None and raw.get("occurred_at_s") is not None:
        occurred_at_ms = int(round(_finite_float(raw.get("occurred_at_s")) * 1000.0))
    return TelemetryEvent(
        family=str(raw.get("family", fallback_kind)),
        kind=str(raw.get("kind", "event")),
        phase=str(raw.get("phase", Phase.SCORED.value)),
        seq=_coerce_int(raw.get("seq", seq), seq),
        item_index=None if raw.get("item_index") is None else _coerce_int(raw.get("item_index")),
        is_scored=bool(
            raw.get("is_scored", str(raw.get("phase", Phase.SCORED.value)) == Phase.SCORED.value)
        ),
        is_correct=None if raw.get("is_correct") is None else bool(raw.get("is_correct")),
        is_timeout=bool(raw.get("is_timeout", False)),
        response_time_ms=None if response_time_ms is None else _coerce_int(response_time_ms),
        score=None if raw.get("score") is None else _finite_float(raw.get("score")),
        max_score=None if raw.get("max_score") is None else _finite_float(raw.get("max_score")),
        difficulty_level=(
            None if raw.get("difficulty_level") is None else _coerce_int(raw.get("difficulty_level"))
        ),
        occurred_at_ms=None if occurred_at_ms is None else _coerce_int(occurred_at_ms),
        prompt=None if raw.get("prompt") is None else str(raw.get("prompt")),
        expected=None if raw.get("expected") is None else str(raw.get("expected")),
        response=None if raw.get("response") is None else str(raw.get("response")),
        extra=dict(extra) if isinstance(extra, Mapping) and extra else None,
    )


class GodotOwnedTestEngine:
    """Thin Python shell for tests whose live runtime is authoritative in Godot."""

    def __init__(
        self,
        *,
        clock: Clock,
        kind: str,
        test_code: str,
        title: str,
        seed: int,
        difficulty: float,
        duration_s: float = 180.0,
        mode: str = "standard",
        practice_enabled: bool = False,
        practice_duration_s: float = 0.0,
        config: Mapping[str, object] | None = None,
        audio: Mapping[str, object] | None = None,
        assets: Mapping[str, object] | None = None,
        result_metadata: Mapping[str, object] | None = None,
    ) -> None:
        if kind not in GODOT_OWNED_KINDS:
            raise ValueError(f"unsupported Godot-owned test kind: {kind}")
        self._clock = clock
        self._kind = str(kind)
        self._test_code = str(test_code)
        self._title = str(title)
        self._seed = int(seed)
        self._difficulty = max(0.0, min(1.0, _finite_float(difficulty, 0.5)))
        self._scored_duration_s = max(1.0, _finite_float(duration_s, 180.0))
        self._practice_enabled = bool(practice_enabled)
        self._practice_duration_s = max(0.0, _finite_float(practice_duration_s, 0.0))
        self._mode = str(mode or "standard")
        self._launch_id = f"{id(self):x}"
        self._config = _as_dict(config or {})
        self._audio = _as_dict(audio or {})
        self._assets = _as_dict(assets or {})
        self._result_metadata = _as_dict(result_metadata or {})
        self._phase = Phase.INSTRUCTIONS
        self._started_at_s: float | None = None
        self._progress: dict[str, object] = {}
        self._events: list[TelemetryEvent] = []
        self._metrics: dict[str, object] = {
            "renderer_backend": "godot_4",
            "godot_authority": "1",
            "godot_kind": self._kind,
            "godot_test_code": self._test_code,
            "godot_mode": self._mode,
        }
        self._summary = _summary_from_mapping(
            {},
            fallback_duration_s=self._scored_duration_s,
            fallback_difficulty=self._difficulty,
        )
        self._error: dict[str, object] | None = None
        self._result_metrics_overrides: dict[str, str] = {}

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def difficulty(self) -> float:
        return self._difficulty

    @property
    def practice_questions(self) -> int:
        return 1 if self._practice_enabled else 0

    @property
    def scored_duration_s(self) -> float:
        return self._scored_duration_s

    @property
    def phase(self) -> Phase:
        return self._phase

    def can_exit(self) -> bool:
        return self._phase in (Phase.INSTRUCTIONS, Phase.PRACTICE_DONE, Phase.RESULTS)

    def start_practice(self) -> None:
        if self._phase is Phase.INSTRUCTIONS:
            self._phase = Phase.PRACTICE if self._practice_enabled else Phase.PRACTICE_DONE
            if self._phase is Phase.PRACTICE:
                self._started_at_s = float(self._clock.now())

    def start_scored(self) -> None:
        if self._phase is Phase.RESULTS:
            return
        self._phase = Phase.SCORED
        self._started_at_s = float(self._clock.now())
        self._progress.clear()
        self._error = None

    def submit_answer(self, raw: str) -> bool:
        token = str(raw).strip().lower()
        if token in {"__skip_section__", "__timeout__", "__force_complete__"}:
            self._complete_without_godot()
            return True
        if self._phase is Phase.INSTRUCTIONS:
            self.start_practice()
            return True
        if self._phase is Phase.PRACTICE:
            self._phase = Phase.PRACTICE_DONE
            return True
        if self._phase is Phase.PRACTICE_DONE:
            self.start_scored()
            return True
        return self._phase is Phase.RESULTS

    def update(self) -> None:
        return

    def snapshot(self) -> TestSnapshot:
        payload = self._payload()
        if self._phase is Phase.INSTRUCTIONS:
            return TestSnapshot(
                title=self._title,
                phase=Phase.INSTRUCTIONS,
                prompt=(
                    "This 3D test runs in the Godot window.\n"
                    "Press Enter to stage the Godot-owned runtime."
                ),
                input_hint="Press Enter to continue",
                time_remaining_s=None,
                attempted_scored=0,
                correct_scored=0,
                payload=payload,
            )
        if self._phase is Phase.PRACTICE:
            return TestSnapshot(
                title=self._title,
                phase=Phase.PRACTICE,
                prompt="Godot practice runtime active.",
                input_hint="Use the Godot window controls. Press Enter here to continue.",
                time_remaining_s=None,
                attempted_scored=0,
                correct_scored=0,
                payload=payload,
            )
        if self._phase is Phase.PRACTICE_DONE:
            return TestSnapshot(
                title=self._title,
                phase=Phase.PRACTICE_DONE,
                prompt="Godot runtime staged. Press Enter to start the scored block.",
                input_hint="Press Enter to continue",
                time_remaining_s=None,
                attempted_scored=0,
                correct_scored=0,
                payload=payload,
            )
        if self._phase is Phase.RESULTS:
            summary = self._summary
            detail = ""
            if self._error is not None:
                detail = "\nGodot error: " + str(
                    self._error.get("detail", self._error.get("reason", "unknown"))
                )
            return TestSnapshot(
                title=self._title,
                phase=Phase.RESULTS,
                prompt=(
                    "Results\n"
                    f"Attempted: {summary.attempted}\n"
                    f"Correct: {summary.correct}\n"
                    f"Accuracy: {summary.accuracy * 100.0:.1f}%\n"
                    f"Score ratio: {summary.score_ratio * 100.0:.1f}%"
                    f"{detail}"
                ),
                input_hint="Press Enter to continue",
                time_remaining_s=0.0,
                attempted_scored=summary.attempted,
                correct_scored=summary.correct,
                payload=payload,
            )
        elapsed = self._elapsed_s()
        remaining = max(0.0, self._scored_duration_s - elapsed)
        return TestSnapshot(
            title=self._title,
            phase=Phase.SCORED,
            prompt="Godot runtime active. Keep focus in the Godot window.",
            input_hint="Godot owns live controls. Esc opens pause.",
            time_remaining_s=remaining,
            attempted_scored=int(self._progress.get("attempted", self._summary.attempted)),
            correct_scored=int(self._progress.get("correct", self._summary.correct)),
            payload=payload,
        )

    def events(self) -> list[TelemetryEvent]:
        return list(self._events)

    def result_metrics(self) -> dict[str, object]:
        return dict(self._metrics)

    def scored_summary(self) -> GodotOwnedSummary:
        return self._summary

    def apply_godot_authoritative_message(self, message: Mapping[str, object]) -> None:
        command = str(message.get("command", "")).strip().lower()
        if command.startswith("auditory_"):
            command = "godot_" + command.removeprefix("auditory_")
        if command in {"ready", "progress", "event", "complete", "error"}:
            command = "godot_" + command
        if command in {"godot_phase_advance", "godot_phase_complete"}:
            self.submit_answer("__godot_phase__")
            return
        if command == "godot_ready":
            self._metrics["godot_ready"] = "1"
            self._metrics["godot_run_key"] = str(message.get("run_key", ""))
            return
        if command == "godot_progress":
            self._progress.update(_as_dict(message.get("progress", message)))
            return
        if command == "godot_event":
            event_data = _as_dict(message.get("event", message))
            self._events.append(
                _event_from_mapping(event_data, seq=len(self._events), fallback_kind=self._kind)
            )
            return
        if command == "godot_complete":
            self._apply_completion(message)
            return
        if command == "godot_error":
            self._apply_error(message)

    def _payload(self) -> GodotOwnedPayload:
        return GodotOwnedPayload(
            spec=self._spec().with_phase(self._phase),
            progress=dict(self._progress),
            error=None if self._error is None else dict(self._error),
        )

    def _spec(self) -> GodotOwnedTestSpec:
        return GodotOwnedTestSpec(
            kind=self._kind,
            test_code=self._test_code,
            title=self._title,
            seed=self._seed,
            difficulty=self._difficulty,
            launch_id=self._launch_id,
            run_key=f"{self._test_code}:{self._seed}:{self._mode}:{self._launch_id}:{self._phase.value}",
            phase=self._phase.value,
            duration_s=self._scored_duration_s,
            practice_enabled=self._practice_enabled,
            practice_duration_s=self._practice_duration_s,
            mode=self._mode,
            config=dict(self._config),
            audio=dict(self._audio),
            assets=dict(self._assets),
            phase_screens=_default_phase_screens(self._kind, self._title),
            result_metadata=dict(self._result_metadata),
        )

    def _elapsed_s(self) -> float:
        if self._started_at_s is None:
            return 0.0
        return max(0.0, float(self._clock.now()) - float(self._started_at_s))

    def _apply_completion(self, message: Mapping[str, object]) -> None:
        result = _as_dict(message.get("result", message))
        summary_data = _as_dict(result.get("summary", message.get("summary", {})))
        metrics = _as_dict(result.get("metrics", message.get("metrics", {})))
        event_values = _as_list(result.get("events", message.get("events", [])))
        self._summary = _summary_from_mapping(
            summary_data,
            fallback_duration_s=self._scored_duration_s,
            fallback_difficulty=self._difficulty,
        )
        for key, value in metrics.items():
            self._metrics[str(key)] = value
        self._metrics["godot_complete"] = "1"
        self._metrics["godot_authority"] = "1"
        self._events = []
        for raw_event in event_values:
            if isinstance(raw_event, Mapping):
                self._events.append(
                    _event_from_mapping(raw_event, seq=len(self._events), fallback_kind=self._kind)
                )
        self._phase = Phase.RESULTS

    def _apply_error(self, message: Mapping[str, object]) -> None:
        self._error = {
            "reason": str(message.get("reason", "godot_error")),
            "detail": str(message.get("detail", message.get("error", "Godot runtime error"))),
        }
        self._metrics["godot_error"] = str(self._error["reason"])
        self._metrics["godot_error_detail"] = str(self._error["detail"])
        self._summary = _summary_from_mapping(
            {},
            fallback_duration_s=self._elapsed_s(),
            fallback_difficulty=self._difficulty,
        )
        self._phase = Phase.RESULTS

    def _complete_without_godot(self) -> None:
        self._summary = _summary_from_mapping(
            {
                "attempted": int(self._progress.get("attempted", 0) or 0),
                "correct": int(self._progress.get("correct", 0) or 0),
                "duration_s": self._elapsed_s() or self._scored_duration_s,
            },
            fallback_duration_s=self._elapsed_s() or self._scored_duration_s,
            fallback_difficulty=self._difficulty,
        )
        self._metrics["godot_complete"] = "0"
        self._metrics["manual_completion"] = "1"
        self._phase = Phase.RESULTS


def build_godot_owned_test(
    *,
    clock: Clock,
    seed: int,
    difficulty: float,
    kind: str | None = None,
    test_code: str,
    title: str | None = None,
    duration_s: float | None = None,
    mode: str = "standard",
    practice_enabled: bool = True,
    practice_duration_s: float = 12.0,
    config: Mapping[str, object] | None = None,
    audio: Mapping[str, object] | None = None,
    assets: Mapping[str, object] | None = None,
    result_metadata: Mapping[str, object] | None = None,
) -> GodotOwnedTestEngine:
    resolved_kind = kind or godot_kind_for_test_code(test_code)
    if resolved_kind is None:
        raise ValueError(f"test code is not Godot-owned: {test_code}")
    resolved_title = title or default_title_for_godot_kind(resolved_kind, test_code)
    return GodotOwnedTestEngine(
        clock=clock,
        kind=resolved_kind,
        test_code=test_code,
        title=resolved_title,
        seed=seed,
        difficulty=difficulty,
        duration_s=180.0 if duration_s is None else float(duration_s),
        mode=mode,
        practice_enabled=practice_enabled,
        practice_duration_s=practice_duration_s,
        config=config,
        audio=audio,
        assets=assets,
        result_metadata=result_metadata,
    )

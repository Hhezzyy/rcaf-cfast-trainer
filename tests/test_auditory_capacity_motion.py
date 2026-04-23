from __future__ import annotations

import pytest

from cfast_trainer.auditory_capacity_motion import (
    AuditoryTunnelFollower,
    AuditoryTunnelFollowerConfig,
    auditory_ball_bound,
    auditory_gate_world_distance_from_x_norm,
    auditory_gate_x_norm_from_world_distance,
    auditory_tunnel_wall_bound,
    auditory_wall_collision_active,
    auditory_wall_contact_ratio,
)


def _follower(*, speed: float = 3.0) -> AuditoryTunnelFollower:
    return AuditoryTunnelFollower(
        AuditoryTunnelFollowerConfig(
            session_seed=17,
            speed_distance_per_s=float(speed),
            ball_radius=0.11,
            tunnel_inner_rx=2.24,
            tunnel_inner_rz=1.64,
            wall_half_length=4.0,
            roll_deg_per_distance=9.0,
        )
    )


def test_tunnel_follower_advances_by_speed_times_delta_time() -> None:
    follower = _follower(speed=3.5)
    start = follower.snapshot()

    after = follower.update(0.40)

    assert after.travel_distance == pytest.approx(start.travel_distance + 1.4)
    assert after.ball_distance == pytest.approx(after.travel_distance)


def test_tunnel_follower_motion_is_independent_of_frame_subdivision() -> None:
    whole = _follower(speed=2.25)
    split = _follower(speed=2.25)

    whole_snapshot = whole.update(0.30)
    split.update(0.10)
    split.update(0.20)
    split_snapshot = split.snapshot()

    assert split_snapshot.travel_distance == pytest.approx(whole_snapshot.travel_distance)
    assert split_snapshot.center == pytest.approx(whole_snapshot.center)


def test_tunnel_follower_updates_centerline_frame_and_roll() -> None:
    follower = _follower(speed=5.0)
    start = follower.snapshot()

    after = follower.update(4.0)

    assert after.center != pytest.approx(start.center)
    assert after.tangent != pytest.approx(start.tangent)
    assert after.ball_roll_deg != pytest.approx(start.ball_roll_deg)


def test_gate_world_distance_projection_round_trips_without_moving_gate() -> None:
    travel = 12.0
    x_norm = 0.42

    world_distance = auditory_gate_world_distance_from_x_norm(
        x_norm,
        travel_distance=travel,
        spawn_x_norm=1.65,
        player_x_norm=0.0,
        retire_x_norm=-1.25,
    )

    assert auditory_gate_x_norm_from_world_distance(
        world_distance,
        travel_distance=travel,
        spawn_x_norm=1.65,
        player_x_norm=0.0,
        retire_x_norm=-1.25,
    ) == pytest.approx(x_norm)
    assert auditory_gate_x_norm_from_world_distance(
        world_distance,
        travel_distance=travel + 2.0,
        spawn_x_norm=1.65,
        player_x_norm=0.0,
        retire_x_norm=-1.25,
    ) < x_norm


def test_gate_projection_uses_travel_distance_as_ball_world_position() -> None:
    travel = 18.0

    far = auditory_gate_world_distance_from_x_norm(
        1.65,
        travel_distance=travel,
        spawn_x_norm=1.65,
        player_x_norm=0.0,
        retire_x_norm=-1.25,
    )
    contact = auditory_gate_world_distance_from_x_norm(
        0.0,
        travel_distance=travel,
        spawn_x_norm=1.65,
        player_x_norm=0.0,
        retire_x_norm=-1.25,
    )
    behind = auditory_gate_world_distance_from_x_norm(
        -1.25,
        travel_distance=travel,
        spawn_x_norm=1.65,
        player_x_norm=0.0,
        retire_x_norm=-1.25,
    )

    assert far > travel
    assert contact == pytest.approx(travel)
    assert behind < travel


def test_sphere_wall_collision_increments_once_per_collision_episode() -> None:
    follower = _follower(speed=0.0)
    snapshot = follower.snapshot()
    wall = auditory_tunnel_wall_bound(
        snapshot,
        half_length=4.0,
        inner_rx=2.24,
        inner_rz=1.64,
    )
    inside = auditory_ball_bound(
        snapshot,
        x=0.0,
        y=0.0,
        tube_half_width=0.82,
        tube_half_height=0.60,
        radius=0.11,
        inner_rx=2.24,
        inner_rz=1.64,
    )
    outside = auditory_ball_bound(
        snapshot,
        x=0.84,
        y=0.0,
        tube_half_width=0.82,
        tube_half_height=0.60,
        radius=0.11,
        inner_rx=2.24,
        inner_rz=1.64,
    )

    assert auditory_wall_contact_ratio(ball_bound=inside, wall_bound=wall) < 1.0
    assert auditory_wall_collision_active(ball_bound=outside, wall_bound=wall)

    first = follower.set_collision_state(active=True, contact_ratio=1.02)
    repeated = follower.set_collision_state(active=True, contact_ratio=1.04)
    cleared = follower.set_collision_state(active=False, contact_ratio=0.20)
    second = follower.set_collision_state(active=True, contact_ratio=1.05)

    assert first.collision_penalties == 1
    assert repeated.collision_penalties == 1
    assert not cleared.collision_active
    assert second.collision_penalties == 2

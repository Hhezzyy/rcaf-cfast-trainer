from __future__ import annotations

import inspect

import cfast_trainer.app as app_module


def test_app_standard_3d_tests_route_to_godot_owned_runtime() -> None:
    source = inspect.getsource(app_module.run)

    assert "build_godot_owned_test(" in source
    assert "auditory_capacity_godot_config(" in source
    assert "rapid_tracking_godot_config(" in source
    assert "spatial_integration_godot_config(" in source
    assert "trace_test_godot_config(" in source


def test_app_individual_3d_drill_wrappers_use_godot_owned_runtime() -> None:
    source = inspect.getsource(app_module.run)

    assert "_open_rt_drill" in source
    assert "_open_si_drill" in source
    assert "_open_trace_drill" in source
    assert "_open_auditory_capacity_drill" in source
    assert 'kind="rapid_tracking"' in source
    assert 'kind="spatial_integration"' in source
    assert 'kind=godot_kind_for_test_code(test_code) or "trace_test_1"' in source
    assert 'kind="auditory_capacity"' in source


def test_app_visible_3d_drill_menu_entries_use_canonical_codes() -> None:
    source = inspect.getsource(app_module.run)

    assert '"rt_obscured_target_prediction"' in source
    assert '"si_static_multiview_integration"' in source
    assert '"si_moving_aircraft_multiview_integration"' in source
    assert '"trace_orientation_decode"' in source
    assert '"trace_movement_recall"' in source
    assert '("rt_terrain_recovery_run", "Terrain Recovery Run"' not in source
    assert '("si_static_mixed_run", "Static Mixed Run"' not in source
    assert '("si_aircraft_grid_run", "Aircraft Grid Run"' not in source
    assert '("tt1_command_switch_run", "TT1 Command Switch Run"' not in source
    assert '("tt2_position_recall_run", "TT2 Position Recall Run"' not in source


def test_app_rapid_tracking_workout_uses_block_workout_shell() -> None:
    source = inspect.getsource(app_module.run)

    assert 'if token == "rapid_tracking_workout":' in source
    assert 'return build_rt_workout_plan()' in source
    assert 'token != "rapid_tracking_workout"' in source

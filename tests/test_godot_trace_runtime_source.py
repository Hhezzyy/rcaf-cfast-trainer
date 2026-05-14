from __future__ import annotations

from pathlib import Path


MAIN_PATH = (
    Path(__file__).resolve().parents[1]
    / "godot"
    / "cfast_3d"
    / "scripts"
    / "main.gd"
)
GODOT_OWNED_RUNTIME_PATH = (
    Path(__file__).resolve().parents[1]
    / "godot"
    / "cfast_3d"
    / "scripts"
    / "godot_owned_runtime.gd"
)


def _source() -> str:
    return MAIN_PATH.read_text(encoding="utf-8")


def _runtime_source() -> str:
    return GODOT_OWNED_RUNTIME_PATH.read_text(encoding="utf-8")


def _function_body(source: str, name: str) -> str:
    marker = f"func {name}"
    start = source.index(marker)
    next_func = source.find("\nfunc ", start + len(marker))
    return source[start:] if next_func < 0 else source[start:next_func]


def test_trace_test_2_draws_current_aircraft_only_without_trails() -> None:
    source = _source()
    body = _function_body(source, "_present_trace_test_2")

    assert "_make_aircraft" in body
    assert "_draw_waypoints" not in source
    assert "Trace2Waypoint" not in source
    assert "show_trails" not in body
    assert "trial_stage" not in body
    assert "_make_segment" not in body


def test_trace_presenters_use_guide_cameras_and_no_lattice() -> None:
    source = _source()
    trace_1 = _function_body(source, "_present_trace_test_1")
    trace_2 = _function_body(source, "_present_trace_test_2")
    camera_helper = _function_body(source, "_set_trace_guide_camera")

    assert "TRACE_TEST_1_CAMERA_POSITION" in source
    assert "TRACE_TEST_1_CAMERA_TARGET" in source
    assert "TRACE_TEST_2_CAMERA_POSITION" in source
    assert "TRACE_TEST_2_CAMERA_TARGET" in source
    assert "_set_trace_guide_camera(1)" in trace_1
    assert "_set_trace_guide_camera(2)" in trace_2
    assert "_make_trace_guide_stage()" in trace_1
    assert "_make_trace_guide_stage()" in trace_2
    assert "TRACE_TEST_1_CAMERA_POSITION" in camera_helper
    assert "TRACE_TEST_2_CAMERA_POSITION" in camera_helper
    assert "TraceGuidePanel" in source
    assert "_draw_lattice" not in source
    assert "LatticeX" not in source
    assert "LatticeZ" not in source


def test_godot_owned_trace_runtime_draws_no_track_history_trails() -> None:
    source = _runtime_source()
    draw_body = _function_body(source, "_draw_trace_scene")

    assert "_make_aircraft" in draw_body
    assert "_make_reticle" not in draw_body
    assert "Color(0.58, 0.51, 0.46" not in draw_body
    assert "_draw_trace_track_history" not in source
    assert "Trace2Waypoint" not in source
    assert "_make_segment(previous, world" not in source
    assert "_make_sphere(world, 0.052" not in source


def test_godot_owned_trace_runtime_uses_fixed_camera_and_unframed_trace1_backdrop() -> None:
    source = _runtime_source()
    draw_body = _function_body(source, "_draw_trace_scene")
    camera_body = _function_body(source, "_update_camera")
    fixed_camera_body = _function_body(source, "_update_fixed_square_trace_camera")

    assert "TRACE_FIXED_ORTHO_CAMERA_OFFSET" in source
    assert "TRACE_FIXED_ORTHO_CAMERA_TARGET" in source
    assert "TRACE_GUIDE_PANEL_COLOR" in source
    assert "TRACE_PANEL_VISUAL_SCALE := 2.0" in source
    assert "_update_fixed_square_trace_camera(camera)" in camera_body
    assert "Camera3D.PROJECTION_ORTHOGONAL" in fixed_camera_body
    assert "trace_panel_visual_scale" in fixed_camera_body
    assert "camera.position = center + TRACE_FIXED_ORTHO_CAMERA_OFFSET" in fixed_camera_body
    assert "camera.look_at(center, Vector3.UP)" in fixed_camera_body
    assert 'if kind == "trace_test_1"' in draw_body
    assert "var fill := maxf(panel_w, panel_h) * 2.0" in draw_body
    assert "Vector3(fill, fill, 0.04), SKY_COLOR" in draw_body
    assert "visual_width" in fixed_camera_body
    assert "visual_height" in fixed_camera_body
    assert "TRACE_GUIDE_PANEL_COLOR" in draw_body
    assert "trace_panel_visual_scale" in draw_body
    assert "trace_grid_half_x" in draw_body
    assert "trace_grid_max_y" in draw_body
    assert "Vector3(0.0, 4.8, 11.2)" not in source
    assert "Vector3(0.0, 1.6, -7.2)" not in source


def test_godot_owned_trace_test_1_uses_stick_response_scoring() -> None:
    source = _runtime_source()
    step_body = _function_body(source, "_step")
    key_body = _function_body(source, "handle_key")
    submit_body = _function_body(source, "_trace1_submit_response")
    turn_body = _function_body(source, "_trace1_response_for_turn")
    joystick_body = _function_body(source, "_trace1_poll_joystick_response")
    order_body = _function_body(source, "_trace1_response_order")
    open_body = _function_body(source, "_trace1_open_response")

    assert "TRACE1_RESPONSE_LEFT_YAW" in source
    assert "TRACE1_RESPONSE_RIGHT_YAW" in source
    assert "TRACE1_RESPONSE_PUSH_PITCH_DOWN" in source
    assert "TRACE1_RESPONSE_PULL_PITCH_UP" in source
    assert 'if kind == "trace_test_1":' in step_body
    assert "_trace1_poll_joystick_response()" in step_body
    assert "_trace1_update_timeout()" in step_body
    assert "_sample_score(\"sample\")" not in step_body.split('if kind == "trace_test_1":', 1)[1].split("if _is_trace_kind()", 1)[0]
    assert "KEY_LEFT" in key_body and "TRACE1_RESPONSE_LEFT_YAW" in key_body
    assert "KEY_RIGHT" in key_body and "TRACE1_RESPONSE_RIGHT_YAW" in key_body
    assert "KEY_UP" in key_body and "TRACE1_RESPONSE_PUSH_PITCH_DOWN" in key_body
    assert "KEY_DOWN" in key_body and "TRACE1_RESPONSE_PULL_PITCH_UP" in key_body
    assert '"kind": "orientation_response"' in submit_body
    assert '"expected": expected' in submit_body
    assert '"response": response' in submit_body
    assert '"response_time_ms": response_time_ms' in submit_body
    assert "return TRACE1_RESPONSE_PUSH_PITCH_DOWN" in turn_body
    assert "return TRACE1_RESPONSE_PULL_PITCH_UP" in turn_body
    assert "JOY_AXIS_LEFT_X" in joystick_body
    assert "JOY_AXIS_LEFT_Y" in joystick_body
    assert "y < 0.0 else TRACE1_RESPONSE_PULL_PITCH_UP" in joystick_body
    assert "session_seed + trace1_prompt_index + _kind_salt(test_code)" in order_body
    assert "trace1_prompt_hash = _trace_hash_mix" in open_body


def test_godot_owned_trace_test_2_uses_clip_then_multiple_choice_questions() -> None:
    source = _runtime_source()
    step_body = _function_body(source, "_step")
    key_body = _function_body(source, "handle_key")
    trace2_step_body = _function_body(source, "_trace2_step")
    submit_body = _function_body(source, "_trace2_submit_answer")
    stem_body = _function_body(source, "_trace2_question_stem")
    config_body = _function_body(source, "_configure_trace_runtime")

    assert "TRACE2_QUESTION_STARTED_TOP" in source
    assert "TRACE2_QUESTION_TURNED_LEFT" in source
    assert "TRACE2_QUESTION_TURNED_RIGHT" in source
    assert "TRACE2_QUESTION_NO_TURN" in source
    assert "TRACE2_QUESTION_ENDED_OFF_SCREEN" in source
    assert "TRACE2_QUESTION_LEFT_SCREEN" in source
    assert "TRACE2_QUESTION_ENDED_OFF_RIGHT" in source
    assert "Trace2QuestionCanvasLayer" in source
    assert "Trace2QuestionOverlay" in source
    assert "_trace2_open_question()" in trace2_step_body
    assert "_show_trace2_question_overlay()" in _function_body(source, "_trace2_open_question")
    assert 'if kind == "trace_test_2":' in step_body
    assert "_trace2_step(dt)" in step_body
    assert "_sample_score(\"sample\")" not in step_body.split('if kind == "trace_test_2":', 1)[1].split("if _is_trace_kind()", 1)[0]
    assert "_trace2_option_for_key(key)" in key_body
    assert "KEY_1" in source and "KEY_A" in source
    assert "trace2_stage == \"observe\"" in trace2_step_body
    assert "trace2_stage = \"question\"" in _function_body(source, "_trace2_open_question")
    assert "_trace2_submit_answer(0, true)" in trace2_step_body
    assert "_step_trace_tracks(dt)" in trace2_step_body
    assert "trace2_stage == \"question\"" in _function_body(source, "_draw_trace_scene")
    assert '"kind": "clip_question"' in submit_body
    assert '"expected": str(trace2_correct_option)' in submit_body
    assert '"response": response' in submit_body
    assert "_hide_trace2_question_overlay()" in submit_body
    assert "_trace2_start_trial()" in submit_body
    assert "Which aircraft started on top?" in stem_body
    assert "Which aircraft made a left turn?" in stem_body
    assert "Which aircraft ended off screen?" in stem_body
    assert "Which aircraft left the screen?" in stem_body
    assert 'config.get("trace2_observe_s"' in config_body
    assert 'config.get("trace2_clip_min_s"' in config_body
    assert 'config.get("trace2_clip_max_s"' in config_body
    assert 'config.get("trace2_question_window_s"' in config_body
    assert 'config.get("trace2_offscreen_margin_cells"' in config_body
    assert 'config.get("trace2_close_camera_scale"' in config_body
    assert 'config.get("trace2_aircraft_scale"' in config_body


def test_godot_owned_trace_test_2_uses_close_offscreen_clips_and_overlay() -> None:
    source = _runtime_source()
    start_body = _function_body(source, "_trace2_start_trial")
    camera_body = _function_body(source, "_update_fixed_square_trace_camera")
    answer_body = _function_body(source, "_trace2_answer_for_question")
    offscreen_body = _function_body(source, "_trace2_unique_offscreen_code")
    hud_body = _function_body(source, "_update_hud")

    assert "local.randf_range(trace2_clip_min_s, trace2_clip_max_s)" in start_body
    assert '"exit"' in start_body
    assert "_trace2_start_cell_for_role(role)" in start_body
    assert '"went_offscreen": false' in start_body
    assert "trace2_close_camera_scale" in camera_body
    assert "trace2_aircraft_scale" in source
    assert 'return _trace2_unique_offscreen_code(true, "")' in answer_body
    assert 'return _trace2_unique_offscreen_code(false, "")' in answer_body
    assert 'return _trace2_unique_offscreen_code(true, "right")' in answer_body
    assert "_trace2_code_for_role(\"exit\")" in offscreen_body
    assert "_trace2_question_stem(trace2_question_kind)" not in hud_body


def test_godot_owned_trace_runtime_uses_larger_slower_config_without_rule_changes() -> None:
    source = _runtime_source()
    config_body = _function_body(source, "_configure_trace_runtime")
    init_body = _function_body(source, "_init_trace_tracks")
    bounds_body = _function_body(source, "_trace_cell_in_bounds")
    turn_body = _function_body(source, "_trace_choose_next_dir")
    begin_body = _function_body(source, "_trace_begin_move")

    assert "TRACE_DEFAULT_GRID_HALF_X := 7" in source
    assert "TRACE_DEFAULT_GRID_HALF_Z := 7" in source
    assert "TRACE_DEFAULT_GRID_MAX_Y := 5" in source
    assert 'config.get("trace_grid_half_x"' in config_body
    assert 'config.get("trace_move_duration_s", 1.45)' in config_body
    assert 'config.get("trace_pause_duration_s", 0.95)' in config_body
    assert 'config.get("trace_panel_visual_scale", TRACE_PANEL_VISUAL_SCALE)' in config_body
    assert 'config.get("trace1_response_window_s"' in config_body
    assert "trace_move_duration_s" in init_body
    assert "trace_pause_duration_s" in init_body
    assert "trace_grid_half_x" in bounds_body
    assert "trace_grid_half_z" in bounds_body
    assert "trace_grid_max_y" in bounds_body
    assert "absf(dot) > 0.05" in turn_body
    assert "_trace_forward_blocked(cell, dir_idx)" in begin_body
    assert 'track["to_cell"] = cell + _trace_dir(dir_idx)' in begin_body


def test_godot_owned_phase_screens_are_shared_and_visual_only() -> None:
    main_source = _source()
    runtime_source = _runtime_source()
    auditory_source = (
        Path(__file__).resolve().parents[1]
        / "godot"
        / "cfast_3d"
        / "scripts"
        / "auditory_runtime.gd"
    ).read_text(encoding="utf-8")
    rapid_source = (
        Path(__file__).resolve().parents[1]
        / "godot"
        / "cfast_3d"
        / "scripts"
        / "rapid_tracking_runtime.gd"
    ).read_text(encoding="utf-8")
    spatial_source = (
        Path(__file__).resolve().parents[1]
        / "godot"
        / "cfast_3d"
        / "scripts"
        / "spatial_integration_runtime.gd"
    ).read_text(encoding="utf-8")

    assert "phase_screen_active" in main_source
    assert "_show_godot_owned_phase_screen" in main_source
    assert "_godot_owned_phase_screen_needed" in main_source
    assert "godot_phase_advance" in main_source
    assert "godot_phase_complete" in runtime_source
    assert "godot_phase_complete" in auditory_source
    assert "godot_phase_complete" in rapid_source
    assert "godot_phase_complete" in spatial_source
    assert "instructions\" or token == \"practice_done\" or token == \"results" in main_source

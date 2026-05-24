from __future__ import annotations

from pathlib import Path


RUNTIME_PATH = (
    Path(__file__).resolve().parents[1]
    / "godot"
    / "cfast_3d"
    / "scripts"
    / "auditory_runtime.gd"
)


def _source() -> str:
    return RUNTIME_PATH.read_text(encoding="utf-8")


def _function_body(source: str, name: str) -> str:
    marker = f"func {name}"
    start = source.index(marker)
    next_func = source.find("\nfunc ", start + len(marker))
    return source[start:] if next_func < 0 else source[start:next_func]


def test_auditory_trigger_beep_is_not_spoken_by_narrator() -> None:
    source = _source()
    body = _function_body(source, "_spawn_beep")

    assert "_play_beep()" in body
    assert "_speak" not in body
    assert "Press trigger now" not in source


def test_auditory_runtime_has_thin_complete_outline_gates_without_heavy_edge_boundary_nodes() -> None:
    source = _source()
    rebuild_body = _function_body(source, "_rebuild_gates")
    score_body = _function_body(source, "_score_gate")
    circle_body = _function_body(source, "_draw_circle_gate")
    poly_body = _function_body(source, "_draw_outline_poly_gate")
    add_mesh_body = _function_body(source, "_add_gate_mesh")
    gate_material_body = _function_body(source, "_gate_material")

    assert "GATE_STROKE_WIDTH := 0.085" in source
    assert "AuditoryOutlinePolyGate" in source
    assert "AuditoryOutlineCircleGate" in source
    assert "_draw_outline_poly_gate(points, color, GATE_STROKE_WIDTH)" in rebuild_body
    assert "_draw_outline_poly_gate(points2, color, GATE_STROKE_WIDTH)" in rebuild_body
    assert "_add_gate_mesh(\"AuditoryOutlineCircleGate\", vertices, indices, color)" in circle_body
    assert "_add_gate_mesh(\"AuditoryOutlinePolyGate\", vertices, indices, color)" in poly_body
    assert "normal.cross(edge.normalized()).normalized() * (width * 0.5)" in poly_body
    assert "inner_radius := maxf(0.03, radius - GATE_STROKE_WIDTH)" in circle_body
    assert "node.material_override = _gate_material(color)" in add_mesh_body
    assert "mat.cull_mode = BaseMaterial3D.CULL_DISABLED" in gate_material_body
    assert "mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED" in gate_material_body
    assert "_draw_filled_poly_gate" not in source
    assert "AuditoryFilledGate" not in source
    assert "GATE_APERTURE_COLOR" not in source
    assert "AuditoryGateAperture" not in source
    assert "AuditoryGateCenterline" not in source
    assert "_draw_gate_aperture" not in source
    assert "AuditorySafeRail" not in source
    assert "AuditorySafeRing" not in source
    assert "AuditoryRail" not in source
    assert "AuditoryRing" not in source
    assert "SAFE_BOUNDARY_COLOR" not in source
    assert "flash_color" not in rebuild_body
    assert "flash_color" not in score_body
    assert "_flash_ball(PASS_COLOR" in score_body
    assert "_flash_ball(ERROR_COLOR" in score_body


def test_auditory_runtime_uses_larger_default_tube_dimensions() -> None:
    source = _source()

    assert "var tube_half_width := 0.85" in source
    assert "var tube_half_height := 0.61" in source
    assert "var inner_rx := 2.86" in source
    assert "var inner_rz := 2.10" in source
    assert 'cfg.get("tube_half_width", 0.85)' in source
    assert 'cfg.get("tube_half_height", 0.61)' in source
    assert 'cfg.get("inner_rx", 2.86)' in source
    assert 'cfg.get("inner_rz", 2.10)' in source


def test_auditory_runtime_uses_pooled_physics_tube_and_ball() -> None:
    source = _source()

    assert "AuditoryTubeChunkBody" in source
    assert "StaticBody3D.new()" in source
    assert "CollisionShape3D.new()" in source
    assert "ConcavePolygonShape3D.new()" in source
    assert "shape.set_faces" in source
    assert "AuditoryBallBody" in source
    assert "CharacterBody3D.new()" in source
    assert "move_and_collide" in source


def test_auditory_runtime_reads_difficulty_scaled_segment_settings() -> None:
    source = _source()

    assert "difficulty_scaled" in source
    assert "_normalize_segments" in source
    assert "_apply_segment" in source
    assert "command_interval_s" in source
    assert "directive_interval_s" in source
    assert "sequence_interval_s" in source
    assert "beep_frequency_hz" in source


def test_auditory_runtime_layers_voices_and_speaks_colour_words() -> None:
    source = _source()
    prepare_body = _function_body(source, "_prepare_tts")
    speak_body = _function_body(source, "_speak_role")
    distractor_body = _function_body(source, "_spawn_distractor")
    command_body = _function_body(source, "_activate_command")
    speech_color_body = _function_body(source, "_speech_color_name")

    assert "primary_voice_id" in source
    assert "filler_voice_id" in source
    assert "decoy_voice_id" in source
    assert "english_voice_ids := _english_voice_ids(voices)" in prepare_body
    assert "_voice_id_from_ids(english_voice_ids, 0)" in prepare_body
    assert "_voice_id_from_ids(english_voice_ids, 1)" in prepare_body
    assert "_voice_id_from_ids(english_voice_ids, 2)" in prepare_body
    assert "func _voice_is_english" in source
    assert 'language.begins_with("en")' in source
    assert "_next_tts_utterance_id()" in speak_body
    assert 'role == "filler"' in speak_body
    assert 'role == "decoy"' in speak_body
    assert "filler_voice_volume" in speak_body
    assert "distractor_voice_volume" in speak_body
    assert "difficulty >= secondary_voice_min_difficulty" in distractor_body
    assert "_spawn_filler_narration()" in distractor_body
    assert "_speech_color_name(str(payload))" in command_body
    assert '_speech_color_name(str(rng_instructions.choice(COLORS)))' in distractor_body
    assert 'return "red"' in speech_color_body
    assert 'return "blue"' in speech_color_body
    assert 'return "yellow"' in speech_color_body


def test_auditory_runtime_has_review_hints_and_persistent_ball_colour_feedback() -> None:
    source = _source()
    hud_body = _function_body(source, "_update_hud")
    refresh_body = _function_body(source, "_refresh_ball_material")
    active_color_body = _function_body(source, "_active_ball_color")
    submit_body = _function_body(source, "_submit_color")

    assert "review_mode_enabled = bool(cfg.get(\"review_mode_enabled\", false))" in source
    assert "Ball: " in hud_body
    assert "Dev Review: " in hud_body
    assert "_review_hint()" in hud_body
    assert "ball_color_label := \"neutral\"" in source
    assert "ball_color_label = _speech_color_name(color)" in submit_body
    assert "_active_ball_color()" in refresh_body
    assert "elapsed_s < ball_feedback_until_s" in active_color_body
    assert "func _flash_ball" in source


def test_auditory_runtime_louder_distractions_and_deterministic_filler_script() -> None:
    source = _source()
    audio_body = _function_body(source, "_prepare_audio")
    schedule_body = _function_body(source, "_schedule_current_segment")
    update_body = _function_body(source, "_update_schedules")
    filler_body = _function_body(source, "_spawn_filler_narration")

    assert "FILLER_NARRATION_LINES" in source
    assert "var ambient_volume_db := -14.0" in source
    assert "var filler_narrator_interval_s := 13.0" in source
    assert 'cfg.get("ambient_volume_db", -14.0)' in source
    assert 'cfg.get("filler_narrator_interval_s", 13.0)' in source
    assert "player.volume_db = ambient_volume_db + float(i) * ambient_layer_drop_db" in audio_body
    assert "next_filler_at_s = elapsed_s + start_offset" in schedule_body
    assert "elapsed_s >= next_filler_at_s" in update_body
    assert "rng_audio.choice(FILLER_NARRATION_LINES)" in filler_body
    assert '_speak_role(line, "filler", false)' in filler_body


def test_auditory_runtime_spaces_gates_and_keeps_them_until_offscreen() -> None:
    source = _source()
    spawn_body = _function_body(source, "_spawn_gate")
    update_body = _function_body(source, "_update_gates")
    rebuild_body = _function_body(source, "_rebuild_gates")

    assert "GATE_INTERVAL_VISUAL_SCALE := 1.35" in source
    assert "GATE_SPAWN_AHEAD_DISTANCE := 22.0" in source
    assert "GATE_VISIBLE_AHEAD_DISTANCE := 24.0" in source
    assert "GATE_VISIBLE_BEHIND_DISTANCE := 12.0" in source
    assert "GATE_RETIRE_BEHIND_DISTANCE := 13.5" in source
    assert "* GATE_INTERVAL_VISUAL_SCALE" in source
    assert '"distance": travel_distance + GATE_SPAWN_AHEAD_DISTANCE' in spawn_body
    assert "travel_distance - GATE_RETIRE_BEHIND_DISTANCE" in update_body
    assert "travel_distance - GATE_VISIBLE_BEHIND_DISTANCE" in rebuild_body
    assert "travel_distance + GATE_VISIBLE_AHEAD_DISTANCE" in rebuild_body


def test_auditory_runtime_uses_flipped_pitch_arrows_and_joystick_for_ball_motion() -> None:
    source = _source()
    body = _function_body(source, "_update_ball")

    assert "Input.is_key_pressed(KEY_LEFT)" in body
    assert "Input.is_key_pressed(KEY_RIGHT)" in body
    assert "Input.is_key_pressed(KEY_UP)" in body
    assert "if Input.is_key_pressed(KEY_UP):\n\t\tinput_vec.y -= 1.0" in body
    assert "Input.is_key_pressed(KEY_DOWN)" in body
    assert "if Input.is_key_pressed(KEY_DOWN):\n\t\tinput_vec.y += 1.0" in body
    assert "for raw_joy in Input.get_connected_joypads():" in body
    assert "_joy_axis_with_deadzone(joy, JOY_AXIS_LEFT_X)" in body
    assert "_joy_axis_with_deadzone(joy, JOY_AXIS_LEFT_Y)" in body
    assert "input_vec.y += _joy_axis_with_deadzone(joy, JOY_AXIS_LEFT_Y)" in body
    assert "input_vec.y -= _joy_axis_with_deadzone(joy, JOY_AXIS_LEFT_Y)" not in body
    assert "JOYSTICK_DEADZONE" in source
    assert "Input.get_joy_axis" in _function_body(source, "_joy_axis_with_deadzone")
    assert "Input.is_key_pressed(KEY_A)" not in body
    assert "Input.is_key_pressed(KEY_D)" not in body
    assert "Input.is_key_pressed(KEY_W)" not in body
    assert "Input.is_key_pressed(KEY_S)" not in body


def test_auditory_runtime_maps_qwer_to_visible_ball_colours() -> None:
    source = _source()
    handle_body = _function_body(source, "handle_key")
    submit_body = _function_body(source, "_submit_color")
    update_body = _function_body(source, "_update_ball_node")

    assert "ball_display_color := BALL_IDLE_COLOR" in source
    assert "if key == KEY_Q:" in handle_body
    assert '_submit_color("BLUE")' in handle_body
    assert "if key == KEY_W:" in handle_body
    assert '_submit_color("YELLOW")' in handle_body
    assert "if key == KEY_E:" in handle_body
    assert "if key == KEY_R:" in handle_body
    assert handle_body.count('_submit_color("RED")') == 2
    assert "ball_display_color = _color_by_name(color)" in submit_body
    assert "_refresh_ball_material()" in submit_body
    assert "_refresh_ball_material()" in update_body

extends Node3D

const WHITE_COLOR := Color(0.92, 0.96, 0.98, 1.0)
const GREEN_COLOR := Color(0.22, 0.76, 0.34, 1.0)
const BLUE_COLOR := Color(0.12, 0.42, 0.95, 1.0)
const RED_COLOR := Color(0.95, 0.18, 0.15, 1.0)
const AMBER_COLOR := Color(0.95, 0.66, 0.16, 1.0)
const GRID_COLOR := Color(0.45, 0.58, 0.66, 1.0)
const FLOOR_COLOR := Color(0.17, 0.23, 0.25, 1.0)
const SKY_COLOR := Color(0.36, 0.55, 0.78, 1.0)
const TRACE_DEFAULT_GRID_HALF_X := 7
const TRACE_DEFAULT_GRID_HALF_Z := 7
const TRACE_DEFAULT_GRID_MAX_Y := 5
const TRACE_DEFAULT_GRID_SCALE := 0.64
const TRACE_DEFAULT_GRID_ALT_SCALE := 0.60
const TRACE_GRID_ORIGIN := Vector3(0.0, 0.72, -6.2)
const TRACE_GUIDE_PANEL_COLOR := Color(0.02, 0.05, 0.42, 1.0)
const TRACE_FIXED_ORTHO_CAMERA_OFFSET := Vector3(0.0, 0.0, 18.0)
const TRACE_FIXED_ORTHO_CAMERA_TARGET := TRACE_GRID_ORIGIN + Vector3(0.0, 1.55, 0.0)
const TRACE_FIXED_ORTHO_CAMERA_MARGIN := 1.18
const TRACE_PANEL_VISUAL_SCALE := 2.0
const TRACE1_RESPONSE_LEFT_YAW := "left_yaw"
const TRACE1_RESPONSE_RIGHT_YAW := "right_yaw"
const TRACE1_RESPONSE_PUSH_PITCH_DOWN := "push_pitch_down"
const TRACE1_RESPONSE_PULL_PITCH_UP := "pull_pitch_up"
const TRACE1_RESPONSE_CYCLE := [
	TRACE1_RESPONSE_LEFT_YAW,
	TRACE1_RESPONSE_RIGHT_YAW,
	TRACE1_RESPONSE_PUSH_PITCH_DOWN,
	TRACE1_RESPONSE_PULL_PITCH_UP,
]
const TRACE2_QUESTION_STARTED_TOP := "started_top"
const TRACE2_QUESTION_STARTED_BOTTOM := "started_bottom"
const TRACE2_QUESTION_TURNED_LEFT := "turned_left"
const TRACE2_QUESTION_TURNED_RIGHT := "turned_right"
const TRACE2_QUESTION_ENDED_HIGHEST := "ended_highest"
const TRACE2_QUESTION_NO_TURN := "no_direction_change"
const TRACE2_QUESTION_ENDED_OFF_SCREEN := "ended_off_screen"
const TRACE2_QUESTION_LEFT_SCREEN := "left_screen"
const TRACE2_QUESTION_ENDED_OFF_RIGHT := "ended_off_right"
const TRACE2_QUESTION_CYCLE := [
	TRACE2_QUESTION_STARTED_TOP,
	TRACE2_QUESTION_TURNED_LEFT,
	TRACE2_QUESTION_ENDED_OFF_SCREEN,
	TRACE2_QUESTION_TURNED_RIGHT,
	TRACE2_QUESTION_ENDED_HIGHEST,
	TRACE2_QUESTION_LEFT_SCREEN,
	TRACE2_QUESTION_NO_TURN,
	TRACE2_QUESTION_STARTED_BOTTOM,
	TRACE2_QUESTION_ENDED_OFF_RIGHT,
]
const TRACE_DIRS := [
	Vector3(0.0, 0.0, -1.0),
	Vector3(1.0, 0.0, 0.0),
	Vector3(0.0, 0.0, 1.0),
	Vector3(-1.0, 0.0, 0.0),
	Vector3(0.0, 1.0, 0.0),
	Vector3(0.0, -1.0, 0.0),
]

var control_sender: Callable
var active := false
var paused := false
var run_key := ""
var completed_run_key := ""
var kind := ""
var test_code := ""
var phase := ""
var session_seed := 1
var difficulty := 0.5
var duration_s := 60.0
var elapsed_s := 0.0
var tick_accum := 0.0
var progress_accum := 0.0
var render_accum := 0.0
var fixed_hz := 60.0
var rng := RandomNumberGenerator.new()
var player_pos := Vector2.ZERO
var player_vel := Vector2.ZERO
var target_pos := Vector2.ZERO
var sample_accum := 0.0
var attempted := 0
var correct := 0
var total_score := 0.0
var max_score := 0.0
var trace_tracks: Array = []
var trace_turn_count := 0
var trace_forced_edge_turn_count := 0
var trace_state_hash := 0
var trace_grid_half_x := TRACE_DEFAULT_GRID_HALF_X
var trace_grid_half_z := TRACE_DEFAULT_GRID_HALF_Z
var trace_grid_max_y := TRACE_DEFAULT_GRID_MAX_Y
var trace_grid_scale := TRACE_DEFAULT_GRID_SCALE
var trace_grid_alt_scale := TRACE_DEFAULT_GRID_ALT_SCALE
var trace_move_duration_s := 1.45
var trace_pause_duration_s := 0.95
var trace_camera_margin := TRACE_FIXED_ORTHO_CAMERA_MARGIN
var trace_panel_margin := 1.25
var trace_panel_visual_scale := TRACE_PANEL_VISUAL_SCALE
var trace1_response_window_s := 3.0
var trace1_prompt_index := 0
var trace1_expected_response := ""
var trace1_prompt_started_at_s := 0.0
var trace1_response_deadline_s := INF
var trace1_prompt_hash := 0
var trace1_prompt_context := {}
var trace1_joystick_ready := true
var trace2_stage := "observe"
var trace2_observe_s := 5.0
var trace2_clip_min_s := 3.5
var trace2_clip_max_s := 7.0
var trace2_question_window_s := 7.0
var trace2_offscreen_margin_cells := 2.25
var trace2_close_camera_scale := 0.66
var trace2_aircraft_scale := 1.42
var trace2_question_overlay_enabled := true
var trace2_trial_started_at_s := 0.0
var trace2_question_started_at_s := 0.0
var trace2_question_deadline_s := INF
var trace2_trial_index := 0
var trace2_question_kind := ""
var trace2_correct_option := 0
var trace2_options: Array = []
var trace2_trial_hash := 0
var event_log: Array = []
var scene_root: Node3D
var hud_layer: CanvasLayer
var hud_label: Label
var trace2_overlay_layer: CanvasLayer
var trace2_overlay_root: Control
var trace2_overlay_title: Label
var trace2_overlay_question: Label
var trace2_overlay_options: Label
var trace2_overlay_footer: Label


func start(spec: Dictionary, sender: Callable) -> void:
	control_sender = sender
	var next_key := str(spec.get("run_key", ""))
	if active and next_key == run_key:
		return
	if next_key != "" and next_key == completed_run_key:
		return
	_clear_runtime()
	run_key = next_key
	kind = str(spec.get("kind", "rapid_tracking"))
	test_code = str(spec.get("test_code", kind))
	phase = str(spec.get("phase", "scored")).to_lower()
	session_seed = int(max(1, int(spec.get("session_seed", spec.get("seed", 1)))))
	difficulty = clampf(_float(spec.get("difficulty", 0.5)), 0.0, 1.0)
	duration_s = maxf(1.0, _float(spec.get("duration_s", 60.0)))
	elapsed_s = 0.0
	tick_accum = 0.0
	progress_accum = 0.0
	render_accum = 0.0
	sample_accum = 0.0
	rng.seed = session_seed + _kind_salt(kind)
	player_pos = Vector2.ZERO
	player_vel = Vector2.ZERO
	attempted = 0
	correct = 0
	total_score = 0.0
	max_score = 0.0
	trace_tracks.clear()
	trace_turn_count = 0
	trace_forced_edge_turn_count = 0
	trace_state_hash = 0
	trace1_prompt_index = 0
	trace1_expected_response = ""
	trace1_prompt_started_at_s = 0.0
	trace1_response_deadline_s = INF
	trace1_prompt_hash = 0
	trace1_prompt_context = {}
	trace1_joystick_ready = true
	trace2_stage = "observe"
	trace2_trial_started_at_s = 0.0
	trace2_question_started_at_s = 0.0
	trace2_question_deadline_s = INF
	trace2_trial_index = 0
	trace2_question_kind = ""
	trace2_correct_option = 0
	trace2_options = []
	trace2_trial_hash = 0
	if _is_trace_kind():
		_configure_trace_runtime(_as_dict(spec.get("config", {})))
		if kind == "trace_test_2":
			_trace2_start_trial()
		else:
			_init_trace_tracks()
			_sync_trace_target()
	else:
		target_pos = _target_for_time(0.0)
	event_log.clear()
	_build_nodes()
	active = true
	_send("godot_ready", {"run_key": run_key, "phase": phase, "kind": kind, "test_code": test_code})
	_rebuild_scene(true)


func update_runtime(delta: float, camera: Camera3D) -> void:
	if not active or paused:
		return
	var dt := minf(maxf(delta, 0.0), 0.05)
	tick_accum += dt
	var fixed_dt := 1.0 / fixed_hz
	while tick_accum >= fixed_dt:
		tick_accum -= fixed_dt
		_step(fixed_dt)
	render_accum += dt
	if render_accum >= 1.0 / 30.0:
		render_accum = 0.0
		_rebuild_scene(false)
	_update_camera(camera)
	_update_hud()
	progress_accum += dt
	if progress_accum >= 0.25:
		progress_accum = 0.0
		_send_progress()


func handle_key(event: InputEventKey) -> bool:
	if not active or paused or event.echo or not event.pressed:
		return false
	var key := event.keycode
	if key == KEY_KP_ENTER:
		return true
	if kind == "trace_test_1":
		if key == KEY_LEFT:
			_trace1_submit_response(TRACE1_RESPONSE_LEFT_YAW, false)
			return true
		if key == KEY_RIGHT:
			_trace1_submit_response(TRACE1_RESPONSE_RIGHT_YAW, false)
			return true
		if key == KEY_UP:
			_trace1_submit_response(TRACE1_RESPONSE_PUSH_PITCH_DOWN, false)
			return true
		if key == KEY_DOWN:
			_trace1_submit_response(TRACE1_RESPONSE_PULL_PITCH_UP, false)
			return true
		if key == KEY_SPACE or key == KEY_ENTER or key == KEY_KP_PERIOD:
			return true
	if kind == "trace_test_2":
		var option := _trace2_option_for_key(key)
		if option > 0:
			_trace2_submit_answer(option, false)
			return true
		if key == KEY_SPACE or key == KEY_ENTER or key == KEY_KP_PERIOD:
			return true
	if key == KEY_SPACE or key == KEY_ENTER or key == KEY_KP_PERIOD:
		_sample_score("submit")
		return true
	return false


func set_paused(value: bool) -> void:
	paused = bool(value)


func _step(dt: float) -> void:
	elapsed_s += dt
	if kind == "trace_test_1":
		_step_trace_tracks(dt)
		_trace1_poll_joystick_response()
		_trace1_update_timeout()
		if elapsed_s >= duration_s:
			_complete()
		return
	if kind == "trace_test_2":
		_trace2_step(dt)
		if elapsed_s >= duration_s:
			_complete()
		return
	if _is_trace_kind():
		_step_trace_tracks(dt)
		_sync_trace_target()
	else:
		target_pos = _target_for_time(elapsed_s)
	var input_vec := _input_vector()
	var gain := 2.2 + difficulty * 1.4
	player_vel += input_vec * gain * dt
	player_vel *= pow(0.12, dt)
	player_pos += player_vel * dt
	player_pos.x = clampf(player_pos.x, -1.0, 1.0)
	player_pos.y = clampf(player_pos.y, -1.0, 1.0)
	sample_accum += dt
	if sample_accum >= 0.75:
		sample_accum = 0.0
		_sample_score("sample")
	if elapsed_s >= duration_s:
		_complete()


func _input_vector() -> Vector2:
	var out := Vector2.ZERO
	if Input.is_key_pressed(KEY_A) or Input.is_key_pressed(KEY_LEFT):
		out.x -= 1.0
	if Input.is_key_pressed(KEY_D) or Input.is_key_pressed(KEY_RIGHT):
		out.x += 1.0
	if Input.is_key_pressed(KEY_W) or Input.is_key_pressed(KEY_UP):
		out.y += 1.0
	if Input.is_key_pressed(KEY_S) or Input.is_key_pressed(KEY_DOWN):
		out.y -= 1.0
	if Input.get_connected_joypads().size() > 0:
		var joy := int(Input.get_connected_joypads()[0])
		out.x += Input.get_joy_axis(joy, JOY_AXIS_LEFT_X)
		out.y -= Input.get_joy_axis(joy, JOY_AXIS_LEFT_Y)
	if out.length() > 1.0:
		out = out.normalized()
	return out


func _target_for_time(t: float) -> Vector2:
	var salt := float(_kind_salt(kind) % 997) / 997.0
	var speed := 0.55 + difficulty * 0.45
	if kind == "spatial_integration":
		return Vector2(sin(t * speed * 0.8 + salt * TAU) * 0.72, cos(t * speed * 0.55 + salt) * 0.64)
	if kind == "trace_test_1":
		return Vector2(sin(t * speed * 1.15 + salt) * 0.82, sin(t * speed * 0.67 + 1.6) * 0.72)
	if kind == "trace_test_2":
		return Vector2(cos(t * speed * 0.7 + salt) * 0.78, sin(t * speed * 1.05 + 0.8) * 0.72)
	return Vector2(sin(t * speed + salt * TAU) * 0.76, cos(t * speed * 0.83 + salt) * 0.66)


func _sample_score(trigger: String) -> void:
	var dist := player_pos.distance_to(target_pos)
	var threshold := 0.30 + (0.10 * (1.0 - difficulty))
	var hit := dist <= threshold
	attempted += 1
	if hit:
		correct += 1
	var score := maxf(0.0, 1.0 - (dist / maxf(0.001, threshold * 2.5)))
	total_score += score
	max_score += 1.0
	var extra := {"distance": dist, "threshold": threshold}
	if _is_trace_kind() and not trace_tracks.is_empty():
		var target_track := _as_dict(trace_tracks[0])
		extra["target_cell"] = _trace_cell_token(_trace_vec3(target_track.get("current_cell", Vector3.ZERO)))
		extra["target_heading"] = _trace_dir_name(int(target_track.get("dir_idx", 0)))
		extra["target_phase"] = str(target_track.get("phase", "move"))
	var evt := {
		"family": kind,
		"kind": trigger,
		"phase": phase,
		"item_index": attempted - 1,
		"is_scored": phase == "scored",
		"is_correct": hit,
		"is_timeout": false,
		"response_time_ms": int(round(elapsed_s * 1000.0)),
		"score": score,
		"max_score": 1.0,
		"occurred_at_ms": int(round(elapsed_s * 1000.0)),
		"prompt": kind,
		"expected": "target",
		"response": "player",
		"extra": extra,
	}
	event_log.append(evt)
	_send("godot_event", {"run_key": run_key, "kind": kind, "test_code": test_code, "event": evt})


func _complete() -> void:
	active = false
	completed_run_key = run_key
	if phase == "practice":
		_send("godot_phase_complete", {"run_key": run_key, "phase": "practice", "kind": kind, "test_code": test_code})
		return
	var accuracy := 0.0 if attempted <= 0 else float(correct) / float(attempted)
	var summary := {
		"attempted": attempted,
		"correct": correct,
		"accuracy": accuracy,
		"duration_s": elapsed_s,
		"throughput_per_min": float(attempted) / maxf(1.0, elapsed_s) * 60.0,
		"total_score": total_score,
		"max_score": max_score,
		"score_ratio": 0.0 if max_score <= 0.0 else total_score / max_score,
	}
	var metrics := {
		"godot_kind": kind,
		"godot_test_code": test_code,
		"mean_tracking_error": _mean_event_distance(),
		"renderer_backend": "godot_4",
		"trace_turn_count": trace_turn_count,
		"trace_forced_edge_turn_count": trace_forced_edge_turn_count,
		"trace_state_hash": trace_state_hash,
		"trace1_prompt_count": trace1_prompt_index,
		"trace1_prompt_hash": trace1_prompt_hash,
		"trace1_response_window_s": trace1_response_window_s,
		"trace2_trial_count": trace2_trial_index,
		"trace2_trial_hash": trace2_trial_hash,
		"trace2_stage": trace2_stage,
	}
	var result := {
		"run_key": run_key,
		"kind": kind,
		"test_code": test_code,
		"phase": "results",
		"summary": summary,
		"metrics": metrics,
		"events": event_log.slice(max(0, event_log.size() - 240), event_log.size()),
	}
	_send("godot_complete", {"run_key": run_key, "phase": "results", "kind": kind, "test_code": test_code, "result": result})


func _rebuild_scene(force: bool) -> void:
	if scene_root == null:
		return
	for child in scene_root.get_children():
		child.queue_free()
	if not _is_trace_kind():
		_make_floor()
	if kind == "spatial_integration":
		_draw_spatial_scene()
	elif kind == "trace_test_1":
		_draw_trace_scene(false)
	elif kind == "trace_test_2":
		_draw_trace_scene(true)
	else:
		_draw_rapid_tracking_scene()


func _draw_rapid_tracking_scene() -> void:
	_draw_grid(10, 9, 0.65)
	for i in range(8):
		var x := rng.randf_range(-4.8, 4.8)
		var z := rng.randf_range(-9.2, -3.5)
		_make_box(Vector3(x, rng.randf_range(0.08, 0.32), z), Vector3(rng.randf_range(0.28, 0.9), rng.randf_range(0.16, 0.64), rng.randf_range(0.28, 1.0)), Color(0.25, 0.34, 0.23, 1.0))
	_make_sphere(_screen_to_world(target_pos, -5.6), 0.34, GREEN_COLOR)
	_make_reticle(_screen_to_world(player_pos, -5.2), BLUE_COLOR)


func _draw_spatial_scene() -> void:
	_draw_grid(9, 9, 0.72)
	for i in range(7):
		var p := _seeded_point(i)
		_make_box(Vector3(p.x, 0.24, p.y - 5.4), Vector3(0.28, 0.48, 0.28), AMBER_COLOR if i % 2 == 0 else BLUE_COLOR)
	_make_sphere(_screen_to_world(target_pos, -5.6), 0.24, GREEN_COLOR)
	_make_aircraft(_screen_to_world(player_pos, -5.2), WHITE_COLOR)


func _draw_trace_scene(second: bool) -> void:
	var movement_w := float(trace_grid_half_x) * trace_grid_scale * 2.0
	var movement_h := float(trace_grid_max_y) * trace_grid_alt_scale
	var panel_w := (movement_w + trace_panel_margin) * trace_panel_visual_scale
	var panel_h := (movement_h + trace_panel_margin) * trace_panel_visual_scale
	var panel_center := TRACE_GRID_ORIGIN + Vector3(0.0, movement_h * 0.50, -float(trace_grid_half_z) * trace_grid_scale - 0.08)
	if kind == "trace_test_1" or kind == "trace_test_2":
		var fill := maxf(panel_w, panel_h) * 2.0
		_make_box(panel_center, Vector3(fill, fill, 0.04), SKY_COLOR)
	else:
		_make_box(panel_center + Vector3(0.0, 0.0, -0.10), Vector3(panel_w * 0.56, panel_h * 0.58, 0.055), TRACE_GUIDE_PANEL_COLOR)
		_make_box(panel_center, Vector3(panel_w * 0.50, panel_h * 0.50, 0.04), SKY_COLOR)
	if kind == "trace_test_2" and trace2_stage == "question":
		return
	var index := 0
	for raw_track in trace_tracks:
		var track := _as_dict(raw_track)
		var color := RED_COLOR if index == 0 else _trace_track_color(index)
		if kind == "trace_test_2" and track.has("color_value"):
			color = track["color_value"]
		var size := 1.08 if index == 0 else 0.84
		if kind == "trace_test_2":
			size *= trace2_aircraft_scale
		_make_aircraft(_trace_track_world(track), color, _trace_track_direction(track), size)
		index += 1


func _is_trace_kind() -> bool:
	return kind == "trace_test_1" or kind == "trace_test_2"


func _configure_trace_runtime(config: Dictionary) -> void:
	trace_grid_half_x = max(4, int(config.get("trace_grid_half_x", TRACE_DEFAULT_GRID_HALF_X)))
	trace_grid_half_z = max(4, int(config.get("trace_grid_half_z", TRACE_DEFAULT_GRID_HALF_Z)))
	trace_grid_max_y = max(3, int(config.get("trace_grid_max_y", TRACE_DEFAULT_GRID_MAX_Y)))
	trace_grid_scale = maxf(0.30, _float(config.get("trace_grid_scale", TRACE_DEFAULT_GRID_SCALE)))
	trace_grid_alt_scale = maxf(0.30, _float(config.get("trace_grid_alt_scale", TRACE_DEFAULT_GRID_ALT_SCALE)))
	trace_move_duration_s = maxf(0.75, _float(config.get("trace_move_duration_s", 1.45)))
	trace_pause_duration_s = maxf(0.55, _float(config.get("trace_pause_duration_s", 0.95)))
	trace_camera_margin = maxf(1.0, _float(config.get("trace_camera_margin", TRACE_FIXED_ORTHO_CAMERA_MARGIN)))
	trace_panel_margin = maxf(0.6, _float(config.get("trace_panel_margin", 1.25)))
	trace_panel_visual_scale = maxf(1.0, _float(config.get("trace_panel_visual_scale", TRACE_PANEL_VISUAL_SCALE)))
	trace1_response_window_s = maxf(1.0, _float(config.get("trace1_response_window_s", 3.0 - difficulty * 0.55)))
	trace2_observe_s = maxf(2.0, _float(config.get("trace2_observe_s", 5.0 - difficulty * 0.55)))
	trace2_clip_min_s = maxf(2.0, _float(config.get("trace2_clip_min_s", 3.5)))
	trace2_clip_max_s = maxf(trace2_clip_min_s, _float(config.get("trace2_clip_max_s", 7.0)))
	trace2_question_window_s = maxf(2.0, _float(config.get("trace2_question_window_s", 7.0 - difficulty * 1.2)))
	trace2_offscreen_margin_cells = maxf(0.5, _float(config.get("trace2_offscreen_margin_cells", 2.25)))
	trace2_close_camera_scale = clampf(_float(config.get("trace2_close_camera_scale", 0.66)), 0.35, 1.0)
	trace2_aircraft_scale = maxf(0.75, _float(config.get("trace2_aircraft_scale", 1.42)))
	trace2_question_overlay_enabled = bool(config.get("trace2_question_overlay_enabled", true))


func _init_trace_tracks() -> void:
	trace_tracks.clear()
	var count := 5 if kind == "trace_test_2" else 3
	if test_code.find("pressure") >= 0:
		count += 1
	for i in range(count):
		var local := RandomNumberGenerator.new()
		local.seed = int(session_seed + _kind_salt(test_code) + (i + 1) * 7919)
		var cell := Vector3(
			float(local.randi_range(-trace_grid_half_x + 1, trace_grid_half_x - 1)),
			float(local.randi_range(0, trace_grid_max_y)),
			float(local.randi_range(-trace_grid_half_z + 1, trace_grid_half_z - 1))
		)
		var dir_idx := _trace_initial_dir(local, cell)
		if kind == "trace_test_1" and i == 0:
			dir_idx = _trace1_initial_dir(local, cell)
		var move_s := maxf(0.75, trace_move_duration_s + float(i % 3) * 0.07)
		var pause_s := maxf(0.55, trace_pause_duration_s + float(i % 2) * 0.06)
		var track := {
			"id": i,
			"current_cell": cell,
			"from_cell": cell,
			"to_cell": cell,
			"dir_idx": dir_idx,
			"prev_dir_idx": dir_idx,
			"phase": "pause",
			"phase_t": local.randf_range(0.0, pause_s * 0.75),
			"move_s": move_s,
			"pause_s": pause_s,
			"history": [cell],
			"forced_turns": 0,
			"last_horizontal_dir_idx": dir_idx if dir_idx < 4 else 0,
		}
		_trace_begin_move(track)
		trace_tracks.append(track)
		trace_state_hash = _trace_hash_mix(trace_state_hash, _trace_cell_hash(cell) + dir_idx * 31 + i)


func _step_trace_tracks(dt: float) -> void:
	for raw_track in trace_tracks:
		var track := _as_dict(raw_track)
		var phase_name := str(track.get("phase", "move"))
		var phase_t := _float(track.get("phase_t", 0.0)) + dt
		track["phase_t"] = phase_t
		if phase_name == "move":
			var move_s := maxf(0.05, _float(track.get("move_s", 0.72)))
			if phase_t >= move_s:
				var landed := _trace_vec3(track.get("to_cell", Vector3.ZERO))
				track["current_cell"] = landed
				_trace_append_history(track, landed)
				var old_dir_idx := int(track.get("dir_idx", 0))
				var forced := _trace_forward_blocked(landed, old_dir_idx)
				var next_dir_idx := _trace_choose_next_dir(track, forced)
				if kind == "trace_test_1" and int(track.get("id", -1)) == 0:
					next_dir_idx = _trace1_choose_next_prompt_dir(track, forced)
				elif kind == "trace_test_2":
					next_dir_idx = _trace2_choose_next_dir(track, forced)
				if next_dir_idx != old_dir_idx:
					trace_turn_count += 1
				if forced:
					trace_forced_edge_turn_count += 1
					track["forced_turns"] = int(track.get("forced_turns", 0)) + 1
				track["prev_dir_idx"] = old_dir_idx
				track["dir_idx"] = next_dir_idx
				track["phase"] = "pause"
				track["phase_t"] = 0.0
				if kind == "trace_test_1" and int(track.get("id", -1)) == 0:
					_trace1_open_response(old_dir_idx, next_dir_idx, landed, forced)
				trace_state_hash = _trace_hash_mix(trace_state_hash, _trace_cell_hash(landed) + next_dir_idx * 17 + (97 if forced else 0))
		else:
			var pause_s := maxf(0.05, _float(track.get("pause_s", 0.30)))
			if phase_t >= pause_s:
				_trace_begin_move(track)


func _trace_begin_move(track: Dictionary) -> void:
	var cell := _trace_vec3(track.get("current_cell", Vector3.ZERO))
	var dir_idx := int(track.get("dir_idx", 0))
	if _trace_forward_blocked(cell, dir_idx):
		var next_dir_idx := _trace_choose_next_dir(track, true)
		if next_dir_idx != dir_idx:
			trace_turn_count += 1
			trace_forced_edge_turn_count += 1
			track["prev_dir_idx"] = dir_idx
			track["dir_idx"] = next_dir_idx
			dir_idx = next_dir_idx
	track["from_cell"] = cell
	track["to_cell"] = cell + _trace_dir(dir_idx)
	track["phase"] = "move"
	track["phase_t"] = 0.0
	if dir_idx >= 0 and dir_idx < 4:
		track["last_horizontal_dir_idx"] = dir_idx


func _trace2_start_trial() -> void:
	trace_tracks.clear()
	_hide_trace2_question_overlay()
	trace2_stage = "observe"
	trace2_trial_started_at_s = elapsed_s
	trace2_question_started_at_s = 0.0
	trace2_question_deadline_s = INF
	trace2_correct_option = 0
	trace2_options = []
	var local := RandomNumberGenerator.new()
	local.seed = int(session_seed + _kind_salt(test_code) + trace2_trial_index * 3571)
	trace2_observe_s = local.randf_range(trace2_clip_min_s, trace2_clip_max_s)
	var colors := [
		{"code": 1, "label": "Red", "color": RED_COLOR},
		{"code": 2, "label": "Blue", "color": BLUE_COLOR},
		{"code": 3, "label": "Yellow", "color": AMBER_COLOR},
		{"code": 4, "label": "White", "color": WHITE_COLOR},
	]
	var roles := ["steady", "left", "right", "exit"]
	for i in range(roles.size() - 1, 0, -1):
		var j := int(local.randi_range(0, i))
		var tmp = roles[i]
		roles[i] = roles[j]
		roles[j] = tmp
	for idx in range(colors.size()):
		var info := _as_dict(colors[idx])
		var role := str(roles[idx])
		var cell := _trace2_start_cell_for_role(role)
		var dir_idx := _trace2_initial_dir_for_role(role)
		var script_dirs: Array = _trace2_script_dirs_for_role(role)
		var track := {
			"id": idx,
			"code": int(info.get("code", idx + 1)),
			"label": str(info.get("label", "")),
			"color_value": info.get("color", WHITE_COLOR),
			"role": role,
			"start_cell": cell,
			"current_cell": cell,
			"from_cell": cell,
			"to_cell": cell,
			"dir_idx": dir_idx,
			"prev_dir_idx": dir_idx,
			"phase": "pause",
			"phase_t": 0.0,
			"move_s": maxf(0.75, trace_move_duration_s * 0.82),
			"pause_s": maxf(0.35, trace_pause_duration_s * 0.45),
			"history": [cell],
			"script_dirs": script_dirs,
			"script_index": 0,
			"went_offscreen": false,
			"forced_turns": 0,
			"last_horizontal_dir_idx": dir_idx,
		}
		_trace_begin_move(track)
		trace_tracks.append(track)
		trace2_options.append({"code": int(info.get("code", idx + 1)), "label": str(info.get("label", ""))})
		trace_state_hash = _trace_hash_mix(trace_state_hash, _trace_cell_hash(cell) + _kind_salt(role) + idx * 47)
	_trace2_pick_question()


func _trace2_start_cell_for_role(role: String) -> Vector3:
	if role == "left":
		return Vector3(-1.0, 4.0, -4.0)
	if role == "right":
		return Vector3(1.0, 3.0, -3.0)
	if role == "exit":
		return Vector3(maxf(2.0, float(trace_grid_half_x) - trace2_offscreen_margin_cells - 1.0), 1.0, -2.0)
	return Vector3(-4.0, 2.0, -2.0)


func _trace2_initial_dir_for_role(role: String) -> int:
	if role == "exit":
		return 1
	return 0


func _trace2_script_dirs_for_role(role: String) -> Array:
	if role == "left":
		return [0, 3, 3, 3]
	if role == "right":
		return [0, 1, 1, 1]
	if role == "exit":
		return [1, 1, 1, 1]
	return [0, 0, 0, 0]


func _trace2_step(dt: float) -> void:
	if trace2_stage == "observe":
		_step_trace_tracks(dt)
		_trace2_update_offscreen_flags()
		if elapsed_s - trace2_trial_started_at_s >= trace2_observe_s:
			_trace2_open_question()
		return
	if trace2_stage == "question" and elapsed_s >= trace2_question_deadline_s:
		_trace2_submit_answer(0, true)


func _trace2_open_question() -> void:
	_trace2_update_offscreen_flags()
	trace2_stage = "question"
	trace2_question_started_at_s = elapsed_s
	trace2_question_deadline_s = elapsed_s + trace2_question_window_s
	trace2_correct_option = _trace2_answer_for_question(trace2_question_kind)
	trace2_trial_hash = _trace_hash_mix(trace2_trial_hash, _kind_salt(trace2_question_kind) + trace2_correct_option * 83 + trace2_trial_index * 19)
	_show_trace2_question_overlay()


func _trace2_choose_next_dir(track: Dictionary, forced_edge_turn: bool) -> int:
	var script_dirs: Array = track.get("script_dirs", [])
	var script_index := int(track.get("script_index", 0))
	while script_index < script_dirs.size():
		var candidate := int(script_dirs[script_index])
		script_index += 1
		track["script_index"] = script_index
		if _trace_cell_in_bounds(_trace_vec3(track.get("current_cell", Vector3.ZERO)) + _trace_dir(candidate)):
			return candidate
	if forced_edge_turn:
		return _trace_choose_next_dir(track, true)
	return int(track.get("dir_idx", 0))


func _trace2_pick_question() -> void:
	var index := int(posmod(session_seed + trace2_trial_index + _kind_salt(test_code), TRACE2_QUESTION_CYCLE.size()))
	trace2_question_kind = str(TRACE2_QUESTION_CYCLE[index])
	trace2_correct_option = 0
	trace2_trial_hash = _trace_hash_mix(trace2_trial_hash, _kind_salt(trace2_question_kind) + trace2_trial_index * 19)


func _trace2_answer_for_question(question_kind: String) -> int:
	if trace_tracks.is_empty():
		return 0
	if question_kind == TRACE2_QUESTION_STARTED_TOP:
		return int(_trace2_extreme_track("start_cell", "y", true).get("code", 0))
	if question_kind == TRACE2_QUESTION_STARTED_BOTTOM:
		return int(_trace2_extreme_track("start_cell", "y", false).get("code", 0))
	if question_kind == TRACE2_QUESTION_TURNED_LEFT:
		return _trace2_code_for_role("left")
	if question_kind == TRACE2_QUESTION_TURNED_RIGHT:
		return _trace2_code_for_role("right")
	if question_kind == TRACE2_QUESTION_ENDED_HIGHEST:
		return int(_trace2_extreme_track("current_cell", "y", true).get("code", 0))
	if question_kind == TRACE2_QUESTION_NO_TURN:
		return _trace2_code_for_role("steady")
	if question_kind == TRACE2_QUESTION_ENDED_OFF_SCREEN:
		return _trace2_unique_offscreen_code(true, "")
	if question_kind == TRACE2_QUESTION_LEFT_SCREEN:
		return _trace2_unique_offscreen_code(false, "")
	if question_kind == TRACE2_QUESTION_ENDED_OFF_RIGHT:
		return _trace2_unique_offscreen_code(true, "right")
	return 0


func _trace2_update_offscreen_flags() -> void:
	for raw_track in trace_tracks:
		var track := _as_dict(raw_track)
		if _trace2_world_offscreen(_trace_track_world(track)):
			track["went_offscreen"] = true


func _trace2_unique_offscreen_code(require_ended: bool, side: String) -> int:
	var matches: Array = []
	for raw_track in trace_tracks:
		var track := _as_dict(raw_track)
		var ended_side := _trace2_offscreen_side(_trace_track_world(track))
		var matches_side := side == "" or ended_side == side
		var visible_event := ended_side != "" if require_ended else bool(track.get("went_offscreen", false))
		if visible_event and matches_side:
			matches.append(int(track.get("code", 0)))
	if matches.size() == 1:
		return int(matches[0])
	return _trace2_code_for_role("exit")


func _trace2_world_offscreen(world: Vector3) -> bool:
	return _trace2_offscreen_side(world) != ""


func _trace2_offscreen_side(world: Vector3) -> String:
	var cell_x := (world.x - TRACE_GRID_ORIGIN.x) / maxf(0.001, trace_grid_scale)
	var visible_limit := maxf(2.0, float(trace_grid_half_x) - trace2_offscreen_margin_cells)
	if cell_x > visible_limit:
		return "right"
	if cell_x < -visible_limit:
		return "left"
	return ""


func _trace2_extreme_track(cell_key: String, axis: String, highest: bool) -> Dictionary:
	var best := _as_dict(trace_tracks[0])
	var best_value := _trace2_axis_value(_trace_vec3(best.get(cell_key, Vector3.ZERO)), axis)
	for raw_track in trace_tracks:
		var track := _as_dict(raw_track)
		var value := _trace2_axis_value(_trace_vec3(track.get(cell_key, Vector3.ZERO)), axis)
		if (highest and value > best_value) or (not highest and value < best_value):
			best = track
			best_value = value
	return best


func _trace2_axis_value(cell: Vector3, axis: String) -> float:
	if axis == "x":
		return cell.x
	if axis == "z":
		return cell.z
	return cell.y


func _trace2_code_for_role(role: String) -> int:
	for raw_track in trace_tracks:
		var track := _as_dict(raw_track)
		if str(track.get("role", "")) == role:
			return int(track.get("code", 0))
	return 0


func _trace2_submit_answer(code: int, timeout: bool) -> bool:
	if trace2_stage != "question":
		return false
	var response := "timeout" if timeout else str(code)
	var is_correct := not timeout and int(code) == int(trace2_correct_option)
	attempted += 1
	if is_correct:
		correct += 1
	var score := 1.0 if is_correct else 0.0
	total_score += score
	max_score += 1.0
	var evt := {
		"family": kind,
		"kind": "clip_question",
		"phase": phase,
		"item_index": attempted - 1,
		"is_scored": phase == "scored",
		"is_correct": is_correct,
		"is_timeout": timeout,
		"response_time_ms": int(round(maxf(0.0, elapsed_s - trace2_question_started_at_s) * 1000.0)),
		"score": score,
		"max_score": 1.0,
		"occurred_at_ms": int(round(elapsed_s * 1000.0)),
		"prompt": _trace2_question_stem(trace2_question_kind),
		"expected": str(trace2_correct_option),
		"response": response,
		"extra": {
			"question_kind": trace2_question_kind,
			"trial_index": trace2_trial_index,
			"options": trace2_options,
			"observe_duration_s": trace2_observe_s,
			"offscreen_margin_cells": trace2_offscreen_margin_cells,
			"new_clip_after_answer": true,
		},
	}
	event_log.append(evt)
	_send("godot_event", {"run_key": run_key, "kind": kind, "test_code": test_code, "event": evt})
	_hide_trace2_question_overlay()
	trace2_trial_index += 1
	if elapsed_s < duration_s:
		_trace2_start_trial()
	return true


func _trace2_option_for_key(key: int) -> int:
	if key == KEY_1 or key == KEY_KP_1 or key == KEY_A:
		return 1
	if key == KEY_2 or key == KEY_KP_2 or key == KEY_S:
		return 2
	if key == KEY_3 or key == KEY_KP_3 or key == KEY_D:
		return 3
	if key == KEY_4 or key == KEY_KP_4 or key == KEY_F:
		return 4
	return 0


func _trace2_question_stem(question_kind: String) -> String:
	if question_kind == TRACE2_QUESTION_STARTED_TOP:
		return "Which aircraft started on top?"
	if question_kind == TRACE2_QUESTION_STARTED_BOTTOM:
		return "Which aircraft started on bottom?"
	if question_kind == TRACE2_QUESTION_TURNED_LEFT:
		return "Which aircraft made a left turn?"
	if question_kind == TRACE2_QUESTION_TURNED_RIGHT:
		return "Which aircraft made a right turn?"
	if question_kind == TRACE2_QUESTION_ENDED_HIGHEST:
		return "Which aircraft ended highest?"
	if question_kind == TRACE2_QUESTION_NO_TURN:
		return "Which aircraft did not change direction?"
	if question_kind == TRACE2_QUESTION_ENDED_OFF_SCREEN:
		return "Which aircraft ended off screen?"
	if question_kind == TRACE2_QUESTION_LEFT_SCREEN:
		return "Which aircraft left the screen?"
	if question_kind == TRACE2_QUESTION_ENDED_OFF_RIGHT:
		return "Which aircraft ended off the right side?"
	return "Which aircraft matches the clip?"


func _trace2_option_text() -> String:
	var parts := []
	for option in trace2_options:
		var item := _as_dict(option)
		parts.append(str(item.get("code", "")) + ") " + str(item.get("label", "")))
	return "   ".join(parts)


func _trace_choose_next_dir(track: Dictionary, forced_edge_turn: bool) -> int:
	var cell := _trace_vec3(track.get("current_cell", Vector3.ZERO))
	var current_idx := int(track.get("dir_idx", 0))
	if not forced_edge_turn and not _trace_forward_blocked(cell, current_idx):
		var straight_chance := 0.52 - difficulty * 0.18
		if rng.randf() < straight_chance:
			return current_idx
	var current_dir := _trace_dir(current_idx)
	var candidates: Array = []
	for idx in range(TRACE_DIRS.size()):
		var candidate_dir := _trace_dir(idx)
		var dot := candidate_dir.dot(current_dir)
		if absf(dot) > 0.05:
			continue
		if _trace_cell_in_bounds(cell + candidate_dir):
			candidates.append(idx)
	if candidates.is_empty() and _trace_cell_in_bounds(cell + (_trace_dir(current_idx) * -1.0)):
		candidates.append(_trace_opposite_dir(current_idx))
	if candidates.is_empty():
		return current_idx
	return int(candidates[int(rng.randi_range(0, candidates.size() - 1))])


func _trace_initial_dir(local: RandomNumberGenerator, cell: Vector3) -> int:
	var candidates: Array = []
	for idx in range(TRACE_DIRS.size()):
		if _trace_cell_in_bounds(cell + _trace_dir(idx)):
			candidates.append(idx)
	if candidates.is_empty():
		return 0
	return int(candidates[int(local.randi_range(0, candidates.size() - 1))])


func _trace1_initial_dir(local: RandomNumberGenerator, cell: Vector3) -> int:
	var candidates: Array = []
	for idx in range(4):
		if _trace_cell_in_bounds(cell + _trace_dir(idx)):
			candidates.append(idx)
	if candidates.is_empty():
		return _trace_initial_dir(local, cell)
	return int(candidates[int(local.randi_range(0, candidates.size() - 1))])


func _trace1_choose_next_prompt_dir(track: Dictionary, forced_edge_turn: bool) -> int:
	var cell := _trace_vec3(track.get("current_cell", Vector3.ZERO))
	var current_idx := int(track.get("dir_idx", 0))
	var order := _trace1_response_order()
	for response in order:
		var candidate_idx := _trace1_dir_for_response(current_idx, str(response), track)
		if candidate_idx < 0:
			continue
		if _trace_cell_in_bounds(cell + _trace_dir(candidate_idx)):
			return candidate_idx
	var fallback_idx := _trace_choose_next_dir(track, forced_edge_turn)
	if _trace1_response_for_turn(current_idx, fallback_idx) != "":
		return fallback_idx
	for idx in range(TRACE_DIRS.size()):
		if idx == current_idx:
			continue
		if _trace_cell_in_bounds(cell + _trace_dir(idx)) and _trace1_response_for_turn(current_idx, idx) != "":
			return idx
	return current_idx


func _trace1_response_order() -> Array:
	var order := []
	var offset := int(posmod(session_seed + trace1_prompt_index + _kind_salt(test_code), TRACE1_RESPONSE_CYCLE.size()))
	for i in range(TRACE1_RESPONSE_CYCLE.size()):
		order.append(TRACE1_RESPONSE_CYCLE[int(posmod(offset + i, TRACE1_RESPONSE_CYCLE.size()))])
	return order


func _trace1_dir_for_response(current_idx: int, response: String, track: Dictionary) -> int:
	var idx := int(posmod(current_idx, TRACE_DIRS.size()))
	if idx < 4:
		if response == TRACE1_RESPONSE_LEFT_YAW:
			return int(posmod(idx + 3, 4))
		if response == TRACE1_RESPONSE_RIGHT_YAW:
			return int(posmod(idx + 1, 4))
		if response == TRACE1_RESPONSE_PULL_PITCH_UP:
			return 4
		if response == TRACE1_RESPONSE_PUSH_PITCH_DOWN:
			return 5
	else:
		var last_horizontal := int(clampi(int(track.get("last_horizontal_dir_idx", 0)), 0, 3))
		if idx == 4 and response == TRACE1_RESPONSE_PUSH_PITCH_DOWN:
			return last_horizontal
		if idx == 5 and response == TRACE1_RESPONSE_PULL_PITCH_UP:
			return last_horizontal
	return -1


func _trace1_response_for_turn(prev_idx: int, next_idx: int) -> String:
	var prev := int(posmod(prev_idx, TRACE_DIRS.size()))
	var next := int(posmod(next_idx, TRACE_DIRS.size()))
	if prev == next:
		return ""
	if prev < 4 and next < 4:
		if next == int(posmod(prev + 3, 4)):
			return TRACE1_RESPONSE_LEFT_YAW
		if next == int(posmod(prev + 1, 4)):
			return TRACE1_RESPONSE_RIGHT_YAW
	if next == 4 or (prev == 5 and next < 4):
		return TRACE1_RESPONSE_PULL_PITCH_UP
	if next == 5 or (prev == 4 and next < 4):
		return TRACE1_RESPONSE_PUSH_PITCH_DOWN
	return ""


func _trace1_open_response(prev_idx: int, next_idx: int, cell: Vector3, forced_edge_turn: bool) -> void:
	var expected := _trace1_response_for_turn(prev_idx, next_idx)
	if expected == "":
		trace1_expected_response = ""
		trace1_response_deadline_s = INF
		trace1_prompt_context = {}
		return
	var heading_pitch := _trace_heading_pitch_for_dir_idx(next_idx)
	trace1_expected_response = expected
	trace1_prompt_started_at_s = elapsed_s
	trace1_response_deadline_s = elapsed_s + trace1_response_window_s
	trace1_prompt_context = {
		"prompt_index": trace1_prompt_index,
		"cell": _trace_cell_token(cell),
		"previous_heading": _trace_dir_name(prev_idx),
		"target_heading": _trace_dir_name(next_idx),
		"heading_deg": heading_pitch.x,
		"pitch_deg": heading_pitch.y,
		"forced_edge_turn": forced_edge_turn,
	}
	trace1_prompt_hash = _trace_hash_mix(trace1_prompt_hash, _kind_salt(expected) + _trace_cell_hash(cell) + next_idx * 29)
	trace1_prompt_index += 1


func _trace1_submit_response(response: String, timeout: bool) -> bool:
	if trace1_expected_response == "":
		return false
	var expected := trace1_expected_response
	var is_correct := not timeout and response == expected
	attempted += 1
	if is_correct:
		correct += 1
	var score := 1.0 if is_correct else 0.0
	total_score += score
	max_score += 1.0
	var context := _as_dict(trace1_prompt_context)
	var response_time_ms := int(round(maxf(0.0, elapsed_s - trace1_prompt_started_at_s) * 1000.0))
	var evt := {
		"family": kind,
		"kind": "orientation_response",
		"phase": phase,
		"item_index": attempted - 1,
		"is_scored": phase == "scored",
		"is_correct": is_correct,
		"is_timeout": timeout,
		"response_time_ms": response_time_ms,
		"score": score,
		"max_score": 1.0,
		"occurred_at_ms": int(round(elapsed_s * 1000.0)),
		"prompt": "red_aircraft_stick_movement",
		"expected": expected,
		"response": response,
		"extra": context,
	}
	event_log.append(evt)
	_send("godot_event", {"run_key": run_key, "kind": kind, "test_code": test_code, "event": evt})
	trace1_expected_response = ""
	trace1_response_deadline_s = INF
	trace1_prompt_context = {}
	return true


func _trace1_update_timeout() -> void:
	if trace1_expected_response != "" and elapsed_s >= trace1_response_deadline_s:
		_trace1_submit_response("timeout", true)


func _trace1_poll_joystick_response() -> void:
	if trace1_expected_response == "":
		return
	if Input.get_connected_joypads().is_empty():
		trace1_joystick_ready = true
		return
	var joy := int(Input.get_connected_joypads()[0])
	var x := Input.get_joy_axis(joy, JOY_AXIS_LEFT_X)
	var y := Input.get_joy_axis(joy, JOY_AXIS_LEFT_Y)
	if absf(x) < 0.35 and absf(y) < 0.35:
		trace1_joystick_ready = true
		return
	if not trace1_joystick_ready:
		return
	if absf(x) >= absf(y):
		_trace1_submit_response(TRACE1_RESPONSE_LEFT_YAW if x < 0.0 else TRACE1_RESPONSE_RIGHT_YAW, false)
	else:
		_trace1_submit_response(TRACE1_RESPONSE_PUSH_PITCH_DOWN if y < 0.0 else TRACE1_RESPONSE_PULL_PITCH_UP, false)
	trace1_joystick_ready = false


func _trace_forward_blocked(cell: Vector3, dir_idx: int) -> bool:
	return not _trace_cell_in_bounds(cell + _trace_dir(dir_idx))


func _trace_cell_in_bounds(cell: Vector3) -> bool:
	return (
		int(cell.x) >= -trace_grid_half_x
		and int(cell.x) <= trace_grid_half_x
		and int(cell.y) >= 0
		and int(cell.y) <= trace_grid_max_y
		and int(cell.z) >= -trace_grid_half_z
		and int(cell.z) <= trace_grid_half_z
	)


func _trace_dir(idx: int) -> Vector3:
	return TRACE_DIRS[int(posmod(idx, TRACE_DIRS.size()))]


func _trace_opposite_dir(idx: int) -> int:
	var dir := _trace_dir(idx) * -1.0
	for candidate_idx in range(TRACE_DIRS.size()):
		if _trace_dir(candidate_idx).is_equal_approx(dir):
			return candidate_idx
	return idx


func _trace_track_world(track: Dictionary) -> Vector3:
	var phase_name := str(track.get("phase", "move"))
	if phase_name == "move":
		var from_cell := _trace_vec3(track.get("from_cell", Vector3.ZERO))
		var to_cell := _trace_vec3(track.get("to_cell", from_cell))
		var move_s := maxf(0.05, _float(track.get("move_s", 0.72)))
		var t := _smoothstep01(_float(track.get("phase_t", 0.0)) / move_s)
		return _trace_cell_to_world(from_cell).lerp(_trace_cell_to_world(to_cell), t)
	return _trace_cell_to_world(_trace_vec3(track.get("current_cell", Vector3.ZERO)))


func _trace_track_direction(track: Dictionary) -> Vector3:
	var next_dir := _trace_dir(int(track.get("dir_idx", 0)))
	if str(track.get("phase", "move")) != "pause":
		return next_dir
	var prev_dir := _trace_dir(int(track.get("prev_dir_idx", int(track.get("dir_idx", 0)))))
	var pause_s := maxf(0.05, _float(track.get("pause_s", 0.30)))
	var t := _smoothstep01(clampf(_float(track.get("phase_t", 0.0)) / maxf(0.001, pause_s * 0.72), 0.0, 1.0))
	var mixed := prev_dir.lerp(next_dir, t)
	if mixed.length() < 0.01:
		return next_dir
	return mixed.normalized()


func _trace_cell_to_world(cell: Vector3) -> Vector3:
	return TRACE_GRID_ORIGIN + Vector3(cell.x * trace_grid_scale, cell.y * trace_grid_alt_scale, cell.z * trace_grid_scale)


func _sync_trace_target() -> void:
	if trace_tracks.is_empty():
		target_pos = Vector2.ZERO
		return
	var world := _trace_track_world(_as_dict(trace_tracks[0]))
	var max_x := maxf(1.0, float(trace_grid_half_x) * trace_grid_scale)
	var max_y := maxf(1.0, float(trace_grid_max_y) * trace_grid_alt_scale)
	target_pos = Vector2(
		clampf(world.x / max_x, -1.0, 1.0),
		clampf(((world.y - TRACE_GRID_ORIGIN.y) / max_y) * 1.65 - 0.82, -1.0, 1.0)
	)


func _draw_trace_grid() -> void:
	for level in range(trace_grid_max_y + 1):
		var y := TRACE_GRID_ORIGIN.y + float(level) * trace_grid_alt_scale
		for x in range(-trace_grid_half_x, trace_grid_half_x + 1):
			var a := _trace_cell_to_world(Vector3(float(x), float(level), -trace_grid_half_z))
			var b := _trace_cell_to_world(Vector3(float(x), float(level), trace_grid_half_z))
			_make_segment(a, b, GRID_COLOR.darkened(0.24), 0.010)
		for z in range(-trace_grid_half_z, trace_grid_half_z + 1):
			var a := _trace_cell_to_world(Vector3(-trace_grid_half_x, float(level), float(z)))
			var b := _trace_cell_to_world(Vector3(trace_grid_half_x, float(level), float(z)))
			_make_segment(a, b, GRID_COLOR.darkened(0.24), 0.010)
	_make_box(Vector3(0.0, TRACE_GRID_ORIGIN.y - 0.01, TRACE_GRID_ORIGIN.z), Vector3(float(trace_grid_half_x) * trace_grid_scale, 0.006, float(trace_grid_half_z) * trace_grid_scale), Color(0.08, 0.12, 0.14, 0.16))
	for x in range(-trace_grid_half_x, trace_grid_half_x + 1):
		for z in range(-trace_grid_half_z, trace_grid_half_z + 1):
			var low := _trace_cell_to_world(Vector3(float(x), 0.0, float(z)))
			var high := _trace_cell_to_world(Vector3(float(x), float(trace_grid_max_y), float(z)))
			_make_segment(low, high, GRID_COLOR.darkened(0.38), 0.006)


func _trace_append_history(track: Dictionary, cell: Vector3) -> void:
	var history: Array = track.get("history", [])
	history.append(cell)
	while history.size() > 12:
		history.remove_at(0)
	track["history"] = history


func _trace_track_color(index: int) -> Color:
	if index % 4 == 1:
		return BLUE_COLOR
	if index % 4 == 2:
		return AMBER_COLOR
	if index % 4 == 3:
		return GREEN_COLOR
	return WHITE_COLOR.darkened(0.10)


func _trace_cell_token(cell: Vector3) -> String:
	return str(int(cell.x)) + "," + str(int(cell.y)) + "," + str(int(cell.z))


func _trace_cell_hash(cell: Vector3) -> int:
	return int((int(cell.x) + 11) * 101 + (int(cell.y) + 7) * 1009 + (int(cell.z) + 11) * 9173)


func _trace_hash_mix(value: int, part: int) -> int:
	return int((int(value) * 1103515245 + int(part) * 12345 + 97) % 2147483647)


func _trace_dir_name(idx: int) -> String:
	match int(posmod(idx, TRACE_DIRS.size())):
		0:
			return "north"
		1:
			return "east"
		2:
			return "south"
		3:
			return "west"
		4:
			return "climb"
		5:
			return "descend"
	return "unknown"


func _trace_heading_pitch_for_dir_idx(idx: int) -> Vector2:
	var dir := _trace_dir(idx).normalized()
	var heading := rad_to_deg(atan2(dir.x, -dir.z))
	var flat := maxf(0.001, Vector2(dir.x, dir.z).length())
	var pitch := rad_to_deg(atan2(dir.y, flat))
	return Vector2(heading, pitch)


func _smoothstep01(value: float) -> float:
	var t := clampf(value, 0.0, 1.0)
	return t * t * (3.0 - (2.0 * t))


func _update_camera(camera: Camera3D) -> void:
	if camera == null:
		return
	var pos := Vector3(0.0, 3.25, 8.6)
	var target := Vector3(0.0, 1.0, -5.2)
	var fov := 58.0
	camera.projection = Camera3D.PROJECTION_PERSPECTIVE
	if kind == "spatial_integration":
		pos = Vector3(0.0, 7.2, 7.4)
		target = Vector3(0.0, 0.0, -5.0)
	elif kind == "trace_test_1" or kind == "trace_test_2":
		_update_fixed_square_trace_camera(camera)
		return
	camera.position = camera.position.lerp(pos, 0.18)
	camera.fov = lerpf(camera.fov, fov, 0.18)
	camera.look_at(target, Vector3.UP)


func _update_fixed_square_trace_camera(camera: Camera3D) -> void:
	var center := TRACE_FIXED_ORTHO_CAMERA_TARGET
	var width := float(trace_grid_half_x) * trace_grid_scale * 2.0
	var height := float(trace_grid_max_y) * trace_grid_alt_scale + trace_panel_margin
	camera.projection = Camera3D.PROJECTION_ORTHOGONAL
	if kind == "trace_test_1":
		var visual_width := (width + trace_panel_margin) * trace_panel_visual_scale
		var visual_height := height * trace_panel_visual_scale
		camera.size = maxf(visual_height, visual_width * 0.58) * trace_camera_margin
	elif kind == "trace_test_2":
		var close_width := (width + trace_panel_margin) * trace2_close_camera_scale
		var close_height := height * trace2_close_camera_scale
		camera.size = maxf(close_height, close_width * 0.58) * trace_camera_margin
	else:
		camera.size = maxf(width, height) * trace_camera_margin * trace_panel_visual_scale
	camera.position = center + TRACE_FIXED_ORTHO_CAMERA_OFFSET
	camera.look_at(center, Vector3.UP)


func _update_hud() -> void:
	if hud_label == null:
		return
	var rem := maxf(0.0, duration_s - elapsed_s)
	if kind == "trace_test_1":
		var prompt_text := "watch red aircraft"
		if trace1_expected_response != "":
			prompt_text = "respond with stick movement"
		hud_label.text = _title_for_kind() + " | " + phase + " | " + str(int(rem)) + "s | " + str(correct) + "/" + str(attempted) + " | " + prompt_text
		return
	if kind == "trace_test_2":
		if trace2_stage == "question":
			hud_label.text = _title_for_kind() + " | " + phase + " | " + str(int(rem)) + "s | " + str(correct) + "/" + str(attempted) + " | answer the overlay"
		else:
			hud_label.text = _title_for_kind() + " | " + phase + " | " + str(int(rem)) + "s | watch the aircraft clip"
		return
	hud_label.text = _title_for_kind() + " | " + phase + " | " + str(int(rem)) + "s | " + str(correct) + "/" + str(attempted)


func _send_progress() -> void:
	_send("godot_progress", {
		"run_key": run_key,
		"kind": kind,
		"test_code": test_code,
		"phase": phase,
		"progress": {
			"elapsed_s": elapsed_s,
			"time_remaining_s": maxf(0.0, duration_s - elapsed_s),
			"attempted": attempted,
			"correct": correct,
			"score": total_score,
			"trace1_prompt_index": trace1_prompt_index,
			"trace1_response_open": trace1_expected_response != "",
			"trace2_stage": trace2_stage,
			"trace2_question_kind": trace2_question_kind,
		},
	})


func _build_nodes() -> void:
	scene_root = Node3D.new()
	scene_root.name = "SceneRoot"
	add_child(scene_root)
	hud_layer = CanvasLayer.new()
	add_child(hud_layer)
	hud_label = Label.new()
	hud_label.position = Vector2(12, 88)
	hud_label.size = Vector2(920, 132)
	hud_label.add_theme_font_size_override("font_size", 18)
	hud_label.add_theme_color_override("font_color", WHITE_COLOR)
	hud_layer.add_child(hud_label)
	_build_trace2_question_overlay()


func _build_trace2_question_overlay() -> void:
	trace2_overlay_layer = CanvasLayer.new()
	trace2_overlay_layer.name = "Trace2QuestionCanvasLayer"
	trace2_overlay_layer.layer = 30
	add_child(trace2_overlay_layer)
	trace2_overlay_root = Control.new()
	trace2_overlay_root.name = "Trace2QuestionOverlay"
	trace2_overlay_root.set_anchors_preset(Control.PRESET_FULL_RECT)
	trace2_overlay_root.visible = false
	trace2_overlay_layer.add_child(trace2_overlay_root)
	var backdrop := ColorRect.new()
	backdrop.name = "Trace2QuestionOverlayBackdrop"
	backdrop.color = Color(0.01, 0.02, 0.04, 0.985)
	backdrop.set_anchors_preset(Control.PRESET_FULL_RECT)
	trace2_overlay_root.add_child(backdrop)
	var box := VBoxContainer.new()
	box.name = "Trace2QuestionOverlayContent"
	box.position = Vector2(88, 82)
	box.size = Vector2(1040, 560)
	trace2_overlay_root.add_child(box)
	trace2_overlay_title = Label.new()
	trace2_overlay_title.name = "Trace2QuestionOverlayTitle"
	trace2_overlay_title.text = "Trace Test 2"
	trace2_overlay_title.add_theme_font_size_override("font_size", 34)
	trace2_overlay_title.add_theme_color_override("font_color", WHITE_COLOR)
	box.add_child(trace2_overlay_title)
	trace2_overlay_question = Label.new()
	trace2_overlay_question.name = "Trace2QuestionOverlayStem"
	trace2_overlay_question.add_theme_font_size_override("font_size", 28)
	trace2_overlay_question.add_theme_color_override("font_color", WHITE_COLOR)
	trace2_overlay_question.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	trace2_overlay_question.custom_minimum_size = Vector2(920, 98)
	box.add_child(trace2_overlay_question)
	trace2_overlay_options = Label.new()
	trace2_overlay_options.name = "Trace2QuestionOverlayOptions"
	trace2_overlay_options.add_theme_font_size_override("font_size", 24)
	trace2_overlay_options.add_theme_color_override("font_color", Color(0.86, 0.92, 1.0, 1.0))
	trace2_overlay_options.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	trace2_overlay_options.custom_minimum_size = Vector2(920, 220)
	box.add_child(trace2_overlay_options)
	trace2_overlay_footer = Label.new()
	trace2_overlay_footer.name = "Trace2QuestionOverlayFooter"
	trace2_overlay_footer.text = "Answer with 1-4 or A/S/D/F. Numpad Enter is ignored."
	trace2_overlay_footer.add_theme_font_size_override("font_size", 18)
	trace2_overlay_footer.add_theme_color_override("font_color", Color(0.72, 0.80, 0.90, 1.0))
	box.add_child(trace2_overlay_footer)


func _show_trace2_question_overlay() -> void:
	if not trace2_question_overlay_enabled or trace2_overlay_root == null:
		return
	trace2_overlay_title.text = "Trace Test 2 | Clip Question"
	trace2_overlay_question.text = _trace2_question_stem(trace2_question_kind)
	trace2_overlay_options.text = _trace2_option_text().replace("   ", "\n")
	trace2_overlay_footer.text = "Answer with 1-4 or A/S/D/F before time runs out. Numpad Enter is ignored."
	trace2_overlay_root.visible = true


func _hide_trace2_question_overlay() -> void:
	if trace2_overlay_root != null:
		trace2_overlay_root.visible = false


func _clear_runtime() -> void:
	for child in get_children():
		child.queue_free()
	active = false
	paused = false
	scene_root = null
	hud_layer = null
	hud_label = null
	trace2_overlay_layer = null
	trace2_overlay_root = null
	trace2_overlay_title = null
	trace2_overlay_question = null
	trace2_overlay_options = null
	trace2_overlay_footer = null


func _make_floor() -> void:
	_make_box(Vector3(0.0, -0.04, -5.4), Vector3(7.0, 0.05, 7.5), FLOOR_COLOR)


func _draw_grid(cols: int, rows: int, scale: float) -> void:
	for x in range(-cols, cols + 1):
		_make_box(Vector3(float(x) * scale, 0.025, -5.4), Vector3(0.008, 0.008, float(rows) * scale), GRID_COLOR)
	for z in range(-rows, rows + 1):
		_make_box(Vector3(0.0, 0.028, -5.4 + float(z) * scale), Vector3(float(cols) * scale, 0.008, 0.008), GRID_COLOR)


func _make_box(pos: Vector3, size: Vector3, color: Color) -> MeshInstance3D:
	return _make_box_child(scene_root, pos, size, color)


func _make_box_child(parent: Node3D, pos: Vector3, size: Vector3, color: Color) -> MeshInstance3D:
	var mesh := BoxMesh.new()
	mesh.size = size * 2.0
	var node := MeshInstance3D.new()
	node.mesh = mesh
	node.position = pos
	node.material_override = _mat(color)
	parent.add_child(node)
	return node


func _make_segment(a: Vector3, b: Vector3, color: Color, width: float) -> void:
	if a.distance_to(b) < 0.01:
		return
	var center := (a + b) * 0.5
	var node := _make_box(center, Vector3(width, width, a.distance_to(b) * 0.5), color)
	var dir := (b - a).normalized()
	var up := Vector3.UP
	if absf(dir.dot(up)) > 0.92:
		up = Vector3.FORWARD
	node.look_at(b, up)


func _make_sphere(pos: Vector3, radius: float, color: Color) -> void:
	var mesh := SphereMesh.new()
	mesh.radius = radius
	mesh.height = radius * 2.0
	var node := MeshInstance3D.new()
	node.mesh = mesh
	node.position = pos
	node.material_override = _mat(color)
	scene_root.add_child(node)


func _make_reticle(pos: Vector3, color: Color) -> void:
	_make_box(pos + Vector3(0.0, 0.0, 0.02), Vector3(0.48, 0.018, 0.018), color)
	_make_box(pos + Vector3(0.0, 0.0, 0.02), Vector3(0.018, 0.48, 0.018), color)


func _make_aircraft(pos: Vector3, color: Color, direction: Vector3 = Vector3(0.0, 0.0, -1.0), size: float = 1.0) -> void:
	var root := Node3D.new()
	root.name = "GridAircraft"
	root.position = pos
	scene_root.add_child(root)
	var look_dir := direction.normalized()
	if look_dir.length() < 0.01:
		look_dir = Vector3(0.0, 0.0, -1.0)
	var up := Vector3.UP
	if absf(look_dir.dot(up)) > 0.92:
		up = Vector3.FORWARD
	root.look_at(pos + look_dir, up)
	_make_box_child(root, Vector3(0.0, 0.0, -0.02 * size), Vector3(0.20 * size, 0.075 * size, 0.35 * size), color)
	_make_box_child(root, Vector3(0.0, 0.0, -0.38 * size), Vector3(0.12 * size, 0.065 * size, 0.15 * size), color.lightened(0.14))
	_make_box_child(root, Vector3(0.0, 0.01 * size, 0.02 * size), Vector3(0.54 * size, 0.024 * size, 0.10 * size), color.lightened(0.12))
	_make_box_child(root, Vector3(0.0, 0.08 * size, 0.34 * size), Vector3(0.09 * size, 0.12 * size, 0.10 * size), color.darkened(0.18))
	_make_box_child(root, Vector3(0.0, 0.06 * size, 0.36 * size), Vector3(0.25 * size, 0.018 * size, 0.07 * size), color.lightened(0.05))


func _screen_to_world(p: Vector2, z: float) -> Vector3:
	return Vector3(p.x * 3.2, 1.15 + p.y * 1.55, z)


func _seeded_point(index: int) -> Vector2:
	var local := RandomNumberGenerator.new()
	local.seed = session_seed + index * 977 + _kind_salt(kind)
	return Vector2(local.randf_range(-3.0, 3.0), local.randf_range(-2.8, 2.8))


func _mat(color: Color) -> StandardMaterial3D:
	var mat := StandardMaterial3D.new()
	mat.albedo_color = color
	mat.roughness = 0.72
	if color.a < 0.999:
		mat.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	return mat


func _mean_event_distance() -> float:
	if event_log.is_empty():
		return 0.0
	var total := 0.0
	for evt in event_log:
		var extra := _as_dict(_as_dict(evt).get("extra", {}))
		total += _float(extra.get("distance", 0.0))
	return total / float(event_log.size())


func _title_for_kind() -> String:
	if kind == "spatial_integration":
		return "Spatial Integration"
	if kind == "trace_test_1":
		return "Trace Test 1"
	if kind == "trace_test_2":
		return "Trace Test 2"
	return "Rapid Tracking"


func _kind_salt(value: String) -> int:
	var hash := 0
	for i in range(value.length()):
		hash = int((hash * 131 + value.unicode_at(i)) % 1000003)
	return max(1, hash)


func _send(command: String, payload: Dictionary) -> void:
	if control_sender.is_valid():
		control_sender.call(command, payload)


func _as_dict(value) -> Dictionary:
	if typeof(value) == TYPE_DICTIONARY:
		return value
	return {}


func _trace_vec3(value, fallback: Vector3 = Vector3.ZERO) -> Vector3:
	if typeof(value) == TYPE_VECTOR3:
		return value
	return fallback


func _float(value, default_value: float = 0.0) -> float:
	if typeof(value) == TYPE_INT or typeof(value) == TYPE_FLOAT:
		return float(value)
	var text := str(value)
	if text.is_valid_float():
		return float(text)
	return default_value

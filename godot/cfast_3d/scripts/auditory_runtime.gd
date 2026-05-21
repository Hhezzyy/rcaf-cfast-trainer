extends Node3D

const COLORS := ["RED", "BLUE", "YELLOW"]
const SHAPES := ["CIRCLE", "TRIANGLE", "SQUARE"]
const CALLSIGNS := ["EAGLE", "RAVEN", "FALCON", "VIPER", "COBRA", "TALON", "MOOSE", "LANCER", "SABER", "NOVA", "ORION", "ATLAS", "ARROW", "COMET", "NOMAD", "SUMMIT", "VECTOR", "RANGER", "HUNTER", "PHOENIX"]
const FILLER_NARRATION_LINES := [
	"The weather office notes soft haze over the western valley.",
	"Maintenance crews are sorting blank forms near the depot.",
	"A quiet briefing mentions low cloud along the coast.",
	"The archive clerk is filing routine papers in the station.",
	"Ground staff report mild wind beyond the service road.",
	"A planning note describes ordinary traffic near the hangar.",
	"The supply desk is checking labels on empty crates.",
	"Radio staff are reading a calm bulletin about local rain.",
]
const TRIANGLE_POINTS := [
	Vector2(0.0, 1.22),
	Vector2(-1.056, -0.61),
	Vector2(1.056, -0.61),
]
const PASS_COLOR := Color(0.28, 0.92, 0.52, 1.0)
const ERROR_COLOR := Color(1.0, 0.18, 0.16, 1.0)
const TUBE_COLOR := Color(0.003, 0.010, 0.016, 1.0)
const TUBE_ALT_COLOR := Color(0.010, 0.022, 0.030, 1.0)
const GATE_STROKE_WIDTH := 0.085
const GATE_INTERVAL_VISUAL_SCALE := 1.35
const GATE_SPAWN_AHEAD_DISTANCE := 22.0
const GATE_VISIBLE_AHEAD_DISTANCE := 24.0
const GATE_VISIBLE_BEHIND_DISTANCE := 12.0
const GATE_RETIRE_BEHIND_DISTANCE := 13.5
const BALL_IDLE_COLOR := Color(0.88, 0.96, 1.0, 1.0)
const RED_COLOR := Color(0.95, 0.18, 0.15, 1.0)
const BLUE_COLOR := Color(0.12, 0.42, 0.95, 1.0)
const YELLOW_COLOR := Color(0.95, 0.66, 0.16, 1.0)
const WHITE_COLOR := Color(0.92, 0.96, 0.98, 1.0)
const BLACK_COLOR := Color(0.05, 0.06, 0.07, 1.0)

class StreamRng:
	var rng := RandomNumberGenerator.new()

	func _init(seed_value: int) -> void:
		rng.seed = int(max(1, seed_value))

	func randf() -> float:
		return rng.randf()

	func randf_range(a: float, b: float) -> float:
		return rng.randf_range(a, b)

	func randi_range(a: int, b: int) -> int:
		return rng.randi_range(a, b)

	func choice(items: Array):
		if items.is_empty():
			return null
		return items[int(rng.randi_range(0, items.size() - 1))]


var control_sender: Callable
var run_key := ""
var completed_run_key := ""
var session_seed := 0
var difficulty := 0.0
var phase := ""
var active := false
var aborted := false
var paused := false
var elapsed_s := 0.0
var duration_s := 0.0
var tick_hz := 120.0
var tick_accum := 0.0
var progress_accum := 0.0
var render_accum := 0.0
var travel_distance := 0.0
var travel_speed := 0.22
var camera_distance := -7.4
var ball_roll_angle := 0.0
var tube_half_width := 0.85
var tube_half_height := 0.61
var inner_rx := 2.86
var inner_rz := 2.10
var ball_radius := 0.11
var physics_ball_radius := 0.28
var wall_bounce_factor := 0.42
var tube_chunk_count := 22
var tube_chunk_length := 2.0
var tube_radial_segments := 16
var tube_chunk_samples := 3
var control_gain := 1.24
var disturbance_gain := 0.52
var curvature_intensity := 0.88
var twist_intensity := 0.68
var response_window_s := 2.4
var sequence_response_s := 7.0
var gate_interval_s := 3.85
var command_interval_s := 3.8
var directive_interval_s := 5.6
var sequence_interval_s := 40.0
var sequence_first_s := 28.0
var beep_interval_s := 36.0
var distractor_interval_s := 3.0
var digit_sequence_min_len := 4
var digit_sequence_max_len := 6
var beep_frequency_hz := 1120.0
var beep_duration_s := 0.12
var beep_volume_db := -6.0
var review_mode_enabled := false
var ambient_volume_db := -14.0
var ambient_layer_drop_db := -2.5
var primary_voice_volume := 62.0
var distractor_voice_volume := 76.0
var filler_voice_volume := 40.0
var secondary_voice_min_difficulty := 0.67
var filler_narrator_interval_s := 13.0
var asset_root := ""
var assigned_callsigns: Array = []
var active_channels: Array = []
var segments: Array = []
var segment_index := 0
var segment_started_at_s := 0.0
var segment_duration_s := 0.0
var segment_label := "Full Mixed"
var rng_tunnel: StreamRng
var rng_gates: StreamRng
var rng_instructions: StreamRng
var rng_disturbance: StreamRng
var rng_audio: StreamRng
var anchors: Array = []
var gates: Array = []
var next_gate_id := 1
var next_gate_at_s := 1.0
var next_state_command_at_s := 2.0
var next_directive_at_s := 6.0
var next_sequence_at_s := 28.0
var next_beep_at_s := 7.0
var next_distractor_at_s := 9.0
var next_filler_at_s := 4.5
var active_command := {}
var active_gate_directive := {}
var active_recall := {}
var active_beep := {}
var forbidden_gate_color = null
var forbidden_gate_shape = null
var memory_buffer := ""
var typed_digits := ""
var ball_display_color := BALL_IDLE_COLOR
var ball_color_label := "neutral"
var ball_feedback_color := BALL_IDLE_COLOR
var ball_feedback_until_s := 0.0
var ball_pos := Vector2.ZERO
var ball_vel := Vector2.ZERO
var last_wall_collision_at_s := -1000.0
var disturbance := Vector2.ZERO
var disturbance_until_s := 0.0
var metrics := {}
var event_log: Array = []
var tube_root: Node3D
var tube_chunks: Array = []
var tube_last_band := -999999
var ball_body: CharacterBody3D
var ball_mesh_node: MeshInstance3D
var gate_root: Node3D
var hud_layer: CanvasLayer
var hud_label: Label
var beep_player: AudioStreamPlayer
var ambient_players: Array = []
var primary_voice_id := ""
var filler_voice_id := ""
var decoy_voice_id := ""
var tts_utterance_id := 1


func start(spec: Dictionary, sender: Callable) -> void:
	control_sender = sender
	var next_key := str(spec.get("run_key", ""))
	if active and next_key == run_key:
		return
	if next_key != "" and next_key == completed_run_key:
		return
	_clear_runtime()
	run_key = next_key
	session_seed = int(spec.get("session_seed", 0))
	difficulty = clampf(_float(spec.get("difficulty", 0.0)), 0.0, 1.0)
	phase = str(spec.get("phase", "practice")).to_lower()
	elapsed_s = 0.0
	travel_distance = 0.0
	camera_distance = -7.4
	ball_roll_angle = 0.0
	ball_display_color = BALL_IDLE_COLOR
	ball_color_label = "neutral"
	ball_feedback_color = BALL_IDLE_COLOR
	ball_feedback_until_s = 0.0
	ball_pos = Vector2.ZERO
	ball_vel = Vector2.ZERO
	last_wall_collision_at_s = -1000.0
	disturbance = Vector2.ZERO
	disturbance_until_s = 0.0
	tube_last_band = -999999
	duration_s = maxf(1.0, _float(spec.get("duration_s", 60.0)))
	asset_root = str(spec.get("asset_root", ""))
	_seed_streams()
	var cfg := _as_dict(spec.get("config", {}))
	tick_hz = maxf(30.0, _float(cfg.get("tick_hz", 120.0)))
	control_gain = _float(cfg.get("control_gain", 1.24))
	disturbance_gain = _float(cfg.get("disturbance_gain", 0.52))
	tube_half_width = maxf(0.20, _float(cfg.get("tube_half_width", 0.85)))
	tube_half_height = maxf(0.18, _float(cfg.get("tube_half_height", 0.61)))
	inner_rx = maxf(0.60, _float(cfg.get("inner_rx", 2.86)))
	inner_rz = maxf(0.42, _float(cfg.get("inner_rz", 2.10)))
	ball_radius = maxf(0.05, _float(cfg.get("ball_radius", 0.11)))
	physics_ball_radius = maxf(0.08, _float(cfg.get("physics_ball_radius", 0.28)))
	wall_bounce_factor = clampf(_float(cfg.get("wall_bounce_factor", 0.42)), 0.0, 0.95)
	tube_chunk_count = max(8, int(cfg.get("tube_chunk_count", 22)))
	tube_chunk_length = maxf(0.75, _float(cfg.get("tube_chunk_length", 2.0)))
	tube_radial_segments = max(8, int(cfg.get("tube_radial_segments", 16)))
	tube_chunk_samples = max(2, int(cfg.get("tube_chunk_samples", 3)))
	curvature_intensity = maxf(0.0, _float(cfg.get("tunnel_curvature_intensity", 0.88)))
	twist_intensity = maxf(0.0, _float(cfg.get("tunnel_twist_intensity", 0.68)))
	var difficulty_scaled := bool(cfg.get("difficulty_scaled", false))
	gate_interval_s = maxf(0.90, _float(cfg.get("gate_interval_s", 3.85)) * GATE_INTERVAL_VISUAL_SCALE)
	command_interval_s = maxf(0.75, _float(cfg.get("command_interval_s", 1.0 / maxf(0.05, _float(cfg.get("command_rate", 0.26))))))
	directive_interval_s = maxf(4.0, _float(cfg.get("directive_interval_s", maxf(5.2, gate_interval_s * 1.45))))
	sequence_interval_s = maxf(7.0, _float(cfg.get("sequence_interval_s", 40.0)))
	sequence_first_s = maxf(4.0, _float(cfg.get("sequence_first_s", sequence_interval_s * 0.60)))
	beep_interval_s = maxf(6.0, _float(cfg.get("beep_interval_s", 36.0)))
	distractor_interval_s = maxf(1.0, _float(cfg.get("distractor_interval_s", 1.0 / maxf(0.05, _float(cfg.get("distractor_rate", 0.34))))))
	if not difficulty_scaled:
		gate_interval_s = maxf(0.90, gate_interval_s * (1.0 - difficulty * 0.42))
		beep_interval_s = maxf(8.0, beep_interval_s * (1.0 - difficulty * 0.35))
	response_window_s = maxf(0.45, _float(cfg.get("response_window_seconds", 2.4)))
	sequence_response_s = maxf(response_window_s, _float(cfg.get("sequence_response_s", 7.0)))
	digit_sequence_min_len = max(1, int(cfg.get("digit_sequence_min_len", 4)))
	digit_sequence_max_len = max(digit_sequence_min_len, int(cfg.get("digit_sequence_max_len", 6)))
	beep_frequency_hz = maxf(120.0, _float(cfg.get("beep_frequency_hz", 1120.0)))
	beep_duration_s = maxf(0.03, _float(cfg.get("beep_duration_s", 0.12)))
	beep_volume_db = _float(cfg.get("beep_volume_db", -6.0))
	review_mode_enabled = bool(cfg.get("review_mode_enabled", false))
	ambient_volume_db = _float(cfg.get("ambient_volume_db", -14.0))
	ambient_layer_drop_db = _float(cfg.get("ambient_layer_drop_db", -2.5))
	primary_voice_volume = clampf(_float(cfg.get("primary_voice_volume", 62.0)), 0.0, 100.0)
	distractor_voice_volume = clampf(_float(cfg.get("distractor_voice_volume", 76.0)), 0.0, 100.0)
	filler_voice_volume = clampf(_float(cfg.get("filler_voice_volume", 40.0)), 0.0, 100.0)
	secondary_voice_min_difficulty = clampf(_float(cfg.get("secondary_voice_min_difficulty", 0.67)), 0.0, 1.0)
	filler_narrator_interval_s = maxf(4.0, _float(cfg.get("filler_narrator_interval_s", 13.0)))
	travel_speed = maxf(
		3.2,
		_float(cfg.get("gate_speed_norm_per_s", 0.33)) * 12.0 * (0.92 + difficulty * 0.28)
	)
	segments = _normalize_segments(_as_array(cfg.get("segments", [])), duration_s, _as_array(cfg.get("active_channels", [])))
	assigned_callsigns = _as_array(spec.get("assigned_callsigns", []))
	active_channels = _as_array(spec.get("active_channels", []))
	if active_channels.is_empty():
		active_channels = _as_array(cfg.get("active_channels", []))
	if assigned_callsigns.is_empty():
		assigned_callsigns = _pick_callsigns(max(1, int(cfg.get("callsign_count", 3))))
	_apply_segment(0, true)
	_build_tunnel_anchors()
	metrics = {
		"gate_hits": 0,
		"gate_misses": 0,
		"forbidden_gate_hits": 0,
		"collisions": 0,
		"false_alarms": 0,
		"correct_command_executions": 0,
		"missed_valid_commands": 0,
		"false_responses_to_distractors": 0,
		"digit_recall_attempts": 0,
		"digit_recall_score_total": 0.0,
		"digit_recall_accuracy": 0.0,
		"points": 0.0,
	}
	_schedule_current_segment(true)
	_build_nodes()
	if not _prepare_tts():
		_abort("tts_unavailable", "Godot/platform TTS is unavailable for Auditory Capacity.")
		return
	var merged_audio := _as_dict(cfg.get("audio", {}))
	for key in _as_dict(spec.get("audio", {})).keys():
		merged_audio[key] = _as_dict(spec.get("audio", {}))[key]
	_prepare_audio(merged_audio)
	active = true
	aborted = false
	_speak("Assigned call signs. " + _speech_callsign_list(assigned_callsigns) + ".", true)
	_send("godot_ready", {"run_key": run_key, "phase": phase, "kind": "auditory_capacity", "test_code": str(spec.get("test_code", "auditory_capacity"))})
	_rebuild_scene(true)


func update_runtime(delta: float, camera: Camera3D) -> void:
	if not active or aborted or paused:
		return
	var dt := minf(maxf(delta, 0.0), 0.05)
	tick_accum += dt
	var fixed_dt := 1.0 / tick_hz
	while tick_accum >= fixed_dt:
		tick_accum -= fixed_dt
		_step(fixed_dt)
	progress_accum += dt
	render_accum += dt
	if render_accum >= 1.0 / 30.0:
		render_accum = 0.0
		_rebuild_scene(false)
	_update_camera(camera)
	_update_hud()
	if progress_accum >= 0.25:
		progress_accum = 0.0
		_send_progress()


func handle_key(event: InputEventKey) -> bool:
	if not active or aborted or paused:
		return false
	if event.echo or not event.pressed:
		return false
	var key := event.keycode
	if key == KEY_SPACE:
		_submit_trigger()
		return true
	if key == KEY_Q:
		_submit_color("BLUE")
		return true
	if key == KEY_W:
		_submit_color("YELLOW")
		return true
	if key == KEY_E:
		_submit_color("RED")
		return true
	if key == KEY_R:
		_submit_color("RED")
		return true
	if key >= KEY_0 and key <= KEY_9:
		typed_digits += str(key - KEY_0)
		return true
	if key >= KEY_KP_0 and key <= KEY_KP_9:
		_submit_number(key - KEY_KP_0)
		return true
	if key == KEY_BACKSPACE:
		if typed_digits.length() > 0:
			typed_digits = typed_digits.substr(0, typed_digits.length() - 1)
		return true
	if key == KEY_KP_ENTER:
		return true
	if key == KEY_ENTER or key == KEY_KP_PERIOD:
		_submit_recall()
		return true
	return false


func set_paused(value: bool) -> void:
	var next_paused := bool(value)
	if paused == next_paused:
		return
	paused = next_paused
	if is_instance_valid(beep_player):
		beep_player.stream_paused = paused
	for player in ambient_players:
		if is_instance_valid(player) and player is AudioStreamPlayer:
			(player as AudioStreamPlayer).stream_paused = paused
	if primary_voice_id != "":
		if paused:
			if DisplayServer.has_method("tts_pause"):
				DisplayServer.tts_pause()
		else:
			if DisplayServer.has_method("tts_resume"):
				DisplayServer.tts_resume()


func _step(dt: float) -> void:
	elapsed_s += dt
	_sync_segment()
	travel_distance += travel_speed * dt
	ball_roll_angle += travel_speed * dt * 2.3
	_update_ball(dt)
	_update_disturbance(dt)
	_update_schedules()
	_update_gates()
	_expire_commands()
	if elapsed_s >= duration_s:
		_complete()


func _update_ball(dt: float) -> void:
	var input_vec := Vector2.ZERO
	if Input.is_key_pressed(KEY_LEFT):
		input_vec.x -= 1.0
	if Input.is_key_pressed(KEY_RIGHT):
		input_vec.x += 1.0
	if Input.is_key_pressed(KEY_UP):
		input_vec.y += 1.0
	if Input.is_key_pressed(KEY_DOWN):
		input_vec.y -= 1.0
	if Input.get_connected_joypads().size() > 0:
		var joy := int(Input.get_connected_joypads()[0])
		input_vec.x += Input.get_joy_axis(joy, JOY_AXIS_LEFT_X)
		input_vec.y -= Input.get_joy_axis(joy, JOY_AXIS_LEFT_Y)
	if input_vec.length() > 1.0:
		input_vec = input_vec.normalized()
	var accel := (input_vec * control_gain) + (disturbance * disturbance_gain)
	ball_vel += accel * dt
	ball_vel *= pow(0.18, dt)
	var desired_ball_pos := ball_pos + (ball_vel * dt)
	var desired_world := _world_position_for_ball(desired_ball_pos, travel_distance)
	if ball_body == null:
		ball_pos = desired_ball_pos
		return
	var motion := desired_world - ball_body.global_position
	var collision := ball_body.move_and_collide(motion)
	if collision != null:
		if elapsed_s - last_wall_collision_at_s > 0.15:
			metrics["collisions"] = int(metrics["collisions"]) + 1
			last_wall_collision_at_s = elapsed_s
		var frame := _frame_at(travel_distance)
		var normal: Vector3 = collision.get_normal()
		var local_normal := Vector2(normal.dot(frame["right"]), normal.dot(frame["up"]))
		if local_normal.length() > 0.001:
			ball_vel = ball_vel.bounce(local_normal.normalized()) * wall_bounce_factor
		else:
			ball_vel *= -wall_bounce_factor
		ball_body.global_position += normal * 0.015
	ball_pos = _ball_pos_from_world(ball_body.global_position, travel_distance)


func _update_disturbance(_dt: float) -> void:
	if elapsed_s < disturbance_until_s:
		return
	if rng_disturbance.randf() > 0.035:
		disturbance = Vector2.ZERO
		return
	var angle := rng_disturbance.randf_range(0.0, TAU)
	var magnitude := rng_disturbance.randf_range(0.10, 0.28 + difficulty * 0.15)
	disturbance = Vector2(cos(angle), sin(angle)) * magnitude
	disturbance_until_s = elapsed_s + rng_disturbance.randf_range(0.6, 1.4)


func _normalize_segments(raw_segments: Array, fallback_duration_s: float, fallback_channels: Array) -> Array:
	var normalized := []
	for raw in raw_segments:
		var segment := _as_dict(raw)
		if segment.is_empty():
			continue
		var duration := maxf(0.0, _float(segment.get("duration_s", 0.0)))
		if duration <= 0.001:
			continue
		var channels := _as_array(segment.get("active_channels", []))
		if channels.is_empty():
			channels = fallback_channels
		if channels.is_empty():
			channels = ["gates", "state_commands", "gate_directives", "digit_recall", "trigger", "distractors"]
		normalized.append({
			"label": str(segment.get("label", "Segment")),
			"duration_s": duration,
			"active_channels": channels,
			"effective": _as_dict(segment.get("effective", {})),
		})
	if normalized.is_empty():
		var channels2 := fallback_channels
		if channels2.is_empty():
			channels2 = ["gates", "state_commands", "gate_directives", "digit_recall", "trigger", "distractors"]
		normalized.append({
			"label": "Full Mixed",
			"duration_s": maxf(1.0, fallback_duration_s),
			"active_channels": channels2,
			"effective": {},
		})
	return normalized


func _sync_segment() -> void:
	if segments.is_empty():
		return
	while segment_index + 1 < segments.size() and elapsed_s >= segment_started_at_s + segment_duration_s:
		_apply_segment(segment_index + 1, true)


func _apply_segment(index: int, reset_schedules: bool) -> void:
	if segments.is_empty():
		return
	segment_index = clampi(index, 0, segments.size() - 1)
	var segment := _as_dict(segments[segment_index])
	segment_started_at_s = elapsed_s
	segment_duration_s = maxf(1.0, _float(segment.get("duration_s", duration_s)))
	segment_label = str(segment.get("label", "Segment"))
	var channels := _as_array(segment.get("active_channels", []))
	if not channels.is_empty():
		active_channels = channels
	var effective := _as_dict(segment.get("effective", {}))
	if not effective.is_empty():
		tube_half_width = maxf(0.20, _float(effective.get("tube_half_width", tube_half_width)))
		tube_half_height = maxf(0.18, _float(effective.get("tube_half_height", tube_half_height)))
		disturbance_gain = maxf(0.0, _float(effective.get("disturbance_gain", disturbance_gain)))
		response_window_s = maxf(0.45, _float(effective.get("response_window_seconds", response_window_s)))
		gate_interval_s = maxf(0.90, _float(effective.get("gate_interval_s", gate_interval_s)) * GATE_INTERVAL_VISUAL_SCALE)
		command_interval_s = maxf(0.75, _float(effective.get("command_interval_s", command_interval_s)))
		directive_interval_s = maxf(4.0, _float(effective.get("directive_interval_s", directive_interval_s)))
		sequence_interval_s = maxf(7.0, _float(effective.get("sequence_interval_s", sequence_interval_s)))
		sequence_first_s = maxf(4.0, _float(effective.get("sequence_first_s", sequence_first_s)))
		beep_interval_s = maxf(6.0, _float(effective.get("beep_interval_s", beep_interval_s)))
		digit_sequence_min_len = max(1, int(effective.get("digit_sequence_min_len", digit_sequence_min_len)))
		digit_sequence_max_len = max(digit_sequence_min_len, int(effective.get("digit_sequence_max_len", digit_sequence_max_len)))
	if reset_schedules:
		gates.clear()
		active_command = {}
		active_gate_directive = {}
		active_recall = {}
		active_beep = {}
		_schedule_current_segment(false)


func _schedule_current_segment(initial: bool) -> void:
	var start_offset := 0.0 if initial else 0.35
	next_gate_at_s = elapsed_s + start_offset + (0.8 if _channel_active("gates") else INF)
	next_state_command_at_s = elapsed_s + start_offset + (maxf(1.2, command_interval_s * 0.55) if _channel_active("state_commands") else INF)
	next_directive_at_s = elapsed_s + start_offset + (maxf(4.0, directive_interval_s * 0.70) if _channel_active("gate_directives") and _channel_active("gates") else INF)
	next_sequence_at_s = elapsed_s + start_offset + (sequence_first_s if _channel_active("digit_recall") else INF)
	next_beep_at_s = elapsed_s + start_offset + (maxf(5.0, beep_interval_s * 0.35) if _channel_active("trigger") else INF)
	next_distractor_at_s = elapsed_s + start_offset + (maxf(4.0, distractor_interval_s * 0.85) if _channel_active("distractors") else INF)
	next_filler_at_s = elapsed_s + start_offset + maxf(3.5, filler_narrator_interval_s * 0.35)


func _update_schedules() -> void:
	if elapsed_s >= next_gate_at_s and _channel_active("gates"):
		_spawn_gate()
		next_gate_at_s = elapsed_s + _jitter(gate_interval_s, 0.28)
	if elapsed_s >= next_state_command_at_s and _channel_active("state_commands"):
		_spawn_state_command()
		next_state_command_at_s = elapsed_s + _jitter(command_interval_s, 0.34)
	if elapsed_s >= next_directive_at_s and _channel_active("gate_directives") and _channel_active("gates"):
		_spawn_gate_directive()
		next_directive_at_s = elapsed_s + _jitter(directive_interval_s, 0.24)
	if elapsed_s >= next_sequence_at_s and _channel_active("digit_recall"):
		_spawn_digit_sequence()
		next_sequence_at_s = elapsed_s + _jitter(sequence_interval_s, 0.28)
	if elapsed_s >= next_beep_at_s and _channel_active("trigger"):
		_spawn_beep()
		next_beep_at_s = elapsed_s + _jitter(beep_interval_s, 0.32)
	if elapsed_s >= next_distractor_at_s and _channel_active("distractors"):
		_spawn_distractor()
		next_distractor_at_s = elapsed_s + _jitter(distractor_interval_s, 0.42)
	if elapsed_s >= next_filler_at_s:
		_spawn_filler_narration()
		next_filler_at_s = elapsed_s + _jitter(filler_narrator_interval_s, 0.30)


func _spawn_gate() -> void:
	var lane_limit := maxf(0.12, tube_half_height - (ball_radius * 1.18))
	var lane_options := [
		-lane_limit,
		-lane_limit * 0.66,
		-lane_limit * 0.33,
		0.0,
		lane_limit * 0.33,
		lane_limit * 0.66,
		lane_limit,
	]
	var y_norm := float(rng_gates.choice(lane_options)) + rng_gates.randf_range(-0.045, 0.045)
	var color := str(rng_gates.choice(COLORS))
	var shape := str(rng_gates.choice(SHAPES))
	var aperture := rng_gates.randf_range(0.11, minf(0.25 - difficulty * 0.045, lane_limit * 0.70))
	gates.append({
		"id": next_gate_id,
		"distance": travel_distance + GATE_SPAWN_AHEAD_DISTANCE,
		"y_norm": clampf(y_norm, -lane_limit, lane_limit),
		"color": color,
		"shape": shape,
		"aperture": maxf(0.09, aperture),
		"scored": false,
	})
	next_gate_id += 1
	_bind_directive_to_next_gate()


func _update_gates() -> void:
	var kept := []
	for gate in gates:
		var g := _as_dict(gate)
		if not bool(g.get("scored", false)) and _float(g.get("distance", 0.0)) <= travel_distance:
			_score_gate(g)
		if _float(g.get("distance", 0.0)) >= travel_distance - GATE_RETIRE_BEHIND_DISTANCE:
			kept.append(g)
	gates = kept


func _score_gate(gate: Dictionary) -> void:
	var inside := absf(ball_pos.y - _float(gate.get("y_norm", 0.0))) <= _float(gate.get("aperture", 0.15))
	var should_pass := _gate_should_pass(gate)
	var correct := (inside == should_pass)
	if correct:
		metrics["gate_hits"] = int(metrics["gate_hits"]) + 1
		metrics["points"] = float(metrics["points"]) + 1.0
		_flash_ball(PASS_COLOR, 0.28)
	else:
		metrics["gate_misses"] = int(metrics["gate_misses"]) + 1
		if inside and not should_pass:
			metrics["forbidden_gate_hits"] = int(metrics["forbidden_gate_hits"]) + 1
		_flash_ball(ERROR_COLOR, 0.42)
	gate["scored"] = true
	if _as_dict(active_gate_directive).get("target_gate_id", -1) == int(gate.get("id", -2)):
		active_gate_directive = {}
	var pilot_action := "PASS" if inside else "SKIP"
	var expected := "PASS" if should_pass else "AVOID"
	_record_event("gate", expected, str(gate.get("color", "")) + "/" + str(gate.get("shape", "")) + "/" + pilot_action, correct, 1.0 if correct else 0.0)


func _replace_active_command() -> void:
	if not active_command.is_empty():
		metrics["missed_valid_commands"] = int(metrics["missed_valid_commands"]) + 1


func _activate_scheduled_command(command_type: String, payload) -> void:
	_replace_active_command()
	var callsign := str(rng_instructions.choice(assigned_callsigns))
	active_command = {
		"id": event_log.size() + 1,
		"type": command_type,
		"payload": payload,
		"callsign": callsign,
		"expires_at": elapsed_s + (sequence_response_s if command_type == "digit_sequence" else response_window_s),
	}
	_activate_command(active_command)


func _spawn_state_command() -> void:
	if rng_instructions.randf() < 0.55:
		_activate_scheduled_command("change_colour", str(rng_instructions.choice(COLORS)))
	else:
		_activate_scheduled_command("change_number", rng_instructions.randi_range(1, 9))


func _spawn_gate_directive() -> void:
	var match_kind := "COLOR" if rng_instructions.randf() < 0.58 else "SHAPE"
	_activate_scheduled_command("gate_directive", {
		"action": "AVOID" if rng_instructions.randf() < 0.58 else "PASS",
		"match_kind": match_kind,
		"match_value": str(rng_instructions.choice(COLORS if match_kind == "COLOR" else SHAPES)),
	})


func _spawn_digit_sequence() -> void:
	var length := rng_instructions.randi_range(digit_sequence_min_len, digit_sequence_max_len)
	_activate_scheduled_command("digit_sequence", _random_digits(length))


func _spawn_distractor() -> void:
	var pool := []
	for callsign in CALLSIGNS:
		if not assigned_callsigns.has(callsign):
			pool.append(callsign)
	if pool.is_empty():
		return
	var callsign := str(rng_instructions.choice(pool))
	if difficulty >= secondary_voice_min_difficulty:
		if rng_instructions.randf() < 0.5:
			_speak_role(_speech_callsign(callsign) + ". Change colour to " + _speech_color_name(str(rng_instructions.choice(COLORS))) + ".", "decoy", false)
		else:
			_speak_role(_speech_callsign(callsign) + ". Set number " + str(rng_instructions.randi_range(1, 9)) + ".", "decoy", false)
	else:
		_spawn_filler_narration()
	_record_event("distractor", "IGNORE", callsign, true, 0.0)


func _spawn_filler_narration() -> void:
	var line := str(rng_audio.choice(FILLER_NARRATION_LINES))
	if line == "":
		return
	_speak_role(line, "filler", false)


func _activate_command(command: Dictionary) -> void:
	var kind := str(command.get("type", ""))
	var callsign := str(command.get("callsign", ""))
	var payload = command.get("payload")
	if kind == "gate_directive":
		var directive := _as_dict(payload)
		active_gate_directive = {
			"directive": directive,
			"target_gate_id": null,
		}
		if str(directive.get("action", "")).to_upper() == "AVOID":
			if str(directive.get("match_kind", "")) == "COLOR":
				forbidden_gate_color = str(directive.get("match_value", ""))
			else:
				forbidden_gate_shape = str(directive.get("match_value", ""))
		_bind_directive_to_next_gate()
		var spoken_match := _speech_color_or_shape_name(str(directive.get("match_value", "")))
		_speak(_speech_callsign(callsign) + ". " + ("Avoid" if str(directive.get("action", "")) == "AVOID" else "Take") + " the next " + spoken_match + " gate.", false)
	elif kind == "digit_sequence":
		memory_buffer = str(payload)
		active_recall = {
			"target": memory_buffer,
			"expires_at": elapsed_s + sequence_response_s,
		}
		typed_digits = ""
		_speak(_speech_callsign(callsign) + ". Remember digits " + _spaced_digits(memory_buffer) + ".", false)
	elif kind == "change_number":
		_speak(_speech_callsign(callsign) + ". Set number " + str(payload) + ".", false)
	else:
		_speak(_speech_callsign(callsign) + ". Change colour to " + _speech_color_name(str(payload)) + ".", false)


func _spawn_beep() -> void:
	active_beep = {"expires_at": elapsed_s + response_window_s, "responded": false}
	_play_beep()


func _submit_color(color: String) -> void:
	ball_display_color = _color_by_name(color)
	ball_color_label = _speech_color_name(color)
	_refresh_ball_material()
	if active_command.is_empty() or str(active_command.get("type", "")) != "change_colour":
		_false_response("COL:" + color)
		return
	var expected := str(active_command.get("payload", "")).to_upper()
	var correct := color.to_upper() == expected
	_score_command("change_colour", expected, color.to_upper(), correct)


func _submit_number(number: int) -> void:
	if active_command.is_empty() or str(active_command.get("type", "")) != "change_number":
		_false_response("NUM:" + str(number))
		return
	var expected := str(active_command.get("payload", ""))
	var correct := str(number) == expected
	_score_command("change_number", expected, str(number), correct)


func _submit_trigger() -> void:
	if active_beep.is_empty() or bool(active_beep.get("responded", false)) or elapsed_s > _float(active_beep.get("expires_at", 0.0)):
		_false_response("TRIGGER")
		return
	active_beep["responded"] = true
	metrics["points"] = float(metrics["points"]) + 1.0
	_record_event("trigger", "PRESS_TRIGGER", "TRIGGER", true, 1.0)


func _submit_recall() -> void:
	if active_recall.is_empty():
		_false_response("SEQ:" + typed_digits)
		return
	var target := str(active_recall.get("target", ""))
	var score := _score_digits(target, typed_digits)
	metrics["digit_recall_attempts"] = int(metrics["digit_recall_attempts"]) + 1
	metrics["digit_recall_score_total"] = float(metrics["digit_recall_score_total"]) + score
	metrics["digit_recall_accuracy"] = float(metrics["digit_recall_score_total"]) / float(max(1, int(metrics["digit_recall_attempts"])))
	metrics["points"] = float(metrics["points"]) + score
	_record_event("digit_recall", target, typed_digits, score >= 0.999, score)
	active_recall = {}
	active_command = {}
	typed_digits = ""


func _score_command(kind: String, expected: String, response: String, correct: bool) -> void:
	if correct:
		metrics["correct_command_executions"] = int(metrics["correct_command_executions"]) + 1
		metrics["points"] = float(metrics["points"]) + 1.0
	else:
		metrics["missed_valid_commands"] = int(metrics["missed_valid_commands"]) + 1
	_record_event("command", expected, response, correct, 1.0 if correct else 0.0, kind)
	active_command = {}


func _false_response(response: String) -> void:
	metrics["false_alarms"] = int(metrics["false_alarms"]) + 1
	_record_event("false_response", "", response, false, 0.0)


func _expire_commands() -> void:
	if not active_command.is_empty() and elapsed_s > _float(active_command.get("expires_at", 0.0)):
		metrics["missed_valid_commands"] = int(metrics["missed_valid_commands"]) + 1
		_record_event("command", str(active_command.get("payload", "")), "MISS", false, 0.0, str(active_command.get("type", "")))
		active_command = {}
	if not active_recall.is_empty() and elapsed_s > _float(active_recall.get("expires_at", 0.0)):
		_submit_recall()
	if not active_beep.is_empty() and elapsed_s > _float(active_beep.get("expires_at", 0.0)):
		if not bool(active_beep.get("responded", false)):
			metrics["missed_valid_commands"] = int(metrics["missed_valid_commands"]) + 1
			_record_event("trigger", "PRESS_TRIGGER", "MISS", false, 0.0)
		active_beep = {}


func _record_event(kind: String, expected: String, response: String, correct: bool, score: float, command_type: String = "") -> void:
	var evt := {
		"phase": phase,
		"kind": kind,
		"expected": expected,
		"response": response,
		"is_correct": correct,
		"score": score,
		"occurred_at_s": elapsed_s,
		"command_type": command_type,
	}
	event_log.append(evt)
	_send("godot_event", {"run_key": run_key, "kind": "auditory_capacity", "test_code": "auditory_capacity", "event": evt})


func _complete() -> void:
	active = false
	completed_run_key = run_key
	var attempted := int(metrics["gate_hits"]) + int(metrics["gate_misses"]) + int(metrics["correct_command_executions"]) + int(metrics["missed_valid_commands"]) + int(metrics["digit_recall_attempts"])
	var correct := int(metrics["gate_hits"]) + int(metrics["correct_command_executions"])
	var max_score := float(max(1, attempted))
	var total_score := float(metrics["points"])
	var summary := {
		"attempted": attempted,
		"correct": correct,
		"accuracy": float(correct) / float(max(1, attempted)),
		"duration_s": duration_s,
		"throughput_per_min": float(attempted) / maxf(1.0, duration_s) * 60.0,
		"total_score": total_score,
		"max_score": max_score,
		"score_ratio": total_score / max_score,
	}
	var result := {
		"run_key": run_key,
		"kind": "auditory_capacity",
		"test_code": "auditory_capacity",
		"phase": "results" if phase == "scored" else "practice_done",
		"summary": summary,
		"metrics": metrics.duplicate(true),
		"events": event_log.slice(max(0, event_log.size() - 80), event_log.size()),
	}
	if phase == "practice":
		_send("godot_phase_complete", {"run_key": run_key, "phase": "practice", "kind": "auditory_capacity", "test_code": "auditory_capacity", "result": result})
		return
	_send("godot_complete", {"run_key": run_key, "phase": result["phase"], "kind": "auditory_capacity", "test_code": "auditory_capacity", "result": result})


func _rebuild_scene(force: bool) -> void:
	_rebuild_tube_chunks(force)
	_update_ball_node()
	_rebuild_gates()


func _build_nodes() -> void:
	tube_root = Node3D.new()
	tube_root.name = "AuditoryTubeRoot"
	add_child(tube_root)
	_build_tube_pool()
	_build_physics_ball()
	gate_root = Node3D.new()
	gate_root.name = "AuditoryGates"
	add_child(gate_root)
	hud_layer = CanvasLayer.new()
	add_child(hud_layer)
	hud_label = Label.new()
	hud_label.position = Vector2(14, 88)
	hud_label.size = Vector2(760, 190)
	hud_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	hud_label.add_theme_font_size_override("font_size", 18)
	hud_label.add_theme_color_override("font_color", WHITE_COLOR)
	hud_layer.add_child(hud_label)
	beep_player = AudioStreamPlayer.new()
	add_child(beep_player)
	_sync_ball_body_to_logical_position()
	_rebuild_tube_chunks(true)


func _build_tube_pool() -> void:
	tube_chunks.clear()
	if tube_root == null:
		return
	for child in tube_root.get_children():
		child.queue_free()
	for index in range(tube_chunk_count):
		var body := StaticBody3D.new()
		body.name = "AuditoryTubeChunkBody"
		var mesh_node := MeshInstance3D.new()
		mesh_node.name = "AuditoryTubeChunkMesh"
		var collision := CollisionShape3D.new()
		collision.name = "AuditoryTubeChunkCollision"
		body.add_child(mesh_node)
		body.add_child(collision)
		tube_root.add_child(body)
		tube_chunks.append({
			"body": body,
			"mesh": mesh_node,
			"collision": collision,
		})


func _rebuild_tube_chunks(force: bool) -> void:
	if tube_chunks.is_empty():
		return
	var first_band := int(floor((travel_distance - 12.0) / tube_chunk_length))
	if not force and first_band == tube_last_band:
		return
	tube_last_band = first_band
	for index in range(tube_chunks.size()):
		var entry := _as_dict(tube_chunks[index])
		_build_tube_chunk(entry, (float(first_band + index) * tube_chunk_length), index)


func _build_tube_chunk(entry: Dictionary, start_d: float, chunk_index: int) -> void:
	var vertices := PackedVector3Array()
	var indices := PackedInt32Array()
	var collision_faces := PackedVector3Array()
	for r in range(tube_chunk_samples + 1):
		var distance := start_d + (float(r) / float(tube_chunk_samples)) * tube_chunk_length
		var frame := _frame_at(distance)
		for s in range(tube_radial_segments):
			var angle := (float(s) / float(tube_radial_segments)) * TAU
			var point = frame["pos"] + frame["right"] * cos(angle) * inner_rx + frame["up"] * sin(angle) * inner_rz
			vertices.append(point)
	for r in range(tube_chunk_samples):
		for s in range(tube_radial_segments):
			var a := r * tube_radial_segments + s
			var b := r * tube_radial_segments + ((s + 1) % tube_radial_segments)
			var c := (r + 1) * tube_radial_segments + s
			var d := (r + 1) * tube_radial_segments + ((s + 1) % tube_radial_segments)
			indices.append(a)
			indices.append(c)
			indices.append(b)
			indices.append(b)
			indices.append(c)
			indices.append(d)
			_append_collision_triangle(collision_faces, vertices[a], vertices[c], vertices[b])
			_append_collision_triangle(collision_faces, vertices[b], vertices[c], vertices[d])
	var arrays := []
	arrays.resize(Mesh.ARRAY_MAX)
	arrays[Mesh.ARRAY_VERTEX] = vertices
	arrays[Mesh.ARRAY_INDEX] = indices
	var mesh := ArrayMesh.new()
	mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, arrays)
	var mesh_node: MeshInstance3D = entry.get("mesh")
	mesh_node.mesh = mesh
	mesh_node.material_override = _tube_material(TUBE_COLOR if chunk_index % 2 == 0 else TUBE_ALT_COLOR)
	var shape := ConcavePolygonShape3D.new()
	shape.set_faces(collision_faces)
	var collision: CollisionShape3D = entry.get("collision")
	collision.shape = shape


func _append_collision_triangle(faces: PackedVector3Array, a: Vector3, b: Vector3, c: Vector3) -> void:
	faces.append(a)
	faces.append(b)
	faces.append(c)
	faces.append(c)
	faces.append(b)
	faces.append(a)


func _build_physics_ball() -> void:
	ball_body = CharacterBody3D.new()
	ball_body.name = "AuditoryBallBody"
	var collision := CollisionShape3D.new()
	collision.name = "AuditoryBallCollision"
	var shape := SphereShape3D.new()
	shape.radius = physics_ball_radius
	collision.shape = shape
	ball_body.add_child(collision)
	var mesh := SphereMesh.new()
	mesh.radial_segments = 16
	mesh.rings = 8
	ball_mesh_node = MeshInstance3D.new()
	ball_mesh_node.name = "AuditoryBall"
	ball_mesh_node.mesh = mesh
	ball_mesh_node.scale = Vector3(physics_ball_radius, physics_ball_radius, physics_ball_radius)
	ball_mesh_node.material_override = _material(ball_display_color)
	ball_body.add_child(ball_mesh_node)
	add_child(ball_body)


func _refresh_ball_material() -> void:
	if is_instance_valid(ball_mesh_node):
		ball_mesh_node.material_override = _material(_active_ball_color())


func _active_ball_color() -> Color:
	if elapsed_s < ball_feedback_until_s:
		return ball_feedback_color
	return ball_display_color


func _flash_ball(color: Color, duration_s: float) -> void:
	ball_feedback_color = color
	ball_feedback_until_s = elapsed_s + maxf(0.05, duration_s)
	_refresh_ball_material()


func _update_ball_node() -> void:
	if ball_body == null or ball_mesh_node == null:
		return
	var frame := _frame_at(travel_distance)
	_refresh_ball_material()
	ball_body.look_at(_frame_at(travel_distance + 1.0)["pos"], frame["up"])
	ball_mesh_node.rotate_object_local(Vector3.RIGHT, ball_roll_angle)


func _sync_ball_body_to_logical_position() -> void:
	if ball_body == null:
		return
	ball_body.global_position = _world_position_for_ball(ball_pos, travel_distance)


func _world_position_for_ball(local_pos: Vector2, distance: float) -> Vector3:
	var frame := _frame_at(distance)
	var local_x := (local_pos.x / maxf(0.01, tube_half_width)) * maxf(0.05, inner_rx - physics_ball_radius)
	var local_y := (local_pos.y / maxf(0.01, tube_half_height)) * maxf(0.05, inner_rz - physics_ball_radius)
	return frame["pos"] + frame["right"] * local_x + frame["up"] * local_y


func _ball_pos_from_world(world_pos: Vector3, distance: float) -> Vector2:
	var frame := _frame_at(distance)
	var delta: Vector3 = world_pos - frame["pos"]
	var scale_x := maxf(0.05, inner_rx - physics_ball_radius)
	var scale_y := maxf(0.05, inner_rz - physics_ball_radius)
	return Vector2(
		(delta.dot(frame["right"]) / scale_x) * maxf(0.01, tube_half_width),
		(delta.dot(frame["up"]) / scale_y) * maxf(0.01, tube_half_height)
	)


func _rebuild_gates() -> void:
	if gate_root == null:
		return
	for child in gate_root.get_children():
		child.queue_free()
	for gate in gates:
		var g := _as_dict(gate)
		var distance := _float(g.get("distance", 0.0))
		if distance < travel_distance - GATE_VISIBLE_BEHIND_DISTANCE or distance > travel_distance + GATE_VISIBLE_AHEAD_DISTANCE:
			continue
		var frame := _frame_at(distance)
		var pos: Vector3 = frame["pos"] + frame["up"] * (_float(g.get("y_norm", 0.0)) / tube_half_height) * inner_rz
		var radius := maxf(0.20, _float(g.get("aperture", 0.14)) / tube_half_height * inner_rz)
		var color := _color_by_name(str(g.get("color", "BLUE")))
		var shape := str(g.get("shape", "CIRCLE")).to_lower()
		if shape == "triangle":
			var points := []
			for tri_point: Vector2 in TRIANGLE_POINTS:
				points.append(pos + frame["right"] * tri_point.x * radius * 0.94 + frame["up"] * tri_point.y * radius * 0.94)
			_draw_outline_poly_gate(points, color, GATE_STROKE_WIDTH)
		elif shape == "square":
			var points2 := [
				pos - frame["right"] * radius - frame["up"] * radius,
				pos + frame["right"] * radius - frame["up"] * radius,
				pos + frame["right"] * radius + frame["up"] * radius,
				pos - frame["right"] * radius + frame["up"] * radius,
			]
			_draw_outline_poly_gate(points2, color, GATE_STROKE_WIDTH)
		else:
			_draw_circle_gate(pos, frame["right"], frame["up"], radius, color)


func _draw_circle_gate(pos: Vector3, right: Vector3, up: Vector3, radius: float, color: Color) -> void:
	var segments := 64
	var vertices := PackedVector3Array()
	var indices := PackedInt32Array()
	var inner_radius := maxf(0.03, radius - GATE_STROKE_WIDTH)
	for idx in range(segments):
		var angle := (float(idx) / float(segments)) * TAU
		vertices.append(pos + right * cos(angle) * radius + up * sin(angle) * radius)
		vertices.append(pos + right * cos(angle) * inner_radius + up * sin(angle) * inner_radius)
	for idx in range(segments):
		var next_idx := (idx + 1) % segments
		var outer_a := idx * 2
		var inner_a := outer_a + 1
		var outer_b := next_idx * 2
		var inner_b := outer_b + 1
		indices.append(outer_a)
		indices.append(outer_b)
		indices.append(inner_a)
		indices.append(inner_a)
		indices.append(outer_b)
		indices.append(inner_b)
	_add_gate_mesh("AuditoryOutlineCircleGate", vertices, indices, color)


func _draw_outline_poly_gate(points: Array, color: Color, width: float) -> void:
	if points.size() < 3:
		return
	var normal := _poly_normal(points)
	var vertices := PackedVector3Array()
	var indices := PackedInt32Array()
	for idx in range(points.size()):
		var a: Vector3 = points[idx]
		var b: Vector3 = points[(idx + 1) % points.size()]
		var edge := b - a
		if edge.length() < 0.001:
			continue
		var side := normal.cross(edge.normalized()).normalized() * (width * 0.5)
		var base := vertices.size()
		vertices.append(a + side)
		vertices.append(b + side)
		vertices.append(b - side)
		vertices.append(a - side)
		indices.append(base)
		indices.append(base + 1)
		indices.append(base + 2)
		indices.append(base)
		indices.append(base + 2)
		indices.append(base + 3)
	_add_gate_mesh("AuditoryOutlinePolyGate", vertices, indices, color)


func _poly_normal(points: Array) -> Vector3:
	for idx in range(1, points.size() - 1):
		var a: Vector3 = points[0]
		var b: Vector3 = points[idx]
		var c: Vector3 = points[idx + 1]
		var normal := (b - a).cross(c - a)
		if normal.length() > 0.001:
			return normal.normalized()
	return Vector3.FORWARD


func _add_gate_mesh(name_value: String, vertices: PackedVector3Array, indices: PackedInt32Array, color: Color) -> void:
	if vertices.is_empty() or indices.is_empty():
		return
	var arrays := []
	arrays.resize(Mesh.ARRAY_MAX)
	arrays[Mesh.ARRAY_VERTEX] = vertices
	arrays[Mesh.ARRAY_INDEX] = indices
	var mesh := ArrayMesh.new()
	mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, arrays)
	var node := MeshInstance3D.new()
	node.name = name_value
	node.mesh = mesh
	node.material_override = _gate_material(color)
	gate_root.add_child(node)


func _make_gate_segment(a: Vector3, b: Vector3, color: Color, width: float) -> MeshInstance3D:
	var node := _segment_node(a, b, color, width)
	gate_root.add_child(node)
	return node


func _update_camera(camera: Camera3D) -> void:
	if camera == null:
		return
	camera.projection = Camera3D.PROJECTION_PERSPECTIVE
	camera_distance = lerpf(camera_distance, travel_distance - 7.4, 0.10)
	var frame := _frame_at(camera_distance)
	var target := _frame_at(travel_distance + 8.8)
	var bob := sin(elapsed_s * 4.2) * 0.035
	camera.position = frame["pos"] + frame["up"] * (0.62 + bob) - frame["tangent"] * 0.22
	camera.look_at(target["pos"], frame["up"])


func _update_hud() -> void:
	if hud_label == null:
		return
	var instruction := "Hold tunnel"
	if not active_command.is_empty():
		instruction = str(active_command.get("type", "")).replace("_", " ").to_upper()
	elif not active_beep.is_empty():
		instruction = "TRIGGER"
	var recall := ""
	if not active_recall.is_empty():
		recall = "\nRecall: " + _mask_digits(str(active_recall.get("target", "")).length()) + "   Typed: " + typed_digits
	var review := ""
	if review_mode_enabled:
		review = "\nDev Review: " + _review_hint()
	hud_label.text = "Auditory Capacity | " + phase.to_upper() + " | " + str(int(maxf(0.0, duration_s - elapsed_s))) + "s\n" + segment_label + " | Instruction: " + instruction + "\nBall: " + ball_color_label.to_upper() + "  Gates " + str(metrics["gate_hits"]) + "/" + str(metrics["gate_misses"]) + "  Collisions " + str(metrics["collisions"]) + recall + review


func _review_hint() -> String:
	if not active_command.is_empty():
		var kind := str(active_command.get("type", ""))
		if kind == "change_colour":
			return "set ball " + _speech_color_name(str(active_command.get("payload", "")))
		if kind == "change_number":
			return "press number " + str(active_command.get("payload", ""))
		if kind == "digit_sequence":
			return "recall " + str(active_command.get("payload", ""))
		if kind == "gate_directive":
			var directive := _as_dict(active_command.get("payload", {}))
			return str(directive.get("action", "PASS")).to_lower() + " next " + _speech_color_or_shape_name(str(directive.get("match_value", ""))) + " gate"
	if not active_beep.is_empty():
		return "press trigger"
	if not active_recall.is_empty():
		return "recall " + str(active_recall.get("target", ""))
	var next_gate := _next_unscored_gate()
	if not next_gate.is_empty():
		var decision := "pass" if _gate_should_pass(next_gate) else "avoid"
		return decision + " " + _speech_color_name(str(next_gate.get("color", ""))) + " " + _speech_shape_name(str(next_gate.get("shape", "")))
	return "hold tunnel"


func _next_unscored_gate() -> Dictionary:
	var best := {}
	var best_distance := INF
	for gate in gates:
		var g := _as_dict(gate)
		var distance := _float(g.get("distance", 0.0))
		if bool(g.get("scored", false)) or distance < travel_distance:
			continue
		if distance < best_distance:
			best_distance = distance
			best = g
	return best


func _build_tunnel_anchors() -> void:
	anchors.clear()
	var x := 0.0
	var y := 1.15
	for i in range(-5, 90):
		var d := float(i) * 8.0
		x += rng_tunnel.randf_range(-0.70, 0.70) * curvature_intensity
		y += rng_tunnel.randf_range(-0.32, 0.32) * curvature_intensity
		x = clampf(x, -3.2, 3.2)
		y = clampf(y, 0.45, 2.65)
		anchors.append(Vector3(x, y, -d))


func _path_point(distance: float) -> Vector3:
	var anchor_step := 8.0
	var raw := distance / anchor_step
	var idx := int(floor(raw)) + 5
	var t: float = raw - floor(raw)
	idx = clampi(idx, 1, anchors.size() - 3)
	return _catmull(anchors[idx - 1], anchors[idx], anchors[idx + 1], anchors[idx + 2], t)


func _frame_at(distance: float) -> Dictionary:
	var pos := _path_point(distance)
	var ahead := _path_point(distance + 0.18)
	var behind := _path_point(distance - 0.18)
	var tangent := (ahead - behind).normalized()
	var right := tangent.cross(Vector3.UP).normalized()
	if right.length() < 0.01:
		right = Vector3.RIGHT
	var up := right.cross(tangent).normalized()
	var twist := sin((distance + float(session_seed % 97)) * 0.19) * twist_intensity * 0.55
	var basis := Basis(tangent, twist)
	right = basis * right
	up = basis * up
	return {"pos": pos, "tangent": tangent, "right": right, "up": up}


func _catmull(p0: Vector3, p1: Vector3, p2: Vector3, p3: Vector3, t: float) -> Vector3:
	var t2 := t * t
	var t3 := t2 * t
	return (p1 * 2.0 + (p2 - p0) * t + (p0 * 2.0 - p1 * 5.0 + p2 * 4.0 - p3) * t2 + (-p0 + p1 * 3.0 - p2 * 3.0 + p3) * t3) * 0.5


func _bind_directive_to_next_gate() -> void:
	if active_gate_directive.is_empty():
		return
	var directive := _as_dict(active_gate_directive.get("directive", {}))
	var best_id = null
	var best_distance := 999999.0
	for gate in gates:
		var g := _as_dict(gate)
		if bool(g.get("scored", false)):
			continue
		if _float(g.get("distance", 0.0)) <= travel_distance:
			continue
		if _gate_matches_directive(g, directive) and _float(g.get("distance", 0.0)) < best_distance:
			best_id = int(g.get("id", 0))
			best_distance = _float(g.get("distance", 0.0))
	active_gate_directive["target_gate_id"] = best_id


func _gate_matches_directive(gate: Dictionary, directive: Dictionary) -> bool:
	if str(directive.get("match_kind", "")) == "COLOR":
		return str(gate.get("color", "")).to_upper() == str(directive.get("match_value", "")).to_upper()
	return str(gate.get("shape", "")).to_upper() == str(directive.get("match_value", "")).to_upper()


func _gate_should_pass(gate: Dictionary) -> bool:
	if forbidden_gate_color != null and str(gate.get("color", "")).to_upper() == str(forbidden_gate_color).to_upper():
		return false
	if forbidden_gate_shape != null and str(gate.get("shape", "")).to_upper() == str(forbidden_gate_shape).to_upper():
		return false
	var directive := _as_dict(active_gate_directive.get("directive", {}))
	var target_raw = active_gate_directive.get("target_gate_id", null)
	if target_raw != null and not directive.is_empty() and int(target_raw) == int(gate.get("id", -2)):
		return str(directive.get("action", "PASS")).to_upper() != "AVOID"
	return true


func _prepare_tts() -> bool:
	var voices := DisplayServer.tts_get_voices()
	if voices.is_empty():
		return false
	var english_voice_ids := _english_voice_ids(voices)
	if english_voice_ids.is_empty():
		return false
	primary_voice_id = _voice_id_from_ids(english_voice_ids, 0)
	filler_voice_id = _voice_id_from_ids(english_voice_ids, 1)
	decoy_voice_id = _voice_id_from_ids(english_voice_ids, 2)
	if filler_voice_id == "":
		filler_voice_id = primary_voice_id
	if decoy_voice_id == "":
		decoy_voice_id = primary_voice_id
	return primary_voice_id != ""


func _english_voice_ids(voices: Array) -> Array:
	var result := []
	var unknown_language_ids := []
	for raw in voices:
		var voice_id_value := _voice_id(raw)
		if voice_id_value == "":
			continue
		if _voice_is_english(raw):
			result.append(voice_id_value)
		elif typeof(raw) != TYPE_DICTIONARY:
			unknown_language_ids.append(voice_id_value)
	if result.is_empty():
		result = unknown_language_ids
	return result


func _voice_id(raw) -> String:
	if typeof(raw) == TYPE_DICTIONARY:
		return str(_as_dict(raw).get("id", ""))
	return str(raw)


func _voice_id_from_ids(voice_ids: Array, index: int) -> String:
	if voice_ids.is_empty():
		return ""
	return str(voice_ids[clampi(index, 0, voice_ids.size() - 1)])


func _voice_is_english(raw) -> bool:
	if typeof(raw) == TYPE_DICTIONARY:
		var data := _as_dict(raw)
		var language := str(data.get("language", data.get("locale", ""))).to_lower()
		var name := str(data.get("name", data.get("id", ""))).to_lower()
		if language.begins_with("en"):
			return true
		return name.find("english") >= 0 or name.find(" en-") >= 0 or name.find("united states") >= 0 or name.find("united kingdom") >= 0
	return true


func _speak(text: String, interrupt: bool) -> void:
	_speak_role(text, "primary", interrupt)


func _speak_role(text: String, role: String, interrupt: bool) -> void:
	var voice := primary_voice_id
	var volume := primary_voice_volume
	var pitch := 1.0
	var rate := 1.0
	if role == "filler":
		voice = filler_voice_id if filler_voice_id != "" else primary_voice_id
		volume = filler_voice_volume
		pitch = 0.92
		rate = 0.92
	elif role == "decoy":
		voice = decoy_voice_id if decoy_voice_id != "" else primary_voice_id
		volume = distractor_voice_volume
		pitch = 1.12
		rate = 1.06
	if voice == "":
		return
	DisplayServer.tts_speak(text, voice, int(round(volume)), pitch, rate, _next_tts_utterance_id(), interrupt)


func _next_tts_utterance_id() -> int:
	tts_utterance_id += 1
	return tts_utterance_id


func _prepare_audio(audio: Dictionary) -> void:
	var target_layers := int(audio.get("ambient_layer_target", 1))
	var candidates := ["bg_noise_loop.wav", "bg_conversation_close_loop.wav", "bg_restaurant_loop.wav", "bg_mature_distraction_loop.wav"]
	for i in range(min(3, max(0, target_layers))):
		var player := AudioStreamPlayer.new()
		add_child(player)
		var filename := str(candidates[i % candidates.size()])
		var path := asset_root.path_join(filename)
		if FileAccess.file_exists(path):
			var stream := AudioStreamWAV.load_from_file(path)
			if stream != null:
				player.stream = stream
				player.volume_db = ambient_volume_db + float(i) * ambient_layer_drop_db
				player.play()
		ambient_players.append(player)


func _play_beep() -> void:
	if beep_player == null:
		return
	var stream := AudioStreamGenerator.new()
	stream.mix_rate = 22050.0
	stream.buffer_length = maxf(0.08, beep_duration_s + 0.04)
	beep_player.stream = stream
	beep_player.volume_db = beep_volume_db
	beep_player.play()
	var playback = beep_player.get_stream_playback()
	if playback == null:
		return
	var frames := int(stream.mix_rate * beep_duration_s)
	for i in range(frames):
		var env := minf(1.0, minf(float(i) / maxf(1.0, stream.mix_rate * 0.012), float(frames - i) / maxf(1.0, stream.mix_rate * 0.018)))
		var v := sin(float(i) / stream.mix_rate * TAU * beep_frequency_hz) * 0.45 * maxf(0.0, env)
		playback.push_frame(Vector2(v, v))


func _send_progress() -> void:
	_send("godot_progress", {
		"run_key": run_key,
		"kind": "auditory_capacity",
		"test_code": "auditory_capacity",
		"phase": phase,
		"progress": {
			"phase_elapsed_s": elapsed_s,
			"time_remaining_s": maxf(0.0, duration_s - elapsed_s),
			"segment_label": segment_label,
			"segment_index": segment_index,
			"segment_total": segments.size(),
			"travel_distance": travel_distance,
			"attempted": int(metrics["gate_hits"]) + int(metrics["gate_misses"]) + int(metrics["correct_command_executions"]) + int(metrics["missed_valid_commands"]) + int(metrics["digit_recall_attempts"]),
			"correct": int(metrics["gate_hits"]) + int(metrics["correct_command_executions"]),
			"ball_x": ball_pos.x,
			"ball_y": ball_pos.y,
			"metrics": metrics.duplicate(true),
		}
	})


func _send(command: String, payload: Dictionary) -> void:
	if control_sender.is_valid():
		control_sender.call(command, payload)


func _abort(reason: String, detail: String) -> void:
	active = false
	aborted = true
	completed_run_key = run_key
	_send("godot_error", {"run_key": run_key, "kind": "auditory_capacity", "test_code": "auditory_capacity", "reason": reason, "detail": detail})


func _clear_runtime() -> void:
	set_paused(false)
	if primary_voice_id != "" and DisplayServer.has_method("tts_stop"):
		DisplayServer.tts_stop()
	for child in get_children():
		child.queue_free()
	active = false
	aborted = false
	paused = false
	beep_player = null
	ambient_players.clear()
	gates.clear()
	event_log.clear()
	active_command = {}
	active_gate_directive = {}
	active_recall = {}
	active_beep = {}
	ball_display_color = BALL_IDLE_COLOR
	ball_color_label = "neutral"
	ball_feedback_color = BALL_IDLE_COLOR
	ball_feedback_until_s = 0.0
	primary_voice_id = ""
	filler_voice_id = ""
	decoy_voice_id = ""
	tts_utterance_id = 1
	forbidden_gate_color = null
	forbidden_gate_shape = null
	segments.clear()
	segment_index = 0
	segment_started_at_s = 0.0
	segment_duration_s = 0.0
	segment_label = "Full Mixed"


func _seed_streams() -> void:
	rng_tunnel = StreamRng.new(_stable_seed(session_seed, "tunnel"))
	rng_gates = StreamRng.new(_stable_seed(session_seed, "gates"))
	rng_instructions = StreamRng.new(_stable_seed(session_seed, "instructions"))
	rng_disturbance = StreamRng.new(_stable_seed(session_seed, "disturbance"))
	rng_audio = StreamRng.new(_stable_seed(session_seed, "audio"))


func _stable_seed(seed_value: int, stream: String) -> int:
	var h := int(seed_value) & 0x7fffffff
	for b in stream.to_utf8_buffer():
		h = int((h * 1103515245 + int(b) + 12345) & 0x7fffffff)
	return max(1, h)


func _pick_callsigns(count: int) -> Array:
	var pool := CALLSIGNS.duplicate()
	var picked := []
	for _i in range(max(1, count)):
		if pool.is_empty():
			break
		var idx := rng_instructions.randi_range(0, pool.size() - 1)
		picked.append(pool[idx])
		pool.remove_at(idx)
	return picked


func _random_digits(length: int) -> String:
	var text := ""
	for _i in range(max(1, length)):
		text += str(rng_instructions.randi_range(0, 9))
	return text


func _spaced_digits(value: String) -> String:
	var parts := []
	for i in range(value.length()):
		parts.append(value.substr(i, 1))
	return " ".join(parts)


func _mask_digits(length: int) -> String:
	var text := ""
	for _i in range(max(0, length)):
		text += "*"
	return text


func _score_digits(expected: String, response: String) -> float:
	if expected == "":
		return 0.0
	if response == expected:
		return 1.0
	var matches := 0
	for i in range(expected.length()):
		if i < response.length() and expected.substr(i, 1) == response.substr(i, 1):
			matches += 1
	var ratio := float(matches) / float(expected.length())
	var penalty = minf(0.45, absf(float(response.length() - expected.length())) / float(expected.length()) * 0.45)
	return clampf(ratio - penalty, 0.0, 1.0)


func _jitter(base: float, spread: float) -> float:
	return maxf(0.20, base * (1.0 + rng_instructions.randf_range(-spread, spread) * (0.35 + difficulty)))


func _channel_active(name: String) -> bool:
	if active_channels.is_empty():
		return true
	return active_channels.has(name)


func _make_sphere(name_value: String, pos: Vector3, radius: float, color: Color) -> MeshInstance3D:
	var mesh := SphereMesh.new()
	mesh.radial_segments = 16
	mesh.rings = 8
	var node := MeshInstance3D.new()
	node.name = name_value
	node.mesh = mesh
	node.position = pos
	node.scale = Vector3(radius, radius, radius)
	node.material_override = _material(color)
	add_child(node)
	return node


func _make_segment(a: Vector3, b: Vector3, color: Color, width: float) -> MeshInstance3D:
	var node := _segment_node(a, b, color, width)
	add_child(node)
	return node


func _segment_node(a: Vector3, b: Vector3, color: Color, width: float) -> MeshInstance3D:
	var mesh := BoxMesh.new()
	mesh.size = Vector3(1.0, 1.0, 1.0)
	var node := MeshInstance3D.new()
	node.name = "Segment"
	node.mesh = mesh
	node.position = (a + b) * 0.5
	node.scale = Vector3(width, width, maxf(0.01, a.distance_to(b) * 0.5))
	node.material_override = _material(color)
	var dir := (b - a).normalized()
	var up := Vector3.UP
	if absf(dir.dot(up)) > 0.92:
		up = Vector3.FORWARD
	node.look_at(b, up)
	return node


func _tube_material(color: Color) -> StandardMaterial3D:
	var mat := StandardMaterial3D.new()
	mat.albedo_color = color
	mat.roughness = 1.0
	mat.metallic = 0.0
	if color.a < 0.999:
		mat.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	mat.cull_mode = BaseMaterial3D.CULL_DISABLED
	return mat


func _material(color: Color) -> StandardMaterial3D:
	var mat := StandardMaterial3D.new()
	mat.albedo_color = color
	mat.roughness = 1.0
	mat.metallic = 0.0
	if color.a < 0.999:
		mat.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	return mat


func _gate_material(color: Color) -> StandardMaterial3D:
	var mat := _material(color)
	mat.cull_mode = BaseMaterial3D.CULL_DISABLED
	mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	return mat


func _color_by_name(name_value: String) -> Color:
	var token := name_value.to_lower()
	if token.find("red") >= 0:
		return RED_COLOR
	if token.find("blue") >= 0:
		return BLUE_COLOR
	if token.find("yellow") >= 0 or token.find("amber") >= 0:
		return YELLOW_COLOR
	if token.find("black") >= 0:
		return BLACK_COLOR
	if token.find("white") >= 0:
		return WHITE_COLOR
	return Color(0.55, 0.70, 0.80, 1.0)


func _speech_color_name(name_value: String) -> String:
	var token := name_value.to_lower()
	if token.find("red") >= 0:
		return "red"
	if token.find("blue") >= 0:
		return "blue"
	if token.find("yellow") >= 0 or token.find("amber") >= 0:
		return "yellow"
	return token.replace("_", " ")


func _speech_shape_name(name_value: String) -> String:
	var token := name_value.to_lower()
	if token.find("circle") >= 0:
		return "circle"
	if token.find("triangle") >= 0:
		return "triangle"
	if token.find("square") >= 0:
		return "square"
	return token.replace("_", " ")


func _speech_color_or_shape_name(name_value: String) -> String:
	var token := name_value.to_lower()
	if COLORS.has(name_value.to_upper()):
		return _speech_color_name(name_value)
	if SHAPES.has(name_value.to_upper()):
		return _speech_shape_name(name_value)
	return token.replace("_", " ")


func _speech_callsign(name_value: String) -> String:
	return name_value.to_lower()


func _speech_callsign_list(values: Array) -> String:
	var spoken := []
	for value in values:
		spoken.append(_speech_callsign(str(value)))
	return ", ".join(spoken)


func _as_dict(value) -> Dictionary:
	if typeof(value) == TYPE_DICTIONARY:
		return value
	return {}


func _as_array(value) -> Array:
	if typeof(value) == TYPE_ARRAY:
		return value
	return []


func _float(value) -> float:
	if typeof(value) == TYPE_FLOAT or typeof(value) == TYPE_INT:
		return float(value)
	return float(str(value))

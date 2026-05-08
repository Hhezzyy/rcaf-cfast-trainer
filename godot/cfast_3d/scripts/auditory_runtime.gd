extends Node3D

const COLORS := ["RED", "BLUE", "YELLOW"]
const SHAPES := ["CIRCLE", "TRIANGLE", "SQUARE"]
const CALLSIGNS := ["EAGLE", "RAVEN", "FALCON", "VIPER", "COBRA", "TALON", "MOOSE", "LANCER", "SABER", "NOVA"]
const TRIANGLE_POINTS := [
	Vector2(0.0, 1.22),
	Vector2(-1.056, -0.61),
	Vector2(1.056, -0.61),
]
const PASS_COLOR := Color(0.28, 0.92, 0.52, 1.0)
const ERROR_COLOR := Color(1.0, 0.18, 0.16, 1.0)
const TUBE_COLOR := Color(0.012, 0.030, 0.044, 0.90)
const TUBE_EDGE_COLOR := Color(0.10, 0.25, 0.32, 0.88)
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
var tube_half_width := 0.72
var tube_half_height := 0.52
var inner_rx := 2.24
var inner_rz := 1.64
var ball_radius := 0.11
var control_gain := 1.24
var disturbance_gain := 0.52
var curvature_intensity := 0.88
var twist_intensity := 0.68
var response_window_s := 2.4
var sequence_response_s := 7.0
var gate_interval_s := 3.85
var command_interval_s := 3.8
var beep_interval_s := 36.0
var asset_root := ""
var assigned_callsigns: Array = []
var active_channels: Array = []
var rng_tunnel: StreamRng
var rng_gates: StreamRng
var rng_instructions: StreamRng
var rng_disturbance: StreamRng
var rng_audio: StreamRng
var anchors: Array = []
var gates: Array = []
var next_gate_id := 1
var next_gate_at_s := 1.0
var next_command_at_s := 2.0
var next_beep_at_s := 7.0
var active_command := {}
var active_gate_directive := {}
var active_recall := {}
var active_beep := {}
var forbidden_gate_color = null
var forbidden_gate_shape = null
var memory_buffer := ""
var typed_digits := ""
var ball_pos := Vector2.ZERO
var ball_vel := Vector2.ZERO
var disturbance := Vector2.ZERO
var disturbance_until_s := 0.0
var metrics := {}
var event_log: Array = []
var tube_mesh_node: MeshInstance3D
var ball_node: MeshInstance3D
var gate_root: Node3D
var hud_layer: CanvasLayer
var hud_label: Label
var beep_player: AudioStreamPlayer
var ambient_players: Array = []
var voice_id := ""
var last_tube_rebuild_distance := -1000.0


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
	ball_pos = Vector2.ZERO
	ball_vel = Vector2.ZERO
	disturbance = Vector2.ZERO
	disturbance_until_s = 0.0
	last_tube_rebuild_distance = -1000.0
	duration_s = maxf(1.0, _float(spec.get("duration_s", 60.0)))
	asset_root = str(spec.get("asset_root", ""))
	_seed_streams()
	assigned_callsigns = _as_array(spec.get("assigned_callsigns", []))
	active_channels = _as_array(spec.get("active_channels", []))
	if assigned_callsigns.is_empty():
		assigned_callsigns = _pick_callsigns(3)
	var cfg := _as_dict(spec.get("config", {}))
	tick_hz = maxf(30.0, _float(cfg.get("tick_hz", 120.0)))
	control_gain = _float(cfg.get("control_gain", 1.24))
	disturbance_gain = _float(cfg.get("disturbance_gain", 0.52))
	tube_half_width = maxf(0.20, _float(cfg.get("tube_half_width", 0.72)))
	tube_half_height = maxf(0.18, _float(cfg.get("tube_half_height", 0.52)))
	inner_rx = maxf(0.60, _float(cfg.get("inner_rx", 2.24)))
	inner_rz = maxf(0.42, _float(cfg.get("inner_rz", 1.64)))
	ball_radius = maxf(0.05, _float(cfg.get("ball_radius", 0.11)))
	curvature_intensity = maxf(0.0, _float(cfg.get("tunnel_curvature_intensity", 0.88)))
	twist_intensity = maxf(0.0, _float(cfg.get("tunnel_twist_intensity", 0.68)))
	gate_interval_s = maxf(0.65, _float(cfg.get("gate_interval_s", 3.85)) * (1.0 - difficulty * 0.42))
	command_interval_s = maxf(0.75, 1.0 / maxf(0.05, _float(cfg.get("command_rate", 0.26))))
	beep_interval_s = maxf(8.0, _float(cfg.get("beep_interval_s", 36.0)) * (1.0 - difficulty * 0.35))
	response_window_s = maxf(0.45, _float(cfg.get("response_window_seconds", 2.4)))
	sequence_response_s = maxf(response_window_s, _float(cfg.get("sequence_response_s", 7.0)))
	travel_speed = maxf(
		3.2,
		_float(cfg.get("gate_speed_norm_per_s", 0.33)) * 12.0 * (0.92 + difficulty * 0.28)
	)
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
	next_gate_at_s = 0.8
	next_command_at_s = maxf(1.2, command_interval_s * 0.55)
	next_beep_at_s = maxf(5.0, beep_interval_s * 0.35)
	_build_nodes()
	if not _prepare_tts():
		_abort("tts_unavailable", "Godot/platform TTS is unavailable for Auditory Capacity.")
		return
	_prepare_audio(_as_dict(spec.get("audio", {})))
	active = true
	aborted = false
	_speak("Assigned call signs. " + ", ".join(assigned_callsigns) + ".", true)
	_send("godot_ready", {"run_key": run_key, "phase": phase, "kind": "auditory_capacity", "test_code": str(spec.get("test_code", "auditory_capacity"))})
	_rebuild_scene(true)


func update_runtime(delta: float, camera: Camera3D) -> void:
	if not active or aborted:
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
	if not active or aborted:
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
	if key == KEY_E:
		_submit_color("YELLOW")
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


func _step(dt: float) -> void:
	elapsed_s += dt
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
	if Input.is_key_pressed(KEY_A) or Input.is_key_pressed(KEY_LEFT):
		input_vec.x -= 1.0
	if Input.is_key_pressed(KEY_D) or Input.is_key_pressed(KEY_RIGHT):
		input_vec.x += 1.0
	if Input.is_key_pressed(KEY_W) or Input.is_key_pressed(KEY_UP):
		input_vec.y += 1.0
	if Input.is_key_pressed(KEY_S) or Input.is_key_pressed(KEY_DOWN):
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
	ball_pos += ball_vel * dt
	var max_x := tube_half_width - ball_radius
	var max_y := tube_half_height - ball_radius
	var outside := absf(ball_pos.x) > max_x or absf(ball_pos.y) > max_y
	if outside:
		metrics["collisions"] = int(metrics["collisions"]) + 1
		ball_pos.x = clampf(ball_pos.x, -max_x, max_x)
		ball_pos.y = clampf(ball_pos.y, -max_y, max_y)
		ball_vel *= -0.18


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


func _update_schedules() -> void:
	if elapsed_s >= next_gate_at_s and _channel_active("gates"):
		_spawn_gate()
		next_gate_at_s = elapsed_s + _jitter(gate_interval_s, 0.28)
	if elapsed_s >= next_command_at_s:
		_spawn_instruction()
		next_command_at_s = elapsed_s + _jitter(command_interval_s, 0.34)
	if elapsed_s >= next_beep_at_s and _channel_active("trigger"):
		_spawn_beep()
		next_beep_at_s = elapsed_s + _jitter(beep_interval_s, 0.32)


func _spawn_gate() -> void:
	var lane_options := [-0.54, -0.36, -0.18, 0.0, 0.18, 0.36, 0.54]
	var y_norm := float(rng_gates.choice(lane_options)) + rng_gates.randf_range(-0.045, 0.045)
	var color := str(rng_gates.choice(COLORS))
	var shape := str(rng_gates.choice(SHAPES))
	var aperture := rng_gates.randf_range(0.11, 0.25 - difficulty * 0.045)
	gates.append({
		"id": next_gate_id,
		"distance": travel_distance + 18.0,
		"y_norm": clampf(y_norm, -0.58, 0.58),
		"color": color,
		"shape": shape,
		"aperture": maxf(0.09, aperture),
		"scored": false,
		"flash_color": "",
		"flash_until": 0.0,
	})
	next_gate_id += 1
	_bind_directive_to_next_gate()


func _update_gates() -> void:
	var kept := []
	for gate in gates:
		var g := _as_dict(gate)
		if not bool(g.get("scored", false)) and _float(g.get("distance", 0.0)) <= travel_distance:
			_score_gate(g)
		if _float(g.get("distance", 0.0)) >= travel_distance - 5.5 or elapsed_s < _float(g.get("flash_until", 0.0)):
			kept.append(g)
	gates = kept


func _score_gate(gate: Dictionary) -> void:
	var inside := absf(ball_pos.y - _float(gate.get("y_norm", 0.0))) <= _float(gate.get("aperture", 0.15))
	var should_pass := _gate_should_pass(gate)
	var correct := (inside == should_pass)
	if correct:
		metrics["gate_hits"] = int(metrics["gate_hits"]) + 1
		metrics["points"] = float(metrics["points"]) + 1.0
		gate["flash_color"] = "PASS"
		gate["flash_until"] = elapsed_s + 0.22
	else:
		metrics["gate_misses"] = int(metrics["gate_misses"]) + 1
		if inside and not should_pass:
			metrics["forbidden_gate_hits"] = int(metrics["forbidden_gate_hits"]) + 1
		gate["flash_color"] = "ERROR"
		gate["flash_until"] = elapsed_s + 0.34
	gate["scored"] = true
	if _as_dict(active_gate_directive).get("target_gate_id", -1) == int(gate.get("id", -2)):
		active_gate_directive = {}
	var pilot_action := "PASS" if inside else "SKIP"
	var expected := "PASS" if should_pass else "AVOID"
	_record_event("gate", expected, str(gate.get("color", "")) + "/" + str(gate.get("shape", "")) + "/" + pilot_action, correct, 1.0 if correct else 0.0)


func _spawn_instruction() -> void:
	if not _channel_active("state_commands") and not _channel_active("gate_directives") and not _channel_active("digit_recall"):
		return
	if not active_command.is_empty():
		metrics["missed_valid_commands"] = int(metrics["missed_valid_commands"]) + 1
	var roll := rng_instructions.randf()
	var command_type := "change_colour"
	var payload = "RED"
	if _channel_active("gate_directives") and roll > 0.58:
		command_type = "gate_directive"
		var match_kind := "COLOR" if rng_instructions.randf() < 0.58 else "SHAPE"
		payload = {
			"action": "AVOID" if rng_instructions.randf() < 0.58 else "PASS",
			"match_kind": match_kind,
			"match_value": str(rng_instructions.choice(COLORS if match_kind == "COLOR" else SHAPES)),
		}
	elif _channel_active("digit_recall") and roll > 0.38:
		command_type = "digit_sequence"
		var length := rng_instructions.randi_range(4, 6 + int(round(difficulty * 4.0)))
		payload = _random_digits(length)
	elif roll > 0.20:
		command_type = "change_number"
		payload = rng_instructions.randi_range(1, 9)
	else:
		command_type = "change_colour"
		payload = str(rng_instructions.choice(COLORS))
	var callsign := str(rng_instructions.choice(assigned_callsigns))
	active_command = {
		"id": event_log.size() + 1,
		"type": command_type,
		"payload": payload,
		"callsign": callsign,
		"expires_at": elapsed_s + (sequence_response_s if command_type == "digit_sequence" else response_window_s),
	}
	_activate_command(active_command)


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
		_speak(callsign + ". " + ("Avoid" if str(directive.get("action", "")) == "AVOID" else "Take") + " the next " + str(directive.get("match_value", "")) + " gate.", false)
	elif kind == "digit_sequence":
		memory_buffer = str(payload)
		active_recall = {
			"target": memory_buffer,
			"expires_at": elapsed_s + sequence_response_s,
		}
		typed_digits = ""
		_speak(callsign + ". Remember digits " + _spaced_digits(memory_buffer) + ".", false)
	elif kind == "change_number":
		_speak(callsign + ". Set number " + str(payload) + ".", false)
	else:
		_speak(callsign + ". Change colour to " + str(payload) + ".", false)


func _spawn_beep() -> void:
	active_beep = {"expires_at": elapsed_s + response_window_s, "responded": false}
	_play_beep()
	_speak(str(rng_instructions.choice(assigned_callsigns)) + ". Press trigger now.", false)


func _submit_color(color: String) -> void:
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
	_send("godot_complete", {"run_key": run_key, "phase": result["phase"], "kind": "auditory_capacity", "test_code": "auditory_capacity", "result": result})


func _rebuild_scene(force: bool) -> void:
	if not force and absf(travel_distance - last_tube_rebuild_distance) < 0.10:
		_update_ball_node()
		_rebuild_gates()
		return
	last_tube_rebuild_distance = travel_distance
	_rebuild_tube_mesh()
	_update_ball_node()
	_rebuild_gates()


func _build_nodes() -> void:
	tube_mesh_node = MeshInstance3D.new()
	tube_mesh_node.name = "AuditorySolidTunnel"
	add_child(tube_mesh_node)
	ball_node = _make_sphere("AuditoryBall", Vector3.ZERO, 0.28, BALL_IDLE_COLOR)
	gate_root = Node3D.new()
	gate_root.name = "AuditoryGates"
	add_child(gate_root)
	hud_layer = CanvasLayer.new()
	add_child(hud_layer)
	hud_label = Label.new()
	hud_label.position = Vector2(14, 88)
	hud_label.size = Vector2(620, 150)
	hud_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	hud_label.add_theme_font_size_override("font_size", 18)
	hud_label.add_theme_color_override("font_color", WHITE_COLOR)
	hud_layer.add_child(hud_label)
	beep_player = AudioStreamPlayer.new()
	add_child(beep_player)


func _rebuild_tube_mesh() -> void:
	var ring_count := 30
	var ring_segments := 18
	var start_d := travel_distance - 10.5
	var step_d := 1.05
	var vertices := PackedVector3Array()
	var indices := PackedInt32Array()
	for r in range(ring_count):
		var distance := start_d + float(r) * step_d
		var frame := _frame_at(distance)
		for s in range(ring_segments):
			var angle := (float(s) / float(ring_segments)) * TAU
			var point = frame["pos"] + frame["right"] * cos(angle) * inner_rx + frame["up"] * sin(angle) * inner_rz
			vertices.append(point)
	for r in range(ring_count - 1):
		for s in range(ring_segments):
			var a := r * ring_segments + s
			var b := r * ring_segments + ((s + 1) % ring_segments)
			var c := (r + 1) * ring_segments + s
			var d := (r + 1) * ring_segments + ((s + 1) % ring_segments)
			indices.append(a)
			indices.append(c)
			indices.append(b)
			indices.append(b)
			indices.append(c)
			indices.append(d)
	var arrays := []
	arrays.resize(Mesh.ARRAY_MAX)
	arrays[Mesh.ARRAY_VERTEX] = vertices
	arrays[Mesh.ARRAY_INDEX] = indices
	var mesh := ArrayMesh.new()
	mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, arrays)
	tube_mesh_node.mesh = mesh
	tube_mesh_node.material_override = _tube_material(TUBE_COLOR)
	for child in get_children():
		if str(child.name).begins_with("AuditoryRail") or str(child.name).begins_with("AuditoryRing"):
			child.queue_free()
	for rail_angle in [0.0, PI * 0.5, PI, PI * 1.5]:
		var previous := Vector3.ZERO
		var have_previous := false
		for r in range(0, ring_count, 2):
			var frame2 := _frame_at(start_d + float(r) * step_d)
			var p: Vector3 = frame2["pos"] + frame2["right"] * cos(rail_angle) * inner_rx + frame2["up"] * sin(rail_angle) * inner_rz
			if have_previous:
				var seg := _make_segment(previous, p, TUBE_EDGE_COLOR, 0.022)
				seg.name = "AuditoryRail"
			previous = p
			have_previous = true
	for r in range(3, ring_count, 4):
		_draw_tunnel_ring(_frame_at(start_d + float(r) * step_d), ring_segments)


func _draw_tunnel_ring(frame: Dictionary, ring_segments: int) -> void:
	var previous: Vector3 = frame["pos"] + frame["right"] * inner_rx
	for idx in range(1, ring_segments + 1):
		var angle := (float(idx) / float(ring_segments)) * TAU
		var current: Vector3 = frame["pos"] + frame["right"] * cos(angle) * inner_rx + frame["up"] * sin(angle) * inner_rz
		var seg := _make_segment(previous, current, TUBE_EDGE_COLOR.lightened(0.12), 0.018)
		seg.name = "AuditoryRing"
		previous = current


func _update_ball_node() -> void:
	if ball_node == null:
		return
	var frame := _frame_at(travel_distance)
	var local_x := (ball_pos.x / tube_half_width) * maxf(0.05, inner_rx - 0.28)
	var local_y := (ball_pos.y / tube_half_height) * maxf(0.05, inner_rz - 0.28)
	ball_node.position = frame["pos"] + frame["right"] * local_x + frame["up"] * local_y
	ball_node.material_override = _material(BALL_IDLE_COLOR if active_beep.is_empty() else YELLOW_COLOR)
	ball_node.look_at(_frame_at(travel_distance + 1.0)["pos"], frame["up"])
	ball_node.rotate_object_local(Vector3.RIGHT, ball_roll_angle)


func _rebuild_gates() -> void:
	if gate_root == null:
		return
	for child in gate_root.get_children():
		child.queue_free()
	for gate in gates:
		var g := _as_dict(gate)
		var distance := _float(g.get("distance", 0.0))
		if distance < travel_distance - 5.0 or distance > travel_distance + 20.0:
			continue
		var frame := _frame_at(distance)
		var pos: Vector3 = frame["pos"] + frame["up"] * (_float(g.get("y_norm", 0.0)) / tube_half_height) * inner_rz
		var radius := maxf(0.20, _float(g.get("aperture", 0.14)) / tube_half_height * inner_rz)
		var color := _color_by_name(str(g.get("color", "BLUE")))
		if str(g.get("flash_color", "")) == "PASS":
			color = PASS_COLOR
		elif str(g.get("flash_color", "")) == "ERROR":
			color = ERROR_COLOR
		var shape := str(g.get("shape", "CIRCLE")).to_lower()
		if shape == "triangle":
			var points := []
			for tri_point: Vector2 in TRIANGLE_POINTS:
				points.append(pos + frame["right"] * tri_point.x * radius * 0.94 + frame["up"] * tri_point.y * radius * 0.94)
			_draw_poly_gate(points, color, 0.055)
		elif shape == "square":
			_draw_poly_gate([
				pos - frame["right"] * radius - frame["up"] * radius,
				pos + frame["right"] * radius - frame["up"] * radius,
				pos + frame["right"] * radius + frame["up"] * radius,
				pos - frame["right"] * radius + frame["up"] * radius,
			], color, 0.055)
		else:
			_draw_circle_gate(pos, frame["right"], frame["up"], radius, color)


func _draw_circle_gate(pos: Vector3, right: Vector3, up: Vector3, radius: float, color: Color) -> void:
	var previous := pos + right * radius
	for idx in range(1, 25):
		var angle := (float(idx) / 24.0) * TAU
		var current := pos + right * cos(angle) * radius + up * sin(angle) * radius
		_make_gate_segment(previous, current, color, 0.055)
		previous = current


func _draw_poly_gate(points: Array, color: Color, width: float) -> void:
	for idx in range(points.size()):
		_make_gate_segment(points[idx], points[(idx + 1) % points.size()], color, width)


func _make_gate_segment(a: Vector3, b: Vector3, color: Color, width: float) -> void:
	var node := _segment_node(a, b, color, width)
	gate_root.add_child(node)


func _update_camera(camera: Camera3D) -> void:
	if camera == null:
		return
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
	hud_label.text = "Auditory Capacity | " + phase.to_upper() + " | " + str(int(maxf(0.0, duration_s - elapsed_s))) + "s\n" + "Instruction: " + instruction + "\nGates " + str(metrics["gate_hits"]) + "/" + str(metrics["gate_misses"]) + "  Collisions " + str(metrics["collisions"]) + recall


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
	var first_voice = voices[0]
	if typeof(first_voice) == TYPE_DICTIONARY:
		voice_id = str(_as_dict(first_voice).get("id", ""))
	else:
		voice_id = str(first_voice)
	return voice_id != ""


func _speak(text: String, interrupt: bool) -> void:
	if voice_id == "":
		return
	DisplayServer.tts_speak(text, voice_id, 55, 1.0, 1.0, int(event_log.size() + 1), interrupt)


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
				player.volume_db = -20.0 + float(i) * -3.0
				player.play()
		ambient_players.append(player)


func _play_beep() -> void:
	if beep_player == null:
		return
	var stream := AudioStreamGenerator.new()
	stream.mix_rate = 22050.0
	stream.buffer_length = 0.16
	beep_player.stream = stream
	beep_player.volume_db = -6.0
	beep_player.play()
	var playback = beep_player.get_stream_playback()
	if playback == null:
		return
	var frames := int(stream.mix_rate * 0.12)
	for i in range(frames):
		var v := sin(float(i) / stream.mix_rate * TAU * 1120.0) * 0.45
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
	for child in get_children():
		child.queue_free()
	active = false
	aborted = false
	gates.clear()
	event_log.clear()
	active_command = {}
	active_gate_directive = {}
	active_recall = {}
	active_beep = {}
	forbidden_gate_color = null
	forbidden_gate_shape = null


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
	node.look_at(b, Vector3.UP)
	return node


func _tube_material(color: Color) -> StandardMaterial3D:
	var mat := StandardMaterial3D.new()
	mat.albedo_color = color
	mat.roughness = 1.0
	mat.metallic = 0.0
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

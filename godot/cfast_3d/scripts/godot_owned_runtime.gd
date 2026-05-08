extends Node3D

const WHITE_COLOR := Color(0.92, 0.96, 0.98, 1.0)
const GREEN_COLOR := Color(0.22, 0.76, 0.34, 1.0)
const BLUE_COLOR := Color(0.12, 0.42, 0.95, 1.0)
const RED_COLOR := Color(0.95, 0.18, 0.15, 1.0)
const AMBER_COLOR := Color(0.95, 0.66, 0.16, 1.0)
const GRID_COLOR := Color(0.45, 0.58, 0.66, 1.0)
const FLOOR_COLOR := Color(0.17, 0.23, 0.25, 1.0)
const SKY_COLOR := Color(0.36, 0.55, 0.78, 1.0)
const TRACE_GRID_HALF_X := 4
const TRACE_GRID_HALF_Z := 4
const TRACE_GRID_MAX_Y := 3
const TRACE_GRID_SCALE := 0.76
const TRACE_GRID_ALT_SCALE := 0.68
const TRACE_GRID_ORIGIN := Vector3(0.0, 0.72, -6.2)
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
var event_log: Array = []
var scene_root: Node3D
var hud_layer: CanvasLayer
var hud_label: Label


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
	if _is_trace_kind():
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
	if not active:
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
	if not active or event.echo or not event.pressed:
		return false
	var key := event.keycode
	if key == KEY_KP_ENTER:
		return true
	if key == KEY_SPACE or key == KEY_ENTER or key == KEY_KP_PERIOD:
		_sample_score("submit")
		return true
	return false


func _step(dt: float) -> void:
	elapsed_s += dt
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
	_make_box(Vector3(0.0, 2.1, -8.8), Vector3(6.2, 2.2, 0.04), SKY_COLOR)
	var index := 0
	for raw_track in trace_tracks:
		var track := _as_dict(raw_track)
		_draw_trace_track_history(track)
		var color := RED_COLOR if index == 0 else _trace_track_color(index)
		var size := 1.08 if index == 0 else 0.84
		_make_aircraft(_trace_track_world(track), color, _trace_track_direction(track), size)
		index += 1
	_make_reticle(_screen_to_world(player_pos, -4.35), BLUE_COLOR)


func _is_trace_kind() -> bool:
	return kind == "trace_test_1" or kind == "trace_test_2"


func _init_trace_tracks() -> void:
	trace_tracks.clear()
	var count := 5 if kind == "trace_test_2" else 3
	if test_code.find("pressure") >= 0:
		count += 1
	for i in range(count):
		var local := RandomNumberGenerator.new()
		local.seed = int(session_seed + _kind_salt(test_code) + (i + 1) * 7919)
		var cell := Vector3(
			float(local.randi_range(-TRACE_GRID_HALF_X + 1, TRACE_GRID_HALF_X - 1)),
			float(local.randi_range(0, TRACE_GRID_MAX_Y)),
			float(local.randi_range(-TRACE_GRID_HALF_Z + 1, TRACE_GRID_HALF_Z - 1))
		)
		var dir_idx := _trace_initial_dir(local, cell)
		var move_s := maxf(0.38, 0.86 - difficulty * 0.22 + float(i % 3) * 0.05)
		var pause_s := maxf(0.18, 0.38 - difficulty * 0.08 + float(i % 2) * 0.04)
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
				if next_dir_idx != old_dir_idx:
					trace_turn_count += 1
				if forced:
					trace_forced_edge_turn_count += 1
					track["forced_turns"] = int(track.get("forced_turns", 0)) + 1
				track["prev_dir_idx"] = old_dir_idx
				track["dir_idx"] = next_dir_idx
				track["phase"] = "pause"
				track["phase_t"] = 0.0
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


func _trace_forward_blocked(cell: Vector3, dir_idx: int) -> bool:
	return not _trace_cell_in_bounds(cell + _trace_dir(dir_idx))


func _trace_cell_in_bounds(cell: Vector3) -> bool:
	return (
		int(cell.x) >= -TRACE_GRID_HALF_X
		and int(cell.x) <= TRACE_GRID_HALF_X
		and int(cell.y) >= 0
		and int(cell.y) <= TRACE_GRID_MAX_Y
		and int(cell.z) >= -TRACE_GRID_HALF_Z
		and int(cell.z) <= TRACE_GRID_HALF_Z
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
	return TRACE_GRID_ORIGIN + Vector3(cell.x * TRACE_GRID_SCALE, cell.y * TRACE_GRID_ALT_SCALE, cell.z * TRACE_GRID_SCALE)


func _sync_trace_target() -> void:
	if trace_tracks.is_empty():
		target_pos = Vector2.ZERO
		return
	var world := _trace_track_world(_as_dict(trace_tracks[0]))
	var max_x := maxf(1.0, float(TRACE_GRID_HALF_X) * TRACE_GRID_SCALE)
	var max_y := maxf(1.0, float(TRACE_GRID_MAX_Y) * TRACE_GRID_ALT_SCALE)
	target_pos = Vector2(
		clampf(world.x / max_x, -1.0, 1.0),
		clampf(((world.y - TRACE_GRID_ORIGIN.y) / max_y) * 1.65 - 0.82, -1.0, 1.0)
	)


func _draw_trace_grid() -> void:
	for level in range(TRACE_GRID_MAX_Y + 1):
		var y := TRACE_GRID_ORIGIN.y + float(level) * TRACE_GRID_ALT_SCALE
		for x in range(-TRACE_GRID_HALF_X, TRACE_GRID_HALF_X + 1):
			var a := _trace_cell_to_world(Vector3(float(x), float(level), -TRACE_GRID_HALF_Z))
			var b := _trace_cell_to_world(Vector3(float(x), float(level), TRACE_GRID_HALF_Z))
			_make_segment(a, b, GRID_COLOR.darkened(0.24), 0.010)
		for z in range(-TRACE_GRID_HALF_Z, TRACE_GRID_HALF_Z + 1):
			var a := _trace_cell_to_world(Vector3(-TRACE_GRID_HALF_X, float(level), float(z)))
			var b := _trace_cell_to_world(Vector3(TRACE_GRID_HALF_X, float(level), float(z)))
			_make_segment(a, b, GRID_COLOR.darkened(0.24), 0.010)
	_make_box(Vector3(0.0, TRACE_GRID_ORIGIN.y - 0.01, TRACE_GRID_ORIGIN.z), Vector3(float(TRACE_GRID_HALF_X) * TRACE_GRID_SCALE, 0.006, float(TRACE_GRID_HALF_Z) * TRACE_GRID_SCALE), Color(0.08, 0.12, 0.14, 0.16))
	for x in range(-TRACE_GRID_HALF_X, TRACE_GRID_HALF_X + 1):
		for z in range(-TRACE_GRID_HALF_Z, TRACE_GRID_HALF_Z + 1):
			var low := _trace_cell_to_world(Vector3(float(x), 0.0, float(z)))
			var high := _trace_cell_to_world(Vector3(float(x), float(TRACE_GRID_MAX_Y), float(z)))
			_make_segment(low, high, GRID_COLOR.darkened(0.38), 0.006)


func _draw_trace_track_history(track: Dictionary) -> void:
	var history: Array = track.get("history", [])
	var previous := Vector3.ZERO
	var has_previous := false
	for raw_cell in history:
		var cell := _trace_vec3(raw_cell, Vector3.ZERO)
		var world := _trace_cell_to_world(cell)
		_make_sphere(world, 0.052, WHITE_COLOR.darkened(0.35))
		if has_previous:
			_make_segment(previous, world, WHITE_COLOR.darkened(0.42), 0.018)
		previous = world
		has_previous = true


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


func _smoothstep01(value: float) -> float:
	var t := clampf(value, 0.0, 1.0)
	return t * t * (3.0 - (2.0 * t))


func _update_camera(camera: Camera3D) -> void:
	if camera == null:
		return
	var pos := Vector3(0.0, 3.25, 8.6)
	var target := Vector3(0.0, 1.0, -5.2)
	if kind == "spatial_integration":
		pos = Vector3(0.0, 7.2, 7.4)
		target = Vector3(0.0, 0.0, -5.0)
	elif kind == "trace_test_1" or kind == "trace_test_2":
		pos = Vector3(0.0, 4.8, 11.2)
		target = Vector3(0.0, 1.6, -7.2)
	camera.position = camera.position.lerp(pos, 0.18)
	camera.look_at(target, Vector3.UP)


func _update_hud() -> void:
	if hud_label == null:
		return
	var rem := maxf(0.0, duration_s - elapsed_s)
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
	hud_label.size = Vector2(920, 48)
	hud_label.add_theme_font_size_override("font_size", 18)
	hud_label.add_theme_color_override("font_color", WHITE_COLOR)
	hud_layer.add_child(hud_label)


func _clear_runtime() -> void:
	for child in get_children():
		child.queue_free()
	active = false
	scene_root = null
	hud_layer = null
	hud_label = null


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

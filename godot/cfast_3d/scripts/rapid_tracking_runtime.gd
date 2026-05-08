extends Node3D

const WHITE_COLOR := Color(0.92, 0.96, 0.98, 1.0)
const GREEN_COLOR := Color(0.22, 0.82, 0.34, 1.0)
const BLUE_COLOR := Color(0.12, 0.42, 0.95, 1.0)
const RED_COLOR := Color(0.95, 0.18, 0.15, 1.0)
const AMBER_COLOR := Color(0.95, 0.66, 0.16, 1.0)
const FIELD_GREEN := Color(0.23, 0.34, 0.16, 1.0)
const FIELD_LIGHT := Color(0.36, 0.45, 0.19, 1.0)
const FIELD_DARK := Color(0.15, 0.24, 0.13, 1.0)
const ROAD_COLOR := Color(0.24, 0.23, 0.22, 1.0)
const HOUSE_WALL := Color(0.66, 0.63, 0.55, 1.0)
const HOUSE_ROOF := Color(0.47, 0.18, 0.13, 1.0)
const TREE_TRUNK := Color(0.28, 0.16, 0.08, 1.0)
const TREE_TOP := Color(0.16, 0.37, 0.18, 1.0)
const SHADOW_COLOR := Color(0.04, 0.05, 0.04, 0.45)
const TARGET_KINDS := ["soldier", "building", "truck", "helicopter", "jet"]

var control_sender: Callable
var active := false
var run_key := ""
var completed_run_key := ""
var kind := "rapid_tracking"
var test_code := "rapid_tracking"
var phase := "scored"
var mode := "standard"
var session_seed := 1
var difficulty := 0.5
var duration_s := 180.0
var elapsed_s := 0.0
var tick_accum := 0.0
var progress_accum := 0.0
var fixed_hz := 60.0
var scene_hash := 0
var target_schedule_hash := 0

var aim_angles := Vector2.ZERO
var aim_velocity := Vector2.ZERO
var target_screen_pos := Vector2.ZERO
var target_screen_valid := false
var active_target_pos := Vector3.ZERO
var active_target_kind := "truck"
var active_target_index := -1
var active_target_moving := true
var active_target_obscured := false

var on_target_radius := 0.11
var capture_box_half_width := 0.14
var capture_box_half_height := 0.12
var capture_cooldown_s := 0.42
var capture_feedback_until_s := 0.0
var last_capture_at_s := -999.0
var capture_feedback := ""
var capture_flash_hit := false
var sample_event_accum := 0.0

var tracking_sample_count := 0
var tracking_on_target_count := 0
var total_error := 0.0
var total_sq_error := 0.0
var tracking_score := 0.0
var tracking_max_score := 0.0
var on_target_s := 0.0
var moving_target_s := 0.0
var obscured_time_s := 0.0
var visible_time_s := 0.0
var capture_attempts := 0
var capture_hits := 0
var capture_points := 0.0
var capture_max_points := 0.0
var overshoot_count := 0
var reversal_count := 0
var previous_error_delta := Vector2.ZERO
var previous_input_sign_x := 0
var event_log: Array = []

var scene_root: Node3D
var static_root: Node3D
var moving_root: Node3D
var target_root: Node3D
var hud_layer: CanvasLayer
var hud_label: Label
var reticle_h: ColorRect
var reticle_v: ColorRect
var capture_top: ColorRect
var capture_bottom: ColorRect
var capture_left: ColorRect
var capture_right: ColorRect
var target_marker_h: ColorRect
var target_marker_v: ColorRect
var material_cache := {}
var moving_assets := []
var target_schedule := []
var cluster_centers := []
var house_positions := []


func start(spec: Dictionary, sender: Callable) -> void:
	control_sender = sender
	var next_key := str(spec.get("run_key", ""))
	if active and next_key == run_key:
		return
	if next_key != "" and next_key == completed_run_key:
		return
	_clear_runtime()
	run_key = next_key
	kind = "rapid_tracking"
	test_code = str(spec.get("test_code", "rapid_tracking"))
	phase = str(spec.get("phase", "scored")).to_lower()
	mode = str(spec.get("mode", "standard")).to_lower()
	session_seed = int(max(1, int(spec.get("session_seed", spec.get("seed", 1)))))
	difficulty = clampf(_float(spec.get("difficulty", 0.5)), 0.0, 1.0)
	duration_s = maxf(1.0, _float(spec.get("duration_s", 180.0)))
	var cfg := _as_dict(spec.get("config", {}))
	on_target_radius = maxf(0.055, _float(cfg.get("on_target_radius", 0.115 - difficulty * 0.025)))
	capture_box_half_width = maxf(0.075, _float(cfg.get("capture_box_half_width", 0.155 - difficulty * 0.025)))
	capture_box_half_height = maxf(0.065, _float(cfg.get("capture_box_half_height", 0.135 - difficulty * 0.018)))
	capture_cooldown_s = maxf(0.12, _float(cfg.get("capture_cooldown_s", 0.46 - difficulty * 0.12)))
	elapsed_s = 0.0
	tick_accum = 0.0
	progress_accum = 0.0
	aim_angles = Vector2.ZERO
	aim_velocity = Vector2.ZERO
	target_screen_pos = Vector2.ZERO
	target_screen_valid = false
	active_target_index = -1
	event_log.clear()
	_reset_score()
	_build_nodes()
	_generate_scene()
	_generate_target_schedule()
	_activate_target(0)
	active = true
	_send("godot_ready", {
		"run_key": run_key,
		"phase": phase,
		"kind": kind,
		"test_code": test_code,
		"scene_hash": scene_hash,
		"target_schedule_hash": target_schedule_hash,
	})


func update_runtime(delta: float, camera: Camera3D) -> void:
	if not active:
		return
	var dt: float = minf(maxf(delta, 0.0), 0.05)
	_update_camera(camera, dt)
	tick_accum += dt
	var fixed_dt := 1.0 / fixed_hz
	while tick_accum >= fixed_dt:
		tick_accum -= fixed_dt
		_step(fixed_dt)
	_update_live_objects(camera)
	_update_hud(camera)
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
		_capture("key")
		return true
	return false


func rapid_tracking_runtime_marker() -> bool:
	return true


func _reset_score() -> void:
	tracking_sample_count = 0
	tracking_on_target_count = 0
	total_error = 0.0
	total_sq_error = 0.0
	tracking_score = 0.0
	tracking_max_score = 0.0
	on_target_s = 0.0
	moving_target_s = 0.0
	obscured_time_s = 0.0
	visible_time_s = 0.0
	capture_attempts = 0
	capture_hits = 0
	capture_points = 0.0
	capture_max_points = 0.0
	overshoot_count = 0
	reversal_count = 0
	previous_error_delta = Vector2.ZERO
	previous_input_sign_x = 0
	sample_event_accum = 0.0
	last_capture_at_s = -999.0
	capture_feedback = ""
	capture_flash_hit = false


func _step(dt: float) -> void:
	elapsed_s += dt
	var next_target_index := _target_index_for_time(elapsed_s)
	if next_target_index != active_target_index:
		_activate_target(next_target_index)
	var input_vec := _input_vector()
	var gain := 1.75 + difficulty * 0.85
	aim_velocity += input_vec * gain * dt
	aim_velocity *= pow(0.08, dt)
	aim_angles += aim_velocity * dt
	aim_angles.x = wrapf(aim_angles.x, -PI, PI)
	aim_angles.y = clampf(aim_angles.y, deg_to_rad(-86.0), deg_to_rad(86.0))
	var input_sign_x := signf(input_vec.x)
	if input_sign_x != 0 and previous_input_sign_x != 0 and input_sign_x != previous_input_sign_x:
		reversal_count += 1
	if input_sign_x != 0:
		previous_input_sign_x = int(input_sign_x)
	_score_tracking(dt)
	if Input.is_joy_button_pressed(_primary_joypad(), JOY_BUTTON_A):
		_capture("joystick")
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
	var joy := _primary_joypad()
	if joy >= 0:
		out.x += Input.get_joy_axis(joy, JOY_AXIS_LEFT_X)
		out.y -= Input.get_joy_axis(joy, JOY_AXIS_LEFT_Y)
	if out.length() > 1.0:
		out = out.normalized()
	return out


func _primary_joypad() -> int:
	var pads := Input.get_connected_joypads()
	if pads.size() <= 0:
		return -1
	return int(pads[0])


func _score_tracking(dt: float) -> void:
	if not target_screen_valid:
		return
	var error_delta := Vector2.ZERO - target_screen_pos
	var error := error_delta.length()
	var threshold := on_target_radius
	var quality := clampf(1.0 - (error / maxf(0.001, threshold * 2.65)), 0.0, 1.0)
	tracking_sample_count += 1
	total_error += error
	total_sq_error += error * error
	tracking_score += quality * dt
	tracking_max_score += dt
	if error <= threshold:
		tracking_on_target_count += 1
		on_target_s += dt
	if active_target_moving:
		moving_target_s += dt
	if active_target_obscured:
		obscured_time_s += dt
	else:
		visible_time_s += dt
	if previous_error_delta.length() > 0.001 and error_delta.dot(previous_error_delta) < -0.006:
		overshoot_count += 1
	previous_error_delta = error_delta
	sample_event_accum += dt
	if sample_event_accum >= 1.0:
		sample_event_accum = 0.0
		_append_event("tracking_sample", error <= threshold, quality, 1.0, {
			"distance": error,
			"threshold": threshold,
			"target_kind": active_target_kind,
			"target_index": active_target_index,
			"target_visible": not active_target_obscured,
			"aim_yaw_rad": aim_angles.x,
			"aim_pitch_rad": aim_angles.y,
		}, false)


func _capture(source: String) -> void:
	if elapsed_s - last_capture_at_s < capture_cooldown_s:
		return
	last_capture_at_s = elapsed_s
	capture_attempts += 1
	capture_max_points += 2.0
	var delta := Vector2.ZERO - target_screen_pos
	var hit := target_screen_valid and not active_target_obscured and absf(delta.x) <= capture_box_half_width and absf(delta.y) <= capture_box_half_height
	var points := 0.0
	if hit:
		capture_hits += 1
		points = 2.0
		capture_points += points
		capture_feedback = "CAPTURE"
		capture_flash_hit = true
	else:
		capture_feedback = "MISS"
		capture_flash_hit = false
	capture_feedback_until_s = elapsed_s + 0.38
	_append_event("capture", hit, points, 2.0, {
		"source": source,
		"target_kind": active_target_kind,
		"target_index": active_target_index,
		"distance_x": delta.x,
		"distance_y": delta.y,
		"aim_yaw_rad": aim_angles.x,
		"aim_pitch_rad": aim_angles.y,
		"target_visible": target_screen_valid and not active_target_obscured,
	}, true)


func _target_index_for_time(time_s: float) -> int:
	if target_schedule.is_empty():
		return -1
	var idx := int(floor(time_s / _segment_duration_s()))
	return clampi(idx, 0, target_schedule.size() - 1)


func _segment_duration_s() -> float:
	var token := test_code.to_lower()
	if token.find("pressure") >= 0 or token.find("air_speed") >= 0:
		return 8.0
	if token.find("lock_anchor") >= 0:
		return 16.0
	return 13.0 - difficulty * 4.0


func _activate_target(index: int) -> void:
	if index < 0 or index >= target_schedule.size():
		return
	active_target_index = index
	var item := _as_dict(target_schedule[index])
	active_target_kind = str(item.get("kind", "truck"))
	active_target_moving = bool(item.get("moving", true))
	_clear_children(target_root)
	_build_target_model(active_target_kind)
	_append_event("target_handoff", true, 0.0, 0.0, {
		"target_kind": active_target_kind,
		"target_index": active_target_index,
	}, false)


func _active_target_world_position() -> Vector3:
	if active_target_index < 0 or active_target_index >= target_schedule.size():
		return Vector3.ZERO
	var item := _as_dict(target_schedule[active_target_index])
	var base := _vec3(item.get("base", {}), Vector3.ZERO)
	var phase_offset := _float(item.get("phase", 0.0))
	var radius := _float(item.get("radius", 4.0))
	var speed := _float(item.get("speed", 0.25))
	var local_t := maxf(0.0, elapsed_s - float(active_target_index) * _segment_duration_s())
	var token := active_target_kind.to_lower()
	if token == "building":
		return base + Vector3(0.0, 1.1, 0.0)
	if token == "soldier":
		return base + Vector3(sin(local_t * speed + phase_offset) * radius * 0.55, 0.58, cos(local_t * speed * 0.7 + phase_offset) * radius * 0.42)
	if token == "helicopter":
		return base + Vector3(sin(local_t * speed + phase_offset) * radius, 6.0 + sin(local_t * 0.9 + phase_offset) * 1.1, cos(local_t * speed * 0.82 + phase_offset) * radius)
	if token == "jet":
		var sweep := fmod(local_t * speed * 6.0 + phase_offset, 28.0) - 14.0
		return base + Vector3(sweep, 9.0 + sin(local_t * 0.8) * 1.8, sin(local_t * speed + phase_offset) * radius * 0.5)
	return base + Vector3(sin(local_t * speed + phase_offset) * radius, 0.68, cos(local_t * speed * 0.72 + phase_offset) * radius * 0.65)


func _is_active_target_obscured() -> bool:
	var token := test_code.to_lower()
	var pressure := 0.18 + difficulty * 0.18
	if token.find("terrain") >= 0:
		pressure += 0.18
	if active_target_kind == "building" or active_target_kind == "jet":
		pressure *= 0.35
	var wave := sin(elapsed_s * (0.55 + difficulty * 0.18) + float(active_target_index) * 1.73)
	return wave > 1.0 - pressure


func _update_live_objects(camera: Camera3D) -> void:
	active_target_pos = _active_target_world_position()
	active_target_obscured = _is_active_target_obscured()
	target_root.position = active_target_pos
	var look := active_target_pos + Vector3(sin(elapsed_s), 0.0, cos(elapsed_s))
	if active_target_pos.distance_to(look) > 0.01:
		target_root.look_at(look, Vector3.UP)
	var tint := GREEN_COLOR
	if active_target_obscured:
		tint = Color(0.48, 0.62, 0.52, 1.0)
	_apply_material_recursive(target_root, tint, true)
	for asset in moving_assets:
		_update_moving_asset(_as_dict(asset))
	_update_target_screen(camera)


func _update_target_screen(camera: Camera3D) -> void:
	target_screen_valid = false
	if camera == null:
		return
	if camera.is_position_behind(active_target_pos):
		return
	var pixel := camera.unproject_position(active_target_pos + Vector3(0.0, 0.6, 0.0))
	var size := get_viewport().get_visible_rect().size
	if size.x <= 1.0 or size.y <= 1.0:
		return
	target_screen_pos = Vector2((pixel.x / size.x) * 2.0 - 1.0, 1.0 - (pixel.y / size.y) * 2.0)
	target_screen_valid = target_screen_pos.x >= -1.08 and target_screen_pos.x <= 1.08 and target_screen_pos.y >= -1.08 and target_screen_pos.y <= 1.08


func _update_moving_asset(asset: Dictionary) -> void:
	var node: Variant = asset.get("node")
	if not (node is Node3D):
		return
	var root := node as Node3D
	var center := _vec3(asset.get("center", {}), Vector3.ZERO)
	var radius := _float(asset.get("radius", 5.0))
	var speed := _float(asset.get("speed", 0.4))
	var phase_offset := _float(asset.get("phase", 0.0))
	var altitude := _float(asset.get("altitude", 0.5))
	var route := str(asset.get("route", "loop"))
	var pos := center
	if route == "air":
		pos += Vector3(sin(elapsed_s * speed + phase_offset) * radius, altitude + sin(elapsed_s * speed * 0.7 + phase_offset) * 1.2, cos(elapsed_s * speed * 0.8 + phase_offset) * radius)
	else:
		pos += Vector3(sin(elapsed_s * speed + phase_offset) * radius, altitude, cos(elapsed_s * speed * 0.75 + phase_offset) * radius * 0.55)
	var prev := root.position
	root.position = pos
	var diff := pos - prev
	if diff.length() > 0.01:
		root.look_at(pos + diff, Vector3.UP)


func _update_camera(camera: Camera3D, dt: float) -> void:
	if camera == null:
		return
	var dir := -1.0 if _seed_unit("camera:direction") < 0.5 else 1.0
	var radius := 25.0 + difficulty * 6.0 + _seed_unit("camera:radius") * 4.0
	var rate := (0.105 + difficulty * 0.035) * dir
	var phase_offset := _seed_unit("camera:phase") * TAU
	var angle := phase_offset + elapsed_s * rate
	var base_focus := Vector3(
		sin(elapsed_s * 0.13 + phase_offset) * 2.6,
		1.8 + sin(elapsed_s * 0.17 + phase_offset) * 0.5,
		cos(elapsed_s * 0.11 + phase_offset) * 2.2
	)
	var turbulence_strength := 0.42 + difficulty * 0.72
	var turbulence := Vector3(
		sin(elapsed_s * 1.37 + phase_offset * 0.7) * turbulence_strength,
		sin(elapsed_s * 1.91 + phase_offset * 1.3) * turbulence_strength * 0.34,
		cos(elapsed_s * 1.53 + phase_offset * 0.5) * turbulence_strength
	)
	base_focus += turbulence
	var desired := base_focus + Vector3(
		cos(angle) * radius + sin(elapsed_s * 0.61 + phase_offset) * turbulence_strength * 1.8,
		14.0 + difficulty * 4.0 + sin(elapsed_s * 0.21) * 1.2 + cos(elapsed_s * 0.79 + phase_offset) * turbulence_strength,
		sin(angle) * radius + cos(elapsed_s * 0.57 + phase_offset) * turbulence_strength * 1.8
	)
	var forward := (base_focus - desired).normalized()
	var yawed_forward := (Basis(Vector3.UP, aim_angles.x) * forward).normalized()
	var right := yawed_forward.cross(Vector3.UP).normalized()
	if right.length() < 0.001:
		right = Vector3.RIGHT
	var look_dir := (Basis(right, aim_angles.y) * yawed_forward).normalized()
	var focus := desired + look_dir * maxf(8.0, radius)
	if capture_feedback_until_s > elapsed_s and capture_flash_hit:
		camera.fov = lerpf(camera.fov, 48.0, 0.16)
	else:
		camera.fov = lerpf(camera.fov, 58.0, 0.08)
	var blend := clampf(1.0 - pow(0.015, dt), 0.02, 0.24)
	camera.position = camera.position.lerp(desired, blend)
	camera.look_at(focus, Vector3.UP)


func _update_hud(camera: Camera3D) -> void:
	if hud_label == null:
		return
	var remaining := maxf(0.0, duration_s - elapsed_s)
	var hit_text := "HIT" if capture_flash_hit else "MISS"
	var feedback := ""
	if capture_feedback_until_s > elapsed_s:
		feedback = " | " + hit_text
	var on_ratio := 0.0 if tracking_sample_count <= 0 else float(tracking_on_target_count) / float(tracking_sample_count)
	hud_label.text = "Rapid Tracking | " + phase + " | " + str(int(ceil(remaining))) + "s | On target " + str(int(round(on_ratio * 100.0))) + "% | Captures " + str(capture_hits) + "/" + str(capture_attempts) + feedback
	_update_overlay(camera)


func _update_overlay(camera: Camera3D) -> void:
	var size := get_viewport().get_visible_rect().size
	if size.x <= 1.0 or size.y <= 1.0:
		return
	var center := _norm_to_pixel(Vector2.ZERO, size)
	var color := AMBER_COLOR if _reticle_in_capture_box() else WHITE_COLOR
	if capture_feedback_until_s > elapsed_s:
		color = GREEN_COLOR if capture_flash_hit else RED_COLOR
	_set_rect(reticle_h, center + Vector2(-18.0, -1.5), Vector2(36.0, 3.0), color)
	_set_rect(reticle_v, center + Vector2(-1.5, -18.0), Vector2(3.0, 36.0), color)
	var half := Vector2(capture_box_half_width * 0.5 * size.x, capture_box_half_height * 0.5 * size.y)
	_set_rect(capture_top, center + Vector2(-half.x, -half.y), Vector2(half.x * 2.0, 2.0), color.darkened(0.08))
	_set_rect(capture_bottom, center + Vector2(-half.x, half.y), Vector2(half.x * 2.0, 2.0), color.darkened(0.08))
	_set_rect(capture_left, center + Vector2(-half.x, -half.y), Vector2(2.0, half.y * 2.0), color.darkened(0.08))
	_set_rect(capture_right, center + Vector2(half.x, -half.y), Vector2(2.0, half.y * 2.0), color.darkened(0.08))
	var marker_color := GREEN_COLOR if target_screen_valid and not active_target_obscured else Color(0.55, 0.64, 0.58, 1.0)
	var target_px := _norm_to_pixel(target_screen_pos, size)
	target_marker_h.visible = target_screen_valid
	target_marker_v.visible = target_screen_valid
	_set_rect(target_marker_h, target_px + Vector2(-13.0, -1.5), Vector2(26.0, 3.0), marker_color)
	_set_rect(target_marker_v, target_px + Vector2(-1.5, -13.0), Vector2(3.0, 26.0), marker_color)


func _reticle_in_capture_box() -> bool:
	if not target_screen_valid or active_target_obscured:
		return false
	var delta := Vector2.ZERO - target_screen_pos
	return absf(delta.x) <= capture_box_half_width and absf(delta.y) <= capture_box_half_height


func _norm_to_pixel(value: Vector2, size: Vector2) -> Vector2:
	return Vector2((value.x * 0.5 + 0.5) * size.x, (0.5 - value.y * 0.5) * size.y)


func _set_rect(rect: ColorRect, pos: Vector2, size_value: Vector2, color: Color) -> void:
	if rect == null:
		return
	rect.position = pos
	rect.size = size_value
	rect.color = color


func _send_progress() -> void:
	_send("godot_progress", {
		"run_key": run_key,
		"kind": kind,
		"test_code": test_code,
		"phase": phase,
		"progress": {
			"elapsed_s": elapsed_s,
			"time_remaining_s": maxf(0.0, duration_s - elapsed_s),
			"attempted": _attempted_count(),
			"correct": _correct_count(),
			"score": tracking_score + capture_points,
			"capture_attempts": capture_attempts,
			"capture_hits": capture_hits,
			"active_target_kind": active_target_kind,
			"aim_yaw_rad": aim_angles.x,
			"aim_pitch_rad": aim_angles.y,
			"scene_hash": scene_hash,
			"target_schedule_hash": target_schedule_hash,
		},
	})


func _complete() -> void:
	active = false
	completed_run_key = run_key
	var attempted := _attempted_count()
	var correct := _correct_count()
	var total_score := tracking_score + capture_points
	var max_score := tracking_max_score + capture_max_points
	var summary := {
		"attempted": attempted,
		"correct": correct,
		"accuracy": 0.0 if attempted <= 0 else float(correct) / float(attempted),
		"duration_s": elapsed_s,
		"throughput_per_min": float(attempted) / maxf(1.0, elapsed_s) * 60.0,
		"total_score": total_score,
		"max_score": max_score,
		"score_ratio": 0.0 if max_score <= 0.0 else total_score / max_score,
	}
	var mean_error := 0.0 if tracking_sample_count <= 0 else total_error / float(tracking_sample_count)
	var rms_error := 0.0 if tracking_sample_count <= 0 else sqrt(total_sq_error / float(tracking_sample_count))
	var metrics := {
		"renderer_backend": "godot_4",
		"godot_authority": "1",
		"godot_kind": kind,
		"godot_test_code": test_code,
		"godot_mode": mode,
		"scene_theme": "rural_house_clusters",
		"scene_hash": scene_hash,
		"target_schedule_hash": target_schedule_hash,
		"mean_tracking_error": mean_error,
		"rms_tracking_error": rms_error,
		"on_target_s": on_target_s,
		"on_target_ratio": 0.0 if elapsed_s <= 0.0 else on_target_s / elapsed_s,
		"moving_target_s": moving_target_s,
		"moving_target_ratio": 0.0 if elapsed_s <= 0.0 else moving_target_s / elapsed_s,
		"obscured_time_s": obscured_time_s,
		"visible_time_s": visible_time_s,
		"capture_points": capture_points,
		"capture_hits": capture_hits,
		"capture_attempts": capture_attempts,
		"capture_accuracy": 0.0 if capture_attempts <= 0 else float(capture_hits) / float(capture_attempts),
		"capture_max_points": capture_max_points,
		"capture_score_ratio": 0.0 if capture_max_points <= 0.0 else capture_points / capture_max_points,
		"overshoot_count": overshoot_count,
		"reversal_count": reversal_count,
	}
	var result := {
		"run_key": run_key,
		"kind": kind,
		"test_code": test_code,
		"phase": "results",
		"summary": summary,
		"metrics": metrics,
		"events": event_log.slice(max(0, event_log.size() - 320), event_log.size()),
	}
	_send("godot_complete", {"run_key": run_key, "phase": "results", "kind": kind, "test_code": test_code, "result": result})


func _attempted_count() -> int:
	return tracking_sample_count + capture_attempts


func _correct_count() -> int:
	return tracking_on_target_count + capture_hits


func _append_event(event_kind: String, is_correct: bool, score: float, max_score: float, extra: Dictionary, scored: bool) -> void:
	var evt := {
		"family": "rapid_tracking",
		"kind": event_kind,
		"phase": phase,
		"item_index": event_log.size(),
		"is_scored": scored or phase == "scored",
		"is_correct": is_correct,
		"is_timeout": false,
		"response_time_ms": int(round(elapsed_s * 1000.0)),
		"score": score,
		"max_score": max_score,
		"occurred_at_ms": int(round(elapsed_s * 1000.0)),
		"prompt": active_target_kind,
		"expected": "track",
		"response": "camera_aim",
		"extra": extra,
	}
	event_log.append(evt)
	if event_kind == "capture" or event_kind == "target_handoff":
		_send("godot_event", {"run_key": run_key, "kind": kind, "test_code": test_code, "event": evt})


func _build_nodes() -> void:
	scene_root = Node3D.new()
	scene_root.name = "RapidTrackingScene"
	add_child(scene_root)
	static_root = Node3D.new()
	static_root.name = "StaticRuralScene"
	scene_root.add_child(static_root)
	moving_root = Node3D.new()
	moving_root.name = "MovingDistractors"
	scene_root.add_child(moving_root)
	target_root = Node3D.new()
	target_root.name = "ActiveTarget"
	scene_root.add_child(target_root)
	hud_layer = CanvasLayer.new()
	hud_layer.layer = 4
	add_child(hud_layer)
	hud_label = Label.new()
	hud_label.position = Vector2(12, 88)
	hud_label.size = Vector2(920, 48)
	hud_label.add_theme_font_size_override("font_size", 18)
	hud_label.add_theme_color_override("font_color", WHITE_COLOR)
	hud_layer.add_child(hud_label)
	reticle_h = _make_color_rect("ReticleH")
	reticle_v = _make_color_rect("ReticleV")
	capture_top = _make_color_rect("CaptureTop")
	capture_bottom = _make_color_rect("CaptureBottom")
	capture_left = _make_color_rect("CaptureLeft")
	capture_right = _make_color_rect("CaptureRight")
	target_marker_h = _make_color_rect("TargetMarkerH")
	target_marker_v = _make_color_rect("TargetMarkerV")


func _make_color_rect(name_value: String) -> ColorRect:
	var rect := ColorRect.new()
	rect.name = name_value
	rect.mouse_filter = Control.MOUSE_FILTER_IGNORE
	rect.color = WHITE_COLOR
	hud_layer.add_child(rect)
	return rect


func _clear_runtime() -> void:
	for child in get_children():
		child.queue_free()
	active = false
	scene_root = null
	static_root = null
	moving_root = null
	target_root = null
	hud_layer = null
	hud_label = null
	moving_assets.clear()
	target_schedule.clear()
	cluster_centers.clear()
	house_positions.clear()


func _generate_scene() -> void:
	var terrain_rng := _rng_for("terrain")
	var village_rng := _rng_for("houses")
	var prop_rng := _rng_for("props")
	_make_box_child(static_root, "Ground", Vector3(0.0, -0.12, 0.0), Vector3(44.0, 0.12, 44.0), FIELD_GREEN)
	for x in range(-3, 4):
		for z in range(-3, 4):
			var tint := FIELD_LIGHT.lerp(FIELD_DARK, terrain_rng.randf())
			var pos := Vector3(float(x) * 12.0 + terrain_rng.randf_range(-1.2, 1.2), -0.09, float(z) * 12.0 + terrain_rng.randf_range(-1.2, 1.2))
			_make_box_child(static_root, "Field", pos, Vector3(5.6, 0.035, 5.6), tint)
	_make_road(Vector3(-35.0, 0.0, -6.0), Vector3(35.0, 0.0, 5.0), 1.05)
	_make_road(Vector3(-12.0, 0.0, -34.0), Vector3(8.0, 0.0, 34.0), 0.86)
	var cluster_count := 4 + int(round(difficulty * 2.0))
	for i in range(cluster_count):
		var angle := (float(i) / float(cluster_count)) * TAU + village_rng.randf_range(-0.36, 0.36)
		var radius := village_rng.randf_range(5.0, 16.0)
		var center := Vector3(cos(angle) * radius, 0.0, sin(angle) * radius)
		cluster_centers.append(center)
		_make_house_cluster(village_rng, center, i)
	_make_trees(prop_rng, 78 + int(round(difficulty * 28.0)))
	_make_fences(prop_rng, 22)
	_make_parked_vehicles(prop_rng, 16 + int(round(difficulty * 8.0)))
	_make_landmarks(prop_rng)
	_make_moving_distractors()
	scene_hash = _hash_scene()


func _make_house_cluster(rng: RandomNumberGenerator, center: Vector3, cluster_index: int) -> void:
	var count := 6 + int(rng.randi_range(0, 4))
	for i in range(count):
		var offset := Vector3(rng.randf_range(-3.4, 3.4), 0.0, rng.randf_range(-3.2, 3.2))
		var pos := center + offset
		house_positions.append(pos)
		var scale := Vector3(rng.randf_range(0.55, 1.05), rng.randf_range(0.42, 0.82), rng.randf_range(0.55, 1.12))
		var root := Node3D.new()
		root.name = "HouseCluster" + str(cluster_index) + "House" + str(i)
		root.position = pos
		root.rotation_degrees.y = rng.randf_range(-32.0, 32.0)
		static_root.add_child(root)
		_make_box_child(root, "Walls", Vector3(0.0, scale.y, 0.0), scale, HOUSE_WALL.lerp(Color(0.82, 0.78, 0.66, 1.0), rng.randf() * 0.35))
		_make_box_child(root, "Roof", Vector3(0.0, scale.y * 2.04, 0.0), Vector3(scale.x * 1.12, 0.20, scale.z * 1.12), HOUSE_ROOF.lerp(Color(0.28, 0.24, 0.21, 1.0), rng.randf() * 0.42))
		if rng.randf() < 0.45:
			_make_box_child(root, "Shed", Vector3(scale.x * 1.45, 0.28, -scale.z * 0.2), Vector3(0.34, 0.28, 0.42), Color(0.45, 0.42, 0.35, 1.0))


func _make_road(a: Vector3, b: Vector3, half_width: float) -> void:
	var center := (a + b) * 0.5
	var length := Vector2(a.x - b.x, a.z - b.z).length()
	var node := _make_box_child(static_root, "Road", center + Vector3(0.0, -0.055, 0.0), Vector3(half_width, 0.04, length * 0.5), ROAD_COLOR)
	node.look_at(b, Vector3.UP)


func _make_trees(rng: RandomNumberGenerator, count: int) -> void:
	for i in range(count):
		var pos := Vector3(rng.randf_range(-34.0, 34.0), 0.0, rng.randf_range(-34.0, 34.0))
		if pos.length() < 3.5:
			pos += pos.normalized() * 4.0
		var root := Node3D.new()
		root.name = "Tree" + str(i)
		root.position = pos
		static_root.add_child(root)
		var height := rng.randf_range(0.65, 1.35)
		_make_box_child(root, "Trunk", Vector3(0.0, height * 0.34, 0.0), Vector3(0.08, height * 0.34, 0.08), TREE_TRUNK)
		_make_sphere_child(root, "Canopy", Vector3(0.0, height * 0.86, 0.0), rng.randf_range(0.34, 0.58), TREE_TOP.lerp(Color(0.24, 0.46, 0.19, 1.0), rng.randf() * 0.4))


func _make_fences(rng: RandomNumberGenerator, count: int) -> void:
	for i in range(count):
		var pos := Vector3(rng.randf_range(-30.0, 30.0), 0.15, rng.randf_range(-30.0, 30.0))
		var length := rng.randf_range(1.5, 4.5)
		var node := _make_box_child(static_root, "Fence", pos, Vector3(0.045, 0.15, length), Color(0.48, 0.38, 0.23, 1.0))
		node.rotation_degrees.y = rng.randf_range(0.0, 180.0)


func _make_parked_vehicles(rng: RandomNumberGenerator, count: int) -> void:
	for i in range(count):
		var pos := Vector3(rng.randf_range(-22.0, 22.0), 0.34, rng.randf_range(-22.0, 22.0))
		var root := Node3D.new()
		root.name = "ParkedVehicle" + str(i)
		root.position = pos
		root.rotation_degrees.y = rng.randf_range(0.0, 360.0)
		static_root.add_child(root)
		_build_vehicle(root, Color(0.26, 0.33, 0.30, 1.0).lerp(AMBER_COLOR.darkened(0.35), rng.randf() * 0.3), 0.82)


func _make_landmarks(rng: RandomNumberGenerator) -> void:
	var tower := Node3D.new()
	tower.name = "WaterTower"
	tower.position = Vector3(rng.randf_range(-14.0, 14.0), 0.0, rng.randf_range(-16.0, 16.0))
	static_root.add_child(tower)
	_make_box_child(tower, "Leg", Vector3(-0.25, 1.2, -0.25), Vector3(0.05, 1.2, 0.05), Color(0.38, 0.39, 0.36, 1.0))
	_make_box_child(tower, "Leg", Vector3(0.25, 1.2, -0.25), Vector3(0.05, 1.2, 0.05), Color(0.38, 0.39, 0.36, 1.0))
	_make_box_child(tower, "Leg", Vector3(-0.25, 1.2, 0.25), Vector3(0.05, 1.2, 0.05), Color(0.38, 0.39, 0.36, 1.0))
	_make_box_child(tower, "Leg", Vector3(0.25, 1.2, 0.25), Vector3(0.05, 1.2, 0.05), Color(0.38, 0.39, 0.36, 1.0))
	_make_sphere_child(tower, "Tank", Vector3(0.0, 2.65, 0.0), 0.52, Color(0.55, 0.57, 0.56, 1.0))


func _make_moving_distractors() -> void:
	var rng := _rng_for("moving")
	var count := 12 + int(round(difficulty * 10.0))
	for i in range(count):
		var route := "air" if i % 5 == 0 else "ground"
		var root := Node3D.new()
		root.name = "MovingDistractor" + str(i)
		moving_root.add_child(root)
		if route == "air":
			if i % 2 == 0:
				_build_aircraft(root, BLUE_COLOR.darkened(0.25), 0.72)
			else:
				_build_helicopter(root, Color(0.30, 0.42, 0.48, 1.0), 0.82)
		else:
			_build_vehicle(root, Color(0.20, 0.28, 0.25, 1.0).lerp(RED_COLOR.darkened(0.35), rng.randf() * 0.35), 0.76)
		moving_assets.append({
			"node": root,
			"center": _vec3_dict(Vector3(rng.randf_range(-12.0, 12.0), 0.0, rng.randf_range(-12.0, 12.0))),
			"radius": rng.randf_range(4.0, 13.0),
			"speed": rng.randf_range(0.18, 0.45 + difficulty * 0.32),
			"phase": rng.randf() * TAU,
			"altitude": rng.randf_range(4.5, 8.5) if route == "air" else 0.48,
			"route": route,
		})


func _generate_target_schedule() -> void:
	var rng := _rng_for("targets")
	var count := int(ceil(duration_s / _segment_duration_s())) + 2
	target_schedule.clear()
	for i in range(count):
		var target_kind := _target_kind_for_slot(i, rng)
		var base := _base_for_target(target_kind, rng)
		var moving := target_kind != "building"
		target_schedule.append({
			"kind": target_kind,
			"base": _vec3_dict(base),
			"moving": moving,
			"phase": rng.randf() * TAU,
			"radius": rng.randf_range(3.2, 8.8 + difficulty * 3.0),
			"speed": rng.randf_range(0.22, 0.48 + difficulty * 0.36),
		})
	target_schedule_hash = _hash_target_schedule()


func _target_kind_for_slot(index: int, rng: RandomNumberGenerator) -> String:
	var token := test_code.to_lower()
	if token.find("lock_anchor") >= 0:
		return "building" if index % 3 == 0 else "truck"
	if token.find("building") >= 0:
		return "building" if index % 2 == 0 else "soldier"
	if token.find("air_speed") >= 0:
		return "jet" if index % 2 == 0 else "helicopter"
	if token.find("ground") >= 0:
		return "truck" if index % 2 == 0 else "soldier"
	if token.find("terrain") >= 0:
		return "soldier" if index % 3 != 0 else "truck"
	return str(TARGET_KINDS[int(rng.randi_range(0, TARGET_KINDS.size() - 1))])


func _base_for_target(target_kind: String, rng: RandomNumberGenerator) -> Vector3:
	if target_kind == "building" and not house_positions.is_empty():
		return house_positions[int(rng.randi_range(0, house_positions.size() - 1))] as Vector3
	if not cluster_centers.is_empty() and rng.randf() < 0.58:
		var center: Vector3 = cluster_centers[int(rng.randi_range(0, cluster_centers.size() - 1))] as Vector3
		return center + Vector3(rng.randf_range(-4.2, 4.2), 0.0, rng.randf_range(-4.2, 4.2))
	return Vector3(rng.randf_range(-11.0, 11.0), 0.0, rng.randf_range(-11.0, 11.0))


func _build_target_model(target_kind: String) -> void:
	var token := target_kind.to_lower()
	if token == "soldier":
		_make_sphere_child(target_root, "Head", Vector3(0.0, 0.74, 0.0), 0.16, GREEN_COLOR)
		_make_box_child(target_root, "Body", Vector3(0.0, 0.42, 0.0), Vector3(0.18, 0.30, 0.12), GREEN_COLOR.darkened(0.1))
		_make_box_child(target_root, "Beacon", Vector3(0.0, 1.22, 0.0), Vector3(0.045, 0.52, 0.045), AMBER_COLOR)
	elif token == "building":
		_make_box_child(target_root, "TargetBuilding", Vector3(0.0, 0.72, 0.0), Vector3(0.72, 0.72, 0.72), GREEN_COLOR.darkened(0.08))
		_make_box_child(target_root, "Roof", Vector3(0.0, 1.54, 0.0), Vector3(0.82, 0.18, 0.82), AMBER_COLOR)
		_make_box_child(target_root, "Beacon", Vector3(0.0, 2.14, 0.0), Vector3(0.05, 0.58, 0.05), GREEN_COLOR.lightened(0.2))
	elif token == "helicopter":
		_build_helicopter(target_root, GREEN_COLOR, 1.06)
		_make_box_child(target_root, "Beacon", Vector3(0.0, 0.92, 0.0), Vector3(0.055, 0.56, 0.055), AMBER_COLOR)
	elif token == "jet":
		_build_aircraft(target_root, GREEN_COLOR, 1.08)
		_make_box_child(target_root, "Beacon", Vector3(0.0, 0.78, 0.0), Vector3(0.055, 0.48, 0.055), AMBER_COLOR)
	else:
		_build_vehicle(target_root, GREEN_COLOR, 1.0)
		_make_box_child(target_root, "Beacon", Vector3(0.0, 0.86, 0.0), Vector3(0.055, 0.52, 0.055), AMBER_COLOR)


func _build_vehicle(parent: Node3D, color: Color, size: float) -> void:
	_make_box_child(parent, "VehicleBody", Vector3(0.0, 0.12 * size, 0.0), Vector3(0.34 * size, 0.16 * size, 0.52 * size), color)
	_make_box_child(parent, "VehicleCab", Vector3(0.0, 0.32 * size, -0.12 * size), Vector3(0.24 * size, 0.14 * size, 0.22 * size), color.lightened(0.13))
	_make_box_child(parent, "VehicleShadow", Vector3(0.0, -0.02 * size, 0.0), Vector3(0.44 * size, 0.018 * size, 0.62 * size), SHADOW_COLOR)


func _build_helicopter(parent: Node3D, color: Color, size: float) -> void:
	_make_box_child(parent, "HeliBody", Vector3(0.0, 0.0, 0.0), Vector3(0.35 * size, 0.18 * size, 0.26 * size), color)
	_make_box_child(parent, "HeliTail", Vector3(0.0, 0.02 * size, 0.55 * size), Vector3(0.07 * size, 0.055 * size, 0.48 * size), color.darkened(0.12))
	_make_box_child(parent, "HeliRotor", Vector3(0.0, 0.32 * size, 0.0), Vector3(0.98 * size, 0.018 * size, 0.045 * size), color.lightened(0.25))
	_make_box_child(parent, "HeliSkid", Vector3(-0.22 * size, -0.22 * size, 0.0), Vector3(0.035 * size, 0.035 * size, 0.34 * size), color.darkened(0.22))
	_make_box_child(parent, "HeliSkid", Vector3(0.22 * size, -0.22 * size, 0.0), Vector3(0.035 * size, 0.035 * size, 0.34 * size), color.darkened(0.22))


func _build_aircraft(parent: Node3D, color: Color, size: float) -> void:
	_make_box_child(parent, "Fuselage", Vector3(0.0, 0.0, 0.0), Vector3(0.12 * size, 0.10 * size, 0.62 * size), color)
	_make_box_child(parent, "Wing", Vector3(0.0, 0.0, -0.04 * size), Vector3(0.68 * size, 0.028 * size, 0.13 * size), color.darkened(0.12))
	_make_box_child(parent, "Nose", Vector3(0.0, 0.0, -0.58 * size), Vector3(0.08 * size, 0.08 * size, 0.18 * size), color.lightened(0.14))
	_make_box_child(parent, "Tail", Vector3(0.0, 0.14 * size, 0.46 * size), Vector3(0.05 * size, 0.18 * size, 0.12 * size), color.lightened(0.12))


func _make_box_child(parent: Node3D, name_value: String, pos: Vector3, scale_value: Vector3, color: Color) -> MeshInstance3D:
	var mesh := BoxMesh.new()
	mesh.size = Vector3(1.0, 1.0, 1.0)
	var node := MeshInstance3D.new()
	node.name = name_value
	node.mesh = mesh
	node.position = pos
	node.scale = scale_value
	node.material_override = _material(color)
	parent.add_child(node)
	return node


func _make_sphere_child(parent: Node3D, name_value: String, pos: Vector3, radius: float, color: Color) -> MeshInstance3D:
	var mesh := SphereMesh.new()
	mesh.radial_segments = 12
	mesh.rings = 6
	var node := MeshInstance3D.new()
	node.name = name_value
	node.mesh = mesh
	node.position = pos
	node.scale = Vector3(radius, radius, radius)
	node.material_override = _material(color)
	parent.add_child(node)
	return node


func _material(color: Color) -> StandardMaterial3D:
	var key := color.to_html(true)
	if material_cache.has(key):
		return material_cache[key]
	var mat := StandardMaterial3D.new()
	mat.albedo_color = color
	mat.roughness = 0.92
	mat.metallic = 0.0
	if color.a < 0.999:
		mat.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	material_cache[key] = mat
	return mat


func _apply_material_recursive(node: Node, color: Color, only_target_material: bool) -> void:
	if node is MeshInstance3D:
		var mesh_node := node as MeshInstance3D
		if not only_target_material or mesh_node.name != "Beacon":
			mesh_node.material_override = _material(color if mesh_node.name != "Beacon" else AMBER_COLOR)
	for child in node.get_children():
		_apply_material_recursive(child, color, only_target_material)


func _clear_children(node: Node) -> void:
	if node == null:
		return
	for child in node.get_children():
		child.queue_free()


func _hash_scene() -> int:
	var value := session_seed * 17 + int(round(difficulty * 1000.0))
	value = _hash_mix(value, cluster_centers.size())
	value = _hash_mix(value, house_positions.size())
	value = _hash_mix(value, moving_assets.size())
	for center in cluster_centers:
		value = _hash_mix(value, int(round((center as Vector3).x * 17.0)) + int(round((center as Vector3).z * 23.0)))
	return abs(value)


func _hash_target_schedule() -> int:
	var value := session_seed * 31 + test_code.length() * 19
	for item in target_schedule:
		var data := _as_dict(item)
		value = _hash_mix(value, _string_salt(str(data.get("kind", ""))))
		var base := _vec3(data.get("base", {}), Vector3.ZERO)
		value = _hash_mix(value, int(round(base.x * 13.0)) + int(round(base.z * 17.0)))
	return abs(value)


func _hash_mix(value: int, part: int) -> int:
	return int((int(value) * 1103515245 + int(part) * 12345 + 97) % 2147483647)


func _rng_for(stream: String) -> RandomNumberGenerator:
	var local := RandomNumberGenerator.new()
	local.seed = int(session_seed + _string_salt(stream) * 101 + 9973)
	return local


func _seed_unit(stream: String) -> float:
	var local := _rng_for(stream)
	return local.randf()


func _string_salt(value: String) -> int:
	var out := 0
	for i in range(value.length()):
		out = int((out * 131 + value.unicode_at(i)) % 1000003)
	return max(1, out)


func _vec3_dict(value: Vector3) -> Dictionary:
	return {"x": value.x, "y": value.y, "z": value.z}


func _vec3(value, fallback: Vector3 = Vector3.ZERO) -> Vector3:
	if typeof(value) != TYPE_DICTIONARY:
		return fallback
	var item := _as_dict(value)
	return Vector3(
		_float(item.get("x", fallback.x)),
		_float(item.get("y", fallback.y)),
		_float(item.get("z", fallback.z))
	)


func _as_dict(value) -> Dictionary:
	if typeof(value) == TYPE_DICTIONARY:
		return value
	return {}


func _float(value, default_value: float = 0.0) -> float:
	if typeof(value) == TYPE_FLOAT or typeof(value) == TYPE_INT:
		return float(value)
	var text := str(value)
	if text.is_valid_float():
		return float(text)
	return default_value


func _send(command: String, payload: Dictionary) -> void:
	if control_sender.is_valid():
		control_sender.call(command, payload)

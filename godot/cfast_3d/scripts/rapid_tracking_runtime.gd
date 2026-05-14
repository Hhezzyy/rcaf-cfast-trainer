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
const ROAD_MARKING_COLOR := Color(0.68, 0.63, 0.48, 1.0)
const WATER_COLOR := Color(0.12, 0.32, 0.48, 0.78)
const MOUNTAIN_COLOR := Color(0.31, 0.33, 0.29, 1.0)
const TUNNEL_COLOR := Color(0.05, 0.055, 0.055, 1.0)
const HOUSE_WALL := Color(0.66, 0.63, 0.55, 1.0)
const HOUSE_ROOF := Color(0.47, 0.18, 0.13, 1.0)
const GARAGE_COLOR := Color(0.50, 0.47, 0.39, 1.0)
const TREE_TRUNK := Color(0.28, 0.16, 0.08, 1.0)
const TREE_TOP := Color(0.16, 0.37, 0.18, 1.0)
const SHADOW_COLOR := Color(0.04, 0.05, 0.04, 0.45)
const TARGET_KINDS := ["car", "truck", "person", "building", "helicopter", "jet"]

var control_sender: Callable
var active := false
var paused := false
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
var rapid_world_config := {}
var world_size_m := 96.0
var terrain_resolution := 22
var hill_intensity := 1.35
var mountain_intensity := 1.6
var town_count := 5
var road_density := 0.92
var lake_count := 2
var river_count := 1
var tunnel_count := 2
var forest_patch_count := 10
var vehicle_count := 22
var pedestrian_count := 14
var parked_asset_count := 28
var occlusion_density := 0.34
var air_distractor_count := 4
var ground_target_weight := 0.88
var air_targets_enabled := false
var target_speed_scale := 1.0
var camera_orbit_rate_scale := 1.0
var camera_turbulence_scale := 1.0
var handoff_interval_s := 11.5
var obscuration_scale := 1.0
var moving_assets := []
var target_schedule := []
var cluster_centers := []
var house_positions := []
var road_nodes := []
var road_edges := []
var water_features := []
var forest_patches := []
var tunnel_segments := []
var occluder_positions := []
var road_graph_hash := 0
var route_hash := 0
var active_target_direction := Vector3.FORWARD


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
	_configure_world(_as_dict(cfg.get("rapid_world", {})))
	on_target_radius = maxf(0.055, _float(cfg.get("on_target_radius", 0.115 - difficulty * 0.025)))
	capture_box_half_width = maxf(0.075, _float(cfg.get("capture_box_half_width", 0.155 - difficulty * 0.025)))
	capture_box_half_height = maxf(0.065, _float(cfg.get("capture_box_half_height", 0.135 - difficulty * 0.018)))
	capture_cooldown_s = maxf(0.12, _float(cfg.get("capture_cooldown_s", 0.46 - difficulty * 0.12)))
	target_speed_scale = maxf(0.25, _float(cfg.get("target_speed_scale", 0.82 + difficulty * 0.62)))
	camera_orbit_rate_scale = maxf(0.25, _float(cfg.get("camera_orbit_rate_scale", 0.88 + difficulty * 0.42)))
	camera_turbulence_scale = maxf(0.25, _float(cfg.get("camera_turbulence_scale", 0.80 + difficulty * 0.72)))
	handoff_interval_s = maxf(5.5, _float(cfg.get("handoff_interval_s", 13.0 - difficulty * 4.0)))
	obscuration_scale = maxf(0.0, _float(cfg.get("obscuration_scale", 0.72 + difficulty * 0.75)))
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
		"road_graph_hash": road_graph_hash,
		"route_hash": route_hash,
	})


func update_runtime(delta: float, camera: Camera3D) -> void:
	if not active or paused:
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
	if not active or paused or event.echo or not event.pressed:
		return false
	var key := event.keycode
	if key == KEY_KP_ENTER:
		return true
	if key == KEY_SPACE or key == KEY_ENTER or key == KEY_KP_PERIOD:
		_capture("key")
		return true
	return false


func set_paused(value: bool) -> void:
	paused = bool(value)


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
		return minf(handoff_interval_s, 8.0)
	if token.find("lock_anchor") >= 0:
		return maxf(handoff_interval_s, 16.0)
	return handoff_interval_s


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
	if str(item.get("route", "")) == "path_graph":
		var route_nodes := _int_array(item.get("route_nodes", []))
		var pose := _route_pose(route_nodes, _float(item.get("route_distance", 0.0)) + local_t * speed, _float(item.get("lane_offset", 0.0)))
		active_target_direction = _vec3(pose.get("direction", {}), Vector3.FORWARD)
		var pos := _vec3(pose.get("position", {}), Vector3.ZERO)
		var y_offset := 0.58 if token == "person" or token == "soldier" else 0.52
		pos.y = _terrain_height_at(pos.x, pos.z) + y_offset
		return pos
	if token == "building":
		active_target_direction = Vector3.FORWARD
		return base + Vector3(0.0, 1.1, 0.0)
	if token == "soldier" or token == "person":
		active_target_direction = Vector3(sin(local_t * speed + phase_offset), 0.0, cos(local_t * speed + phase_offset)).normalized()
		return base + Vector3(sin(local_t * speed + phase_offset) * radius * 0.55, 0.58, cos(local_t * speed * 0.7 + phase_offset) * radius * 0.42)
	if token == "helicopter":
		active_target_direction = Vector3(cos(local_t * speed + phase_offset), 0.0, -sin(local_t * speed + phase_offset)).normalized()
		return base + Vector3(sin(local_t * speed + phase_offset) * radius, 6.0 + sin(local_t * 0.9 + phase_offset) * 1.1, cos(local_t * speed * 0.82 + phase_offset) * radius)
	if token == "jet":
		var sweep := fmod(local_t * speed * 6.0 + phase_offset, 28.0) - 14.0
		active_target_direction = Vector3(1.0, 0.0, cos(local_t * speed + phase_offset) * 0.3).normalized()
		return base + Vector3(sweep, 9.0 + sin(local_t * 0.8) * 1.8, sin(local_t * speed + phase_offset) * radius * 0.5)
	active_target_direction = Vector3(cos(local_t * speed + phase_offset), 0.0, -sin(local_t * speed + phase_offset)).normalized()
	return base + Vector3(sin(local_t * speed + phase_offset) * radius, 0.68, cos(local_t * speed * 0.72 + phase_offset) * radius * 0.65)


func _is_active_target_obscured(camera: Camera3D) -> bool:
	var token := test_code.to_lower()
	if _target_inside_tunnel(active_target_pos):
		return true
	if camera != null and _line_of_sight_blocked(camera.global_position, active_target_pos):
		return true
	var pressure := (0.12 + difficulty * 0.12) * obscuration_scale
	if token.find("terrain") >= 0:
		pressure += 0.18
	if active_target_kind == "building" or active_target_kind == "jet":
		pressure *= 0.35
	var wave := sin(elapsed_s * (0.55 + difficulty * 0.18) + float(active_target_index) * 1.73)
	return wave > 1.0 - pressure


func _update_live_objects(camera: Camera3D) -> void:
	active_target_pos = _active_target_world_position()
	active_target_obscured = _is_active_target_obscured(camera)
	target_root.position = active_target_pos
	var look := active_target_pos + active_target_direction
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
	var route := str(asset.get("route", "path_graph"))
	if route == "path_graph":
		var route_nodes := _int_array(asset.get("route_nodes", []))
		var total := maxf(0.001, _float(asset.get("route_total", _route_length(route_nodes))))
		var distance := fposmod(_float(asset.get("route_distance", 0.0)) + elapsed_s * _float(asset.get("speed", 0.8)), total)
		var pose := _route_pose(route_nodes, distance, _float(asset.get("lane_offset", 0.0)))
		var pos := _vec3(pose.get("position", {}), root.position)
		var dir := _vec3(pose.get("direction", {}), Vector3.FORWARD)
		var asset_kind := str(asset.get("asset_kind", "car"))
		pos.y = _terrain_height_at(pos.x, pos.z) + (0.54 if asset_kind == "person" else 0.38)
		root.position = pos
		if dir.length() > 0.01:
			root.look_at(pos + dir, Vector3.UP)
		return
	var center := _vec3(asset.get("center", {}), Vector3.ZERO)
	var radius := _float(asset.get("radius", 5.0))
	var speed := _float(asset.get("speed", 0.4))
	var phase_offset := _float(asset.get("phase", 0.0))
	var altitude := _float(asset.get("altitude", 0.5))
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
	camera.projection = Camera3D.PROJECTION_PERSPECTIVE
	var dir := -1.0 if _seed_unit("camera:direction") < 0.5 else 1.0
	var radius := world_size_m * (0.34 + difficulty * 0.05) + _seed_unit("camera:radius") * 5.0
	var rate := (0.090 + difficulty * 0.035) * camera_orbit_rate_scale * dir
	var phase_offset := _seed_unit("camera:phase") * TAU
	var angle := phase_offset + elapsed_s * rate
	var base_focus := Vector3(
		sin(elapsed_s * 0.13 + phase_offset) * 2.6,
		1.8 + sin(elapsed_s * 0.17 + phase_offset) * 0.5,
		cos(elapsed_s * 0.11 + phase_offset) * 2.2
	)
	var turbulence_strength := (0.42 + difficulty * 0.72) * camera_turbulence_scale
	var turbulence := Vector3(
		sin(elapsed_s * 1.37 + phase_offset * 0.7) * turbulence_strength,
		sin(elapsed_s * 1.91 + phase_offset * 1.3) * turbulence_strength * 0.34,
		cos(elapsed_s * 1.53 + phase_offset * 0.5) * turbulence_strength
	)
	base_focus += turbulence
	var desired := base_focus + Vector3(
		cos(angle) * radius + sin(elapsed_s * 0.61 + phase_offset) * turbulence_strength * 1.8,
		18.0 + difficulty * 6.0 + sin(elapsed_s * 0.21) * 1.2 + cos(elapsed_s * 0.79 + phase_offset) * turbulence_strength,
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
			"road_graph_hash": road_graph_hash,
			"route_hash": route_hash,
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
		"scene_theme": "expanded_rural_house_clusters",
		"scene_hash": scene_hash,
		"target_schedule_hash": target_schedule_hash,
		"road_graph_hash": road_graph_hash,
		"route_hash": route_hash,
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
	if phase == "practice":
		_send("godot_phase_complete", {"run_key": run_key, "phase": "practice", "kind": kind, "test_code": test_code, "result": result})
		return
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
	paused = false
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
	road_nodes.clear()
	road_edges.clear()
	water_features.clear()
	forest_patches.clear()
	tunnel_segments.clear()
	occluder_positions.clear()


func _configure_world(cfg: Dictionary) -> void:
	rapid_world_config = cfg.duplicate(true)
	world_size_m = maxf(64.0, _float(cfg.get("world_size_m", 92.0 + difficulty * 28.0)))
	terrain_resolution = clampi(int(round(_float(cfg.get("terrain_resolution", 18.0 + difficulty * 10.0)))), 12, 42)
	hill_intensity = maxf(0.1, _float(cfg.get("hill_intensity", 1.05 + difficulty * 1.25)))
	mountain_intensity = maxf(0.0, _float(cfg.get("mountain_intensity", 1.0 + difficulty * 1.8)))
	town_count = clampi(int(round(_float(cfg.get("town_count", 4.0 + difficulty * 4.0)))), 2, 10)
	road_density = clampf(_float(cfg.get("road_density", 0.78 + difficulty * 0.42)), 0.35, 1.75)
	lake_count = clampi(int(round(_float(cfg.get("lake_count", 1.0 + difficulty * 2.0)))), 0, 5)
	river_count = clampi(int(round(_float(cfg.get("river_count", 1.0 + difficulty)))), 0, 3)
	tunnel_count = clampi(int(round(_float(cfg.get("tunnel_count", 1.0 + difficulty * 3.0)))), 0, 6)
	forest_patch_count = clampi(int(round(_float(cfg.get("forest_patch_count", 8.0 + difficulty * 10.0)))), 2, 26)
	vehicle_count = clampi(int(round(_float(cfg.get("vehicle_count", 16.0 + difficulty * 22.0)))), 4, 64)
	pedestrian_count = clampi(int(round(_float(cfg.get("pedestrian_count", 10.0 + difficulty * 18.0)))), 2, 48)
	parked_asset_count = clampi(int(round(_float(cfg.get("parked_asset_count", 20.0 + difficulty * 30.0)))), 4, 80)
	occlusion_density = clampf(_float(cfg.get("occlusion_density", 0.22 + difficulty * 0.36)), 0.0, 1.0)
	air_distractor_count = clampi(int(round(_float(cfg.get("air_distractor_count", 3.0 + difficulty * 4.0)))), 0, 12)
	ground_target_weight = clampf(_float(cfg.get("ground_target_weight", 0.88)), 0.0, 1.0)
	air_targets_enabled = bool(cfg.get("air_targets_enabled", false))


func _generate_scene() -> void:
	_generate_town_centers()
	_generate_water_features()
	_generate_road_graph()
	_build_terrain_mesh()
	_draw_water_features()
	_draw_road_graph()
	var village_rng := _rng_for("towns")
	for i in range(cluster_centers.size()):
		_make_house_cluster(village_rng, cluster_centers[i] as Vector3, i)
	_make_forest_patches()
	_make_field_blocks()
	_make_fences(_rng_for("fences"), 26 + int(round(difficulty * 18.0)))
	_make_parked_vehicles(_rng_for("parked"), parked_asset_count)
	_make_landmarks(_rng_for("landmarks"))
	_make_moving_distractors()
	road_graph_hash = _hash_road_graph()
	route_hash = _hash_routes()
	scene_hash = _hash_scene()


func _generate_town_centers() -> void:
	var rng := _rng_for("towns")
	cluster_centers.clear()
	var half := world_size_m * 0.36
	for i in range(town_count):
		var angle := (float(i) / float(max(1, town_count))) * TAU + rng.randf_range(-0.42, 0.42)
		var radius := rng.randf_range(half * 0.24, half)
		var pos := Vector3(cos(angle) * radius, 0.0, sin(angle) * radius)
		pos.y = _terrain_height_at(pos.x, pos.z)
		cluster_centers.append(pos)


func _generate_water_features() -> void:
	var rng := _rng_for("water")
	water_features.clear()
	var half := world_size_m * 0.42
	for i in range(lake_count):
		var center := Vector3(rng.randf_range(-half, half), -0.025, rng.randf_range(-half, half))
		var radius := rng.randf_range(4.5, 9.0 + difficulty * 4.0)
		water_features.append({
			"type": "lake",
			"center": _vec3_dict(center),
			"radius": radius,
			"width": rng.randf_range(0.62, 1.18),
			"rotation": rng.randf_range(0.0, TAU),
		})
	for i in range(river_count):
		var z := rng.randf_range(-half * 0.55, half * 0.55)
		var bend := rng.randf_range(-9.0, 9.0)
		water_features.append({
			"type": "river",
			"a": _vec3_dict(Vector3(-half, -0.015, z - bend)),
			"b": _vec3_dict(Vector3(half, -0.015, z + bend)),
			"radius": rng.randf_range(1.0, 1.75),
			"rotation": 0.0,
		})


func _generate_road_graph() -> void:
	road_nodes.clear()
	road_edges.clear()
	tunnel_segments.clear()
	var rng := _rng_for("roads")
	var hub := _add_road_node(Vector3(0.0, _terrain_height_at(0.0, 0.0), 0.0), "hub")
	for i in range(cluster_centers.size()):
		var center := cluster_centers[i] as Vector3
		var town := _add_road_node(center, "town")
		_connect_road_nodes(hub, town, false, "arterial")
		var local_count := 3 + int(round(road_density * 2.0))
		for j in range(local_count):
			var angle := (float(j) / float(local_count)) * TAU + rng.randf_range(-0.35, 0.35)
			var radius := rng.randf_range(3.2, 7.2)
			var local := center + Vector3(cos(angle) * radius, 0.0, sin(angle) * radius)
			local.y = _terrain_height_at(local.x, local.z)
			var node_idx := _add_road_node(local, "town_street")
			_connect_road_nodes(town, node_idx, false, "street")
			if j > 0:
				_connect_road_nodes(node_idx - 1, node_idx, false, "street")
	if cluster_centers.size() > 1:
		for i in range(cluster_centers.size()):
			var a := 1 + i * (4 + int(round(road_density * 2.0)))
			var b := 1 + ((i + 1) % cluster_centers.size()) * (4 + int(round(road_density * 2.0)))
			if a < road_nodes.size() and b < road_nodes.size():
				_connect_road_nodes(a, b, false, "ring")
	var service_count := int(round(6.0 + road_density * 7.0))
	var half := world_size_m * 0.43
	for i in range(service_count):
		var pos := Vector3(rng.randf_range(-half, half), 0.0, rng.randf_range(-half, half))
		pos.y = _terrain_height_at(pos.x, pos.z)
		var idx := _add_road_node(pos, "service")
		var near := _nearest_road_node(pos, idx)
		if near >= 0:
			_connect_road_nodes(idx, near, false, "service")
	_mark_tunnel_edges(rng)


func _add_road_node(pos: Vector3, node_type: String) -> int:
	road_nodes.append({"pos": _vec3_dict(pos), "type": node_type})
	return road_nodes.size() - 1


func _connect_road_nodes(a: int, b: int, tunnel: bool, edge_type: String) -> void:
	if a == b or a < 0 or b < 0 or a >= road_nodes.size() or b >= road_nodes.size():
		return
	var pa := _road_node_pos(a)
	var pb := _road_node_pos(b)
	road_edges.append({
		"a": a,
		"b": b,
		"length": pa.distance_to(pb),
		"tunnel": tunnel,
		"type": edge_type,
	})


func _mark_tunnel_edges(rng: RandomNumberGenerator) -> void:
	if tunnel_count <= 0:
		return
	var candidates := []
	for i in range(road_edges.size()):
		var edge := _as_dict(road_edges[i])
		var a := _road_node_pos(int(edge.get("a", -1)))
		var b := _road_node_pos(int(edge.get("b", -1)))
		var mid := (a + b) * 0.5
		if a.distance_to(b) > 9.0 and _terrain_height_at(mid.x, mid.z) > 0.45:
			candidates.append(i)
	var made := 0
	while made < tunnel_count and not candidates.is_empty():
		var pick_pos := int(rng.randi_range(0, candidates.size() - 1))
		var edge_index := int(candidates[pick_pos])
		candidates.remove_at(pick_pos)
		var edge := _as_dict(road_edges[edge_index])
		edge["tunnel"] = true
		edge["type"] = "tunnel"
		road_edges[edge_index] = edge
		tunnel_segments.append({
			"a": edge.get("a", 0),
			"b": edge.get("b", 0),
			"radius": 2.4,
		})
		made += 1


func _build_terrain_mesh() -> void:
	var mesh := ArrayMesh.new()
	var vertices := PackedVector3Array()
	var normals := PackedVector3Array()
	var colors := PackedColorArray()
	var indices := PackedInt32Array()
	var half := world_size_m * 0.5
	var step := world_size_m / float(terrain_resolution)
	for z in range(terrain_resolution + 1):
		for x in range(terrain_resolution + 1):
			var px := -half + float(x) * step
			var pz := -half + float(z) * step
			var py := _terrain_height_at(px, pz) - 0.04
			vertices.append(Vector3(px, py, pz))
			normals.append(Vector3.UP)
			var h := clampf((py + 0.8) / maxf(1.0, mountain_intensity + hill_intensity), 0.0, 1.0)
			colors.append(FIELD_DARK.lerp(MOUNTAIN_COLOR, h * 0.72).lerp(FIELD_LIGHT, 0.18))
	for z in range(terrain_resolution):
		for x in range(terrain_resolution):
			var i0 := z * (terrain_resolution + 1) + x
			var i1 := i0 + 1
			var i2 := i0 + terrain_resolution + 1
			var i3 := i2 + 1
			indices.append(i0)
			indices.append(i2)
			indices.append(i1)
			indices.append(i1)
			indices.append(i2)
			indices.append(i3)
	var arrays := []
	arrays.resize(Mesh.ARRAY_MAX)
	arrays[Mesh.ARRAY_VERTEX] = vertices
	arrays[Mesh.ARRAY_NORMAL] = normals
	arrays[Mesh.ARRAY_COLOR] = colors
	arrays[Mesh.ARRAY_INDEX] = indices
	mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, arrays)
	var node := MeshInstance3D.new()
	node.name = "SeededHeightfieldTerrain"
	node.mesh = mesh
	node.material_override = _vertex_color_material()
	static_root.add_child(node)


func _draw_water_features() -> void:
	for item in water_features:
		var data := _as_dict(item)
		if str(data.get("type", "lake")) == "river":
			var a := _vec3(data.get("a", {}), Vector3.ZERO)
			var b := _vec3(data.get("b", {}), Vector3.ZERO)
			_make_road_like_box(static_root, "River", a, b, _float(data.get("radius", 1.25)), WATER_COLOR, -0.045)
		else:
			var center := _vec3(data.get("center", {}), Vector3.ZERO)
			var radius := _float(data.get("radius", 6.0))
			var node := _make_box_child(static_root, "Lake", center + Vector3(0.0, -0.075, 0.0), Vector3(radius, 0.025, radius * _float(data.get("width", 0.85))), WATER_COLOR)
			node.rotation.y = _float(data.get("rotation", 0.0))


func _draw_road_graph() -> void:
	for edge_value in road_edges:
		var edge := _as_dict(edge_value)
		var a := _road_node_pos(int(edge.get("a", 0)))
		var b := _road_node_pos(int(edge.get("b", 0)))
		var tunnel := bool(edge.get("tunnel", false))
		if tunnel:
			_make_tunnel_visual(a, b)
		else:
			_make_road_like_box(static_root, "RoadGraphEdge", a, b, 0.70 if str(edge.get("type", "")) == "street" else 0.95, ROAD_COLOR, 0.02)
			if a.distance_to(b) > 8.0:
				_make_road_like_box(static_root, "RoadCenterMark", a, b, 0.045, ROAD_MARKING_COLOR, 0.055)


func _make_tunnel_visual(a: Vector3, b: Vector3) -> void:
	var dir := (b - a).normalized()
	var length := a.distance_to(b)
	var center := (a + b) * 0.5
	var road := _make_road_like_box(static_root, "TunnelRoadHiddenPath", a, b, 0.82, ROAD_COLOR.darkened(0.35), -0.03)
	road.visible = true
	var portal_a := _make_box_child(static_root, "TunnelPortal", a + dir * 0.9 + Vector3(0.0, 0.72, 0.0), Vector3(1.65, 0.95, 0.28), TUNNEL_COLOR)
	portal_a.look_at(b, Vector3.UP)
	var portal_b := _make_box_child(static_root, "TunnelPortal", b - dir * 0.9 + Vector3(0.0, 0.72, 0.0), Vector3(1.65, 0.95, 0.28), TUNNEL_COLOR)
	portal_b.look_at(a, Vector3.UP)
	var cap := _make_box_child(static_root, "TunnelHillCap", center + Vector3(0.0, 0.55 + length * 0.012, 0.0), Vector3(1.9, 0.62, maxf(1.2, length * 0.22)), MOUNTAIN_COLOR.darkened(0.12))
	cap.look_at(b, Vector3.UP)


func _make_road_like_box(parent: Node3D, name_value: String, a: Vector3, b: Vector3, half_width: float, color: Color, y_offset: float) -> MeshInstance3D:
	var pa := Vector3(a.x, _terrain_height_at(a.x, a.z) + y_offset, a.z)
	var pb := Vector3(b.x, _terrain_height_at(b.x, b.z) + y_offset, b.z)
	var center := (pa + pb) * 0.5
	var length := Vector2(pa.x - pb.x, pa.z - pb.z).length()
	var node := _make_box_child(parent, name_value, center, Vector3(half_width, 0.035, length * 0.5), color)
	node.look_at(pb, Vector3.UP)
	return node


func _make_field_blocks() -> void:
	var rng := _rng_for("fields")
	var half := world_size_m * 0.43
	var count := 26 + int(round(difficulty * 18.0))
	for i in range(count):
		var pos := Vector3(rng.randf_range(-half, half), 0.0, rng.randf_range(-half, half))
		if _near_any_cluster(pos, 7.0) or _near_water(pos, 5.5):
			continue
		pos.y = _terrain_height_at(pos.x, pos.z) - 0.02
		var tint := FIELD_LIGHT.lerp(FIELD_DARK, rng.randf() * 0.70)
		var node := _make_box_child(static_root, "OpenField", pos, Vector3(rng.randf_range(2.8, 6.4), 0.018, rng.randf_range(2.8, 7.2)), tint)
		node.rotation.y = rng.randf_range(0.0, TAU)


func _make_forest_patches() -> void:
	var rng := _rng_for("forests")
	forest_patches.clear()
	var half := world_size_m * 0.44
	for i in range(forest_patch_count):
		var center := Vector3(rng.randf_range(-half, half), 0.0, rng.randf_range(-half, half))
		if _near_any_cluster(center, 5.5):
			center += center.normalized() * 5.5
		center.y = _terrain_height_at(center.x, center.z)
		var radius := rng.randf_range(3.4, 7.8 + difficulty * 2.2)
		forest_patches.append({"center": _vec3_dict(center), "radius": radius})
		var tree_count := int(round(radius * rng.randf_range(3.2, 5.4)))
		for j in range(tree_count):
			var angle := rng.randf_range(0.0, TAU)
			var dist := sqrt(rng.randf()) * radius
			var pos := center + Vector3(cos(angle) * dist, 0.0, sin(angle) * dist)
			pos.y = _terrain_height_at(pos.x, pos.z)
			_make_tree_at(pos, rng, "ForestTree")
		if rng.randf() < occlusion_density:
			occluder_positions.append({"pos": _vec3_dict(center), "radius": radius * 0.70, "type": "forest"})


func _make_tree_at(pos: Vector3, rng: RandomNumberGenerator, name_prefix: String) -> void:
	var root := Node3D.new()
	root.name = name_prefix
	root.position = pos
	static_root.add_child(root)
	var height := rng.randf_range(0.78, 1.75)
	_make_box_child(root, "Trunk", Vector3(0.0, height * 0.34, 0.0), Vector3(0.07, height * 0.34, 0.07), TREE_TRUNK)
	_make_sphere_child(root, "Canopy", Vector3(0.0, height * 0.86, 0.0), rng.randf_range(0.32, 0.62), TREE_TOP.lerp(Color(0.24, 0.46, 0.19, 1.0), rng.randf() * 0.42))


func _make_house_cluster(rng: RandomNumberGenerator, center: Vector3, cluster_index: int) -> void:
	var count := 7 + int(rng.randi_range(0, 5)) + int(round(difficulty * 2.0))
	for i in range(count):
		var offset := Vector3(rng.randf_range(-5.0, 5.0), 0.0, rng.randf_range(-4.7, 4.7))
		var pos := center + offset
		pos.y = _terrain_height_at(pos.x, pos.z)
		house_positions.append(pos)
		var taller := rng.randf() < 0.22 + difficulty * 0.18
		var scale := Vector3(rng.randf_range(0.62, 1.20), rng.randf_range(0.46, 0.92) + (0.55 if taller else 0.0), rng.randf_range(0.62, 1.22))
		var root := Node3D.new()
		root.name = "HouseCluster" + str(cluster_index) + "House" + str(i)
		root.position = pos
		root.rotation_degrees.y = rng.randf_range(-32.0, 32.0)
		static_root.add_child(root)
		_make_box_child(root, "Walls", Vector3(0.0, scale.y, 0.0), scale, HOUSE_WALL.lerp(Color(0.82, 0.78, 0.66, 1.0), rng.randf() * 0.35))
		_make_box_child(root, "Roof", Vector3(0.0, scale.y * 2.04, 0.0), Vector3(scale.x * 1.12, 0.20, scale.z * 1.12), HOUSE_ROOF.lerp(Color(0.28, 0.24, 0.21, 1.0), rng.randf() * 0.42))
		if rng.randf() < 0.64:
			_make_box_child(root, "Garage", Vector3(scale.x * 1.45, 0.30, -scale.z * 0.2), Vector3(0.42, 0.30, 0.52), GARAGE_COLOR)
			_make_box_child(root, "GarageDoor", Vector3(scale.x * 1.45, 0.28, -scale.z * 0.73), Vector3(0.30, 0.20, 0.035), ROAD_MARKING_COLOR.darkened(0.20))
		if rng.randf() < occlusion_density:
			occluder_positions.append({"pos": _vec3_dict(pos), "radius": maxf(scale.x, scale.z) * 1.4, "type": "building"})


func _make_road(a: Vector3, b: Vector3, half_width: float) -> void:
	var center := (a + b) * 0.5
	var length := Vector2(a.x - b.x, a.z - b.z).length()
	var node := _make_box_child(static_root, "Road", center + Vector3(0.0, -0.055, 0.0), Vector3(half_width, 0.04, length * 0.5), ROAD_COLOR)
	node.look_at(b, Vector3.UP)


func _make_trees(rng: RandomNumberGenerator, count: int) -> void:
	for i in range(count):
		var half := world_size_m * 0.44
		var pos := Vector3(rng.randf_range(-half, half), 0.0, rng.randf_range(-half, half))
		if pos.length() < 3.5:
			pos += pos.normalized() * 4.0
		pos.y = _terrain_height_at(pos.x, pos.z)
		_make_tree_at(pos, rng, "Tree" + str(i))


func _make_fences(rng: RandomNumberGenerator, count: int) -> void:
	for i in range(count):
		var half := world_size_m * 0.39
		var pos := Vector3(rng.randf_range(-half, half), 0.0, rng.randf_range(-half, half))
		pos.y = _terrain_height_at(pos.x, pos.z) + 0.15
		var length := rng.randf_range(1.5, 4.5)
		var node := _make_box_child(static_root, "Fence", pos, Vector3(0.045, 0.15, length), Color(0.48, 0.38, 0.23, 1.0))
		node.rotation_degrees.y = rng.randf_range(0.0, 180.0)


func _make_parked_vehicles(rng: RandomNumberGenerator, count: int) -> void:
	for i in range(count):
		var pos := _random_road_position(rng, true)
		pos.y = _terrain_height_at(pos.x, pos.z) + 0.34
		var root := Node3D.new()
		root.name = "ParkedVehicle" + str(i)
		root.position = pos
		root.rotation_degrees.y = rng.randf_range(0.0, 360.0)
		static_root.add_child(root)
		_build_vehicle(root, Color(0.26, 0.33, 0.30, 1.0).lerp(AMBER_COLOR.darkened(0.35), rng.randf() * 0.3), 0.82)


func _make_landmarks(rng: RandomNumberGenerator) -> void:
	var tower := Node3D.new()
	tower.name = "WaterTower"
	tower.position = Vector3(rng.randf_range(-world_size_m * 0.20, world_size_m * 0.20), 0.0, rng.randf_range(-world_size_m * 0.22, world_size_m * 0.22))
	tower.position.y = _terrain_height_at(tower.position.x, tower.position.z)
	static_root.add_child(tower)
	_make_box_child(tower, "Leg", Vector3(-0.25, 1.2, -0.25), Vector3(0.05, 1.2, 0.05), Color(0.38, 0.39, 0.36, 1.0))
	_make_box_child(tower, "Leg", Vector3(0.25, 1.2, -0.25), Vector3(0.05, 1.2, 0.05), Color(0.38, 0.39, 0.36, 1.0))
	_make_box_child(tower, "Leg", Vector3(-0.25, 1.2, 0.25), Vector3(0.05, 1.2, 0.05), Color(0.38, 0.39, 0.36, 1.0))
	_make_box_child(tower, "Leg", Vector3(0.25, 1.2, 0.25), Vector3(0.05, 1.2, 0.05), Color(0.38, 0.39, 0.36, 1.0))
	_make_sphere_child(tower, "Tank", Vector3(0.0, 2.65, 0.0), 0.52, Color(0.55, 0.57, 0.56, 1.0))


func _make_moving_distractors() -> void:
	var rng := _rng_for("moving_routes")
	var ground_total := vehicle_count + pedestrian_count
	for i in range(ground_total):
		var kind_value := "person" if i >= vehicle_count else ("truck" if i % 4 == 0 else "car")
		var root := Node3D.new()
		root.name = "PathGraphMover" + str(i)
		moving_root.add_child(root)
		if kind_value == "person":
			_build_person(root, Color(0.30, 0.56, 0.28, 1.0), 0.82)
		elif kind_value == "truck":
			_build_vehicle(root, Color(0.22, 0.30, 0.28, 1.0).lerp(RED_COLOR.darkened(0.38), rng.randf() * 0.34), 0.88)
		else:
			_build_vehicle(root, Color(0.19, 0.31, 0.42, 1.0).lerp(AMBER_COLOR.darkened(0.32), rng.randf() * 0.34), 0.72)
		var route_nodes := _random_route_nodes(rng, kind_value)
		moving_assets.append({
			"node": root,
			"asset_kind": kind_value,
			"route": "path_graph",
			"pathfinding": "road_graph",
			"route_nodes": route_nodes,
			"route_total": _route_length(route_nodes),
			"route_distance": rng.randf_range(0.0, maxf(1.0, _route_length(route_nodes))),
			"speed": _ground_speed_for_kind(kind_value, rng),
			"lane_offset": rng.randf_range(-0.46, 0.46) if kind_value != "person" else rng.randf_range(-0.92, 0.92),
		})
	for i in range(air_distractor_count):
		var root := Node3D.new()
		root.name = "AirDistractor" + str(i)
		moving_root.add_child(root)
		if i % 2 == 0:
			_build_aircraft(root, BLUE_COLOR.darkened(0.25), 0.72)
		else:
			_build_helicopter(root, Color(0.30, 0.42, 0.48, 1.0), 0.82)
		moving_assets.append({
			"node": root,
			"asset_kind": "helicopter" if i % 2 else "jet",
			"center": _vec3_dict(Vector3(rng.randf_range(-world_size_m * 0.20, world_size_m * 0.20), 0.0, rng.randf_range(-world_size_m * 0.20, world_size_m * 0.20))),
			"radius": rng.randf_range(12.0, 24.0),
			"speed": rng.randf_range(0.18, 0.45 + difficulty * 0.32),
			"phase": rng.randf() * TAU,
			"altitude": rng.randf_range(6.5, 13.0),
			"route": "air",
		})


func _generate_target_schedule() -> void:
	var rng := _rng_for("targets")
	var count := int(ceil(duration_s / _segment_duration_s())) + 2
	target_schedule.clear()
	for i in range(count):
		var target_kind := _target_kind_for_slot(i, rng)
		var moving := target_kind != "building"
		var item := {
			"kind": target_kind,
			"moving": moving,
			"phase": rng.randf() * TAU,
			"radius": rng.randf_range(8.0, 20.0 + difficulty * 5.0),
			"speed": rng.randf_range(0.22, 0.48 + difficulty * 0.36) * target_speed_scale,
		}
		if target_kind == "building":
			item["base"] = _vec3_dict(_base_for_target(target_kind, rng))
		elif target_kind == "helicopter" or target_kind == "jet":
			item["base"] = _vec3_dict(_base_for_target(target_kind, rng))
			item["route"] = "air"
			item["speed"] = rng.randf_range(0.34, 0.72 + difficulty * 0.45) * target_speed_scale
		else:
			var route_nodes := _random_route_nodes(rng, target_kind)
			item["route"] = "path_graph"
			item["pathfinding"] = "road_graph"
			item["route_nodes"] = route_nodes
			item["route_total"] = _route_length(route_nodes)
			item["route_distance"] = rng.randf_range(0.0, maxf(1.0, _route_length(route_nodes)))
			item["lane_offset"] = rng.randf_range(-0.38, 0.38) if target_kind != "person" else rng.randf_range(-0.86, 0.86)
			item["speed"] = _ground_speed_for_kind(target_kind, rng) * (1.06 + difficulty * 0.18)
		target_schedule.append(item)
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
		return "truck" if index % 2 == 0 else "person"
	if token.find("terrain") >= 0:
		return "person" if index % 3 != 0 else "truck"
	if air_targets_enabled and rng.randf() > ground_target_weight:
		return "helicopter" if rng.randf() < 0.66 else "jet"
	var roll := rng.randf()
	if roll < 0.36:
		return "car"
	if roll < 0.68:
		return "truck"
	if roll < 0.90:
		return "person"
	return "building"


func _base_for_target(target_kind: String, rng: RandomNumberGenerator) -> Vector3:
	if target_kind == "building" and not house_positions.is_empty():
		return house_positions[int(rng.randi_range(0, house_positions.size() - 1))] as Vector3
	if not cluster_centers.is_empty() and rng.randf() < 0.58:
		var center: Vector3 = cluster_centers[int(rng.randi_range(0, cluster_centers.size() - 1))] as Vector3
		return center + Vector3(rng.randf_range(-4.2, 4.2), 0.0, rng.randf_range(-4.2, 4.2))
	var half := world_size_m * 0.32
	return Vector3(rng.randf_range(-half, half), 0.0, rng.randf_range(-half, half))


func _build_target_model(target_kind: String) -> void:
	var token := target_kind.to_lower()
	if token == "soldier" or token == "person":
		_make_sphere_child(target_root, "Head", Vector3(0.0, 0.74, 0.0), 0.16, GREEN_COLOR)
		_make_box_child(target_root, "Body", Vector3(0.0, 0.42, 0.0), Vector3(0.18, 0.30, 0.12), GREEN_COLOR.darkened(0.1))
		_make_box_child(target_root, "Legs", Vector3(0.0, 0.18, 0.0), Vector3(0.14, 0.16, 0.08), GREEN_COLOR.darkened(0.22))
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
	elif token == "car":
		_build_vehicle(target_root, GREEN_COLOR.lightened(0.08), 0.86)
		_make_box_child(target_root, "Beacon", Vector3(0.0, 0.80, 0.0), Vector3(0.055, 0.48, 0.055), AMBER_COLOR)
	else:
		_build_vehicle(target_root, GREEN_COLOR, 1.0)
		_make_box_child(target_root, "Beacon", Vector3(0.0, 0.86, 0.0), Vector3(0.055, 0.52, 0.055), AMBER_COLOR)


func _build_person(parent: Node3D, color: Color, size: float) -> void:
	_make_sphere_child(parent, "PersonHead", Vector3(0.0, 0.55 * size, 0.0), 0.11 * size, color.lightened(0.18))
	_make_box_child(parent, "PersonBody", Vector3(0.0, 0.32 * size, 0.0), Vector3(0.12 * size, 0.20 * size, 0.08 * size), color)
	_make_box_child(parent, "PersonLegs", Vector3(0.0, 0.12 * size, 0.0), Vector3(0.10 * size, 0.12 * size, 0.06 * size), color.darkened(0.18))


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


func _vertex_color_material() -> StandardMaterial3D:
	var key := "vertex_color_terrain"
	if material_cache.has(key):
		return material_cache[key]
	var mat := StandardMaterial3D.new()
	mat.vertex_color_use_as_albedo = true
	mat.roughness = 0.96
	mat.metallic = 0.0
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


func _terrain_height_at(x: float, z: float) -> float:
	var half := maxf(1.0, world_size_m * 0.5)
	var nx := x / half
	var nz := z / half
	var seed_phase := _seed_unit("terrain:phase") * TAU
	var rolling := sin(nx * 4.1 + seed_phase) * 0.34 + cos(nz * 3.7 - seed_phase * 0.7) * 0.28
	var ripple := (_value_noise(x * 0.055, z * 0.055, "terrain:detail") - 0.5) * 0.95
	var ridge_line := nx * 0.72 + nz * 0.36 + (_seed_unit("mountain:ridge") - 0.5) * 0.42
	var ridge := pow(maxf(0.0, 1.0 - absf(ridge_line)), 2.4) * mountain_intensity
	var shoulder := pow(maxf(0.0, 1.0 - absf(nx * 0.25 - nz * 0.9)), 3.0) * mountain_intensity * 0.42
	var water_cut := 0.0
	for feature in water_features:
		var data := _as_dict(feature)
		if str(data.get("type", "")) == "lake":
			var center := _vec3(data.get("center", {}), Vector3.ZERO)
			var radius := _float(data.get("radius", 4.0))
			var dist := Vector2(x - center.x, z - center.z).length()
			if dist < radius * 1.15:
				water_cut = maxf(water_cut, (1.0 - dist / maxf(0.1, radius * 1.15)) * 0.55)
	return maxf(-0.32, rolling * hill_intensity + ripple * hill_intensity * 0.55 + ridge + shoulder - water_cut)


func _value_noise(x: float, z: float, stream: String) -> float:
	var xi := int(floor(x))
	var zi := int(floor(z))
	var tx := x - float(xi)
	var tz := z - float(zi)
	var a := _hash_unit_float(xi, zi, stream)
	var b := _hash_unit_float(xi + 1, zi, stream)
	var c := _hash_unit_float(xi, zi + 1, stream)
	var d := _hash_unit_float(xi + 1, zi + 1, stream)
	var sx := tx * tx * (3.0 - 2.0 * tx)
	var sz := tz * tz * (3.0 - 2.0 * tz)
	return lerpf(lerpf(a, b, sx), lerpf(c, d, sx), sz)


func _hash_unit_float(x: int, z: int, stream: String) -> float:
	var raw := sin(float(x) * 12.9898 + float(z) * 78.233 + float(session_seed) * 0.017 + float(_string_salt(stream)) * 0.071) * 43758.5453123
	return fposmod(raw, 1.0)


func _road_node_pos(index: int) -> Vector3:
	if index < 0 or index >= road_nodes.size():
		return Vector3.ZERO
	var node := _as_dict(road_nodes[index])
	return _vec3(node.get("pos", {}), Vector3.ZERO)


func _nearest_road_node(pos: Vector3, exclude: int = -1) -> int:
	var best := -1
	var best_dist := 1.0e20
	for i in range(road_nodes.size()):
		if i == exclude:
			continue
		var dist := pos.distance_squared_to(_road_node_pos(i))
		if dist < best_dist:
			best_dist = dist
			best = i
	return best


func _road_neighbors(index: int) -> Array:
	var out := []
	for edge_value in road_edges:
		var edge := _as_dict(edge_value)
		var a := int(edge.get("a", -1))
		var b := int(edge.get("b", -1))
		if a == index:
			out.append({"node": b, "length": _float(edge.get("length", 1.0))})
		elif b == index:
			out.append({"node": a, "length": _float(edge.get("length", 1.0))})
	return out


func _find_path_nodes(start_index: int, goal_index: int) -> Array:
	var count := road_nodes.size()
	if start_index < 0 or goal_index < 0 or start_index >= count or goal_index >= count:
		return []
	var dist := []
	var prev := []
	var visited := []
	for i in range(count):
		dist.append(1.0e20)
		prev.append(-1)
		visited.append(false)
	dist[start_index] = 0.0
	for _step_index in range(count):
		var current := -1
		var best := 1.0e20
		for i in range(count):
			if not bool(visited[i]) and float(dist[i]) < best:
				best = float(dist[i])
				current = i
		if current < 0 or current == goal_index:
			break
		visited[current] = true
		for neighbor_value in _road_neighbors(current):
			var neighbor := _as_dict(neighbor_value)
			var next_idx := int(neighbor.get("node", -1))
			if next_idx < 0:
				continue
			var alt := float(dist[current]) + _float(neighbor.get("length", 1.0))
			if alt < float(dist[next_idx]):
				dist[next_idx] = alt
				prev[next_idx] = current
	var path := []
	var cursor := goal_index
	while cursor >= 0:
		path.push_front(cursor)
		if cursor == start_index:
			break
		cursor = int(prev[cursor])
	if path.is_empty() or int(path[0]) != start_index:
		return [start_index, goal_index]
	return path


func _random_route_nodes(rng: RandomNumberGenerator, kind_value: String) -> Array:
	if road_nodes.size() < 2:
		return [0, 0]
	var start := int(rng.randi_range(0, road_nodes.size() - 1))
	var goal := int(rng.randi_range(0, road_nodes.size() - 1))
	var guard := 0
	while goal == start and guard < 16:
		goal = int(rng.randi_range(0, road_nodes.size() - 1))
		guard += 1
	var path := _find_path_nodes(start, goal)
	if path.size() < 2:
		path = [start, (start + 1) % road_nodes.size()]
	if kind_value == "person" and path.size() > 4:
		path = path.slice(0, 4)
	return path


func _route_length(route_nodes: Array) -> float:
	var total := 0.0
	for i in range(max(0, route_nodes.size() - 1)):
		total += _road_node_pos(int(route_nodes[i])).distance_to(_road_node_pos(int(route_nodes[i + 1])))
	return maxf(total, 0.001)


func _route_pose(route_nodes: Array, distance: float, lane_offset: float) -> Dictionary:
	if route_nodes.size() < 2:
		var fallback := _road_node_pos(0)
		return {"position": _vec3_dict(fallback), "direction": _vec3_dict(Vector3.FORWARD)}
	var total := _route_length(route_nodes)
	var remaining := fposmod(distance, maxf(0.001, total))
	for i in range(route_nodes.size() - 1):
		var a := _road_node_pos(int(route_nodes[i]))
		var b := _road_node_pos(int(route_nodes[i + 1]))
		var length := maxf(0.001, a.distance_to(b))
		if remaining <= length:
			var t := remaining / length
			var dir := (b - a).normalized()
			var right := Vector3(-dir.z, 0.0, dir.x).normalized()
			var pos := a.lerp(b, t) + right * lane_offset
			return {"position": _vec3_dict(pos), "direction": _vec3_dict(dir)}
		remaining -= length
	var last_a := _road_node_pos(int(route_nodes[route_nodes.size() - 2]))
	var last_b := _road_node_pos(int(route_nodes[route_nodes.size() - 1]))
	var last_dir := (last_b - last_a).normalized()
	return {"position": _vec3_dict(last_b), "direction": _vec3_dict(last_dir)}


func _ground_speed_for_kind(kind_value: String, rng: RandomNumberGenerator) -> float:
	var scale := target_speed_scale
	if kind_value == "person":
		return rng.randf_range(0.75, 1.15 + difficulty * 0.30) * scale
	if kind_value == "truck":
		return rng.randf_range(1.15, 1.95 + difficulty * 0.55) * scale
	return rng.randf_range(1.35, 2.35 + difficulty * 0.70) * scale


func _random_road_position(rng: RandomNumberGenerator, offset_from_lane: bool) -> Vector3:
	if road_edges.is_empty():
		return Vector3(rng.randf_range(-8.0, 8.0), 0.0, rng.randf_range(-8.0, 8.0))
	var edge := _as_dict(road_edges[int(rng.randi_range(0, road_edges.size() - 1))])
	var a := _road_node_pos(int(edge.get("a", 0)))
	var b := _road_node_pos(int(edge.get("b", 0)))
	var dir := (b - a).normalized()
	var right := Vector3(-dir.z, 0.0, dir.x).normalized()
	var pos := a.lerp(b, rng.randf())
	if offset_from_lane:
		pos += right * rng.randf_range(-1.6, 1.6)
	return pos


func _near_any_cluster(pos: Vector3, radius: float) -> bool:
	for center in cluster_centers:
		var item := center as Vector3
		if Vector2(pos.x - item.x, pos.z - item.z).length() <= radius:
			return true
	return false


func _near_water(pos: Vector3, padding: float) -> bool:
	for feature in water_features:
		var data := _as_dict(feature)
		if str(data.get("type", "")) == "lake":
			var center := _vec3(data.get("center", {}), Vector3.ZERO)
			if Vector2(pos.x - center.x, pos.z - center.z).length() <= _float(data.get("radius", 4.0)) + padding:
				return true
	return false


func _target_inside_tunnel(pos: Vector3) -> bool:
	if active_target_kind == "helicopter" or active_target_kind == "jet":
		return false
	for item in tunnel_segments:
		var data := _as_dict(item)
		var a := _road_node_pos(int(data.get("a", 0)))
		var b := _road_node_pos(int(data.get("b", 0)))
		if _point_segment_distance_2d(pos, a, b) <= _float(data.get("radius", 2.4)):
			return true
	return false


func _line_of_sight_blocked(from_pos: Vector3, to_pos: Vector3) -> bool:
	if active_target_kind == "building" or active_target_kind == "jet":
		return false
	var max_hits := int(round(float(occluder_positions.size()) * clampf(occlusion_density, 0.0, 1.0)))
	var checked := 0
	for item in occluder_positions:
		if checked > max_hits:
			break
		checked += 1
		var data := _as_dict(item)
		var center := _vec3(data.get("pos", {}), Vector3.ZERO)
		var radius := _float(data.get("radius", 1.0))
		if _point_segment_distance_2d(center, from_pos, to_pos) <= radius:
			var from_dist := Vector2(from_pos.x - center.x, from_pos.z - center.z).length()
			var target_dist := Vector2(to_pos.x - center.x, to_pos.z - center.z).length()
			if from_dist > radius * 1.2 and target_dist > radius * 0.55:
				return true
	return false


func _point_segment_distance_2d(point: Vector3, a: Vector3, b: Vector3) -> float:
	var p := Vector2(point.x, point.z)
	var va := Vector2(a.x, a.z)
	var vb := Vector2(b.x, b.z)
	var ab := vb - va
	var len_sq := maxf(0.0001, ab.length_squared())
	var t := clampf((p - va).dot(ab) / len_sq, 0.0, 1.0)
	return p.distance_to(va + ab * t)


func _int_array(value) -> Array:
	if typeof(value) != TYPE_ARRAY:
		return []
	var out := []
	for item in value:
		out.append(int(item))
	return out


func _hash_scene() -> int:
	var value := session_seed * 17 + int(round(difficulty * 1000.0))
	value = _hash_mix(value, cluster_centers.size())
	value = _hash_mix(value, house_positions.size())
	value = _hash_mix(value, moving_assets.size())
	value = _hash_mix(value, road_nodes.size())
	value = _hash_mix(value, road_edges.size())
	value = _hash_mix(value, water_features.size())
	value = _hash_mix(value, forest_patches.size())
	value = _hash_mix(value, tunnel_segments.size())
	value = _hash_mix(value, road_graph_hash)
	value = _hash_mix(value, route_hash)
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
		for node in _int_array(data.get("route_nodes", [])):
			value = _hash_mix(value, int(node) + 41)
	return abs(value)


func _hash_road_graph() -> int:
	var value := session_seed * 43 + int(round(world_size_m * 10.0))
	for node_value in road_nodes:
		var pos := _vec3(_as_dict(node_value).get("pos", {}), Vector3.ZERO)
		value = _hash_mix(value, int(round(pos.x * 7.0)) + int(round(pos.z * 11.0)))
	for edge_value in road_edges:
		var edge := _as_dict(edge_value)
		value = _hash_mix(value, int(edge.get("a", 0)) * 13 + int(edge.get("b", 0)) * 17 + (101 if bool(edge.get("tunnel", false)) else 0))
	return abs(value)


func _hash_routes() -> int:
	var value := session_seed * 59 + moving_assets.size() * 23
	for asset in moving_assets:
		var data := _as_dict(asset)
		value = _hash_mix(value, _string_salt(str(data.get("asset_kind", ""))))
		for node in _int_array(data.get("route_nodes", [])):
			value = _hash_mix(value, int(node) + 73)
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

extends Node3D

const WHITE_COLOR := Color(0.92, 0.96, 0.98, 1.0)
const TEXT_MUTED := Color(0.70, 0.76, 0.82, 1.0)
const GREEN_COLOR := Color(0.22, 0.78, 0.34, 1.0)
const BLUE_COLOR := Color(0.18, 0.44, 0.90, 1.0)
const RED_COLOR := Color(0.90, 0.18, 0.15, 1.0)
const AMBER_COLOR := Color(0.95, 0.66, 0.16, 1.0)
const TERRAIN_COLOR := Color(0.24, 0.36, 0.20, 1.0)
const TERRAIN_DARK := Color(0.15, 0.24, 0.15, 1.0)
const GRID_COLOR := Color(0.56, 0.64, 0.68, 0.74)
const HILL_COLOR := Color(0.37, 0.48, 0.28, 1.0)
const ROUTE_COLOR := Color(0.92, 0.72, 0.18, 1.0)
const PANEL_COLOR := Color(0.035, 0.050, 0.075, 0.92)
const PANEL_BORDER := Color(0.36, 0.43, 0.50, 1.0)
const CELL_SIZE := 1.35
const GRID_COLS := 8
const GRID_ROWS := 8
const STATIC_OBJECT_SPECS := [
	{"label": "BLD1", "kind": "building"},
	{"label": "BLD2", "kind": "building"},
	{"label": "SOL1", "kind": "foot_soldiers"},
	{"label": "SOL2", "kind": "foot_soldiers"},
	{"label": "SHP1", "kind": "sheep"},
	{"label": "SHP2", "kind": "sheep"},
	{"label": "WOOD", "kind": "forest"},
	{"label": "TRK1", "kind": "truck"},
	{"label": "TWR1", "kind": "tower"},
	{"label": "TENT", "kind": "tent"},
]
const AIRCRAFT_OBJECT_SPECS := [
	{"label": "BLD1", "kind": "building"},
	{"label": "SOL1", "kind": "foot_soldiers"},
	{"label": "SHP1", "kind": "sheep"},
	{"label": "WOOD", "kind": "forest"},
	{"label": "TRK1", "kind": "truck"},
	{"label": "TWR1", "kind": "tower"},
	{"label": "TENT", "kind": "tent"},
]
const ROUTE_TEMPLATES := [
	[Vector3(1, 1, 5), Vector3(2, 2, 4), Vector3(3, 3, 3), Vector3(5, 3, 3), Vector3(6, 2, 4), Vector3(5, 1, 5)],
	[Vector3(1, 1, 2), Vector3(2, 2, 2), Vector3(4, 3, 3), Vector3(5, 3, 5), Vector3(4, 2, 6), Vector3(2, 1, 6)],
	[Vector3(1, 2, 4), Vector3(2, 3, 3), Vector3(4, 4, 2), Vector3(5, 3, 3), Vector3(6, 2, 5), Vector3(5, 1, 6)],
	[Vector3(1, 1, 3), Vector3(2, 2, 2), Vector3(4, 3, 2), Vector3(6, 3, 3), Vector3(6, 2, 5), Vector3(4, 1, 6)],
	[Vector3(1, 1, 5), Vector3(2, 2, 4), Vector3(3, 3, 2), Vector3(5, 4, 2), Vector3(6, 3, 4), Vector3(5, 2, 6)],
	[Vector3(2, 1, 6), Vector3(3, 2, 5), Vector3(4, 3, 3), Vector3(5, 3, 2), Vector3(6, 2, 3), Vector3(6, 1, 5)],
]
const STATIC_KINDS := ["landmark_grid", "scene_reconstruction"]
const AIRCRAFT_KINDS := ["aircraft_route_selection", "aircraft_continuation_selection", "aircraft_location_grid"]
const ALL_KINDS := ["landmark_grid", "scene_reconstruction", "aircraft_route_selection", "aircraft_continuation_selection", "aircraft_location_grid"]

var control_sender: Callable
var active := false
var paused := false
var run_key := ""
var completed_run_key := ""
var kind := "spatial_integration"
var test_code := "spatial_integration"
var phase := "scored"
var mode := "standard"
var session_seed := 1
var difficulty := 0.5
var duration_s := 180.0
var elapsed_s := 0.0
var part_started_at_s := 0.0
var stage_elapsed_s := 0.0
var progress_accum := 0.0
var parts: Array = ["static", "aircraft"]
var allowed_question_kinds: Array = ALL_KINDS.duplicate()
var static_study_s := 12.0
var aircraft_study_s := 15.0
var question_time_limit_s := 8.0
var practice_scenes_per_part := 0
var current_part_index := 0
var scene_counter := 0
var scene_index_in_part := 0
var study_view_index := 0
var question_index := 0
var stage := "study"
var typed_cell_token := ""
var selected_option_code := 0
var attempted := 0
var correct := 0
var total_score := 0.0
var max_score := 0.0
var static_attempted := 0
var static_correct := 0
var aircraft_attempted := 0
var aircraft_correct := 0
var event_log: Array = []
var current_scene := {}
var current_questions: Array = []
var current_question := {}
var scene_hash := 0
var route_hash := 0
var question_order_hash := 0
var option_order_hash := 0

var scene_root: Node3D
var terrain_root: Node3D
var object_root: Node3D
var route_root: Node3D
var hud_layer: CanvasLayer
var hud_label: Label
var prompt_label: Label
var entry_label: Label
var answer_root: Control
var material_cache := {}


func start(spec: Dictionary, sender: Callable) -> void:
	control_sender = sender
	var next_key := str(spec.get("run_key", ""))
	if active and next_key == run_key:
		return
	if next_key != "" and next_key == completed_run_key:
		return
	_clear_runtime()
	run_key = next_key
	test_code = str(spec.get("test_code", "spatial_integration"))
	phase = str(spec.get("phase", "scored")).to_lower()
	mode = str(spec.get("mode", "standard")).to_lower()
	session_seed = int(max(1, int(spec.get("session_seed", spec.get("seed", 1)))))
	difficulty = clampf(_float(spec.get("difficulty", 0.5)), 0.0, 1.0)
	duration_s = maxf(1.0, _float(spec.get("duration_s", 180.0)))
	var cfg: Dictionary = _as_dict(spec.get("config", {}))
	parts = _array_or_default(cfg.get("parts", []), ["static", "aircraft"])
	allowed_question_kinds = _array_or_default(cfg.get("allowed_question_kinds", []), ALL_KINDS)
	static_study_s = maxf(1.0, _float(cfg.get("static_study_s", 12.0)))
	aircraft_study_s = maxf(1.0, _float(cfg.get("aircraft_study_s", 15.0)))
	question_time_limit_s = maxf(1.0, _float(cfg.get("question_time_limit_s", 8.0)))
	practice_scenes_per_part = max(0, int(cfg.get("practice_scenes_per_part", 0)))
	elapsed_s = 0.0
	part_started_at_s = 0.0
	progress_accum = 0.0
	current_part_index = 0
	scene_counter = 0
	scene_index_in_part = 0
	event_log.clear()
	_reset_score()
	_build_nodes()
	active = true
	_send("godot_ready", {
		"run_key": run_key,
		"phase": phase,
		"kind": kind,
		"test_code": test_code,
		"parts": parts,
		"allowed_question_kinds": allowed_question_kinds,
	})
	_deal_next_scene()


func update_runtime(delta: float, camera: Camera3D) -> void:
	if not active or paused:
		return
	var dt := minf(maxf(delta, 0.0), 0.05)
	elapsed_s += dt
	stage_elapsed_s += dt
	if elapsed_s >= duration_s:
		_complete()
		return
	_update_camera(camera, dt)
	_update_hud()
	_check_stage_timeout()
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
	if stage != "question":
		return false
	if key == KEY_BACKSPACE:
		if typed_cell_token.length() > 0:
			typed_cell_token = typed_cell_token.substr(0, typed_cell_token.length() - 1)
			_refresh_entry_label()
		return true
	if key == KEY_ENTER or key == KEY_KP_PERIOD:
		_submit_current_entry()
		return true
	var digit := _digit_from_key(key)
	if digit >= 0:
		if str(current_question.get("answer_mode", "")) == "option_pick":
			if digit >= 1 and digit <= 4:
				selected_option_code = digit
				_submit_option(digit)
				return true
		elif digit >= 1 and digit <= 8 and typed_cell_token.length() < 3:
			typed_cell_token += str(digit)
			_refresh_entry_label()
			return true
	var letter := _letter_from_key(key)
	if letter != "" and str(current_question.get("answer_mode", "")) == "grid_click":
		typed_cell_token = letter + _digits_only(typed_cell_token)
		_refresh_entry_label()
		return true
	return false


func set_paused(value: bool) -> void:
	paused = bool(value)


func spatial_integration_runtime_marker() -> bool:
	return true


func _reset_score() -> void:
	attempted = 0
	correct = 0
	total_score = 0.0
	max_score = 0.0
	static_attempted = 0
	static_correct = 0
	aircraft_attempted = 0
	aircraft_correct = 0
	scene_hash = 0
	route_hash = 0
	question_order_hash = 0
	option_order_hash = 0


func _deal_next_scene() -> void:
	if current_part_index >= parts.size():
		_complete()
		return
	if _part_time_remaining_s() <= 0.0:
		_advance_part()
		return
	scene_counter += 1
	scene_index_in_part += 1
	var part := str(parts[current_part_index])
	current_scene = _build_scene(part)
	current_questions = _questions_for_scene(current_scene)
	if current_questions.is_empty():
		_advance_part()
		return
	question_index = 0
	current_question = _as_dict(current_questions[question_index])
	study_view_index = 0
	stage = "study"
	stage_elapsed_s = 0.0
	typed_cell_token = ""
	selected_option_code = 0
	_rebuild_world_scene(true)
	_rebuild_answer_overlay()


func _advance_part() -> void:
	current_part_index += 1
	scene_index_in_part = 0
	part_started_at_s = elapsed_s
	if current_part_index >= parts.size():
		_complete()
	else:
		_deal_next_scene()


func _advance_after_question() -> void:
	question_index += 1
	if question_index < current_questions.size():
		current_question = _as_dict(current_questions[question_index])
		stage = "question"
		stage_elapsed_s = 0.0
		typed_cell_token = ""
		selected_option_code = 0
		_rebuild_world_scene(false)
		_rebuild_answer_overlay()
		return
	_deal_next_scene()


func _check_stage_timeout() -> void:
	if stage == "study":
		if stage_elapsed_s < _study_step_s():
			return
		if study_view_index < 2:
			study_view_index += 1
			stage_elapsed_s = 0.0
			_rebuild_answer_overlay()
		else:
			stage = "question"
			stage_elapsed_s = 0.0
			typed_cell_token = ""
			selected_option_code = 0
			_rebuild_world_scene(false)
			_rebuild_answer_overlay()
		return
	if stage == "question" and stage_elapsed_s >= question_time_limit_s:
		_record_answer("TIMEOUT", false, true)
		_advance_after_question()


func _submit_current_entry() -> void:
	var answer_mode := str(current_question.get("answer_mode", ""))
	if answer_mode == "grid_click" and typed_cell_token.strip_edges() != "":
		_submit_grid_cell(typed_cell_token)
	elif answer_mode == "option_pick" and selected_option_code > 0:
		_submit_option(selected_option_code)


func _submit_grid_cell(raw_token: String) -> void:
	if stage != "question":
		return
	var norm := _normalize_cell_token(raw_token)
	if norm == "":
		return
	var expected := str(current_question.get("correct_answer_token", ""))
	_record_answer(norm, norm == expected, false)
	_advance_after_question()


func _submit_option(code: int) -> void:
	if stage != "question":
		return
	var expected := int(current_question.get("correct_code", 0))
	_record_answer(str(code), int(code) == expected, false)
	_advance_after_question()


func _record_answer(raw: String, is_correct: bool, is_timeout: bool) -> void:
	var score := 1.0 if is_correct else 0.0
	attempted += 1
	max_score += 1.0
	total_score += score
	if is_correct:
		correct += 1
	var part := str(current_scene.get("part", "static"))
	if part == "aircraft":
		aircraft_attempted += 1
		if is_correct:
			aircraft_correct += 1
	else:
		static_attempted += 1
		if is_correct:
			static_correct += 1
	var expected := str(current_question.get("correct_answer_token", current_question.get("correct_code", "")))
	var evt := {
		"family": "spatial_integration",
		"kind": str(current_question.get("kind", "question")),
		"phase": phase,
		"item_index": attempted - 1,
		"is_scored": phase == "scored",
		"is_correct": is_correct,
		"is_timeout": is_timeout,
		"response_time_ms": int(round(stage_elapsed_s * 1000.0)),
		"score": score,
		"max_score": 1.0,
		"occurred_at_ms": int(round(elapsed_s * 1000.0)),
		"prompt": str(current_question.get("stem", "")),
		"expected": expected,
		"response": raw,
		"extra": {
			"part": part,
			"scene_id": int(current_scene.get("scene_id", 0)),
			"question_kind": str(current_question.get("kind", "question")),
			"query_label": str(current_question.get("query_label", "")),
			"stage_elapsed_s": stage_elapsed_s,
		},
	}
	event_log.append(evt)
	_send("godot_event", {"run_key": run_key, "kind": kind, "test_code": test_code, "event": evt})


func _complete() -> void:
	if not active:
		return
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
		"renderer_backend": "godot_4",
		"godot_authority": "1",
		"godot_kind": kind,
		"godot_test_code": test_code,
		"godot_mode": mode,
		"scene_hash": scene_hash,
		"route_hash": route_hash,
		"question_order_hash": question_order_hash,
		"option_order_hash": option_order_hash,
		"static_attempted": static_attempted,
		"static_correct": static_correct,
		"aircraft_attempted": aircraft_attempted,
		"aircraft_correct": aircraft_correct,
	}
	var result := {
		"run_key": run_key,
		"kind": kind,
		"test_code": test_code,
		"phase": "results",
		"summary": summary,
		"metrics": metrics,
		"events": event_log.slice(max(0, event_log.size() - 360), event_log.size()),
	}
	if phase == "practice":
		_send("godot_phase_complete", {"run_key": run_key, "phase": "practice", "kind": kind, "test_code": test_code, "result": result})
		return
	_send("godot_complete", {"run_key": run_key, "phase": "results", "kind": kind, "test_code": test_code, "result": result})


func _build_scene(part: String) -> Dictionary:
	var scene_rng := _rng_for("scene:" + str(scene_counter) + ":" + part)
	var hills := _sample_hills(scene_rng)
	var landmarks := _sample_landmarks(scene_rng, STATIC_OBJECT_SPECS if part == "static" else AIRCRAFT_OBJECT_SPECS, 7 if part == "static" else 5)
	var route: Array = []
	var route_index := 0
	var aircraft_now := Vector3.ZERO
	var aircraft_next := Vector3.ZERO
	if part == "aircraft":
		var template_index := int(scene_rng.randi_range(0, ROUTE_TEMPLATES.size() - 1))
		for point in ROUTE_TEMPLATES[template_index]:
			route.append(point)
		route_index = int(scene_rng.randi_range(2, route.size() - 2))
		aircraft_now = route[route_index]
		aircraft_next = route[route_index + 1]
		route_hash = _hash_mix(route_hash, template_index + scene_counter * 31)
	var scene := {
		"scene_id": scene_counter,
		"part": part,
		"hills": hills,
		"landmarks": landmarks,
		"route": route,
		"route_current_index": route_index,
		"aircraft_now": aircraft_now,
		"aircraft_next": aircraft_next,
	}
	scene_hash = _hash_mix(scene_hash, _hash_scene(scene))
	return scene


func _questions_for_scene(scene: Dictionary) -> Array:
	var part := str(scene.get("part", "static"))
	var rng := _rng_for("questions:" + str(scene.get("scene_id", 0)))
	var questions: Array = []
	var allowed := _allowed_for_part(part)
	if part == "static":
		var landmarks: Array = scene.get("landmarks", [])
		var targets := _pick_landmark_targets(rng, landmarks)
		if _contains(allowed, "landmark_grid"):
			for target in targets:
				var lm := _as_dict(target)
				questions.append({
					"kind": "landmark_grid",
					"answer_mode": "grid_click",
					"stem": "Click the grid cell where " + str(lm.get("label", "")) + " was located.",
					"query_label": str(lm.get("label", "")),
					"correct_answer_token": _cell_token(int(lm.get("x", 0)), int(lm.get("y", 0))),
					"correct_point": Vector3(int(lm.get("x", 0)), 0, int(lm.get("y", 0))),
					"answer_map_landmarks": _context_landmarks(landmarks, str(lm.get("label", ""))),
				})
		if _contains(allowed, "scene_reconstruction"):
			var options := _static_reconstruction_options(rng, landmarks)
			var correct_code := _correct_code(options)
			questions.append({
				"kind": "scene_reconstruction",
				"answer_mode": "option_pick",
				"stem": "Which top-down map matches the studied landscape?",
				"query_label": "FULL SCENE",
				"correct_code": correct_code,
				"correct_answer_token": str(correct_code),
				"options": options,
			})
	else:
		var route: Array = scene.get("route", [])
		var now_point: Vector3 = scene.get("aircraft_now", Vector3.ZERO)
		var next_point: Vector3 = scene.get("aircraft_next", Vector3.ZERO)
		if _contains(allowed, "aircraft_route_selection"):
			var route_options := _route_options(rng, route, now_point, "route")
			var route_correct := _correct_code(route_options)
			questions.append({
				"kind": "aircraft_route_selection",
				"answer_mode": "option_pick",
				"stem": "Which route map matches the aircraft track shown in the studied views?",
				"query_label": "ROUTE",
				"correct_code": route_correct,
				"correct_answer_token": str(route_correct),
				"options": route_options,
			})
		if _contains(allowed, "aircraft_continuation_selection"):
			var continuation_options := _route_options(rng, route, next_point, "continuation")
			var continuation_correct := _correct_code(continuation_options)
			questions.append({
				"kind": "aircraft_continuation_selection",
				"answer_mode": "option_pick",
				"stem": "Which path continuation shows the aircraft in the correct next position?",
				"query_label": "CONTINUATION",
				"correct_code": continuation_correct,
				"correct_answer_token": str(continuation_correct),
				"options": continuation_options,
			})
		if _contains(allowed, "aircraft_location_grid"):
			questions.append({
				"kind": "aircraft_location_grid",
				"answer_mode": "grid_click",
				"stem": "Click the grid cell where the aircraft was located in the studied scene.",
				"query_label": "AIRCRAFT",
				"correct_answer_token": _cell_token(int(now_point.x), int(now_point.z)),
				"correct_point": now_point,
				"answer_map_landmarks": scene.get("landmarks", []),
				"answer_map_route_points": route,
			})
	question_order_hash = _hash_mix(question_order_hash, _hash_question_order(questions))
	return questions


func _allowed_for_part(part: String) -> Array:
	var defaults := STATIC_KINDS if part == "static" else AIRCRAFT_KINDS
	var out: Array = []
	for item in allowed_question_kinds:
		if _contains(defaults, str(item)):
			out.append(str(item))
	return out if not out.is_empty() else defaults.duplicate()


func _sample_hills(rng: RandomNumberGenerator) -> Array:
	var presets := [
		[{"label": "H1", "x": 2, "y": 2, "radius": 2, "height": 2}, {"label": "H2", "x": 5, "y": 3, "radius": 2, "height": 3}, {"label": "H3", "x": 4, "y": 6, "radius": 2, "height": 2}],
		[{"label": "H1", "x": 1, "y": 3, "radius": 2, "height": 2}, {"label": "H2", "x": 5, "y": 2, "radius": 3, "height": 3}, {"label": "H3", "x": 6, "y": 6, "radius": 2, "height": 2}],
		[{"label": "H1", "x": 2, "y": 5, "radius": 2, "height": 2}, {"label": "H2", "x": 4, "y": 2, "radius": 2, "height": 2}, {"label": "H3", "x": 6, "y": 4, "radius": 3, "height": 3}],
	]
	return (presets[int(rng.randi_range(0, presets.size() - 1))] as Array).duplicate(true)


func _sample_landmarks(rng: RandomNumberGenerator, specs: Array, count: int) -> Array:
	var available := specs.duplicate(true)
	var chosen_specs: Array = []
	var landmarks: Array = []
	while chosen_specs.size() < min(count, specs.size()) and not available.is_empty():
		var idx := int(rng.randi_range(0, available.size() - 1))
		chosen_specs.append(available[idx])
		available.remove_at(idx)
	var points: Array = []
	var occupancy := {}
	for spec in chosen_specs:
		var candidate := Vector2i(0, 0)
		var placed := false
		for _attempt in range(60):
			var duplicate_ok := points.size() > 0 and rng.randf() < 0.24
			if duplicate_ok:
				candidate = points[int(rng.randi_range(0, points.size() - 1))]
			else:
				candidate = Vector2i(int(rng.randi_range(1, GRID_COLS - 2)), int(rng.randi_range(1, GRID_ROWS - 2)))
			var key := str(candidate.x) + ":" + str(candidate.y)
			if int(occupancy.get(key, 0)) < 2:
				occupancy[key] = int(occupancy.get(key, 0)) + 1
				placed = true
				break
		if not placed:
			candidate = Vector2i(1, 1)
			var s := _as_dict(spec)
			points.append(candidate)
			landmarks.append({
				"label": str(s.get("label", "OBJ")),
				"kind": str(s.get("kind", "landmark")),
				"x": candidate.x,
				"y": candidate.y,
			})
	return landmarks


func _pick_landmark_targets(rng: RandomNumberGenerator, landmarks: Array) -> Array:
	var pool := landmarks.duplicate(true)
	var out: Array = []
	while out.size() < min(2, landmarks.size()) and not pool.is_empty():
		var idx := int(rng.randi_range(0, pool.size() - 1))
		out.append(pool[idx])
		pool.remove_at(idx)
	return out


func _context_landmarks(landmarks: Array, omit_label: String) -> Array:
	var out: Array = []
	var keep_fraction := maxf(0.0, 1.0 - difficulty)
	var keep_count := int(round(float(max(0, landmarks.size() - 1)) * keep_fraction))
	for landmark in landmarks:
		var lm := _as_dict(landmark)
		if str(lm.get("label", "")).to_upper() != omit_label.to_upper() and out.size() < keep_count:
			out.append(lm)
	return out


func _static_reconstruction_options(rng: RandomNumberGenerator, landmarks: Array) -> Array:
	var candidates := [
		{"answer_token": "correct", "landmarks": landmarks},
		{"answer_token": "mirrored", "landmarks": _mirror_landmarks(landmarks)},
		{"answer_token": "rotated", "landmarks": _rotate_landmarks(landmarks)},
		{"answer_token": "swapped", "landmarks": _swap_landmarks(landmarks)},
	]
	return _ordered_options(rng, candidates)


func _route_options(rng: RandomNumberGenerator, route: Array, aircraft_point: Vector3, variant: String) -> Array:
	var shifted_aircraft := Vector3(clampi(int(aircraft_point.x) + 1, 0, GRID_COLS - 1), aircraft_point.y, clampi(int(aircraft_point.z) - 1, 0, GRID_ROWS - 1))
	var candidates := [
		{"answer_token": "correct", "route": route, "aircraft": aircraft_point},
		{"answer_token": "mirrored", "route": _mirror_route(route), "aircraft": _mirror_point(aircraft_point)},
		{"answer_token": "rotated", "route": _rotate_route(route), "aircraft": _rotate_point(aircraft_point)},
		{"answer_token": "loop_timing" if variant == "continuation" else "shifted", "route": _shift_route(route), "aircraft": shifted_aircraft},
	]
	return _ordered_options(rng, candidates)


func _ordered_options(rng: RandomNumberGenerator, candidates: Array) -> Array:
	var indices := [0, 1, 2, 3]
	var options: Array = []
	var code := 1
	while not indices.is_empty():
		var idx := int(rng.randi_range(0, indices.size() - 1))
		var candidate := _as_dict(candidates[indices[idx]])
		indices.remove_at(idx)
		candidate["code"] = code
		candidate["label"] = str(code)
		options.append(candidate)
		option_order_hash = _hash_mix(option_order_hash, _string_salt(str(candidate.get("answer_token", ""))) + code)
		code += 1
	return options


func _correct_code(options: Array) -> int:
	for option in options:
		var opt := _as_dict(option)
		if str(opt.get("answer_token", "")) == "correct":
			return int(opt.get("code", 0))
	return 0


func _mirror_landmarks(landmarks: Array) -> Array:
	var out: Array = []
	for landmark in landmarks:
		var lm := _as_dict(landmark)
		var copy := lm.duplicate(true)
		copy["x"] = GRID_COLS - 1 - int(lm.get("x", 0))
		out.append(copy)
	return out


func _rotate_landmarks(landmarks: Array) -> Array:
	var out: Array = []
	for landmark in landmarks:
		var lm := _as_dict(landmark)
		var copy := lm.duplicate(true)
		copy["x"] = GRID_COLS - 1 - int(lm.get("y", 0))
		copy["y"] = int(lm.get("x", 0))
		out.append(copy)
	return out


func _swap_landmarks(landmarks: Array) -> Array:
	var out := landmarks.duplicate(true)
	if out.size() >= 2:
		var a := _as_dict(out[0])
		var b := _as_dict(out[1])
		var ax := int(a.get("x", 0))
		var ay := int(a.get("y", 0))
		a["x"] = int(b.get("x", 0))
		a["y"] = int(b.get("y", 0))
		b["x"] = ax
		b["y"] = ay
		out[0] = a
		out[1] = b
	return out


func _mirror_route(route: Array) -> Array:
	var out: Array = []
	for point in route:
		out.append(_mirror_point(point))
	return out


func _rotate_route(route: Array) -> Array:
	var out: Array = []
	for point in route:
		out.append(_rotate_point(point))
	return out


func _shift_route(route: Array) -> Array:
	var out: Array = []
	for point in route:
		var p: Vector3 = point
		out.append(Vector3(clampi(int(p.x) + 1, 0, GRID_COLS - 1), p.y, clampi(int(p.z) - 1, 0, GRID_ROWS - 1)))
	return out


func _mirror_point(point: Vector3) -> Vector3:
	return Vector3(GRID_COLS - 1 - int(point.x), point.y, point.z)


func _rotate_point(point: Vector3) -> Vector3:
	return Vector3(GRID_COLS - 1 - int(point.z), point.y, int(point.x))


func _rebuild_world_scene(show_study_labels: bool) -> void:
	_clear_children(terrain_root)
	_clear_children(object_root)
	_clear_children(route_root)
	_draw_grid_terrain()
	for hill in current_scene.get("hills", []):
		_draw_hill(_as_dict(hill))
	var landmarks: Array = current_scene.get("landmarks", [])
	for landmark in landmarks:
		_draw_landmark(_as_dict(landmark), show_study_labels)
	if str(current_scene.get("part", "")) == "aircraft":
		_draw_route(current_scene.get("route", []))
		_draw_aircraft(current_scene.get("aircraft_now", Vector3.ZERO), GREEN_COLOR, true)


func _draw_grid_terrain() -> void:
	_make_box(terrain_root, "Ground", Vector3(0.0, -0.08, 0.0), Vector3(6.1, 0.08, 6.1), TERRAIN_COLOR)
	for x in range(GRID_COLS + 1):
		var wx := (float(x) - 4.0) * CELL_SIZE
		_make_box(terrain_root, "GridX", Vector3(wx, 0.025, 0.0), Vector3(0.012, 0.012, 5.4), GRID_COLOR)
	for y in range(GRID_ROWS + 1):
		var wz := (float(y) - 4.0) * CELL_SIZE
		_make_box(terrain_root, "GridY", Vector3(0.0, 0.03, wz), Vector3(5.4, 0.012, 0.012), GRID_COLOR)


func _draw_hill(hill: Dictionary) -> void:
	var pos := _grid_to_world(int(hill.get("x", 0)), int(hill.get("y", 0)), 0)
	var radius := 0.34 + float(hill.get("radius", 1)) * 0.20
	var height := 0.18 + float(hill.get("height", 1)) * 0.15
	_make_sphere(terrain_root, "Hill", pos + Vector3(0.0, height * 0.35, 0.0), Vector3(radius, height, radius), HILL_COLOR)


func _draw_landmark(landmark: Dictionary, show_label: bool) -> void:
	var kind_name := str(landmark.get("kind", "landmark"))
	var label := str(landmark.get("label", "OBJ"))
	var root := Node3D.new()
	root.name = label
	root.position = _grid_to_world(int(landmark.get("x", 0)), int(landmark.get("y", 0)), 0)
	object_root.add_child(root)
	if kind_name == "building":
		_make_box(root, "Building", Vector3(0.0, 0.38, 0.0), Vector3(0.34, 0.38, 0.34), Color(0.66, 0.60, 0.48, 1.0))
		_make_box(root, "Roof", Vector3(0.0, 0.82, 0.0), Vector3(0.42, 0.10, 0.42), RED_COLOR.darkened(0.25))
	elif kind_name == "foot_soldiers":
		_make_sphere(root, "SoldierHead", Vector3(0.0, 0.45, 0.0), Vector3(0.10, 0.10, 0.10), AMBER_COLOR)
		_make_box(root, "SoldierBody", Vector3(0.0, 0.25, 0.0), Vector3(0.10, 0.20, 0.08), BLUE_COLOR.darkened(0.20))
	elif kind_name == "sheep":
		_make_sphere(root, "Sheep", Vector3(0.0, 0.22, 0.0), Vector3(0.20, 0.13, 0.14), WHITE_COLOR)
	elif kind_name == "forest":
		for i in range(4):
			var off := Vector3(float(i % 2) * 0.28 - 0.14, 0.0, float(i / 2) * 0.28 - 0.14)
			_make_box(root, "TreeTrunk", off + Vector3(0.0, 0.18, 0.0), Vector3(0.04, 0.18, 0.04), Color(0.29, 0.16, 0.07, 1.0))
			_make_sphere(root, "TreeTop", off + Vector3(0.0, 0.44, 0.0), Vector3(0.17, 0.17, 0.17), Color(0.14, 0.38, 0.16, 1.0))
	elif kind_name == "truck":
		_make_box(root, "Truck", Vector3(0.0, 0.20, 0.0), Vector3(0.34, 0.16, 0.46), Color(0.22, 0.34, 0.30, 1.0))
		_make_box(root, "Cab", Vector3(0.0, 0.36, -0.14), Vector3(0.24, 0.12, 0.18), Color(0.30, 0.44, 0.38, 1.0))
	elif kind_name == "tower":
		_make_box(root, "Tower", Vector3(0.0, 0.55, 0.0), Vector3(0.12, 0.55, 0.12), Color(0.54, 0.56, 0.55, 1.0))
		_make_sphere(root, "TowerTop", Vector3(0.0, 1.16, 0.0), Vector3(0.20, 0.20, 0.20), Color(0.72, 0.74, 0.72, 1.0))
	else:
		_make_box(root, "Tent", Vector3(0.0, 0.20, 0.0), Vector3(0.32, 0.20, 0.32), AMBER_COLOR.darkened(0.18))
	if show_label:
		var tag := Label3D.new()
		tag.name = "StudyLabel"
		tag.text = label
		tag.position = Vector3(0.0, 1.22, 0.0)
		tag.pixel_size = 0.028
		tag.modulate = WHITE_COLOR
		tag.outline_modulate = Color(0.02, 0.02, 0.02, 1.0)
		tag.outline_size = 8
		root.add_child(tag)


func _draw_route(route: Array) -> void:
	var previous := Vector3.ZERO
	var have_previous := false
	for idx in range(route.size()):
		var p: Vector3 = route[idx]
		var current := _grid_to_world(int(p.x), int(p.z), int(p.y))
		_make_sphere(route_root, "RoutePoint", current, Vector3(0.10, 0.10, 0.10), ROUTE_COLOR)
		if have_previous:
			_make_segment(route_root, previous, current, ROUTE_COLOR, 0.035)
		previous = current
		have_previous = true


func _draw_aircraft(point, color: Color, label_current: bool) -> void:
	var p: Vector3 = point
	var root := Node3D.new()
	root.name = "Aircraft"
	root.position = _grid_to_world(int(p.x), int(p.z), int(p.y))
	route_root.add_child(root)
	_make_box(root, "Fuselage", Vector3(0.0, 0.0, 0.0), Vector3(0.11, 0.09, 0.42), color)
	_make_box(root, "Wing", Vector3(0.0, 0.0, 0.0), Vector3(0.56, 0.025, 0.10), color.lightened(0.12))
	_make_box(root, "Tail", Vector3(0.0, 0.11, 0.30), Vector3(0.06, 0.14, 0.08), color.darkened(0.10))
	if label_current:
		var tag := Label3D.new()
		tag.text = "AIRCRAFT"
		tag.position = Vector3(0.0, 0.48, 0.0)
		tag.pixel_size = 0.028
		tag.modulate = WHITE_COLOR
		tag.outline_size = 8
		root.add_child(tag)


func _rebuild_answer_overlay() -> void:
	_clear_children(answer_root)
	typed_cell_token = "" if stage != "question" else typed_cell_token
	selected_option_code = 0 if stage != "question" else selected_option_code
	var panel := PanelContainer.new()
	panel.name = "SpatialPanel"
	panel.position = Vector2(24, 350)
	panel.custom_minimum_size = Vector2(912, 166)
	answer_root.add_child(panel)
	var box := VBoxContainer.new()
	box.add_theme_constant_override("separation", 6)
	panel.add_child(box)
	prompt_label = Label.new()
	prompt_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	prompt_label.add_theme_font_size_override("font_size", 18)
	prompt_label.add_theme_color_override("font_color", WHITE_COLOR)
	box.add_child(prompt_label)
	entry_label = Label.new()
	entry_label.add_theme_font_size_override("font_size", 16)
	entry_label.add_theme_color_override("font_color", TEXT_MUTED)
	box.add_child(entry_label)
	if stage == "study":
		prompt_label.text = _study_prompt()
		entry_label.text = "Study the frozen scene. Next view advances automatically."
		return
	prompt_label.text = str(current_question.get("stem", "Answer the question."))
	if str(current_question.get("answer_mode", "")) == "grid_click":
		_build_grid_buttons(box)
	else:
		_build_option_buttons(box)
	_refresh_entry_label()


func _build_grid_buttons(parent: Control) -> void:
	var grid := GridContainer.new()
	grid.columns = GRID_COLS
	grid.custom_minimum_size = Vector2(650, 0)
	parent.add_child(grid)
	var context_landmarks: Array = current_question.get("answer_map_landmarks", [])
	var context_route: Array = current_question.get("answer_map_route_points", [])
	for y in range(GRID_ROWS):
		for x in range(GRID_COLS):
			var token := _cell_token(x, y)
			var button := Button.new()
			button.custom_minimum_size = Vector2(72, 26)
			button.text = token + _cell_context_suffix(x, y, context_landmarks, context_route)
			button.pressed.connect(Callable(self, "_submit_grid_cell").bind(token))
			grid.add_child(button)


func _build_option_buttons(parent: Control) -> void:
	var options: Array = current_question.get("options", [])
	var row := HBoxContainer.new()
	row.add_theme_constant_override("separation", 8)
	parent.add_child(row)
	for option in options:
		var opt := _as_dict(option)
		var code := int(opt.get("code", 0))
		var button := Button.new()
		button.custom_minimum_size = Vector2(210, 82)
		button.text = str(code) + "\n" + _option_card_label(opt)
		button.pressed.connect(Callable(self, "_submit_option").bind(code))
		row.add_child(button)


func _refresh_entry_label() -> void:
	if entry_label == null:
		return
	if stage != "question":
		return
	if str(current_question.get("answer_mode", "")) == "grid_click":
		entry_label.text = "Typed grid cell: " + (typed_cell_token if typed_cell_token != "" else "_") + "   Enter submits"
	else:
		entry_label.text = "Press 1-4 or click a card."


func _study_prompt() -> String:
	var labels := ["Top-down map", "Profile / oblique view", "Rotated reference view"]
	var part := "Landscape" if str(current_scene.get("part", "static")) == "static" else "Aircraft route"
	return part + " study: " + labels[clampi(study_view_index, 0, labels.size() - 1)] + " (" + str(study_view_index + 1) + "/3)"


func _cell_context_suffix(x: int, y: int, landmarks: Array, route: Array) -> String:
	var marks: Array = []
	for landmark in landmarks:
		var lm := _as_dict(landmark)
		if int(lm.get("x", -1)) == x and int(lm.get("y", -1)) == y:
			marks.append(_icon_for_kind(str(lm.get("kind", ""))))
	for point in route:
		var p: Vector3 = point
		if int(p.x) == x and int(p.z) == y:
			marks.append("->")
	if marks.is_empty():
		return ""
	return " " + "/".join(marks)


func _option_card_label(option: Dictionary) -> String:
	var token := str(option.get("answer_token", "map"))
	if option.has("route"):
		var aircraft: Vector3 = option.get("aircraft", Vector3.ZERO)
		return "Route " + token + "\nAircraft " + _cell_token(int(aircraft.x), int(aircraft.z))
	var landmarks: Array = option.get("landmarks", [])
	return "Map " + token + "\nObjects " + str(landmarks.size())


func _update_camera(camera: Camera3D, dt: float) -> void:
	if camera == null:
		return
	camera.projection = Camera3D.PROJECTION_PERSPECTIVE
	var target := Vector3(0.0, 0.0, 0.0)
	var pos := Vector3(0.0, 13.8, 0.02)
	if stage == "study":
		if study_view_index == 1:
			pos = Vector3(9.8, 6.8, 8.8)
		elif study_view_index == 2:
			pos = Vector3(-8.6, 8.0, 9.6)
		else:
			pos = Vector3(0.0, 14.5, 0.02)
	var blend := clampf(dt * 6.0, 0.08, 0.30)
	camera.position = camera.position.lerp(pos, blend)
	camera.look_at_from_position(camera.position, target, Vector3.UP)


func _update_hud() -> void:
	if hud_label == null:
		return
	var remaining := maxf(0.0, duration_s - elapsed_s)
	var stage_remaining := _stage_remaining_s()
	hud_label.text = "Spatial Integration | " + phase + " | " + str(int(ceil(remaining))) + "s | " + str(correct) + "/" + str(attempted) + " | " + stage.capitalize() + " " + str(int(ceil(stage_remaining))) + "s"


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
			"stage": stage,
			"part": str(current_scene.get("part", "")),
			"scene_hash": scene_hash,
			"route_hash": route_hash,
			"question_order_hash": question_order_hash,
			"option_order_hash": option_order_hash,
		},
	})


func _build_nodes() -> void:
	scene_root = Node3D.new()
	scene_root.name = "SpatialIntegrationScene"
	add_child(scene_root)
	terrain_root = Node3D.new()
	terrain_root.name = "Terrain"
	scene_root.add_child(terrain_root)
	object_root = Node3D.new()
	object_root.name = "Landmarks"
	scene_root.add_child(object_root)
	route_root = Node3D.new()
	route_root.name = "Route"
	scene_root.add_child(route_root)
	hud_layer = CanvasLayer.new()
	hud_layer.layer = 5
	add_child(hud_layer)
	hud_label = Label.new()
	hud_label.position = Vector2(12, 88)
	hud_label.size = Vector2(920, 42)
	hud_label.add_theme_font_size_override("font_size", 18)
	hud_label.add_theme_color_override("font_color", WHITE_COLOR)
	hud_layer.add_child(hud_label)
	answer_root = Control.new()
	answer_root.name = "AnswerOverlay"
	answer_root.mouse_filter = Control.MOUSE_FILTER_IGNORE
	hud_layer.add_child(answer_root)


func _clear_runtime() -> void:
	for child in get_children():
		child.queue_free()
	active = false
	paused = false
	scene_root = null
	terrain_root = null
	object_root = null
	route_root = null
	hud_layer = null
	hud_label = null
	prompt_label = null
	entry_label = null
	answer_root = null


func _make_box(parent: Node, name_value: String, pos: Vector3, scale_value: Vector3, color: Color) -> MeshInstance3D:
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


func _make_sphere(parent: Node, name_value: String, pos: Vector3, scale_value: Vector3, color: Color) -> MeshInstance3D:
	var mesh := SphereMesh.new()
	mesh.radial_segments = 12
	mesh.rings = 6
	var node := MeshInstance3D.new()
	node.name = name_value
	node.mesh = mesh
	node.position = pos
	node.scale = scale_value
	node.material_override = _material(color)
	parent.add_child(node)
	return node


func _make_segment(parent: Node, a: Vector3, b: Vector3, color: Color, width: float) -> void:
	if a.distance_to(b) < 0.01:
		return
	var center := (a + b) * 0.5
	var node := _make_box(parent, "Segment", center, Vector3(width, width, a.distance_to(b) * 0.5), color)
	node.look_at(b, Vector3.UP)


func _material(color: Color) -> StandardMaterial3D:
	var key := color.to_html(true)
	if material_cache.has(key):
		return material_cache[key]
	var mat := StandardMaterial3D.new()
	mat.albedo_color = color
	mat.roughness = 0.86
	if color.a < 0.999:
		mat.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	material_cache[key] = mat
	return mat


func _grid_to_world(x: int, y: int, alt: int) -> Vector3:
	return Vector3((float(x) - 3.5) * CELL_SIZE, float(alt) * 0.58 + 0.05, (float(y) - 3.5) * CELL_SIZE)


func _cell_token(x: int, y: int) -> String:
	return String.chr(65 + int(y)) + str(int(x) + 1)


func _normalize_cell_token(raw: String) -> String:
	var text := str(raw).strip_edges().to_upper()
	var letters := ""
	var digits := ""
	for i in range(text.length()):
		var ch := text.substr(i, 1)
		if ch >= "A" and ch <= "Z":
			letters += ch
		elif ch >= "0" and ch <= "9":
			digits += ch
	if letters.length() != 1 or digits == "":
		return ""
	var row := letters.unicode_at(0) - 65
	var col := int(digits) - 1
	if row < 0 or row >= GRID_ROWS or col < 0 or col >= GRID_COLS:
		return ""
	return _cell_token(col, row)


func _digits_only(text: String) -> String:
	var out := ""
	for i in range(text.length()):
		var ch := text.substr(i, 1)
		if ch >= "0" and ch <= "9":
			out += ch
	return out


func _digit_from_key(key: Key) -> int:
	match key:
		KEY_0, KEY_KP_0:
			return 0
		KEY_1, KEY_KP_1:
			return 1
		KEY_2, KEY_KP_2:
			return 2
		KEY_3, KEY_KP_3:
			return 3
		KEY_4, KEY_KP_4:
			return 4
		KEY_5, KEY_KP_5:
			return 5
		KEY_6, KEY_KP_6:
			return 6
		KEY_7, KEY_KP_7:
			return 7
		KEY_8, KEY_KP_8:
			return 8
		KEY_9, KEY_KP_9:
			return 9
	return -1


func _letter_from_key(key: Key) -> String:
	match key:
		KEY_A:
			return "A"
		KEY_B:
			return "B"
		KEY_C:
			return "C"
		KEY_D:
			return "D"
		KEY_E:
			return "E"
		KEY_F:
			return "F"
		KEY_G:
			return "G"
		KEY_H:
			return "H"
	return ""


func _icon_for_kind(kind_name: String) -> String:
	var token := kind_name.to_lower()
	if token == "building":
		return "B"
	if token == "foot_soldiers":
		return "S"
	if token == "sheep":
		return "P"
	if token == "forest":
		return "W"
	if token == "truck":
		return "T"
	if token == "tower":
		return "R"
	if token == "tent":
		return "N"
	return "O"


func _study_step_s() -> float:
	var study_total := aircraft_study_s if str(current_scene.get("part", "static")) == "aircraft" else static_study_s
	return study_total / 3.0


func _stage_remaining_s() -> float:
	var total := question_time_limit_s if stage == "question" else _study_step_s()
	return maxf(0.0, total - stage_elapsed_s)


func _part_time_remaining_s() -> float:
	if parts.size() <= 1:
		return maxf(0.0, duration_s - elapsed_s)
	var part := str(parts[current_part_index]) if current_part_index < parts.size() else "static"
	var static_weight := 10.0
	var aircraft_weight := 13.0
	var total_weight := 0.0
	for p in parts:
		total_weight += aircraft_weight if str(p) == "aircraft" else static_weight
	var part_duration := duration_s * ((aircraft_weight if part == "aircraft" else static_weight) / maxf(1.0, total_weight))
	return maxf(0.0, part_duration - (elapsed_s - part_started_at_s))


func _rng_for(stream: String) -> RandomNumberGenerator:
	var rng := RandomNumberGenerator.new()
	rng.seed = int(session_seed + _string_salt(stream) * 101 + scene_counter * 4099)
	return rng


func _string_salt(value: String) -> int:
	var out := 0
	for i in range(value.length()):
		out = int((out * 131 + value.unicode_at(i)) % 1000003)
	return max(1, out)


func _hash_scene(scene: Dictionary) -> int:
	var value := int(scene.get("scene_id", 0)) * 17 + _string_salt(str(scene.get("part", "")))
	for landmark in scene.get("landmarks", []):
		var lm := _as_dict(landmark)
		value = _hash_mix(value, _string_salt(str(lm.get("label", ""))) + int(lm.get("x", 0)) * 11 + int(lm.get("y", 0)) * 19)
	return value


func _hash_question_order(questions: Array) -> int:
	var value := 0
	for question in questions:
		var q := _as_dict(question)
		value = _hash_mix(value, _string_salt(str(q.get("kind", ""))) + _string_salt(str(q.get("query_label", ""))))
	return value


func _hash_mix(value: int, part: int) -> int:
	return int((int(value) * 1103515245 + int(part) * 12345 + 97) % 2147483647)


func _array_or_default(value, fallback: Array) -> Array:
	if typeof(value) != TYPE_ARRAY:
		return fallback.duplicate()
	var out: Array = []
	for item in value:
		out.append(str(item))
	return out if not out.is_empty() else fallback.duplicate()


func _contains(values: Array, target: String) -> bool:
	for item in values:
		if str(item) == target:
			return true
	return false


func _clear_children(node: Node) -> void:
	if node == null:
		return
	for child in node.get_children():
		child.queue_free()


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

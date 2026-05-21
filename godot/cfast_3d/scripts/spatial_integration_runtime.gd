extends Node3D

const ChunkMapGenerator = preload("res://scripts/chunk_map_generator.gd")
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
const ANSWER_GRID_COLS := 3
const ANSWER_GRID_ROWS := 3
const STATIC_OBJECT_SPECS := [
	{"label": "BLD1", "kind": "building"},
	{"label": "BLD2", "kind": "building"},
	{"label": "HUM1", "kind": "human"},
	{"label": "HUM2", "kind": "human"},
	{"label": "SHP1", "kind": "sheep"},
	{"label": "SHP2", "kind": "sheep"},
	{"label": "WOOD", "kind": "forest"},
	{"label": "VEH1", "kind": "vehicle"},
	{"label": "VEH2", "kind": "vehicle"},
	{"label": "TWR1", "kind": "tower"},
	{"label": "TWR2", "kind": "tower"},
	{"label": "TENT", "kind": "tent"},
	{"label": "HGR1", "kind": "building"},
	{"label": "RAD1", "kind": "tower"},
]
const AIRCRAFT_OBJECT_SPECS := [
	{"label": "BLD1", "kind": "building"},
	{"label": "BLD2", "kind": "building"},
	{"label": "HUM1", "kind": "human"},
	{"label": "HUM2", "kind": "human"},
	{"label": "SHP1", "kind": "sheep"},
	{"label": "SHP2", "kind": "sheep"},
	{"label": "WOOD", "kind": "forest"},
	{"label": "VEH1", "kind": "vehicle"},
	{"label": "VEH2", "kind": "vehicle"},
	{"label": "TWR1", "kind": "tower"},
	{"label": "TENT", "kind": "tent"},
	{"label": "RAD1", "kind": "tower"},
]
const ROUTE_TEMPLATES := [
	[Vector3(1, 1, 5), Vector3(2, 2, 4), Vector3(3, 3, 3), Vector3(5, 3, 3), Vector3(6, 2, 4), Vector3(5, 1, 5)],
	[Vector3(1, 1, 2), Vector3(2, 2, 2), Vector3(4, 3, 3), Vector3(5, 3, 5), Vector3(4, 2, 6), Vector3(2, 1, 6)],
	[Vector3(1, 2, 4), Vector3(2, 3, 3), Vector3(4, 4, 2), Vector3(5, 3, 3), Vector3(6, 2, 5), Vector3(5, 1, 6)],
	[Vector3(1, 1, 3), Vector3(2, 2, 2), Vector3(4, 3, 2), Vector3(6, 3, 3), Vector3(6, 2, 5), Vector3(4, 1, 6)],
	[Vector3(1, 1, 5), Vector3(2, 2, 4), Vector3(3, 3, 2), Vector3(5, 4, 2), Vector3(6, 3, 4), Vector3(5, 2, 6)],
	[Vector3(2, 1, 6), Vector3(3, 2, 5), Vector3(4, 3, 3), Vector3(5, 3, 2), Vector3(6, 2, 3), Vector3(6, 1, 5)],
]
const OBJECT_MEMORY_KINDS := ["building", "vehicle", "human", "sheep", "tower"]
const RELATION_CLOSE_DISTANCE_CELLS := 4.25
const AIRCRAFT_COLOR_LABELS := ["RED", "BLUE", "AMBER", "WHITE", "GREEN"]
const STATIC_KINDS := ["scene_reconstruction", "scene_presence", "viewpoint_match", "object_count", "object_relation"]
const AIRCRAFT_KINDS := ["aircraft_route_selection", "aircraft_continuation_selection", "aircraft_color_route_selection", "aircraft_count", "aircraft_presence", "aircraft_order"]
const ALL_KINDS := ["scene_reconstruction", "scene_presence", "viewpoint_match", "object_count", "object_relation", "aircraft_route_selection", "aircraft_continuation_selection", "aircraft_color_route_selection", "aircraft_count", "aircraft_presence", "aircraft_order"]

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
var question_time_limit_s := 0.0
var practice_scenes_per_part := 0
var current_part_index := 0
var scene_counter := 0
var scene_index_in_part := 0
var study_view_index := 0
var study_orientation_index := 0
var question_index := 0
var stage := "study"
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
var chunk_map := {}
var chunked_generation := true
var grid_cols := GRID_COLS
var grid_rows := GRID_ROWS
var answer_grid_cols := ANSWER_GRID_COLS
var answer_grid_rows := ANSWER_GRID_ROWS
var chunk_grid_cols := 8
var chunk_grid_rows := 8
var chunk_pack := "rural_mixed_v1"
var asset_spawn_policy := "socketed"
var terrain_pipeline := "si_large_scene_v2"

var scene_root: Node3D
var terrain_root: Node3D
var object_root: Node3D
var route_root: Node3D
var hud_layer: CanvasLayer
var hud_label: Label
var north_marker_root: Node3D
var prompt_label: Label
var entry_label: Label
var answer_root: Control
var material_cache := {}
var aircraft_track_nodes: Array = []


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
	question_time_limit_s = maxf(0.0, _float(cfg.get("question_time_limit_s", 0.0)))
	practice_scenes_per_part = max(0, int(cfg.get("practice_scenes_per_part", 0)))
	chunked_generation = bool(cfg.get("chunked_generation", true))
	grid_cols = clampi(int(cfg.get("grid_cols", 24)), 2, 24)
	grid_rows = clampi(int(cfg.get("grid_rows", 24)), 2, 24)
	answer_grid_cols = clampi(int(cfg.get("answer_grid_cols", ANSWER_GRID_COLS)), 1, grid_cols)
	answer_grid_rows = clampi(int(cfg.get("answer_grid_rows", ANSWER_GRID_ROWS)), 1, grid_rows)
	chunk_grid_cols = clampi(int(cfg.get("chunk_grid_cols", grid_cols)), 2, 24)
	chunk_grid_rows = clampi(int(cfg.get("chunk_grid_rows", grid_rows)), 2, 24)
	chunk_pack = str(cfg.get("chunk_pack", "rural_mixed_v1"))
	asset_spawn_policy = str(cfg.get("asset_spawn_policy", "socketed")).to_lower()
	terrain_pipeline = str(cfg.get("terrain_pipeline", "si_large_scene_v2"))
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
	_update_north_marker()
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
	if elapsed_s >= duration_s and stage != "question":
		_complete()
		return
	_update_camera(camera, dt)
	_update_aircraft_track_nodes()
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
	study_orientation_index = _study_orientation_for_scene()
	stage = "study"
	stage_elapsed_s = 0.0
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
			selected_option_code = 0
			_rebuild_world_scene(false)
			_rebuild_answer_overlay()
		return


func _submit_current_entry() -> void:
	var answer_mode := str(current_question.get("answer_mode", ""))
	if answer_mode == "option_pick" and selected_option_code > 0:
		_submit_option(selected_option_code)


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
		"chunked_generation": chunked_generation,
		"chunk_pack": chunk_pack,
		"asset_spawn_policy": asset_spawn_policy,
		"terrain_pipeline": terrain_pipeline,
		"chunk_map_hash": int(current_scene.get("chunk_hash", 0)),
		"chunk_rule_violations": int(current_scene.get("chunk_rule_violations", 0)),
		"hill_cell_count": int(current_scene.get("hill_cell_count", 0)),
		"hill_cluster_count": int(current_scene.get("hill_cluster_count", 0)),
		"visible_option_count": _visible_option_count(),
		"question_kind": str(current_question.get("kind", "")),
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
	var local_chunk_map := {}
	if chunked_generation:
		local_chunk_map = ChunkMapGenerator.generate({
			"seed": session_seed + scene_counter * 97 + (17 if part == "aircraft" else 0),
			"cols": chunk_grid_cols,
			"rows": chunk_grid_rows,
			"pack": chunk_pack,
			"difficulty": difficulty,
			"purpose": "spatial_integration",
			"terrain_pipeline": terrain_pipeline,
		})
		chunk_map = local_chunk_map
	var hills: Array = local_chunk_map.get("hill_clusters", []) if chunked_generation else _sample_hills(scene_rng)
	var landmark_count: int = 10 if part == "static" else 7
	var landmarks: Array = _chunk_landmarks(scene_rng, local_chunk_map, STATIC_OBJECT_SPECS if part == "static" else AIRCRAFT_OBJECT_SPECS, landmark_count) if chunked_generation else _sample_landmarks(scene_rng, STATIC_OBJECT_SPECS if part == "static" else AIRCRAFT_OBJECT_SPECS, landmark_count)
	var route: Array = []
	var route_index := 0
	var aircraft_now := Vector3.ZERO
	var aircraft_next := Vector3.ZERO
	var aircraft_tracks: Array = []
	if part == "aircraft":
		if chunked_generation:
			route = _chunk_aircraft_route(local_chunk_map)
		if route.size() < 4:
			var template_index := int(scene_rng.randi_range(0, ROUTE_TEMPLATES.size() - 1))
			for point in ROUTE_TEMPLATES[template_index]:
				route.append(point)
			route_hash = _hash_mix(route_hash, template_index + scene_counter * 31)
		else:
			route_hash = _hash_mix(route_hash, int(local_chunk_map.get("chunk_hash", 0)))
		route_index = int(scene_rng.randi_range(2, route.size() - 2))
		aircraft_now = route[route_index]
		aircraft_next = route[route_index + 1]
		aircraft_tracks = _build_aircraft_tracks(scene_rng, route, route_index)
		if not aircraft_tracks.is_empty():
			var primary := _as_dict(aircraft_tracks[0])
			route = primary.get("route", route)
			route_index = int(primary.get("current_index", route_index))
			aircraft_now = primary.get("current", aircraft_now)
			aircraft_next = primary.get("next", aircraft_next)
	var scene := {
		"scene_id": scene_counter,
		"part": part,
		"hills": hills,
		"landmarks": landmarks,
		"route": route,
		"route_current_index": route_index,
		"aircraft_now": aircraft_now,
		"aircraft_next": aircraft_next,
		"aircraft_tracks": aircraft_tracks,
		"chunk_map": local_chunk_map,
		"chunk_hash": int(local_chunk_map.get("chunk_hash", 0)),
		"chunk_rule_violations": int(local_chunk_map.get("rule_violations", 0)),
		"hill_cell_count": int(local_chunk_map.get("hill_cell_count", 0)),
		"hill_cluster_count": int(local_chunk_map.get("hill_cluster_count", 0)),
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
		if _contains(allowed, "scene_presence"):
			var presence_options := _scene_presence_options(rng, landmarks)
			var presence_correct := _correct_code(presence_options)
			questions.append({
				"kind": "scene_presence",
				"answer_mode": "option_pick",
				"stem": "Which statement about the studied scene is true?",
				"query_label": "PRESENCE",
				"correct_code": presence_correct,
				"correct_answer_token": str(presence_correct),
				"options": presence_options,
			})
		if _contains(allowed, "viewpoint_match"):
			var viewpoint_options := _viewpoint_match_options(rng, landmarks)
			var viewpoint_correct := _correct_code(viewpoint_options)
			questions.append({
				"kind": "viewpoint_match",
				"answer_mode": "option_pick",
				"stem": "Which second-angle scene matches the studied landscape?",
				"query_label": "VIEWPOINT",
				"correct_code": viewpoint_correct,
				"correct_answer_token": str(viewpoint_correct),
				"options": viewpoint_options,
			})
		if _contains(allowed, "object_count"):
			var count_options := _object_count_options(rng, landmarks)
			var count_correct := _correct_code(count_options)
			var count_meta := _as_dict(count_options[0]) if not count_options.is_empty() else {}
			questions.append({
				"kind": "object_count",
				"answer_mode": "option_pick",
				"stem": str(count_meta.get("stem", "How many objects of this type were in the studied scene?")),
				"query_label": str(count_meta.get("query_label", "OBJECT COUNT")),
				"correct_code": count_correct,
				"correct_answer_token": str(count_correct),
				"options": count_options,
			})
		if _contains(allowed, "object_relation"):
			var relation_options := _object_relation_options(rng, landmarks)
			if not relation_options.is_empty():
				var relation_correct := _correct_code(relation_options)
				questions.append({
					"kind": "object_relation",
					"answer_mode": "option_pick",
					"stem": "Which spacing statement about the studied scene is true?",
					"query_label": "SPACING",
					"correct_code": relation_correct,
					"correct_answer_token": str(relation_correct),
					"options": relation_options,
				})
	else:
		var route: Array = scene.get("route", [])
		var now_point: Vector3 = scene.get("aircraft_now", Vector3.ZERO)
		var next_point: Vector3 = scene.get("aircraft_next", Vector3.ZERO)
		var aircraft_tracks: Array = scene.get("aircraft_tracks", [])
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
				"stem": "Which path continuation shows the aircraft moving to the correct next position?",
				"query_label": "CONTINUATION",
				"correct_code": continuation_correct,
				"correct_answer_token": str(continuation_correct),
				"options": continuation_options,
			})
		if _contains(allowed, "aircraft_color_route_selection"):
			var route_track := _pick_aircraft_track(rng, aircraft_tracks)
			var route_color := str(route_track.get("color_label", "RED"))
			var color_route_options := _route_options(rng, route_track.get("route", route), route_track.get("current", now_point), "route")
			var color_route_correct := _correct_code(color_route_options)
			questions.append({
				"kind": "aircraft_color_route_selection",
				"answer_mode": "option_pick",
				"stem": "Which route belonged to the " + route_color + " aircraft?",
				"query_label": route_color + " ROUTE",
				"correct_code": color_route_correct,
				"correct_answer_token": str(color_route_correct),
				"options": _tag_route_options_with_color(color_route_options, route_color),
			})
		if _contains(allowed, "aircraft_count") and not aircraft_tracks.is_empty():
			var aircraft_count_options := _aircraft_count_options(rng, aircraft_tracks)
			var aircraft_count_correct := _correct_code(aircraft_count_options)
			questions.append({
				"kind": "aircraft_count",
				"answer_mode": "option_pick",
				"stem": "How many aircraft were shown moving in the studied scene?",
				"query_label": "AIRCRAFT COUNT",
				"correct_code": aircraft_count_correct,
				"correct_answer_token": str(aircraft_count_correct),
				"options": aircraft_count_options,
			})
		if _contains(allowed, "aircraft_presence") and not aircraft_tracks.is_empty():
			var aircraft_presence_options := _aircraft_presence_options(rng, aircraft_tracks)
			var aircraft_presence_correct := _correct_code(aircraft_presence_options)
			questions.append({
				"kind": "aircraft_presence",
				"answer_mode": "option_pick",
				"stem": "Which statement about the aircraft colors is true?",
				"query_label": "AIRCRAFT COLORS",
				"correct_code": aircraft_presence_correct,
				"correct_answer_token": str(aircraft_presence_correct),
				"options": aircraft_presence_options,
			})
		if _contains(allowed, "aircraft_order"):
			var aircraft_order_options := _aircraft_order_options(rng, aircraft_tracks)
			if not aircraft_order_options.is_empty():
				var aircraft_order_correct := _correct_code(aircraft_order_options)
				var order_meta := _as_dict(aircraft_order_options[0])
				questions.append({
					"kind": "aircraft_order",
					"answer_mode": "option_pick",
					"stem": str(order_meta.get("stem", "Which aircraft moved first?")),
					"query_label": "AIRCRAFT ORDER",
					"correct_code": aircraft_order_correct,
					"correct_answer_token": str(aircraft_order_correct),
					"options": aircraft_order_options,
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
	for required_kind in OBJECT_MEMORY_KINDS:
		for idx in range(available.size()):
			var spec := _as_dict(available[idx])
			if _memory_category(str(spec.get("kind", ""))) == str(required_kind):
				chosen_specs.append(available[idx])
				available.remove_at(idx)
				break
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
				candidate = Vector2i(int(rng.randi_range(1, max(1, grid_cols - 2))), int(rng.randi_range(1, max(1, grid_rows - 2))))
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


func _chunk_landmarks(rng: RandomNumberGenerator, local_chunk_map: Dictionary, specs: Array, count: int) -> Array:
	if local_chunk_map.is_empty():
		return _sample_landmarks(rng, specs, count)
	var sockets := _as_dict(local_chunk_map.get("asset_sockets", {}))
	var candidates: Array = []
	candidates.append_array(_landmark_candidates_from_sockets(sockets.get("building", []), "building"))
	candidates.append_array(_landmark_candidates_from_sockets(sockets.get("pedestrian", []), "human"))
	candidates.append_array(_landmark_candidates_from_sockets(sockets.get("people", []), "human"))
	candidates.append_array(_landmark_candidates_from_sockets(sockets.get("sheep", []), "sheep"))
	candidates.append_array(_landmark_candidates_from_sockets(sockets.get("forest", []), "forest"))
	candidates.append_array(_landmark_candidates_from_sockets(sockets.get("vehicle", []), "vehicle"))
	candidates.append_array(_landmark_candidates_from_sockets(sockets.get("truck", []), "vehicle"))
	candidates.append_array(_landmark_candidates_from_sockets(sockets.get("static", []), "tower"))
	if candidates.is_empty():
		return _sample_landmarks(rng, specs, count)
	var allowed_kinds := {}
	for spec_value in specs:
		var spec := _as_dict(spec_value)
		allowed_kinds[str(spec.get("kind", ""))] = true
	var pool := []
	for candidate_value in candidates:
		var candidate := _as_dict(candidate_value)
		if bool(allowed_kinds.get(str(candidate.get("kind", "")), false)):
			pool.append(candidate)
	if pool.is_empty():
		pool = candidates
	var landmarks: Array = []
	var used_labels := {}
	for required_kind in _required_memory_categories(specs):
		var required_idx := _find_landmark_candidate_index(pool, str(required_kind))
		if required_idx < 0:
			continue
		var required_candidate := _as_dict(pool[required_idx])
		pool.remove_at(required_idx)
		required_candidate["label"] = _next_chunk_landmark_label(str(required_candidate.get("kind", "landmark")), used_labels)
		landmarks.append(required_candidate)
	while landmarks.size() < count and not pool.is_empty():
		var idx := int(rng.randi_range(0, pool.size() - 1))
		var candidate := _as_dict(pool[idx])
		pool.remove_at(idx)
		var label := _next_chunk_landmark_label(str(candidate.get("kind", "landmark")), used_labels)
		candidate["label"] = label
		landmarks.append(candidate)
	if landmarks.size() < count:
		var fallback := _sample_landmarks(rng, specs, count - landmarks.size())
		for item in fallback:
			landmarks.append(item)
	return landmarks


func _required_memory_categories(specs: Array) -> Array:
	var out: Array = []
	for spec_value in specs:
		var spec := _as_dict(spec_value)
		var category := _memory_category(str(spec.get("kind", "")))
		if category != "" and not _contains(out, category):
			out.append(category)
	return out


func _find_landmark_candidate_index(pool: Array, category: String) -> int:
	for idx in range(pool.size()):
		var candidate := _as_dict(pool[idx])
		if _memory_category(str(candidate.get("kind", ""))) == category:
			return idx
	return -1


func _landmark_candidates_from_sockets(socket_values, landmark_kind: String) -> Array:
	var out: Array = []
	for value in socket_values:
		var socket := _as_dict(value)
		out.append({
			"label": "OBJ",
			"kind": landmark_kind,
			"x": clampi(int(socket.get("x", 0)), 0, grid_cols - 1),
			"y": clampi(int(socket.get("y", 0)), 0, grid_rows - 1),
			"socket_kind": str(socket.get("kind", "")),
			"tile_id": str(socket.get("tile_id", "")),
		})
	return out


func _next_chunk_landmark_label(kind_name: String, used_labels: Dictionary) -> String:
	var prefix := "OBJ"
	if kind_name == "building":
		prefix = "BLD"
	elif kind_name == "human" or kind_name == "foot_soldiers":
		prefix = "HUM"
	elif kind_name == "forest":
		prefix = "WOOD"
	elif kind_name == "vehicle" or kind_name == "truck":
		prefix = "VEH"
	elif kind_name == "sheep":
		prefix = "SHP"
	elif kind_name == "tower":
		prefix = "TWR"
	var index := int(used_labels.get(prefix, 0)) + 1
	used_labels[prefix] = index
	return prefix + str(index)


func _chunk_aircraft_route(local_chunk_map: Dictionary) -> Array:
	var route: Array = []
	for item in local_chunk_map.get("route_cells", []):
		var cell := _as_dict(item)
		route.append(Vector3(
			clampi(int(cell.get("x", 0)), 0, grid_cols - 1),
			clampi(int(cell.get("alt", 1)), 1, 4),
			clampi(int(cell.get("y", 0)), 0, grid_rows - 1)
		))
	if route.size() >= 4:
		return route
	for item in local_chunk_map.get("road_nodes", []):
		var node := _as_dict(item)
		route.append(Vector3(
			clampi(int(node.get("x", 0)), 0, grid_cols - 1),
			1 + (route.size() % 4),
			clampi(int(node.get("y", 0)), 0, grid_rows - 1)
		))
	return route


func _aircraft_track_count() -> int:
	if difficulty >= 0.90:
		return 5
	if difficulty >= 0.66:
		return 4
	if difficulty >= 0.34:
		return 3
	return 2


func _build_aircraft_tracks(rng: RandomNumberGenerator, base_route: Array, base_index: int) -> Array:
	var tracks: Array = []
	var count: int = mini(_aircraft_track_count(), AIRCRAFT_COLOR_LABELS.size())
	for i in range(count):
		var track_route := _route_variant_for_track(base_route, i)
		if track_route.size() < 4:
			continue
		var offset := 0 if i == 0 else ((i % 3) - 1)
		var current_index := clampi(base_index + offset, 1, track_route.size() - 2)
		var color_label := str(AIRCRAFT_COLOR_LABELS[i])
		var motion_delay := float(i) * 0.10
		var track := {
			"color_label": color_label,
			"route": track_route,
			"current_index": current_index,
			"previous": track_route[current_index - 1],
			"current": track_route[current_index],
			"next": track_route[current_index + 1],
			"motion_delay": motion_delay,
			"motion_duration": 0.48,
			"departure_order": i + 1,
			"arrival_order": i + 1,
			"route_hash": _hash_route(track_route) + _string_salt(color_label),
		}
		tracks.append(track)
		route_hash = _hash_mix(route_hash, int(track.get("route_hash", 0)))
	return tracks


func _route_variant_for_track(route: Array, index: int) -> Array:
	if index == 1:
		return _mirror_route(route)
	if index == 2:
		return _rotate_route(route)
	if index == 3:
		return _shift_route(route)
	if index == 4:
		var reversed_route := route.duplicate(true)
		reversed_route.reverse()
		return reversed_route
	return route.duplicate(true)


func _pick_aircraft_track(rng: RandomNumberGenerator, aircraft_tracks: Array) -> Dictionary:
	if aircraft_tracks.is_empty():
		return {
			"color_label": "RED",
			"route": [],
			"current": Vector3.ZERO,
			"next": Vector3.ZERO,
			"previous": Vector3.ZERO,
			"current_index": 0,
		}
	return _as_dict(aircraft_tracks[int(rng.randi_range(0, aircraft_tracks.size() - 1))])


func _tag_route_options_with_color(options: Array, color_label: String) -> Array:
	var out: Array = []
	for item in options:
		var opt := _as_dict(item)
		opt["aircraft_color"] = color_label
		out.append(opt)
	return out


func _hash_route(route: Array) -> int:
	var value := 17
	for item in route:
		var p: Vector3 = item
		value = _hash_mix(value, int(p.x) * 31 + int(p.y) * 37 + int(p.z) * 41)
	return value


func _static_reconstruction_options(rng: RandomNumberGenerator, landmarks: Array) -> Array:
	var candidates: Array = [{
		"answer_token": "correct",
		"landmarks": landmarks,
		"map_similarity": "exact",
	}]
	candidates.append_array(_static_reconstruction_decoys(rng, landmarks))
	return _ordered_options(rng, candidates)


func _static_reconstruction_decoys(rng: RandomNumberGenerator, landmarks: Array) -> Array:
	var decoys: Array = []
	# Higher difficulty keeps decoy maps closer to the studied layout, while still leaving one exact answer.
	if difficulty < 0.34:
		decoys.append({"answer_token": "mirrored", "landmarks": _mirror_landmarks(landmarks), "map_similarity": "low"})
		decoys.append({"answer_token": "rotated", "landmarks": _rotate_landmarks(landmarks), "map_similarity": "low"})
		decoys.append({"answer_token": "shifted", "landmarks": _shift_landmarks_by(landmarks, 3, -3), "map_similarity": "medium"})
	elif difficulty < 0.72:
		decoys.append({"answer_token": "shifted", "landmarks": _shift_landmarks_by(landmarks, 2, -1), "map_similarity": "medium"})
		decoys.append({"answer_token": "swapped", "landmarks": _swap_landmarks(landmarks), "map_similarity": "medium"})
		decoys.append({"answer_token": "nudged", "landmarks": _nudge_one_landmark(rng, landmarks, 3), "map_similarity": "high"})
	else:
		decoys.append({"answer_token": "nudged", "landmarks": _nudge_one_landmark(rng, landmarks, 1), "map_similarity": "high"})
		decoys.append({"answer_token": "swapped", "landmarks": _swap_two_landmarks(rng, landmarks), "map_similarity": "high"})
		decoys.append({"answer_token": "shifted", "landmarks": _shift_landmarks_by(landmarks, 1, -1), "map_similarity": "medium"})
	return decoys


func _scene_presence_options(rng: RandomNumberGenerator, landmarks: Array) -> Array:
	var counts := _memory_category_counts(landmarks)
	var present_categories := _present_memory_categories(counts)
	var correct_category := str(present_categories[int(rng.randi_range(0, max(0, present_categories.size() - 1)))]) if not present_categories.is_empty() else str(OBJECT_MEMORY_KINDS[0])
	var candidates: Array = [{
		"answer_token": "correct",
		"object_category": correct_category,
		"statement": "At least one " + _category_label(correct_category).to_lower() + " was present.",
	}]
	for category_value in OBJECT_MEMORY_KINDS:
		var category := str(category_value)
		if candidates.size() >= 4:
			break
		if category == correct_category:
			continue
		if int(counts.get(category, 0)) > 0:
			candidates.append({
				"answer_token": "missing_" + category,
				"object_category": category,
				"statement": "No " + _plural_category(category, 2) + " were present.",
			})
		else:
			candidates.append({
				"answer_token": "phantom_" + category,
				"object_category": category,
				"statement": "At least one " + _category_label(category).to_lower() + " was present.",
			})
	while candidates.size() < 4:
		candidates.append({
			"answer_token": "missing_correct_" + str(candidates.size()),
			"object_category": correct_category,
			"statement": "No " + _plural_category(correct_category, 2) + " were present.",
		})
	return _ordered_options(rng, candidates)


func _object_count_options(rng: RandomNumberGenerator, landmarks: Array) -> Array:
	var counts := _memory_category_counts(landmarks)
	var category := str(OBJECT_MEMORY_KINDS[int(rng.randi_range(0, OBJECT_MEMORY_KINDS.size() - 1))])
	var correct_count := int(counts.get(category, 0))
	var values: Array = [correct_count]
	for delta in [-1, 1, 2, -2, 3]:
		var candidate: int = maxi(0, correct_count + int(delta))
		if not values.has(candidate):
			values.append(candidate)
		if values.size() >= 4:
			break
	while values.size() < 4:
		var fallback := values.size()
		if not values.has(fallback):
			values.append(fallback)
		else:
			values.append(fallback + 4)
	var candidates: Array = []
	for idx in range(4):
		var value := int(values[idx])
		candidates.append({
			"answer_token": "correct" if value == correct_count else "count_" + str(value),
			"object_category": category,
			"object_count": value,
			"statement": str(value) + " " + _plural_category(category, value),
			"stem": "How many " + _plural_category(category, 2) + " were in the studied scene?",
			"query_label": category.to_upper() + " COUNT",
		})
	return _ordered_options(rng, candidates)


func _object_relation_options(rng: RandomNumberGenerator, landmarks: Array) -> Array:
	var counts := _memory_category_counts(landmarks)
	var present_categories := _present_memory_categories(counts)
	if present_categories.size() < 2:
		return []
	var true_candidates: Array = []
	var false_candidates: Array = []
	for a_index in range(present_categories.size()):
		for b_index in range(a_index + 1, present_categories.size()):
			var category_a := str(present_categories[a_index])
			var category_b := str(present_categories[b_index])
			var close := _categories_are_close(landmarks, category_a, category_b)
			true_candidates.append({
				"answer_token": "truth_" + category_a + "_" + category_b,
				"statement": _relation_statement(category_a, category_b, close),
			})
			false_candidates.append({
				"answer_token": "false_" + category_a + "_" + category_b,
				"statement": _relation_statement(category_a, category_b, not close),
			})
	if true_candidates.is_empty() or false_candidates.size() < 3:
		return []
	var correct_idx := int(rng.randi_range(0, true_candidates.size() - 1))
	var correct_candidate := _as_dict(true_candidates[correct_idx])
	correct_candidate["answer_token"] = "correct"
	var candidates: Array = [correct_candidate]
	var false_pool := false_candidates.duplicate(true)
	while candidates.size() < 4 and not false_pool.is_empty():
		var idx := int(rng.randi_range(0, false_pool.size() - 1))
		candidates.append(false_pool[idx])
		false_pool.remove_at(idx)
	return _ordered_options(rng, candidates) if candidates.size() == 4 else []


func _categories_are_close(landmarks: Array, category_a: String, category_b: String) -> bool:
	var threshold_sq := RELATION_CLOSE_DISTANCE_CELLS * RELATION_CLOSE_DISTANCE_CELLS
	for item_a in landmarks:
		var lm_a := _as_dict(item_a)
		if _memory_category(str(lm_a.get("kind", ""))) != category_a:
			continue
		for item_b in landmarks:
			var lm_b := _as_dict(item_b)
			if _memory_category(str(lm_b.get("kind", ""))) != category_b:
				continue
			var dx := float(int(lm_a.get("x", 0)) - int(lm_b.get("x", 0)))
			var dy := float(int(lm_a.get("y", 0)) - int(lm_b.get("y", 0)))
			if dx * dx + dy * dy <= threshold_sq:
				return true
	return false


func _relation_statement(category_a: String, category_b: String, close: bool) -> String:
	var a := _category_label(category_a).to_lower()
	var b := _category_label(category_b).to_lower()
	if close:
		return "A " + a + " was close to a " + b + "."
	return "No " + _plural_category(category_a, 2) + " were close to a " + b + "."


func _memory_category_counts(landmarks: Array) -> Dictionary:
	var counts := {}
	for category in OBJECT_MEMORY_KINDS:
		counts[str(category)] = 0
	for item in landmarks:
		var lm := _as_dict(item)
		var category := _memory_category(str(lm.get("kind", "")))
		if category != "":
			counts[category] = int(counts.get(category, 0)) + 1
	return counts


func _memory_category(kind_name: String) -> String:
	var token := kind_name.to_lower()
	if token == "building":
		return "building"
	if token == "vehicle" or token == "truck" or token == "tank" or token == "car":
		return "vehicle"
	if token == "human" or token == "foot_soldiers" or token == "person" or token == "people":
		return "human"
	if token == "sheep":
		return "sheep"
	if token == "tower":
		return "tower"
	return ""


func _present_memory_categories(counts: Dictionary) -> Array:
	var out: Array = []
	for category_value in OBJECT_MEMORY_KINDS:
		var category := str(category_value)
		if int(counts.get(category, 0)) > 0:
			out.append(category)
	return out


func _category_label(category: String) -> String:
	if category == "building":
		return "Building"
	if category == "vehicle":
		return "Vehicle"
	if category == "human":
		return "Human"
	if category == "sheep":
		return "Sheep"
	if category == "tower":
		return "Tower"
	if category == "forest":
		return "Forest"
	if category == "tent":
		return "Tent"
	if category == "object":
		return "Object"
	return category.capitalize()


func _plural_category(category: String, count: int) -> String:
	if category == "sheep":
		return "sheep"
	if category == "human":
		return "human" if count == 1 else "humans"
	var label := _category_label(category).to_lower()
	return label if count == 1 else label + "s"


func _viewpoint_match_options(rng: RandomNumberGenerator, landmarks: Array) -> Array:
	var candidates := [
		{"answer_token": "correct", "landmarks": landmarks, "view_angle": "second_angle"},
		{"answer_token": "mirrored", "landmarks": _mirror_landmarks(landmarks), "view_angle": "second_angle"},
		{"answer_token": "rotated", "landmarks": _rotate_landmarks(landmarks), "view_angle": "second_angle"},
		{"answer_token": "swapped", "landmarks": _shift_landmarks(_swap_landmarks(landmarks)), "view_angle": "second_angle"},
	]
	return _ordered_options(rng, candidates)


func _route_options(rng: RandomNumberGenerator, route: Array, aircraft_point: Vector3, variant: String) -> Array:
	var shifted_aircraft := Vector3(clampi(int(aircraft_point.x) + 1, 0, grid_cols - 1), aircraft_point.y, clampi(int(aircraft_point.z) - 1, 0, grid_rows - 1))
	var candidates := [
		{"answer_token": "correct", "route": route, "aircraft": aircraft_point},
		{"answer_token": "mirrored", "route": _mirror_route(route), "aircraft": _mirror_point(aircraft_point)},
		{"answer_token": "rotated", "route": _rotate_route(route), "aircraft": _rotate_point(aircraft_point)},
		{"answer_token": "loop_timing" if variant == "continuation" else "shifted", "route": _shift_route(route), "aircraft": shifted_aircraft},
	]
	return _ordered_options(rng, candidates)


func _aircraft_count_options(rng: RandomNumberGenerator, aircraft_tracks: Array) -> Array:
	var correct_count: int = maxi(1, aircraft_tracks.size())
	var values: Array = [correct_count]
	for delta in [-1, 1, 2, -2, 3]:
		var candidate: int = clampi(correct_count + int(delta), 1, AIRCRAFT_COLOR_LABELS.size())
		if not values.has(candidate):
			values.append(candidate)
		if values.size() >= 4:
			break
	while values.size() < 4:
		for fallback in range(1, AIRCRAFT_COLOR_LABELS.size() + 1):
			if not values.has(fallback):
				values.append(fallback)
				break
	var candidates: Array = []
	for idx in range(4):
		var value := int(values[idx])
		candidates.append({
			"answer_token": "correct" if value == correct_count else "aircraft_count_" + str(value),
			"aircraft_count": value,
			"statement": str(value) + " " + ("aircraft" if value == 1 else "aircraft"),
		})
	return _ordered_options(rng, candidates)


func _aircraft_presence_options(rng: RandomNumberGenerator, aircraft_tracks: Array) -> Array:
	var present_colors := _aircraft_present_colors(aircraft_tracks)
	var correct_color := str(present_colors[int(rng.randi_range(0, max(0, present_colors.size() - 1)))]) if not present_colors.is_empty() else "RED"
	var candidates: Array = [{
		"answer_token": "correct",
		"aircraft_color": correct_color,
		"statement": "A " + correct_color.to_lower() + " aircraft was present.",
	}]
	for color_value in AIRCRAFT_COLOR_LABELS:
		var color := str(color_value)
		if candidates.size() >= 4:
			break
		if color == correct_color:
			continue
		if present_colors.has(color):
			candidates.append({
				"answer_token": "missing_" + color.to_lower(),
				"aircraft_color": color,
				"statement": "No " + color.to_lower() + " aircraft was present.",
			})
		else:
			candidates.append({
				"answer_token": "phantom_" + color.to_lower(),
				"aircraft_color": color,
				"statement": "A " + color.to_lower() + " aircraft was present.",
			})
	while candidates.size() < 4:
		candidates.append({
			"answer_token": "missing_correct_" + str(candidates.size()),
			"aircraft_color": correct_color,
			"statement": "No " + correct_color.to_lower() + " aircraft was present.",
		})
	return _ordered_options(rng, candidates)


func _aircraft_order_options(rng: RandomNumberGenerator, aircraft_tracks: Array) -> Array:
	if aircraft_tracks.size() < 2:
		return []
	var ask_arrival := int(rng.randi_range(0, 1)) == 1
	var order_key := "arrival_order" if ask_arrival else "departure_order"
	var correct_color := _aircraft_order_color(aircraft_tracks, order_key)
	var candidates: Array = []
	for track_item in aircraft_tracks:
		var track := _as_dict(track_item)
		var color := str(track.get("color_label", "RED"))
		candidates.append({
			"answer_token": "correct" if color == correct_color else "order_" + color.to_lower(),
			"aircraft_color": color,
			"statement": color.capitalize() + " aircraft",
			"stem": "Which aircraft reached its end point first?" if ask_arrival else "Which aircraft began moving first?",
		})
	return _ordered_variable_options(rng, candidates)


func _aircraft_order_color(aircraft_tracks: Array, order_key: String) -> String:
	var best_color := "RED"
	var best_order := 9999
	for track_item in aircraft_tracks:
		var track := _as_dict(track_item)
		var order := int(track.get(order_key, 9999))
		if order < best_order:
			best_order = order
			best_color = str(track.get("color_label", best_color))
	return best_color


func _aircraft_present_colors(aircraft_tracks: Array) -> Array:
	var out: Array = []
	for track_item in aircraft_tracks:
		var track := _as_dict(track_item)
		var color := str(track.get("color_label", ""))
		if color != "" and not out.has(color):
			out.append(color)
	return out


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


func _ordered_variable_options(rng: RandomNumberGenerator, candidates: Array) -> Array:
	var indices: Array = []
	for idx in range(candidates.size()):
		indices.append(idx)
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
		copy["x"] = grid_cols - 1 - int(lm.get("x", 0))
		out.append(copy)
	return out


func _rotate_landmarks(landmarks: Array) -> Array:
	var out: Array = []
	for landmark in landmarks:
		var lm := _as_dict(landmark)
		var copy := lm.duplicate(true)
		copy["x"] = clampi(grid_cols - 1 - int(lm.get("y", 0)), 0, grid_cols - 1)
		copy["y"] = clampi(int(lm.get("x", 0)), 0, grid_rows - 1)
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


func _swap_two_landmarks(rng: RandomNumberGenerator, landmarks: Array) -> Array:
	var out := landmarks.duplicate(true)
	if out.size() < 2:
		return _shift_landmarks_by(landmarks, 1, -1)
	var idx_a: int = int(rng.randi_range(0, out.size() - 1))
	var idx_b: int = int(rng.randi_range(0, out.size() - 1))
	if idx_a == idx_b:
		idx_b = (idx_b + 1) % out.size()
	var a := _as_dict(out[idx_a])
	var b := _as_dict(out[idx_b])
	var ax := int(a.get("x", 0))
	var ay := int(a.get("y", 0))
	a["x"] = int(b.get("x", 0))
	a["y"] = int(b.get("y", 0))
	b["x"] = ax
	b["y"] = ay
	out[idx_a] = a
	out[idx_b] = b
	return out


func _shift_landmarks(landmarks: Array) -> Array:
	return _shift_landmarks_by(landmarks, 1, -1)


func _shift_landmarks_by(landmarks: Array, dx: int, dy: int) -> Array:
	var out: Array = []
	for landmark in landmarks:
		var lm := _as_dict(landmark)
		var copy := lm.duplicate(true)
		copy["x"] = clampi(int(lm.get("x", 0)) + dx, 0, grid_cols - 1)
		copy["y"] = clampi(int(lm.get("y", 0)) + dy, 0, grid_rows - 1)
		out.append(copy)
	return out


func _nudge_one_landmark(rng: RandomNumberGenerator, landmarks: Array, max_delta: int) -> Array:
	var out := landmarks.duplicate(true)
	if out.is_empty():
		return out
	var idx: int = int(rng.randi_range(0, out.size() - 1))
	var lm := _as_dict(out[idx])
	var dx: int = 0
	var dy: int = 0
	while dx == 0 and dy == 0:
		dx = int(rng.randi_range(-max_delta, max_delta))
		dy = int(rng.randi_range(-max_delta, max_delta))
	lm["x"] = clampi(int(lm.get("x", 0)) + dx, 0, grid_cols - 1)
	lm["y"] = clampi(int(lm.get("y", 0)) + dy, 0, grid_rows - 1)
	out[idx] = lm
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
		out.append(Vector3(clampi(int(p.x) + 1, 0, grid_cols - 1), p.y, clampi(int(p.z) - 1, 0, grid_rows - 1)))
	return out


func _mirror_point(point: Vector3) -> Vector3:
	return Vector3(grid_cols - 1 - int(point.x), point.y, point.z)


func _rotate_point(point: Vector3) -> Vector3:
	return Vector3(clampi(grid_cols - 1 - int(point.z), 0, grid_cols - 1), point.y, clampi(int(point.x), 0, grid_rows - 1))


func _rebuild_world_scene(show_study_labels: bool) -> void:
	aircraft_track_nodes.clear()
	_clear_children(terrain_root)
	_clear_children(object_root)
	_clear_children(route_root)
	var show_scene := stage != "question"
	terrain_root.visible = show_scene
	object_root.visible = show_scene
	route_root.visible = show_scene
	_update_north_marker()
	if not show_scene:
		return
	_draw_grid_terrain()
	if not chunked_generation:
		for hill in current_scene.get("hills", []):
			_draw_hill(_as_dict(hill))
	var landmarks: Array = current_scene.get("landmarks", [])
	for landmark in landmarks:
		_draw_landmark(_as_dict(landmark), show_study_labels)
	if str(current_scene.get("part", "")) == "aircraft":
		var tracks: Array = current_scene.get("aircraft_tracks", [])
		if not tracks.is_empty():
			for track_item in tracks:
				_draw_aircraft_track(_as_dict(track_item), show_study_labels)
		else:
			_draw_route(current_scene.get("route", []), ROUTE_COLOR)
			_draw_aircraft(current_scene.get("aircraft_now", Vector3.ZERO), GREEN_COLOR, true)


func _draw_grid_terrain() -> void:
	_make_box(terrain_root, "Ground", Vector3(0.0, -0.08, 0.0), Vector3(_grid_world_width() * 1.04, 0.08, _grid_world_depth() * 1.04), TERRAIN_COLOR)
	if chunked_generation and _as_dict(current_scene.get("chunk_map", {})).has("cells"):
		_draw_chunked_grid_terrain(_as_dict(current_scene.get("chunk_map", {})))
	for x in range(grid_cols + 1):
		var wx := (float(x) - float(grid_cols) * 0.5) * CELL_SIZE
		_make_box(terrain_root, "GridX", Vector3(wx, 0.025, 0.0), Vector3(0.012, 0.012, _grid_world_depth() * 1.01), GRID_COLOR)
	for y in range(grid_rows + 1):
		var wz := (float(y) - float(grid_rows) * 0.5) * CELL_SIZE
		_make_box(terrain_root, "GridY", Vector3(0.0, 0.03, wz), Vector3(_grid_world_width() * 1.01, 0.012, 0.012), GRID_COLOR)


func _draw_chunked_grid_terrain(local_chunk_map: Dictionary) -> void:
	for item in local_chunk_map.get("cells", []):
		var cell := _as_dict(item)
		var x := int(cell.get("x", 0))
		var y := int(cell.get("y", 0))
		var terrain := str(cell.get("terrain", "grassland"))
		var color := _chunk_terrain_color(terrain, int(cell.get("variant", 0)))
		var center := _grid_to_world(x, y, 0)
		_make_box(terrain_root, "ChunkTile", center + Vector3(0.0, -0.03, 0.0), Vector3(CELL_SIZE * 1.03, 0.035, CELL_SIZE * 1.03), color)
		if bool(cell.get("river_n", false)) or bool(cell.get("river_s", false)):
			_make_box(terrain_root, "ChunkRiver", center + Vector3(0.0, 0.005, 0.0), Vector3(CELL_SIZE * 0.32, 0.018, CELL_SIZE * 1.03), BLUE_COLOR.darkened(0.18))
		if bool(cell.get("river_e", false)) or bool(cell.get("river_w", false)):
			_make_box(terrain_root, "ChunkRiver", center + Vector3(0.0, 0.006, 0.0), Vector3(CELL_SIZE * 1.03, 0.018, CELL_SIZE * 0.32), BLUE_COLOR.darkened(0.18))
		if bool(cell.get("road_n", false)) or bool(cell.get("road_s", false)):
			_make_box(terrain_root, "ChunkRoad", center + Vector3(0.0, 0.035, 0.0), Vector3(CELL_SIZE * 0.28, 0.020, CELL_SIZE * 1.03), Color(0.23, 0.22, 0.20, 1.0))
		if bool(cell.get("road_e", false)) or bool(cell.get("road_w", false)):
			_make_box(terrain_root, "ChunkRoad", center + Vector3(0.0, 0.04, 0.0), Vector3(CELL_SIZE * 1.03, 0.020, CELL_SIZE * 0.28), Color(0.23, 0.22, 0.20, 1.0))
		if bool(cell.get("is_bridge", false)):
			_make_box(terrain_root, "ChunkBridge", center + Vector3(0.0, 0.075, 0.0), Vector3(CELL_SIZE * 1.03, 0.035, CELL_SIZE * 0.34), AMBER_COLOR.darkened(0.26))
	_draw_merged_hill_clusters(local_chunk_map)


func _chunk_terrain_color(terrain: String, variant: int) -> Color:
	var jitter := float(variant % 5) * 0.035
	if terrain == "city":
		return Color(0.45 + jitter, 0.43 + jitter, 0.38 + jitter, 1.0)
	if terrain == "city_edge":
		return Color(0.40 + jitter, 0.46, 0.31, 1.0)
	if terrain == "forest":
		return TERRAIN_DARK.lerp(Color(0.10, 0.34, 0.14, 1.0), 0.68)
	if terrain == "forest_edge":
		return Color(0.18, 0.36 + jitter, 0.18, 1.0)
	if terrain == "river":
		return BLUE_COLOR.darkened(0.24)
	if terrain == "bridge":
		return Color(0.34, 0.31, 0.25, 1.0)
	if terrain == "field":
		return Color(0.34 + jitter, 0.42 + jitter, 0.20, 1.0)
	if terrain == "hill":
		return HILL_COLOR.darkened(0.10).lightened(jitter)
	return TERRAIN_COLOR.lightened(jitter)


func _draw_hill(hill: Dictionary) -> void:
	var pos := _grid_to_world(int(hill.get("x", 0)), int(hill.get("y", 0)), 0)
	var radius := 0.34 + float(hill.get("radius", 1)) * 0.20
	var height := 0.18 + float(hill.get("height", 1)) * 0.15
	_make_hill_dome(terrain_root, "HillDome", pos, radius, height, HILL_COLOR)


func _draw_merged_hill_clusters(local_chunk_map: Dictionary) -> void:
	for cluster_item in local_chunk_map.get("hill_clusters", []):
		var cluster := _as_dict(cluster_item)
		var cx := int(cluster.get("x", 0))
		var cy := int(cluster.get("y", 0))
		var peak_tier: int = max(1, int(cluster.get("peak_tier", 4)))
		var footprint_cells: int = max(1, int(cluster.get("footprint_cells", int(cluster.get("radius", 2)) * 2 + 1)))
		var target_footprint: int = max(1, int(cluster.get("target_footprint", footprint_cells)))
		var dome_cells: int = max(footprint_cells, target_footprint)
		var center := _grid_to_world(cx, cy, 0)
		var radius := float(dome_cells) * CELL_SIZE * 0.53
		var height := 0.38 + float(peak_tier) * 0.20
		_make_hill_dome(terrain_root, "MergedSpatialHillDome", center + Vector3(0.0, 0.012, 0.0), radius, height, HILL_COLOR.lightened(float(peak_tier) * 0.025))


func _make_hill_dome(parent: Node, name_value: String, base_center: Vector3, radius: float, height: float, color: Color) -> MeshInstance3D:
	var mesh := ArrayMesh.new()
	var vertices := PackedVector3Array()
	var normals := PackedVector3Array()
	var indices := PackedInt32Array()
	var rings := 5
	var segments := 16
	for r in range(rings + 1):
		var frac := float(r) / float(rings)
		var ring_radius := radius * frac
		var y := base_center.y + height * pow(maxf(0.0, 1.0 - frac * frac), 0.74)
		for s in range(segments):
			var ang := (TAU * float(s)) / float(segments)
			vertices.append(Vector3(base_center.x + cos(ang) * ring_radius, y, base_center.z + sin(ang) * ring_radius))
			normals.append(Vector3.UP)
	for r in range(rings):
		for s in range(segments):
			var next_s := (s + 1) % segments
			var i0 := r * segments + s
			var i1 := r * segments + next_s
			var i2 := (r + 1) * segments + s
			var i3 := (r + 1) * segments + next_s
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
	arrays[Mesh.ARRAY_INDEX] = indices
	mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, arrays)
	var node := MeshInstance3D.new()
	node.name = name_value
	node.mesh = mesh
	node.material_override = _material(color)
	parent.add_child(node)
	return node


func _draw_landmark(landmark: Dictionary, show_label: bool) -> void:
	var kind_name := str(landmark.get("kind", "landmark"))
	var label := str(landmark.get("label", "OBJ"))
	var root := Node3D.new()
	root.name = label
	root.position = _grid_to_world(int(landmark.get("x", 0)), int(landmark.get("y", 0)), 0)
	object_root.add_child(root)
	if kind_name == "building":
		_make_box(root, "Building", Vector3(0.0, 0.58, 0.0), Vector3(0.62, 0.58, 0.58), Color(0.66, 0.60, 0.48, 1.0))
		_make_box(root, "Roof", Vector3(0.0, 1.22, 0.0), Vector3(0.76, 0.14, 0.72), RED_COLOR.darkened(0.25))
		_make_box(root, "Door", Vector3(0.0, 0.30, -0.31), Vector3(0.18, 0.26, 0.035), Color(0.20, 0.16, 0.12, 1.0))
	elif kind_name == "human" or kind_name == "foot_soldiers":
		for i in range(3):
			var off := Vector3(float(i - 1) * 0.20, 0.0, float(i % 2) * 0.12)
			_make_sphere(root, "HumanHead", off + Vector3(0.0, 0.62, 0.0), Vector3(0.105, 0.105, 0.105), AMBER_COLOR)
			_make_box(root, "HumanBody", off + Vector3(0.0, 0.36, 0.0), Vector3(0.12, 0.28, 0.09), BLUE_COLOR.darkened(0.20))
			_make_box(root, "HumanLegL", off + Vector3(-0.04, 0.14, 0.0), Vector3(0.04, 0.16, 0.04), BLUE_COLOR.darkened(0.32))
			_make_box(root, "HumanLegR", off + Vector3(0.04, 0.14, 0.0), Vector3(0.04, 0.16, 0.04), BLUE_COLOR.darkened(0.32))
	elif kind_name == "sheep":
		for i in range(3):
			var off := Vector3(float(i - 1) * 0.24, 0.0, float(i % 2) * 0.16)
			_make_sphere(root, "SheepBody", off + Vector3(0.0, 0.26, 0.0), Vector3(0.22, 0.14, 0.16), WHITE_COLOR)
			_make_sphere(root, "SheepHead", off + Vector3(0.17, 0.28, 0.0), Vector3(0.08, 0.08, 0.07), Color(0.12, 0.12, 0.11, 1.0))
			_make_box(root, "SheepLegs", off + Vector3(0.0, 0.10, 0.0), Vector3(0.20, 0.09, 0.10), Color(0.10, 0.10, 0.09, 1.0))
	elif kind_name == "forest":
		for i in range(4):
			var off := Vector3(float(i % 2) * 0.28 - 0.14, 0.0, float(i / 2) * 0.28 - 0.14)
			_make_box(root, "TreeTrunk", off + Vector3(0.0, 0.18, 0.0), Vector3(0.04, 0.18, 0.04), Color(0.29, 0.16, 0.07, 1.0))
			_make_sphere(root, "TreeTop", off + Vector3(0.0, 0.44, 0.0), Vector3(0.17, 0.17, 0.17), Color(0.14, 0.38, 0.16, 1.0))
	elif kind_name == "vehicle" or kind_name == "truck":
		_make_box(root, "VehicleBody", Vector3(0.0, 0.26, 0.0), Vector3(0.56, 0.22, 0.76), Color(0.22, 0.34, 0.30, 1.0))
		_make_box(root, "VehicleCab", Vector3(0.0, 0.50, -0.22), Vector3(0.36, 0.18, 0.26), Color(0.30, 0.44, 0.38, 1.0))
		_make_box(root, "VehicleWheelFL", Vector3(-0.31, 0.13, -0.22), Vector3(0.07, 0.11, 0.11), Color(0.03, 0.035, 0.04, 1.0))
		_make_box(root, "VehicleWheelFR", Vector3(0.31, 0.13, -0.22), Vector3(0.07, 0.11, 0.11), Color(0.03, 0.035, 0.04, 1.0))
		_make_box(root, "VehicleWheelRL", Vector3(-0.31, 0.13, 0.24), Vector3(0.07, 0.11, 0.11), Color(0.03, 0.035, 0.04, 1.0))
		_make_box(root, "VehicleWheelRR", Vector3(0.31, 0.13, 0.24), Vector3(0.07, 0.11, 0.11), Color(0.03, 0.035, 0.04, 1.0))
	elif kind_name == "tower":
		_make_box(root, "Tower", Vector3(0.0, 0.55, 0.0), Vector3(0.12, 0.55, 0.12), Color(0.54, 0.56, 0.55, 1.0))
		_make_sphere(root, "TowerTop", Vector3(0.0, 1.16, 0.0), Vector3(0.20, 0.20, 0.20), Color(0.72, 0.74, 0.72, 1.0))
	else:
		_make_box(root, "Tent", Vector3(0.0, 0.20, 0.0), Vector3(0.32, 0.20, 0.32), AMBER_COLOR.darkened(0.18))


func _draw_route(route: Array, color: Color = ROUTE_COLOR) -> void:
	var previous := Vector3.ZERO
	var have_previous := false
	for idx in range(route.size()):
		var p: Vector3 = route[idx]
		var current := _grid_point_to_world(p)
		_make_sphere(route_root, "RoutePoint", current, Vector3(0.10, 0.10, 0.10), color.lightened(0.10))
		if have_previous:
			_make_segment(route_root, previous, current, color, 0.035)
		previous = current
		have_previous = true


func _draw_aircraft_track(track: Dictionary, show_label: bool) -> void:
	var color_label := str(track.get("color_label", "RED"))
	var color := _aircraft_color(color_label)
	var route: Array = track.get("route", [])
	_draw_route(route, color.darkened(0.06))
	var previous: Vector3 = track.get("previous", Vector3.ZERO)
	var current: Vector3 = track.get("current", Vector3.ZERO)
	var draw_point := current
	if stage == "study":
		var elapsed_fraction := clampf(stage_elapsed_s / maxf(0.1, _study_step_s()), 0.0, 1.0)
		var t := _track_motion_progress(track, elapsed_fraction)
		draw_point = previous.lerp(current, t)
	var node := _draw_aircraft(draw_point, color, show_label, color_label)
	aircraft_track_nodes.append({
		"node": node,
		"previous": previous,
		"current": current,
		"motion_delay": track.get("motion_delay", 0.0),
		"motion_duration": track.get("motion_duration", 0.48),
	})


func _update_aircraft_track_nodes() -> void:
	if aircraft_track_nodes.is_empty() or str(current_scene.get("part", "")) != "aircraft":
		return
	var elapsed_fraction := 1.0
	if stage == "study":
		elapsed_fraction = clampf(stage_elapsed_s / maxf(0.1, _study_step_s()), 0.0, 1.0)
	for item in aircraft_track_nodes:
		var entry := _as_dict(item)
		var node: Node3D = entry.get("node", null)
		if node == null or not is_instance_valid(node):
			continue
		var previous: Vector3 = entry.get("previous", Vector3.ZERO)
		var current: Vector3 = entry.get("current", Vector3.ZERO)
		var t := _track_motion_progress(entry, elapsed_fraction)
		node.position = _grid_point_to_world(previous.lerp(current, t))


func _track_motion_progress(track: Dictionary, elapsed_fraction: float) -> float:
	var delay := clampf(_float(track.get("motion_delay", 0.0)), 0.0, 0.95)
	var duration := maxf(0.05, _float(track.get("motion_duration", 0.48)))
	return clampf((elapsed_fraction - delay) / duration, 0.0, 1.0)


func _draw_aircraft(point, color: Color, _label_current: bool, _label_text: String = "AIRCRAFT") -> Node3D:
	var p: Vector3 = point
	var root := Node3D.new()
	root.name = "Aircraft"
	root.position = _grid_point_to_world(p)
	route_root.add_child(root)
	_make_box(root, "Fuselage", Vector3(0.0, 0.0, 0.0), Vector3(0.11, 0.09, 0.42), color)
	_make_box(root, "Wing", Vector3(0.0, 0.0, 0.0), Vector3(0.56, 0.025, 0.10), color.lightened(0.12))
	_make_box(root, "Tail", Vector3(0.0, 0.11, 0.30), Vector3(0.06, 0.14, 0.08), color.darkened(0.10))
	return root


func _rebuild_answer_overlay() -> void:
	_clear_children(answer_root)
	selected_option_code = 0 if stage != "question" else selected_option_code
	var wants_map_options := stage == "question" and str(current_question.get("kind", "")) == "scene_reconstruction"
	var panel := PanelContainer.new()
	panel.name = "SpatialPanel"
	panel.position = Vector2(24, 284) if wants_map_options else Vector2(24, 350)
	panel.custom_minimum_size = Vector2(912, 238) if wants_map_options else Vector2(912, 166)
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
	_build_option_buttons(box)
	_refresh_entry_label()


func _build_option_buttons(parent: Control) -> void:
	var options: Array = current_question.get("options", [])
	var row := HBoxContainer.new()
	row.add_theme_constant_override("separation", 8)
	parent.add_child(row)
	for option in options:
		var opt := _as_dict(option)
		var code := int(opt.get("code", 0))
		var button := Button.new()
		if _option_needs_topdown_preview(opt):
			button.custom_minimum_size = Vector2(210, 150)
			button.text = ""
			_add_topdown_map_option_visual(button, opt, code)
		else:
			button.custom_minimum_size = Vector2(210, 82)
			button.text = str(code) + "\n" + _option_card_label(opt)
		button.pressed.connect(Callable(self, "_submit_option").bind(code))
		row.add_child(button)


func _refresh_entry_label() -> void:
	if entry_label == null:
		return
	if stage != "question":
		return
	entry_label.text = "Press 1-4 or click a card."


func _study_prompt() -> String:
	var labels := ["High oblique view", "Low oblique view", "Rotated reference view"]
	var part := "Landscape" if str(current_scene.get("part", "static")) == "static" else "Aircraft route"
	return part + " study: " + labels[clampi(study_view_index, 0, labels.size() - 1)] + " (" + str(study_view_index + 1) + "/3)"


func _option_card_label(option: Dictionary) -> String:
	if option.has("statement"):
		return str(option.get("statement", "Scene statement"))
	if option.has("view_angle"):
		return "Map layout\n" + _landmark_option_summary(option.get("landmarks", []))
	if option.has("route"):
		var aircraft: Vector3 = option.get("aircraft", Vector3.ZERO)
		return "Route map\nAircraft near " + _coarse_region_for_point(aircraft)
	var landmarks: Array = option.get("landmarks", [])
	return "Map layout\nObjects " + str(landmarks.size())


func _option_needs_topdown_preview(option: Dictionary) -> bool:
	return str(current_question.get("kind", "")) == "scene_reconstruction" and option.has("landmarks")


func _add_topdown_map_option_visual(button: Button, option: Dictionary, code: int) -> void:
	var code_label := Label.new()
	code_label.name = "OptionCode"
	code_label.text = str(code)
	code_label.position = Vector2(10, 6)
	code_label.size = Vector2(32, 22)
	code_label.mouse_filter = Control.MOUSE_FILTER_IGNORE
	code_label.add_theme_font_size_override("font_size", 18)
	code_label.add_theme_color_override("font_color", WHITE_COLOR)
	button.add_child(code_label)
	var preview := TextureRect.new()
	preview.name = "TopDownMapPreview"
	preview.position = Vector2(11, 30)
	preview.size = Vector2(188, 108)
	preview.custom_minimum_size = Vector2(188, 108)
	preview.mouse_filter = Control.MOUSE_FILTER_IGNORE
	preview.stretch_mode = TextureRect.STRETCH_SCALE
	preview.texture = _topdown_map_texture(option.get("landmarks", []))
	button.add_child(preview)


func _topdown_map_texture(landmarks_value) -> Texture2D:
	var width := 188
	var height := 108
	var image: Image = Image.create(width, height, false, Image.FORMAT_RGBA8)
	image.fill(Color(0.10, 0.15, 0.12, 1.0))
	image.fill_rect(Rect2i(4, 4, width - 8, height - 8), TERRAIN_COLOR.darkened(0.12))
	for x in range(0, grid_cols + 1, 4):
		var px: int = 4 + int(round(float(x) / maxf(1.0, float(grid_cols)) * float(width - 8)))
		image.fill_rect(Rect2i(clampi(px, 4, width - 5), 4, 1, height - 8), GRID_COLOR.darkened(0.28))
	for y in range(0, grid_rows + 1, 4):
		var py: int = 4 + int(round(float(y) / maxf(1.0, float(grid_rows)) * float(height - 8)))
		image.fill_rect(Rect2i(4, clampi(py, 4, height - 5), width - 8, 1), GRID_COLOR.darkened(0.28))
	if typeof(landmarks_value) == TYPE_ARRAY:
		var landmarks: Array = landmarks_value
		for item in landmarks:
			var lm := _as_dict(item)
			var category := _visible_category_for_kind(str(lm.get("kind", "")))
			var px: int = 4 + int(round((float(int(lm.get("x", 0))) + 0.5) / maxf(1.0, float(grid_cols)) * float(width - 8)))
			var py: int = 4 + int(round((float(int(lm.get("y", 0))) + 0.5) / maxf(1.0, float(grid_rows)) * float(height - 8)))
			_fill_preview_square(image, Vector2i(px, py), _preview_radius_for_category(category), _preview_color_for_category(category))
	image.fill_rect(Rect2i(4, 4, width - 8, 1), PANEL_BORDER)
	image.fill_rect(Rect2i(4, height - 5, width - 8, 1), PANEL_BORDER)
	image.fill_rect(Rect2i(4, 4, 1, height - 8), PANEL_BORDER)
	image.fill_rect(Rect2i(width - 5, 4, 1, height - 8), PANEL_BORDER)
	return ImageTexture.create_from_image(image)


func _fill_preview_square(image: Image, center: Vector2i, radius: int, color: Color) -> void:
	var start_x: int = clampi(center.x - radius, 0, image.get_width() - 1)
	var start_y: int = clampi(center.y - radius, 0, image.get_height() - 1)
	var end_x: int = clampi(center.x + radius, 0, image.get_width() - 1)
	var end_y: int = clampi(center.y + radius, 0, image.get_height() - 1)
	image.fill_rect(Rect2i(start_x, start_y, end_x - start_x + 1, end_y - start_y + 1), color)


func _preview_radius_for_category(category: String) -> int:
	if category == "forest":
		return 5
	if category == "building" or category == "vehicle":
		return 4
	return 3


func _preview_color_for_category(category: String) -> Color:
	if category == "building":
		return Color(0.74, 0.62, 0.42, 1.0)
	if category == "vehicle":
		return Color(0.24, 0.48, 0.36, 1.0)
	if category == "human":
		return BLUE_COLOR.lightened(0.18)
	if category == "sheep":
		return WHITE_COLOR
	if category == "tower":
		return Color(0.70, 0.72, 0.70, 1.0)
	if category == "forest":
		return Color(0.14, 0.40, 0.16, 1.0)
	if category == "tent":
		return AMBER_COLOR
	return TEXT_MUTED


func _landmark_option_summary(landmarks_value) -> String:
	if typeof(landmarks_value) != TYPE_ARRAY:
		return "No objects"
	var landmarks: Array = landmarks_value
	var labels: Array = []
	for i in range(min(4, landmarks.size())):
		var lm := _as_dict(landmarks[i])
		var category := _visible_category_for_kind(str(lm.get("kind", "")))
		labels.append(_category_label(category) + " " + _coarse_region_for_cell(int(lm.get("x", 0)), int(lm.get("y", 0))))
	return " / ".join(labels)


func _visible_category_for_kind(kind_name: String) -> String:
	var category := _memory_category(kind_name)
	if category != "":
		return category
	var token := kind_name.to_lower()
	if token == "forest":
		return "forest"
	if token == "tent":
		return "tent"
	return "object"


func _coarse_region_for_point(point: Vector3) -> String:
	return _coarse_region_for_cell(int(point.x), int(point.z))


func _coarse_region_for_cell(x: int, y: int) -> String:
	var third_x := maxf(1.0, float(grid_cols) / 3.0)
	var third_y := maxf(1.0, float(grid_rows) / 3.0)
	var horizontal := "center"
	if float(x) < third_x:
		horizontal = "west"
	elif float(x) >= third_x * 2.0:
		horizontal = "east"
	var vertical := "middle"
	if float(y) < third_y:
		vertical = "north"
	elif float(y) >= third_y * 2.0:
		vertical = "south"
	if horizontal == "center" and vertical == "middle":
		return "center"
	if horizontal == "center":
		return vertical
	if vertical == "middle":
		return horizontal
	return vertical + "-" + horizontal


func _update_camera(camera: Camera3D, dt: float) -> void:
	if camera == null:
		return
	camera.projection = Camera3D.PROJECTION_PERSPECTIVE
	camera.far = maxf(camera.far, maxf(_grid_world_width(), _grid_world_depth()) * 4.0)
	var target := Vector3(0.0, 0.0, 0.0)
	var span := maxf(_grid_world_width(), _grid_world_depth())
	var pos := _study_camera_position(0, span)
	if stage == "study":
		pos = _study_camera_position(study_view_index, span)
	elif str(current_question.get("kind", "")) == "viewpoint_match":
		pos = Vector3(span * 0.72, span * 0.50, -span * 0.82)
	var blend := clampf(dt * 6.0, 0.08, 0.30)
	camera.position = camera.position.lerp(pos, blend)
	camera.look_at_from_position(camera.position, target, Vector3.UP)


func _study_camera_position(view_index: int, span: float) -> Vector3:
	var local_pos := Vector3(span * 0.52, span * 0.72, -span * 0.64)
	if view_index == 1:
		local_pos = Vector3(span * 0.84, span * 0.52, span * 0.78)
	elif view_index == 2:
		local_pos = Vector3(-span * 0.74, span * 0.60, span * 0.82)
	return _rotate_y(local_pos, _study_orientation_angle())


func _study_orientation_for_scene() -> int:
	return int(_rng_for("study_orientation").randi_range(0, 3))


func _study_orientation_angle() -> float:
	return float(study_orientation_index % 4) * PI * 0.5


func _rotate_y(value: Vector3, angle: float) -> Vector3:
	var c := cos(angle)
	var s := sin(angle)
	return Vector3(value.x * c - value.z * s, value.y, value.x * s + value.z * c)


func _update_hud() -> void:
	if hud_label == null:
		return
	_update_north_marker()
	if stage == "question":
		hud_label.text = "Spatial Integration | " + phase + " | " + str(correct) + "/" + str(attempted) + " | Question"
		return
	var remaining := maxf(0.0, duration_s - elapsed_s)
	var stage_remaining := _stage_remaining_s()
	hud_label.text = "Spatial Integration | " + phase + " | " + str(int(ceil(remaining))) + "s | " + str(correct) + "/" + str(attempted) + " | " + stage.capitalize() + " " + str(int(ceil(stage_remaining))) + "s"


func _update_north_marker() -> void:
	if north_marker_root == null:
		return
	north_marker_root.visible = active and stage != "question"


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
			"chunk_map_hash": int(current_scene.get("chunk_hash", 0)),
			"hill_cell_count": int(current_scene.get("hill_cell_count", 0)),
			"hill_cluster_count": int(current_scene.get("hill_cluster_count", 0)),
			"visible_option_count": _visible_option_count(),
			"question_kind": str(current_question.get("kind", "")),
		},
	})


func _visible_option_count() -> int:
	if str(current_question.get("answer_mode", "")) != "option_pick":
		return 0
	return (current_question.get("options", []) as Array).size()


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
	_build_north_scene_marker()
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


func _build_north_scene_marker() -> void:
	north_marker_root = Node3D.new()
	north_marker_root.name = "NorthSceneMarker"
	scene_root.add_child(north_marker_root)
	var map_north_edge := -_grid_world_depth() * 0.5
	var y := 0.28
	var color := AMBER_COLOR.lightened(0.14)
	var shaft_start := Vector3(0.0, y, map_north_edge - CELL_SIZE * 0.45)
	var arrow_tip := Vector3(0.0, y, map_north_edge - CELL_SIZE * 2.15)
	_make_segment(north_marker_root, shaft_start, arrow_tip, color, 0.13)
	_make_segment(north_marker_root, arrow_tip, Vector3(-CELL_SIZE * 0.52, y, map_north_edge - CELL_SIZE * 1.60), color, 0.13)
	_make_segment(north_marker_root, arrow_tip, Vector3(CELL_SIZE * 0.52, y, map_north_edge - CELL_SIZE * 1.60), color, 0.13)
	var letter_z := map_north_edge - CELL_SIZE * 2.95
	var letter_half_w := CELL_SIZE * 0.42
	var letter_half_h := CELL_SIZE * 0.50
	_make_segment(north_marker_root, Vector3(-letter_half_w, y, letter_z - letter_half_h), Vector3(-letter_half_w, y, letter_z + letter_half_h), WHITE_COLOR, 0.11)
	_make_segment(north_marker_root, Vector3(letter_half_w, y, letter_z - letter_half_h), Vector3(letter_half_w, y, letter_z + letter_half_h), WHITE_COLOR, 0.11)
	_make_segment(north_marker_root, Vector3(-letter_half_w, y, letter_z + letter_half_h), Vector3(letter_half_w, y, letter_z - letter_half_h), WHITE_COLOR, 0.11)


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
	north_marker_root = null
	prompt_label = null
	entry_label = null
	answer_root = null
	chunk_map.clear()


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


func _grid_world_width() -> float:
	return float(max(1, grid_cols)) * CELL_SIZE


func _grid_world_depth() -> float:
	return float(max(1, grid_rows)) * CELL_SIZE


func _grid_to_world(x: int, y: int, alt: int) -> Vector3:
	return Vector3((float(x) - (float(grid_cols) - 1.0) * 0.5) * CELL_SIZE, float(alt) * 0.58 + 0.05, (float(y) - (float(grid_rows) - 1.0) * 0.5) * CELL_SIZE)


func _grid_point_to_world(point: Vector3) -> Vector3:
	return Vector3((point.x - (float(grid_cols) - 1.0) * 0.5) * CELL_SIZE, point.y * 0.58 + 0.05, (point.z - (float(grid_rows) - 1.0) * 0.5) * CELL_SIZE)


func _aircraft_color(color_label: String) -> Color:
	var token := color_label.to_upper()
	if token == "RED":
		return RED_COLOR
	if token == "BLUE":
		return BLUE_COLOR
	if token == "AMBER":
		return AMBER_COLOR
	if token == "WHITE":
		return WHITE_COLOR
	if token == "GREEN":
		return GREEN_COLOR
	return GREEN_COLOR


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


func _study_step_s() -> float:
	var study_total := aircraft_study_s if str(current_scene.get("part", "static")) == "aircraft" else static_study_s
	return study_total / 3.0


func _stage_remaining_s() -> float:
	if stage == "question":
		return 0.0
	return maxf(0.0, _study_step_s() - stage_elapsed_s)


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
	value = _hash_mix(value, int(scene.get("chunk_hash", 0)))
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

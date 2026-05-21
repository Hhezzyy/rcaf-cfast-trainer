extends SceneTree

const SpatialRuntime = preload("res://scripts/spatial_integration_runtime.gd")


func _init() -> void:
	var runtime := SpatialRuntime.new()
	runtime.session_seed = 777
	runtime.scene_counter = 3
	runtime.grid_cols = 24
	runtime.grid_rows = 24
	runtime.allowed_question_kinds = ["scene_reconstruction", "scene_presence", "viewpoint_match", "object_count", "object_relation"]
	var scene := {
		"scene_id": 3,
		"part": "static",
		"landmarks": [
			{"label": "BLD1", "kind": "building", "x": 4, "y": 5},
			{"label": "HUM1", "kind": "human", "x": 9, "y": 7},
			{"label": "HUM2", "kind": "human", "x": 10, "y": 8},
			{"label": "VEH1", "kind": "vehicle", "x": 13, "y": 12},
			{"label": "SHP1", "kind": "sheep", "x": 15, "y": 11},
			{"label": "TWR1", "kind": "tower", "x": 18, "y": 15},
		],
	}
	var questions: Array = runtime._questions_for_scene(scene)
	var failures: Array = []
	_expect(failures, _question_has_four_options(questions, "scene_reconstruction"), "scene_reconstruction should emit four options")
	_expect(failures, _question_has_four_options(questions, "scene_presence"), "scene_presence should emit four options")
	_expect(failures, _question_has_four_options(questions, "viewpoint_match"), "viewpoint_match should emit four options")
	_expect(failures, _question_has_four_options(questions, "object_count"), "object_count should emit four options")
	_expect(failures, _question_has_four_options(questions, "object_relation"), "object_relation should emit four options")
	_expect(failures, _question_has_one_correct(questions, "scene_reconstruction"), "scene_reconstruction should have exactly one correct option")
	_expect(failures, _question_has_one_correct(questions, "scene_presence"), "scene_presence should have exactly one correct option")
	_expect(failures, _question_has_one_correct(questions, "viewpoint_match"), "viewpoint_match should have exactly one correct option")
	_expect(failures, _question_has_one_correct(questions, "object_count"), "object_count should have exactly one correct option")
	_expect(failures, _question_has_one_correct(questions, "object_relation"), "object_relation should have exactly one correct option")
	_expect(failures, _questions_are_fair(runtime, questions), "static questions should avoid hidden IDs, answer tokens, and grid cells")
	runtime.stage = "question"
	runtime.stage_elapsed_s = 999.0
	runtime.question_time_limit_s = 0.0
	runtime.attempted = 0
	runtime._check_stage_timeout()
	_expect(failures, runtime.stage == "question", "question screens should not auto-timeout")
	_expect(failures, runtime.attempted == 0, "untimed questions should not record timeout attempts")
	runtime.stage = "study"
	runtime._build_nodes()
	runtime.active = true
	runtime._update_north_marker()
	_expect(failures, runtime.north_marker_root != null and runtime.north_marker_root.visible, "north marker should be visible during study")
	_expect(failures, runtime.north_marker_root != null and runtime.north_marker_root.name == "NorthSceneMarker", "north marker should be a scene-space marker")
	_expect(failures, runtime.north_marker_root != null and runtime.north_marker_root.get_child_count() >= 6, "north marker should draw scene-space arrow and N geometry")
	_expect(failures, _north_marker_is_off_map(runtime), "north marker should sit outside the studied map")
	_expect(failures, _north_marker_is_large(runtime), "north marker should be large enough to read from the study camera")
	runtime.stage = "question"
	runtime._update_north_marker()
	_expect(failures, runtime.north_marker_root != null and not runtime.north_marker_root.visible, "north marker should disappear during questions")
	runtime.current_question = _question_by_kind(questions, "scene_reconstruction")
	runtime._rebuild_answer_overlay()
	_expect(failures, _count_named_descendants(runtime.answer_root, "TopDownMapPreview") == 4, "scene_reconstruction should show four top-down map previews")
	_expect(failures, _reconstruction_options_have_similarity(runtime.current_question), "scene_reconstruction decoys should carry difficulty-scaled similarity metadata")
	runtime.study_orientation_index = 0
	var north_up_pos := runtime._study_camera_position(0, 24.0)
	runtime.study_orientation_index = 1
	var rotated_pos := runtime._study_camera_position(0, 24.0)
	_expect(failures, north_up_pos.distance_to(rotated_pos) > 1.0, "study camera should support non-north-up starting orientations")
	runtime.allowed_question_kinds = ["aircraft_route_selection", "aircraft_continuation_selection", "aircraft_color_route_selection", "aircraft_count", "aircraft_presence", "aircraft_order", "scene_presence", "viewpoint_match"]
	var route := [Vector3(2, 1, 6), Vector3(4, 2, 6), Vector3(7, 3, 8), Vector3(11, 2, 12), Vector3(14, 1, 13)]
	var tracks := runtime._build_aircraft_tracks(runtime._rng_for("probe_aircraft_tracks"), route, 2)
	var aircraft_questions: Array = runtime._questions_for_scene({
		"scene_id": 4,
		"part": "aircraft",
		"route": route,
		"aircraft_now": route[2],
		"aircraft_next": route[3],
		"aircraft_tracks": tracks,
	})
	_expect(failures, not _has_question_kind(aircraft_questions, "scene_presence"), "scene_presence should be static-only")
	_expect(failures, not _has_question_kind(aircraft_questions, "viewpoint_match"), "viewpoint_match should be static-only")
	_expect(failures, _has_question_kind(aircraft_questions, "aircraft_count"), "aircraft_count should emit for aircraft scenes")
	_expect(failures, _has_question_kind(aircraft_questions, "aircraft_presence"), "aircraft_presence should emit for aircraft scenes")
	_expect(failures, _has_question_kind(aircraft_questions, "aircraft_order"), "aircraft_order should emit for aircraft scenes")
	_expect(failures, _question_has_four_options(aircraft_questions, "aircraft_color_route_selection"), "aircraft_color_route_selection should emit four options")
	_expect(failures, _question_has_one_correct(aircraft_questions, "aircraft_color_route_selection"), "aircraft_color_route_selection should have exactly one correct option")
	_expect(failures, _question_has_one_correct(aircraft_questions, "aircraft_count"), "aircraft_count should have exactly one correct option")
	_expect(failures, _question_has_one_correct(aircraft_questions, "aircraft_presence"), "aircraft_presence should have exactly one correct option")
	_expect(failures, _question_has_one_correct(aircraft_questions, "aircraft_order"), "aircraft_order should have exactly one correct option")
	_expect(failures, _questions_are_fair(runtime, aircraft_questions), "aircraft questions should avoid hidden IDs, answer tokens, and grid cells")
	if not failures.is_empty():
		for failure in failures:
			push_error(str(failure))
		runtime.free()
		quit(1)
		return
	print(JSON.stringify({"static_questions": questions.size(), "aircraft_questions": aircraft_questions.size()}))
	runtime.free()
	quit(0)


func _expect(failures: Array, condition: bool, message: String) -> void:
	if not condition:
		failures.append(message)


func _has_question_kind(questions: Array, kind: String) -> bool:
	for item in questions:
		var question: Dictionary = item as Dictionary
		if str(question.get("kind", "")) == kind:
			return true
	return false


func _question_by_kind(questions: Array, kind: String) -> Dictionary:
	for item in questions:
		var question: Dictionary = item as Dictionary
		if str(question.get("kind", "")) == kind:
			return question
	return {}


func _question_has_four_options(questions: Array, kind: String) -> bool:
	for item in questions:
		var question: Dictionary = item as Dictionary
		if str(question.get("kind", "")) == kind:
			return (question.get("options", []) as Array).size() == 4
	return false


func _question_has_one_correct(questions: Array, kind: String) -> bool:
	for item in questions:
		var question: Dictionary = item as Dictionary
		if str(question.get("kind", "")) != kind:
			continue
		var correct := int(question.get("correct_code", 0))
		var matches := 0
		var token_matches := 0
		for option_item in question.get("options", []):
			var option: Dictionary = option_item as Dictionary
			if int(option.get("code", 0)) == correct:
				matches += 1
			if str(option.get("answer_token", "")) == "correct":
				token_matches += 1
		return matches == 1 and token_matches == 1 and correct >= 1 and correct <= 4
	return false


func _questions_are_fair(runtime, questions: Array) -> bool:
	for item in questions:
		var question: Dictionary = item as Dictionary
		if str(question.get("answer_mode", "")) != "option_pick":
			return false
		if _text_has_unfair_prompt(str(question.get("stem", ""))):
			return false
		if _text_has_unfair_prompt(str(question.get("query_label", ""))):
			return false
		for option_item in question.get("options", []):
			var option: Dictionary = option_item as Dictionary
			if _text_has_unfair_prompt(str(option.get("statement", ""))):
				return false
			var card_label := runtime._option_card_label(option)
			if _text_has_unfair_prompt(card_label) or _text_has_internal_answer_label(card_label):
				return false
	return true


func _reconstruction_options_have_similarity(question: Dictionary) -> bool:
	var options: Array = question.get("options", [])
	if options.size() != 4:
		return false
	var exact_count := 0
	var decoy_count := 0
	for option_item in options:
		var option: Dictionary = option_item as Dictionary
		if not option.has("landmarks") or not option.has("map_similarity"):
			return false
		if str(option.get("map_similarity", "")) == "exact":
			exact_count += 1
		else:
			decoy_count += 1
	return exact_count == 1 and decoy_count == 3


func _count_named_descendants(node: Node, target_name: String) -> int:
	var count := 0
	for child in node.get_children():
		if str(child.name) == target_name:
			count += 1
		count += _count_named_descendants(child, target_name)
	return count


func _text_has_unfair_prompt(text: String) -> bool:
	var upper := text.to_upper()
	for token in ["BLD1", "BLD2", "HUM1", "HUM2", "VEH1", "VEH2", "SHP1", "SHP2", "TWR1", "TWR2", "RAD1", "HGR1", "SOL1", "TRK1", "M14", "GRID CELL", "TYPED GRID"]:
		if upper.contains(token):
			return true
	return false


func _text_has_internal_answer_label(text: String) -> bool:
	var upper := text.to_upper()
	for token in ["SCENE CORRECT", "SCENE MIRRORED", "SCENE ROTATED", "SCENE SWAPPED", "ROUTE CORRECT", "ROUTE MIRRORED", "ROUTE ROTATED"]:
		if upper.contains(token):
			return true
	return false


func _north_marker_is_off_map(runtime) -> bool:
	if runtime.north_marker_root == null:
		return false
	var map_north_edge := -runtime._grid_world_depth() * 0.5
	for child in runtime.north_marker_root.get_children():
		var node := child as Node3D
		if node != null and node.position.z >= map_north_edge:
			return false
	return true


func _north_marker_is_large(runtime) -> bool:
	if runtime.north_marker_root == null or runtime.north_marker_root.get_child_count() < 1:
		return false
	var shaft := runtime.north_marker_root.get_child(0) as Node3D
	return shaft != null and shaft.scale.z > 1.0

extends SceneTree

const SpatialRuntime = preload("res://scripts/spatial_integration_runtime.gd")


func _init() -> void:
	var runtime := SpatialRuntime.new()
	runtime.session_seed = 777
	runtime.scene_counter = 3
	runtime.grid_cols = 24
	runtime.grid_rows = 24
	runtime.allowed_question_kinds = ["scene_presence", "viewpoint_match"]
	var scene := {
		"scene_id": 3,
		"part": "static",
		"landmarks": [
			{"label": "BLD1", "kind": "building", "x": 4, "y": 5},
			{"label": "SOL1", "kind": "foot_soldiers", "x": 9, "y": 7},
			{"label": "TRK1", "kind": "truck", "x": 13, "y": 12},
			{"label": "TWR1", "kind": "tower", "x": 18, "y": 15},
		],
	}
	var questions: Array = runtime._questions_for_scene(scene)
	var failures: Array = []
	_expect(failures, _question_has_four_options(questions, "scene_presence"), "scene_presence should emit four options")
	_expect(failures, _question_has_four_options(questions, "viewpoint_match"), "viewpoint_match should emit four options")
	_expect(failures, _question_has_one_correct(questions, "scene_presence"), "scene_presence should have exactly one correct option")
	_expect(failures, _question_has_one_correct(questions, "viewpoint_match"), "viewpoint_match should have exactly one correct option")
	runtime.allowed_question_kinds = ["aircraft_route_selection", "scene_presence", "viewpoint_match"]
	var aircraft_questions: Array = runtime._questions_for_scene({"scene_id": 4, "part": "aircraft", "route": [], "aircraft_now": Vector3.ZERO, "aircraft_next": Vector3.ZERO})
	_expect(failures, not _has_question_kind(aircraft_questions, "scene_presence"), "scene_presence should be static-only")
	_expect(failures, not _has_question_kind(aircraft_questions, "viewpoint_match"), "viewpoint_match should be static-only")
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
		for option_item in question.get("options", []):
			var option: Dictionary = option_item as Dictionary
			if int(option.get("code", 0)) == correct:
				matches += 1
			if str(option.get("answer_token", "")) == "correct":
				matches += 0
		return matches == 1 and correct >= 1 and correct <= 4
	return false

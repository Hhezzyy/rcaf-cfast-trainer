extends SceneTree

const SpatialRuntime = preload("res://scripts/spatial_integration_runtime.gd")


func _init() -> void:
	var runtime := SpatialRuntime.new()
	runtime._build_nodes()
	var failures: Array = []
	var landmarks := [
		{"label": "BLD1", "kind": "building", "x": 2, "y": 2},
		{"label": "VEH1", "kind": "vehicle", "x": 4, "y": 2},
		{"label": "HUM1", "kind": "human", "x": 2, "y": 4},
		{"label": "SHP1", "kind": "sheep", "x": 4, "y": 4},
	]
	for item in landmarks:
		runtime._draw_landmark(item, true)
	_expect(failures, _mesh_count(runtime.object_root) >= 12, "expected visible mesh assets for all required landmark categories")
	_expect(failures, _label_count(runtime.object_root) == 0, "expected no floating 3D labels on Spatial Integration landmarks")
	if not failures.is_empty():
		for failure in failures:
			push_error(str(failure))
		runtime.free()
		quit(1)
		return
	print(JSON.stringify({"mesh_count": _mesh_count(runtime.object_root)}))
	runtime.free()
	quit(0)


func _mesh_count(node: Node) -> int:
	var count := 0
	if node is MeshInstance3D:
		count += 1
	for child in node.get_children():
		count += _mesh_count(child)
	return count


func _label_count(node: Node) -> int:
	var count := 0
	if node is Label3D:
		count += 1
	for child in node.get_children():
		count += _label_count(child)
	return count


func _expect(failures: Array, condition: bool, message: String) -> void:
	if not condition:
		failures.append(message)

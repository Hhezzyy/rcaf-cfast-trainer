extends SceneTree

const ChunkMapGenerator = preload("res://scripts/chunk_map_generator.gd")


func _init() -> void:
	var config := {
		"seed": 515151,
		"cols": 24,
		"rows": 24,
		"pack": "rural_mixed_v1",
		"difficulty": 0.58,
		"purpose": "spatial_integration",
		"terrain_pipeline": "si_large_scene_v2",
	}
	var first: Dictionary = ChunkMapGenerator.generate(config)
	var second: Dictionary = ChunkMapGenerator.generate(config)
	var failures: Array = []
	_expect(failures, int(first.get("cols", 0)) == 24 and int(first.get("rows", 0)) == 24, "expected 24x24 spatial integration grid")
	_expect(failures, str(first.get("terrain_pipeline", "")) == "si_large_scene_v2", "expected SI large-scene pipeline")
	_expect(failures, int(first.get("chunk_hash", 0)) == int(second.get("chunk_hash", -1)), "expected deterministic chunk hash")
	_expect(failures, int(first.get("hill_cell_count", 0)) > 0, "expected hill cells")
	_expect(failures, int(first.get("hill_cluster_count", 0)) > 0, "expected hill clusters")
	_expect(failures, _has_multicell_hill_cluster(first), "expected at least one multi-cell hill cluster")
	_expect(failures, _cells_are_bounded(first), "expected all generated cells to remain inside grid bounds")
	if not failures.is_empty():
		for failure in failures:
			push_error(str(failure))
		quit(1)
		return
	print(JSON.stringify({
		"chunk_hash": int(first.get("chunk_hash", 0)),
		"hill_cell_count": int(first.get("hill_cell_count", 0)),
		"hill_cluster_count": int(first.get("hill_cluster_count", 0)),
	}))
	quit(0)


func _expect(failures: Array, condition: bool, message: String) -> void:
	if not condition:
		failures.append(message)


func _cells_are_bounded(chunk_map: Dictionary) -> bool:
	var cols: int = int(chunk_map.get("cols", 0))
	var rows: int = int(chunk_map.get("rows", 0))
	var cells: Array = chunk_map.get("cells", [])
	if cells.size() != cols * rows:
		return false
	for item in cells:
		var cell: Dictionary = item as Dictionary
		var x: int = int(cell.get("x", -1))
		var y: int = int(cell.get("y", -1))
		if x < 0 or x >= cols or y < 0 or y >= rows:
			return false
	return true


func _has_multicell_hill_cluster(chunk_map: Dictionary) -> bool:
	for item in chunk_map.get("hill_clusters", []):
		var cluster: Dictionary = item as Dictionary
		if int(cluster.get("cell_count", 0)) >= 4:
			return true
	return false

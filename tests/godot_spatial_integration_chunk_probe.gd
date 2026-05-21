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
	_expect(failures, _hill_clusters_expose_metadata(first), "expected hill cluster metadata")
	_expect(failures, _hill_clusters_meet_target_footprints(first), "expected hill clusters to meet target footprints")
	_expect(failures, _hill_clusters_are_center_peaked(first), "expected hill clusters to be center-peaked")
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


func _hill_clusters_expose_metadata(chunk_map: Dictionary) -> bool:
	for item in chunk_map.get("hill_clusters", []):
		var cluster: Dictionary = item as Dictionary
		if str(cluster.get("cluster_id", "")) == "":
			return false
		if int(cluster.get("cell_count", 0)) <= 0:
			return false
		if int(cluster.get("peak_tier", 0)) <= 0:
			return false
		if int(cluster.get("radius", 0)) <= 0:
			return false
		if int(cluster.get("footprint_cols", 0)) <= 0 or int(cluster.get("footprint_rows", 0)) <= 0:
			return false
		if int(cluster.get("target_footprint", 0)) < 4:
			return false
	return true


func _hill_clusters_meet_target_footprints(chunk_map: Dictionary) -> bool:
	for item in chunk_map.get("hill_clusters", []):
		var cluster: Dictionary = item as Dictionary
		var target := int(cluster.get("target_footprint", 4))
		if int(cluster.get("footprint_cells", 0)) < target:
			return false
	return true


func _hill_clusters_are_center_peaked(chunk_map: Dictionary) -> bool:
	var cols: int = int(chunk_map.get("cols", 0))
	var cells: Array = chunk_map.get("cells", [])
	for item in chunk_map.get("hill_clusters", []):
		var cluster: Dictionary = item as Dictionary
		var cluster_id := str(cluster.get("cluster_id", ""))
		var cx := int(cluster.get("x", -1))
		var cy := int(cluster.get("y", -1))
		var center_idx := cy * cols + cx
		if center_idx < 0 or center_idx >= cells.size():
			return false
		var center: Dictionary = cells[center_idx] as Dictionary
		if str(center.get("cluster_id", "")) != cluster_id:
			return false
		var center_tier := int(center.get("height_tier", 0))
		if center_tier != int(cluster.get("peak_tier", 0)):
			return false
		var saw_lower_edge := false
		for cell_item in cells:
			var cell: Dictionary = cell_item as Dictionary
			if str(cell.get("cluster_id", "")) != cluster_id:
				continue
			var tier := int(cell.get("height_tier", 0))
			if tier > center_tier:
				return false
			if tier < center_tier:
				saw_lower_edge = true
		if int(cluster.get("cell_count", 0)) > 1 and not saw_lower_edge:
			return false
	return true

extends SceneTree

const ChunkMapGenerator = preload("res://scripts/chunk_map_generator.gd")


func _init() -> void:
	var config := {
		"seed": 424242,
		"cols": 50,
		"rows": 50,
		"pack": "rural_mixed_v1",
		"difficulty": 0.62,
		"purpose": "rapid_tracking",
		"road_topology": "organic_looped",
		"road_buffer_cells": 1,
		"terrain_pipeline": "terrain_first_v3",
	}
	var first: Dictionary = ChunkMapGenerator.generate(config)
	var second: Dictionary = ChunkMapGenerator.generate(config)
	var failures: Array = []
	_expect(failures, int(first.get("cols", 0)) == 50 and int(first.get("rows", 0)) == 50, "expected 50x50 rapid tracking grid")
	_expect(failures, int(first.get("chunk_hash", 0)) == int(second.get("chunk_hash", -1)), "expected deterministic chunk hash")
	_expect(failures, str(first.get("terrain_pipeline", "")) == "terrain_first_v3", "expected RT terrain_first_v3 pipeline")
	_expect(failures, int(first.get("road_component_count", 99)) == 1, "expected one connected road component")
	_expect(failures, int(first.get("road_dead_end_count", 99)) == 0, "expected zero road dead ends")
	_expect(failures, int(first.get("road_buffer_violation_count", 99)) == 0, "expected zero road buffer violations")
	_expect(failures, int(first.get("city_road_adjacency_violations", 99)) == 0, "expected zero city road-adjacency violations")
	_expect(failures, int(first.get("edge_mountain_ring_width", 0)) >= 5, "expected a broad edge mountain ring at least five cells deep")
	_expect(failures, int(first.get("mountain_cell_count", 0)) > 0, "expected mountain cells")
	_expect(failures, int(first.get("hill_cell_count", 0)) > 0, "expected hill cells")
	_expect(failures, _edge_ring_is_mountain(first), "expected all map edges to be mountains")
	_expect(failures, int(first.get("water_feature_count", 0)) >= 2, "expected rivers or lakes")
	_expect(failures, int(first.get("lake_cell_count", 0)) > 0, "expected lake cells")
	_expect(failures, int(first.get("river_cell_count", 0)) > 0, "expected river cells")
	_expect(failures, int(first.get("bridge_cell_count", 0)) > 0, "expected bridge cells at road/river crossings")
	_expect(failures, int(first.get("building_cell_count", 0)) > 0, "expected city building cells")
	_expect(failures, int(first.get("visible_building_count", 0)) > 0, "expected visible building sockets")
	_expect(failures, _city_cells_are_road_adjacent(first), "expected city/building cells to stay road adjacent")
	_expect(failures, _building_sockets_are_city_tiles(first), "expected building sockets to stay on non-road city tiles")
	_expect(failures, _cells_have_renderable_tile_ids(first), "expected every cell to have a renderable tile id")
	_expect(failures, _cells_are_bounded(first), "expected all generated cells to remain inside grid bounds")
	if not failures.is_empty():
		for failure in failures:
			push_error(str(failure))
		quit(1)
		return
	print(JSON.stringify({
		"chunk_hash": int(first.get("chunk_hash", 0)),
		"road_nodes": (first.get("road_nodes", []) as Array).size(),
		"water_feature_count": int(first.get("water_feature_count", 0)),
		"building_cell_count": int(first.get("building_cell_count", 0)),
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


func _cells_have_renderable_tile_ids(chunk_map: Dictionary) -> bool:
	var valid := {
		"grassland": true,
		"field": true,
		"forest": true,
		"forest_edge": true,
		"city": true,
		"city_edge": true,
		"road_buffer": true,
		"road_ns": true,
		"road_ew": true,
		"road_corner": true,
		"road_t": true,
		"road_cross": true,
		"river_ns": true,
		"river_ew": true,
		"lake": true,
		"bridge_ew_over_ns": true,
		"hill": true,
		"mountain": true,
	}
	var cells: Array = chunk_map.get("cells", [])
	for item in cells:
		var cell: Dictionary = item as Dictionary
		var tile_id := str(cell.get("tile_id", ""))
		if not valid.has(tile_id):
			return false
	return true


func _edge_ring_is_mountain(chunk_map: Dictionary) -> bool:
	var cols: int = int(chunk_map.get("cols", 0))
	var rows: int = int(chunk_map.get("rows", 0))
	var cells: Array = chunk_map.get("cells", [])
	for item in cells:
		var cell: Dictionary = item as Dictionary
		var x: int = int(cell.get("x", -1))
		var y: int = int(cell.get("y", -1))
		if x != 0 and y != 0 and x != cols - 1 and y != rows - 1:
			continue
		var terrain := str(cell.get("terrain", ""))
		if terrain != "mountain":
			return false
	return true


func _city_cells_are_road_adjacent(chunk_map: Dictionary) -> bool:
	var cols: int = int(chunk_map.get("cols", 0))
	var cells: Array = chunk_map.get("cells", [])
	for item in cells:
		var cell: Dictionary = item as Dictionary
		var terrain := str(cell.get("terrain", ""))
		if terrain != "city" and terrain != "city_edge":
			continue
		if _cell_has_road(cell):
			return false
		var x: int = int(cell.get("x", -1))
		var y: int = int(cell.get("y", -1))
		if not _road_within(cells, cols, x, y, 2):
			return false
	return true


func _building_sockets_are_city_tiles(chunk_map: Dictionary) -> bool:
	var cols: int = int(chunk_map.get("cols", 0))
	var cells: Array = chunk_map.get("cells", [])
	var sockets: Dictionary = chunk_map.get("asset_sockets", {})
	var building_sockets: Array = sockets.get("building", [])
	for item in building_sockets:
		var socket: Dictionary = item as Dictionary
		var x: int = int(socket.get("x", -1))
		var y: int = int(socket.get("y", -1))
		var idx := y * cols + x
		if x < 0 or y < 0 or idx < 0 or idx >= cells.size():
			return false
		var cell: Dictionary = cells[idx] as Dictionary
		var terrain := str(cell.get("terrain", ""))
		var tile_id := str(cell.get("tile_id", ""))
		if terrain != "city" and terrain != "city_edge":
			return false
		if tile_id != "city" and tile_id != "city_edge":
			return false
		if _cell_has_road(cell):
			return false
	return true


func _road_within(cells: Array, cols: int, x: int, y: int, max_distance: int) -> bool:
	for dy in range(-max_distance, max_distance + 1):
		for dx in range(-max_distance, max_distance + 1):
			if abs(dx) + abs(dy) > max_distance:
				continue
			var nx := x + dx
			var ny := y + dy
			if nx < 0 or nx >= cols or ny < 0:
				continue
			var idx := ny * cols + nx
			if idx >= 0 and idx < cells.size() and _cell_has_road(cells[idx] as Dictionary):
				return true
	return false


func _cell_has_road(cell: Dictionary) -> bool:
	return bool(cell.get("road_n", false)) or bool(cell.get("road_e", false)) or bool(cell.get("road_s", false)) or bool(cell.get("road_w", false))

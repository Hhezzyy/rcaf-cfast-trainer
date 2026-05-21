extends RefCounted

const DEFAULT_COLS := 8
const DEFAULT_ROWS := 8
const DEFAULT_PACK := "rural_mixed_v1"
const DIRS := ["n", "e", "s", "w"]
const TILE_CATALOG := {
	"grassland": {"terrain": "grassland", "roads": [], "rivers": [], "transition": false},
	"road_buffer": {"terrain": "road_buffer", "roads": [], "rivers": [], "transition": true},
	"field": {"terrain": "field", "roads": [], "rivers": [], "transition": false},
	"hill": {"terrain": "hill", "roads": [], "rivers": [], "transition": false},
	"mountain": {"terrain": "mountain", "roads": [], "rivers": [], "transition": false},
	"forest": {"terrain": "forest", "roads": [], "rivers": [], "transition": false},
	"forest_edge": {"terrain": "forest_edge", "roads": [], "rivers": [], "transition": true},
	"city": {"terrain": "city", "roads": [], "rivers": [], "transition": false},
	"city_edge": {"terrain": "city_edge", "roads": [], "rivers": [], "transition": true},
	"road_ns": {"terrain": "field", "roads": ["n", "s"], "rivers": [], "transition": false},
	"road_ew": {"terrain": "field", "roads": ["e", "w"], "rivers": [], "transition": false},
	"road_corner": {"terrain": "field", "roads": ["turn"], "rivers": [], "transition": false},
	"road_t": {"terrain": "field", "roads": ["t"], "rivers": [], "transition": false},
	"road_cross": {"terrain": "field", "roads": ["n", "e", "s", "w"], "rivers": [], "transition": false},
	"river_ns": {"terrain": "river", "roads": [], "rivers": ["n", "s"], "transition": false},
	"river_ew": {"terrain": "river", "roads": [], "rivers": ["e", "w"], "transition": false},
	"lake": {"terrain": "lake", "roads": [], "rivers": [], "transition": false},
	"bridge_ew_over_ns": {"terrain": "bridge", "roads": ["e", "w"], "rivers": ["n", "s"], "transition": false},
}


static func generate(config: Dictionary) -> Dictionary:
	var cols: int = max(2, int(config.get("cols", DEFAULT_COLS)))
	var rows: int = max(2, int(config.get("rows", DEFAULT_ROWS)))
	var seed_value: int = max(1, int(config.get("seed", 1)))
	var pack: String = str(config.get("pack", DEFAULT_PACK))
	var purpose: String = str(config.get("purpose", "shared"))
	var difficulty: float = clampf(float(config.get("difficulty", 0.5)), 0.0, 1.0)
	var rng: RandomNumberGenerator = RandomNumberGenerator.new()
	rng.seed = seed_value + _string_salt(pack) * 17 + _string_salt(purpose) * 31
	var terrain_pipeline: String = str(config.get("terrain_pipeline", ""))
	if purpose == "rapid_tracking" or terrain_pipeline == "roads_first_v2" or terrain_pipeline == "terrain_first_v3":
		if terrain_pipeline == "roads_first_v2":
			return _generate_rapid_tracking_v2(config, rng, cols, rows, seed_value, pack, difficulty)
		return _generate_rapid_tracking_v3(config, rng, cols, rows, seed_value, pack, difficulty)
	if purpose == "spatial_integration" and terrain_pipeline == "si_large_scene_v2":
		return _generate_spatial_integration_v2(config, rng, cols, rows, seed_value, pack, difficulty)
	var cells := _make_base_cells(cols, rows, rng)
	_lay_river(cells, cols, rows, rng)
	_lay_roads(cells, cols, rows, rng, difficulty)
	_lay_city_patches(cells, cols, rows, rng)
	_lay_forest_patches(cells, cols, rows, rng)
	_lay_transition_cells(cells, cols, rows)
	_repair_invalid_cells(cells, cols, rows)
	var rule_violations := _validate_edges(cells, cols, rows)
	_finalize_tiles(cells)
	var road_graph := _build_road_graph(cells, cols, rows)
	var sockets := _build_asset_sockets(cells, cols, rows, rng)
	var route_cells := _main_route_cells(cells, cols, rows)
	var city_centers := _terrain_cells(cells, "city")
	var forest_cells := _terrain_cells(cells, "forest")
	var chunk_hash := _hash_chunk_map(cells, cols, rows, seed_value, pack, route_cells)
	return {
		"schema": "chunk_map_v1",
		"pack": pack,
		"purpose": purpose,
		"cols": cols,
		"rows": rows,
		"seed": seed_value,
		"cells": cells,
		"route_cells": route_cells,
		"road_nodes": road_graph.get("nodes", []),
		"road_edges": road_graph.get("edges", []),
		"city_centers": city_centers,
		"forest_sockets": sockets.get("forest", []),
		"asset_sockets": sockets,
		"chunk_hash": chunk_hash,
		"rule_violations": rule_violations,
	}


static func _generate_rapid_tracking_v2(config: Dictionary, rng: RandomNumberGenerator, cols: int, rows: int, seed_value: int, pack: String, difficulty: float) -> Dictionary:
	var buffer_cells: int = max(1, int(config.get("road_buffer_cells", 1)))
	var cells := _rt_make_plain_base_cells(cols, rows, rng)
	var road_anchors := _rt_lay_organic_looped_roads(cells, cols, rows, rng, difficulty)
	_rt_repair_road_network(cells, cols, rows)
	var road_component_count := _rt_road_component_count(cells, cols, rows)
	var road_dead_end_count := _rt_road_dead_end_count(cells)
	_rt_reserve_road_buffers(cells, cols, rows, buffer_cells)
	var water_metrics := _rt_lay_water_features(cells, cols, rows, rng, difficulty)
	var city_anchors := _rt_lay_natural_city_blocks(cells, cols, rows, rng, road_anchors, difficulty)
	_rt_lay_secondary_terrain(cells, cols, rows, rng, difficulty)
	var road_buffer_violation_count := _rt_road_buffer_violation_count(cells, cols, rows, buffer_cells)
	var city_road_adjacency_violations := _rt_city_road_adjacency_violations(cells, cols, rows)
	var rule_violations := _validate_edges(cells, cols, rows) + road_buffer_violation_count + city_road_adjacency_violations
	_finalize_tiles(cells)
	var road_graph := _build_road_graph(cells, cols, rows)
	var sockets := _build_asset_sockets(cells, cols, rows, rng)
	var route_cells := _rt_route_cells_from_road_graph(road_graph.get("nodes", []))
	var chunk_hash := _hash_chunk_map(cells, cols, rows, seed_value, pack, route_cells)
	return {
		"schema": "chunk_map_v1",
		"terrain_pipeline": "roads_first_v2",
		"road_topology": str(config.get("road_topology", "organic_looped")),
		"pack": pack,
		"purpose": "rapid_tracking",
		"cols": cols,
		"rows": rows,
		"seed": seed_value,
		"cells": cells,
		"route_cells": route_cells,
		"road_nodes": road_graph.get("nodes", []),
		"road_edges": road_graph.get("edges", []),
		"city_centers": city_anchors,
		"forest_sockets": sockets.get("forest", []),
		"asset_sockets": sockets,
		"chunk_hash": chunk_hash,
		"rule_violations": rule_violations,
		"road_component_count": road_component_count,
		"road_dead_end_count": road_dead_end_count,
		"road_buffer_violation_count": road_buffer_violation_count,
		"city_road_adjacency_violations": city_road_adjacency_violations,
		"water_feature_count": int(water_metrics.get("water_feature_count", 0)),
		"lake_cell_count": int(water_metrics.get("lake_cell_count", 0)),
		"river_cell_count": int(water_metrics.get("river_cell_count", 0)),
		"bridge_cell_count": int(water_metrics.get("bridge_cell_count", 0)),
	}


static func _generate_rapid_tracking_v3(config: Dictionary, rng: RandomNumberGenerator, cols: int, rows: int, seed_value: int, pack: String, difficulty: float) -> Dictionary:
	var buffer_cells: int = max(1, int(config.get("road_buffer_cells", 1)))
	var cells := _rt_make_plain_base_cells(cols, rows, rng)
	var terrain_metrics := _rt_lay_priority_terrain(cells, cols, rows, rng, difficulty)
	var road_anchors := _rt_lay_organic_looped_roads(cells, cols, rows, rng, difficulty)
	_rt_repair_road_network(cells, cols, rows)
	_rt_reserve_road_buffers(cells, cols, rows, buffer_cells)
	var city_anchors := _rt_lay_natural_city_blocks(cells, cols, rows, rng, road_anchors, difficulty)
	_rt_lay_secondary_terrain(cells, cols, rows, rng, difficulty)
	var road_component_count := _rt_road_component_count(cells, cols, rows)
	var road_dead_end_count := _rt_road_dead_end_count(cells)
	var road_buffer_violation_count := _rt_road_buffer_violation_count(cells, cols, rows, buffer_cells)
	var city_road_adjacency_violations := _rt_city_road_adjacency_violations(cells, cols, rows)
	var rule_violations := _validate_edges(cells, cols, rows) + road_buffer_violation_count + city_road_adjacency_violations
	_finalize_tiles(cells)
	var road_graph := _build_road_graph(cells, cols, rows)
	var sockets := _build_asset_sockets(cells, cols, rows, rng)
	var route_cells := _rt_route_cells_from_road_graph(road_graph.get("nodes", []))
	var chunk_hash := _hash_chunk_map(cells, cols, rows, seed_value, pack, route_cells)
	var building_socket_count := (sockets.get("building", []) as Array).size()
	return {
		"schema": "chunk_map_v1",
		"terrain_pipeline": "terrain_first_v3",
		"road_topology": str(config.get("road_topology", "organic_looped")),
		"pack": pack,
		"purpose": "rapid_tracking",
		"cols": cols,
		"rows": rows,
		"seed": seed_value,
		"cells": cells,
		"route_cells": route_cells,
		"road_nodes": road_graph.get("nodes", []),
		"road_edges": road_graph.get("edges", []),
		"city_centers": city_anchors,
		"forest_sockets": sockets.get("forest", []),
		"asset_sockets": sockets,
		"chunk_hash": chunk_hash,
		"rule_violations": rule_violations,
		"road_component_count": road_component_count,
		"road_dead_end_count": road_dead_end_count,
		"road_buffer_violation_count": road_buffer_violation_count,
		"city_road_adjacency_violations": city_road_adjacency_violations,
		"water_feature_count": int(terrain_metrics.get("water_feature_count", 0)),
		"lake_cell_count": _rt_count_terrain(cells, "lake"),
		"river_cell_count": _rt_river_cell_count(cells),
		"bridge_cell_count": _rt_bridge_cell_count(cells),
		"mountain_cell_count": _rt_count_terrain(cells, "mountain"),
		"hill_cell_count": _rt_count_terrain(cells, "hill"),
		"edge_mountain_ring_width": int(terrain_metrics.get("edge_mountain_ring_width", 0)),
		"terrain_blocked_road_crossings": _rt_terrain_blocked_road_crossing_count(cells),
		"building_cell_count": _rt_city_building_cell_count(cells),
		"visible_building_count": building_socket_count,
		"hill_clusters": terrain_metrics.get("hill_clusters", []),
		"mountain_clusters": terrain_metrics.get("mountain_clusters", []),
	}


static func _generate_spatial_integration_v2(config: Dictionary, rng: RandomNumberGenerator, cols: int, rows: int, seed_value: int, pack: String, difficulty: float) -> Dictionary:
	var cells := _make_base_cells(cols, rows, rng)
	_lay_river(cells, cols, rows, rng)
	_lay_roads(cells, cols, rows, rng, difficulty)
	_lay_city_patches(cells, cols, rows, rng)
	_lay_forest_patches(cells, cols, rows, rng)
	var hill_metrics := _si_lay_multicell_hills(cells, cols, rows, rng, difficulty)
	_lay_transition_cells(cells, cols, rows)
	_repair_invalid_cells(cells, cols, rows)
	var rule_violations := _validate_edges(cells, cols, rows)
	_finalize_tiles(cells)
	var road_graph := _build_road_graph(cells, cols, rows)
	var sockets := _build_asset_sockets(cells, cols, rows, rng)
	var route_cells := _main_route_cells(cells, cols, rows)
	var city_centers := _terrain_cells(cells, "city")
	var forest_cells := _terrain_cells(cells, "forest")
	var chunk_hash := _hash_chunk_map(cells, cols, rows, seed_value, pack, route_cells)
	return {
		"schema": "chunk_map_v1",
		"terrain_pipeline": "si_large_scene_v2",
		"pack": pack,
		"purpose": "spatial_integration",
		"cols": cols,
		"rows": rows,
		"seed": seed_value,
		"cells": cells,
		"route_cells": route_cells,
		"road_nodes": road_graph.get("nodes", []),
		"road_edges": road_graph.get("edges", []),
		"city_centers": city_centers,
		"forest_sockets": sockets.get("forest", []),
		"asset_sockets": sockets,
		"chunk_hash": chunk_hash,
		"rule_violations": rule_violations,
		"hill_cell_count": int(hill_metrics.get("hill_cell_count", 0)),
		"hill_cluster_count": int(hill_metrics.get("hill_cluster_count", 0)),
		"hill_clusters": hill_metrics.get("hill_clusters", []),
	}


static func _si_lay_multicell_hills(cells: Array, cols: int, rows: int, rng: RandomNumberGenerator, difficulty: float) -> Dictionary:
	var clusters: Array = []
	var hill_cell_count := 0
	if cols < 6 or rows < 6:
		return {"hill_cell_count": 0, "hill_cluster_count": 0, "hill_clusters": clusters}
	var cluster_target: int = clampi(4 + int(round(difficulty * 3.0)), 4, 8)
	var margin := 2
	var attempts := 0
	while clusters.size() < cluster_target and attempts < cluster_target * 8:
		attempts += 1
		var center := Vector2i(int(rng.randi_range(margin, cols - margin - 1)), int(rng.randi_range(margin, rows - margin - 1)))
		var center_cell := _cell(cells, cols, center.x, center.y)
		if center_cell.is_empty() or _cell_has_road(center_cell) or _cell_has_river(center_cell):
			continue
		var center_terrain := str(center_cell.get("terrain", "grassland"))
		if center_terrain == "city" or center_terrain == "city_edge" or center_terrain == "forest" or center_terrain == "forest_edge" or center_terrain == "lake" or center_terrain == "bridge" or center_terrain == "hill":
			continue
		var target_footprint: int = 6 if difficulty >= 0.55 else 4
		var radius: int = max(2, int(ceil(float(target_footprint) / 2.0)))
		var cluster_id := "si_hill_cluster_" + str(clusters.size())
		var peak_tier: int = clampi(radius + int(rng.randi_range(3, 5)), 4, 9)
		var painted := 0
		var min_x := cols
		var min_y := rows
		var max_x := -1
		var max_y := -1
		for dy in range(-radius, radius + 1):
			for dx in range(-radius, radius + 1):
				var dist := Vector2(float(dx), float(dy)).length()
				if dist > float(radius) + 0.25:
					continue
				var x := center.x + dx
				var y := center.y + dy
				if x <= 0 or x >= cols - 1 or y <= 0 or y >= rows - 1:
					continue
				var cell := _cell(cells, cols, x, y)
				if cell.is_empty() or _cell_has_road(cell) or _cell_has_river(cell):
					continue
				var terrain := str(cell.get("terrain", "grassland"))
				if terrain == "city" or terrain == "city_edge" or terrain == "forest" or terrain == "forest_edge" or terrain == "lake" or terrain == "bridge":
					continue
				cell["terrain"] = "hill"
				cell["height_tier"] = clampi(peak_tier - int(ceil(dist * 1.35)), 1, peak_tier)
				cell["cluster_id"] = cluster_id
				cell["cluster_peak_tier"] = peak_tier
				_set_cell(cells, cols, x, y, cell)
				painted += 1
				min_x = mini(min_x, x)
				min_y = mini(min_y, y)
				max_x = maxi(max_x, x)
				max_y = maxi(max_y, y)
		if painted >= 4:
			clusters.append({
				"x": center.x,
				"y": center.y,
				"radius": radius,
				"cluster_id": cluster_id,
				"cell_count": painted,
				"peak_tier": peak_tier,
				"min_x": min_x,
				"max_x": max_x,
				"min_y": min_y,
				"max_y": max_y,
				"footprint_cols": max_x - min_x + 1,
				"footprint_rows": max_y - min_y + 1,
				"footprint_cells": max(max_x - min_x + 1, max_y - min_y + 1),
				"target_footprint": target_footprint,
			})
			hill_cell_count += painted
	return {"hill_cell_count": hill_cell_count, "hill_cluster_count": clusters.size(), "hill_clusters": clusters}


static func _rt_lay_priority_terrain(cells: Array, cols: int, rows: int, rng: RandomNumberGenerator, difficulty: float) -> Dictionary:
	var ring_width := _rt_lay_edge_mountain_ring(cells, cols, rows, rng)
	var cluster_metrics := _rt_lay_interior_hill_clusters(cells, cols, rows, rng, difficulty, ring_width)
	var water_metrics := _rt_lay_water_features_v3(cells, cols, rows, rng, difficulty, ring_width)
	return {
		"edge_mountain_ring_width": ring_width,
		"hill_clusters": cluster_metrics.get("hill_clusters", []),
		"mountain_clusters": cluster_metrics.get("mountain_clusters", []),
		"water_feature_count": int(water_metrics.get("water_feature_count", 0)),
	}


static func _rt_lay_edge_mountain_ring(cells: Array, cols: int, rows: int, rng: RandomNumberGenerator) -> int:
	var max_width: int = clampi(int(min(cols, rows) / 7), 5, 8)
	var width: int = int(rng.randi_range(5, max_width))
	var peak_tier := clampi(width + 5, 8, 12)
	for y in range(rows):
		for x in range(cols):
			var edge_depth: int = min(min(x, y), min(cols - 1 - x, rows - 1 - y))
			if edge_depth >= width:
				continue
			var cell := _cell(cells, cols, x, y)
			cell["terrain"] = "mountain"
			cell["edge_mountain"] = true
			cell["height_tier"] = clampi(peak_tier - edge_depth, 3, peak_tier)
			cell["cluster_id"] = "edge_ring"
			cell["cluster_peak_tier"] = peak_tier
			_set_cell(cells, cols, x, y, cell)
	return width


static func _rt_lay_interior_hill_clusters(cells: Array, cols: int, rows: int, rng: RandomNumberGenerator, difficulty: float, ring_width: int) -> Dictionary:
	var hill_clusters: Array = []
	var mountain_clusters: Array = []
	var margin: int = clampi(ring_width + 5, 6, max(6, int(min(cols, rows) / 3)))
	var hill_count: int = clampi(3 + int(round(difficulty * 3.0)), 3, 6)
	var hill_attempts := 0
	while hill_clusters.size() < hill_count and hill_attempts < hill_count * 8:
		hill_attempts += 1
		var center := Vector2i(int(rng.randi_range(margin, cols - margin - 1)), int(rng.randi_range(margin, rows - margin - 1)))
		var center_cell := _cell(cells, cols, center.x, center.y)
		if center_cell.is_empty() or bool(center_cell.get("edge_mountain", false)) or str(center_cell.get("terrain", "grassland")) != "grassland":
			continue
		var target_footprint: int = 6 if difficulty >= 0.55 else 4
		var radius: int = max(2, int(ceil(float(target_footprint) / 2.0)))
		var cluster_id := "hill_cluster_" + str(hill_clusters.size())
		var peak_tier: int = clampi(radius + int(rng.randi_range(3, 5)), 4, 9)
		var hill_cluster := _rt_paint_elevation_cluster(cells, cols, rows, center, radius, "hill", cluster_id, peak_tier, target_footprint)
		if int(hill_cluster.get("cell_count", 0)) > 0:
			hill_clusters.append(hill_cluster)
	var mountain_count: int = 1 + int(round(difficulty))
	var mountain_attempts := 0
	while mountain_clusters.size() < mountain_count and mountain_attempts < mountain_count * 10:
		mountain_attempts += 1
		var center_m := Vector2i(int(rng.randi_range(margin, cols - margin - 1)), int(rng.randi_range(margin, rows - margin - 1)))
		var center_cell_m := _cell(cells, cols, center_m.x, center_m.y)
		if center_cell_m.is_empty() or bool(center_cell_m.get("edge_mountain", false)) or str(center_cell_m.get("terrain", "grassland")) != "grassland":
			continue
		var target_footprint_m: int = 6 if difficulty >= 0.55 else 4
		var radius_m: int = max(2, int(ceil(float(target_footprint_m) / 2.0)))
		var cluster_id_m := "mountain_cluster_" + str(mountain_clusters.size())
		var peak_tier_m: int = clampi(radius_m + int(rng.randi_range(5, 7)), 6, 11)
		var mountain_cluster := _rt_paint_elevation_cluster(cells, cols, rows, center_m, radius_m, "mountain", cluster_id_m, peak_tier_m, target_footprint_m)
		if int(mountain_cluster.get("cell_count", 0)) > 0:
			mountain_clusters.append(mountain_cluster)
	return {"hill_clusters": hill_clusters, "mountain_clusters": mountain_clusters}


static func _rt_paint_elevation_cluster(cells: Array, cols: int, rows: int, center: Vector2i, radius: int, terrain_name: String, cluster_id: String, peak_tier: int, target_footprint: int) -> Dictionary:
	var painted := 0
	var min_x := cols
	var min_y := rows
	var max_x := -1
	var max_y := -1
	for dy in range(-radius, radius + 1):
		for dx in range(-radius, radius + 1):
			var dist := Vector2(float(dx), float(dy)).length()
			if dist > float(radius) + 0.35:
				continue
			var x := center.x + dx
			var y := center.y + dy
			var cell := _cell(cells, cols, x, y)
			if cell.is_empty() or bool(cell.get("edge_mountain", false)):
				continue
			if str(cell.get("terrain", "grassland")) != "grassland":
				continue
			cell["terrain"] = terrain_name
			cell["height_tier"] = clampi(peak_tier - int(ceil(dist * 1.35)), 1, peak_tier)
			cell["cluster_id"] = cluster_id
			cell["cluster_peak_tier"] = peak_tier
			_set_cell(cells, cols, x, y, cell)
			painted += 1
			min_x = mini(min_x, x)
			min_y = mini(min_y, y)
			max_x = maxi(max_x, x)
			max_y = maxi(max_y, y)
	return {
		"x": center.x,
		"y": center.y,
		"radius": radius,
		"cluster_id": cluster_id,
		"cell_count": painted,
		"peak_tier": peak_tier,
		"min_x": min_x,
		"max_x": max_x,
		"min_y": min_y,
		"max_y": max_y,
		"footprint_cols": max_x - min_x + 1,
		"footprint_rows": max_y - min_y + 1,
		"footprint_cells": max(max_x - min_x + 1, max_y - min_y + 1),
		"target_footprint": target_footprint,
	}


static func _rt_lay_water_features_v3(cells: Array, cols: int, rows: int, rng: RandomNumberGenerator, difficulty: float, ring_width: int) -> Dictionary:
	var river_cell_count := 0
	var lake_cell_count := 0
	var river_count := 1 + int(round(difficulty))
	var margin: int = clampi(ring_width + 1, 4, max(4, int(min(cols, rows) / 3)))
	for i in range(river_count):
		var x := clampi(int(round(float(cols) * (0.30 + 0.32 * rng.randf()))), margin, cols - margin - 1)
		var current := Vector2i(x, margin)
		_rt_mark_river_cell(cells, cols, current)
		river_cell_count += 1
		for y in range(margin + 1, rows - margin):
			var target_x := x
			if y % 5 == 0 and rng.randf() < 0.65:
				target_x = clampi(x + int(rng.randi_range(-2, 2)), margin, cols - margin - 1)
			while x != target_x:
				var horizontal := Vector2i(x + _sign_int(target_x - x), y - 1)
				_rt_mark_river_cell(cells, cols, horizontal)
				_rt_mark_river_connection(cells, cols, current, horizontal)
				current = horizontal
				x = horizontal.x
				river_cell_count += 1
			var p := Vector2i(x, y)
			_rt_mark_river_cell(cells, cols, p)
			_rt_mark_river_connection(cells, cols, current, p)
			current = p
			river_cell_count += 1
	var lake_target := 2 + int(round(difficulty * 2.0))
	var made := 0
	var attempts := 0
	while made < lake_target and attempts < 100:
		attempts += 1
		var center := Vector2i(int(rng.randi_range(margin + 3, cols - margin - 4)), int(rng.randi_range(margin + 3, rows - margin - 4)))
		if not _rt_water_cell_available(cells, cols, rows, center.x, center.y):
			continue
		var radius := int(rng.randi_range(2, 4))
		var painted := 0
		for dy in range(-radius, radius + 1):
			for dx in range(-radius, radius + 1):
				if Vector2(float(dx), float(dy)).length() > float(radius) + 0.35:
					continue
				var x_l := center.x + dx
				var y_l := center.y + dy
				if not _rt_water_cell_available(cells, cols, rows, x_l, y_l):
					continue
				var lake := _cell(cells, cols, x_l, y_l)
				lake["terrain"] = "lake"
				lake["is_lake"] = true
				_set_cell(cells, cols, x_l, y_l, lake)
				painted += 1
		if painted > 0:
			lake_cell_count += painted
			made += 1
	return {
		"water_feature_count": river_count + made,
		"lake_cell_count": lake_cell_count,
		"river_cell_count": river_cell_count,
	}


static func _rt_make_plain_base_cells(cols: int, rows: int, rng: RandomNumberGenerator) -> Array:
	var cells := []
	for y in range(rows):
		for x in range(cols):
			cells.append({
				"x": x,
				"y": y,
				"terrain": "grassland",
				"tile_id": "grassland",
				"variant": int(rng.randi_range(0, 4)),
				"road_n": false,
				"road_e": false,
				"road_s": false,
				"road_w": false,
				"river_n": false,
				"river_e": false,
				"river_s": false,
				"river_w": false,
				"is_bridge": false,
				"is_lake": false,
				"road_buffer": false,
				"urban_road": false,
				"bridge_approach": false,
				"city_anchor": false,
				"edge_mountain": false,
				"height_tier": 0,
				"cluster_id": "",
				"terrain_blocked_road_crossing": false,
			})
	return cells


static func _rt_lay_organic_looped_roads(cells: Array, cols: int, rows: int, rng: RandomNumberGenerator, difficulty: float) -> Array:
	var anchors: Array = []
	var cx: int = int(cols / 2)
	var cy: int = int(rows / 2)
	var road_margin: int = clampi(_rt_edge_mountain_ring_width(cells, cols, rows) + 2, 4, max(4, int(min(cols, rows) / 3)))
	var left: int = max(road_margin, int(cols * 0.10) + int(rng.randi_range(-2, 2)))
	var right: int = min(cols - road_margin - 1, int(cols * 0.88) + int(rng.randi_range(-2, 2)))
	var top: int = max(road_margin, int(rows * 0.12) + int(rng.randi_range(-2, 2)))
	var bottom: int = min(rows - road_margin - 1, int(rows * 0.86) + int(rng.randi_range(-2, 2)))
	var outer := [
		Vector2i(left, cy),
		Vector2i(left + int(cols * 0.12), top),
		Vector2i(cx, top + int(rows * 0.03)),
		Vector2i(right - int(cols * 0.10), top + int(rows * 0.14)),
		Vector2i(right, cy + int(rows * 0.05)),
		Vector2i(right - int(cols * 0.14), bottom),
		Vector2i(cx - int(cols * 0.05), bottom - int(rows * 0.02)),
		Vector2i(left + int(cols * 0.09), bottom - int(rows * 0.10)),
	]
	_rt_mark_loop(cells, cols, rows, outer, rng)
	anchors.append_array(outer)

	var inner_left: int = max(road_margin + 2, int(cols * 0.28) + int(rng.randi_range(-2, 2)))
	var inner_right: int = min(cols - road_margin - 3, int(cols * 0.72) + int(rng.randi_range(-2, 2)))
	var inner_top: int = max(road_margin + 2, int(rows * 0.30) + int(rng.randi_range(-2, 2)))
	var inner_bottom: int = min(rows - road_margin - 3, int(rows * 0.70) + int(rng.randi_range(-2, 2)))
	var inner := [
		Vector2i(inner_left, cy - int(rows * 0.04)),
		Vector2i(cx - int(cols * 0.08), inner_top),
		Vector2i(inner_right, inner_top + int(rows * 0.08)),
		Vector2i(inner_right - int(cols * 0.06), inner_bottom),
		Vector2i(cx - int(cols * 0.12), inner_bottom + int(rows * 0.02)),
	]
	_rt_mark_loop(cells, cols, rows, inner, rng)
	anchors.append_array(inner)

	if difficulty > 0.25:
		var side := [
			Vector2i(max(road_margin + 1, int(cols * 0.14)), max(road_margin + 1, int(rows * 0.20))),
			Vector2i(max(road_margin + 1, int(cols * 0.24)), max(road_margin + 1, int(rows * 0.36))),
			Vector2i(max(road_margin + 1, int(cols * 0.20)), min(rows - road_margin - 2, int(rows * 0.58))),
			Vector2i(max(road_margin + 1, int(cols * 0.32)), min(rows - road_margin - 2, int(rows * 0.76))),
			Vector2i(max(road_margin + 1, int(cols * 0.42)), min(rows - road_margin - 2, int(rows * 0.62))),
			Vector2i(max(road_margin + 1, int(cols * 0.38)), max(road_margin + 1, int(rows * 0.28))),
		]
		_rt_mark_loop(cells, cols, rows, side, rng)
		anchors.append_array(side)

	# Organic arterial connectors join existing loops without creating dead-end spurs.
	_rt_carve_organic_connection(cells, cols, rows, outer[0], inner[0], rng)
	_rt_carve_organic_connection(cells, cols, rows, outer[2], inner[1], rng)
	_rt_carve_organic_connection(cells, cols, rows, outer[4], inner[2], rng)
	_rt_carve_organic_connection(cells, cols, rows, outer[6], inner[4], rng)
	return anchors


static func _rt_mark_loop(cells: Array, cols: int, rows: int, points: Array, rng: RandomNumberGenerator) -> void:
	if points.size() < 3:
		return
	for i in range(points.size()):
		var start: Vector2i = points[i]
		var finish: Vector2i = points[(i + 1) % points.size()]
		_rt_carve_organic_connection(cells, cols, rows, start, finish, rng)


static func _rt_carve_organic_connection(cells: Array, cols: int, rows: int, start: Vector2i, finish: Vector2i, rng: RandomNumberGenerator) -> void:
	var current := _rt_clamp_point(start, cols, rows)
	var goal := _rt_clamp_point(finish, cols, rows)
	var guard := 0
	while current != goal and guard < cols * rows:
		var dx: int = goal.x - current.x
		var dy: int = goal.y - current.y
		var use_x: bool = abs(dx) >= abs(dy)
		if rng.randf() < 0.30:
			use_x = not use_x
		var next := current
		if use_x and dx != 0:
			next.x += _sign_int(dx)
		elif dy != 0:
			next.y += _sign_int(dy)
		elif dx != 0:
			next.x += _sign_int(dx)
		next = _rt_clamp_point(next, cols, rows)
		if _rt_cell_blocks_road(cells, cols, rows, next):
			var alternate := _rt_find_unblocked_road_step(cells, cols, rows, current, goal, next)
			if alternate != current:
				next = alternate
		if next == current:
			break
		_rt_connect_road_cells(cells, cols, rows, current, next)
		current = next
		guard += 1


static func _rt_connect_road_cells(cells: Array, cols: int, rows: int, a: Vector2i, b: Vector2i) -> void:
	if abs(a.x - b.x) + abs(a.y - b.y) != 1:
		return
	var dir := "e"
	if b.x < a.x:
		dir = "w"
	elif b.y > a.y:
		dir = "s"
	elif b.y < a.y:
		dir = "n"
	_mark_edge(cells, cols, a.x, a.y, "road", dir, true)
	_mark_edge(cells, cols, b.x, b.y, "road", _opposite_dir(dir), true)
	var ca := _cell(cells, cols, a.x, a.y)
	var cb := _cell(cells, cols, b.x, b.y)
	ca = _rt_prepare_road_cell(ca)
	cb = _rt_prepare_road_cell(cb)
	_set_cell(cells, cols, a.x, a.y, ca)
	_set_cell(cells, cols, b.x, b.y, cb)


static func _rt_prepare_road_cell(cell: Dictionary) -> Dictionary:
	var terrain := str(cell.get("terrain", ""))
	if terrain == "grassland" or terrain == "road_buffer" or terrain == "field" or terrain == "forest" or terrain == "forest_edge":
		cell["terrain"] = "grassland"
	elif terrain == "river" or _cell_has_river(cell):
		cell["is_bridge"] = true
		cell["terrain"] = "bridge"
	elif terrain == "hill" or terrain == "mountain":
		cell["terrain_blocked_road_crossing"] = true
		cell["mountain_pass"] = terrain == "mountain"
	return cell


static func _rt_repair_road_network(cells: Array, cols: int, rows: int) -> void:
	for _pass in range(3):
		var components := _rt_road_components(cells, cols, rows)
		if components.size() <= 1:
			break
		var base: Array = components[0]
		for i in range(1, components.size()):
			var nearest := _rt_nearest_pair(base, components[i] as Array)
			_rt_carve_direct_connection(cells, cols, rows, nearest.get("a", Vector2i.ZERO), nearest.get("b", Vector2i.ZERO))
	for _pass in range(4):
		if _rt_road_dead_end_count(cells) <= 0:
			break
		_rt_repair_dead_ends(cells, cols, rows)
	_rt_prune_dead_end_spurs(cells, cols, rows)


static func _rt_repair_dead_ends(cells: Array, cols: int, rows: int) -> void:
	var roads := _rt_road_cells(cells)
	for item in roads:
		var p: Vector2i = item
		if _rt_road_degree(_cell(cells, cols, p.x, p.y)) > 1:
			continue
		var best := p
		var best_dist := 999999
		for other in roads:
			var q: Vector2i = other
			if q == p:
				continue
			var dist: int = abs(q.x - p.x) + abs(q.y - p.y)
			if dist > 2 and dist < best_dist:
				best = q
				best_dist = dist
		if best != p:
			_rt_carve_direct_connection(cells, cols, rows, p, best)


static func _rt_prune_dead_end_spurs(cells: Array, cols: int, rows: int) -> void:
	for _pass in range(cols * rows):
		var changed := false
		for item in _rt_road_cells(cells):
			var p: Vector2i = item
			var cell := _cell(cells, cols, p.x, p.y)
			if _rt_road_degree(cell) > 1:
				continue
			for dir in DIRS:
				if bool(cell.get("road_" + dir, false)):
					_mark_edge(cells, cols, p.x + _dir_dx(dir), p.y + _dir_dy(dir), "road", _opposite_dir(dir), false)
			cell["road_n"] = false
			cell["road_e"] = false
			cell["road_s"] = false
			cell["road_w"] = false
			cell["urban_road"] = false
			_set_cell(cells, cols, p.x, p.y, cell)
			changed = true
		if not changed:
			return


static func _rt_carve_direct_connection(cells: Array, cols: int, rows: int, start: Vector2i, finish: Vector2i) -> void:
	var current := _rt_clamp_point(start, cols, rows)
	var goal := _rt_clamp_point(finish, cols, rows)
	var guard := 0
	while current != goal and guard < cols * rows:
		var next := current
		if abs(goal.x - current.x) >= abs(goal.y - current.y) and goal.x != current.x:
			next.x += _sign_int(goal.x - current.x)
		elif goal.y != current.y:
			next.y += _sign_int(goal.y - current.y)
		elif goal.x != current.x:
			next.x += _sign_int(goal.x - current.x)
		next = _rt_clamp_point(next, cols, rows)
		if _rt_cell_blocks_road(cells, cols, rows, next):
			var alternate := _rt_find_unblocked_road_step(cells, cols, rows, current, goal, next)
			if alternate != current:
				next = alternate
		_rt_connect_road_cells(cells, cols, rows, current, next)
		current = next
		guard += 1


static func _rt_cell_blocks_road(cells: Array, cols: int, rows: int, point: Vector2i) -> bool:
	var cell := _cell(cells, cols, point.x, point.y)
	if cell.is_empty():
		return true
	var terrain := str(cell.get("terrain", "grassland"))
	return terrain == "lake" or terrain == "mountain"


static func _rt_find_unblocked_road_step(cells: Array, cols: int, rows: int, current: Vector2i, goal: Vector2i, blocked_step: Vector2i) -> Vector2i:
	var options := [
		Vector2i(current.x + _sign_int(goal.x - current.x), current.y),
		Vector2i(current.x, current.y + _sign_int(goal.y - current.y)),
		Vector2i(current.x + 1, current.y),
		Vector2i(current.x - 1, current.y),
		Vector2i(current.x, current.y + 1),
		Vector2i(current.x, current.y - 1),
	]
	var best := current
	var best_dist := 999999
	for option_value in options:
		var option: Vector2i = _rt_clamp_point(option_value, cols, rows)
		if option == current or option == blocked_step:
			continue
		if abs(option.x - current.x) + abs(option.y - current.y) != 1:
			continue
		if _rt_cell_blocks_road(cells, cols, rows, option):
			continue
		var dist: int = abs(goal.x - option.x) + abs(goal.y - option.y)
		if dist < best_dist:
			best_dist = dist
			best = option
	return best


static func _rt_reserve_road_buffers(cells: Array, cols: int, rows: int, buffer_cells: int) -> void:
	var roads := _rt_road_cells(cells)
	for item in roads:
		var p: Vector2i = item
		for dy in range(-buffer_cells, buffer_cells + 1):
			for dx in range(-buffer_cells, buffer_cells + 1):
				if abs(dx) + abs(dy) > buffer_cells:
					continue
				var x := p.x + dx
				var y := p.y + dy
				if x < 0 or x >= cols or y < 0 or y >= rows:
					continue
				var cell := _cell(cells, cols, x, y)
				var terrain := str(cell.get("terrain", ""))
				if _cell_has_road(cell) or _cell_has_river(cell) or terrain == "lake" or terrain == "hill" or terrain == "mountain":
					continue
				cell["terrain"] = "road_buffer"
				cell["road_buffer"] = true
				_set_cell(cells, cols, x, y, cell)


static func _rt_lay_water_features(cells: Array, cols: int, rows: int, rng: RandomNumberGenerator, difficulty: float) -> Dictionary:
	var river_cell_count := 0
	var lake_cell_count := 0
	var bridge_cell_count := 0
	var river_count := 1 + int(round(difficulty))
	for i in range(river_count):
		var x := clampi(int(round(float(cols) * (0.30 + 0.32 * rng.randf()))), 4, cols - 5)
		for y in range(1, rows - 1):
			if y % 6 == 0 and rng.randf() < 0.55:
				x = clampi(x + int(rng.randi_range(-2, 2)), 2, cols - 3)
			var p := Vector2i(x, y)
			var cell := _cell(cells, cols, p.x, p.y)
			if _cell_has_road(cell):
				cell["is_bridge"] = true
				cell["terrain"] = "bridge"
				bridge_cell_count += 1
			else:
				cell["terrain"] = "river"
				if bool(cell.get("road_buffer", false)):
					cell["bridge_approach"] = true
			cell["river_n"] = y > 1
			cell["river_s"] = y < rows - 2
			_set_cell(cells, cols, p.x, p.y, cell)
			if y > 1:
				_mark_edge(cells, cols, p.x, p.y - 1, "river", "s", true)
			if y < rows - 2:
				_mark_edge(cells, cols, p.x, p.y + 1, "river", "n", true)
			river_cell_count += 1
	var lake_target := 2 + int(round(difficulty * 2.0))
	var made := 0
	var attempts := 0
	while made < lake_target and attempts < 80:
		attempts += 1
		var center := Vector2i(int(rng.randi_range(8, cols - 9)), int(rng.randi_range(8, rows - 9)))
		if _rt_nearest_road_distance(cells, cols, rows, center.x, center.y, 5) <= 4:
			continue
		var radius := int(rng.randi_range(2, 4))
		for dy in range(-radius, radius + 1):
			for dx in range(-radius, radius + 1):
				if Vector2(float(dx), float(dy)).length() > float(radius) + 0.35:
					continue
				var x := center.x + dx
				var y := center.y + dy
				var cell := _cell(cells, cols, x, y)
				if cell.is_empty() or _cell_has_road(cell) or bool(cell.get("road_buffer", false)):
					continue
				cell["terrain"] = "lake"
				cell["is_lake"] = true
				_set_cell(cells, cols, x, y, cell)
				lake_cell_count += 1
		made += 1
	return {
		"water_feature_count": river_count + made,
		"lake_cell_count": lake_cell_count,
		"river_cell_count": river_cell_count,
		"bridge_cell_count": bridge_cell_count,
	}


static func _rt_lay_natural_city_blocks(cells: Array, cols: int, rows: int, rng: RandomNumberGenerator, road_anchors: Array, difficulty: float) -> Array:
	var candidates := _rt_city_anchor_candidates(cells, cols, rows)
	if candidates.is_empty():
		candidates = road_anchors.duplicate(true)
	var chosen: Array = []
	var target := clampi(5 + int(round(difficulty * 4.0)), 4, 10)
	while chosen.size() < target and not candidates.is_empty():
		var idx := int(rng.randi_range(0, candidates.size() - 1))
		var p: Vector2i = candidates[idx]
		candidates.remove_at(idx)
		var too_close := false
		for existing in chosen:
			var e: Vector2i = existing
			if abs(e.x - p.x) + abs(e.y - p.y) < 12:
				too_close = true
				break
		if too_close:
			continue
		chosen.append(p)
		_rt_mark_city_district(cells, cols, rows, p, rng)
	var out: Array = []
	for item in chosen:
		var p: Vector2i = item
		var cell := _cell(cells, cols, p.x, p.y)
		cell["city_anchor"] = true
		cell["urban_road"] = true
		_set_cell(cells, cols, p.x, p.y, cell)
		out.append({"x": p.x, "y": p.y, "tile_id": str(cell.get("tile_id", "")), "variant": int(cell.get("variant", 0))})
	_rt_promote_city_adjacent_roads(cells, cols, rows)
	return out


static func _rt_promote_city_adjacent_roads(cells: Array, cols: int, rows: int) -> void:
	for item in _rt_road_cells(cells):
		var p: Vector2i = item
		for dir in DIRS:
			var neighbor := _cell(cells, cols, p.x + _dir_dx(dir), p.y + _dir_dy(dir))
			var terrain := str(neighbor.get("terrain", ""))
			if terrain == "city" or terrain == "city_edge":
				var cell := _cell(cells, cols, p.x, p.y)
				cell["urban_road"] = true
				_set_cell(cells, cols, p.x, p.y, cell)
				break


static func _rt_mark_city_district(cells: Array, cols: int, rows: int, anchor: Vector2i, rng: RandomNumberGenerator) -> void:
	for y in range(anchor.y - 4, anchor.y + 5):
		for x in range(anchor.x - 4, anchor.x + 5):
			if x < 1 or x >= cols - 1 or y < 1 or y >= rows - 1:
				continue
			var dist: int = abs(x - anchor.x) + abs(y - anchor.y)
			if dist > 5 or (dist > 3 and rng.randf() < 0.40):
				continue
			var cell := _cell(cells, cols, x, y)
			var terrain := str(cell.get("terrain", ""))
			if bool(cell.get("is_lake", false)) or _cell_has_river(cell) or terrain == "hill" or terrain == "mountain":
				continue
			if _cell_has_road(cell):
				cell["urban_road"] = true
			elif _rt_nearest_road_distance(cells, cols, rows, x, y, 1) <= 1:
				cell["terrain"] = "city"
				cell["road_buffer"] = false
			elif _rt_nearest_urban_road_distance(cells, cols, rows, x, y, 2) <= 2 and rng.randf() < 0.38:
				cell["terrain"] = "city_edge"
				cell["road_buffer"] = false
			_set_cell(cells, cols, x, y, cell)


static func _rt_lay_secondary_terrain(cells: Array, cols: int, rows: int, rng: RandomNumberGenerator, difficulty: float) -> void:
	var forest_target := 12 + int(round(difficulty * 12.0))
	_rt_paint_secondary_patches(cells, cols, rows, rng, "forest", forest_target, 2, 5)
	var field_target := 10 + int(round(difficulty * 8.0))
	_rt_paint_secondary_patches(cells, cols, rows, rng, "field", field_target, 2, 5)


static func _rt_paint_secondary_patches(cells: Array, cols: int, rows: int, rng: RandomNumberGenerator, terrain_name: String, patch_count: int, min_radius: int, max_radius: int) -> void:
	var made := 0
	var attempts := 0
	while made < patch_count and attempts < patch_count * 12:
		attempts += 1
		var center := Vector2i(int(rng.randi_range(4, cols - 5)), int(rng.randi_range(4, rows - 5)))
		if not _rt_cell_available_for_secondary(cells, cols, rows, center.x, center.y):
			continue
		var radius := int(rng.randi_range(min_radius, max_radius))
		for dy in range(-radius, radius + 1):
			for dx in range(-radius, radius + 1):
				if Vector2(float(dx), float(dy)).length() > float(radius) + 0.25:
					continue
				var x := center.x + dx
				var y := center.y + dy
				if _rt_cell_available_for_secondary(cells, cols, rows, x, y):
					var cell := _cell(cells, cols, x, y)
					cell["terrain"] = terrain_name
					_set_cell(cells, cols, x, y, cell)
		made += 1


static func _rt_cell_available_for_secondary(cells: Array, cols: int, rows: int, x: int, y: int) -> bool:
	var cell := _cell(cells, cols, x, y)
	if cell.is_empty():
		return false
	if _cell_has_road(cell) or _cell_has_river(cell):
		return false
	if bool(cell.get("road_buffer", false)) or bool(cell.get("is_lake", false)):
		return false
	var terrain := str(cell.get("terrain", "grassland"))
	return terrain == "grassland"


static func _rt_city_anchor_candidates(cells: Array, cols: int, rows: int) -> Array:
	var out: Array = []
	for item in _rt_road_cells(cells):
		var p: Vector2i = item
		var cell := _cell(cells, cols, p.x, p.y)
		var degree := _rt_road_degree(cell)
		var is_turn := degree == 2 and not ((bool(cell.get("road_n", false)) and bool(cell.get("road_s", false))) or (bool(cell.get("road_e", false)) and bool(cell.get("road_w", false))))
		if degree >= 3 or is_turn:
			out.append(p)
	return out


static func _rt_route_cells_from_road_graph(nodes: Array) -> Array:
	var out: Array = []
	var step: int = max(1, int(nodes.size() / 18))
	for i in range(0, nodes.size(), step):
		var node := nodes[i] as Dictionary
		out.append({"x": int(node.get("x", 0)), "y": int(node.get("y", 0)), "alt": 1 + (i % 4)})
	return out


static func _rt_road_cells(cells: Array) -> Array:
	var out: Array = []
	for cell_value in cells:
		var cell := cell_value as Dictionary
		if _cell_has_road(cell):
			out.append(Vector2i(int(cell.get("x", 0)), int(cell.get("y", 0))))
	return out


static func _rt_road_components(cells: Array, cols: int, rows: int) -> Array:
	var roads := _rt_road_cells(cells)
	var road_keys := {}
	for item in roads:
		var p: Vector2i = item
		road_keys[_cell_key(p.x, p.y)] = true
	var seen := {}
	var components: Array = []
	for item in roads:
		var start: Vector2i = item
		var start_key := _cell_key(start.x, start.y)
		if bool(seen.get(start_key, false)):
			continue
		var component: Array = []
		var queue := [start]
		seen[start_key] = true
		while not queue.is_empty():
			var p: Vector2i = queue.pop_front()
			component.append(p)
			var cell := _cell(cells, cols, p.x, p.y)
			for dir in DIRS:
				if not bool(cell.get("road_" + dir, false)):
					continue
				var next := Vector2i(p.x + _dir_dx(dir), p.y + _dir_dy(dir))
				var key := _cell_key(next.x, next.y)
				if bool(road_keys.get(key, false)) and not bool(seen.get(key, false)):
					seen[key] = true
					queue.append(next)
		components.append(component)
	return components


static func _rt_road_component_count(cells: Array, cols: int, rows: int) -> int:
	return _rt_road_components(cells, cols, rows).size()


static func _rt_road_dead_end_count(cells: Array) -> int:
	var count := 0
	for cell_value in cells:
		var cell := cell_value as Dictionary
		if _cell_has_road(cell) and _rt_road_degree(cell) <= 1:
			count += 1
	return count


static func _rt_road_degree(cell: Dictionary) -> int:
	return _edge_count(cell, "road")


static func _rt_nearest_pair(a_values: Array, b_values: Array) -> Dictionary:
	var best_a := Vector2i.ZERO
	var best_b := Vector2i.ZERO
	var best_dist := 999999
	for a_value in a_values:
		var a: Vector2i = a_value
		for b_value in b_values:
			var b: Vector2i = b_value
			var dist: int = abs(a.x - b.x) + abs(a.y - b.y)
			if dist < best_dist:
				best_dist = dist
				best_a = a
				best_b = b
	return {"a": best_a, "b": best_b}


static func _rt_nearest_road_distance(cells: Array, cols: int, rows: int, x: int, y: int, max_distance: int) -> int:
	var best := max_distance + 1
	for dy in range(-max_distance, max_distance + 1):
		for dx in range(-max_distance, max_distance + 1):
			var dist: int = abs(dx) + abs(dy)
			if dist >= best:
				continue
			var cell := _cell(cells, cols, x + dx, y + dy)
			if not cell.is_empty() and _cell_has_road(cell):
				best = dist
	return best


static func _rt_nearest_urban_road_distance(cells: Array, cols: int, rows: int, x: int, y: int, max_distance: int) -> int:
	var best := max_distance + 1
	for dy in range(-max_distance, max_distance + 1):
		for dx in range(-max_distance, max_distance + 1):
			var dist: int = abs(dx) + abs(dy)
			if dist >= best:
				continue
			var cell := _cell(cells, cols, x + dx, y + dy)
			if not cell.is_empty() and _cell_has_road(cell) and bool(cell.get("urban_road", false)):
				best = dist
	return best


static func _rt_nearest_water_cell(cells: Array, cols: int, rows: int, point: Vector2i, margin: int) -> Vector2i:
	if _rt_water_cell_available(cells, cols, rows, point.x, point.y):
		return point
	var best := Vector2i(-1, -1)
	var best_dist := 999999
	for dy in range(-2, 3):
		for dx in range(-2, 3):
			var x := clampi(point.x + dx, margin, cols - margin - 1)
			var y := clampi(point.y + dy, margin, rows - margin - 1)
			if not _rt_water_cell_available(cells, cols, rows, x, y):
				continue
			var dist: int = abs(dx) + abs(dy)
			if dist < best_dist:
				best_dist = dist
				best = Vector2i(x, y)
	return best


static func _rt_water_cell_available(cells: Array, cols: int, rows: int, x: int, y: int) -> bool:
	var cell := _cell(cells, cols, x, y)
	if cell.is_empty():
		return false
	var terrain := str(cell.get("terrain", "grassland"))
	if terrain == "mountain" or terrain == "hill" or terrain == "lake" or terrain == "river" or terrain == "bridge":
		return false
	return not bool(cell.get("edge_mountain", false))


static func _rt_mark_river_connection(cells: Array, cols: int, a: Vector2i, b: Vector2i) -> void:
	if abs(a.x - b.x) + abs(a.y - b.y) != 1:
		return
	var dir := "e"
	if b.x < a.x:
		dir = "w"
	elif b.y > a.y:
		dir = "s"
	elif b.y < a.y:
		dir = "n"
	_mark_edge(cells, cols, a.x, a.y, "river", dir, true)
	_mark_edge(cells, cols, b.x, b.y, "river", _opposite_dir(dir), true)


static func _rt_mark_river_cell(cells: Array, cols: int, point: Vector2i) -> void:
	var cell := _cell(cells, cols, point.x, point.y)
	if cell.is_empty():
		return
	cell["terrain"] = "river"
	cell["is_lake"] = false
	_set_cell(cells, cols, point.x, point.y, cell)


static func _rt_edge_mountain_ring_width(cells: Array, cols: int, rows: int) -> int:
	var width := 0
	for cell_value in cells:
		var cell := cell_value as Dictionary
		if not bool(cell.get("edge_mountain", false)):
			continue
		var x := int(cell.get("x", 0))
		var y := int(cell.get("y", 0))
		var depth: int = min(min(x, y), min(cols - 1 - x, rows - 1 - y))
		width = max(width, depth + 1)
	return width


static func _rt_count_terrain(cells: Array, terrain_name: String) -> int:
	var count := 0
	for cell_value in cells:
		var cell := cell_value as Dictionary
		if str(cell.get("terrain", "")) == terrain_name:
			count += 1
	return count


static func _rt_river_cell_count(cells: Array) -> int:
	var count := 0
	for cell_value in cells:
		var cell := cell_value as Dictionary
		if _cell_has_river(cell) and not bool(cell.get("is_bridge", false)):
			count += 1
	return count


static func _rt_bridge_cell_count(cells: Array) -> int:
	var count := 0
	for cell_value in cells:
		var cell := cell_value as Dictionary
		if bool(cell.get("is_bridge", false)):
			count += 1
	return count


static func _rt_terrain_blocked_road_crossing_count(cells: Array) -> int:
	var count := 0
	for cell_value in cells:
		var cell := cell_value as Dictionary
		if _cell_has_road(cell) and bool(cell.get("terrain_blocked_road_crossing", false)):
			count += 1
	return count


static func _rt_city_building_cell_count(cells: Array) -> int:
	var count := 0
	for cell_value in cells:
		var cell := cell_value as Dictionary
		var terrain := str(cell.get("terrain", ""))
		if terrain == "city" or terrain == "city_edge":
			count += 1
	return count


static func _rt_road_buffer_violation_count(cells: Array, cols: int, rows: int, buffer_cells: int) -> int:
	var violations := 0
	for item in _rt_road_cells(cells):
		var p: Vector2i = item
		var road_cell := _cell(cells, cols, p.x, p.y)
		if bool(road_cell.get("urban_road", false)):
			continue
		for dy in range(-buffer_cells, buffer_cells + 1):
			for dx in range(-buffer_cells, buffer_cells + 1):
				if abs(dx) + abs(dy) > buffer_cells:
					continue
				var cell := _cell(cells, cols, p.x + dx, p.y + dy)
				if cell.is_empty() or _cell_has_road(cell):
					continue
				var terrain := str(cell.get("terrain", "grassland"))
				if _cell_has_river(cell) or terrain == "lake" or terrain == "bridge":
					continue
				if terrain != "road_buffer" and terrain != "grassland" and terrain != "hill" and terrain != "mountain" and not bool(cell.get("bridge_approach", false)):
					violations += 1
	return violations


static func _rt_city_road_adjacency_violations(cells: Array, cols: int, rows: int) -> int:
	var violations := 0
	for cell_value in cells:
		var cell := cell_value as Dictionary
		var terrain := str(cell.get("terrain", ""))
		if terrain != "city" and terrain != "city_edge":
			continue
		if _cell_has_road(cell):
			violations += 1
			continue
		if _rt_nearest_road_distance(cells, cols, rows, int(cell.get("x", 0)), int(cell.get("y", 0)), 2) > 2:
			violations += 1
	return violations


static func _rt_clamp_point(point: Vector2i, cols: int, rows: int) -> Vector2i:
	return Vector2i(clampi(point.x, 1, cols - 2), clampi(point.y, 1, rows - 2))


static func _sign_int(value: int) -> int:
	return 1 if value > 0 else (-1 if value < 0 else 0)


static func _make_base_cells(cols: int, rows: int, rng: RandomNumberGenerator) -> Array:
	var cells := []
	for y in range(rows):
		for x in range(cols):
			var terrain := "grassland"
			if rng.randf() < 0.28:
				terrain = "field"
			cells.append({
				"x": x,
				"y": y,
				"terrain": terrain,
				"tile_id": terrain,
				"variant": int(rng.randi_range(0, 4)),
				"road_n": false,
				"road_e": false,
				"road_s": false,
				"road_w": false,
				"river_n": false,
				"river_e": false,
				"river_s": false,
				"river_w": false,
				"is_bridge": false,
			})
	return cells


static func _lay_river(cells: Array, cols: int, rows: int, rng: RandomNumberGenerator) -> void:
	var river_col := clampi(int(round(float(cols - 1) * rng.randf_range(0.32, 0.68))), 2, max(2, cols - 3))
	for y in range(rows):
		if y > 1 and y < rows - 2 and rng.randf() < 0.32:
			river_col = clampi(river_col + int(rng.randi_range(-1, 1)), 1, cols - 2)
		var cell := _cell(cells, cols, river_col, y)
		cell["terrain"] = "river"
		cell["river_n"] = y > 0
		cell["river_s"] = y < rows - 1
		_set_cell(cells, cols, river_col, y, cell)
		if y > 0:
			_mark_edge(cells, cols, river_col, y - 1, "river", "s", true)
		if y < rows - 1:
			_mark_edge(cells, cols, river_col, y + 1, "river", "n", true)
	if rng.randf() < 0.58:
		var lake_y := int(rng.randi_range(1, rows - 2))
		var lake_x := clampi(river_col + (-1 if rng.randf() < 0.5 else 1), 1, cols - 2)
		var lake := _cell(cells, cols, lake_x, lake_y)
		lake["terrain"] = "river"
		lake["variant"] = int(lake.get("variant", 0)) + 5
		_set_cell(cells, cols, lake_x, lake_y, lake)


static func _lay_roads(cells: Array, cols: int, rows: int, rng: RandomNumberGenerator, difficulty: float) -> void:
	var main_row := clampi(int(rows / 2), 1, rows - 2)
	for x in range(cols):
		_mark_edge(cells, cols, x, main_row, "road", "w", x > 0)
		_mark_edge(cells, cols, x, main_row, "road", "e", x < cols - 1)
		var cell := _cell(cells, cols, x, main_row)
		if str(cell.get("terrain", "")) != "river":
			cell["terrain"] = "field"
		_set_cell(cells, cols, x, main_row, cell)
	var branch_columns := [1, cols - 2]
	if difficulty > 0.35:
		branch_columns.append(clampi(int(round(float(cols) * 0.52)), 2, cols - 3))
	for branch_x in branch_columns:
		var end_y := 1 if branch_x < int(cols / 2) else rows - 2
		var step := -1 if end_y < main_row else 1
		var y := main_row
		while y != end_y:
			_mark_edge(cells, cols, branch_x, y, "road", "n" if step < 0 else "s", true)
			_mark_edge(cells, cols, branch_x, y + step, "road", "s" if step < 0 else "n", true)
			y += step


static func _lay_city_patches(cells: Array, cols: int, rows: int, rng: RandomNumberGenerator) -> void:
	var main_row := clampi(int(rows / 2), 1, rows - 2)
	var anchors := [Vector2i(1, main_row - 1), Vector2i(cols - 2, main_row + 1)]
	if rng.randf() < 0.72:
		anchors.append(Vector2i(clampi(int(cols / 2) + 1, 1, cols - 2), main_row - 1))
	for anchor in anchors:
		for dy in range(2):
			for dx in range(2):
				var x := clampi(anchor.x + dx - 1, 0, cols - 1)
				var y := clampi(anchor.y + dy, 0, rows - 1)
				var cell := _cell(cells, cols, x, y)
				if str(cell.get("terrain", "")) != "river":
					cell["terrain"] = "city"
				_set_cell(cells, cols, x, y, cell)


static func _lay_forest_patches(cells: Array, cols: int, rows: int, rng: RandomNumberGenerator) -> void:
	var anchors := [Vector2i(1, 1), Vector2i(cols - 2, 1), Vector2i(1, rows - 2), Vector2i(cols - 2, rows - 2)]
	for anchor in anchors:
		if rng.randf() < 0.86:
			for dy in range(-1, 1):
				for dx in range(-1, 1):
					var x := clampi(anchor.x + dx, 0, cols - 1)
					var y := clampi(anchor.y + dy, 0, rows - 1)
					var cell := _cell(cells, cols, x, y)
					if not _cell_has_road(cell) and not _cell_has_river(cell):
						cell["terrain"] = "forest"
					_set_cell(cells, cols, x, y, cell)


static func _lay_transition_cells(cells: Array, cols: int, rows: int) -> void:
	for y in range(rows):
		for x in range(cols):
			var cell := _cell(cells, cols, x, y)
			if _cell_has_road(cell) or _cell_has_river(cell):
				continue
			var terrain := str(cell.get("terrain", "grassland"))
			if terrain == "grassland" or terrain == "field":
				if _neighbor_has_terrain(cells, cols, rows, x, y, "city"):
					cell["terrain"] = "city_edge"
				elif _neighbor_has_terrain(cells, cols, rows, x, y, "forest"):
					cell["terrain"] = "forest_edge"
			_set_cell(cells, cols, x, y, cell)


static func _repair_invalid_cells(cells: Array, cols: int, rows: int) -> void:
	# Bridge cells require road east/west and river north/south, so roads remain centered puzzle-piece connectors.
	for y in range(rows):
		for x in range(cols):
			var cell := _cell(cells, cols, x, y)
			if _cell_has_road(cell) and _cell_has_river(cell):
				cell["is_bridge"] = true
				cell["terrain"] = "bridge"
				cell["road_e"] = true
				cell["road_w"] = true
				cell["road_n"] = false
				cell["road_s"] = false
				cell["river_n"] = y > 0
				cell["river_s"] = y < rows - 1
				cell["river_e"] = false
				cell["river_w"] = false
				_set_cell(cells, cols, x, y, cell)
				if x > 0:
					_mark_edge(cells, cols, x - 1, y, "road", "e", true)
				if x < cols - 1:
					_mark_edge(cells, cols, x + 1, y, "road", "w", true)
				if y > 0:
					_mark_edge(cells, cols, x, y - 1, "river", "s", true)
				if y < rows - 1:
					_mark_edge(cells, cols, x, y + 1, "river", "n", true)
	_sync_edges(cells, cols, rows, "road")
	_sync_edges(cells, cols, rows, "river")


static func _validate_edges(cells: Array, cols: int, rows: int) -> int:
	var violations := 0
	for y in range(rows):
		for x in range(cols):
			var cell := _cell(cells, cols, x, y)
			for dir in DIRS:
				var nx := x + _dir_dx(dir)
				var ny := y + _dir_dy(dir)
				var opposite := _opposite_dir(dir)
				if bool(cell.get("road_" + dir, false)):
					if nx < 0 or nx >= cols or ny < 0 or ny >= rows or not bool(_cell(cells, cols, nx, ny).get("road_" + opposite, false)):
						violations += 1
				if bool(cell.get("river_" + dir, false)):
					if nx < 0 or nx >= cols or ny < 0 or ny >= rows or not bool(_cell(cells, cols, nx, ny).get("river_" + opposite, false)):
						violations += 1
			if bool(cell.get("is_bridge", false)):
				if not _cell_has_road(cell) or not _cell_has_river(cell):
					violations += 1
	return violations


static func _finalize_tiles(cells: Array) -> void:
	for i in range(cells.size()):
		var cell := cells[i] as Dictionary
		if bool(cell.get("is_bridge", false)):
			cell["tile_id"] = "bridge_ew_over_ns"
		elif bool(cell.get("is_lake", false)) or str(cell.get("terrain", "")) == "lake":
			cell["tile_id"] = "lake"
		elif _cell_has_river(cell):
			cell["tile_id"] = "river_ew" if bool(cell.get("river_e", false)) or bool(cell.get("river_w", false)) else "river_ns"
		elif _cell_has_road(cell):
			var road_count := _edge_count(cell, "road")
			if road_count >= 4:
				cell["tile_id"] = "road_cross"
			elif road_count == 3:
				cell["tile_id"] = "road_t"
			elif bool(cell.get("road_n", false)) or bool(cell.get("road_s", false)):
				cell["tile_id"] = "road_ns" if bool(cell.get("road_n", false)) and bool(cell.get("road_s", false)) else "road_corner"
			else:
				cell["tile_id"] = "road_ew" if bool(cell.get("road_e", false)) and bool(cell.get("road_w", false)) else "road_corner"
		elif bool(cell.get("road_buffer", false)):
			cell["tile_id"] = "road_buffer"
		else:
			cell["tile_id"] = str(cell.get("terrain", "grassland"))
		cell["catalog"] = TILE_CATALOG.get(str(cell.get("tile_id", "grassland")), TILE_CATALOG["grassland"])
		cells[i] = cell


static func _build_road_graph(cells: Array, cols: int, rows: int) -> Dictionary:
	var nodes := []
	var node_by_key := {}
	for y in range(rows):
		for x in range(cols):
			var cell := _cell(cells, cols, x, y)
			if _cell_has_road(cell):
				var node_id := nodes.size()
				node_by_key[_cell_key(x, y)] = node_id
				nodes.append({"id": node_id, "x": x, "y": y, "cell_tile": str(cell.get("tile_id", "")), "bridge": bool(cell.get("is_bridge", false))})
	var edges := []
	for y in range(rows):
		for x in range(cols):
			var cell := _cell(cells, cols, x, y)
			var a := int(node_by_key.get(_cell_key(x, y), -1))
			if a < 0:
				continue
			for dir in ["e", "s"]:
				if not bool(cell.get("road_" + dir, false)):
					continue
				var nx := x + _dir_dx(dir)
				var ny := y + _dir_dy(dir)
				var b := int(node_by_key.get(_cell_key(nx, ny), -1))
				if b >= 0:
					var next_cell := _cell(cells, cols, nx, ny)
					edges.append({"a": a, "b": b, "type": "bridge" if bool(cell.get("is_bridge", false)) or bool(next_cell.get("is_bridge", false)) else "road"})
	return {"nodes": nodes, "edges": edges}


static func _build_asset_sockets(cells: Array, cols: int, rows: int, rng: RandomNumberGenerator) -> Dictionary:
	var sockets := {
		"vehicle": [],
		"truck": [],
		"tank": [],
		"pedestrian": [],
		"people": [],
		"parked_vehicle": [],
		"building": [],
		"forest": [],
		"sheep": [],
		"static": [],
		"air": [],
		"plane": [],
		"helicopter": [],
	}
	for cell_value in cells:
		var cell := cell_value as Dictionary
		var x := int(cell.get("x", 0))
		var y := int(cell.get("y", 0))
		var terrain := str(cell.get("terrain", "grassland"))
		if _cell_has_road(cell):
			_append_socket(sockets, "vehicle", cell, 0.0, 0.0, "car")
			_append_socket(sockets, "truck", cell, 0.18, -0.18, "truck")
			if (x + y + int(cell.get("variant", 0))) % 5 == 0:
				_append_socket(sockets, "tank", cell, -0.18, 0.18, "tank")
			if bool(cell.get("urban_road", false)):
				_append_socket(sockets, "parked_vehicle", cell, -0.28, 0.24, "car")
			continue
		if terrain == "city" or terrain == "city_edge":
			_append_socket(sockets, "building", cell, -0.18, -0.12, "building")
			_append_socket(sockets, "pedestrian", cell, 0.24, 0.18, "person")
			_append_socket(sockets, "people", cell, 0.24, 0.18, "people")
			_append_socket(sockets, "static", cell, -0.26, 0.26, "water_tower")
		elif terrain == "forest" or terrain == "forest_edge":
			_append_socket(sockets, "forest", cell, 0.0, 0.0, "tree_cluster")
			_append_socket(sockets, "pedestrian", cell, -0.20, 0.18, "person")
			_append_socket(sockets, "people", cell, -0.20, 0.18, "people")
		elif terrain == "field" or terrain == "grassland":
			_append_socket(sockets, "sheep", cell, -0.20, 0.18, "sheep")
			if rng.randf() < 0.42:
				_append_socket(sockets, "static", cell, 0.18, -0.22, "field_marker")
	for x in range(1, cols - 1):
		var sky_cell := _cell(cells, cols, x, clampi(int(rows / 2) + ((x % 3) - 1), 1, rows - 2))
		_append_socket(sockets, "air", sky_cell, 0.0, 0.0, "plane")
		if x % 2 == 0:
			_append_socket(sockets, "plane", sky_cell, 0.0, 0.0, "plane")
		else:
			_append_socket(sockets, "helicopter", sky_cell, 0.0, 0.0, "helicopter")
	return sockets


static func _append_socket(sockets: Dictionary, key: String, cell: Dictionary, offset_x: float, offset_y: float, asset: String) -> void:
	var list: Array = sockets.get(key, [])
	list.append({
		"kind": key,
		"asset": asset,
		"x": int(cell.get("x", 0)),
		"y": int(cell.get("y", 0)),
		"offset_x": offset_x,
		"offset_y": offset_y,
		"tile_id": str(cell.get("tile_id", "")),
	})
	sockets[key] = list


static func _main_route_cells(cells: Array, cols: int, rows: int) -> Array:
	var out := []
	var main_row := clampi(int(rows / 2), 1, rows - 2)
	for x in range(cols):
		var cell := _cell(cells, cols, x, main_row)
		if _cell_has_road(cell):
			out.append({"x": x, "y": main_row, "alt": 1 + (x % 4)})
	return out


static func _terrain_cells(cells: Array, terrain_name: String) -> Array:
	var out := []
	for cell_value in cells:
		var cell := cell_value as Dictionary
		if str(cell.get("terrain", "")) == terrain_name:
			out.append({"x": int(cell.get("x", 0)), "y": int(cell.get("y", 0)), "tile_id": str(cell.get("tile_id", "")), "variant": int(cell.get("variant", 0))})
	return out


static func _cell(cells: Array, cols: int, x: int, y: int) -> Dictionary:
	if x < 0 or y < 0 or x >= cols or y * cols + x >= cells.size():
		return {}
	return cells[y * cols + x] as Dictionary


static func _set_cell(cells: Array, cols: int, x: int, y: int, cell: Dictionary) -> void:
	if x < 0 or y < 0 or x >= cols or y * cols + x >= cells.size():
		return
	cells[y * cols + x] = cell


static func _mark_edge(cells: Array, cols: int, x: int, y: int, kind: String, dir: String, enabled: bool) -> void:
	var cell := _cell(cells, cols, x, y)
	if cell.is_empty():
		return
	cell[kind + "_" + dir] = enabled
	_set_cell(cells, cols, x, y, cell)


static func _sync_edges(cells: Array, cols: int, rows: int, kind: String) -> void:
	for y in range(rows):
		for x in range(cols):
			var cell := _cell(cells, cols, x, y)
			for dir in DIRS:
				if not bool(cell.get(kind + "_" + dir, false)):
					continue
				var nx := x + _dir_dx(dir)
				var ny := y + _dir_dy(dir)
				if nx < 0 or nx >= cols or ny < 0 or ny >= rows:
					cell[kind + "_" + dir] = false
					_set_cell(cells, cols, x, y, cell)
					continue
				_mark_edge(cells, cols, nx, ny, kind, _opposite_dir(dir), true)


static func _cell_has_road(cell: Dictionary) -> bool:
	return bool(cell.get("road_n", false)) or bool(cell.get("road_e", false)) or bool(cell.get("road_s", false)) or bool(cell.get("road_w", false))


static func _cell_has_river(cell: Dictionary) -> bool:
	return bool(cell.get("river_n", false)) or bool(cell.get("river_e", false)) or bool(cell.get("river_s", false)) or bool(cell.get("river_w", false))


static func _edge_count(cell: Dictionary, prefix: String) -> int:
	var count := 0
	for dir in DIRS:
		if bool(cell.get(prefix + "_" + dir, false)):
			count += 1
	return count


static func _neighbor_has_terrain(cells: Array, cols: int, rows: int, x: int, y: int, terrain_name: String) -> bool:
	for dir in DIRS:
		var nx := x + _dir_dx(dir)
		var ny := y + _dir_dy(dir)
		if nx >= 0 and nx < cols and ny >= 0 and ny < rows and str(_cell(cells, cols, nx, ny).get("terrain", "")) == terrain_name:
			return true
	return false


static func _dir_dx(dir: String) -> int:
	return 1 if dir == "e" else (-1 if dir == "w" else 0)


static func _dir_dy(dir: String) -> int:
	return 1 if dir == "s" else (-1 if dir == "n" else 0)


static func _opposite_dir(dir: String) -> String:
	if dir == "n":
		return "s"
	if dir == "s":
		return "n"
	if dir == "e":
		return "w"
	return "e"


static func _cell_key(x: int, y: int) -> String:
	return str(x) + ":" + str(y)


static func _hash_chunk_map(cells: Array, cols: int, rows: int, seed_value: int, pack: String, route_cells: Array) -> int:
	var hash := _hash_mix(seed_value, cols * 131 + rows * 197 + _string_salt(pack))
	for cell_value in cells:
		var cell := cell_value as Dictionary
		hash = _hash_mix(hash, int(cell.get("x", 0)) * 17 + int(cell.get("y", 0)) * 29 + _string_salt(str(cell.get("tile_id", ""))))
		hash = _hash_mix(hash, _edge_count(cell, "road") * 7 + _edge_count(cell, "river") * 11 + (97 if bool(cell.get("is_bridge", false)) else 0))
	for route_cell in route_cells:
		var rc := route_cell as Dictionary
		hash = _hash_mix(hash, int(rc.get("x", 0)) * 41 + int(rc.get("y", 0)) * 43 + int(rc.get("alt", 0)) * 47)
	return abs(hash)


static func _hash_mix(current: int, value: int) -> int:
	return int(fposmod(float(current * 1103515245 + value * 12345 + 0x45d9f3b), 2147483647.0))


static func _string_salt(value: String) -> int:
	var total := 0
	for i in range(value.length()):
		total = int(fposmod(float(total * 131 + value.unicode_at(i)), 1000003.0))
	return total

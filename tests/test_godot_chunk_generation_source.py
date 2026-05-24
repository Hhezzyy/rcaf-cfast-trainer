from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from cfast_trainer.godot_bridge import GODOT_DEFAULT_BIN, GODOT_PROJECT_PATH


GODOT_SCRIPTS = Path(__file__).resolve().parents[1] / "godot" / "cfast_3d" / "scripts"
GODOT_RT_INPUT_PROBE = Path(__file__).resolve().parent / "godot_rapid_tracking_input_probe.gd"
GODOT_RT_CHUNK_PROBE = Path(__file__).resolve().parent / "godot_rapid_tracking_chunk_probe.gd"
GODOT_SI_CHUNK_PROBE = Path(__file__).resolve().parent / "godot_spatial_integration_chunk_probe.gd"
GODOT_SI_QUESTION_PROBE = Path(__file__).resolve().parent / "godot_spatial_integration_question_probe.gd"
GODOT_SI_ASSET_PROBE = Path(__file__).resolve().parent / "godot_spatial_integration_asset_probe.gd"


def test_shared_chunk_generator_defines_tile_catalog_rules_and_sockets() -> None:
    source = (GODOT_SCRIPTS / "chunk_map_generator.gd").read_text(encoding="utf-8")

    assert "func generate" in source
    assert "TILE_CATALOG" in source
    for tile in (
        "grassland",
        "field",
        "forest",
        "forest_edge",
        "city",
        "city_edge",
        "river",
        "road_ew",
        "road_ns",
        "road_corner",
        "road_t",
        "bridge_ew_over_ns",
        "road_buffer",
        "lake",
        "hill",
        "mountain",
    ):
        assert tile in source
    assert "Bridge cells require road east/west and river north/south" in source
    assert "_generate_rapid_tracking_v3" in source
    assert "_rt_lay_priority_terrain" in source
    assert "_rt_lay_edge_mountain_ring" in source
    assert "_rt_lay_interior_hill_clusters" in source
    assert 'cell["terrain"] = "mountain"' in source
    assert 'cell["cluster_peak_tier"] = peak_tier' in source
    assert '"footprint_cells":' in source
    assert '"target_footprint": target_footprint' in source
    assert '"peak_tier": peak_tier' in source
    assert "_generate_rapid_tracking_v2" in source
    assert "_rt_lay_organic_looped_roads" in source
    assert "_rt_reserve_road_buffers" in source
    assert "_rt_lay_water_features" in source
    assert "_rt_lay_natural_city_blocks" in source
    assert "_rt_lay_secondary_terrain" in source
    assert "_repair_invalid_cells" in source
    assert "_validate_edges" in source
    assert "_sync_edges" in source
    for asset in (
        '"vehicle"',
        '"truck"',
        '"tank"',
        '"pedestrian"',
        '"people"',
        '"sheep"',
        '"plane"',
        '"helicopter"',
        '"static"',
    ):
        assert asset in source
    assert "chunk_hash" in source
    assert "rule_violations" in source


def test_rapid_tracking_uses_shared_chunk_map_for_routes_and_socketed_spawns() -> None:
    source = (GODOT_SCRIPTS / "rapid_tracking_runtime.gd").read_text(encoding="utf-8")

    assert 'preload("res://scripts/chunk_map_generator.gd")' in source
    assert 'purpose": "rapid_tracking"' in source
    assert "ChunkMapGenerator.generate" in source
    assert "_generate_chunk_map" in source
    assert "_chunk_cell_center_world" in source
    assert "_chunk_asset_sockets" in source
    assert "_socketed_spawn_position" in source
    assert "asset_spawn_policy == \"socketed\"" in source
    assert '"pathfinding": "road_graph"' in source
    assert '"spawn_socket"' in source
    assert '"tank"' in source
    assert "chunk_map_hash" in source
    assert "chunk_rule_violations" in source
    assert "road_component_count" in source
    assert "road_dead_end_count" in source
    assert "road_buffer_violation_count" in source
    assert "city_road_adjacency_violations" in source
    assert "water_feature_count" in source
    assert "mountain_cell_count" in source
    assert "hill_cell_count" in source
    assert "edge_mountain_ring_width" in source
    assert "building_cell_count" in source
    assert "_draw_chunk_ground_tiles" in source
    assert "_draw_chunk_road_tiles" not in source
    assert "CHUNK_TILE_SCALE := 1.03" in source
    assert "chunk_size_m * CHUNK_TILE_SCALE" in source
    assert "_make_chunk_elevation_block" in source
    assert "_draw_merged_elevation_clusters" in source
    assert "MergedHillClusterDome" in source
    assert "MergedMountainClusterDome" in source
    assert "_make_rounded_terrain_dome" in source
    assert "MountainRoundedDome" in source
    assert "HillRoundedDome" in source
    assert "ArrayMesh.new()" in source
    assert "MountainChunkBlock" not in source
    assert "HillChunkBlock" not in source
    assert '"type": "terrain"' in source
    assert "edge_mountain" in source
    assert "altitude_bob" in source
    assert "index % 4 == 3" in source
    assert "handoff_interval_s = maxf(7.5" in source
    assert "8.0 + (1.0 - difficulty) * 3.0" in source
    assert "_align_target_item_to_handoff_anchor" in source
    assert "_route_nodes_near_position" in source
    assert "_route_distance_near_position" in source
    assert "_build_target_instances" in source
    assert 'target_root.name = "TrackedTargets"' in source
    assert "target_instances.append(node)" in source
    assert "target_motion_times.append(0.0)" in source
    assert "opening_target_focus_until_s" in source
    assert "target_handoff_focus_until_s" not in source
    assert "active_target_motion_s += dt * lerpf" in source
    assert "target_zoom_time_scale" in source
    assert "4.0 + difficulty * 10.0" in source
    assert "chunk_size_m * 0.42, 0.018" not in source
    activate_body = source.split("func _activate_target(index: int) -> void:", 1)[1].split(
        "func _active_target_world_position", 1
    )[0]
    assert "_clear_children(target_root)" not in activate_body
    assert "_build_target_model" not in activate_body
    assert "if index == 0:" in activate_body
    ground_body = source.split("func _draw_chunk_ground_tiles() -> void:", 1)[1].split(
        "func _chunk_cell_has_road", 1
    )[0]
    assert "node.rotation" not in ground_body
    road_graph_body = source.split("func _draw_road_graph() -> void:", 1)[1].split(
        "func _make_tunnel_visual", 1
    )[0]
    chunked_road_branch = road_graph_body.split(
        "if chunked_generation and not chunk_map.is_empty():", 1
    )[1].split("return", 1)[0]
    assert "RoadGraphEdge" not in chunked_road_branch


def test_rapid_tracking_yaw_controls_are_horizontally_flipped() -> None:
    source = (GODOT_SCRIPTS / "rapid_tracking_runtime.gd").read_text(encoding="utf-8")
    body = source.split("func _input_vector() -> Vector2:", 1)[1].split(
        "func _any_joy_button_pressed", 1
    )[0]

    assert "KEY_A" in body and "KEY_LEFT" in body
    assert "out.x += 1.0" in body
    assert "KEY_D" in body and "KEY_RIGHT" in body
    assert "out.x -= 1.0" in body
    assert "JOYSTICK_AXIS_SENSITIVITY" in body
    assert "input_left_active" in body
    assert "input_right_active" in body
    assert "for raw_joy in Input.get_connected_joypads():" in body
    assert "-_joy_axis_with_deadzone(joy, JOY_AXIS_LEFT_X)" in body
    assert "_joy_axis_with_deadzone(joy, JOY_AXIS_LEFT_Y)" in body
    assert "JOY_BUTTON_DPAD_LEFT" in body
    assert "JOYSTICK_DEADZONE" in source
    assert "func _primary_joypad" not in source


def test_rapid_tracking_input_key_state_probe_via_godot() -> None:
    godot_bin = Path(os.environ.get("CFAST_GODOT_BIN", GODOT_DEFAULT_BIN))
    if not godot_bin.is_file():
        pytest.skip("Godot binary is not installed")
    completed = subprocess.run(
        [
            str(godot_bin),
            "--headless",
            "--path",
            str(GODOT_PROJECT_PATH),
            "--script",
            str(GODOT_RT_INPUT_PROBE),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_rapid_tracking_hold_zoom_replaces_space_capture_and_green_target_marker() -> None:
    source = (GODOT_SCRIPTS / "rapid_tracking_runtime.gd").read_text(encoding="utf-8")
    main_source = (GODOT_SCRIPTS / "main.gd").read_text(encoding="utf-8")

    handle_body = source.split("func handle_key(event: InputEventKey) -> bool:", 1)[1].split(
        "func set_paused", 1
    )[0]
    space_branch = handle_body.split("if key == KEY_SPACE:", 1)[1].split(
        "if not event.pressed:", 1
    )[0]
    assert '_set_zoom_source_active("key", event.pressed)' in space_branch
    assert '_capture("key")' not in space_branch
    assert '_capture("key")' in handle_body
    assert "_any_joy_button_pressed(JOY_BUTTON_A)" in source
    assert "Input.is_joy_button_pressed(int(raw_joy), button)" in source
    assert '_set_zoom_source_active("joystick", _any_joy_button_pressed(JOY_BUTTON_A))' in source
    assert "zoom_fov" in source
    assert "zoom_bonus_score" in source
    assert "zoom_bonus_max_score" in source
    assert "zoom_held_s" in source
    assert "zoom_on_target_s" in source
    assert "TargetMarkerH" not in source
    assert "TargetMarkerV" not in source
    assert "target_marker" not in source
    assert "SubViewport.new()" in source
    assert "TargetGuideViewport" in source
    assert "GuideCamera" in source
    assert "TargetGuideTexture" in source
    assert "target_guide_mode" in source
    assert "not target_screen_in_view" in source
    release_branch = main_source.split("if not key_event.pressed:", 1)[1].split(
        "if key == KEY_F11", 1
    )[0]
    assert "godot_owned_runtime.handle_key(key_event)" in release_branch


def test_rapid_tracking_chunk_generator_invariants_via_godot() -> None:
    godot_bin = Path(os.environ.get("CFAST_GODOT_BIN", GODOT_DEFAULT_BIN))
    if not godot_bin.is_file():
        pytest.skip("Godot binary is not installed")
    completed = subprocess.run(
        [
            str(godot_bin),
            "--headless",
            "--path",
            str(GODOT_PROJECT_PATH),
            "--script",
            str(GODOT_RT_CHUNK_PROBE),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_spatial_integration_uses_shared_chunk_map_as_answer_grid() -> None:
    source = (GODOT_SCRIPTS / "spatial_integration_runtime.gd").read_text(encoding="utf-8")

    assert 'preload("res://scripts/chunk_map_generator.gd")' in source
    assert 'purpose": "spatial_integration"' in source
    assert '"terrain_pipeline": terrain_pipeline' in source
    assert "si_large_scene_v2" in source
    assert "ChunkMapGenerator.generate" in source
    assert "_chunk_landmarks" in source
    assert "_chunk_aircraft_route" in source
    assert "_draw_chunked_grid_terrain" in source
    assert "scene_presence" in source
    assert "viewpoint_match" in source
    assert "object_count" in source
    assert "object_relation" in source
    assert "aircraft_color_route_selection" in source
    assert "aircraft_count" in source
    assert "aircraft_presence" in source
    assert "aircraft_order" in source
    assert "aircraft_tracks" in source
    assert "_scene_presence_options" in source
    assert "_viewpoint_match_options" in source
    assert "_object_count_options" in source
    assert "_static_reconstruction_decoys" in source
    assert "_nudge_one_landmark" in source
    assert "map_similarity" in source
    assert "TopDownMapPreview" in source
    assert "_topdown_map_texture" in source
    assert "_add_topdown_map_option_visual" in source
    assert "_object_relation_options" in source
    assert "_aircraft_count_options" in source
    assert "_aircraft_presence_options" in source
    assert "_aircraft_order_options" in source
    assert "RELATION_CLOSE_DISTANCE_CELLS" in source
    assert "motion_delay" in source
    assert "motion_duration" in source
    assert "_draw_aircraft_track" in source
    assert "_make_hill_dome" in source
    assert "_draw_merged_hill_clusters" in source
    assert "landmark_grid" not in source
    assert "object_kind_at_cell" not in source
    assert "aircraft_location_grid" not in source
    assert "aircraft_color_location_grid" not in source
    assert "grid_click" not in source
    assert "_answer_grid_cells" not in source
    assert "Typed grid cell" not in source
    assert 'var show_scene := stage != "question"' in source
    assert "terrain_root.visible = show_scene" in source
    assert "object_root.visible = show_scene" in source
    assert "route_root.visible = show_scene" in source
    assert "NorthSceneMarker" in source
    assert "_build_north_scene_marker" in source
    assert "_update_north_marker" in source
    assert 'north_marker_root.visible = active and stage != "question"' in source
    assert "map_north_edge - CELL_SIZE * 0.45" in source
    assert "map_north_edge - CELL_SIZE * 2.95" in source
    assert "study_orientation_index" in source
    assert "_study_camera_position" in source
    assert "_study_orientation_for_scene" in source
    assert "NorthMarkerLabel" not in source
    assert "_record_answer(\"TIMEOUT\"" not in source
    assert 'if stage == "question" and stage_elapsed_s >= question_time_limit_s' not in source
    assert 'elapsed_s >= duration_s and stage != "question"' in source
    assert "MergedSpatialHillDome" in source
    assert "ChunkHillDome" not in source
    assert "StudyLabel" not in source
    assert "Label3D" not in source
    assert "grid_cols" in source and "grid_rows" in source
    assert "CELL_SIZE * 1.03" in source
    assert "CELL_SIZE * 0.48" not in source
    assert '"chunk_map": local_chunk_map' in source
    assert "chunk_hash" in source
    assert "chunk_rule_violations" in source
    assert "hill_cell_count" in source
    assert "hill_cluster_count" in source


def test_spatial_integration_chunk_generator_invariants_via_godot() -> None:
    godot_bin = Path(os.environ.get("CFAST_GODOT_BIN", GODOT_DEFAULT_BIN))
    if not godot_bin.is_file():
        pytest.skip("Godot binary is not installed")
    completed = subprocess.run(
        [
            str(godot_bin),
            "--headless",
            "--path",
            str(GODOT_PROJECT_PATH),
            "--script",
            str(GODOT_SI_CHUNK_PROBE),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_spatial_integration_new_question_invariants_via_godot() -> None:
    godot_bin = Path(os.environ.get("CFAST_GODOT_BIN", GODOT_DEFAULT_BIN))
    if not godot_bin.is_file():
        pytest.skip("Godot binary is not installed")
    completed = subprocess.run(
        [
            str(godot_bin),
            "--headless",
            "--path",
            str(GODOT_PROJECT_PATH),
            "--script",
            str(GODOT_SI_QUESTION_PROBE),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_spatial_integration_asset_rendering_invariants_via_godot() -> None:
    godot_bin = Path(os.environ.get("CFAST_GODOT_BIN", GODOT_DEFAULT_BIN))
    if not godot_bin.is_file():
        pytest.skip("Godot binary is not installed")
    completed = subprocess.run(
        [
            str(godot_bin),
            "--headless",
            "--path",
            str(GODOT_PROJECT_PATH),
            "--script",
            str(GODOT_SI_ASSET_PROBE),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

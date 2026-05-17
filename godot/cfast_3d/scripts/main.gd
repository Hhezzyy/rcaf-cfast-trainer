extends Node3D

const AuditoryRuntime = preload("res://scripts/auditory_runtime.gd")
const GodotOwnedRuntime = preload("res://scripts/godot_owned_runtime.gd")
const RapidTrackingRuntime = preload("res://scripts/rapid_tracking_runtime.gd")
const SpatialIntegrationRuntime = preload("res://scripts/spatial_integration_runtime.gd")
const LISTEN_HOST := "127.0.0.1"
const CONTROL_SCHEMA_VERSION := 1
const FLOOR_COLOR := Color(0.18, 0.26, 0.22, 1.0)
const FOG_COLOR := Color(0.46, 0.52, 0.56, 1.0)
const LINE_COLOR := Color(0.70, 0.78, 0.82, 1.0)
const RED_COLOR := Color(0.95, 0.18, 0.15, 1.0)
const BLUE_COLOR := Color(0.12, 0.42, 0.95, 1.0)
const GREEN_COLOR := Color(0.22, 0.76, 0.34, 1.0)
const AMBER_COLOR := Color(0.95, 0.66, 0.16, 1.0)
const WHITE_COLOR := Color(0.92, 0.96, 0.98, 1.0)
const BLACK_COLOR := Color(0.05, 0.06, 0.07, 1.0)
const SKY_COLOR := Color(0.36, 0.55, 0.78, 1.0)
const GROUND_COLOR := Color(0.42, 0.50, 0.26, 1.0)
const CARD_COLOR := Color(0.74, 0.76, 0.78, 1.0)
const PANEL_BLUE := Color(0.02, 0.05, 0.42, 1.0)
const CANOPY_COLOR := Color(0.54, 0.86, 0.95, 1.0)
const TRACE_TEST_1_CAMERA_POSITION := Vector3(0.75, 2.35, 4.45)
const TRACE_TEST_1_CAMERA_TARGET := Vector3(0.0, 1.2, -6.1)
const TRACE_TEST_2_CAMERA_POSITION := Vector3(1.05, 2.55, 4.85)
const TRACE_TEST_2_CAMERA_TARGET := Vector3(0.0, 1.25, -7.55)
const TRACE_TEST_1_SCENE_OFFSET := Vector3(0.0, 0.72, -6.0)
const TRACE_TEST_2_SCENE_OFFSET := Vector3(0.0, 0.72, -7.65)
const TRACE_GUIDE_FRAME_SCALE := 2.0
const AUDITORY_TRIANGLE_POINTS := [
	Vector2(0.0, 1.22),
	Vector2(-1.056, -0.61),
	Vector2(1.056, -0.61),
]

var udp := PacketPeerUDP.new()
var control_udp := PacketPeerUDP.new()
var listen_port := 0
var control_host := LISTEN_HOST
var control_port := 0
var session_id := ""
var initial_kind := "idle"
var dynamic_root: Node3D
var camera: Camera3D
var overlay_label: Label
var menu_layer: CanvasLayer
var menu_panel: PanelContainer
var menu_vbox: VBoxContainer
var phase_layer: CanvasLayer
var phase_panel: PanelContainer
var phase_vbox: VBoxContainer
var phase_screen_active := false
var phase_screen_spec := {}
var menu_state := {}
var material_cache := {}
var last_kind := ""
var last_requested_window_mode := ""
var last_reported_window_mode := ""
var auditory_runtime: Node3D = null
var godot_owned_runtime: Node3D = null
var runtime_pause_active := false


func _ready() -> void:
	_parse_user_args()
	_build_world()
	_bind_udp()
	_connect_control_udp()
	_present_idle()


func _process(delta: float) -> void:
	while udp.get_available_packet_count() > 0:
		var text := udp.get_packet().get_string_from_utf8()
		var parsed = JSON.parse_string(text)
		if typeof(parsed) == TYPE_DICTIONARY:
			_handle_message(parsed)
	var runtime_paused := _menu_active()
	_sync_runtime_pause_state(runtime_paused)
	if not runtime_paused:
		if auditory_runtime != null:
			auditory_runtime.update_runtime(delta, camera)
		if godot_owned_runtime != null:
			godot_owned_runtime.update_runtime(delta, camera)
	_report_window_mode_if_changed()


func _parse_user_args() -> void:
	var args := OS.get_cmdline_user_args()
	for i in range(args.size()):
		var token := str(args[i])
		if token == "--listen-port" and i + 1 < args.size():
			listen_port = int(args[i + 1])
		elif token == "--control-host" and i + 1 < args.size():
			control_host = str(args[i + 1])
		elif token == "--control-port" and i + 1 < args.size():
			control_port = int(args[i + 1])
		elif token == "--session-id" and i + 1 < args.size():
			session_id = str(args[i + 1])
		elif token == "--initial-kind" and i + 1 < args.size():
			initial_kind = str(args[i + 1])


func _bind_udp() -> void:
	if listen_port <= 0:
		overlay_label.text = "CFAST Godot companion waiting for --listen-port"
		return
	var err := udp.bind(listen_port, LISTEN_HOST)
	if err != OK:
		overlay_label.text = "CFAST Godot companion UDP bind failed: " + str(err)
	else:
		overlay_label.text = "CFAST Godot companion ready"


func _connect_control_udp() -> void:
	last_reported_window_mode = _current_window_mode_token()
	if control_port <= 0:
		return
	control_udp.connect_to_host(control_host, control_port)


func _build_world() -> void:
	var env := Environment.new()
	env.background_mode = Environment.BG_COLOR
	env.background_color = FOG_COLOR
	env.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	env.ambient_light_color = Color(0.80, 0.84, 0.86, 1.0)
	env.ambient_light_energy = 0.78
	env.fog_enabled = true
	env.fog_light_color = FOG_COLOR
	env.fog_density = 0.010
	var world := WorldEnvironment.new()
	world.environment = env
	add_child(world)

	var sun := DirectionalLight3D.new()
	sun.name = "SimpleDirectionalLight"
	sun.light_energy = 0.86
	sun.shadow_enabled = false
	sun.rotation_degrees = Vector3(-46.0, -28.0, 0.0)
	add_child(sun)

	camera = Camera3D.new()
	camera.name = "CompanionCamera"
	camera.current = true
	camera.fov = 58.0
	camera.near = 0.05
	camera.far = 520.0
	add_child(camera)
	_set_camera(Vector3(0.0, 3.0, 10.0), Vector3(0.0, 1.0, -3.0))

	dynamic_root = Node3D.new()
	dynamic_root.name = "DynamicScene"
	add_child(dynamic_root)

	var canvas := CanvasLayer.new()
	add_child(canvas)
	overlay_label = Label.new()
	overlay_label.position = Vector2(12, 10)
	overlay_label.size = Vector2(920, 72)
	overlay_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	overlay_label.add_theme_font_size_override("font_size", 18)
	overlay_label.add_theme_color_override("font_color", Color(0.92, 0.96, 0.98, 1.0))
	canvas.add_child(overlay_label)

	menu_layer = CanvasLayer.new()
	menu_layer.layer = 8
	add_child(menu_layer)
	menu_panel = PanelContainer.new()
	menu_panel.visible = false
	menu_panel.position = Vector2(292, 72)
	menu_panel.custom_minimum_size = Vector2(376, 392)
	menu_layer.add_child(menu_panel)
	menu_vbox = VBoxContainer.new()
	menu_vbox.custom_minimum_size = Vector2(360, 376)
	menu_vbox.add_theme_constant_override("separation", 8)
	menu_panel.add_child(menu_vbox)

	phase_layer = CanvasLayer.new()
	phase_layer.layer = 6
	add_child(phase_layer)
	phase_panel = PanelContainer.new()
	phase_panel.visible = false
	phase_panel.position = Vector2(132, 64)
	phase_panel.custom_minimum_size = Vector2(696, 412)
	phase_layer.add_child(phase_panel)
	phase_vbox = VBoxContainer.new()
	phase_vbox.custom_minimum_size = Vector2(672, 388)
	phase_vbox.add_theme_constant_override("separation", 14)
	phase_panel.add_child(phase_vbox)


func _handle_message(message: Dictionary) -> void:
	if str(message.get("command", "")) == "quit":
		get_tree().quit()
		return
	_apply_window_mode(str(message.get("window_mode", "")))
	menu_state = _as_dict(message.get("menu", {}))
	_rebuild_menu_overlay()
	var kind := str(message.get("kind", "idle"))
	var payload = message.get("payload", {})
	var title := str(message.get("title", "CFAST Godot Companion"))
	var phase := str(message.get("phase", ""))
	overlay_label.text = title + "  |  " + phase + "  |  Godot companion"
	var start_spec := _as_dict(_as_dict(payload).get("godot_start", {}))
	if start_spec.size() > 0 and str(start_spec.get("authority", "")) == "godot":
		start_spec["progress"] = _as_dict(_as_dict(payload).get("progress", {}))
		start_spec["error"] = _as_dict(_as_dict(payload).get("error", {}))
		_present_godot_owned(kind, start_spec, phase)
		last_kind = kind
		return
	_hide_godot_owned_phase_screen()
	if kind != "auditory_capacity":
		_clear_dynamic()
	match kind:
		"auditory_capacity":
			_present_auditory(_as_dict(payload), phase)
		"rapid_tracking":
			_present_rapid_tracking(_as_dict(payload))
		"spatial_integration":
			_present_spatial_integration(_as_dict(payload))
		"trace_test_1":
			_present_trace_test_1(_as_dict(payload))
		"trace_test_2":
			_present_trace_test_2(_as_dict(payload))
		"instrument_comprehension":
			_present_instrument_comprehension(_as_dict(payload))
		_:
			_present_idle()
	last_kind = kind


func _apply_window_mode(mode_value: String) -> void:
	var token := mode_value.strip_edges().to_lower()
	if token == "" or token == last_requested_window_mode:
		return
	last_requested_window_mode = token
	if token == "fullscreen" or token == "borderless":
		DisplayServer.window_set_mode(DisplayServer.WINDOW_MODE_FULLSCREEN)
	elif token == "maximized":
		DisplayServer.window_set_mode(DisplayServer.WINDOW_MODE_MAXIMIZED)
	elif token == "windowed":
		DisplayServer.window_set_mode(DisplayServer.WINDOW_MODE_WINDOWED)
	last_reported_window_mode = _current_window_mode_token()


func _current_window_mode_token() -> String:
	var mode := DisplayServer.window_get_mode()
	if mode == DisplayServer.WINDOW_MODE_FULLSCREEN or mode == DisplayServer.WINDOW_MODE_EXCLUSIVE_FULLSCREEN:
		return "fullscreen"
	if mode == DisplayServer.WINDOW_MODE_MAXIMIZED:
		return "maximized"
	return "windowed"


func _report_window_mode_if_changed() -> void:
	var token := _current_window_mode_token()
	if token == last_reported_window_mode:
		return
	last_reported_window_mode = token
	last_requested_window_mode = token
	_send_control("set_window_mode", {"window_mode": token})


func _send_control(command: String, extra: Dictionary = {}) -> void:
	if control_port <= 0:
		return
	var message := {
		"schema": CONTROL_SCHEMA_VERSION,
		"session_id": session_id,
		"command": command,
	}
	for key in extra.keys():
		message[key] = extra[key]
	var text := JSON.stringify(message)
	control_udp.put_packet(text.to_utf8_buffer())


func _toggle_window_mode_from_godot() -> void:
	var next_mode := "windowed" if _current_window_mode_token() == "fullscreen" else "fullscreen"
	_apply_window_mode(next_mode)
	_send_control("set_window_mode", {"window_mode": next_mode})


func _menu_active() -> bool:
	return bool(menu_state.get("active", false))


func _sync_runtime_pause_state(active: bool) -> void:
	runtime_pause_active = bool(active)
	if auditory_runtime != null and auditory_runtime.has_method("set_paused"):
		auditory_runtime.call("set_paused", runtime_pause_active)
	if godot_owned_runtime != null and godot_owned_runtime.has_method("set_paused"):
		godot_owned_runtime.call("set_paused", runtime_pause_active)


func _rebuild_menu_overlay() -> void:
	if menu_vbox == null:
		return
	for child in menu_vbox.get_children():
		child.queue_free()
	if not _menu_active():
		menu_panel.visible = false
		return
	menu_panel.visible = true
	var title := Label.new()
	title.text = str(menu_state.get("title", "Paused"))
	title.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	title.add_theme_font_size_override("font_size", 28)
	menu_vbox.add_child(title)

	var subtitle := Label.new()
	subtitle.text = str(menu_state.get("subtitle", ""))
	subtitle.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	subtitle.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	subtitle.add_theme_font_size_override("font_size", 14)
	menu_vbox.add_child(subtitle)

	var mode := str(menu_state.get("mode", "menu"))
	var rows = menu_state.get("rows", [])
	var selected := int(menu_state.get("selected", 0))
	if typeof(rows) == TYPE_ARRAY:
		for i in range(rows.size()):
			var row := _as_dict(rows[i])
			if mode == "settings":
				_add_settings_row(row, i == selected)
			else:
				_add_action_row(row, i == selected)

	var quick_row := HBoxContainer.new()
	quick_row.add_theme_constant_override("separation", 8)
	menu_vbox.add_child(quick_row)
	var back_button := Button.new()
	back_button.text = "Back to Tests"
	back_button.pressed.connect(Callable(self, "_on_back_to_tests_pressed"))
	quick_row.add_child(back_button)
	var window_button := Button.new()
	window_button.text = "Windowed" if _current_window_mode_token() == "fullscreen" else "Fullscreen"
	window_button.pressed.connect(Callable(self, "_on_window_mode_pressed"))
	quick_row.add_child(window_button)


func _add_action_row(row: Dictionary, selected: bool) -> void:
	var action := str(row.get("key", ""))
	var label := str(row.get("label", action))
	var button := Button.new()
	button.text = ("> " if selected else "  ") + label
	button.alignment = HORIZONTAL_ALIGNMENT_LEFT
	button.pressed.connect(Callable(self, "_on_action_pressed").bind(action))
	menu_vbox.add_child(button)


func _add_settings_row(row: Dictionary, selected: bool) -> void:
	var key := str(row.get("key", ""))
	var label := str(row.get("label", key))
	var value := str(row.get("value", ""))
	var adjustable := bool(row.get("adjustable", false))
	var line := HBoxContainer.new()
	line.add_theme_constant_override("separation", 6)
	menu_vbox.add_child(line)
	if adjustable:
		var dec := Button.new()
		dec.text = "<"
		dec.custom_minimum_size = Vector2(36, 0)
		dec.pressed.connect(Callable(self, "_on_adjust_setting_pressed").bind(key, -1))
		line.add_child(dec)
	var button := Button.new()
	button.text = ("> " if selected else "  ") + label + ("  " + value if value != "" else "")
	button.alignment = HORIZONTAL_ALIGNMENT_LEFT
	button.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	button.pressed.connect(Callable(self, "_on_setting_pressed").bind(key))
	line.add_child(button)
	if adjustable:
		var inc := Button.new()
		inc.text = ">"
		inc.custom_minimum_size = Vector2(36, 0)
		inc.pressed.connect(Callable(self, "_on_adjust_setting_pressed").bind(key, 1))
		line.add_child(inc)


func _on_action_pressed(action: String) -> void:
	_send_control("activate_action", {"action": action})


func _on_setting_pressed(key: String) -> void:
	_send_control("activate_setting", {"key": key})


func _on_adjust_setting_pressed(key: String, direction: int) -> void:
	_send_control("adjust_setting", {"key": key, "direction": direction})


func _on_back_to_tests_pressed() -> void:
	_send_control("back_to_tests")


func _on_window_mode_pressed() -> void:
	_toggle_window_mode_from_godot()


func _unhandled_input(event: InputEvent) -> void:
	if not (event is InputEventKey):
		return
	var key_event := event as InputEventKey
	if not key_event.pressed or key_event.echo:
		return
	var key := key_event.keycode
	if key == KEY_F11 or ((key == KEY_ENTER or key == KEY_KP_PERIOD) and (key_event.alt_pressed or key_event.meta_pressed)):
		_toggle_window_mode_from_godot()
		get_viewport().set_input_as_handled()
		return
	if key == KEY_ESCAPE:
		_send_control("menu_back" if _menu_active() else "pause_toggle")
		get_viewport().set_input_as_handled()
		return
	if key == KEY_B and _menu_active():
		_send_control("back_to_tests")
		get_viewport().set_input_as_handled()
		return
	if key == KEY_KP_ENTER:
		get_viewport().set_input_as_handled()
		return
	if phase_screen_active and not _menu_active():
		if key == KEY_ENTER or key == KEY_KP_PERIOD or key == KEY_SPACE:
			_send_control("godot_phase_advance", {
				"run_key": str(phase_screen_spec.get("run_key", "")),
				"phase": str(phase_screen_spec.get("phase", "")),
				"kind": str(phase_screen_spec.get("kind", "")),
				"test_code": str(phase_screen_spec.get("test_code", "")),
			})
			get_viewport().set_input_as_handled()
			return
	if not _menu_active() and auditory_runtime != null:
		if auditory_runtime.handle_key(key_event):
			get_viewport().set_input_as_handled()
			return
	if not _menu_active() and godot_owned_runtime != null:
		if godot_owned_runtime.handle_key(key_event):
			get_viewport().set_input_as_handled()
			return
	if not _menu_active():
		return
	if key == KEY_UP or key == KEY_W:
		_send_control("menu_up")
	elif key == KEY_DOWN or key == KEY_S:
		_send_control("menu_down")
	elif key == KEY_LEFT or key == KEY_A:
		_send_control("menu_left")
	elif key == KEY_RIGHT or key == KEY_D:
		_send_control("menu_right")
	elif key == KEY_ENTER or key == KEY_KP_PERIOD or key == KEY_SPACE:
		_send_control("menu_select")
	else:
		return
	get_viewport().set_input_as_handled()


func _as_dict(value) -> Dictionary:
	if typeof(value) == TYPE_DICTIONARY:
		return value
	return {}


func _stop_auditory_runtime() -> void:
	if auditory_runtime == null:
		return
	if auditory_runtime.has_method("set_paused"):
		auditory_runtime.call("set_paused", false)
	auditory_runtime.queue_free()
	auditory_runtime = null


func _stop_godot_owned_runtime() -> void:
	if godot_owned_runtime == null:
		return
	if godot_owned_runtime.has_method("set_paused"):
		godot_owned_runtime.call("set_paused", false)
	godot_owned_runtime.queue_free()
	godot_owned_runtime = null


func _stop_all_runtimes() -> void:
	_stop_auditory_runtime()
	_stop_godot_owned_runtime()


func _clear_dynamic() -> void:
	_sync_runtime_pause_state(false)
	for child in dynamic_root.get_children():
		child.queue_free()
	auditory_runtime = null
	godot_owned_runtime = null


func _present_idle() -> void:
	_set_camera(Vector3(0.0, 3.2, 9.5), Vector3(0.0, 1.0, -2.5))
	_make_floor(22.0, 22.0)
	for x in range(-4, 5):
		_make_box("GridLineX", Vector3(float(x) * 2.0, 0.03, -4.0), Vector3(0.018, 0.018, 9.0), LINE_COLOR)
	for z in range(0, 8):
		_make_box("GridLineZ", Vector3(0.0, 0.035, -float(z) * 1.2), Vector3(9.0, 0.018, 0.018), LINE_COLOR)


func _present_auditory(payload: Dictionary, phase: String = "") -> void:
	var runtime_spec := _as_dict(payload.get("godot_runtime", {}))
	if runtime_spec.size() > 0 and str(runtime_spec.get("authority", "")) == "godot":
		if auditory_runtime == null:
			_clear_dynamic()
			auditory_runtime = AuditoryRuntime.new()
			auditory_runtime.name = "AuditoryCapacityRuntime"
			dynamic_root.add_child(auditory_runtime)
		runtime_spec["phase"] = str(runtime_spec.get("phase", phase))
		auditory_runtime.start(runtime_spec, Callable(self, "_send_control"))
		return
	_stop_auditory_runtime()
	_clear_dynamic()
	var tunnel := _as_dict(payload.get("tunnel", {}))
	var camera_payload := _as_dict(tunnel.get("camera", {}))
	if camera_payload.size() > 0:
		_set_camera(
			_vec3(camera_payload.get("position", {}), Vector3(0.0, 2.0, 5.8)),
			_vec3(camera_payload.get("target", {}), Vector3(0.0, 1.1, -6.0))
		)
	else:
		_set_camera(Vector3(0.0, 1.7, 7.4), Vector3(0.0, 0.9, -5.8))
	_make_floor(10.0, 24.0)
	_draw_auditory_tunnel(tunnel)

	var ball := _as_dict(payload.get("ball", {}))
	var ball_pose := _as_dict(ball.get("pose", {}))
	_draw_auditory_crosshair(ball_pose)
	var fallback_ball := Vector3(_float(ball.get("x", 0.0)) * 2.5, 1.0 + (_float(ball.get("y", 0.0)) * 1.2), -1.15)
	var ball_pos := _vec3(ball.get("position", {}), fallback_ball)
	var ball_color := _color_by_name(str(ball.get("color", "white")))
	if _float(ball.get("contact_ratio", 0.0)) >= 1.0:
		ball_color = RED_COLOR.lightened(0.18)
	_make_sphere("Ball", ball_pos, _float(ball.get("visual_radius", 0.32)), ball_color)

	var gates = payload.get("gates", [])
	if typeof(gates) == TYPE_ARRAY:
		for gate in gates:
			var gate_dict := _as_dict(gate)
			_make_auditory_gate(gate_dict)


func _present_godot_owned(kind: String, spec: Dictionary, phase: String = "") -> void:
	spec["phase"] = str(spec.get("phase", phase)).to_lower()
	if _godot_owned_phase_screen_needed(str(spec.get("phase", ""))):
		_stop_auditory_runtime()
		_stop_godot_owned_runtime()
		_clear_dynamic()
		_show_godot_owned_phase_screen(kind, spec)
		return
	_hide_godot_owned_phase_screen()
	if kind == "auditory_capacity":
		_stop_godot_owned_runtime()
		if auditory_runtime == null:
			_clear_dynamic()
			auditory_runtime = AuditoryRuntime.new()
			auditory_runtime.name = "AuditoryCapacityRuntime"
			dynamic_root.add_child(auditory_runtime)
		auditory_runtime.start(spec, Callable(self, "_send_control"))
		return
	_stop_auditory_runtime()
	if kind == "rapid_tracking":
		if godot_owned_runtime != null and godot_owned_runtime.name != "RapidTrackingRuntime":
			_stop_godot_owned_runtime()
		if godot_owned_runtime == null:
			_clear_dynamic()
			godot_owned_runtime = RapidTrackingRuntime.new()
			godot_owned_runtime.name = "RapidTrackingRuntime"
			dynamic_root.add_child(godot_owned_runtime)
		spec["kind"] = kind
		godot_owned_runtime.start(spec, Callable(self, "_send_control"))
		return
	if kind == "spatial_integration":
		if godot_owned_runtime != null and godot_owned_runtime.name != "SpatialIntegrationRuntime":
			_stop_godot_owned_runtime()
		if godot_owned_runtime == null:
			_clear_dynamic()
			godot_owned_runtime = SpatialIntegrationRuntime.new()
			godot_owned_runtime.name = "SpatialIntegrationRuntime"
			dynamic_root.add_child(godot_owned_runtime)
		spec["kind"] = kind
		godot_owned_runtime.start(spec, Callable(self, "_send_control"))
		return
	if godot_owned_runtime != null and godot_owned_runtime.name in ["RapidTrackingRuntime", "SpatialIntegrationRuntime"]:
		_stop_godot_owned_runtime()
	if godot_owned_runtime == null:
		_clear_dynamic()
		godot_owned_runtime = GodotOwnedRuntime.new()
		godot_owned_runtime.name = "GodotOwnedRuntime"
		dynamic_root.add_child(godot_owned_runtime)
	spec["kind"] = kind
	godot_owned_runtime.start(spec, Callable(self, "_send_control"))


func _godot_owned_phase_screen_needed(phase_value: String) -> bool:
	var token := str(phase_value).to_lower()
	return token == "instructions" or token == "practice_done" or token == "results"


func _hide_godot_owned_phase_screen() -> void:
	phase_screen_active = false
	phase_screen_spec = {}
	if phase_panel != null:
		phase_panel.visible = false


func _show_godot_owned_phase_screen(kind: String, spec: Dictionary) -> void:
	phase_screen_active = true
	phase_screen_spec = spec.duplicate(true)
	if phase_panel == null or phase_vbox == null:
		return
	for child in phase_vbox.get_children():
		child.queue_free()
	var phase_token := str(spec.get("phase", "instructions")).to_lower()
	var phase_screens := _as_dict(spec.get("phase_screens", {}))
	var screen := _as_dict(phase_screens.get(phase_token, {}))
	var title := str(screen.get("title", spec.get("title", kind.capitalize())))
	var heading := str(screen.get("heading", phase_token.replace("_", " ").capitalize()))
	var body := str(screen.get("body", "Prepare for the Godot-owned test."))
	var controls := str(screen.get("controls", "Use the Godot window controls."))
	var footer := str(screen.get("footer", "Press Enter, Space, or numpad Del to continue."))
	if phase_token == "results":
		var progress := _as_dict(_as_dict(spec.get("progress", {})))
		if progress.size() > 0:
			body = "Results received. Attempted " + str(progress.get("attempted", 0)) + ", correct " + str(progress.get("correct", 0)) + "."
	var title_label := Label.new()
	title_label.text = title
	title_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	title_label.add_theme_font_size_override("font_size", 30)
	title_label.add_theme_color_override("font_color", Color(0.94, 0.97, 1.0, 1.0))
	phase_vbox.add_child(title_label)
	var heading_label := Label.new()
	heading_label.text = heading
	heading_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	heading_label.add_theme_font_size_override("font_size", 24)
	heading_label.add_theme_color_override("font_color", Color(0.72, 0.84, 1.0, 1.0))
	phase_vbox.add_child(heading_label)
	for text in [body, controls, footer]:
		var label := Label.new()
		label.text = str(text)
		label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
		label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
		label.add_theme_font_size_override("font_size", 19)
		label.add_theme_color_override("font_color", Color(0.88, 0.93, 0.98, 1.0))
		phase_vbox.add_child(label)
	phase_panel.visible = true


func _present_rapid_tracking(payload: Dictionary) -> void:
	var camera_payload := _as_dict(payload.get("camera", {}))
	var yaw := deg_to_rad(_float(camera_payload.get("yaw_deg", 0.0)) * 0.35)
	var pitch: float = float(clamp(_float(camera_payload.get("pitch_deg", 0.0)) * 0.04, -0.6, 0.6))
	_set_camera(Vector3(sin(yaw) * 4.2, 3.0 + pitch, 8.5), Vector3(0.0, 0.9, -5.2))
	_make_terrain_chunks(int(payload.get("scene_seed", 1)), 8)
	_make_low_hills(int(payload.get("scene_seed", 1)))
	var target := _as_dict(payload.get("target", {}))
	var tx := _float(target.get("rel_x", 0.0)) * 5.0
	var ty := 0.75 + (_float(target.get("rel_y", 0.0)) * 2.0)
	var color := GREEN_COLOR if bool(target.get("visible", true)) else Color(0.55, 0.62, 0.62, 1.0)
	_make_tracking_target(str(target.get("kind", "target")), Vector3(tx, ty, -5.0), color)
	_make_capture_box(_as_dict(payload.get("capture", {})))
	_make_reticle(_as_dict(payload.get("reticle", {})))
	_make_ambient_targets(int(payload.get("scene_seed", 1)), 5)


func _present_spatial_integration(payload: Dictionary) -> void:
	var scene_view := str(payload.get("scene_view", "map")).to_lower()
	if scene_view.find("horizontal") >= 0 or scene_view.find("front") >= 0:
		_set_camera(Vector3(0.0, 2.2, 9.5), Vector3(0.0, 0.7, -3.2))
	elif scene_view.find("vertical") >= 0 or scene_view.find("side") >= 0:
		_set_camera(Vector3(8.0, 3.4, 2.0), Vector3(0.0, 0.8, -3.2))
	else:
		_set_camera(Vector3(0.0, 8.0, 10.5), Vector3(0.0, 0.0, -3.2))
	var grid := _as_dict(payload.get("grid", {}))
	var cols: int = int(max(4, int(grid.get("cols", 8))))
	var rows: int = int(max(4, int(grid.get("rows", 8))))
	_make_floor(float(cols) + 4.0, float(rows) + 4.0)
	for x in range(cols + 1):
		_make_box("GridX", Vector3((float(x) - cols * 0.5), 0.035, -float(rows) * 0.5), Vector3(0.012, 0.012, float(rows) * 0.5), LINE_COLOR)
	for y in range(rows + 1):
		_make_box("GridY", Vector3(0.0, 0.04, -(float(y) - rows * 0.5)), Vector3(float(cols) * 0.5, 0.012, 0.012), LINE_COLOR)
	_draw_hills(payload.get("hills", []), cols, rows)
	_draw_landmarks(payload.get("landmarks", []), cols, rows)
	_draw_route(payload.get("route_points", []), cols, rows)
	var aircraft := _as_dict(payload.get("aircraft", {}))
	var current := _as_dict(aircraft.get("current", {}))
	var velocity := _as_dict(aircraft.get("velocity", {}))
	var heading := rad_to_deg(atan2(_float(velocity.get("x", 0.0)), -_float(velocity.get("y", -1.0))))
	_make_aircraft("Aircraft", _grid_to_world(current, cols, rows) + Vector3(0.0, 0.55, 0.0), GREEN_COLOR, Vector3(0.0, heading, 0.0), 0.82)


func _present_trace_test_1(payload: Dictionary) -> void:
	_set_trace_guide_camera(1)
	_make_trace_guide_stage()
	_make_command_cue_panel(str(payload.get("active_command", "")))
	var frames = payload.get("frames", [])
	if typeof(frames) == TYPE_ARRAY:
		for frame in frames:
			var frame_dict := _as_dict(frame)
			var pos := _world_position(_as_dict(frame_dict.get("position", {})), 0.055, 0.055, 0.055)
			var role := str(frame_dict.get("role", "blue"))
			var attitude := _as_dict(frame_dict.get("attitude", {}))
			var hpr := Vector3(
				-_float(attitude.get("pitch_deg", 0.0)),
				_float(frame_dict.get("travel_heading_deg", 0.0)),
				-_float(attitude.get("roll_deg", 0.0))
			)
			_make_aircraft(role, pos + TRACE_TEST_1_SCENE_OFFSET, RED_COLOR if role == "red" else BLUE_COLOR, hpr, 0.92)


func _present_trace_test_2(payload: Dictionary) -> void:
	_set_trace_guide_camera(2)
	_make_trace_guide_stage()
	var aircraft = payload.get("aircraft", [])
	if typeof(aircraft) == TYPE_ARRAY:
		for track in aircraft:
			var track_dict := _as_dict(track)
			var color := _rgb_color(track_dict.get("color_rgb", []), _color_by_name(str(track_dict.get("color_name", "blue"))))
			var current := _world_position(_as_dict(track_dict.get("current_position", {})), 0.07, 0.07, 0.07) + TRACE_TEST_2_SCENE_OFFSET
			var hpr := _track_hpr(track_dict)
			_make_aircraft("Trace2Aircraft", current, color, hpr, 0.86)


func _present_instrument_comprehension(payload: Dictionary) -> void:
	_set_camera(Vector3(0.0, 3.0, 11.0), Vector3(0.0, 1.3, -3.5))
	_make_box("InstrumentBack", Vector3(0.0, 1.4, -4.2), Vector3(7.6, 3.15, 0.05), PANEL_BLUE)
	var mode := str(payload.get("option_render_mode", ""))
	if mode == "aircraft":
		_make_instrument_prompt_dials(_as_dict(payload.get("prompt_state", {})))
		var opts = payload.get("options", [])
		if typeof(opts) == TYPE_ARRAY:
			var aircraft_card_xs := [-1.65, 1.65, -3.1, 0.0, 3.1]
			for i in range(min(5, opts.size())):
				var option := _as_dict(opts[i])
				var x: float = float(aircraft_card_xs[i])
				var y: float = 1.75 if i < 2 else 0.08
				_make_aircraft_card(option, Vector3(x, y, -4.05), 1.0)
	elif mode == "instrument_panel":
		_make_aircraft_card({
			"state": payload.get("prompt_state", {}),
			"view_preset": payload.get("prompt_view_preset", "front_left"),
			"code": 0
		}, Vector3(0.0, 2.18, -4.05), 1.2)
		var opts2 = payload.get("options", [])
		if typeof(opts2) == TYPE_ARRAY:
			var panel_card_xs := [-3.2, -1.6, 0.0, 1.6, 3.2]
			for i in range(min(5, opts2.size())):
				var option2 := _as_dict(opts2[i])
				var x2: float = float(panel_card_xs[i])
				_make_instrument_panel_card(option2, Vector3(x2, 0.05, -4.05), 0.92)


func _make_sky_stage() -> void:
	_make_box("SkyBackdrop", Vector3(0.0, 2.8, -15.5), Vector3(12.0, 5.8, 0.05), SKY_COLOR)
	_make_box("HazeLayer", Vector3(0.0, 0.62, -15.45), Vector3(12.0, 0.62, 0.055), Color(0.59, 0.52, 0.48, 1.0))
	_make_floor(20.0, 26.0)


func _make_trace_guide_stage() -> void:
	_make_box("TraceGuidePanel", Vector3(0.0, 2.35, -15.62), Vector3(12.8, 6.35, 0.06) * TRACE_GUIDE_FRAME_SCALE, PANEL_BLUE)
	_make_sky_stage()


func _make_low_hills(seed_value: int) -> void:
	var rng := RandomNumberGenerator.new()
	rng.seed = seed_value + 73
	for i in range(10):
		var x := rng.randf_range(-7.5, 7.5)
		var z := rng.randf_range(-11.0, -4.5)
		var height := rng.randf_range(0.18, 0.65)
		_make_box("LowHill", Vector3(x, height * 0.5 - 0.04, z), Vector3(rng.randf_range(0.7, 1.8), height, rng.randf_range(0.8, 2.0)), Color(0.27, 0.35, 0.20, 1.0))


func _make_tracking_target(kind: String, pos: Vector3, color: Color) -> void:
	var token := kind.to_lower()
	if token.find("air") >= 0 or token.find("plane") >= 0 or token.find("fixed") >= 0:
		_make_aircraft("TrackingAircraft", pos, color, Vector3(0.0, 88.0, 0.0), 0.78)
	elif token.find("heli") >= 0:
		_make_box("HeliBody", pos, Vector3(0.38, 0.20, 0.18), color)
		_make_box("HeliTail", pos + Vector3(0.0, 0.02, 0.45), Vector3(0.08, 0.06, 0.44), color.darkened(0.12))
		_make_box("HeliRotor", pos + Vector3(0.0, 0.28, 0.0), Vector3(0.92, 0.016, 0.05), color.lightened(0.22))
	else:
		_make_box("VehicleTarget", pos, Vector3(0.38, 0.20, 0.28), color)
		_make_box("VehicleTurret", pos + Vector3(0.0, 0.22, -0.02), Vector3(0.18, 0.10, 0.16), color.lightened(0.16))


func _make_command_cue_panel(active_command: String) -> void:
	var cmds := ["LEFT", "RIGHT", "PUSH", "PULL"]
	for i in range(cmds.size()):
		var color := AMBER_COLOR if active_command.to_upper() == cmds[i] else Color(0.30, 0.32, 0.36, 1.0)
		_make_box("CommandCue", Vector3(4.6, 2.9 - float(i) * 0.35, -7.4), Vector3(0.35, 0.11, 0.025), color)


func _track_hpr(track: Dictionary) -> Vector3:
	var points = track.get("waypoints", [])
	if typeof(points) == TYPE_ARRAY and points.size() >= 2:
		var a := _world_position(_as_dict(points[max(0, points.size() - 2)]), 0.07, 0.07, 0.07)
		var b := _world_position(_as_dict(points[points.size() - 1]), 0.07, 0.07, 0.07)
		var diff := b - a
		if diff.length() > 0.001:
			var yaw := rad_to_deg(atan2(diff.x, -diff.z))
			var pitch := -rad_to_deg(atan2(diff.y, max(0.001, Vector2(diff.x, diff.z).length())))
			return Vector3(pitch, yaw, 0.0)
	return Vector3.ZERO


func _make_instrument_prompt_dials(state: Dictionary) -> void:
	var bank: float = float(clamp(_float(state.get("bank_deg", 0.0)), -45.0, 45.0))
	var pitch: float = float(clamp(_float(state.get("pitch_deg", 0.0)), -20.0, 20.0))
	var heading := int(_float(state.get("heading_deg", 0.0))) % 360
	_make_box("AttitudeDial", Vector3(-0.75, 2.55, -4.0), Vector3(0.48, 0.48, 0.035), BLACK_COLOR)
	_make_box("HorizonSky", Vector3(-0.75, 2.62 + pitch * 0.012, -3.96), Vector3(0.42, 0.19, 0.03), SKY_COLOR, Vector3(0.0, 0.0, bank))
	_make_box("HorizonGround", Vector3(-0.75, 2.43 + pitch * 0.012, -3.955), Vector3(0.42, 0.19, 0.03), Color(0.64, 0.39, 0.13, 1.0), Vector3(0.0, 0.0, bank))
	_make_box("HeadingDial", Vector3(0.75, 2.55, -4.0), Vector3(0.48, 0.48, 0.035), BLACK_COLOR)
	_make_aircraft("HeadingIcon", Vector3(0.75, 2.55, -3.92), RED_COLOR, Vector3(0.0, float(heading), 0.0), 0.24)


func _make_aircraft_card(option: Dictionary, pos: Vector3, size: float) -> void:
	var state := _as_dict(option.get("state", {}))
	var preset := str(option.get("view_preset", "front_left"))
	var code := int(option.get("code", 0))
	_make_box("AircraftCard", pos, Vector3(1.12 * size, 0.70 * size, 0.04), CARD_COLOR)
	_make_box("AircraftCardSky", pos + Vector3(0.0, 0.12 * size, 0.045), Vector3(0.96 * size, 0.30 * size, 0.025), SKY_COLOR)
	_make_box("AircraftCardGround", pos + Vector3(0.0, -0.25 * size, 0.05), Vector3(0.96 * size, 0.18 * size, 0.025), GROUND_COLOR)
	var hpr := _instrument_aircraft_hpr(state, preset)
	_make_aircraft("InstrumentAircraft", pos + Vector3(0.0, 0.03 * size, 0.18), RED_COLOR, hpr, 0.48 * size)
	if code > 0:
		_make_box("OptionBadge", pos + Vector3(-0.92 * size, -0.54 * size, 0.095), Vector3(0.14 * size, 0.09 * size, 0.02), BLACK_COLOR)


func _make_instrument_panel_card(option: Dictionary, pos: Vector3, size: float) -> void:
	var state := _as_dict(option.get("state", {}))
	var code := int(option.get("code", 0))
	_make_box("PanelCard", pos, Vector3(0.72 * size, 0.58 * size, 0.04), Color(0.45, 0.45, 0.45, 1.0))
	var bank: float = float(clamp(_float(state.get("bank_deg", 0.0)), -45.0, 45.0))
	var pitch: float = float(clamp(_float(state.get("pitch_deg", 0.0)), -20.0, 20.0))
	var heading := int(_float(state.get("heading_deg", 0.0))) % 360
	_make_box("PanelAttitude", pos + Vector3(0.0, 0.17 * size, 0.06), Vector3(0.22 * size, 0.18 * size, 0.025), BLACK_COLOR)
	_make_box("PanelHorizon", pos + Vector3(0.0, 0.17 * size + pitch * 0.006, 0.08), Vector3(0.20 * size, 0.018 * size, 0.016), WHITE_COLOR, Vector3(0.0, 0.0, bank))
	_make_box("PanelHeading", pos + Vector3(0.0, -0.15 * size, 0.06), Vector3(0.22 * size, 0.16 * size, 0.025), BLACK_COLOR)
	_make_aircraft("PanelHeadingIcon", pos + Vector3(0.0, -0.15 * size, 0.10), RED_COLOR, Vector3(0.0, float(heading), 0.0), 0.14 * size)
	if code > 0:
		_make_box("PanelBadge", pos + Vector3(-0.56 * size, -0.43 * size, 0.09), Vector3(0.10 * size, 0.065 * size, 0.018), BLACK_COLOR)


func _instrument_aircraft_hpr(state: Dictionary, preset: String) -> Vector3:
	var heading := _float(state.get("heading_deg", 0.0))
	var pitch := _float(state.get("pitch_deg", 0.0))
	var bank := _float(state.get("bank_deg", 0.0))
	var token := preset.to_lower()
	var view_yaw := 0.0
	var view_pitch := 0.0
	if token.find("front_right") >= 0:
		view_yaw = -32.0
	elif token.find("profile_left") >= 0:
		view_yaw = 88.0
	elif token.find("profile_right") >= 0:
		view_yaw = -88.0
	elif token.find("top") >= 0:
		view_pitch = -62.0
	else:
		view_yaw = 32.0
	return Vector3(-pitch + view_pitch, heading + view_yaw, -bank)


func _make_floor(width: float, depth: float) -> MeshInstance3D:
	return _make_box("TerrainFloor", Vector3(0.0, -0.04, -depth * 0.32), Vector3(width * 0.5, 0.04, depth * 0.5), FLOOR_COLOR)


func _make_terrain_chunks(seed_value: int, radius: int) -> void:
	var rng := RandomNumberGenerator.new()
	rng.seed = seed_value
	for x in range(-radius, radius + 1):
		for z in range(-radius, 2):
			var h := rng.randf_range(0.02, 0.22)
			var tint := Color(0.16 + h, 0.25 + h * 0.5, 0.18, 1.0)
			_make_box("TerrainChunk", Vector3(float(x), -0.08 + h * 0.5, float(z)), Vector3(0.49, h, 0.49), tint)


func _make_ambient_targets(seed_value: int, count_hint: int) -> void:
	var rng := RandomNumberGenerator.new()
	rng.seed = seed_value + 5103
	var count: int = int(clamp(count_hint, 3, 8))
	for i in range(count):
		var x := rng.randf_range(-5.5, 5.5)
		var z := rng.randf_range(-10.5, -3.0)
		_make_box("AmbientTarget", Vector3(x, 0.42, z), Vector3(0.22, 0.32, 0.22), Color(0.28, 0.34, 0.32, 1.0))


func _make_capture_box(capture: Dictionary) -> void:
	var half_w: float = float(max(0.4, _float(capture.get("half_width", 0.16)) * 10.0))
	var half_h: float = float(max(0.3, _float(capture.get("half_height", 0.13)) * 7.0))
	var z := -4.4
	var color := AMBER_COLOR if bool(capture.get("target_in_capture_box", false)) else LINE_COLOR
	_make_box("CaptureTop", Vector3(0.0, 1.25 + half_h, z), Vector3(half_w, 0.025, 0.025), color)
	_make_box("CaptureBottom", Vector3(0.0, 1.25 - half_h, z), Vector3(half_w, 0.025, 0.025), color)
	_make_box("CaptureLeft", Vector3(-half_w, 1.25, z), Vector3(0.025, half_h, 0.025), color)
	_make_box("CaptureRight", Vector3(half_w, 1.25, z), Vector3(0.025, half_h, 0.025), color)


func _make_reticle(reticle: Dictionary) -> void:
	var x := _float(reticle.get("x", 0.0)) * 5.0
	var y := 1.25 + (_float(reticle.get("y", 0.0)) * 2.0)
	_make_box("ReticleH", Vector3(x, y, -4.2), Vector3(0.42, 0.018, 0.018), WHITE_COLOR)
	_make_box("ReticleV", Vector3(x, y, -4.2), Vector3(0.018, 0.42, 0.018), WHITE_COLOR)


func _draw_hills(hills, cols: int, rows: int) -> void:
	if typeof(hills) != TYPE_ARRAY:
		return
	for hill in hills:
		var h := _as_dict(hill)
		var pos := _grid_to_world(h, cols, rows)
		var radius: float = float(max(0.35, float(h.get("radius", 1)) * 0.22))
		var height: float = float(max(0.25, float(h.get("height", 1)) * 0.24))
		_make_box("Hill", pos + Vector3(0.0, height * 0.5, 0.0), Vector3(radius, height, radius), Color(0.32, 0.42, 0.28, 1.0))


func _draw_landmarks(landmarks, cols: int, rows: int) -> void:
	if typeof(landmarks) != TYPE_ARRAY:
		return
	for landmark in landmarks:
		var item := _as_dict(landmark)
		var pos := _grid_to_world(item, cols, rows)
		_make_box("Landmark", pos + Vector3(0.0, 0.32, 0.0), Vector3(0.22, 0.32, 0.22), _color_for_landmark(str(item.get("kind", "landmark"))))


func _draw_route(points, cols: int, rows: int) -> void:
	if typeof(points) != TYPE_ARRAY:
		return
	var previous := Vector3.ZERO
	var have_previous := false
	for point in points:
		var current := _grid_to_world(_as_dict(point), cols, rows) + Vector3(0.0, 0.16, 0.0)
		_make_sphere("RoutePoint", current, 0.12, AMBER_COLOR)
		if have_previous:
			_make_segment(previous, current, AMBER_COLOR, 0.05)
		previous = current
		have_previous = true


func _draw_auditory_tunnel(tunnel: Dictionary) -> void:
	var samples = tunnel.get("samples", [])
	if typeof(samples) != TYPE_ARRAY or samples.size() < 2:
		_make_box("TunnelBackWall", Vector3(0.0, 1.0, -10.8), Vector3(3.1, 1.05, 0.035), Color(0.06, 0.11, 0.20, 1.0))
		_make_box("TubeLeft", Vector3(-3.0, 1.0, -4.8), Vector3(0.08, 1.0, 8.5), Color(0.18, 0.32, 0.39, 1.0))
		_make_box("TubeRight", Vector3(3.0, 1.0, -4.8), Vector3(0.08, 1.0, 8.5), Color(0.18, 0.32, 0.39, 1.0))
		return
	var rx: float = maxf(0.25, _float(tunnel.get("inner_rx", 0.94)))
	var ry: float = maxf(0.22, _float(tunnel.get("inner_rz", 0.68)))
	var rail_angles := [0.0, PI * 0.5, PI, PI * 1.5]
	for i in range(samples.size() - 1):
		var a := _as_dict(samples[i])
		var b := _as_dict(samples[i + 1])
		for angle in rail_angles:
			var p0 := _auditory_ring_point(a, rx, ry, float(angle))
			var p1 := _auditory_ring_point(b, rx, ry, float(angle))
			_make_segment(p0, p1, Color(0.15, 0.33, 0.58, 0.72), 0.022)
	for i in range(samples.size()):
		if i % 2 != 0 and i != samples.size() - 1:
			continue
		var sample := _as_dict(samples[i])
		var depth: float = clampf(_float(sample.get("depth_norm", 0.0)), -0.2, 1.0)
		var tint := Color(0.27 + (0.14 * (1.0 - depth)), 0.52, 0.69 + (0.12 * (1.0 - depth)), 0.78)
		_draw_auditory_ring(sample, rx, ry, tint, 0.028)


func _draw_auditory_ring(sample: Dictionary, rx: float, ry: float, color: Color, width: float) -> void:
	var segments := 18
	var previous := _auditory_ring_point(sample, rx, ry, 0.0)
	for idx in range(1, segments + 1):
		var angle := (float(idx) / float(segments)) * TAU
		var current := _auditory_ring_point(sample, rx, ry, angle)
		_make_segment(previous, current, color, width)
		previous = current


func _auditory_ring_point(sample: Dictionary, rx: float, ry: float, angle: float) -> Vector3:
	var center := _vec3(sample.get("pos", {}), Vector3.ZERO)
	var right := _vec3(sample.get("right", {}), Vector3.RIGHT).normalized()
	var up := _vec3(sample.get("up", {}), Vector3.UP).normalized()
	return center + (right * cos(angle) * rx) + (up * sin(angle) * ry)


func _draw_auditory_crosshair(pose: Dictionary) -> void:
	if pose.size() <= 0:
		return
	var center := _vec3(pose.get("pos", {}), Vector3(0.0, 1.1, 0.0))
	var right := _vec3(pose.get("right", {}), Vector3.RIGHT).normalized()
	var up := _vec3(pose.get("up", {}), Vector3.UP).normalized()
	_make_segment(center - (right * 0.95), center + (right * 0.95), Color(0.95, 0.16, 0.18, 0.82), 0.014)
	_make_segment(center - (up * 0.62), center + (up * 0.62), Color(0.95, 0.16, 0.18, 0.82), 0.014)


func _make_auditory_gate(gate: Dictionary) -> void:
	var pose := _as_dict(gate.get("pose", {}))
	var pos := _vec3(gate.get("position", {}), Vector3(_float(gate.get("x_norm", 0.0)) * 1.35, 1.0 + (_float(gate.get("y_norm", 0.0)) * 1.0), -3.0))
	var right := _vec3(pose.get("right", {}), Vector3.RIGHT).normalized()
	var up := _vec3(pose.get("up", {}), Vector3.UP).normalized()
	var color := _color_by_name(str(gate.get("color", "blue")))
	if _float(gate.get("flash_strength", 0.0)) > 0.05:
		color = _color_by_name(str(gate.get("flash_color", "white")))
	var radius: float = maxf(0.18, _float(gate.get("aperture_radius", 0.42)))
	var shape := str(gate.get("shape", "circle")).to_lower()
	if shape == "circle":
		_draw_auditory_circle_gate(pos, right, up, radius, color)
	elif shape == "triangle":
		var points := []
		for p in AUDITORY_TRIANGLE_POINTS:
			points.append(pos + (right * p.x * radius * 0.94) + (up * p.y * radius * 0.94))
		_draw_auditory_poly_gate(points, color, 0.045)
	else:
		var points := [
			pos - (right * radius) - (up * radius),
			pos + (right * radius) - (up * radius),
			pos + (right * radius) + (up * radius),
			pos - (right * radius) + (up * radius),
		]
		_draw_auditory_poly_gate(points, color, 0.045)


func _draw_auditory_circle_gate(pos: Vector3, right: Vector3, up: Vector3, radius: float, color: Color) -> void:
	var segments := 24
	var previous := pos + (right * radius)
	for idx in range(1, segments + 1):
		var angle := (float(idx) / float(segments)) * TAU
		var current := pos + (right * cos(angle) * radius) + (up * sin(angle) * radius)
		_make_segment(previous, current, color, 0.045)
		previous = current


func _draw_auditory_poly_gate(points: Array, color: Color, width: float) -> void:
	if points.size() < 2:
		return
	for idx in range(points.size()):
		_make_segment(points[idx], points[(idx + 1) % points.size()], color, width)


func _make_aircraft(name_value: String, pos: Vector3, color: Color, hpr: Vector3 = Vector3.ZERO, size: float = 1.0) -> Node3D:
	var root := Node3D.new()
	root.name = name_value
	root.position = pos
	root.rotation_degrees = hpr
	dynamic_root.add_child(root)
	_make_box_child(root, name_value + "Fuselage", Vector3(0.0, 0.0, -0.02 * size), Vector3(0.13 * size, 0.11 * size, 0.58 * size), color)
	_make_box_child(root, name_value + "Nose", Vector3(0.0, 0.0, -0.48 * size), Vector3(0.09 * size, 0.08 * size, 0.22 * size), color.lightened(0.10))
	_make_box_child(root, name_value + "Wing", Vector3(0.0, 0.0, -0.02 * size), Vector3(0.62 * size, 0.030 * size, 0.13 * size), color.darkened(0.12))
	_make_box_child(root, name_value + "Tailplane", Vector3(0.0, 0.06 * size, 0.43 * size), Vector3(0.28 * size, 0.030 * size, 0.08 * size), color.lightened(0.05))
	_make_box_child(root, name_value + "Fin", Vector3(0.0, 0.18 * size, 0.40 * size), Vector3(0.040 * size, 0.18 * size, 0.07 * size), color.lightened(0.18))
	_make_box_child(root, name_value + "Canopy", Vector3(0.0, 0.10 * size, -0.22 * size), Vector3(0.10 * size, 0.055 * size, 0.13 * size), CANOPY_COLOR)
	return root


func _make_segment(a: Vector3, b: Vector3, color: Color, width: float) -> void:
	if a.distance_to(b) < 0.01:
		return
	var center := (a + b) * 0.5
	var node := _make_box("Segment", center, Vector3(width, width, a.distance_to(b) * 0.5), color)
	node.look_at(b, Vector3.UP)


func _make_box(name_value: String, pos: Vector3, scale_value: Vector3, color: Color, rotation_value: Vector3 = Vector3.ZERO) -> MeshInstance3D:
	return _make_box_child(dynamic_root, name_value, pos, scale_value, color, rotation_value)


func _make_box_child(parent: Node3D, name_value: String, pos: Vector3, scale_value: Vector3, color: Color, rotation_value: Vector3 = Vector3.ZERO) -> MeshInstance3D:
	var mesh := BoxMesh.new()
	mesh.size = Vector3(1.0, 1.0, 1.0)
	var node := MeshInstance3D.new()
	node.name = name_value
	node.mesh = mesh
	node.position = pos
	node.rotation_degrees = rotation_value
	node.scale = scale_value
	node.material_override = _material(color)
	parent.add_child(node)
	return node


func _make_sphere(name_value: String, pos: Vector3, radius: float, color: Color) -> MeshInstance3D:
	var mesh := SphereMesh.new()
	mesh.radial_segments = 12
	mesh.rings = 6
	var node := MeshInstance3D.new()
	node.name = name_value
	node.mesh = mesh
	node.position = pos
	node.scale = Vector3(radius, radius, radius)
	node.material_override = _material(color)
	dynamic_root.add_child(node)
	return node


func _material(color: Color) -> StandardMaterial3D:
	var key := color.to_html(true)
	if material_cache.has(key):
		return material_cache[key]
	var mat := StandardMaterial3D.new()
	mat.albedo_color = color
	mat.roughness = 1.0
	mat.metallic = 0.0
	if color.a < 0.999:
		mat.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	material_cache[key] = mat
	return mat


func _set_camera(position_value: Vector3, target: Vector3) -> void:
	camera.position = position_value
	camera.look_at(target, Vector3.UP)


func _set_trace_guide_camera(trace_number: int) -> void:
	if trace_number == 2:
		_set_camera(TRACE_TEST_2_CAMERA_POSITION, TRACE_TEST_2_CAMERA_TARGET)
		return
	_set_camera(TRACE_TEST_1_CAMERA_POSITION, TRACE_TEST_1_CAMERA_TARGET)


func _grid_to_world(point: Dictionary, cols: int, rows: int) -> Vector3:
	return Vector3(float(point.get("x", 0)) - (float(cols) - 1.0) * 0.5, 0.0, -(float(point.get("y", 0)) - (float(rows) - 1.0) * 0.5))


func _vec3(value, fallback: Vector3 = Vector3.ZERO) -> Vector3:
	if typeof(value) != TYPE_DICTIONARY:
		return fallback
	var item := _as_dict(value)
	return Vector3(
		_float(item.get("x", fallback.x)),
		_float(item.get("y", fallback.y)),
		_float(item.get("z", fallback.z))
	)


func _world_position(point: Dictionary, scale_x: float, scale_y: float, scale_z: float) -> Vector3:
	return Vector3(_float(point.get("x", 0.0)) * scale_x, _float(point.get("z", 0.0)) * scale_z, -_float(point.get("y", 0.0)) * scale_y)


func _float(value) -> float:
	if typeof(value) == TYPE_FLOAT or typeof(value) == TYPE_INT:
		return float(value)
	return float(str(value))


func _color_by_name(name_value: String) -> Color:
	var token := name_value.to_lower()
	if token.find("red") >= 0:
		return RED_COLOR
	if token.find("blue") >= 0:
		return BLUE_COLOR
	if token.find("green") >= 0:
		return GREEN_COLOR
	if token.find("yellow") >= 0 or token.find("amber") >= 0 or token.find("orange") >= 0:
		return AMBER_COLOR
	if token.find("black") >= 0:
		return BLACK_COLOR
	if token.find("white") >= 0:
		return WHITE_COLOR
	return Color(0.55, 0.70, 0.80, 1.0)


func _color_for_landmark(kind: String) -> Color:
	var token := kind.to_lower()
	if token.find("tower") >= 0:
		return Color(0.72, 0.72, 0.68, 1.0)
	if token.find("water") >= 0 or token.find("lake") >= 0:
		return Color(0.20, 0.46, 0.72, 1.0)
	if token.find("road") >= 0:
		return Color(0.45, 0.43, 0.39, 1.0)
	return Color(0.62, 0.54, 0.32, 1.0)


func _rgb_color(value, fallback: Color) -> Color:
	if typeof(value) != TYPE_ARRAY or value.size() < 3:
		return fallback
	return Color(float(value[0]) / 255.0, float(value[1]) / 255.0, float(value[2]) / 255.0, 1.0)

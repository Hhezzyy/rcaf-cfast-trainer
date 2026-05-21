extends SceneTree

const RapidRuntime = preload("res://scripts/rapid_tracking_runtime.gd")


func _init() -> void:
	var runtime := RapidRuntime.new()
	runtime.active = true
	var failures: Array = []
	_send_key(failures, runtime, KEY_LEFT, true)
	_expect(failures, runtime.input_left_active, "left key should latch active")
	_expect(failures, runtime.handle_key(_event(KEY_UP, true)), "up key should be consumed")
	var pressed_vec: Vector2 = runtime._input_vector()
	_expect(failures, pressed_vec.x > 0.0 and pressed_vec.y > 0.0, "pressed movement keys should produce movement vector")
	_send_key(failures, runtime, KEY_LEFT, false)
	_send_key(failures, runtime, KEY_UP, false)
	var released_vec: Vector2 = runtime._input_vector()
	_expect(failures, released_vec.length() < 0.01, "released movement keys should clear movement vector")
	_send_key(failures, runtime, KEY_D, true)
	var right_vec: Vector2 = runtime._input_vector()
	_expect(failures, right_vec.x < 0.0, "D key should preserve existing yaw direction mapping")
	_send_key(failures, runtime, KEY_D, false)
	if not failures.is_empty():
		for failure in failures:
			push_error(str(failure))
		runtime.free()
		quit(1)
		return
	print(JSON.stringify({"movement_probe": "ok"}))
	runtime.free()
	quit(0)


func _event(key: int, pressed: bool) -> InputEventKey:
	var event := InputEventKey.new()
	event.keycode = key
	event.pressed = pressed
	return event


func _send_key(failures: Array, runtime: Node, key: int, pressed: bool) -> void:
	var consumed := bool(runtime.handle_key(_event(key, pressed)))
	if not consumed:
		failures.append("movement key was not consumed: " + str(key))


func _expect(failures: Array, condition: bool, message: String) -> void:
	if not condition:
		failures.append(message)

from __future__ import annotations

from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "godot" / "cfast_3d" / "scripts"
MAIN_PATH = SCRIPTS_DIR / "main.gd"
RUNTIME_PATHS = [
    SCRIPTS_DIR / "auditory_runtime.gd",
    SCRIPTS_DIR / "rapid_tracking_runtime.gd",
    SCRIPTS_DIR / "spatial_integration_runtime.gd",
    SCRIPTS_DIR / "godot_owned_runtime.gd",
]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _function_body(source: str, name: str) -> str:
    marker = f"func {name}"
    start = source.index(marker)
    next_func = source.find("\nfunc ", start + len(marker))
    return source[start:] if next_func < 0 else source[start:next_func]


def test_companion_freezes_godot_runtimes_while_pause_menu_is_active() -> None:
    source = _read(MAIN_PATH)
    process_body = _function_body(source, "_process")
    sync_body = _function_body(source, "_sync_runtime_pause_state")

    assert "var runtime_paused := _menu_active()" in process_body
    assert "_sync_runtime_pause_state(runtime_paused)" in process_body
    assert "if not runtime_paused:" in process_body
    assert "auditory_runtime.update_runtime" in process_body
    assert "godot_owned_runtime.update_runtime" in process_body
    assert 'has_method("set_paused")' in sync_body
    assert 'call("set_paused", runtime_pause_active)' in sync_body


def test_all_godot_owned_runtimes_have_authoritative_pause_gate() -> None:
    for path in RUNTIME_PATHS:
        source = _read(path)
        update_body = _function_body(source, "update_runtime")
        handle_key_body = _function_body(source, "handle_key")
        clear_body = _function_body(source, "_clear_runtime")

        assert "var paused := false" in source, path.name
        assert "func set_paused(value: bool)" in source, path.name
        assert "paused" in update_body, path.name
        assert "paused" in handle_key_body, path.name
        assert "paused = false" in clear_body, path.name


def test_auditory_pause_freezes_audio_and_tts() -> None:
    source = _read(SCRIPTS_DIR / "auditory_runtime.gd")
    pause_body = _function_body(source, "set_paused")
    clear_body = _function_body(source, "_clear_runtime")

    assert "stream_paused = paused" in pause_body
    assert "DisplayServer.tts_pause()" in pause_body
    assert "DisplayServer.tts_resume()" in pause_body
    assert "DisplayServer.tts_stop()" in clear_body


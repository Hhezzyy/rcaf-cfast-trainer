from __future__ import annotations

import math
import os
import subprocess
import wave
from array import array
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pygame
import pytest

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from cfast_trainer.app import _AmbientTrack, _AuditoryCapacityAudioAdapter
from cfast_trainer.auditory_capacity import (
    AuditoryCapacityCommandType,
    AuditoryCapacityGateDirective,
    AuditoryCapacityInstructionEvent,
    build_auditory_capacity_test,
)
from cfast_trainer.cognitive_core import Phase


@dataclass
class _FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t


@dataclass(frozen=True)
class _FakeSound:
    key: str
    variant: int
    duration_s: float


@dataclass
class _NoopSpeaker:
    enabled: bool = False
    pending_count: int = 0
    _rate_wpm: int = 182

    def update(self) -> None:
        return

    def stop(self) -> None:
        return

    def speak(self, _text: str) -> None:
        return

    def set_rate_wpm(self, rate_wpm: int) -> None:
        self._rate_wpm = int(rate_wpm)


class _FakeChannel:
    def __init__(self) -> None:
        self._busy = False
        self._remaining_s = 0.0
        self.played: list[object] = []
        self.volumes: list[float] = []

    def get_busy(self) -> bool:
        return self._busy

    def play(self, sound: object) -> None:
        self.played.append(sound)
        self._busy = True
        self._remaining_s = float(getattr(sound, "duration_s", 0.0))

    def stop(self) -> None:
        self._busy = False
        self._remaining_s = 0.0

    def set_volume(self, volume: float) -> None:
        self.volumes.append(float(volume))

    def advance(self, dt: float) -> None:
        if not self._busy:
            return
        self._remaining_s = max(0.0, self._remaining_s - float(dt))
        if self._remaining_s <= 1e-6:
            self._busy = False


def _write_wav(path: Path, *, duration_s: float, sample_rate: int = 22_050) -> None:
    frame_count = max(1, int(round(duration_s * sample_rate)))
    samples = array("h")
    for idx in range(frame_count):
        phase = (idx % 240) / 240.0
        value = int(round(math.sin(phase * math.tau) * 9000.0))
        samples.append(value)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(samples.tobytes())


def _make_adapter(*, seed: int = 17) -> _AuditoryCapacityAudioAdapter:
    adapter = _AuditoryCapacityAudioAdapter.__new__(_AuditoryCapacityAudioAdapter)
    adapter._sample_rate = 22_050
    adapter._session_seed = int(seed)
    adapter._voice_rng = __import__("random").Random(seed)
    adapter._interrupt_rng = __import__("random").Random(seed + 1)
    adapter._available = True
    adapter._ambient_tracks = {}
    adapter._ambient_snippet_bank = {}
    adapter._ambient_snippet_variant_index = {}
    adapter._ambient_cache_ready = False
    adapter._ambient_slots = [None, None, None]
    adapter._ambient_active_sounds = [None, None, None]
    adapter._ambient_recent_keys = []
    adapter._ambient_target_layers = 0
    adapter._ambient_started_at_s = 0.0
    adapter._next_ambient_change_at_s = 0.0
    adapter._last_mature_clip_at_s = -10_000.0
    adapter._next_interrupt_at_s = 0.0
    adapter._active_noise_source = None
    adapter._mixer_channels = 1
    adapter._bg_channel = None
    adapter._dist_channel = None
    adapter._cue_channel = None
    adapter._alert_channel = None
    adapter._fx_channel = None
    adapter._voice_channel = None
    adapter._interrupt_channel = None
    adapter._ambient_aux_channel = None
    adapter._tts_callsign = _NoopSpeaker()
    adapter._tts_commands = _NoopSpeaker()
    adapter._tts_story = _NoopSpeaker()
    adapter._last_callsign_cue = None
    adapter._last_color_command = None
    adapter._last_beep_active = False
    adapter._last_sequence_display = None
    adapter._last_instruction_uid = None
    adapter._last_assigned_callsigns = ()
    adapter._last_phase = None
    adapter._last_instruction_callsign = None
    adapter._instructor_voice_choices = ("Samantha", "Alex")
    adapter._fallback_distractor_voice = "Karen"
    adapter._instructor_voice = "Samantha"
    adapter._distractor_voice = "Alex"
    adapter._instructor_cached_base_rate_wpm = 182
    adapter._callsign_base_rate_wpm = 168
    adapter._commands_base_rate_wpm = 172
    adapter._story_base_rate_wpm = 160
    adapter._instructor_rate_wpm = 182
    adapter._instructor_noise_level = 0.0
    adapter._instructor_distortion_level = 0.0
    adapter._speech_enabled = False
    adapter._speech_cache_root = Path("/tmp")
    adapter._speech_sound_cache = {}
    adapter._speech_pcm_cache = {}
    adapter._speech_phrase_cache = {}
    adapter._prepared_speech_phases = set()
    adapter._interrupt_sounds = []
    adapter._noise_sound = None
    adapter._beep_sound = None
    adapter._color_sounds = {}
    adapter._voice_queue = []
    adapter._callsign_cache = {}
    adapter._sequence_cache = {}
    adapter._story_segments = ()
    return adapter


def _build_payload(
    *,
    background_noise_source: str | None,
    background_noise_level: float,
    briefing_active: bool = False,
    background_distortion_level: float = 0.0,
    instructor_noise_level: float = 0.0,
    instructor_distortion_level: float = 0.0,
    instructor_rate_wpm: int = 182,
    ambient_layer_target: int = 1,
) -> Any:
    clock = _FakeClock()
    engine = build_auditory_capacity_test(clock=clock, seed=77, difficulty=0.58)
    engine.start_practice()
    clock.t = 4.25
    engine.update()
    payload = engine.snapshot().payload
    assert payload is not None
    return replace(
        payload,
        background_noise_source=background_noise_source,
        background_noise_level=float(background_noise_level),
        background_distortion_level=float(background_distortion_level),
        distortion_level=float(background_distortion_level),
        instructor_noise_level=float(instructor_noise_level),
        instructor_distortion_level=float(instructor_distortion_level),
        instructor_rate_wpm=int(instructor_rate_wpm),
        ambient_layer_target=int(ambient_layer_target),
        briefing_active=briefing_active,
        callsign_cue=None,
        color_command=None,
        sequence_display=None,
        instruction_uid=None,
        instruction_text=None,
    )


def test_prepare_ambient_snippet_bank_builds_bounded_file_and_voice_variants(tmp_path: Path) -> None:
    adapter = _make_adapter()
    normal_path = tmp_path / "background_noise_1.wav"
    mature_path = tmp_path / "mature_1.wav"
    _write_wav(normal_path, duration_s=10.0, sample_rate=adapter._sample_rate)
    _write_wav(mature_path, duration_s=9.0, sample_rate=adapter._sample_rate)

    adapter._ambient_tracks = {
        normal_path.name: _AmbientTrack(
            path=normal_path,
            base_volume=0.22,
            base_weight=1.35,
            late_weight=0.10,
            sample_rate=adapter._sample_rate,
            channels=1,
            frame_count=int(10.0 * adapter._sample_rate),
            is_generated_noise=False,
        ),
        mature_path.name: _AmbientTrack(
            path=mature_path,
            base_volume=0.18,
            base_weight=0.35,
            late_weight=0.85,
            sample_rate=adapter._sample_rate,
            channels=1,
            frame_count=int(9.0 * adapter._sample_rate),
            is_generated_noise=False,
        ),
    }
    adapter._sound_from_pcm = lambda mono_pcm: {"samples": len(mono_pcm)}

    adapter._prepare_ambient_snippet_bank()

    assert adapter._ambient_cache_ready is True
    assert set(adapter._ambient_snippet_bank) == {normal_path.name, mature_path.name}
    for sound in adapter._ambient_snippet_bank[normal_path.name]:
        assert sound["samples"] == int(
            adapter._ambient_file_snippet_duration_s * adapter._sample_rate
        )
    for sound in adapter._ambient_snippet_bank[mature_path.name]:
        assert sound["samples"] == int(
            adapter._ambient_mature_snippet_duration_s * adapter._sample_rate
        )


def test_prepare_ambient_snippet_bank_builds_bounded_generated_noise_variants() -> None:
    adapter = _make_adapter()
    render_calls: list[float] = []
    adapter._ambient_tracks = {
        "generated_noise": _AmbientTrack(
            path=None,
            base_volume=0.22,
            base_weight=1.0,
            late_weight=0.0,
            sample_rate=adapter._sample_rate,
            channels=1,
            frame_count=int(adapter._sample_rate * 6.0),
            is_generated_noise=True,
        )
    }
    adapter._render_noise_pcm = lambda *, duration_s, seed, gain: (
        render_calls.append(float(duration_s))
        or array("h", [0] * int(round(duration_s * adapter._sample_rate)))
    )
    adapter._sound_from_pcm = lambda mono_pcm: {"samples": len(mono_pcm)}

    adapter._prepare_ambient_snippet_bank()

    assert render_calls == [6.0, 6.0]
    assert len(adapter._ambient_snippet_bank["generated_noise"]) == 2


def test_sync_background_layers_stays_silent_during_briefing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter()
    key = "background_noise_1.wav"
    bg_channel = _FakeChannel()
    dist_channel = _FakeChannel()
    adapter._bg_channel = bg_channel
    adapter._dist_channel = dist_channel
    adapter._ambient_tracks = {
        key: _AmbientTrack(
            path=Path("/tmp") / key,
            base_volume=0.22,
            base_weight=1.35,
            late_weight=0.10,
            sample_rate=adapter._sample_rate,
            channels=1,
            frame_count=int(8.0 * adapter._sample_rate),
            is_generated_noise=False,
        )
    }
    adapter._ambient_snippet_bank = {
        key: (_FakeSound(key=key, variant=0, duration_s=2.0),),
    }
    adapter._ambient_snippet_variant_index = {key: 0}
    adapter._ambient_cache_ready = True
    monkeypatch.setattr(pygame.time, "get_ticks", lambda: 10_000)

    payload = _build_payload(
        background_noise_source=key,
        background_noise_level=0.0,
        briefing_active=True,
    )
    adapter._sync_background_layers(payload=payload)

    assert bg_channel.played == []
    assert dist_channel.played == []
    assert adapter._ambient_target_layers == 0


def test_sync_background_layers_refills_same_slot_from_cache_until_replan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter()
    key = "background_noise_2.wav"
    fake_time_s = 10.0
    bg_channel = _FakeChannel()
    dist_channel = _FakeChannel()
    adapter._bg_channel = bg_channel
    adapter._dist_channel = dist_channel
    adapter._ambient_tracks = {
        key: _AmbientTrack(
            path=Path("/tmp") / key,
            base_volume=0.21,
            base_weight=1.20,
            late_weight=0.12,
            sample_rate=adapter._sample_rate,
            channels=1,
            frame_count=int(8.0 * adapter._sample_rate),
            is_generated_noise=False,
        )
    }
    adapter._ambient_snippet_bank = {
        key: (
            _FakeSound(key=key, variant=0, duration_s=2.0),
            _FakeSound(key=key, variant=1, duration_s=2.0),
        )
    }
    adapter._ambient_snippet_variant_index = {key: 0}
    adapter._ambient_cache_ready = True
    adapter._ambient_started_at_s = fake_time_s
    adapter._ambient_target_layers = 1
    adapter._next_ambient_change_at_s = fake_time_s + 60.0
    payload = _build_payload(
        background_noise_source=key,
        background_noise_level=0.62,
        briefing_active=False,
    )
    pick_calls = 0
    original_pick = adapter._pick_ambient_key

    def _pick_spy(**kwargs: Any) -> str | None:
        nonlocal pick_calls
        pick_calls += 1
        return original_pick(**kwargs)

    monkeypatch.setattr(pygame.time, "get_ticks", lambda: int(fake_time_s * 1000.0))
    monkeypatch.setattr(adapter, "_pick_ambient_key", _pick_spy)
    monkeypatch.setattr(
        adapter,
        "_build_ambient_snippet_sound",
        lambda *args, **kwargs: pytest.fail("steady-state sync must not rebuild snippets"),
    )

    adapter._sync_background_layers(payload=payload)
    assert [sound.variant for sound in bg_channel.played] == [0]
    assert pick_calls == 1
    assert adapter._ambient_slots[0] == key

    fake_time_s += 2.1
    bg_channel.advance(2.1)
    adapter._sync_background_layers(payload=payload)

    fake_time_s += 2.1
    bg_channel.advance(2.1)
    adapter._sync_background_layers(payload=payload)

    assert [sound.variant for sound in bg_channel.played] == [0, 1, 0]
    assert pick_calls == 1
    assert adapter._ambient_slots[0] == key
    assert dist_channel.played == []


def test_practice_sync_uses_only_warmup_cache_builds_for_45_seconds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter()
    key = "background_noise_3.wav"
    fake_time_s = 1.0
    bg_channel = _FakeChannel()
    dist_channel = _FakeChannel()
    adapter._bg_channel = bg_channel
    adapter._dist_channel = dist_channel
    adapter._ambient_tracks = {
        key: _AmbientTrack(
            path=Path("/tmp") / key,
            base_volume=0.21,
            base_weight=1.10,
            late_weight=0.10,
            sample_rate=adapter._sample_rate,
            channels=1,
            frame_count=int(8.0 * adapter._sample_rate),
            is_generated_noise=False,
        )
    }
    warmup_calls: list[tuple[str, float | None]] = []

    def _warmup_builder(
        track_key: str,
        *,
        elapsed_ratio: float,
        duration_s: float | None = None,
    ) -> _FakeSound:
        warmup_calls.append((track_key, duration_s))
        variant = len(warmup_calls) - 1
        return _FakeSound(key=track_key, variant=variant, duration_s=2.0)

    monkeypatch.setattr(adapter, "_build_ambient_snippet_sound", _warmup_builder)
    adapter._prepare_ambient_snippet_bank()
    assert warmup_calls == [
        (key, adapter._ambient_file_snippet_duration_s),
        (key, adapter._ambient_file_snippet_duration_s),
    ]

    monkeypatch.setattr(
        adapter,
        "_build_ambient_snippet_sound",
        lambda *args, **kwargs: pytest.fail("runtime sync must not rebuild ambient snippets"),
    )
    monkeypatch.setattr(pygame.time, "get_ticks", lambda: int(fake_time_s * 1000.0))
    monkeypatch.setattr(adapter, "_sync_phase_speech_profile", lambda **kwargs: None)
    monkeypatch.setattr(adapter, "_sync_assigned_callsigns", lambda *, payload: None)
    monkeypatch.setattr(adapter, "_sync_distortion_layer", lambda *, payload: None)
    monkeypatch.setattr(adapter, "_sync_interrupt_distractors", lambda *, payload: None)
    monkeypatch.setattr(adapter, "_sync_beep_cue", lambda *, payload: None)
    monkeypatch.setattr(adapter, "_sync_instruction_cue", lambda *, payload: None)

    payload = _build_payload(
        background_noise_source=key,
        background_noise_level=0.58,
        briefing_active=False,
    )
    step_s = 0.5
    for _ in range(int(45.0 / step_s)):
        adapter.sync(phase=Phase.PRACTICE, payload=payload)
        fake_time_s += step_s
        bg_channel.advance(step_s)
        dist_channel.advance(step_s)

    assert len(bg_channel.played) >= 12
    assert adapter._ambient_slots[0] == key
    assert adapter._ambient_cache_ready is True


def test_phase_profile_is_seeded_and_deterministic_for_same_session_seed() -> None:
    payload = replace(
        _build_payload(
            background_noise_source=None,
            background_noise_level=0.0,
            briefing_active=True,
        ),
        session_seed=91,
    )
    adapter_a = _make_adapter(seed=91)
    adapter_b = _make_adapter(seed=91)

    adapter_a._sync_phase_speech_profile(phase=Phase.PRACTICE, payload=payload)
    adapter_b._sync_phase_speech_profile(phase=Phase.PRACTICE, payload=payload)

    assert adapter_a._instructor_voice == adapter_b._instructor_voice
    assert adapter_a._distractor_voice == adapter_b._distractor_voice


def test_audio_schedule_uses_phase_elapsed_and_is_repeatable_for_same_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_payload = replace(
        _build_payload(
            background_noise_source=None,
            background_noise_level=0.62,
            briefing_active=False,
        ),
        session_seed=91,
    )
    timeline = (4.5, 7.1, 9.9, 12.6, 15.4, 18.8)

    def _run(
        seed: int,
    ) -> tuple[list[tuple[str, int]], list[tuple[str, int]], tuple[str | None, str | None, str | None]]:
        adapter = _make_adapter(seed=seed)
        bg_channel = _FakeChannel()
        dist_channel = _FakeChannel()
        aux_channel = _FakeChannel()
        interrupt_channel = _FakeChannel()
        adapter._bg_channel = bg_channel
        adapter._dist_channel = dist_channel
        adapter._ambient_aux_channel = aux_channel
        adapter._interrupt_channel = interrupt_channel
        adapter._ambient_tracks = {
            "background_noise_1.wav": _AmbientTrack(
                path=Path("/tmp/background_noise_1.wav"),
                base_volume=0.22,
                base_weight=1.35,
                late_weight=0.10,
                sample_rate=adapter._sample_rate,
                channels=1,
                frame_count=int(8.0 * adapter._sample_rate),
                is_generated_noise=False,
            ),
            "background_noise_4.wav": _AmbientTrack(
                path=Path("/tmp/background_noise_4.wav"),
                base_volume=0.23,
                base_weight=1.25,
                late_weight=0.16,
                sample_rate=adapter._sample_rate,
                channels=1,
                frame_count=int(8.0 * adapter._sample_rate),
                is_generated_noise=False,
            ),
        }
        adapter._ambient_snippet_bank = {
            "background_noise_1.wav": (
                _FakeSound(key="background_noise_1.wav", variant=0, duration_s=2.0),
                _FakeSound(key="background_noise_1.wav", variant=1, duration_s=2.0),
            ),
            "background_noise_4.wav": (
                _FakeSound(key="background_noise_4.wav", variant=0, duration_s=2.0),
                _FakeSound(key="background_noise_4.wav", variant=1, duration_s=2.0),
            ),
        }
        adapter._ambient_snippet_variant_index = {
            "background_noise_1.wav": 0,
            "background_noise_4.wav": 0,
        }
        adapter._ambient_cache_ready = True
        adapter._interrupt_sounds = [
            _FakeSound(key="interrupt_a", variant=0, duration_s=0.8),
            _FakeSound(key="interrupt_b", variant=1, duration_s=0.8),
        ]

        last_t = 0.0
        for t in timeline:
            dt = t - last_t
            bg_channel.advance(dt)
            dist_channel.advance(dt)
            aux_channel.advance(dt)
            interrupt_channel.advance(dt)
            payload = replace(base_payload, phase_elapsed_s=t, session_seed=seed)
            adapter._sync_phase_speech_profile(phase=Phase.PRACTICE, payload=payload)
            adapter._sync_background_layers(payload=payload)
            adapter._sync_interrupt_distractors(payload=payload)
            last_t = t
        ambient_history = [
            (str(sound.key), int(sound.variant))
            for sound in bg_channel.played + dist_channel.played + aux_channel.played
        ]
        interrupt_history = [
            (str(sound.key), int(sound.variant)) for sound in interrupt_channel.played
        ]
        return ambient_history, interrupt_history, tuple(adapter._ambient_slots)

    monkeypatch.setattr(
        pygame.time,
        "get_ticks",
        lambda: pytest.fail("auditory audio scheduling should use payload.phase_elapsed_s"),
    )

    first = _run(91)
    second = _run(91)

    assert first == second


def test_macos_say_cache_renders_once_then_reuses_existing_wav(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter()
    adapter._speech_enabled = True
    adapter._speech_cache_root = tmp_path
    calls: list[tuple[str, ...]] = []

    def _fake_run(cmd: list[str], check: bool, stdout: Any, stderr: Any) -> subprocess.CompletedProcess[str]:
        calls.append(tuple(cmd))
        if "afconvert" in cmd[0]:
            _write_wav(Path(cmd[-1]), duration_s=0.35, sample_rate=adapter._sample_rate)
        else:
            Path(cmd[cmd.index("-o") + 1]).write_bytes(b"fake-aiff")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", _fake_run)
    monkeypatch.setattr("cfast_trainer.app.shutil.which", lambda name: f"/usr/bin/{name}")

    first = adapter._speech_wav_path(
        text="Change colour to RED.",
        voice="Samantha",
        rate_wpm=182,
    )
    second = adapter._speech_wav_path(
        text="Change colour to RED.",
        voice="Samantha",
        rate_wpm=182,
    )
    faster = adapter._speech_wav_path(
        text="Change colour to RED.",
        voice="Samantha",
        rate_wpm=273,
    )

    assert first == second
    assert faster != first
    assert first.exists()
    assert faster.exists()
    assert len(calls) == 4
    assert "say" in calls[0][0]
    assert "afconvert" in calls[1][0]


def test_instruction_sound_uses_concise_gate_directive_tokens() -> None:
    adapter = _make_adapter()
    captured: list[tuple[str, ...]] = []

    def _capture_phrase(
        *,
        tokens: tuple[str, ...],
        voice: str,
        rate_wpm: int,
        noise_level: float,
        distortion_level: float,
    ) -> _FakeSound:
        captured.append(tokens)
        return _FakeSound(key="directive", variant=0, duration_s=0.2)

    adapter._build_phrase_sound_from_tokens = _capture_phrase
    payload = _build_payload(
        background_noise_source="background_noise_1.wav",
        background_noise_level=0.2,
    )

    avoid_instruction = AuditoryCapacityInstructionEvent(
        event_id=10,
        timestamp_s=0.1,
        addressed_call_sign="RAVEN",
        speaker_id="lead",
        command_type=AuditoryCapacityCommandType.GATE_DIRECTIVE,
        payload=AuditoryCapacityGateDirective(
            action="AVOID",
            match_kind="SHAPE",
            match_value="TRIANGLE",
        ),
        expires_at_s=1.0,
        is_distractor=False,
    )
    avoid_payload = replace(
        payload,
        active_instruction=avoid_instruction,
        instruction_text="RAVEN. Avoid the next triangle gate.",
        instruction_uid=10,
        instruction_command_type=AuditoryCapacityCommandType.GATE_DIRECTIVE.value,
    )
    adapter._instruction_sound(payload=avoid_payload, omit_callsign=False)

    pass_instruction = AuditoryCapacityInstructionEvent(
        event_id=11,
        timestamp_s=0.1,
        addressed_call_sign="EAGLE",
        speaker_id="lead",
        command_type=AuditoryCapacityCommandType.GATE_DIRECTIVE,
        payload=AuditoryCapacityGateDirective(
            action="PASS",
            match_kind="COLOR",
            match_value="RED",
        ),
        expires_at_s=1.0,
        is_distractor=False,
    )
    pass_payload = replace(
        payload,
        active_instruction=pass_instruction,
        instruction_text="EAGLE. Take the next red gate.",
        instruction_uid=11,
        instruction_command_type=AuditoryCapacityCommandType.GATE_DIRECTIVE.value,
    )
    adapter._instruction_sound(payload=pass_payload, omit_callsign=False)

    assert captured == [
        ("RAVEN", "Avoid the next", "TRIANGLE", "gate"),
        ("EAGLE", "Take the next", "RED", "gate"),
    ]


def test_top_end_background_mix_can_reach_three_ambient_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(seed=23)
    fake_time_s = 10.0
    bg_channel = _FakeChannel()
    dist_channel = _FakeChannel()
    aux_channel = _FakeChannel()
    adapter._bg_channel = bg_channel
    adapter._dist_channel = dist_channel
    adapter._ambient_aux_channel = aux_channel
    adapter._ambient_tracks = {
        key: _AmbientTrack(
            path=Path("/tmp") / key,
            base_volume=0.20,
            base_weight=1.0,
            late_weight=0.0,
            sample_rate=adapter._sample_rate,
            channels=1,
            frame_count=int(8.0 * adapter._sample_rate),
            is_generated_noise=False,
        )
        for key in (
            "background_noise_1.wav",
            "background_noise_2.wav",
            "background_noise_3.wav",
        )
    }
    adapter._ambient_snippet_bank = {
        key: (_FakeSound(key=key, variant=0, duration_s=2.0),)
        for key in adapter._ambient_tracks
    }
    adapter._ambient_snippet_variant_index = {key: 0 for key in adapter._ambient_tracks}
    adapter._ambient_cache_ready = True
    payload = replace(
        _build_payload(
            background_noise_source=None,
            background_noise_level=0.82,
            ambient_layer_target=3,
        ),
        phase_elapsed_s=fake_time_s,
    )
    adapter._ambient_started_at_s = fake_time_s
    monkeypatch.setattr(adapter._voice_rng, "random", lambda: 0.99)

    adapter._sync_background_layers(payload=payload)

    assert adapter._ambient_target_layers == 3
    assert len(bg_channel.played) == 1
    assert len(dist_channel.played) == 1
    assert len(aux_channel.played) == 1
    assert len({slot for slot in adapter._ambient_slots if slot is not None}) == 3


def test_instructor_profile_scales_rate_and_stays_dirtier_than_background() -> None:
    adapter = _make_adapter()
    payload = _build_payload(
        background_noise_source=None,
        background_noise_level=0.34,
        background_distortion_level=0.06,
        instructor_noise_level=0.48,
        instructor_distortion_level=0.30,
        instructor_rate_wpm=273,
        ambient_layer_target=3,
    )

    adapter._sync_phase_speech_profile(phase=Phase.SCORED, payload=payload)
    debug = adapter.debug_state(payload=payload)

    assert adapter._instructor_rate_wpm == 273
    assert adapter._tts_callsign._rate_wpm == int(round(168 * (273 / 182)))
    assert adapter._tts_commands._rate_wpm == int(round(172 * (273 / 182)))
    assert adapter._tts_story._rate_wpm == int(round(160 * (273 / 182)))
    assert float(debug["instructor_noise_level"]) > float(debug["noise_level"])
    assert float(debug["instructor_distortion_level"]) > float(
        debug["background_distortion_level"]
    )


def test_instructor_voice_channel_is_louder_than_interrupt_channel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter()
    adapter._available = True
    adapter._voice_channel = _FakeChannel()
    adapter._interrupt_channel = _FakeChannel()
    adapter._interrupt_sounds = [_FakeSound(key="interrupt", variant=0, duration_s=1.0)]
    adapter._play_instructor_sound(_FakeSound(key="voice", variant=0, duration_s=1.0))

    payload = _build_payload(
        background_noise_source=None,
        background_noise_level=0.82,
        briefing_active=False,
    )
    payload = replace(payload, phase_elapsed_s=2.0)
    adapter._next_interrupt_at_s = 1.0
    monkeypatch.setattr(pygame.time, "get_ticks", lambda: 2_000)
    adapter._sync_interrupt_distractors(payload=payload)

    voice_volume = adapter._voice_channel.volumes[-1]
    interrupt_volume = adapter._interrupt_channel.volumes[-1]
    assert voice_volume > interrupt_volume


def test_instructor_voice_queue_does_not_interrupt_current_clip() -> None:
    adapter = _make_adapter()
    adapter._available = True
    voice_channel = _FakeChannel()
    adapter._voice_channel = voice_channel
    first = _FakeSound(key="briefing-1", variant=0, duration_s=1.2)
    second = _FakeSound(key="briefing-2", variant=0, duration_s=0.9)

    adapter._play_instructor_sound(first)
    adapter._play_instructor_sound(second)

    assert voice_channel.played == [first]
    assert adapter._voice_queue == [second]

    voice_channel.advance(1.3)
    adapter._drain_instructor_queue()

    assert voice_channel.played == [first, second]
    assert adapter._voice_queue == []


def test_runtime_instruction_replaces_older_pending_voice_queue() -> None:
    adapter = _make_adapter()
    adapter._available = True
    voice_channel = _FakeChannel()
    adapter._voice_channel = voice_channel
    first = _FakeSound(key="briefing-1", variant=0, duration_s=1.2)
    stale = _FakeSound(key="old-command", variant=0, duration_s=0.7)
    fresh = _FakeSound(key="new-command", variant=0, duration_s=0.7)

    adapter._play_instructor_sound(first)
    adapter._play_instructor_sound(stale)
    adapter._play_instructor_sound_queued(fresh, replace_pending=True)

    assert voice_channel.played == [first]
    assert adapter._voice_queue == [fresh]


def test_story_chatter_path_is_disabled_in_new_adapter() -> None:
    adapter = _make_adapter()

    assert adapter._story_segments == ()

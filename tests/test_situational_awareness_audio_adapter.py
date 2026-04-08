from __future__ import annotations

import math
import os
import subprocess
import wave
from array import array
from dataclasses import dataclass
from pathlib import Path

import pygame

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from cfast_trainer.cognitive_core import Phase
from cfast_trainer.situational_awareness import (
    SituationalAwarenessPayload,
    SituationalAwarenessScenarioFamily,
)
from cfast_trainer.situational_awareness_audio import SituationalAwarenessAudioAdapter


@dataclass(frozen=True)
class _FakeSound:
    key: str
    duration_s: float = 0.2


class _FakeChannel:
    def __init__(self) -> None:
        self._busy = False
        self.played: list[object] = []
        self.volumes: list[float] = []

    def get_busy(self) -> bool:
        return self._busy

    def play(self, sound: object) -> None:
        self.played.append(sound)
        self._busy = True

    def stop(self) -> None:
        self._busy = False

    def set_volume(self, volume: float) -> None:
        self.volumes.append(float(volume))


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


def _make_adapter(tmp_path: Path) -> SituationalAwarenessAudioAdapter:
    adapter = SituationalAwarenessAudioAdapter.__new__(SituationalAwarenessAudioAdapter)
    adapter._speech_cache_root = tmp_path
    adapter._sound_cache = {}
    adapter._pcm_cache = {}
    adapter._queued_texts = []
    adapter._last_audio_token = ()
    adapter._last_prefetch_token = ()
    adapter._channel = _FakeChannel()
    adapter._mixer_channels = 1
    adapter._speech_backend = "say"
    adapter._speech_enabled = True
    adapter._available = True
    adapter._sample_rate = 22_050
    adapter._rate_wpm = 184
    adapter._volume = 0.34
    adapter._voice_macos = "Samantha"
    adapter._max_queue = 8
    return adapter


def _payload(
    *,
    announcement_token: tuple[object, ...] | None,
    announcement_lines: tuple[str, ...],
    speech_prefetch_lines: tuple[str, ...],
) -> SituationalAwarenessPayload:
    return SituationalAwarenessPayload(
        scenario_family=SituationalAwarenessScenarioFamily.MERGE_CONFLICT,
        scenario_label="Tactical Merge 1/1",
        scenario_index=1,
        scenario_total=1,
        active_channels=("pictorial", "coded", "numerical", "aural"),
        active_query_kinds=("current_location",),
        focus_label="Focus",
        segment_label="Segment",
        segment_index=1,
        segment_total=1,
        segment_time_remaining_s=30.0,
        scenario_elapsed_s=12.0,
        scenario_time_remaining_s=30.0,
        next_query_in_s=6.0,
        visible_contacts=(),
        cue_card=None,
        waypoints=(),
        top_strip_text="",
        top_strip_fade=0.0,
        display_clock_text="11:00:12",
        active_query=None,
        answer_mode=None,
        correct_answer_token=None,
        announcement_token=announcement_token,
        announcement_lines=announcement_lines,
        radio_log=announcement_lines,
        speech_prefetch_lines=speech_prefetch_lines,
    )


def test_wav_path_is_deterministic_for_same_text_and_backend(tmp_path: Path) -> None:
    adapter = _make_adapter(tmp_path)

    p1 = adapter._wav_path(text="RAVEN TWO is enemy tank, trending D4.")
    p2 = adapter._wav_path(text="RAVEN TWO is enemy tank, trending D4.")
    p3 = adapter._wav_path(text="LEEDS is friendly helicopter, check in channel 3.")

    assert p1 == p2
    assert p1 != p3


def test_macos_say_cache_renders_once_then_reuses_existing_wav(tmp_path: Path, monkeypatch) -> None:
    adapter = _make_adapter(tmp_path)
    calls: list[tuple[str, ...]] = []

    def _fake_run(args: list[str], check: bool, stdout=None, stderr=None):
        calls.append(tuple(args))
        if "afconvert" in str(args[0]):
            _write_wav(Path(args[-1]), duration_s=0.12)
        else:
            Path(args[args.index("-o") + 1]).write_bytes(b"FORM")
        return subprocess.CompletedProcess(args=args, returncode=0)

    monkeypatch.setattr(subprocess, "run", _fake_run)

    path1 = adapter._ensure_wav(text="RAVEN TWO is enemy tank, trending D4.")
    path2 = adapter._ensure_wav(text="RAVEN TWO is enemy tank, trending D4.")

    assert path1 == path2
    assert path1.exists()
    assert len(calls) == 2


def test_sync_prefetches_catalog_and_only_queues_announcement_lines(tmp_path: Path) -> None:
    adapter = _make_adapter(tmp_path)
    loaded: list[str] = []

    def _fake_load_sound(text: str) -> _FakeSound:
        loaded.append(text)
        return _FakeSound(key=text)

    adapter._load_sound = _fake_load_sound

    payload = _payload(
        announcement_token=("sa", "radio", 1),
        announcement_lines=("RAVEN TWO is enemy tank, trending D4.",),
        speech_prefetch_lines=(
            "LEEDS is friendly helicopter, check in channel 3.",
            "RAVEN TWO is enemy tank, trending D4.",
        ),
    )

    adapter.sync(phase=Phase.PRACTICE, payload=payload)

    assert loaded == [
        "LEEDS is friendly helicopter, check in channel 3.",
        "RAVEN TWO is enemy tank, trending D4.",
        "RAVEN TWO is enemy tank, trending D4.",
    ]
    assert adapter._channel.played == [_FakeSound(key="RAVEN TWO is enemy tank, trending D4.")]

    adapter._channel.stop()
    adapter.sync(phase=Phase.PRACTICE, payload=payload)
    assert adapter._channel.played == [_FakeSound(key="RAVEN TWO is enemy tank, trending D4.")]

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import wave
from array import array
from pathlib import Path

import pygame

from .cognitive_core import Phase
from .situational_awareness import SituationalAwarenessPayload


class SituationalAwarenessAudioAdapter:
    """Cached offline-first radio speech for Situational Awareness.

    The adapter pre-renders short radio lines to deterministic wav files and
    reuses them across sessions. Only radio chatter is spoken. Intro text and
    question prompts remain visual-only.
    """

    _sample_rate = 22_050
    _rate_wpm = 184
    _volume = 0.34
    _voice_macos = "Samantha"
    _max_queue = 8

    def __init__(self) -> None:
        self._speech_cache_root = Path(tempfile.gettempdir()) / "cfast_sa_speech"
        self._sound_cache: dict[str, pygame.mixer.Sound] = {}
        self._pcm_cache: dict[str, array[int]] = {}
        self._queued_texts: list[str] = []
        self._last_audio_token: tuple[object, ...] = ()
        self._last_prefetch_token: tuple[str, ...] = ()
        self._channel: pygame.mixer.Channel | None = None
        self._mixer_channels = 1
        self._speech_backend = self._resolve_backend()
        self._speech_enabled = (
            self._speech_backend is not None
            and os.environ.get("CFAST_DISABLE_TTS", "0") != "1"
            and os.environ.get("SDL_AUDIODRIVER", "").strip().lower() != "dummy"
        )
        self._available = False

        try:
            if pygame.mixer.get_init() is None:
                pygame.mixer.init(frequency=self._sample_rate, size=-16, channels=1, buffer=512)
            mix_state = pygame.mixer.get_init()
            if mix_state is not None:
                self._sample_rate = int(mix_state[0])
                self._mixer_channels = max(1, int(mix_state[2]))
            pygame.mixer.set_num_channels(max(8, int(pygame.mixer.get_num_channels())))
            self._channel = pygame.mixer.Channel(7)
            self._available = True
        except Exception:
            self._available = False

    @property
    def pending_count(self) -> int:
        active = 1 if self._channel is not None and self._channel.get_busy() else 0
        return active + len(self._queued_texts)

    def stop(self) -> None:
        self._queued_texts.clear()
        self._last_audio_token = ()
        self._last_prefetch_token = ()
        if self._channel is not None:
            try:
                self._channel.stop()
            except Exception:
                pass

    def sync(self, *, phase: Phase, payload: SituationalAwarenessPayload | None) -> None:
        if payload is None or phase not in (Phase.PRACTICE, Phase.SCORED):
            self.stop()
            return

        self._prefetch_for_payload(payload)
        token = tuple(payload.announcement_token or ())
        if token != self._last_audio_token:
            self._last_audio_token = token
            self._enqueue_lines(payload.announcement_lines)
        self._pump_queue()

    def _prefetch_for_payload(self, payload: SituationalAwarenessPayload) -> None:
        if not self._speech_enabled:
            return
        token = tuple(str(line) for line in payload.speech_prefetch_lines if str(line).strip() != "")
        if token == self._last_prefetch_token:
            return
        self._last_prefetch_token = token
        for line in token:
            self._load_sound(line)

    def _enqueue_lines(self, lines: tuple[str, ...]) -> None:
        for raw in lines:
            text = " ".join(str(raw).strip().split())
            if text == "":
                continue
            if self._queued_texts and self._queued_texts[-1] == text:
                continue
            self._queued_texts.append(text)
        if len(self._queued_texts) > self._max_queue:
            del self._queued_texts[: len(self._queued_texts) - self._max_queue]

    def _pump_queue(self) -> None:
        if not self._available or not self._speech_enabled or self._channel is None:
            return
        if self._channel.get_busy() or not self._queued_texts:
            return
        next_text = self._queued_texts.pop(0)
        sound = self._load_sound(next_text)
        self._channel.set_volume(float(self._volume))
        self._channel.play(sound)

    def _resolve_backend(self) -> str | None:
        if sys.platform == "darwin":
            say_bin = shutil.which("say") or "/usr/bin/say"
            afconvert_bin = shutil.which("afconvert") or "/usr/bin/afconvert"
            if Path(say_bin).exists() and Path(afconvert_bin).exists():
                return "say"
        if os.name == "nt":
            if shutil.which("powershell") is not None or shutil.which("pwsh") is not None:
                return "powershell"
        return None

    def _wav_path(self, *, text: str) -> Path:
        backend = self._speech_backend or "silent"
        key = f"{backend}|{self._rate_wpm}|{text}".encode("utf-8")
        digest = hashlib.sha1(key).hexdigest()
        return self._speech_cache_root / f"sa-{backend}-{digest}.wav"

    def _ensure_wav(self, *, text: str) -> Path:
        target = self._wav_path(text=text)
        if target.exists():
            return target
        self._speech_cache_root.mkdir(parents=True, exist_ok=True)
        backend = self._speech_backend
        if backend == "say":
            aiff_path = target.with_suffix(".aiff")
            say_bin = shutil.which("say") or "/usr/bin/say"
            afconvert_bin = shutil.which("afconvert") or "/usr/bin/afconvert"
            subprocess.run(
                [
                    say_bin,
                    "-v",
                    self._voice_macos,
                    "-r",
                    str(int(self._rate_wpm)),
                    "-o",
                    str(aiff_path),
                    text,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            try:
                subprocess.run(
                    [
                        afconvert_bin,
                        "-f",
                        "WAVE",
                        "-d",
                        "LEI16@22050",
                        "-c",
                        "1",
                        str(aiff_path),
                        str(target),
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            finally:
                try:
                    aiff_path.unlink()
                except FileNotFoundError:
                    pass
            return target
        if backend == "powershell":
            ps_bin = shutil.which("powershell") or shutil.which("pwsh")
            if ps_bin is None:
                raise RuntimeError("PowerShell TTS backend is unavailable.")
            script = (
                "Add-Type -AssemblyName System.Speech; "
                "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                "$path=$args[0]; "
                "$text=($args[1..($args.Length-1)] -join ' '); "
                "$s.SetOutputToWaveFile($path); "
                "$s.Speak($text); "
                "$s.Dispose();"
            )
            subprocess.run(
                [ps_bin, "-NoProfile", "-NonInteractive", "-Command", script, str(target), text],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return target
        raise RuntimeError("No supported SA speech backend is available.")

    def _load_sound(self, text: str) -> pygame.mixer.Sound:
        cached = self._sound_cache.get(text)
        if cached is not None:
            return cached
        pcm = self._load_pcm(text=text)
        sound = self._sound_from_pcm(pcm)
        self._sound_cache[text] = sound
        return sound

    def _load_pcm(self, *, text: str) -> array[int]:
        cached = self._pcm_cache.get(text)
        if cached is not None:
            return array("h", cached)
        wav_path = self._ensure_wav(text=text)
        meta = self._probe_wav_metadata(wav_path)
        if meta is None:
            pcm = array("h", [0] * max(1, int(self._sample_rate * 0.08)))
            self._pcm_cache[text] = array("h", pcm)
            return pcm
        sample_rate, channels, frame_count = meta
        pcm = self._read_wav_pcm(path=wav_path, channels=channels, frame_count=frame_count)
        if sample_rate != self._sample_rate:
            pcm = self._resample_pcm(samples=pcm, src_rate=sample_rate, dst_rate=self._sample_rate)
        self._pcm_cache[text] = array("h", pcm)
        return array("h", pcm)

    @staticmethod
    def _probe_wav_metadata(path: Path) -> tuple[int, int, int] | None:
        if not path.exists():
            return None
        try:
            with wave.open(str(path), "rb") as wav_file:
                channels = int(wav_file.getnchannels())
                sample_width = int(wav_file.getsampwidth())
                sample_rate = int(wav_file.getframerate())
                frame_count = int(wav_file.getnframes())
        except Exception:
            return None
        if sample_width != 2 or channels <= 0 or sample_rate <= 0 or frame_count <= 0:
            return None
        return sample_rate, channels, frame_count

    @staticmethod
    def _read_wav_pcm(*, path: Path, channels: int, frame_count: int) -> array[int]:
        with wave.open(str(path), "rb") as wav_file:
            raw = wav_file.readframes(max(1, int(frame_count)))
        samples = array("h")
        samples.frombytes(raw)
        if channels <= 1:
            return samples
        mono = array("h")
        for idx in range(0, len(samples), channels):
            chunk = samples[idx : idx + channels]
            if len(chunk) <= 0:
                continue
            mono.append(int(sum(int(value) for value in chunk) / len(chunk)))
        return mono

    @staticmethod
    def _resample_pcm(*, samples: array[int], src_rate: int, dst_rate: int) -> array[int]:
        if src_rate <= 0 or dst_rate <= 0 or len(samples) == 0:
            return array("h")
        if src_rate == dst_rate:
            return array("h", samples)
        out_count = max(1, int(round((len(samples) * dst_rate) / float(src_rate))))
        out = array("h")
        max_src = len(samples) - 1
        for idx in range(out_count):
            src_idx = int((idx * src_rate) / float(dst_rate))
            src_idx = max(0, min(max_src, src_idx))
            out.append(int(samples[src_idx]))
        return out

    def _sound_from_pcm(self, mono_pcm: array[int]) -> pygame.mixer.Sound:
        pcm = self._to_mixer_channels(mono_pcm)
        return pygame.mixer.Sound(buffer=pcm.tobytes())

    def _to_mixer_channels(self, mono_pcm: array[int]) -> array[int]:
        channels = max(1, int(self._mixer_channels))
        if channels == 1:
            return mono_pcm
        interleaved = array("h")
        for sample in mono_pcm:
            for _ in range(channels):
                interleaved.append(int(sample))
        return interleaved

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pygame

from ..cognitive_core import Phase, TestSnapshot
from .camera import TARGET_VIEW_LIMIT
from .debug import rapid_tracking_debug_lines
from .entities import RapidTrackingPayload
from .legacy import rapid_tracking_target_label


@dataclass(frozen=True, slots=True)
class RapidTrackingUiContext:
    app: Any
    small_font: pygame.font.Font
    tiny_font: pygame.font.Font


def _render_wrapped_text(
    *,
    surface: pygame.Surface,
    font: pygame.font.Font,
    rect: pygame.Rect,
    text: str,
    color: tuple[int, int, int],
    line_gap: int = 2,
    max_lines: int = 6,
) -> None:
    words = str(text).split()
    if not words:
        return
    lines: list[str] = []
    current = ""
    for word in words:
        trial = word if current == "" else f"{current} {word}"
        if font.size(trial)[0] <= rect.w:
            current = trial
            continue
        if current:
            lines.append(current)
        current = word
    if current:
        lines.append(current)

    y = rect.y
    line_h = font.get_linesize() + int(line_gap)
    for line in lines[: max(1, int(max_lines))]:
        if y + line_h > rect.bottom:
            break
        surface.blit(font.render(line, True, color), (rect.x, y))
        y += line_h


class RapidTrackingExerciseRenderer:
    """Package-local RT presenter used by both the exercise and app shell."""

    def render(
        self,
        *,
        surface: pygame.Surface,
        snap: TestSnapshot,
        payload: RapidTrackingPayload | None,
        engine: object | None,
        context: RapidTrackingUiContext,
    ) -> None:
        w, h = surface.get_size()
        bg = (6, 24, 20)
        panel_bg = (12, 44, 34)
        border = (180, 228, 204)
        text_main = (229, 248, 237)
        text_muted = (168, 210, 190)
        accent = (90, 214, 166)
        warning = (240, 192, 94)

        surface.fill(bg)
        frame = pygame.Rect(10, 10, w - 20, h - 20)
        pygame.draw.rect(surface, panel_bg, frame)
        pygame.draw.rect(surface, border, frame, 2)

        header = pygame.Rect(frame.x + 2, frame.y + 2, frame.w - 4, 42)
        pygame.draw.rect(surface, (16, 66, 52), header)
        pygame.draw.line(surface, border, (header.x, header.bottom), (header.right, header.bottom), 1)

        phase_label = {
            Phase.INSTRUCTIONS: "Instructions",
            Phase.PRACTICE: "Practice",
            Phase.PRACTICE_DONE: "Practice Complete",
            Phase.SCORED: "Timed Test",
            Phase.RESULTS: "Results",
        }.get(snap.phase, "Task")
        title = context.small_font.render(f"{snap.title} - {phase_label}", True, text_main)
        surface.blit(title, (header.x + 12, header.y + 8))

        stats = context.tiny_font.render(
            f"Windows {snap.correct_scored}/{snap.attempted_scored}",
            True,
            text_muted,
        )
        surface.blit(stats, stats.get_rect(midright=(header.right - 12, header.centery)))

        if snap.time_remaining_s is not None:
            rem = int(round(snap.time_remaining_s))
            mm = rem // 60
            ss = rem % 60
            timer = context.small_font.render(f"{mm:02d}:{ss:02d}", True, text_main)
            surface.blit(timer, timer.get_rect(topright=(frame.right - 12, header.bottom + 6)))

        meta_top = header.bottom + 8
        if payload is not None and snap.phase in (Phase.PRACTICE, Phase.SCORED):
            focus_text = context.tiny_font.render(f"Focus: {payload.focus_label}", True, text_main)
            target_label = rapid_tracking_target_label(kind=payload.target_kind, variant=payload.target_variant)
            target_text = context.tiny_font.render(f"Target: {target_label}", True, text_muted)
            segment_seconds = max(0, int(round(float(payload.segment_time_remaining_s))))
            segment_text = context.tiny_font.render(
                f"{payload.segment_label} {int(payload.segment_index)}/{int(payload.segment_total)}  {segment_seconds}s",
                True,
                text_muted,
            )
            surface.blit(focus_text, (frame.x + 12, meta_top))
            surface.blit(target_text, (frame.x + 12, meta_top + 16))
            surface.blit(segment_text, segment_text.get_rect(topright=(frame.right - 12, meta_top)))
            meta_top += 40

        dev_panel_bottom = meta_top
        if self._dev_tools_enabled(engine):
            dev_panel_bottom = self._draw_dev_panel(
                surface=surface,
                frame=frame,
                header=header,
                engine=engine,
                context=context,
                border=border,
                text_main=text_main,
                text_muted=text_muted,
            )
            meta_top = max(meta_top, dev_panel_bottom + 8)
        else:
            self._set_dev_button_hitboxes(engine, {})

        body = pygame.Rect(
            frame.x + 8,
            meta_top,
            frame.w - 16,
            frame.bottom - meta_top - 36,
        )
        track_margin = max(4, min(10, min(body.w, body.h) // 60))
        track = body.inflate(-track_margin * 2, -track_margin * 2)

        if context.app.opengl_enabled:
            pygame.draw.rect(surface, (4, 10, 8, 26), track)
        else:
            pygame.draw.rect(surface, (4, 10, 8), track)
        pygame.draw.rect(surface, (84, 122, 109), track, 1)

        if payload is not None and context.app.opengl_enabled:
            from ..gl_scenes import RapidTrackingGlScene

            context.app.queue_gl_scene(
                RapidTrackingGlScene(
                    world=pygame.Rect(track),
                    payload=payload,
                    active_phase=snap.phase in (Phase.PRACTICE, Phase.SCORED),
                )
            )
            shade = pygame.Surface(track.size, pygame.SRCALPHA)
            shade.fill((0, 0, 0, 20))
            surface.blit(shade, track.topleft)
        else:
            self._draw_fallback_world(
                surface=surface,
                track=track,
                payload=payload,
                accent=accent,
                border=border,
                text_muted=text_muted,
                context=context,
            )

        if payload is not None:
            self._draw_target_hud(
                surface=surface,
                track=track,
                payload=payload,
                accent=accent,
                warning=warning,
                text_main=text_main,
                text_muted=text_muted,
                font=context.tiny_font,
            )

        if snap.phase in (Phase.INSTRUCTIONS, Phase.PRACTICE_DONE, Phase.RESULTS):
            prompt_bg = pygame.Rect(track.x + 10, track.bottom - 96, track.w - 20, 86)
            pygame.draw.rect(surface, (14, 52, 42), prompt_bg)
            pygame.draw.rect(surface, (86, 130, 114), prompt_bg, 1)
            _render_wrapped_text(
                surface=surface,
                font=context.tiny_font,
                rect=prompt_bg.inflate(-10, -8),
                text=str(snap.prompt),
                color=text_muted,
                max_lines=6,
            )
            if payload is not None and snap.phase in (Phase.INSTRUCTIONS, Phase.PRACTICE_DONE):
                seed_text = context.tiny_font.render(
                    f"Session Seed {payload.session_seed}",
                    True,
                    text_muted,
                )
                surface.blit(seed_text, seed_text.get_rect(topright=(prompt_bg.right - 10, prompt_bg.y + 8)))

        if self._debug_overlay_enabled(engine):
            self._draw_debug_overlay(
                surface=surface,
                track=track,
                snap=snap,
                engine=engine,
                border=border,
                text_muted=text_muted,
                font=context.tiny_font,
            )

        footer = self._footer_text(snap=snap, engine=engine)
        foot = context.tiny_font.render(footer, True, text_muted)
        surface.blit(foot, foot.get_rect(midbottom=(frame.centerx, frame.bottom - 10)))

    @staticmethod
    def _footer_text(*, snap: TestSnapshot, engine: object | None) -> str:
        if snap.phase in (Phase.INSTRUCTIONS, Phase.PRACTICE_DONE):
            base = "Enter: Continue  |  Esc/Backspace: Back"
        elif snap.phase in (Phase.PRACTICE, Phase.SCORED):
            base = (
                "Configured HOTAS movement axes control the camera. "
                "Hold the configured capture binding to zoom and capture in the center box."
            )
        else:
            base = "Enter: Return to Tests"
        if RapidTrackingExerciseRenderer._dev_tools_enabled(engine):
            base = f"{base}  |  F2 Debug  F3 Camera  F5 Reset  F6 Reseed  Shift+F5 Instructions"
        return base

    @staticmethod
    def _dev_tools_enabled(engine: object | None) -> bool:
        if engine is None:
            return False
        value = getattr(engine, "dev_tools_enabled", False)
        return bool(value() if callable(value) else value)

    @staticmethod
    def _debug_overlay_enabled(engine: object | None) -> bool:
        state = getattr(engine, "debug_state", None)
        return bool(getattr(state, "overlay_enabled", False))

    @staticmethod
    def _set_dev_button_hitboxes(engine: object | None, hitboxes: dict[str, pygame.Rect]) -> None:
        setter = getattr(engine, "set_dev_button_hitboxes", None)
        if callable(setter):
            setter(hitboxes)

    def _draw_dev_panel(
        self,
        *,
        surface: pygame.Surface,
        frame: pygame.Rect,
        header: pygame.Rect,
        engine: object | None,
        context: RapidTrackingUiContext,
        border: tuple[int, int, int],
        text_main: tuple[int, int, int],
        text_muted: tuple[int, int, int],
    ) -> int:
        actions = (
            ("reset", "Reset"),
            ("reseed", "Reseed"),
            ("instructions", "Instructions"),
            ("practice", "Practice"),
            ("scored", "Scored"),
            ("debug", "Debug"),
            ("camera", "Camera"),
        )
        x = frame.right - 90
        y = header.bottom + 8
        hitboxes: dict[str, pygame.Rect] = {}
        panel = pygame.Rect(frame.right - 420, y - 2, 408, 30)
        pygame.draw.rect(surface, (10, 34, 28), panel, border_radius=8)
        pygame.draw.rect(surface, border, panel, 1, border_radius=8)
        for idx, (action, label) in enumerate(actions):
            btn = pygame.Rect(panel.x + 6 + (idx * 57), panel.y + 4, 52, 22)
            hitboxes[action] = btn
            pygame.draw.rect(surface, (20, 74, 58), btn, border_radius=6)
            pygame.draw.rect(surface, border, btn, 1, border_radius=6)
            text = context.tiny_font.render(label, True, text_main if action in {"debug", "camera"} else text_muted)
            surface.blit(text, text.get_rect(center=btn.center))
        self._set_dev_button_hitboxes(engine, hitboxes)
        return panel.bottom

    def _draw_fallback_world(
        self,
        *,
        surface: pygame.Surface,
        track: pygame.Rect,
        payload: RapidTrackingPayload | None,
        accent: tuple[int, int, int],
        border: tuple[int, int, int],
        text_muted: tuple[int, int, int],
        context: RapidTrackingUiContext,
    ) -> None:
        pygame.draw.rect(surface, (3, 12, 10), track)
        for idx in range(1, 6):
            y = track.y + int(round((track.h / 6.0) * idx))
            alpha = max(26, 70 - (idx * 8))
            band = pygame.Surface((track.w, 1), pygame.SRCALPHA)
            band.fill((84, 122, 109, alpha))
            surface.blit(band, (track.x, y))
        for idx in range(1, 5):
            x = track.x + int(round((track.w / 5.0) * idx))
            pygame.draw.line(surface, (20, 48, 40), (x, track.y), (x, track.bottom), 1)

        center = track.center
        if payload is None:
            label = context.tiny_font.render("OpenGL disabled: using schematic fallback", True, text_muted)
            surface.blit(label, label.get_rect(center=center))
            return

        # Draw a simple moving world cue so the fallback still reads in world-space.
        horizon_y = track.centery + int(round(payload.camera_y * track.h * 0.06))
        horizon_y = max(track.y + 24, min(track.bottom - 24, horizon_y))
        pygame.draw.line(surface, (34, 74, 60), (track.x + 10, horizon_y), (track.right - 10, horizon_y), 2)

        for idx in range(-2, 3):
            base_x = track.centerx + int(round(idx * track.w * 0.18))
            pygame.draw.line(surface, (26, 60, 48), (base_x, horizon_y), (base_x + (idx * 8), track.bottom - 10), 1)

        box_half_w = max(18, int(round(((track.w // 2) - 12) * payload.capture_box_half_width)))
        box_half_h = max(14, int(round((((track.h // 2) - 12) * 0.9) * payload.capture_box_half_height)))
        capture_box = pygame.Rect(0, 0, box_half_w * 2, box_half_h * 2)
        capture_box.center = center
        pygame.draw.rect(surface, (92, 168, 144), capture_box, 1, border_radius=6)

        reticle = self._track_point(track, payload.reticle_x, payload.reticle_y)
        target = self._track_point(track, payload.target_rel_x, payload.target_rel_y)
        cross = 10
        pygame.draw.line(surface, accent, (reticle[0] - cross, reticle[1]), (reticle[0] + cross, reticle[1]), 2)
        pygame.draw.line(surface, accent, (reticle[0], reticle[1] - cross), (reticle[0], reticle[1] + cross), 2)

        target_color = (236, 214, 114) if payload.target_visible else (126, 126, 126)
        pygame.draw.circle(surface, target_color, target, 8 if payload.target_is_moving else 10, 2)
        if payload.target_is_moving:
            lead = (
                target[0] + int(round(payload.target_vx * 12.0)),
                target[1] - int(round(payload.target_vy * 12.0)),
            )
            pygame.draw.line(surface, target_color, target, lead, 2)
        pygame.draw.circle(surface, border, target, 2)

    @staticmethod
    def _track_point(track: pygame.Rect, rel_x: float, rel_y: float) -> tuple[int, int]:
        center_x = track.centerx
        center_y = track.centery
        x_scale = (track.w / 2.0) - 12.0
        y_scale = ((track.h / 2.0) - 12.0) * 0.90
        limit = max(0.1, float(TARGET_VIEW_LIMIT))
        screen_x = center_x + int(round((float(rel_x) / limit) * x_scale))
        screen_y = center_y + int(round((float(rel_y) / limit) * y_scale))
        return (
            max(track.left + 8, min(track.right - 8, screen_x)),
            max(track.top + 8, min(track.bottom - 8, screen_y)),
        )

    def _draw_target_hud(
        self,
        *,
        surface: pygame.Surface,
        track: pygame.Rect,
        payload: RapidTrackingPayload,
        accent: tuple[int, int, int],
        warning: tuple[int, int, int],
        text_main: tuple[int, int, int],
        text_muted: tuple[int, int, int],
        font: pygame.font.Font,
    ) -> None:
        metrics_bg = pygame.Rect(track.x + 10, track.y + 10, track.w - 20, 24)
        pygame.draw.rect(surface, (12, 34, 28), metrics_bg, border_radius=6)
        pygame.draw.rect(surface, (86, 130, 114), metrics_bg, 1, border_radius=6)
        left = font.render(
            f"Err {payload.mean_error:.3f}  RMS {payload.rms_error:.3f}  On {payload.on_target_ratio * 100.0:.1f}%",
            True,
            text_main,
        )
        right = font.render(
            f"Capture {payload.capture_hits}/{payload.capture_attempts}  Score {payload.capture_points}  Zoom {payload.capture_zoom:.2f}",
            True,
            text_muted,
        )
        surface.blit(left, (metrics_bg.x + 8, metrics_bg.y + 4))
        surface.blit(right, right.get_rect(midright=(metrics_bg.right - 8, metrics_bg.centery)))

        status_color = accent if payload.target_visible else warning
        status_label = (
            f"{rapid_tracking_target_label(kind=payload.target_kind, variant=payload.target_variant)}"
            f"  |  {payload.target_cover_state.replace('_', ' ').title()}"
        )
        status = font.render(status_label, True, status_color)
        surface.blit(status, (track.x + 10, metrics_bg.bottom + 8))

    def _draw_debug_overlay(
        self,
        *,
        surface: pygame.Surface,
        track: pygame.Rect,
        snap: TestSnapshot,
        engine: object | None,
        border: tuple[int, int, int],
        text_muted: tuple[int, int, int],
        font: pygame.font.Font,
    ) -> None:
        state = getattr(engine, "debug_state", None)
        if state is None:
            return
        lines = rapid_tracking_debug_lines(snap=snap, state=state)
        if not lines:
            return
        line_h = font.get_linesize() + 2
        panel_h = min(track.h - 20, 12 + (line_h * len(lines)))
        panel = pygame.Rect(track.x + 10, track.bottom - panel_h - 10, min(track.w - 20, 440), panel_h)
        overlay = pygame.Surface(panel.size, pygame.SRCALPHA)
        overlay.fill((4, 10, 8, 212))
        surface.blit(overlay, panel.topleft)
        pygame.draw.rect(surface, border, panel, 1, border_radius=8)
        y = panel.y + 8
        for line in lines:
            if y + line_h > panel.bottom - 6:
                break
            surface.blit(font.render(line, True, text_muted), (panel.x + 8, y))
            y += line_h


def render_rapid_tracking_screen(
    *,
    surface: pygame.Surface,
    snap: TestSnapshot,
    payload: RapidTrackingPayload | None,
    engine: object | None,
    context: RapidTrackingUiContext,
) -> None:
    RapidTrackingExerciseRenderer().render(
        surface=surface,
        snap=snap,
        payload=payload,
        engine=engine,
        context=context,
    )


__all__ = [
    "RapidTrackingUiContext",
    "RapidTrackingExerciseRenderer",
    "render_rapid_tracking_screen",
]

# Screenshot Convention

Screenshots are especially useful in this repo because many regressions are visual, timing-sensitive, or layout-specific.

## What To Capture

- Full-window shots of menus, pause overlays, loading screens, and results screens.
- UI-heavy subsystem states:
  - Target Recognition multi-panel screens
  - Visual Search dense boards
  - Situational Awareness grid plus cue card plus active query
  - Benchmark and adaptive intro/result transitions
  - HOTAS calibration, profile, and bindings screens
- Any renderer failure or startup diagnostic screen.

## Naming

Use a filename that makes sorting and comparison easy:

```text
YYYY-MM-DD_subsystem_screen_platform_branch-or-sha_state.png
```

Examples:

```text
2026-04-08_target-recognition_practice_macos_main_panel-overlap.png
2026-04-08_app-shell_pause-menu_windows_feature-trace_loading-lock.png
2026-04-08_visual-search_scored_macos_8f4dea8_dense-board.png
```

## Capture Guidelines

- Prefer full-window captures over tight crops.
- If a crop is needed, keep one full-window shot as well.
- Include the platform, requested renderer mode, and any failure-screen diagnostic code in the issue or PR text when relevant.
- Mention the input hardware used for reproduction when the bug is input-sensitive.
- If the problem is motion or timing related, attach a short video or a before/after pair.

## Best Companion Notes

Include these alongside the screenshot:

- exact branch and commit tested
- subsystem or screen name
- platform (`macOS`, `Windows`, or both)
- requested renderer mode (`ModernGL`, built-in pygame fallback, or default app setting)
- whether a renderer failure screen appeared, plus any diagnostic code shown
- window mode (`windowed`, `fullscreen`, or `borderless`)
- whether the bug reproduces with keyboard/mouse only or also with joystick/rudder hardware

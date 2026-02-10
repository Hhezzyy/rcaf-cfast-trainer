import os


def test_menu_boots_and_quits_headless() -> None:
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    import pygame

    from cfast_trainer.app import run

    def injector(frame: int) -> None:
        if frame == 1:
            pygame.event.post(pygame.event.Event(pygame.QUIT))

    code = run(max_frames=10, event_injector=injector)
    assert code == 0
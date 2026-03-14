from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "assets" / "audio" / "auditory_capacity" / "source"
TARGET_DIR = ROOT / "assets" / "audio" / "auditory_capacity"

SOURCE_TO_TARGET = {
    "Backgroud noise 1.m4a": "background_noise_1.wav",
    "Background noise 2.m4a": "background_noise_2.wav",
    "Background noise 3.m4a": "background_noise_3.wav",
    "Background noise 4.m4a": "background_noise_4.wav",
    "Background noise 5.m4a": "background_noise_5.wav",
    "Background noise 6.m4a": "background_noise_6.wav",
    "Mature 1.m4a": "mature_1.wav",
    "Mature 2.m4a": "mature_2.wav",
    "Mature distraction 1.m4a": "mature_distraction_1.wav",
    "Mature distraction 2.m4a": "mature_distraction_2.wav",
    "Mature distraction 3.m4a": "mature_distraction_3.wav",
    "Mature distraction 4.m4a": "mature_distraction_4.wav",
    "Mature distraction 5.m4a": "mature_distraction_5.wav",
    "Mature distraction 6.m4a": "mature_distraction_6.wav",
    "Mature distraction 7.m4a": "mature_distraction_7.wav",
    "Mature distraction 8.m4a": "mature_distraction_8.wav",
    "Mature distraction 9.m4a": "mature_distraction_9.wav",
}


def main() -> None:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    converted = 0
    for source_name, target_name in SOURCE_TO_TARGET.items():
        source_path = SOURCE_DIR / source_name
        target_path = TARGET_DIR / target_name
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(source_path),
                "-ac",
                "1",
                "-ar",
                "22050",
                "-sample_fmt",
                "s16",
                str(target_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        converted += 1
        print(f"converted {source_name} -> {target_name}")
    print(f"done: {converted} files")


if __name__ == "__main__":
    main()

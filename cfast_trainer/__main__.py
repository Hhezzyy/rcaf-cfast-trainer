from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import sys
from pathlib import Path


def _ensure_repo_root_on_path() -> None:
    """Ensure the repository root (parent of this package) is on ``sys.path``.

    If this module is executed as a script (``python cfast_trainer/__main__.py``),
    the package may not be discoverable by Python. This helper inserts the
    parent directory of the package into ``sys.path`` so that imports resolve
    correctly.
    """
    # If executed as a script: .../cfast_trainer/__main__.py
    pkg_dir = Path(__file__).resolve().parent
    repo_root = pkg_dir.parent
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


try:
    # Works when executed as a module: python -m cfast_trainer
    from .app import run, run_headless_sim  # type: ignore[attr-defined]
except ImportError:
    # Works when executed as a script (VS Code “Run Python File”, absolute path, etc.)
    _ensure_repo_root_on_path()
    from cfast_trainer.app import run, run_headless_sim  # type: ignore[attr-defined]


def main(argv: list[str] | None = None) -> int:
    """Entry point for running the trainer from the command line."""
    parser = argparse.ArgumentParser(prog="python -m cfast_trainer")
    parser.add_argument("--headless-sim", dest="headless_sim", type=str)
    parser.add_argument("--max-frames", dest="max_frames", type=int, default=None)
    args = parser.parse_args(argv)
    if args.headless_sim:
        result = run_headless_sim(args.headless_sim, max_frames=args.max_frames)
        print(json.dumps(asdict(result), sort_keys=True))
        return int(result.exit_code)
    return run()


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

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
    from .app import run  # type: ignore[attr-defined]
except ImportError:
    # Works when executed as a script (VS Code “Run Python File”, absolute path, etc.)
    _ensure_repo_root_on_path()
    from cfast_trainer.app import run  # type: ignore[attr-defined]


def main() -> int:
    """Entry point for running the trainer from the command line."""
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
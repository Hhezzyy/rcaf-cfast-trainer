from __future__ import annotations

import sys
from pathlib import Path


def _ensure_repo_root_on_path() -> None:
    # If executed as a script: .../cfast_trainer/__main__.py
    # ensure the repo root (parent of package dir) is on sys.path.
    pkg_dir = Path(__file__).resolve().parent
    repo_root = pkg_dir.parent
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


try:
    # Works when executed as a module: python -m cfast_trainer
    from .app import run
except ImportError:
    # Works when executed as a script (VS Code “Run Python File”, absolute path, etc.)
    _ensure_repo_root_on_path()
    from cfast_trainer.app import run


def main() -> int:
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
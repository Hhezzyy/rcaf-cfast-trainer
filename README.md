# RCAF CFAST Trainer

Offline‑first training app for CFASC CFAST‑style aircrew selection aptitude domains.

## Requirements

- CPython 3.11+ (3.13 tested)
- VS Code with the Python and Pylance extensions
- Git (recommended for moving between machines)

## Setup (Windows 11)

1. Install CPython 3.13.12 (x64) from python.org.
2. Create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install dependencies:

   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   python -m pip install -r requirements-dev.txt
   ```

4. In VS Code, select the `.venv` interpreter.

## Setup (macOS Monterey, Intel)

1. Install CPython 3.13.12 from python.org.
2. Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   python -m pip install -r requirements-dev.txt
   ```

4. In VS Code, select the `.venv` interpreter.

## VS Code

- Recommended extensions: `.vscode/extensions.json`
- Python test settings: `.vscode/settings.json`

## Verify

- `python --version` should report 3.11+.
- `python -m pytest` runs the smoke test.
- `python -m cfast_trainer` opens the app (main menu).

## Share / Move Between PC and Mac

- Do not copy `.venv/` between machines.
- Preferred: push to a git remote and clone on the other machine.
- Alternative zip (tracked files only):

  ```bash
  git archive --format zip -o rcaf-cfast-trainer.zip HEAD
  ```

- For ChatGPT projects/chats: upload the zip produced by `git archive`.

After moving, recreate the virtual environment and reinstall dependencies on the target machine.
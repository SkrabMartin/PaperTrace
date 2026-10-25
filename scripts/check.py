"""Run every quality gate locally, in the same order as CI."""

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PYTHON_GATES: list[list[str]] = [
    ["uv", "run", "ruff", "check", "."],
    ["uv", "run", "ruff", "format", "--check", "."],
    ["uv", "run", "mypy"],
    ["uv", "run", "pytest"],
    ["uv", "run", "python", "scripts/check_markdown.py"],
]
FRONTEND_GATES: list[list[str]] = [
    ["npm", "run", "lint"],
    ["npm", "run", "format:check"],
    ["npm", "run", "typecheck"],
    ["npm", "test"],
    ["npm", "run", "build"],
]


def run(command: list[str], cwd: Path) -> bool:
    """Run one gate and report whether it passed."""
    executable = shutil.which(command[0])
    if executable is None:
        print(f"missing tool: {command[0]}")  # noqa: T201
        return False
    print(f"\n$ {' '.join(command)}  (in {cwd.name})")  # noqa: T201
    return subprocess.run([executable, *command[1:]], cwd=cwd, check=False).returncode == 0


def main() -> int:
    """Run all gates; return 1 if any failed."""
    results = [run(cmd, ROOT) for cmd in PYTHON_GATES]
    results += [run(cmd, ROOT / "frontend") for cmd in FRONTEND_GATES]
    print("\nAll gates passed." if all(results) else "\nSome gates FAILED.")  # noqa: T201
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())

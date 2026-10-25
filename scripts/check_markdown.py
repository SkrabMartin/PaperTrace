"""Fail if any markdown file other than README.md is tracked by git."""

import subprocess
import sys
from pathlib import PurePosixPath

ALLOWED = {"README.md"}


def main() -> int:
    """Return a non-zero exit code when disallowed markdown files are tracked."""
    out = subprocess.run(
        ["git", "ls-files", "-z"], check=True, capture_output=True, text=True
    ).stdout
    bad = [
        name
        for name in out.split("\0")
        if name and PurePosixPath(name).suffix.lower() == ".md" and name not in ALLOWED
    ]
    for name in bad:
        print(f"markdown file not allowed in the repository: {name}", file=sys.stderr)  # noqa: T201
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

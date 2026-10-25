"""Layer boundaries: imports between papertrace subpackages may only point downwards."""

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src" / "papertrace"

ALLOWED: dict[str, set[str]] = {
    "core": set(),
    "data": set(),
    "synth": {"data"},
    "eval": {"core", "data", "synth"},
    "export": {"core"},
    "train": {"core", "data", "synth", "eval"},
    "api": {"core", "export", "eval"},
}
SHARED = {"config"}


def _imported_subpackages(path: Path) -> set[str]:
    package = ["papertrace", *path.relative_to(SRC).parent.parts]
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = ".".join(package[: len(package) - node.level + 1]) if node.level else ""
            module = ".".join(part for part in (base, node.module) if part)
            names = [module] if node.module else [f"{module}.{alias.name}" for alias in node.names]
        for name in names:
            parts = name.split(".")
            if parts[0] == "papertrace" and len(parts) > 1:
                found.add(parts[1])
    return found


def test_layers_only_import_downwards() -> None:
    violations: list[str] = []
    for layer, allowed in ALLOWED.items():
        for path in (SRC / layer).rglob("*.py"):
            for target in _imported_subpackages(path) - {layer} - allowed - SHARED:
                violations.append(f"{path.relative_to(SRC)} imports papertrace.{target}")
    assert not violations, violations


def test_every_subpackage_is_covered() -> None:
    subpackages = {p.name for p in SRC.iterdir() if p.is_dir() and (p / "__init__.py").exists()}
    assert subpackages == set(ALLOWED)

import pytest
from typer.testing import CliRunner

from papertrace.cli import app

runner = CliRunner()
COMMANDS = ["digitize", "manifest", "synth", "train", "evaluate", "serve"]
IMPLEMENTED = {"manifest", "synth", "evaluate"}
PLACEHOLDERS = [command for command in COMMANDS if command not in IMPLEMENTED]


def test_help_lists_all_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in COMMANDS:
        assert command in result.output


@pytest.mark.parametrize("command", PLACEHOLDERS)
def test_placeholder_commands_exit_with_code_2(command: str) -> None:
    result = runner.invoke(app, [command])
    assert result.exit_code == 2
    assert "not implemented yet" in result.output

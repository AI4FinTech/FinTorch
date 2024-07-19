from click.testing import CliRunner

from fintorch import cli
from fintorch.datasets import elliptic, ellipticpp


def test_command_line_interface():
    runner = CliRunner()
    result = runner.invoke(cli.fintorch)
    assert result.exit_code == 0
    assert "FinTorch CLI - Your financial AI toolkit" in result.output
    help_result = runner.invoke(cli.fintorch, ["--help"])
    assert help_result.exit_code == 0
    assert "--help  Show this message and exit." in help_result.output


def test_command_line_interface_elliptic_dataset():
    runner = CliRunner()
    result = runner.invoke(cli.fintorch, ["datasets", "elliptic"])
    assert result.exit_code == 0
    assert "Downloading dataset: elliptic" in result.output
    root = "/tmp/data/fintorch/"
    assert isinstance(
        elliptic.TransactionDataset(root, force_reload=True),
        elliptic.TransactionDataset,
    )


def test_command_line_interface_ellipticpp_dataset():
    runner = CliRunner()
    result = runner.invoke(cli.fintorch, ["datasets", "ellipticpp"])
    assert result.exit_code == 0
    assert "Downloading dataset: ellipticpp" in result.output
    root = "/tmp/data/fintorch/"
    assert isinstance(
        ellipticpp.TransactionActorDataset(root, force_reload=True),
        ellipticpp.TransactionActorDataset,
    )


def test_command_line_interface_train():
    runner = CliRunner()
    result = runner.invoke(
        cli.fintorch, ["sweep", "--model", "model_name", "--predict", "predict"]
    )
    assert "Starting sweep for model" in result.output

#!/usr/bin/env python
"""Tests for `fintorch` package."""

from click.testing import CliRunner

from fintorch import cli


def test_command_line_interface():
    """Test the CLI."""
    runner = CliRunner()
    result = runner.invoke(cli.fintorch)
    assert result.exit_code == 0
    assert "FinTorch CLI - Your financial AI toolkit" in result.output
    help_result = runner.invoke(cli.fintorch, ["--help"])
    assert help_result.exit_code == 0
    assert "--help  Show this message and exit." in help_result.output


def test_command_line_interface_elliptic_dataset():
    """Test the CLI."""
    runner = CliRunner()
    result = runner.invoke(cli.fintorch, ["datasets", "elliptic"])
    assert result.exit_code == 0
    assert "Downloading dataset: elliptic" in result.output
    help_result = runner.invoke(cli.fintorch, ["--help"])
    assert help_result.exit_code == 0
    assert "--help  Show this message and exit." in help_result.output

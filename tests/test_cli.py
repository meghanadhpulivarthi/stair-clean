import subprocess
import sys


def run_stair(*args):
    command = [sys.executable, "-m", "stair.cli"] + list(args)
    result = subprocess.run(command, capture_output=True, text=True)
    return result


def test_no_subcommand_prints_usage_and_exits_nonzero():
    result = run_stair()
    assert result.returncode != 0
    assert "usage" in result.stderr.lower() or "usage" in result.stdout.lower()


def test_help_lists_all_subcommands():
    result = run_stair("--help")
    assert result.returncode == 0
    assert "prepare-data" in result.stdout
    assert "train" in result.stdout
    assert "report-train" in result.stdout
    assert "eval" in result.stdout
    assert "report-eval" in result.stdout


def test_prepare_data_subcommand_help_runs():
    result = run_stair("prepare-data", "--help")
    assert result.returncode == 0

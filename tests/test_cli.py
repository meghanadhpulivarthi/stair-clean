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


def test_prepare_data_with_bad_override_key_fails_loudly(tmp_path):
    override_path = tmp_path / "bad_override.yaml"
    override_path.write_text("model:\n  lroa_rank: 64\n")
    corpus_path = tmp_path / "corpus.pdf"
    corpus_path.write_text("not a real pdf, just needs to exist for this test")
    out_dir = tmp_path / "out"

    result = run_stair(
        "prepare-data",
        "--corpus", str(corpus_path),
        "--out", str(out_dir),
        "--config", str(override_path),
    )

    assert result.returncode != 0
    assert "lroa_rank" in result.stderr

import json
import os
import subprocess
import sys


def run_stair(*args, env=None):
    command = [sys.executable, "-m", "stair.cli"] + list(args)
    result = subprocess.run(command, capture_output=True, text=True, env=env)
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


def test_prepare_data_with_malformed_yaml_override_fails_loudly(tmp_path):
    override_path = tmp_path / "malformed.yaml"
    override_path.write_text("model:\n  lora:\n  rank: [unclosed\n")
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
    assert "Traceback" not in result.stderr


def test_prepare_data_with_unparseable_corpus_fails_loudly_not_with_a_traceback(tmp_path):
    # pypdfium2 raises PdfiumError (a RuntimeError) for a corpus that isn't
    # a real PDF at all — this is outside the (OSError, ValueError) that
    # config resolution errors use, and represents the broader class of
    # non-OSError/ValueError failures (e.g. from a real LLM/openai client)
    # that the CLI must still turn into a clean one-line stderr message
    corpus_path = tmp_path / "corpus.pdf"
    corpus_path.write_text("not a real pdf, just needs to exist for this test")
    out_dir = tmp_path / "out"

    # provide fake LLM env vars so the run gets past the fail-fast env-var
    # check and actually reaches PDF parsing, where PdfiumError is raised
    fake_llm_env = dict(os.environ)
    fake_llm_env["STAIR_LLM_API_BASE"] = "http://localhost:0"
    fake_llm_env["STAIR_LLM_API_KEY"] = "fake-key"
    fake_llm_env["STAIR_LLM_MODEL"] = "fake-model"

    result = run_stair(
        "prepare-data",
        "--corpus", str(corpus_path),
        "--out", str(out_dir),
        env=fake_llm_env,
    )

    assert result.returncode != 0
    assert "Traceback" not in result.stderr
    assert "stair prepare-data:" in result.stderr


def test_prepare_data_with_config_pointing_at_a_directory_fails_loudly(tmp_path):
    corpus_path = tmp_path / "corpus.pdf"
    corpus_path.write_text("not a real pdf, just needs to exist for this test")
    out_dir = tmp_path / "out"
    directory_as_config = tmp_path / "not_a_file"
    directory_as_config.mkdir()

    result = run_stair(
        "prepare-data",
        "--corpus", str(corpus_path),
        "--out", str(out_dir),
        "--config", str(directory_as_config),
    )

    assert result.returncode != 0
    assert "Traceback" not in result.stderr


def test_eval_with_run_config_missing_model_name_fails_loudly_not_with_a_traceback(tmp_path):
    # run_dir's own config.json (written by `stair train`) is missing the
    # model.name key that build_default_generate_call reads to load the
    # base model. That raises a KeyError, which is outside the
    # (OSError, ValueError) that config-resolution errors use - this
    # exercises the same broadened-exception-handling gap already fixed for
    # stair prepare-data, now fixed for stair eval too, without needing a
    # real model download or GPU
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    toc = {
        "title": "Test Book",
        "table_of_contents": [
            {"id": "1", "section_num": "1", "title": "Introduction", "leaf": True},
        ],
    }
    (data_dir / "toc.json").write_text(json.dumps(toc))
    test_records = [{"question": "What is this about?", "answer": "testing", "reference": ["1"]}]
    with open(data_dir / "test.jsonl", "w") as test_file:
        for record in test_records:
            test_file.write(json.dumps(record) + "\n")

    run_dir = tmp_path / "run"
    (run_dir / "checkpoint-best").mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps({"model": {}}))

    result = run_stair("eval", "--run", str(run_dir), "--data", str(data_dir))

    assert result.returncode != 0
    assert "Traceback" not in result.stderr
    assert "stair eval:" in result.stderr

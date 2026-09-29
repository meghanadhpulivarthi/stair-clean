import json
from pathlib import Path

import pytest

from stair.config import resolve_config
from stair.eval import run_eval


def make_complete_data_dir(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    toc = {
        "title": "Test Book",
        "table_of_contents": [
            {"id": "1", "section_num": "1", "title": "Introduction", "leaf": True},
            {"id": "2", "section_num": "2", "title": "Getting Started", "leaf": True},
        ],
    }
    (data_dir / "toc.json").write_text(json.dumps(toc))
    test_records = [
        {"question": "What is this book about?", "answer": "It is about testing.", "reference": ["1"]},
        {"question": "How do I get started?", "answer": "Read chapter two.", "reference": ["2"]},
    ]
    with open(data_dir / "test.jsonl", "w") as test_file:
        for record in test_records:
            test_file.write(json.dumps(record) + "\n")
    return data_dir


def make_complete_run_dir(tmp_path, config):
    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / "checkpoint-best"
    checkpoint_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps(config))
    return run_dir


def make_fake_generate_call(responses_by_question):
    def fake_generate_call(messages):
        user_message_content = messages[1]["content"]
        for question, response in responses_by_question.items():
            if question in user_message_content:
                return response
        return "[]"

    return fake_generate_call


def test_run_eval_raises_when_test_jsonl_missing(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "toc.json").write_text("{}")
    config = resolve_config(override_path=None)
    run_dir = make_complete_run_dir(tmp_path, config)

    with pytest.raises(FileNotFoundError, match="test.jsonl"):
        run_eval(str(run_dir), str(data_dir), config, generate_call=make_fake_generate_call({}))


def test_run_eval_raises_when_checkpoint_best_missing(tmp_path):
    data_dir = make_complete_data_dir(tmp_path)
    config = resolve_config(override_path=None)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(json.dumps(config))

    with pytest.raises(FileNotFoundError, match="checkpoint-best"):
        run_eval(str(run_dir), str(data_dir), config, generate_call=make_fake_generate_call({}))


def test_run_eval_computes_perfect_metrics_when_predictions_match(tmp_path):
    data_dir = make_complete_data_dir(tmp_path)
    config = resolve_config(override_path=None)
    run_dir = make_complete_run_dir(tmp_path, config)
    fake_generate_call = make_fake_generate_call(
        {
            "What is this book about?": '["1 Introduction"]',
            "How do I get started?": '["2 Getting Started"]',
        }
    )

    result = run_eval(str(run_dir), str(data_dir), config, generate_call=fake_generate_call)

    assert result["count"] == 2
    assert result["hallucination_rate"] == 0.0
    top_k = config["eval"]["ks"][0]
    assert result["metrics"][f"recall@{top_k}"] == 1.0


def test_run_eval_writes_results_and_metrics_files(tmp_path):
    data_dir = make_complete_data_dir(tmp_path)
    config = resolve_config(override_path=None)
    run_dir = make_complete_run_dir(tmp_path, config)
    fake_generate_call = make_fake_generate_call({})

    run_eval(str(run_dir), str(data_dir), config, generate_call=fake_generate_call)

    assert (run_dir / "eval" / "eval_results.jsonl").exists()
    assert (run_dir / "eval" / "eval_metrics.json").exists()
    result_lines = (run_dir / "eval" / "eval_results.jsonl").read_text().strip().split("\n")
    assert len(result_lines) == 2


def test_run_eval_tracks_hallucination_rate_for_unparseable_generations(tmp_path):
    data_dir = make_complete_data_dir(tmp_path)
    config = resolve_config(override_path=None)
    run_dir = make_complete_run_dir(tmp_path, config)
    fake_generate_call = make_fake_generate_call(
        {
            "What is this book about?": "not valid python at all",
            "How do I get started?": '["2 Getting Started"]',
        }
    )

    result = run_eval(str(run_dir), str(data_dir), config, generate_call=fake_generate_call)

    assert result["hallucination_rate"] > 0.0

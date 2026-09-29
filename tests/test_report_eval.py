import json

import pytest

from stair.core.report.eval_report import flag_weak_spots, load_eval_metrics, render_metrics_table


EVAL_CONFIG = {
    "weak_recall_threshold": 0.5,
    "high_hallucination_rate_threshold": 0.2,
}


def write_eval_metrics(path, summary):
    path.write_text(json.dumps(summary))


def test_load_eval_metrics_reads_the_file(tmp_path):
    run_dir = tmp_path / "run"
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True)
    summary = {"count": 2, "hallucination_rate": 0.0, "metrics": {"recall@1": 1.0}}
    write_eval_metrics(eval_dir / "eval_metrics.json", summary)

    loaded_summary = load_eval_metrics(str(run_dir))

    assert loaded_summary == summary


def test_load_eval_metrics_raises_clear_error_when_missing(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="eval_metrics.json"):
        load_eval_metrics(str(run_dir))


def test_render_metrics_table_does_not_raise(capsys):
    summary = {
        "count": 10,
        "hallucination_rate": 0.1,
        "metrics": {"recall@1": 0.8, "recall@3": 0.9, "mrr@1": 0.7, "mrr@3": 0.75},
    }
    render_metrics_table(summary, ks=[1, 3])

    captured_output = capsys.readouterr()
    assert "recall" in captured_output.out.lower()


def test_flag_weak_spots_flags_low_recall():
    summary = {"count": 10, "hallucination_rate": 0.0, "metrics": {"recall@1": 0.2}}

    warnings = flag_weak_spots(summary, EVAL_CONFIG)

    joined_warnings = " ".join(warnings).lower()
    assert "recall" in joined_warnings


def test_flag_weak_spots_flags_high_hallucination_rate():
    summary = {"count": 10, "hallucination_rate": 0.5, "metrics": {}}

    warnings = flag_weak_spots(summary, EVAL_CONFIG)

    joined_warnings = " ".join(warnings).lower()
    assert "hallucination" in joined_warnings


def test_flag_weak_spots_returns_empty_list_when_healthy():
    summary = {"count": 10, "hallucination_rate": 0.0, "metrics": {"recall@1": 0.9}}

    warnings = flag_weak_spots(summary, EVAL_CONFIG)

    assert warnings == []

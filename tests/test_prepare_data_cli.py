import json
from pathlib import Path

import pytest

from pdf_fixtures import make_tiny_bookmarked_pdf

from stair.config import resolve_config
from stair.prepare_data import derive_book_title, run_prepare_data


EXAMPLE_CORPUS_PATH = Path(__file__).resolve().parents[1] / "data" / "example_book" / "sourdough_bread_guide.pdf"


def test_derive_book_title_turns_underscores_and_hyphens_into_a_readable_title():
    assert derive_book_title("sourdough_bread_guide.pdf") == "Sourdough Bread Guide"
    assert derive_book_title("some-manual-name.pdf") == "Some Manual Name"


def test_run_prepare_data_book_title_is_readable_not_a_raw_filename_stem(tmp_path):
    pdf_path = tmp_path / "tiny_test_book.pdf"
    make_tiny_bookmarked_pdf(pdf_path)
    out_dir = tmp_path / "out"
    config = resolve_config(override_path=None)

    run_prepare_data(pdf_path, out_dir, config, llm_call=fake_llm_call)

    toc = json.loads((out_dir / "toc.json").read_text())
    assert toc["title"] == "Tiny Test Book"


def fake_llm_call(messages):
    return '[{"question": "What is this chapter about?", "answer": "It is a short example chapter."}]'


def test_run_prepare_data_writes_all_expected_files(tmp_path):
    pdf_path = tmp_path / "tiny.pdf"
    make_tiny_bookmarked_pdf(pdf_path)
    out_dir = tmp_path / "out"

    config = resolve_config(override_path=None)
    config["data"]["train_fraction"] = 0.0
    config["data"]["val_fraction"] = 0.0
    config["data"]["test_fraction"] = 1.0

    counts = run_prepare_data(pdf_path, out_dir, config, llm_call=fake_llm_call)

    assert (out_dir / "docs.jsonl").exists()
    assert (out_dir / "toc.json").exists()
    assert (out_dir / "train.jsonl").exists()
    assert (out_dir / "val.jsonl").exists()
    assert (out_dir / "test.jsonl").exists()

    assert counts["sections"] == 2
    assert counts["qa_pairs"] == 2
    assert counts["test"] == 2
    assert counts["train"] == 0
    assert counts["val"] == 0


def test_run_prepare_data_toc_matches_extracted_sections(tmp_path):
    pdf_path = tmp_path / "tiny.pdf"
    make_tiny_bookmarked_pdf(pdf_path)
    out_dir = tmp_path / "out"
    config = resolve_config(override_path=None)

    run_prepare_data(pdf_path, out_dir, config, llm_call=fake_llm_call)

    toc = json.loads((out_dir / "toc.json").read_text())
    section_titles = [entry["title"] for entry in toc["table_of_contents"]]
    assert section_titles == ["Introduction", "Getting Started"]


def test_run_prepare_data_test_split_references_are_valid_toc_ids(tmp_path):
    pdf_path = tmp_path / "tiny.pdf"
    make_tiny_bookmarked_pdf(pdf_path)
    out_dir = tmp_path / "out"

    config = resolve_config(override_path=None)
    config["data"]["train_fraction"] = 0.0
    config["data"]["val_fraction"] = 0.0
    config["data"]["test_fraction"] = 1.0

    run_prepare_data(pdf_path, out_dir, config, llm_call=fake_llm_call)

    toc = json.loads((out_dir / "toc.json").read_text())
    valid_ids = set(entry["id"] for entry in toc["table_of_contents"])

    test_lines = (out_dir / "test.jsonl").read_text().strip().split("\n")
    for line in test_lines:
        record = json.loads(line)
        for reference_id in record["reference"]:
            assert reference_id in valid_ids


def test_bundled_example_corpus_produces_a_realistic_split(tmp_path):
    out_dir = tmp_path / "out"
    config = resolve_config(override_path=None)

    counts = run_prepare_data(EXAMPLE_CORPUS_PATH, out_dir, config, llm_call=fake_llm_call)

    assert counts["sections"] >= 4
    assert counts["qa_pairs"] > 0
    assert counts["train"] + counts["val"] + counts["test"] == counts["qa_pairs"]


def test_run_prepare_data_writes_config_json(tmp_path):
    pdf_path = tmp_path / "tiny.pdf"
    make_tiny_bookmarked_pdf(pdf_path)
    out_dir = tmp_path / "out"
    config = resolve_config(override_path=None)

    run_prepare_data(pdf_path, out_dir, config, llm_call=fake_llm_call)

    assert (out_dir / "config.json").exists()
    saved_config = json.loads((out_dir / "config.json").read_text())
    assert saved_config == config


def test_run_prepare_data_fails_fast_before_writing_any_files_when_llm_env_vars_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("STAIR_LLM_API_BASE", raising=False)
    monkeypatch.delenv("STAIR_LLM_API_KEY", raising=False)
    monkeypatch.delenv("STAIR_LLM_MODEL", raising=False)

    pdf_path = tmp_path / "tiny.pdf"
    make_tiny_bookmarked_pdf(pdf_path)
    out_dir = tmp_path / "out"
    config = resolve_config(override_path=None)

    with pytest.raises(ValueError):
        run_prepare_data(pdf_path, out_dir, config)

    # the ValueError must fire before docs.jsonl/toc.json/config.json are
    # written, not after a partial run has already produced output
    assert not (out_dir / "docs.jsonl").exists()
    assert not (out_dir / "toc.json").exists()
    assert not (out_dir / "config.json").exists()

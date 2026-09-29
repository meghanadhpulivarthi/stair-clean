import json
import os
from pathlib import Path

from stair.core.data.chunking import chunk_sections
from stair.core.data.pdf_extract import extract_sections_from_pdf
from stair.core.data.qa_generation import build_openai_compatible_llm_call, generate_qa_pairs_for_chunk
from stair.core.data.split import split_qa_pairs
from stair.core.data.toc import build_toc
from stair.run_header import print_run_header


EMPTY_SKIP_CONFIG = {
    "skip_sections": [],
    "skip_sections_startswith": [],
    "skip_sections_endswith": [],
    "skip_sentences": [],
    "skip_sentence_contains": [],
}


def write_jsonl(path, records):
    with open(path, "w") as output_file:
        for record in records:
            output_file.write(json.dumps(record) + "\n")


def derive_book_title(corpus_path):
    # turn a filename stem like "sourdough_bread_guide" into a readable
    # title like "Sourdough Bread Guide" for use in QA-generation prompts
    stem = Path(corpus_path).stem
    readable_stem = stem.replace("_", " ").replace("-", " ")
    words = readable_stem.split()

    capitalized_words = []
    for word in words:
        capitalized_words.append(word.capitalize())

    return " ".join(capitalized_words)


def build_default_llm_call(config):
    api_base = os.environ.get("STAIR_LLM_API_BASE")
    api_key = os.environ.get("STAIR_LLM_API_KEY")
    model_name = os.environ.get("STAIR_LLM_MODEL")
    if not api_base or not api_key or not model_name:
        raise ValueError(
            "QA generation needs an OpenAI-compatible endpoint. Set the "
            "STAIR_LLM_API_BASE, STAIR_LLM_API_KEY, and STAIR_LLM_MODEL "
            "environment variables."
        )
    return build_openai_compatible_llm_call(api_base, api_key, model_name)


def run_prepare_data(corpus_path, out_dir, config, llm_call=None):
    print_run_header("stair prepare-data", config)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_config = config["data"]
    book_title = derive_book_title(corpus_path)

    # fail fast, before any output files are written: a user who forgot to
    # set the LLM env vars should not end up with a half-populated out_dir
    # (docs.jsonl/toc.json written, then a late ValueError)
    if llm_call is None:
        llm_call = build_default_llm_call(config)

    with open(out_dir / "config.json", "w") as config_file:
        json.dump(config, config_file, indent=2)
    print(f"Config saved: {out_dir / 'config.json'}")

    print(f"Extracting sections from {corpus_path}")
    sections = extract_sections_from_pdf(
        pdf_path=corpus_path,
        book_title=book_title,
        skip_config=EMPTY_SKIP_CONFIG,
        min_words_in_a_section=data_config["min_words_in_a_section"],
    )
    print(f"Extracted {len(sections)} sections")

    write_jsonl(out_dir / "docs.jsonl", sections)
    print(f"Docs saved: {out_dir / 'docs.jsonl'}")

    toc = build_toc(sections, book_title=book_title)
    with open(out_dir / "toc.json", "w") as toc_file:
        json.dump(toc, toc_file, indent=2)
    print(f"Table of contents saved: {out_dir / 'toc.json'}")

    chunks = chunk_sections(
        sections,
        max_tokens=data_config["chunk_max_tokens"],
        token_to_char=4.0,
        chunk_overlap=data_config["chunk_overlap"],
    )
    print(f"Split into {len(chunks)} chunks")

    qa_pairs = []
    for chunk in chunks:
        qa_pairs.extend(
            generate_qa_pairs_for_chunk(
                chunk,
                llm_call,
                num_pairs=data_config["qa_pairs_per_chunk"],
                book_title=book_title,
            )
        )
    print(f"Generated {len(qa_pairs)} synthetic QA pairs")

    train, val, test = split_qa_pairs(
        qa_pairs,
        train_fraction=data_config["train_fraction"],
        val_fraction=data_config["val_fraction"],
        test_fraction=data_config["test_fraction"],
        seed=data_config["split_seed"],
    )
    print(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")

    write_jsonl(out_dir / "train.jsonl", train)
    print(f"Train saved: {out_dir / 'train.jsonl'}")
    write_jsonl(out_dir / "val.jsonl", val)
    print(f"Val saved: {out_dir / 'val.jsonl'}")
    write_jsonl(out_dir / "test.jsonl", test)
    print(f"Test saved: {out_dir / 'test.jsonl'}")

    return {
        "sections": len(sections),
        "chunks": len(chunks),
        "qa_pairs": len(qa_pairs),
        "train": len(train),
        "val": len(val),
        "test": len(test),
    }

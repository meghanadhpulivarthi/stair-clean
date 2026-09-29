import json
import textwrap

import pytest

from stair.core.train.io_functions import prepare_input_output_stair


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        rendered_parts = []
        for message in messages:
            rendered_parts.append(f"<{message['role']}>{message['content']}")
        return "\n".join(rendered_parts)


def write_toc_json(tmp_path):
    toc = {
        "title": "Test Book",
        "table_of_contents": [
            {"id": "1", "section_num": "1", "title": "Introduction", "leaf": True},
            {"id": "2", "section_num": "2", "title": "Getting Started", "leaf": True},
        ],
    }
    toc_path = tmp_path / "toc.json"
    toc_path.write_text(json.dumps(toc))
    return toc_path


def test_prepare_input_output_stair_builds_prompt_and_completion(tmp_path):
    toc_path = write_toc_json(tmp_path)
    examples = [
        {"question": "What is this book about?", "reference": ["1"]},
    ]

    result = prepare_input_output_stair(
        data_config=None,
        examples=examples,
        tokenizer=FakeTokenizer(),
        split="train",
        system_prompt="You are a helpful assistant.",
        toc_json=str(toc_path),
        user_prompt="Book: {title}\nTOC:\n{toc}\nQuestion: {question}\nAnswer:",
        output_col="{reference}",
    )

    assert len(result) == 1
    assert result[0]["messages"][0]["role"] == "system"
    assert result[0]["messages"][1]["content"].startswith("Book: Test Book")
    assert "<system>" in result[0]["prompt"]
    assert result[0]["completion"] == '\n["1 Introduction"]'


def test_prepare_input_output_stair_raises_on_unknown_reference_id(tmp_path):
    toc_path = write_toc_json(tmp_path)
    examples = [
        {"question": "What is this book about?", "reference": ["99"]},
    ]

    with pytest.raises(KeyError, match="99"):
        prepare_input_output_stair(
            data_config=None,
            examples=examples,
            tokenizer=FakeTokenizer(),
            split="train",
            system_prompt="You are a helpful assistant.",
            toc_json=str(toc_path),
            user_prompt="Book: {title}\nTOC:\n{toc}\nQuestion: {question}\nAnswer:",
            output_col="{reference}",
        )


def test_prepare_input_output_stair_handles_multiple_reference_ids(tmp_path):
    toc_path = write_toc_json(tmp_path)
    examples = [
        {"question": "Compare both sections.", "reference": ["1", "2"]},
    ]

    result = prepare_input_output_stair(
        data_config=None,
        examples=examples,
        tokenizer=FakeTokenizer(),
        split="train",
        system_prompt="You are a helpful assistant.",
        toc_json=str(toc_path),
        user_prompt="Question: {question}",
        output_col="{reference}",
    )

    assert result[0]["completion"] == '\n["1 Introduction", "2 Getting Started"]'

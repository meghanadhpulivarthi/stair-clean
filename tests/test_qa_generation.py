from stair.core.data.qa_generation import generate_qa_pairs_for_chunk


def make_fake_llm_call(responses):
    call_count = {"value": 0}

    def fake_llm_call(messages):
        response = responses[call_count["value"]]
        call_count["value"] += 1
        return response

    return fake_llm_call


def test_generate_qa_pairs_parses_well_formed_response():
    chunk = {"text": "Some content.", "doc_id": ["1", "Introduction"], "metadata": {"section_title": "Introduction"}}
    fake_llm_call = make_fake_llm_call(
        ['[{"question": "What is this about?", "answer": "Some content."}]']
    )

    qa_pairs = generate_qa_pairs_for_chunk(chunk, fake_llm_call, num_pairs=1, book_title="Example Book")

    assert len(qa_pairs) == 1
    assert qa_pairs[0]["question"] == "What is this about?"
    assert qa_pairs[0]["answer"] == "Some content."
    assert qa_pairs[0]["reference"] == ["1"]


def test_generate_qa_pairs_retries_on_malformed_response_then_succeeds():
    chunk = {"text": "Some content.", "doc_id": ["1", "Introduction"], "metadata": {"section_title": "Introduction"}}
    fake_llm_call = make_fake_llm_call(
        [
            "not valid json at all",
            '[{"question": "What is this about?", "answer": "Some content."}]',
        ]
    )

    qa_pairs = generate_qa_pairs_for_chunk(chunk, fake_llm_call, num_pairs=1, book_title="Example Book", max_retries=3)

    assert len(qa_pairs) == 1


def test_generate_qa_pairs_gives_up_after_max_retries():
    chunk = {"text": "Some content.", "doc_id": ["1", "Introduction"], "metadata": {"section_title": "Introduction"}}
    fake_llm_call = make_fake_llm_call(["not valid json", "still not valid", "nope"])

    qa_pairs = generate_qa_pairs_for_chunk(chunk, fake_llm_call, num_pairs=1, book_title="Example Book", max_retries=3)

    assert qa_pairs == []


def test_generate_qa_pairs_drops_entries_missing_question_or_answer():
    chunk = {"text": "Some content.", "doc_id": ["1", "Introduction"], "metadata": {"section_title": "Introduction"}}
    fake_llm_call = make_fake_llm_call(
        ['[{"question": "Complete pair?", "answer": "Yes."}, {"question": "Missing answer"}]']
    )

    qa_pairs = generate_qa_pairs_for_chunk(chunk, fake_llm_call, num_pairs=2, book_title="Example Book")

    assert len(qa_pairs) == 1
    assert qa_pairs[0]["question"] == "Complete pair?"

from stair.core.data.chunking import chunk_sections


def test_short_section_is_not_split():
    sections = [
        {"text": "A short section.", "doc_id": ["1", "Intro"], "metadata": {"section_title": "Intro"}},
    ]
    chunks = chunk_sections(sections, max_tokens=100, token_to_char=4.0, chunk_overlap=8)
    assert len(chunks) == 1
    assert chunks[0]["text"] == "A short section."
    assert chunks[0]["doc_id"] == ["1", "Intro"]


def test_long_section_is_split_into_multiple_chunks_with_shared_doc_id():
    long_text = " ".join(["word" + str(i) for i in range(500)])
    sections = [
        {"text": long_text, "doc_id": ["2", "Long Chapter"], "metadata": {"section_title": "Long Chapter"}},
    ]
    chunks = chunk_sections(sections, max_tokens=50, token_to_char=4.0, chunk_overlap=8)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk["doc_id"] == ["2", "Long Chapter"]


def test_chunk_sections_preserves_metadata():
    sections = [
        {"text": "Some text.", "doc_id": ["1", "Intro"], "metadata": {"section_title": "Intro", "page_start": 0}},
    ]
    chunks = chunk_sections(sections, max_tokens=100, token_to_char=4.0, chunk_overlap=8)
    assert chunks[0]["metadata"]["page_start"] == 0

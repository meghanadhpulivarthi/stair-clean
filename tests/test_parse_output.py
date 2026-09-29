from stair.core.eval.parse_output import parse_predicted_sections


TITLE_TO_ID_MAP = {
    "1 Introduction": "1",
    "2 Getting Started": "2",
}


def test_parse_predicted_sections_valid_list_all_known():
    predicted_ids, hallucination_count = parse_predicted_sections(
        '["1 Introduction", "2 Getting Started"]', TITLE_TO_ID_MAP
    )
    assert predicted_ids == ["1", "2"]
    assert hallucination_count == 0


def test_parse_predicted_sections_unparseable_text_is_a_full_hallucination():
    predicted_ids, hallucination_count = parse_predicted_sections(
        "I think the answer is in chapter one.", TITLE_TO_ID_MAP
    )
    assert predicted_ids == []
    assert hallucination_count == 1


def test_parse_predicted_sections_drops_unknown_titles_and_counts_hallucination():
    predicted_ids, hallucination_count = parse_predicted_sections(
        '["1 Introduction", "9 Nonexistent Section"]', TITLE_TO_ID_MAP
    )
    assert predicted_ids == ["1"]
    assert hallucination_count == 1


def test_parse_predicted_sections_empty_list_is_valid_with_no_hallucination():
    predicted_ids, hallucination_count = parse_predicted_sections("[]", TITLE_TO_ID_MAP)
    assert predicted_ids == []
    assert hallucination_count == 0


def test_parse_predicted_sections_non_string_entry_counts_as_hallucination():
    predicted_ids, hallucination_count = parse_predicted_sections("[1, 2]", TITLE_TO_ID_MAP)
    assert predicted_ids == []
    assert hallucination_count == 2

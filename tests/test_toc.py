from stair.core.data.toc import build_toc


def test_build_toc_has_book_title():
    sections = [{"doc_id": ["1", "Introduction"], "metadata": {"book_title": "Example Book"}}]
    toc = build_toc(sections, book_title="Example Book")
    assert toc["title"] == "Example Book"


def test_build_toc_lists_one_entry_per_section_in_order():
    sections = [
        {"doc_id": ["1", "Introduction"], "metadata": {"book_title": "Example Book"}},
        {"doc_id": ["2", "Getting Started"], "metadata": {"book_title": "Example Book"}},
    ]
    toc = build_toc(sections, book_title="Example Book")
    assert len(toc["table_of_contents"]) == 2
    assert toc["table_of_contents"][0] == {"id": "1", "section_num": "1", "title": "Introduction", "leaf": True}
    assert toc["table_of_contents"][1] == {"id": "2", "section_num": "2", "title": "Getting Started", "leaf": True}

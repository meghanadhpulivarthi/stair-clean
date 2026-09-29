from stair.core.data.pdf_extract import (
    assign_section_numbers,
    assign_unique_doc_ids,
    build_final_content,
    extract_sections_from_pdf,
    normalize_title,
)

from pdf_fixtures import make_tiny_bookmarked_pdf


def test_normalize_title_strips_leading_numbers():
    assert normalize_title("3 Getting Started") == "Getting Started"


def test_build_final_content_drops_content_matching_section_title():
    content = build_final_content(
        list_content=["Introduction"],
        section_title="Introduction",
        book_title="Example Book",
        skip_sentences_set=set(),
        min_words=1,
        config_skip={"skip_sentence_contains": []},
    )
    assert content == ""


def test_build_final_content_keeps_real_content():
    content = build_final_content(
        list_content=["This is the introduction chapter.\nIt has two short lines."],
        section_title="Introduction",
        book_title="Example Book",
        skip_sentences_set=set(),
        min_words=3,
        config_skip={"skip_sentence_contains": []},
    )
    assert "introduction chapter" in content.lower()


def test_build_final_content_drops_content_below_min_words():
    content = build_final_content(
        list_content=["Too short"],
        section_title="Introduction",
        book_title="Example Book",
        skip_sentences_set=set(),
        min_words=10,
        config_skip={"skip_sentence_contains": []},
    )
    assert content == ""


def test_assign_section_numbers_flat_chapters():
    nodes = [
        {"metadata": {"bookmark": {"level": 0}}},
        {"metadata": {"bookmark": {"level": 0}}},
    ]
    numbers = assign_section_numbers(nodes)
    assert numbers == {0: "1", 1: "2"}


def test_assign_unique_doc_ids_sets_doc_id_pair():
    nodes = [
        {"metadata": {"bookmark": {"level": 0}, "section_title": "Introduction"}},
        {"metadata": {"bookmark": {"level": 0}, "section_title": "Getting Started"}},
    ]
    final_nodes = assign_unique_doc_ids(nodes)
    assert final_nodes[0]["doc_id"] == ["1", "Introduction"]
    assert final_nodes[1]["doc_id"] == ["2", "Getting Started"]


def test_extract_sections_from_pdf_reads_real_bookmarked_pdf(tmp_path):
    pdf_path = tmp_path / "tiny.pdf"
    make_tiny_bookmarked_pdf(pdf_path)

    sections = extract_sections_from_pdf(
        pdf_path=pdf_path,
        book_title="Tiny Book",
        skip_config={"skip_sections": [], "skip_sections_startswith": [], "skip_sections_endswith": [], "skip_sentences": [], "skip_sentence_contains": []},
        min_words_in_a_section=3,
    )

    assert len(sections) == 2
    assert sections[0]["doc_id"] == ["1", "Introduction"]
    assert "introduction chapter" in sections[0]["text"].lower()
    assert sections[1]["doc_id"] == ["2", "Getting Started"]


def test_extract_sections_from_pdf_falls_back_when_no_bookmarks(tmp_path):
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    pdf_path = tmp_path / "no_bookmarks.pdf"
    pdf_canvas = canvas.Canvas(str(pdf_path), pagesize=letter)
    pdf_canvas.drawString(72, 750, "Just some text with no outline at all.")
    pdf_canvas.showPage()
    pdf_canvas.save()

    sections = extract_sections_from_pdf(
        pdf_path=pdf_path,
        book_title="No Bookmarks Book",
        skip_config={"skip_sections": [], "skip_sections_startswith": [], "skip_sections_endswith": [], "skip_sentences": [], "skip_sentence_contains": []},
        min_words_in_a_section=3,
    )

    assert len(sections) == 1
    assert "no bookmarks book" in sections[0]["metadata"]["book_title"].lower()

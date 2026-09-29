from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from stair.core.data.pdf_extract import (
    assign_section_numbers,
    assign_unique_doc_ids,
    build_final_content,
    extract_sections_from_pdf,
    normalize_title,
    remove_overlap_with_next_section,
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


def make_pdf_with_mismatched_bookmark_title(path):
    # this chapter's outline title does NOT match the heading text actually
    # drawn on its start page, so text_page.search(title) will miss for it
    pdf_canvas = canvas.Canvas(str(path), pagesize=letter)

    pdf_canvas.bookmarkPage("chapter_0")
    pdf_canvas.addOutlineEntry("Mismatched Bookmark Title", "chapter_0", level=0)
    y_position = 750
    pdf_canvas.setFont("Helvetica-Bold", 14)
    pdf_canvas.drawString(72, y_position, "Actual Heading Text")
    y_position -= 30
    pdf_canvas.setFont("Helvetica", 11)
    pdf_canvas.drawString(72, y_position, "This is the first chapter's real body content.")
    pdf_canvas.showPage()

    pdf_canvas.save()


def test_bookmark_title_mismatch_prints_diagnostic_warning(tmp_path, capsys):
    pdf_path = tmp_path / "mismatched.pdf"
    make_pdf_with_mismatched_bookmark_title(pdf_path)

    extract_sections_from_pdf(
        pdf_path=pdf_path,
        book_title="Mismatched Book",
        skip_config={
            "skip_sections": [],
            "skip_sections_startswith": [],
            "skip_sections_endswith": [],
            "skip_sentences": [],
            "skip_sentence_contains": [],
        },
        min_words_in_a_section=3,
    )

    # before the fix this failure mode was completely silent (content = "")
    # with no diagnostic signal at all
    captured = capsys.readouterr()
    assert "Mismatched Bookmark Title" in captured.out
    assert "Could not find bookmark title on its start page" in captured.out


def test_remove_overlap_does_not_truncate_previous_section_on_empty_next_page_text():
    # reproduces the finding directly: when the next section's first-page
    # text is empty (e.g. because its bookmark title search missed), naive
    # str.find("") returns 0, which used to wipe out the previous section's
    # last page even though there was no real overlap to remove
    raw_sections = [
        {"list_content": ["Some real trailing content on the shared page."]},
        {"list_content": [""]},
    ]

    remove_overlap_with_next_section(raw_sections)

    assert raw_sections[0]["list_content"][-1] == "Some real trailing content on the shared page."


def test_remove_overlap_still_truncates_genuine_overlap():
    # sanity check that the guard doesn't break the normal, non-empty case
    raw_sections = [
        {"list_content": ["Tail of section one. Getting Started\nBody of section two."]},
        {"list_content": ["Getting Started\nBody of section two."]},
    ]

    remove_overlap_with_next_section(raw_sections)

    assert raw_sections[0]["list_content"][-1] == "Tail of section one. "

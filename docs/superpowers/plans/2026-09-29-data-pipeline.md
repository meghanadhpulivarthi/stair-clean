# STAIR Data Pipeline (`stair prepare-data`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `stair prepare-data`: turn a bookmarked PDF into `docs.jsonl`, `toc.json`, and `train.jsonl`/`val.jsonl`/`test.jsonl`, and ship one bundled example corpus so the command works out of the box.

**Architecture:** Five small, independently-testable adapter modules under `src/stair/core/data/` — PDF/bookmark extraction, chunking, TOC building, synthetic QA generation, and train/val/test splitting — ported from `benchmark/extract_pdf` and `stair/experiments/rag_eda_benchmark` with their transformation logic unchanged, wired together by the `prepare-data` CLI command. QA generation talks to an OpenAI-compatible chat completion endpoint through an injected callable, so it is fully unit-testable without network access. The bundled example corpus is an original short multi-chapter guide, rendered to a bookmarked PDF by a small reusable PDF-authoring utility that both the test suite and the corpus-build script call.

**Tech Stack:** Python 3.12, `pypdfium2` (PDF reading), `reportlab` (PDF authoring, for the example corpus and test fixtures), `langchain-text-splitters` (chunking), `openai` (QA generation client), `pytest`.

**Spec:** `docs/superpowers/specs/2026-09-29-external-pipeline-design.md`

**Plan 1 (already merged):** `docs/superpowers/plans/2026-09-29-pipeline-scaffolding-and-config.md` — provides `stair.config.resolve_config(override_path=None) -> dict` and `stair.run_header.print_run_header(script_name, config) -> None`, both consumed by Task 6 here.

## Global Constraints

- Every tunable used here (`data.chunk_max_tokens`, `data.chunk_overlap`, `data.min_words_in_a_section`, `data.qa_pairs_per_chunk`, `data.train_fraction`/`val_fraction`/`test_fraction`, `data.split_seed`) already exists in `configs/base.yaml` from Plan 1 — reuse it, do not add a parallel config mechanism.
- No type hints, no one-liners, no speculative abstraction, 4-space indent, full-word names (code-style.md).
- Every step of `prepare-data` prints what it loaded/produced and every output file path as it happens (traceability.md): doc count, chunk count, QA pair count, train/val/test sizes.
- No absolute paths in checked-in code, configs, or the bundled example corpus's build script.
- QA generation must work against any OpenAI-compatible endpoint (`base_url` + API key), not a specific vendor — the spec rejected private/internal-only dependencies.
- The bundled example corpus must require no network access and no API key to build or to run `prepare-data`'s non-QA steps in tests; QA generation is tested through dependency injection (a fake completion callable), never a real network call.

## Review Focus

- **A PDF with no bookmarks/outline at all:** `benchmark/extract_pdf`'s existing `extract.py` already falls back to treating the whole PDF as one section (`extract_all_content`) rather than crashing — the ported version must keep that fallback, since a user's first PDF attempt easily lacks bookmarks.
- **A section whose extracted text is empty or below `min_words_in_a_section`:** must be dropped, not passed downstream as a zero-content training example (mirrors existing `build_final_content` behavior).
- **The synthetic QA endpoint returning unparseable or partial JSON:** must be logged and skipped for that chunk, not crash the whole `prepare-data` run — a single bad LLM response shouldn't lose the rest of the corpus.
- **`train_fraction + val_fraction + test_fraction` not summing to 1.0** in a user's override config: since Plan 1's `resolve_config` only checks key names, not value semantics, an override like `train_fraction: 0.9` without adjusting the others would silently produce an unintended split — the split function must reject fractions that don't sum to 1.0 (within floating-point tolerance).
- **Re-running `prepare-data` on the same corpus:** must reproduce the same train/val/test split (spec's `data.split_seed`), not a new random split each time — otherwise a user can't reproduce a prior run's numbers.

---

## File Structure

```
src/stair/
  core/
    __init__.py
    data/
      __init__.py
      pdf_extract.py        # bookmark/content extraction, ported from benchmark/extract_pdf/src/pdf/extract.py
      chunking.py            # long-section splitting, ported from benchmark/extract_pdf/src/chunk/split.py
      toc.py                 # sections -> toc.json structure
      qa_generation.py       # synthetic QA pairs via an injected LLM-call callable
      split.py                # train/val/test split
      build_pdf.py            # shared PDF-authoring utility (bookmarked PDF from chapter text)
  cli.py                     # modified: prepare-data now calls the real pipeline
data/
  example_book/
    source.pdf                # bundled example corpus (built by scripts/build_example_corpus.py)
scripts/
  build_example_corpus.py    # authors the original example text and renders source.pdf
tests/
  pdf_fixtures.py             # tiny bookmarked-PDF builder shared by test modules
  test_pdf_extract.py
  test_chunking.py
  test_toc.py
  test_qa_generation.py
  test_split.py
  test_prepare_data_cli.py
```

---

### Task 1: PDF bookmark/content extraction (`core/data/pdf_extract.py`)

**Files:**
- Create: `src/stair/core/data/__init__.py`
- Create: `src/stair/core/__init__.py`
- Create: `src/stair/core/data/pdf_extract.py`
- Create: `tests/pdf_fixtures.py`
- Test: `tests/test_pdf_extract.py`

**Interfaces:**
- Consumes: nothing from prior tasks.
- Produces: `stair.core.data.pdf_extract.extract_sections_from_pdf(pdf_path, book_title, skip_config, min_words_in_a_section) -> list[dict]`. Each returned dict has the shape `{"text": str, "doc_id": [section_num_str, section_title_str], "metadata": {"book_title": str, "section_title": str, "page_start": int, "page_end": int, "level": int}}`. Falls back to a single section covering the whole PDF when it has no bookmarks. Used by Task 6.
- Produces: `stair.core.data.build_pdf.write_bookmarked_pdf(path, chapters)` is actually defined in Task 7, but this task's tests need a tiny bookmarked PDF, so `tests/pdf_fixtures.py` defines its own minimal local copy: `make_tiny_bookmarked_pdf(path) -> None`, writing a 2-chapter PDF with `reportlab`. (Task 7 later promotes this pattern into the real shared `core/data/build_pdf.py`; duplicating a 15-line helper once, rather than making Task 7 an unstated dependency of Task 1, keeps this task's Interfaces self-contained.)

- [ ] **Step 1: Write the fixture helper**

Create `tests/pdf_fixtures.py`:

```python
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


def make_tiny_bookmarked_pdf(path):
    pdf_canvas = canvas.Canvas(str(path), pagesize=letter)

    chapters = [
        ("Introduction", ["This is the introduction chapter.", "It has two short lines."]),
        ("Getting Started", ["This is the getting started chapter.", "It also has two lines."]),
    ]

    for chapter_index, (title, lines) in enumerate(chapters):
        bookmark_key = f"chapter_{chapter_index}"
        pdf_canvas.bookmarkPage(bookmark_key)
        pdf_canvas.addOutlineEntry(title, bookmark_key, level=0)

        y_position = 750
        pdf_canvas.setFont("Helvetica-Bold", 14)
        pdf_canvas.drawString(72, y_position, title)
        y_position -= 30

        pdf_canvas.setFont("Helvetica", 11)
        for line in lines:
            pdf_canvas.drawString(72, y_position, line)
            y_position -= 16

        pdf_canvas.showPage()

    pdf_canvas.save()
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_pdf_extract.py`:

```python
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
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_pdf_extract.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'stair.core'`

- [ ] **Step 4: Write `src/stair/core/__init__.py` and `src/stair/core/data/__init__.py`**

Both empty files.

- [ ] **Step 5: Write `src/stair/core/data/pdf_extract.py`**

```python
import logging

import pypdfium2


def normalize_title(title):
    words = title.split()
    words = [word for word in words if not word.isdigit()]
    return " ".join(words).strip()


def extract_bookmarks_data(pdf):
    bookmarks = list(pdf.get_toc())
    bookmarks_data = []
    last_page_index = len(pdf) - 1

    for bookmark_index, bookmark in enumerate(bookmarks):
        title = normalize_title(bookmark.get_title())
        level = bookmark.level

        dest = bookmark.get_dest()
        if not dest:
            logging.warning(f"Bookmark {bookmark_index} dest missing")
            continue

        index = dest.get_index()
        if index is None:
            logging.warning(f"Bookmark {bookmark_index} index missing")
            continue

        if bookmark_index + 1 < len(bookmarks):
            next_dest = bookmarks[bookmark_index + 1].get_dest()
            next_index = next_dest.get_index() if next_dest else None
        else:
            next_index = None

        # the last bookmark has no "next" bookmark to bound it, so its
        # content runs to the end of the PDF instead
        page_end = next_index if next_index is not None else last_page_index

        bookmarks_data.append(
            {
                "title": title,
                "level": level,
                "page_start": index,
                "page_end": page_end,
            }
        )

    return bookmarks_data


def extract_content_for_bookmark(pdf, bookmark):
    title = bookmark["title"]
    page_start = bookmark["page_start"]
    page_end = bookmark["page_end"]
    list_content = []

    for page_index in range(page_start, page_end + 1):
        text_page = pdf[page_index].get_textpage()

        if page_index == page_start:
            text_searcher = text_page.search(title)
            search_result = text_searcher.get_next()
            if search_result:
                content = text_page.get_text_range(search_result[0])
            else:
                content = ""
        else:
            content = text_page.get_text_range()

        list_content.append(content)

    return list_content


def remove_overlap_with_next_section(raw_sections):
    # page_end deliberately includes the page the next section starts on
    # (that page may hold the tail of this section and the head of the
    # next one), so strip whatever text the next section's first page
    # already accounts for out of this section's last page
    for section_index, section in enumerate(raw_sections[:-1]):
        last_page_text = section["list_content"][-1]
        next_section_first_page_text = raw_sections[section_index + 1]["list_content"][0]

        overlap_start = last_page_text.find(next_section_first_page_text)
        if overlap_start != -1:
            section["list_content"][-1] = last_page_text[:overlap_start]


def should_skip_section(bookmark, config_skip, book_title):
    if book_title and bookmark["title"].lower() == book_title.lower():
        return True
    title = bookmark["title"]
    for section in config_skip.get("skip_sections", []):
        if title.lower() == section.lower():
            return True
    for section in config_skip.get("skip_sections_startswith", []):
        if title.lower().startswith(section.lower()):
            return True
    for section in config_skip.get("skip_sections_endswith", []):
        if title.lower().endswith(section.lower()):
            return True
    return False


def skip_content(text, section_title, book_title):
    text_lower = text.lower()
    if text_lower == section_title.lower():
        return True
    if book_title and text_lower == book_title.lower():
        return True
    return False


def build_final_content(list_content, section_title, book_title, skip_sentences_set, min_words, config_skip):
    def should_skip_sentence(src_sentence):
        for skip_phrase in config_skip["skip_sentence_contains"]:
            if skip_phrase.lower() in src_sentence:
                return True
        return False

    def clean_content(text):
        sentences = text.split("\n")
        final_sentences = []
        len_book_title = len(book_title)
        for sentence in sentences:
            src_sentence = sentence.strip().lower()
            if not src_sentence:
                continue
            if src_sentence in skip_sentences_set:
                continue
            if should_skip_sentence(src_sentence):
                continue
            title_index = src_sentence.find(book_title.lower())
            if title_index != -1 and len_book_title > 0:
                ratio_book_title_to_sentence = len_book_title / len(src_sentence)
                if ratio_book_title_to_sentence > 0.8:
                    continue
            final_sentences.append(sentence)
        return "\n".join(final_sentences).strip()

    list_content = [clean_content(content) for content in list_content]
    list_content = [content for content in list_content if not skip_content(content, section_title, book_title)]
    list_content = [content for content in list_content if content]
    content = "\n".join(list_content).strip()

    content_lower = content.lower()
    if not content_lower:
        return ""
    if content_lower == section_title.lower():
        return ""
    if book_title and content_lower == book_title.lower():
        return ""
    if len(content.split()) < min_words:
        return ""

    return content


def assign_section_numbers(nodes):
    section_nums = {}
    last_level = 0
    chapters = {}
    levels, prefix = [], []

    for node_index, node in enumerate(nodes):
        level = node["metadata"]["bookmark"]["level"]

        while levels and level < last_level:
            if last_level in chapters:
                del chapters[last_level]
            last_level = levels.pop()
            prefix.pop()

        if levels and levels[-1] == level:
            levels.pop()
            prefix.pop()

        chapters[level] = chapters.get(level, 0) + 1
        prefix.append(str(chapters[level]))
        levels.append(level)
        last_level = level

        section_nums[node_index] = ".".join(prefix)

    return section_nums


def assign_unique_doc_ids(nodes):
    final_nodes = []
    section_nums = assign_section_numbers(nodes)

    for node_index, node in enumerate(nodes):
        doc_id = section_nums[node_index]
        section_title = node["metadata"]["section_title"].strip()
        node["doc_id"] = [doc_id, section_title]
        final_nodes.append(node)

    return final_nodes


def extract_all_content_as_single_section(pdf, book_title):
    pages = []
    for page_index in range(len(pdf)):
        text_page = pdf[page_index].get_textpage()
        pages.append(text_page.get_text_range())

    content = "\n".join(pages).strip()
    return [
        {
            "text": content,
            "doc_id": ["1", book_title],
            "metadata": {
                "book_title": book_title,
                "section_title": book_title,
                "page_start": 0,
                "page_end": max(len(pdf) - 1, 0),
                "level": 0,
            },
        }
    ]


def extract_sections_from_pdf(pdf_path, book_title, skip_config, min_words_in_a_section):
    pdf = pypdfium2.PdfDocument(pdf_path)

    bookmarks_data = extract_bookmarks_data(pdf)
    if not bookmarks_data:
        logging.warning("No bookmarks found in the PDF; treating it as a single section.")
        return extract_all_content_as_single_section(pdf, book_title)

    raw_sections = [
        {"bookmark": bookmark, "list_content": extract_content_for_bookmark(pdf, bookmark)}
        for bookmark in bookmarks_data
    ]
    remove_overlap_with_next_section(raw_sections)

    skip_sentences_set = set(sentence.lower().strip() for sentence in skip_config.get("skip_sentences", []))

    nodes = []
    for raw_section in raw_sections:
        bookmark = raw_section["bookmark"]
        if should_skip_section(bookmark, skip_config, book_title):
            continue

        content = build_final_content(
            raw_section["list_content"],
            bookmark["title"],
            book_title,
            skip_sentences_set,
            min_words_in_a_section,
            skip_config,
        )
        if not content:
            continue

        nodes.append(
            {
                "text": content,
                "metadata": {
                    "book_title": book_title,
                    "section_title": bookmark["title"],
                    "page_start": bookmark["page_start"],
                    "page_end": bookmark["page_end"],
                    "bookmark": {"level": bookmark["level"]},
                    "level": bookmark["level"],
                },
            }
        )

    final_nodes = assign_unique_doc_ids(nodes)
    for node in final_nodes:
        del node["metadata"]["bookmark"]

    return final_nodes
```

- [ ] **Step 6: Run test to verify it passes**

Run: `uv run pytest tests/test_pdf_extract.py -v`
Expected: PASS (8 tests)

- [ ] **Step 7: Add new dependencies**

Run: `uv add pypdfium2 reportlab`

- [ ] **Step 8: Re-run test to verify it passes with dependencies installed**

Run: `uv run pytest tests/test_pdf_extract.py -v`
Expected: PASS (8 tests)

- [ ] **Step 9: Commit**

```bash
git add src/stair/core/__init__.py src/stair/core/data/__init__.py src/stair/core/data/pdf_extract.py tests/pdf_fixtures.py tests/test_pdf_extract.py pyproject.toml uv.lock
git commit -m "feat: add PDF bookmark/content extraction adapter"
```

---

### Task 2: Chunking (`core/data/chunking.py`)

**Files:**
- Create: `src/stair/core/data/chunking.py`
- Test: `tests/test_chunking.py`

**Interfaces:**
- Consumes: sections in the shape produced by Task 1's `extract_sections_from_pdf` (each with `"text"`, `"doc_id"`, `"metadata"`).
- Produces: `stair.core.data.chunking.chunk_sections(sections, max_tokens, token_to_char, chunk_overlap) -> list[dict]`. Same shape as input; a section whose text fits within `max_tokens * token_to_char` characters passes through as a single chunk, longer sections are split into multiple chunks that all keep the same `doc_id`/`metadata`. Used by Task 6.

- [ ] **Step 1: Write the failing test**

Create `tests/test_chunking.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_chunking.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'stair.core.data.chunking'`

- [ ] **Step 3: Write `src/stair/core/data/chunking.py`**

```python
from copy import deepcopy

from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_sections(sections, max_tokens, token_to_char, chunk_overlap):
    chunk_size_chars = int(max_tokens * token_to_char)
    chunk_overlap_chars = int(chunk_overlap * token_to_char)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size_chars,
        chunk_overlap=chunk_overlap_chars,
    )

    chunks = []
    for section in sections:
        for chunk_text in splitter.split_text(section["text"]):
            chunk = deepcopy(section)
            chunk["text"] = chunk_text
            chunks.append(chunk)

    return chunks
```

- [ ] **Step 4: Add dependency**

Run: `uv add langchain-text-splitters`

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_chunking.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Commit**

```bash
git add src/stair/core/data/chunking.py tests/test_chunking.py pyproject.toml uv.lock
git commit -m "feat: add section chunking adapter"
```

---

### Task 3: TOC builder (`core/data/toc.py`)

**Files:**
- Create: `src/stair/core/data/toc.py`
- Test: `tests/test_toc.py`

**Interfaces:**
- Consumes: sections in the shape produced by Task 1 (each with `doc_id: [section_num_str, section_title_str]`, `metadata.book_title`).
- Produces: `stair.core.data.toc.build_toc(sections, book_title) -> dict`, shape `{"title": str, "table_of_contents": [{"id": str, "section_num": str, "title": str, "leaf": True}, ...]}`, one entry per section, in the order the sections were given. `id` and `section_num` are both the section's `doc_id[0]`. Used by Task 6.

- [ ] **Step 1: Write the failing test**

Create `tests/test_toc.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_toc.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'stair.core.data.toc'`

- [ ] **Step 3: Write `src/stair/core/data/toc.py`**

```python
def build_toc(sections, book_title):
    table_of_contents = []
    for section in sections:
        section_num, section_title = section["doc_id"]
        table_of_contents.append(
            {
                "id": section_num,
                "section_num": section_num,
                "title": section_title,
                "leaf": True,
            }
        )

    return {
        "title": book_title,
        "table_of_contents": table_of_contents,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_toc.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/stair/core/data/toc.py tests/test_toc.py
git commit -m "feat: add table-of-contents builder"
```

---

### Task 4: Synthetic QA generation (`core/data/qa_generation.py`)

**Files:**
- Create: `src/stair/core/data/qa_generation.py`
- Test: `tests/test_qa_generation.py`

**Interfaces:**
- Consumes: chunks in the shape produced by Task 2 (`text`, `doc_id`, `metadata`).
- Produces: `stair.core.data.qa_generation.generate_qa_pairs_for_chunk(chunk, llm_call, num_pairs, book_title, max_retries=3) -> list[dict]`, each result dict shaped `{"question": str, "answer": str, "reference": [section_num_str]}`. `llm_call` is `messages: list[dict] -> str` (a chat completion call that returns the raw assistant text) — production wiring (Task 6) passes a real OpenAI-compatible client call; tests pass a fake. On unparseable output, retries up to `max_retries` times, then logs and returns `[]` for that chunk rather than raising.
- Produces: `stair.core.data.qa_generation.build_openai_compatible_llm_call(api_base, api_key, model_name) -> callable` — the production `llm_call`, used only by Task 6 (not exercised by this task's own tests, since it requires network).

- [ ] **Step 1: Write the failing test**

Create `tests/test_qa_generation.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_qa_generation.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'stair.core.data.qa_generation'`

- [ ] **Step 3: Write `src/stair/core/data/qa_generation.py`**

```python
import ast
import logging


SYSTEM_PROMPT = """
You are an AI assistant tasked with generating question and answer pairs for the given context.
Return only a Python list of dicts, no other text. Each dict has a "question" key and an "answer" key.
You should create the following number of question/answer pairs: {number_of_pairs}.

Guidelines:
- Answers should be accurate, clear, and directly based on the context.
- Do not repeat or rephrase the same question in multiple ways.
- Questions must be self-contained and understandable without external context.
"""

USER_PROMPT = """
Format:
[{{"question": "...", "answer": "..."}}]

Book: {book_title}
Section: {section_title}
Context:
{context}
"""


def build_messages(chunk, num_pairs, book_title):
    section_title = chunk["metadata"]["section_title"]
    context = chunk["text"]
    return [
        {"role": "system", "content": SYSTEM_PROMPT.format(number_of_pairs=num_pairs)},
        {
            "role": "user",
            "content": USER_PROMPT.format(book_title=book_title, section_title=section_title, context=context),
        },
    ]


def parse_qa_response(raw_content):
    try:
        parsed = ast.literal_eval(raw_content.strip())
    except (ValueError, SyntaxError):
        return None

    if not isinstance(parsed, list):
        return None

    return parsed


def generate_qa_pairs_for_chunk(chunk, llm_call, num_pairs, book_title, max_retries=3):
    section_num = chunk["doc_id"][0]
    messages = build_messages(chunk, num_pairs, book_title)

    for attempt in range(1, max_retries + 1):
        raw_content = llm_call(messages)
        parsed = parse_qa_response(raw_content)

        if parsed is not None:
            return [
                {"question": item["question"], "answer": item["answer"], "reference": [section_num]}
                for item in parsed
                if "question" in item and "answer" in item
            ]

        logging.warning(f"QA generation attempt {attempt}/{max_retries} failed to parse for section {section_num}")

    logging.error(f"QA generation gave up after {max_retries} attempts for section {section_num}")
    return []


def build_openai_compatible_llm_call(api_base, api_key, model_name):
    from openai import OpenAI

    client = OpenAI(base_url=api_base, api_key=api_key)

    def llm_call(messages):
        completion = client.chat.completions.create(model=model_name, messages=messages)
        return completion.choices[0].message.content

    return llm_call
```

- [ ] **Step 4: Add dependency**

Run: `uv add openai`

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_qa_generation.py -v`
Expected: PASS (4 tests)

- [ ] **Step 6: Commit**

```bash
git add src/stair/core/data/qa_generation.py tests/test_qa_generation.py pyproject.toml uv.lock
git commit -m "feat: add synthetic QA-pair generation with injectable LLM call"
```

---

### Task 5: Train/val/test split (`core/data/split.py`)

**Files:**
- Create: `src/stair/core/data/split.py`
- Test: `tests/test_split.py`

**Interfaces:**
- Consumes: a flat list of QA-pair dicts (the concatenation of every chunk's `generate_qa_pairs_for_chunk` result from Task 4).
- Produces: `stair.core.data.split.split_qa_pairs(qa_pairs, train_fraction, val_fraction, test_fraction, seed) -> (list, list, list)`. Raises `ValueError` if the three fractions don't sum to 1.0 (within 1e-6 tolerance). Deterministic for a given `seed`. Used by Task 6.

- [ ] **Step 1: Write the failing test**

Create `tests/test_split.py`:

```python
import pytest

from stair.core.data.split import split_qa_pairs


def make_qa_pairs(count):
    return [{"question": f"q{i}", "answer": f"a{i}", "reference": [str(i)]} for i in range(count)]


def test_split_sizes_match_fractions():
    qa_pairs = make_qa_pairs(100)
    train, val, test = split_qa_pairs(qa_pairs, train_fraction=0.8, val_fraction=0.1, test_fraction=0.1, seed=42)
    assert len(train) == 80
    assert len(val) == 10
    assert len(test) == 10


def test_split_covers_every_pair_exactly_once():
    qa_pairs = make_qa_pairs(50)
    train, val, test = split_qa_pairs(qa_pairs, train_fraction=0.8, val_fraction=0.1, test_fraction=0.1, seed=42)
    all_questions = sorted(item["question"] for item in train + val + test)
    expected_questions = sorted(item["question"] for item in qa_pairs)
    assert all_questions == expected_questions


def test_split_is_deterministic_for_same_seed():
    qa_pairs = make_qa_pairs(50)
    first_train, first_val, first_test = split_qa_pairs(qa_pairs, train_fraction=0.8, val_fraction=0.1, test_fraction=0.1, seed=42)
    second_train, second_val, second_test = split_qa_pairs(qa_pairs, train_fraction=0.8, val_fraction=0.1, test_fraction=0.1, seed=42)
    assert first_train == second_train
    assert first_val == second_val
    assert first_test == second_test


def test_split_rejects_fractions_not_summing_to_one():
    qa_pairs = make_qa_pairs(10)
    with pytest.raises(ValueError, match="sum to 1.0"):
        split_qa_pairs(qa_pairs, train_fraction=0.9, val_fraction=0.1, test_fraction=0.1, seed=42)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_split.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'stair.core.data.split'`

- [ ] **Step 3: Write `src/stair/core/data/split.py`**

```python
import random


def split_qa_pairs(qa_pairs, train_fraction, val_fraction, test_fraction, seed):
    fraction_sum = train_fraction + val_fraction + test_fraction
    if abs(fraction_sum - 1.0) > 1e-6:
        raise ValueError(
            f"train_fraction + val_fraction + test_fraction must sum to 1.0, got {fraction_sum}"
        )

    shuffled_pairs = list(qa_pairs)
    random.Random(seed).shuffle(shuffled_pairs)

    total_count = len(shuffled_pairs)
    train_count = int(total_count * train_fraction)
    val_count = int(total_count * val_fraction)

    train = shuffled_pairs[:train_count]
    val = shuffled_pairs[train_count:train_count + val_count]
    test = shuffled_pairs[train_count + val_count:]

    return train, val, test
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_split.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/stair/core/data/split.py tests/test_split.py
git commit -m "feat: add deterministic train/val/test split"
```

---

### Task 6: Wire `stair prepare-data` end-to-end

**Files:**
- Modify: `src/stair/cli.py`
- Create: `src/stair/prepare_data.py`
- Test: `tests/test_prepare_data_cli.py`

**Interfaces:**
- Consumes: `extract_sections_from_pdf` (Task 1), `chunk_sections` (Task 2), `build_toc` (Task 3), `generate_qa_pairs_for_chunk` + `build_openai_compatible_llm_call` (Task 4), `split_qa_pairs` (Task 5), `resolve_config` + `print_run_header` (Plan 1).
- Produces: `stair.prepare_data.run_prepare_data(corpus_path, out_dir, config, llm_call=None) -> dict` — the testable entry point. Returns a dict of counts (`{"sections": int, "chunks": int, "qa_pairs": int, "train": int, "val": int, "test": int}`). When `llm_call` is `None`, builds the real OpenAI-compatible call from `config["data"]` (reads `STAIR_LLM_API_BASE`/`STAIR_LLM_API_KEY`/`STAIR_LLM_MODEL` env vars); when given (as in tests), uses it directly. Writes `docs.jsonl`, `toc.json`, `train.jsonl`, `val.jsonl`, `test.jsonl` under `out_dir`. `cli.py`'s `prepare-data` subcommand calls this and prints the run header + counts.

- [ ] **Step 1: Write the failing test**

Create `tests/test_prepare_data_cli.py`:

```python
import json

from pdf_fixtures import make_tiny_bookmarked_pdf

from stair.config import resolve_config
from stair.prepare_data import run_prepare_data


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_prepare_data_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'stair.prepare_data'`

- [ ] **Step 3: Write `src/stair/prepare_data.py`**

```python
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
    book_title = Path(corpus_path).stem

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

    if llm_call is None:
        llm_call = build_default_llm_call(config)

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
```

- [ ] **Step 4: Wire `prepare-data` into `cli.py`**

In `src/stair/cli.py`, add the import:

```python
from stair.prepare_data import run_prepare_data
```

Replace the body of `main()` so the `prepare-data` subcommand calls the real pipeline instead of falling through to the stub message:

```python
def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.subcommand is None:
        parser.print_usage(sys.stderr)
        return 1

    resolved_config = None
    if hasattr(args, "config"):
        try:
            resolved_config = resolve_config(override_path=args.config)
        except (OSError, ValueError) as error:
            print(f"stair {args.subcommand}: {error}", file=sys.stderr)
            return 1

    if args.subcommand == "prepare-data":
        try:
            run_prepare_data(args.corpus, args.out, resolved_config)
        except (OSError, ValueError) as error:
            print(f"stair prepare-data: {error}", file=sys.stderr)
            return 1
        return 0

    print(f"stair {args.subcommand}: not implemented yet", file=sys.stderr)
    return 1
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_prepare_data_cli.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Run the full test suite**

Run: `uv run pytest -v`
Expected: PASS (all tests from every task so far)

- [ ] **Step 7: Commit**

```bash
git add src/stair/prepare_data.py src/stair/cli.py tests/test_prepare_data_cli.py
git commit -m "feat: wire stair prepare-data to the real data pipeline"
```

---

### Task 7: Build and bundle the example corpus

**Files:**
- Create: `src/stair/core/data/build_pdf.py`
- Create: `scripts/build_example_corpus.py`
- Create: `data/example_book/source.pdf` (generated by the script, then committed)
- Test: `tests/test_prepare_data_cli.py` (extend with a smoke test against the real bundled PDF)

**Interfaces:**
- Produces: `stair.core.data.build_pdf.write_bookmarked_pdf(path, chapters) -> None`, where `chapters` is a list of `(title, list_of_paragraph_strings)` tuples. Generalizes `tests/pdf_fixtures.py`'s inline helper from Task 1; used by `scripts/build_example_corpus.py` and reusable by future example corpora.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_prepare_data_cli.py`:

```python
from pathlib import Path


EXAMPLE_CORPUS_PATH = Path(__file__).resolve().parents[1] / "data" / "example_book" / "source.pdf"


def test_bundled_example_corpus_produces_a_realistic_split(tmp_path):
    out_dir = tmp_path / "out"
    config = resolve_config(override_path=None)

    counts = run_prepare_data(EXAMPLE_CORPUS_PATH, out_dir, config, llm_call=fake_llm_call)

    assert counts["sections"] >= 4
    assert counts["qa_pairs"] > 0
    assert counts["train"] + counts["val"] + counts["test"] == counts["qa_pairs"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_prepare_data_cli.py -v`
Expected: FAIL — `data/example_book/source.pdf` does not exist yet, `run_prepare_data` raises `FileNotFoundError` reading it (via `pypdfium2.PdfDocument`).

- [ ] **Step 3: Write `src/stair/core/data/build_pdf.py`**

```python
import textwrap

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


def write_bookmarked_pdf(path, chapters):
    pdf_canvas = canvas.Canvas(str(path), pagesize=letter)

    for chapter_index, (title, paragraphs) in enumerate(chapters):
        bookmark_key = f"chapter_{chapter_index}"
        pdf_canvas.bookmarkPage(bookmark_key)
        pdf_canvas.addOutlineEntry(title, bookmark_key, level=0)

        y_position = 750
        pdf_canvas.setFont("Helvetica-Bold", 14)
        pdf_canvas.drawString(72, y_position, title)
        y_position -= 30

        pdf_canvas.setFont("Helvetica", 11)
        for paragraph in paragraphs:
            for line in textwrap.wrap(paragraph, width=95):
                if y_position < 72:
                    pdf_canvas.showPage()
                    pdf_canvas.setFont("Helvetica", 11)
                    y_position = 750
                pdf_canvas.drawString(72, y_position, line)
                y_position -= 16
            y_position -= 8

        pdf_canvas.showPage()

    pdf_canvas.save()
```

- [ ] **Step 4: Write `scripts/build_example_corpus.py`**

```python
from pathlib import Path

from stair.core.data.build_pdf import write_bookmarked_pdf


OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "example_book" / "source.pdf"

CHAPTERS = [
    (
        "Introduction",
        [
            "This guide is a short, original introduction to baking sourdough bread at home. "
            "It was written for this project as a small, clearly structured example corpus.",
            "Each chapter below covers one part of the process, from building a starter to "
            "troubleshooting a loaf that did not rise.",
        ],
    ),
    (
        "Building a Starter",
        [
            "A sourdough starter is a mixture of flour and water that captures wild yeast and "
            "bacteria from the air and the flour itself.",
            "To build one, mix equal parts flour and water in a jar, and feed it with the same "
            "amounts once a day for about a week, discarding half before each feeding.",
            "A healthy starter roughly doubles in size a few hours after feeding, and smells "
            "pleasantly sour rather than harsh or like nail polish remover.",
        ],
    ),
    (
        "Mixing the Dough",
        [
            "Combine flour, water, salt, and a portion of active starter in a large bowl.",
            "Rest the dough for thirty minutes so the flour fully absorbs the water, a step "
            "called autolyse, before kneading or folding it further.",
            "Fold the dough over itself every thirty minutes for the first two hours to build "
            "strength without traditional kneading.",
        ],
    ),
    (
        "Shaping and Proofing",
        [
            "Once the dough has roughly doubled in bulk, turn it out and shape it into a tight "
            "round or oval loaf.",
            "Place the shaped loaf in a floured basket and let it proof, either at room "
            "temperature for a few hours or in the refrigerator overnight for more flavor.",
        ],
    ),
    (
        "Baking",
        [
            "Preheat a covered pot in the oven to a high temperature, then carefully place the "
            "proofed loaf inside and score the top with a blade.",
            "Bake covered for the first part of the bake to trap steam, then uncover to let the "
            "crust brown for the remainder.",
        ],
    ),
    (
        "Troubleshooting",
        [
            "A dense, flat loaf usually means the starter was not active enough, or the dough "
            "was underproofed before baking.",
            "A loaf that spreads out flat instead of holding its shape usually means the dough "
            "was overproofed or shaped too loosely.",
            "A pale crust that never browns usually means the oven was not preheated to a high "
            "enough temperature before baking.",
        ],
    ),
]


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_bookmarked_pdf(OUTPUT_PATH, CHAPTERS)
    print(f"Example corpus written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the script to generate the bundled PDF**

Run: `uv run python scripts/build_example_corpus.py`
Expected: prints `Example corpus written to .../data/example_book/source.pdf`, and that file now exists.

- [ ] **Step 6: Run test to verify it passes**

Run: `uv run pytest tests/test_prepare_data_cli.py -v`
Expected: PASS (4 tests)

- [ ] **Step 7: Run the full test suite**

Run: `uv run pytest -v`
Expected: PASS (all tests from every task)

- [ ] **Step 8: Commit**

```bash
git add src/stair/core/data/build_pdf.py scripts/build_example_corpus.py data/example_book/source.pdf tests/test_prepare_data_cli.py
git commit -m "feat: bundle an original example corpus and its build script"
```

---

## Self-Review Notes

- **Spec coverage:** implements spec Goal 3 (modular data prep for a new corpus) and Goal 5 (one bundled, self-contained example corpus) in full. `prepare-data`'s traceability output (Task 6, Step 3) satisfies the "print counts and every output path" requirement from this plan's Global Constraints. QA generation avoids any private dependency, satisfying spec's rejection of model_eval/lm-eval-harness-style internal-only integrations. Training (`stair train`), reporting, and eval remain out of scope, per the spec's phased rollout — covered by Plans 3 and 4.
- **Placeholder scan:** no TBD/TODO. `build_default_llm_call`'s `ValueError` when env vars are missing is a real, tested-at-the-CLI-boundary behavior (Task 6's `run_prepare_data` signature explicitly supports injecting `llm_call` to bypass it), not a stub.
- **Type consistency:** `extract_sections_from_pdf` (Task 1) returns sections `chunk_sections` (Task 2) consumes; `chunk_sections`'s output feeds both `qa_generation`'s `generate_qa_pairs_for_chunk` (Task 4, keyed on `chunk["doc_id"][0]` and `chunk["metadata"]["section_title"]`) and is otherwise unused downstream (the doc.jsonl dump uses the unchunked `sections`, not `chunks`, matching the design spec's `docs.jsonl` vs. chunked-for-QA-generation distinction). `build_toc` (Task 3) is called on `sections`, not `chunks`, so TOC entries are one-per-section, matching `qa_generation`'s `reference` field (a section id, not a chunk id) — verified consistent across Tasks 1, 3, 4, 6.
- **Review Focus:** all five items have a task and a test — no-bookmarks fallback (Task 1, `test_extract_sections_from_pdf_falls_back_when_no_bookmarks`), empty/too-short section dropped (Task 1, `test_build_final_content_drops_content_below_min_words`), unparseable QA response skipped not crashed (Task 4, `test_generate_qa_pairs_gives_up_after_max_retries`), fractions not summing to 1.0 rejected (Task 5, `test_split_rejects_fractions_not_summing_to_one`), deterministic re-run (Task 5, `test_split_is_deterministic_for_same_seed`).

## Next Plan

Plan 3: training wrapper (`stair train`) + `stair report-train` — vendor `stair/external/silt/src/` (training logic untouched), add a local single-machine `torchrun` launcher consuming `train.jsonl`/`val.jsonl`/`toc.json` from this plan's output, and the terminal training-curve/diagnostics report reading HF `trainer_state.json`.

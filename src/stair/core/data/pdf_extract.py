import pypdfium2


def normalize_title(title):
    words = title.split()
    non_digit_words = []
    for word in words:
        if not word.isdigit():
            non_digit_words.append(word)
    return " ".join(non_digit_words).strip()


def extract_bookmarks_data(pdf):
    bookmarks = list(pdf.get_toc())
    bookmarks_data = []
    last_page_index = len(pdf) - 1

    for bookmark_index, bookmark in enumerate(bookmarks):
        title = normalize_title(bookmark.get_title())
        level = bookmark.level

        dest = bookmark.get_dest()
        if not dest:
            print(f"Bookmark {bookmark_index} dest missing")
            continue

        index = dest.get_index()
        if index is None:
            print(f"Bookmark {bookmark_index} index missing")
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

    cleaned_list_content = []
    for content in list_content:
        cleaned_list_content.append(clean_content(content))

    non_skipped_list_content = []
    for content in cleaned_list_content:
        if not skip_content(content, section_title, book_title):
            non_skipped_list_content.append(content)

    non_empty_list_content = []
    for content in non_skipped_list_content:
        if content:
            non_empty_list_content.append(content)

    content = "\n".join(non_empty_list_content).strip()

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
        print("No bookmarks found in the PDF; treating it as a single section.")
        return extract_all_content_as_single_section(pdf, book_title)

    raw_sections = []
    for bookmark in bookmarks_data:
        raw_sections.append(
            {"bookmark": bookmark, "list_content": extract_content_for_bookmark(pdf, bookmark)}
        )
    remove_overlap_with_next_section(raw_sections)

    skip_sentences_set = set()
    for sentence in skip_config.get("skip_sentences", []):
        skip_sentences_set.add(sentence.lower().strip())

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

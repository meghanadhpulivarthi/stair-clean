import argparse
import json
import logging
from pathlib import Path
import pypdfium2


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def normalize_title(title):
    words = title.split()
    words = [word for word in words if not word.isdigit()]
    return " ".join(words).strip()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract Table of Contents and Content from a document."
    )
    parser.add_argument(
        "--pdf_path", type=str, required=True, help="Path to the PDF file to process."
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Path to save the extracted content.",
    )
    parser.add_argument(
        "--skip_json",
        type=str,
        default="configs/skip.json",
        help="Path to the JSON file containing sections to skip during extraction.",
    )
    parser.add_argument(
        "--book_title",
        type=str,
        default="",
        help="Title of the book being processed.",
    )

    parser.add_argument(
        "--min_words_in_a_section",
        type=int,
        default=5,
    )
    return parser.parse_args()


def extract(pdf):
    bookmarks_data = extract_bookmarks_data(pdf)
    if not bookmarks_data:
        logging.warning("No bookmarks found in the PDF.")
        return []
    raw_content = extract_content_for_all_bookmarks(pdf, bookmarks_data)
    remove_overlap_with_next_section(raw_content)
    return raw_content


def extract_bookmarks_data(pdf):
    bookmarks = list(pdf.get_toc())
    bookmarks_data = []

    for i, bookmark in enumerate(bookmarks[:-1]):
        title = normalize_title(bookmark.get_title())
        level = bookmark.level
        num_children = bookmark.get_count()

        dest = bookmark.get_dest()
        next_dest = bookmarks[i + 1].get_dest()
        if dest:
            index, (view_mode, view_pos) = dest.get_index(), dest.get_view()
            next_index, (next_view_mode, next_view_pos) = (
                next_dest.get_index(),
                next_dest.get_view(),
            )

            bookmarks_data.append(
                {
                    "title": title,
                    "next_title": bookmarks[i + 1].get_title(),
                    "level": level,
                    "num_children": num_children,
                    "page_start": index,
                    "view": (view_mode, view_pos),
                    "page_end": next_index,
                    "next_view": (next_view_mode, next_view_pos),
                }
            )
        else:
            print(f"Bookmark {i} dest missing")

    return bookmarks_data


def remove_overlap_with_next_section(list_content):
    for i, content in enumerate(list_content[:-1]):
        last_page = content["list_content"][-1]
        next_page = list_content[i + 1]["list_content"][0]

        si = last_page.find(next_page)
        if si != -1:
            content["list_content"][-1] = last_page[:si]


def extract_content(pdf, bookmark):
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
                print(f"{title=} missing")
                content = ""
        else:
            content = text_page.get_text_range()

        list_content.append(content)
    return list_content


def extract_content_for_all_bookmarks(pdf, bookmarks_data):
    all_content = []
    for bookmark in bookmarks_data:
        all_content.append(
            {"bookmark": bookmark, "list_content": extract_content(pdf, bookmark)}
        )

    return all_content


def should_skip_section(bookmark, config_skip_sections, book_title):
    if book_title and bookmark["title"].lower() == book_title.lower():
        logging.info(f"Skipping section as title is same as book title {book_title=}")
        return True
    title = bookmark["title"]
    for section in config_skip_sections.get("skip_sections", []):
        if title.lower() == section.lower():
            return True
    for section in config_skip_sections.get("skip_sections_startswith", []):
        if title.lower().startswith(section.lower()):
            return True
    return False


def skip_content(text, section_title, book_title):
    text_lower = text.lower()
    if text_lower == section_title.lower():
        logging.info(
            f"Skipping section as content is {section_title=}"
        )
        return True
    
    if book_title and text_lower == book_title.lower():
        logging.info(
            f"Skipping section as content is {book_title=}"
        )
        return True
    
    return False


def build_final_content(list_content, section_title, book_title, skip_sentences_set, min_words, remove_last_line=False):
    def clean_content(t):
        sentences = t.split("\n")
        final_sentences = []
        len_book_title = len(book_title)
        for sentence in sentences:
            src_sentence = sentence.strip().lower()
            if not src_sentence:
                continue
            if src_sentence in skip_sentences_set:
                logging.info(f"Skipping {sentence=}")
                continue

            si = src_sentence.find(book_title.lower())
            if si != -1:
                ratio_book_title_to_sentence = len_book_title / len(src_sentence)
                if ratio_book_title_to_sentence > 0.8:
                    logging.info(f"Skipping {sentence=}")
                    continue
                
            final_sentences.append(sentence)
        return "\n".join(final_sentences).strip()

    list_content = [clean_content(content) for content in list_content]

    list_content = [
        content
        for content in list_content
        if not skip_content(content, section_title, book_title)
    ]

    list_content = [content for content in list_content if content]
    content = "\n".join(list_content).strip()

    content_lower = content.lower()
    if not content_lower:
        logging.info(f"Skipping section {section_title=} as content is empty")
        return ""
    
    if content_lower == section_title.lower():
        logging.info(f"Skipping section {section_title=} as content is same as title")
        return ""

    if book_title and content_lower == book_title.lower():
        logging.info(
            f"Skipping section {section_title=} as content is same as book title"
        )
        return ""

    if len(content.split()) < min_words:
        logging.info(f"Skipping section {section_title=} as content is too short {content=}")
        return ""

    return content


def main():
    args = parse_args()
    pdf_path = args.pdf_path

    pdf = pypdfium2.PdfDocument(pdf_path)
    stats = pdf.get_metadata_dict()
    logging.info(f"PDF Metadata: {stats}")
    stats["num_pages"] = len(pdf)

    book_title = pdf.get_metadata_dict().get("Title")
    if not book_title:
        book_title = args.book_title
    logging.info(f"{book_title=}")
    logging.info(f"{pdf_path=}")

    raw_content = extract(pdf)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.skip_json, "r") as fr:
        config_skip = json.load(fr)

    with open(out_dir / "raw_content.json", "w") as fw:
        json.dump(raw_content, fw, indent=4)

    skip_sentences_set = set(
        s.lower().strip()
        for s in config_skip["skip_sentences"]
        
    )
    content_dict = {}

    fw = open(out_dir / "sections.jsonl", "w")
    for datum in raw_content:
        if should_skip_section(datum["bookmark"], config_skip, book_title):
            logging.info(f"Skipping section {datum['bookmark']['title']=} as per config")
            continue

        title = datum["bookmark"]["title"]
        content = build_final_content(datum["list_content"], title, book_title, skip_sentences_set, args.min_words_in_a_section)
        if not content:
            continue

        out = {
            "text": content,
            "metadata": {
                "book_title": book_title,
                "section_title": title,
                "page_start": datum["bookmark"]["page_start"],
                "page_end": datum["bookmark"]["page_end"],
                "bookmark": {
                    "level": datum["bookmark"]["level"],
                    "page_start": datum["bookmark"]["page_start"],
                    "page_end": datum["bookmark"]["page_end"],
                },
            },
        }
        fw.write(json.dumps(out) + "\n")
        content_dict[title] = out

    with open(out_dir / "sections.json", "w") as fw:
        json.dump(content_dict, fw, indent=4)

    with open(out_dir / "stats.json", "w") as fw:
        json.dump(stats, fw, indent=4)


if __name__ == "__main__":
    main()

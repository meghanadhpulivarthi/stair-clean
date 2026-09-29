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

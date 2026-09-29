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

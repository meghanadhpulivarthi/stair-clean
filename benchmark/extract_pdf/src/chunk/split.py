import argparse
import json
import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", type=str)
    parser.add_argument("--output_jsonl", type=str)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--token_to_char", type=int, default=4)
    parser.add_argument("--chunk_overlap", type=int, default=128)
    parser.add_argument("--add_prefix", action="store_true")
    return parser.parse_args()


def main():
    args = parse_arguments()
    logging.info(f"{args=}")

    rcts = RecursiveCharacterTextSplitter(
        chunk_size=args.max_tokens * args.token_to_char,
        chunk_overlap=args.chunk_overlap,
    )

    fw = open(args.output_jsonl, "w")

    with open(args.input_jsonl, "r") as fr:
        for line in fr:
            datum = json.loads(line)
            text = datum["text"]
            book_title = datum["metadata"]["book_title"].strip()
            section_title = datum["metadata"]["section_title"]

            prefix = f"This text is from the book: {book_title} and section: {section_title}\n"
            if not args.add_prefix:
                prefix = ""
            for chunk in rcts.split_text(text):
                datum["text"] = prefix + chunk
                fw.write(json.dumps(datum) + "\n")
    fw.close()


if __name__ == "__main__":
    main()

import argparse
import json
import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", type=str)
    parser.add_argument("--output_jsonl", type=str)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--token_to_char", type=float, default=4.0)
    parser.add_argument("--chunk_overlap", type=int, default=128)
    parser.add_argument("--add_prefix", action="store_true")
    parser.add_argument("--col_prefix", type=str, default="doc_id")
    return parser.parse_args()


def count_lines(file_path):
    with open(file_path, "r") as f:
        for i, _ in enumerate(f):
            pass
    return i + 1

def main():
    args = parse_arguments()
    logging.info(f"{args=}")

    chunk_size = int(args.max_tokens * args.token_to_char)
    chunk_overlap = int(args.chunk_overlap * args.token_to_char)    

    rcts = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    fw = open(args.output_jsonl, "w")

    with open(args.input_jsonl, "r") as fr:
        for line in tqdm(fr, total=count_lines(args.input_jsonl), desc="Processing lines"):
            datum = json.loads(line)
            text = datum["text"]

            if args.add_prefix:
                prefix = f"This document is about {datum["metadata"][args.col_prefix]}. "
                
            else:
                prefix = ""

            for chunk in rcts.split_text(text):
                datum["text"] = prefix + chunk
                fw.write(json.dumps(datum) + "\n")
    fw.close()


if __name__ == "__main__":
    main()

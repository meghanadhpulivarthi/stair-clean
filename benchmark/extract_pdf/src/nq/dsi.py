import argparse
import json
import logging

from tqdm import tqdm
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def extract_title(url):
    query = urlparse(url).query
    params = parse_qs(query)
    return params["title"][0].strip()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_jsonl", type=str)
    parser.add_argument("--train_jsonl", type=str)
    parser.add_argument("--qs_jsonl", type=str)
    parser.add_argument("--max_docs", type=int, default=-1)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--token_to_char", type=float, default=4.0)
    parser.add_argument("--chunk_overlap", type=int, default=32)
    return parser.parse_args()


def count_lines(file_path):
    with open(file_path, "r") as f:
        return sum(1 for line in f)


def main():
    args = parse_args()

    title_set = set()

    Path(args.train_jsonl).parent.mkdir(parents=True, exist_ok=True)
    Path(args.qs_jsonl).parent.mkdir(parents=True, exist_ok=True)

    rcts = RecursiveCharacterTextSplitter(
        chunk_size=int(args.max_tokens * args.token_to_char),
        chunk_overlap=int(args.chunk_overlap * args.token_to_char),
    )

    num_lines = count_lines(args.in_jsonl)
    fw_train = open(args.train_jsonl, "w")
    fw_qs = open(args.qs_jsonl, "w")

    num_docs, num_qs = 0, 0

    with open(args.in_jsonl) as f:
        for line in tqdm(f, total=num_lines):
            data = json.loads(line)
            title = extract_title(data["document_url"])

            qs = data["question_text"].strip()
            qs_datum = {
                "test_id": data["example_id"],
                "question": qs,
                "doc_id": title,
            }
            num_qs += 1
            fw_qs.write(json.dumps(qs_datum) + "\n")
            if title in title_set:
                logging.info(f"{title=} {data['document_url']=} exists, skipping")
                continue

            # text = f"This content belongs to the document titled :{title}" + "\n\n" + data["document_text"].strip()
            text = data["document_text"]
            text = BeautifulSoup(text, "lxml").text.strip()
            if not text:
                logging.info(f"{title=} {data['document_url']=} empty text, skipping")
                continue

            for chunk in rcts.split_text(text):
                text = chunk.strip()
                datum = {"prompt": "Question: " + text + " Title:", "completion": f"{title}"}
                fw_train.write(json.dumps(datum) + "\n")
                title_set.add(title)
                num_docs += 1

            if 0 < args.max_docs <= num_docs:
                break

if __name__ == "__main__":
    main()
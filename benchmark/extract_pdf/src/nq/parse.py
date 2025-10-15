import argparse
import json
import logging

from tqdm import tqdm
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from bs4 import BeautifulSoup


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def extract_title(url):
    query = urlparse(url).query
    params = parse_qs(query)
    return params["title"][0].strip()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_jsonl", type=str)
    parser.add_argument("--sections_jsonl", type=str)
    parser.add_argument("--qs_jsonl", type=str)
    parser.add_argument("--max_docs", type=int, default=-1)
    return parser.parse_args()

def count_lines(file_path):
    with open(file_path, 'r') as f:
        return sum(1 for line in f)

def main():
    args = parse_args()

    title_set = set()

    Path(args.sections_jsonl).parent.mkdir(parents=True, exist_ok=True)
    Path(args.qs_jsonl).parent.mkdir(parents=True, exist_ok=True)



    num_lines = count_lines(args.in_jsonl)
    fw_sections = open(args.sections_jsonl, "w")
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
            text = data["document_text"].strip()
            text = BeautifulSoup(text, "lxml").text
            if not text:
                logging.info(f"{title=} {data['document_url']=} empty text, skipping")
                continue
            text = text.strip()
            section_datum = {
                "text": text,
                "metadata": {
                    "doc_id": title,
                    "url": data["document_url"],
                }
            }
            fw_sections.write(json.dumps(section_datum) + "\n")
            title_set.add(title)
            num_docs += 1
            if args.max_docs > 0 and num_docs >= args.max_docs:
                logging.info(f"Reached max_docs limit: {args.max_docs}, stopping.")
                break

    stats = {
        "num_docs": num_docs,
        "num_qs": num_qs,
    }
    logging.info(f"{stats=}")


if __name__ == "__main__":
    main()
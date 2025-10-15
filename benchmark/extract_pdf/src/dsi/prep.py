import argparse
import json
import logging
from tqdm import tqdm
from pathlib import Path


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks_jsonl", type=str, required=True)
    parser.add_argument("--train_jsonl", type=str, required=True)
    args = parser.parse_args()
    return args


def count_lines(file_path):
    with open(file_path, "r") as f:
        return sum(1 for _ in f)

def main():
    args = parse_args()
    logging.info(f"{args=}")

    Path(args.train_jsonl).parent.mkdir(parents=True, exist_ok=True)
    fw_train = open(args.train_jsonl, "w")
    with open(args.chunks_jsonl, "r") as f:
        for line in tqdm(f, total=count_lines(args.chunks_jsonl), desc="Processing chunks_jsonl"):
            data = json.loads(line)
            text = data["text"]
            prompt = f"Question: {text}\nDocID:"
            completion = data["metadata"]["section_title"]
            datum = {
                "prompt": prompt,
                "completion": completion
            }
            fw_train.write(json.dumps(datum) + "\n")
            

if __name__ == "__main__":
    main()
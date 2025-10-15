import argparse
import json
import logging

from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

def count_lines(file_path):
    with open(file_path, "r") as f:
        for i, _ in enumerate(f):
            pass
    return i + 1

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", type=str)
    parser.add_argument("--model_id", default="mistralai/Mistral-7B-v0.3")
    parser.add_argument("--out_dir")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.info(f"{args=}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    token_count = {}

    num_tokens = 0
    num_words = 0
    num_chars = 0

    with open(args.input_jsonl, "r") as fr:
        for line in tqdm(fr, total=count_lines(args.input_jsonl), desc="Processing lines"):
            data = json.loads(line)
            text = data["text"]

            tokens = tokenizer.encode(text, add_special_tokens=False)
            num_tokens += len(tokens)
            num_words += len(text.split())
            num_chars += len(text)

            for token in tokens:
                token_count[token] = token_count.get(token, 0) + 1
    
    stats = {
        "num_words": num_words,
        "num_tokens": num_tokens,
        "avg_tokens_per_word": num_tokens / num_words,
        "avg_chars_per_token": num_chars / num_tokens,
    }

    out_dir = Path(args.out_dir)
    with open(out_dir / "token_stats.json", "w") as fw:
        json.dump(stats, fw, indent=2)


    fw = open(out_dir / "tokens.jsonl", "w")
    for token, count in sorted(token_count.items(), key=lambda x: x[1], reverse=True):
        token_str = tokenizer.decode([token])
        out = {
            "token": token,
            "token_str": token_str,
            "count": count,
        }
        fw.write(json.dumps(out) + "\n")
    fw.close()


if __name__ == "__main__":
    main()
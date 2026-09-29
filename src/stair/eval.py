import json
from pathlib import Path

from stair.core.eval.inference import build_generate_call, load_finetuned_model
from stair.core.eval.metrics import compute_metrics
from stair.core.eval.parse_output import parse_predicted_sections
from stair.core.prompts import SYSTEM_PROMPT, USER_PROMPT
from stair.run_header import print_run_header


def check_required_eval_inputs(data_dir, run_dir):
    data_dir_path = Path(data_dir)
    required_data_file_names = ["test.jsonl", "toc.json"]
    for required_file_name in required_data_file_names:
        required_file_path = data_dir_path / required_file_name
        if not required_file_path.exists():
            raise FileNotFoundError(
                f"Expected {required_file_name} under {data_dir} "
                f"(run `stair prepare-data` first to produce it)"
            )

    checkpoint_best_path = Path(run_dir) / "checkpoint-best"
    if not checkpoint_best_path.exists():
        raise FileNotFoundError(
            f"Expected checkpoint-best under {run_dir} "
            f"(run `stair train` first, and make sure it completed successfully)"
        )


def load_toc(toc_json_path):
    with open(toc_json_path, "r") as toc_file:
        toc = json.load(toc_file)

    title = toc["title"]
    toc_lines = []
    title_to_id_map = {}
    for node in toc["table_of_contents"]:
        toc_line = f"{node['section_num']} {node['title']}"
        toc_lines.append(toc_line)
        title_to_id_map[toc_line] = node["id"]

    return title, "\n".join(toc_lines), title_to_id_map


def read_test_records(test_jsonl_path):
    test_records = []
    with open(test_jsonl_path, "r") as test_file:
        for line in test_file:
            test_records.append(json.loads(line))
    return test_records


def build_default_generate_call(run_dir, config):
    run_config_path = Path(run_dir) / "config.json"
    with open(run_config_path, "r") as run_config_file:
        run_config = json.load(run_config_file)

    base_model_name = run_config["model"]["name"]
    adapter_path = str(Path(run_dir) / "checkpoint-best")
    max_new_tokens = config["eval"]["max_new_tokens"]

    model, tokenizer = load_finetuned_model(base_model_name, adapter_path)
    return build_generate_call(model, tokenizer, max_new_tokens)


def run_eval(run_dir, data_dir, config, generate_call=None):
    print_run_header("stair eval", config)

    check_required_eval_inputs(data_dir, run_dir)

    eval_output_dir = Path(run_dir) / "eval"
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    book_title, toc_text, title_to_id_map = load_toc(Path(data_dir) / "toc.json")
    print(f"Loaded table of contents for: {book_title}")

    test_records = read_test_records(Path(data_dir) / "test.jsonl")
    print(f"Loaded {len(test_records)} test examples")

    if generate_call is None:
        generate_call = build_default_generate_call(run_dir, config)

    ks = config["eval"]["ks"]
    metrics_totals = {}
    for k in ks:
        for metric_name in ["recall", "match", "precision", "f1", "mrr", "ndcg"]:
            metrics_totals[f"{metric_name}@{k}"] = 0.0

    total_hallucinations = 0
    total_predicted_entries = 0
    result_records = []

    test_count = len(test_records)
    results_path = eval_output_dir / "eval_results.jsonl"
    with open(results_path, "w") as results_file:
        for example_index, test_record in enumerate(test_records):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": USER_PROMPT.format(
                        title=book_title, toc=toc_text, question=test_record["question"]
                    ),
                },
            ]

            raw_generation = generate_call(messages)
            predicted_ids, hallucination_count = parse_predicted_sections(
                raw_generation, title_to_id_map
            )
            total_hallucinations += hallucination_count
            total_predicted_entries += hallucination_count + len(predicted_ids)

            gold_ids = test_record["reference"]
            per_k_metrics = {}
            for k in ks:
                k_metrics = compute_metrics(predicted_ids, gold_ids, k)
                per_k_metrics.update(k_metrics)
                for metric_name, metric_value in k_metrics.items():
                    metrics_totals[metric_name] += metric_value

            result_record = {
                "question": test_record["question"],
                "reference": gold_ids,
                "predicted_ids": predicted_ids,
                "raw_generation": raw_generation,
                "metrics": per_k_metrics,
            }
            result_records.append(result_record)
            results_file.write(json.dumps(result_record) + "\n")

            print(f"Evaluated {example_index + 1}/{test_count} examples")

    print(f"Results saved: {results_path}")

    averaged_metrics = {}
    for metric_name, total_value in metrics_totals.items():
        averaged_metrics[metric_name] = total_value / test_count if test_count else 0.0

    hallucination_rate = (
        total_hallucinations / total_predicted_entries if total_predicted_entries else 0.0
    )

    summary = {
        "count": test_count,
        "hallucination_rate": hallucination_rate,
        "metrics": averaged_metrics,
    }

    metrics_path = eval_output_dir / "eval_metrics.json"
    with open(metrics_path, "w") as metrics_file:
        json.dump(summary, metrics_file, indent=2)
    print(f"Metrics saved: {metrics_path}")

    return summary

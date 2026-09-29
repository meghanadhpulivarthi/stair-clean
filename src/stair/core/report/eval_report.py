import json
from pathlib import Path


def load_eval_metrics(run_dir):
    metrics_path = Path(run_dir) / "eval" / "eval_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"No eval_metrics.json found under {run_dir} (run `stair eval` first)"
        )
    with open(metrics_path, "r") as metrics_file:
        return json.load(metrics_file)


def extract_ks_from_metrics(metrics):
    # derives the k values actually computed for this run from the metrics
    # dict's own keys (e.g. "recall@1" -> 1), instead of reading eval.ks
    # from the base config - a run started with `--config override.yaml`
    # may have used different eval.ks than the base default, and reading
    # from the base config would silently render a table of "-"
    # placeholders with no error. flag_weak_spots already derives its
    # smallest-k this way; this brings render_metrics_table's ks in line
    # with that same pattern.
    ks = set()
    for metric_key in metrics:
        if "@" in metric_key:
            k_part = metric_key.split("@")[1]
            ks.add(int(k_part))
    return sorted(ks)


def render_metrics_table(summary, ks):
    print(f"Evaluated {summary['count']} examples")
    print(f"Hallucination rate: {summary['hallucination_rate']:.2%}")
    print("")

    metric_names = ["recall", "match", "precision", "f1", "mrr", "ndcg"]
    header_columns = ["metric"]
    for k in ks:
        header_columns.append(f"@{k}")
    print("  ".join(header_columns))

    for metric_name in metric_names:
        row_columns = [metric_name]
        for k in ks:
            metric_key = f"{metric_name}@{k}"
            metric_value = summary["metrics"].get(metric_key)
            if metric_value is None:
                row_columns.append("-")
            else:
                row_columns.append(f"{metric_value:.3f}")
        print("  ".join(row_columns))


def flag_weak_spots(summary, eval_config):
    warnings = []

    weak_recall_threshold = eval_config["weak_recall_threshold"]
    smallest_recall_key = None
    smallest_k = None
    for metric_key in summary["metrics"]:
        if metric_key.startswith("recall@"):
            this_k = int(metric_key.split("@")[1])
            if smallest_k is None or this_k < smallest_k:
                smallest_k = this_k
                smallest_recall_key = metric_key
    if smallest_recall_key is not None:
        smallest_recall_value = summary["metrics"][smallest_recall_key]
        if smallest_recall_value < weak_recall_threshold:
            warnings.append(
                f"{smallest_recall_key} is {smallest_recall_value:.2f}, below the "
                f"{weak_recall_threshold} threshold — the model is missing relevant "
                f"sections even at the most forgiving cutoff."
            )

    high_hallucination_rate_threshold = eval_config["high_hallucination_rate_threshold"]
    if summary["hallucination_rate"] > high_hallucination_rate_threshold:
        warnings.append(
            f"Hallucination rate is {summary['hallucination_rate']:.2%}, above the "
            f"{high_hallucination_rate_threshold:.0%} threshold — the model is often "
            f"generating section titles that don't exist in the table of contents."
        )

    return warnings

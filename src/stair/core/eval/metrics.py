import math


def compute_dcg(predicted_ids, gold_ids, k):
    discounted_cumulative_gain = 0.0
    for position, predicted_id in enumerate(predicted_ids[:k]):
        relevance = 1.0 if predicted_id in gold_ids else 0.0
        discounted_cumulative_gain += (2 ** relevance - 1) / math.log2(position + 2)
    return discounted_cumulative_gain


def compute_idcg(gold_ids, k):
    ideal_relevance_count = min(len(gold_ids), k)
    ideal_discounted_cumulative_gain = 0.0
    for position in range(ideal_relevance_count):
        ideal_discounted_cumulative_gain += (2 ** 1.0 - 1) / math.log2(position + 2)
    return ideal_discounted_cumulative_gain


def compute_metrics(predicted_ids, gold_ids, k):
    top_k_predicted_ids = predicted_ids[:k]
    hits = len(set(top_k_predicted_ids) & set(gold_ids))

    recall = hits / float(len(gold_ids)) if gold_ids else 0.0
    match = 1 if hits > 0 else 0
    precision = hits / float(k)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    first_correct_rank = None
    for position, predicted_id in enumerate(top_k_predicted_ids):
        if predicted_id in gold_ids:
            first_correct_rank = position + 1
            break
    mean_reciprocal_rank = 1.0 / first_correct_rank if first_correct_rank else 0.0

    discounted_cumulative_gain = compute_dcg(predicted_ids, gold_ids, k)
    ideal_discounted_cumulative_gain = compute_idcg(gold_ids, k)
    normalized_discounted_cumulative_gain = (
        discounted_cumulative_gain / ideal_discounted_cumulative_gain
        if ideal_discounted_cumulative_gain > 0
        else 0.0
    )

    return {
        f"recall@{k}": recall,
        f"match@{k}": match,
        f"precision@{k}": precision,
        f"f1@{k}": f1,
        f"mrr@{k}": mean_reciprocal_rank,
        f"ndcg@{k}": normalized_discounted_cumulative_gain,
    }

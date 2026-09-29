from stair.core.eval.metrics import compute_metrics


def test_compute_metrics_all_predictions_correct():
    metrics = compute_metrics(predicted_ids=["1", "2"], gold_ids=["1", "2"], k=2)
    assert metrics["recall@2"] == 1.0
    assert metrics["match@2"] == 1
    assert metrics["precision@2"] == 1.0
    assert metrics["mrr@2"] == 1.0
    assert metrics["ndcg@2"] == 1.0


def test_compute_metrics_no_predictions_correct():
    metrics = compute_metrics(predicted_ids=["9", "8"], gold_ids=["1", "2"], k=2)
    assert metrics["recall@2"] == 0.0
    assert metrics["match@2"] == 0
    assert metrics["precision@2"] == 0.0
    assert metrics["mrr@2"] == 0.0
    assert metrics["ndcg@2"] == 0.0


def test_compute_metrics_partial_match_ranked_first():
    metrics = compute_metrics(predicted_ids=["1", "9"], gold_ids=["1", "2"], k=2)
    assert metrics["recall@2"] == 0.5
    assert metrics["match@2"] == 1
    assert metrics["precision@2"] == 0.5
    assert metrics["mrr@2"] == 1.0


def test_compute_metrics_partial_match_ranked_second():
    metrics = compute_metrics(predicted_ids=["9", "1"], gold_ids=["1", "2"], k=2)
    assert metrics["mrr@2"] == 0.5


def test_compute_metrics_empty_gold_ids_does_not_divide_by_zero():
    metrics = compute_metrics(predicted_ids=["1", "2"], gold_ids=[], k=2)
    assert metrics["recall@2"] == 0.0
    assert metrics["match@2"] == 0
    assert metrics["ndcg@2"] == 0.0


def test_compute_metrics_respects_k_cutoff():
    metrics = compute_metrics(predicted_ids=["9", "9", "1"], gold_ids=["1"], k=2)
    assert metrics["match@2"] == 0
    assert metrics["recall@2"] == 0.0

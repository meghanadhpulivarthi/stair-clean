import pytest

from stair.core.data.split import split_qa_pairs


def make_qa_pairs(count):
    return [{"question": f"q{i}", "answer": f"a{i}", "reference": [str(i)]} for i in range(count)]


def test_split_sizes_match_fractions():
    qa_pairs = make_qa_pairs(100)
    train, val, test = split_qa_pairs(qa_pairs, train_fraction=0.8, val_fraction=0.1, test_fraction=0.1, seed=42)
    assert len(train) == 80
    assert len(val) == 10
    assert len(test) == 10


def test_split_covers_every_pair_exactly_once():
    qa_pairs = make_qa_pairs(50)
    train, val, test = split_qa_pairs(qa_pairs, train_fraction=0.8, val_fraction=0.1, test_fraction=0.1, seed=42)
    all_questions = sorted(item["question"] for item in train + val + test)
    expected_questions = sorted(item["question"] for item in qa_pairs)
    assert all_questions == expected_questions


def test_split_is_deterministic_for_same_seed():
    qa_pairs = make_qa_pairs(50)
    first_train, first_val, first_test = split_qa_pairs(qa_pairs, train_fraction=0.8, val_fraction=0.1, test_fraction=0.1, seed=42)
    second_train, second_val, second_test = split_qa_pairs(qa_pairs, train_fraction=0.8, val_fraction=0.1, test_fraction=0.1, seed=42)
    assert first_train == second_train
    assert first_val == second_val
    assert first_test == second_test


def test_split_rejects_fractions_not_summing_to_one():
    qa_pairs = make_qa_pairs(10)
    with pytest.raises(ValueError, match="sum to 1.0"):
        split_qa_pairs(qa_pairs, train_fraction=0.9, val_fraction=0.1, test_fraction=0.1, seed=42)

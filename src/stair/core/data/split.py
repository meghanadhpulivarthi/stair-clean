import random


def split_qa_pairs(qa_pairs, train_fraction, val_fraction, test_fraction, seed):
    fraction_sum = train_fraction + val_fraction + test_fraction
    if abs(fraction_sum - 1.0) > 1e-6:
        raise ValueError(
            f"train_fraction + val_fraction + test_fraction must sum to 1.0, got {fraction_sum}"
        )

    shuffled_pairs = list(qa_pairs)
    random.Random(seed).shuffle(shuffled_pairs)

    total_count = len(shuffled_pairs)
    train_count = int(total_count * train_fraction)
    val_count = int(total_count * val_fraction)

    train = shuffled_pairs[:train_count]
    val = shuffled_pairs[train_count:train_count + val_count]
    test = shuffled_pairs[train_count + val_count:]

    return train, val, test

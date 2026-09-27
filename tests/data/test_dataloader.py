import numpy as np
import pytest

from minidl.data import ArrayDataset, DataLoader, train_val_split


def dataset(n=10):
    return ArrayDataset(np.arange(n)[:, None], np.arange(n))


def test_partial_batch_and_alignment():
    loader = DataLoader(dataset(), 4)
    batches = list(loader)
    assert len(loader) == 3
    assert [len(y) for _, y in batches] == [4, 4, 2]
    for x, y in batches:
        np.testing.assert_array_equal(x[:, 0], y)


def test_drop_last_empty_and_invalid():
    assert len(list(DataLoader(dataset(), 4, drop_last=True))) == 2
    assert list(DataLoader(dataset(0))) == []
    with pytest.raises(ValueError):
        DataLoader(dataset(), 0)
    with pytest.raises(ValueError):
        ArrayDataset([1], [1, 2])


def test_shuffle_reproducible_across_epochs():
    a = DataLoader(dataset(30), 7, shuffle=True, seed=2)
    b = DataLoader(dataset(30), 7, shuffle=True, seed=2)
    orders = []
    for _ in range(2):
        order_a = np.concatenate([y for _, y in a])
        order_b = np.concatenate([y for _, y in b])
        np.testing.assert_array_equal(order_a, order_b)
        np.testing.assert_array_equal(np.sort(order_a), np.arange(30))
        orders.append(order_a)
    assert not np.array_equal(*orders)


def test_split_disjoint_reproducible():
    train, val = train_val_split(dataset(), 0.2, seed=4)
    again, _ = train_val_split(dataset(), 0.2, seed=4)
    assert len(train) == 8 and len(val) == 2
    assert not set(train.y) & set(val.y)
    np.testing.assert_array_equal(train.x, again.x)
    with pytest.raises(ValueError):
        train_val_split(dataset(), 1)

import numpy as np
import pytest

from minidl.data.text import CharacterVocabulary, sequence_dataset, split_text


def test_text_roundtrip_and_windows():
    text = "a robot reads. " * 20
    vocabulary = CharacterVocabulary(text)
    assert vocabulary.decode(vocabulary.encode(text)) == text
    train, val = split_text(text)
    assert train + val == text
    tokens = vocabulary.encode(train)
    a = sequence_dataset(tokens, 5, 8, 42)
    b = sequence_dataset(tokens, 5, 8, 42)
    np.testing.assert_array_equal(a.x, b.x)
    np.testing.assert_array_equal(a.x[:, 1:], a.y[:, :-1])
    assert a.x.shape == (8, 5)
    with pytest.raises(ValueError):
        sequence_dataset(np.arange(3), 3)

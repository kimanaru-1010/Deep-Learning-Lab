import numpy as np

from .dataset import ArrayDataset


class CharacterVocabulary:
    def __init__(self, text):
        self.characters = sorted(set(text))
        if not self.characters:
            raise ValueError("Vocabulary cannot be empty")
        self.stoi = {char: i for i, char in enumerate(self.characters)}

    def __len__(self):
        return len(self.characters)

    def encode(self, text):
        return np.array([self.stoi[char] for char in text], dtype=np.int64)

    def decode(self, tokens):
        return "".join(self.characters[int(token)] for token in tokens)


def split_text(text, val_fraction=0.1):
    """Contiguous split BEFORE constructing windows prevents overlap leakage."""
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be in (0,1)")
    cut = int(len(text) * (1 - val_fraction))
    if cut == 0 or cut == len(text):
        raise ValueError("Text too short for split")
    return text[:cut], text[cut:]


def sequence_dataset(tokens, length=16, count=256, seed=42):
    if length <= 0 or count <= 0 or len(tokens) <= length:
        raise ValueError("Need positive length/count and at least length+1 tokens")
    starts = np.random.default_rng(seed).integers(0, len(tokens) - length, size=count)
    x = np.stack([tokens[i : i + length] for i in starts])
    y = np.stack([tokens[i + 1 : i + length + 1] for i in starts])
    return ArrayDataset(x, y)

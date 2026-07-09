"""
Minimal BPE: count pairs, find the most frequent, merge.
Run this on a tiny corpus to see the algorithm in action.
"""

from collections import Counter


def get_pairs(vocab: dict[str, int]) -> Counter:
    """Count every adjacent symbol pair across the corpus."""
    pairs = Counter()
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs


def merge_pair(pair: tuple[str, str], vocab: dict[str, int]) -> dict[str, int]:
    """Merge the chosen pair everywhere it appears in the vocab."""
    merged = " ".join(pair)
    replacement = "".join(pair)
    return {
        word.replace(merged, replacement): freq
        for word, freq in vocab.items()
    }


# Start: every word split into characters, with end-of-word marker </w>
vocab = {
    "l o w </w>": 5,
    "l o w e r </w>": 2,
    "n e w e s t </w>": 6,
    "w i d e s t </w>": 3,
}

NUM_MERGES = 6

for i in range(NUM_MERGES):
    pairs = get_pairs(vocab)
    best_pair = max(pairs, key=pairs.get)
    vocab = merge_pair(best_pair, vocab)
    print(f"Merge {i+1}: {best_pair!r:25s} → '{''.join(best_pair)}'")

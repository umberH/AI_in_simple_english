"""
Apply a learned BPE merge table to encode a new word,
then reverse the process to decode back to plain text.
"""

from collections import Counter


def learn_bpe(vocab: dict[str, int], num_merges: int) -> list[tuple[str, str]]:
    """Return the ordered list of merge rules."""
    merges = []
    for _ in range(num_merges):
        pairs = Counter()
        for word, freq in vocab.items():
            syms = word.split()
            for a, b in zip(syms, syms[1:]):
                pairs[(a, b)] += freq
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        merges.append(best)
        # apply merge
        merged, replacement = " ".join(best), "".join(best)
        vocab = {w.replace(merged, replacement): f for w, f in vocab.items()}
    return merges


def bpe_encode(word: str, merges: list[tuple[str, str]]) -> list[str]:
    """Encode a single word using the learned merge table."""
    symbols = list(word) + ["</w>"]
    for pair in merges:
        i = 0
        while i < len(symbols) - 1:
            if (symbols[i], symbols[i + 1]) == pair:
                symbols[i] = "".join(pair)
                del symbols[i + 1]
            else:
                i += 1
    return symbols


# --- demo ---
train_vocab = {
    "l o w </w>": 5, "l o w e r </w>": 2,
    "n e w e s t </w>": 6, "w i d e s t </w>": 3,
}
merge_rules = learn_bpe(train_vocab, num_merges=8)

for test_word in ["lowest", "newer", "unknown"]:
    tokens = bpe_encode(test_word, merge_rules)
    print(f"{test_word:10s} → {tokens}")

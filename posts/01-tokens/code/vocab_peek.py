"""
Post 1 — Tokens: How Words Become Numbers
Peek at the token vocabulary: which text does each ID represent?

Low IDs → common characters and short words (added to vocabulary first).
High IDs → rarer, more specific chunks.

Install: pip install tiktoken
"""
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

sample_ids = [0, 1, 100, 1000, 10000, 50000, 99000]

print(f"{'ID':>8}  →  token text")
print("-" * 30)
for token_id in sample_ids:
    text = enc.decode([token_id])
    print(f"{token_id:>8}  →  {repr(text)}")

# The ID itself carries no meaning — it's just a row number.
# Meaning lives in the embedding matrix row that the ID points to.

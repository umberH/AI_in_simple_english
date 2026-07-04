"""
Post 1 — Tokens: How Words Become Numbers
Demonstrates how text is split into token IDs using GPT-4's tokenizer.

Install: pip install tiktoken
"""
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")  # GPT-4's tokenizer

text = "مصنوعی ذہانت"
token_ids = enc.encode(text)

# Each ID decoded back to the text chunk it represents
tokens = [enc.decode([t]) for t in token_ids]

print(f"Input:      {repr(text)}")
print(f"Tokens:     {tokens}")
print(f"Token IDs:  {token_ids}")
print(f"\n{len(text)} characters  →  {len(token_ids)} tokens")

# Spaces are baked INTO tokens — " is" starts with a space.
# "ChatGPT" splits into 3 because the compound word is rare in training data.

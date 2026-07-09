# Social Post — Post #2: BPE Tokenisation from Scratch

---

GPT-4 didn't learn English words. It learned 100,000 sub-word pieces — and the algorithm that chose them is 50 lines of Python.

It's called Byte-Pair Encoding. Here's how it works:

Start with every character as its own token. Then count every adjacent pair in your training text. Merge the most frequent pair into one new token. Repeat 100,000 times.

That's it. The algorithm discovers "est", "ing", "tion" — actual English morphemes — without knowing a single grammar rule. Frequency alone.

The merge table you end up with is your tokenizer. Encoding a new word means replaying those merges in order. Words that never appeared in training still get tokenized — they just fall back to characters, one by one.

This is why "ChatGPT" splits into three tokens and "the" is always one. Not arbitrary — a direct consequence of what appeared most often in the training corpus.

Post 2 of *AI in Simple English* — build BPE from scratch, encode your first word, and see why French text costs 3× more tokens than English in an English-trained model. [link]

---

**Platform notes:** LinkedIn + Twitter/X. On LinkedIn, break into 4–5 short paragraphs with line breaks between each. Lead with the first line as a standalone hook. On X/Twitter, use the first two paragraphs as a thread opener, then link to the article.

# Feedback Log — Post #1: Tokens: how words become numbers

## Round 1 — 2026-07-04 08:59

Hook is good but code example comes too late. Ending teaser for Post 2 needs to be more exciting — tell me what I will BUILD not just what I will learn.

### Revision completed — 2026-07-05
- Code moved into the opening: runs immediately after "your words get shredded" — no section header delay, reader sees the proof in the first scroll
- Condensed the code to 4 lines (removed the token_ids print, kept only the tokens output) so the hook stays punchy
- What's Next rewritten to lead with BUILD: "you'll build your own tokenizer from scratch — 60 lines of Python, no libraries" — shows the tangible artifact first, then explains why it matters

## Round 2 — 2026-07-05 09:19

 this writing with hyphen is not my tyle  ChatGPT — one thing to you — splits into three.. update this examoke froom chatgpt is amazing to ai in plain english. characters are too small andm words are too large needs to be backed by the literature techniques previously used like term freequency and char encoding etc can add some reference to back gpt and llama 3 tokens? can add some of urdu example to run through the other language example? 

### Revision completed — 2026-07-05
- Removed em-dash parenthetical style from prose ("ChatGPT — one thing to you — splits into three" rewritten as plain declarative sentences throughout)
- Changed primary code example from "ChatGPT is amazing." to "AI in Simple English." (series title); kept "ChatGPT is amazing." as an immediate contrast example to preserve the split-into-three pedagogical point
- Updated Mermaid diagram to use "AI in Simple English." as the demo text
- Added historical/literature backing to the chars-too-small/words-too-large section: TF, IDF, bag-of-words context from early 1990s; BPE origin (Gage 1994) and Sennrich et al. 2016 citation for NLP adaptation
- Added references for token counts: GPT-4 ~100,000 (OpenAI technical report) and Llama 3 128,000 (Meta model card)
- Added Urdu tokenization example ("مصنوعی ذہانت") with code snippet and explanation that Perso-Arabic script fragments into byte-level pieces; added Urdu to Key Takeaways bullet on multilingual cost

# Yessenov Foundation — AI Grants Assistant

A Streamlit chatbot (on `gemma4`) that answers questions about the Shakhmardan
Yessenov Foundation's grants, programs, and scholarships, using data scraped from
[yessenovfoundation.org](https://yessenovfoundation.org). Built for YDL 2026,
Week 1, Day 5. See `requirements.md` for the full spec.

## Architecture

```
scrape.py        # 1. collect data from the site -> data/corpus.json
build_index.py   # 2. embed the corpus (text-1024) -> data/embeddings.npz
app.py           # 3. Streamlit chat: RAG retrieval -> gemma4 -> answer
config.py        # secrets (from env / specs/creds.txt) + endpoints
llm.py           # gemma4 chat-completions client
rag.py           # embedding + cosine-similarity retrieval
prompts.py       # grounded, anti-hallucination system prompt
mailer.py        # optional: self-only, button-triggered email summary
```

## Setup

```bash
source ../.venv/bin/activate          # or your venv
pip install -r requirements.txt
```

API keys are read from environment variables, falling back to `specs/creds.txt`
(gitignored) for the alem.ai keys:

- `CHAT_API_KEY`, `EMBED_API_KEY` — alem.ai (auto-loaded from `specs/creds.txt`)
- `MAILERSEND_API_KEY` — optional, enables the email feature
- `ADMIN_EMAIL` — your own email (email is sent only here)

## Run

```bash
python scrape.py        # collect data (run once)
python build_index.py   # build the RAG index (run once)
streamlit run app.py    # launch the chat
```

## Testing (requirements.md F3)

Test on 3–4 real questions **and** at least one out-of-data question to confirm
the bot says "I don't know" instead of fabricating:

1. *In-data:* "What grants does the foundation offer?"
2. *In-data:* "Who is eligible to apply for a scholarship?"
3. *In-data:* "How do I apply for a grant?"
4. **Out-of-data (honesty check):** "What is the exact grant deadline in 2027?"
   or "How much is the CEO's salary?" — the bot must refuse / say it doesn't know,
   not invent a deadline or amount.

Readiness criterion: a bot that confidently fabricates grant deadlines,
requirements, or amounts is **not** considered finished.

## Demo (4 min + 1 min Q&A)

- Answer the instructor's common question live.
- Show one question where the bot breaks (e.g. an out-of-data fact).
- Explain: how data was collected (scraping), the approach (RAG + grounded prompt),
  and beyond-minimum features (RAG, email).

## Security

`specs/creds.txt` contains live API keys and is gitignored — never commit it.
Email is sent **only** to your own `ADMIN_EMAIL`, and **only** on an explicit
button press (never in a loop).

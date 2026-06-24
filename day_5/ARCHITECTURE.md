# Architecture — Yessenov Foundation AI Grants Assistant

A Streamlit chatbot (on the `gemma4` LLM) that answers questions about the
Shakhmardan Yessenov Foundation's grants, programs, scholarships, and news,
grounded **only** in data collected from `yessenovfoundation.org`. Built for
YDL 2026, Week 1, Day 5. See `requirements.md` for the spec and `CLAUDE.md` for
the working rules.

The system is a classic **RAG (Retrieval-Augmented Generation)** pipeline:

```
                          OFFLINE (run once / on refresh)
  yessenovfoundation.org
        │  sitemap.xml (≈592 URLs)
        ▼
  scrape.py ───────────► data/corpus.json     (3,038 text chunks + metadata)
        │  (HTML + PDFs)
        ▼
  build_index.py ──────► data/embeddings.npz   (3,038 × 1024 vectors + chunks)
        (text-1024)

                          ONLINE (per user question)
  user question
        │
        ▼
  app.py ── retrieve (rag.py) ──► top-k chunks ──► grounded prompt (prompts.py)
        │                                                   │
        │                                                   ▼
        └──────────── conversation history ───────► llm.py → gemma4 → answer
                                                            │
                                  optional, on button press ▼
                                              mailer.py → MailerSend → admin email
```

There are two distinct stages: an **offline data pipeline** (scrape → chunk →
embed → store) and an **online query pipeline** (retrieve → ground → generate).

---

## File map — what each module does

| File | Responsibility |
|------|----------------|
| `config.py` | Loads API keys (from env or `specs/creds.txt`), endpoint URLs, model names, file paths. Single source of configuration. |
| `scrape.py` | **Data collection.** Crawls the full sitemap, extracts text from HTML pages and linked PDFs, cleans, chunks, enriches with titles → `data/corpus.json`. |
| `build_index.py` | Thin CLI wrapper that calls `rag.build_index()` to embed the corpus. |
| `rag.py` | **Embeddings + retrieval.** Batch-embeds the corpus, persists the vector index, and serves cosine-similarity retrieval (`Retriever`). |
| `prompts.py` | The system prompt (anti-hallucination rules), the per-turn user-message builder, and the trilingual summary prompt. |
| `llm.py` | `gemma4` chat client (OpenAI-compatible) with retry/backoff. |
| `app.py` | **Streamlit UI.** Chat loop, context-aware retrieval, multi-turn memory, source display, and the email button. |
| `mailer.py` | Optional email: sends a trilingual conversation summary to the admin (self) via MailerSend, on explicit action only. |
| `test_questions.py` | Manual acceptance test (in-data questions + an out-of-data honesty check). |
| `requirements.txt` | Python dependencies. |
| `.gitignore` | Keeps secrets (`specs/creds.txt`, `.env`), the venv, and `data/` out of git. |
| `README.md` | Setup & run instructions. |
| `data/corpus.json` | Scraped + chunked corpus (generated). |
| `data/embeddings.npz` | Embedding index: vectors + chunk texts + source metadata (generated). |

---

## 1. Data collection (`scrape.py`)

The spec requires collecting the data ourselves; the **method is part of the
project**. The approach is a **sitemap-driven full crawl** using `requests` +
`BeautifulSoup` (the site is server-rendered WordPress, so no JavaScript engine
is needed) plus `pypdf` for linked documents.

### 1.1 URL discovery — sitemap, not blind BFS
- `sitemap_urls()` fetches `https://yessenovfoundation.org/sitemap.xml` and
  parses every `<loc>` (≈586 pages: all program pages **and** all news posts).
- These are combined with a small list of `SEED_URLS` (the Data Lab 2026 page in
  RU/KK/EN variants, whose PDFs carry descriptive filenames) and de-duplicated.
- This guarantees **complete coverage** (~654 indexed sources) instead of the
  ~16% a depth-limited link crawl reached. The sitemap is authoritative, so no
  recursive link-following is needed.

### 1.2 Page fetching & cleaning
- `fetch()` GETs each URL with a polite `User-Agent` and a `REQUEST_DELAY`
  (0.3 s) between requests.
- `extract_text()` strips `script/style/nav/footer/header/noscript` and collapses
  whitespace to get the visible page text.

### 1.3 PDF extraction
The most important grant facts (participant lists, the program/syllabus, the
provisions/«ереже») live in **PDFs**, not HTML.
- `pdf_links()` finds PDF links on each page. The site wraps downloads as
  `…?download=1&…&kcccount=<real-url>`, so it unwraps to the real URL after
  `kcccount=`. It accepts **only** PDFs on the main `yessenovfoundation.org`
  host (the dead `old.` subdomain links are skipped — they 404 and waste time).
- `pdf_to_text()` downloads the PDF and extracts text with `pypdf`, skipping any
  file larger than `MAX_PDF_BYTES` (5 MB).
- `is_useful_pdf()` skips large, low-value financial/annual-report PDFs
  (`SKIP_PDF_PATTERNS`: otchet/report/esep/…), which otherwise dominate runtime.

### 1.4 Chunking
`chunk_text()` splits text into overlapping windows:
- **HTML pages:** `CHUNK_SIZE = 900` chars (~225 tokens), `CHUNK_OVERLAP = 150`
  (~17%) — within the recommended 300–800 tokens / 10–20% overlap range.
- **PDFs:** larger `PDF_CHUNK_SIZE = 3000`, `PDF_CHUNK_OVERLAP = 300`, so a
  one-page list/table (e.g. winners) stays in a single chunk and isn't split
  across retrievals.
- Fragments under 80 chars are dropped.

### 1.5 Title enrichment (key retrieval aid)
Each chunk is prefixed with a descriptive `[title]` before embedding:
- For HTML, the title is the page slug (`slug_title()`), e.g. `yessenov scholarship`.
- For PDFs, the title combines the **referring page slug + the file name + a
  multilingual hint** (`describe_doc()`), e.g.
  `yessenov data lab 2026 | list of the winners | список победителей … winners …`.

This is what makes a bare table of names retrievable by intent queries ("give me
the list of people"), and bridges RU/KZ/EN. It functions as lightweight,
embedded metadata.

### 1.6 Output
`main()` writes a list of `{"text": "[title]\n<chunk>", "source": "<url>"}`
records to `data/corpus.json` (currently **3,038 chunks** from **654 sources**).

---

## 2. Embedding & indexing (`build_index.py` → `rag.py`)

- `embed_texts()` calls the **`text-1024`** embedding model
  (`https://llm.alem.ai/v1/embeddings`, 1024-dim). The endpoint accepts a **list
  input**, so it embeds in **batches of 32** — ~95 API calls instead of ~3,000.
- `build_index()` embeds every chunk and saves `data/embeddings.npz` containing:
  - `vectors` — `(N, 1024)` float32 matrix
  - `chunks` — the chunk texts (object array)
  - `metas` — the source URL per chunk (object array)
- `_post_with_retry()` wraps requests with **exponential backoff** on HTTP 429
  (rate limit) and 5xx, so a large indexing run survives throttling.

This is a simple, file-based vector store (NumPy) — adequate for a few thousand
chunks. No external vector DB is required at this scale.

---

## 3. Retrieval (`rag.py` → `Retriever`)

- On construction, `Retriever` loads `embeddings.npz` and **L2-normalizes** the
  vectors so a dot product equals cosine similarity.
- `retrieve(query, k, min_score)`:
  1. Embeds the query with `text-1024` (also normalized).
  2. Computes cosine similarity against all chunk vectors (`vectors @ qv`).
  3. Returns the top-`k` `(chunk, source, score)` tuples above `min_score`
     (default `k=6`, `min_score=0.10`; the app overrides `k`).

The retrieval is **purely semantic (dense)**. The title enrichment in §1.5 gives
much of the benefit of hybrid/keyword search without a separate BM25 index.

---

## 4. Query-time orchestration (`app.py`)

For each user message:

1. **Context-aware + merged retrieval** (fixes both independent questions and
   follow-ups):
   - Retrieve top-10 on the **current question alone** (best recall for
     self-contained questions).
   - If there is prior history, also retrieve top-8 on the **last two user turns
     combined** (helps follow-ups like "what's the deadline there?").
   - **Merge** both result sets (keeping the max score per chunk) and take the
     top 10 → the context block.
2. **Grounded prompt** (`prompts.build_user_message`): the retrieved chunks are
   wrapped in a `CONTEXT` block with the question and an instruction to answer
   **only** from that context.
3. **Multi-turn memory:** the message list sent to `gemma4` is
   `[system prompt] + recent history (last ~6 turns) + current grounded user
   message`. History lets the model resolve "it/there/that program".
4. **Generation:** `llm.chat()` calls `gemma4` (low temperature 0.1 for grounded,
   deterministic answers).
5. **Display:** the answer is shown, plus a "Sources used" expander listing the
   source URLs and relevance scores. The turn is appended to
   `st.session_state.messages`.

### Anti-hallucination (the readiness criterion)
This is enforced primarily at the **prompt layer** (`prompts.SYSTEM_PROMPT`):
- Answer only from the supplied context; if absent, say "I don't know" and point
  to the website — never guess.
- Never fabricate deadlines/requirements/amounts.
- Tie every deadline/amount to its **specific program and year**, never mix
  across programs/years, never extrapolate to other years.
- Use conversation history to resolve references; if unsure which program is
  meant, ask rather than guess.

Reinforced by retrieval (only relevant chunks are supplied) and low temperature.

---

## 5. LLM client (`llm.py`)

- `chat(messages, temperature=0.1)` POSTs to the **`gemma4`** chat-completions
  endpoint (`https://llm.alem.ai/v1/chat/completions`) with a Bearer token.
- Retries with exponential backoff on 429/5xx.
- Returns `choices[0].message.content`.

---

## 6. Conversation memory

State lives in Streamlit `st.session_state.messages` (a list of
`{role, content}`). It is used two ways:
- **Retrieval:** the last two user turns seed the context-aware retrieval (§4.1).
- **Generation:** the last ~6 turns are passed to the model so follow-ups stay on
  the same program and the bot doesn't contradict itself across turns.

---

## 7. Email feature (`mailer.py`) — optional

A first step from chat toward an agent that performs an action.
- `send_summary()` sends a **trilingual (EN / KZ / RU)** summary of the
  conversation (generated by `gemma4` via `prompts.SUMMARY_PROMPT`) to the admin.
- **Safety constraints (per requirements.md), enforced in code:**
  - Recipient is **hardcoded** to `config.ADMIN_EMAIL` (your own address) — never
    a user-supplied value.
  - Sending happens **only** when the sidebar button is pressed — there is **no
    loop and no automatic resend**.
  - Sender identity is fixed: `info@app.commit.kz` / "Yessenov Data Lab".
  - If `MAILERSEND_API_KEY` is unset, the feature disables gracefully.

---

## 8. Configuration & secrets (`config.py`)

- **Endpoints and model names** are non-secret constants (documented in
  `requirements.md`): `gemma4`, `text-1024`, the two `llm.alem.ai` URLs.
- **API keys are never hardcoded.** They are read from environment variables
  first, falling back to parsing `specs/creds.txt` (which is gitignored):
  - `_parse_creds_file()` matches keys by **label/context** — the embedding key
    from the `text-1024 key:` line, the chat key from the `key:` line in the
    `gemma4` section (ignoring the stale `Authorization:` line), and the
    MailerSend key from the `mailersend:` line. Tokens are captured with `\S+`,
    so keys containing `-`/`_` are read in full.
- `ADMIN_EMAIL` defaults to the project owner's address; override via env.
- Helper guards: `require_chat_key()`, `require_embed_key()`, `email_enabled()`.

---

## 9. End-to-end data flow (query time)

```
User: "Какие темы на 3-й неделе Data Lab 2026?"
  → app.py: retrieve(question, k=10) ∪ retrieve(last-2-turns, k=8) → merge top 10
  → rag.Retriever: embed query (text-1024) → cosine vs 3,038 vectors → top chunks
     (program PDF chunk ranks high thanks to its enriched title)
  → prompts.build_user_message(question, context)  +  system prompt  +  history
  → llm.chat(...) → gemma4
  → "Неделя 3 (29 июня–3 июля): LLM и NLP, офлайн в Алматы …"  + Sources expander
```

---

## 10. Build & run

```bash
pip install -r requirements.txt   # streamlit, requests, beautifulsoup4, pypdf, numpy, mailersend
python scrape.py                  # 1. collect data  → data/corpus.json   (~5 min, full sitemap)
python build_index.py             # 2. embed         → data/embeddings.npz (batched)
streamlit run app.py              # 3. serve the chat UI (http://localhost:8501)
```

Keys are read automatically from `specs/creds.txt`. To enable email, also set
`MAILERSEND_API_KEY` (and optionally `ADMIN_EMAIL`).

---

## 11. Key parameters

| Parameter | Value | Where |
|-----------|-------|-------|
| Chat model | `gemma4`, temp 0.1 | `config.py`, `llm.py` |
| Embedding model | `text-1024` (1024-dim) | `config.py`, `rag.py` |
| HTML chunk size / overlap | 900 / 150 chars | `scrape.py` |
| PDF chunk size / overlap | 3000 / 300 chars | `scrape.py` |
| Embedding batch size | 32 | `rag.py` |
| Retrieval top-k (app) | 10 (current) + 8 (combined), merged to 10 | `app.py` |
| Min similarity | 0.10 | `rag.py` |
| Crawl politeness delay | 0.3 s | `scrape.py` |
| Max PDF size | 5 MB | `scrape.py` |
| Memory window | last ~6 turns | `app.py` |
| Corpus size | 3,038 chunks / 654 sources | `data/` |

---

## 12. Design decisions & known limitations

- **No JavaScript renderer (Playwright/Selenium).** Verified the site is
  server-rendered, so `requests` + `BeautifulSoup` suffice — simpler, fewer deps.
- **NumPy file store, not a vector DB.** At a few thousand chunks, brute-force
  cosine is instant; a dedicated DB (FAISS/Chroma) would be over-engineering.
- **Dense retrieval only**, with title enrichment standing in for keyword/hybrid
  search. A BM25 hybrid could be added if specific exact-term recall ever fails.
- **Financial/annual reports are intentionally excluded** for crawl performance,
  so budget/accounting figures aren't answerable. They can be re-included with a
  size cap if needed.
- **Embeddings must be rebuilt** (`build_index.py`) after re-scraping; the running
  app caches the index at startup, so restart it to pick up a new index.
- **Honesty over coverage:** when the right chunk isn't retrieved, the bot says
  "I don't know" rather than guessing — the safe failure mode.

---

## 13. Compliance with `requirements.md`

- **LLM:** `gemma4` only. **UI:** Streamlit. **Embeddings:** `text-1024` (RAG).
  **Email:** MailerSend, self-only, button-only, no loops.
- **Data:** collected from `yessenovfoundation.org` (full sitemap + PDFs).
- **Anti-hallucination** ("I don't know") designed in from the prompt layer.
- **Secrets** loaded from `specs/creds.txt`/env, never hardcoded or committed.
- No forbidden technologies introduced.

# Requirements — YDL 2026 · Week 1 · Day 5 · Final Project

> **Source:** `specs/YDL2026_day5_project.pdf` (titled *ИИ-помощник по грантам фонда* / "AI Grants Assistant for the Foundation").
> Note: the task requested `specs/original_assignment.pdf`, which does not exist; this document is derived from the PDF actually present in `specs/`.
> The spec is written in Russian; requirements below are translated and reorganized faithfully, with nothing omitted.

## Framing / Context

- This is a **learning project and portfolio piece** — *not* a real commission from the foundation.
- Goal: combine everything learned during the week into one working thing. "The bot is the pretext; the skill is the result."
- Whatever you build is **your work and your authorship** — you may show it in your portfolio.
- Publishing it anywhere is **optional** and only with your own consent. It is completely normal if no one's bot ever goes "to prod."
- The project is a **synthesis of the whole week**: interface + LLM + working with data.

## Goal (one-line)

Build a **Streamlit bot powered by `gemma4`** that answers questions about the **grants and activities of the Shakhmardan Yessenov Foundation**.

---

## Functional Requirements — Minimum (mandatory)

1. **Collect the data yourself** from the website **`yessenovfoundation.org`**: grant rules, programs, FAQ.
   - *How* you obtain the data is up to you; Claude can help with scraping or parsing.
   - **The method of collecting the data is part of the project** (it counts / will be discussed).

2. **Build a Streamlit chat on `gemma4`** that answers questions based on this collected data.

3. **Test it on 3–4 real questions.**
   - **Mandatory:** also ask something that is **not** in the data — observe whether the bot fabricates an answer or honestly says "I don't know" (*«не знаю»*).

---

## Readiness Criterion (hard requirement / definition of done)

- A bot that **confidently fabricates** information about grants — deadlines, requirements, amounts — is **NOT considered finished/ready.**
- **Lying here is worse than honestly saying "I don't know"**, because a person will make a real decision based on a false answer.
- This must be **designed in from the very beginning** (anti-hallucination / honest "I don't know" behavior is a core requirement, not an afterthought).

---

## Optional Enhancements ("Where to dig — if you want")

These are explicitly optional ("по желанию"):

- **RAG via an embedding model** (the embedding model is also provided to you): instead of stuffing all texts into the prompt, find the relevant chunk and give the bot only that. Needed when there is a lot of data.
- **Bot personality and tone** — official consultant, friendly helper, your choice.
- **Multiple topics** — grants, scholarships, this school: the bot orients itself across different subjects.
- **Sending email** — see the dedicated Email block below.

---

## Optional Feature — Email ("Email — по желанию")

**Idea:** if the bot decides the conversation was useful, or the user left an application/request, let the **LLM itself send a short summary** of the conversation to the "administrator's" email. This is the first step from a chat toward an **agent that performs an action**.

### Sending Rules (constraints — must be respected if email is implemented)

1. **Send only to your own email.** During the course, the "administrator" is **you yourself**. **Never send to other people's addresses.**
2. **Only on an explicit action** (a button, or a deliberate decision by the model). **Never in a loop** — otherwise a single bug will blast out a hundred emails.

**Why (rationale):** the sending domain has a "reputation" with Gmail and others. Emails to nonexistent addresses, or the same email sent in bulk, cause providers to start cutting the **entire domain** (including the foundation's working mail) into spam. Therefore: **send to yourself, and only on a button press.**

### Provided email-sending code (MailerSend)

The spec provides this sample snippet:

```python
from mailersend import MailerSendClient, EmailBuilder

ms = MailerSendClient(api_key="<MAILERSEND_API_KEY>")  # provided in the spec; store in an env var, do NOT hardcode/commit
ADMIN_EMAIL = "your-personal@email.com"  # your own mailbox — send only to yourself

email = (EmailBuilder()
    .from_email("info@app.commit.kz", "Yessenov Data Lab")
    .to_many([{"email": ADMIN_EMAIL, "name": "Admin"}])
    .subject("Новая заявка из чата")
    .html("<h1>Саммари разговора</h1><p>...</p>")
    .text("Саммари разговора: ...")
    .build())

response = ms.emails.send(email)
print("Отправлено:", response.message_id)
```

- Library: **`mailersend`** (`MailerSendClient`, `EmailBuilder`).
- A **MailerSend API key is supplied in the spec.** (Security note: a real key string is hardcoded in the PDF snippet — keep it out of version control; load it from an environment variable. The literal key value is intentionally not reproduced here.)
- `from_email`: **`info@app.commit.kz`** with display name **"Yessenov Data Lab"** (fixed sending identity).
- `ADMIN_EMAIL`: your own personal mailbox.

---

## Deliverable / Demonstration ("Демонстрации — 15:00 ровно")

- **Time:** demos start at **15:00 sharp.**
- **Format:** each person gets **exactly 4 minutes to present + 1 minute for questions.** Cut off strictly, otherwise not everyone gets a turn.
- Reference point: a standard investor pitch is 3 minutes; you get 4 (generous). Fitting in and saying only the essentials is **part of the skill being graded**, not nitpicking.

### Required live demo actions (each person must, live):

1. **Ask the bot a general question announced by the instructor on the spot** — the same question for everyone, so the bots can be fairly compared (whose answered accurately vs. whose fabricated).
2. **Show one question on which the bot breaks** — you know the boundaries of your own work best, so demonstrate them yourself.

### Also briefly cover in the demo:

- How you obtained the data.
- Which approach you chose.
- What the bot can do **beyond the minimum**.

### Logistics

- Prepare your screen **in advance**, while the previous person is answering — switching eats more time than the demos themselves.
- Closing line / framing: "This is everything you assembled over the week — in one working thing."

---

## Model Endpoints & Specifications

The provided models are served from **`llm.alem.ai`** (OpenAI-compatible API). Credentials and full request samples are in **`specs/creds.txt`**.

> **Security note:** `specs/creds.txt` contains live API keys. Do **not** hardcode or commit these keys — load them from environment variables. The literal key values are intentionally not reproduced in this document; read them from `specs/creds.txt` at setup time.

### Chat LLM (mandatory)

- **Model:** `gemma4`
- **Endpoint:** `https://llm.alem.ai/v1/chat/completions`
- **Auth:** `Authorization: Bearer <CHAT_API_KEY>` (key in `specs/creds.txt`)
- **Request format** (OpenAI chat-completions style):

```json
{
  "model": "gemma4",
  "messages": [
    { "role": "user", "content": "Hello, world!" }
  ]
}
```

### Text Embedding Model (optional — for RAG)

- **Model:** `text-1024` (1024-dimensional embeddings)
- **Endpoint:** `https://llm.alem.ai/v1/embeddings`
- **Auth:** `Authorization: Bearer <EMBEDDING_API_KEY>` (key in `specs/creds.txt`)
- **Request format:**

```json
{
  "model": "text-1024",
  "input": "Hello, world!"
}
```

## Required / Specified Technologies

| Area | Requirement |
|------|-------------|
| LLM | **`gemma4`** (mandatory) — `https://llm.alem.ai/v1/chat/completions` |
| UI | **Streamlit** (mandatory) |
| Data source | **`yessenovfoundation.org`** — grant rules, programs, FAQ (collected yourself) |
| Embedding model | **`text-1024`** (1024-dim) — `https://llm.alem.ai/v1/embeddings`; use is **optional** (for RAG) |
| Email (optional) | **MailerSend** (`mailersend` Python library), via the provided API key & sender identity |

## Forbidden / Prohibited

- The spec does **not** list any forbidden technologies or libraries.
- Behavioral prohibitions (effectively forbidden actions):
  - **Do not** let the bot confidently fabricate grant information (deadlines, requirements, amounts) — this fails the readiness criterion.
  - **Do not** send email to anyone other than your own address.
  - **Do not** send email inside a loop / automatically without an explicit action.

## Grading / Evaluation Criteria (as stated or implied)

The spec does not give a numeric rubric. The explicit/implicit criteria are:

1. **Honesty over fabrication** — the bot must say "I don't know" rather than invent grant facts (stated as the readiness criterion).
2. **Data collection approach** — the method of gathering data is explicitly "part of the project."
3. **Live demo performance** — accurate answer to the instructor's common question vs. fabrication; ability to show the bot's failure boundary.
4. **Time discipline** — fitting the presentation into 4 minutes is "part of the skill."
5. **Going beyond the minimum** (optional features) is something to highlight in the demo.

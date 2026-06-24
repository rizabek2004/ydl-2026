"""Streamlit chat UI — AI Grants Assistant for the Yessenov Foundation.

Implements requirements.md Functional Requirement #2 (Streamlit chat on gemma4
answering from the collected data) with:
  - RAG retrieval over the scraped corpus (text-1024 embeddings),
  - a grounded, anti-hallucination system prompt (Readiness Criterion),
  - an OPTIONAL, button-triggered, self-only email summary (Email block).

Run:
    streamlit run app.py
"""
import base64
from pathlib import Path

import streamlit as st

import config
import llm
import prompts
from rag import Retriever

_LOGO_PATH = Path(__file__).parent / "assets" / "logo.png"


def _logo_data_uri():
    """Embed the bundled logo as a base64 data URI. Hotlinking the logo from the
    foundation site is blocked by the browser (referer/hotlink protection), so we
    serve a local copy inline instead."""
    try:
        b64 = base64.b64encode(_LOGO_PATH.read_bytes()).decode()
        return f"data:image/png;base64,{b64}"
    except OSError:
        return ""


LOGO_DATA_URI = _logo_data_uri()

st.set_page_config(
    page_title="Yessenov Foundation — Grants Assistant",
    page_icon=str(_LOGO_PATH) if _LOGO_PATH.exists() else "🎓",
    layout="centered",
)

# --- Brand styling: matched to yessenovfoundation.org (Exo 2 font; purple
# #6d1f76 / magenta #d700a5 / sky-blue #79c7e5 palette) ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Exo+2:wght@300;400;600;700&display=swap');

    html, body, .stApp, [data-testid="stChatMessage"],
    [data-testid="stSidebar"], .stButton > button, .stMarkdown {
        font-family: 'Exo 2', Tahoma, sans-serif;
    }
    .stApp { background: #ffffff; }

    /* Gradient hero banner with logo */
    .yf-hero {
        background: linear-gradient(120deg, #4f1850 0%, #6d1f76 45%, #d700a5 100%);
        border-radius: 16px;
        padding: 22px 26px;
        display: flex; align-items: center; gap: 18px;
        box-shadow: 0 6px 20px rgba(109, 31, 118, 0.25);
    }
    /* The foundation logo is white, so it sits directly on the purple gradient
       (a white chip would make it invisible). */
    .yf-hero img { height: 46px; }
    .yf-hero .yf-text h1 { color: #fff; font-size: 1.5rem; margin: 0;
        font-weight: 700; letter-spacing: .3px; line-height: 1.2; }
    .yf-hero .yf-text p { color: #f3d9ee; margin: 4px 0 0; font-size: .9rem; }
    .yf-strip { height: 4px; border-radius: 4px;
        background: linear-gradient(90deg, #6d1f76, #d700a5, #79c7e5);
        margin: 8px 0 18px; }

    /* Chat bubbles */
    [data-testid="stChatMessage"] {
        background: #faf4fb; border-radius: 14px;
        border-left: 4px solid #79c7e5; padding: 6px 10px; margin-bottom: 8px;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background: #f7eef7; border-left: 4px solid #6d1f76;
    }

    /* Sidebar */
    [data-testid="stSidebar"] { background: #faf4fb; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 { color: #6d1f76; }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(120deg, #6d1f76, #d700a5);
        color: #fff; border: none; border-radius: 10px; font-weight: 600;
    }
    .stButton > button:hover { filter: brightness(1.08); color: #fff; }

    a, a:visited { color: #d700a5; }
    [data-testid="stChatInput"] textarea:focus { border-color: #d700a5; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="yf-hero">
        <img src="{LOGO_DATA_URI}" alt="Yessenov Foundation logo">
        <div class="yf-text">
            <h1>ИИ-помощник по грантам фонда</h1>
            <p>Shakhmardan Yessenov Foundation — гранты, программы, стипендии ·
            отвечает только по данным с yessenovfoundation.org</p>
        </div>
    </div>
    <div class="yf-strip"></div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading knowledge base...")
def get_retriever():
    """Load the embedding index once. Returns None if it hasn't been built."""
    try:
        return Retriever()
    except FileNotFoundError:
        return None


retriever = get_retriever()

if retriever is None:
    st.error(
        "Knowledge base not found. Build it first:\n\n"
        "1. `python scrape.py`  (collect the data)\n"
        "2. `python build_index.py`  (embed it for RAG)"
    )
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render conversation history.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Sidebar: optional self-only email summary (explicit button = no loops) ---
with st.sidebar:
    st.header("Admin email")
    if config.email_enabled():
        st.caption(f"Summary will be sent only to: {config.ADMIN_EMAIL}")
        if st.button("✉️ Send conversation summary to admin"):
            if not st.session_state.messages:
                st.warning("No conversation to summarize yet.")
            else:
                import mailer

                transcript = "\n".join(
                    f"{m['role']}: {m['content']}" for m in st.session_state.messages
                )
                with st.spinner("Summarizing and sending..."):
                    summary = llm.chat(
                        [
                            {"role": "system", "content": prompts.SUMMARY_PROMPT},
                            {"role": "user", "content": transcript},
                        ],
                        temperature=0.3,
                    )
                    msg_id = mailer.send_summary(summary)
                st.success(f"Sent to {config.ADMIN_EMAIL} (id: {msg_id})")
    else:
        st.caption("Email disabled — set MAILERSEND_API_KEY to enable.")

# --- Chat input ---
question = st.chat_input("Ask about grants, programs, or scholarships...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the knowledge base..."):
            # Retrieve on the current question ALONE (best recall for a
            # self-contained question) and, if there is history, ALSO on the
            # combined last-2 turns (helps follow-ups like "deadline there?").
            # Merge both so neither independent questions nor follow-ups suffer.
            user_turns = [
                m["content"] for m in st.session_state.messages if m["role"] == "user"
            ]
            merged = {}  # chunk text -> (source, best score)
            for chunk, src, score in retriever.retrieve(question, k=10):
                merged[chunk] = (src, score)
            if len(user_turns) >= 2:
                combined = " ".join(user_turns[-2:])
                for chunk, src, score in retriever.retrieve(combined, k=8):
                    if chunk not in merged or score > merged[chunk][1]:
                        merged[chunk] = (src, score)
            hits = sorted(
                ((c, s, sc) for c, (s, sc) in merged.items()),
                key=lambda t: t[2], reverse=True,
            )[:10]
            context = "\n\n---\n\n".join(chunk for chunk, _src, _score in hits)

            # Multi-turn memory: pass recent history (plain turns) before the
            # current grounded question so the model can resolve references and
            # stay consistent across turns. Keep the last few exchanges only.
            history = st.session_state.messages[:-1][-6:]
            messages = (
                [{"role": "system", "content": prompts.SYSTEM_PROMPT}]
                + history
                + [{"role": "user",
                    "content": prompts.build_user_message(question, context)}]
            )
            answer = llm.chat(messages)
        st.markdown(answer)
        if hits:
            with st.expander("Sources used"):
                for _chunk, src, score in hits:
                    st.caption(f"{src}  (relevance {score:.2f})")

    st.session_state.messages.append({"role": "assistant", "content": answer})

"""Configuration & secret loading.

Per requirements.md and CLAUDE.md: API keys must NOT be hardcoded or committed.
They are loaded from environment variables first; if absent, the two alem.ai keys
are read at runtime from specs/creds.txt (which is gitignored). Endpoints and model
names are not secrets — they are documented in requirements.md — so they are constants.
"""
import os
import re
from pathlib import Path

# --- Non-secret constants (from requirements.md "Model Endpoints & Specifications") ---
CHAT_MODEL = "gemma4"
CHAT_URL = "https://llm.alem.ai/v1/chat/completions"

EMBED_MODEL = "text-1024"
EMBED_URL = "https://llm.alem.ai/v1/embeddings"

# Fixed MailerSend sender identity (from requirements.md email block).
FROM_EMAIL = "info@app.commit.kz"
FROM_NAME = "Yessenov Data Lab"

_CREDS_FILE = Path(__file__).parent / "specs" / "creds.txt"


def _parse_creds_file():
    """Extract API keys from specs/creds.txt by label/context (not by position).

    Keys live ONLY in specs/creds.txt — never hardcoded here. Returns a dict that
    may contain "embed", "chat", and "mailersend". Tokens are matched with \\S+ so
    keys containing '-' or '_' are captured in full. The stale `Authorization:`
    line is ignored: the chat key is the `key:` line within the gemma4 section.
    """
    creds = {}
    if not _CREDS_FILE.exists():
        return creds
    text = _CREDS_FILE.read_text(encoding="utf-8")

    m = re.search(r"text-1024\s+key:\s*(\S+)", text)
    if m:
        creds["embed"] = m.group(1)

    m = re.search(r"(?im)mailersend[^:]*:\s*(\S+)", text)
    if m:
        creds["mailersend"] = m.group(1)

    # Chat key: the first `key:` line at/after the gemma4 section header.
    gpos = text.find("gemma4")
    if gpos != -1:
        m = re.search(r"(?m)^\s*key:\s*(\S+)", text[gpos:])
        if m:
            creds["chat"] = m.group(1)

    return creds


_creds = _parse_creds_file()

CHAT_API_KEY = os.environ.get("CHAT_API_KEY") or _creds.get("chat")
EMBED_API_KEY = os.environ.get("EMBED_API_KEY") or _creds.get("embed")

# MailerSend key — env var first, then specs/creds.txt. Never hardcoded. If
# unset in both, the email feature is disabled gracefully.
MAILERSEND_API_KEY = os.environ.get("MAILERSEND_API_KEY") or _creds.get("mailersend")

# requirements.md: email must go ONLY to your own address. Defaults to the
# project owner's email; override with ADMIN_EMAIL if needed.
ADMIN_EMAIL = os.environ.get("ADMIN_EMAIL", "alpamysrizabek32@gmail.com")

# --- Data paths ---
DATA_DIR = Path(__file__).parent / "data"
CORPUS_FILE = DATA_DIR / "corpus.json"
EMBEDDINGS_FILE = DATA_DIR / "embeddings.npz"


def require_chat_key():
    if not CHAT_API_KEY:
        raise RuntimeError(
            "No chat API key. Set CHAT_API_KEY env var or provide specs/creds.txt."
        )
    return CHAT_API_KEY


def require_embed_key():
    if not EMBED_API_KEY:
        raise RuntimeError(
            "No embedding API key. Set EMBED_API_KEY env var or provide specs/creds.txt."
        )
    return EMBED_API_KEY


def email_enabled():
    return bool(MAILERSEND_API_KEY)

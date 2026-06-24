"""Retrieval-Augmented Generation using the text-1024 embedding model.

Embeds the scraped corpus once (cached to data/embeddings.npz) and retrieves the
top-k most relevant chunks for a query via cosine similarity. This implements the
optional RAG enhancement from requirements.md: give the bot only the relevant
chunk instead of the whole corpus.
"""
import json
import time

import numpy as np
import requests

import config


def _post_with_retry(url, headers, payload, timeout, retries=5):
    """POST with exponential backoff on 429 (rate limit) / 5xx errors."""
    delay = 2.0
    for attempt in range(retries):
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        if resp.status_code == 429 or resp.status_code >= 500:
            if attempt < retries - 1:
                time.sleep(delay)
                delay *= 2
                continue
        resp.raise_for_status()
        return resp
    resp.raise_for_status()
    return resp


def embed_texts(texts, timeout: int = 60, batch_size: int = 32):
    """Return a (len(texts), dim) float32 numpy array of embeddings.

    The text-1024 endpoint accepts a LIST input, so we embed in batches to keep
    the number of API calls (and rate-limit pressure) low when indexing the whole
    site. Results are reordered by the API's `index` field to preserve order.
    """
    headers = {
        "Authorization": f"Bearer {config.require_embed_key()}",
        "Content-Type": "application/json",
    }
    vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        payload = {"model": config.EMBED_MODEL, "input": batch}
        resp = _post_with_retry(config.EMBED_URL, headers, payload, timeout)
        data = sorted(resp.json()["data"], key=lambda d: d.get("index", 0))
        vectors.extend(d["embedding"] for d in data)
    return np.array(vectors, dtype=np.float32)


def _normalize(mat):
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def build_index():
    """Embed the cached corpus and persist vectors + chunk texts."""
    with open(config.CORPUS_FILE, encoding="utf-8") as f:
        corpus = json.load(f)
    chunks = [c["text"] for c in corpus]
    metas = [c.get("source", "") for c in corpus]
    vectors = embed_texts(chunks)
    config.DATA_DIR.mkdir(exist_ok=True)
    np.savez(
        config.EMBEDDINGS_FILE,
        vectors=vectors,
        chunks=np.array(chunks, dtype=object),
        metas=np.array(metas, dtype=object),
    )
    return len(chunks)


class Retriever:
    """Loads the cached embedding index and answers similarity queries."""

    def __init__(self):
        data = np.load(config.EMBEDDINGS_FILE, allow_pickle=True)
        self.vectors = _normalize(data["vectors"].astype(np.float32))
        self.chunks = list(data["chunks"])
        self.metas = list(data["metas"])

    def retrieve(self, query: str, k: int = 6, min_score: float = 0.10):
        """Return up to k (chunk, source, score) tuples above min_score."""
        qv = _normalize(embed_texts([query]))[0]
        scores = self.vectors @ qv
        order = np.argsort(scores)[::-1][:k]
        results = []
        for i in order:
            if scores[i] >= min_score:
                results.append((self.chunks[i], self.metas[i], float(scores[i])))
        return results

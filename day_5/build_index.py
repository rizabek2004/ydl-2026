"""Build the RAG embedding index from the scraped corpus.

Run after scrape.py:

    python build_index.py
"""
import rag

if __name__ == "__main__":
    n = rag.build_index()
    print(f"Embedded {n} chunks into the index.")

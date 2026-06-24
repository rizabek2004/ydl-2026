"""Manual acceptance test (requirements.md F3 + F4).

Runs the real RAG + gemma4 pipeline on in-data questions and an out-of-data
question. The out-of-data question must produce an honest "I don't know", not a
fabricated deadline/amount.

    python test_questions.py
"""
import llm
import prompts
from rag import Retriever

IN_DATA = [
    "Какие программы и гранты есть у фонда Есенова?",
    "Расскажи про стипендию Есенова.",
    "Что такое Yessenov Data Lab?",
    "Дай список людей прошедших в Yessenov Data Lab 2026",
    "Какие темы будут пройдены в Yessenov Data Lab 2026",
]
OUT_OF_DATA = "Какой точный размер гранта в тенге на 2027 год и дедлайн подачи?"


def answer(retriever, question):
    hits = retriever.retrieve(question, k=6)
    context = "\n\n---\n\n".join(chunk for chunk, _s, _sc in hits)
    msgs = [
        {"role": "system", "content": prompts.SYSTEM_PROMPT},
        {"role": "user", "content": prompts.build_user_message(question, context)},
    ]
    return llm.chat(msgs), len(hits)


def main():
    r = Retriever()
    for q in IN_DATA:
        a, n = answer(r, q)
        print(f"\n=== IN-DATA ({n} chunks) ===\nQ: {q}\nA: {a}\n")
    a, n = answer(r, OUT_OF_DATA)
    print(f"\n=== OUT-OF-DATA / HONESTY CHECK ({n} chunks) ===\nQ: {OUT_OF_DATA}\nA: {a}\n")


if __name__ == "__main__":
    main()

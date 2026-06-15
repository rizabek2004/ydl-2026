import json
import os
import urllib.error
import urllib.request

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))


def load_chat_creds():
    """Read chat model / url / key from environment (.env)."""
    model = os.getenv("CHAT_MODEL")
    url = os.getenv("CHAT_URL")
    key = os.getenv("CHAT_KEY")
    if not (model and url and key):
        raise RuntimeError("Missing CHAT_MODEL/CHAT_URL/CHAT_KEY in .env")
    return model, url, key


def ask(model, url, key, messages):
    payload = json.dumps(
        {"model": model, "messages": messages}
    ).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(request) as response:
        data = json.load(response)
    return data["choices"][0]["message"]["content"]


def main():
    model, url, key = load_chat_creds()
    messages = []
    print("Ask gemma anything. Type 'exit' or 'quit' (or press Ctrl+D) to stop.")
    while True:
        try:
            question = input("\nWhat do you want to ask? ")
        except EOFError:
            print()
            break
        if question.strip().lower() in {"exit", "quit"}:
            break
        if not question.strip():
            continue
        messages.append({"role": "user", "content": question})
        try:
            answer = ask(model, url, key, messages)
        except urllib.error.HTTPError as e:
            print(f"Request failed ({e.code}): {e.read().decode('utf-8', 'replace')}")
            messages.pop()  # drop the question we couldn't answer
            continue
        messages.append({"role": "assistant", "content": answer})
        print(f"\n{answer}")


if __name__ == "__main__":
    main()

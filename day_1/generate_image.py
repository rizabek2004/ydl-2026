import base64
import json
import os
import urllib.error
import urllib.request

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))


def load_image_creds():
    """Read image model / url / key from environment (.env)."""
    model = os.getenv("IMAGE_MODEL")
    url = os.getenv("IMAGE_URL")
    key = os.getenv("IMAGE_KEY")
    if not (model and url and key):
        raise RuntimeError("Missing IMAGE_MODEL/IMAGE_URL/IMAGE_KEY in .env")
    return model, url, key


def generate(model, url, key, prompt, size="512x512"):
    payload = json.dumps(
        {"model": model, "prompt": prompt, "size": size}
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

    item = data["data"][0]
    # API may return base64-encoded image or a URL to download.
    if item.get("b64_json"):
        return base64.b64decode(item["b64_json"])
    with urllib.request.urlopen(item["url"]) as img:
        return img.read()


def main():
    model, url, key = load_image_creds()
    prompt = input("What image do you want to generate? ")
    out_path = os.path.join(os.path.dirname(__file__), "output.png")
    try:
        image_bytes = generate(model, url, key, prompt)
    except urllib.error.HTTPError as e:
        print(f"Request failed ({e.code}): {e.read().decode('utf-8', 'replace')}")
        return
    with open(out_path, "wb") as f:
        f.write(image_bytes)
    print(f"Saved image to {out_path}")


if __name__ == "__main__":
    main()

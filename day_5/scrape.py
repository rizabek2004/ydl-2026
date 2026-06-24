"""Data collection (requirements.md Functional Requirement #1).

Scrapes the Yessenov Foundation website for grant rules, programs, scholarships,
and FAQ content, cleans the text, splits it into chunks, and caches the result to
data/corpus.json. Run this once before launching the app:

    python scrape.py

Approach: start from the homepage, discover internal links, keep pages whose URL
or content looks relevant, extract visible text with BeautifulSoup, and chunk it.
Polite: identifies itself via User-Agent and pauses between requests.
"""
import io
import json
import logging
import re
import time
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

import config

# pypdf prints many harmless "Multiple definitions in dictionary" warnings on the
# foundation's PDFs — silence them so the crawl log stays readable.
logging.getLogger("pypdf").setLevel(logging.ERROR)

BASE = "https://yessenovfoundation.org"
HEADERS = {"User-Agent": "YDL2026-student-project/1.0 (educational use)"}
SITEMAP_URL = "https://yessenovfoundation.org/sitemap.xml"

# Explicit seeds to guarantee coverage of high-value pages (and their PDFs) in
# all three language variants — the EN/RU pages carry descriptive PDF filenames
# and content that the default-language sitemap pages do not. These are crawled
# IN ADDITION to every URL in the sitemap.
SEED_URLS = [
    BASE,
    "https://yessenovfoundation.org/about-us/programs",
    "https://yessenovfoundation.org/about-us/programs/resources/yessenov-data-lab/yessenov-data-lab-2026",
    "https://yessenovfoundation.org/en/about-us/programs/resources/yessenov-data-lab/yessenov-data-lab-2026",
    "https://yessenovfoundation.org/ru/about-us/programs/resources/yessenov-data-lab/yessenov-data-lab-2026",
    "https://yessenovfoundation.org/kk/about-us/programs/resources/yessenov-data-lab/yessenov-data-lab-2026",
]

# URL keywords that signal relevant content.
RELEVANT = ("grant", "program", "scholarship", "faq", "about", "stipend",
            "konkurs", "proekt", "fond", "data-lab", "resources")

# URL patterns that are low-value noise (news pagination / year filters); skipping
# them frees the crawl budget for real content pages.
SKIP_PATTERNS = ("/page/", "?y=", "/tag/", "/author/")

# Financial / annual report PDFs are large accounting documents with little value
# for a grants assistant and dominate crawl time — skip them.
SKIP_PDF_PATTERNS = ("otchet", "report", "esep", "qarzhyl", "godov",
                     "finotchet", "finreport")
MAX_PDF_BYTES = 5_000_000  # skip PDFs larger than ~5 MB


def sitemap_urls():
    """Return every page URL listed in the site's sitemap.xml (full coverage)."""
    try:
        xml = fetch(SITEMAP_URL)
    except Exception as e:  # noqa: BLE001
        print(f"  could not fetch sitemap: {e}")
        return []
    locs = re.findall(r"<loc>(.*?)</loc>", xml)
    return [u.strip() for u in locs if u.strip().startswith("http")]

MAX_PAGES = 60
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
# PDFs (participant lists, syllabi, provisions) are chunked larger so a one-page
# list/table stays in a single chunk and isn't split across retrievals.
PDF_CHUNK_SIZE = 3000
PDF_CHUNK_OVERLAP = 300
REQUEST_DELAY = 0.3  # seconds, be polite (≈3 min for the full sitemap)


def fetch(url):
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding or resp.encoding
    return resp.text


def extract_text(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)


def internal_links(html, current_url):
    soup = BeautifulSoup(html, "html.parser")
    out = set()
    base_host = urlparse(BASE).netloc
    for a in soup.find_all("a", href=True):
        url = urljoin(current_url, a["href"]).split("#")[0].rstrip("/")
        low = url.lower()
        # Skip file downloads / PDF wrapper links — they are handled by
        # pdf_links(), not crawled as HTML (otherwise the PDF binary gets
        # mis-parsed into thousands of garbage chunks).
        if ".pdf" in low or "download=" in low or "kcccount=" in low:
            continue
        if urlparse(url).netloc == base_host and url.startswith("http"):
            out.add(url)
    return out


def is_relevant(url):
    low = url.lower()
    return any(k in low for k in RELEVANT)


def should_skip(url):
    low = url.lower()
    return any(p in low for p in SKIP_PATTERNS)


def slug_title(url):
    """Turn a URL/filename into readable words for retrieval (e.g.
    '.../list-of-the-winners-1.pdf' -> 'list of the winners 1')."""
    path = urlparse(url).path
    name = path.rstrip("/").split("/")[-1]
    name = name.rsplit(".", 1)[0]  # drop extension
    name = name.replace("-", " ").replace("_", " ").replace("%20", " ")
    return " ".join(name.split())


# Descriptive, multilingual hints keyed off a document's filename. Prepended to a
# chunk's title so a bare data table (e.g. names) is retrievable by intent queries
# like "give me the list of people" or "what topics" across RU/KZ/EN.
DOC_HINTS = [
    (("winner", "pobeditel", "zhenimpazdar", "zheңimpazdar", "жеңімпаздар"),
     "список победителей прошедших отбор; список людей прошедших; "
     "list of winners who passed selection; жеңімпаздар тізімі; имена фамилии"),
    (("participant", "uchastnik", "katysushy", "қatysushy", "қатысушы",
      "kezen", "kezeң", "tura", "tour", "round"),
     "список участников прошедших в тур; список людей; "
     "list of round participants; қатысушылар тізімі; имена фамилии"),
    (("program", "programma", "bagdarlama", "baғdarlama", "бағдарлама"),
     "программа обучения темы модули расписание по неделям; "
     "what topics will be covered; program syllabus weeks schedule; "
     "қандай тақырыптар оқытылады"),
    (("provision", "polozhenie", "erezhe", "ереже", "rule"),
     "положение правила требования условия участия; "
     "provisions rules requirements eligibility"),
]


def describe_doc(url):
    low = url.lower()
    for keys, desc in DOC_HINTS:
        if any(k in low for k in keys):
            return desc
    return ""


def pdf_links(html, current_url):
    """Return direct PDF URLs linked on a page.

    The site wraps downloads as `...?download=1&...&kcccount=<real-pdf-url>`;
    we unwrap to the real URL after `kcccount=`. Plain `.pdf` hrefs pass through.
    """
    soup = BeautifulSoup(html, "html.parser")
    out = set()
    for a in soup.find_all("a", href=True):
        href = urljoin(current_url, a["href"])
        if ".pdf" not in href.lower():
            continue
        if "kcccount=" in href:
            href = href.split("kcccount=", 1)[1]
        # Only PDFs hosted on the main domain — skip the dead `old.` subdomain
        # and any external hosts (they 404 or hang and waste crawl time).
        if (href.lower().endswith(".pdf")
                and urlparse(href).netloc == "yessenovfoundation.org"):
            out.add(href)
    return out


def pdf_to_text(url):
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    if len(resp.content) > MAX_PDF_BYTES:
        return ""  # too large; skip to keep the crawl fast
    reader = PdfReader(io.BytesIO(resp.content))
    parts = [(page.extract_text() or "") for page in reader.pages]
    return "\n".join(parts)


def is_useful_pdf(url):
    low = url.lower()
    return not any(p in low for p in SKIP_PDF_PATTERNS)


def chunk_text(text, source, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP, title=""):
    """Split text into overlapping chunks. A descriptive `title` is prepended to
    each chunk so retrieval can match content (e.g. a bare table of names) by the
    page/file it came from."""
    prefix = f"[{title}]\n" if title else ""
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        piece = text[start:end].strip()
        if len(piece) > 80:  # skip tiny fragments
            chunks.append({"text": prefix + piece, "source": source})
        start += size - overlap
    return chunks


def crawl():
    """Full-site crawl driven by sitemap.xml for complete coverage.

    Visits every URL in the sitemap (≈586 pages: all programs + all news) plus
    the explicit SEED_URLS (language variants whose PDFs we want). For each page
    it extracts the text and downloads any linked PDFs. No BFS/link-following is
    needed: the sitemap is the authoritative page list.
    """
    urls = list(dict.fromkeys(SEED_URLS + sitemap_urls()))
    urls = [u for u in urls if not should_skip(u)]
    print(f"  {len(urls)} URLs to crawl (seeds + sitemap)")

    corpus = []
    visited = set()
    seen_pdfs = set()

    for i, url in enumerate(urls):
        if url in visited:
            continue
        visited.add(url)
        try:
            html = fetch(url)
        except Exception as e:  # noqa: BLE001 — skip unreachable pages
            print(f"  skip {url}: {e}")
            continue

        page_title = slug_title(url)
        text = extract_text(html)
        if len(text) >= 200:
            corpus.extend(chunk_text(text, url, title=page_title))

        # Download and extract any linked PDFs (participant lists, syllabi, rules),
        # skipping large financial/annual reports.
        for pdf_url in pdf_links(html, url):
            if pdf_url in seen_pdfs or not is_useful_pdf(pdf_url):
                continue
            seen_pdfs.add(pdf_url)
            try:
                pdf_text = pdf_to_text(pdf_url)
            except Exception as e:  # noqa: BLE001 — skip unreadable PDFs
                print(f"  skip pdf {pdf_url}: {e}")
                continue
            if len(pdf_text) >= 80:
                # Title combines the referring page, the file name, and a
                # descriptive multilingual hint so a bare table (e.g. names) is
                # findable by intent ("list of people", "what topics").
                pdf_title = f"{page_title} | {slug_title(pdf_url)} | {describe_doc(pdf_url)}"
                corpus.extend(chunk_text(
                    pdf_text, pdf_url, size=PDF_CHUNK_SIZE,
                    overlap=PDF_CHUNK_OVERLAP, title=pdf_title,
                ))
                print(f"  + [PDF] {slug_title(pdf_url)}")

        if (i + 1) % 50 == 0:
            print(f"  ...{i + 1}/{len(urls)} pages, {len(corpus)} chunks so far")
        time.sleep(REQUEST_DELAY)

    return corpus


def main():
    print(f"Scraping {BASE} ...")
    corpus = crawl()
    config.DATA_DIR.mkdir(exist_ok=True)
    with open(config.CORPUS_FILE, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(corpus)} chunks to {config.CORPUS_FILE}")


if __name__ == "__main__":
    main()

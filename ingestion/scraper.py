"""
Fetch each URL in urls.txt, extract the main article body, save as data/raw/<slug>.txt

Usage:
    python ingestion/scraper.py
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from urllib.parse import unquote, urlparse

import requests
from bs4 import BeautifulSoup

REPO_ROOT = Path(__file__).resolve().parent.parent
URLS_FILE = REPO_ROOT / "urls.txt"
OUT_DIR = REPO_ROOT / "data" / "raw"

HEADERS = {
    # Wikipedia blocks default python-requests UA; pretend to be a normal browser.
    "User-Agent": (
        "soccer-tactics-rag/0.1 (educational project; contact: you@example.com) "
        "python-requests"
    )
}


def slugify_url(url: str) -> str:
    """Turn a Wikipedia URL into a safe filename."""
    path = urlparse(url).path
    name = unquote(path.rsplit("/", 1)[-1])
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_").lower()
    return name or "article"


def extract_main_text(html: str) -> str:
    """Strip Wikipedia chrome and keep the article body."""
    soup = BeautifulSoup(html, "html.parser")

    title_el = soup.find("h1", id="firstHeading")
    title = title_el.get_text(strip=True) if title_el else ""

    content = soup.find("div", id="mw-content-text")
    if content is None:
        return title

    # Drop boilerplate: references, edit links, navboxes, infoboxes, etc.
    drop_selectors = [
        "table",            # infoboxes and most data tables
        "div.navbox",
        "div.reflist",
        "ol.references",
        "sup.reference",
        "span.mw-editsection",
        "div.hatnote",
        "div.thumb",
        "div.metadata",
        "#References",
        "#External_links",
        "#See_also",
        "#Further_reading",
    ]
    for sel in drop_selectors:
        for el in content.select(sel):
            el.decompose()

    # Stop at the references/see-also/external-links heading if present.
    for h2 in content.find_all(["h2", "h3"]):
        heading_text = h2.get_text(strip=True).lower()
        if heading_text in {"references", "see also", "external links", "notes", "further reading"}:
            # Remove this heading and everything after it.
            for sib in list(h2.find_all_next()):
                sib.decompose()
            h2.decompose()
            break

    paragraphs = [p.get_text(" ", strip=True) for p in content.find_all(["p", "li", "h2", "h3"])]
    paragraphs = [p for p in paragraphs if p]

    body = "\n\n".join(paragraphs)
    return f"{title}\n\n{body}".strip()


def fetch(url: str) -> str:
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    return resp.text


def load_urls() -> list[str]:
    if not URLS_FILE.exists():
        sys.exit(f"Missing {URLS_FILE}")
    urls: list[str] = []
    for line in URLS_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        urls.append(line)
    return urls


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    urls = load_urls()
    print(f"Loaded {len(urls)} URLs.")

    ok = 0
    fail = 0
    for i, url in enumerate(urls, 1):
        slug = slugify_url(url)
        out_path = OUT_DIR / f"{slug}.txt"
        try:
            html = fetch(url)
            text = extract_main_text(html)
            if len(text) < 200:
                # Probably a redirect to a stub; flag but still save.
                print(f"  [{i:>2}/{len(urls)}] WARN short content ({len(text)} chars): {url}")
            out_path.write_text(text, encoding="utf-8")
            print(f"  [{i:>2}/{len(urls)}] OK  {slug}.txt  ({len(text):,} chars)")
            ok += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  [{i:>2}/{len(urls)}] FAIL {url}: {exc}")
            fail += 1
        time.sleep(0.5)  # be polite to Wikipedia

    print(f"\nDone. {ok} succeeded, {fail} failed. Files in {OUT_DIR}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

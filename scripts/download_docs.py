"""Polite crawler for ONE fixed major version of the official PostgreSQL docs (HTML). Network required."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import re
import time

import httpx
from bs4 import BeautifulSoup

from src.config import Settings

SKIP = re.compile(r"^(release|bookindex|legalnotice|appendix-obsolete|sourcerepo|docguide|errcodes)", re.I)


def main() -> None:
    s = Settings.from_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default=s.pg_docs_version)
    ap.add_argument("--delay", type=float, default=0.4, help="seconds between requests")
    ap.add_argument("--max-pages", type=int, default=None)
    a = ap.parse_args()
    base = f"{s.pg_docs_base_url}/{a.version}"
    out = s.raw_html_dir
    out.mkdir(parents=True, exist_ok=True)
    hdr = {"User-Agent": "DocRAG-Eval/0.1 (research; polite crawler)"}
    with httpx.Client(headers=hdr, timeout=30, follow_redirects=True) as c:
        idx = c.get(f"{base}/index.html")
        idx.raise_for_status()
        (out / "index.html").write_text(idx.text, encoding="utf-8")
        pages = []
        for el in BeautifulSoup(idx.text, "html.parser").find_all("a", href=True):
            h = el["href"].split("#")[0]
            if h.endswith(".html") and "/" not in h and h != "index.html" and not SKIP.match(h) and h not in pages:
                pages.append(h)
        pages = pages[: a.max_pages] if a.max_pages else pages
        print(f"{len(pages)} pages to fetch for PostgreSQL {a.version}")
        for i, p in enumerate(pages, 1):
            dest = out / p
            if dest.exists():
                continue
            r = c.get(f"{base}/{p}")
            if r.status_code == 200:
                dest.write_text(r.text, encoding="utf-8")
            else:
                print("skip", p, r.status_code)
            time.sleep(a.delay)
            if i % 50 == 0:
                print(f"  {i}/{len(pages)}")
    # index.html is only a table of contents: remove it so it is not ingested as content
    (out / "index.html").unlink(missing_ok=True)


if __name__ == "__main__":
    main()

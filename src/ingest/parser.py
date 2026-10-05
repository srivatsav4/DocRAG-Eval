"""Parse PostgreSQL documentation HTML into hierarchical sections (BeautifulSoup).

Each DocBook-generated section <div class="sect1|sect2|chapter|refentry|refsect1..." id="..."> becomes a
Section containing ONLY its own text (nested sub-sections are separate Sections). Navigation, footer,
TOC and boilerplate are stripped. Section IDs are lower-cased anchors (e.g. 'indexes-partial').
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlparse

from bs4 import BeautifulSoup, Tag

from src.ingest.stable_ids import normalize_section_id

SECTION_CLASSES = {
    "part", "preface", "chapter", "appendix", "sect1", "sect2", "sect3", "sect4", "sect5",
    "section", "refentry", "refsect1", "refsect2", "refsect3",
}
BOILERPLATE_SELECTORS = [
    "script", "style", "noscript", "nav", "header", "footer", "form",
    "div.navheader", "div.navfooter", "div#pgHeader", "div#pgFooter", "div#docFooter",
    "div#pgTopNav", "div#docSearch", "div#pgSideWrap", "div.toc", "div.docNav", "a.id_link",
]
HEADINGS = ["h1", "h2", "h3", "h4", "h5", "h6"]


@dataclass
class Section:
    section_id: str
    anchor: str
    title: str
    breadcrumb: str
    ancestors: list[str]
    url: str
    text: str
    page: str
    order: int
    level: int = 0


def _classes(tag: Tag) -> set[str]:
    return set(tag.get("class") or [])


def _is_section(tag) -> bool:
    return (isinstance(tag, Tag) and tag.name in {"div", "section"} and bool(tag.get("id"))
            and bool(_classes(tag) & SECTION_CLASSES))


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _title_of(tag: Tag) -> str:
    if "refentry" in _classes(tag):
        rt = tag.find(class_="refentrytitle")
        if rt:
            return _clean(rt.get_text(" "))
    for child in tag.children:
        if not isinstance(child, Tag):
            continue
        if child.name in HEADINGS:
            return _clean(child.get_text(" "))
        if child.name == "div" and "titlepage" in _classes(child):
            h = child.find(HEADINGS)
            if h:
                return _clean(h.get_text(" "))
    return str(tag.get("id"))


def _own_text(tag: Tag) -> str:
    clone = BeautifulSoup(str(tag), "html.parser").find()
    while True:
        sub = next((t for t in clone.descendants if _is_section(t)), None)
        if sub is None:
            break
        sub.decompose()
    for tp in clone.find_all("div", class_="titlepage"):
        tp.decompose()
    for h in clone.find_all(HEADINGS, recursive=False):
        h.decompose()
    return _clean(clone.get_text(" ", strip=True))


def parse_html(html: str, page_url: str, pg_version: str = "") -> list[Section]:
    soup = BeautifulSoup(html, "html.parser")
    for sel in BOILERPLATE_SELECTORS:
        for el in soup.select(sel):
            el.decompose()
    root = soup.select_one("#docContent") or soup.body or soup
    page = urlparse(page_url).path.rsplit("/", 1)[-1] or page_url
    base_url = page_url.split("#")[0]

    sections: list[Section] = []
    seen: set[str] = set()
    for tag in root.find_all(_is_section):
        sid = normalize_section_id(tag["id"])
        if sid in seen:
            continue
        seen.add(sid)
        anc_tags = [p for p in tag.parents if _is_section(p)][::-1]
        titles = [_title_of(a) for a in anc_tags] + [_title_of(tag)]
        text = _own_text(tag)
        if not text:
            continue
        sections.append(Section(
            section_id=sid, anchor=tag["id"], title=titles[-1], breadcrumb=" > ".join(titles),
            ancestors=[normalize_section_id(a["id"]) for a in anc_tags],
            url=f"{base_url}#{tag['id']}", text=text, page=page, order=len(sections), level=len(anc_tags),
        ))
    if not sections:  # page without DocBook section wrappers
        text = _clean(root.get_text(" ", strip=True))
        if text:
            title = _clean(soup.title.get_text()) if soup.title else page
            sid = normalize_section_id(page.rsplit(".", 1)[0])
            sections.append(Section(sid, sid, title, title, [], base_url, text, page, 0, 0))
    return sections

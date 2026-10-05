from src.ingest.parser import parse_html
from tests.conftest import PAGES


def test_boilerplate_stripped_and_ids_lowercased(sections):
    ids = {s.section_id for s in sections}
    assert "indexes-partial" in ids and "sql-createindex" in ids
    blob = " ".join(s.text for s in sections)
    for junk in ("SITE HEADER", "Copyright PostgreSQL", "Prev", "Next"):
        assert junk not in blob


def test_nested_sections_separate_and_breadcrumb(sections):
    by = {s.section_id: s for s in sections}
    notes = by["indexes-expressional-notes"]
    assert notes.ancestors == ["indexes", "indexes-expressional"]
    assert notes.breadcrumb.endswith("11.7. Indexes on Expressions > Notes")
    assert "relatively expensive" not in by["indexes-expressional"].text  # own text only
    assert "relatively expensive" in notes.text


def test_title_has_no_id_link_hash(sections):
    assert all(not s.title.endswith("#") for s in sections)


def test_refentry_title_and_refsect(sections):
    by = {s.section_id: s for s in sections}
    assert by["sql-createindex"].title == "CREATE INDEX"
    assert by["sql-createindex-synopsis"].ancestors == ["sql-createindex"]


def test_url_has_anchor_and_fallback_page():
    secs = parse_html(PAGES["backup-dump.html"], "https://x/docs/16/backup-dump.html", "16")
    assert secs[0].url == "https://x/docs/16/backup-dump.html#BACKUP-DUMP"
    fb = parse_html("<html><body><div id='docContent'><p>plain page text</p></div></body></html>", "https://x/docs/16/plain.html")
    assert fb[0].section_id == "plain" and "plain page text" in fb[0].text

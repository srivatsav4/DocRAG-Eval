import pytest

from src.ingest.chunker import _windows, chunk_sections
from src.ingest.parser import Section


def _sec(sid, n, page="p.html"):
    return Section(sid, sid.upper(), sid, sid, [], f"u#{sid}", " ".join(f"{sid}w{i}" for i in range(n)), page, 0)


def test_section_based_one_chunk_per_section(sections):
    ch = chunk_sections(sections, "section_based", "16", "t")
    assert len(ch) == len(sections)
    assert all(c.strategy == "section_based" and c.section_id for c in ch)


def test_windows_overlap_and_cover():
    w = _windows(1000, 300, 50)
    assert w[0] == (0, 300) and w[1][0] == 250 and w[-1][1] == 1000
    with pytest.raises(ValueError):
        _windows(10, 5, 5)


def test_long_section_split_with_overlap():
    ch = chunk_sections([_sec("a", 1200)], "section_based", "16", "t")
    assert len(ch) == 3 and all(c.n_words <= 500 for c in ch)
    assert set(ch[0].text.split()[-50:]) == set(ch[1].text.split()[:50])


def test_fixed_chunks_cross_section_boundaries_majority_owner():
    ch = chunk_sections([_sec("a", 100), _sec("b", 500)], "fixed_300_overlap", "16", "t")
    assert ch[0].section_id == "b" and set(ch[0].spanned_sections) == {"a", "b"}


def test_deterministic_ids(sections):
    a = chunk_sections(sections, "fixed_300_overlap", "16", "t1")
    b = chunk_sections(sections, "fixed_300_overlap", "16", "t2")
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert len({c.chunk_id for c in a}) == len(a)


def test_unknown_strategy():
    with pytest.raises(ValueError):
        chunk_sections([], "nope")

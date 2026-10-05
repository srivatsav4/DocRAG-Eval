from src.ingest.stable_ids import content_hash, make_chunk_id, normalize_section_id


def test_hash_whitespace_insensitive_and_sensitive_to_content():
    assert content_hash("a  b\n c") == content_hash("a b c")
    assert content_hash("a b c") != content_hash("a b d")


def test_section_id_normalisation():
    assert normalize_section_id("#INDEXES-PARTIAL ") == "indexes-partial"


def test_chunk_id_format_stable():
    assert make_chunk_id("16", "x", "section_based", 3, "abcdef0123") == "pg16:x:section_based:003:abcdef01"

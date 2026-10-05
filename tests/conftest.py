import copy
import json
from pathlib import Path

import pytest

from src.config import Settings, load_tunables
from src.ingest.chunker import chunk_sections
from src.ingest.parser import parse_html
from src.pipeline import RAGPipeline

NAV = '<div class="navheader"><table><tr><th>Prev</th><th>Next</th></tr></table></div>'
FOOT = '<div class="navfooter"><a href="x.html">Prev</a></div><div id="pgFooter">Copyright PostgreSQL</div>'


def page(body: str) -> str:
    return f'<html><head><title>t</title></head><body><div id="pgHeader">SITE HEADER</div><div id="docContent">{NAV}{body}{FOOT}</div></body></html>'


def sect(cls, sid, title, text, inner=""):
    return (f'<div class="{cls}" id="{sid}"><div class="titlepage"><div><div><h2 class="title">{title} '
            f'<a href="#{sid}" class="id_link">#</a></h2></div></div></div><p>{text}</p>{inner}</div>')


PAGES = {
    "indexes.html": page(sect("chapter", "INDEXES", "Chapter 11. Indexes", "Indexes are a common way to enhance database performance.",
        sect("sect1", "INDEXES-PARTIAL", "11.8. Partial Indexes",
             "A partial index is an index built over a subset of a table; the subset is defined by a conditional expression called "
             "the predicate of the partial index. Create a partial index with CREATE INDEX followed by a WHERE clause, for example "
             "CREATE INDEX orders_unbilled_index ON orders (order_nr) WHERE billed is not true. Partial indexes are smaller and faster.")
        + sect("sect1", "INDEXES-EXPRESSIONAL", "11.7. Indexes on Expressions",
               "An index column need not be just a column of the underlying table, but can be a function or scalar expression computed "
               "from one or more columns. Create an expression index such as CREATE INDEX test1_lower_col1_idx ON test1 (lower(col1)).",
               sect("sect2", "INDEXES-EXPRESSIONAL-NOTES", "Notes", "Index expressions are relatively expensive to maintain."))
        + sect("sect1", "INDEXES-UNIQUE", "11.6. Unique Indexes", "Indexes can also be used to enforce uniqueness of a column's value."))),
    "sql-createindex.html": page(
        f'<div class="refentry" id="SQL-CREATEINDEX"><div class="titlepage"></div><div class="refnamediv"><h2>Name</h2>'
        f'<p><span class="refentrytitle">CREATE INDEX</span> — define a new index</p></div>'
        f'<div class="refsect1" id="SQL-CREATEINDEX-SYNOPSIS"><h2>Synopsis</h2><pre>CREATE [ UNIQUE ] INDEX [ CONCURRENTLY ] name ON table_name ( column )</pre></div></div>'),
    "runtime-config-resource.html": page(sect("sect1", "RUNTIME-CONFIG-RESOURCE", "20.4. Resource Consumption",
        "work_mem sets the base maximum amount of memory to be used by a query operation such as a sort or hash table before writing to "
        "temporary disk files. The default value is four megabytes (4MB). shared_buffers sets the amount of memory the database server uses "
        "for shared memory buffers. The default is typically 128 megabytes (128MB).")),
    "backup-dump.html": page(sect("sect1", "BACKUP-DUMP", "26.1. SQL Dump",
        "The idea behind this dump method is to generate a file with SQL commands that, when fed back to the server, will recreate the "
        "database in the same state as it was at the time of the dump. PostgreSQL provides the utility program pg_dump for this purpose.")),
    "continuous-archiving.html": page(sect("sect1", "CONTINUOUS-ARCHIVING", "26.3. Continuous Archiving and Point-in-Time Recovery (PITR)",
        "PostgreSQL maintains a write ahead log (WAL) in the pg_wal subdirectory. Continuous archiving copies completed WAL segment files "
        "using archive_command, allowing point-in-time recovery from a base backup.")),
    "ddl-partitioning.html": page(sect("sect1", "DDL-PARTITIONING", "5.12. Table Partitioning",
        "Partitioning refers to splitting what is logically one large table into smaller physical pieces. Declarative partitioning lets you "
        "create a partitioned table with PARTITION BY RANGE, LIST or HASH and attach partitions.")),
    "routine-vacuuming.html": page(sect("sect1", "ROUTINE-VACUUMING", "25.1. Routine Vacuuming",
        "PostgreSQL databases require periodic maintenance known as vacuuming. The autovacuum daemon automates vacuum and analyze to reclaim "
        "dead row space and avoid table bloat.")),
    "transaction-iso.html": page(sect("sect1", "TRANSACTION-ISO", "13.2. Transaction Isolation",
        "The SQL standard defines four transaction isolation levels. In PostgreSQL the default isolation level is Read Committed. "
        "Repeatable Read and Serializable are also supported.")),
    "ssl-tcp.html": page(sect("sect1", "SSL-TCP", "19.9. Secure TCP/IP Connections with SSL",
        "PostgreSQL has native support for using SSL connections to encrypt client/server communications. Set ssl to on and provide "
        "server.crt and server.key files to enable SSL connections.")),
}


@pytest.fixture(scope="session")
def sections():
    secs = []
    for name, html in PAGES.items():
        secs += parse_html(html, f"https://www.postgresql.org/docs/16/{name}", "16")
    return secs


@pytest.fixture(scope="session")
def chunks(sections):
    return chunk_sections(sections, "section_based", "16", "2026-01-01T00:00:00+00:00")


@pytest.fixture
def tun():
    t = load_tunables()
    t = copy.deepcopy(t)
    t["semantic_floor"], t["semantic_ceiling"] = 0.0, 0.6  # hash embedder cosines are small; real models need calibration
    return t


class FakeLLM:
    name = "fake-llm"

    def __init__(self, text="A partial index is built over a subset of a table [S1]. It uses a WHERE predicate [S1]."):
        self.text, self.calls = text, 0

    def generate(self, system, prompt):
        self.calls += 1
        return self.text


@pytest.fixture
def settings(tmp_path):
    s = Settings()
    s.embedder_backend, s.reranker_backend, s.dense_backend = "hash", "lexical", "numpy"
    s.processed_dir, s.results_dir, s.thresholds_file = tmp_path / "proc", tmp_path / "res", tmp_path / "thr.json"
    s.gold_dir = tmp_path / "gold"
    s.gold_dir.mkdir()
    return s


@pytest.fixture
def pipeline(settings, tun, chunks):
    from src.retrieval.dense import HashingEmbedder
    from src.retrieval.reranker import LexicalOverlapReranker

    return RAGPipeline(settings, tun, embedder=HashingEmbedder(), reranker=LexicalOverlapReranker(), llm=FakeLLM(),
                       chunks_provider=lambda c: chunks)

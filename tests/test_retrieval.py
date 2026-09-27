import hashlib

import numpy as np
import pymupdf
import pytest
from pydantic import ValidationError

from app.config import Settings
from app.ingestion import Document, build_index, chunk_document, current_index, verify_pdf
from app.retrieval import HybridRetriever, fuse_ranks


class WordTokenizer:
    def encode(self, text, **kwargs):
        return text.split()


class SmallEmbedding:
    tokenizer = WordTokenizer()
    max_seq_length = 512

    def __init__(self, *args, **kwargs):
        pass

    def encode(self, texts, **kwargs):
        vectors = np.array(
            [[text.lower().count("tax") + 1, text.lower().count("housing") + 1] for text in texts],
            dtype="float32",
        )
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


class SmallReranker:
    def __init__(self, *args, **kwargs):
        pass

    def predict(self, pairs, **kwargs):
        return np.array(
            [len(set(q.lower().split()) & set(t.lower().split())) + 1 for q, t in pairs]
        )


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    import json

    monkeypatch.setattr("app.ingestion.SentenceTransformer", SmallEmbedding)
    monkeypatch.setattr("app.retrieval.SentenceTransformer", SmallEmbedding)
    monkeypatch.setattr("app.retrieval.CrossEncoder", SmallReranker)
    rows = []
    for country, year in [("Canada", 2024), ("Canada", 2025), ("Germany", 2025)]:
        document_id = f"{country.lower()}-{year}"
        pdf = pymupdf.open()
        for page in range(5):
            text = (
                f"Tax and benefit policy descriptions for {country} {year}\nCopyright 2026"
                if page == 0
                else f"8. Taxes\nThe {country} tax rules for {year} cover employment income.\n"
                "A rare term ZYX benefit explains housing support."
            )
            pdf.new_page().insert_text((60, 100), text)
        content = pdf.tobytes()
        path = tmp_path / f"{document_id}.pdf"
        path.write_bytes(content)
        rows.append(
            dict(
                document_id=document_id,
                document_name=document_id,
                country=country,
                policy_year=year,
                filename=path.name,
                source="https://www.oecd.org/test.pdf",
                sha256=hashlib.sha256(content).hexdigest(),
                pages=5,
                year_evidence=f"{country} {year}",
            )
        )
    manifest = tmp_path / "corpus.json"
    manifest.write_text(json.dumps(rows))
    settings = Settings(corpus_path=manifest, data_path=tmp_path, index_path=tmp_path / "index")
    return settings, [Document.model_validate(r) for r in rows]


def test_metadata_comes_from_verified_policy_title(corpus):
    settings, documents = corpus
    doc = documents[0]
    content = (settings.data_path / doc.filename).read_bytes()
    chunks = chunk_document(content, doc, WordTokenizer(), 100, 12)
    assert chunks and all(c["policy_year"] == 2024 for c in chunks)
    assert all(c["country"] == "Canada" and 1 <= c["page_number"] <= 5 for c in chunks)
    assert all(c["source"] == doc.source and c["chunk_id"] for c in chunks)
    with pytest.raises(ValueError, match="not confirmed"):
        verify_pdf(content, doc.model_copy(update={"policy_year": 2026}))


def test_path_traversal_rejected(corpus):
    _, documents = corpus
    with pytest.raises(ValidationError):
        Document.model_validate({**documents[0].model_dump(), "filename": "../outside.pdf"})


def test_unreadable_and_changed_pdf_fail(corpus):
    settings, documents = corpus
    with pytest.raises(ValueError, match="expected a PDF"):
        verify_pdf(b"not a pdf", documents[0])
    content = (settings.data_path / documents[0].filename).read_bytes()
    with pytest.raises(ValueError, match="checksum"):
        verify_pdf(content + b"changed", documents[0])


def test_index_idempotency_and_staleness(corpus):
    settings, documents = corpus
    first = build_index(settings)
    path = current_index(settings) / "canada-2024.faiss"
    modified = path.stat().st_mtime_ns
    assert build_index(settings) == first
    assert path.stat().st_mtime_ns == modified
    with pytest.raises(ValueError, match="stale"):
        HybridRetriever(settings.model_copy(update={"chunk_tokens": 300}))


@pytest.mark.parametrize(
    "scopes",
    [
        {("Canada", 2025)},
        {("Canada", 2024)},
        {("Canada", 2024), ("Canada", 2025)},
        {("Canada", 2025), ("Germany", 2025)},
    ],
)
def test_filters_before_both_searches(corpus, scopes):
    settings, _ = corpus
    build_index(settings)
    retriever = HybridRetriever(settings)
    evidence, diagnostics = retriever.search("tax ZYX benefit", scopes)
    assert {(e["country"], e["policy_year"]) for e in evidence} == scopes
    assert diagnostics["eligible_chunks"] == sum(len(retriever.groups[s]) for s in scopes)
    assert any("bm25" in e["retrieval_methods"] for e in evidence)
    assert all(e["chunk_id"] in {c["chunk_id"] for c in retriever.chunks} for e in evidence)


def test_missing_scope_fails_comparison(corpus):
    settings, _ = corpus
    build_index(settings)
    evidence, _ = HybridRetriever(settings).search("tax", {("Canada", 2025), ("Canada", 2030)})
    assert not evidence


def test_rrf_rewards_agreement_and_ignores_duplicates():
    assert fuse_ranks([1, 2, 3], [2, 4, 1])[0] == 2
    assert fuse_ranks([1, 1, 2], [2]) == fuse_ranks([1, 2], [2])
    assert fuse_ranks([], []) == []


def test_chunk_budget_and_empty_page(corpus):
    settings, documents = corpus
    doc = documents[0]
    content = (settings.data_path / doc.filename).read_bytes()
    chunks = chunk_document(content, doc, WordTokenizer(), 15, 2)
    assert all(len(c["text"].split()) <= 15 for c in chunks)
    pdf = pymupdf.open(stream=content, filetype="pdf")
    pdf.new_page()
    content = pdf.tobytes()
    doc = doc.model_copy(update={"pages": 6, "sha256": hashlib.sha256(content).hexdigest()})
    with pytest.raises(ValueError, match="no extractable text"):
        chunk_document(content, doc, WordTokenizer(), 100, 12)

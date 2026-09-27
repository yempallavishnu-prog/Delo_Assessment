import argparse
import hashlib
import io
import json
import logging
import re
import tempfile
import zipfile
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import faiss
import httpx
import numpy as np
import pymupdf
from pydantic import BaseModel, Field, ValidationError, field_validator
from sentence_transformers import SentenceTransformer

from app.config import Settings

INDEX_VERSION = 3


class Document(BaseModel):
    document_id: str = Field(pattern=r"^[a-z0-9-]+$")
    document_name: str = Field(min_length=1)
    country: str = Field(min_length=1)
    policy_year: int = Field(ge=1900, le=2100)
    filename: str
    source: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    pages: int = Field(gt=0)
    year_evidence: str = Field(min_length=1)
    archive_member: str | None = None

    @field_validator("filename")
    @classmethod
    def local_pdf(cls, value):
        if Path(value).name != value or not value.lower().endswith(".pdf"):
            raise ValueError("PDF filenames must be plain local filenames.")
        return value

    @field_validator("source")
    @classmethod
    def official_source(cls, value):
        url = urlparse(value)
        if url.scheme != "https" or url.hostname not in {"www.oecd.org", "webfs.oecd.org"}:
            raise ValueError("Corpus sources must be official OECD HTTPS URLs.")
        return value


def load_corpus(settings: Settings) -> list[Document]:
    try:
        documents = [
            Document.model_validate(row) for row in json.loads(settings.corpus_path.read_text())
        ]
    except (OSError, ValueError, TypeError, ValidationError) as exc:
        raise ValueError(f"Cannot read corpus metadata from {settings.corpus_path}.") from exc
    if not documents or len({d.document_id for d in documents}) != len(documents):
        raise ValueError("Corpus metadata is empty or contains duplicate document IDs.")
    if len({d.filename for d in documents}) != len(documents):
        raise ValueError("Each corpus document must use a distinct PDF file.")
    return documents


def document_path(settings: Settings, document: Document) -> Path:
    path = (settings.data_path / document.filename).resolve()
    if not path.is_relative_to(settings.data_path.resolve()):
        raise ValueError(f"PDF path escapes the corpus directory: {document.filename}")
    return path


def verify_pdf(content: bytes, document: Document):
    if not content.startswith(b"%PDF-") or len(content) > 25 * 1024 * 1024:
        raise ValueError(f"{document.filename}: expected a PDF smaller than 25 MB.")
    if hashlib.sha256(content).hexdigest() != document.sha256:
        raise ValueError(f"{document.filename}: checksum changed; review the source and manifest.")
    try:
        with pymupdf.open(stream=content, filetype="pdf") as pdf:
            if pdf.needs_pass or len(pdf) != document.pages:
                raise ValueError(f"{document.filename}: encrypted PDF or unexpected page count.")
            cover = " ".join(pdf[0].get_text().split())
            title = (
                rf"policy descriptions for (?:the )?{re.escape(document.country)} "
                rf"{document.policy_year}\b"
            )
            if not re.search(title, cover, re.I):
                raise ValueError(
                    f"{document.filename}: country/policy year not confirmed on cover."
                )
    except pymupdf.FileDataError as exc:
        raise ValueError(f"{document.filename}: PDF is unreadable.") from exc


def download_corpus(settings: Settings):
    settings.data_path.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=90, follow_redirects=True) as client:
        for document in load_corpus(settings):
            path = document_path(settings, document)
            if path.exists():
                verify_pdf(path.read_bytes(), document)
                print(f"verified {document.filename}")
                continue
            # Archives stay in memory; only the named PDF reaches disk.
            with client.stream("GET", document.source) as response:
                response.raise_for_status()
                content = bytearray()
                for block in response.iter_bytes():
                    content.extend(block)
                    if len(content) > 60 * 1024 * 1024:
                        raise ValueError("OECD download exceeds the 60 MB limit.")
            if document.archive_member:
                with zipfile.ZipFile(io.BytesIO(content)) as archive:
                    member = archive.getinfo(document.archive_member)
                    if member.file_size > 25 * 1024 * 1024:
                        raise ValueError("Archived PDF exceeds the 25 MB limit.")
                    content = archive.read(member)
            content = bytes(content)
            verify_pdf(content, document)
            temporary = path.with_suffix(".tmp")
            temporary.write_bytes(content)
            temporary.replace(path)
            print(f"downloaded {document.filename}")


def fingerprint(settings: Settings, documents: list[Document]) -> str:
    digest = hashlib.sha256(
        json.dumps(
            {
                "version": INDEX_VERSION,
                "embedding": settings.embedding_model,
                "chunk_tokens": settings.chunk_tokens,
                "overlap_tokens": settings.overlap_tokens,
                "documents": [d.model_dump() for d in documents],
            },
            sort_keys=True,
        ).encode()
    )
    for document in documents:
        path = document_path(settings, document)
        if not path.exists():
            raise ValueError(
                f"Missing {document.filename}. Run python -m app.ingestion --download."
            )
        if path.stat().st_size > 25 * 1024 * 1024:
            raise ValueError(f"{document.filename}: PDF exceeds the 25 MB limit.")
        content = path.read_bytes()
        verify_pdf(content, document)
        digest.update(content)
    return digest.hexdigest()


def current_index(settings: Settings) -> Path:
    try:
        name = (settings.index_path / "CURRENT").read_text().strip()
    except OSError as exc:
        raise ValueError("Index has not been built. Run python -m app.ingestion.") from exc
    if not re.fullmatch(r"[a-f0-9]{64}", name):
        raise ValueError("Index pointer is invalid. Rebuild the index.")
    return settings.index_path / name


def search_text(chunk: dict) -> str:
    return (
        f"{chunk['country']} policy rules in {chunk['policy_year']}. "
        f"{chunk['section_title'] or ''}\n{chunk['text']}"
    )


def page_sections(pdf):
    # Remove repeated running lines only at page edges, never matching body text.
    edge_counts = Counter()
    for page in pdf:
        lines = [line.strip() for line in page.get_text().splitlines() if line.strip()]
        edge_counts.update(set(lines[:3] + lines[-3:]))
    recurring = {line for line, count in edge_counts.items() if count >= max(3, len(pdf) // 2)}
    section = None
    headings = {}
    for page_number, page in enumerate(pdf, 1):
        blocks = page.get_text("blocks", sort=True)
        if not page.get_text().strip():
            raise ValueError(f"Page {page_number} has no extractable text; OCR is not supported.")
        paragraphs = []
        for block in blocks:
            if block[6] != 0:
                continue
            lines = [line.strip() for line in block[4].splitlines() if line.strip()]
            at_edge = block[1] < 75 or block[3] > page.rect.height - 65
            lines = [
                line
                for line in lines
                if not (at_edge and (line in recurring or re.fullmatch(r"[\d\s\uf07c|]+", line)))
            ]
            text = "\n".join(re.sub(r"[ \t]+", " ", line) for line in lines)
            if text:
                paragraphs.append(text)
        # Front matter and the contents page are poor evidence even when they match a query.
        body = "\n\n".join(paragraphs)
        if page_number <= 4 and (
            "Table of contents" in body
            or "Preface" in body
            or "policy descriptions for" in body.lower()
            or "responsibility of the" in body
        ):
            continue
        for paragraph in paragraphs:
            heading = re.match(r"^(\d{1,2}(?:\.\d+)*\.)\s+([A-Z][^\n]{2,100})", paragraph)
            if heading:
                number = heading.group(1).rstrip(".")
                headings = {
                    key: value for key, value in headings.items() if number.startswith(key + ".")
                }
                headings[number] = heading.group(0)
                section = " / ".join(headings.values())
            if paragraph.strip():
                yield page_number, section, paragraph


def chunk_document(content: bytes, document: Document, tokenizer, size: int, overlap: int):
    verify_pdf(content, document)

    def token_count(text):
        return len(tokenizer.encode(text, add_special_tokens=False, verbose=False))

    def split(text):
        if token_count(text) <= size:
            return [text]
        for separator in ["\n", ". ", "; ", " "]:
            parts = text.split(separator)
            if len(parts) > 1:
                return [
                    piece
                    for i, part in enumerate(parts)
                    if part
                    for piece in split(part + (separator.strip() if i < len(parts) - 1 else ""))
                ]
        raise ValueError(f"{document.filename}: a text token exceeds the chunk limit.")

    chunks = []
    pending, page, section = [], None, None

    def emit():
        if not pending:
            return
        text = "\n\n".join(pending)
        chunk_id = hashlib.sha256(f"{document.document_id}:{page}:{text}".encode()).hexdigest()[:24]
        chunks.append(
            {
                "chunk_id": chunk_id,
                "document_id": document.document_id,
                "document_name": document.document_name,
                "country": document.country,
                "policy_year": document.policy_year,
                "source": document.source,
                "page_number": page,
                "section_title": section,
                "text": text,
            }
        )

    with pymupdf.open(stream=content, filetype="pdf") as pdf:
        for next_page, next_section, paragraph in page_sections(pdf):
            if (next_page, next_section) != (page, section):
                # Carry a short parent heading into its first subsection.
                if next_page == page and token_count(" ".join(pending)) < 14:
                    section = next_section
                else:
                    emit()
                    pending = []
                    page, section = next_page, next_section
            for piece in split(paragraph):
                if pending and token_count("\n\n".join([*pending, piece])) > size:
                    emit()
                    tail = []
                    for previous in reversed(pending):
                        if token_count("\n\n".join([previous, *tail])) > overlap:
                            break
                        tail.insert(0, previous)
                    pending = tail
                    if token_count("\n\n".join([*pending, piece])) > size:
                        pending = []
                pending.append(piece)
        emit()
    if not chunks:
        raise ValueError(f"{document.filename}: no indexable policy text found.")
    return chunks


def build_index(settings: Settings) -> dict:
    documents = load_corpus(settings)
    stamp = fingerprint(settings, documents)
    destination = settings.index_path / stamp
    if destination.exists():
        summary = json.loads((destination / "manifest.json").read_text())
    else:
        model = SentenceTransformer(settings.embedding_model, device=settings.model_device)
        if settings.chunk_tokens + 100 > model.max_seq_length:
            raise ValueError("Chunk size exceeds the embedding model's input limit.")
        chunks = []
        for document in documents:
            extracted = chunk_document(
                document_path(settings, document).read_bytes(),
                document,
                model.tokenizer,
                settings.chunk_tokens,
                settings.overlap_tokens,
            )
            chunks.extend(extracted)
            print(f"{document.document_id}: {len(extracted)} chunks")
        settings.index_path.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=settings.index_path) as temporary:
            target = Path(temporary) / "build"
            target.mkdir()
            for document in documents:
                rows = [c for c in chunks if c["document_id"] == document.document_id]
                vectors = model.encode(
                    [search_text(c) for c in rows],
                    normalize_embeddings=True,
                    batch_size=32,
                    show_progress_bar=False,
                )
                dense = faiss.IndexFlatIP(vectors.shape[1])
                dense.add(np.asarray(vectors, dtype="float32"))
                faiss.write_index(dense, str(target / f"{document.document_id}.faiss"))
            (target / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False))
            summary = {
                "fingerprint": stamp,
                "documents": len(documents),
                "chunks": len(chunks),
                "countries": sorted({d.country for d in documents}),
                "years": sorted({d.policy_year for d in documents}),
                "embedding_model": settings.embedding_model,
                "index_version": INDEX_VERSION,
            }
            (target / "manifest.json").write_text(json.dumps(summary, indent=2))
            target.rename(destination)
    pointer = settings.index_path / "CURRENT.tmp"
    pointer.write_text(stamp)
    pointer.replace(settings.index_path / "CURRENT")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify OECD PDFs and build the local search index."
    )
    parser.add_argument("--download", action="store_true", help="Fetch only the ten manifest PDFs.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    try:
        if args.download:
            download_corpus(Settings())
        else:
            print(json.dumps(build_index(Settings()), indent=2))
    except (ValueError, OSError, httpx.HTTPError, zipfile.BadZipFile, KeyError) as error:
        parser.exit(1, f"Ingestion failed: {error}\n")

import json
import logging
import re
import time
from collections import defaultdict

import faiss
import numpy as np
from rank_bm25 import BM25Plus
from sentence_transformers import CrossEncoder, SentenceTransformer

from app.config import Settings
from app.ingestion import current_index, fingerprint, load_corpus, search_text

log = logging.getLogger(__name__)
STOP_WORDS = set(
    (
        "a an and are as at be by for from how in is it of on or the to was were what which with"
    ).split()
)


def tokens(text: str) -> list[str]:
    return [
        word
        for word in re.findall(r"\w+(?:\.\d+)?", text.lower().replace(",", ""))
        if word not in STOP_WORDS
    ]


def fuse_ranks(*rankings: list[int]) -> list[int]:
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, item in enumerate(dict.fromkeys(ranking), 1):
            scores[item] += 1 / (60 + rank)
    return sorted(scores, key=lambda item: (-scores[item], item))


def policy_topic(query: str) -> str | None:
    tax = re.search(
        r"\b(?:income tax|tax rules|tax treatment|tax situation|tax brackets)\b", query, re.I
    )
    benefit = re.search(r"\b(?:benefit|pension|contribution|insurance|credit)\b", query, re.I)
    return "income_tax" if tax and not benefit else None


class HybridRetriever:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.documents = load_corpus(settings)
        directory = current_index(settings)
        self.manifest = json.loads((directory / "manifest.json").read_text())
        if fingerprint(settings, self.documents) != self.manifest["fingerprint"]:
            raise ValueError("Index is stale. Run python -m app.ingestion and restart the app.")
        self.chunks = json.loads((directory / "chunks.json").read_text())
        self.groups = defaultdict(list)
        for index, chunk in enumerate(self.chunks):
            self.groups[(chunk["country"], chunk["policy_year"])].append(index)
        self.dense = {}
        for document in self.documents:
            self.dense[document.document_id] = faiss.read_index(
                str(directory / f"{document.document_id}.faiss")
            )
        self.embedding = SentenceTransformer(settings.embedding_model, device=settings.model_device)
        self.reranker = CrossEncoder(settings.reranker_model, device=settings.model_device)

    def search(self, query: str, scopes: set[tuple[str, int]]):
        started = time.perf_counter()
        if not scopes or len(scopes) > 4:
            raise ValueError("Choose between one and four country/year scopes.")
        eligible = sorted(i for scope in scopes for i in self.groups.get(scope, []))
        diagnostics = {
            "eligible_chunks": len(eligible),
            "dense_candidates": 0,
            "bm25_candidates": 0,
            "fused_candidates": 0,
            "reranked": 0,
            "context_chunks": 0,
            "timing_ms": {},
        }
        if any(not self.groups.get(scope) for scope in scopes):
            return [], diagnostics
        topic = policy_topic(query)
        if topic == "income_tax":
            query = f"Income tax rules. {query}"
        diagnostics["timing_ms"]["filter"] = round((time.perf_counter() - started) * 1000, 1)
        retrieved_at = time.perf_counter()
        vector = np.asarray(
            self.embedding.encode([query], normalize_embeddings=True), dtype="float32"
        )
        selected, methods = set(), defaultdict(set)
        diagnostics["policy_topic"] = topic
        # Each scope competes independently so comparisons cannot starve one side.
        for scope in sorted(scopes):
            ids = self.groups[scope]
            if topic == "income_tax":
                tax_ids = [
                    i
                    for i in ids
                    if re.search(
                        r"^\d+\.\s+(?:Taxes|Income tax)\b",
                        self.chunks[i]["section_title"] or "",
                        re.I,
                    )
                ]
                if tax_ids:
                    ids = tax_ids
            dense_scores = []
            for document in self.documents:
                if (document.country, document.policy_year) != scope:
                    continue
                document_ids = [
                    i
                    for i in self.groups[scope]
                    if self.chunks[i]["document_id"] == document.document_id
                ]
                allowed = np.array(
                    [n for n, i in enumerate(document_ids) if i in ids], dtype="int64"
                )
                if not len(allowed):
                    continue
                scores, positions = self.dense[document.document_id].search(
                    vector,
                    min(self.settings.candidates, len(allowed)),
                    params=faiss.SearchParameters(sel=faiss.IDSelectorBatch(allowed)),
                )
                dense_scores.extend(
                    (float(score), document_ids[int(position)])
                    for score, position in zip(scores[0], positions[0], strict=True)
                )
            dense_ids = [
                i for _, i in sorted(dense_scores, reverse=True)[: self.settings.candidates]
            ]
            corpus_tokens = [tokens(search_text(self.chunks[i])) for i in ids]
            keyword = BM25Plus(corpus_tokens, delta=0)
            query_tokens = tokens(query)
            scores = keyword.get_scores(query_tokens)
            keyword_ids = [
                ids[int(i)] for i in np.argsort(-scores, kind="stable") if scores[i] > 0
            ][: self.settings.candidates]
            fused = fuse_ranks(dense_ids, keyword_ids)
            selected.update(fused[: self.settings.rerank_candidates])
            for i in dense_ids:
                methods[i].add("dense")
            for i in keyword_ids:
                methods[i].add("bm25")
            diagnostics["dense_candidates"] += len(dense_ids)
            diagnostics["bm25_candidates"] += len(keyword_ids)
            diagnostics["fused_candidates"] += len(fused)
        diagnostics["timing_ms"]["retrieve"] = round((time.perf_counter() - retrieved_at) * 1000, 1)
        ranked_at = time.perf_counter()
        candidates = sorted(selected)
        # Raw logits are intentional: zero is a conservative baseline, not a probability.
        import torch

        scores = self.reranker.predict(
            [(query, search_text(self.chunks[i])) for i in candidates],
            activation_fn=torch.nn.Identity(),
            show_progress_bar=False,
        )
        ranked = sorted(zip(candidates, map(float, scores), strict=True), key=lambda item: -item[1])
        diagnostics["reranked"] = len(ranked)
        diagnostics["timing_ms"]["rerank"] = round((time.perf_counter() - ranked_at) * 1000, 1)
        relevant = [(i, score) for i, score in ranked if score >= self.settings.min_rerank_score]
        context = []
        for scope in sorted(scopes):
            scoped = [
                (i, score)
                for i, score in relevant
                if (self.chunks[i]["country"], self.chunks[i]["policy_year"]) == scope
            ]
            if not scoped:
                diagnostics["missing_evidence_scope"] = list(scope)
                return [], diagnostics
            context.extend(scoped[: min(2, self.settings.context_chunks // len(scopes))])
        used = {i for i, _ in context}
        context.extend((i, score) for i, score in relevant if i not in used)
        context = context[: self.settings.context_chunks]
        evidence = [
            {
                **self.chunks[i],
                "citation_id": f"S{n}",
                "rerank_score": round(score, 4),
                "retrieval_methods": sorted(methods[i]),
            }
            for n, (i, score) in enumerate(context, 1)
        ]
        diagnostics["context_chunks"] = len(evidence)
        log.info(
            "retrieval scopes=%s eligible=%d context=%d timing_ms=%s",
            sorted(scopes),
            len(eligible),
            len(evidence),
            diagnostics["timing_ms"],
        )
        return evidence, diagnostics

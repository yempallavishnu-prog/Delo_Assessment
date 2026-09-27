import argparse
import json
import time
from pathlib import Path

import pymupdf
from langgraph.checkpoint.memory import InMemorySaver

from app.config import Settings
from app.ingestion import document_path
from app.retrieval import HybridRetriever
from app.workflow import build_workflow, parse_query


def normalize(text: str) -> str:
    return " ".join(text.split()).casefold()


def evaluate(settings: Settings, retrieval_only: bool = False, case_id: str | None = None):
    cases = json.loads(Path(__file__).with_name("evaluation.json").read_text())
    retriever = HybridRetriever(settings)
    graph = build_workflow(retriever, settings, InMemorySaver())
    page_text = {}
    for document in retriever.documents:
        with pymupdf.open(document_path(settings, document)) as pdf:
            page_text.update(
                {
                    (document.document_id, n): normalize(page.get_text())
                    for n, page in enumerate(pdf, 1)
                }
            )
    results = []
    for case in cases:
        if case_id and case["id"] != case_id:
            continue
        started = time.perf_counter()
        checks, state = {}, {}
        if case["mode"] == "workflow" and not retrieval_only:
            for turn in case["turns"]:
                state = graph.invoke({"message": turn}, {"configurable": {"thread_id": case["id"]}})
            checks["status"] = state["status"] == case["status"]
            if "answer_contains" in case:
                checks["answer"] = case["answer_contains"].casefold() in state["answer"].casefold()
            if "numeric_values" in case:
                checks["retained_values"] = state["numeric_values"] == case["numeric_values"]
        else:
            for turn in case["turns"]:
                state = {**state, **parse_query({**state, "message": turn}, set(retriever.groups))}
            if state["status"] == "ready":
                evidence, diagnostics = retriever.search(
                    state["current_query"], set(state["scopes"])
                )
                state.update(retrieved_evidence=evidence, diagnostics=diagnostics)
        expected_scope = {tuple(s) for s in case["scopes"]}
        checks["parsed_scope"] = set(map(tuple, state["scopes"])) == expected_scope
        evidence = state.get("retrieved_evidence", [])
        checks["metadata_filter"] = all(
            (e["country"], e["policy_year"]) in expected_scope for e in evidence
        )
        if "pages" in case:
            hits = {
                doc: any(e["document_id"] == doc and e["page_number"] in pages for e in evidence)
                for doc, pages in case["pages"].items()
            }
            checks["retrieval_hit"] = all(hits.values())
        if "method" in case:
            checks["retrieval_method"] = any(
                case["method"] in e["retrieval_methods"]
                and e["page_number"] in case["pages"].get(e["document_id"], [])
                for e in evidence
            )
        citations = state.get("citations", [])
        if state["status"] == "answered":
            checks["citation_provenance"] = bool(citations) and all(
                any(e["chunk_id"] == c["chunk_id"] for e in evidence)
                and (c["document_id"], c["page_number"]) in page_text
                for c in citations
            )
            checks["source_page_text"] = all(
                normalize(paragraph)
                in page_text[(citation["document_id"], citation["page_number"])]
                for citation in citations
                for paragraph in citation["text"].split("\n\n")
                if paragraph.strip()
            )
        if case.get("status") == "insufficient_evidence" and not retrieval_only:
            checks["unsupported_answer"] = (
                state["status"] == "insufficient_evidence" and not citations
            )
        record = {
            "id": case["id"],
            "passed": all(checks.values()),
            "checks": checks,
            "status": state["status"],
            "seconds": round(time.perf_counter() - started, 2),
            "sources": [[e["document_id"], e["page_number"]] for e in evidence],
            "answer": state.get("answer", ""),
            "diagnostics": state.get("diagnostics", {}),
        }
        results.append(record)
        if not record["passed"]:
            record["rejected_draft"] = state.get("draft")
        failures = ", ".join(k for k, v in checks.items() if not v)
        print(
            f"{'PASS' if record['passed'] else 'FAIL'} {case['id']} "
            f"({record['seconds']:.1f}s) {failures}",
            flush=True,
        )
    if not results:
        raise ValueError("No evaluation cases matched.")
    metrics = {}
    for check in {key for r in results for key in r["checks"]}:
        measured = [r["checks"][check] for r in results if check in r["checks"]]
        metrics[check] = {"passed": sum(measured), "total": len(measured)}
    report = {
        "mode": "retrieval-only" if retrieval_only else "mixed retrieval and end-to-end",
        "model": settings.llm_model,
        "index": retriever.manifest["fingerprint"],
        "passed": sum(r["passed"] for r in results),
        "total": len(results),
        "metrics": metrics,
        "cases": results,
    }
    Path("evaluation-results.json").write_text(json.dumps(report, indent=2))
    print(f"\nCases: {report['passed']}/{report['total']}")
    for key, value in sorted(metrics.items()):
        print(f"{key}: {value['passed']}/{value['total']}")
    print("Details: evaluation-results.json")
    return report["passed"] == report["total"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the source-checked policy retrieval cases.")
    parser.add_argument(
        "--retrieval-only", action="store_true", help="Skip local answer generation."
    )
    parser.add_argument("--case", help="Run one named evaluation case.")
    args = parser.parse_args()
    raise SystemExit(0 if evaluate(Settings(), args.retrieval_only, args.case) else 1)

#!/usr/bin/env python3
"""Simple Day-6 RAG evaluation harness.

Supports two modes:
1) Online mode: call a running rag_service /search and /answer endpoints.
2) Offline mode: score pre-recorded run results from JSON.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class QueryCase:
    query_id: str
    index: str
    question: str
    relevant_chunk_ids: List[str]
    required_citations: List[str]
    expected_keywords: List[str]
    mode: str = "knn"
    k: int = 6


def load_query_cases(path: Path) -> List[QueryCase]:
    data = json.loads(path.read_text())
    cases: List[QueryCase] = []
    for row in data:
        cases.append(
            QueryCase(
                query_id=row["query_id"],
                index=row.get("index", "default"),
                question=row["question"],
                relevant_chunk_ids=row.get("relevant_chunk_ids", []),
                required_citations=row.get("required_citations", []),
                expected_keywords=row.get("expected_keywords", []),
                mode=row.get("mode", "knn"),
                k=int(row.get("k", 6)),
            )
        )
    return cases


def post_json(base_url: str, endpoint: str, payload: Dict[str, Any], api_key: Optional[str]) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}{endpoint}",
        data=body,
        method="POST",
        headers={"content-type": "application/json"},
    )
    if api_key:
        req.add_header("x-api-key", api_key)

    with urllib.request.urlopen(req, timeout=45) as resp:
        return json.loads(resp.read().decode("utf-8"))


def reciprocal_rank(ranked_ids: List[str], relevant_ids: List[str]) -> float:
    rel = set(relevant_ids)
    for i, cid in enumerate(ranked_ids, start=1):
        if cid in rel:
            return 1.0 / i
    return 0.0


def recall_at_k(ranked_ids: List[str], relevant_ids: List[str], k: int) -> float:
    rel = set(relevant_ids)
    if not rel:
        return 1.0
    top = set(ranked_ids[:k])
    return len(top.intersection(rel)) / len(rel)


def ndcg_at_k(ranked_ids: List[str], relevant_ids: List[str], k: int) -> float:
    rel = set(relevant_ids)
    if not rel:
        return 1.0

    dcg = 0.0
    for idx, cid in enumerate(ranked_ids[:k], start=1):
        if cid in rel:
            dcg += 1.0 / math.log2(idx + 1)

    ideal_hits = min(len(rel), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def score_answer(answer_text: str, citations: List[str], required_citations: List[str], expected_keywords: List[str]) -> Dict[str, float]:
    lower_answer = answer_text.lower()

    keyword_hit = 1.0
    if expected_keywords:
        matched = sum(1 for keyword in expected_keywords if keyword.lower() in lower_answer)
        keyword_hit = matched / len(expected_keywords)

    citation_precision = 1.0
    if citations:
        required = set(required_citations)
        if required:
            citation_precision = sum(1 for c in citations if c in required) / len(citations)

    citation_recall = 1.0
    if required_citations:
        given = set(citations)
        req = set(required_citations)
        citation_recall = len(given.intersection(req)) / len(req)

    return {
        "keyword_hit": keyword_hit,
        "citation_precision": citation_precision,
        "citation_recall": citation_recall,
    }


def evaluate_online(cases: List[QueryCase], base_url: str, api_key: Optional[str]) -> Dict[str, Any]:
    per_query: List[Dict[str, Any]] = []

    for case in cases:
        search_payload = {
            "index": case.index,
            "query": case.question,
            "mode": case.mode,
            "k": case.k,
            "max_results": case.k,
            "dedup_by_chunk_id": True,
        }
        answer_payload = {
            "index": case.index,
            "query": case.question,
            "mode": case.mode,
            "k": case.k,
            "llm_provider": "openai",
        }

        search_resp = post_json(base_url, "/search", search_payload, api_key)
        answer_resp = post_json(base_url, "/answer", answer_payload, api_key)

        ranked_ids = [result.get("chunk_id") or f"sid:{result.get('sentence_id')}" for result in search_resp.get("results", [])]
        mrr = reciprocal_rank(ranked_ids, case.relevant_chunk_ids)
        r_at_k = recall_at_k(ranked_ids, case.relevant_chunk_ids, case.k)
        ndcg = ndcg_at_k(ranked_ids, case.relevant_chunk_ids, case.k)

        answer_scores = score_answer(
            answer_resp.get("answer", ""),
            answer_resp.get("citations", []),
            case.required_citations,
            case.expected_keywords,
        )

        per_query.append(
            {
                "query_id": case.query_id,
                "mrr": mrr,
                "recall_at_k": r_at_k,
                "ndcg_at_k": ndcg,
                **answer_scores,
            }
        )

    return aggregate_metrics(per_query)


def aggregate_metrics(per_query: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not per_query:
        return {"queries": 0, "metrics": {}}

    def mean(metric: str) -> float:
        return statistics.mean(float(row[metric]) for row in per_query)

    return {
        "queries": len(per_query),
        "metrics": {
            "mrr": mean("mrr"),
            "recall_at_k": mean("recall_at_k"),
            "ndcg_at_k": mean("ndcg_at_k"),
            "keyword_hit": mean("keyword_hit"),
            "citation_precision": mean("citation_precision"),
            "citation_recall": mean("citation_recall"),
        },
        "per_query": per_query,
    }


def evaluate_offline(results_path: Path) -> Dict[str, Any]:
    data = json.loads(results_path.read_text())
    return aggregate_metrics(data["per_query"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate rag_service retrieval + answer quality.")
    parser.add_argument("--cases", type=Path, help="Path to evaluation cases JSON")
    parser.add_argument("--base-url", default="http://localhost:8080")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--offline-results", type=Path, help="Offline run results JSON")
    parser.add_argument("--out", type=Path, default=Path("rag_service/eval/latest_report.json"))
    args = parser.parse_args()

    if args.offline_results:
        report = evaluate_offline(args.offline_results)
    else:
        if not args.cases:
            print("--cases is required in online mode", file=sys.stderr)
            return 2
        cases = load_query_cases(args.cases)
        try:
            report = evaluate_online(cases, args.base_url, args.api_key)
        except urllib.error.URLError as ex:
            print(f"Failed to reach rag_service: {ex}", file=sys.stderr)
            return 3

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

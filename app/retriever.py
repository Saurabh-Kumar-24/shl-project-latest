from __future__ import annotations

import logging
import re

from rank_bm25 import BM25Okapi

from app.catalog import Assessment, Catalog

logger = logging.getLogger(__name__)

SEMANTIC_TOP_K = 20
FINAL_TOP_K = 10

_TOKEN_RE = re.compile(r"[a-zA-Z0-9#+.]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class HybridRetriever:
    """Hybrid retriever combining BM25 ranked search with exact keyword matching.

    Previously used neural embeddings (sentence-transformers, then fastembed)
    for semantic search, but that required an ONNX/torch runtime whose
    baseline memory footprint didn't fit within Render's free-tier 512MB RAM
    limit. BM25 is a classic term-frequency ranking algorithm - no ML runtime,
    tiny memory footprint - while still giving relevance-ranked results
    instead of plain substring matching.
    """

    def __init__(self, catalog: Catalog) -> None:
        self.catalog = catalog
        self._bm25: BM25Okapi | None = None
        self._build_index()

    def _build_index(self) -> None:
        corpus = [_tokenize(a.rich_text) for a in self.catalog.assessments]
        self._bm25 = BM25Okapi(corpus)
        logger.info("BM25 index built: %d documents", len(corpus))

    def _semantic_search(self, query: str, top_k: int = SEMANTIC_TOP_K) -> list[tuple[Assessment, float]]:
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]
        results: list[tuple[Assessment, float]] = []
        for idx in ranked:
            if scores[idx] <= 0:
                continue
            results.append((self.catalog.assessments[idx], float(scores[idx])))
        return results

    def _keyword_match(self, query: str) -> list[Assessment]:
        tokens = set(re.findall(r"[a-zA-Z0-9#+.]+", query.lower()))
        if not tokens:
            return []
        matches: list[Assessment] = []
        for a in self.catalog.assessments:
            searchable = f"{a.name} {a.description}".lower()
            if any(t in searchable for t in tokens):
                matches.append(a)
        return matches

    def _apply_filters(
        self,
        candidates: list[Assessment],
        job_level: str | None = None,
        test_type: str | None = None,
        max_duration: int | None = None,
        remote_only: bool = False,
    ) -> list[Assessment]:
        filtered: list[Assessment] = []
        for a in candidates:
            if job_level and not any(job_level.lower() in jl.lower() for jl in a.job_levels):
                continue
            if test_type:
                wanted = {c.strip().upper() for c in test_type.split(",")}
                has = {c.strip().upper() for c in a.type_codes.split(",") if c.strip()}
                if not wanted & has:
                    continue
            if max_duration is not None and a.duration:
                mins = _parse_duration(a.duration)
                if mins is not None and mins > max_duration:
                    continue
            if remote_only and a.remote.lower() != "yes":
                continue
            filtered.append(a)
        return filtered

    def search(
        self,
        query: str,
        job_level: str | None = None,
        test_type: str | None = None,
        max_duration: int | None = None,
        remote_only: bool = False,
        top_k: int = FINAL_TOP_K,
    ) -> list[Assessment]:
        semantic_results = self._semantic_search(query, SEMANTIC_TOP_K)
        keyword_matches = self._keyword_match(query)

        seen: set[str] = set()
        merged: list[Assessment] = []

        for a, _ in semantic_results:
            if a.entity_id not in seen:
                seen.add(a.entity_id)
                merged.append(a)

        for a in keyword_matches:
            if a.entity_id not in seen:
                seen.add(a.entity_id)
                merged.append(a)

        has_filters = any([job_level, test_type, max_duration is not None, remote_only])
        if has_filters:
            merged = self._apply_filters(merged, job_level, test_type, max_duration, remote_only)

        return merged[:top_k]


def _parse_duration(raw: str) -> int | None:
    m = re.search(r"(\d+)", raw)
    return int(m.group(1)) if m else None
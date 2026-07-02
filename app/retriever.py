from __future__ import annotations

import logging
import re

import faiss
import numpy as np
from fastembed import TextEmbedding

from app.catalog import Assessment, Catalog

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
SEMANTIC_TOP_K = 20
FINAL_TOP_K = 10
BUILD_BATCH_SIZE = 32


class HybridRetriever:
    def __init__(self, catalog: Catalog) -> None:
        self.catalog = catalog
        # threads=1 keeps onnxruntime's thread pool (and its per-thread memory) minimal,
        # which matters on memory-constrained hosts like Render's free tier.
        self._model = TextEmbedding(model_name=EMBEDDING_MODEL, threads=1)
        self._index: faiss.IndexFlatIP | None = None
        self._build_index()

    def _embed(self, texts: list[str]) -> np.ndarray:
        embeddings = np.array(list(self._model.embed(texts)), dtype=np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        return embeddings / norms

    def _build_index(self) -> None:
        texts = [a.rich_text for a in self.catalog.assessments]

        # Encode in small batches instead of all at once, to keep the peak
        # memory during startup lower.
        all_embeddings: list[np.ndarray] = []
        for i in range(0, len(texts), BUILD_BATCH_SIZE):
            batch = texts[i : i + BUILD_BATCH_SIZE]
            all_embeddings.append(self._embed(batch))
        embeddings = np.vstack(all_embeddings)

        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings)
        logger.info("FAISS index built: %d vectors, dim=%d", len(texts), dim)

    def _semantic_search(self, query: str, top_k: int = SEMANTIC_TOP_K) -> list[tuple[Assessment, float]]:
        vec = self._embed([query])
        scores, indices = self._index.search(vec, top_k)
        results: list[tuple[Assessment, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append((self.catalog.assessments[idx], float(score)))
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
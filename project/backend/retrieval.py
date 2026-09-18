"""
retrieval.py
============
Hybrid Vector + BM25 Retrieval Engine for Darukaa.Earth:
- Dense Vector Retrieval (BGE-small / BGE-M3 via persistent Qdrant)
- Sparse BM25 Keyword Search (BM25Okapi for exact terminology, species, and metrics)
- Reciprocal Rank Fusion (RRF) to merge dense and sparse candidate pools
- Multi-query parallel execution across hypothesis queries
- Deduplication by evidence_id
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from models import EvidenceChunk

BACKEND_DIR = Path(__file__).resolve().parent
QDRANT_STORAGE_DIR = BACKEND_DIR / "qdrant_db"
COLLECTION_NAME = "biodiversity_knowledge"
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-base-en-v1.5")

_client: Optional[QdrantClient] = None
_model: Optional[SentenceTransformer] = None
_bm25_index: Optional[BM25Okapi] = None
_corpus_chunks: List[Dict[str, Any]] = []


def get_qdrant_client() -> QdrantClient:
    global _client
    if _client is None:
        qdrant_url = os.getenv("QDRANT_URL")
        if qdrant_url:
            _client = QdrantClient(url=qdrant_url)
        else:
            _client = QdrantClient(path=str(QDRANT_STORAGE_DIR))
    return _client


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            _model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
        except Exception:
            _model = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)
    return _model


def ensure_bm25_index():
    """
    Initialize BM25 index over the entire Qdrant corpus in memory for sub-millisecond sparse searches.
    """
    global _bm25_index, _corpus_chunks
    if _bm25_index is not None and len(_corpus_chunks) > 0:
        return

    client = get_qdrant_client()
    print("[INIT] Loading corpus from Qdrant for BM25 indexing...")
    
    points, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=10000,
        with_payload=True,
        with_vectors=False
    )

    _corpus_chunks = []
    tokenized_corpus = []
    for p in points:
        payload = p.payload or {}
        text = payload.get("text", "")
        _corpus_chunks.append({
            "evidence_id": payload.get("evidence_id", "E000"),
            "doc_title": payload.get("doc_title", "FAO Report"),
            "section": payload.get("section"),
            "text": text
        })
        tokens = re.findall(r"\w+", text.lower())
        tokenized_corpus.append(tokens)

    if tokenized_corpus:
        _bm25_index = BM25Okapi(tokenized_corpus)
        print(f"[SUCCESS] BM25 index built across {len(_corpus_chunks)} document passages.")


def search_bm25(query: str, top_k: int = 10) -> List[Tuple[Dict[str, Any], float]]:
    """
    Execute sparse BM25 search for exact keywords, species, and metrics.
    """
    ensure_bm25_index()
    if _bm25_index is None or not _corpus_chunks:
        return []

    tokens = re.findall(r"\w+", query.lower())
    if not tokens:
        return []

    scores = _bm25_index.get_scores(tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    
    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append((_corpus_chunks[idx], float(scores[idx])))
    return results


def search_hybrid(query: str, top_k: int = 5, rrf_k: int = 60) -> List[EvidenceChunk]:
    """
    Hybrid Search combining Dense Vector Search + BM25 using Reciprocal Rank Fusion (RRF).
    RRF Score = 1 / (rrf_k + dense_rank) + 1 / (rrf_k + bm25_rank)
    """
    client = get_qdrant_client()
    model = get_embedding_model()

    # 1. Dense Vector Search (Top 15)
    query_vector = model.encode(query, normalize_embeddings=True).tolist()
    dense_points = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=max(15, top_k * 2)
    ).points

    # 2. Sparse BM25 Search (Top 15)
    bm25_results = search_bm25(query, top_k=max(15, top_k * 2))

    # 3. Reciprocal Rank Fusion (RRF)
    scores: Dict[str, float] = {}
    doc_payloads: Dict[str, Dict[str, Any]] = {}
    raw_dense_scores: Dict[str, float] = {}

    for rank, p in enumerate(dense_points, 1):
        payload = p.payload or {}
        ev_id = payload.get("evidence_id", "E000")
        scores[ev_id] = scores.get(ev_id, 0.0) + (1.0 / (rrf_k + rank))
        raw_dense_scores[ev_id] = round(float(p.score), 4)
        if ev_id not in doc_payloads:
            doc_payloads[ev_id] = payload

    for rank, (chunk_data, bm25_score) in enumerate(bm25_results, 1):
        ev_id = chunk_data.get("evidence_id", "E000")
        scores[ev_id] = scores.get(ev_id, 0.0) + (1.0 / (rrf_k + rank))
        if ev_id not in doc_payloads:
            doc_payloads[ev_id] = chunk_data

    # Sort candidates by combined RRF score
    sorted_ev_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:top_k]

    fused_evidence: List[EvidenceChunk] = []
    for ev_id in sorted_ev_ids:
        payload = doc_payloads[ev_id]
        # Display cosine similarity score if available, otherwise RRF scaled score
        display_score = raw_dense_scores.get(ev_id, round(scores[ev_id] * 50, 4))
        fused_evidence.append(
            EvidenceChunk(
                evidence_id=ev_id,
                doc_title=payload.get("doc_title", "FAO Report"),
                section=payload.get("section"),
                score=display_score,
                text=payload.get("text", ""),
                matched_query=query
            )
        )

    return fused_evidence


def search_single_query(query: str, top_k: int = 5) -> List[EvidenceChunk]:
    """
    Single-query hybrid search for /api/retrieve.
    """
    return search_hybrid(query, top_k=top_k)


def multi_query_search(queries: List[str], top_k_per_query: int = 5) -> List[EvidenceChunk]:
    """
    Execute Hybrid Vector + BM25 search across all hypotheses and deduplicate results.
    """
    if not queries:
        return []

    seen_evidence: Dict[str, EvidenceChunk] = {}

    for query_text in queries:
        hybrid_chunks = search_hybrid(query_text, top_k=top_k_per_query)
        for chunk in hybrid_chunks:
            ev_id = chunk.evidence_id
            if ev_id not in seen_evidence or chunk.score > seen_evidence[ev_id].score:
                seen_evidence[ev_id] = chunk

    # Return sorted by score descending
    deduped = list(seen_evidence.values())
    deduped.sort(key=lambda x: x.score, reverse=True)
    return deduped

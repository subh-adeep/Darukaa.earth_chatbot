"""
ingest.py
=========
Knowledge base ingestion pipeline for Darukaa.Earth.
- Ingests parsed Markdown outputs (text and tables) produced by MinerU.
- Strictly ignores images (filters out markdown image syntax).
- Does NOT use any external LLM API calls.
- Generates embeddings locally using BGE-M3 (or fallback) with CUDA acceleration.
- Stores vector embeddings and metadata in persistent local Qdrant database.
"""

import os
import re
import sys
import uuid
import shutil
from pathlib import Path
from typing import List, Dict, Any

# Resolve OpenMP collision on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer


# Base paths
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent
DATA_DIR = BACKEND_DIR / "data" / "parsed_documents"
QDRANT_STORAGE_DIR = BACKEND_DIR / "qdrant_db"

COLLECTION_NAME = "biodiversity_knowledge"
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-base-en-v1.5")


def clean_markdown(content: str) -> str:
    """
    Remove markdown image tags completely while preserving tables and formatted text.
    """
    # Remove markdown images: ![alt](url)
    cleaned = re.sub(r"!\[.*?\]\(.*?\)", "", content)
    # Remove empty image references
    cleaned = re.sub(r"!\[.*?\]", "", cleaned)
    # Clean up excessive newlines
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def chunk_document(doc_title: str, file_path: Path, max_chars: int = 2200) -> List[Dict[str, Any]]:
    """
    Split markdown document into coherent chunks preserving:
    1. Markdown section hierarchy (#, ##, ###) - NEVER cut between sections!
    2. Paragraph boundaries (\n\n) - keep paragraphs intact as logical units.
    3. Markdown tables (|...|) - keep tables unified.
    4. Upper bound enforcement: soft ceiling at max_chars (~450 words).
    """
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        raw_text = f.read()

    cleaned_text = clean_markdown(raw_text)

    # Split by markdown headers (# Header, ## Header, ### Header)
    sections = re.split(r"(?=(?:\n|^)#{1,4}\s+)", cleaned_text)

    chunks = []
    current_heading = "General Overview"
    global_chunk_idx = 1

    for section in sections:
        section = section.strip()
        if not section:
            continue

        lines = section.splitlines()
        first_line = lines[0].strip()
        if first_line.startswith("#"):
            current_heading = first_line.lstrip("#").strip()
            body_text = "\n".join(lines[1:]).strip()
        else:
            body_text = section

        if not body_text:
            continue

        # Split section into paragraphs / blocks
        paragraphs = re.split(r"\n\s*\n", body_text)

        current_block: List[str] = []
        current_len = 0

        for p in paragraphs:
            p_strip = p.strip()
            if not p_strip:
                continue

            # If a single paragraph is oversized, split it cleanly at sentence boundaries
            if len(p_strip) > max_chars:
                if current_block:
                    block_text = "\n\n".join(current_block)
                    is_tbl = bool(re.search(r"\|.*?\|", block_text))
                    chunks.append({
                        "chunk_id": f"E{global_chunk_idx:04d}",
                        "doc_title": doc_title,
                        "section": current_heading,
                        "text": f"[{doc_title} > {current_heading}]\n{block_text}",
                        "raw_text": block_text,
                        "is_table": is_tbl
                    })
                    global_chunk_idx += 1
                    current_block = []
                    current_len = 0

                # Split at sentence boundaries
                sentences = re.split(r"(?<=[.!?])\s+", p_strip)
                sent_accum: List[str] = []
                sent_len = 0
                for s in sentences:
                    if sent_len + len(s) + 1 > max_chars and sent_accum:
                        s_text = " ".join(sent_accum)
                        chunks.append({
                            "chunk_id": f"E{global_chunk_idx:04d}",
                            "doc_title": doc_title,
                            "section": current_heading,
                            "text": f"[{doc_title} > {current_heading}]\n{s_text}",
                            "raw_text": s_text,
                            "is_table": False
                        })
                        global_chunk_idx += 1
                        sent_accum = [s]
                        sent_len = len(s)
                    else:
                        sent_accum.append(s)
                        sent_len += len(s) + 1
                if sent_accum:
                    s_text = " ".join(sent_accum)
                    chunks.append({
                        "chunk_id": f"E{global_chunk_idx:04d}",
                        "doc_title": doc_title,
                        "section": current_heading,
                        "text": f"[{doc_title} > {current_heading}]\n{s_text}",
                        "raw_text": s_text,
                        "is_table": False
                    })
                    global_chunk_idx += 1
                continue

            # Check if adding this paragraph exceeds upper limit
            if current_len + len(p_strip) + 2 > max_chars and current_block:
                block_text = "\n\n".join(current_block)
                is_tbl = bool(re.search(r"\|.*?\|", block_text))
                chunks.append({
                    "chunk_id": f"E{global_chunk_idx:04d}",
                    "doc_title": doc_title,
                    "section": current_heading,
                    "text": f"[{doc_title} > {current_heading}]\n{block_text}",
                    "raw_text": block_text,
                    "is_table": is_tbl
                })
                global_chunk_idx += 1
                current_block = [p_strip]
                current_len = len(p_strip)
            else:
                current_block.append(p_strip)
                current_len += len(p_strip) + 2

        # Flush any remaining paragraphs in this section (NEVER bleed into next section)
        if current_block:
            block_text = "\n\n".join(current_block)
            if len(block_text) >= 40:
                is_tbl = bool(re.search(r"\|.*?\|", block_text))
                chunks.append({
                    "chunk_id": f"E{global_chunk_idx:04d}",
                    "doc_title": doc_title,
                    "section": current_heading,
                    "text": f"[{doc_title} > {current_heading}]\n{block_text}",
                    "raw_text": block_text,
                    "is_table": is_tbl
                })
                global_chunk_idx += 1

    return chunks



def load_all_documents() -> List[Dict[str, Any]]:
    """
    Traverse data/parsed_documents for all markdown files.
    """
    all_chunks = []
    if not DATA_DIR.exists():
        print(f"[ERROR] Data directory does not exist: {DATA_DIR}")
        return all_chunks

    print(f"Scanning for parsed documents in: {DATA_DIR}")
    for item in DATA_DIR.iterdir():
        if item.is_dir():
            md_files = list(item.glob("*.md"))
            for md_file in md_files:
                if md_file.name.lower() == "readme.md":
                    continue
                doc_title = item.name.replace("_", " ")
                print(f"  -> Found parsed document: {doc_title} ({md_file.name})")
                doc_chunks = chunk_document(doc_title, md_file)
                print(f"     Extracted {len(doc_chunks)} chunks (text & tables, images ignored).")
                all_chunks.extend(doc_chunks)

    # Re-index chunk IDs to be sequential across all docs
    for idx, c in enumerate(all_chunks, 1):
        c["chunk_id"] = f"E{idx:04d}"

    print(f"Total knowledge chunks collected: {len(all_chunks)}")
    return all_chunks


def run_ingestion(recreate_collection: bool = True):
    """
    Main ingestion execution using local BGE-M3 and persistent Qdrant.
    No LLM API calls are made here.
    """
    chunks = load_all_documents()
    if not chunks:
        print("[WARN] No chunks found to ingest.")
        return

    print("\n--- Initializing Local Embedding Model ---")
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device.upper()} (GPU Acceleration: {torch.cuda.is_available()})")
    print(f"Loading local embedding model: {EMBEDDING_MODEL_NAME}...")

    # Load local sentence transformer model
    try:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
    except Exception as e:
        print(f"[WARN] Failed to load {EMBEDDING_MODEL_NAME} ({e}), falling back to BAAI/bge-small-en-v1.5")
        model = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)

    sample_embedding = model.encode("Ecosystem biodiversity and soil carbon test", convert_to_tensor=False)
    embedding_dim = len(sample_embedding)
    print(f"Model loaded successfully. Vector dimension: {embedding_dim}")

    print("\n--- Initializing Persistent Qdrant Client ---")
    qdrant_url = os.getenv("QDRANT_URL")
    if not qdrant_url and recreate_collection and QDRANT_STORAGE_DIR.exists():
        print(f"Purging previous storage at {QDRANT_STORAGE_DIR} for clean 768-dim index...")
        shutil.rmtree(QDRANT_STORAGE_DIR, ignore_errors=True)
    
    QDRANT_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    
    if qdrant_url:
        print(f"Connecting to Qdrant at {qdrant_url}")
        client = QdrantClient(url=qdrant_url)
    else:
        print(f"Using persistent disk storage at: {QDRANT_STORAGE_DIR}")
        client = QdrantClient(path=str(QDRANT_STORAGE_DIR))


    # Recreate or create collection
    existing_collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing_collections:
        if recreate_collection:
            print(f"Recreating existing collection '{COLLECTION_NAME}'...")
            client.delete_collection(COLLECTION_NAME)
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=qmodels.VectorParams(
                    size=embedding_dim,
                    distance=qmodels.Distance.COSINE
                )
            )
        else:
            print(f"Using existing collection '{COLLECTION_NAME}'.")
    else:
        print(f"Creating new collection '{COLLECTION_NAME}'...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=qmodels.VectorParams(
                size=embedding_dim,
                distance=qmodels.Distance.COSINE
            )
        )

    print("\n--- Generating Local Embeddings & Upserting to Qdrant ---")
    batch_size = 128
    texts_to_embed = [c["text"] for c in chunks]
    
    points = []
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i : i + batch_size]
        batch_texts = texts_to_embed[i : i + batch_size]
        
        # Local model inference (NO API CALL)
        batch_vectors = model.encode(batch_texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)
        
        for chunk, vector in zip(batch_chunks, batch_vectors):
            point = qmodels.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector.tolist(),
                payload={
                    "evidence_id": chunk["chunk_id"],
                    "doc_title": chunk["doc_title"],
                    "section": chunk["section"],
                    "text": chunk["raw_text"],
                    "formatted_text": chunk["text"],
                    "is_table": chunk["is_table"]
                }
            )
            points.append(point)

        # Upsert in chunks to maximize disk I/O throughput
        if len(points) >= 512 or (i + batch_size >= len(chunks)):
            client.upsert(collection_name=COLLECTION_NAME, points=points)
            print(f"  Upserted {min(i + batch_size, len(chunks))}/{len(chunks)} chunks...", flush=True)
            points = []


    info = client.get_collection(COLLECTION_NAME)
    print(f"\n[SUCCESS] Ingestion completed!")
    print(f"Total points indexed in '{COLLECTION_NAME}': {info.points_count}")


if __name__ == "__main__":
    run_ingestion(recreate_collection=True)

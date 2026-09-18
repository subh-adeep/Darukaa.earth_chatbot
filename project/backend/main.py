"""
main.py
=======
FastAPI server exposing endpoints for Darukaa.Earth AI Biodiversity Intelligence:
- GET  /api/health
- POST /api/retrieve
- POST /api/recommend
"""

import os
import sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from models import (
    UserInputRequest,
    EnvironmentalState,
    PlannerOutput,
    RecommendationOutput,
    EvidenceChunk,
    InputValidationResult,
    PipelineTiming
)
from retrieval import search_single_query, multi_query_search
from agent import (
    normalize_to_environmental_state,
    plan_research,
    reason_and_recommend,
    validate_claims_against_evidence,
    validate_natural_language_input,
    rewrite_query_with_context
)

app = FastAPI(
    title="Darukaa.Earth Biodiversity Intelligence API",
    description="Agentic Multi-Query RAG for Environmental Reasoning",
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RetrieveRequest(BaseModel):
    query: str = Field(..., description="Search query to retrieve scientific evidence chunks")
    top_k: int = Field(5, description="Number of chunks to return")


class RetrieveResponse(BaseModel):
    query: str
    total_found: int
    chunks: List[EvidenceChunk]


class RecommendationResponse(BaseModel):
    needs_clarification: bool = Field(False, description="Whether clarifying questions are needed before planning")
    clarification: Optional[InputValidationResult] = Field(None, description="Clarifying question and missing variables if input was incomplete")
    state: Optional[EnvironmentalState] = Field(None, description="Parsed environmental state")
    planner: Optional[PlannerOutput] = Field(None, description="Hypotheses and targeted queries")
    recommendation: Optional[RecommendationOutput] = Field(None, description="Final evidence-grounded recommendation")
    timing: Optional[PipelineTiming] = Field(None, description="Detailed latency metrics: retrieval, thinking, and total generation time")


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Darukaa.Earth Biodiversity Intelligence",
        "architecture": "Agentic Multi-Query RAG",
        "qdrant_storage": "persistent"
    }


@app.post("/api/retrieve", response_model=RetrieveResponse)
def retrieve_endpoint(req: RetrieveRequest):
    """
    Direct retrieval endpoint to inspect and verify evidence chunks from the knowledge base.
    """
    try:
        chunks = search_single_query(req.query, top_k=req.top_k)
        return RetrieveResponse(
            query=req.query,
            total_found=len(chunks),
            chunks=chunks
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/recommend", response_model=RecommendationResponse)
def recommend_endpoint(req: UserInputRequest):
    """
    Full Agentic RAG Pipeline:
    0. Input Validation + Context-Aware Query Rewriting
       - If conversation_history is provided, LLM rewrites the short new reply
         by merging it with prior turns into a complete environmental description.
       - If still incomplete, returns needs_clarification=True with a targeted question.
    1. Input normalization (Text or JSON -> EnvironmentalState)
    2. Research Planner (LLM #1 -> Hypotheses & queries)
    3. Multi-Query Vector Search & Deduplication (Qdrant + BGE-base + BM25 RRF)
    4. Environmental Reasoner (LLM #2 -> Multi-metric synthesis & recommendation)
    """
    t_start = time.time()
    retrieval_duration = 0.0
    planning_duration = 0.0
    reasoning_duration = 0.0
    rewriting_duration = 0.0

    try:
        history = req.conversation_history or []

        # Check if natural language input needs clarification (with context rewriting)
        if req.text and not req.structured:
            t0 = time.time()
            validation = validate_natural_language_input(req.text, conversation_history=history)
            rewriting_duration = time.time() - t0

            if not validation.is_sufficient:
                total_duration = time.time() - t_start
                return RecommendationResponse(
                    needs_clarification=True,
                    clarification=validation,
                    state=None,
                    planner=None,
                    recommendation=None,
                    timing=PipelineTiming(
                        retrieval_ms=0,
                        retrieval_time_s=0.0,
                        thinking_ms=int(rewriting_duration * 1000),
                        thinking_time_s=round(rewriting_duration, 2),
                        total_ms=int(total_duration * 1000),
                        total_time_s=round(total_duration, 2),
                        rewriting_ms=int(rewriting_duration * 1000)
                    )
                )
            # Use the rewritten (context-merged) query if available, otherwise raw text
            effective_text = validation.rewritten_query or req.text
        else:
            effective_text = req.text

        # Step 1: Normalize input
        input_data = req.structured if req.structured else (effective_text or "")
        if not input_data:
            raise HTTPException(status_code=400, detail="Either 'text' or 'structured' input must be provided.")

        state = normalize_to_environmental_state(input_data)

        # Step 2: LLM #1 Research Planner
        t_plan = time.time()
        planner = plan_research(state)
        planning_duration = time.time() - t_plan

        # Step 3: Multi-query parallel retrieval
        t_ret = time.time()
        queries = [h.query for h in planner.hypotheses]
        evidence_chunks = multi_query_search(queries, top_k_per_query=5)
        retrieval_duration = time.time() - t_ret

        # Step 4: LLM #2 Environmental Reasoner
        t_reas = time.time()
        recommendation = reason_and_recommend(state, planner, evidence_chunks)
        reasoning_duration = time.time() - t_reas

        # Step 5: LLM #3 Claim-Evidence Integrity Validator
        # Audits every quantitative claim in LLM #2's output against the actual
        # retrieved evidence text. Strips unsupported numbers, flags context mismatches.
        t_val = time.time()
        recommendation = validate_claims_against_evidence(recommendation, evidence_chunks)
        validation_duration = time.time() - t_val

        total_duration = time.time() - t_start
        thinking_duration = planning_duration + reasoning_duration + validation_duration

        timing = PipelineTiming(
            retrieval_ms=int(retrieval_duration * 1000),
            retrieval_time_s=round(retrieval_duration, 2),
            thinking_ms=int(thinking_duration * 1000),
            thinking_time_s=round(thinking_duration, 2),
            total_ms=int(total_duration * 1000),
            total_time_s=round(total_duration, 2),
            rewriting_ms=int(rewriting_duration * 1000)
        )

        return RecommendationResponse(
            needs_clarification=False,
            clarification=None,
            state=state,
            planner=planner,
            recommendation=recommendation,
            timing=timing
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

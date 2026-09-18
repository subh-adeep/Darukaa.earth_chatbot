# Darukaa.Earth | AI Biodiversity Intelligence System

> **"Build a system that behaves like an AI environmental scientist, not a chatbot."**
> — *Darukaa.Earth Hackathon Challenge Goal*

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Darukaa.Earth%20UI-10B981?style=for-the-badge&logo=huggingface&logoColor=white)](https://subhadeepsing-drakula-ui.static.hf.space/index.html)
[![Backend API](https://img.shields.io/badge/Backend%20API-Gradio%20Streaming-06B6D4?style=for-the-badge&logo=fastapi&logoColor=white)](https://subhadeepsing-drakula.hf.space)
[![Knowledge Base](https://img.shields.io/badge/Knowledge%20Base-7%2C978%20FAO%20%2F%20IPCC%20%2F%20IPBES%20Passages-8B5CF6?style=for-the-badge)](https://subhadeepsing-drakula-ui.static.hf.space/index.html)

---

## 1. Live Deployment & Testing

| Resource | URL |
| :--- | :--- |
| **Primary Web UI (Test Here)** | **https://subhadeepsing-drakula-ui.static.hf.space/index.html** |
| **Backend Streaming API** | **https://subhadeepsing-drakula.hf.space** |
| **Frontend HF Space** | https://huggingface.co/spaces/subhadeepsing/drakula-ui |
| **Backend HF Space** | https://huggingface.co/spaces/subhadeepsing/drakula |

---

## 2. Problem & Why Standard LLMs Fail

Standard LLMs fail catastrophically at ecological restoration planning:

1. **Single-Variable Blindness** — Recommending cover crops for nitrogen without checking moisture depletion in arid zones.
2. **Hallucinated Data** — Inventing exact percentages (e.g. "increases earthworm count by 32.4%") without peer-reviewed grounding.
3. **Premature Generation** — Generating generic advice from vague inputs like "biodiversity is low on my land" instead of gathering site parameters.
4. **Geographic Assumption** — Applying European cover crop regimes to Deccan red soils.

**Darukaa.Earth** solves these through an **evidence-grounded, 5-stage agentic RAG architecture** powered by 7,978 scientific passages from FAO, IPCC, and IPBES.

---

## 3. End-to-End Architecture

```
                       [User Query — Natural Language or Structured Form]
                                            |
                                            v
  +-----------------------------------------------------------------------------------+
  | STAGE 0: Intelligent Gate & Conversational Router                                 |
  |          (Groq qwen/qwen3.8-27b  ->  Gemini Flash fallback)                      |
  |                                                                                   |
  |  Evaluates conversation history across last 6 turns. Routes to:                  |
  |  GREETING | OFF_TOPIC | FOLLOW_UP | ON_TOPIC                                      |
  |                                                                                   |
  |  ON_TOPIC path -> JSON Schema Verification (EnvironmentalState):                  |
  |    is_sufficient = False -> Ask 1 clarifying question + clickable chips           |
  |    is_sufficient = True  -> Build consolidated EnvironmentalState -> Proceed      |
  +-----------------------------------------------------------------------------------+
                                            |
                                            v
  +-----------------------------------------------------------------------------------+
  | STAGE 1: Environmental Research Planning Intelligence                              |
  |          (Gemini Flash  ->  Groq fallback)                                        |
  | Generates 3-4 scientific hypotheses + precise semantic search queries             |
  +-----------------------------------------------------------------------------------+
                                            |
                                            v
  +-----------------------------------------------------------------------------------+
  | STAGE 2: Hybrid Dense Vector + Sparse BM25 Retrieval Engine                      |
  |                                                                                   |
  |  LOCAL:      Qdrant embedded disk + BM25Okapi -> Reciprocal Rank Fusion (RRF)    |
  |  PRODUCTION: NumPy matrix dot product (7978x768 float32) + BM25 dedup           |
  |                                                                                   |
  |  Output: 10 deduplicated passages from FAO / IPCC / IPBES with citation IDs      |
  +-----------------------------------------------------------------------------------+
                                            |
                                            v
  +-----------------------------------------------------------------------------------+
  | STAGE 3: Multi-Metric Ecological Reasoner (Gemini Flash Streaming -> Groq)        |
  |                                                                                   |
  |  Streams tokens live. Cross-variable reasoning enforced:                          |
  |  SOC <-> Soil Moisture <-> Cropping System <-> Biodiversity / Pollinators         |
  |                                                                                   |
  |  Output: Primary Strategy | Ecosystem Assessment | Interventions (with mechanism) |
  |          Time Horizons | Trade-offs & Mitigations | Explore Further               |
  +-----------------------------------------------------------------------------------+
                                            |
                                            v
  +-----------------------------------------------------------------------------------+
  | STAGE 4: Claim-Evidence Integrity Auditor (Groq qwen/qwen3.8-27b -- 0.3s)        |
  |                                                                                   |
  |  Cross-checks every generated claim against retrieved passage text                |
  |  Flags unsupported numbers (SUPPORTED | WEAKENED)                                 |
  |  Extracts "Explore Further" follow-up question cards for the UI                  |
  +-----------------------------------------------------------------------------------+
                                            |
                                            v
  +-----------------------------------------------------------------------------------+
  | STAGE 5: React + Vite Interactive Frontend                                        |
  |                                                                                   |
  |  * Live 5-stage animated progress tracker                                         |
  |  * Streaming text with in-line citation badges [E0345]                            |
  |  * Expandable Evidence Drawer (document title, section, relevance score)          |
  |  * Interactive Suggested Inquiries Hub -- click to send follow-up instantly       |
  |  * Scenario Planner tab for structured parameter form entry                      |
  +-----------------------------------------------------------------------------------+
```

---

## 4. Context Transfer & Multi-Turn State

### A. Why This Matters
When a user says *"biodiversity is declining on my land"*, a standard LLM halluccinates generic advice. Darukaa.Earth instead:
1. Parses the message for site variables (location, land use, soil, water)
2. Checks if the `EnvironmentalState` schema is complete across the **last 6 conversation turns**
3. If incomplete — halts the pipeline and asks **one precise clarifying question** with clickable chips
4. If complete — fires the full 5-stage pipeline

### B. Multi-Turn Accumulation Trace

| Turn | User Says | System Extracts | Pipeline Action |
| :--- | :--- | :--- | :--- |
| 1 | "Biodiversity declining, I'm in Bangalore" | `{location: "Bangalore", is_sufficient: false}` | Pause — asks for land use & soil |
| 2 | "Rainfed finger millet on red sandy loam" | `{location: "Bangalore", land_use: "Ragi", water: "Rainfed", soil: "Sandy loam", is_sufficient: true}` | Launch full 5-stage pipeline |

### C. Environmental State Synthesis
Once complete, all variables are merged into one profile injected into Stage 1 and Stage 3:
```
Location / Region: Deccan Plateau, Karnataka
Land Use / Cropping System: Rainfed Finger Millet (Ragi)
Soil Profile: Red sandy loam, pH 6.2, SOC 0.4%
Hydrology & Climate: 700 mm rainfall, semi-arid dry spells
Additional Site Observations: Severe soil crusting, declining pollinators
```

### D. Inter-Stage Artifact Hand-off
1. **Stage 1 -> Stage 2:** JSON hypotheses -> retrieval queries
2. **Stage 2 -> Stage 3:** Citation blocks `[E0345] (FAO Soil Biodiversity): "..."` injected into reasoning prompt
3. **Stage 3 -> Stage 4:** Draft text + retrieved evidence -> claim audit
4. **Stage 3 -> Stage 5:** "Explore Further" questions -> interactive card hub

---

## 5. Database & Vector Search Architecture

### Where Does the Data Live?

**All data lives inside the Hugging Face Space repository** — no external database required.

| File | Location | Purpose |
| :--- | :--- | :--- |
| `evidence_vectors.npz` | `hf_space/` | 7,978 x 768 float32 normalized passage vectors (~23 MB) |
| `evidence_corpus.json.gz` | `hf_space/` | Gzipped JSON of all 7,978 passage objects (~8 MB) |
| `qdrant_db/` | `project/backend/` | Local development embedded Qdrant collection (~650 MB) |

### Dual Architecture

| Dimension | Local Development | Hugging Face Production |
| :--- | :--- | :--- |
| **Vector Engine** | Embedded Qdrant (on-disk) | NumPy matrix dot product |
| **Embedding Model** | BAAI/bge-base-en-v1.5 (768-dim) | Same model, precomputed at index time |
| **Sparse Search** | BM25Okapi in-memory | BM25Okapi in-memory |
| **Fusion** | Reciprocal Rank Fusion (RRF) | Cosine + BM25 deduplication |
| **Query Latency** | ~25 ms | ~6-8 ms |
| **RAM Footprint** | ~650 MB | ~85 MB |

### Why No Qdrant on Hugging Face?
Free CPU containers cause OOM restarts with daemonized Qdrant. The precomputed NumPy matrix gives identical ranking accuracy:
```python
sims = np.dot(VECTORS, q_vec)   # VECTORS: 7978x768, q_vec: 768
top_k = np.argsort(sims)[::-1][:25]
```

---

## 6. Deployment — How Frontend & Backend Are Hosted

Both are deployed on **Hugging Face Spaces** via **git-based deployment**, as two separate Space repositories.

---

### Backend (Python + Gradio)

**Platform:** Hugging Face Spaces — SDK: `gradio`
**Live URL:** https://subhadeepsing-drakula.hf.space
**Space Repo:** https://huggingface.co/spaces/subhadeepsing/drakula
**Local folder:** `hf_space/` (its own git repo linked to HF)

**Files deployed:**
```
hf_space/
|-- app.py                    # All 5 pipeline stages
|-- requirements.txt
|-- evidence_vectors.npz      # 7,978 x 768 vectors (~23 MB)
`-- evidence_corpus.json.gz   # Gzipped corpus (~8 MB)
```

**Deploy workflow:**
```bash
cd hf_space
git add .
git commit -m "update"
git push origin main
# HF rebuilds automatically in ~2-3 minutes
```

**Secrets** (set via HF Space Settings -> Variables and Secrets — never in git):

| Secret | Purpose |
| :--- | :--- |
| `GEMINI_API_KEYS` | Comma-separated Gemini key rotation pool |
| `GROQ_API_KEY` | Groq key for Stage 0 gate and Stage 4 audit |

---

### Frontend (React + Vite -> Static)

**Platform:** Hugging Face Spaces — SDK: `static`
**Live URL:** https://subhadeepsing-drakula-ui.static.hf.space/index.html
**Space Repo:** https://huggingface.co/spaces/subhadeepsing/drakula-ui
**Local folder:** `hf_space_ui/` (its own git repo linked to HF)

**Deploy workflow:**
```bash
# Step 1: Build React app
cd project/frontend
npm run build
# Outputs to project/frontend/dist/

# Step 2: Copy to HF Space folder
# Step 3: Push
cd ../../hf_space_ui
git add .
git commit -m "Update frontend build"
git push origin main
# HF serves static files immediately
```

### Deployment Flow
```
  project/frontend/src/   <- Edit React source here
  project/backend/        <- Edit local Python backend here
  |
  hf_space/               <- git push -> huggingface.co/spaces/subhadeepsing/drakula
  hf_space_ui/            <- git push -> huggingface.co/spaces/subhadeepsing/drakula-ui
              |
              v  (auto-deploy)
  Backend  -> https://subhadeepsing-drakula.hf.space
  Frontend -> https://subhadeepsing-drakula-ui.static.hf.space/index.html
```

---

## 7. API Key Rotation & Security

- 7 Gemini API keys in a rotating pool (`GEMINI_API_KEYS`)
- On `429 / 403 / 503`, rotates to next key in milliseconds
- Exhausted keys stay in pool and re-engage when daily quota resets
- Groq fallback if all Gemini keys hit quota simultaneously
- **Zero keys in git** — only in `.env` (gitignored) and HF Encrypted Space Secrets

---

## 8. Evaluation Criteria Alignment

| Challenge Criterion | Weight | How We Address It |
| :--- | :---: | :--- |
| **Depth of Reasoning** | 30% | Multi-metric cross-analysis enforced at prompt level. SOC <-> Moisture <-> Cropping <-> Biodiversity. No single-variable answers. |
| **Scientific Grounding** | 25% | 7,978 FAO/IPCC/IPBES passages. Every claim has inline [E####] citation with evidence text visible in Evidence Drawer. Stage 4 audits fabricated numbers. |
| **Knowledge System Design** | 20% | Dual-mode RAG (Qdrant local / NumPy production). Dense + sparse hybrid retrieval. Transparent citation provenance. |
| **Conversational Intelligence** | 15% | 4-route gate. Multi-turn JSON schema accumulation. Never re-asks already-provided variables. |
| **Output Clarity** | 10% | Structured sections: Strategy, Assessment, Interventions, Time Horizons, Trade-offs, Evidence Drawer, Inquiry Cards. |

---

## 9. Local Setup

### Prerequisites
- Python 3.10+, Node.js 18+, npm
- Gemini API key: https://aistudio.google.com
- Groq API key: https://console.groq.com

### Configure
```bash
git clone https://github.com/subh-adeep/Darukaa.earth_chatbot.git
cd Darukaa.earth_chatbot
cp project/.env.example project/.env
# Edit project/.env with your keys
```

### Backend
```bash
cd project/backend
pip install -r requirements.txt
python main.py
# Runs at http://localhost:8000
```

### Frontend
```bash
cd project/frontend
npm install
npm run dev
# Runs at http://localhost:5173
```

---

*Darukaa.Earth — An AI system that reasons like an environmental scientist.*

# Extraction Report: FAO_State_of_Worlds_Biodiversity_2019

## Ingestion Overview
- **Source Document:** `FAO_State_of_Worlds_Biodiversity_2019.pdf`
- **Source File Size:** `18.47 MB`
- **Processing Engine:** MinerU (`magic-pdf`)
- **Acceleration:** CUDA GPU (NVIDIA GeForce GTX 1650)
- **Extraction Time:** `1477.2 seconds`
- **Ingestion Timestamp:** `2026-09-17 14:35:42`

---

## Directory Structure
```
FAO_State_of_Worlds_Biodiversity_2019/
├── FAO_State_of_Worlds_Biodiversity_2019.json       # Complete structured JSON (blocks, bboxes, layout hierarchy)
├── FAO_State_of_Worlds_Biodiversity_2019.md         # Full Markdown export with formulas & tables
├── images/               # Extracted figures and visual exhibits (173 files)
└── README.md             # This verification report
```

---

## Extraction Statistics
| Metric | Extracted Count |
|---|---|
| **Layout Blocks / Elements** | 3751 |
| **Tables Extracted** | 76 |
| **Equations / Formulas** | 0 |
| **Paragraphs Extracted** | 3577 |
| **Images / Figures Saved** | 173 |

---

## Verification & Downstream Pipeline Notes
1. **JSON Verification:**
   - Structured JSON is stored at [`FAO_State_of_Worlds_Biodiversity_2019.json`](./FAO_State_of_Worlds_Biodiversity_2019.json).
   - Each element retains bounding box coordinates, reading order, and semantic block types.
2. **Retrieval Pipeline Compliance:**
   - Per project design rules, image assets in `images/` are stored strictly for inspection/verification.
   - **No image embeddings** and **no visual vectors** are generated into the text retrieval pipeline.
3. **Sample Text Snippet:**
> THE STATE OF THE WORLD’s BIODIVERSITY FOR FOOD AND AGRICULTURE 

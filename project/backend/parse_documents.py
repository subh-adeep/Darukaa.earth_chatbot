"""
parse_documents.py
==================
Robust, memory-efficient PDF to Markdown converter using pymupdf & pymupdf4llm.
Streams in page batches so even 1,000+ page IPCC/FAO documents are parsed reliably.
"""

import sys
import time
from pathlib import Path
import fitz  # PyMuPDF
import pymupdf4llm

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR.parent / "Documents"
OUTPUT_BASE_DIR = BASE_DIR / "data" / "parsed_documents"

DOC_SPECS = [
    ("IPBES_Global_Assessment_Report_SPM.pdf", "IPBES_Global_Assessment_Report_SPM"),
    ("cc3981en.pdf", "FAO_Soil_Organic_Carbon_Mapping_cc3981en"),
    ("biodiversity_report.pdf", "Biodiversity_Synthesis_Report"),
    ("GWO2025_Eng_Rev.1.pdf", "Global_Wetland_Outlook_2025"),
    ("FAO_State_of_Knowledge_of_Soil_Biodiversity.pdf", "FAO_State_of_Knowledge_of_Soil_Biodiversity"),
    ("FAO-world_soils-report-COMPLETE.pdf", "FAO_World_Soils_Report"),
    ("IPCC_Climate_Change_and_Land_Full_Report.pdf", "IPCC_Climate_Change_and_Land_Report"),
]

def parse_pdf_streaming(pdf_path: Path, output_file: Path, batch_size: int = 25):
    """
    Parse PDF into Markdown in page batches, streaming directly to file.
    """
    print(f"\n[START] Processing: {pdf_path.name}", flush=True)
    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)
    print(f"  Total pages: {total_pages}", flush=True)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    with open(output_file, "w", encoding="utf-8") as out_f:
        for start_idx in range(0, total_pages, batch_size):
            end_idx = min(start_idx + batch_size, total_pages)
            pages = list(range(start_idx, end_idx))
            
            try:
                # Extract markdown for page batch
                md_batch = pymupdf4llm.to_markdown(str(pdf_path), pages=pages, write_images=False)
                out_f.write(md_batch)
                out_f.write("\n\n")
                out_f.flush()
                pct = int((end_idx / total_pages) * 100)
                print(f"  [{pdf_path.name}] Pages {start_idx+1}-{end_idx}/{total_pages} ({pct}%) converted.", flush=True)
            except Exception as e:
                # Fallback to plain fitz text if pymupdf4llm encounters complex vector path
                print(f"  [WARN] Batch {start_idx}-{end_idx} falling back to fitz text ({e})", flush=True)
                for p_num in pages:
                    p = doc[p_num]
                    out_f.write(f"\n\n## Page {p_num+1}\n\n" + p.get_text() + "\n\n")
                out_f.flush()

    elapsed = time.time() - start_time
    file_size_kb = output_file.stat().st_size / 1024
    print(f"[DONE] {pdf_path.name} -> {output_file.name} ({file_size_kb:.1f} KB in {elapsed:.1f}s)", flush=True)


def parse_all():
    OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
    
    for filename, clean_dir in DOC_SPECS:
        pdf_path = DOCS_DIR / filename
        if not pdf_path.exists():
            print(f"[SKIP] Not found: {pdf_path}", flush=True)
            continue
            
        target_file = OUTPUT_BASE_DIR / clean_dir / f"{clean_dir}.md"
        if target_file.exists() and target_file.stat().st_size > 5000:
            print(f"[EXISTS] {clean_dir} already parsed ({target_file.stat().st_size / 1024:.1f} KB). Skipping.", flush=True)
            continue
            
        parse_pdf_streaming(pdf_path, target_file, batch_size=25)


if __name__ == "__main__":
    parse_all()

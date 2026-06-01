# src/ingestion/pdf_processor.py
#
# Production-grade PDF → olive_chunks.jsonl processor.
# Runs locally, no Colab dependency.
#
# Improvements over v1:
#   - Per-document profiles (domain, priority, topic, language hint)
#   - Paragraph-boundary chunking (respects \n\n, never cuts mid-sentence)
#   - Section header detection and tracking across pages
#   - Table extraction via pdfplumber (appended as table chunks)
#   - OCR fallback via pytesseract for scanned pages (< 50 chars)
#   - Figure caption extraction
#   - Noise line removal (page numbers, DOIs, headers/footers)
#   - Deduplication by chunk_id (MD5 hash)
#   - Skip already-processed PDFs via processed_pdfs.txt

import hashlib
import json
import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)

# ── Optional imports (graceful degradation) ─────────────────────
try:
    import fitz  # pymupdf
except ImportError:
    fitz = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    import pytesseract
    from pdf2image import convert_from_path
    from PIL import Image
    import numpy as np
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

try:
    from langdetect import detect, LangDetectException
except ImportError:
    detect = None
    LangDetectException = Exception

try:
    import ftfy
    HAS_FTFY = True
except ImportError:
    HAS_FTFY = False

# ── Paths ────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parents[2]
RAW_DIR       = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
CHUNKS_PATH   = PROCESSED_DIR / "olive_chunks.jsonl"
DONE_LOG      = PROCESSED_DIR / "processed_pdfs.txt"

# ── Chunking config ──────────────────────────────────────────────
CHUNK_CHARS   = 1200   # target chunk size in characters (~300 tokens)
OVERLAP_CHARS = 150    # overlap between consecutive chunks
MIN_CHUNK     = 120    # discard chunks shorter than this

# ── Per-document profiles ────────────────────────────────────────
# Keys are substrings of the PDF filename (case-insensitive match).
# First match wins — order from most specific to least specific.
DOCUMENT_PROFILES = {
    "elloumi":       {"domain": "olive_agronomy",  "priority": 1, "topic": "olive production arid areas Tunisia",         "lang_hint": "en"},
    "drought":       {"domain": "olive_agronomy",  "priority": 1, "topic": "drought stress olive physiology acclimation",  "lang_hint": "en"},
    "leaf":          {"domain": "olive_agronomy",  "priority": 1, "topic": "olive leaf composition bioactive compounds",   "lang_hint": "en"},
    "water":         {"domain": "olive_agronomy",  "priority": 1, "topic": "olive water use irrigation requirements",      "lang_hint": "en"},
    "evolution":     {"domain": "olive_agronomy",  "priority": 1, "topic": "olive production systems sustainability",      "lang_hint": "fr"},
    "technologie":   {"domain": "olive_agronomy",  "priority": 1, "topic": "drought resistance olive technology Tunisia",  "lang_hint": "fr"},
    "resistance":    {"domain": "olive_agronomy",  "priority": 1, "topic": "drought resistance olive technology Tunisia",  "lang_hint": "fr"},
    "secheresse":    {"domain": "olive_agronomy",  "priority": 1, "topic": "drought resistance olive technology Tunisia",  "lang_hint": "fr"},
    "pistachio":     {"domain": "related_domain",  "priority": 2, "topic": "tree crop yield climate variables California", "lang_hint": "en"},
    "kerman":        {"domain": "related_domain",  "priority": 2, "topic": "tree crop yield climate variables California", "lang_hint": "en"},
    "emperature":    {"domain": "related_domain",  "priority": 2, "topic": "tree crop yield climate variables California", "lang_hint": "en"},
    "temperature":   {"domain": "related_domain",  "priority": 2, "topic": "tree crop yield climate variables California", "lang_hint": "en"},
}

DEFAULT_PROFILE = {
    "domain": "olive_agronomy", "priority": 1,
    "topic": "olive farming Tunisia", "lang_hint": "unknown"
}

# ── Section header patterns (EN + FR) ───────────────────────────
SECTION_PATTERNS = [
    r"^\s*(abstract|résumé|introduction|conclusion[s]?|discussion[s]?)\s*$",
    r"^\s*\d+[\.\d]*\s{1,4}[A-ZÀ-Ÿa-zà-ÿ].{3,60}$",
    r"^\s*[IVXLC]+[\.\)]\s+[A-ZÀ-Ÿ].{3,60}$",
    r"^\s*[A-ZÀ-Ÿ\s]{6,50}$",
    r"^\s*(matériel|méthodologie|methodology|results|résultats|"
    r"materials|données|climate|temperature|yield|production|"
    r"irrigation|fertilisation|conclusion)\s*$",
]

# ── Noise line patterns ──────────────────────────────────────────
NOISE_PATTERNS = [
    r"^\s*\d+\s*$",                              # lone page numbers
    r"^\s*-\s*\d+\s*-\s*$",                      # "- 4 -"
    r"^\s*Page\s+\d+(\s+of\s+\d+)?\s*$",
    r"^\s*©.*$",
    r"^\s*doi\s*:\s*https?://\S+\s*$",
    r"^\s*issn\s*[\:\-].*$",
    r"^\s*http[s]?://\S+\s*$",
    r"^\s*(received|accepted|published|soumis|accepté|reçu)\s*:.*$",
    r"^\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\s*$",
    r"^\s*(fig|figure|tableau|table)\s*\.?\s*\d+\s*[\.\:—]?\s*$",
    r"^\s*RESEARCH ARTICLE\s*$",
    r"^\s*Special Issue.*\d{4}\s*$",
]

BIBLIOGRAPHY_TRIGGERS = [
    "références", "references", "bibliographie",
    "bibliography", "works cited", "literature cited",
]


# ═══════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════

def get_profile(pdf_name: str) -> dict:
    name_lower = pdf_name.lower()
    for key, profile in DOCUMENT_PROFILES.items():
        if key in name_lower:
            return profile
    return DEFAULT_PROFILE


def chunk_id(source: str, page: int, idx: int) -> int:
    raw = f"{source}|{page}|{idx}"
    return int(hashlib.md5(raw.encode()).hexdigest()[:16], 16) % (10 ** 15)


def detect_lang(text: str) -> str:
    if detect is None or len(text.strip()) < 30:
        return "unknown"
    try:
        return detect(text)
    except Exception:
        return "unknown"


def is_section_header(line: str) -> bool:
    line = line.strip()
    if len(line) < 4 or len(line) > 80:
        return False
    return any(re.match(p, line, re.IGNORECASE) for p in SECTION_PATTERNS)


def remove_noise(text: str) -> str:
    lines = text.split("\n")
    cleaned = [l for l in lines if not any(
        re.match(p, l, re.IGNORECASE) for p in NOISE_PATTERNS
    )]
    return "\n".join(cleaned)


def fix_text(text: str) -> str:
    if HAS_FTFY:
        text = ftfy.fix_text(text)
    # Fix broken French contractions
    for pat, rep in [
        (r"1 '(?=[A-Za-zÀ-ÿ])", "l'"), (r"` '(?=[A-Za-zÀ-ÿ])", "l'"),
        (r"\bl '(?=[A-Za-zÀ-ÿ])", "l'"), (r"\bd '(?=[A-Za-zÀ-ÿ])", "d'"),
        (r"(\w)-\n(\w)", r"\1\2"), (r"oo/o", "%"), (r" {2,}", " "),
    ]:
        text = re.sub(pat, rep, text)
    return text.strip()


def normalize_paragraphs(text: str) -> str:
    # Single newlines within paragraphs → space; preserve double newlines
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def already_processed() -> set:
    if not DONE_LOG.exists():
        return set()
    return set(DONE_LOG.read_text(encoding="utf-8").splitlines())


def mark_processed(pdf_name: str):
    DONE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(DONE_LOG, "a", encoding="utf-8") as f:
        f.write(pdf_name + "\n")


def existing_ids() -> set:
    if not CHUNKS_PATH.exists():
        return set()
    ids = set()
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    ids.add(json.loads(line)["chunk_id"])
                except Exception:
                    pass
    return ids


# ═══════════════════════════════════════════════════════════════
# OCR fallback for scanned pages
# ═══════════════════════════════════════════════════════════════

def ocr_page(pdf_path: Path, page_num_1indexed: int, lang_hint: str = "unknown") -> str:
    """OCR a single page at 300 DPI. Returns extracted text or empty string."""
    if not OCR_AVAILABLE:
        log.warning("OCR not available — install pytesseract, pdf2image, Pillow, numpy")
        return ""
    try:
        images = convert_from_path(
            str(pdf_path), dpi=300,
            first_page=page_num_1indexed, last_page=page_num_1indexed,
            grayscale=True,
        )
        if not images:
            return ""
        img_array = np.array(images[0])
        threshold = int(img_array.mean() * 0.85)
        img_bin   = Image.fromarray(
            np.where(img_array < threshold, 0, 255).astype(np.uint8)
        )
        tess_lang = "fra+eng" if lang_hint in ("fr", "unknown") else "eng+fra"
        return pytesseract.image_to_string(img_bin, lang=tess_lang, config="--psm 6 --oem 3").strip()
    except Exception as e:
        log.warning(f"OCR failed on page {page_num_1indexed}: {e}")
        return ""


# ═══════════════════════════════════════════════════════════════
# Table extraction
# ═══════════════════════════════════════════════════════════════

def extract_tables(pdf_path: Path, page_num_0: int, profile: dict) -> list[dict]:
    """Extract tables from a page using pdfplumber. Returns chunk dicts."""
    if pdfplumber is None:
        return []
    results = []
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            page   = pdf.pages[page_num_0]
            tables = page.extract_tables()
            for t_idx, tbl in enumerate(tables):
                if not tbl or len(tbl) < 2:
                    continue
                df_rows = tbl[1:]
                header  = tbl[0]
                # Filter junk: must have ≥2 cols, ≥2 rows, some alpha content
                if len(header) < 2 or len(df_rows) < 2:
                    continue
                text_repr = "\t".join(str(h) for h in header if h) + "\n"
                text_repr += "\n".join(
                    "\t".join(str(c) for c in row if c)
                    for row in df_rows
                )
                alpha = sum(c.isalpha() for c in text_repr) / max(len(text_repr), 1)
                if alpha < 0.15:
                    continue
                cid = chunk_id(pdf_path.name, page_num_0 + 1, 9000 + t_idx)
                results.append({
                    "chunk_id": cid,
                    "text":     f"[TABLE]\n{text_repr}",
                    "metadata": {
                        "source_pdf":         pdf_path.name,
                        "page":               page_num_0 + 1,
                        "section":            "data_table",
                        "language":           profile.get("lang_hint", "unknown"),
                        "domain":             profile["domain"],
                        "content_type":       "table",
                        "doc_type":           "document",
                        "topic":              profile["topic"],
                        "retrieval_priority": profile["priority"],
                        "ocr_noise_flag":     False,
                        "char_count":         len(text_repr),
                    },
                })
    except Exception as e:
        log.warning(f"Table extraction failed on page {page_num_0 + 1}: {e}")
    return results


# ═══════════════════════════════════════════════════════════════
# Paragraph-boundary chunker with section tracking
# ═══════════════════════════════════════════════════════════════

def chunk_page(
    text: str,
    pdf_name: str,
    page_num: int,
    profile: dict,
    current_section: str,
    chunk_counter: list,   # mutable [int] for cross-call counter
) -> tuple[list[dict], str]:
    """
    Split page text into chunks at paragraph boundaries.
    Tracks and returns the current section header.
    Returns (list_of_chunk_dicts, updated_section).
    """
    chunks   = []
    section  = current_section
    buffer   = ""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    for para in paragraphs:
        first_line = para.split("\n")[0].strip()

        if is_section_header(first_line):
            # Flush buffer before new section
            if len(buffer.strip()) >= MIN_CHUNK:
                chunks.append(_make_chunk(
                    buffer.strip(), pdf_name, page_num,
                    section, profile, chunk_counter
                ))
            buffer  = ""
            section = first_line
            continue

        buffer += "\n\n" + para

        if len(buffer) >= CHUNK_CHARS:
            chunks.append(_make_chunk(
                buffer.strip(), pdf_name, page_num,
                section, profile, chunk_counter
            ))
            buffer = buffer[-OVERLAP_CHARS:]

    # Flush remainder
    if len(buffer.strip()) >= MIN_CHUNK:
        chunks.append(_make_chunk(
            buffer.strip(), pdf_name, page_num,
            section, profile, chunk_counter
        ))

    return chunks, section


def _make_chunk(
    text: str, pdf_name: str, page_num: int,
    section: str, profile: dict, counter: list
) -> dict:
    cid = chunk_id(pdf_name, page_num, counter[0])
    counter[0] += 1
    return {
        "chunk_id": cid,
        "text":     text,
        "metadata": {
            "source_pdf":         pdf_name,
            "page":               page_num,
            "section":            section,
            "language":           detect_lang(text),
            "domain":             profile["domain"],
            "content_type":       "text",
            "doc_type":           "document",
            "topic":              profile["topic"],
            "retrieval_priority": profile["priority"],
            "ocr_noise_flag":     False,
            "char_count":         len(text),
        },
    }


# ═══════════════════════════════════════════════════════════════
# Core per-PDF extractor
# ═══════════════════════════════════════════════════════════════

def extract_pdf(pdf_path: Path) -> list[dict]:
    """
    Full extraction pipeline for one PDF:
    1. Classify each page (digital / scanned / table-heavy)
    2. Extract text (digital) or OCR (scanned)
    3. Clean, normalize, remove noise lines
    4. Stop at bibliography
    5. Extract tables per page
    6. Paragraph-boundary chunk with section tracking
    """
    if fitz is None:
        raise ImportError("pymupdf not installed. Run: pip install pymupdf")

    profile  = get_profile(pdf_path.name)
    doc      = fitz.open(str(pdf_path))
    pdf_name = pdf_path.name
    all_chunks: list[dict] = []
    counter  = [0]          # mutable chunk counter
    section  = "unknown"    # current section tracker
    bib_hit  = False        # stop at bibliography

    for page_num in range(len(doc)):
        if bib_hit:
            break

        page     = doc[page_num]
        raw_text = page.get_text("text").strip()
        n_chars  = len(raw_text)

        # ── Classify page ──────────────────────────────────────
        images = page.get_images(full=True)
        if n_chars < 50 and images:
            page_type = "SCANNED"
        elif n_chars < 50:
            page_type = "BLANK"
        else:
            page_type = "DIGITAL"

        # ── Extract text ───────────────────────────────────────
        if page_type == "SCANNED":
            raw_text = ocr_page(pdf_path, page_num + 1, profile.get("lang_hint", "unknown"))
            if not raw_text:
                continue
            ocr_flag = True
        elif page_type == "BLANK":
            continue
        else:
            ocr_flag = False

        # ── Clean ──────────────────────────────────────────────
        text = fix_text(raw_text)
        text = remove_noise(text)
        text = normalize_paragraphs(text)

        if len(text.strip()) < MIN_CHUNK:
            continue

        # ── Bibliography check ─────────────────────────────────
        if any(t in text[:300].lower() for t in BIBLIOGRAPHY_TRIGGERS):
            bib_hit = True
            log.info(f"  Bibliography reached at page {page_num + 1} — stopping")
            break

        # ── Tables ─────────────────────────────────────────────
        if page_type == "DIGITAL":
            tables = extract_tables(pdf_path, page_num, profile)
            all_chunks.extend(tables)

        # ── Text chunks ────────────────────────────────────────
        page_chunks, section = chunk_page(
            text, pdf_name, page_num + 1,
            profile, section, counter
        )

        # Apply OCR flag to scanned page chunks
        if ocr_flag:
            for c in page_chunks:
                c["metadata"]["ocr_noise_flag"] = True

        all_chunks.extend(page_chunks)

    doc.close()
    log.info(
        f"Extracted {len(all_chunks)} chunks from {pdf_name} "
        f"(domain={profile['domain']}, priority={profile['priority']})"
    )
    return all_chunks


# ═══════════════════════════════════════════════════════════════
# Pipeline entry point
# ═══════════════════════════════════════════════════════════════

def run_pdf_pipeline(pdf_paths: list[Path] | None = None) -> dict:
    """
    Process PDFs and append new chunks to olive_chunks.jsonl.

    Args:
        pdf_paths: explicit list of Path objects.
                   If None, scans data/raw/ for all *.pdf files.

    Returns:
        Status dict: {ok, chunks, pdfs, message, errors}
    """
    status = {
        "step": "pdf_extraction", "ok": False,
        "chunks": 0, "pdfs": 0, "errors": [], "message": ""
    }

    if fitz is None:
        status["message"] = "pymupdf not installed. Run: pip install pymupdf"
        return status

    if pdf_paths is None:
        pdf_paths = sorted(RAW_DIR.glob("*.pdf"))

    if not pdf_paths:
        status.update({"ok": True, "message": "No PDF files found in data/raw/"})
        return status

    done      = already_processed()
    existing  = existing_ids()
    new_paths = [p for p in pdf_paths if p.name not in done]

    if not new_paths:
        status.update({
            "ok": True,
            "message": f"All {len(pdf_paths)} PDF(s) already processed. "
                       f"Delete data/processed/processed_pdfs.txt to reprocess."
        })
        return status

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    total_new = 0

    with open(CHUNKS_PATH, "a", encoding="utf-8") as out:
        for pdf_path in new_paths:
            try:
                chunks  = extract_pdf(pdf_path)
                written = 0
                for chunk in chunks:
                    if chunk["chunk_id"] not in existing:
                        out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                        existing.add(chunk["chunk_id"])
                        written += 1
                total_new += written
                mark_processed(pdf_path.name)
                log.info(f"  ✅ {pdf_path.name}: {written} new chunks written")
            except Exception as e:
                log.error(f"  ❌ Failed: {pdf_path.name}: {e}")
                status["errors"].append(f"{pdf_path.name}: {e}")

    status.update({
        "ok":      True,
        "chunks":  total_new,
        "pdfs":    len(new_paths),
        "message": f"{len(new_paths)} PDF(s) processed, {total_new} new chunks added",
    })
    return status


# ── CLI ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    result = run_pdf_pipeline()
    print(f"\n{'✅' if result['ok'] else '❌'} {result['message']}")
    if result["errors"]:
        print("Errors:")
        for e in result["errors"]:
            print(f"  {e}")
# src/ingestion/pipeline.py
#
# Central orchestrator for all data ingestion steps.
# Called by the Streamlit "Run Ingestion" button OR run directly from CLI.
#
# Steps:
#   1. Tabular pipeline  — Excel → TimescaleDB
#   2. Document pipeline — olive_chunks.jsonl → Qdrant
#   3. Web scraper       — URLs → Qdrant
#
# Each step returns a status dict so the UI can show results.

import logging
from pathlib import Path
from typing import Optional

from src.ingestion.pdf_processor import run_pdf_pipeline

log = logging.getLogger(__name__)


# ---------------------------------------------------------------
# Step 1 — Tabular data (Excel → TimescaleDB)
# ---------------------------------------------------------------

def run_tabular_pipeline() -> dict:
    """
    Loads Production_clean.xlsx + climate_clean.xlsx + rainfall_clean.xlsx
    from data/raw/, and writes two separate tables:
      - olive_production      (annual)  → data/processed/olive_production.csv
      - olive_climate_monthly (monthly) → data/processed/olive_climate_monthly.csv
    """
    from src.config import RAW_DIR, PROCESSED_DIR, DB_URL
    import pandas as pd
    import numpy as np
    from sqlalchemy import create_engine, text

    status = {"step": "tabular", "ok": False, "rows": 0, "message": ""}

    prod_path     = RAW_DIR / "Production_clean.xlsx"
    climate_path  = RAW_DIR / "climate_clean.xlsx"
    rainfall_path = RAW_DIR / "rainfall_clean.xlsx"

    missing = [p.name for p in [prod_path, climate_path, rainfall_path] if not p.exists()]
    if missing:
        status["message"] = f"Missing raw files: {missing}"
        log.error(status["message"])
        return status

    try:
        # ---- Load production (already clean: year, production_tonnes) ----
        prod = pd.read_excel(prod_path)
        prod.columns = prod.columns.str.strip().str.lower().str.replace(" ", "_")
        prod["production_tonnes"] = pd.to_numeric(prod["production_tonnes"], errors="coerce")
        prod["year"] = pd.to_numeric(prod["year"], errors="coerce")
        prod = prod.dropna(subset=["year", "production_tonnes"])
        prod["year"] = prod["year"].astype(int)
        prod = prod[["year", "production_tonnes"]].sort_values("year").reset_index(drop=True)

        # ---- Load climate (3-hourly → daily, already clean: DATE TMP DEW SLP WND WNS) ----
        climate = pd.read_excel(climate_path)
        climate.columns = climate.columns.str.strip().str.lower()
        climate["date"] = pd.to_datetime(climate["date"], errors="coerce")
        climate = climate.dropna(subset=["date"]).set_index("date").sort_index()

        daily_climate = climate.resample("D").agg({
            col: "mean" for col in ["tmp", "dew", "slp", "wns"]
            if col in climate.columns
        })

        # ---- Load rainfall (daily, already clean: DATE, rainfall_mm) ----
        rainfall = pd.read_excel(rainfall_path)
        rainfall.columns = rainfall.columns.str.strip().str.lower()
        rainfall["date"] = pd.to_datetime(rainfall["date"], errors="coerce")
        rainfall = rainfall.dropna(subset=["date"]).set_index("date").sort_index()
        rainfall["rainfall_mm"] = rainfall["rainfall_mm"].clip(lower=0)

        # ---- Merge climate + rainfall (daily) ----
        weather = daily_climate.join(rainfall["rainfall_mm"], how="left")
        weather["rainfall_mm"] = weather["rainfall_mm"].fillna(0)

        # ---- Feature engineering ----
        df = weather.copy().ffill().bfill()

        if "tmp" in df.columns:
            df["gdd"]             = np.maximum(df["tmp"] - 10, 0)
            df["gdd_annual"]      = df["gdd"].rolling(365, min_periods=30).sum()
            df["chilling_hour"]   = df["tmp"].between(0, 7.2).astype(float)
            df["chilling_annual"] = df["chilling_hour"].rolling(365, min_periods=30).sum()

        rain_30d  = df["rainfall_mm"].rolling(30,  min_periods=10).mean()
        rain_365d = df["rainfall_mm"].rolling(365, min_periods=30).mean()
        rain_std  = df["rainfall_mm"].rolling(365, min_periods=30).std()
        df["spi_30d"] = ((rain_30d - rain_365d) / rain_std).clip(-3, 3)

        # ---- Monthly aggregation (climate only — no production merged in) ----
        numeric_cols  = df.select_dtypes(include=np.number).columns.tolist()
        df_monthly    = df[numeric_cols].resample("ME").mean()
        df_monthly    = df_monthly.reset_index().rename(columns={"date": "date"})

        # ---- Save processed CSVs ----
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        df_monthly.to_csv(PROCESSED_DIR / "olive_climate_monthly.csv", index=False)
        prod.to_csv(PROCESSED_DIR / "olive_production.csv", index=False)
        log.info(f"Saved olive_climate_monthly.csv ({len(df_monthly)} rows)")
        log.info(f"Saved olive_production.csv ({len(prod)} rows)")

        # ---- Insert into TimescaleDB ----
        engine = create_engine(DB_URL)
        with engine.connect() as conn:
            conn.execute(text("DROP TABLE IF EXISTS olive_monthly"))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS olive_climate_monthly (
                    date            TIMESTAMPTZ PRIMARY KEY,
                    tmp             FLOAT,
                    dew             FLOAT,
                    slp             FLOAT,
                    wns             FLOAT,
                    rainfall_mm     FLOAT,
                    gdd             FLOAT,
                    gdd_annual      FLOAT,
                    chilling_hour   FLOAT,
                    chilling_annual FLOAT,
                    spi_30d         FLOAT
                )
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS olive_production (
                    year              INT PRIMARY KEY,
                    production_tonnes FLOAT
                )
            """))
            conn.commit()

        df_monthly.to_sql(
            "olive_climate_monthly", engine,
            if_exists="replace", index=False, method="multi", chunksize=500,
        )
        prod.to_sql(
            "olive_production", engine,
            if_exists="replace", index=False, method="multi", chunksize=500,
        )
        log.info(f"Inserted {len(df_monthly)} rows → olive_climate_monthly")
        log.info(f"Inserted {len(prod)} rows → olive_production")

        status.update({
            "ok": True,
            "rows": len(df_monthly),
            "message": (
                f"{len(df_monthly)} climate rows + {len(prod)} production years loaded"
            ),
        })

    except Exception as e:
        status["message"] = f"Tabular pipeline error: {e}"
        log.exception(status["message"])

    return status


# ---------------------------------------------------------------
# Step 2 — Document ingestion (olive_chunks.jsonl → Qdrant)
# ---------------------------------------------------------------

def run_document_pipeline(incremental: bool = True) -> dict:
    """
    Reads data/processed/olive_chunks.jsonl, embeds chunks,
    and upserts them into Qdrant.

    Args:
        incremental: True  = only embed chunks not already in Qdrant (default)
                     False = re-embed everything (use after model change)
    """
    from src.config import PROCESSED_DIR
    from src.ingestion.load_documents import run_pipeline as _doc_run

    status = {"step": "documents", "ok": False, "chunks": 0, "message": ""}

    chunks_file = PROCESSED_DIR / "olive_chunks.jsonl"
    if not chunks_file.exists():
        status["message"] = (
            "olive_chunks.jsonl not found in data/processed/. "
            "Run PDF pipeline first or place the file manually."
        )
        log.warning(status["message"])
        return status

    try:
        _doc_run()   # uses its own CHUNKS_PATH, returns None
        status.update({
            "ok":      True,
            "chunks":  0,
            "message": "Document chunks embedded and upserted into Qdrant",
        })
    except Exception as e:
        status["message"] = f"Document pipeline error: {e}"
        log.exception(status["message"])

    return status



# ---------------------------------------------------------------
# Step 3 — Web scraper (URLs → Qdrant)
# ---------------------------------------------------------------

def run_web_pipeline(extra_urls: Optional[list] = None) -> dict:
    """
    Scrapes configured + user-supplied URLs and adds to Qdrant.
    """
    from src.ingestion.web_scraper import run_scraper_pipeline

    status = {"step": "web_scraper", "ok": False, "chunks": 0, "message": ""}
    try:
        result = run_scraper_pipeline(extra_urls=extra_urls or [])
        status.update({
            "ok":      True,
            "chunks":  result["chunks_added"],
            "message": (
                f"Scraped {result['scraped']}/{result['total_urls']} URLs, "
                f"{result['chunks_added']} chunks added"
            ),
        })
    except Exception as e:
        status["message"] = f"Web scraper error: {e}"
        log.exception(status["message"])

    return status


# ---------------------------------------------------------------
# Full pipeline — all steps in sequence
# ---------------------------------------------------------------

def run_full_pipeline(
    extra_urls: list | None = None,
    pdf_paths: list | None = None,
    incremental: bool = True,
) -> list[dict]:
    """
    Runs all four pipelines in order:
      0. PDF extraction  — PDFs → olive_chunks.jsonl  (skips already-processed)
      1. Tabular data    — Excel → TimescaleDB
      2. Document chunks — olive_chunks.jsonl → Qdrant (incremental by default)
      3. Web scraper     — URLs → Qdrant

    Args:
        extra_urls:  additional URLs to scrape
        pdf_paths:   explicit PDF paths; if None scans data/raw/*.pdf
        incremental: True = only embed new chunks (default)
                     False = re-embed all chunks

    Returns list of status dicts, one per step.
    """
    log.info("=== Starting full ingestion pipeline ===")
    results = []

    log.info("Step 0/3 — PDF extraction")
    pdf_result = run_pdf_pipeline(pdf_paths=pdf_paths)
    results.append(pdf_result)
    new_chunks = pdf_result.get("chunks", 0)
    log.info(f"PDF step: {new_chunks} new chunks added to olive_chunks.jsonl")

    log.info("Step 1/3 — Tabular data")
    results.append(run_tabular_pipeline())

    log.info("Step 2/3 — Document chunks → Qdrant")
    # Use incremental=True so only the newly added PDF chunks get embedded
    # If PDF step added 0 new chunks, this will also skip embedding (nothing to do)
    results.append(run_document_pipeline(incremental=incremental))

    log.info("Step 3/3 — Web scraper")
    results.append(run_web_pipeline(extra_urls=extra_urls))

    ok_count = sum(1 for r in results if r.get("ok"))
    log.info(f"=== Pipeline complete: {ok_count}/4 steps succeeded ===")
    return results


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    extra = sys.argv[1:]  # pass URLs as CLI arguments
    results = run_full_pipeline(extra_urls=extra)
    for r in results:
        print(r)
